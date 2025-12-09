#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) Training for MIDAS.

Features:
- PyTorch TFT implementation
- Multi-horizon forecasting capability
- Attention-based interpretable predictions
- Variable selection networks
"""
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from utils import (
    get_logger, setup_experiment_dir, save_metrics,
    compute_regression_metrics, compute_directional_accuracy,
    plot_predictions, plot_training_curves,
    check_gpu_availability, get_device_info, OUTPUT_DIR
)
from dataset import MIDASDataset
from preprocessing import Preprocessor, prepare_sequences_with_target
from splitter import OFISafeSplitter

logger = get_logger("train_tft")


# ============================================
# TFT Components
# ============================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for feature gating."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x):
        out = self.fc(x)
        return out[:, :, :self.output_dim] * torch.sigmoid(out[:, :, self.output_dim:])


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for non-linear processing."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Dense layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # GLU
        self.glu = GatedLinearUnit(output_dim, output_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Skip connection
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context=None):
        # Primary pathway
        hidden = F.elu(self.fc1(x))
        
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)
        
        hidden = self.dropout(F.elu(self.fc2(hidden)))
        hidden = self.glu(hidden.unsqueeze(1) if hidden.dim() == 2 else hidden)
        if hidden.dim() == 3 and hidden.size(1) == 1:
            hidden = hidden.squeeze(1)
        
        # Skip connection
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x
        
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""
    
    def __init__(
        self,
        input_dim: int,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Feature-wise GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout, context_dim)
            for _ in range(n_features)
        ])
        
        # Variable selection weights
        self.selection_grn = GatedResidualNetwork(
            n_features * hidden_dim, hidden_dim, n_features, dropout, context_dim
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context=None):
        # x: (batch, seq_len, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # Process each feature
        processed_features = []
        for i in range(n_features):
            feat = x[:, :, i:i+1]  # (batch, seq_len, 1)
            processed = self.feature_grns[i](feat, context)
            processed_features.append(processed)
        
        # Stack and flatten
        stacked = torch.cat(processed_features, dim=-1)  # (batch, seq_len, n_features * hidden)
        
        # Variable selection weights
        weights = self.selection_grn(stacked, context)
        weights = self.softmax(weights)  # (batch, seq_len, n_features)
        
        # Weighted combination
        selected = (x * weights).sum(dim=-1, keepdim=True)
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretable weights."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_linear(context)
        
        return output, attn_weights


class TemporalFusionDecoder(nn.Module):
    """Temporal fusion decoder with self-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x, mask)
        
        # Add & norm
        x = self.layer_norm(x + attn_out)
        
        # GRN
        out = self.grn(x)
        
        return out, attn_weights


# ============================================
# Full TFT Model
# ============================================

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for HFT prediction.
    
    Adapted for single-target regression without future known inputs.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(n_features, hidden_dim)
        
        # Variable selection
        self.var_selection = VariableSelectionNetwork(
            hidden_dim, n_features, hidden_dim, dropout
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=1,  # After variable selection
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=False,
        )
        
        # Temporal fusion decoder
        self.decoder = TemporalFusionDecoder(hidden_dim, n_heads, dropout)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        selected, var_weights = self.var_selection(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(selected)
        
        # Temporal fusion decoding
        decoded, attn_weights = self.decoder(lstm_out)
        
        # Use last timestep for prediction
        final_hidden = decoded[:, -1, :]
        
        # Output
        output = self.output_layer(final_hidden)
        
        return output.squeeze(-1), {
            "variable_weights": var_weights,
            "attention_weights": attn_weights,
        }


class SimpleTFT(nn.Module):
    """
    Simplified TFT that's easier to train.
    Uses the core TFT concepts without full complexity.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Feature embedding
        self.feature_embedding = nn.Linear(n_features, hidden_dim)
        
        # Positional encoding
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        
        # Embed features
        embedded = self.feature_embedding(x)
        embedded = self.pos_dropout(embedded)
        
        # Transformer
        encoded = self.transformer(embedded)
        
        # Attention pooling
        attn_weights = self.attention_pool(encoded)  # (batch, seq_len, 1)
        pooled = (encoded * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        
        # Output
        output = self.output(pooled)
        
        return output.squeeze(-1), {"attention_weights": attn_weights}


# ============================================
# Training Functions
# ============================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred, _ = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    return total_loss / n_batches, np.concatenate(all_preds), np.concatenate(all_targets)


def train_tft(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train TFT model."""
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Data loaders
    from train_lstm import SequenceDataset
    
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    batch_size = config.get("batch_size", 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Model
    n_features = X_train.shape[2]
    
    if config.get("use_simple", True):
        model = SimpleTFT(
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 64),
            n_heads=config.get("n_heads", 4),
            dropout=config.get("dropout", 0.1),
        )
    else:
        model = TemporalFusionTransformer(
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 64),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.1),
        )
    
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    # Scheduler
    n_epochs = config.get("n_epochs", 100)
    total_steps = len(train_loader) * n_epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.get("learning_rate", 1e-3),
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 15)
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - "
                   f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    train_time = time.time() - start_time
    
    return model, {
        "train_time_seconds": train_time,
        "best_epoch": len(history["train_loss"]) - patience_counter,
        "best_val_loss": best_val_loss,
        "n_parameters": n_params,
        "history": history,
    }


def run_training_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    target_type: str = "return",
    target_horizon: int = 10,
    sequence_length: int = 50,
    train_ratio: float = 0.8,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full TFT training pipeline."""
    if output_dir is None:
        output_dir = setup_experiment_dir("tft")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if config is None:
        config = {
            "hidden_dim": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "use_simple": True,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "n_epochs": 100,
            "early_stopping_patience": 15,
        }
    
    results = {
        "model": "tft",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_info": get_device_info(),
        "config": {
            "target_type": target_type,
            "target_horizon": target_horizon,
            "sequence_length": sequence_length,
            **config,
        }
    }
    
    logger.info("=" * 60)
    logger.info("MIDAS TFT Training Pipeline")
    logger.info("=" * 60)
    
    # Load and prepare data (same as LSTM)
    dataset = MIDASDataset(data_dir)
    dataset.load()
    dataset.create_target(target_type, target_horizon)
    dataset.select_features()
    feature_names = dataset.feature_names
    
    splitter = OFISafeSplitter(train_ratio=train_ratio)
    splitter.fit(dataset.df)
    train_df, test_df = splitter.get_all_train_test(dataset.df)
    
    train_df = train_df.drop_nulls(["target"])
    test_df = test_df.drop_nulls(["target"])
    
    prep = Preprocessor(normalize="zscore", clip_outliers_std=5.0)
    train_df = prep.fit_transform(train_df, feature_names)
    test_df = prep.transform(test_df)
    prep.save(output_dir / "preprocessor")
    
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["target"].to_numpy()
    
    X_train_seq, y_train_seq = prepare_sequences_with_target(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = prepare_sequences_with_target(X_test, y_test, sequence_length)
    
    val_size = int(len(X_train_seq) * 0.15)
    X_val_seq = X_train_seq[-val_size:]
    y_val_seq = y_train_seq[-val_size:]
    X_train_seq = X_train_seq[:-val_size]
    y_train_seq = y_train_seq[:-val_size]
    
    # Train
    model, train_info = train_tft(X_train_seq, y_train_seq, X_val_seq, y_val_seq, config)
    results["training"] = {k: v for k, v in train_info.items() if k != "history"}
    
    # Evaluate
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    from train_lstm import SequenceDataset
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    _, y_pred, y_true = validate(model, test_loader, nn.MSELoss(), device)
    
    test_metrics = {
        "regression": compute_regression_metrics(y_true, y_pred),
        "directional": compute_directional_accuracy(y_true, y_pred),
    }
    results["metrics"] = test_metrics
    
    logger.info(f"\n{'='*40}")
    logger.info("Test Results:")
    logger.info(f"  MSE: {test_metrics['regression']['mse']:.6f}")
    logger.info(f"  R²: {test_metrics['regression']['r2']:.4f}")
    logger.info(f"  Directional Accuracy: {test_metrics['directional']['accuracy']:.2%}")
    logger.info(f"{'='*40}")
    
    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_names": feature_names,
        "sequence_length": sequence_length,
    }, output_dir / "model.pt")
    
    save_metrics(results, output_dir / "results.json")
    plot_predictions(y_true, y_pred, output_dir / "predictions.png", n_points=1000)
    plot_training_curves(train_info["history"], output_dir / "training_curves.png")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TFT model for MIDAS")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--target-type", type=str, default="return")
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=100)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data/features")
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    config = {
        "hidden_dim": args.hidden_dim,
        "n_epochs": args.n_epochs,
    }
    
    results = run_training_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        target_type=args.target_type,
        target_horizon=args.target_horizon,
        sequence_length=args.sequence_length,
        config=config,
    )
    
    return results


if __name__ == "__main__":
    main()
