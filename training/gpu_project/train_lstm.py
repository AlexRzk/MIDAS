#!/usr/bin/env python3
"""
LSTM GPU Training for MIDAS.

Features:
- PyTorch LSTM with GPU acceleration
- Sequence handling for time series
- Learning rate scheduling
- Early stopping
- Gradient clipping
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from utils import (
    get_logger, setup_experiment_dir, save_metrics,
    compute_regression_metrics, compute_directional_accuracy,
    compute_trading_metrics, plot_predictions, plot_training_curves,
    check_gpu_availability, get_device_info, OUTPUT_DIR
)
from dataset import MIDASDataset
from preprocessing import Preprocessor, prepare_sequences_with_target
from splitter import OFISafeSplitter

logger = get_logger("train_lstm")


# ============================================
# PyTorch Dataset
# ============================================

class SequenceDataset(Dataset):
    """PyTorch dataset for LSTM sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Sequences of shape (n_sequences, seq_len, n_features)
            y: Targets of shape (n_sequences,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# LSTM Model Architecture
# ============================================

class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Predict
        out = self.fc(last_output)
        return out.squeeze(-1)


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Attention weights
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        
        # Output
        out = self.fc(context)
        return out.squeeze(-1)


# ============================================
# Training Functions
# ============================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
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
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / n_batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    return avg_loss, y_pred, y_true


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train LSTM model.
    
    Args:
        X_train: Training sequences (n_samples, seq_len, n_features)
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        config: Training configuration
        
    Returns:
        Trained model and training history
    """
    # Device
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Data loaders
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Model
    input_size = X_train.shape[2]
    
    if config.get("use_attention", False):
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
        )
    else:
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            bidirectional=config.get("bidirectional", False),
        )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )
    
    # Loss
    criterion = nn.MSELoss()
    
    # Training loop
    n_epochs = config.get("n_epochs", 100)
    early_stopping_patience = config.get("early_stopping_patience", 15)
    grad_clip = config.get("grad_clip", 1.0)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        
        # Validate
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - "
                   f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                   f"LR: {current_lr:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    info = {
        "train_time_seconds": train_time,
        "best_epoch": len(history["train_loss"]) - patience_counter,
        "best_val_loss": best_val_loss,
        "n_parameters": n_params,
        "history": history,
    }
    
    logger.info(f"Training complete in {train_time:.1f}s")
    
    return model, info


def run_training_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    target_type: str = "return",
    target_horizon: int = 10,
    sequence_length: int = 50,
    train_ratio: float = 0.8,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run full LSTM training pipeline.
    """
    # Setup
    if output_dir is None:
        output_dir = setup_experiment_dir("lstm")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default config
    if config is None:
        config = {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
            "use_attention": True,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "n_epochs": 100,
            "early_stopping_patience": 15,
            "grad_clip": 1.0,
        }
    
    results = {
        "model": "lstm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_info": get_device_info(),
        "config": {
            "target_type": target_type,
            "target_horizon": target_horizon,
            "sequence_length": sequence_length,
            "train_ratio": train_ratio,
            **config,
        }
    }
    
    logger.info("=" * 60)
    logger.info("MIDAS LSTM Training Pipeline")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n[1/6] Loading data...")
    dataset = MIDASDataset(data_dir)
    dataset.load()
    
    # 2. Create target
    logger.info("\n[2/6] Creating target...")
    dataset.create_target(target_type, target_horizon)
    dataset.select_features()
    feature_names = dataset.feature_names
    
    # 3. Split data
    logger.info("\n[3/6] Splitting data...")
    splitter = OFISafeSplitter(train_ratio=train_ratio)
    splitter.fit(dataset.df)
    train_df, test_df = splitter.get_all_train_test(dataset.df)
    
    train_df = train_df.drop_nulls(["target"])
    test_df = test_df.drop_nulls(["target"])
    
    # 4. Preprocess
    logger.info("\n[4/6] Preprocessing...")
    prep = Preprocessor(normalize="zscore", clip_outliers_std=5.0)
    train_df = prep.fit_transform(train_df, feature_names)
    test_df = prep.transform(test_df)
    prep.save(output_dir / "preprocessor")
    
    # Convert to numpy
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["target"].to_numpy()
    
    # 5. Prepare sequences
    logger.info("\n[5/6] Preparing sequences...")
    X_train_seq, y_train_seq = prepare_sequences_with_target(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = prepare_sequences_with_target(X_test, y_test, sequence_length)
    
    logger.info(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")
    
    # Split train for validation
    val_size = int(len(X_train_seq) * 0.15)
    X_val_seq = X_train_seq[-val_size:]
    y_val_seq = y_train_seq[-val_size:]
    X_train_seq = X_train_seq[:-val_size]
    y_train_seq = y_train_seq[:-val_size]
    
    # 6. Train
    logger.info("\n[6/6] Training model...")
    model, train_info = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, config)
    results["training"] = {k: v for k, v in train_info.items() if k != "history"}
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    device = torch.device("cuda" if check_gpu_availability() else "cpu")
    model.eval()
    
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    _, y_pred, y_true = validate(model, test_loader, nn.MSELoss(), device)
    
    test_metrics = {
        "regression": compute_regression_metrics(y_true, y_pred),
        "directional": compute_directional_accuracy(y_true, y_pred),
        "n_test_samples": len(y_true),
    }
    results["metrics"] = test_metrics
    
    logger.info(f"\n{'='*40}")
    logger.info("Test Set Results:")
    logger.info(f"  MSE: {test_metrics['regression']['mse']:.6f}")
    logger.info(f"  MAE: {test_metrics['regression']['mae']:.6f}")
    logger.info(f"  R²: {test_metrics['regression']['r2']:.4f}")
    logger.info(f"  Directional Accuracy: {test_metrics['directional']['accuracy']:.2%}")
    logger.info(f"{'='*40}")
    
    # Save outputs
    logger.info("\nSaving outputs...")
    
    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "feature_names": feature_names,
        "sequence_length": sequence_length,
    }, output_dir / "model.pt")
    
    # Save metrics
    save_metrics(results, output_dir / "results.json")
    
    # Generate plots
    plot_predictions(y_true, y_pred, output_dir / "predictions.png", n_points=1000)
    plot_training_curves(train_info["history"], output_dir / "training_curves.png")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LSTM model for MIDAS")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--target-type", type=str, default="return")
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--use-attention", action="store_true")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data/features")
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    config = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "use_attention": args.use_attention,
    }
    
    results = run_training_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        target_type=args.target_type,
        target_horizon=args.target_horizon,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        config=config,
    )
    
    return results


if __name__ == "__main__":
    main()
