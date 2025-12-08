"""
Temporal Fusion Transformer (TFT) model implementation.

Provides two options:
1. pytorch_forecasting TFT (preferred for production)
2. Compact PyTorch implementation (for minimal dependencies)
"""
import logging
from typing import Optional, Dict, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

try:
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    HAS_FORECASTING = True
except ImportError:
    HAS_FORECASTING = False
    TemporalFusionTransformer = None
    QuantileLoss = None

logger = logging.getLogger(__name__)


class QuantileRegressionLoss(nn.Module):
    """
    Quantile regression loss for probabilistic forecasting.
    """
    
    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_quantiles = len(self.quantiles)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, output_length, n_targets * n_quantiles)
            targets: (batch, output_length, n_targets)
        """
        batch_size, output_length, n_targets = targets.shape
        
        # Reshape predictions to (batch, output_length, n_targets, n_quantiles)
        predictions = predictions.view(batch_size, output_length, n_targets, self.n_quantiles)
        
        # Expand targets to match quantiles
        targets = targets.unsqueeze(-1).expand(-1, -1, -1, self.n_quantiles)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets[..., i] - predictions[..., i]
            losses.append(torch.max(q * errors, (q - 1) * errors))
        
        loss = torch.stack(losses, dim=-1).mean()
        return loss


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for TFT."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc(x)
        return output[..., :output.size(-1)//2] * torch.sigmoid(output[..., output.size(-1)//2:])


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - core building block of TFT.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Optional context conditioning
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim else None
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Skip connection
        residual = self.skip(x) if self.skip else x
        
        # First layer
        hidden = F.elu(self.fc1(x))
        
        # Add context if provided
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        
        # Second layer
        hidden = self.dropout(F.elu(self.fc2(hidden)))
        
        # Gating
        hidden = self.gate(hidden)
        
        # Residual + Layer Norm
        output = self.layer_norm(hidden + residual)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for TFT.
    Learns to select and weight input variables.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_variables: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim
        
        # Variable-wise GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
            for _ in range(n_variables)
        ])
        
        # Softmax weights GRN
        self.weight_grn = GatedResidualNetwork(
            n_variables * hidden_dim, hidden_dim, n_variables, dropout, context_dim
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_variables, input_dim)
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            weights: (batch, seq_len, n_variables)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Process each variable
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_x = x[..., i, :]  # (batch, seq_len, input_dim)
            var_outputs.append(grn(var_x, context))
        
        # Stack: (batch, seq_len, n_variables, hidden_dim)
        var_outputs = torch.stack(var_outputs, dim=-2)
        
        # Compute attention weights
        flat_vars = var_outputs.view(batch_size, seq_len, -1)
        weights = F.softmax(self.weight_grn(flat_vars, context), dim=-1)
        
        # Weighted sum
        output = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable weights for TFT.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        output = self.out_linear(context)
        
        # Return average attention for interpretability
        avg_attention = attention.mean(dim=1)
        
        return output, avg_attention


class TemporalFusionTransformerModel(nn.Module):
    """
    Compact Temporal Fusion Transformer implementation.
    
    Architecture:
    1. Variable Selection Network
    2. LSTM Encoder/Decoder
    3. Static Enrichment
    4. Temporal Self-Attention
    5. Position-wise Feed Forward
    6. Quantile Outputs
    """
    
    def __init__(
        self,
        n_features: int,
        n_targets: int = 1,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        input_length: int = 60,
        output_length: int = 10,
        quantiles: List[float] = None,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_dim = hidden_dim
        self.input_length = input_length
        self.output_length = output_length
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.n_quantiles = len(self.quantiles)
        
        # Input embedding
        self.input_embedding = nn.Linear(n_features, hidden_dim)
        
        # Variable Selection
        self.variable_selection = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            hidden_dim, hidden_dim, n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0
        )
        
        # LSTM Decoder
        self.lstm_decoder = nn.LSTM(
            hidden_dim, hidden_dim, n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0
        )
        
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Temporal self-attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim, n_heads, dropout
        )
        self.attention_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Position-wise feed forward
        self.ff_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim * 4, hidden_dim, dropout
        )
        
        # Output projection (quantile outputs)
        self.output_proj = nn.Linear(
            hidden_dim, n_targets * self.n_quantiles
        )
    
    def forward(
        self,
        x: torch.Tensor,
        future_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input features (batch, input_length, n_features)
            future_features: Known future features (batch, output_length, n_known_features)
            
        Returns:
            predictions: (batch, output_length, n_targets * n_quantiles)
            attention_weights: Dict with attention visualizations
        """
        batch_size = x.size(0)
        
        # Input embedding
        embedded = self.input_embedding(x)  # (batch, input_length, hidden_dim)
        
        # Variable selection
        selected = self.variable_selection(embedded)
        
        # LSTM Encoder
        encoder_output, (h_n, c_n) = self.lstm_encoder(selected)
        
        # Prepare decoder input (use last encoder state + zeros for future)
        decoder_input = torch.zeros(
            batch_size, self.output_length, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )
        
        # LSTM Decoder
        decoder_output, _ = self.lstm_decoder(decoder_input, (h_n, c_n))
        
        # Combine encoder and decoder outputs for attention
        combined = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Static enrichment
        enriched = self.static_enrichment(combined)
        
        # Self-attention (only on decoder outputs attending to all)
        query = enriched[:, self.input_length:, :]  # Decoder positions
        key = value = enriched  # All positions
        
        # Create causal mask for decoder
        attn_output, attention_weights = self.attention(query, key, value)
        
        # Gated skip connection
        attn_output = self.attention_norm(
            decoder_output + self.attention_gate(attn_output)
        )
        
        # Feed forward
        output = self.ff_grn(attn_output)
        
        # Quantile outputs
        predictions = self.output_proj(output)
        
        return predictions, {"attention": attention_weights}
    
    def predict_point(self, x: torch.Tensor) -> torch.Tensor:
        """Return median (50th percentile) predictions."""
        predictions, _ = self.forward(x)
        batch_size, output_length, _ = predictions.shape
        
        # Reshape to (batch, output_length, n_targets, n_quantiles)
        predictions = predictions.view(
            batch_size, output_length, self.n_targets, self.n_quantiles
        )
        
        # Return median (index 1 for [0.1, 0.5, 0.9])
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        return predictions[..., median_idx]


if HAS_LIGHTNING:
    class TFTLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for TFT training.
        """
        
        def __init__(
            self,
            n_features: int,
            n_targets: int = 1,
            hidden_dim: int = 64,
            n_heads: int = 4,
            n_lstm_layers: int = 2,
            dropout: float = 0.1,
            input_length: int = 60,
            output_length: int = 10,
            quantiles: List[float] = None,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
        ):
            super().__init__()
            self.save_hyperparameters()
            
            self.model = TemporalFusionTransformerModel(
                n_features=n_features,
                n_targets=n_targets,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_lstm_layers=n_lstm_layers,
                dropout=dropout,
                input_length=input_length,
                output_length=output_length,
                quantiles=quantiles,
            )
            
            self.loss_fn = QuantileRegressionLoss(quantiles or [0.1, 0.5, 0.9])
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            predictions, _ = self.model(x)
            return predictions
        
        def training_step(self, batch, batch_idx):
            x, y, _ = batch
            predictions, _ = self.model(x)
            loss = self.loss_fn(predictions, y)
            
            self.log("train_loss", loss, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y, _ = batch
            predictions, _ = self.model(x)
            loss = self.loss_fn(predictions, y)
            
            self.log("val_loss", loss, prog_bar=True)
            return loss
        
        def test_step(self, batch, batch_idx):
            x, y, _ = batch
            predictions, _ = self.model(x)
            loss = self.loss_fn(predictions, y)
            
            self.log("test_loss", loss)
            return loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }


def create_model(
    n_features: int,
    n_targets: int = 1,
    use_pytorch_forecasting: bool = True,
    **kwargs
):
    """
    Factory function to create TFT model.
    
    Args:
        n_features: Number of input features
        n_targets: Number of target variables
        use_pytorch_forecasting: Use pytorch_forecasting TFT if available
        **kwargs: Additional model arguments
        
    Returns:
        TFT model (Lightning module if available)
    """
    if use_pytorch_forecasting and HAS_FORECASTING:
        logger.info("Using pytorch_forecasting TFT")
        # Note: pytorch_forecasting requires specific data format
        # Return placeholder for now
        raise NotImplementedError(
            "pytorch_forecasting TFT requires TimeSeriesDataSet. "
            "Use use_pytorch_forecasting=False for custom implementation."
        )
    
    if HAS_LIGHTNING:
        logger.info("Using custom TFT with PyTorch Lightning")
        return TFTLightningModule(
            n_features=n_features,
            n_targets=n_targets,
            **kwargs
        )
    
    logger.info("Using custom TFT (no Lightning)")
    return TemporalFusionTransformerModel(
        n_features=n_features,
        n_targets=n_targets,
        **kwargs
    )
