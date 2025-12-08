#!/usr/bin/env python3
"""
TFT Training CLI for MIDAS.

Usage:
    python training/train.py --data-dir data/features/ --epochs 100
    python training/train.py --data-dir data/features/ --gpus 1 --batch-size 128
    
Environment variables:
    CUDA_VISIBLE_DEVICES: GPU device IDs to use
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import MIDASDataModule, DEFAULT_FEATURES, DEFAULT_TARGETS
from training.model import create_model, TFTLightningModule

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from pytorch_lightning.loggers import TensorBoardLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    print("WARNING: pytorch_lightning not installed. Training will use basic PyTorch.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_lightning(
    data_module: MIDASDataModule,
    model: TFTLightningModule,
    args: argparse.Namespace,
):
    """Train using PyTorch Lightning."""
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.model_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="tft",
        version=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=callbacks,
        logger=tb_logger,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad,
        precision=args.precision,
        deterministic=args.deterministic,
        enable_progress_bar=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test
    if args.test:
        logger.info("Running test evaluation...")
        trainer.test(model, data_module)
    
    # Save final model
    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model saved to: {best_model_path}")
    
    # Export to TorchScript if requested
    if args.export_torchscript:
        export_path = Path(args.model_dir) / "tft_scripted.pt"
        try:
            scripted = torch.jit.script(model.model)
            scripted.save(str(export_path))
            logger.info(f"Exported TorchScript to: {export_path}")
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")
    
    return trainer


def train_basic(
    data_module: MIDASDataModule,
    model: torch.nn.Module,
    args: argparse.Namespace,
):
    """Basic PyTorch training loop (no Lightning)."""
    from training.model import QuantileRegressionLoss
    
    device = torch.device("cuda" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = QuantileRegressionLoss()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                predictions, _ = model(x)
                loss = loss_fn(predictions, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = Path(args.model_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TFT model on MIDAS features")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing feature Parquet files",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=60,
        help="Number of historical timesteps for input",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=10,
        help="Number of future timesteps to predict",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation",
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for TFT",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n-lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 to disable)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--accumulate-grad",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "32", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    
    # Output arguments
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/tensorboard"),
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test evaluation after training",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Export model to TorchScript",
    )
    
    args = parser.parse_args()
    
    # Auto-detect GPUs
    if args.gpus is None:
        args.gpus = torch.cuda.device_count()
    
    logger.info(f"Using {args.gpus} GPU(s)")
    
    # Create output directories
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = MIDASDataModule(
        data_dir=args.data_dir,
        feature_columns=DEFAULT_FEATURES,
        target_columns=DEFAULT_TARGETS,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )
    data_module.setup()
    
    # Save normalizer
    normalizer_path = args.model_dir / "normalizer.json"
    data_module.save_normalizer(normalizer_path)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        n_features=data_module.n_features,
        n_targets=data_module.n_targets,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_lstm_layers=args.n_lstm_layers,
        dropout=args.dropout,
        input_length=args.input_length,
        output_length=args.output_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_pytorch_forecasting=False,
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    if HAS_LIGHTNING and isinstance(model, TFTLightningModule):
        # Convert to Lightning DataModule
        from training.dataset import MIDASLightningDataModule
        lightning_dm = MIDASLightningDataModule(
            data_dir=args.data_dir,
            feature_columns=DEFAULT_FEATURES,
            target_columns=DEFAULT_TARGETS,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
        )
        lightning_dm.setup()
        train_with_lightning(lightning_dm, model, args)
    else:
        train_basic(data_module, model, args)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
