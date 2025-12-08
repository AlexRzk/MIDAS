"""
PyTorch Dataset and DataModule for MIDAS feature data.

Handles:
- Loading Parquet feature files
- Feature selection and normalization
- Sliding window sequence creation for TFT
- Time-based train/val/test splits
"""
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pytorch_lightning as pl_lightning
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl_lightning = None

logger = logging.getLogger(__name__)


# Default feature columns for TFT
DEFAULT_FEATURES = [
    "midprice",
    "spread",
    "ofi",
    "imbalance_1",
    "imbalance_5",
    "microprice",
    "taker_buy_volume",
    "taker_sell_volume",
    "liquidity_1",
    "liquidity_5",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
]

# Target columns for prediction
DEFAULT_TARGETS = ["close"]  # Predict future close price


class FeatureNormalizer:
    """
    Normalizes features using train set statistics.
    Saves/loads normalization params as JSON.
    """
    
    def __init__(self):
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.fitted = False
    
    def fit(self, df: pl.DataFrame, columns: List[str]) -> "FeatureNormalizer":
        """Compute mean and std from training data."""
        for col in columns:
            if col in df.columns:
                col_data = df[col].drop_nulls()
                self.mean[col] = float(col_data.mean()) if len(col_data) > 0 else 0.0
                self.std[col] = float(col_data.std()) if len(col_data) > 0 else 1.0
                # Prevent division by zero
                if self.std[col] == 0 or self.std[col] is None:
                    self.std[col] = 1.0
        
        self.fitted = True
        return self
    
    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Apply normalization to DataFrame."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        exprs = []
        for col in columns:
            if col in df.columns and col in self.mean:
                exprs.append(
                    ((pl.col(col) - self.mean[col]) / self.std[col]).alias(col)
                )
        
        if exprs:
            df = df.with_columns(exprs)
        
        return df
    
    def inverse_transform(self, values: np.ndarray, column: str) -> np.ndarray:
        """Reverse normalization for a single column."""
        if column not in self.mean:
            return values
        return values * self.std[column] + self.mean[column]
    
    def save(self, filepath: Path):
        """Save normalization parameters to JSON."""
        data = {
            "mean": self.mean,
            "std": self.std,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved normalizer to {filepath}")
    
    def load(self, filepath: Path) -> "FeatureNormalizer":
        """Load normalization parameters from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        self.mean = data["mean"]
        self.std = data["std"]
        self.fitted = True
        logger.info(f"Loaded normalizer from {filepath}")
        return self


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with sliding windows.
    
    Creates sequences of (input_length, output_length) for TFT training.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        timestamps: np.ndarray,
        input_length: int = 60,
        output_length: int = 10,
        stride: int = 1,
    ):
        """
        Args:
            data: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples, n_targets)
            timestamps: Timestamp array for reference
            input_length: Number of historical timesteps for input
            output_length: Number of future timesteps to predict
            stride: Step size between sequences
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.timestamps = timestamps
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        
        # Calculate valid indices
        total_length = input_length + output_length
        self.n_sequences = max(0, (len(data) - total_length) // stride + 1)
    
    def __len__(self) -> int:
        return self.n_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input sequence (input_length, n_features)
            y: Target sequence (output_length, n_targets)
            ts: Timestamps for the sequence
        """
        start_idx = idx * self.stride
        input_end = start_idx + self.input_length
        output_end = input_end + self.output_length
        
        x = self.data[start_idx:input_end]
        y = self.targets[input_end:output_end]
        ts = torch.LongTensor(self.timestamps[start_idx:output_end])
        
        return x, y, ts


class MIDASDataModule:
    """
    Data module for loading and preparing MIDAS feature data.
    
    Handles:
    - Loading Parquet files
    - Feature selection
    - Normalization
    - Time-based splits
    - DataLoader creation
    """
    
    def __init__(
        self,
        data_dir: Path,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        input_length: int = 60,
        output_length: int = 10,
        batch_size: int = 64,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stride: int = 1,
        num_workers: int = 4,
    ):
        """
        Args:
            data_dir: Directory containing feature Parquet files
            feature_columns: List of feature column names (default: DEFAULT_FEATURES)
            target_columns: List of target column names (default: DEFAULT_TARGETS)
            input_length: Number of historical timesteps
            output_length: Number of future timesteps to predict
            batch_size: Batch size for DataLoaders
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            stride: Step size between sequences
            num_workers: Number of DataLoader workers
        """
        self.data_dir = Path(data_dir)
        self.feature_columns = feature_columns or DEFAULT_FEATURES
        self.target_columns = target_columns or DEFAULT_TARGETS
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.stride = stride
        self.num_workers = num_workers
        
        self.normalizer = FeatureNormalizer()
        self.train_dataset: Optional[TimeSeriesDataset] = None
        self.val_dataset: Optional[TimeSeriesDataset] = None
        self.test_dataset: Optional[TimeSeriesDataset] = None
        
        self._df: Optional[pl.DataFrame] = None
    
    def load_data(self) -> pl.DataFrame:
        """Load all feature files and concatenate."""
        files = sorted(self.data_dir.glob("features_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No feature files found in {self.data_dir}")
        
        logger.info(f"Loading {len(files)} feature files from {self.data_dir}")
        
        dfs = [pl.read_parquet(f) for f in files]
        df = pl.concat(dfs).sort("ts")
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def setup(self, stage: Optional[str] = None):
        """
        Prepare datasets for training/validation/testing.
        
        Args:
            stage: 'fit', 'test', or None (all)
        """
        if self._df is None:
            self._df = self.load_data()
        
        df = self._df
        
        # Filter to available columns
        available_features = [c for c in self.feature_columns if c in df.columns]
        available_targets = [c for c in self.target_columns if c in df.columns]
        
        if not available_features:
            raise ValueError(f"No feature columns found. Available: {df.columns}")
        if not available_targets:
            raise ValueError(f"No target columns found. Available: {df.columns}")
        
        logger.info(f"Using features: {available_features}")
        logger.info(f"Using targets: {available_targets}")
        
        # Time-based split
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        df_train = df[:train_end]
        df_val = df[train_end:val_end]
        df_test = df[val_end:]
        
        logger.info(f"Split sizes - Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,}")
        
        # Fit normalizer on training data
        all_columns = available_features + available_targets
        self.normalizer.fit(df_train, all_columns)
        
        # Normalize all splits
        df_train = self.normalizer.transform(df_train, all_columns)
        df_val = self.normalizer.transform(df_val, all_columns)
        df_test = self.normalizer.transform(df_test, all_columns)
        
        # Convert to numpy arrays
        def to_arrays(df: pl.DataFrame):
            # Handle nulls
            df = df.fill_null(0.0)
            
            features = df.select(available_features).to_numpy()
            targets = df.select(available_targets).to_numpy()
            timestamps = df["ts"].to_numpy() if "ts" in df.columns else np.arange(len(df))
            
            return features, targets, timestamps
        
        train_features, train_targets, train_ts = to_arrays(df_train)
        val_features, val_targets, val_ts = to_arrays(df_val)
        test_features, test_targets, test_ts = to_arrays(df_test)
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = TimeSeriesDataset(
                train_features, train_targets, train_ts,
                self.input_length, self.output_length, self.stride
            )
            self.val_dataset = TimeSeriesDataset(
                val_features, val_targets, val_ts,
                self.input_length, self.output_length, self.stride
            )
            logger.info(f"Train sequences: {len(self.train_dataset):,}")
            logger.info(f"Val sequences: {len(self.val_dataset):,}")
        
        if stage == "test" or stage is None:
            self.test_dataset = TimeSeriesDataset(
                test_features, test_targets, test_ts,
                self.input_length, self.output_length, self.stride
            )
            logger.info(f"Test sequences: {len(self.test_dataset):,}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def save_normalizer(self, filepath: Path):
        """Save normalizer parameters."""
        self.normalizer.save(filepath)
    
    def load_normalizer(self, filepath: Path):
        """Load normalizer parameters."""
        self.normalizer.load(filepath)
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return len([c for c in self.feature_columns if self._df is not None and c in self._df.columns])
    
    @property
    def n_targets(self) -> int:
        """Number of target columns."""
        return len([c for c in self.target_columns if self._df is not None and c in self._df.columns])


# PyTorch Lightning DataModule wrapper (if available)
if HAS_LIGHTNING:
    class MIDASLightningDataModule(pl_lightning.LightningDataModule):
        """PyTorch Lightning wrapper for MIDASDataModule."""
        
        def __init__(self, **kwargs):
            super().__init__()
            self._module = MIDASDataModule(**kwargs)
        
        def setup(self, stage: Optional[str] = None):
            self._module.setup(stage)
        
        def train_dataloader(self):
            return self._module.train_dataloader()
        
        def val_dataloader(self):
            return self._module.val_dataloader()
        
        def test_dataloader(self):
            return self._module.test_dataloader()
        
        @property
        def normalizer(self):
            return self._module.normalizer
        
        @property
        def n_features(self):
            return self._module.n_features
        
        @property
        def n_targets(self):
            return self._module.n_targets
