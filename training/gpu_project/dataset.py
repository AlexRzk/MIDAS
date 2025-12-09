#!/usr/bin/env python3
"""
Dataset loading and management for MIDAS GPU training.
Handles parquet feature files with OFI and microstructure data.
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from utils import get_logger, DATA_DIR

logger = get_logger("dataset")


# ============================================
# Feature Configuration
# ============================================

# Core orderbook features
ORDERBOOK_FEATURES = [
    "bid_px_01", "bid_sz_01", "ask_px_01", "ask_sz_01",
    "bid_px_02", "bid_sz_02", "ask_px_02", "ask_sz_02",
    "bid_px_03", "bid_sz_03", "ask_px_03", "ask_sz_03",
    "bid_px_04", "bid_sz_04", "ask_px_04", "ask_sz_04",
    "bid_px_05", "bid_sz_05", "ask_px_05", "ask_sz_05",
]

# Microstructure features (including OFI)
MICROSTRUCTURE_FEATURES = [
    "midprice", "spread", "spread_bps",
    "microprice",
    "imbalance", "imbalance_1", "imbalance_5", "imbalance_10",
    "ofi", "ofi_10", "ofi_cumulative",
    "taker_buy_volume", "taker_sell_volume",
    "signed_volume", "volume_imbalance",
    "volatility_20", "volatility_100",
    "kyle_lambda", "vpin",
]

# Advanced features
ADVANCED_FEATURES = [
    "bid_ladder_slope", "ask_ladder_slope",
    "bid_slope_ratio", "ask_slope_ratio",
    "queue_imb_1", "queue_imb_2", "queue_imb_3", "queue_imb_4", "queue_imb_5",
    "vol_of_vol",
]

# All trainable features (excluding timestamp and target)
ALL_FEATURES = ORDERBOOK_FEATURES + MICROSTRUCTURE_FEATURES + ADVANCED_FEATURES


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    total_rows: int
    total_files: int
    time_range_hours: float
    start_ts: int
    end_ts: int
    bucket_size_us: int
    available_features: List[str]
    missing_features: List[str]


def discover_features(data_dir: Path = DATA_DIR) -> List[Path]:
    """Discover all feature parquet files."""
    # Try normalized files first, then fall back to regular features
    patterns = ["normalized_features_*.parquet", "features_*.parquet"]
    
    files = []
    for pattern in patterns:
        files = sorted(data_dir.glob(pattern))
        if files:
            break
        # Try nested directory
        files = sorted(data_dir.glob(f"**/{pattern}"))
        if files:
            break
    
    logger.info(f"Discovered {len(files)} feature files in {data_dir}")
    return files


def load_single_file(filepath: Path) -> pl.DataFrame:
    """Load a single parquet file."""
    df = pl.read_parquet(filepath)
    logger.info(f"Loaded {filepath.name}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_all_features(
    data_dir: Path = DATA_DIR,
    files: Optional[List[Path]] = None,
) -> pl.DataFrame:
    """Load and concatenate all feature files."""
    if files is None:
        files = discover_features(data_dir)
    
    if not files:
        raise ValueError(f"No feature files found in {data_dir}")
    
    # Load all files
    dfs = []
    for f in files:
        df = load_single_file(f)
        dfs.append(df)
    
    # Concatenate
    combined = pl.concat(dfs, how="vertical")
    
    # Sort by timestamp
    if "ts" in combined.columns:
        combined = combined.sort("ts")
    
    logger.info(f"Combined dataset: {len(combined):,} rows")
    return combined


def get_dataset_info(df: pl.DataFrame) -> DatasetInfo:
    """Get dataset metadata."""
    ts_col = "ts" if "ts" in df.columns else df.columns[0]
    
    start_ts = df[ts_col].min()
    end_ts = df[ts_col].max()
    time_range_us = end_ts - start_ts
    time_range_hours = time_range_us / (1_000_000 * 3600)
    
    # Estimate bucket size from median time difference
    if len(df) > 1:
        ts_diff = df[ts_col].diff().drop_nulls()
        bucket_size_us = int(ts_diff.median())
    else:
        bucket_size_us = 100_000  # Default 100ms
    
    # Check available features
    available = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    
    return DatasetInfo(
        total_rows=len(df),
        total_files=1,  # Updated by caller if needed
        time_range_hours=time_range_hours,
        start_ts=start_ts,
        end_ts=end_ts,
        bucket_size_us=bucket_size_us,
        available_features=available,
        missing_features=missing,
    )


def select_features(
    df: pl.DataFrame,
    feature_list: Optional[List[str]] = None,
    include_target: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Select features for training.
    
    Returns:
        df: DataFrame with selected columns
        feature_names: List of feature column names (excluding ts and target)
    """
    if feature_list is None:
        # Use all available features
        feature_list = [f for f in ALL_FEATURES if f in df.columns]
    else:
        # Filter to available
        feature_list = [f for f in feature_list if f in df.columns]
    
    # Always include timestamp
    columns = ["ts"] + feature_list
    
    # Add target if present and requested
    if include_target and "target" in df.columns:
        columns.append("target")
    
    df = df.select([c for c in columns if c in df.columns])
    
    # Feature names exclude ts and target
    feature_names = [c for c in df.columns if c not in ["ts", "target"]]
    
    return df, feature_names


def create_target(
    df: pl.DataFrame,
    target_type: str = "return",  # "return", "direction", "price_delta"
    horizon: int = 10,  # Number of ticks to look ahead
    threshold_bps: float = 0.5,  # For direction classification
) -> pl.DataFrame:
    """
    Create prediction target.
    
    Args:
        df: Input DataFrame with midprice
        target_type: Type of target to create
        horizon: Number of ticks to look ahead
        threshold_bps: Threshold for direction classification
        
    Returns:
        DataFrame with target column added
    """
    price_col = "midprice" if "midprice" in df.columns else "bid_px_01"
    
    if target_type == "return":
        # Future return in basis points
        df = df.with_columns([
            ((pl.col(price_col).shift(-horizon) - pl.col(price_col)) / 
             pl.col(price_col) * 10000).alias("target")
        ])
    
    elif target_type == "price_delta":
        # Raw price difference
        df = df.with_columns([
            (pl.col(price_col).shift(-horizon) - pl.col(price_col)).alias("target")
        ])
    
    elif target_type == "direction":
        # Classification: 1 = up, 0 = down/flat
        df = df.with_columns([
            ((pl.col(price_col).shift(-horizon) - pl.col(price_col)) / 
             pl.col(price_col) * 10000).alias("_return_bps")
        ])
        df = df.with_columns([
            (pl.col("_return_bps") > threshold_bps).cast(pl.Int8).alias("target")
        ])
        df = df.drop("_return_bps")
    
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    logger.info(f"Created target: {target_type}, horizon={horizon}")
    return df


def add_lag_features(
    df: pl.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 5, 10, 20],
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add lagged versions of specified columns.
    
    Returns:
        df: DataFrame with lag features added
        new_columns: List of new column names
    """
    new_columns = []
    expressions = []
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            new_col = f"{col}_lag_{lag}"
            expressions.append(pl.col(col).shift(lag).alias(new_col))
            new_columns.append(new_col)
    
    if expressions:
        df = df.with_columns(expressions)
    
    logger.info(f"Added {len(new_columns)} lag features")
    return df, new_columns


def add_rolling_features(
    df: pl.DataFrame,
    columns: List[str],
    windows: List[int] = [10, 30, 60, 120],
    operations: List[str] = ["mean", "std"],
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add rolling window features.
    
    Returns:
        df: DataFrame with rolling features added
        new_columns: List of new column names
    """
    new_columns = []
    expressions = []
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            for op in operations:
                new_col = f"{col}_{op}_{window}"
                if op == "mean":
                    expressions.append(pl.col(col).rolling_mean(window).alias(new_col))
                elif op == "std":
                    expressions.append(pl.col(col).rolling_std(window).alias(new_col))
                elif op == "sum":
                    expressions.append(pl.col(col).rolling_sum(window).alias(new_col))
                elif op == "min":
                    expressions.append(pl.col(col).rolling_min(window).alias(new_col))
                elif op == "max":
                    expressions.append(pl.col(col).rolling_max(window).alias(new_col))
                new_columns.append(new_col)
    
    if expressions:
        df = df.with_columns(expressions)
    
    logger.info(f"Added {len(new_columns)} rolling features")
    return df, new_columns


class MIDASDataset:
    """
    Main dataset class for MIDAS training.
    Handles loading, preprocessing, and feature engineering.
    """
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.df: Optional[pl.DataFrame] = None
        self.info: Optional[DatasetInfo] = None
        self.feature_names: List[str] = []
        
    def load(self, files: Optional[List[Path]] = None) -> "MIDASDataset":
        """Load data from parquet files."""
        self.df = load_all_features(self.data_dir, files)
        self.info = get_dataset_info(self.df)
        return self
    
    def create_target(
        self,
        target_type: str = "return",
        horizon: int = 10,
        threshold_bps: float = 0.5,
    ) -> "MIDASDataset":
        """Add prediction target."""
        self.df = create_target(self.df, target_type, horizon, threshold_bps)
        return self
    
    def add_features(
        self,
        lag_columns: Optional[List[str]] = None,
        lag_values: List[int] = [1, 2, 5, 10],
        rolling_columns: Optional[List[str]] = None,
        rolling_windows: List[int] = [10, 30, 60],
    ) -> "MIDASDataset":
        """Add lag and rolling features."""
        if lag_columns:
            self.df, new_cols = add_lag_features(self.df, lag_columns, lag_values)
        
        if rolling_columns:
            self.df, new_cols = add_rolling_features(
                self.df, rolling_columns, rolling_windows, ["mean", "std"]
            )
        
        return self
    
    def select_features(
        self,
        feature_list: Optional[List[str]] = None,
    ) -> "MIDASDataset":
        """Select features for training."""
        self.df, self.feature_names = select_features(self.df, feature_list)
        return self
    
    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data as numpy arrays.
        
        Returns:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
            ts: Timestamp array (n_samples,)
        """
        feature_cols = [c for c in self.df.columns if c not in ["ts", "target"]]
        
        X = self.df.select(feature_cols).to_numpy()
        y = self.df["target"].to_numpy() if "target" in self.df.columns else np.zeros(len(X))
        ts = self.df["ts"].to_numpy() if "ts" in self.df.columns else np.arange(len(X))
        
        return X, y, ts
    
    def summary(self) -> str:
        """Get dataset summary string."""
        if self.info is None:
            return "Dataset not loaded"
        
        return f"""
Dataset Summary:
  Total rows: {self.info.total_rows:,}
  Time range: {self.info.time_range_hours:.2f} hours
  Bucket size: {self.info.bucket_size_us / 1000:.1f} ms
  Available features: {len(self.info.available_features)}
  Missing features: {len(self.info.missing_features)}
  Feature columns: {len(self.feature_names)}
"""


if __name__ == "__main__":
    # Test dataset loading
    dataset = MIDASDataset()
    
    try:
        dataset.load()
        print(dataset.summary())
        print(f"\nAvailable features: {dataset.info.available_features[:10]}...")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure data is uploaded to {DATA_DIR}")
