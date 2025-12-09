#!/usr/bin/env python3
"""
Data preprocessing for MIDAS GPU training.

Handles:
- Gap detection and forward-filling
- Normalization with training statistics only (prevent leakage)
- NaN handling
- Feature engineering preparation
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import json

from utils import get_logger

logger = get_logger("preprocessing")


@dataclass
class NormalizationStats:
    """Statistics for feature normalization."""
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    min: Dict[str, float] = field(default_factory=dict)
    max: Dict[str, float] = field(default_factory=dict)
    
    def save(self, path: Path):
        """Save stats to JSON."""
        data = {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved normalization stats to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Load stats from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def check_sorted(df: pl.DataFrame, ts_col: str = "ts") -> bool:
    """Check if DataFrame is sorted by timestamp."""
    ts = df[ts_col].to_numpy()
    is_sorted = np.all(ts[:-1] <= ts[1:])
    if not is_sorted:
        logger.warning("Data is not sorted by timestamp!")
    return is_sorted


def sort_by_timestamp(df: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
    """Sort DataFrame by timestamp."""
    return df.sort(ts_col)


def count_nulls(df: pl.DataFrame) -> Dict[str, int]:
    """Count null values per column."""
    null_counts = {}
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            null_counts[col] = null_count
    return null_counts


def forward_fill_nulls(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Forward-fill null values in specified columns."""
    if columns is None:
        columns = [c for c in df.columns if c not in ["ts"]]
    
    expressions = []
    for col in columns:
        if col in df.columns:
            expressions.append(pl.col(col).forward_fill().alias(col))
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def backward_fill_nulls(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Backward-fill null values (for start-of-segment nulls)."""
    if columns is None:
        columns = [c for c in df.columns if c not in ["ts"]]
    
    expressions = []
    for col in columns:
        if col in df.columns:
            expressions.append(pl.col(col).backward_fill().alias(col))
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def fill_remaining_nulls(
    df: pl.DataFrame,
    fill_value: float = 0.0,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Fill any remaining nulls with a constant value."""
    if columns is None:
        columns = [c for c in df.columns if c not in ["ts"]]
    
    expressions = []
    for col in columns:
        if col in df.columns:
            expressions.append(pl.col(col).fill_null(fill_value).alias(col))
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def drop_rows_with_nulls(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Drop rows that have null values in specified columns."""
    if columns is None:
        return df.drop_nulls()
    return df.drop_nulls(subset=columns)


def compute_normalization_stats(
    df: pl.DataFrame,
    columns: List[str],
) -> NormalizationStats:
    """
    Compute normalization statistics from training data only.
    
    CRITICAL: Only call this on training data to prevent leakage!
    """
    stats = NormalizationStats()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = df[col]
        stats.mean[col] = float(series.mean())
        stats.std[col] = float(series.std())
        stats.min[col] = float(series.min())
        stats.max[col] = float(series.max())
        
        # Handle zero std
        if stats.std[col] == 0 or np.isnan(stats.std[col]):
            stats.std[col] = 1.0
            logger.warning(f"Column {col} has zero std, using 1.0")
    
    logger.info(f"Computed normalization stats for {len(columns)} columns")
    return stats


def normalize_zscore(
    df: pl.DataFrame,
    stats: NormalizationStats,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Apply z-score normalization: (x - mean) / std
    
    Uses pre-computed statistics to prevent leakage.
    """
    if columns is None:
        columns = list(stats.mean.keys())
    
    expressions = []
    for col in columns:
        if col in df.columns and col in stats.mean:
            mean = stats.mean[col]
            std = stats.std[col]
            expressions.append(
                ((pl.col(col) - mean) / std).alias(col)
            )
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def normalize_minmax(
    df: pl.DataFrame,
    stats: NormalizationStats,
    columns: Optional[List[str]] = None,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> pl.DataFrame:
    """
    Apply min-max normalization: (x - min) / (max - min) * (target_max - target_min) + target_min
    """
    if columns is None:
        columns = list(stats.min.keys())
    
    expressions = []
    for col in columns:
        if col in df.columns and col in stats.min:
            min_val = stats.min[col]
            max_val = stats.max[col]
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            
            scale = target_max - target_min
            expressions.append(
                ((pl.col(col) - min_val) / range_val * scale + target_min).alias(col)
            )
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def clip_outliers(
    df: pl.DataFrame,
    columns: List[str],
    n_std: float = 5.0,
    stats: Optional[NormalizationStats] = None,
) -> pl.DataFrame:
    """
    Clip outliers to +/- n_std standard deviations.
    
    If stats are provided, uses pre-computed mean/std.
    Otherwise computes from data (only use on training data).
    """
    expressions = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if stats and col in stats.mean:
            mean = stats.mean[col]
            std = stats.std[col]
        else:
            mean = df[col].mean()
            std = df[col].std()
        
        lower = mean - n_std * std
        upper = mean + n_std * std
        
        expressions.append(
            pl.col(col).clip(lower, upper).alias(col)
        )
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


def remove_infinite_values(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    replace_with: float = 0.0,
) -> pl.DataFrame:
    """Replace infinite values with a constant."""
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64]]
    
    expressions = []
    for col in columns:
        if col in df.columns:
            expressions.append(
                pl.when(pl.col(col).is_infinite())
                .then(replace_with)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    if expressions:
        df = df.with_columns(expressions)
    
    return df


class Preprocessor:
    """
    Complete preprocessing pipeline for MIDAS data.
    
    Usage:
        # Fit on training data
        prep = Preprocessor()
        train_df = prep.fit_transform(train_df, feature_cols)
        
        # Transform test data with same statistics
        test_df = prep.transform(test_df)
    """
    
    def __init__(
        self,
        normalize: str = "zscore",  # "zscore", "minmax", or None
        clip_outliers_std: Optional[float] = 5.0,
        fill_strategy: str = "forward",  # "forward", "zero", "drop"
    ):
        self.normalize = normalize
        self.clip_outliers_std = clip_outliers_std
        self.fill_strategy = fill_strategy
        
        self.stats: Optional[NormalizationStats] = None
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False
    
    def fit(self, df: pl.DataFrame, feature_columns: List[str]) -> "Preprocessor":
        """
        Fit preprocessor on training data.
        
        CRITICAL: Only call this on training data!
        """
        self.feature_columns = feature_columns
        
        # Compute statistics from training data
        self.stats = compute_normalization_stats(df, feature_columns)
        self.is_fitted = True
        
        logger.info(f"Preprocessor fitted on {len(feature_columns)} features")
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Transform data using fitted statistics.
        
        Safe to use on both training and test data.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # 1. Sort by timestamp
        if "ts" in df.columns and not check_sorted(df, "ts"):
            df = sort_by_timestamp(df, "ts")
        
        # 2. Handle null values
        if self.fill_strategy == "forward":
            df = forward_fill_nulls(df, self.feature_columns)
            df = backward_fill_nulls(df, self.feature_columns)  # For leading nulls
            df = fill_remaining_nulls(df, 0.0, self.feature_columns)
        elif self.fill_strategy == "zero":
            df = fill_remaining_nulls(df, 0.0, self.feature_columns)
        elif self.fill_strategy == "drop":
            df = drop_rows_with_nulls(df, self.feature_columns)
        
        # 3. Handle infinite values
        df = remove_infinite_values(df, self.feature_columns)
        
        # 4. Clip outliers
        if self.clip_outliers_std is not None:
            df = clip_outliers(df, self.feature_columns, self.clip_outliers_std, self.stats)
        
        # 5. Normalize
        if self.normalize == "zscore":
            df = normalize_zscore(df, self.stats, self.feature_columns)
        elif self.normalize == "minmax":
            df = normalize_minmax(df, self.stats, self.feature_columns)
        
        return df
    
    def fit_transform(
        self,
        df: pl.DataFrame,
        feature_columns: List[str],
    ) -> pl.DataFrame:
        """Fit on data and transform it."""
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def save(self, path: Path):
        """Save preprocessor state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save stats
        self.stats.save(path / "normalization_stats.json")
        
        # Save config
        config = {
            "normalize": self.normalize,
            "clip_outliers_std": self.clip_outliers_std,
            "fill_strategy": self.fill_strategy,
            "feature_columns": self.feature_columns,
        }
        with open(path / "preprocessor_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved preprocessor to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """Load preprocessor state."""
        path = Path(path)
        
        # Load config
        with open(path / "preprocessor_config.json", "r") as f:
            config = json.load(f)
        
        prep = cls(
            normalize=config["normalize"],
            clip_outliers_std=config["clip_outliers_std"],
            fill_strategy=config["fill_strategy"],
        )
        prep.feature_columns = config["feature_columns"]
        prep.stats = NormalizationStats.load(path / "normalization_stats.json")
        prep.is_fitted = True
        
        return prep


def prepare_for_lstm(
    X: np.ndarray,
    sequence_length: int = 50,
    stride: int = 1,
) -> np.ndarray:
    """
    Reshape data for LSTM input.
    
    Args:
        X: Input array of shape (n_samples, n_features)
        sequence_length: Number of timesteps per sequence
        stride: Step size between sequences
        
    Returns:
        Array of shape (n_sequences, sequence_length, n_features)
    """
    n_samples, n_features = X.shape
    n_sequences = (n_samples - sequence_length) // stride + 1
    
    sequences = np.zeros((n_sequences, sequence_length, n_features))
    
    for i in range(n_sequences):
        start = i * stride
        end = start + sequence_length
        sequences[i] = X[start:end]
    
    return sequences


def prepare_sequences_with_target(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 50,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM with corresponding targets.
    
    Target corresponds to the LAST timestep in each sequence.
    """
    X_seq = prepare_for_lstm(X, sequence_length, stride)
    
    # Target is at the end of each sequence
    n_sequences = len(X_seq)
    y_seq = np.zeros(n_sequences)
    
    for i in range(n_sequences):
        target_idx = i * stride + sequence_length - 1
        y_seq[i] = y[target_idx]
    
    return X_seq, y_seq


if __name__ == "__main__":
    # Test preprocessing
    import numpy as np
    
    # Create test data
    n_samples = 1000
    df = pl.DataFrame({
        "ts": np.arange(n_samples) * 100_000,
        "feature1": np.random.randn(n_samples) * 10 + 50,
        "feature2": np.random.randn(n_samples) * 5 + 20,
        "target": np.random.randn(n_samples),
    })
    
    # Add some nulls
    df = df.with_columns([
        pl.when(pl.col("feature1") > 55).then(None).otherwise(pl.col("feature1")).alias("feature1")
    ])
    
    feature_cols = ["feature1", "feature2"]
    
    # Fit and transform
    prep = Preprocessor(normalize="zscore", clip_outliers_std=3.0)
    df_processed = prep.fit_transform(df, feature_cols)
    
    print("Original stats:")
    print(f"  feature1 mean: {df['feature1'].mean():.2f}, std: {df['feature1'].std():.2f}")
    
    print("\nProcessed stats:")
    print(f"  feature1 mean: {df_processed['feature1'].mean():.2f}, std: {df_processed['feature1'].std():.2f}")
    print(f"  Nulls remaining: {df_processed['feature1'].null_count()}")
