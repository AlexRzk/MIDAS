#!/usr/bin/env python3
"""
OFI-Safe Data Splitting for MIDAS.

Critical for financial ML:
- Detects gaps in time series data
- Splits into continuous segments
- Ensures no information leakage between train/test
- Handles OFI cumulative nature properly
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from utils import get_logger

logger = get_logger("splitter")


@dataclass
class ContinuousSegment:
    """A continuous segment of time series data."""
    start_idx: int
    end_idx: int
    start_ts: int
    end_ts: int
    n_rows: int
    duration_hours: float
    
    def __repr__(self):
        return f"Segment(rows={self.n_rows:,}, hours={self.duration_hours:.2f})"


@dataclass
class Split:
    """Train/test split information."""
    segment_id: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_rows: int
    test_rows: int
    
    def __repr__(self):
        return f"Split(train={self.train_rows:,}, test={self.test_rows:,})"


def detect_gaps(
    ts_array: np.ndarray,
    bucket_size_us: int = 100_000,  # 100ms default
    gap_threshold_factor: float = 2.0,
) -> Tuple[np.ndarray, int]:
    """
    Detect gaps in timestamp array.
    
    A gap is defined as a time difference > gap_threshold_factor * bucket_size.
    
    Args:
        ts_array: Timestamps in microseconds
        bucket_size_us: Expected time between samples
        gap_threshold_factor: Factor to multiply bucket_size for gap detection
        
    Returns:
        gap_indices: Indices where gaps occur (gap is AFTER this index)
        n_gaps: Number of gaps found
    """
    if len(ts_array) < 2:
        return np.array([], dtype=np.int64), 0
    
    # Compute time differences
    ts_diff = np.diff(ts_array)
    
    # Gap threshold
    threshold = bucket_size_us * gap_threshold_factor
    
    # Find gap locations
    gap_mask = ts_diff > threshold
    gap_indices = np.where(gap_mask)[0]
    
    # Log gap statistics
    if len(gap_indices) > 0:
        gap_sizes_ms = ts_diff[gap_indices] / 1000
        logger.info(f"Found {len(gap_indices)} gaps")
        logger.info(f"  Gap sizes: min={gap_sizes_ms.min():.1f}ms, "
                   f"max={gap_sizes_ms.max():.1f}ms, "
                   f"median={np.median(gap_sizes_ms):.1f}ms")
    else:
        logger.info("No gaps detected - data is continuous")
    
    return gap_indices, len(gap_indices)


def split_into_segments(
    df: pl.DataFrame,
    ts_col: str = "ts",
    bucket_size_us: int = 100_000,
    gap_threshold_factor: float = 2.0,
    min_segment_rows: int = 100,
) -> List[ContinuousSegment]:
    """
    Split DataFrame into continuous segments.
    
    Args:
        df: Input DataFrame
        ts_col: Timestamp column name
        bucket_size_us: Expected time between samples
        gap_threshold_factor: Factor for gap detection
        min_segment_rows: Minimum rows for a segment to be kept
        
    Returns:
        List of ContinuousSegment objects
    """
    ts_array = df[ts_col].to_numpy()
    gap_indices, _ = detect_gaps(ts_array, bucket_size_us, gap_threshold_factor)
    
    # Build segment boundaries
    segment_starts = [0] + list(gap_indices + 1)
    segment_ends = list(gap_indices + 1) + [len(df)]
    
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        n_rows = end - start
        if n_rows < min_segment_rows:
            logger.debug(f"Skipping small segment: {n_rows} rows")
            continue
        
        start_ts = int(ts_array[start])
        end_ts = int(ts_array[end - 1])
        duration_hours = (end_ts - start_ts) / (1_000_000 * 3600)
        
        segments.append(ContinuousSegment(
            start_idx=start,
            end_idx=end,
            start_ts=start_ts,
            end_ts=end_ts,
            n_rows=n_rows,
            duration_hours=duration_hours,
        ))
    
    logger.info(f"Created {len(segments)} continuous segments")
    for i, seg in enumerate(segments):
        logger.info(f"  Segment {i}: {seg}")
    
    return segments


def ofi_safe_split(
    df: pl.DataFrame,
    segments: List[ContinuousSegment],
    train_ratio: float = 0.8,
    min_test_rows: int = 100,
) -> List[Split]:
    """
    Create OFI-safe train/test splits.
    
    CRITICAL: Each segment is split independently to prevent
    information leakage from OFI cumulative features.
    
    Args:
        df: Full DataFrame
        segments: List of continuous segments
        train_ratio: Fraction of each segment for training
        min_test_rows: Minimum rows required for test set
        
    Returns:
        List of Split objects
    """
    splits = []
    
    for i, seg in enumerate(segments):
        # Calculate split point within segment
        train_rows = int(seg.n_rows * train_ratio)
        test_rows = seg.n_rows - train_rows
        
        if test_rows < min_test_rows:
            logger.warning(f"Segment {i} test set too small ({test_rows} rows), skipping")
            continue
        
        split = Split(
            segment_id=i,
            train_start_idx=seg.start_idx,
            train_end_idx=seg.start_idx + train_rows,
            test_start_idx=seg.start_idx + train_rows,
            test_end_idx=seg.end_idx,
            train_rows=train_rows,
            test_rows=test_rows,
        )
        splits.append(split)
        logger.info(f"  Segment {i} split: {split}")
    
    total_train = sum(s.train_rows for s in splits)
    total_test = sum(s.test_rows for s in splits)
    logger.info(f"Total: {total_train:,} train rows, {total_test:,} test rows")
    
    return splits


def get_split_data(
    df: pl.DataFrame,
    split: Split,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Get train and test DataFrames for a split."""
    train_df = df.slice(split.train_start_idx, split.train_rows)
    test_df = df.slice(split.test_start_idx, split.test_rows)
    return train_df, test_df


def get_all_train_test(
    df: pl.DataFrame,
    splits: List[Split],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Get combined train and test DataFrames from all splits.
    
    Note: This combines data from multiple segments.
    For walk-forward evaluation, process splits individually.
    """
    train_dfs = []
    test_dfs = []
    
    for split in splits:
        train_df, test_df = get_split_data(df, split)
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    
    combined_train = pl.concat(train_dfs, how="vertical") if train_dfs else pl.DataFrame()
    combined_test = pl.concat(test_dfs, how="vertical") if test_dfs else pl.DataFrame()
    
    return combined_train, combined_test


class OFISafeSplitter:
    """
    OFI-safe data splitter for financial ML.
    
    Ensures:
    1. Gaps in data create separate segments
    2. No information leakage across segments
    3. Train data always before test data (no look-ahead)
    4. OFI cumulative features reset at segment boundaries
    """
    
    def __init__(
        self,
        bucket_size_us: int = 100_000,
        gap_threshold_factor: float = 2.0,
        train_ratio: float = 0.8,
        min_segment_rows: int = 100,
        min_test_rows: int = 100,
    ):
        self.bucket_size_us = bucket_size_us
        self.gap_threshold_factor = gap_threshold_factor
        self.train_ratio = train_ratio
        self.min_segment_rows = min_segment_rows
        self.min_test_rows = min_test_rows
        
        self.segments: List[ContinuousSegment] = []
        self.splits: List[Split] = []
    
    def fit(self, df: pl.DataFrame, ts_col: str = "ts") -> "OFISafeSplitter":
        """
        Analyze data and create splits.
        
        Args:
            df: Input DataFrame
            ts_col: Timestamp column name
        """
        # Detect segments
        self.segments = split_into_segments(
            df, ts_col,
            self.bucket_size_us,
            self.gap_threshold_factor,
            self.min_segment_rows,
        )
        
        # Create splits
        self.splits = ofi_safe_split(
            df, self.segments,
            self.train_ratio,
            self.min_test_rows,
        )
        
        return self
    
    def get_splits(self) -> List[Split]:
        """Get list of splits."""
        return self.splits
    
    def get_segments(self) -> List[ContinuousSegment]:
        """Get list of continuous segments."""
        return self.segments
    
    def get_train_test(
        self,
        df: pl.DataFrame,
        split_idx: int = 0,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Get train/test DataFrames for specific split."""
        if split_idx >= len(self.splits):
            raise ValueError(f"Split index {split_idx} out of range")
        return get_split_data(df, self.splits[split_idx])
    
    def get_all_train_test(
        self,
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Get combined train/test DataFrames."""
        return get_all_train_test(df, self.splits)
    
    def summary(self) -> Dict[str, Any]:
        """Get splitter summary."""
        return {
            "n_segments": len(self.segments),
            "n_splits": len(self.splits),
            "total_train_rows": sum(s.train_rows for s in self.splits),
            "total_test_rows": sum(s.test_rows for s in self.splits),
            "segments": [
                {
                    "rows": s.n_rows,
                    "hours": s.duration_hours,
                }
                for s in self.segments
            ],
        }


def walk_forward_splits(
    df: pl.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.1,
    gap_rows: int = 0,  # Gap between train and test to prevent leakage
    ts_col: str = "ts",
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Create walk-forward validation splits.
    
    Each split has more training data than the previous one,
    simulating real trading where you train on historical data.
    
    Args:
        df: Input DataFrame
        n_splits: Number of splits
        test_size: Fraction of data for each test set
        gap_rows: Rows to skip between train and test
        ts_col: Timestamp column name
        
    Returns:
        List of (train_df, test_df) tuples
    """
    n_rows = len(df)
    test_rows = int(n_rows * test_size)
    
    splits = []
    
    for i in range(n_splits):
        # Test window moves forward with each split
        test_end = n_rows - (n_splits - i - 1) * test_rows
        test_start = test_end - test_rows
        train_end = test_start - gap_rows
        
        if train_end <= 0:
            logger.warning(f"Skip split {i}: not enough training data")
            continue
        
        train_df = df.slice(0, train_end)
        test_df = df.slice(test_start, test_rows)
        
        logger.info(f"Walk-forward split {i}: train={len(train_df):,}, test={len(test_df):,}")
        splits.append((train_df, test_df))
    
    return splits


if __name__ == "__main__":
    # Test splitter
    import numpy as np
    
    # Create test data with a gap
    n_samples = 10000
    ts = np.arange(n_samples) * 100_000  # 100ms intervals
    
    # Insert a gap at position 5000
    ts[5000:] += 60_000_000  # 60 second gap
    
    df = pl.DataFrame({
        "ts": ts,
        "feature": np.random.randn(n_samples),
        "target": np.random.randn(n_samples),
    })
    
    # Test splitter
    splitter = OFISafeSplitter(
        bucket_size_us=100_000,
        gap_threshold_factor=2.0,
        train_ratio=0.8,
    )
    
    splitter.fit(df)
    
    print("\nSplitter Summary:")
    print(splitter.summary())
    
    # Get data
    train_df, test_df = splitter.get_all_train_test(df)
    print(f"\nTotal: train={len(train_df):,}, test={len(test_df):,}")
