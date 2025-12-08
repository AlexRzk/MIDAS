#!/usr/bin/env python3
"""Quick script to inspect feature files - show first and last rows."""

import polars as pl
from pathlib import Path
import sys


def inspect_features(data_dir: str = "data/features"):
    """Inspect feature files and show sample data."""
    data_path = Path(data_dir)
    
    # Find feature files
    feature_files = sorted(data_path.glob("features_*.parquet"))
    
    if not feature_files:
        print("No feature files found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(feature_files)} feature files")
    print(f"{'='*80}\n")
    
    # Read first file
    first_file = feature_files[0]
    print(f"First file: {first_file.name}")
    df_first = pl.read_parquet(first_file)
    
    # Read last file
    last_file = feature_files[-1]
    print(f"Last file: {last_file.name}")
    df_last = pl.read_parquet(last_file)
    
    # Combine for analysis
    print(f"\nTotal rows in first file: {len(df_first):,}")
    print(f"Total rows in last file: {len(df_last):,}")
    print(f"Total columns: {len(df_first.columns)}")
    
    print(f"\n{'='*80}")
    print("COLUMN NAMES")
    print(f"{'='*80}")
    print(f"\n{', '.join(df_first.columns)}\n")
    
    print(f"{'='*80}")
    print("FIRST 5 ROWS (from first file)")
    print(f"{'='*80}\n")
    print(df_first.head(5))
    
    print(f"\n{'='*80}")
    print("LAST 5 ROWS (from last file)")
    print(f"{'='*80}\n")
    print(df_last.tail(5))
    
    print(f"\n{'='*80}")
    print("COLUMN STATISTICS (from first file)")
    print(f"{'='*80}\n")
    
    # Show stats for key columns
    key_cols = ["ts", "midprice", "spread", "ofi", "imbalance_1", "volatility_20"]
    available_cols = [c for c in key_cols if c in df_first.columns]
    
    if available_cols:
        stats = df_first.select(available_cols).describe()
        print(stats)
    
    print(f"\n{'='*80}")
    print("DATA TYPES")
    print(f"{'='*80}\n")
    for col, dtype in zip(df_first.columns, df_first.dtypes):
        print(f"{col:30s} {dtype}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/features"
    inspect_features(data_dir)
