"""
MIDAS Data Inspector and Visualizer

This script provides comprehensive data inspection, validation, and visualization
for the MIDAS pipeline output.
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import zstandard as zstd
import polars as pl
import numpy as np
from typing import Optional, Dict, List


class DataInspector:
    """Inspect and validate MIDAS pipeline data."""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            base_path = Path("/data") if Path("/data").exists() else Path("./data")
        
        self.base_path = base_path
        self.raw_path = base_path / "raw"
        self.clean_path = base_path / "clean"
        self.features_path = base_path / "features"
    
    def print_section(self, title: str):
        """Print a section header."""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def inspect_raw_data(self, sample_size: int = 100):
        """Inspect raw WebSocket data."""
        self.print_section("RAW DATA INSPECTION")
        
        files = sorted(self.raw_path.glob("*.jsonl.zst"))
        print(f"\nFound {len(files)} raw data files")
        
        if not files:
            print("⚠️  No raw data files found yet. Collector may still be initializing.")
            return
        
        # Show file details
        print("\n📁 Files:")
        total_size = 0
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name}: {size_mb:.2f} MB (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        print(f"\n📊 Total raw data size: {total_size:.2f} MB")
        
        # Read and analyze first file
        if files:
            print(f"\n🔍 Analyzing first file: {files[0].name}")
            
            dctx = zstd.ZstdDecompressor()
            records = []
            
            with open(files[0], "rb") as fh:
                with dctx.stream_reader(fh) as reader:
                    data = reader.read(1024 * 1024)  # Read 1MB
                    lines = data.decode().split("\n")
                    
                    for line in lines[:sample_size]:
                        if line.strip():
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            
            if records:
                print(f"\n✅ Successfully parsed {len(records)} records")
                
                # Count record types
                depth_count = sum(1 for r in records if r.get("type") == "depth")
                trade_count = sum(1 for r in records if r.get("type") == "trade")
                
                print(f"\n📈 Record types:")
                print(f"  - Depth updates: {depth_count}")
                print(f"  - Trades: {trade_count}")
                
                # Show sample records
                print(f"\n📝 Sample depth update:")
                depth_sample = next((r for r in records if r.get("type") == "depth"), None)
                if depth_sample:
                    print(f"  Timestamp: {depth_sample.get('exchange_ts')} μs")
                    print(f"  Symbol: {depth_sample.get('symbol')}")
                    print(f"  Update IDs: {depth_sample.get('first_update_id')} → {depth_sample.get('last_update_id')}")
                    print(f"  Bids: {len(depth_sample.get('bids', []))} levels")
                    print(f"  Asks: {len(depth_sample.get('asks', []))} levels")
                    if depth_sample.get('bids'):
                        print(f"  Best bid: {depth_sample['bids'][0]}")
                    if depth_sample.get('asks'):
                        print(f"  Best ask: {depth_sample['asks'][0]}")
                
                print(f"\n📝 Sample trade:")
                trade_sample = next((r for r in records if r.get("type") == "trade"), None)
                if trade_sample:
                    trade_data = trade_sample.get('trade', {})
                    print(f"  Timestamp: {trade_sample.get('exchange_ts')} μs")
                    print(f"  Price: {trade_data.get('price')}")
                    print(f"  Quantity: {trade_data.get('quantity')}")
                    print(f"  Buyer is maker: {trade_data.get('buyer_is_maker')}")
                
                # Validate data quality
                print(f"\n✅ Data Quality Checks:")
                
                # Check timestamp ordering
                timestamps = [r.get('exchange_ts', 0) for r in records]
                if timestamps and all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                    print(f"  ✓ Timestamps are ordered")
                else:
                    print(f"  ⚠️  Timestamps may be out of order")
                
                # Check for missing fields
                required_fields = ['exchange_ts', 'local_ts', 'type', 'symbol']
                missing = []
                for field in required_fields:
                    if not all(field in r for r in records):
                        missing.append(field)
                
                if not missing:
                    print(f"  ✓ All required fields present")
                else:
                    print(f"  ⚠️  Missing fields: {missing}")
    
    def inspect_clean_data(self):
        """Inspect cleaned/processed data."""
        self.print_section("CLEAN DATA INSPECTION")
        
        files = sorted(self.clean_path.glob("clean_*.parquet"))
        print(f"\nFound {len(files)} clean data files")
        
        if not files:
            print("⚠️  No clean data files found. Processor may still be working.")
            return
        
        # Show file details
        print("\n📁 Files:")
        total_size = 0
        total_rows = 0
        
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            # Quick row count
            df_info = pl.read_parquet(f, n_rows=0)
            df = pl.read_parquet(f)
            rows = len(df)
            total_rows += rows
            
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name}: {size_mb:.2f} MB, {rows:,} rows (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        print(f"\n📊 Total: {total_size:.2f} MB, {total_rows:,} rows")
        
        # Analyze first file in detail
        if files:
            print(f"\n🔍 Analyzing: {files[0].name}")
            df = pl.read_parquet(files[0])
            
            print(f"\n📋 Schema ({len(df.columns)} columns):")
            for col, dtype in list(zip(df.columns, df.dtypes))[:20]:
                print(f"  {col:20s} {str(dtype):15s}")
            if len(df.columns) > 20:
                print(f"  ... and {len(df.columns) - 20} more columns")
            
            # Show sample data
            print(f"\n📊 Sample rows:")
            print(df.head(3))
            
            # Statistics
            print(f"\n📈 Statistics:")
            if 'ts' in df.columns:
                ts_min = df['ts'].min()
                ts_max = df['ts'].max()
                duration_sec = (ts_max - ts_min) / 1_000_000
                print(f"  Time range: {duration_sec:.2f} seconds")
                print(f"  Rows per second: {len(df) / duration_sec:.2f}")
            
            if 'bid_px_01' in df.columns and 'ask_px_01' in df.columns:
                print(f"  Bid range: {df['bid_px_01'].min():.2f} - {df['bid_px_01'].max():.2f}")
                print(f"  Ask range: {df['ask_px_01'].min():.2f} - {df['ask_px_01'].max():.2f}")
            
            # Data quality checks
            print(f"\n✅ Data Quality:")
            
            # Check for nulls
            null_counts = df.null_count()
            has_nulls = any(null_counts.row(0))
            if has_nulls:
                print(f"  ⚠️  Some columns contain nulls")
                for col, count in zip(df.columns, null_counts.row(0)):
                    if count > 0:
                        print(f"    {col}: {count} nulls ({count/len(df)*100:.1f}%)")
            else:
                print(f"  ✓ No null values")
            
            # Check timestamp ordering
            if 'ts' in df.columns:
                is_sorted = df['ts'].is_sorted()
                if is_sorted:
                    print(f"  ✓ Timestamps are ordered")
                else:
                    print(f"  ⚠️  Timestamps not ordered")
            
            # Check for crossed book
            if 'bid_px_01' in df.columns and 'ask_px_01' in df.columns:
                crossed = (df['bid_px_01'] >= df['ask_px_01']).sum()
                if crossed == 0:
                    print(f"  ✓ No crossed books")
                else:
                    print(f"  ⚠️  {crossed} crossed book instances")
    
    def inspect_features(self):
        """Inspect feature data."""
        self.print_section("FEATURES DATA INSPECTION")
        
        files = sorted(self.features_path.glob("features_*.parquet"))
        print(f"\nFound {len(files)} feature files")
        
        if not files:
            print("⚠️  No feature files found. Feature generator may still be working.")
            return
        
        # Show file details
        print("\n📁 Files:")
        total_size = 0
        total_rows = 0
        
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            df = pl.read_parquet(f)
            rows = len(df)
            total_rows += rows
            
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name}: {size_mb:.2f} MB, {rows:,} rows (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        print(f"\n📊 Total: {total_size:.2f} MB, {total_rows:,} rows")
        
        # Analyze features
        if files:
            print(f"\n🔍 Analyzing: {files[0].name}")
            df = pl.read_parquet(files[0])
            
            print(f"\n📋 Features ({len(df.columns)} columns):")
            
            # Group columns by category
            price_cols = [c for c in df.columns if 'px' in c]
            size_cols = [c for c in df.columns if 'sz' in c]
            feature_cols = [c for c in df.columns if c not in price_cols + size_cols and c != 'ts']
            
            print(f"\n  Price columns ({len(price_cols)}): {', '.join(price_cols[:5])}...")
            print(f"  Size columns ({len(size_cols)}): {', '.join(size_cols[:5])}...")
            print(f"\n  Feature columns ({len(feature_cols)}):")
            for col in feature_cols:
                dtype = df[col].dtype
                print(f"    {col:25s} {str(dtype):15s}")
            
            # Show statistics for key features
            print(f"\n📈 Feature Statistics:")
            
            key_features = ['midprice', 'spread', 'imbalance', 'ofi', 'microprice', 
                          'taker_buy_volume', 'taker_sell_volume', 'signed_volume']
            
            for feat in key_features:
                if feat in df.columns:
                    col_data = df[feat]
                    print(f"\n  {feat}:")
                    print(f"    Min:    {col_data.min():.6f}")
                    print(f"    Max:    {col_data.max():.6f}")
                    print(f"    Mean:   {col_data.mean():.6f}")
                    print(f"    Median: {col_data.median():.6f}")
                    print(f"    Std:    {col_data.std():.6f}")
                    
                    # Check for infinities or NaNs
                    if col_data.dtype in [pl.Float64, pl.Float32]:
                        null_count = col_data.null_count()
                        if null_count > 0:
                            print(f"    ⚠️  NaNs: {null_count}")
            
            # Show sample rows
            print(f"\n📊 Sample Feature Rows:")
            sample_cols = ['ts', 'midprice', 'spread', 'imbalance', 'ofi', 'signed_volume']
            sample_cols = [c for c in sample_cols if c in df.columns]
            print(df.select(sample_cols).head(5))
    
    def validate_pipeline(self):
        """Validate the entire pipeline."""
        self.print_section("PIPELINE VALIDATION")
        
        print("\n🔍 Checking data flow:")
        
        raw_files = list(self.raw_path.glob("*.jsonl.zst"))
        clean_files = list(self.clean_path.glob("clean_*.parquet"))
        feature_files = list(self.features_path.glob("features_*.parquet"))
        
        print(f"  Raw data files:     {len(raw_files):3d} {'✓' if raw_files else '⚠️'}")
        print(f"  Clean data files:   {len(clean_files):3d} {'✓' if clean_files else '⚠️'}")
        print(f"  Feature files:      {len(feature_files):3d} {'✓' if feature_files else '⚠️'}")
        
        # Check file ages
        print(f"\n🕒 File freshness:")
        now = datetime.now().timestamp()
        
        if raw_files:
            latest_raw = max(f.stat().st_mtime for f in raw_files)
            age_sec = now - latest_raw
            print(f"  Latest raw file:    {age_sec:.0f}s ago {'✓' if age_sec < 120 else '⚠️'}")
        
        if clean_files:
            latest_clean = max(f.stat().st_mtime for f in clean_files)
            age_sec = now - latest_clean
            print(f"  Latest clean file:  {age_sec:.0f}s ago {'✓' if age_sec < 300 else '⚠️'}")
        
        if feature_files:
            latest_feat = max(f.stat().st_mtime for f in feature_files)
            age_sec = now - latest_feat
            print(f"  Latest feature file: {age_sec:.0f}s ago {'✓' if age_sec < 300 else '⚠️'}")
        
        # Check data consistency
        print(f"\n✅ Data Consistency:")
        
        if clean_files:
            df_clean = pl.read_parquet(clean_files[0])
            expected_cols = ['ts', 'bid_px_01', 'ask_px_01', 'bid_sz_01', 'ask_sz_01']
            missing = [c for c in expected_cols if c not in df_clean.columns]
            if not missing:
                print(f"  ✓ Clean data has expected columns")
            else:
                print(f"  ⚠️  Clean data missing: {missing}")
        
        if feature_files:
            df_feat = pl.read_parquet(feature_files[0])
            expected_features = ['midprice', 'spread', 'imbalance', 'ofi']
            missing = [f for f in expected_features if f not in df_feat.columns]
            if not missing:
                print(f"  ✓ Features data has expected columns")
            else:
                print(f"  ⚠️  Features missing: {missing}")
    
    def run_full_inspection(self):
        """Run complete inspection."""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "MIDAS DATA INSPECTOR" + " " * 38 + "║")
        print("╚" + "═" * 78 + "╝")
        
        self.inspect_raw_data()
        self.inspect_clean_data()
        self.inspect_features()
        self.validate_pipeline()
        
        print("\n" + "=" * 80)
        print("  INSPECTION COMPLETE")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect MIDAS pipeline data")
    parser.add_argument("--path", type=str, help="Base data path", default=None)
    parser.add_argument("--raw", action="store_true", help="Inspect only raw data")
    parser.add_argument("--clean", action="store_true", help="Inspect only clean data")
    parser.add_argument("--features", action="store_true", help="Inspect only features")
    
    args = parser.parse_args()
    
    base_path = Path(args.path) if args.path else None
    inspector = DataInspector(base_path)
    
    if args.raw:
        inspector.inspect_raw_data()
    elif args.clean:
        inspector.inspect_clean_data()
    elif args.features:
        inspector.inspect_features()
    else:
        inspector.run_full_inspection()


if __name__ == "__main__":
    main()
