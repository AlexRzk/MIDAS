#!/usr/bin/env python3
"""
Fix normalization issues - re-normalize features with proper column exclusions.

Usage:
    python3 scripts/fix_normalization.py --input-dir data/features --scaler-dir data/scalers --output-dir data/features_normalized
"""
import argparse
import sys
from pathlib import Path

# Add features module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from features.normalize import normalize_existing_features
import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Fix normalization with proper column exclusions")
    parser.add_argument("--input-dir", type=str, default="data/features", help="Input feature directory")
    parser.add_argument("--scaler-dir", type=str, default="data/scalers", help="Scaler output directory")
    parser.add_argument("--output-dir", type=str, default="data/features_normalized", help="Normalized output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scalers")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    scaler_dir = Path(args.scaler_dir)
    output_dir = Path(args.output_dir)
    
    # Check input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
        
    # Find first parquet file to inspect columns
    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {input_dir}")
        return 1
        
    print(f"📊 Inspecting columns from {parquet_files[0].name}...")
    df = pl.read_parquet(parquet_files[0])
    
    # Columns that should NOT be normalized
    # - Identifiers: timestamp, time_idx
    # - Price levels (used as targets): microprice, close, open, high, low
    # - Volume (already in original units for trading): taker_buy_volume, taker_sell_volume, volume
    exclude_columns = [
        'timestamp', 'time_idx',  # Identifiers
        'microprice', 'close', 'open', 'high', 'low',  # Price targets
    ]
    
    # Check which excluded columns actually exist
    existing_excludes = [c for c in exclude_columns if c in df.columns]
    print(f"\n📋 Columns to EXCLUDE from normalization: {existing_excludes}")
    
    # Show what will be normalized
    normalize_cols = [c for c in df.columns if c not in existing_excludes]
    print(f"📋 Columns to NORMALIZE ({len(normalize_cols)}): {normalize_cols[:10]}{'...' if len(normalize_cols) > 10 else ''}")
    
    print(f"\n🔧 Running normalization...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Scalers: {scaler_dir}")
    
    try:
        normalize_existing_features(
            input_dir=str(input_dir),
            scaler_dir=str(scaler_dir),
            output_dir=str(output_dir),
            exclude_columns=existing_excludes,
            overwrite=args.overwrite
        )
        print("\n✅ Normalization complete!")
        print(f"\n📁 Outputs:")
        print(f"   Normalized data: {output_dir}")
        print(f"   Scalers: {scaler_dir}/normalization_manifest.json")
        
        # Show scaler stats
        import json
        manifest_path = scaler_dir / "normalization_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"\n📊 Fitted scalers:")
            for scaler_type, features in manifest.items():
                if scaler_type != "metadata":
                    print(f"   {scaler_type}: {len(features)} features")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
