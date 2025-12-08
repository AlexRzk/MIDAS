#!/usr/bin/env python3
"""
Data Validation Script for MIDAS Training Pipeline

Verifies that collected feature data is:
1. Present and accessible
2. In correct Parquet format
3. Contains all required columns
4. Has proper data types and ranges
5. Sufficient quantity for training
6. Ready for TFT model consumption

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py --min-hours 6
    python scripts/validate_training_data.py --output report.json
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import argparse

try:
    import polars as pl
except ImportError:
    print("ERROR: polars not installed. Run: pip install polars")
    sys.exit(1)

# Required columns for training (minimum set - just OHLC)
REQUIRED_COLUMNS_MIN = [
    "open", "high", "low", "close",
]

# Full feature set (preferred but not all required)
PREFERRED_COLUMNS = [
    # OHLCV (volume is preferred but not required)
    "volume", "vwap",
    # Core features
    "midprice", "microprice", "spread",
    # Order flow
    "ofi", "ofi_10",
    # Imbalances
    "imbalance_1", "imbalance_5", "imbalance_10",
    # Volume metrics
    "taker_buy_volume", "taker_sell_volume",
    # Liquidity
    "liquidity_1", "liquidity_5", "liquidity_10",
]

# Timestamp column name (can be 'ts' or 'bucket')
TIMESTAMP_COLUMNS = ["ts", "bucket"]

OPTIONAL_COLUMNS = [
    "spread_bps", "signed_volume", "volume_imbalance",
    "returns", "volatility_20", "volatility_100",
    "kyle_lambda", "vpin", "number_of_trades",
]

class TrainingDataValidator:
    """Validates feature data for training readiness."""
    
    def __init__(self, data_dir: Path, min_hours: float = 6.0):
        self.data_dir = Path(data_dir)
        self.min_hours = min_hours
        self.issues = []
        self.warnings = []
        self.info = []
        
    def log_issue(self, msg: str):
        """Log a critical issue."""
        self.issues.append(msg)
        print(f"❌ ERROR: {msg}")
    
    def log_warning(self, msg: str):
        """Log a warning."""
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")
    
    def log_info(self, msg: str):
        """Log informational message."""
        self.info.append(msg)
        print(f"✓ {msg}")
    
    def check_directory_exists(self) -> bool:
        """Check if data directory exists."""
        if not self.data_dir.exists():
            self.log_issue(f"Data directory not found: {self.data_dir}")
            return False
        
        if not self.data_dir.is_dir():
            self.log_issue(f"Path is not a directory: {self.data_dir}")
            return False
        
        self.log_info(f"Data directory exists: {self.data_dir}")
        return True
    
    def find_feature_files(self) -> List[Path]:
        """Find all feature Parquet files."""
        patterns = ["features_*.parquet", "aggregated_*.parquet"]
        files = []
        
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))
        
        # Remove duplicates and sort
        files = sorted(set(files))
        
        if not files:
            self.log_issue(f"No feature files found matching patterns: {patterns}")
            return []
        
        self.log_info(f"Found {len(files)} feature files")
        return files
    
    def check_file_readability(self, files: List[Path]) -> List[Path]:
        """Check which files are readable."""
        readable = []
        
        for file in files:
            try:
                # Try to read schema only (fast)
                schema = pl.read_parquet_schema(file)
                readable.append(file)
            except Exception as e:
                self.log_warning(f"Cannot read {file.name}: {e}")
        
        if len(readable) < len(files):
            self.log_warning(f"Only {len(readable)}/{len(files)} files are readable")
        else:
            self.log_info(f"All {len(files)} files are readable")
        
        return readable
    
    def check_columns(self, files: List[Path]) -> Tuple[bool, str]:
        """Verify all required columns are present. Returns (success, timestamp_col_name)."""
        if not files:
            return False, ""
        
        # Check first file as representative
        sample_file = files[0]
        
        try:
            schema = pl.read_parquet_schema(sample_file)
            columns = set(schema.names())
            
            self.log_info(f"Available columns: {sorted(columns)}")
            
            # Find timestamp column
            ts_col = None
            for col in TIMESTAMP_COLUMNS:
                if col in columns:
                    ts_col = col
                    break
            
            if not ts_col:
                self.log_issue(f"Missing timestamp column (expected one of: {TIMESTAMP_COLUMNS})")
                return False, ""
            
            self.log_info(f"Using timestamp column: '{ts_col}'")
            
            # Check minimum required columns (OHLC - volume not always available)
            required_ohlc = ["open", "high", "low", "close"]
            missing_ohlc = set(required_ohlc) - columns
            if missing_ohlc:
                self.log_issue(f"Missing minimum required columns: {sorted(missing_ohlc)}")
                return False, ts_col
            
            self.log_info(f"All {len(required_ohlc)} minimum required OHLC columns present")
            
            # Check for volume (nice to have but not required)
            if "volume" not in columns:
                self.log_info("Note: 'volume' column not present (using other features)")
            
            # Check preferred columns (nice to have)
            missing_pref = set(PREFERRED_COLUMNS) - columns
            if missing_pref:
                present_features = [c for c in PREFERRED_COLUMNS if c in columns]
                if present_features:
                    self.log_info(f"Present features: {sorted(present_features)}")
                self.log_info(f"Missing features (optional): {sorted(missing_pref)}")
            else:
                self.log_info("All preferred features present")
            
            # Check optional
            present_optional = set(OPTIONAL_COLUMNS) & columns
            if present_optional:
                self.log_info(f"Optional columns present: {sorted(present_optional)}")
            
            return True, ts_col
            
        except Exception as e:
            self.log_issue(f"Failed to read schema from {sample_file.name}: {e}")
            return False, ""
    
    def check_data_types(self, files: List[Path], ts_col: str) -> bool:
        """Check data types are appropriate."""
        if not files or not ts_col:
            return False
        
        sample_file = files[0]
        
        try:
            df = pl.read_parquet(sample_file)
            
            # Check timestamp is integer (microseconds)
            if df[ts_col].dtype not in [pl.Int64, pl.UInt64]:
                self.log_warning(f"Timestamp column '{ts_col}' has unexpected type: {df[ts_col].dtype}")
            
            # Check numeric columns
            for col in REQUIRED_COLUMNS_MIN:
                if col in df.columns:
                    if df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                        self.log_warning(f"Column '{col}' has non-numeric type: {df[col].dtype}")
            
            self.log_info("Data types look correct")
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to check data types: {e}")
            return False
    
    def check_data_quantity(self, files: List[Path], ts_col: str) -> Tuple[int, float]:
        """Check total rows and time coverage."""
        if not files or not ts_col:
            return 0, 0.0
        
        total_rows = 0
        min_ts = None
        max_ts = None
        
        for file in files:
            try:
                df = pl.read_parquet(file, columns=[ts_col])
                total_rows += len(df)
                
                file_min = df[ts_col].min()
                file_max = df[ts_col].max()
                
                if min_ts is None or file_min < min_ts:
                    min_ts = file_min
                if max_ts is None or file_max > max_ts:
                    max_ts = file_max
                    
            except Exception as e:
                self.log_warning(f"Failed to read {file.name}: {e}")
        
        if min_ts and max_ts:
            # Convert microseconds to hours
            duration_us = max_ts - min_ts
            duration_hours = duration_us / (1e6 * 3600)
            
            self.log_info(f"Total rows: {total_rows:,}")
            self.log_info(f"Time coverage: {duration_hours:.2f} hours")
            self.log_info(f"Period: {datetime.fromtimestamp(min_ts/1e6)} to {datetime.fromtimestamp(max_ts/1e6)}")
            
            if duration_hours < self.min_hours:
                self.log_warning(
                    f"Only {duration_hours:.2f} hours of data (minimum recommended: {self.min_hours})"
                )
            
            # Estimate if we have enough for training (60 input + 10 output = 70 minutes minimum)
            expected_rows_per_hour = 60  # 1-minute buckets
            min_rows_needed = 70  # For one training sample
            
            if total_rows < min_rows_needed:
                self.log_issue(f"Insufficient data: {total_rows} rows < {min_rows_needed} minimum")
            else:
                max_samples = total_rows - 70
                self.log_info(f"Can generate ~{max_samples:,} training samples (input_len=60, output_len=10)")
            
            return total_rows, duration_hours
        
        return total_rows, 0.0
    
    def check_data_quality(self, files: List[Path], ts_col: str) -> bool:
        """Check for null values, outliers, and monotonicity."""
        if not files or not ts_col:
            return False
        
        sample_file = files[0]
        
        try:
            df = pl.read_parquet(sample_file)
            
            # Check for nulls in minimum required columns
            null_counts = df.null_count()
            has_nulls = False
            
            for col in REQUIRED_COLUMNS_MIN:
                if col in df.columns:
                    null_count = null_counts[col][0]
                    if null_count > 0:
                        null_pct = 100.0 * null_count / len(df)
                        self.log_warning(f"Column '{col}' has {null_count} nulls ({null_pct:.2f}%)")
                        has_nulls = True
            
            if not has_nulls:
                self.log_info("No null values in required columns")
            
            # Check timestamp monotonicity
            ts_diff = df[ts_col].diff()
            if (ts_diff < 0).any():
                self.log_warning("Timestamps are not monotonically increasing")
            else:
                self.log_info("Timestamps are monotonically increasing")
            
            # Check for reasonable value ranges
            if "close" in df.columns:
                close_min = df["close"].min()
                close_max = df["close"].max()
                
                if close_min <= 0:
                    self.log_warning(f"Close price has invalid values: min={close_min}")
                elif close_max / close_min > 100:
                    self.log_warning(f"Extreme price range: {close_min:.2f} to {close_max:.2f}")
                else:
                    self.log_info(f"Price range looks reasonable: {close_min:.2f} to {close_max:.2f}")
            
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to check data quality: {e}")
            return False
    
    def check_time_consistency(self, files: List[Path], ts_col: str) -> bool:
        """Check that time buckets are consistent (e.g., 1-minute intervals)."""
        if not files or not ts_col:
            return False
        
        sample_file = files[0]
        
        try:
            df = pl.read_parquet(sample_file, n_rows=100)
            
            if len(df) < 2:
                self.log_warning("Not enough rows to check time consistency")
                return True
            
            # Calculate time differences
            ts_diff = df[ts_col].diff().drop_nulls()
            
            if len(ts_diff) == 0:
                return True
            
            # Expected: 60 seconds = 60,000,000 microseconds
            expected_interval_us = 60_000_000
            
            median_diff = ts_diff.median()
            
            # Check if close to expected
            tolerance = 0.1  # 10%
            if abs(median_diff - expected_interval_us) / expected_interval_us < tolerance:
                self.log_info(f"Time buckets are consistent: ~{median_diff/1e6:.1f} seconds")
            else:
                self.log_warning(
                    f"Unexpected time bucket size: {median_diff/1e6:.1f}s (expected 60s)"
                )
            
            return True
            
        except Exception as e:
            self.log_warning(f"Failed to check time consistency: {e}")
            return True
    
    def validate(self) -> Dict:
        """Run all validation checks."""
        print("\n" + "="*60)
        print("MIDAS Training Data Validation")
        print("="*60 + "\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "min_hours_required": self.min_hours,
            "checks": {},
            "passed": False,
        }
        
        # Check 1: Directory exists
        if not self.check_directory_exists():
            results["passed"] = False
            return results
        
        # Check 2: Find files
        files = self.find_feature_files()
        results["file_count"] = len(files)
        
        if not files:
            results["passed"] = False
            return results
        
        # Check 3: Readability
        readable_files = self.check_file_readability(files)
        results["readable_files"] = len(readable_files)
        
        if not readable_files:
            results["passed"] = False
            return results
        
        # Check 4: Columns
        columns_ok, ts_col = self.check_columns(readable_files)
        results["checks"]["columns"] = columns_ok
        results["timestamp_column"] = ts_col
        
        if not columns_ok or not ts_col:
            results["passed"] = False
            return results
        
        # Check 5: Data types
        types_ok = self.check_data_types(readable_files, ts_col)
        results["checks"]["data_types"] = types_ok
        
        # Check 6: Quantity
        total_rows, hours = self.check_data_quantity(readable_files, ts_col)
        results["total_rows"] = total_rows
        results["hours_coverage"] = hours
        results["checks"]["sufficient_data"] = hours >= self.min_hours
        
        # Check 7: Quality
        quality_ok = self.check_data_quality(readable_files, ts_col)
        results["checks"]["data_quality"] = quality_ok
        
        # Check 8: Time consistency
        time_ok = self.check_time_consistency(readable_files, ts_col)
        results["checks"]["time_consistency"] = time_ok
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        all_checks_passed = (
            columns_ok and 
            types_ok and 
            hours >= self.min_hours and
            quality_ok and
            time_ok
        )
        
        results["passed"] = all_checks_passed
        results["issues"] = self.issues
        results["warnings"] = self.warnings
        
        if all_checks_passed:
            print("✅ ALL CHECKS PASSED - Data is ready for training!")
        else:
            print(f"❌ VALIDATION FAILED - {len(self.issues)} issue(s) found")
        
        print(f"Warnings: {len(self.warnings)}")
        print(f"Files: {len(readable_files)}")
        print(f"Rows: {total_rows:,}")
        print(f"Coverage: {hours:.2f} hours")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Validate MIDAS training data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing feature Parquet files",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=6.0,
        help="Minimum hours of data required (default: 6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report file (optional)",
    )
    
    args = parser.parse_args()
    
    validator = TrainingDataValidator(args.data_dir, args.min_hours)
    results = validator.validate()
    
    # Save report if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
