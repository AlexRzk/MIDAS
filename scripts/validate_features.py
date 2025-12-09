#!/usr/bin/env python3
"""
Feature validation script for MIDAS pipeline.

Validates Parquet feature files for:
- Required column presence and types
- Timestamp monotonicity
- Bucket counts for given intervals
- Data quality metrics

Usage:
    python scripts/validate_features.py --dir data/features/
    python scripts/validate_features.py --sample data/features/features_20241208_180000.parquet
    python scripts/validate_features.py --dir data/features/ --interval 60000 --output reports/validation.json
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import polars as pl

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from features.features.ts_utils import detect_timestamp_unit, validate_timestamp_monotonicity, TimestampUnit
except ImportError:
    # Fallback if running outside package
    from enum import Enum
    
    class TimestampUnit(str, Enum):
        MICROSECONDS = "us"
        MILLISECONDS = "ms"
        SECONDS = "s"
        UNKNOWN = "unknown"
    
    def detect_timestamp_unit(df, ts_col="ts"):
        if ts_col not in df.columns or len(df) == 0:
            return TimestampUnit.UNKNOWN
        median_ts = df[ts_col].median()
        if median_ts > 1e15:
            return TimestampUnit.MICROSECONDS
        elif median_ts > 1e12:
            return TimestampUnit.MILLISECONDS
        elif median_ts > 1e9:
            return TimestampUnit.SECONDS
        return TimestampUnit.UNKNOWN
    
    def validate_timestamp_monotonicity(df, ts_col="ts"):
        if len(df) <= 1:
            return True, 0, []
        violations = df.with_row_index().filter(pl.col(ts_col) < pl.col(ts_col).shift(1))
        return len(violations) == 0, len(violations), violations["index"].head(10).to_list() if len(violations) > 0 else []


# Required columns for ML features
REQUIRED_COLUMNS = {
    "core": [
        "ts",
        "midprice",
        "microprice", 
        "spread",
    ],
    "ofi": [
        "ofi",
        "ofi_10",
    ],
    "imbalance": [
        "imbalance_1",
        "imbalance_5",
        "imbalance_10",
    ],
    "volume": [
        "taker_buy_volume",
        "taker_sell_volume",
    ],
    "liquidity": [
        "liquidity_1",
        "liquidity_5",
        "liquidity_10",
    ],
    "kline": [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
    ],
}

# Expected data types
EXPECTED_TYPES = {
    "ts": [pl.Int64, pl.UInt64],
    "midprice": [pl.Float64, pl.Float32],
    "spread": [pl.Float64, pl.Float32],
    "ofi": [pl.Float64, pl.Float32, pl.Int64],
    "open": [pl.Float64, pl.Float32],
    "high": [pl.Float64, pl.Float32],
    "low": [pl.Float64, pl.Float32],
    "close": [pl.Float64, pl.Float32],
    "volume": [pl.Float64, pl.Float32],
}


def validate_file(
    filepath: Path,
    interval_ms: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Validate a single Parquet feature file.
    
    Args:
        filepath: Path to Parquet file
        interval_ms: Expected bucket interval in milliseconds (optional)
        verbose: Print progress messages
        
    Returns:
        Validation report dict
    """
    report = {
        "file": str(filepath),
        "valid": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
        "stats": {},
        "columns": {
            "present": [],
            "missing": [],
            "types": {},
        },
    }
    
    # Check file exists
    if not filepath.exists():
        report["valid"] = False
        report["errors"].append(f"File not found: {filepath}")
        return report
    
    # Read file
    try:
        df = pl.read_parquet(filepath)
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to read Parquet: {str(e)}")
        return report
    
    report["stats"]["rows"] = len(df)
    report["stats"]["columns"] = len(df.columns)
    
    if verbose:
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Check required columns
    all_required = []
    for category, cols in REQUIRED_COLUMNS.items():
        all_required.extend(cols)
    
    present_cols = set(df.columns)
    report["columns"]["present"] = list(present_cols)
    
    for col in all_required:
        if col not in present_cols:
            report["columns"]["missing"].append(col)
            # Kline columns are warnings, not errors
            if col in REQUIRED_COLUMNS.get("kline", []):
                report["warnings"].append(f"Missing kline column: {col}")
            else:
                report["warnings"].append(f"Missing expected column: {col}")
    
    # Check data types
    for col, expected_types in EXPECTED_TYPES.items():
        if col in df.columns:
            actual_type = df[col].dtype
            report["columns"]["types"][col] = str(actual_type)
            if actual_type not in expected_types:
                report["warnings"].append(
                    f"Column '{col}' has type {actual_type}, expected one of {expected_types}"
                )
    
    # Timestamp validation
    if "ts" in df.columns:
        # Detect unit
        ts_unit = detect_timestamp_unit(df, "ts")
        report["stats"]["ts_unit"] = ts_unit.value if hasattr(ts_unit, 'value') else str(ts_unit)
        
        # Check monotonicity
        is_mono, num_violations, violation_indices = validate_timestamp_monotonicity(df, "ts")
        report["stats"]["ts_monotonic"] = is_mono
        report["stats"]["ts_violations"] = num_violations
        
        if not is_mono:
            report["warnings"].append(
                f"Timestamp not monotonic: {num_violations} violations at indices {violation_indices[:5]}..."
            )
        
        # Time range
        ts_min = df["ts"].min()
        ts_max = df["ts"].max()
        report["stats"]["ts_min"] = int(ts_min) if ts_min is not None else None
        report["stats"]["ts_max"] = int(ts_max) if ts_max is not None else None
        
        # Bucket count validation
        if interval_ms is not None and ts_unit != TimestampUnit.UNKNOWN:
            # Convert interval to timestamp units
            if ts_unit == TimestampUnit.MICROSECONDS or ts_unit.value == "us":
                interval_in_ts = interval_ms * 1000
            elif ts_unit == TimestampUnit.MILLISECONDS or ts_unit.value == "ms":
                interval_in_ts = interval_ms
            else:  # seconds
                interval_in_ts = interval_ms / 1000
            
            # Count unique buckets
            df_buckets = df.with_columns([
                (pl.col("ts") // int(interval_in_ts)).alias("_bucket")
            ])
            n_buckets = df_buckets["_bucket"].n_unique()
            
            # Expected buckets based on time range
            if ts_min is not None and ts_max is not None:
                expected_buckets = int((ts_max - ts_min) / interval_in_ts) + 1
                report["stats"]["interval_ms"] = interval_ms
                report["stats"]["actual_buckets"] = n_buckets
                report["stats"]["expected_buckets"] = expected_buckets
                report["stats"]["rows_per_bucket"] = round(len(df) / n_buckets, 2) if n_buckets > 0 else 0
                
                # Check for reasonable bucket count
                if n_buckets > expected_buckets * 10:
                    report["warnings"].append(
                        f"Too many buckets ({n_buckets}) for interval {interval_ms}ms. "
                        f"Expected ~{expected_buckets}. Possible timestamp unit mismatch."
                    )
    
    # Data quality checks
    if "midprice" in df.columns:
        null_count = df["midprice"].null_count()
        if null_count > 0:
            pct = null_count / len(df) * 100
            report["stats"]["midprice_null_pct"] = round(pct, 2)
            if pct > 5:
                report["warnings"].append(f"High null rate in midprice: {pct:.1f}%")
    
    if "spread" in df.columns:
        spread_stats = df["spread"].describe()
        report["stats"]["spread_mean"] = float(df["spread"].mean()) if df["spread"].mean() is not None else None
        
        # Check for negative spreads
        neg_spreads = df.filter(pl.col("spread") < 0)
        if len(neg_spreads) > 0:
            report["errors"].append(f"Found {len(neg_spreads)} negative spreads")
            report["valid"] = False
    
    # Volume checks
    for vol_col in ["taker_buy_volume", "taker_sell_volume", "volume"]:
        if vol_col in df.columns:
            neg_vol = df.filter(pl.col(vol_col) < 0)
            if len(neg_vol) > 0:
                report["errors"].append(f"Found {len(neg_vol)} negative {vol_col}")
                report["valid"] = False
    
    # Overall validation status
    if report["errors"]:
        report["valid"] = False
    
    return report


def validate_directory(
    dir_path: Path,
    interval_ms: Optional[int] = None,
    sample_size: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Validate all feature files in a directory.
    
    Args:
        dir_path: Directory containing Parquet files
        interval_ms: Expected bucket interval
        sample_size: Number of files to fully validate (0 = all)
        verbose: Print progress
        
    Returns:
        Summary validation report
    """
    report = {
        "directory": str(dir_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_files": 0,
            "validated_files": 0,
            "valid_files": 0,
            "total_rows": 0,
            "total_errors": 0,
            "total_warnings": 0,
        },
        "files": [],
        "common_issues": [],
    }
    
    # List files
    files = sorted(dir_path.glob("features_*.parquet"))
    report["summary"]["total_files"] = len(files)
    
    if verbose:
        print(f"Found {len(files)} feature files in {dir_path}")
    
    if len(files) == 0:
        report["common_issues"].append("No feature files found")
        return report
    
    # Select files to validate
    if sample_size > 0 and len(files) > sample_size:
        # Sample: first, last, and random middle files
        import random
        middle_indices = random.sample(range(1, len(files) - 1), min(sample_size - 2, len(files) - 2))
        selected_indices = [0] + sorted(middle_indices) + [len(files) - 1]
        files_to_validate = [files[i] for i in selected_indices[:sample_size]]
    else:
        files_to_validate = files
    
    # Validate each file
    all_missing_cols = set()
    
    for filepath in files_to_validate:
        if verbose:
            print(f"\nValidating: {filepath.name}")
        
        file_report = validate_file(filepath, interval_ms=interval_ms, verbose=verbose)
        report["files"].append(file_report)
        report["summary"]["validated_files"] += 1
        report["summary"]["total_rows"] += file_report["stats"].get("rows", 0)
        report["summary"]["total_errors"] += len(file_report["errors"])
        report["summary"]["total_warnings"] += len(file_report["warnings"])
        
        if file_report["valid"]:
            report["summary"]["valid_files"] += 1
        
        all_missing_cols.update(file_report["columns"]["missing"])
    
    # Common issues summary
    if all_missing_cols:
        report["common_issues"].append(f"Missing columns across files: {sorted(all_missing_cols)}")
    
    return report


def print_report(report: dict, detailed: bool = False):
    """Print human-friendly validation report."""
    print("\n" + "=" * 60)
    print("MIDAS Feature Validation Report")
    print("=" * 60)
    
    if "directory" in report:
        # Directory report
        print(f"\nDirectory: {report['directory']}")
        print(f"Generated: {report['timestamp']}")
        
        summary = report["summary"]
        print(f"\nSummary:")
        print(f"  Total files:     {summary['total_files']}")
        print(f"  Validated:       {summary['validated_files']}")
        print(f"  Valid:           {summary['valid_files']}")
        print(f"  Total rows:      {summary['total_rows']:,}")
        print(f"  Total errors:    {summary['total_errors']}")
        print(f"  Total warnings:  {summary['total_warnings']}")
        
        if report["common_issues"]:
            print(f"\nCommon Issues:")
            for issue in report["common_issues"]:
                print(f"  ⚠ {issue}")
        
        if detailed:
            for file_report in report["files"]:
                print(f"\n--- {Path(file_report['file']).name} ---")
                print_file_report(file_report)
    else:
        # Single file report
        print_file_report(report)


def print_file_report(report: dict):
    """Print report for a single file."""
    status = "✅ VALID" if report["valid"] else "❌ INVALID"
    print(f"  Status: {status}")
    print(f"  Rows: {report['stats'].get('rows', 'N/A'):,}")
    
    if "ts_unit" in report["stats"]:
        print(f"  Timestamp unit: {report['stats']['ts_unit']}")
    
    if "ts_monotonic" in report["stats"]:
        mono_status = "✓" if report["stats"]["ts_monotonic"] else "✗"
        print(f"  Timestamp monotonic: {mono_status}")
    
    if "actual_buckets" in report["stats"]:
        print(f"  Buckets: {report['stats']['actual_buckets']} (expected ~{report['stats']['expected_buckets']})")
        print(f"  Rows/bucket: {report['stats']['rows_per_bucket']}")
    
    if report["columns"]["missing"]:
        print(f"  Missing columns: {', '.join(report['columns']['missing'][:10])}")
        if len(report["columns"]["missing"]) > 10:
            print(f"    ... and {len(report['columns']['missing']) - 10} more")
    
    if report["errors"]:
        print(f"  Errors:")
        for err in report["errors"][:5]:
            print(f"    ❌ {err}")
    
    if report["warnings"]:
        print(f"  Warnings:")
        for warn in report["warnings"][:5]:
            print(f"    ⚠ {warn}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MIDAS feature Parquet files"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing feature files",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="Single file to validate",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Expected bucket interval in milliseconds (e.g., 60000 for 1 minute)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of files to validate when using --dir (0 = all)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-file report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    
    args = parser.parse_args()
    
    if not args.dir and not args.sample:
        # Default to data/features/ relative to repo root
        args.dir = Path(__file__).parent.parent / "data" / "features"
    
    verbose = not args.quiet
    
    if args.sample:
        report = validate_file(args.sample, interval_ms=args.interval, verbose=verbose)
    else:
        report = validate_directory(
            args.dir,
            interval_ms=args.interval,
            sample_size=args.sample_size,
            verbose=verbose,
        )
    
    # Print human-friendly report
    print_report(report, detailed=args.detailed)
    
    # Save JSON report if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nJSON report saved to: {args.output}")
    
    # Exit code based on validation status
    if "summary" in report:
        sys.exit(0 if report["summary"]["total_errors"] == 0 else 1)
    else:
        sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
