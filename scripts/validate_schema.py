#!/usr/bin/env python3
"""
Schema validation script.

Validates that Parquet files have expected schema structure:
- Required columns present
- Correct data types
- Consistent schemas across files
"""
import sys
from pathlib import Path
import polars as pl
from dataclasses import dataclass


@dataclass
class SchemaDefinition:
    """Expected schema definition."""
    name: str
    required_columns: dict[str, pl.DataType]
    optional_columns: dict[str, pl.DataType]


# Define expected schemas
CLEAN_SCHEMA = SchemaDefinition(
    name="clean_data",
    required_columns={
        "ts": pl.Int64,
        "bid_px_01": pl.Float64,
        "ask_px_01": pl.Float64,
        "bid_sz_01": pl.Float64,
        "ask_sz_01": pl.Float64,
    },
    optional_columns={
        f"bid_px_{i:02d}": pl.Float64 for i in range(2, 11)
    } | {
        f"ask_px_{i:02d}": pl.Float64 for i in range(2, 11)
    } | {
        f"bid_sz_{i:02d}": pl.Float64 for i in range(2, 11)
    } | {
        f"ask_sz_{i:02d}": pl.Float64 for i in range(2, 11)
    },
)

FEATURES_SCHEMA = SchemaDefinition(
    name="features_data",
    required_columns={
        "ts": pl.Int64,
        "midprice": pl.Float64,
        "spread": pl.Float64,
        "spread_bps": pl.Float64,
        "imbalance": pl.Float64,
    },
    optional_columns={
        "ofi": pl.Float64,
        "ofi_cumulative": pl.Float64,
        "microprice": pl.Float64,
        "returns": pl.Float64,
        "volatility_20": pl.Float64,
        "volatility_100": pl.Float64,
        "kyle_lambda": pl.Float64,
        "vpin": pl.Float64,
        "ladder_slope_bid": pl.Float64,
        "ladder_slope_ask": pl.Float64,
        "queue_imbalance_bid": pl.Float64,
        "queue_imbalance_ask": pl.Float64,
        "vol_of_vol": pl.Float64,
    },
)


def validate_schema(filepath: Path, expected: SchemaDefinition) -> tuple[bool, list[str]]:
    """
    Validate a Parquet file against expected schema.
    
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    try:
        # Read just the schema, not the data
        schema = pl.read_parquet_schema(filepath)
    except Exception as e:
        return False, [f"Cannot read schema: {e}"]
    
    # Check required columns
    for col, expected_type in expected.required_columns.items():
        if col not in schema:
            errors.append(f"Missing required column: {col}")
        elif not _types_compatible(schema[col], expected_type):
            errors.append(f"Column {col}: expected {expected_type}, got {schema[col]}")
    
    # Check optional columns if present
    for col, expected_type in expected.optional_columns.items():
        if col in schema:
            if not _types_compatible(schema[col], expected_type):
                errors.append(f"Column {col}: expected {expected_type}, got {schema[col]}")
    
    return len(errors) == 0, errors


def _types_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check if actual type is compatible with expected."""
    # Allow some flexibility (e.g., Int64 vs UInt64)
    if actual == expected:
        return True
    
    # Int types are compatible
    int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    if actual in int_types and expected in int_types:
        return True
    
    # Float types are compatible
    float_types = {pl.Float32, pl.Float64}
    if actual in float_types and expected in float_types:
        return True
    
    return False


def validate_directory(data_dir: Path, pattern: str, schema: SchemaDefinition) -> tuple[int, int]:
    """
    Validate all files in a directory.
    
    Returns (passed_count, failed_count).
    """
    files = sorted(data_dir.glob(pattern))
    
    if not files:
        print(f"No files matching {pattern} found in {data_dir}")
        return 0, 0
    
    print(f"\nValidating {len(files)} files against {schema.name} schema")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for filepath in files:
        is_valid, errors = validate_schema(filepath, schema)
        
        if is_valid:
            print(f"  ✓ {filepath.name}")
            passed += 1
        else:
            print(f"  ✗ {filepath.name}")
            for error in errors:
                print(f"      - {error}")
            failed += 1
    
    return passed, failed


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Parquet schemas")
    parser.add_argument("--clean", action="store_true", help="Validate clean data schema")
    parser.add_argument("--features", action="store_true", help="Validate feature data schema")
    parser.add_argument("--all", action="store_true", help="Validate all schemas")
    
    args = parser.parse_args()
    
    if not any([args.clean, args.features, args.all]):
        args.all = True
    
    base_dir = Path(__file__).parent.parent / "data"
    total_passed = 0
    total_failed = 0
    
    if args.all or args.clean:
        print("\n" + "=" * 60)
        print("CLEAN DATA SCHEMA VALIDATION")
        print("=" * 60)
        clean_dir = base_dir / "clean"
        if clean_dir.exists():
            passed, failed = validate_directory(clean_dir, "clean_*.parquet", CLEAN_SCHEMA)
            total_passed += passed
            total_failed += failed
    
    if args.all or args.features:
        print("\n" + "=" * 60)
        print("FEATURES DATA SCHEMA VALIDATION")
        print("=" * 60)
        features_dir = base_dir / "features"
        if features_dir.exists():
            passed, failed = validate_directory(features_dir, "features_*.parquet", FEATURES_SCHEMA)
            total_passed += passed
            total_failed += failed
    
    # Summary
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
