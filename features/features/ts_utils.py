"""
Timestamp unit detection and normalization utilities.

Handles microseconds (μs), milliseconds (ms), and seconds (s) timestamps,
with automatic detection and conversion.

This module fixes the critical bug where bucket_ms=60000 was being applied
to microsecond timestamps, creating 60ms buckets instead of 60-second buckets.
"""
import polars as pl
import structlog
from typing import Literal, Optional, Tuple
from enum import Enum

logger = structlog.get_logger()


class TimestampUnit(str, Enum):
    """Supported timestamp units."""
    MICROSECONDS = "us"
    MILLISECONDS = "ms"
    SECONDS = "s"
    UNKNOWN = "unknown"


# Conversion factors to microseconds (canonical unit)
TO_MICROSECONDS = {
    TimestampUnit.MICROSECONDS: 1,
    TimestampUnit.MILLISECONDS: 1_000,
    TimestampUnit.SECONDS: 1_000_000,
}

# Reference timestamps for detection (2020-01-01 to 2030-01-01 range)
# Used to determine magnitude-based unit classification
REFERENCE_BOUNDS = {
    TimestampUnit.MICROSECONDS: (1577836800_000_000, 1893456000_000_000),  # 2020-2030 in μs
    TimestampUnit.MILLISECONDS: (1577836800_000, 1893456000_000),          # 2020-2030 in ms
    TimestampUnit.SECONDS: (1577836800, 1893456000),                        # 2020-2030 in s
}


def detect_timestamp_unit(
    df: pl.DataFrame,
    ts_col: str = "ts",
    sample_size: int = 1000,
) -> TimestampUnit:
    """
    Auto-detect timestamp unit by examining value magnitudes.
    
    Detection heuristic based on expected timestamp ranges (2020-2030):
    - μs: 1.58e15 to 1.89e15
    - ms: 1.58e12 to 1.89e12
    - s:  1.58e9  to 1.89e9
    
    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
        sample_size: Number of rows to sample for detection
        
    Returns:
        Detected TimestampUnit
    """
    if ts_col not in df.columns:
        logger.error("timestamp_column_not_found", column=ts_col)
        return TimestampUnit.UNKNOWN
    
    if len(df) == 0:
        logger.warning("empty_dataframe_for_unit_detection")
        return TimestampUnit.UNKNOWN
    
    # Sample for efficiency on large DataFrames
    sample_df = df.head(min(sample_size, len(df)))
    
    # Get median timestamp (robust to outliers)
    median_ts = sample_df[ts_col].median()
    
    if median_ts is None:
        logger.warning("null_median_timestamp")
        return TimestampUnit.UNKNOWN
    
    median_ts = float(median_ts)
    
    # Check each unit's expected range
    for unit, (low, high) in REFERENCE_BOUNDS.items():
        if low <= median_ts <= high:
            logger.info(
                "detected_timestamp_unit",
                column=ts_col,
                median_value=median_ts,
                unit=unit.value,
            )
            return unit
    
    # Fallback: magnitude-based classification
    if median_ts > 1e15:
        detected = TimestampUnit.MICROSECONDS
    elif median_ts > 1e12:
        detected = TimestampUnit.MILLISECONDS
    elif median_ts > 1e9:
        detected = TimestampUnit.SECONDS
    else:
        detected = TimestampUnit.UNKNOWN
    
    logger.info(
        "detected_timestamp_unit_fallback",
        column=ts_col,
        median_value=median_ts,
        unit=detected.value,
    )
    
    return detected


def normalize_timestamp(
    df: pl.DataFrame,
    ts_col: str = "ts",
    source_unit: Optional[TimestampUnit] = None,
    target_unit: TimestampUnit = TimestampUnit.MICROSECONDS,
    add_converted_columns: bool = False,
) -> pl.DataFrame:
    """
    Normalize timestamp column to a standard unit.
    
    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
        source_unit: Source unit (if None, auto-detect)
        target_unit: Target unit for normalization
        add_converted_columns: If True, add ts_us and ts_ms columns
        
    Returns:
        DataFrame with normalized timestamp
    """
    if source_unit is None:
        source_unit = detect_timestamp_unit(df, ts_col)
    
    if source_unit == TimestampUnit.UNKNOWN:
        logger.warning("cannot_normalize_unknown_unit", column=ts_col)
        return df
    
    # Calculate conversion factor
    source_to_us = TO_MICROSECONDS[source_unit]
    target_to_us = TO_MICROSECONDS[target_unit]
    
    # Convert: source -> μs -> target
    if source_to_us == target_to_us:
        # No conversion needed
        logger.debug("no_timestamp_conversion_needed", unit=source_unit.value)
        converted_df = df
    else:
        # Apply conversion factor
        factor = source_to_us / target_to_us
        
        if factor >= 1:
            # Multiplying (e.g., ms -> μs)
            converted_df = df.with_columns([
                (pl.col(ts_col) * int(factor)).alias(ts_col)
            ])
        else:
            # Dividing (e.g., μs -> ms)
            converted_df = df.with_columns([
                (pl.col(ts_col) // int(1 / factor)).alias(ts_col)
            ])
        
        logger.info(
            "normalized_timestamp",
            column=ts_col,
            source_unit=source_unit.value,
            target_unit=target_unit.value,
            factor=factor,
        )
    
    # Optionally add both ts_us and ts_ms columns
    if add_converted_columns:
        current_unit = target_unit
        current_to_us = TO_MICROSECONDS[current_unit]
        
        # Add ts_us (microseconds)
        if current_unit == TimestampUnit.MICROSECONDS:
            converted_df = converted_df.with_columns([
                pl.col(ts_col).alias("ts_us")
            ])
        else:
            factor_to_us = current_to_us
            converted_df = converted_df.with_columns([
                (pl.col(ts_col) * factor_to_us).alias("ts_us")
            ])
        
        # Add ts_ms (milliseconds)
        converted_df = converted_df.with_columns([
            (pl.col("ts_us") // 1000).alias("ts_ms")
        ])
    
    return converted_df


def create_time_buckets(
    df: pl.DataFrame,
    bucket_size: int,
    bucket_unit: Literal["us", "ms", "s"] = "ms",
    ts_col: str = "ts",
    ts_unit: Optional[TimestampUnit] = None,
    output_col: str = "bucket",
) -> pl.DataFrame:
    """
    Create time bucket column with automatic unit handling.
    
    This function properly handles the unit mismatch that caused the bug
    where 680k rows were created instead of ~1.2k for 1-minute aggregation.
    
    Args:
        df: DataFrame with timestamp column
        bucket_size: Size of bucket in bucket_unit (e.g., 60000 for 60 seconds when bucket_unit='ms')
        bucket_unit: Unit of bucket_size ('us', 'ms', or 's')
        ts_col: Name of timestamp column
        ts_unit: Unit of timestamp column (if None, auto-detect)
        output_col: Name of output bucket column
        
    Returns:
        DataFrame with bucket column added
        
    Example:
        # 1-minute buckets (60000 ms = 60 seconds)
        >>> df = create_time_buckets(df, bucket_size=60000, bucket_unit='ms')
        
        # Same as above but explicit
        >>> df = create_time_buckets(df, bucket_size=60, bucket_unit='s')
    """
    if ts_unit is None:
        ts_unit = detect_timestamp_unit(df, ts_col)
    
    if ts_unit == TimestampUnit.UNKNOWN:
        logger.error("cannot_create_buckets_unknown_unit", column=ts_col)
        return df
    
    # Convert bucket_unit string to enum
    bucket_unit_enum = {
        "us": TimestampUnit.MICROSECONDS,
        "ms": TimestampUnit.MILLISECONDS,
        "s": TimestampUnit.SECONDS,
    }[bucket_unit]
    
    # Convert bucket_size to timestamp units
    bucket_to_us = TO_MICROSECONDS[bucket_unit_enum]
    ts_to_us = TO_MICROSECONDS[ts_unit]
    
    # Calculate bucket size in timestamp units
    bucket_size_in_ts_units = int(bucket_size * bucket_to_us / ts_to_us)
    
    logger.info(
        "creating_time_buckets",
        bucket_size=bucket_size,
        bucket_unit=bucket_unit,
        ts_unit=ts_unit.value,
        bucket_size_in_ts_units=bucket_size_in_ts_units,
    )
    
    # Create bucket column
    df = df.with_columns([
        ((pl.col(ts_col) // bucket_size_in_ts_units) * bucket_size_in_ts_units).alias(output_col)
    ])
    
    # Log statistics
    n_buckets = df[output_col].n_unique()
    logger.info(
        "buckets_created",
        total_rows=len(df),
        unique_buckets=n_buckets,
        avg_rows_per_bucket=round(len(df) / n_buckets, 2) if n_buckets > 0 else 0,
    )
    
    return df


def validate_timestamp_monotonicity(
    df: pl.DataFrame,
    ts_col: str = "ts",
) -> Tuple[bool, int, list]:
    """
    Check if timestamps are monotonically increasing.
    
    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
        
    Returns:
        Tuple of (is_monotonic, num_violations, violation_indices)
    """
    if len(df) <= 1:
        return True, 0, []
    
    # Check for decreasing timestamps
    violations = df.with_row_index().filter(
        pl.col(ts_col) < pl.col(ts_col).shift(1)
    )
    
    num_violations = len(violations)
    is_monotonic = num_violations == 0
    
    # Get indices of violations (limit to first 10)
    violation_indices = violations["index"].head(10).to_list() if num_violations > 0 else []
    
    if not is_monotonic:
        logger.warning(
            "timestamp_monotonicity_violated",
            num_violations=num_violations,
            sample_indices=violation_indices,
        )
    
    return is_monotonic, num_violations, violation_indices
