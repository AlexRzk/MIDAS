"""
Kline (OHLCV) computation from trade data.

Generates standard candlestick data aligned to configurable time intervals.
"""
import polars as pl
import structlog
from typing import Optional, Literal
from .ts_utils import detect_timestamp_unit, TimestampUnit, TO_MICROSECONDS, create_time_buckets

logger = structlog.get_logger()


# Standard kline columns that will be added to features
KLINE_COLUMNS = [
    "open",
    "high", 
    "low",
    "close",
    "volume",
    "vwap",
    "number_of_trades",
]


def compute_klines_from_trades(
    df: pl.DataFrame,
    interval_ms: int = 60000,
    price_col: str = "last_trade_px",
    qty_col: str = "last_trade_qty",
    ts_col: str = "ts",
    ts_unit: Optional[TimestampUnit] = None,
) -> pl.DataFrame:
    """
    Compute OHLCV klines from trade data.
    
    Args:
        df: DataFrame with trade data (price, quantity, timestamp)
        interval_ms: Kline interval in milliseconds (default: 60000 = 1 minute)
        price_col: Column name for trade price
        qty_col: Column name for trade quantity
        ts_col: Column name for timestamp
        ts_unit: Timestamp unit (auto-detected if None)
        
    Returns:
        DataFrame with kline columns added (open, high, low, close, volume, vwap, number_of_trades)
    """
    if price_col not in df.columns:
        logger.warning("price_column_missing", column=price_col)
        # Return df with null kline columns
        return _add_null_kline_columns(df)
    
    if qty_col not in df.columns:
        logger.warning("qty_column_missing", column=qty_col)
        return _add_null_kline_columns(df)
    
    # Detect timestamp unit if not provided
    if ts_unit is None:
        ts_unit = detect_timestamp_unit(df, ts_col)
    
    if ts_unit == TimestampUnit.UNKNOWN:
        logger.error("cannot_compute_klines_unknown_ts_unit")
        return _add_null_kline_columns(df)
    
    # Convert interval_ms to timestamp units
    ms_to_us = TO_MICROSECONDS[TimestampUnit.MILLISECONDS]
    ts_to_us = TO_MICROSECONDS[ts_unit]
    
    interval_in_ts_units = int(interval_ms * ms_to_us / ts_to_us)
    
    logger.info(
        "computing_klines",
        interval_ms=interval_ms,
        ts_unit=ts_unit.value,
        interval_in_ts_units=interval_in_ts_units,
    )
    
    # Create bucket column for grouping
    df_with_bucket = df.with_columns([
        ((pl.col(ts_col) // interval_in_ts_units) * interval_in_ts_units).alias("_kline_bucket")
    ])
    
    # Filter out rows with null prices (no trade occurred)
    valid_trades = df_with_bucket.filter(
        pl.col(price_col).is_not_null() & (pl.col(price_col) > 0)
    )
    
    if len(valid_trades) == 0:
        logger.warning("no_valid_trades_for_klines")
        return _add_null_kline_columns(df)
    
    # Compute OHLCV per bucket
    klines = valid_trades.group_by("_kline_bucket").agg([
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.col(qty_col).sum().alias("volume"),
        # VWAP = sum(price * qty) / sum(qty)
        ((pl.col(price_col) * pl.col(qty_col)).sum() / pl.col(qty_col).sum()).alias("vwap"),
        pl.count().alias("number_of_trades"),
    ]).sort("_kline_bucket")
    
    # Join klines back to original DataFrame
    df_with_bucket = df_with_bucket.join(
        klines,
        on="_kline_bucket",
        how="left",
    )
    
    # Drop temporary bucket column
    df_with_bucket = df_with_bucket.drop("_kline_bucket")
    
    # Forward fill kline values within buckets (all rows in same bucket get same OHLCV)
    # This ensures each 100ms snapshot has the corresponding kline values
    
    logger.info(
        "klines_computed",
        unique_klines=len(klines),
        total_rows=len(df_with_bucket),
    )
    
    return df_with_bucket


def compute_klines_aggregated(
    df: pl.DataFrame,
    interval_ms: int = 60000,
    price_col: str = "midprice",
    ts_col: str = "ts",
    ts_unit: Optional[TimestampUnit] = None,
    volume_cols: Optional[list[str]] = None,
) -> pl.DataFrame:
    """
    Compute OHLCV klines from aggregated price data (e.g., midprice snapshots).
    
    Use this when trade data is not available but you have high-frequency price snapshots.
    
    Args:
        df: DataFrame with price snapshots
        interval_ms: Kline interval in milliseconds
        price_col: Column name for price (default: midprice)
        ts_col: Column name for timestamp
        ts_unit: Timestamp unit (auto-detected if None)
        volume_cols: List of volume columns to sum (default: taker_buy_volume, taker_sell_volume)
        
    Returns:
        DataFrame aggregated to kline interval with OHLCV columns
    """
    if volume_cols is None:
        volume_cols = ["taker_buy_volume", "taker_sell_volume"]
    
    if price_col not in df.columns:
        logger.error("price_column_missing_for_klines", column=price_col)
        raise ValueError(f"Price column '{price_col}' not found")
    
    # Detect timestamp unit
    if ts_unit is None:
        ts_unit = detect_timestamp_unit(df, ts_col)
    
    if ts_unit == TimestampUnit.UNKNOWN:
        logger.error("cannot_aggregate_klines_unknown_ts_unit")
        raise ValueError("Cannot determine timestamp unit")
    
    # Convert interval to timestamp units
    ms_to_us = TO_MICROSECONDS[TimestampUnit.MILLISECONDS]
    ts_to_us = TO_MICROSECONDS[ts_unit]
    interval_in_ts_units = int(interval_ms * ms_to_us / ts_to_us)
    
    # Create bucket
    df = df.with_columns([
        ((pl.col(ts_col) // interval_in_ts_units) * interval_in_ts_units).alias("_bucket")
    ])
    
    # Build aggregation expressions
    agg_exprs = [
        pl.col(ts_col).first().alias(ts_col),
        pl.col(price_col).first().alias("open"),
        pl.col(price_col).max().alias("high"),
        pl.col(price_col).min().alias("low"),
        pl.col(price_col).last().alias("close"),
        pl.count().alias("number_of_trades"),
    ]
    
    # Add volume aggregation
    existing_vol_cols = [c for c in volume_cols if c in df.columns]
    if existing_vol_cols:
        vol_sum = sum(pl.col(c) for c in existing_vol_cols)
        agg_exprs.append(vol_sum.sum().alias("volume"))
        
        # VWAP calculation
        agg_exprs.append(
            ((pl.col(price_col) * vol_sum).sum() / vol_sum.sum()).alias("vwap")
        )
    else:
        # No volume data - set to null
        agg_exprs.append(pl.lit(None).cast(pl.Float64).alias("volume"))
        agg_exprs.append(pl.lit(None).cast(pl.Float64).alias("vwap"))
    
    # Aggregate
    result = df.group_by("_bucket").agg(agg_exprs).sort("_bucket")
    result = result.drop("_bucket")
    
    logger.info(
        "klines_aggregated",
        input_rows=len(df),
        output_rows=len(result),
        interval_ms=interval_ms,
    )
    
    return result


def _add_null_kline_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add null kline columns to DataFrame when klines cannot be computed."""
    return df.with_columns([
        pl.lit(None).cast(pl.Float64).alias(col)
        for col in KLINE_COLUMNS
    ])


def validate_kline_columns(df: pl.DataFrame) -> dict:
    """
    Validate presence and quality of kline columns.
    
    Returns:
        Dict with validation results
    """
    results = {
        "has_all_columns": True,
        "missing_columns": [],
        "null_counts": {},
        "value_ranges": {},
    }
    
    for col in KLINE_COLUMNS:
        if col not in df.columns:
            results["has_all_columns"] = False
            results["missing_columns"].append(col)
        else:
            null_count = df[col].null_count()
            results["null_counts"][col] = null_count
            
            # Get value range for numeric columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                non_null = df.filter(pl.col(col).is_not_null())
                if len(non_null) > 0:
                    results["value_ranges"][col] = {
                        "min": float(non_null[col].min()),
                        "max": float(non_null[col].max()),
                        "mean": float(non_null[col].mean()),
                    }
    
    return results
