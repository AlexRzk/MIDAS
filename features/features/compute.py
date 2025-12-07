"""
Feature computation for order book data.

Implements standard microstructure features:
- OFI (Order Flow Imbalance)
- Spread
- Midprice
- Book Imbalance
- Microprice
- Liquidity metrics
"""
import polars as pl
import numpy as np
from typing import Optional
import structlog

logger = structlog.get_logger()


class FeatureComputer:
    """
    Computes ML-ready features from cleaned order book data.
    """
    
    def __init__(self, depth: int = 10, ofi_window: int = 10):
        self.depth = depth
        self.ofi_window = ofi_window
    
    def compute_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute basic order book features.
        
        Adds:
        - midprice
        - spread
        - spread_bps (spread in basis points)
        """
        return df.with_columns([
            # Midprice: (best_bid + best_ask) / 2
            ((pl.col("bid_px_01") + pl.col("ask_px_01")) / 2).alias("midprice"),
            
            # Spread: best_ask - best_bid
            (pl.col("ask_px_01") - pl.col("bid_px_01")).alias("spread"),
            
            # Spread in basis points
            (
                (pl.col("ask_px_01") - pl.col("bid_px_01")) 
                / ((pl.col("bid_px_01") + pl.col("ask_px_01")) / 2) 
                * 10000
            ).alias("spread_bps"),
        ])
    
    def compute_imbalance(self, df: pl.DataFrame, levels: list[int] = None) -> pl.DataFrame:
        """
        Compute order book imbalance at various depths.
        
        Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Range: [-1, 1] where positive means more bid volume
        """
        if levels is None:
            levels = [1, 5, 10]
        
        for level in levels:
            if level > self.depth:
                continue
            
            # Sum bid volumes up to level
            bid_cols = [f"bid_sz_{i:02d}" for i in range(1, level + 1)]
            ask_cols = [f"ask_sz_{i:02d}" for i in range(1, level + 1)]
            
            # Check which columns exist
            existing_bid_cols = [c for c in bid_cols if c in df.columns]
            existing_ask_cols = [c for c in ask_cols if c in df.columns]
            
            if existing_bid_cols and existing_ask_cols:
                bid_vol_expr = sum(pl.col(c) for c in existing_bid_cols)
                ask_vol_expr = sum(pl.col(c) for c in existing_ask_cols)
                
                df = df.with_columns([
                    bid_vol_expr.alias(f"bid_vol_{level}"),
                    ask_vol_expr.alias(f"ask_vol_{level}"),
                    (
                        (bid_vol_expr - ask_vol_expr) 
                        / (bid_vol_expr + ask_vol_expr)
                    ).alias(f"imbalance_{level}"),
                ])
        
        # Default imbalance (level 1)
        if "imbalance_1" in df.columns:
            df = df.with_columns(pl.col("imbalance_1").alias("imbalance"))
        
        return df
    
    def compute_ofi(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute Order Flow Imbalance (OFI).
        
        OFI = ΔBidSize - ΔAskSize
        
        More specifically, for the best level:
        OFI_t = (BidSize_t - BidSize_{t-1}) - (AskSize_t - AskSize_{t-1})
        
        Positive OFI indicates net buying pressure.
        """
        # Compute changes in best bid/ask sizes
        df = df.with_columns([
            (pl.col("bid_sz_01") - pl.col("bid_sz_01").shift(1)).alias("delta_bid_sz"),
            (pl.col("ask_sz_01") - pl.col("ask_sz_01").shift(1)).alias("delta_ask_sz"),
        ])
        
        # OFI = ΔBid - ΔAsk
        df = df.with_columns([
            (pl.col("delta_bid_sz") - pl.col("delta_ask_sz")).alias("ofi"),
        ])
        
        # Rolling OFI (sum over window)
        if self.ofi_window > 1:
            df = df.with_columns([
                pl.col("ofi").rolling_sum(self.ofi_window).alias(f"ofi_{self.ofi_window}"),
            ])
        
        # Cumulative OFI
        df = df.with_columns([
            pl.col("ofi").cum_sum().alias("ofi_cumulative"),
        ])
        
        return df
    
    def compute_microprice(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute microprice - a better estimate of true price than midprice.
        
        Microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        
        Weights the prices by the opposite side's size.
        """
        return df.with_columns([
            (
                (pl.col("bid_sz_01") * pl.col("ask_px_01") + pl.col("ask_sz_01") * pl.col("bid_px_01"))
                / (pl.col("bid_sz_01") + pl.col("ask_sz_01"))
            ).alias("microprice"),
        ])
    
    def compute_liquidity_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute liquidity metrics at various depth levels.
        """
        for level in [1, 5, 10]:
            if level > self.depth:
                continue
            
            bid_cols = [f"bid_sz_{i:02d}" for i in range(1, level + 1)]
            ask_cols = [f"ask_sz_{i:02d}" for i in range(1, level + 1)]
            
            existing_bid_cols = [c for c in bid_cols if c in df.columns]
            existing_ask_cols = [c for c in ask_cols if c in df.columns]
            
            if existing_bid_cols and existing_ask_cols:
                df = df.with_columns([
                    sum(pl.col(c) for c in existing_bid_cols).alias(f"liquidity_bid_{level}"),
                    sum(pl.col(c) for c in existing_ask_cols).alias(f"liquidity_ask_{level}"),
                ])
                
                # Total liquidity at depth
                df = df.with_columns([
                    (pl.col(f"liquidity_bid_{level}") + pl.col(f"liquidity_ask_{level}")).alias(f"liquidity_{level}"),
                ])
        
        return df
    
    def compute_trade_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute features from trade data.
        
        - taker_buy_volume (already in data)
        - taker_sell_volume (already in data)
        - signed_volume = buy - sell
        - volume_imbalance
        """
        # Ensure columns exist
        if "taker_buy_vol" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("taker_buy_vol"))
        if "taker_sell_vol" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("taker_sell_vol"))
        
        # Rename for output
        df = df.with_columns([
            pl.col("taker_buy_vol").alias("taker_buy_volume"),
            pl.col("taker_sell_vol").alias("taker_sell_volume"),
        ])
        
        # Signed volume
        df = df.with_columns([
            (pl.col("taker_buy_volume") - pl.col("taker_sell_volume")).alias("signed_volume"),
        ])
        
        # Volume imbalance
        total_vol = pl.col("taker_buy_volume") + pl.col("taker_sell_volume")
        df = df.with_columns([
            pl.when(total_vol > 0)
            .then((pl.col("taker_buy_volume") - pl.col("taker_sell_volume")) / total_vol)
            .otherwise(0.0)
            .alias("volume_imbalance"),
        ])
        
        return df
    
    def compute_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute price-related features.
        
        - returns (log returns of midprice)
        - volatility (rolling std of returns)
        """
        df = df.with_columns([
            (pl.col("midprice").log() - pl.col("midprice").log().shift(1)).alias("returns"),
        ])
        
        # Rolling volatility
        df = df.with_columns([
            pl.col("returns").rolling_std(20).alias("volatility_20"),
            pl.col("returns").rolling_std(100).alias("volatility_100"),
        ])
        
        return df
    
    def compute_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all features on the input DataFrame.
        
        Returns DataFrame with all computed features.
        """
        logger.info("computing_features", input_rows=len(df))
        
        # Apply all feature computations
        df = self.compute_basic_features(df)
        df = self.compute_imbalance(df)
        df = self.compute_ofi(df)
        df = self.compute_microprice(df)
        df = self.compute_liquidity_metrics(df)
        df = self.compute_trade_features(df)
        df = self.compute_price_features(df)
        
        logger.info("features_computed", output_columns=len(df.columns))
        
        return df
    
    def get_output_schema(self) -> list[str]:
        """Get the list of output columns for the feature DataFrame."""
        cols = ["ts"]
        
        # Price/size levels
        for i in range(1, self.depth + 1):
            cols.extend([
                f"bid_px_{i:02d}",
                f"bid_sz_{i:02d}",
                f"ask_px_{i:02d}",
                f"ask_sz_{i:02d}",
            ])
        
        # Core features
        cols.extend([
            "midprice",
            "spread",
            "spread_bps",
            "imbalance",
            "ofi",
            "microprice",
            "taker_buy_volume",
            "taker_sell_volume",
            "signed_volume",
            "last_trade_px",
            "last_trade_qty",
        ])
        
        # Liquidity at depths
        for level in [1, 5, 10]:
            if level <= self.depth:
                cols.extend([
                    f"liquidity_bid_{level}",
                    f"liquidity_ask_{level}",
                    f"liquidity_{level}",
                ])
        
        return cols


def time_bucket_aggregate(
    df: pl.DataFrame,
    bucket_ms: int,
    timestamp_col: str = "ts",
) -> pl.DataFrame:
    """
    Aggregate data into fixed time buckets.
    
    For each bucket, takes:
    - Last value for prices/sizes
    - Sum for volumes
    - Mean for imbalance/OFI
    """
    # Convert bucket to microseconds (assuming ts is in microseconds)
    bucket_us = bucket_ms * 1000
    
    # Create bucket column
    df = df.with_columns([
        (pl.col(timestamp_col) // bucket_us * bucket_us).alias("bucket_ts"),
    ])
    
    # Define aggregation strategy
    # Last value columns (prices, sizes)
    last_cols = [c for c in df.columns if any(
        c.startswith(prefix) for prefix in ["bid_px", "ask_px", "bid_sz", "ask_sz", "last_trade"]
    )]
    
    # Sum columns (volumes)
    sum_cols = [c for c in df.columns if "volume" in c.lower() or "vol" in c.lower()]
    
    # Mean columns
    mean_cols = ["ofi", "imbalance", "spread", "midprice", "microprice"]
    mean_cols = [c for c in mean_cols if c in df.columns]
    
    # Build aggregation expressions
    agg_exprs = []
    
    for col in last_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).last().alias(col))
    
    for col in sum_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).sum().alias(col))
    
    for col in mean_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().alias(col))
    
    # Aggregate
    result = df.group_by("bucket_ts").agg(agg_exprs).sort("bucket_ts")
    
    # Rename bucket_ts to ts
    result = result.rename({"bucket_ts": "ts"})
    
    return result
