"""
Data cleaning and validation utilities.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import structlog

from .orderbook import OrderBookSnapshot

logger = structlog.get_logger()


@dataclass
class CleaningStats:
    """Statistics from the cleaning process."""
    total_snapshots: int = 0
    valid_snapshots: int = 0
    invalid_timestamps: int = 0
    invalid_prices: int = 0
    invalid_spreads: int = 0
    stale_snapshots: int = 0


class DataCleaner:
    """
    Cleans and validates order book snapshots.
    """
    
    def __init__(
        self,
        max_spread_pct: float = 0.1,  # Max 10% spread
        min_price: float = 0.0,
        max_price: float = float("inf"),
        max_staleness_us: int = 60_000_000,  # 60 seconds
    ):
        self.max_spread_pct = max_spread_pct
        self.min_price = min_price
        self.max_price = max_price
        self.max_staleness_us = max_staleness_us
        
        self.stats = CleaningStats()
        self._last_valid_ts: int = 0
    
    def is_valid(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Validate a single snapshot.
        Returns True if valid, False otherwise.
        """
        self.stats.total_snapshots += 1
        
        # Check timestamp
        if snapshot.timestamp <= 0:
            self.stats.invalid_timestamps += 1
            return False
        
        # Check for staleness (if we have previous timestamp)
        if self._last_valid_ts > 0:
            if snapshot.timestamp < self._last_valid_ts:
                self.stats.invalid_timestamps += 1
                return False
            if snapshot.timestamp - self._last_valid_ts > self.max_staleness_us:
                self.stats.stale_snapshots += 1
                logger.warning(
                    "stale_snapshot",
                    gap_ms=(snapshot.timestamp - self._last_valid_ts) / 1000,
                )
        
        # Check prices exist
        if not snapshot.bids or not snapshot.asks:
            self.stats.invalid_prices += 1
            return False
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid is None or best_ask is None:
            self.stats.invalid_prices += 1
            return False
        
        # Check price bounds
        if not (self.min_price <= best_bid <= self.max_price):
            self.stats.invalid_prices += 1
            return False
        
        if not (self.min_price <= best_ask <= self.max_price):
            self.stats.invalid_prices += 1
            return False
        
        # Check crossed book
        if best_bid >= best_ask:
            self.stats.invalid_spreads += 1
            logger.warning("crossed_book", bid=best_bid, ask=best_ask)
            return False
        
        # Check spread
        spread_pct = (best_ask - best_bid) / best_bid
        if spread_pct > self.max_spread_pct:
            self.stats.invalid_spreads += 1
            return False
        
        self.stats.valid_snapshots += 1
        self._last_valid_ts = snapshot.timestamp
        return True
    
    def clean_batch(
        self, snapshots: list[OrderBookSnapshot]
    ) -> list[OrderBookSnapshot]:
        """
        Clean a batch of snapshots, removing invalid ones.
        """
        return [s for s in snapshots if self.is_valid(s)]
    
    def get_stats(self) -> dict:
        """Get cleaning statistics."""
        return {
            "total_snapshots": self.stats.total_snapshots,
            "valid_snapshots": self.stats.valid_snapshots,
            "invalid_timestamps": self.stats.invalid_timestamps,
            "invalid_prices": self.stats.invalid_prices,
            "invalid_spreads": self.stats.invalid_spreads,
            "stale_snapshots": self.stats.stale_snapshots,
            "acceptance_rate": (
                self.stats.valid_snapshots / self.stats.total_snapshots
                if self.stats.total_snapshots > 0
                else 0.0
            ),
        }
    
    def reset(self):
        """Reset statistics."""
        self.stats = CleaningStats()
        self._last_valid_ts = 0


def remove_outliers(
    values: np.ndarray,
    method: str = "iqr",
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Remove outliers from a numpy array.
    
    Args:
        values: Input array
        method: 'iqr' for interquartile range, 'zscore' for z-score
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        Boolean mask of valid values
    """
    if method == "iqr":
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (values >= lower) & (values <= upper)
    
    elif method == "zscore":
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.ones(len(values), dtype=bool)
        z_scores = np.abs((values - mean) / std)
        return z_scores <= threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


def forward_fill_gaps(
    timestamps: np.ndarray,
    values: np.ndarray,
    max_gap_us: int = 1_000_000,  # 1 second
) -> np.ndarray:
    """
    Forward fill small gaps in time series data.
    
    Args:
        timestamps: Timestamp array in microseconds
        values: Value array to fill
        max_gap_us: Maximum gap to fill
    
    Returns:
        Filled value array
    """
    filled = values.copy()
    
    for i in range(1, len(filled)):
        if np.isnan(filled[i]):
            gap = timestamps[i] - timestamps[i - 1]
            if gap <= max_gap_us and not np.isnan(filled[i - 1]):
                filled[i] = filled[i - 1]
    
    return filled
