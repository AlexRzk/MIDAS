"""
Tests for kline (OHLCV) computation from trade data.
"""
import pytest
import polars as pl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from features.kline import (
    compute_klines_from_trades,
    compute_klines_aggregated,
    validate_kline_columns,
    KLINE_COLUMNS,
)
from features.ts_utils import TimestampUnit


class TestKlineFromTrades:
    """Tests for computing klines from trade data."""
    
    @pytest.fixture
    def sample_trades(self):
        """Create deterministic sample trade data."""
        # 10 trades over 2 minutes
        base_ts = 1705320000_000_000  # microseconds
        
        return pl.DataFrame({
            "ts": [
                base_ts,                    # Minute 1, trade 1
                base_ts + 10_000_000,       # Minute 1, trade 2 (+10s)
                base_ts + 30_000_000,       # Minute 1, trade 3 (+30s)
                base_ts + 50_000_000,       # Minute 1, trade 4 (+50s)
                base_ts + 60_000_000,       # Minute 2, trade 1 (+60s)
                base_ts + 70_000_000,       # Minute 2, trade 2 (+70s)
                base_ts + 90_000_000,       # Minute 2, trade 3 (+90s)
                base_ts + 110_000_000,      # Minute 2, trade 4 (+110s)
            ],
            "last_trade_px": [
                100.0,  # Open minute 1
                101.0,  # High minute 1
                99.0,   # Low minute 1
                100.5,  # Close minute 1
                100.5,  # Open minute 2
                102.0,  # High minute 2
                98.0,   # Low minute 2
                101.0,  # Close minute 2
            ],
            "last_trade_qty": [
                1.0,
                2.0,
                1.5,
                0.5,  # Total minute 1: 5.0
                1.0,
                3.0,
                2.0,
                1.0,  # Total minute 2: 7.0
            ],
        })
    
    def test_kline_ohlc_values(self, sample_trades):
        """Should compute correct OHLC values."""
        result = compute_klines_from_trades(
            sample_trades,
            interval_ms=60000,  # 1 minute
            price_col="last_trade_px",
            qty_col="last_trade_qty",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        # Check that kline columns exist
        for col in ["open", "high", "low", "close"]:
            assert col in result.columns, f"Missing column: {col}"
        
        # Get unique klines (by bucket)
        klines = result.select(["open", "high", "low", "close"]).unique()
        
        # Should have 2 unique klines (2 minutes)
        assert len(klines) == 2
    
    def test_kline_volume(self, sample_trades):
        """Should compute correct volume."""
        result = compute_klines_from_trades(
            sample_trades,
            interval_ms=60000,
            price_col="last_trade_px",
            qty_col="last_trade_qty",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        assert "volume" in result.columns
        
        # Get unique volumes
        volumes = result["volume"].unique().sort()
        
        # Should have volumes 5.0 and 7.0
        assert len(volumes) == 2
    
    def test_kline_vwap(self, sample_trades):
        """Should compute correct VWAP."""
        result = compute_klines_from_trades(
            sample_trades,
            interval_ms=60000,
            price_col="last_trade_px",
            qty_col="last_trade_qty",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        assert "vwap" in result.columns
        
        # VWAP for minute 1: (100*1 + 101*2 + 99*1.5 + 100.5*0.5) / 5 = 100.3
        vwaps = result["vwap"].unique().sort()
        assert len(vwaps) == 2
    
    def test_kline_trade_count(self, sample_trades):
        """Should count trades per bucket."""
        result = compute_klines_from_trades(
            sample_trades,
            interval_ms=60000,
            price_col="last_trade_px",
            qty_col="last_trade_qty",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        assert "number_of_trades" in result.columns
        
        # Each minute has 4 trades
        trade_counts = result["number_of_trades"].unique()
        assert 4 in trade_counts.to_list()
    
    def test_kline_missing_price_column(self):
        """Should handle missing price column gracefully."""
        df = pl.DataFrame({
            "ts": [1705320000_000_000],
            "other_col": [1.0],
        })
        
        result = compute_klines_from_trades(
            df,
            interval_ms=60000,
            price_col="last_trade_px",  # Missing
            qty_col="last_trade_qty",
        )
        
        # Should have null kline columns
        assert "open" in result.columns
        assert result["open"].null_count() == len(result)
    
    def test_kline_auto_detect_ts_unit(self, sample_trades):
        """Should auto-detect timestamp unit."""
        result = compute_klines_from_trades(
            sample_trades,
            interval_ms=60000,
            price_col="last_trade_px",
            qty_col="last_trade_qty",
            ts_unit=None,  # Auto-detect
        )
        
        # Should successfully compute klines
        assert "open" in result.columns
        assert result["open"].null_count() < len(result)


class TestKlineAggregated:
    """Tests for aggregating snapshots to klines."""
    
    @pytest.fixture
    def sample_snapshots(self):
        """Create sample 100ms snapshots."""
        base_ts = 1705320000_000_000  # microseconds
        
        # 20 snapshots (2 seconds at 100ms)
        return pl.DataFrame({
            "ts": [base_ts + i * 100_000 for i in range(20)],
            "midprice": [100.0 + i * 0.1 for i in range(20)],
            "taker_buy_volume": [1.0] * 20,
            "taker_sell_volume": [0.5] * 20,
        })
    
    def test_aggregated_ohlc(self, sample_snapshots):
        """Should aggregate snapshots to OHLC."""
        result = compute_klines_aggregated(
            sample_snapshots,
            interval_ms=1000,  # 1 second
            price_col="midprice",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        # 2 seconds of data -> 2 klines
        assert len(result) == 2
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
    
    def test_aggregated_volume(self, sample_snapshots):
        """Should sum volumes correctly."""
        result = compute_klines_aggregated(
            sample_snapshots,
            interval_ms=1000,
            price_col="midprice",
            ts_unit=TimestampUnit.MICROSECONDS,
        )
        
        # Each second has 10 snapshots, each with 1.5 total volume
        # So each kline should have 15.0 volume
        assert "volume" in result.columns
        assert result["volume"][0] == 15.0


class TestKlineValidation:
    """Tests for kline column validation."""
    
    def test_validate_all_columns_present(self):
        """Should pass when all columns present."""
        df = pl.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
            "vwap": [100.2],
            "number_of_trades": [50],
        })
        
        result = validate_kline_columns(df)
        
        assert result["has_all_columns"] is True
        assert result["missing_columns"] == []
    
    def test_validate_missing_columns(self):
        """Should report missing columns."""
        df = pl.DataFrame({
            "open": [100.0],
            "close": [100.5],
        })
        
        result = validate_kline_columns(df)
        
        assert result["has_all_columns"] is False
        assert "high" in result["missing_columns"]
        assert "low" in result["missing_columns"]
        assert "volume" in result["missing_columns"]
    
    def test_validate_value_ranges(self):
        """Should compute value ranges."""
        df = pl.DataFrame({
            "open": [100.0, 105.0, 98.0],
            "high": [101.0, 106.0, 99.0],
            "low": [99.0, 104.0, 97.0],
            "close": [100.5, 105.5, 98.5],
            "volume": [1000.0, 2000.0, 1500.0],
            "vwap": [100.2, 105.2, 98.2],
            "number_of_trades": [50, 75, 60],
        })
        
        result = validate_kline_columns(df)
        
        assert "open" in result["value_ranges"]
        assert result["value_ranges"]["open"]["min"] == 98.0
        assert result["value_ranges"]["open"]["max"] == 105.0


class TestKlineColumnList:
    """Tests for KLINE_COLUMNS constant."""
    
    def test_kline_columns_complete(self):
        """Should have all standard kline columns."""
        expected = ["open", "high", "low", "close", "volume", "vwap", "number_of_trades"]
        
        for col in expected:
            assert col in KLINE_COLUMNS, f"Missing column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
