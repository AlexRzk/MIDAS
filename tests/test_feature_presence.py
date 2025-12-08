"""
Tests for feature presence and writer functionality.
"""
import pytest
import polars as pl
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from features.writer import FeatureWriter
from features.compute import FeatureComputer, time_bucket_aggregate


class TestFeatureWriter:
    """Tests for FeatureWriter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature DataFrame."""
        return pl.DataFrame({
            "ts": [1705320000_000_000, 1705320000_100_000, 1705320000_200_000],
            "midprice": [100.0, 100.1, 100.2],
            "spread": [0.01, 0.02, 0.01],
            "spread_bps": [1.0, 2.0, 1.0],
            "ofi": [10.0, -5.0, 8.0],
            "imbalance": [0.1, -0.1, 0.05],
            "microprice": [100.05, 100.08, 100.15],
            "taker_buy_volume": [100.0, 150.0, 120.0],
            "taker_sell_volume": [80.0, 90.0, 100.0],
            "bid_px_01": [99.99, 100.0, 100.1],
            "ask_px_01": [100.0, 100.02, 100.11],
            "bid_sz_01": [10.0, 12.0, 11.0],
            "ask_sz_01": [8.0, 10.0, 9.0],
        })
    
    def test_writer_creates_file(self, temp_dir, sample_features):
        """Should create Parquet file."""
        writer = FeatureWriter(output_path=temp_dir)
        
        filepath = writer.write(sample_features, filename="test_features.parquet")
        
        assert filepath.exists()
        assert filepath.suffix == ".parquet"
    
    def test_writer_preserves_columns(self, temp_dir, sample_features):
        """Should preserve all columns in output."""
        writer = FeatureWriter(output_path=temp_dir)
        
        filepath = writer.write(sample_features, filename="test_features.parquet")
        
        # Read back
        df_read = pl.read_parquet(filepath)
        
        assert set(df_read.columns) == set(sample_features.columns)
    
    def test_writer_preserves_data(self, temp_dir, sample_features):
        """Should preserve data values."""
        writer = FeatureWriter(output_path=temp_dir)
        
        filepath = writer.write(sample_features, filename="test_features.parquet")
        df_read = pl.read_parquet(filepath)
        
        # Check values match
        assert df_read["midprice"].to_list() == sample_features["midprice"].to_list()
        assert df_read["ts"].to_list() == sample_features["ts"].to_list()
    
    def test_writer_stats(self, temp_dir, sample_features):
        """Should track write statistics."""
        writer = FeatureWriter(output_path=temp_dir)
        
        writer.write(sample_features, filename="test1.parquet")
        writer.write(sample_features, filename="test2.parquet")
        
        stats = writer.get_stats()
        
        assert stats["files_written"] == 2
        assert stats["rows_written"] == 6  # 3 rows * 2 files
    
    def test_writer_compression(self, temp_dir, sample_features):
        """Should apply ZSTD compression."""
        writer = FeatureWriter(
            output_path=temp_dir,
            compression="zstd",
            compression_level=3,
        )
        
        filepath = writer.write(sample_features, filename="test_compressed.parquet")
        
        # File should exist and be readable
        df_read = pl.read_parquet(filepath)
        assert len(df_read) == len(sample_features)


class TestFeaturePresence:
    """Tests for required feature columns."""
    
    REQUIRED_CORE_FEATURES = [
        "ts",
        "midprice",
        "spread",
        "ofi",
        "microprice",
    ]
    
    REQUIRED_IMBALANCE_FEATURES = [
        "imbalance",
        "imbalance_1",
    ]
    
    REQUIRED_VOLUME_FEATURES = [
        "taker_buy_volume",
        "taker_sell_volume",
    ]
    
    REQUIRED_KLINE_FEATURES = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "number_of_trades",
    ]
    
    @pytest.fixture
    def sample_orderbook_data(self):
        """Create sample order book data for feature computation."""
        n = 100
        base_ts = 1705320000_000_000
        
        return pl.DataFrame({
            "ts": [base_ts + i * 100_000 for i in range(n)],
            "bid_px_01": [100.0 - 0.01 + i * 0.001 for i in range(n)],
            "ask_px_01": [100.0 + 0.01 + i * 0.001 for i in range(n)],
            "bid_sz_01": [10.0 + i * 0.1 for i in range(n)],
            "ask_sz_01": [10.0 - i * 0.05 for i in range(n)],
            "bid_px_02": [99.9 - 0.01 + i * 0.001 for i in range(n)],
            "ask_px_02": [100.1 + 0.01 + i * 0.001 for i in range(n)],
            "bid_sz_02": [20.0] * n,
            "ask_sz_02": [20.0] * n,
            "taker_buy_vol": [1.0] * n,
            "taker_sell_vol": [0.5] * n,
            "last_trade_px": [100.0 + i * 0.001 for i in range(n)],
            "last_trade_qty": [0.1] * n,
        })
    
    def test_compute_produces_core_features(self, sample_orderbook_data):
        """FeatureComputer should produce core features."""
        computer = FeatureComputer(depth=10, ofi_window=10)
        
        result = computer.compute_all_features(sample_orderbook_data)
        
        for col in self.REQUIRED_CORE_FEATURES:
            assert col in result.columns, f"Missing core feature: {col}"
    
    def test_compute_produces_imbalance_features(self, sample_orderbook_data):
        """FeatureComputer should produce imbalance features."""
        computer = FeatureComputer(depth=10, ofi_window=10)
        
        result = computer.compute_all_features(sample_orderbook_data)
        
        for col in self.REQUIRED_IMBALANCE_FEATURES:
            assert col in result.columns, f"Missing imbalance feature: {col}"
    
    def test_compute_produces_volume_features(self, sample_orderbook_data):
        """FeatureComputer should produce volume features."""
        computer = FeatureComputer(depth=10, ofi_window=10)
        
        result = computer.compute_all_features(sample_orderbook_data)
        
        for col in self.REQUIRED_VOLUME_FEATURES:
            assert col in result.columns, f"Missing volume feature: {col}"
    
    def test_aggregation_produces_kline_features(self, sample_orderbook_data):
        """time_bucket_aggregate should produce kline features."""
        computer = FeatureComputer(depth=10, ofi_window=10)
        features = computer.compute_all_features(sample_orderbook_data)
        
        # Aggregate to 1-second buckets
        result = time_bucket_aggregate(features, bucket_ms=1000)
        
        # Check for OHLC columns
        for col in ["open", "high", "low", "close"]:
            assert col in result.columns, f"Missing kline feature: {col}"
    
    def test_aggregation_produces_vwap(self, sample_orderbook_data):
        """time_bucket_aggregate should compute VWAP."""
        computer = FeatureComputer(depth=10, ofi_window=10)
        features = computer.compute_all_features(sample_orderbook_data)
        
        result = time_bucket_aggregate(features, bucket_ms=1000)
        
        assert "vwap" in result.columns


class TestTimeBucketAggregateIntegration:
    """Integration tests for time bucket aggregation."""
    
    @pytest.fixture
    def hourly_data(self):
        """Create 1 hour of 100ms data."""
        n = 36_000  # 1 hour at 100ms
        base_ts = 1705320000_000_000
        
        return pl.DataFrame({
            "ts": [base_ts + i * 100_000 for i in range(n)],
            "midprice": [100.0 + (i % 100) * 0.01 for i in range(n)],
            "spread": [0.01] * n,
            "ofi": [1.0 if i % 2 == 0 else -1.0 for i in range(n)],
            "imbalance": [0.1 if i % 3 == 0 else -0.1 for i in range(n)],
            "microprice": [100.0 + (i % 100) * 0.01 for i in range(n)],
            "taker_buy_volume": [1.0] * n,
            "taker_sell_volume": [0.5] * n,
        })
    
    def test_1_minute_aggregation(self, hourly_data):
        """Should create ~60 buckets for 1 hour at 1-minute intervals."""
        result = time_bucket_aggregate(hourly_data, bucket_ms=60000)
        
        # 1 hour / 1 minute = 60 buckets
        assert len(result) == 60
    
    def test_5_minute_aggregation(self, hourly_data):
        """Should create ~12 buckets for 1 hour at 5-minute intervals."""
        result = time_bucket_aggregate(hourly_data, bucket_ms=300000)
        
        # 1 hour / 5 minutes = 12 buckets
        assert len(result) == 12
    
    def test_aggregation_volume_sum(self, hourly_data):
        """Should sum volumes correctly in aggregation."""
        result = time_bucket_aggregate(hourly_data, bucket_ms=60000)
        
        # Each minute has 600 samples (60s / 100ms)
        # Total buy volume per minute = 600 * 1.0 = 600
        first_bucket_buy_vol = result.filter(pl.col("ts") == result["ts"].min())["taker_buy_volume"][0]
        
        assert first_bucket_buy_vol == 600.0
    
    def test_aggregation_preserves_ts_column(self, hourly_data):
        """Should have ts column in output."""
        result = time_bucket_aggregate(hourly_data, bucket_ms=60000)
        
        assert "ts" in result.columns
        assert result["ts"].dtype in [pl.Int64, pl.UInt64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
