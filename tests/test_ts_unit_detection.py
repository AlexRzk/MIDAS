"""
Tests for timestamp unit detection and normalization.
"""
import pytest
import polars as pl
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

from features.ts_utils import (
    TimestampUnit,
    detect_timestamp_unit,
    normalize_timestamp,
    create_time_buckets,
    validate_timestamp_monotonicity,
)


class TestTimestampUnitDetection:
    """Tests for detect_timestamp_unit function."""
    
    def test_detect_microseconds(self):
        """Should detect microsecond timestamps correctly."""
        # Timestamp for 2024-01-15 12:00:00 in microseconds
        ts_us = 1705320000_000_000
        df = pl.DataFrame({"ts": [ts_us, ts_us + 100_000, ts_us + 200_000]})
        
        unit = detect_timestamp_unit(df, "ts")
        assert unit == TimestampUnit.MICROSECONDS
    
    def test_detect_milliseconds(self):
        """Should detect millisecond timestamps correctly."""
        # Timestamp for 2024-01-15 12:00:00 in milliseconds
        ts_ms = 1705320000_000
        df = pl.DataFrame({"ts": [ts_ms, ts_ms + 100, ts_ms + 200]})
        
        unit = detect_timestamp_unit(df, "ts")
        assert unit == TimestampUnit.MILLISECONDS
    
    def test_detect_seconds(self):
        """Should detect second timestamps correctly."""
        # Timestamp for 2024-01-15 12:00:00 in seconds
        ts_s = 1705320000
        df = pl.DataFrame({"ts": [ts_s, ts_s + 1, ts_s + 2]})
        
        unit = detect_timestamp_unit(df, "ts")
        assert unit == TimestampUnit.SECONDS
    
    def test_empty_dataframe(self):
        """Should return UNKNOWN for empty DataFrame."""
        df = pl.DataFrame({"ts": []}).cast({"ts": pl.Int64})
        
        unit = detect_timestamp_unit(df, "ts")
        assert unit == TimestampUnit.UNKNOWN
    
    def test_missing_column(self):
        """Should return UNKNOWN for missing column."""
        df = pl.DataFrame({"other": [1, 2, 3]})
        
        unit = detect_timestamp_unit(df, "ts")
        assert unit == TimestampUnit.UNKNOWN


class TestTimestampNormalization:
    """Tests for normalize_timestamp function."""
    
    def test_normalize_ms_to_us(self):
        """Should convert milliseconds to microseconds."""
        ts_ms = 1705320000_000
        df = pl.DataFrame({"ts": [ts_ms, ts_ms + 100]})
        
        result = normalize_timestamp(
            df,
            ts_col="ts",
            source_unit=TimestampUnit.MILLISECONDS,
            target_unit=TimestampUnit.MICROSECONDS,
        )
        
        expected_us = ts_ms * 1000
        assert result["ts"][0] == expected_us
    
    def test_normalize_us_to_ms(self):
        """Should convert microseconds to milliseconds."""
        ts_us = 1705320000_000_000
        df = pl.DataFrame({"ts": [ts_us, ts_us + 1000]})
        
        result = normalize_timestamp(
            df,
            ts_col="ts",
            source_unit=TimestampUnit.MICROSECONDS,
            target_unit=TimestampUnit.MILLISECONDS,
        )
        
        expected_ms = ts_us // 1000
        assert result["ts"][0] == expected_ms
    
    def test_normalize_same_unit(self):
        """Should not change values when source == target."""
        ts_us = 1705320000_000_000
        df = pl.DataFrame({"ts": [ts_us]})
        
        result = normalize_timestamp(
            df,
            ts_col="ts",
            source_unit=TimestampUnit.MICROSECONDS,
            target_unit=TimestampUnit.MICROSECONDS,
        )
        
        assert result["ts"][0] == ts_us
    
    def test_add_converted_columns(self):
        """Should add ts_us and ts_ms columns when requested."""
        ts_us = 1705320000_000_000
        df = pl.DataFrame({"ts": [ts_us]})
        
        result = normalize_timestamp(
            df,
            ts_col="ts",
            source_unit=TimestampUnit.MICROSECONDS,
            add_converted_columns=True,
        )
        
        assert "ts_us" in result.columns
        assert "ts_ms" in result.columns
        assert result["ts_us"][0] == ts_us
        assert result["ts_ms"][0] == ts_us // 1000


class TestTimeBuckets:
    """Tests for create_time_buckets function."""
    
    def test_bucket_creation_us_timestamps(self):
        """Should create correct buckets for microsecond timestamps."""
        # Create 10 timestamps, 100ms apart (100_000 us)
        base_ts = 1705320000_000_000
        timestamps = [base_ts + i * 100_000 for i in range(10)]
        df = pl.DataFrame({"ts": timestamps})
        
        # Create 1-second buckets (1000 ms)
        result = create_time_buckets(
            df,
            bucket_size=1000,  # 1000 ms = 1 second
            bucket_unit="ms",
            ts_col="ts",
        )
        
        # All 10 samples (1 second of data at 100ms intervals) should be in 1 bucket
        assert result["bucket"].n_unique() == 1
    
    def test_bucket_60_seconds(self):
        """Should create correct 60-second buckets."""
        # Create timestamps spanning 3 minutes
        base_ts = 1705320000_000_000  # microseconds
        # 180 samples at 1-second intervals
        timestamps = [base_ts + i * 1_000_000 for i in range(180)]
        df = pl.DataFrame({"ts": timestamps})
        
        # Create 1-minute buckets (60000 ms = 60 seconds)
        result = create_time_buckets(
            df,
            bucket_size=60000,  # 60000 ms = 60 seconds
            bucket_unit="ms",
            ts_col="ts",
        )
        
        # Should have 3 buckets (3 minutes of data)
        assert result["bucket"].n_unique() == 3
    
    def test_bucket_with_seconds_unit(self):
        """Should work with bucket_unit='s'."""
        base_ts = 1705320000_000_000  # microseconds
        timestamps = [base_ts + i * 1_000_000 for i in range(180)]  # 180 seconds
        df = pl.DataFrame({"ts": timestamps})
        
        # Create 60-second buckets using 's' unit
        result = create_time_buckets(
            df,
            bucket_size=60,  # 60 seconds
            bucket_unit="s",
            ts_col="ts",
        )
        
        assert result["bucket"].n_unique() == 3
    
    def test_bucket_auto_detect_unit(self):
        """Should auto-detect timestamp unit."""
        base_ts = 1705320000_000_000  # microseconds
        timestamps = [base_ts + i * 100_000 for i in range(600)]  # 60 seconds at 100ms
        df = pl.DataFrame({"ts": timestamps})
        
        # Create 1-minute buckets without specifying ts_unit
        result = create_time_buckets(
            df,
            bucket_size=60000,  # 60000 ms = 1 minute
            bucket_unit="ms",
            ts_col="ts",
            ts_unit=None,  # Auto-detect
        )
        
        assert result["bucket"].n_unique() == 1  # All in same minute


class TestTimestampMonotonicity:
    """Tests for validate_timestamp_monotonicity function."""
    
    def test_monotonic_timestamps(self):
        """Should return True for monotonically increasing timestamps."""
        df = pl.DataFrame({"ts": [100, 200, 300, 400]})
        
        is_mono, num_violations, indices = validate_timestamp_monotonicity(df, "ts")
        
        assert is_mono is True
        assert num_violations == 0
        assert indices == []
    
    def test_non_monotonic_timestamps(self):
        """Should detect non-monotonic timestamps."""
        df = pl.DataFrame({"ts": [100, 200, 150, 400]})  # 150 < 200
        
        is_mono, num_violations, indices = validate_timestamp_monotonicity(df, "ts")
        
        assert is_mono is False
        assert num_violations == 1
        assert 2 in indices  # Index 2 has the violation
    
    def test_single_row(self):
        """Should return True for single row."""
        df = pl.DataFrame({"ts": [100]})
        
        is_mono, num_violations, indices = validate_timestamp_monotonicity(df, "ts")
        
        assert is_mono is True


class TestBugFix680kRows:
    """
    Regression test for the bug where bucket_ms=60000 created 680k rows
    instead of ~1.2k for 1-minute aggregation.
    
    The bug was caused by applying bucket_ms directly to microsecond timestamps
    without unit conversion.
    """
    
    def test_60s_bucket_on_us_timestamps(self):
        """
        Ensure bucket_ms=60000 creates ~60-second buckets even when
        timestamps are in microseconds.
        """
        # Simulate ~22 hours of data at 100ms intervals (792,000 samples)
        # This is similar to the real data that had 680k rows
        base_ts = 1705320000_000_000  # Start timestamp in microseconds
        
        # Create 1 hour of 100ms data (36,000 samples)
        n_samples = 36_000
        timestamps = [base_ts + i * 100_000 for i in range(n_samples)]
        df = pl.DataFrame({"ts": timestamps})
        
        # Create 1-minute buckets
        result = create_time_buckets(
            df,
            bucket_size=60000,  # 60000 ms = 60 seconds = 1 minute
            bucket_unit="ms",
            ts_col="ts",
        )
        
        # Should have 60 buckets (1 hour / 1 minute)
        expected_buckets = 60
        actual_buckets = result["bucket"].n_unique()
        
        # Allow some tolerance for edge effects
        assert abs(actual_buckets - expected_buckets) <= 1, (
            f"Expected ~{expected_buckets} buckets, got {actual_buckets}. "
            f"This indicates the timestamp unit conversion is not working correctly."
        )
        
        # Verify rows per bucket is reasonable (~600 per minute at 100ms intervals)
        rows_per_bucket = n_samples / actual_buckets
        assert 500 <= rows_per_bucket <= 700, (
            f"Expected ~600 rows per bucket, got {rows_per_bucket:.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
