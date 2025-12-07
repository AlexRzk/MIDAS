"""
Data Quality Validation Module

Implements comprehensive data quality checks:
- No crossed books
- No negative volumes
- Monotonic timestamps
- Midprice stability bounds
- Raw JSON validation
- Microstructure anomaly detection (spoofing patterns)
"""
import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path
import json
import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    check_name: str
    message: str
    failed_count: int = 0
    total_count: int = 0
    details: dict = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return (self.total_count - self.failed_count) / self.total_count


@dataclass 
class DataQualityReport:
    """Comprehensive data quality report."""
    timestamp: str
    file_path: str
    total_rows: int
    results: list[ValidationResult] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return all(r.valid for r in self.results)
    
    @property
    def overall_quality_score(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.pass_rate for r in self.results) / len(self.results)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "total_rows": self.total_rows,
            "all_passed": self.all_passed,
            "quality_score": self.overall_quality_score,
            "checks": [
                {
                    "name": r.check_name,
                    "valid": r.valid,
                    "message": r.message,
                    "pass_rate": r.pass_rate,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class DataValidator:
    """
    Validates order book and feature data for quality issues.
    """
    
    def __init__(
        self,
        max_spread_bps: float = 100.0,  # 1%
        max_price_change_pct: float = 5.0,  # 5% per tick
        min_timestamp_delta_us: int = 0,
        max_timestamp_gap_us: int = 60_000_000,  # 60 seconds
        depth: int = 10,
    ):
        self.max_spread_bps = max_spread_bps
        self.max_price_change_pct = max_price_change_pct
        self.min_timestamp_delta_us = min_timestamp_delta_us
        self.max_timestamp_gap_us = max_timestamp_gap_us
        self.depth = depth
    
    def validate_no_crossed_book(self, df: pl.DataFrame) -> ValidationResult:
        """Check that bid < ask (no crossed book)."""
        if "bid_px_01" not in df.columns or "ask_px_01" not in df.columns:
            return ValidationResult(
                valid=True,
                check_name="no_crossed_book",
                message="Price columns not found, skipping check",
                total_count=len(df),
            )
        
        crossed = df.filter(pl.col("bid_px_01") >= pl.col("ask_px_01"))
        failed = len(crossed)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="no_crossed_book",
            message=f"Found {failed} crossed book instances" if failed > 0 else "No crossed books",
            failed_count=failed,
            total_count=len(df),
            details={
                "first_crossed_ts": crossed["ts"][0] if failed > 0 and "ts" in crossed.columns else None,
            },
        )
    
    def validate_no_negative_volumes(self, df: pl.DataFrame) -> ValidationResult:
        """Check that all volumes are non-negative."""
        vol_cols = [c for c in df.columns if "sz" in c or "vol" in c.lower() or "volume" in c.lower()]
        
        if not vol_cols:
            return ValidationResult(
                valid=True,
                check_name="no_negative_volumes",
                message="No volume columns found",
                total_count=len(df),
            )
        
        negative_mask = pl.lit(False)
        for col in vol_cols:
            negative_mask = negative_mask | (pl.col(col) < 0)
        
        negative_rows = df.filter(negative_mask)
        failed = len(negative_rows)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="no_negative_volumes",
            message=f"Found {failed} rows with negative volumes" if failed > 0 else "All volumes non-negative",
            failed_count=failed,
            total_count=len(df),
            details={"checked_columns": vol_cols},
        )
    
    def validate_monotonic_timestamps(self, df: pl.DataFrame) -> ValidationResult:
        """Check that timestamps are monotonically increasing."""
        if "ts" not in df.columns:
            return ValidationResult(
                valid=True,
                check_name="monotonic_timestamps",
                message="No timestamp column found",
                total_count=len(df),
            )
        
        # Check for non-increasing timestamps
        ts_diff = df.select([
            (pl.col("ts") - pl.col("ts").shift(1)).alias("ts_diff")
        ])
        
        non_monotonic = ts_diff.filter(pl.col("ts_diff") < self.min_timestamp_delta_us)
        failed = len(non_monotonic)
        
        # Also check for large gaps
        large_gaps = ts_diff.filter(pl.col("ts_diff") > self.max_timestamp_gap_us)
        gap_count = len(large_gaps)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="monotonic_timestamps",
            message=f"Found {failed} non-monotonic timestamps, {gap_count} large gaps" if failed > 0 or gap_count > 0 else "Timestamps monotonic",
            failed_count=failed,
            total_count=len(df),
            details={
                "large_gaps": gap_count,
                "max_gap_us": ts_diff["ts_diff"].max() if len(ts_diff) > 0 else 0,
            },
        )
    
    def validate_midprice_stability(self, df: pl.DataFrame) -> ValidationResult:
        """Check that midprice doesn't jump too much between ticks."""
        if "bid_px_01" not in df.columns or "ask_px_01" not in df.columns:
            return ValidationResult(
                valid=True,
                check_name="midprice_stability",
                message="Price columns not found",
                total_count=len(df),
            )
        
        # Calculate midprice and its changes
        mid = (df["bid_px_01"] + df["ask_px_01"]) / 2
        mid_pct_change = ((mid - mid.shift(1)) / mid.shift(1) * 100).abs()
        
        # Filter out NaN and check for large jumps
        large_jumps = mid_pct_change.filter(mid_pct_change > self.max_price_change_pct)
        failed = len(large_jumps)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="midprice_stability",
            message=f"Found {failed} large price jumps (>{self.max_price_change_pct}%)" if failed > 0 else "Midprice stable",
            failed_count=failed,
            total_count=len(df),
            details={
                "max_jump_pct": float(mid_pct_change.max()) if len(mid_pct_change) > 0 else 0,
                "threshold_pct": self.max_price_change_pct,
            },
        )
    
    def validate_spread_bounds(self, df: pl.DataFrame) -> ValidationResult:
        """Check that spread is within reasonable bounds."""
        if "bid_px_01" not in df.columns or "ask_px_01" not in df.columns:
            return ValidationResult(
                valid=True,
                check_name="spread_bounds",
                message="Price columns not found",
                total_count=len(df),
            )
        
        mid = (df["bid_px_01"] + df["ask_px_01"]) / 2
        spread_bps = (df["ask_px_01"] - df["bid_px_01"]) / mid * 10000
        
        wide_spreads = spread_bps.filter(spread_bps > self.max_spread_bps)
        failed = len(wide_spreads)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="spread_bounds",
            message=f"Found {failed} rows with spread > {self.max_spread_bps} bps" if failed > 0 else "Spread within bounds",
            failed_count=failed,
            total_count=len(df),
            details={
                "max_spread_bps": float(spread_bps.max()) if len(spread_bps) > 0 else 0,
                "mean_spread_bps": float(spread_bps.mean()) if len(spread_bps) > 0 else 0,
            },
        )
    
    def validate_price_levels_ordered(self, df: pl.DataFrame) -> ValidationResult:
        """Check that price levels are correctly ordered (bid descending, ask ascending)."""
        failed = 0
        
        # Check bids are descending
        for i in range(1, self.depth):
            bid_col = f"bid_px_{i:02d}"
            bid_next = f"bid_px_{i+1:02d}"
            
            if bid_col in df.columns and bid_next in df.columns:
                inverted = df.filter(
                    (pl.col(bid_col) < pl.col(bid_next)) & 
                    (pl.col(bid_next) > 0)
                )
                failed += len(inverted)
        
        # Check asks are ascending
        for i in range(1, self.depth):
            ask_col = f"ask_px_{i:02d}"
            ask_next = f"ask_px_{i+1:02d}"
            
            if ask_col in df.columns and ask_next in df.columns:
                inverted = df.filter(
                    (pl.col(ask_col) > pl.col(ask_next)) & 
                    (pl.col(ask_next) > 0)
                )
                failed += len(inverted)
        
        return ValidationResult(
            valid=failed == 0,
            check_name="price_levels_ordered",
            message=f"Found {failed} rows with inverted price levels" if failed > 0 else "Price levels correctly ordered",
            failed_count=failed,
            total_count=len(df),
        )
    
    def detect_spoofing_patterns(self, df: pl.DataFrame, window: int = 10) -> ValidationResult:
        """
        Detect potential spoofing patterns.
        
        Spoofing indicators:
        - Large orders placed and quickly removed
        - Orders that don't trade but move the market
        - Layering (multiple large orders at different levels)
        """
        if "bid_sz_01" not in df.columns:
            return ValidationResult(
                valid=True,
                check_name="spoofing_detection",
                message="Size columns not found",
                total_count=len(df),
            )
        
        # Detect rapid size changes (potential spoofing)
        df_analysis = df.with_columns([
            (pl.col("bid_sz_01") - pl.col("bid_sz_01").shift(1)).alias("bid_size_change"),
            (pl.col("ask_sz_01") - pl.col("ask_sz_01").shift(1)).alias("ask_size_change"),
        ])
        
        # Large additions followed by quick removals
        df_analysis = df_analysis.with_columns([
            pl.col("bid_size_change").rolling_sum(window).alias("bid_change_sum"),
            pl.col("ask_size_change").rolling_sum(window).alias("ask_change_sum"),
        ])
        
        # Pattern: large positive then large negative within window
        # This is a simplified heuristic
        suspicious = df_analysis.filter(
            (pl.col("bid_change_sum").abs() > pl.col("bid_sz_01") * 2) |
            (pl.col("ask_change_sum").abs() > pl.col("ask_sz_01") * 2)
        )
        
        suspicious_count = len(suspicious)
        
        return ValidationResult(
            valid=True,  # Informational, not a hard failure
            check_name="spoofing_detection",
            message=f"Detected {suspicious_count} potential spoofing patterns" if suspicious_count > 0 else "No suspicious patterns detected",
            failed_count=0,  # Don't fail on this
            total_count=len(df),
            details={
                "suspicious_patterns": suspicious_count,
                "window_size": window,
            },
        )
    
    def validate_dataframe(self, df: pl.DataFrame, file_path: str = "") -> DataQualityReport:
        """Run all validation checks on a DataFrame."""
        from datetime import datetime
        
        report = DataQualityReport(
            timestamp=datetime.utcnow().isoformat(),
            file_path=file_path,
            total_rows=len(df),
        )
        
        # Run all checks
        report.results.append(self.validate_no_crossed_book(df))
        report.results.append(self.validate_no_negative_volumes(df))
        report.results.append(self.validate_monotonic_timestamps(df))
        report.results.append(self.validate_midprice_stability(df))
        report.results.append(self.validate_spread_bounds(df))
        report.results.append(self.validate_price_levels_ordered(df))
        report.results.append(self.detect_spoofing_patterns(df))
        
        # Log summary
        logger.info(
            "data_quality_validated",
            file=file_path,
            total_rows=len(df),
            all_passed=report.all_passed,
            quality_score=round(report.overall_quality_score, 4),
        )
        
        return report


class RawJsonValidator:
    """Validates raw JSONL messages before compression."""
    
    REQUIRED_DEPTH_FIELDS = ["exchange_ts", "local_ts", "type", "symbol", "bids", "asks"]
    REQUIRED_TRADE_FIELDS = ["exchange_ts", "local_ts", "type", "symbol", "trade"]
    
    def __init__(self):
        self.valid_count = 0
        self.invalid_count = 0
        self.errors: list[str] = []
    
    def validate_message(self, msg: dict) -> tuple[bool, Optional[str]]:
        """Validate a single message."""
        msg_type = msg.get("type")
        
        if msg_type == "depth":
            return self._validate_depth(msg)
        elif msg_type == "trade":
            return self._validate_trade(msg)
        elif msg.get("_header") or msg.get("_footer"):
            return True, None  # Metadata is valid
        else:
            return False, f"Unknown message type: {msg_type}"
    
    def _validate_depth(self, msg: dict) -> tuple[bool, Optional[str]]:
        """Validate depth message."""
        for field in self.REQUIRED_DEPTH_FIELDS:
            if field not in msg:
                return False, f"Missing required field: {field}"
        
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        
        # Validate bid/ask structure
        for bid in bids:
            if len(bid) != 2:
                return False, f"Invalid bid format: {bid}"
            try:
                float(bid[0])
                float(bid[1])
            except (ValueError, TypeError):
                return False, f"Invalid bid values: {bid}"
        
        for ask in asks:
            if len(ask) != 2:
                return False, f"Invalid ask format: {ask}"
            try:
                float(ask[0])
                float(ask[1])
            except (ValueError, TypeError):
                return False, f"Invalid ask values: {ask}"
        
        return True, None
    
    def _validate_trade(self, msg: dict) -> tuple[bool, Optional[str]]:
        """Validate trade message."""
        for field in self.REQUIRED_TRADE_FIELDS:
            if field not in msg:
                return False, f"Missing required field: {field}"
        
        trade = msg.get("trade", {})
        required_trade_fields = ["trade_id", "price", "quantity"]
        
        for field in required_trade_fields:
            if field not in trade:
                return False, f"Missing trade field: {field}"
        
        return True, None
    
    def validate_line(self, line: str) -> tuple[bool, Optional[str]]:
        """Validate a single JSONL line."""
        try:
            msg = json.loads(line)
            valid, error = self.validate_message(msg)
            
            if valid:
                self.valid_count += 1
            else:
                self.invalid_count += 1
                if error:
                    self.errors.append(error)
            
            return valid, error
            
        except json.JSONDecodeError as e:
            self.invalid_count += 1
            error = f"JSON parse error: {e}"
            self.errors.append(error)
            return False, error
    
    def get_stats(self) -> dict:
        """Get validation statistics."""
        total = self.valid_count + self.invalid_count
        return {
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "total_count": total,
            "valid_rate": self.valid_count / total if total > 0 else 1.0,
            "unique_errors": len(set(self.errors)),
        }
    
    def reset(self):
        """Reset counters."""
        self.valid_count = 0
        self.invalid_count = 0
        self.errors = []


def create_parquet_metadata(
    symbol: str,
    snapshot_interval_ms: int,
    order_book_depth: int,
    start_ts: int,
    end_ts: int,
    row_count: int,
    quality_report: Optional[DataQualityReport] = None,
) -> dict:
    """
    Create metadata for Parquet file footer.
    
    This metadata helps downstream consumers understand the data.
    """
    from datetime import datetime
    
    metadata = {
        "midas_version": "2.0",
        "created_at": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "snapshot_interval_ms": snapshot_interval_ms,
        "order_book_depth": order_book_depth,
        "time_range": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_sec": (end_ts - start_ts) / 1_000_000,
        },
        "row_count": row_count,
    }
    
    if quality_report:
        metadata["quality"] = {
            "score": quality_report.overall_quality_score,
            "all_passed": quality_report.all_passed,
        }
    
    return metadata
