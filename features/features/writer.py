"""
Parquet writer for feature data.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
import json

logger = structlog.get_logger()


class FeatureWriter:
    """
    Writes computed features to Parquet files with ZSTD compression.
    """
    
    def __init__(
        self,
        output_path: Path,
        row_group_size: int = 50000,
        compression: str = "zstd",
        compression_level: int = 3,
    ):
        self.output_path = output_path
        self.row_group_size = row_group_size
        self.compression = compression
        self.compression_level = compression_level
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._files_written = 0
        self._rows_written = 0
    
    def write(
        self,
        df: pl.DataFrame,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write a DataFrame to Parquet.
        
        Args:
            df: Polars DataFrame with features
            filename: Optional custom filename
            
        Returns:
            Path to written file
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"features_{ts}_{self._files_written:04d}.parquet"
        
        filepath = self.output_path / filename
        
        # Basic validation
        if "midprice" not in df.columns:
            logger.warning("midprice_missing", file=str(filepath))
        if "spread_bps" in df.columns:
            try:
                max_spread = float(df["spread_bps"].max())
                if max_spread > 10000:
                    logger.warning("abnormal_spread_bps", file=str(filepath), max_spread=max_spread)
            except Exception:
                pass

        # Write using Arrow so we can attach metadata
        metadata = {
            "midas_version": "2.0",
            "created_at": datetime.utcnow().isoformat(),
            "row_count": len(df),
        }

        try:
            table = pa.Table.from_batches(list(df.to_arrow().to_batches()))
            raw_md = {k: json.dumps(v).encode("utf8") for k, v in metadata.items()}
            table = table.replace_schema_metadata(raw_md)
            pq.write_table(table, filepath, compression=self.compression, compression_level=self.compression_level, row_group_size=self.row_group_size)
        except Exception:
            # Fallback to Polars
            df.write_parquet(
                filepath,
                compression=self.compression,
                compression_level=self.compression_level,
                row_group_size=self.row_group_size,
            )
        
        self._files_written += 1
        self._rows_written += len(df)
        
        logger.info(
            "wrote_feature_file",
            file=str(filepath),
            rows=len(df),
            columns=len(df.columns),
        )
        
        return filepath
    
    def get_stats(self) -> dict:
        """Get writer statistics."""
        return {
            "files_written": self._files_written,
            "rows_written": self._rows_written,
        }


class FeatureReader:
    """
    Reads feature data from Parquet files.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def list_files(self) -> list[Path]:
        """List all feature files sorted by name."""
        return sorted(self.data_path.glob("features_*.parquet"))
    
    def read_file(self, file_path: Path) -> pl.DataFrame:
        """Read a single Parquet file."""
        return pl.read_parquet(file_path)
    
    def read_all(self) -> pl.DataFrame:
        """Read and concatenate all files."""
        files = self.list_files()
        if not files:
            return pl.DataFrame()
        
        dfs = [self.read_file(f) for f in files]
        return pl.concat(dfs).sort("ts")
    
    def read_time_range(
        self,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> pl.DataFrame:
        """Read data within a time range."""
        df = self.read_all()
        
        if start_ts is not None:
            df = df.filter(pl.col("ts") >= start_ts)
        
        if end_ts is not None:
            df = df.filter(pl.col("ts") <= end_ts)
        
        return df
    
    def get_schema(self) -> dict:
        """Get schema of feature files."""
        files = self.list_files()
        if not files:
            return {}
        
        df = pl.read_parquet(files[0], n_rows=0)
        return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}


def create_example_schema() -> dict:
    """
    Create an example schema showing the output Parquet structure.
    
    This demonstrates the expected output format for ML consumption.
    """
    schema = {
        "ts": "int64",  # Timestamp in microseconds
        
        # Order book levels (top 10)
        **{f"bid_px_{i:02d}": "float64" for i in range(1, 11)},
        **{f"bid_sz_{i:02d}": "float64" for i in range(1, 11)},
        **{f"ask_px_{i:02d}": "float64" for i in range(1, 11)},
        **{f"ask_sz_{i:02d}": "float64" for i in range(1, 11)},
        
        # Core features
        "midprice": "float64",
        "spread": "float64",
        "spread_bps": "float64",
        "imbalance": "float64",
        "ofi": "float64",
        "microprice": "float64",
        
        # Trade features
        "taker_buy_volume": "float64",
        "taker_sell_volume": "float64",
        "signed_volume": "float64",
        "last_trade_px": "float64",
        "last_trade_qty": "float64",
        
        # Liquidity metrics
        "liquidity_bid_1": "float64",
        "liquidity_ask_1": "float64",
        "liquidity_1": "float64",
        "liquidity_bid_5": "float64",
        "liquidity_ask_5": "float64",
        "liquidity_5": "float64",
        "liquidity_bid_10": "float64",
        "liquidity_ask_10": "float64",
        "liquidity_10": "float64",
    }
    
    return schema


def print_example_row():
    """Print an example output row for documentation."""
    import json
    
    example = {
        "ts": 1701936000000000,  # 2023-12-07 12:00:00 UTC in microseconds
        
        # Top 3 bid levels (example)
        "bid_px_01": 43250.50,
        "bid_sz_01": 2.5,
        "bid_px_02": 43250.00,
        "bid_sz_02": 5.2,
        "bid_px_03": 43249.50,
        "bid_sz_03": 3.1,
        
        # Top 3 ask levels (example)
        "ask_px_01": 43251.00,
        "ask_sz_01": 1.8,
        "ask_px_02": 43251.50,
        "ask_sz_02": 4.3,
        "ask_px_03": 43252.00,
        "ask_sz_03": 2.9,
        
        # Features
        "midprice": 43250.75,
        "spread": 0.50,
        "spread_bps": 1.16,
        "imbalance": 0.163,  # (2.5 - 1.8) / (2.5 + 1.8)
        "ofi": 0.7,
        "microprice": 43250.66,
        
        # Trade features
        "taker_buy_volume": 15.3,
        "taker_sell_volume": 12.1,
        "signed_volume": 3.2,
        "last_trade_px": 43250.50,
        "last_trade_qty": 0.5,
        
        # Liquidity
        "liquidity_1": 4.3,
        "liquidity_5": 25.6,
        "liquidity_10": 48.2,
    }
    
    print("Example Parquet Row:")
    print(json.dumps(example, indent=2))
    return example
