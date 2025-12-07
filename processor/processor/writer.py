"""
Clean data writer for intermediate storage.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import polars as pl
import structlog

from .orderbook import OrderBookSnapshot

logger = structlog.get_logger()


class CleanDataWriter:
    """
    Writes cleaned order book snapshots to intermediate storage.
    Uses Polars for efficient DataFrame operations.
    """
    
    def __init__(self, output_path: Path, depth: int = 10):
        self.output_path = output_path
        self.depth = depth
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._buffer: list[dict] = []
        self._buffer_size = 50000
        self._files_written = 0
    
    def _snapshot_to_dict(self, snapshot: OrderBookSnapshot) -> dict:
        """Convert a snapshot to a flat dictionary."""
        row = {
            "ts": snapshot.timestamp,
            "local_ts": snapshot.local_timestamp,
            "last_update_id": snapshot.last_update_id,
        }
        
        # Bid levels
        for i in range(self.depth):
            if i < len(snapshot.bids):
                row[f"bid_px_{i+1:02d}"] = snapshot.bids[i].price
                row[f"bid_sz_{i+1:02d}"] = snapshot.bids[i].size
            else:
                row[f"bid_px_{i+1:02d}"] = None
                row[f"bid_sz_{i+1:02d}"] = None
        
        # Ask levels
        for i in range(self.depth):
            if i < len(snapshot.asks):
                row[f"ask_px_{i+1:02d}"] = snapshot.asks[i].price
                row[f"ask_sz_{i+1:02d}"] = snapshot.asks[i].size
            else:
                row[f"ask_px_{i+1:02d}"] = None
                row[f"ask_sz_{i+1:02d}"] = None
        
        # Trade info
        row["last_trade_px"] = snapshot.last_trade_price
        row["last_trade_qty"] = snapshot.last_trade_qty
        row["taker_buy_vol"] = snapshot.taker_buy_volume
        row["taker_sell_vol"] = snapshot.taker_sell_volume
        
        return row
    
    def write_snapshot(self, snapshot: OrderBookSnapshot):
        """Add a snapshot to the buffer."""
        self._buffer.append(self._snapshot_to_dict(snapshot))
        
        if len(self._buffer) >= self._buffer_size:
            self.flush()
    
    def write_snapshots(self, snapshots: list[OrderBookSnapshot]):
        """Write multiple snapshots."""
        for snapshot in snapshots:
            self.write_snapshot(snapshot)
    
    def flush(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        # Create DataFrame
        df = pl.DataFrame(self._buffer)
        
        # Generate filename with timestamp
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"clean_{ts}_{self._files_written:04d}.parquet"
        filepath = self.output_path / filename
        
        # Write to Parquet
        df.write_parquet(
            filepath,
            compression="zstd",
            compression_level=3,
        )
        
        logger.info(
            "wrote_clean_data",
            file=str(filepath),
            rows=len(self._buffer),
        )
        
        self._buffer.clear()
        self._files_written += 1
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class CleanDataReader:
    """
    Reads cleaned data from intermediate Parquet files.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
    
    def list_files(self) -> list[Path]:
        """List all clean data files sorted by name."""
        return sorted(self.data_path.glob("clean_*.parquet"))
    
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
