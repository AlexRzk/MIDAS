"""
Configuration management for MIDAS Processor.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class ProcessorConfig:
    """Configuration for the processor service."""
    
    symbol: str
    raw_data_path: Path
    clean_data_path: Path
    
    # Processing settings
    snapshot_interval_ms: int
    order_book_depth: int
    process_batch_size: int
    
    @classmethod
    def from_env(cls) -> "ProcessorConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        
        return cls(
            symbol=os.getenv("SYMBOL", "btcusdt").upper(),
            raw_data_path=Path(os.getenv("RAW_DATA_PATH", "/data/raw")),
            clean_data_path=Path(os.getenv("CLEAN_DATA_PATH", "/data/clean")),
            snapshot_interval_ms=int(os.getenv("SNAPSHOT_INTERVAL_MS", "100")),
            order_book_depth=int(os.getenv("ORDER_BOOK_DEPTH", "10")),
            process_batch_size=int(os.getenv("PROCESS_BATCH_SIZE", "100000")),
        )
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.clean_data_path.mkdir(parents=True, exist_ok=True)
