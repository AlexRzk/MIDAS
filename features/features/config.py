"""
Configuration management for MIDAS Feature Generator.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class FeatureConfig:
    """Configuration for the feature generator service."""
    
    symbol: str
    clean_data_path: Path
    features_data_path: Path
    
    # Feature settings
    time_bucket_ms: int
    ofi_window: int
    order_book_depth: int
    parquet_row_group_size: int
    
    @classmethod
    def from_env(cls) -> "FeatureConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        
        return cls(
            symbol=os.getenv("SYMBOL", "btcusdt").upper(),
            clean_data_path=Path(os.getenv("CLEAN_DATA_PATH", "/data/clean")),
            features_data_path=Path(os.getenv("FEATURES_DATA_PATH", "/data/features")),
            time_bucket_ms=int(os.getenv("TIME_BUCKET_MS", "100")),
            ofi_window=int(os.getenv("OFI_WINDOW", "10")),
            order_book_depth=int(os.getenv("ORDER_BOOK_DEPTH", "10")),
            parquet_row_group_size=int(os.getenv("PARQUET_ROW_GROUP_SIZE", "50000")),
        )
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.clean_data_path.mkdir(parents=True, exist_ok=True)
        self.features_data_path.mkdir(parents=True, exist_ok=True)
