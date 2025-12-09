"""
Main feature generator service.
"""
import time
import os
import signal
import logging
from pathlib import Path
from datetime import datetime
import polars as pl
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .config import FeatureConfig
from .compute import time_bucket_aggregate
from .writer import FeatureWriter
from .normalize import FeatureNormalizer

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
# Optional Prometheus metrics
_prometheus_enabled = os.getenv("ENABLE_PROMETHEUS", "false").lower() in ("1", "true")
if _prometheus_enabled:
    try:
        from prometheus_client import start_http_server, Counter
        start_http_server(8001)
        FILES_PROCESSED = Counter("midas_features_files_processed_total", "Files processed by features")
    except Exception:
        logger.warning("prometheus_client_not_available")
        _prometheus_enabled = False


class NewFileHandler(FileSystemEventHandler):
    """Handler for new clean data files."""
    
    def __init__(self):
        self._pending_files = set()
    
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            if event.src_path.endswith(".parquet") and "clean_" in event.src_path:
                self._pending_files.add(event.src_path)
                logger.info("new_clean_file_detected", path=event.src_path)


class FeatureService:
    """
    Main feature generator service that:
    1. Reads cleaned order book data
    2. Computes features (OFI, imbalance, spread, etc.)
    3. Aggregates to time buckets
    4. Writes ML-ready Parquet files
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        # Prefer v2 compute if available (backwards compatible)
        try:
            from .compute_v2 import FeatureComputer as FeatureComputerV2, time_bucket_aggregate as tb_v2
            self.computer = FeatureComputerV2(
                depth=config.order_book_depth,
                ofi_window=config.ofi_window,
            )
            # prefer v2 aggregation function
            self.time_bucket_aggregate = tb_v2
            logger.info("using_feature_computer_v2")
        except Exception:
            from .compute import FeatureComputer as FeatureComputerV1
            self.computer = FeatureComputerV1(
                depth=config.order_book_depth,
                ofi_window=config.ofi_window,
            )
            logger.info("using_feature_computer_v1")
        # Default aggregation function
        if not hasattr(self, "time_bucket_aggregate"):
            self.time_bucket_aggregate = time_bucket_aggregate
        self.writer = FeatureWriter(
            output_path=config.features_data_path,
            row_group_size=config.parquet_row_group_size,
        )
        
        # Initialize normalizer
        scaler_dir = config.features_data_path.parent / "scalers"
        self.normalizer = FeatureNormalizer(scaler_dir)
        self._normalization_enabled = os.getenv("ENABLE_NORMALIZATION", "true").lower() in ("1", "true")
        self._is_first_file = True  # Track first file for fitting scalers
        
        self._running = True
        self._processed_files: set[str] = set()
        self._processed_files_path = config.features_data_path / ".processed_files"
        
        self._load_processed_files()
    
    def _load_processed_files(self):
        """Load list of already processed files."""
        if self._processed_files_path.exists():
            with open(self._processed_files_path) as f:
                self._processed_files = set(line.strip() for line in f)
            logger.info("loaded_processed_files", count=len(self._processed_files))
    
    def _save_processed_file(self, filepath: str):
        """Mark a file as processed."""
        self._processed_files.add(filepath)
        with open(self._processed_files_path, "a") as f:
            f.write(f"{filepath}\n")
    
    def list_clean_files(self) -> list[Path]:
        """List all clean data files."""
        return sorted(self.config.clean_data_path.glob("clean_*.parquet"))
    
    def process_file(self, filepath: Path):
        """Process a single clean data file."""
        filepath_str = str(filepath)
        
        if filepath_str in self._processed_files:
            logger.debug("skipping_already_processed", path=filepath_str)
            return
        
        logger.info("processing_clean_file", path=filepath_str)
        start_time = time.time()
        
        try:
            # Read clean data
            df = pl.read_parquet(filepath)
            
            if len(df) == 0:
                logger.warning("empty_file", path=filepath_str)
                self._save_processed_file(filepath_str)
                return
            
            # Compute features
            df = self.computer.compute_all_features(df)
            
            # Time bucket aggregation (optional)
            if self.config.time_bucket_ms > 0:
                df = self.time_bucket_aggregate(df, self.config.time_bucket_ms)
            
            # Select output columns (drop intermediate columns)
            output_cols = self._get_output_columns(df)
            df = df.select([c for c in output_cols if c in df.columns])
            
            # Apply normalization (if enabled)
            if self._normalization_enabled:
                df = self._apply_normalization(df, filepath_str)
            
            # Write to Parquet
            output_name = f"features_{filepath.stem.replace('clean_', '')}.parquet"
            self.writer.write(df, filename=output_name)
            if _prometheus_enabled:
                try:
                    FILES_PROCESSED.inc()
                except Exception:
                    pass
            
            elapsed = time.time() - start_time
            logger.info(
                "file_processed",
                path=filepath_str,
                elapsed_sec=round(elapsed, 2),
                input_rows=len(df),
                output_columns=len(df.columns),
            )
            
            self._save_processed_file(filepath_str)
            
        except Exception as e:
            logger.error("processing_error", path=filepath_str, error=str(e))
            raise
    
    def _get_output_columns(self, df: pl.DataFrame) -> list[str]:
        """Get the list of columns to include in output."""
        # Core columns
        cols = ["ts"]
        
        # Price/size levels
        for i in range(1, self.config.order_book_depth + 1):
            cols.extend([
                f"bid_px_{i:02d}",
                f"bid_sz_{i:02d}",
                f"ask_px_{i:02d}",
                f"ask_sz_{i:02d}",
            ])
        
        # Features
        cols.extend([
            "midprice",
            "spread",
            "spread_bps",
            "imbalance",
            "imbalance_1",
            "imbalance_5",
            "imbalance_10",
            "ofi",
            f"ofi_{self.config.ofi_window}",
            "ofi_cumulative",
            "microprice",
            "taker_buy_volume",
            "taker_sell_volume",
            "signed_volume",
            "volume_imbalance",
            "last_trade_px",
            "last_trade_qty",
            "liquidity_bid_1",
            "liquidity_ask_1",
            "liquidity_1",
            "liquidity_bid_5",
            "liquidity_ask_5",
            "liquidity_5",
            "liquidity_bid_10",
            "liquidity_ask_10",
            "liquidity_10",
            "returns",
            "volatility_20",
            "volatility_100",
            # Advanced microstructure features
            "kyle_lambda",
            "vpin",
            "bid_ladder_slope",
            "ask_ladder_slope",
            "bid_slope_ratio",
            "ask_slope_ratio",
            "queue_imb_1",
            "queue_imb_2",
            "queue_imb_3",
            "queue_imb_4",
            "queue_imb_5",
            "vol_of_vol",
            # OHLCV kline columns (computed from aggregation)
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "number_of_trades",
        ])
        
        return cols
    
    def _apply_normalization(self, df: pl.DataFrame, filepath: str) -> pl.DataFrame:
        """
        Apply feature normalization.
        
        On first file: fit scalers and save them
        On subsequent files: load and apply scalers
        """
        try:
            # Check if scalers already exist
            scaler_manifest = self.normalizer.scaler_dir / "normalization_manifest.json"
            
            if scaler_manifest.exists() and not self._is_first_file:
                # Load existing scalers and transform
                self.normalizer = FeatureNormalizer.load(self.normalizer.scaler_dir)
                df_norm = self.normalizer.transform(df)
                logger.info("applied_normalization", path=filepath, loaded_scalers=True)
            else:
                # First file or no scalers yet - fit and transform
                logger.info("fitting_normalization_scalers", path=filepath)
                df_norm = self.normalizer.fit_transform(df, is_training=True)
                self.normalizer.save()
                logger.info("saved_normalization_scalers", scaler_dir=str(self.normalizer.scaler_dir))
                self._is_first_file = False
            
            return df_norm
            
        except Exception as e:
            logger.error("normalization_error", path=filepath, error=str(e))
            logger.warning("skipping_normalization_for_this_file")
            return df
    
    def process_all_pending(self):
        """Process all unprocessed files."""
        files = self.list_clean_files()
        
        for filepath in files:
            if not self._running:
                break
            
            # Skip files that might still be written to
            try:
                mtime = filepath.stat().st_mtime
                if time.time() - mtime < 30:
                    logger.debug("skipping_recent_file", path=str(filepath))
                    continue
            except OSError:
                continue
            
            self.process_file(filepath)
    
    def run(self):
        """Main run loop."""
        logger.info(
            "feature_service_starting",
            clean_path=str(self.config.clean_data_path),
            features_path=str(self.config.features_data_path),
        )
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("shutdown_signal_received")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Set up file watcher
        event_handler = NewFileHandler()
        observer = Observer()
        observer.schedule(event_handler, str(self.config.clean_data_path), recursive=False)
        observer.start()
        
        try:
            while self._running:
                self.process_all_pending()
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        finally:
            observer.stop()
            observer.join()
            
            logger.info(
                "feature_service_stopped",
                files_processed=len(self._processed_files),
                writer_stats=self.writer.get_stats(),
            )


def main():
    """Entry point."""
    config = FeatureConfig.from_env()
    service = FeatureService(config)
    service.run()


if __name__ == "__main__":
    main()
