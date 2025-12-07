"""
Main processor service - orchestrates reading, reconstruction, and cleaning.
"""
import time
import signal
import sys
import logging
from pathlib import Path
from datetime import datetime
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from .config import ProcessorConfig
from .reader import RawDataReader
from .orderbook import OrderBookReconstructor
from .cleaner import DataCleaner
from .writer import CleanDataWriter

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


class NewFileHandler(FileSystemEventHandler):
    """Handler for new raw data files."""
    
    def __init__(self, callback):
        self.callback = callback
        self._pending_files = set()
    
    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            if event.src_path.endswith(".jsonl.zst"):
                self._pending_files.add(event.src_path)
                logger.info("new_raw_file_detected", path=event.src_path)


class ProcessorService:
    """
    Main processor service that:
    1. Reads raw WebSocket logs
    2. Reconstructs order book snapshots
    3. Cleans and validates data
    4. Writes clean intermediate data
    """
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.reader = RawDataReader(config.raw_data_path)
        self.reconstructor = OrderBookReconstructor(
            depth=config.order_book_depth,
            snapshot_interval_ms=config.snapshot_interval_ms,
        )
        self.cleaner = DataCleaner()
        
        self._running = True
        self._processed_files: set[str] = set()
        self._processed_files_path = config.clean_data_path / ".processed_files"
        
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
    
    def process_file(self, filepath: Path):
        """Process a single raw data file."""
        filepath_str = str(filepath)
        
        if filepath_str in self._processed_files:
            logger.debug("skipping_already_processed", path=filepath_str)
            return
        
        logger.info("processing_file", path=filepath_str)
        start_time = time.time()
        
        # Batch events for processing
        batch = []
        batch_size = self.config.process_batch_size
        
        with CleanDataWriter(
            self.config.clean_data_path,
            depth=self.config.order_book_depth,
        ) as writer:
            for event in self.reader.read_file(filepath):
                batch.append(event)
                
                if len(batch) >= batch_size:
                    snapshots = self.reconstructor.process_events(batch)
                    clean_snapshots = self.cleaner.clean_batch(snapshots)
                    writer.write_snapshots(clean_snapshots)
                    batch.clear()
            
            # Process remaining events
            if batch:
                snapshots = self.reconstructor.process_events(batch)
                clean_snapshots = self.cleaner.clean_batch(snapshots)
                writer.write_snapshots(clean_snapshots)
        
        elapsed = time.time() - start_time
        stats = self.reconstructor.get_stats()
        clean_stats = self.cleaner.get_stats()
        
        logger.info(
            "file_processed",
            path=filepath_str,
            elapsed_sec=round(elapsed, 2),
            events=stats["events_processed"],
            sequence_errors=stats["sequence_errors"],
            valid_snapshots=clean_stats["valid_snapshots"],
            acceptance_rate=round(clean_stats["acceptance_rate"], 4),
        )
        
        self._save_processed_file(filepath_str)
    
    def process_all_pending(self):
        """Process all unprocessed files."""
        files = self.reader.list_files(self.config.symbol)
        
        for filepath in files:
            if not self._running:
                break
            
            # Skip files that might still be written to
            # (check if file was modified in last 30 seconds)
            try:
                mtime = filepath.stat().st_mtime
                if time.time() - mtime < 30:
                    logger.debug("skipping_recent_file", path=str(filepath))
                    continue
            except OSError:
                continue
            
            self.process_file(filepath)
    
    def run(self):
        """Main run loop - process existing files and watch for new ones."""
        logger.info(
            "processor_starting",
            symbol=self.config.symbol,
            raw_path=str(self.config.raw_data_path),
            clean_path=str(self.config.clean_data_path),
        )
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("shutdown_signal_received")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Set up file watcher
        event_handler = NewFileHandler(self.process_file)
        observer = Observer()
        observer.schedule(event_handler, str(self.config.raw_data_path), recursive=False)
        observer.start()
        
        try:
            while self._running:
                # Process any pending files
                self.process_all_pending()
                
                # Sleep before next check
                time.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt")
        finally:
            observer.stop()
            observer.join()
            
            # Log final stats
            logger.info(
                "processor_stopped",
                files_processed=len(self._processed_files),
                reconstructor_stats=self.reconstructor.get_stats(),
                cleaner_stats=self.cleaner.get_stats(),
            )


def main():
    """Entry point."""
    config = ProcessorConfig.from_env()
    service = ProcessorService(config)
    service.run()


if __name__ == "__main__":
    main()
