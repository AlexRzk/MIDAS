"""
Raw data reader for ZSTD-compressed JSONL files.
"""
import json
from pathlib import Path
from typing import Iterator, Optional
import zstandard as zstd
import structlog

logger = structlog.get_logger()


class RawDataReader:
    """
    Reads ZSTD-compressed JSONL files containing raw WebSocket messages.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.dctx = zstd.ZstdDecompressor()
    
    def list_files(self, symbol: Optional[str] = None) -> list[Path]:
        """
        List all raw data files, optionally filtered by symbol.
        Returns files sorted by name (chronological order).
        """
        pattern = f"{symbol.lower()}_*.jsonl.zst" if symbol else "*.jsonl.zst"
        files = sorted(self.data_path.glob(pattern))
        logger.info("found_raw_files", count=len(files), pattern=pattern)
        return files
    
    def read_file(self, file_path: Path) -> Iterator[dict]:
        """
        Read and decompress a single JSONL.zst file.
        Yields parsed JSON records.
        """
        logger.info("reading_file", path=str(file_path))
        
        try:
            with open(file_path, "rb") as f:
                with self.dctx.stream_reader(f) as reader:
                    # Read in chunks and process line by line
                    buffer = b""
                    while True:
                        chunk = reader.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        
                        buffer += chunk
                        lines = buffer.split(b"\n")
                        
                        # Process complete lines, keep partial line in buffer
                        for line in lines[:-1]:
                            if line.strip():
                                try:
                                    yield json.loads(line)
                                except json.JSONDecodeError as e:
                                    logger.warning("json_decode_error", error=str(e))
                        
                        buffer = lines[-1]
                    
                    # Process any remaining data
                    if buffer.strip():
                        try:
                            yield json.loads(buffer)
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
            logger.error("file_read_error", path=str(file_path), error=str(e))
            raise
    
    def read_all(self, symbol: Optional[str] = None) -> Iterator[dict]:
        """
        Read all files for a symbol in chronological order.
        """
        for file_path in self.list_files(symbol):
            yield from self.read_file(file_path)
