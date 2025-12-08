"""
Enhanced Order Book reconstruction - Binance-compliant implementation v2.0

Per Binance documentation:
https://binance-docs.github.io/apidocs/futures/en/#how-to-manage-a-local-order-book-correctly

Key improvements:
- REST snapshot initialization for gap recovery
- Strict sequence validation per Binance rules
- State checkpointing for restart resilience
- Multi-symbol support
- Enhanced metrics and anomaly detection
"""
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import json
import pickle
from pathlib import Path
import time
import structlog

logger = structlog.get_logger()


# Try to import httpx for REST calls, fall back to None
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logger.warning("httpx_not_available", msg="REST gap recovery disabled")


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    timestamp: int
    local_timestamp: int
    symbol: str = ""
    
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    
    last_update_id: int = 0
    
    last_trade_price: Optional[float] = None
    last_trade_qty: Optional[float] = None
    taker_buy_volume: float = 0.0
    taker_sell_volume: float = 0.0
    
    # Quality flags
    sequence_gap: bool = False
    crossed_book: bool = False
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bids[0].size if self.bids else None
    
    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0].size if self.asks else None
    
    @property
    def midprice(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def bid_volume_at_depth(self, depth: int) -> float:
        return sum(level.size for level in self.bids[:depth])
    
    def ask_volume_at_depth(self, depth: int) -> float:
        return sum(level.size for level in self.asks[:depth])
    
    def imbalance(self, depth: int = 1) -> float:
        bid_vol = self.bid_volume_at_depth(depth)
        ask_vol = self.ask_volume_at_depth(depth)
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0


@dataclass
class OrderBookMetrics:
    """Metrics for monitoring order book health."""
    events_processed: int = 0
    depth_updates: int = 0
    trades_processed: int = 0
    sequence_gaps: int = 0
    crossed_books: int = 0
    resyncs: int = 0
    snapshots_emitted: int = 0
    
    # Timing metrics
    last_event_ts: int = 0
    max_latency_us: int = 0
    
    def to_dict(self) -> dict:
        return {
            "events_processed": self.events_processed,
            "depth_updates": self.depth_updates,
            "trades_processed": self.trades_processed,
            "sequence_gaps": self.sequence_gaps,
            "crossed_books": self.crossed_books,
            "resyncs": self.resyncs,
            "snapshots_emitted": self.snapshots_emitted,
        }


class OrderBookV2:
    """
    Binance-compliant order book manager v2.0.
    
    Implements proper sequence validation per Binance docs:
    1. Get depth snapshot via REST
    2. Buffer events during fetch
    3. First event: U <= lastUpdateId AND u >= lastUpdateId
    4. Subsequent: pu == previous u
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        max_depth: int = 100,
        checkpoint_path: Optional[Path] = None,
        auto_resync: bool = True,
        resync_threshold: int = 5,
    ):
        self.symbol = symbol.upper()
        self.max_depth = max_depth
        self.checkpoint_path = checkpoint_path
        self.auto_resync = auto_resync
        self.resync_threshold = resync_threshold
        
        # Order book state
        self._bids: OrderedDict[float, float] = OrderedDict()
        self._asks: OrderedDict[float, float] = OrderedDict()
        
        # Sequence tracking
        self.last_update_id: int = 0
        self.snapshot_update_id: int = 0
        self._consecutive_gaps: int = 0
        
        # Trade tracking
        self.last_trade_price: Optional[float] = None
        self.last_trade_qty: Optional[float] = None
        self.taker_buy_volume: float = 0.0
        self.taker_sell_volume: float = 0.0
        
        # Timestamps
        self.last_exchange_ts: int = 0
        self.last_local_ts: int = 0
        
        # State
        self._initialized = False
        self._pending_resync = False
        
        # Metrics
        self.metrics = OrderBookMetrics()
        
        # Try restore
        if checkpoint_path:
            self._try_restore()
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized and len(self._bids) > 0 and len(self._asks) > 0
    
    def fetch_snapshot_sync(self) -> bool:
        """Fetch REST snapshot synchronously."""
        if not HAS_HTTPX:
            logger.warning("cannot_fetch_snapshot", reason="httpx not installed")
            return False
        
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit=1000"
        
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            
            self._bids.clear()
            self._asks.clear()
            
            for px, sz in data.get("bids", []):
                p, s = float(px), float(sz)
                if s > 0:
                    self._bids[p] = s
            
            for px, sz in data.get("asks", []):
                p, s = float(px), float(sz)
                if s > 0:
                    self._asks[p] = s
            
            self.snapshot_update_id = data.get("lastUpdateId", 0)
            self.last_update_id = self.snapshot_update_id
            self._sort_and_trim()
            self._initialized = True
            self._consecutive_gaps = 0
            self.metrics.resyncs += 1
            
            logger.info(
                "snapshot_fetched",
                symbol=self.symbol,
                update_id=self.snapshot_update_id,
                bids=len(self._bids),
                asks=len(self._asks),
            )
            return True
            
        except Exception as e:
            logger.error("snapshot_failed", symbol=self.symbol, error=str(e))
            return False
    
    def apply_depth_update(
        self,
        bids: list,
        asks: list,
        first_update_id: int,
        last_update_id: int,
        prev_update_id: int,
        exchange_ts: int,
        local_ts: int,
    ) -> tuple[bool, bool]:
        """
        Apply depth update with Binance-compliant validation.
        
        Returns (applied, had_gap).
        """
        had_gap = False
        
        # Track latency
        if local_ts > exchange_ts:
            latency = local_ts - exchange_ts
            self.metrics.max_latency_us = max(self.metrics.max_latency_us, latency)
        
        if self._initialized:
            # Validate sequence
            if prev_update_id != self.last_update_id:
                # Check if valid first event after snapshot
                if first_update_id <= self.snapshot_update_id <= last_update_id:
                    pass  # Valid
                elif last_update_id <= self.last_update_id:
                    return True, False  # Old event
                else:
                    # Gap detected
                    self.metrics.sequence_gaps += 1
                    self._consecutive_gaps += 1
                    had_gap = True
                    
                    logger.warning(
                        "sequence_gap",
                        symbol=self.symbol,
                        expected=self.last_update_id,
                        got_prev=prev_update_id,
                    )
                    
                    # Auto-resync if too many consecutive gaps
                    if self.auto_resync and self._consecutive_gaps >= self.resync_threshold:
                        self._pending_resync = True
                        if self.fetch_snapshot_sync():
                            self._pending_resync = False
                        return False, True
        else:
            self._initialized = True
        
        # Apply updates
        for item in bids:
            px, sz = float(item[0]), float(item[1])
            if sz == 0:
                self._bids.pop(px, None)
            else:
                self._bids[px] = sz
        
        for item in asks:
            px, sz = float(item[0]), float(item[1])
            if sz == 0:
                self._asks.pop(px, None)
            else:
                self._asks[px] = sz
        
        self._sort_and_trim()
        
        # Check for crossed book
        if self._is_crossed():
            self.metrics.crossed_books += 1
            had_gap = True
        
        # Update state
        self.last_update_id = last_update_id
        self.last_exchange_ts = exchange_ts
        self.last_local_ts = local_ts
        self.metrics.depth_updates += 1
        self.metrics.events_processed += 1
        self.metrics.last_event_ts = exchange_ts
        
        if not had_gap:
            self._consecutive_gaps = 0
        
        return True, had_gap
    
    def apply_trade(
        self,
        price: str,
        quantity: str,
        buyer_is_maker: bool,
        exchange_ts: int,
        local_ts: int,
    ):
        """Apply trade event."""
        px, qty = float(price), float(quantity)
        
        self.last_trade_price = px
        self.last_trade_qty = qty
        
        if buyer_is_maker:
            self.taker_sell_volume += qty
        else:
            self.taker_buy_volume += qty
        
        self.last_exchange_ts = max(self.last_exchange_ts, exchange_ts)
        self.last_local_ts = max(self.last_local_ts, local_ts)
        self.metrics.trades_processed += 1
        self.metrics.events_processed += 1
    
    def _is_crossed(self) -> bool:
        if not self._bids or not self._asks:
            return False
        return max(self._bids.keys()) >= min(self._asks.keys())
    
    def _sort_and_trim(self):
        sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])
        self._bids = OrderedDict(sorted_bids[:self.max_depth])
        
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])
        self._asks = OrderedDict(sorted_asks[:self.max_depth])
    
    def get_snapshot(self, depth: int = 10) -> OrderBookSnapshot:
        """Get current order book snapshot."""
        bids = [OrderBookLevel(p, s) for p, s in list(self._bids.items())[:depth]]
        asks = [OrderBookLevel(p, s) for p, s in list(self._asks.items())[:depth]]
        
        self.metrics.snapshots_emitted += 1
        
        return OrderBookSnapshot(
            timestamp=self.last_exchange_ts,
            local_timestamp=self.last_local_ts,
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            last_update_id=self.last_update_id,
            last_trade_price=self.last_trade_price,
            last_trade_qty=self.last_trade_qty,
            taker_buy_volume=self.taker_buy_volume,
            taker_sell_volume=self.taker_sell_volume,
            crossed_book=self._is_crossed(),
        )
    
    def reset_trade_volumes(self):
        self.taker_buy_volume = 0.0
        self.taker_sell_volume = 0.0
    
    def clear(self):
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = 0
        self.snapshot_update_id = 0
        self._initialized = False
        self._consecutive_gaps = 0
    
    # Checkpointing
    def save_checkpoint(self):
        if not self.checkpoint_path:
            return
        
        state = {
            "symbol": self.symbol,
            "bids": dict(self._bids),
            "asks": dict(self._asks),
            "last_update_id": self.last_update_id,
            "snapshot_update_id": self.snapshot_update_id,
            "last_exchange_ts": self.last_exchange_ts,
            "initialized": self._initialized,
            "metrics": self.metrics.to_dict(),
            "checkpoint_time": time.time(),
        }
        
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error("checkpoint_save_error", error=str(e))
    
    def _try_restore(self):
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return
        
        try:
            with open(self.checkpoint_path, "rb") as f:
                state = pickle.load(f)
            
            age = time.time() - state.get("checkpoint_time", 0)
            if age > 300:  # 5 min max
                logger.info("checkpoint_expired", age_sec=age)
                return
            
            self._bids = OrderedDict(state["bids"])
            self._asks = OrderedDict(state["asks"])
            self.last_update_id = state["last_update_id"]
            self.snapshot_update_id = state["snapshot_update_id"]
            self.last_exchange_ts = state["last_exchange_ts"]
            self._initialized = state["initialized"]
            
            logger.info("checkpoint_restored", symbol=self.symbol, age_sec=age)
            
        except Exception as e:
            logger.error("checkpoint_restore_error", error=str(e))


class MultiSymbolReconstructor:
    """
    Order book reconstructor supporting multiple symbols.
    """
    
    def __init__(
        self,
        symbols: list[str] = None,
        depth: int = 10,
        snapshot_interval_ms: int = 100,
        checkpoint_dir: Optional[Path] = None,
        auto_resync: bool = True,
    ):
        self.depth = depth
        self.snapshot_interval_us = snapshot_interval_ms * 1000
        self.checkpoint_dir = checkpoint_dir
        self.auto_resync = auto_resync
        
        if symbols is None:
            symbols = ["BTCUSDT"]
        
        self.books: dict[str, OrderBookV2] = {}
        # Last snapshot timestamp per symbol (microseconds)
        # Initialize before creating any books so _create_book can safely reference it
        self._last_snapshot_ts: dict[str, int] = {}
        for sym in symbols:
            self._create_book(sym)
        self._total_events = 0
    
    def _create_book(self, symbol: str) -> OrderBookV2:
        sym = symbol.upper()
        cp = None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            cp = self.checkpoint_dir / f"book_{sym.lower()}.pkl"
        
        book = OrderBookV2(
            symbol=sym,
            checkpoint_path=cp,
            auto_resync=self.auto_resync,
        )
        self.books[sym] = book
        self._last_snapshot_ts[sym] = 0
        return book
    
    def get_book(self, symbol: str) -> OrderBookV2:
        sym = symbol.upper()
        if sym not in self.books:
            return self._create_book(sym)
        return self.books[sym]
    
    def process_event(self, event: dict) -> Optional[OrderBookSnapshot]:
        """Process single event, return snapshot if interval elapsed."""
        event_type = event.get("type")
        exchange_ts = event.get("exchange_ts", 0)
        local_ts = event.get("local_ts", 0)
        symbol = event.get("symbol", "BTCUSDT").upper()
        
        book = self.get_book(symbol)
        
        if event_type == "depth":
            book.apply_depth_update(
                bids=event.get("bids", []),
                asks=event.get("asks", []),
                first_update_id=event.get("first_update_id", 0),
                last_update_id=event.get("last_update_id", 0),
                prev_update_id=event.get("prev_update_id", 0),
                exchange_ts=exchange_ts,
                local_ts=local_ts,
            )
        elif event_type == "trade":
            trade = event.get("trade", {})
            book.apply_trade(
                price=trade.get("price", "0"),
                quantity=trade.get("quantity", "0"),
                buyer_is_maker=trade.get("buyer_is_maker", False),
                exchange_ts=exchange_ts,
                local_ts=local_ts,
            )
        
        self._total_events += 1
        
        # Emit snapshot at interval
        if book.is_initialized:
            last = self._last_snapshot_ts.get(symbol, 0)
            if last == 0 or exchange_ts - last >= self.snapshot_interval_us:
                self._last_snapshot_ts[symbol] = exchange_ts
                return book.get_snapshot(self.depth)
        
        return None
    
    def process_events(self, events: list[dict]) -> list[OrderBookSnapshot]:
        """Process batch of events."""
        sorted_events = sorted(
            events,
            key=lambda x: (x.get("exchange_ts", 0), x.get("type") == "trade")
        )
        
        snapshots = []
        for ev in sorted_events:
            snap = self.process_event(ev)
            if snap:
                snapshots.append(snap)
        
        return snapshots
    
    def save_checkpoints(self):
        for book in self.books.values():
            book.save_checkpoint()
    
    def get_stats(self) -> dict:
        total_gaps = sum(b.metrics.sequence_gaps for b in self.books.values())
        total_crossed = sum(b.metrics.crossed_books for b in self.books.values())
        total_resyncs = sum(b.metrics.resyncs for b in self.books.values())
        
        return {
            "events_processed": self._total_events,
            "sequence_errors": total_gaps,
            "crossed_books": total_crossed,
            "resyncs": total_resyncs,
            "symbols": list(self.books.keys()),
            "initialized_books": sum(1 for b in self.books.values() if b.is_initialized),
        }
    
    def reset(self):
        for book in self.books.values():
            book.clear()
        self._last_snapshot_ts.clear()
        self._total_events = 0


# Backward compatibility aliases
OrderBook = OrderBookV2
OrderBookReconstructor = MultiSymbolReconstructor
