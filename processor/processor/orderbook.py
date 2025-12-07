"""
Order Book reconstruction from incremental updates.
"""
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Snapshot of the order book at a point in time."""
    timestamp: int  # Exchange timestamp in microseconds
    local_timestamp: int  # Local arrival timestamp
    
    # Top N bid/ask levels
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    
    # Sequence tracking
    last_update_id: int = 0
    
    # Trade info (most recent)
    last_trade_price: Optional[float] = None
    last_trade_qty: Optional[float] = None
    taker_buy_volume: float = 0.0
    taker_sell_volume: float = 0.0
    
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
        """Total bid volume up to depth levels."""
        return sum(level.size for level in self.bids[:depth])
    
    def ask_volume_at_depth(self, depth: int) -> float:
        """Total ask volume up to depth levels."""
        return sum(level.size for level in self.asks[:depth])
    
    def imbalance(self, depth: int = 1) -> float:
        """
        Order book imbalance at specified depth.
        Returns (bid_vol - ask_vol) / (bid_vol + ask_vol)
        """
        bid_vol = self.bid_volume_at_depth(depth)
        ask_vol = self.ask_volume_at_depth(depth)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


class OrderBook:
    """
    Maintains the full order book state and applies incremental updates.
    """
    
    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        
        # Price -> Size mappings (sorted)
        self._bids: OrderedDict[float, float] = OrderedDict()
        self._asks: OrderedDict[float, float] = OrderedDict()
        
        # Sequence tracking
        self.last_update_id: int = 0
        self.first_update_id: int = 0
        
        # Trade tracking (for aggregation)
        self.last_trade_price: Optional[float] = None
        self.last_trade_qty: Optional[float] = None
        self.taker_buy_volume: float = 0.0
        self.taker_sell_volume: float = 0.0
        
        # Timestamps
        self.last_exchange_ts: int = 0
        self.last_local_ts: int = 0
        
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized and len(self._bids) > 0 and len(self._asks) > 0
    
    def apply_depth_update(
        self,
        bids: list[tuple[str, str]],
        asks: list[tuple[str, str]],
        first_update_id: int,
        last_update_id: int,
        prev_update_id: int,
        exchange_ts: int,
        local_ts: int,
    ) -> bool:
        """
        Apply an incremental depth update to the order book.
        Returns True if successful, False if update is out of sequence.
        """
        # Sequence validation
        if self._initialized:
            # Check for gaps in update sequence
            if prev_update_id != self.last_update_id:
                if first_update_id > self.last_update_id + 1:
                    logger.warning(
                        "sequence_gap",
                        expected=self.last_update_id + 1,
                        got_first=first_update_id,
                    )
                    return False
                elif last_update_id <= self.last_update_id:
                    # Duplicate or old update, skip
                    return True
        
        # Apply bid updates
        for price_str, size_str in bids:
            price = float(price_str)
            size = float(size_str)
            
            if size == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = size
        
        # Apply ask updates
        for price_str, size_str in asks:
            price = float(price_str)
            size = float(size_str)
            
            if size == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = size
        
        # Sort and trim
        self._sort_and_trim()
        
        # Update state
        self.last_update_id = last_update_id
        self.first_update_id = first_update_id
        self.last_exchange_ts = exchange_ts
        self.last_local_ts = local_ts
        self._initialized = True
        
        return True
    
    def apply_trade(
        self,
        price: str,
        quantity: str,
        buyer_is_maker: bool,
        exchange_ts: int,
        local_ts: int,
    ):
        """Apply a trade event."""
        price_f = float(price)
        qty_f = float(quantity)
        
        self.last_trade_price = price_f
        self.last_trade_qty = qty_f
        
        # buyer_is_maker = True means the trade was initiated by a seller (taker sell)
        if buyer_is_maker:
            self.taker_sell_volume += qty_f
        else:
            self.taker_buy_volume += qty_f
        
        self.last_exchange_ts = max(self.last_exchange_ts, exchange_ts)
        self.last_local_ts = max(self.last_local_ts, local_ts)
    
    def _sort_and_trim(self):
        """Sort order book levels and trim to max depth."""
        # Bids: highest price first
        sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])
        self._bids = OrderedDict(sorted_bids[:self.max_depth])
        
        # Asks: lowest price first
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])
        self._asks = OrderedDict(sorted_asks[:self.max_depth])
    
    def get_snapshot(self, depth: int = 10) -> OrderBookSnapshot:
        """Get a snapshot of the current order book state."""
        bids = [
            OrderBookLevel(price=p, size=s)
            for p, s in list(self._bids.items())[:depth]
        ]
        asks = [
            OrderBookLevel(price=p, size=s)
            for p, s in list(self._asks.items())[:depth]
        ]
        
        return OrderBookSnapshot(
            timestamp=self.last_exchange_ts,
            local_timestamp=self.last_local_ts,
            bids=bids,
            asks=asks,
            last_update_id=self.last_update_id,
            last_trade_price=self.last_trade_price,
            last_trade_qty=self.last_trade_qty,
            taker_buy_volume=self.taker_buy_volume,
            taker_sell_volume=self.taker_sell_volume,
        )
    
    def reset_trade_volumes(self):
        """Reset accumulated trade volumes (called after snapshot)."""
        self.taker_buy_volume = 0.0
        self.taker_sell_volume = 0.0
    
    def clear(self):
        """Clear all state."""
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = 0
        self.first_update_id = 0
        self.last_trade_price = None
        self.last_trade_qty = None
        self.taker_buy_volume = 0.0
        self.taker_sell_volume = 0.0
        self._initialized = False


class OrderBookReconstructor:
    """
    Reconstructs order book snapshots from raw WebSocket messages.
    """
    
    def __init__(self, depth: int = 10, snapshot_interval_ms: int = 100):
        self.depth = depth
        self.snapshot_interval_us = snapshot_interval_ms * 1000  # Convert to microseconds
        self.order_book = OrderBook(max_depth=100)
        
        self._last_snapshot_ts: int = 0
        self._snapshots: list[OrderBookSnapshot] = []
        self._events_processed: int = 0
        self._sequence_errors: int = 0
    
    def process_event(self, event: dict) -> Optional[OrderBookSnapshot]:
        """
        Process a single event and return a snapshot if interval elapsed.
        """
        event_type = event.get("type")
        exchange_ts = event.get("exchange_ts", 0)
        local_ts = event.get("local_ts", 0)
        
        if event_type == "depth":
            success = self.order_book.apply_depth_update(
                bids=event.get("bids", []),
                asks=event.get("asks", []),
                first_update_id=event.get("first_update_id", 0),
                last_update_id=event.get("last_update_id", 0),
                prev_update_id=event.get("prev_update_id", 0),
                exchange_ts=exchange_ts,
                local_ts=local_ts,
            )
            if not success:
                self._sequence_errors += 1
                
        elif event_type == "trade":
            trade = event.get("trade", {})
            self.order_book.apply_trade(
                price=trade.get("price", "0"),
                quantity=trade.get("quantity", "0"),
                buyer_is_maker=trade.get("buyer_is_maker", False),
                exchange_ts=exchange_ts,
                local_ts=local_ts,
            )
        
        self._events_processed += 1
        
        # Check if we should emit a snapshot
        if self.order_book.is_initialized:
            if self._last_snapshot_ts == 0:
                self._last_snapshot_ts = exchange_ts
            
            if exchange_ts - self._last_snapshot_ts >= self.snapshot_interval_us:
                snapshot = self.order_book.get_snapshot(self.depth)
                self._last_snapshot_ts = exchange_ts
                return snapshot
        
        return None
    
    def process_events(self, events: list[dict]) -> list[OrderBookSnapshot]:
        """
        Process a batch of events and return all generated snapshots.
        """
        snapshots = []
        
        # Sort events by exchange timestamp
        sorted_events = sorted(events, key=lambda x: (x.get("exchange_ts", 0), x.get("type") == "trade"))
        
        for event in sorted_events:
            snapshot = self.process_event(event)
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "events_processed": self._events_processed,
            "sequence_errors": self._sequence_errors,
            "order_book_initialized": self.order_book.is_initialized,
            "last_update_id": self.order_book.last_update_id,
        }
    
    def reset(self):
        """Reset state."""
        self.order_book.clear()
        self._last_snapshot_ts = 0
        self._events_processed = 0
        self._sequence_errors = 0
