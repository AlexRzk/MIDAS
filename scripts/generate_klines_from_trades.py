"""
Generate OHLCV klines from raw trade events stored in data/raw.

Usage:
    python scripts/generate_klines_from_trades.py --interval 1m --source raw --output data/features/klines_1m.parquet

Options:
    --interval   Interval in minutes (e.g., 1m, 5m). Default: 1m
    --source     Where to read trades from: raw|clean. Default: raw
    --symbol     Symbol filter (e.g., btcusdt). Default: all
    --output     Output Parquet path. Default: data/features/klines_<interval>.parquet
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Iterator

import zstandard as zstd
import polars as pl


def parse_interval(interval: str) -> int:
    # simple util: supports '1m', '5m', '15m', '1h'
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return minutes * 60_000
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return hours * 60 * 60_000
    raise ValueError("Unsupported interval format. Use '1m', '5m', '1h', etc.")


def iter_trade_events_from_raw(path_pattern: str, symbol: str | None) -> Iterator[dict]:
    for fn in glob.glob(path_pattern):
        with open(fn, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buffer = b""
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    buffer += chunk
                    parts = buffer.split(b"\n")
                    for line in parts[:-1]:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # Collector writes: {"type": "trade", "symbol": "...", "trade": {"trade_id":..., "price":"...", "quantity":"...", "buyer_is_maker":...}, "exchange_ts":...}
                        if obj.get("type") == "trade":
                            trade = obj.get("trade", {})
                            sym = obj.get("symbol")
                            if sym and symbol and sym.lower() != symbol.lower():
                                continue
                            ts = obj.get("exchange_ts")
                            price_str = trade.get("price", "0")
                            qty_str = trade.get("quantity", "0")
                            yield {
                                "ts": int(ts) if ts is not None else None,
                                "price": float(price_str),
                                "quantity": float(qty_str),
                                "symbol": sym,
                            }
                    buffer = parts[-1]
                # last part
                if buffer.strip():
                    try:
                        obj = json.loads(buffer)
                    except Exception:
                        obj = None
                    if obj and obj.get("type") == "trade":
                        trade = obj.get("trade", {})
                        ts = obj.get("exchange_ts")
                        price_str = trade.get("price", "0")
                        qty_str = trade.get("quantity", "0")
                        yield {
                            "ts": int(ts) if ts is not None else None,
                            "price": float(price_str),
                            "quantity": float(qty_str),
                            "symbol": obj.get("symbol"),
                        }


def iter_trade_events_from_clean(clean_dir: str, symbol: str | None) -> Iterator[dict]:
    # Parse 'clean_*.parquet' files that contain last_trade_px / last_trade_qty & local ts
    # We assume each 'snapshot' carries a last_trade_px & last_trade_qty
    files = sorted(glob.glob(os.path.join(clean_dir, "clean_*.parquet")))
    for fn in files:
        try:
            df = pl.read_parquet(fn)
        except Exception:
            continue
        if symbol:
            # clean df may not have symbol column per snapshot; so skip this filter
            pass
        if "last_trade_px" not in df.columns:
            continue
        # Build trade-like rows for each snapshot that had a last_trade_px non-null
        df2 = df.filter(pl.col("last_trade_px").is_not_null())
        if df2.is_empty():
            continue
        for row in df2.select(["ts", "last_trade_px", "last_trade_qty"]).iter_rows():
            ts, px, qty = row
            yield {"ts": int(ts), "price": float(px), "quantity": float(qty), "symbol": symbol}


def make_candles(trades_df: pl.DataFrame, interval_ms: int) -> pl.DataFrame:
    # Convert ts in ms to bucket timestamp (floor) by interval
    trades_df = trades_df.lazy().with_columns([
        (pl.col("ts") // interval_ms * interval_ms).alias("bucket_ts"),
    ]).collect()

    # Aggregation: open, high, low, close, volume, trades
    grouped = trades_df.groupby("bucket_ts").agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("quantity").sum().alias("volume"),
        pl.col("price").count().alias("trades"),
    ]).sort("bucket_ts")

    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1m", help="Interval like 1m, 5m, 1h")
    parser.add_argument("--source", default="raw", choices=["raw", "clean"], help="raw or clean")
    parser.add_argument("--symbol", default=None, help="Symbol filter e.g., btcusdt")
    parser.add_argument("--raw-path", default="data/raw", help="Raw dir")
    parser.add_argument("--clean-path", default="data/clean", help="Clean dir")
    parser.add_argument("--output", default=None, help="Output parquet path")
    args = parser.parse_args()

    interval_ms = parse_interval(args.interval)
    if args.output is None:
        out_dir = Path("data/features")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"klines_{args.interval}.parquet"
    else:
        out = Path(args.output)

    rows = []
    if args.source == "raw":
        pattern = os.path.join(args.raw_path, "*.jsonl.zst")
        for t in iter_trade_events_from_raw(pattern, args.symbol):
            if t["ts"] is None:
                continue
            rows.append((t["ts"], t["price"], t["quantity"], t.get("symbol")))
    else:
        for t in iter_trade_events_from_clean(args.clean_path, args.symbol):
            if t["ts"] is None:
                continue
            rows.append((t["ts"], t["price"], t["quantity"], t.get("symbol")))

    if not rows:
        print("No trade events found in the selected source.")
        return
    df = pl.DataFrame(rows, schema=["ts", "price", "quantity", "symbol"])

    klines = make_candles(df, interval_ms)
    # Add human-friendly columns
    klines = klines.with_columns([
        (pl.col("bucket_ts")).alias("ts"),
    ])
    klines.write_parquet(out, compression="zstd", compression_level=3)
    print(f"Wrote {out} with {klines.shape[0]} buckets")


if __name__ == "__main__":
    main()
