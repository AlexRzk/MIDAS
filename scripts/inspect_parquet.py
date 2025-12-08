#!/usr/bin/env python3
"""
Inspect Parquet files useful for debugging the MIDAS pipeline.

Prints:
- Path and file size
- Row count and columns
- Parquet metadata (footer)
- First N rows
- ts column min/max and derived datetime ranges
- Basic sanity checks (midprice exists, non-empty rows)

Usage:
  python scripts/inspect_parquet.py --file data/features/features_20251208_065623_0000.parquet --rows 5
  python scripts/inspect_parquet.py --dir data/features --latest --rows 10
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone

try:
    import polars as pl
except Exception:
    pl = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def read_parquet_head(path: Path, n: int = 5):
    if pl is None:
        raise SystemExit("Polars not installed. Please install polars to use this script.")
    return pl.read_parquet(path, n_rows=n)


def inspect_metadata_pq(path: Path) -> dict:
    md = {}
    if pq is None:
        return md
    try:
        pf = pq.ParquetFile(str(path))
        md['num_rows'] = pf.metadata.num_rows
        md['num_row_groups'] = pf.metadata.num_row_groups
        schema = pf.schema_arrow
        md['columns'] = [f.name for f in schema]
        meta = pf.metadata.metadata
        if meta:
            decoded = {k.decode('utf8'): v.decode('utf8') for k, v in meta.items()}
            md['footer_metadata'] = decoded
    except Exception as e:
        md['error'] = str(e)
    return md


def pick_latest_parquet(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    return files[-1]


def format_ts_us_to_iso(ts_us: int) -> str:
    try:
        ts = int(ts_us)
    except Exception:
        return "-"
    # ts is microseconds since epoch
    try:
        dt = datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return str(ts)


def main():
    parser = argparse.ArgumentParser(description="Inspect Parquet files and print useful debug info.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=Path, help="Specific Parquet file to inspect")
    group.add_argument("--dir", "-d", type=Path, help="Directory with Parquet files to inspect")
    parser.add_argument("--latest", action="store_true", help="Pick latest file in directory")
    parser.add_argument("--rows", type=int, default=5, help="Number of head rows to show")
    parser.add_argument("--show-columns", action="store_true", help="Display full list of columns")
    parser.add_argument("--recompute", action="store_true", help="Recompute missing columns (e.g., spread_bps) for basic checks")
    parser.add_argument("--raw", action="store_true", help="Also attempt to print a column sample for raw columns")

    args = parser.parse_args()

    if args.file:
        path = args.file.expanduser().resolve()
    else:
        directory = args.dir.expanduser().resolve()
        if not directory.exists():
            print(f"Directory {directory} does not exist.")
            raise SystemExit(1)
        if args.latest:
            path = pick_latest_parquet(directory)
            if not path:
                print(f"No Parquet files in directory {directory}")
                raise SystemExit(1)
        else:
            print("You must specify --latest when passing a directory to inspect.")
            raise SystemExit(2)

    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    print("\n=== PARQUET INSPECTOR ===")
    print(f"File: {path}")
    try:
        st = path.stat()
        print(f"Size: {human_bytes(st.st_size)}")
    except Exception:
        pass

    metadata = inspect_metadata_pq(path)
    if 'error' in metadata:
        print("Warning: Could not read Parquet metadata:", metadata['error'])
    else:
        print(f"Row groups: {metadata.get('num_row_groups', 'unknown')}")
        print(f"Num rows: {metadata.get('num_rows', 'unknown')}")
        if 'footer_metadata' in metadata:
            print("Parquet footer metadata:")
            for k, v in metadata['footer_metadata'].items():
                try:
                    js = json.loads(v)
                    print(f"  {k}: {json.dumps(js, indent=2)}")
                except Exception:
                    print(f"  {k}: {v}")

    # Display column names and types (read schema via Polars if possible)
    if pl is None:
        print("Polars is not installed; cannot show schema/rows. Install polars to use this script.")
        raise SystemExit(1)

    print("\nSchema & first rows:")
    try:
        head = read_parquet_head(path, args.rows)
    except Exception as e:
        print("Error reading parquet head:", e)
        raise SystemExit(1)

    # Print schema
    print("Columns and types:")
    for c, t in zip(head.columns, head.dtypes):
        print(f"  {c}: {t}")
    if args.show_columns:
        print("\nFull columns:")
        print(head.columns)

    print("\nFirst rows:")
    print(head)

    # If ts column exists, show min/max time
    if 'ts' in head.columns or 'timestamp' in head.columns:
        ts_col_name = 'ts' if 'ts' in head.columns else 'timestamp'
        try:
            ts_df = pl.read_parquet(path, columns=[ts_col_name])
            ts_min = int(ts_df[ts_col_name].min())
            ts_max = int(ts_df[ts_col_name].max())
            print(f"\nPrimary timestamp column: {ts_col_name}")
            print("  min (us):", ts_min)
            print("  min (iso):", format_ts_us_to_iso(ts_min))
            print("  max (us):", ts_max)
            print("  max (iso):", format_ts_us_to_iso(ts_max))
        except Exception as e:
            print("  Failed to compute min/max ts:", e)

    # Sanity checks
    print("\nSanity checks:")
    # Check basic expected features/columns
    sas = []
    required_columns = ['midprice', 'spread', 'spread_bps', 'imbalance']
    missing = [c for c in required_columns if c not in head.columns]
    if missing:
        print("  Missing expected columns:", missing)
    else:
        print("  All expected core feature columns present.")

    # If features present, show basic stats for a few
    numeric_cols = [c for c, t in zip(head.columns, head.dtypes) if str(t).startswith('Float') or str(t).startswith('Int')]
    stats_cols = [c for c in ['midprice', 'spread_bps', 'ofi', 'kyle_lambda', 'vpin'] if c in numeric_cols]
    if stats_cols:
        df_stats = pl.read_parquet(path, columns=stats_cols)
        print("\nColumn sample stats (min, mean, max):")
        for c in stats_cols:
            try:
                vmin = float(df_stats[c].min())
                vmax = float(df_stats[c].max())
                mean = float(df_stats[c].mean())
                print(f"  {c}: min={vmin:.6g}, mean={mean:.6g}, max={vmax:.6g}")
            except Exception as e:
                print(f"  {c}: error computing stats: {e}")

    # If spread_bps missing but spread exists, optionally recompute
    if args.recompute and 'spread_bps' not in head.columns and 'bid_px_01' in head.columns and 'ask_px_01' in head.columns and 'midprice' in head.columns:
        print("\nRecomputing spread_bps: (ask_px_01 - bid_px_01) / midprice * 10000")
        try:
            full_df = pl.read_parquet(path, columns=['bid_px_01', 'ask_px_01', 'midprice'])
            spread_bps = ((full_df['ask_px_01'] - full_df['bid_px_01']) / full_df['midprice']) * 10000
            print("  spread_bps stats: min=", float(spread_bps.min()), "mean=", float(spread_bps.mean()), "max=", float(spread_bps.max()))
        except Exception as e:
            print("  Failed to compute spread_bps:", e)

    print("\nDone.")


if __name__ == '__main__':
    main()
