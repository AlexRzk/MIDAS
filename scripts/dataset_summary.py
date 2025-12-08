"""
Dataset summary tool for MIDAS features.

Usage:
  python3 scripts/dataset_summary.py --dir data/features

Outputs:
  - total files / rows
  - per-file row counts
  - min / max timestamps (human readable)
  - non-zero counts for advanced features
  - per-second aggregate counts
"""
import argparse
from pathlib import Path
import polars as pl
import datetime

ADVANCED = [
    'kyle_lambda','vpin','bid_ladder_slope','ask_ladder_slope',
    'bid_slope_ratio','ask_slope_ratio','queue_imb_1','queue_imb_2',
    'queue_imb_3','queue_imb_4','queue_imb_5','vol_of_vol'
]


def ts_us_to_iso(ts_us: int) -> str:
    if ts_us is None:
        return 'N/A'
    return datetime.datetime.utcfromtimestamp(ts_us / 1_000_000.0).isoformat()


def summarize_feature_dir(path: Path):
    files = sorted(path.glob('features_*.parquet'))
    if not files:
        print('No feature files in', path)
        return

    total_rows = 0
    min_ts = None
    max_ts = None
    per_file = []
    nonzero_counts = {k: 0 for k in ADVANCED}

    for f in files:
        df = pl.read_parquet(f, columns=['ts'] + [c for c in ADVANCED if c in pl.read_parquet(f, n_rows=1).columns])
        n = df.shape[0]
        total_rows += n
        ts_min = int(df['ts'].min()) if n > 0 else None
        ts_max = int(df['ts'].max()) if n > 0 else None
        if ts_min is not None:
            min_ts = ts_min if min_ts is None or ts_min < min_ts else min_ts
        if ts_max is not None:
            max_ts = ts_max if max_ts is None or ts_max > max_ts else max_ts
        per_file.append((f.name, n))

        # count nonzero advanced features
        for c in ADVANCED:
            if c in df.columns:
                nonzero = df.filter(pl.col(c) != 0.0).shape[0]
                nonzero_counts[c] += nonzero

    print('Files:', len(files))
    print('Rows total:', total_rows)
    print('Min ts:', ts_us_to_iso(min_ts))
    print('Max ts:', ts_us_to_iso(max_ts))
    print('\nPer-file rows:')
    for fn, n in per_file:
        print(f'  {fn}: {n}')

    print('\nAdvanced feature non-zero counts:')
    for k, v in nonzero_counts.items():
        print(f'  {k}: {v}')

    # compute per-second aggregate totals
    dfs = []
    for f in files:
        df = pl.read_parquet(f, columns=['ts','midprice'])
        if df.shape[0] == 0:
            continue
        df = df.with_columns(((pl.col('ts') // 1_000_000).alias('ts_s')))
        df2 = df.groupby('ts_s').agg([pl.col('midprice').last().alias('mid')])
        dfs.append(df2)
    if dfs:
        big = pl.concat(dfs)
        print('\nPer-second points (unique ts):', big.shape[0])
    else:
        print('\nPer-second points (unique ts): 0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data/features', help='features directory')
    args = parser.parse_args()
    summarize_feature_dir(Path(args.dir))
