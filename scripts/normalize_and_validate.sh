#!/usr/bin/env bash
# normalize_and_validate.sh
# Usage:
#   ./scripts/normalize_and_validate.sh --input-dir data/features --scaler-dir data/scalers --output-dir data/features_normalized

set -euo pipefail

# Defaults
INPUT_DIR="data/features"
SCALER_DIR="data/scalers"
OUTPUT_DIR=""
OVERWRITE=false

usage() {
  echo "Usage: $0 [--input-dir PATH] [--scaler-dir PATH] [--output-dir PATH] [--overwrite]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --input-dir)
      INPUT_DIR="$2"; shift 2;;
    --scaler-dir)
      SCALER_DIR="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --overwrite)
      OVERWRITE=true; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"; usage;;
  esac
done

if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" && "$OVERWRITE" != "true" ]]; then
  echo "Output dir $OUTPUT_DIR already exists. Use --overwrite to replace." >&2
  exit 1
fi

# Ensure the scaler dir exists so script can save or load
mkdir -p "$SCALER_DIR"

# Run normalization script (this script will fit on first file, transform others)
python3 scripts/normalize_existing_features.py --input-dir "$INPUT_DIR" --scaler-dir "$SCALER_DIR" --output-dir "${OUTPUT_DIR:-$INPUT_DIR}"

# Run GPU project env check or run_env_check
if [[ -f training/gpu_project/run_env_check.py ]]; then
  python3 training/gpu_project/run_env_check.py
else
  echo "Warning: run_env_check.py not found in training/gpu_project; skipping env validation."
fi

# Basic validation: show means/std for a small set of features to inspect distribution
python3 - <<'PY'
import polars as pl
from pathlib import Path

input_dir = Path("${OUTPUT_DIR:-$INPUT_DIR}")
files = sorted(input_dir.glob("*.parquet"))
if not files:
    print('No parquet files found in', input_dir)
    raise SystemExit(0)

# Read small sample
df = pl.read_parquet(files[0]).sample(1000) if len(files) > 0 else None
if df is None:
    print('No sample')
    raise SystemExit(0)

# Print mean/std for numeric columns
num_cols = [c for c, t in df.schema.items() if t in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
print('Numeric columns sample mean/std (first file):')
for col in num_cols[:25]:
    s = df[col].drop_nulls()
    try:
        mean = s.mean()
        std = s.std()
        print(f"{col}: mean={mean:.6f}, std={std:.6f}")
    except Exception:
        pass
PY

echo "Normalization complete. Scalerdir: $SCALER_DIR"

echo "Done." 