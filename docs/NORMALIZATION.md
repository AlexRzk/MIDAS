# MIDAS Feature Normalization

Production-ready normalization layer for high-frequency trading features.

## Overview

The normalization system applies **optimal scaling** to each feature type based on its statistical properties:

- **StandardScaler (z-score)**: For normally distributed features (OFI, imbalances, spreads)
- **RobustScaler (median + IQR)**: For heavy-tailed features (order sizes, volumes) with log1p pre-transform
- **MinMaxScaler (0-1)**: For bounded features (positional encodings)
- **Passthrough**: Raw prices and timestamps (no normalization)

## Architecture

```
features/
├── feature_schema.py    # Feature classification by scaling type
├── normalize.py         # Normalization implementation
└── main.py             # Pipeline integration

data/
└── scalers/
    └── normalization_manifest.json  # Fitted scaler statistics
```

## Feature Classification

### StandardScaler Features (z-score: mean=0, std=1)

```python
STANDARD_FEATURES = [
    "ofi", "ofi_10", "ofi_cumulative",
    "imbalance", "imbalance_1", "imbalance_5", "imbalance_10",
    "microprice", "spread", "spread_bps",
    "volatility_20", "volatility_100",
    "signed_volume", "volume_imbalance",
    "liquidity_1", "liquidity_5", "liquidity_10",
    "kyle_lambda", "vpin",
    "bid_ladder_slope", "ask_ladder_slope",
    # ... see feature_schema.py for complete list
]
```

**Formula**: `(x - mean_train) / std_train`

**Why**: These features approximate normal distributions and benefit from centering and standardization.

### RobustScaler Features (log1p + median/IQR)

```python
ROBUST_FEATURES = [
    "bid_sz_01", "bid_sz_02", ..., "bid_sz_10",
    "ask_sz_01", "ask_sz_02", ..., "ask_sz_10",
    "taker_buy_volume", "taker_sell_volume",
    "queue_imb_1", "queue_imb_2", ..., "queue_imb_5",
]
```

**Formula**: `(log1p(x) - median_train) / IQR_train`

**Why**: Order sizes and volumes have extreme outliers and fat tails. Log1p compresses the range, robust statistics handle outliers.

### Passthrough (No Normalization)

```python
RAW_PASSTHROUGH_FEATURES = [
    "ts",  # Timestamp for alignment
    "bid_px_01", ..., "bid_px_10",  # Raw prices
    "ask_px_01", ..., "ask_px_10",
    "midprice",  # Reference price
]
```

**Why**: Preserve actual price levels for reconstruction and reference.

## Usage

### Automatic (Pipeline Integration)

Normalization is **automatically applied** in the features pipeline when enabled:

```bash
export ENABLE_NORMALIZATION=true  # Default
docker compose up features
```

The first processed file will **fit** scalers. Subsequent files **transform** using those scalers.

### Manual (Existing Features)

To normalize features that were already generated:

```bash
# Normalize in-place
python scripts/normalize_existing_features.py \
    --input-dir /data/features \
    --scaler-dir /data/scalers

# Normalize to new directory
python scripts/normalize_existing_features.py \
    --input-dir /data/features \
    --output-dir /data/features_normalized \
    --scaler-dir /data/scalers
```

### Programmatic

```python
from features.features.normalize import FeatureNormalizer
import polars as pl

# Training: fit scalers
normalizer = FeatureNormalizer(scaler_dir="data/scalers")
train_df = pl.read_parquet("train_features.parquet")
train_normalized = normalizer.fit_transform(train_df, is_training=True)
normalizer.save()

# Inference: load and transform
normalizer = FeatureNormalizer.load("data/scalers")
test_df = pl.read_parquet("test_features.parquet")
test_normalized = normalizer.transform(test_df)
```

## Validation

### Automatic Validation

The environment check validates normalization:

```bash
python training/gpu_project/run_env_check.py
```

Checks:
- ✓ Scalers exist in `/data/scalers/`
- ✓ No NaN or infinite values
- ✓ Standard features: `|mean| < 0.15`, `0.8 < std < 1.2`
- ✓ Robust features: `-25 < values < 25`
- ✓ MinMax features: `0 <= values <= 1`

### Programmatic Validation

```python
normalizer = FeatureNormalizer.load("data/scalers")
report = normalizer.validate_normalized_data(df)

if report["valid"]:
    print("✓ Normalization validation passed")
else:
    print("Warnings:")
    for warning in report["warnings"]:
        print(f"  - {warning}")
```

## Scaler Manifest Format

```json
{
  "version": "1.0.0",
  "created_at": "2025-12-09T10:30:00Z",
  "n_samples_fitted": 343250,
  "scalers": {
    "ofi": {
      "feature_name": "ofi",
      "scaler_type": "standard",
      "mean": 0.0234,
      "std": 1.456,
      "n_samples": 343250,
      "n_nulls": 0
    },
    "bid_sz_01": {
      "feature_name": "bid_sz_01",
      "scaler_type": "robust",
      "median": 8.234,
      "iqr": 2.456,
      "q25": 7.123,
      "q75": 9.579,
      "n_samples": 343250,
      "n_nulls": 0
    }
  }
}
```

## Causality Guarantee

**CRITICAL**: Normalization is strictly **causal** to prevent look-ahead bias:

1. **Training Phase**: Scalers fitted ONLY on training data
2. **Test Phase**: Same scalers applied to test data (never refitted)
3. **Production**: Scalers saved at training time, loaded at inference time

```python
# CORRECT - causal normalization
normalizer.fit(train_df, is_training=True)
train_norm = normalizer.transform(train_df)
test_norm = normalizer.transform(test_df)  # Uses SAME scalers

# WRONG - data leakage!
all_data = pl.concat([train_df, test_df])
normalizer.fit(all_data)  # ❌ Test statistics leak into training!
```

## Performance

- **Polars lazy frames**: Vectorized operations for speed
- **Batch processing**: Efficiently normalizes 100K+ rows/second
- **Low memory**: Streaming transformations
- **Scaler size**: ~1KB per feature (~100KB total for 72 features)

## Configuration

### Enable/Disable Normalization

```bash
# Enable (default)
export ENABLE_NORMALIZATION=true

# Disable
export ENABLE_NORMALIZATION=false
```

### Add New Features

Edit `features/features/feature_schema.py`:

```python
STANDARD_FEATURES = [
    "ofi",
    # ... existing features ...
    "my_new_feature",  # Add here
]
```

Rebuild features pipeline:

```bash
docker compose restart features
```

## Troubleshooting

### Issue: Scalers not found

**Solution**: Run features pipeline to generate scalers:

```bash
docker compose up features
# Wait for first file to process
ls data/scalers/  # Should see normalization_manifest.json
```

### Issue: Normalization validation warnings

**Symptoms**:
- `mean=-0.234` (expected ≈0)
- `std=0.456` (expected ≈1)

**Possible causes**:
1. Scalers fitted on small sample
2. Feature distribution changed
3. Wrong scaler type for feature

**Solution**: Refit scalers on representative training data:

```bash
# Delete old scalers
rm data/scalers/normalization_manifest.json

# Reprocess features (will refit scalers)
docker compose restart features
```

### Issue: Features still unnormalized

**Check**:
1. Is normalization enabled? `echo $ENABLE_NORMALIZATION`
2. Are scalers loaded? Check logs for `loaded_normalizer_manifest`
3. Is feature in schema? Check `feature_schema.py`

## Examples

### Example 1: Training Pipeline

```python
from features.features.normalize import FeatureNormalizer
from features.features.splitter import OFISafeSplitter
import polars as pl

# Load features
df = pl.read_parquet("data/features/features_btcusdt_20251207.parquet")

# OFI-safe split
splitter = OFISafeSplitter(train_ratio=0.8)
splitter.fit(df)
train_df, test_df = splitter.get_all_train_test(df)

# Normalize (fit on training only!)
normalizer = FeatureNormalizer("data/scalers")
train_norm = normalizer.fit_transform(train_df, is_training=True)
normalizer.save()

# Apply same scalers to test
test_norm = normalizer.transform(test_df)

# Train model
# ... use train_norm for training
```

### Example 2: Inference Pipeline

```python
from features.features.normalize import FeatureNormalizer
import polars as pl

# Load new data
new_df = pl.read_parquet("new_features.parquet")

# Load fitted scalers
normalizer = FeatureNormalizer.load("data/scalers")

# Normalize new data
new_norm = normalizer.transform(new_df)

# Make predictions
# ... use new_norm for inference
```

### Example 3: Custom Feature Addition

```python
# 1. Add to schema
# Edit features/features/feature_schema.py
STANDARD_FEATURES.append("my_custom_momentum")

# 2. Compute in features pipeline
# Edit features/features/compute_v2.py
def compute_all_features(self, df):
    # ... existing features ...
    df = df.with_columns([
        (pl.col("midprice").pct_change(10)).alias("my_custom_momentum")
    ])
    return df

# 3. Restart pipeline
docker compose restart features

# Scalers automatically fit for new feature
```

## Best Practices

1. **Always fit on training data only** - Never fit on test/validation data
2. **Save scalers immediately after fitting** - Prevents loss if process crashes
3. **Version your scalers** - Track which data they were fitted on
4. **Validate after normalization** - Check distributions match expectations
5. **Monitor for drift** - Periodically refit if data distribution changes
6. **Use robust scaling for volumes** - Handles extreme outliers better
7. **Keep raw prices unnormalized** - Needed for P&L calculation

## References

- **StandardScaler**: Scikit-learn standard scaling
- **RobustScaler**: Median/IQR scaling resistant to outliers
- **OFI-safe splitting**: Prevents cumulative feature leakage
- **Polars**: High-performance DataFrame library
