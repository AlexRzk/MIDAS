"""
Feature schema for normalization.

Classifies features by their statistical properties and appropriate scaling method.
This ensures optimal normalization for each feature type in HFT ML pipelines.
"""
from typing import List, Dict, Set


# ============================================================================
# STANDARD SCALER (Z-Score Normalization)
# ============================================================================
# Use for: (value - mean) / std
# Best for: Normally distributed features, features with outliers already handled

STANDARD_FEATURES: List[str] = [
    # Order Flow Imbalance (OFI) - core alpha signal
    "ofi",
    "ofi_10",
    "ofi_cumulative",
    
    # Imbalance metrics - price-weighted order book imbalances
    "imbalance",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    
    # Microprice - theoretical mid-price accounting for depth
    "microprice",
    
    # Spread metrics - transaction cost indicators
    "spread",
    "spread_bps",
    
    # Volatility - realized variance measures
    "volatility_20",
    "volatility_100",
    "vol_of_vol",  # Volatility of volatility
    
    # Volume metrics - signed and directional volume
    "signed_volume",
    "volume_imbalance",
    
    # Liquidity metrics - available depth at various levels
    "liquidity_1",
    "liquidity_5",
    "liquidity_10",
    
    # Kyle Lambda - price impact measure
    "kyle_lambda",
    
    # VPIN - Volume-synchronized probability of informed trading
    "vpin",
    
    # Ladder slope metrics - order book shape
    "bid_ladder_slope",
    "ask_ladder_slope",
    "bid_slope_ratio",
    "ask_slope_ratio",
]


# ============================================================================
# ROBUST SCALER (Median + IQR) with LOG1P Pre-transformation
# ============================================================================
# Use for: value = log1p(value); value_norm = (value - median) / IQR
# Best for: Heavy-tailed distributions, features with extreme outliers

ROBUST_FEATURES: List[str] = [
    # Order book sizes at all levels - highly skewed, fat-tailed
    "bid_sz_01", "bid_sz_02", "bid_sz_03", "bid_sz_04", "bid_sz_05",
    "bid_sz_06", "bid_sz_07", "bid_sz_08", "bid_sz_09", "bid_sz_10",
    "ask_sz_01", "ask_sz_02", "ask_sz_03", "ask_sz_04", "ask_sz_05",
    "ask_sz_06", "ask_sz_07", "ask_sz_08", "ask_sz_09", "ask_sz_10",
    
    # Trade volume metrics - extreme spikes common
    "taker_buy_volume",
    "taker_sell_volume",
    
    # Queue imbalance at levels - can have extreme values
    "queue_imb_1",
    "queue_imb_2", 
    "queue_imb_3",
    "queue_imb_4",
    "queue_imb_5",
]


# ============================================================================
# MINMAX SCALER (0-1 Normalization)
# ============================================================================
# Use for: (value - min) / (max - min)
# Best for: Bounded features, positional encodings

MINMAX_FEATURES: List[str] = [
    # Positional encodings (if used)
    # "time_of_day_norm",  # 0-1 within trading day
    # "day_of_week_norm",  # 0-1 within week
]


# ============================================================================
# RAW PASSTHROUGH (No Normalization)
# ============================================================================
# Never normalize these - they are used for indexing, alignment, or are already normalized

RAW_PASSTHROUGH_FEATURES: List[str] = [
    # Timestamp - used for time alignment
    "ts",
    
    # Raw prices - preserve actual price levels for reconstruction
    "bid_px_01", "bid_px_02", "bid_px_03", "bid_px_04", "bid_px_05",
    "bid_px_06", "bid_px_07", "bid_px_08", "bid_px_09", "bid_px_10",
    "ask_px_01", "ask_px_02", "ask_px_03", "ask_px_04", "ask_px_05",
    "ask_px_06", "ask_px_07", "ask_px_08", "ask_px_09", "ask_px_10",
    
    # Midprice raw - keep for reference, use microprice for ML
    "midprice",
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_feature_sets() -> Dict[str, List[str]]:
    """Get all feature sets organized by scaling method."""
    return {
        "standard": STANDARD_FEATURES,
        "robust": ROBUST_FEATURES,
        "minmax": MINMAX_FEATURES,
        "passthrough": RAW_PASSTHROUGH_FEATURES,
    }


def get_features_to_normalize() -> Dict[str, List[str]]:
    """Get only features that require normalization."""
    return {
        "standard": STANDARD_FEATURES,
        "robust": ROBUST_FEATURES,
        "minmax": MINMAX_FEATURES,
    }


def classify_feature(feature_name: str) -> str:
    """
    Classify a feature by its normalization method.
    
    Args:
        feature_name: Name of the feature column
        
    Returns:
        One of: "standard", "robust", "minmax", "passthrough", "unknown"
    """
    if feature_name in STANDARD_FEATURES:
        return "standard"
    elif feature_name in ROBUST_FEATURES:
        return "robust"
    elif feature_name in MINMAX_FEATURES:
        return "minmax"
    elif feature_name in RAW_PASSTHROUGH_FEATURES:
        return "passthrough"
    else:
        return "unknown"


def get_all_known_features() -> Set[str]:
    """Get set of all known features across all categories."""
    return set(
        STANDARD_FEATURES +
        ROBUST_FEATURES +
        MINMAX_FEATURES +
        RAW_PASSTHROUGH_FEATURES
    )


def validate_dataframe_columns(columns: List[str]) -> Dict[str, List[str]]:
    """
    Validate DataFrame columns and classify them.
    
    Args:
        columns: List of column names from DataFrame
        
    Returns:
        Dictionary with classification results and warnings
    """
    known_features = get_all_known_features()
    
    result = {
        "standard": [],
        "robust": [],
        "minmax": [],
        "passthrough": [],
        "unknown": [],
    }
    
    for col in columns:
        category = classify_feature(col)
        result[category].append(col)
    
    return result


# ============================================================================
# Normalization Metadata
# ============================================================================

NORMALIZATION_CONFIG = {
    "standard": {
        "method": "z-score",
        "formula": "(x - mean) / std",
        "pre_transform": None,
        "expected_mean": 0.0,
        "expected_std": 1.0,
        "tolerance_mean": 0.1,
        "tolerance_std": 0.2,
    },
    "robust": {
        "method": "robust",
        "formula": "(log1p(x) - median) / IQR",
        "pre_transform": "log1p",
        "expected_range": (-20, 20),  # Reasonable range after robust scaling
        "tolerance_outliers": 0.01,  # Allow 1% outliers beyond range
    },
    "minmax": {
        "method": "min-max",
        "formula": "(x - min) / (max - min)",
        "pre_transform": None,
        "expected_min": 0.0,
        "expected_max": 1.0,
        "tolerance": 0.01,
    },
}


if __name__ == "__main__":
    # Print feature schema for debugging
    print("Feature Classification Schema")
    print("=" * 80)
    
    for category, features in get_all_feature_sets().items():
        print(f"\n{category.upper()} ({len(features)} features):")
        for feat in features[:5]:
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")
    
    print(f"\nTotal features: {len(get_all_known_features())}")
