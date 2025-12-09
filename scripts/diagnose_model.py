#!/usr/bin/env python3
"""
Diagnostic script to understand why the model has poor directional accuracy.
Analyzes feature-target relationships and model predictions.
"""

import polars as pl
import numpy as np
from pathlib import Path
import json

def main():
    print("\n" + "="*60)
    print("MIDAS Model Diagnostics")
    print("="*60)
    
    # 1. Load latest training output
    output_dir = Path("outputs")
    latest_run = sorted(output_dir.glob("xgboost_bt_*"))[-1]
    print(f"\nAnalyzing run: {latest_run.name}")
    
    # Load results
    with open(latest_run / "results.json") as f:
        results = json.load(f)
    
    print("\n--- Training Results ---")
    print(json.dumps(results, indent=2))
    
    # Extract metrics (handle different structures)
    test_metrics = results.get('test_metrics', results)
    
    if 'mse' in test_metrics:
        print(f"\nMSE: {test_metrics['mse']:.6f}")
        print(f"MAE: {test_metrics['mae']:.6f}")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}")
    
    if 'backtest' in results:
        print(f"\n--- Backtest Results ---")
        print(f"Best iteration: {results['backtest']['best_iteration']}")
        print(f"Best {results['backtest'].get('metric', 'N/A')}: {results['backtest']['best_value']:.4f}")
    
    # 2. Load raw data
    print("\n--- Analyzing Raw Features ---")
    data_dir = Path("data/features_normalized")
    df = pl.read_parquet(data_dir / "normalized_features_*.parquet")
    
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {df.columns[:10]}... ({len(df.columns)} total)")
    
    # Check key features
    print("\n--- Key Feature Statistics ---")
    key_features = ['microprice', 'ofi', 'imbalance', 'spread_bps', 'signed_volume']
    for feat in key_features:
        if feat in df.columns:
            vals = df[feat].drop_nulls()
            print(f"{feat:20s}: mean={vals.mean():.6f}, std={vals.std():.6f}, "
                  f"min={vals.min():.6f}, max={vals.max():.6f}")
    
    # 3. Analyze target creation
    print("\n--- Target Analysis ---")
    print("Computing targets manually to verify...")
    
    # Sort by timestamp
    df = df.sort("ts")
    
    # Compute different target types
    microprice = df["microprice"].to_numpy()
    
    # Returns at different horizons
    for horizon in [1, 5, 10]:
        returns = np.zeros(len(microprice))
        returns[:-horizon] = (microprice[horizon:] - microprice[:-horizon]) / microprice[:-horizon]
        returns[-horizon:] = np.nan
        
        # Direction stats
        up = np.sum(returns > 0.0001)  # Up if return > 1 bps
        down = np.sum(returns < -0.0001)  # Down if return < -1 bps
        flat = np.sum(np.abs(returns) <= 0.0001)  # Flat otherwise
        total = up + down + flat
        
        print(f"\nHorizon {horizon}:")
        print(f"  Up:   {up:8,} ({up/total*100:5.2f}%)")
        print(f"  Down: {down:8,} ({down/total*100:5.2f}%)")
        print(f"  Flat: {flat:8,} ({flat/total*100:5.2f}%)")
        print(f"  Mean return: {np.nanmean(returns)*10000:.2f} bps")
        print(f"  Std return:  {np.nanstd(returns)*10000:.2f} bps")
    
    # 4. Check feature-target correlation
    print("\n--- Feature-Target Correlations (horizon=1) ---")
    returns_1 = np.zeros(len(microprice))
    returns_1[:-1] = (microprice[1:] - microprice[:-1]) / microprice[:-1]
    returns_1[-1] = np.nan
    
    for feat in ['ofi', 'imbalance', 'signed_volume', 'spread_bps', 'volatility_20']:
        if feat in df.columns:
            feat_vals = df[feat].to_numpy()
            # Remove NaN pairs
            mask = ~(np.isnan(feat_vals) | np.isnan(returns_1))
            if np.sum(mask) > 0:
                corr = np.corrcoef(feat_vals[mask], returns_1[mask])[0, 1]
                print(f"{feat:20s}: {corr:+.4f}")
    
    # 5. Check for data issues
    print("\n--- Data Quality Checks ---")
    
    # Check for constant features
    constant_feats = []
    for col in df.columns:
        if col not in ['ts']:
            vals = df[col].drop_nulls()
            if len(vals) > 0 and vals.std() < 1e-10:
                constant_feats.append(col)
    
    if constant_feats:
        print(f"⚠️  Warning: {len(constant_feats)} constant features found:")
        print(f"   {constant_feats[:5]}...")
    else:
        print("✓ No constant features")
    
    # Check for extreme values
    extreme_cols = []
    for col in df.columns:
        if col not in ['ts', 'bid_px_01', 'ask_px_01', 'microprice', 'last_trade_px']:
            vals = df[col].drop_nulls().to_numpy()
            if len(vals) > 0:
                if np.any(np.abs(vals) > 100):  # Normalized features should be ~[-10, 10]
                    extreme_cols.append((col, np.abs(vals).max()))
    
    if extreme_cols:
        print(f"\n⚠️  Warning: {len(extreme_cols)} features with extreme values:")
        for col, max_val in sorted(extreme_cols, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {col:20s}: max={max_val:.2f}")
    else:
        print("✓ No extreme values in normalized features")
    
    # 6. Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Based on directional accuracy
    dir_acc = test_metrics.get('directional_accuracy', 0)
    if dir_acc < 0.30:
        print("\n🔴 CRITICAL: Directional accuracy < 30% (worse than random)")
        print("   Possible causes:")
        print("   1. Target may be inverted (try negating predictions)")
        print("   2. Features don't capture price movement")
        print("   3. Data leakage or look-ahead bias in features")
        print("   4. Insufficient feature engineering")
    
    # Based on R²
    r2 = test_metrics.get('r2', 0)
    if r2 < 0.1:
        print("\n⚠️  LOW R²: Model explains < 10% of variance")
        print("   This is somewhat normal for HFT, but suggests:")
        print("   1. Try longer prediction horizons (5-10 steps)")
        print("   2. Add more predictive features")
        print("   3. Consider ensemble methods")
    
    print("\n" + "="*60)
    print("Next steps to try:")
    print("  1. Train with horizon=5:  python training/gpu_project/train_xgboost.py --target-horizon 5")
    print("  2. Try direction target:   python training/gpu_project/train_xgboost.py --target-type direction")
    print("  3. Check feature computation in features/features/compute_v2.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
