#!/usr/bin/env python3
"""Quick test to verify percentile threshold calculation."""

import numpy as np
import sys
sys.path.insert(0, 'training/gpu_project')

from utils import compute_directional_accuracy

# Create test data similar to your actual returns
np.random.seed(42)
y_true = np.random.normal(0, 0.002, 10000)  # Mean 0, std 0.002 (similar to your data)
y_pred = y_true + np.random.normal(0, 0.001, 10000)  # Add some noise

print("Testing directional accuracy with percentile threshold...")
print(f"\ny_true stats:")
print(f"  Mean: {y_true.mean():.6f}")
print(f"  Std: {y_true.std():.6f}")
print(f"  33rd percentile of |y_true|: {np.percentile(np.abs(y_true), 33):.6f}")

# Test with percentile
result_pct = compute_directional_accuracy(y_true, y_pred, threshold='percentile')
print(f"\nWith threshold='percentile':")
print(f"  Threshold used: {result_pct['threshold_used']:.6f}")
print(f"  Directional accuracy: {result_pct['directional_accuracy']:.2%}")
print(f"  Up count: {result_pct['up_count']:,}")
print(f"  Down count: {result_pct['down_count']:,}")
print(f"  Flat count: {result_pct['flat_count']:,}")

# Test with sign-based
result_sign = compute_directional_accuracy(y_true, y_pred, threshold=None)
print(f"\nWith threshold=None (sign-based):")
print(f"  Threshold used: {result_sign['threshold_used']:.6f}")
print(f"  Directional accuracy: {result_sign['directional_accuracy']:.2%}")
print(f"  Up count: {result_sign['up_count']:,}")
print(f"  Down count: {result_sign['down_count']:,}")
print(f"  Flat count: {result_sign['flat_count']:,}")

print("\n✅ Test complete!")
