#!/usr/bin/env python3
"""
Environment Check Script for MIDAS GPU Training.

Run this script first to verify your environment is properly configured.
Checks:
- Python version
- Required packages and versions
- GPU availability (CUDA)
- Memory availability
- Data directory access
"""
import sys
import os
from pathlib import Path


def print_header(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_ok(msg: str):
    """Print success message."""
    print(f"  ✓ {msg}")


def print_warn(msg: str):
    """Print warning message."""
    print(f"  ⚠ {msg}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"  ✗ {msg}")


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 9:
        print_ok(f"Python {version_str} (>=3.9 required)")
        return True
    else:
        print_fail(f"Python {version_str} (>=3.9 required)")
        return False


def check_packages():
    """Check required packages."""
    print_header("Required Packages")
    
    packages = {
        "numpy": "1.20.0",
        "polars": "0.18.0",
        "torch": "2.0.0",
        "xgboost": "1.7.0",
        "scikit-learn": "1.0.0",
        "matplotlib": "3.5.0",
    }
    
    optional_packages = {
        "optuna": "3.0.0",
        "pytorch_lightning": "2.0.0",
    }
    
    all_ok = True
    
    print("\n  Core packages:")
    for package, min_version in packages.items():
        try:
            if package == "scikit-learn":
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(package)
                version = getattr(mod, "__version__", "unknown")
            print_ok(f"{package}: {version}")
        except ImportError:
            print_fail(f"{package}: NOT INSTALLED (required)")
            all_ok = False
    
    print("\n  Optional packages:")
    for package, min_version in optional_packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print_ok(f"{package}: {version}")
        except ImportError:
            print_warn(f"{package}: NOT INSTALLED (optional)")
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print_header("GPU (CUDA) Check")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print_ok(f"CUDA available: True")
            print_ok(f"CUDA version: {torch.version.cuda}")
            print_ok(f"cuDNN version: {torch.backends.cudnn.version()}")
            
            n_gpus = torch.cuda.device_count()
            print_ok(f"Number of GPUs: {n_gpus}")
            
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print_ok(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
            # Test GPU computation
            try:
                x = torch.randn(1000, 1000, device="cuda")
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                print_ok("GPU computation test: PASSED")
            except Exception as e:
                print_fail(f"GPU computation test: FAILED ({e})")
                return False
            
            return True
        else:
            print_warn("CUDA not available - will use CPU")
            print_warn("Training will be significantly slower")
            return True  # Not a failure, just a warning
            
    except ImportError:
        print_fail("PyTorch not installed")
        return False


def check_xgboost_gpu():
    """Check XGBoost GPU support."""
    print_header("XGBoost GPU Check")
    
    try:
        import xgboost as xgb
        
        # Try to create a GPU-enabled model
        try:
            import numpy as np
            X = np.random.randn(100, 10)
            y = np.random.randn(100)
            
            dtrain = xgb.DMatrix(X, label=y)
            # XGBoost 3.x: use 'hist' with device='cuda' instead of deprecated 'gpu_hist'
            params = {
                "tree_method": "hist",
                "device": "cuda",
            }
            
            # Train a small model
            bst = xgb.train(params, dtrain, num_boost_round=2)
            del bst
            
            print_ok("XGBoost GPU support: AVAILABLE (tree_method='hist', device='cuda')")
            return True
            
        except Exception as e:
            if "CUDA" in str(e) or "gpu" in str(e).lower() or "cuda" in str(e).lower():
                print_warn(f"XGBoost GPU not available: {e}")
                print_warn("Will use CPU for XGBoost (tree_method='hist', device='cpu')")
            else:
                print_warn(f"XGBoost GPU test failed: {e}")
            return True  # Not a critical failure
            
    except ImportError:
        print_fail("XGBoost not installed")
        return False


def check_memory():
    """Check system memory."""
    print_header("Memory Check")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print_ok(f"Total RAM: {total_gb:.1f} GB")
        print_ok(f"Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 8:
            print_warn("Less than 8 GB RAM available - may be insufficient for large datasets")
        
        return True
        
    except ImportError:
        print_warn("psutil not installed - cannot check memory")
        return True


def check_data_directory():
    """Check data directory."""
    print_header("Data Directory Check")
    
    # Check common data locations
    data_paths = [
        Path("/data/features"),
        Path("./data/features"),
        Path("../data/features"),
    ]
    
    found = False
    for data_path in data_paths:
        if data_path.exists():
            print_ok(f"Data directory found: {data_path.absolute()}")
            
            # Check for parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if parquet_files:
                print_ok(f"Found {len(parquet_files)} parquet files")
                
                # Check one file
                try:
                    import polars as pl
                    df = pl.read_parquet(parquet_files[0])
                    print_ok(f"Sample file: {parquet_files[0].name}")
                    print_ok(f"  Rows: {len(df):,}")
                    print_ok(f"  Columns: {len(df.columns)}")
                except Exception as e:
                    print_warn(f"Could not read parquet file: {e}")
            else:
                print_warn("No parquet files found in data directory")
            
            found = True
            break
    
    if not found:
        print_warn("Data directory not found at standard locations")
        print_warn("You may need to specify --data-dir when running training scripts")
    
    return True


def check_normalization():
    """Check normalization scalers and validate normalized data."""
    print_header("Normalization Check")
    
    # Check for scaler directory
    scaler_paths = [
        Path("/data/scalers"),
        Path("./data/scalers"),
        Path("../data/scalers"),
    ]
    
    scaler_dir = None
    for path in scaler_paths:
        if path.exists():
            scaler_dir = path
            break
    
    if scaler_dir is None:
        print_warn("Scaler directory not found")
        print_warn("Run features pipeline first to generate scalers")
        return True
    
    print_ok(f"Scaler directory found: {scaler_dir.absolute()}")
    
    # Check for manifest file
    manifest_file = scaler_dir / "normalization_manifest.json"
    if not manifest_file.exists():
        print_warn("Normalization manifest not found")
        print_warn("Scalers may not have been fitted yet")
        return True
    
    print_ok("Normalization manifest found")
    
    # Load and validate manifest
    try:
        import json
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        n_scalers = len(manifest.get("scalers", {}))
        print_ok(f"Loaded {n_scalers} scaler configurations")
        
        # Count by type
        scaler_types = {}
        for scaler_name, scaler_data in manifest.get("scalers", {}).items():
            scaler_type = scaler_data.get("scaler_type", "unknown")
            scaler_types[scaler_type] = scaler_types.get(scaler_type, 0) + 1
        
        for scaler_type, count in scaler_types.items():
            print_ok(f"  {scaler_type}: {count} features")
        
    except Exception as e:
        print_fail(f"Failed to load manifest: {e}")
        return False
    
    # Validate normalized data
    print("\n  Validating normalized data...")
    
    data_paths = [
        Path("/data/features_normalized"),
        Path("./data/features_normalized"),
        Path("../data/features_normalized"),
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print_warn("No normalized feature data to validate")
        print_warn("Run normalization script first")
        return True
    
    parquet_files = list(data_path.glob("normalized_features_*.parquet"))
    if not parquet_files:
        print_warn("No feature files to validate")
        return True
    
    # Check the most recent file
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
    
    try:
        import polars as pl
        import numpy as np
        
        df = pl.read_parquet(latest_file)
        print_ok(f"Validating file: {latest_file.name}")
        
        validation_passed = True
        issues = []
        
        # Check for NaN/inf values
        for col in df.columns:
            if col == "ts":  # Skip timestamp
                continue
            
            series = df[col]
            
            # Check for nulls
            n_nulls = series.null_count()
            if n_nulls > 0:
                issues.append(f"{col}: {n_nulls} null values")
                validation_passed = False
            
            # Check for infinities (only for numeric columns)
            if series.dtype in [pl.Float32, pl.Float64]:
                arr = series.to_numpy()
                n_inf = np.sum(np.isinf(arr))
                if n_inf > 0:
                    issues.append(f"{col}: {n_inf} infinite values")
                    validation_passed = False
        
        if not issues:
            print_ok("No NaN or infinite values found")
        else:
            for issue in issues[:5]:  # Show first 5 issues
                print_warn(issue)
            if len(issues) > 5:
                print_warn(f"... and {len(issues) - 5} more issues")
        
        # Validate normalized feature distributions
        print("\n  Checking normalized distributions...")
        
        # Standard scaler features should have mean≈0, std≈1
        # Note: microprice, last_trade_px, etc. are RAW_PASSTHROUGH and not normalized
        standard_features = [
            "ofi", "ofi_10", "ofi_cumulative",
            "imbalance", "imbalance_1", "imbalance_5",
            "spread", "spread_bps",
        ]
        
        for feat in standard_features:
            if feat not in df.columns:
                continue
            
            series = df[feat].drop_nulls()
            if len(series) == 0:
                continue
            
            mean = float(series.mean())
            std = float(series.std())
            
            # Check if properly normalized
            if abs(mean) < 0.15 and 0.8 < std < 1.2:
                print_ok(f"{feat}: mean={mean:.3f}, std={std:.3f}")
            else:
                print_warn(f"{feat}: mean={mean:.3f}, std={std:.3f} (expected ≈0, ≈1)")
                validation_passed = False
        
        # Robust scaler features should be in reasonable range
        robust_features = [
            "bid_sz_01", "ask_sz_01", "taker_buy_volume", "taker_sell_volume"
        ]
        
        for feat in robust_features:
            if feat not in df.columns:
                continue
            
            series = df[feat].drop_nulls()
            if len(series) == 0:
                continue
            
            min_val = float(series.min())
            max_val = float(series.max())
            
            if -25 < min_val and max_val < 25:
                print_ok(f"{feat}: range=[{min_val:.1f}, {max_val:.1f}]")
            else:
                print_warn(f"{feat}: range=[{min_val:.1f}, {max_val:.1f}] (expected ≈[-20, 20])")
        
        if validation_passed:
            print_ok("\n  ✓ Normalization validation PASSED")
        else:
            print_warn("\n  ⚠ Normalization validation had warnings")
        
        return validation_passed
        
    except Exception as e:
        print_fail(f"Validation failed: {e}")
        return False


def check_output_directory():
    """Check data directory."""
    print_header("Data Directory Check")
    
    # Check common data locations
    data_paths = [
        Path("/data/features"),
        Path("./data/features"),
        Path("../data/features"),
    ]
    
    found = False
    for data_path in data_paths:
        if data_path.exists():
            print_ok(f"Data directory found: {data_path.absolute()}")
            
            # Check for parquet files
            parquet_files = list(data_path.glob("*.parquet"))
            if parquet_files:
                print_ok(f"Found {len(parquet_files)} parquet files")
                
                # Check one file
                try:
                    import polars as pl
                    df = pl.read_parquet(parquet_files[0])
                    print_ok(f"Sample file: {parquet_files[0].name}")
                    print_ok(f"  Rows: {len(df):,}")
                    print_ok(f"  Columns: {len(df.columns)}")
                except Exception as e:
                    print_warn(f"Could not read parquet file: {e}")
            else:
                print_warn("No parquet files found in data directory")
            
            found = True
            break
    
    if not found:
        print_warn("Data directory not found at standard locations")
        print_warn("You may need to specify --data-dir when running training scripts")
    
    return True


def check_output_directory():
    """Check output directory permissions."""
    print_header("Output Directory Check")
    
    output_paths = [
        Path("/outputs"),
        Path("./outputs"),
    ]
    
    for output_path in output_paths:
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Test write
            test_file = output_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            print_ok(f"Output directory writable: {output_path.absolute()}")
            return True
            
        except Exception as e:
            continue
    
    print_warn("Could not create writable output directory")
    return True


def run_all_checks():
    """Run all environment checks."""
    print("\n" + "="*60)
    print(" MIDAS GPU Training Environment Check")
    print("="*60)
    
    results = {
        "python": check_python_version(),
        "packages": check_packages(),
        "gpu": check_gpu(),
        "xgboost_gpu": check_xgboost_gpu(),
        "memory": check_memory(),
        "data": check_data_directory(),
        "normalization": check_normalization(),
        "output": check_output_directory(),
    }
    
    # Summary
    print_header("Summary")
    
    all_ok = all(results.values())
    
    if all_ok:
        print_ok("All checks passed!")
        print("\n  Your environment is ready for training.")
        print("\n  Quick start commands:")
        print("    python train_linear.py --model-type ridge")
        print("    python train_xgboost.py")
        print("    python train_lstm.py")
        print("    python backtest.py --model-type xgboost")
    else:
        print_fail("Some checks failed. Please fix the issues above.")
        failed = [k for k, v in results.items() if not v]
        print(f"\n  Failed checks: {', '.join(failed)}")
    
    return all_ok


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
