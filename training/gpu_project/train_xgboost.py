#!/usr/bin/env python3
"""
XGBoost GPU Training for MIDAS.

Features:
- GPU-accelerated training (tree_method='hist' with device='cuda')
- Hyperparameter tuning with Optuna
- Feature importance analysis
- OFI-safe cross-validation
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from utils import (
    get_logger, setup_experiment_dir, save_metrics, 
    compute_regression_metrics, compute_directional_accuracy,
    compute_trading_metrics, plot_predictions, plot_feature_importance,
    check_gpu_availability, get_device_info, OUTPUT_DIR
)
from dataset import MIDASDataset, ALL_FEATURES
from preprocessing import Preprocessor
from splitter import OFISafeSplitter

logger = get_logger("train_xgboost")


# ============================================
# Default XGBoost Configuration
# ============================================

DEFAULT_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",  # XGBoost 3.x uses 'hist' for both CPU and GPU
    "device": "cpu",  # Will be updated to "cuda" if GPU available
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "verbosity": 1,
}


def get_gpu_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Update params for GPU training (XGBoost 3.x compatible)."""
    params = base_params.copy()
    
    if check_gpu_availability():
        logger.info("GPU available - enabling GPU training")
        # XGBoost 3.x: use 'hist' with device='cuda' instead of deprecated 'gpu_hist'
        params["tree_method"] = "hist"
        params["device"] = "cuda"
        # GPU doesn't need n_jobs
        params.pop("n_jobs", None)
    else:
        logger.info("GPU not available - using CPU training")
        params["tree_method"] = "hist"
        params["device"] = "cpu"
    
    return params


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        params: XGBoost parameters
        feature_names: List of feature names
        
    Returns:
        Trained model and training info
    """
    if params is None:
        params = get_gpu_params(DEFAULT_PARAMS.copy())
    
    # Extract fit params
    n_estimators = params.pop("n_estimators", 500)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)
    
    # Create model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **params
    )
    
    # Prepare eval set
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    # Train
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,
    )
    
    train_time = time.time() - start_time
    
    # Get best iteration
    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else n_estimators
    
    info = {
        "train_time_seconds": train_time,
        "best_iteration": best_iteration,
        "n_features": X_train.shape[1],
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val) if X_val is not None else 0,
    }
    
    logger.info(f"Training complete in {train_time:.1f}s, best iteration: {best_iteration}")
    
    return model, info


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ts_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute metrics
    reg_metrics = compute_regression_metrics(y_test, y_pred)
    dir_metrics = compute_directional_accuracy(y_test, y_pred)
    
    # Trading metrics if timestamps available
    if ts_test is not None:
        trade_metrics = compute_trading_metrics(y_test, y_pred, ts_test)
    else:
        trade_metrics = {}
    
    return {
        "regression": reg_metrics,
        "directional": dir_metrics,
        "trading": trade_metrics,
        "n_test_samples": len(y_test),
    }


def get_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: List[str],
    importance_type: str = "gain",
) -> Dict[str, float]:
    """Get feature importance scores."""
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Map to feature names
    result = {}
    for i, name in enumerate(feature_names):
        key = f"f{i}"
        result[name] = importance.get(key, 0)
    
    # Sort by importance
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    return result


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Hyperparameter search using Optuna.
    
    Returns best parameters found.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return DEFAULT_PARAMS
    
    base_params = get_gpu_params({})
    
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": base_params.get("tree_method", "hist"),
            "device": base_params.get("device", "cpu"),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "early_stopping_rounds": 30,
            "verbosity": 0,
        }
        
        model, _ = train_xgboost(X_train, y_train, X_val, y_val, params)
        
        y_pred = model.predict(X_val)
        mse = np.mean((y_val - y_pred) ** 2)
        
        return mse
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best trial MSE: {study.best_trial.value:.6f}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    # Merge with base params
    best_params = {**base_params, **study.best_trial.params}
    best_params["objective"] = "reg:squarederror"
    
    return best_params


def run_training_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    target_type: str = "return",
    target_horizon: int = 10,
    train_ratio: float = 0.8,
    do_hyperparameter_search: bool = False,
    n_trials: int = 30,
) -> Dict[str, Any]:
    """
    Run full XGBoost training pipeline.
    
    Args:
        data_dir: Directory with feature parquet files
        output_dir: Directory for outputs
        target_type: Type of target ("return", "direction", "price_delta")
        target_horizon: Number of ticks to look ahead
        train_ratio: Train/test split ratio
        do_hyperparameter_search: Whether to run Optuna search
        n_trials: Number of Optuna trials
        
    Returns:
        Results dictionary
    """
    # Setup
    if output_dir is None:
        output_dir = setup_experiment_dir("xgboost")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": "xgboost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_info": get_device_info(),
        "config": {
            "target_type": target_type,
            "target_horizon": target_horizon,
            "train_ratio": train_ratio,
        }
    }
    
    logger.info("=" * 60)
    logger.info("MIDAS XGBoost Training Pipeline")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n[1/6] Loading data...")
    dataset = MIDASDataset(data_dir)
    dataset.load()
    logger.info(dataset.summary())
    
    # 2. Create target
    logger.info("\n[2/6] Creating target...")
    dataset.create_target(target_type, target_horizon)
    
    # 3. Select features
    logger.info("\n[3/6] Selecting features...")
    dataset.select_features()
    feature_names = dataset.feature_names
    logger.info(f"Using {len(feature_names)} features")
    
    # 4. Split data (OFI-safe)
    logger.info("\n[4/6] Splitting data (OFI-safe)...")
    splitter = OFISafeSplitter(
        bucket_size_us=100_000,
        train_ratio=train_ratio,
    )
    splitter.fit(dataset.df)
    train_df, test_df = splitter.get_all_train_test(dataset.df)
    
    # Drop nulls in target
    train_df = train_df.drop_nulls(["target"])
    test_df = test_df.drop_nulls(["target"])
    
    logger.info(f"Train: {len(train_df):,} samples, Test: {len(test_df):,} samples")
    
    # 5. Preprocess
    logger.info("\n[5/6] Preprocessing...")
    prep = Preprocessor(normalize="zscore", clip_outliers_std=5.0)
    train_df = prep.fit_transform(train_df, feature_names)
    test_df = prep.transform(test_df)
    
    # Save preprocessor
    prep.save(output_dir / "preprocessor")
    
    # Convert to numpy
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["target"].to_numpy()
    ts_test = test_df["ts"].to_numpy() if "ts" in test_df.columns else None
    
    # Split train into train/val for early stopping
    val_size = int(len(X_train) * 0.15)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]
    
    # 6. Train model
    logger.info("\n[6/6] Training model...")
    
    if do_hyperparameter_search:
        logger.info("Running hyperparameter search...")
        params = hyperparameter_search(X_train_fit, y_train_fit, X_val, y_val, n_trials)
    else:
        params = get_gpu_params(DEFAULT_PARAMS.copy())
    
    model, train_info = train_xgboost(
        X_train_fit, y_train_fit,
        X_val, y_val,
        params,
        feature_names,
    )
    
    results["training"] = train_info
    results["params"] = {k: v for k, v in params.items() if not callable(v)}
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, ts_test)
    results["metrics"] = test_metrics
    
    logger.info(f"\n{'='*40}")
    logger.info("Test Set Results:")
    logger.info(f"  MSE: {test_metrics['regression']['mse']:.6f}")
    logger.info(f"  MAE: {test_metrics['regression']['mae']:.6f}")
    logger.info(f"  R²: {test_metrics['regression']['r2']:.4f}")
    logger.info(f"  Directional Accuracy: {test_metrics['directional']['directional_accuracy']:.2%}")
    logger.info(f"{'='*40}")
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    results["feature_importance"] = importance
    
    # Save outputs
    logger.info("\nSaving outputs...")
    
    # Save model
    model.save_model(str(output_dir / "model.json"))
    
    # Save metrics
    save_metrics(results, output_dir / "results.json")
    
    # Save feature importance
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    
    # Generate plots
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, output_dir / "predictions.png", n_points=1000)
    plot_feature_importance(importance, output_dir / "feature_importance.png", top_n=20)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model for MIDAS")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--target-type", type=str, default="return", 
                       choices=["return", "direction", "price_delta"])
    parser.add_argument("--target-horizon", type=int, default=10, help="Prediction horizon in ticks")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--hyperparameter-search", action="store_true", help="Run Optuna search")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data/features")
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    results = run_training_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        target_type=args.target_type,
        target_horizon=args.target_horizon,
        train_ratio=args.train_ratio,
        do_hyperparameter_search=args.hyperparameter_search,
        n_trials=args.n_trials,
    )
    
    return results


if __name__ == "__main__":
    main()
