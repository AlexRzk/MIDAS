#!/usr/bin/env python3
"""
Linear Model Baseline Training for MIDAS.

Provides simple baselines:
- Ridge Regression
- Lasso Regression  
- ElasticNet
- Simple Linear Regression

Essential for comparing against complex models.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib

from utils import (
    get_logger, setup_experiment_dir, save_metrics,
    compute_regression_metrics, compute_directional_accuracy,
    compute_trading_metrics, plot_predictions, plot_feature_importance,
    get_device_info, OUTPUT_DIR
)
from dataset import MIDASDataset
from preprocessing import Preprocessor
from splitter import OFISafeSplitter

logger = get_logger("train_linear")


# ============================================
# Model Configurations
# ============================================

MODEL_CONFIGS = {
    "ridge": {
        "class": Ridge,
        "default_params": {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "auto",
        },
    },
    "lasso": {
        "class": Lasso,
        "default_params": {
            "alpha": 0.01,
            "fit_intercept": True,
            "max_iter": 10000,
        },
    },
    "elasticnet": {
        "class": ElasticNet,
        "default_params": {
            "alpha": 0.01,
            "l1_ratio": 0.5,
            "fit_intercept": True,
            "max_iter": 10000,
        },
    },
    "linear": {
        "class": LinearRegression,
        "default_params": {
            "fit_intercept": True,
        },
    },
}


def train_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "ridge",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a linear model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model ("ridge", "lasso", "elasticnet", "linear")
        params: Model parameters
        
    Returns:
        Trained model and training info
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    model_class = config["class"]
    
    # Merge default params with provided params
    model_params = config["default_params"].copy()
    if params:
        model_params.update(params)
    
    # Create and train model
    model = model_class(**model_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    info = {
        "train_time_seconds": train_time,
        "model_type": model_type,
        "params": model_params,
        "n_features": X_train.shape[1],
        "n_train_samples": len(X_train),
    }
    
    logger.info(f"Training complete in {train_time:.3f}s")
    
    return model, info


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ts_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    
    reg_metrics = compute_regression_metrics(y_test, y_pred)
    dir_metrics = compute_directional_accuracy(y_test, y_pred)
    
    trade_metrics = {}
    if ts_test is not None:
        trade_metrics = compute_trading_metrics(y_test, y_pred, ts_test)
    
    return {
        "regression": reg_metrics,
        "directional": dir_metrics,
        "trading": trade_metrics,
        "n_test_samples": len(y_test),
    }


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Dict[str, float]:
    """Get feature importance from linear model coefficients."""
    if hasattr(model, "coef_"):
        coef = model.coef_
        importance = {name: abs(float(c)) for name, c in zip(feature_names, coef)}
        # Sort by absolute importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance
    return {}


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "ridge",
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Simple grid search for linear model hyperparameters.
    """
    logger.info(f"Running hyperparameter search for {model_type}")
    
    if model_type == "ridge":
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = 1.0
        best_score = float("-inf")
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
            mean_score = scores.mean()
            
            logger.info(f"  alpha={alpha}: MSE={-mean_score:.6f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        return {"alpha": best_alpha}
    
    elif model_type == "lasso":
        alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
        best_alpha = 0.01
        best_score = float("-inf")
        
        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=10000)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
            mean_score = scores.mean()
            
            logger.info(f"  alpha={alpha}: MSE={-mean_score:.6f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        return {"alpha": best_alpha}
    
    elif model_type == "elasticnet":
        alphas = [0.001, 0.01, 0.1]
        l1_ratios = [0.2, 0.5, 0.8]
        best_params = {"alpha": 0.01, "l1_ratio": 0.5}
        best_score = float("-inf")
        
        for alpha in alphas:
            for l1_ratio in l1_ratios:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
                mean_score = scores.mean()
                
                logger.info(f"  alpha={alpha}, l1_ratio={l1_ratio}: MSE={-mean_score:.6f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"alpha": alpha, "l1_ratio": l1_ratio}
        
        return best_params
    
    return {}


def run_training_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    model_type: str = "ridge",
    target_type: str = "return",
    target_horizon: int = 10,
    train_ratio: float = 0.8,
    do_hyperparameter_search: bool = False,
) -> Dict[str, Any]:
    """
    Run full linear model training pipeline.
    """
    if output_dir is None:
        output_dir = setup_experiment_dir(f"linear_{model_type}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": f"linear_{model_type}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_info": get_device_info(),
        "config": {
            "model_type": model_type,
            "target_type": target_type,
            "target_horizon": target_horizon,
            "train_ratio": train_ratio,
        }
    }
    
    logger.info("=" * 60)
    logger.info(f"MIDAS Linear ({model_type.upper()}) Training Pipeline")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n[1/5] Loading data...")
    dataset = MIDASDataset(data_dir)
    dataset.load()
    logger.info(dataset.summary())
    
    # 2. Create target
    logger.info("\n[2/5] Creating target...")
    dataset.create_target(target_type, target_horizon)
    dataset.select_features()
    feature_names = dataset.feature_names
    logger.info(f"Using {len(feature_names)} features")
    
    # 3. Split data
    logger.info("\n[3/5] Splitting data (OFI-safe)...")
    splitter = OFISafeSplitter(
        bucket_size_us=100_000,
        train_ratio=train_ratio,
    )
    splitter.fit(dataset.df)
    train_df, test_df = splitter.get_all_train_test(dataset.df)
    
    train_df = train_df.drop_nulls(["target"])
    test_df = test_df.drop_nulls(["target"])
    
    logger.info(f"Train: {len(train_df):,} samples, Test: {len(test_df):,} samples")
    
    # 4. Preprocess
    logger.info("\n[4/5] Preprocessing...")
    prep = Preprocessor(normalize="zscore", clip_outliers_std=5.0)
    train_df = prep.fit_transform(train_df, feature_names)
    test_df = prep.transform(test_df)
    prep.save(output_dir / "preprocessor")
    
    X_train = train_df.select(feature_names).to_numpy()
    y_train = train_df["target"].to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["target"].to_numpy()
    ts_test = test_df["ts"].to_numpy() if "ts" in test_df.columns else None
    
    # 5. Train
    logger.info("\n[5/5] Training model...")
    
    params = None
    if do_hyperparameter_search:
        params = hyperparameter_search(X_train, y_train, model_type)
        logger.info(f"Best params: {params}")
    
    model, train_info = train_linear_model(X_train, y_train, model_type, params)
    results["training"] = train_info
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, ts_test)
    results["metrics"] = test_metrics
    
    logger.info(f"\n{'='*40}")
    logger.info("Test Set Results:")
    logger.info(f"  MSE: {test_metrics['regression']['mse']:.6f}")
    logger.info(f"  MAE: {test_metrics['regression']['mae']:.6f}")
    logger.info(f"  R²: {test_metrics['regression']['r2']:.4f}")
    logger.info(f"  Directional Accuracy: {test_metrics['directional']['accuracy']:.2%}")
    logger.info(f"{'='*40}")
    
    # Feature importance
    importance = get_feature_importance(model, feature_names)
    results["feature_importance"] = importance
    
    # Save outputs
    logger.info("\nSaving outputs...")
    
    # Save model
    joblib.dump(model, output_dir / "model.joblib")
    
    # Save metrics
    save_metrics(results, output_dir / "results.json")
    
    # Save feature importance
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    
    # Generate plots
    y_pred = model.predict(X_test)
    plot_predictions(y_test, y_pred, output_dir / "predictions.png", n_points=1000)
    
    if importance:
        plot_feature_importance(importance, output_dir / "feature_importance.png", top_n=20)
    
    # Save coefficients
    if hasattr(model, "coef_"):
        coef_dict = {name: float(c) for name, c in zip(feature_names, model.coef_)}
        with open(output_dir / "coefficients.json", "w") as f:
            json.dump(coef_dict, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results


def run_all_linear_models(
    data_dir: Path,
    output_base_dir: Optional[Path] = None,
    target_type: str = "return",
    target_horizon: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all linear model types for comparison.
    """
    if output_base_dir is None:
        output_base_dir = OUTPUT_DIR / f"linear_comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for model_type in ["ridge", "lasso", "elasticnet", "linear"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*60}")
        
        output_dir = output_base_dir / model_type
        
        try:
            results = run_training_pipeline(
                data_dir=data_dir,
                output_dir=output_dir,
                model_type=model_type,
                target_type=target_type,
                target_horizon=target_horizon,
                do_hyperparameter_search=True,
            )
            all_results[model_type] = results
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            all_results[model_type] = {"error": str(e)}
    
    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Summary")
    logger.info(f"{'='*60}")
    
    for model_type, results in all_results.items():
        if "error" in results:
            logger.info(f"  {model_type}: ERROR - {results['error']}")
        else:
            metrics = results["metrics"]["regression"]
            dir_acc = results["metrics"]["directional"]["accuracy"]
            logger.info(f"  {model_type}: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}, DirAcc={dir_acc:.2%}")
    
    # Save comparison
    comparison_path = output_base_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train linear models for MIDAS")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model-type", type=str, default="ridge",
                       choices=["ridge", "lasso", "elasticnet", "linear", "all"])
    parser.add_argument("--target-type", type=str, default="return")
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hyperparameter-search", action="store_true")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data/features")
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.model_type == "all":
        results = run_all_linear_models(
            data_dir=data_dir,
            output_base_dir=output_dir,
            target_type=args.target_type,
            target_horizon=args.target_horizon,
        )
    else:
        results = run_training_pipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            model_type=args.model_type,
            target_type=args.target_type,
            target_horizon=args.target_horizon,
            train_ratio=args.train_ratio,
            do_hyperparameter_search=args.hyperparameter_search,
        )
    
    return results


if __name__ == "__main__":
    main()
