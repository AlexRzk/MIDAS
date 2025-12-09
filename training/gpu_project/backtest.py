#!/usr/bin/env python3
"""
Walk-Forward Backtesting Engine for MIDAS.

Features:
- Walk-forward validation (rolling training windows)
- Multiple trading strategies
- PnL curves with transaction costs
- Performance metrics (Sharpe, Sortino, Max Drawdown)
- JSON output for results
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from utils import (
    get_logger, setup_experiment_dir, save_metrics,
    compute_regression_metrics, compute_directional_accuracy,
    OUTPUT_DIR
)
from dataset import MIDASDataset
from preprocessing import Preprocessor
from splitter import OFISafeSplitter, ContinuousSegment

logger = get_logger("backtest")


# ============================================
# Trading Strategies
# ============================================

def threshold_strategy(
    predictions: np.ndarray,
    threshold: float = 0.5,  # In basis points
) -> np.ndarray:
    """
    Simple threshold strategy.
    
    Buy (+1) if prediction > threshold
    Sell (-1) if prediction < -threshold
    Hold (0) otherwise
    """
    signals = np.zeros(len(predictions))
    signals[predictions > threshold] = 1
    signals[predictions < -threshold] = -1
    return signals


def proportional_strategy(
    predictions: np.ndarray,
    scale: float = 1.0,
    clip: float = 1.0,
) -> np.ndarray:
    """
    Position proportional to prediction magnitude.
    
    Signal = clip(prediction * scale, -1, 1)
    """
    signals = predictions * scale
    return np.clip(signals, -clip, clip)


def sign_strategy(predictions: np.ndarray) -> np.ndarray:
    """Simple sign strategy: +1 for positive, -1 for negative."""
    return np.sign(predictions)


# ============================================
# PnL Calculation
# ============================================

@dataclass
class BacktestResult:
    """Results from a single backtest."""
    timestamps: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray
    signals: np.ndarray
    returns: np.ndarray
    cumulative_pnl: np.ndarray
    n_trades: int
    
    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Model metrics
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0


def calculate_pnl(
    signals: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost_bps: float = 0.1,  # 0.1 bps per trade
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate PnL from signals and actual returns.
    
    Args:
        signals: Position signals (-1, 0, 1 or continuous)
        actual_returns: Actual returns in basis points
        transaction_cost_bps: Cost per trade in basis points
        
    Returns:
        returns: Per-period returns
        cumulative_pnl: Cumulative PnL
        n_trades: Number of position changes
    """
    # Calculate strategy returns
    strategy_returns = signals[:-1] * actual_returns[1:]
    
    # Calculate transaction costs
    position_changes = np.abs(np.diff(signals))
    transaction_costs = position_changes[:-1] * transaction_cost_bps
    
    # Pad to match lengths
    strategy_returns = np.concatenate([[0], strategy_returns])
    transaction_costs = np.concatenate([[0], transaction_costs, [0]])
    
    # Net returns
    net_returns = strategy_returns - transaction_costs[:len(strategy_returns)]
    
    # Cumulative PnL
    cumulative_pnl = np.cumsum(net_returns)
    
    # Count trades
    n_trades = int(np.sum(position_changes > 0))
    
    return net_returns, cumulative_pnl, n_trades


def calculate_performance_metrics(
    returns: np.ndarray,
    cumulative_pnl: np.ndarray,
    annualization_factor: float = 252 * 24 * 60 * 60 * 10,  # For 100ms data
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    """
    # Filter out zero returns for win rate
    nonzero_returns = returns[returns != 0]
    
    if len(nonzero_returns) == 0:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    
    # Total return
    total_return = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0
    
    # Sharpe ratio (annualized)
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor)
    else:
        sharpe = 0.0
    
    # Sortino ratio (only downside volatility)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino = returns.mean() / negative_returns.std() * np.sqrt(annualization_factor)
    else:
        sortino = sharpe
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
    
    # Win rate
    wins = np.sum(nonzero_returns > 0)
    win_rate = wins / len(nonzero_returns) if len(nonzero_returns) > 0 else 0.0
    
    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }


# ============================================
# Walk-Forward Backtester
# ============================================

class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.
    
    Trains models on expanding/rolling windows and tests
    on subsequent out-of-sample data.
    """
    
    def __init__(
        self,
        model_factory: Callable,
        strategy: Callable = threshold_strategy,
        strategy_params: Dict[str, Any] = None,
        n_splits: int = 5,
        train_window: Optional[int] = None,  # None = expanding window
        test_window: Optional[int] = None,  # None = use remaining data
        transaction_cost_bps: float = 0.1,
    ):
        """
        Args:
            model_factory: Callable that returns (train_fn, predict_fn)
            strategy: Signal generation strategy
            strategy_params: Parameters for strategy
            n_splits: Number of walk-forward splits
            train_window: Size of training window (None = expanding)
            test_window: Size of test window
            transaction_cost_bps: Transaction cost in basis points
        """
        self.model_factory = model_factory
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost_bps = transaction_cost_bps
        
        self.results: List[BacktestResult] = []
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        feature_names: List[str],
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtest.
        """
        n_samples = len(X)
        
        # Calculate split points
        if self.test_window:
            test_size = self.test_window
        else:
            test_size = n_samples // (self.n_splits + 1)
        
        self.results = []
        
        for i in range(self.n_splits):
            logger.info(f"\n{'='*40}")
            logger.info(f"Walk-Forward Split {i+1}/{self.n_splits}")
            logger.info(f"{'='*40}")
            
            # Calculate indices
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            
            if self.train_window:
                train_start = max(0, test_start - self.train_window)
            else:
                train_start = 0
            
            train_end = test_start
            
            if train_end <= train_start:
                logger.warning(f"Skipping split {i+1}: not enough training data")
                continue
            
            # Split data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            ts_test = ts[test_start:test_end]
            
            logger.info(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
            
            # Train model
            train_fn, predict_fn = self.model_factory()
            train_fn(X_train, y_train)
            
            # Predict
            predictions = predict_fn(X_test)
            
            # Generate signals
            signals = self.strategy(predictions, **self.strategy_params)
            
            # Calculate PnL
            returns, cumulative_pnl, n_trades = calculate_pnl(
                signals, y_test, self.transaction_cost_bps
            )
            
            # Performance metrics
            perf_metrics = calculate_performance_metrics(returns, cumulative_pnl)
            
            # Model metrics
            reg_metrics = compute_regression_metrics(y_test, predictions)
            dir_metrics = compute_directional_accuracy(y_test, predictions)
            
            # Create result
            result = BacktestResult(
                timestamps=ts_test,
                predictions=predictions,
                actuals=y_test,
                signals=signals,
                returns=returns,
                cumulative_pnl=cumulative_pnl,
                n_trades=n_trades,
                total_return=perf_metrics["total_return"],
                sharpe_ratio=perf_metrics["sharpe_ratio"],
                sortino_ratio=perf_metrics["sortino_ratio"],
                max_drawdown=perf_metrics["max_drawdown"],
                win_rate=perf_metrics["win_rate"],
                profit_factor=perf_metrics["profit_factor"],
                mse=reg_metrics["mse"],
                mae=reg_metrics["mae"],
                r2=reg_metrics["r2"],
                directional_accuracy=dir_metrics["accuracy"],
            )
            
            self.results.append(result)
            
            # Log results
            logger.info(f"Total Return: {result.total_return:.2f} bps")
            logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {result.max_drawdown:.2f} bps")
            logger.info(f"Win Rate: {result.win_rate:.2%}")
            logger.info(f"N Trades: {result.n_trades}")
            logger.info(f"Directional Accuracy: {result.directional_accuracy:.2%}")
        
        return self.results
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all splits."""
        if not self.results:
            return {}
        
        metrics = {
            "n_splits": len(self.results),
            "avg_total_return": np.mean([r.total_return for r in self.results]),
            "std_total_return": np.std([r.total_return for r in self.results]),
            "avg_sharpe": np.mean([r.sharpe_ratio for r in self.results]),
            "avg_max_drawdown": np.mean([r.max_drawdown for r in self.results]),
            "avg_win_rate": np.mean([r.win_rate for r in self.results]),
            "total_trades": sum(r.n_trades for r in self.results),
            "avg_directional_accuracy": np.mean([r.directional_accuracy for r in self.results]),
            "avg_mse": np.mean([r.mse for r in self.results]),
            "avg_r2": np.mean([r.r2 for r in self.results]),
        }
        
        return metrics


def plot_backtest_results(
    results: List[BacktestResult],
    output_path: Path,
    title: str = "Walk-Forward Backtest Results",
):
    """Plot backtest results."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Cumulative PnL across all splits
    ax1 = axes[0]
    cumulative_offset = 0
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        pnl = result.cumulative_pnl + cumulative_offset
        ax1.plot(pnl, color=colors[i], label=f"Split {i+1}")
        cumulative_offset = pnl[-1]
    
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Cumulative PnL (bps)")
    ax1.set_title(f"{title} - Cumulative PnL")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-split returns
    ax2 = axes[1]
    split_returns = [r.total_return for r in results]
    colors_bar = ["green" if r > 0 else "red" for r in split_returns]
    ax2.bar(range(1, len(results) + 1), split_returns, color=colors_bar)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Total Return (bps)")
    ax2.set_title("Per-Split Returns")
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics comparison
    ax3 = axes[2]
    metrics_to_plot = ["sharpe_ratio", "win_rate", "directional_accuracy"]
    x = np.arange(len(results))
    width = 0.25
    
    for j, metric in enumerate(metrics_to_plot):
        values = [getattr(r, metric) for r in results]
        if metric in ["win_rate", "directional_accuracy"]:
            values = [v * 100 for v in values]  # Convert to percentage
        ax3.bar(x + j * width, values, width, label=metric.replace("_", " ").title())
    
    ax3.set_xlabel("Split")
    ax3.set_ylabel("Value")
    ax3.set_title("Performance Metrics by Split")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f"Split {i+1}" for i in range(len(results))])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved backtest plot to {output_path}")


def create_xgboost_factory():
    """Factory for XGBoost model."""
    import xgboost as xgb
    
    model = None
    
    def train_fn(X, y):
        nonlocal model
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
    
    def predict_fn(X):
        return model.predict(X)
    
    return train_fn, predict_fn


def create_ridge_factory():
    """Factory for Ridge regression model."""
    from sklearn.linear_model import Ridge
    
    model = None
    
    def train_fn(X, y):
        nonlocal model
        model = Ridge(alpha=1.0)
        model.fit(X, y)
    
    def predict_fn(X):
        return model.predict(X)
    
    return train_fn, predict_fn


def run_backtest_pipeline(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    model_type: str = "xgboost",
    target_type: str = "return",
    target_horizon: int = 10,
    n_splits: int = 5,
    strategy_threshold: float = 0.5,
    transaction_cost_bps: float = 0.1,
) -> Dict[str, Any]:
    """
    Run full backtesting pipeline.
    """
    if output_dir is None:
        output_dir = setup_experiment_dir(f"backtest_{model_type}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target_type": target_type,
            "target_horizon": target_horizon,
            "n_splits": n_splits,
            "strategy_threshold": strategy_threshold,
            "transaction_cost_bps": transaction_cost_bps,
        }
    }
    
    logger.info("=" * 60)
    logger.info(f"MIDAS Walk-Forward Backtest ({model_type.upper()})")
    logger.info("=" * 60)
    
    # Load data
    logger.info("\n[1/4] Loading data...")
    dataset = MIDASDataset(data_dir)
    dataset.load()
    dataset.create_target(target_type, target_horizon)
    dataset.select_features()
    feature_names = dataset.feature_names
    
    # Prepare data
    logger.info("\n[2/4] Preparing data...")
    df = dataset.df.drop_nulls(["target"])
    
    prep = Preprocessor(normalize="zscore", clip_outliers_std=5.0)
    df = prep.fit_transform(df, feature_names)
    
    X = df.select(feature_names).to_numpy()
    y = df["target"].to_numpy()
    ts = df["ts"].to_numpy() if "ts" in df.columns else np.arange(len(X))
    
    logger.info(f"Total samples: {len(X):,}")
    
    # Create model factory
    if model_type == "xgboost":
        model_factory = create_xgboost_factory
    elif model_type == "ridge":
        model_factory = create_ridge_factory
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Run backtest
    logger.info("\n[3/4] Running walk-forward backtest...")
    backtester = WalkForwardBacktester(
        model_factory=model_factory,
        strategy=threshold_strategy,
        strategy_params={"threshold": strategy_threshold},
        n_splits=n_splits,
        transaction_cost_bps=transaction_cost_bps,
    )
    
    bt_results = backtester.run(X, y, ts, feature_names)
    
    # Aggregate metrics
    aggregate_metrics = backtester.get_aggregate_metrics()
    results["aggregate_metrics"] = aggregate_metrics
    
    # Per-split metrics
    results["splits"] = []
    for i, r in enumerate(bt_results):
        split_data = {
            "split": i + 1,
            "total_return": r.total_return,
            "sharpe_ratio": r.sharpe_ratio,
            "sortino_ratio": r.sortino_ratio,
            "max_drawdown": r.max_drawdown,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "n_trades": r.n_trades,
            "mse": r.mse,
            "r2": r.r2,
            "directional_accuracy": r.directional_accuracy,
        }
        results["splits"].append(split_data)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Backtest Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Avg Total Return: {aggregate_metrics['avg_total_return']:.2f} ± {aggregate_metrics['std_total_return']:.2f} bps")
    logger.info(f"Avg Sharpe Ratio: {aggregate_metrics['avg_sharpe']:.2f}")
    logger.info(f"Avg Max Drawdown: {aggregate_metrics['avg_max_drawdown']:.2f} bps")
    logger.info(f"Avg Win Rate: {aggregate_metrics['avg_win_rate']:.2%}")
    logger.info(f"Avg Directional Accuracy: {aggregate_metrics['avg_directional_accuracy']:.2%}")
    logger.info(f"Total Trades: {aggregate_metrics['total_trades']}")
    
    # Save outputs
    logger.info("\n[4/4] Saving outputs...")
    
    save_metrics(results, output_dir / "backtest_results.json")
    plot_backtest_results(bt_results, output_dir / "backtest_plot.png")
    
    # Save PnL curves as CSV
    all_pnl = []
    offset = 0
    for i, r in enumerate(bt_results):
        for j, pnl in enumerate(r.cumulative_pnl):
            all_pnl.append({
                "split": i + 1,
                "idx": j,
                "cumulative_pnl": pnl + offset,
            })
        offset += r.cumulative_pnl[-1]
    
    pnl_df = pl.DataFrame(all_pnl)
    pnl_df.write_csv(output_dir / "pnl_curve.csv")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run walk-forward backtest for MIDAS")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "ridge"])
    parser.add_argument("--target-type", type=str, default="return")
    parser.add_argument("--target-horizon", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--transaction-cost", type=float, default=0.1)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else Path("/data/features")
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    results = run_backtest_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model_type=args.model_type,
        target_type=args.target_type,
        target_horizon=args.target_horizon,
        n_splits=args.n_splits,
        strategy_threshold=args.threshold,
        transaction_cost_bps=args.transaction_cost,
    )
    
    return results


if __name__ == "__main__":
    main()
