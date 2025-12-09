#!/usr/bin/env python3
"""
Utility functions for MIDAS GPU training pipeline.
Includes logging, paths, metrics computation, and common helpers.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# ============================================
# Path Configuration
# ============================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path("/root/data")
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_model_dir(model_name: str) -> Path:
    """Get or create model-specific directory."""
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_results_dir(model_name: str) -> Path:
    """Get or create results directory for a model."""
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================
# Logging
# ============================================

class Logger:
    """Simple logger with timestamps and file output."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.name = name
        self.log_file = log_file
        
    def _log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] [{self.name}] {message}"
        print(formatted)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")
    
    def info(self, message: str):
        self._log("INFO", message)
    
    def warning(self, message: str):
        self._log("WARN", message)
    
    def error(self, message: str):
        self._log("ERROR", message)
    
    def debug(self, message: str):
        self._log("DEBUG", message)


def get_logger(name: str, model_name: Optional[str] = None) -> Logger:
    """Get a logger instance."""
    log_file = None
    if model_name:
        log_file = get_results_dir(model_name) / "training.log"
    return Logger(name, log_file)


# ============================================
# Metrics Computation
# ============================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    
    # MSE, RMSE, MAE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_directional_accuracy(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """Compute directional accuracy metrics.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
        threshold: If None, uses sign-based (any non-zero). 
                   If provided, classifies as flat if |return| < threshold.
                   Can also be 'percentile' to auto-compute from data distribution.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Auto-compute threshold based on percentiles if requested
    if threshold == 'percentile':
        # Use 33rd/67th percentile to create balanced classes
        threshold = np.percentile(np.abs(y_true), 33)
    
    # Classify directions
    if threshold is None or threshold == 0:
        # Original behavior: strict sign
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)
    else:
        # Threshold-based classification
        true_dir = np.where(y_true > threshold, 1, np.where(y_true < -threshold, -1, 0))
        pred_dir = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    
    # Overall accuracy
    correct = (true_dir == pred_dir).sum()
    total = len(true_dir)
    accuracy = correct / total if total > 0 else 0.0
    
    # Up accuracy (only on non-flat true movements)
    up_mask = true_dir > 0
    up_correct = ((pred_dir > 0) & up_mask).sum()
    up_total = up_mask.sum()
    up_accuracy = up_correct / up_total if up_total > 0 else 0.0
    
    # Down accuracy (only on non-flat true movements)
    down_mask = true_dir < 0
    down_correct = ((pred_dir < 0) & down_mask).sum()
    down_total = down_mask.sum()
    down_accuracy = down_correct / down_total if down_total > 0 else 0.0
    
    return {
        "directional_accuracy": float(accuracy),
        "up_accuracy": float(up_accuracy),
        "down_accuracy": float(down_accuracy),
        "up_count": int(up_total),
        "down_count": int(down_total),
        "flat_count": int((true_dir == 0).sum()),
        "threshold_used": float(threshold) if threshold not in [None, 'percentile'] else 0.0,
    }


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: Optional[np.ndarray] = None,
    transaction_cost_bps: float = 0.5,
) -> Dict[str, float]:
    """Compute trading-specific metrics."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Simple PnL: go long if predict up, short if predict down
    positions = np.sign(y_pred)
    returns = y_true * positions
    
    # Transaction costs (on position changes)
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * transaction_cost_bps / 10000
    net_returns = returns - costs
    
    # Cumulative PnL
    cumulative_pnl = np.cumsum(net_returns)
    
    # Sharpe ratio (assuming returns are in bps)
    if len(net_returns) > 1 and np.std(net_returns) > 0:
        sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252 * 24 * 60)  # Annualized
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Win rate
    wins = (net_returns > 0).sum()
    losses = (net_returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    # Profit factor
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "total_pnl_bps": float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0,
        "sharpe_ratio": float(sharpe),
        "max_drawdown_bps": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "num_trades": int(position_changes.sum() / 2),
        "avg_return_bps": float(np.mean(net_returns)),
    }


# ============================================
# File I/O
# ============================================

def save_metrics(metrics: Dict[str, Any], filepath: Path):
    """Save metrics to JSON file."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(filepath: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: Path,
):
    """Save predictions to parquet."""
    import polars as pl
    
    df = pl.DataFrame({
        "ts": timestamps,
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_true - y_pred,
    })
    df.write_parquet(filepath)


# ============================================
# Plotting
# ============================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filepath: Path,
    max_points: int = 2000,
):
    """Plot actual vs predicted values."""
    import matplotlib.pyplot as plt
    
    # Subsample for visualization
    if len(y_true) > max_points:
        indices = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series
    ax1 = axes[0, 0]
    ax1.plot(y_true, label="Actual", alpha=0.7, linewidth=0.5)
    ax1.plot(y_pred, label="Predicted", alpha=0.7, linewidth=0.5)
    ax1.set_title(f"{title} - Time Series")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=1)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
    ax2.set_title("Actual vs Predicted")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    ax3.axvline(0, color="r", linestyle="--")
    ax3.set_title("Residual Distribution")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    
    # Cumulative PnL
    positions = np.sign(y_pred)
    pnl = np.cumsum(y_true * positions)
    ax4 = axes[1, 1]
    ax4.plot(pnl, color="green", linewidth=1)
    ax4.axhline(0, color="r", linestyle="--", alpha=0.5)
    ax4.fill_between(range(len(pnl)), pnl, 0, alpha=0.3, color="green")
    ax4.set_title("Cumulative PnL (Simple Strategy)")
    ax4.set_xlabel("Sample")
    ax4.set_ylabel("Cumulative PnL (bps)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_pnl_curve(
    cumulative_pnl: np.ndarray,
    title: str,
    filepath: Path,
):
    """Plot PnL curve with drawdown."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # PnL curve
    ax1 = axes[0]
    ax1.plot(cumulative_pnl, color="blue", linewidth=1)
    ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=(cumulative_pnl >= 0), alpha=0.3, color="green")
    ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=(cumulative_pnl < 0), alpha=0.3, color="red")
    ax1.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax1.set_title(title)
    ax1.set_ylabel("Cumulative PnL (bps)")
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    ax2 = axes[1]
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color="red")
    ax2.set_title("Drawdown")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Drawdown (bps)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    title: str,
    filepath: Path,
    top_n: int = 30,
):
    """Plot feature importance."""
    import matplotlib.pyplot as plt
    
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, align="center", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


# ============================================
# GPU Utilities
# ============================================

def check_gpu() -> Dict[str, Any]:
    """Check GPU availability and info."""
    result = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "recommended_device": "cpu",
    }
    
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["device_count"] = torch.cuda.device_count()
        
        if result["cuda_available"]:
            for i in range(result["device_count"]):
                props = torch.cuda.get_device_properties(i)
                result["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1e9,
                    "major": props.major,
                    "minor": props.minor,
                })
            result["recommended_device"] = "cuda:0"
    except ImportError:
        pass
    
    return result


def get_device():
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda:0")
    except ImportError:
        pass
    return "cpu"


# ============================================
# Data Utilities
# ============================================

def microseconds_to_datetime(ts: int) -> datetime:
    """Convert microsecond timestamp to datetime."""
    return datetime.fromtimestamp(ts / 1_000_000)


def datetime_to_microseconds(dt: datetime) -> int:
    """Convert datetime to microsecond timestamp."""
    return int(dt.timestamp() * 1_000_000)


# ============================================
# Experiment Setup
# ============================================

OUTPUT_DIR = Path("/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_experiment_dir(experiment_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Create and return an experiment directory with timestamp.
    
    Args:
        experiment_name: Name of the experiment (e.g., 'xgboost', 'tft')
        base_dir: Base directory for experiments (default: OUTPUT_DIR)
    
    Returns:
        Path to the created experiment directory
    """
    if base_dir is None:
        base_dir = OUTPUT_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    return exp_dir


def check_gpu_availability() -> bool:
    """Check if GPU is available for training."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    return check_gpu()
