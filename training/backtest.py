#!/usr/bin/env python3
"""
Walk-forward backtester for TFT model predictions.

Usage:
    python training/backtest.py --model models/best_model.pt --data-dir data/features/
    python training/backtest.py --model models/tft-10-0.0012.ckpt --output reports/backtest_results.csv
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import MIDASDataModule, FeatureNormalizer, DEFAULT_FEATURES, DEFAULT_TARGETS
from training.model import TemporalFusionTransformerModel, TFTLightningModule

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NaiveExecutionModel:
    """
    Simple execution model for backtesting.
    
    Strategy: 
    - If predicted price > current price by threshold: BUY
    - If predicted price < current price by threshold: SELL
    - Otherwise: HOLD
    """
    
    def __init__(
        self,
        threshold_pct: float = 0.001,  # 0.1% threshold
        position_size: float = 1.0,
        transaction_cost: float = 0.0002,  # 0.02% per trade
    ):
        self.threshold_pct = threshold_pct
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        
        self.position = 0.0
        self.cash = 10000.0
        self.initial_cash = self.cash
        self.trades: List[Dict] = []
    
    def decide(self, current_price: float, predicted_price: float) -> str:
        """Make trading decision."""
        predicted_return = (predicted_price - current_price) / current_price
        
        if predicted_return > self.threshold_pct:
            return "BUY"
        elif predicted_return < -self.threshold_pct:
            return "SELL"
        return "HOLD"
    
    def execute(
        self,
        timestamp: int,
        current_price: float,
        predicted_price: float,
    ) -> Dict:
        """Execute trading decision."""
        action = self.decide(current_price, predicted_price)
        
        trade = {
            "timestamp": timestamp,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "action": action,
            "position_before": self.position,
            "cash_before": self.cash,
        }
        
        if action == "BUY" and self.position <= 0:
            # Close any short position and go long
            if self.position < 0:
                # Close short
                cost = abs(self.position) * current_price * (1 + self.transaction_cost)
                self.cash -= cost
            
            # Open long
            shares = (self.cash * self.position_size) / (current_price * (1 + self.transaction_cost))
            self.position = shares
            self.cash -= shares * current_price * (1 + self.transaction_cost)
            
        elif action == "SELL" and self.position >= 0:
            # Close any long position and go short
            if self.position > 0:
                # Close long
                self.cash += self.position * current_price * (1 - self.transaction_cost)
            
            # Open short (simplified - just track as negative position)
            shares = (self.cash * self.position_size) / (current_price * (1 + self.transaction_cost))
            self.position = -shares
            self.cash += shares * current_price * (1 - self.transaction_cost)
        
        trade["position_after"] = self.position
        trade["cash_after"] = self.cash
        trade["portfolio_value"] = self.cash + self.position * current_price
        
        self.trades.append(trade)
        return trade
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Get current portfolio value."""
        return self.cash + self.position * current_price
    
    def get_returns(self, current_price: float) -> float:
        """Get total returns."""
        return (self.get_portfolio_value(current_price) / self.initial_cash) - 1


def calculate_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics from returns series.
    """
    if len(returns) == 0:
        return {}
    
    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns[-1] - 1
    
    # Sharpe ratio (annualized, assuming minute data)
    # 525600 minutes per year
    periods_per_year = 525600 / 60  # Assuming 1-hour data
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = np.sqrt(periods_per_year) * mean_return / std_return if std_return > 0 else 0
    
    # Maximum drawdown
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Win rate
    n_positive = (returns > 0).sum()
    win_rate = n_positive / len(returns) if len(returns) > 0 else 0
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "n_trades": len(returns),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
    }


def walk_forward_backtest(
    model: torch.nn.Module,
    data_module: MIDASDataModule,
    device: torch.device,
    execution_model: NaiveExecutionModel,
    normalizer: FeatureNormalizer,
    target_col: str = "close",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run walk-forward backtest.
    
    Returns:
        trades_df: DataFrame of all trades
        metrics: Performance metrics dict
    """
    model.eval()
    
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_actuals = []
    all_timestamps = []
    
    logger.info("Running predictions on test set...")
    
    with torch.no_grad():
        for batch_idx, (x, y, ts) in enumerate(test_loader):
            x = x.to(device)
            
            # Get predictions (using median quantile)
            if hasattr(model, "predict_point"):
                predictions = model.predict_point(x)
            else:
                predictions, _ = model(x)
                # Extract median from quantile predictions
                batch_size, output_length, n_out = predictions.shape
                n_targets = y.shape[-1]
                n_quantiles = n_out // n_targets
                predictions = predictions.view(batch_size, output_length, n_targets, n_quantiles)
                predictions = predictions[..., n_quantiles // 2]  # Median
            
            # Take first prediction step for execution
            pred_step1 = predictions[:, 0, :].cpu().numpy()
            actual_step1 = y[:, 0, :].cpu().numpy()
            ts_step1 = ts[:, 0].numpy()
            
            all_predictions.extend(pred_step1.tolist())
            all_actuals.extend(actual_step1.tolist())
            all_timestamps.extend(ts_step1.tolist())
    
    logger.info(f"Generated {len(all_predictions)} predictions")
    
    # Denormalize predictions and actuals
    predictions_denorm = normalizer.inverse_transform(
        np.array(all_predictions), target_col
    )
    actuals_denorm = normalizer.inverse_transform(
        np.array(all_actuals), target_col
    )
    
    # Execute trades
    logger.info("Executing trades...")
    
    portfolio_values = []
    
    for i in range(len(predictions_denorm)):
        current_price = float(actuals_denorm[i])
        predicted_price = float(predictions_denorm[i])
        timestamp = int(all_timestamps[i])
        
        trade = execution_model.execute(timestamp, current_price, predicted_price)
        portfolio_values.append(trade["portfolio_value"])
    
    # Calculate returns
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate metrics
    metrics = calculate_metrics(returns)
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(execution_model.trades)
    
    # Add prediction error
    trades_df["prediction_error"] = (
        trades_df["predicted_price"] - trades_df["current_price"]
    ) / trades_df["current_price"]
    
    return trades_df, metrics


def plot_results(
    trades_df: pd.DataFrame,
    metrics: Dict,
    output_path: Path,
):
    """Generate backtest visualization plots."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Portfolio value
    ax1 = axes[0]
    ax1.plot(trades_df["portfolio_value"], label="Portfolio Value")
    ax1.set_title("Portfolio Value Over Time")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Value ($)")
    ax1.legend()
    ax1.grid(True)
    
    # Cumulative returns
    ax2 = axes[1]
    returns = trades_df["portfolio_value"].pct_change().fillna(0)
    cum_returns = (1 + returns).cumprod() - 1
    ax2.plot(cum_returns * 100, label="Cumulative Return", color="green")
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax2.set_title("Cumulative Returns")
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Return (%)")
    ax2.legend()
    ax2.grid(True)
    
    # Prediction accuracy
    ax3 = axes[2]
    ax3.scatter(
        range(len(trades_df)),
        trades_df["prediction_error"] * 100,
        alpha=0.5,
        s=5,
    )
    ax3.axhline(y=0, color="r", linestyle="--")
    ax3.set_title("Prediction Error Distribution")
    ax3.set_xlabel("Trade #")
    ax3.set_ylabel("Error (%)")
    ax3.grid(True)
    
    plt.tight_layout()
    
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for TFT model")
    
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing feature Parquet files",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=None,
        help="Path to normalizer JSON (default: models/normalizer.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/backtest_results.csv"),
        help="Output CSV file for trades",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Trading threshold (0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0002,
        help="Transaction cost per trade",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=60,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=10,
        help="Output sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load normalizer
    normalizer_path = args.normalizer or Path("models/normalizer.json")
    if not normalizer_path.exists():
        logger.error(f"Normalizer not found at {normalizer_path}")
        sys.exit(1)
    
    normalizer = FeatureNormalizer()
    normalizer.load(normalizer_path)
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = MIDASDataModule(
        data_dir=args.data_dir,
        feature_columns=DEFAULT_FEATURES,
        target_columns=DEFAULT_TARGETS,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
    )
    data_module.setup(stage="test")
    data_module.normalizer = normalizer  # Use loaded normalizer
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    
    if args.model.suffix == ".ckpt":
        # Lightning checkpoint
        model = TFTLightningModule.load_from_checkpoint(str(args.model))
    else:
        # Standard PyTorch checkpoint
        checkpoint = torch.load(args.model, map_location=device)
        
        # Recreate model (need to know architecture)
        model = TemporalFusionTransformerModel(
            n_features=data_module.n_features,
            n_targets=data_module.n_targets,
            input_length=args.input_length,
            output_length=args.output_length,
        )
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Create execution model
    execution_model = NaiveExecutionModel(
        threshold_pct=args.threshold,
        transaction_cost=args.transaction_cost,
    )
    
    # Run backtest
    trades_df, metrics = walk_forward_backtest(
        model=model if not hasattr(model, "model") else model.model,
        data_module=data_module,
        device=device,
        execution_model=execution_model,
        normalizer=normalizer,
        target_col="close",
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Number of Trades: {metrics['n_trades']}")
    print("=" * 60)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(args.output, index=False)
    logger.info(f"Saved trades to {args.output}")
    
    # Save metrics JSON
    metrics_path = args.output.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate plots
    if not args.no_plot:
        plot_results(trades_df, metrics, args.output)


if __name__ == "__main__":
    main()
