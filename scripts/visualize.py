"""
MIDAS Data Visualization Tool

Interactive visualizations for order book, trades, and features.
Requires: matplotlib, seaborn (optional)

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --feature ofi
    python scripts/visualize.py --trades
"""
import sys
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    HAS_PLT = True
except ImportError:
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
    HAS_PLT = False


class DataVisualizer:
    """Visualize MIDAS data."""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            base_path = Path("/data") if Path("/data").exists() else Path("./data")
        
        self.base_path = base_path
        self.raw_path = base_path / "raw"
        self.clean_path = base_path / "clean"
        self.features_path = base_path / "features"
        
        if not HAS_PLT:
            raise ImportError("matplotlib is required for visualization")
    
    def load_latest_features(self, limit: int = 10000) -> pl.DataFrame:
        """Load the most recent feature data."""
        files = sorted(self.features_path.glob("features_*.parquet"))
        if not files:
            raise FileNotFoundError("No feature files found")
        
        df = pl.read_parquet(files[-1])
        
        # Convert timestamp to datetime for plotting
        df = df.with_columns([
            (pl.col("ts") / 1_000_000).cast(pl.Int64).alias("ts_sec")
        ])
        
        # Take last N rows
        if len(df) > limit:
            df = df.tail(limit)
        
        return df
    
    def load_latest_clean(self, limit: int = 10000) -> pl.DataFrame:
        """Load the most recent clean data."""
        files = sorted(self.clean_path.glob("clean_*.parquet"))
        if not files:
            raise FileNotFoundError("No clean files found")
        
        df = pl.read_parquet(files[-1])
        
        # Convert timestamp
        df = df.with_columns([
            (pl.col("ts") / 1_000_000).cast(pl.Int64).alias("ts_sec")
        ])
        
        if len(df) > limit:
            df = df.tail(limit)
        
        return df
    
    def plot_orderbook_evolution(self, limit: int = 500):
        """Plot order book evolution over time."""
        print("Loading data...")
        df = self.load_latest_clean(limit=limit)
        
        # Convert to numpy for plotting
        timestamps = df['ts_sec'].to_numpy()
        t_plot = timestamps - timestamps[0]  # Relative time
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Order Book Evolution', fontsize=16, fontweight='bold')
        
        # Plot 1: Best bid/ask prices
        ax = axes[0]
        ax.plot(t_plot, df['bid_px_01'].to_numpy(), label='Best Bid', color='green', linewidth=1)
        ax.plot(t_plot, df['ask_px_01'].to_numpy(), label='Best Ask', color='red', linewidth=1)
        
        if 'last_trade_px' in df.columns:
            trades = df.filter(pl.col('last_trade_px').is_not_null())
            if len(trades) > 0:
                t_trades = (trades['ts_sec'].to_numpy() - timestamps[0])
                ax.scatter(t_trades, trades['last_trade_px'].to_numpy(), 
                          marker='o', s=10, alpha=0.5, color='blue', label='Trades')
        
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Best Bid/Ask Prices and Trades')
        
        # Plot 2: Spread
        ax = axes[1]
        spread = (df['ask_px_01'] - df['bid_px_01']).to_numpy()
        ax.plot(t_plot, spread, color='purple', linewidth=1)
        ax.fill_between(t_plot, 0, spread, alpha=0.3, color='purple')
        ax.set_ylabel('Spread (USD)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Bid-Ask Spread')
        
        # Plot 3: Book depth (top 5 levels)
        ax = axes[2]
        bid_vol = sum(df[f'bid_sz_{i:02d}'].to_numpy() for i in range(1, 6))
        ask_vol = sum(df[f'ask_sz_{i:02d}'].to_numpy() for i in range(1, 6))
        
        ax.plot(t_plot, bid_vol, label='Bid Volume (5 levels)', color='green', linewidth=1)
        ax.plot(t_plot, ask_vol, label='Ask Volume (5 levels)', color='red', linewidth=1)
        ax.set_ylabel('Volume')
        ax.set_xlabel('Time (seconds)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Liquidity at Top 5 Levels')
        
        plt.tight_layout()
        
        output_path = self.base_path / "orderbook_evolution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.show()
    
    def plot_features(self, limit: int = 1000):
        """Plot computed features."""
        print("Loading feature data...")
        df = self.load_latest_features(limit=limit)
        
        timestamps = df['ts_sec'].to_numpy()
        t_plot = timestamps - timestamps[0]
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('MIDAS Features Analysis', fontsize=16, fontweight='bold')
        
        # 1. Midprice
        ax = fig.add_subplot(gs[0, :])
        if 'midprice' in df.columns:
            ax.plot(t_plot, df['midprice'].to_numpy(), color='black', linewidth=1)
            ax.set_ylabel('Midprice (USD)')
            ax.set_title('Midprice Evolution')
            ax.grid(True, alpha=0.3)
        
        # 2. OFI (Order Flow Imbalance)
        ax = fig.add_subplot(gs[1, 0])
        if 'ofi' in df.columns:
            ofi = df['ofi'].to_numpy()
            colors = ['green' if x > 0 else 'red' for x in ofi]
            ax.bar(t_plot, ofi, color=colors, alpha=0.6, width=(t_plot[-1]-t_plot[0])/len(t_plot))
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('OFI')
            ax.set_title('Order Flow Imbalance (OFI)')
            ax.grid(True, alpha=0.3)
        
        # 3. Imbalance
        ax = fig.add_subplot(gs[1, 1])
        if 'imbalance' in df.columns:
            imb = df['imbalance'].to_numpy()
            ax.plot(t_plot, imb, color='purple', linewidth=1)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax.fill_between(t_plot, 0, imb, where=(imb>0), alpha=0.3, color='green', label='Bid heavy')
            ax.fill_between(t_plot, 0, imb, where=(imb<0), alpha=0.3, color='red', label='Ask heavy')
            ax.set_ylabel('Imbalance')
            ax.set_title('Book Imbalance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Spread
        ax = fig.add_subplot(gs[2, 0])
        if 'spread_bps' in df.columns:
            spread_bps = df['spread_bps'].to_numpy()
            ax.plot(t_plot, spread_bps, color='orange', linewidth=1)
            ax.set_ylabel('Spread (bps)')
            ax.set_title('Spread in Basis Points')
            ax.grid(True, alpha=0.3)
        
        # 5. Trade flow
        ax = fig.add_subplot(gs[2, 1])
        if 'signed_volume' in df.columns:
            signed_vol = df['signed_volume'].to_numpy()
            colors = ['green' if x > 0 else 'red' for x in signed_vol]
            ax.bar(t_plot, signed_vol, color=colors, alpha=0.6, width=(t_plot[-1]-t_plot[0])/len(t_plot))
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Signed Volume')
            ax.set_title('Signed Trade Volume (Buy - Sell)')
            ax.grid(True, alpha=0.3)
        
        # 6. Liquidity
        ax = fig.add_subplot(gs[3, 0])
        if 'liquidity_1' in df.columns and 'liquidity_5' in df.columns:
            ax.plot(t_plot, df['liquidity_1'].to_numpy(), label='L1', linewidth=1)
            ax.plot(t_plot, df['liquidity_5'].to_numpy(), label='L5', linewidth=1)
            if 'liquidity_10' in df.columns:
                ax.plot(t_plot, df['liquidity_10'].to_numpy(), label='L10', linewidth=1)
            ax.set_ylabel('Total Liquidity')
            ax.set_xlabel('Time (seconds)')
            ax.set_title('Liquidity at Different Depths')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 7. Volatility (if available)
        ax = fig.add_subplot(gs[3, 1])
        if 'returns' in df.columns:
            returns = df['returns'].to_numpy()
            # Filter out NaNs
            valid_idx = ~np.isnan(returns)
            ax.hist(returns[valid_idx], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Returns')
            ax.set_ylabel('Frequency')
            ax.set_title('Returns Distribution')
            ax.grid(True, alpha=0.3)
        
        output_path = self.base_path / "features_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.show()
    
    def plot_orderbook_snapshot(self, num_levels: int = 10):
        """Plot a single order book snapshot (depth chart)."""
        print("Loading latest order book...")
        df = self.load_latest_clean(limit=1)
        
        if len(df) == 0:
            print("No data available")
            return
        
        row = df.row(0, named=True)
        
        # Extract bid and ask levels
        bids = []
        asks = []
        
        for i in range(1, num_levels + 1):
            bid_px_key = f'bid_px_{i:02d}'
            bid_sz_key = f'bid_sz_{i:02d}'
            ask_px_key = f'ask_px_{i:02d}'
            ask_sz_key = f'ask_sz_{i:02d}'
            
            if row.get(bid_px_key) is not None and row.get(bid_sz_key) is not None:
                bids.append((row[bid_px_key], row[bid_sz_key]))
            
            if row.get(ask_px_key) is not None and row.get(ask_sz_key) is not None:
                asks.append((row[ask_px_key], row[ask_sz_key]))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bids (green)
        bid_prices = [b[0] for b in bids]
        bid_cumvol = np.cumsum([b[1] for b in bids])
        
        # Plot asks (red)
        ask_prices = [a[0] for a in asks]
        ask_cumvol = np.cumsum([a[1] for a in asks])
        
        ax.fill_betweenx(bid_prices, 0, bid_cumvol, step='post', alpha=0.5, color='green', label='Bids')
        ax.fill_betweenx(ask_prices, 0, ask_cumvol, step='pre', alpha=0.5, color='red', label='Asks')
        
        # Mark midprice
        if bids and asks:
            midprice = (bids[0][0] + asks[0][0]) / 2
            ax.axhline(y=midprice, color='black', linestyle='--', linewidth=1, label=f'Mid: ${midprice:.2f}')
        
        ax.set_xlabel('Cumulative Volume')
        ax.set_ylabel('Price (USD)')
        ax.set_title('Order Book Depth (Current Snapshot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.base_path / "orderbook_snapshot.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.show()
    
    def plot_trades_analysis(self, limit: int = 1000):
        """Analyze trade patterns."""
        print("Loading clean data for trades...")
        df = self.load_latest_clean(limit=limit)
        
        # Filter rows with trades
        df_trades = df.filter(pl.col('last_trade_px').is_not_null())
        
        if len(df_trades) == 0:
            print("No trade data found")
            return
        
        print(f"Found {len(df_trades)} trades")
        
        timestamps = df_trades['ts_sec'].to_numpy()
        t_plot = timestamps - timestamps[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trade prices
        ax = axes[0]
        prices = df_trades['last_trade_px'].to_numpy()
        ax.scatter(t_plot, prices, s=20, alpha=0.6, color='blue')
        ax.plot(t_plot, prices, alpha=0.3, color='blue', linewidth=0.5)
        ax.set_ylabel('Trade Price (USD)')
        ax.set_title('Trade Prices Over Time')
        ax.grid(True, alpha=0.3)
        
        # 2. Trade sizes
        ax = axes[1]
        sizes = df_trades['last_trade_qty'].to_numpy()
        ax.scatter(t_plot, sizes, s=20, alpha=0.6, color='purple')
        ax.set_ylabel('Trade Size')
        ax.set_title('Trade Sizes')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visibility
        
        # 3. Cumulative volume by side
        ax = axes[2]
        if 'taker_buy_vol' in df_trades.columns and 'taker_sell_vol' in df_trades.columns:
            buy_vol = df_trades['taker_buy_vol'].to_numpy()
            sell_vol = df_trades['taker_sell_vol'].to_numpy()
            
            ax.fill_between(t_plot, 0, buy_vol, alpha=0.5, color='green', label='Taker Buy')
            ax.fill_between(t_plot, 0, -sell_vol, alpha=0.5, color='red', label='Taker Sell')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Volume')
            ax.set_xlabel('Time (seconds)')
            ax.set_title('Taker Buy vs Sell Volume')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.base_path / "trades_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")
        plt.show()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize MIDAS data")
    parser.add_argument("--path", type=str, help="Base data path", default=None)
    parser.add_argument("--orderbook", action="store_true", help="Plot order book evolution")
    parser.add_argument("--features", action="store_true", help="Plot features")
    parser.add_argument("--snapshot", action="store_true", help="Plot current order book snapshot")
    parser.add_argument("--trades", action="store_true", help="Plot trade analysis")
    parser.add_argument("--limit", type=int, default=1000, help="Number of data points")
    
    args = parser.parse_args()
    
    base_path = Path(args.path) if args.path else None
    viz = DataVisualizer(base_path)
    
    if args.orderbook:
        viz.plot_orderbook_evolution(limit=args.limit)
    elif args.features:
        viz.plot_features(limit=args.limit)
    elif args.snapshot:
        viz.plot_orderbook_snapshot()
    elif args.trades:
        viz.plot_trades_analysis(limit=args.limit)
    else:
        # Default: show all
        print("Generating all visualizations...\n")
        viz.plot_orderbook_snapshot()
        viz.plot_orderbook_evolution(limit=min(500, args.limit))
        viz.plot_trades_analysis(limit=args.limit)
        viz.plot_features(limit=args.limit)


if __name__ == "__main__":
    main()
