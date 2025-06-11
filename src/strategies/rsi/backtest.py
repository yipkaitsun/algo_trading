import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path when running directly
if __name__ == "__main__":
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(src_path)

from strategies.rsi.rsi_strategy import RSIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from settings.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'settings.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_rolling_metrics(returns, window=252):
    """
    Calculate rolling performance metrics for a series of returns.
    
    Args:
        returns (pd.Series): Series of returns
        window (int): Rolling window size (default: 252 for daily data)
    
    Returns:
        tuple: (sharpe_ratio, annual_return, drawdown, calmar_ratio)
    """
    # Calculate rolling mean and std for Sharpe ratio
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    sharpe_ratio = np.sqrt(252) * (rolling_mean / rolling_std)
    
    # Calculate annual return
    annual_return = returns.rolling(window=window).mean() * 252
    
    # Calculate drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.rolling(window=window).max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    # Calculate Calmar ratio
    calmar_ratio = annual_return / abs(drawdown.rolling(window=window).min())
    
    return sharpe_ratio, annual_return, drawdown, calmar_ratio

def plot_performance_metrics(df, metrics, strategy):
    """
    Plot all performance metrics over time in subplots.
    
    Args:
        df (pd.DataFrame): Original price data with datetime
        metrics (tuple): Tuple of (sharpe_ratio, annual_return, drawdown, calmar_ratio)
        strategy: Strategy instance containing parameters
    """
    sharpe_ratio, annual_return, drawdown, calmar_ratio = metrics
    
    # Create figure with subplots and extra space for parameter box
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.1, 1, 1])
    
    # Create subplots
    axes = [
        fig.add_subplot(gs[1, 0]),  # Sharpe Ratio
        fig.add_subplot(gs[1, 1]),  # Annual Return
        fig.add_subplot(gs[2, 0]),  # Drawdown
        fig.add_subplot(gs[2, 1])   # Calmar Ratio
    ]
    
    # Convert datetime strings to datetime objects if needed
    if isinstance(df['datetime'].iloc[0], str):
        dates = pd.to_datetime(df['datetime'])
    else:
        dates = df['datetime']
    
    # Plot Sharpe Ratio
    axes[0].plot(dates, sharpe_ratio, label='Rolling Sharpe Ratio', color='blue')
    # Add horizontal lines for Sharpe ratio
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (1.0)')
    axes[0].axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Excellent (2.0)')
    axes[0].set_title('Sharpe Ratio')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot Annual Return
    axes[1].plot(dates, annual_return, label='Annual Return', color='green')
    # Add horizontal lines for Annual Return
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    axes[1].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='10%')
    axes[1].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='20%')
    axes[1].set_title('Annual Return')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].grid(True)
    axes[1].legend()
    
    # Plot Drawdown
    axes[2].fill_between(dates, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    # Add horizontal lines for Drawdown
    axes[2].axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5, label='-10%')
    axes[2].axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label='-20%')
    axes[2].set_title('Drawdown')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Drawdown')
    axes[2].grid(True)
    axes[2].legend()
    
    # Plot Calmar Ratio
    axes[3].plot(dates, calmar_ratio, label='Calmar Ratio', color='purple')
    # Add horizontal lines for Calmar Ratio
    axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero')
    axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Good (1.0)')
    axes[3].axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Excellent (2.0)')
    axes[3].set_title('Calmar Ratio')
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel('Calmar Ratio')
    axes[3].grid(True)
    axes[3].legend()
    
    # Add strategy parameters text box in the top row
    param_text = (
        f"Strategy Parameters:\n"
        f"Window Size: {strategy.window}\n"
        f"Overbought Level: {strategy.overbought}\n"
        f"Oversold Level: {strategy.oversold}\n"
        f"Position Size: {strategy.position_size}"
    )
    # Create a new subplot for the parameter box
    param_ax = fig.add_subplot(gs[0, :])
    param_ax.axis('off')
    param_ax.text(0.5, 0.5, param_text,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 ha='center', va='center', fontsize=12)
    
    # Rotate x-axis labels for better readability
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'rsi_performance_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load configuration
    config = load_config()
    strategy_config = config['strategies']['rsi']
    
    # Create strategy instance with parameters from config
    strategy = RSIStrategy(
        window=strategy_config['window'],
        overbought=strategy_config['overbought'],
        oversold=strategy_config['oversold'],
        position_size=strategy_config['position_size']
    )
    
    logger.info(f"Initializing backtest with parameters:")
    logger.info(f"Window size: {strategy.window}")
    logger.info(f"Overbought level: {strategy.overbought}")
    logger.info(f"Oversold level: {strategy.oversold}")
    logger.info(f"Position size: {strategy.position_size}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'BTC_HOURLY.csv')
    df = pd.read_csv(data_path)
    
    # Rename columns to match strategy requirements
    df = df.rename(columns={
        'Date': 'datetime',
        'Close': 'close'
    })
    
    # Calculate indicators
    df = strategy.calculate_indicators(df)
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate rolling metrics
    metrics = calculate_rolling_metrics(df['returns'], window=252)  # Using 252 hours as window
    
    # Plot all performance metrics
    plot_performance_metrics(df, metrics, strategy)
    logger.info("Performance metrics plot saved to backtest/results/rsi_performance_metrics.png")
    
    # Calculate final performance metrics
    metrics = strategy.calculate_performance_metrics(df)
    
    # Print detailed results
    print("\nBacktest Results:")
    print("-------------------")
    print(f"Strategy Parameters:")
    print(f"  Window Size: {strategy.window}")
    print(f"  Overbought Level: {strategy.overbought}")
    print(f"  Oversold Level: {strategy.oversold}")
    print(f"  Position Size: {strategy.position_size}")
    
    print("\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    # Save detailed results to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'rsi_results.csv')
    df.to_csv(results_file, index=False)
    logger.info(f"Detailed results saved to '{results_file}'")

if __name__ == "__main__":
    main() 