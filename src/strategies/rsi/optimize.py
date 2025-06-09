"""
RSI Strategy Optimization Script
Performs grid search to find optimal RSI parameters.
"""

import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.strategies.rsi.rsi_strategy import RSIStrategy

def load_config():
    """Load configuration from settings.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'settings.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def optimize_parameters(df, windows, overbought_levels, oversold_levels):
    """
    Optimize RSI parameters by testing different combinations.
    
    Args:
        df (pd.DataFrame): Price data
        windows (list): List of window sizes to test
        overbought_levels (list): List of overbought threshold values
        oversold_levels (list): List of oversold threshold values
    
    Returns:
        pd.DataFrame: Results for each parameter combination
    """
    results = []
    
    for window in windows:
        # Create strategy instance with current window size
        strategy = RSIStrategy(window=window)
        
        # Calculate indicators once for this window size
        df_test = df.copy()
        df_test = strategy.calculate_indicators(df_test)
        
        for overbought, oversold in zip(overbought_levels, oversold_levels):
                # Generate signals with current thresholds
                df_signals = strategy.generate_signals(df_test, overbought=overbought, oversold=oversold) # type: ignore
                
                # Calculate performance metrics
                metrics = strategy.calculate_performance_metrics(df_signals)# type: ignore
                
                # Store results
                results.append({
                    'window': window, 
                    'overbought': overbought,
                    'oversold': oversold,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'annual_return': metrics['annual_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'calmar_ratio': metrics['calmar_ratio']
                })
        
    return pd.DataFrame(results)

def plot_results(results):
    """Plot optimization results"""
    # Create a figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('RSI Strategy Parameter Optimization Results', fontsize=16)
    
    # Plot Sharpe Ratio heatmap
    pivot_sharpe = results.pivot_table(
        index='window', 
        columns='overbought', 
        values='sharpe_ratio',
        aggfunc='mean'
    )
    im1 = axes[0,0].imshow(pivot_sharpe, cmap='viridis', aspect='auto')
    axes[0,0].set_xlabel('Overbought Level')
    axes[0,0].set_ylabel('Window Size')
    axes[0,0].set_title('Sharpe Ratio Heatmap')
    plt.colorbar(im1, ax=axes[0,0], label='Sharpe Ratio')
    
    # Plot Annual Return heatmap
    pivot_return = results.pivot_table(
        index='window', 
        columns='overbought', 
        values='annual_return',
        aggfunc='mean'
    )
    im2 = axes[0,1].imshow(pivot_return, cmap='viridis', aspect='auto')
    axes[0,1].set_xlabel('Overbought Level')
    axes[0,1].set_ylabel('Window Size')
    axes[0,1].set_title('Annual Return Heatmap')
    plt.colorbar(im2, ax=axes[0,1], label='Annual Return')
    
    # Plot Max Drawdown heatmap
    pivot_dd = results.pivot_table(
        index='window', 
        columns='overbought', 
        values='max_drawdown',
        aggfunc='mean'
    )
    im3 = axes[1,0].imshow(pivot_dd, cmap='viridis', aspect='auto')
    axes[1,0].set_xlabel('Overbought Level')
    axes[1,0].set_ylabel('Window Size')
    axes[1,0].set_title('Max Drawdown Heatmap')
    plt.colorbar(im3, ax=axes[1,0], label='Max Drawdown')
    
    # Plot Calmar Ratio heatmap
    pivot_calmar = results.pivot_table(
        index='window', 
        columns='overbought', 
        values='calmar_ratio',
        aggfunc='mean'
    )
    im4 = axes[1,1].imshow(pivot_calmar, cmap='viridis', aspect='auto')
    axes[1,1].set_xlabel('Overbought Level')
    axes[1,1].set_ylabel('Window Size')
    axes[1,1].set_title('Calmar Ratio Heatmap')
    plt.colorbar(im4, ax=axes[1,1], label='Calmar Ratio')
    
    # Set y-axis ticks to show actual window sizes
    for ax in axes.flat:
        ax.set_yticks(range(len(pivot_sharpe.index)))
        ax.set_yticklabels(pivot_sharpe.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot in the results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'rsi_parameter_optimization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'BTC_HOURLY.csv')
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        'Date': 'datetime',
        'Close': 'close'
    })
    
    # Define parameters to test
    windows = np.arange(1, 50, 1)  # Test window sizes from 5 to 30
    overbought_levels = np.arange(10, 100, 5)  # 60, 65, ..., 90
    oversold_levels = 100 - overbought_levels  # 40, 35, ..., 10
    
    print(f"\nTesting {len(windows)} window sizes from {windows[0]} to {windows[-1]}")
    print(f"Testing {len(overbought_levels)} overbought/oversold pairs:")
    print(f"Total combinations to test: {len(windows) * len(overbought_levels)}")
    
    # Run optimization
    results = optimize_parameters(df, windows, overbought_levels, oversold_levels)
    
    # Find best combination based on Sharpe ratio
    best_params = results.loc[results['sharpe_ratio'].idxmax()]
    
    # Print results
    print("\nOptimization Results:")
    print("-------------------")
    print(f"Best Window Size: {best_params['window']}")
    print(f"Best Overbought Level: {best_params['overbought']}")
    print(f"Best Oversold Level: {best_params['oversold']}")
    print(f"Best Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Annual Return: {best_params['annual_return']:.2f}")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2f}")
    print(f"Calmar Ratio: {best_params['calmar_ratio']:.2f}")
    
    # Create final strategy with best parameters
    final_strategy = RSIStrategy(
        window=int(best_params['window']), # type: ignore
        overbought=int(best_params['overbought']), # type: ignore
        oversold=int(best_params['oversold']) # type: ignore
    )
    print("\nFinal strategy created with optimized parameters:")
    print(f"window={final_strategy.window}")
    print(f"overbought={final_strategy.overbought}")
    print(f"oversold={final_strategy.oversold}")
    
    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'rsi_parameter_optimization_results.csv')
    results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to '{results_file}'")
    
    # Plot results
    plot_results(results)
    print(f"Plot saved as '{os.path.join(results_dir, 'rsi_parameter_optimization.png')}'")

if __name__ == "__main__":
    main() 