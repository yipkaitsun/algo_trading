import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.strategies.zscore.zscore_strategy import ZScoreStrategy

def load_config():
    """Load configuration from settings.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'settings.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def optimize_threshold(df, windows, thresholds):
    """
    Optimize the z-score threshold and window size by testing different combinations.
    
    Args:
        df (pd.DataFrame): Price data
        windows (list): List of window sizes to test
        thresholds (list): List of threshold values to test
    
    Returns:
        pd.DataFrame: Results for each threshold and window combination
    """
    results = []
    
    for window in windows:
        for threshold in thresholds:
            # Create strategy instance with current window size
            strategy = ZScoreStrategy(window=window)
            
            # Calculate indicators and signals
            df_test = df.copy()
            df_test = strategy.calculate_indicators(df_test)
            df_test = strategy.generate_signals(df_test, threshold=threshold)
            
            # Calculate performance metrics
            metrics = strategy.calculate_performance_metrics(df_test)
            
            # Store results
            results.append({
                'window': window,
                'threshold': threshold,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'annual_return': metrics['annual_return'],
                'max_drawdown': metrics['max_drawdown'],
                'calmar_ratio': metrics['calmar_ratio']
            })
    
    return pd.DataFrame(results)

def plot_results(results):
    """Plot optimization results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    
    # Plot Sharpe Ratio heatmap
    pivot_sharpe = results.pivot(index='window', columns='threshold', values='sharpe_ratio')
    im1 = ax1.imshow(pivot_sharpe, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Window Size')
    ax1.set_title('Sharpe Ratio Heatmap')
    plt.colorbar(im1, ax=ax1, label='Sharpe Ratio')
    
    # Set y-axis ticks to show actual window sizes
    ax1.set_yticks(range(len(pivot_sharpe.index)))
    ax1.set_yticklabels(pivot_sharpe.index)
    
    # Plot Annual Return heatmap
    pivot_return = results.pivot(index='window', columns='threshold', values='annual_return')
    im2 = ax2.imshow(pivot_return, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Window Size')
    ax2.set_title('Annual Return Heatmap')
    plt.colorbar(im2, ax=ax2, label='Annual Return')
    
    # Set y-axis ticks to show actual window sizes
    ax2.set_yticks(range(len(pivot_return.index)))
    ax2.set_yticklabels(pivot_return.index)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot in the results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, 'parameter_optimization.png')
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
    windows = np.arange(10, 100, 10) 
    thresholds = np.arange(0, 2.5, 0.25) 
    
    print(f"\nTesting {len(windows)} window sizes from {windows[0]} to {windows[-1]}")
    print(f"Testing {len(thresholds)} thresholds from {thresholds[0]} to {thresholds[-1]}")
    print(f"Total combinations to test: {len(windows) * len(thresholds)}")
    
    # Run optimization
    results = optimize_threshold(df, windows, thresholds)
    
    # Find best combination based on Sharpe ratio
    best_params = results.loc[results['sharpe_ratio'].idxmax()]
    
    # Print results
    print("\nOptimization Results:")
    print("-------------------")
    print(f"Best Window Size: {best_params['window']}")
    print(f"Best Threshold: {best_params['threshold']:.2f}")
    print(f"Best Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Annual Return: {best_params['annual_return']:.2f}")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2f}")
    print(f"Calmar Ratio: {best_params['calmar_ratio']:.2f}")
    
    # Create final strategy with best parameters
    final_strategy = ZScoreStrategy(window=int(best_params['window']))
    print("\nFinal strategy created with optimized parameters:")
    print(f"window={final_strategy.window}")
    print(f"threshold={best_params['threshold']:.2f} (to be passed to generate_signals)")
    
    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'parameter_optimization_results.csv')
    results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to '{results_file}'")
    
    # Plot results
    plot_results(results)
    print(f"Plot saved as '{os.path.join(results_dir, 'parameter_optimization.png')}'")

if __name__ == "__main__":
    main() 