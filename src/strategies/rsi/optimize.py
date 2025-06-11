import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from ..base_optimize import BaseOptimizer
from src.strategies.rsi.rsi_strategy import RSIStrategy

class RsiOptimizer(BaseOptimizer):
    def __init__(self):
        super().__init__(RSIStrategy)
    
    def optimize_parameters(self, df, param_grid):
        results = []
        
        for window in param_grid['windows']:
            # Create strategy instance with current window size
            strategy = RSIStrategy(window=window)
            
            # Calculate indicators once for this window size
            df_test = df.copy()
            df_test = strategy.calculate_indicators(df_test)
            
            for overbought, oversold in zip(param_grid['overbought_levels'], param_grid['oversold_levels']):
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

    def setup_parameters(self):
        """Define the parameter ranges to test."""
        windows = np.arange(10, 100, 10)
        overbought_levels = np.arange(10, 100, 5)  # 60, 65, ..., 90
        oversold_levels = 100 - overbought_levels  # 40, 35, ..., 10

        print(f"\nTesting {len(windows)} window sizes from {windows[0]} to {windows[-1]}")
        print(f"Testing {len(overbought_levels)} overbought/oversold pairs:")
        print(f"Total combinations to test: {len(windows) * len(overbought_levels)}")
        return {'windows':windows, 'overbought_levels':overbought_levels, 'oversold_levels':oversold_levels}
    
    def plot_results(self, results, metric1='sharpe_ratio', metric2='annual_return'):
        """
        Plot optimization results as heatmaps for RSI strategy.
        
        Args:
            results (pd.DataFrame): Optimization results
            metric1 (str): First metric to plot
            metric2 (str): Second metric to plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
        
        # Plot first metric heatmap
        pivot1 = results.pivot(index='window', columns='overbought', values=metric1)
        im1 = ax1.imshow(pivot1, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Overbought Level')
        ax1.set_ylabel('Window Size')
        ax1.set_title(f'{metric1.replace("_", " ").title()} Heatmap')
        plt.colorbar(im1, ax=ax1, label=metric1.replace("_", " ").title())
        
        # Add text annotations
        for i in range(len(pivot1.index)):
            for j in range(len(pivot1.columns)):
                value = pivot1.iloc[i, j]
                if not np.isnan(value):
                    text = ax1.text(j, i, f'{value:.2f}',
                                  ha='center', va='center',
                                  color='white' if value < pivot1.mean().mean() else 'black')
        
        # Plot second metric heatmap
        pivot2 = results.pivot(index='window', columns='overbought', values=metric2)
        im2 = ax2.imshow(pivot2, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Overbought Level')
        ax2.set_ylabel('Window Size')
        ax2.set_title(f'{metric2.replace("_", " ").title()} Heatmap')
        plt.colorbar(im2, ax=ax2, label=metric2.replace("_", " ").title())
        
        # Set y-axis ticks
        ax1.set_yticks(range(len(pivot1.index)))
        ax1.set_yticklabels(pivot1.index)
        ax2.set_yticks(range(len(pivot2.index)))
        ax2.set_yticklabels(pivot2.index)
        
        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'{self.strategy_class.__name__.lower()}_parameter_optimization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'BTC_HOURLY.csv')
    optimizer = RsiOptimizer()
    optimizer.main(data_path)


if __name__ == "__main__":
    main() 