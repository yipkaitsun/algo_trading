import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from ..base_optimize import BaseOptimizer
from .zscore_strategy import ZscoreStrategy


class ZScoreOptimizer(BaseOptimizer):
    def __init__(self):
        super().__init__(ZscoreStrategy)
    
    def optimize_parameters(self, df, param_grid):
        results = []
        
        for window in param_grid['windows']:
            for threshold in param_grid['thresholds']:
                strategy = ZscoreStrategy(window=window, threshold=threshold, position_size=0.1)

                df_test = df.copy()
                df_test = strategy.calculate_indicators(df_test)
                df_test = strategy.generate_signals(df_test, threshold=threshold)
                
                # Calculate performance metrics
                metrics = strategy.calculate_performance_metrics(df_test)
                
                results.append({
                    'window': window,
                    'threshold': threshold,
                    'profit_factor': metrics['profit_factor'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'annual_return': metrics['annual_return'],
                    'max_drawdown': metrics['max_drawdown'],
                    'calmar_ratio': metrics['calmar_ratio']
                })
        
        return pd.DataFrame(results)

    def setup_parameters(self):
        """
        Setup the parameter grid for ZScore optimization.
        
        Returns:
            dict: Dictionary of parameter names and their values to test
        """
        windows = np.arange(10, 100, 10)  # Test different window sizes
        thresholds = np.arange(0, 2.5, 0.25)
        
        print(f"\nTesting {len(windows)} window sizes from {windows[0]} to {windows[-1]}")
        print(f"Testing {len(thresholds)} thresholds from {thresholds[0]} to {thresholds[-1]}")
        print(f"Total combinations to test: {len(windows) * len(thresholds)}")
        
        return {'windows': windows, 'thresholds': thresholds}

    def plot_results(self, results, metric1='sharpe_ratio', metric2='annual_return'):
        """
        Plot optimization results as heatmaps.
        
        Args:
            results (pd.DataFrame): Optimization results
            metric1 (str): First metric to plot
            metric2 (str): Second metric to plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
        
        # Plot first metric heatmap
        pivot1 = results.pivot(index='window', columns='threshold', values=metric1)
        im1 = ax1.imshow(pivot1, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Threshold')
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
        pivot2 = results.pivot(index='window', columns='threshold', values=metric2)
        im2 = ax2.imshow(pivot2, cmap='viridis', aspect='auto')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Window Size')
        ax2.set_title(f'{metric2.replace("_", " ").title()} Heatmap')
        plt.colorbar(im2, ax=ax2, label=metric2.replace("_", " ").title())
        
        # Set x and y axis ticks with actual values
        ax1.set_xticks(range(len(pivot1.columns)))
        ax1.set_xticklabels(pivot1.columns)
        ax1.set_yticks(range(len(pivot1.index)))
        ax1.set_yticklabels(pivot1.index)
        
        ax2.set_xticks(range(len(pivot2.columns)))
        ax2.set_xticklabels(pivot2.columns)
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
    # Initialize optimizer
    optimizer = ZScoreOptimizer()
    # Run optimization
    optimizer.main('btcusdt_3year_hourly_binance')

if __name__ == "__main__":
    main() 