import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BaseOptimizer(ABC):
    def __init__(self, strategy_class):
        """
        Initialize the base optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
        """
        self.strategy_class = strategy_class
    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    def load_config(self):
        """Load configuration from settings.yaml"""
        config_path = self._get_project_root() / 'config' / 'settings.yaml'
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @abstractmethod
    def optimize_parameters(self, df, param_grid) -> pd.DataFrame:
        """
        Optimize strategy parameters.
        
        Args:
            df (pd.DataFrame): Price data
            param_grid (dict): Dictionary of parameter names and their values to test
            
        Returns:
            pd.DataFrame: Results of optimization
        """
        pass
    
    @abstractmethod
    def plot_results(self, results, metric1='sharpe_ratio', metric2='annual_return'):
        """
        Plot optimization results.
        
        Args:
            results (pd.DataFrame): Optimization results
            metric1 (str): First metric to plot
            metric2 (str): Second metric to plot
        """
        pass
    
    def print_best_parameters(self, results, metric_name, metric_key, is_min=False):
        if is_min:
            best_params = results.loc[results[metric_key].idxmin()]
        else:
            best_params = results.loc[results[metric_key].idxmax()]
        
        print(f"\nBest Parameters by {metric_name}:")
        for col in results.columns:
            if col not in ['total_pl', 'cumulative_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'trades', 'equity_curve']:
                print(f"{col.replace('_', ' ').title()}: {best_params[col]}")
        
        print(f"Total P/L: {best_params['total_pl']:.2f}")
        print(f"Cumulative Return: {best_params['cumulative_return']:.4f}")
        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.4f}" if best_params['sharpe_ratio'] else "Sharpe Ratio: N/A")
        print(f"Sortino Ratio: {best_params['sortino_ratio']:.4f}" if best_params['sortino_ratio'] else "Sortino Ratio: N/A")
        print(f"Maximum Drawdown: {best_params['max_drawdown']:.4f}")
        print(f"Number of Trades: {best_params['trades']}")
        print(f"Final Equity: {best_params['equity_curve']:.2f}")
        
        return best_params
    
    
    def load_data(self, symbol) -> pd.DataFrame:
        data_path =  Path(__file__).parent .parent .parent / 'data' / f'{symbol}.csv'
        print(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Validate and rename columns
        required_columns = ['datetime', 'close']
        column_mapping = {'Date': 'datetime', 'Close': 'close'}
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df.rename(columns=column_mapping)
        
        # Convert datetime if needed
        if isinstance(df['datetime'].iloc[0], str):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df

    def main(self, data_filename):
        """
        Main optimization workflow that can be used by any strategy optimizer.
        
        Args:
            data_filename (str): Name of the data file to use for optimization
        """
        # Load data
        df = self.load_data(data_filename)
        
        # Setup parameters - to be implemented by child classes
        param_grid = self.setup_parameters()
        
        # Run optimization
        results = self.optimize_parameters(df, param_grid)
        self.plot_results(results)
        # Print results
        self.print_best_parameters(results, "Sharpe Ratio", "sharpe_ratio")
        self.print_best_parameters(results, "Total P/L", "total_pl")
        self.print_best_parameters(results, "Maximum Drawdown", "max_drawdown", is_min=True)
        self.print_best_parameters(results, "Cumulative Return", "cumulative_return")
        self.print_best_parameters(results, "Sortino Ratio", "sortino_ratio")
        self.print_best_parameters(results, "Number of Trades", "trades")
        self.print_best_parameters(results, "Final Equity", "equity_curve")
        

        # Save results
    @abstractmethod
    def setup_parameters(self) -> dict[str, Any]:
        """
        Setup the parameter grid for optimization.
        To be implemented by child classes.
        
        Returns:
            dict: Dictionary of parameter names and their values to test
        """
        pass 