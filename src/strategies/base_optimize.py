import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from pathlib import Path
class BaseOptimizer(ABC):
    def __init__(self, strategy_class):
        """
        Initialize the base optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
        """
        self.strategy_class = strategy_class
        
    def load_config(self):
        """Load configuration from settings.yaml"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'settings.yaml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @abstractmethod
    def optimize_parameters(self, df, param_grid):
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
            if col not in ['sharpe_ratio', 'annual_return', 'max_drawdown', 'calmar_ratio']:
                print(f"{col.replace('_', ' ').title()}: {best_params[col]}")
        print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
        print(f"Annual Return: {best_params['annual_return']:.2f}")
        print(f"Max Drawdown: {best_params['max_drawdown']:.2f}")
        print(f"Calmar Ratio: {best_params['calmar_ratio']:.2f}")
        
        return best_params
    
    def save_results(self, results):
        """
        Save optimization results to CSV and generate plots.
        
        Args:
            results (pd.DataFrame): Optimization results
        """
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to CSV
        results_file = os.path.join(results_dir, f'{self.strategy_class.__name__.lower()}_parameter_optimization_results.csv')
        results.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to '{results_file}'")
        
        # Generate and save plot
        self.plot_results(results)
        print(f"Plot saved as '{os.path.join(results_dir, f'{self.strategy_class.__name__.lower()}_parameter_optimization.png')}'")
    
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
        
        # Print results
        self.print_best_parameters(results, "Sharpe Ratio", "sharpe_ratio")
        self.print_best_parameters(results, "Annual Return", "annual_return")
        self.print_best_parameters(results, "Minimum Drawdown", "max_drawdown", is_min=True)
        self.print_best_parameters(results, "Calmar Ratio", "calmar_ratio")
        
        # Save results
        self.save_results(results)

    @abstractmethod
    def setup_parameters(self):
        """
        Setup the parameter grid for optimization.
        To be implemented by child classes.
        
        Returns:
            dict: Dictionary of parameter names and their values to test
        """
        pass 