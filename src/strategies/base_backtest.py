from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from src.strategies.base_strategy import BaseStrategy
from src.utils.utils import CalculationUtils

logger = logging.getLogger(__name__)
class BaseBacktest(ABC):
    """Abstract base class for backtesting strategies."""
    
    def __init__(self, strategy_name: str):
        self.strategy : BaseStrategy
        self.strategy_name = strategy_name
        self.results_dir = self._get_results_dir()
        self.results_dir.mkdir(exist_ok=True)
    
    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent
    
    def _get_results_dir(self) -> Path:
        """Get the results directory for this strategy."""
        return self._get_project_root() / 'backtest' / 'results' / self.strategy_name
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load strategy configuration."""
        pass
    
    @abstractmethod
    def initialize_strategy(self):
        """Initialize the strategy with configuration."""
        pass
    
    @abstractmethod
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load and preprocess the data."""
        pass
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.calculate_indicators(df)
        
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ZScore-specific signals."""
        return self.strategy.generate_signals(df)
    
    def save_results(self, df: pd.DataFrame) -> str:
        """Save results to CSV file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.results_dir / f'{self.strategy_name}_results_{timestamp}.csv'
        
        df.index.name = 'timestamp'
        df.to_csv(output_file)
        logger.info(f"Results saved to: {output_file}")
        return str(output_file)
    
    def plot_cumulative_return(self, df: pd.DataFrame) -> None:
        """Plot and save cumulative returns of the strategy."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.results_dir / f'cumulative_return_plot_{timestamp}.png'
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['cumulative_return'], label='Cumulative Return')
        plt.title(f'{self.strategy_name} Strategy Cumulative Return')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Plot saved to: {plot_file}")
    
    def save_performance_metrics(self, metrics: Dict[str, float], df: pd.DataFrame) -> None:
        """Print performance metrics to console."""
        print(f"\nBacktest Results for {self.strategy_name}:")
        print("-------------------")
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in metric.lower():
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")
        
        logger.info("Performance metrics printed to console")
    
    def log_strategy_parameters(self) -> None:
        """Log strategy parameters before saving metrics."""
        logger.info(f"Initializing backtest with parameters:")
        if hasattr(self, 'strategy'):
            for param_name, param_value in vars(self.strategy).items():
                if not param_name.startswith('_'):  # Skip private attributes
                    logger.info(f"{param_name}: {param_value}")

    def run_backtest(self, symbol: str = 'BTC_HOURLY') -> None:
        """Run the backtest with common flow for all strategies."""
        # Initialize strategy
        self.initialize_strategy()
        
        # Load and process data
        df = self.load_data(symbol)
        
        # Calculate indicators and generate signals
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        # Calculate returns
        df = CalculationUtils.calculate_returns(df)
        
        # Save results and plot
        self.save_results(df)
        self.plot_cumulative_return(df)
        
        # Calculate and save performance metrics
        metrics = self.strategy.calculate_performance_metrics(df)
        
        # Log strategy parameters and save metrics
        self.log_strategy_parameters()
        self.save_performance_metrics(metrics, df) 