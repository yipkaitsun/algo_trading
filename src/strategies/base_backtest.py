from abc import ABC, abstractmethod
import pandas as pd
from typing import  Any
from pathlib import Path
import logging
from src.strategies.base_strategy import BaseStrategy
from src.data.fetcher import DataFetcher
import yaml

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
    
    def load_config(self):
        """Load configuration from settings.yaml"""
        config_path = self._get_project_root() / 'config' / 'settings.yaml'
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @abstractmethod
    def initialize_strategy(self):
        """Initialize the strategy with configuration."""
        pass
    
    def load_data(self, symbol) -> pd.DataFrame:
        return DataFetcher.fetch_data_ssh(symbol=symbol)
    
    def save_performance_metrics(self, metrics: Any) -> None:
        """Print performance metrics to console."""
        print(f"\nBacktest Results for {self.strategy_name}:")
        print("-------------------")
        print("Performance Metrics:")
        print(f"Total Profit/Loss: {metrics['total_pl']:.2f}")
        print(f"Cumulative Return: {metrics['cumulative_return']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}" if metrics['sharpe_ratio'] else "Sharpe Ratio: N/A")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}" if metrics['sortino_ratio'] else "Sortino Ratio: N/A")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f}")
        print("Trades:", metrics['trades'])
        print("Equity Curve:", metrics['equity_curve'])
    
    def log_strategy_parameters(self) -> None:
        """Log strategy parameters before saving metrics."""
        logger.info(f"Initializing backtest with parameters:")
        if hasattr(self, 'strategy'):
            for param_name, param_value in vars(self.strategy).items():
                if not param_name.startswith('_'):  # Skip private attributes
                    logger.info(f"{param_name}: {param_value}")

    def run_backtest(self, symbol) -> None:
        """Run the backtest with common flow for all strategies."""
        self.initialize_strategy()
        df = self.load_data(symbol)
        metrics = self.strategy.run(df,True)
        self.log_strategy_parameters()
        self.save_performance_metrics(metrics) 