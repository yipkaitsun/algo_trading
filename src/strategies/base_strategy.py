"""
Abstract base class for trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class that defines the interface for all trading strategies.
    All strategy implementations must inherit from this class and implement
    all abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Strategy configuration parameters
        """
        self.config = config
        
        # Get trading parameters
        trading_config = config.get('trading', {})
        self.initial_capital = trading_config.get('initial_capital', 100000)
        self.commission = trading_config.get('commission', 0.001)
        self.slippage = trading_config.get('slippage', 0.0005)
        
        # Get risk management parameters
        risk_config = config.get('risk', {})
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.stop_loss = risk_config.get('stop_loss', 0.02)
        self.take_profit = risk_config.get('take_profit', 0.05)
        
        # Initialize tracking variables
        self.positions = {}  # Current positions
        self.trades = []     # Trade history
        self.current_capital = self.initial_capital
    
    # ============= Abstract Methods (Must be implemented by child classes) =============
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required columns.
        This method must be implemented by all strategy classes.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        This method must be implemented by all strategy classes.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Data with calculated indicators
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators.
        This method must be implemented by all strategy classes.
        
        Args:
            data (pd.DataFrame): Market data with indicators
            
        Returns:
            pd.DataFrame: Data with signal column
        """
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        This method must be implemented by all strategy classes.
        
        Args:
            data (pd.DataFrame): Market data with signals
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        pass
    
    # ============= Concrete Methods (Implemented in base class) =============
    
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the trading strategy on the provided market data.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Dict[str, Any]: Strategy results including trades and metrics
        """
        logger.info("Starting strategy execution")
        
        if not self.validate_data(data):
            logger.error("Invalid data format")
            return self._get_empty_results()
        
        try:
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Generate signals
            data_with_signals = self.generate_signals(data_with_indicators)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(data_with_signals)
            
            results = {
                'data': data_with_signals,
                'metrics': metrics
            }
            
            logger.info("Strategy execution completed")
            logger.info(f"Performance metrics: {metrics}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during strategy execution: {str(e)}")
            return self._get_empty_results()
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary."""
        return {
            'data': pd.DataFrame(),
            'metrics': {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        }
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data."""
        return data['close'].pct_change()
    
    def _calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod()
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        rolling_max = cumulative_returns.expanding().max()
        return cumulative_returns / rolling_max - 1
    
    def _calculate_max_drawdown(self, drawdown: pd.Series) -> float:
        """Calculate maximum drawdown."""
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns[returns != 0])
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else float('inf') 