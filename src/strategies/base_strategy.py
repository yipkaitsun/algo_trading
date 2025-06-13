"""
Abstract base class for trading strategies.
"""
import os
import pandas as pd
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from src.utils.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class that defines the interface for all trading strategies.
    All strategy implementations must inherit from this class and implement
    all abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
     
        self.config = config

        # Set initial capital (default to 100000 if not specified)
        self.current_capital:Any
        
        # Get trading parameters
        trading_config = config.get('trading', {})
        self.commission = trading_config.get('commission', 0.001)
        self.slippage = trading_config.get('slippage', 0.0005)
        
        # Get risk management parameters
        risk_config = config.get('risk', {})
        self.max_position_size = risk_config.get('max_position_size', 0.1)
        self.stop_loss = risk_config.get('stop_loss', 0.02)
        self.take_profit = risk_config.get('take_profit', 0.05)

    
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

    def calculate_performance_metrics(self, data: pd.DataFrame, initial_capital, savecsv) -> Dict[str, float]:
        return PerformanceMetrics.calculate_trading_metrics(data, initial_capital,save_csv=savecsv)
    

    # ============= Concrete Methods (Implemented in base class) =============
    
    def run(self, data: pd.DataFrame, savecsv) -> Dict[str, Any]:
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
            data_with_indicators = self.calculate_indicators(data)
            
            # Generate signals
            
            data_with_signals = self.generate_signals(data_with_indicators)
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(data_with_signals, self.current_capital, savecsv=savecsv)
            
            return metrics
            
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

    
   