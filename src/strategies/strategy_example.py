"""
Example trading strategy implementation using Moving Average Crossover.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ExampleStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading strategy with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Strategy configuration parameters
        """
        super().__init__(config)
        
        # Strategy parameters with defaults
        strategy_config = config.get('strategy', {})
        self.short_window = strategy_config.get('short_window', 10)
        self.long_window = strategy_config.get('long_window', 20)
        self.position_size = strategy_config.get('position_size', 1.0)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with additional indicator columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
            
        df = data.copy()
        
        # Calculate moving averages
        df['SMA_short'] = df['close'].rolling(window=self.short_window).mean()
        df['SMA_long'] = df['close'].rolling(window=self.long_window).mean()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on indicators.
        
        Args:
            data (pd.DataFrame): Market data with indicators
            
        Returns:
            pd.DataFrame: Data with signal column
        """
        df = data.copy()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'signal'] = 1  # Buy signal
        df.loc[df['SMA_short'] < df['SMA_long'], 'signal'] = -1  # Sell signal
        
        return df
    
    def calculate_position_size(self, signal: int, price: float) -> float:
        """
        Calculate the position size for a trade.
        
        Args:
            signal (int): Trading signal (1 for buy, -1 for sell)
            price (float): Current price
            
        Returns:
            float: Position size
        """
        # Simple position sizing based on available capital
        return self.position_size * (self.current_capital / price)
