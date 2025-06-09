"""
RSI (Relative Strength Index) based trading strategy implementation.
"""

import pandas as pd
from typing import Dict
import logging
from ..base_strategy import BaseStrategy
from src.utils.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    def __init__(self, window=14, overbought=70, oversold=30, position_size=1.0):
        self.window = window
        self.overbought = overbought  # Default value
        self.oversold = oversold  # Default value
        self.position_size = position_size
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
            
        df = data.copy()
        
        # Calculate price changes
        df['delta'] = df['close'].diff()
        
        # Separate gains and losses
        df['gain'] = df['delta'].where(df['delta'] > 0, 0)
        df['loss'] = -df['delta'].where(df['delta'] < 0, 0)
        
        # Calculate average gains and losses
        df['avg_gain'] = df['gain'].rolling(window=self.window).mean()
        df['avg_loss'] = df['loss'].rolling(window=self.window).mean()
        
        # Calculate RS and RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Calculate price change for performance metrics
        df['chg'] = df['close'].pct_change()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, overbought: int = None, oversold: int = None) -> pd.DataFrame:
        """
        Generate trading signals based on RSI thresholds.
        
        Args:
            data (pd.DataFrame): DataFrame with RSI values
            overbought (int, optional): Overbought threshold. If None, uses instance default
            oversold (int, optional): Oversold threshold. If None, uses instance default
            
        Returns:
            pd.DataFrame: DataFrame with signals added
        """
        df = data.copy()
        
        # Use provided thresholds or fall back to instance defaults
        overbought = overbought if overbought is not None else self.overbought
        oversold = oversold if oversold is not None else self.oversold
        
        # Generate signals based on RSI thresholds
        # 1 for oversold (buy), -1 for overbought (sell), 0 for no action
        df['signal'] = 0
        df.loc[df['rsi'] > oversold, 'signal'] = 1
        
        return df
    
    def calculate_position_size(self, signal: int, price: float) -> float:
        return self.position_size * (self.current_capital / price)
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        return PerformanceMetrics.calculate_metrics(data)