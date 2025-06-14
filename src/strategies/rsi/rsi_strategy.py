"""
RSI (Relative Strength Index) based trading strategy implementation.
"""

import pandas as pd
from typing import Dict
import logging
from ..base_strategy import BaseStrategy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RsiStrategy(BaseStrategy):
    def __init__(self, window, overbought, oversold, initial_capital ,config: Dict[str, Any]):
        self.window = window
        self.overbought = overbought  # Default value
        self.oversold = oversold  # Default value
        self.positions = config['position_size']
        self.current_capital = initial_capital  * self.positions
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)
        

    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = data['rsi'].apply(lambda z: self.calculate_signals({'rsi': z}))
        return data

    def calculate_last_window_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        last_window = data.tail(self.window)
        df = self.calculate_indicator(last_window)
        rsi = df['rsi'].iloc[-1]
        return {'rsi': rsi}

    def calculate_signals(self, data: Dict[str, Any]):
        return 1 if data['rsi'] > self.oversold else 0
    
    def calculate_indicator(self, df: pd.DataFrame)->  pd.DataFrame:
        df = df.copy()  # Create an explicit copy to avoid the warning
        
        # Calculate price changes
        df.loc[:, 'delta'] = df['close'].diff()

        # Separate gains and losses
        df.loc[:, 'gain'] = df['delta'].where(df['delta'] > 0, 0)
        df.loc[:, 'loss'] = -df['delta'].where(df['delta'] < 0, 0)
        
        # Calculate average gains and losses
        df.loc[:, 'avg_gain'] = df['gain'].rolling(window=self.window).mean()
        df.loc[:, 'avg_loss'] = df['loss'].rolling(window=self.window).mean()
        
        # Calculate RS and RSI
        df.loc[:, 'rs'] = df['avg_gain'] / df['avg_loss']
        df.loc[:, 'rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Calculate price change for performance metrics
        df.loc[:, 'chg'] = df['close'].pct_change()
        return df
        