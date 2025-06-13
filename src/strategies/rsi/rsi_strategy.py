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
    
    def generate_signals(self, data: pd.DataFrame, overbought: Optional[int] =None, oversold: Optional[int] =None) -> pd.DataFrame:

        df = data.copy()
        
        overbought = overbought if overbought is not None else self.overbought
        oversold = oversold if oversold is not None else self.oversold
        
        df['signal'] = 0
        df.loc[df['rsi'] > oversold, 'signal'] = 1
        
        return df
