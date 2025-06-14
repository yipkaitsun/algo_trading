"""
Z-Score based trading strategy implementation.

This strategy generates trading signals based on the z-score of price movements,
where z-score is calculated as (price - moving average) / standard deviation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from ..base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ZscoreStrategy(BaseStrategy):
    def __init__(self, window: int, threshold: float, initial_capital: float, config: Dict[str, Any]):
        self.window = window
        self.threshold = threshold
        self.positions = config['position_size']
        self.current_capital = initial_capital * self.positions
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)
        
    
    def calculate_last_window_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        last_window = data.tail(self.window)
        zscore = self.calculate_indicator(last_window)['z'].iloc[-1]
        return {'z': float(zscore)}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data['signal'] = data['z'].apply(lambda z: self.calculate_signals({'z': z}))
        return data

    def calculate_signals(self, data: Dict[str, Any]) -> int:
        return 1 if data['z'] > self.threshold else 0
    
    def calculate_indicator(self, df: pd.DataFrame)->  pd.DataFrame:
        df = df.copy()  # Create an explicit copy to avoid the warning
        df.loc[:, 'ma'] = df['close'].rolling(window=self.window).mean()
        df.loc[:, 'sd'] = df['close'].rolling(window=self.window).std()
        df.loc[:, 'z'] = (df['close'] - df['ma']) / df['sd']
        return df
    
