"""
Z-Score based trading strategy implementation.
"""

from turtle import position
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from ..base_strategy import BaseStrategy
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ZscoreStrategy(BaseStrategy):
    def __init__(self, window, threshold,initial_capital ,config: Dict[str, Any]):
        self.window = window
        self.threshold = threshold
        self.positions = config['position_size']
        self.current_capital = initial_capital  * self.positions

        
    def validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)
        
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
            
        df = data.copy()
        
        # Calculate rolling window statistics
        df['ma'] = df['close'].rolling(window=self.window).mean()
        df['sd'] = df['close'].rolling(window=self.window).std()
        
        # Calculate z-score using the selected normalization method
        df['z'] = (df['close'] - df['ma']) / df['sd']
     
        # Calculate price change
        
        return df
    
    def generate_signals(self, data: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        if threshold is None:
            threshold = self.threshold
        data['signal'] = np.where(data['z'] > threshold, 1, 0)
        return data
