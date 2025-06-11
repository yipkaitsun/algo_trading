"""
Z-Score based trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from ..base_strategy import BaseStrategy
from src.utils.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class ZScoreStrategy(BaseStrategy):
    def __init__(self, window=60, threshold=0.75, position_size=1.0):
        """
        Initialize Z-Score strategy with parameters.
        
        Args:
            window (int): Window size for z-score calculation
            threshold (float): Z-score threshold for signals
            position_size (float): Position size multiplier
            normalization_method (str): Method to use for normalization
                Options: 'z_score', 'min_max', 'robust', 'decimal', 'sigmoid'
        """
        self.window = window
        self.threshold = threshold
        self.position_size = position_size
        
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
    
    def calculate_position_size(self, signal: int, price: float) -> float:
        return self.position_size * (self.current_capital / price)
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        return PerformanceMetrics.calculate_metrics(data)
    
