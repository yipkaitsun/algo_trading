"""
Normalization utilities for time series data.
Supports both full series and rolling window normalization methods.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

def z_score_normalize(data: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Z-score normalization (standardization) transforms data to have mean=0 and std=1.
    Formula: (x - mean(x)) / std(x)
    
    Args:
        data: Input data to normalize (numpy array or pandas Series)
        window: If provided, performs rolling window normalization
        
    Returns:
        Normalized data with mean=0 and std=1
    """
    if isinstance(data, pd.Series):
        if window is None:
            mean = data.mean()
            std = data.std()
            return (data - mean) / std
        else:
            rolling_mean = data.rolling(window=window).mean()
            rolling_std = data.rolling(window=window).std()
            return (data - rolling_mean) / rolling_std
    else:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

def min_max_normalize(data: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Min-Max normalization scales the data to a fixed range [0, 1].
    Formula: (x - min(x)) / (max(x) - min(x))
    
    Args:
        data: Input data to normalize (numpy array or pandas Series)
        window: If provided, performs rolling window normalization
        
    Returns:
        Normalized data in range [0, 1]
    """
    if isinstance(data, pd.Series):
        if window is None:
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val)
        else:
            rolling_min = data.rolling(window=window).min()
            rolling_max = data.rolling(window=window).max()
            return (data - rolling_min) / (rolling_max - rolling_min)
    else:
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)

def robust_normalize(data: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Robust normalization using median and interquartile range.
    Less sensitive to outliers than z-score normalization.
    
    Args:
        data: Input data to normalize (numpy array or pandas Series)
        window: If provided, performs rolling window normalization
        
    Returns:
        Normalized data using robust scaling
    """
    if isinstance(data, pd.Series):
        if window is None:
            median = data.median()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            return (data - median) / iqr
        else:
            rolling_median = data.rolling(window=window).median()
            rolling_q1 = data.rolling(window=window).quantile(0.25)
            rolling_q3 = data.rolling(window=window).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            return (data - rolling_median) / rolling_iqr
    else:
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        return (data - median) / iqr

def decimal_scaling(data: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Decimal scaling normalizes data by moving the decimal point.
    Formula: x / 10^k where k is the number of digits in the maximum absolute value.
    
    Args:
        data: Input data to normalize (numpy array or pandas Series)
        window: If provided, performs rolling window normalization
        
    Returns:
        Normalized data using decimal scaling
    """
    if isinstance(data, pd.Series):
        if window is None:
            max_abs = data.abs().max()
            k = len(str(int(max_abs)))
            return data / (10 ** k)
        else:
            rolling_max_abs = data.abs().rolling(window=window).max()
            k = rolling_max_abs.apply(lambda x: len(str(int(x))))
            return data / (10 ** k)
    else:
        max_abs = np.max(np.abs(data))
        k = len(str(int(max_abs)))
        return data / (10 ** k)

def sigmoid_normalize(data: Union[np.ndarray, pd.Series], window: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Sigmoid normalization transforms data using the sigmoid function.
    Formula: 1 / (1 + e^(-x))
    Results in values between 0 and 1.
    
    Args:
        data: Input data to normalize (numpy array or pandas Series)
        window: If provided, performs rolling window normalization
        
    Returns:
        Normalized data using sigmoid function
    """
    if isinstance(data, pd.Series):
        if window is None:
            return 1 / (1 + np.exp(-data))
        else:
            # For rolling window, first normalize using z-score then apply sigmoid
            normalized = z_score_normalize(data, window=window)
            return 1 / (1 + np.exp(-normalized))
    else:
        return 1 / (1 + np.exp(-data))

def normalize_series(series: pd.Series, method: str = 'z_score', window: Optional[int] = None) -> pd.Series:
    """
    Apply normalization to a pandas Series using the specified method.
    
    Args:
        series: Input series to normalize
        method: Normalization method to use
            Options: 'min_max', 'z_score', 'robust', 'decimal', 'sigmoid'
        window: If provided, performs rolling window normalization
            
    Returns:
        Normalized series
        
    Raises:
        ValueError: If an unknown normalization method is specified
    """
    if method == 'min_max':
        return min_max_normalize(series, window=window)
    elif method == 'z_score':
        return z_score_normalize(series, window=window)
    elif method == 'robust':
        return robust_normalize(series, window=window)
    elif method == 'decimal':
        return decimal_scaling(series, window=window)
    elif method == 'sigmoid':
        return sigmoid_normalize(series, window=window)
    else:
        raise ValueError(f"Unknown normalization method: {method}") 