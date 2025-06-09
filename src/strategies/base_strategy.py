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
        self.positions = {}  # Current positions
        self.trades = []     # Trade history
        self.initial_capital = config.get('trading', {}).get('initial_capital', 100000)
        self.current_capital = self.initial_capital
        
    # ============= Abstract Methods (Must be implemented by child classes) =============
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        This method must be implemented by all strategy classes.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with additional indicator columns
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
    def calculate_position_size(self, signal: int, price: float) -> float:
        """
        Calculate the position size for a trade.
        This method must be implemented by all strategy classes.
        
        Args:
            signal (int): Trading signal (1 for buy, -1 for sell)
            price (float): Current price
            
        Returns:
            float: Position size
        """
        pass
    
    # ============= Concrete Methods (Implemented in base class) =============
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required columns.
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def calculate_metrics(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        
        Args:
            trades (List[Dict[str, Any]]): List of executed trades
            data (pd.DataFrame): Market data
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if not trades:
            return self._get_empty_metrics()
            
        returns = self._calculate_returns(trades)
        metrics = {
            'total_trades': len(trades),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'total_return': self._calculate_total_return(returns),
            'annualized_return': self._calculate_annualized_return(returns)
        }
        
        return metrics
    
    def run(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run the trading strategy on the provided market data.
        
        Args:
            market_data (Dict[str, pd.DataFrame]): Dictionary of market data for different symbols
            
        Returns:
            Dict[str, Any]: Strategy results including trades and metrics
        """
        results = {}
        
        for symbol, data in market_data.items():
            logger.info(f"Processing {symbol}")
            
            if not self.validate_data(data):
                logger.error(f"Invalid data format for {symbol}")
                continue
                
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Generate signals
            data_with_signals = self.generate_signals(data_with_indicators)
            
            # Generate trades
            trades = self._generate_trades(data_with_signals)
            
            # Calculate metrics
            metrics = self.calculate_metrics(trades, data)
            
            results[symbol] = {
                'trades': trades,
                'metrics': metrics
            }
            
            logger.info(f"Completed processing {symbol}")
            logger.info(f"Metrics: {metrics}")
        
        return results
    
    # ============= Private Helper Methods =============
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0
        }
    
    def _calculate_returns(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate returns from trades."""
        returns = []
        for i in range(1, len(trades)):
            if trades[i]['type'] == 'SELL':
                returns.append((trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price'])
        return np.array(returns)
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade['type'] == 'SELL' and 
                           trade['price'] > trades[trades.index(trade)-1]['price'])
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor from returns."""
        if len(returns) == 0:
            return 0.0
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        if len(returns) == 0:
            return 0.0
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        return abs(drawdowns.min())
    
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return from returns."""
        if len(returns) == 0:
            return 0.0
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return from returns."""
        if len(returns) == 0:
            return 0.0
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252  # Assuming daily data
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    def _generate_trades(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trades from signals."""
        trades = []
        current_position = 0
        
        for index, row in data.iterrows():
            if row['signal'] == 1 and current_position <= 0:  # Buy signal
                position_size = self.calculate_position_size(1, row['close'])
                trades.append({
                    'timestamp': index,
                    'type': 'BUY',
                    'price': row['close'],
                    'size': position_size
                })
                current_position = 1
            elif row['signal'] == -1 and current_position >= 0:  # Sell signal
                position_size = self.calculate_position_size(-1, row['close'])
                trades.append({
                    'timestamp': index,
                    'type': 'SELL',
                    'price': row['close'],
                    'size': position_size
                })
                current_position = -1
                
        return trades 