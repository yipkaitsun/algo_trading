"""
Moving Average Crossover Strategy implementation.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy implementation.
    Uses short and long moving averages to generate trading signals.
    """
    
    def __init__(self, config):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            config (dict): Strategy configuration dictionary
        """
        super().__init__(config)
        
        # Initialize strategy parameters from config
        strategy_config = config.get('strategies', {}).get('ma_crossover', {})
        self.short_window = strategy_config.get('short_window', 15)
        self.long_window = strategy_config.get('long_window', 50)
        self.position_size = strategy_config.get('position_size', 0.1)
        
        # Get risk management parameters
        risk_config = config.get('risk', {})
        self.stop_loss = risk_config.get('stop_loss', 0.03)
        self.take_profit = risk_config.get('take_profit', 0.02)
        
        # Get trading parameters
        trading_config = config.get('trading', {})
        self.initial_capital = trading_config.get('initial_capital', 100000)
        self.commission = trading_config.get('commission', 0.001)
        self.slippage = trading_config.get('slippage', 0.0005)
    
    def validate_data(self, data):
        """
        Validate that the data has the required columns.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['close']
        return all(col in data.columns for col in required_columns)
    
    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with calculated indicators
        """
        # Calculate moving averages
        data['SMA_short'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['close'].rolling(window=self.long_window).mean()
        
        # Calculate crossover signal
        data['crossover'] = np.where(
            data['SMA_short'] > data['SMA_long'],
            1,  # Bullish crossover
            np.where(
                data['SMA_short'] < data['SMA_long'],
                -1,  # Bearish crossover
                0  # No crossover
            )
        )
        
        # Calculate trend strength using ADX
        data['TR'] = data['close'].rolling(window=14).apply(
            lambda x: max(x) - min(x)
        )
        data['ATR'] = data['TR'].rolling(window=14).mean()
        
        # Calculate price volatility
        data['volatility'] = data['close'].pct_change().rolling(window=20).std()
        
        return data
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy rules.
        
        Args:
            data (pd.DataFrame): Price data with indicators
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        # Initialize signal column
        data['signal'] = 0
        
        # Generate signals based on moving average crossover with additional filters
        for i in range(1, len(data)):
            # Skip if we don't have enough data for indicators
            if pd.isna(data['SMA_short'].iloc[i]) or pd.isna(data['SMA_long'].iloc[i]):
                continue
            
            # Get current crossover state
            current_crossover = data['crossover'].iloc[i]
            prev_crossover = data['crossover'].iloc[i-1]
            
            # Only generate signals on actual crossovers
            if current_crossover != prev_crossover:
                # Check volatility
                current_volatility = data['volatility'].iloc[i]
                avg_volatility = data['volatility'].rolling(window=20).mean().iloc[i]
                
                # Only trade if volatility is not too high
                if current_volatility <= avg_volatility * 1.5:
                    # Generate signal based on crossover direction
                    data['signal'].iloc[i] = current_crossover
        
        # Apply stop loss and take profit
        data['returns'] = data['close'].pct_change()
        data['cumulative_returns'] = (1 + data['returns']).cumprod()
        
        # Calculate stop loss and take profit levels
        data['stop_loss_level'] = data['close'].rolling(window=1).apply(
            lambda x: x[0] * (1 - self.stop_loss)
        )
        data['take_profit_level'] = data['close'].rolling(window=1).apply(
            lambda x: x[0] * (1 + self.take_profit)
        )
        
        # Apply stop loss and take profit to signals
        for i in range(1, len(data)):
            if data['signal'].iloc[i-1] != 0:  # If we have an open position
                if data['close'].iloc[i] <= data['stop_loss_level'].iloc[i]:
                    data['signal'].iloc[i] = 0  # Close position at stop loss
                elif data['close'].iloc[i] >= data['take_profit_level'].iloc[i]:
                    data['signal'].iloc[i] = 0  # Close position at take profit
        
        return data
    
    def calculate_performance_metrics(self, data):
        """
        Calculate strategy performance metrics.
        
        Args:
            data (pd.DataFrame): Price data with signals
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Calculate strategy returns
        data['strategy_returns'] = data['signal'].shift(1) * data['returns'] * self.position_size
        
        # Calculate cumulative returns
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
        
        # Calculate performance metrics
        total_return = data['cumulative_returns'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(data)) - 1
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = data['strategy_returns'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate maximum drawdown
        rolling_max = data['cumulative_returns'].expanding().max()
        drawdowns = data['cumulative_returns'] / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate and profit factor
        winning_trades = data[data['strategy_returns'] > 0]['strategy_returns']
        losing_trades = data[data['strategy_returns'] < 0]['strategy_returns']
        
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades))
        profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
