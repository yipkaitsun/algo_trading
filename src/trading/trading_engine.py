from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from src.strategies import base_strategy
from src.database.db_connection import DatabaseConnection
from src.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, db: DatabaseConnection, symbol: str):
        self.db = db
        self.symbol = symbol
        self.strategy_signal_map: Dict[str, int] = {}
        self.current_timestamp = datetime.now()

    def fetch_current_signals(self, strategy_names: List[str]) -> None:
        """Fetch current signals from the database for all strategies"""
        try:
            results = self.db.execute_query(
                """
                SELECT strategy, signal FROM latest_signals WHERE strategy = ANY(%s)
                """,
                (strategy_names,)
            )
            if results is not None and len(results) > 0:
                for strategy_name, signal in results:
                    self.strategy_signal_map[strategy_name] = signal
            else:
                logger.warning("No results returned from latest_signals query.")
        except Exception as e:
            logger.error(f"Error fetching signals from database: {str(e)}")

    def get_historical_data(self) -> pd.DataFrame:
        """Fetch historical data for the current symbol"""
        try:
            df = DataFetcher.fetch_data_ssh(symbol=self.symbol)
            logger.info(f"Successfully loaded historical data from ssh for {self.symbol}")
            return df
        except Exception as e:
            logger.error(f"Error reading historical data for {self.symbol}: {str(e)}")
            return pd.DataFrame()

    def process_strategy(self, strategy: base_strategy.BaseStrategy, df: pd.DataFrame) -> int:
        """Process a single strategy and return its signal"""
        try:
            df_with_indicators = strategy.calculate_last_window_indicators(df)
            return strategy.calculate_signals(df_with_indicators)
        except Exception as e:
            logger.error(f"Error processing data with strategy {strategy.__class__.__name__}: {str(e)}")
            return self.strategy_signal_map.get(strategy.__class__.__name__, 0)

    def update_strategy_signal(self, strategy_name: str, signal: int, latest_close: Optional[float]) -> None:
        """Update the signal for a strategy in the database"""
        try:
            # Always update latest_signals table with current signal
            self.db.execute_query(
                """
                INSERT INTO latest_signals (strategy, signal, timestamp)
                VALUES (%s, %s, %s)
                ON CONFLICT (strategy) DO UPDATE
                SET signal = EXCLUDED.signal, timestamp = EXCLUDED.timestamp
                """,
                (strategy_name, signal, self.current_timestamp)
            )
            
            # Only record in trading_track when signal changes
            current_signal = self.strategy_signal_map.get(strategy_name, 0)
            if signal != current_signal and latest_close is not None:
                # Convert numpy float to Python float if necessary
                close_price = float(latest_close) if isinstance(latest_close, np.floating) else latest_close
                
                self.db.execute_query(
                    """
                    INSERT INTO trading_track (symbol, close_price, signal, timestamp, strategy)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (self.symbol, close_price, signal, self.current_timestamp, strategy_name)
                )
                logger.info(f"Signal change recorded for {strategy_name}: {current_signal} -> {signal}")
            
            # Update the strategy_signal_map with the new signal
            self.strategy_signal_map[strategy_name] = signal
            
        except Exception as e:
            logger.error(f"Error storing data for {strategy_name}: {str(e)}")

    def run(self, strategies: List[base_strategy.BaseStrategy]) -> None:
        """Run the trading engine for all strategies"""
        # Initialize signal map
        self.strategy_signal_map = {strategy.__class__.__name__: 0 for strategy in strategies}
        strategy_names = [strategy.__class__.__name__ for strategy in strategies]
        
        # Fetch current signals from database
        self.fetch_current_signals(strategy_names)
        
        # Get historical data
        df = self.get_historical_data()
        latest_close = df['close'].iloc[-1] if not df.empty else None
        
        # Process each strategy
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            signal = self.process_strategy(strategy, df)
            self.update_strategy_signal(strategy_name, signal, latest_close)
        
        # Log the current signals for all strategies
        logger.info("Current strategy signals:")
        for strategy_name, signal in self.strategy_signal_map.items():
            logger.info(f"{strategy_name}: {signal}") 