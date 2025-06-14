#!/usr/bin/env python3
"""
Main entry point for the algorithmic trading system.
"""

from datetime import datetime
import pytz
import logging
from src.system.system_initializer import SystemInitializer
from src.trading.trading_engine import TradingEngine

# Configure logging with Hong Kong timezone
hkt = pytz.timezone('Asia/Hong_Kong')
class HKTHandler(logging.StreamHandler):
    def emit(self, record):
        record.created = datetime.now(hkt).timestamp()
        super().emit(record)

def setup_logging() -> logging.Logger:
    """Configure and return the logger instance"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            HKTHandler(),
            logging.FileHandler('trading_system.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def main():
    """Main entry point for the algorithmic trading system"""
    try:
        # Initialize the system
        initializer = SystemInitializer()
        strategies, db = initializer.init_system()
        
        # Create and run the trading engine
        trading_engine = TradingEngine(db, symbol="BTCUSDT")
        trading_engine.run(strategies)
        
    except Exception as e:
        logger.error(f"Fatal error in trading system: {str(e)}")
        raise

if __name__ == "__main__":
    main()

