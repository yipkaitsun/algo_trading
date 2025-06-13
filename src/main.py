#!/usr/bin/env python3
"""
Main entry point for the algorithmic trading system.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import importlib
import logging
from typing import Dict, List, Any
from src.data.fetcher import DataFetcher

# Configure logging with Hong Kong timezone
hkt = pytz.timezone('Asia/Hong_Kong')
class HKTHandler(logging.StreamHandler):
    def emit(self, record):
        record.created = datetime.now(hkt).timestamp()
        super().emit(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        HKTHandler(),
        logging.FileHandler('trading_system.log')
    ]
)

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml"""
    config_path = os.path.join('config', 'settings.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_strategies(config: Dict[str, Any]) -> List[Any]:
    """Initialize trading strategies based on configuration"""
    strategies = []
    active_strategies = config['system_state']['active_strategies']
    
    for strategy_name in active_strategies:
        try:
            # Construct module and class names based on strategy name
            module_name = f"src.strategies.{strategy_name}.{strategy_name}_strategy"
            class_name = f"{strategy_name.capitalize()}Strategy"
            
            # Dynamically import the strategy module
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # Get strategy parameters from config
            strategy_params = config['strategies'][strategy_name]
            
            # Initialize strategy with parameters
            strategy = strategy_class(**strategy_params)
            strategies.append(strategy)
            
            logger.info(f"Successfully initialized {strategy_name} strategy")
            
        except Exception as e:
            logger.error(f"Error initializing {strategy_name} strategy: {str(e)}")
            raise
    
    return strategies

def main():
    """Main entry point for the algorithmic trading system"""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Check system mode
        system_mode = config['system_state']['mode']
        logger.info(f"System running in {system_mode} mode")
        
        # Initialize strategies
        strategies = initialize_strategies(config)
        logger.info(f"Initialized {len(strategies)} trading strategies")
        
        # Get BTCUSDT hourly data
        df = DataFetcher.get_btcusdt_hourly('BTCUSDT')
        logger.info(f"Retrieved {len(df)} hourly BTCUSDT data points")
        
        # Print DataFrame with formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: '%.8f' % float(x))
        
        # Process data through each strategy
        for strategy in strategies:
            try:
                # Calculate indicators
                df_with_indicators = strategy.calculate_indicators(df)
                
                # Generate signals
                df_with_signals = strategy.generate_signals(df_with_indicators)
                
                logger.info(f"Strategy {strategy.__class__.__name__} processed data successfully")
                
            except Exception as e:
                logger.error(f"Error processing data with strategy {strategy.__class__.__name__}: {str(e)}")
                continue
        
        logger.info("Algorithmic trading system initialized and data processed")
        
        # Print the final DataFrame
        logger.info("\nBTCUSDT Hourly Data (HKT):")
        logger.info(df.tail().to_string())
        
    except Exception as e:
        logger.error(f"Error initializing trading system: {str(e)}")
        raise

if __name__ == "__main__":
    main() 