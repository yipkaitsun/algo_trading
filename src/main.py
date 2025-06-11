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
from typing import Dict, List, Any
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
            print(f"Loading strategy: {module_name} with class {class_name}")
            
            # Dynamically import the strategy module
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # Get strategy parameters from config
            strategy_params = config['strategies'][strategy_name]
            
            # Initialize strategy with parameters
            strategy = strategy_class(**strategy_params)
            strategies.append(strategy)
            
            print(f"Successfully initialized {strategy_name} strategy")
            
        except Exception as e:
            print(f"Error initializing {strategy_name} strategy: {str(e)}")
            raise
    
    return strategies

def main():
    """Main entry point for the algorithmic trading system"""
    try:
        # Load configuration
        config = load_config()
        print("Configuration loaded successfully")
        
        # Check system mode
        system_mode = config['system_state']['mode']
        print(f"System running in {system_mode} mode")
        
        # Initialize strategies
        strategies = initialize_strategies(config)
        print(f"Initialized {len(strategies)} trading strategies")
        
        # TODO: Set up data pipeline
        # TODO: Implement trading strategy execution
        # TODO: Set up risk management
        # TODO: Initialize backtesting if needed
        
        print("Algorithmic trading system initialized")
        
    except Exception as e:
        print(f"Error initializing trading system: {str(e)}")
        raise

if __name__ == "__main__":
    main() 