from typing import Dict, List, Any, Tuple, Optional
import os
import yaml
import importlib
import logging
from src.strategies import base_strategy
from src.database.db_connection import DatabaseConnection

logger = logging.getLogger(__name__)

class SystemInitializer:
    def __init__(self):
        self.config: Dict[str, Any] = {}

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from settings.yaml"""
        config_path = os.path.join('config', 'settings.yaml')
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def initialize_strategy(self, strategy_name: str) -> Optional[base_strategy.BaseStrategy]:
        """Initialize a single trading strategy"""
        if not self.config:
            raise RuntimeError("Configuration not loaded. Call init_system() first.")
            
        try:
            module_name = f"src.strategies.{strategy_name}.{strategy_name}_strategy"
            class_name = f"{strategy_name.capitalize()}Strategy"
            
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            strategy_params = self.config['strategies'][strategy_name]
            init_params = {k: v for k, v in strategy_params.items() if k != 'position_size'}
            
            strategy = strategy_class(
                **init_params,
                initial_capital=self.config['initial_capital'],
                config=strategy_params
            )
            
            logger.info(f"Successfully initialized {strategy_name} strategy")
            return strategy
            
        except Exception as e:
            logger.error(f"Error initializing {strategy_name} strategy: {str(e)}")
            return None

    def initialize_strategies(self) -> List[base_strategy.BaseStrategy]:
        """Initialize all trading strategies based on configuration"""
        if not self.config:
            raise RuntimeError("Configuration not loaded. Call init_system() first.")
            
        strategies = []
        active_strategies = self.config['system_state']['active_strategies']
        
        for strategy_name in active_strategies:
            strategy = self.initialize_strategy(strategy_name)
            if strategy:
                strategies.append(strategy)
        
        if not strategies:
            raise RuntimeError("No strategies were successfully initialized")
        
        return strategies

    def init_system(self) -> Tuple[List[base_strategy.BaseStrategy], DatabaseConnection]:
        """Initialize the trading system by loading config and setting up strategies"""
        self.config = self.load_config()
        system_mode = self.config['system_state']['mode']
        logger.info(f"System running in {system_mode} mode")
        
        strategies = self.initialize_strategies()
        db = DatabaseConnection()
        return strategies, db 