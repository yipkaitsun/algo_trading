"""
Main entry point for the algorithmic trading system.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from .utils.data_loader import DataLoader
from .strategies.strategy_example import ExampleStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config/settings.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def main():
    """Main entry point for the algorithmic trading system."""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize data loader
        data_loader = DataLoader(config)
        logger.info("Data loader initialized")

        # Load market data
        market_data = data_loader.load_data()
        logger.info(f"Loaded market data for {len(market_data)} symbols")

        # Initialize and run strategy
        strategy = ExampleStrategy(config)
        results = strategy.run(market_data)
        logger.info("Strategy execution completed")

        # Process and display results
        logger.info(f"Strategy results: {results}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
