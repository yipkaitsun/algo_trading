import os
import sys
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the Python path when running directly
if __name__ == "__main__":
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(src_path)

from strategies.rsi.rsi_strategy import RSIStrategy
from strategies.base_backtest import BaseBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add the src directory to the Python path when running directly
if __name__ == "__main__":
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(src_path)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)





class RsiBacket(BaseBacktest):
    """ZScore strategy backtest implementation."""
    
    def __init__(self):
        super().__init__(strategy_name='rsi')
        self.strategy = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from settings.yaml"""
        config_path = self._get_project_root() / 'config' / 'settings.yaml'
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)['strategies']['rsi']
    
    def initialize_strategy(self) -> None:
        """Initialize the ZScore strategy with configuration."""
        strategy_config = self.load_config()
        self.strategy  = RSIStrategy(
            window=strategy_config['window'],
            overbought=strategy_config['overbought'],
            oversold=strategy_config['oversold'],
            position_size=strategy_config['position_size']
    )
    
    
    def load_data(self, symbol: str= 'BTC_HOURLY') -> pd.DataFrame:
        data_path =  Path(__file__).parent.parent.parent.parent / 'data' / f'{symbol}.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Validate and rename columns
        required_columns = ['datetime', 'close']
        column_mapping = {'Date': 'datetime', 'Close': 'close'}
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df = df.rename(columns=column_mapping)
        
        # Convert datetime if needed
        if isinstance(df['datetime'].iloc[0], str):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df

def main():
    """Main execution function."""
    backtest = RsiBacket()
    backtest.run_backtest()

if __name__ == "__main__":
    main() 