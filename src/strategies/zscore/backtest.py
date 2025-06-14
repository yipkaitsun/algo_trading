import os
import sys
import logging

# Add the src directory to the Python path when running directly
if __name__ == "__main__":
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(src_path)

from .zscore_strategy import ZscoreStrategy
from strategies.base_backtest import BaseBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZScoreBacktest(BaseBacktest):
    """ZScore strategy backtest implementation."""
    
    def __init__(self):
        super().__init__(strategy_name='zscore')
    
    def initialize_strategy(self) -> None:
        """Initialize the ZScore strategy with configuration."""
        config = self.load_config()
        strategy_config = config['strategies']['zscore']
        self.strategy = ZscoreStrategy(
            window=strategy_config['window'],
            threshold=strategy_config['threshold'],
            config=strategy_config,
            initial_capital = config['initial_capital']
        )
    


def main():
    """Main execution function."""
    backtest = ZScoreBacktest()
    backtest.run_backtest('BTCUSDT')

if __name__ == "__main__":
    main() 