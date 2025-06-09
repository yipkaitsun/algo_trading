import os
import sys
import pandas as pd
import yaml
import logging

# Add the src directory to the Python path when running directly
if __name__ == "__main__":
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(src_path)

from strategies.zscore.zscore_strategy import ZScoreStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from settings.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config', 'settings.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config = load_config()
    strategy_config = config['strategies']['zscore']
    
    # Create strategy instance with parameters from config
    strategy = ZScoreStrategy(
        window=strategy_config['window'],
        threshold=strategy_config['threshold'],
        position_size=strategy_config['position_size']
    )
    
    logger.info(f"Initializing backtest with parameters:")
    logger.info(f"Window size: {strategy.window}")
    logger.info(f"Threshold: {strategy.threshold}")
    logger.info(f"Position size: {strategy.position_size}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'BTC_HOURLY.csv')
    df = pd.read_csv(data_path)
    
    # Rename columns to match strategy requirements
    df = df.rename(columns={
        'Date': 'datetime',
        'Close': 'close'
    })
    
    # Calculate indicators
    df = strategy.calculate_indicators(df)
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(df)
    
    # Print detailed results
    print("\nBacktest Results:")
    print("-------------------")
    print(f"Strategy Parameters:")
    print(f"  Window Size: {strategy.window}")
    print(f"  Threshold: {strategy.threshold}")
    print(f"  Position Size: {strategy.position_size}")
    
    print("\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    
    # Save detailed results to CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'backtest', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'zscore_results.csv')
    df.to_csv(results_file, index=False)
    logger.info(f"Detailed results saved to '{results_file}'")

if __name__ == "__main__":
    main() 