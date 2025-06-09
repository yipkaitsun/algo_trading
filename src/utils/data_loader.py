import pandas as pd
from typing import Dict, Any

class DataLoader:
    """
    Loads market data based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def load_data(self):
        data_sources = self.config.get("data_sources", {})
        market_data = {}

        for symbol, path in data_sources.items():
            try:
                df = pd.read_csv(path)
                market_data[symbol] = df
            except Exception as e:
                print(f"Failed to load data for {symbol} from {path}: {e}")

        return market_data