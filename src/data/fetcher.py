"""
Data fetcher module for retrieving cryptocurrency price data from Binance.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List
import yaml
import os
import time
import pytz

logger = logging.getLogger(__name__)

class DataFetcher:
    BASE_URL = "https://data-api.binance.vision/api/v3"
    
    @staticmethod
    def get_btcusdt_hourly(symbol: str) -> pd.DataFrame:
        url = (f"{DataFetcher.BASE_URL}/uiKlines?"
               f"symbol={symbol}&interval=1h&limit=1")
        
        # Make the API request
        response = requests.get(url)
        
        # Process the new data
        new_df = DataFetcher.process_btcusdt_klines(response.json())
        
        if not new_df.empty:
            # Define the CSV file path
            csv_path = 'data/btcusdt_3year_hourly_binance.csv'
            
            try:
                # Append new data to the end of the file
                new_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
                logger.info(f"Successfully appended new data to {csv_path}")
                
            except Exception as e:
                logger.error(f"Error appending data to CSV: {str(e)}")
        
        return new_df
    
    @staticmethod
    def process_btcusdt_klines(raw_klines: List[list]) -> Optional[pd.DataFrame]:
        try:
            if not raw_klines:
                logger.error("No kline data provided")
                return pd.DataFrame()
            
            # Create DataFrame from raw kline data
            df = pd.DataFrame(raw_klines, columns=[
                'datetime', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime (Binance API returns UTC time)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            # Convert UTC to Hong Kong time
            hkt = pytz.timezone('Asia/Hong_Kong')
            df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(hkt)
            
            # Convert numeric columns to float
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Format datetime to string in HKT
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Select and reorder columns to match desired format
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Format numbers to 8 decimal places
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].apply(lambda x: f"{x:.8f}")
            
            logger.info(f"Successfully processed {len(df)} klines")
            return df
            
        except Exception as e:
            logger.error(f"Error processing kline data: {str(e)}")
            return pd.DataFrame()
