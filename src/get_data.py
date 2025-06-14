import requests
import csv
import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://data-api.binance.vision/api/v3"

def to_unix_timestamp_ms(dt):
    """Convert datetime to Unix timestamp in milliseconds"""
    return int(dt.timestamp() * 1000)

# Function to fetch kline data for a date range
def fetch_kline_data(symbol, interval, start_time_ms, end_time_ms, limit=1000):
    """Fetch kline data from Binance API"""
    url = (
        f"{BASE_URL}/uiKlines?"
        f"symbol={symbol}"
        f"&interval={interval}"
        f"&startTime={start_time_ms}"
        f"&endTime={end_time_ms}"
        f"&limit={limit}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Error fetching data for {start_time_ms} to {end_time_ms}: {response.status_code}")
        return None

# Main script to fetch 3 years of hourly data
def fetch_3year_hourly_data_csv(symbol, interval="1h"):
    """Fetch 3 years of hourly data and save to CSV"""
    # Use UTC time for consistency
    end_date = datetime.now(timezone.utc)
    start_date = end_date - relativedelta(years=3)
    # Round start_date to the nearest hour
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    start_time_ms = to_unix_timestamp_ms(start_date)
    end_time_ms = to_unix_timestamp_ms(end_date)
    
    # Initialize CSV output file
    output_file = Path(__file__).parent.parent / 'data' / f'{symbol}_3year_hourly_binance.csv'
    # Create data directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    header = ["datetime", "open", "high", "low", "close", "volume"]
    total_rows = 0
    
    # Use a dictionary to store the latest data for each timestamp
    timestamp_data = {}
    
    current_start_ms = start_time_ms
    while current_start_ms < end_time_ms:
        logger.info(f"Fetching data from {datetime.fromtimestamp(current_start_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        klines = fetch_kline_data(symbol, interval, current_start_ms, end_time_ms)
        if klines:
            for kline in klines:
                # Convert open_time (ms) to readable timestamp in UTC
                timestamp = datetime.fromtimestamp(kline[0] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                
                # Store the latest data for each timestamp
                timestamp_data[timestamp] = [timestamp, kline[1], kline[2], kline[3], kline[4], kline[5]]
            
            # Update start time to the next hour
            if klines:
                next_hour = datetime.fromtimestamp(klines[-1][0] / 1000) + relativedelta(hours=1)
                current_start_ms = to_unix_timestamp_ms(next_hour)
            else:
                break
        else:
            logger.error(f"No data for {current_start_ms}")
            # Retry from 1 hour ago
            retry_time = datetime.fromtimestamp(current_start_ms/1000) - relativedelta(hours=1)
            current_start_ms = to_unix_timestamp_ms(retry_time)
            logger.info(f"Retrying from 1 hour ago: {retry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            continue
        # Respect rate limits (1200 requests/minute ≈ 20 requests/second)
        time.sleep(0.05)  # 50ms delay
    
    # Write the data to CSV, sorted by timestamp
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        
        # Sort timestamps and write data
        for timestamp in sorted(timestamp_data.keys()):
            csv_writer.writerow(timestamp_data[timestamp])
            total_rows += 1
    
    logger.info(f"Saved {total_rows} hourly data points to {output_file}")
    return total_rows

# Execute the script
if __name__ == "__main__":
    fetch_3year_hourly_data_csv("BTCUSDT")