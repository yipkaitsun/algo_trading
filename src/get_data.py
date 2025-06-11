import requests
import csv
import time
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path

BASE_URL = "https://data-api.binance.vision/api/v3"

def to_unix_timestamp_ms(dt):
    return int(dt.timestamp() * 1000)

# Function to fetch kline data for a date range
def fetch_kline_data(symbol, interval, start_time_ms, end_time_ms, limit=1000):
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
        print(f"Error fetching data for {start_time_ms} to {end_time_ms}: {response.status_code}")
        return None

# Main script to fetch 3 years of hourly data
def fetch_3year_hourly_data_csv(symbol="BTCUSDT", interval="1h"):
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=3)
    start_time_ms = to_unix_timestamp_ms(start_date)
    end_time_ms = to_unix_timestamp_ms(end_date)
    
    # Initialize CSV output file
    output_file = Path(__file__).parent.parent / 'data' / 'btcusdt_3year_hourly_binance.csv'
    # Create data directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    header = ["timestamp", "open", "high", "low", "close", "volume"]
    total_rows = 0
    
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        
        current_start_ms = start_time_ms
        while current_start_ms < end_time_ms:
            print(f"Fetching data from {datetime.fromtimestamp(current_start_ms/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            klines = fetch_kline_data(symbol, interval, current_start_ms, end_time_ms)
            if klines:
                for kline in klines:
                    # Convert open_time (ms) to readable timestamp
                    timestamp = datetime.fromtimestamp(kline[0] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([timestamp, kline[1], kline[2], kline[3], kline[4], kline[5]])
                    total_rows += 1
                # Update start time to the next kline (last kline's close_time + 1ms)
                if klines:
                    current_start_ms = klines[-1][6] + 1  # close_time + 1ms
                else:
                    break
            else:
                print(f"No data for {current_start_ms}")
                break
            # Respect rate limits (1200 requests/minute ≈ 20 requests/second)
            time.sleep(0.05)  # 50ms delay
    
    print(f"Saved {total_rows} hourly data points to {output_file}")
    return total_rows

# Execute the script
if __name__ == "__main__":
    fetch_3year_hourly_data_csv()