"""
Data fetcher module for retrieving cryptocurrency price data from Binance.
"""
import pandas as pd
import logging
import os
from ..utils.ssh_client import SSHClient

logger = logging.getLogger(__name__)

class DataFetcher:
    HOSTNAME ="ykt-piserver.zapto.org"
    PORT = 2222

    @staticmethod
    def fetch_data_ssh(symbol):
        temp_path = "temp_data.csv"
        try:
            with SSHClient(
                hostname=DataFetcher.HOSTNAME,
                username=DataFetcher.USERNAME,
                password=DataFetcher.PASSWORD,
                port=DataFetcher.PORT
            ) as ssh:
                ssh.get_file(
                    f"/home/kaitsunyip/Server/data/{symbol}_3year_hourly_binance.csv",
                    temp_path
                )
                df = pd.read_csv(temp_path)
                return df
        except Exception as e:
            logger.error(f"Error in data fetch: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
