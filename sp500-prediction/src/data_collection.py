"""
Data Collection Module for S&P 500 Prediction System
Responsible for fetching historical and current S&P 500 data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500DataCollector:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def fetch_sp500_data(self, period="5y", interval="1d"):
        """
        Fetch S&P 500 historical data
        
        Args:
            period (str): Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: S&P 500 data
        """
        try:
            logger.info(f"Fetching S&P 500 data for period: {period}")
            
            # Use ^GSPC ticker for S&P 500
            sp500 = yf.Ticker("^GSPC")
            data = sp500.history(period=period, interval=interval)
            
            # Clean column names
            data.columns = data.columns.str.lower()
            
            # Add date column
            data['date'] = data.index
            data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 data: {e}")
            return None
    
    def fetch_vix_data(self, period="5y"):
        """
        Fetch VIX (Volatility Index) data as additional feature
        
        Args:
            period (str): Period to fetch data for
            
        Returns:
            pd.DataFrame: VIX data
        """
        try:
            logger.info(f"Fetching VIX data for period: {period}")
            
            vix = yf.Ticker("^VIX")
            data = vix.history(period=period)
            
            # Clean column names and add prefix
            data.columns = ['vix_' + col.lower() for col in data.columns]
            data['date'] = data.index
            data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} VIX records")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return None
    
    def save_raw_data(self, data, filename):
        """
        Save raw data to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Filename for the CSV
        """
        try:
            filepath = os.path.join(self.raw_dir, filename)
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def get_latest_data(self):
        """
        Get the latest day's data for S&P 500
        
        Returns:
            pd.DataFrame: Latest day's data
        """
        try:
            logger.info("Fetching latest S&P 500 data")
            
            sp500 = yf.Ticker("^GSPC")
            data = sp500.history(period="2d", interval="1d")  # Get last 2 days to ensure we have latest
            
            # Clean and format
            data.columns = data.columns.str.lower()
            data['date'] = data.index
            data.reset_index(drop=True, inplace=True)
            
            return data.tail(1)  # Return only the latest record
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return None
    
    def collect_all_data(self):
        """
        Collect all required data and save to files
        """
        logger.info("Starting comprehensive data collection...")
        
        # Fetch S&P 500 data
        sp500_data = self.fetch_sp500_data(period="5y")
        if sp500_data is not None:
            self.save_raw_data(sp500_data, "sp500_historical.csv")
        
        # Fetch VIX data
        vix_data = self.fetch_vix_data(period="5y")
        if vix_data is not None:
            self.save_raw_data(vix_data, "vix_historical.csv")
        
        logger.info("Data collection completed!")
        
        return sp500_data, vix_data

def main():
    """
    Main function to run data collection
    """
    collector = SP500DataCollector()
    sp500_data, vix_data = collector.collect_all_data()
    
    if sp500_data is not None:
        print(f"S&P 500 Data Shape: {sp500_data.shape}")
        print(f"Date Range: {sp500_data['date'].min()} to {sp500_data['date'].max()}")
        print("\nS&P 500 Data Sample:")
        print(sp500_data.head())
    
    if vix_data is not None:
        print(f"\nVIX Data Shape: {vix_data.shape}")
        print("VIX Data Sample:")
        print(vix_data.head())

if __name__ == "__main__":
    main()
