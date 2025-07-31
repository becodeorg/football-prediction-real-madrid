"""
Feature Engineering Module for S&P 500 Prediction System
Responsible for creating technical indicators and preparing features for ML models
"""

import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
    def load_raw_data(self):
        """
        Load raw S&P 500 and VIX data
        
        Returns:
            tuple: (sp500_data, vix_data)
        """
        try:
            sp500_file = os.path.join(self.raw_dir, "sp500_historical.csv")
            vix_file = os.path.join(self.raw_dir, "vix_historical.csv")
            
            sp500_data = pd.read_csv(sp500_file)
            vix_data = pd.read_csv(vix_file)
            
            # Convert date columns
            sp500_data['date'] = pd.to_datetime(sp500_data['date'])
            vix_data['date'] = pd.to_datetime(vix_data['date'])
            
            logger.info(f"Loaded S&P 500 data: {len(sp500_data)} records")
            logger.info(f"Loaded VIX data: {len(vix_data)} records")
            
            return sp500_data, vix_data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            return None, None
    
    def create_target_variable(self, data):
        """
        Create target variable for next day price movement
        
        Args:
            data (pd.DataFrame): S&P 500 data
            
        Returns:
            pd.DataFrame: Data with target variable
        """
        data = data.copy()
        
        # Calculate next day's close price
        data['next_close'] = data['close'].shift(-1)
        
        # Calculate price change percentage
        data['price_change_pct'] = ((data['next_close'] - data['close']) / data['close']) * 100
        
        # Create binary target: 1 if price goes up, 0 if down
        data['target'] = (data['price_change_pct'] > 0).astype(int)
        
        # Create multi-class target for more nuanced prediction
        # 0: significant down (< -1%), 1: stable (-1% to 1%), 2: significant up (> 1%)
        data['target_multiclass'] = 1  # Default to stable
        data.loc[data['price_change_pct'] < -1, 'target_multiclass'] = 0  # Down
        data.loc[data['price_change_pct'] > 1, 'target_multiclass'] = 2   # Up
        
        logger.info("Created target variables")
        return data
    
    def create_technical_indicators(self, data):
        """
        Create technical indicators for feature engineering
        
        Args:
            data (pd.DataFrame): S&P 500 data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        data = data.copy()
        
        # Ensure we have the required columns
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Moving Averages (using pandas rolling)
        data['sma_5'] = close.rolling(window=5).mean()
        data['sma_10'] = close.rolling(window=10).mean()
        data['sma_20'] = close.rolling(window=20).mean()
        data['sma_50'] = close.rolling(window=50).mean()
        data['sma_200'] = close.rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['ema_12'] = close.ewm(span=12).mean()
        data['ema_26'] = close.ewm(span=26).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # RSI (simplified calculation)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_mid'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        data['bb_high'] = data['bb_mid'] + (bb_std * 2)
        data['bb_low'] = data['bb_mid'] - (bb_std * 2)
        data['bb_width'] = data['bb_high'] - data['bb_low']
        data['bb_position'] = (close - data['bb_low']) / (data['bb_high'] - data['bb_low'])
        
        # Stochastic Oscillator
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        data['stoch_k'] = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Volume indicators
        data['volume_sma'] = volume.rolling(window=20).mean()
        data['volume_ratio'] = volume / data['volume_sma']
        
        # Price-based features
        data['price_range'] = high - low
        data['price_range_pct'] = (data['price_range'] / close) * 100
        
        # Returns
        data['return_1d'] = close.pct_change(1)
        data['return_5d'] = close.pct_change(5)
        data['return_10d'] = close.pct_change(10)
        
        # Volatility (rolling standard deviation of returns)
        data['volatility_5d'] = data['return_1d'].rolling(window=5).std()
        data['volatility_20d'] = data['return_1d'].rolling(window=20).std()
        
        logger.info("Created technical indicators")
        return data
    
    def create_lag_features(self, data, lag_periods=[1, 2, 3, 5]):
        """
        Create lagged features
        
        Args:
            data (pd.DataFrame): Data with indicators
            lag_periods (list): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        data = data.copy()
        
        # Features to lag
        features_to_lag = ['close', 'volume', 'rsi', 'macd', 'return_1d', 'volatility_5d']
        
        for feature in features_to_lag:
            if feature in data.columns:
                for lag in lag_periods:
                    data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        logger.info(f"Created lag features for periods: {lag_periods}")
        return data
    
    def merge_with_vix(self, sp500_data, vix_data):
        """
        Merge S&P 500 data with VIX data
        
        Args:
            sp500_data (pd.DataFrame): S&P 500 data
            vix_data (pd.DataFrame): VIX data
            
        Returns:
            pd.DataFrame: Merged data
        """
        # Prepare VIX data for merging
        vix_features = vix_data[['date', 'vix_close', 'vix_volume']].copy()
        vix_features['vix_change'] = vix_features['vix_close'].pct_change()
        
        # Merge on date
        merged_data = pd.merge(sp500_data, vix_features, on='date', how='left')
        
        logger.info("Merged S&P 500 data with VIX data")
        return merged_data
    
    def clean_and_prepare_data(self, data):
        """
        Clean data and prepare for modeling
        
        Args:
            data (pd.DataFrame): Data with all features
            
        Returns:
            pd.DataFrame: Cleaned data ready for modeling
        """
        data = data.copy()
        
        # Drop rows with NaN targets (last row due to shift)
        data = data.dropna(subset=['target'])
        
        # Fill NaN values in features with forward fill then backward fill
        feature_columns = [col for col in data.columns if col not in ['date', 'target', 'target_multiclass', 'next_close', 'price_change_pct']]
        data[feature_columns] = data[feature_columns].ffill().bfill()
        
        # Only remove rows where ALL features are NaN (which shouldn't happen with proper ffill/bfill)
        data = data.dropna(subset=['close'])  # Only check for essential column
        
        logger.info(f"Cleaned data shape: {data.shape}")
        return data
    
    def save_processed_data(self, data, filename):
        """
        Save processed data to CSV
        
        Args:
            data (pd.DataFrame): Processed data
            filename (str): Filename for saving
        """
        try:
            filepath = os.path.join(self.processed_dir, filename)
            data.to_csv(filepath, index=False)
            logger.info(f"Processed data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def engineer_features(self):
        """
        Main method to run complete feature engineering pipeline
        
        Returns:
            pd.DataFrame: Fully engineered dataset
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Load raw data
        sp500_data, vix_data = self.load_raw_data()
        if sp500_data is None or vix_data is None:
            return None
        
        # Create target variable
        sp500_data = self.create_target_variable(sp500_data)
        
        # Create technical indicators
        sp500_data = self.create_technical_indicators(sp500_data)
        
        # Create lag features
        sp500_data = self.create_lag_features(sp500_data)
        
        # Merge with VIX data
        merged_data = self.merge_with_vix(sp500_data, vix_data)
        
        # Clean and prepare
        final_data = self.clean_and_prepare_data(merged_data)
        
        # Save processed data
        self.save_processed_data(final_data, "sp500_features.csv")
        
        logger.info("Feature engineering completed!")
        return final_data

def main():
    """
    Main function to run feature engineering
    """
    engineer = FeatureEngineer()
    engineered_data = engineer.engineer_features()
    
    if engineered_data is not None:
        print(f"Engineered Data Shape: {engineered_data.shape}")
        print(f"Features created: {len(engineered_data.columns)}")
        print(f"\nTarget distribution:")
        print(engineered_data['target'].value_counts())
        print(f"\nMulti-class target distribution:")
        print(engineered_data['target_multiclass'].value_counts())
        
        # Show some feature statistics
        print(f"\nSample of engineered features:")
        feature_cols = [col for col in engineered_data.columns if col not in ['date', 'target', 'target_multiclass', 'next_close', 'price_change_pct']]
        print(engineered_data[feature_cols[:10]].describe())

if __name__ == "__main__":
    main()
