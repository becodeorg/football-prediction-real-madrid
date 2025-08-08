"""
Dataset module for S&P 500 machine learning models
Provides data preprocessing and feature engineering for training ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data.fetch_data import DataFetcher


class SP500Dataset:
    """
    Dataset class for S&P 500 machine learning models
    Handles data collection, preprocessing, feature engineering, and train/test splits
    """
    
    def __init__(self, symbol: str = '^GSPC', start_date: str = '2020-01-01', 
                 interval: str = '1d', test_size: float = 0.2, save_csv: bool = True):
        """
        Initialize the dataset
        
        Args:
            symbol (str): Stock symbol to fetch data for
            start_date (str): Start date for data collection (YYYY-MM-DD)
            interval (str): Data interval (1d for daily)
            test_size (float): Proportion of data to use for testing
            save_csv (bool): Whether to automatically save processed data as CSV
        """
        self.symbol = symbol
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.now()
        self.interval = interval
        self.test_size = test_size
        self.save_csv = save_csv
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
        # Scalers for normalization
        self.feature_scaler = None
        self.target_scaler = None
        
        # Train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Data fetcher
        self.fetcher = DataFetcher()
        
        # CSV file paths
        self.csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'csv')
        os.makedirs(self.csv_dir, exist_ok=True)
        
        date_str = datetime.now().strftime('%Y%m%d')
        self.raw_csv_path = os.path.join(self.csv_dir, f"{symbol.replace('^', '')}_raw_{start_date}_{date_str}.csv")
        self.processed_csv_path = os.path.join(self.csv_dir, f"{symbol.replace('^', '')}_processed_{start_date}_{date_str}.csv")
        self.train_csv_path = os.path.join(self.csv_dir, f"{symbol.replace('^', '')}_train_{start_date}_{date_str}.csv")
        self.test_csv_path = os.path.join(self.csv_dir, f"{symbol.replace('^', '')}_test_{start_date}_{date_str}.csv")
        
        print(f"📊 SP500Dataset initialized for {symbol}")
        print(f"   📅 Date range: {start_date} to {datetime.now().strftime('%Y-%m-%d')}")
        print(f"   ⏱️ Interval: {interval}")
        print(f"   🎯 Test size: {test_size:.1%}")
        if save_csv:
            print(f"   💾 CSV files will be saved to: {self.csv_dir}")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch raw data from the data source
        
        Returns:
            pd.DataFrame: Raw OHLCV data
        """
        print(f"\n📥 Fetching data for {self.symbol}...")
        
        self.raw_data = self.fetcher.get_historical_data_by_dates(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval
        )
        
        if self.raw_data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Clean data
        self.raw_data = self.raw_data.dropna()
        
        print(f"✅ Fetched {len(self.raw_data)} records")
        print(f"   📊 Date range: {self.raw_data.index.min().strftime('%Y-%m-%d')} to {self.raw_data.index.max().strftime('%Y-%m-%d')}")
        print(f"   💰 Price range: ${self.raw_data['low'].min():.2f} - ${self.raw_data['high'].max():.2f}")
        
        # Save raw data to CSV
        if self.save_csv:
            self.raw_data.to_csv(self.raw_csv_path)
            print(f"   💾 Raw data saved to: {self.raw_csv_path}")
        
        return self.raw_data
    
    def create_technical_indicators(self) -> pd.DataFrame:
        """
        Create technical indicators as features
        
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        print("\n🔧 Creating technical indicators...")
        
        if self.raw_data is None:
            raise ValueError("Raw data not available. Run fetch_data() first.")
        
        df = self.raw_data.copy()
        
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_10d'] = df['close'].pct_change(periods=10)
        
        # Volatility
        df['volatility_10d'] = df['price_change'].rolling(window=10).std()
        df['volatility_20d'] = df['price_change'].rolling(window=20).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_avg'] = df['hl_spread'].rolling(window=10).mean()
        
        # Support and Resistance levels (simplified)
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support_distance'] = (df['close'] - df['support']) / df['close']
        df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        print(f"✅ Created {df.shape[1] - self.raw_data.shape[1]} technical indicators")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, prediction_days: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Create target variables for prediction
        
        Args:
            df (pd.DataFrame): DataFrame with features
            prediction_days (List[int]): Days ahead to predict
        
        Returns:
            pd.DataFrame: Data with target variables
        """
        print(f"\n🎯 Creating target variables for {prediction_days} days ahead...")
        
        for days in prediction_days:
            # Future price
            df[f'target_price_{days}d'] = df['close'].shift(-days)
            
            # Future return (calculate correctly)
            df[f'target_return_{days}d'] = (df['close'].shift(-days) - df['close']) / df['close']
            
            # Binary classification: up/down
            df[f'target_direction_{days}d'] = (df[f'target_return_{days}d'] > 0).astype(int)
            
            # Skip volatility prediction for now (it's causing issues)
            # df[f'target_volatility_{days}d'] = ...
        
        print(f"✅ Created target variables for {len(prediction_days)} prediction horizons")
        
        return df
    
    def preprocess_data(self, prediction_days: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline
        
        Args:
            prediction_days (List[int]): Days ahead to predict
        
        Returns:
            pd.DataFrame: Processed data ready for ML
        """
        print("\n🔄 Starting data preprocessing pipeline...")
        
        # Create technical indicators
        df = self.create_technical_indicators()
        
        # Create targets
        df = self.create_target_variables(df, prediction_days)
        
        # Remove rows with NaN values, but be more conservative
        initial_rows = len(df)
        
        # First, remove rows where basic technical indicators are NaN (first 50 days due to SMA50)
        df = df.iloc[50:]  # Skip first 50 days to allow for technical indicators
        
        # Then remove the last max(prediction_days) rows since they don't have targets
        max_pred_days = max(prediction_days) if prediction_days else 1
        if len(df) > max_pred_days:
            df = df.iloc[:-max_pred_days]
        
        # Finally, drop any remaining NaN values
        df = df.dropna()
        
        final_rows = len(df)
        
        print(f"📊 Data cleaned: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed due to NaN and edge effects)")
        
        if final_rows == 0:
            raise ValueError("No data remaining after preprocessing. Check your date range and prediction days.")
        
        self.processed_data = df
        
        # Save processed data to CSV
        if self.save_csv:
            self.processed_data.to_csv(self.processed_csv_path)
            print(f"💾 Processed data saved to: {self.processed_csv_path}")
        
        return df
    
    def prepare_features_and_targets(self, target_type: str = 'return_1d', 
                                   exclude_future_data: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and targets for ML training
        
        Args:
            target_type (str): Type of target to predict ('return_1d', 'direction_1d', 'price_1d', etc.)
            exclude_future_data (bool): Whether to exclude features that might contain future information
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and targets
        """
        print(f"\n🎯 Preparing features and targets for '{target_type}' prediction...")
        
        if self.processed_data is None:
            raise ValueError("Processed data not available. Run preprocess_data() first.")
        
        df = self.processed_data.copy()
        
        # Define feature columns (exclude original OHLCV, targets, and potentially future-leaking data)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits', 'symbol', 'date']
        
        # Exclude all target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        exclude_cols.extend(target_cols)
        
        # Select target
        if f'target_{target_type}' not in df.columns:
            available_targets = [col.replace('target_', '') for col in target_cols]
            raise ValueError(f"Target '{target_type}' not found. Available targets: {available_targets}")
        
        target = df[f'target_{target_type}']
        
        # Select features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols]
        
        print(f"✅ Selected {len(feature_cols)} features and 1 target")
        print(f"   📊 Features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        print(f"   🎯 Target: target_{target_type}")
        print(f"   📈 Target stats: min={target.min():.4f}, max={target.max():.4f}, mean={target.mean():.4f}")
        
        self.features = features
        self.targets = target
        
        return features, target
    
    def create_train_test_split(self, scale_data: bool = True, 
                               scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split and optionally scale the data
        
        Args:
            scale_data (bool): Whether to scale the features and targets
            scaler_type (str): Type of scaler ('standard' or 'minmax')
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        print(f"\n📊 Creating train/test split (test_size={self.test_size:.1%})...")
        
        if self.features is None or self.targets is None:
            raise ValueError("Features and targets not available. Run prepare_features_and_targets() first.")
        
        if len(self.features) == 0:
            raise ValueError("No data available for train/test split. Check your preprocessing pipeline.")
        
        # Time-based split (more realistic for time series)
        split_index = int(len(self.features) * (1 - self.test_size))
        
        # Ensure we have at least some data in both splits
        if split_index <= 1:
            split_index = max(1, len(self.features) - 1)
        
        self.X_train = self.features.iloc[:split_index]
        self.X_test = self.features.iloc[split_index:]
        self.y_train = self.targets.iloc[:split_index]
        self.y_test = self.targets.iloc[split_index:]
        
        print(f"✅ Split created:")
        print(f"   🏋️ Training set: {len(self.X_train)} samples")
        print(f"   🧪 Test set: {len(self.X_test)} samples")
        
        if len(self.X_train) > 0 and len(self.X_test) > 0:
            print(f"   📅 Train period: {self.X_train.index.min().strftime('%Y-%m-%d')} to {self.X_train.index.max().strftime('%Y-%m-%d')}")
            print(f"   📅 Test period: {self.X_test.index.min().strftime('%Y-%m-%d')} to {self.X_test.index.max().strftime('%Y-%m-%d')}")
        else:
            print("   ⚠️ Warning: One or both splits are empty")
        
        # Scale data if requested
        if scale_data:
            print(f"\n🔧 Scaling data using {scaler_type} scaler...")
            
            if scaler_type == 'standard':
                self.feature_scaler = StandardScaler()
                self.target_scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.feature_scaler = MinMaxScaler()
                self.target_scaler = MinMaxScaler()
            else:
                raise ValueError("scaler_type must be 'standard' or 'minmax'")
            
            # Fit on training data only
            self.X_train = pd.DataFrame(
                self.feature_scaler.fit_transform(self.X_train),
                columns=self.X_train.columns,
                index=self.X_train.index
            )
            
            self.X_test = pd.DataFrame(
                self.feature_scaler.transform(self.X_test),
                columns=self.X_test.columns,
                index=self.X_test.index
            )
            
            # Scale targets (for regression tasks)
            if not self.targets.dtype == 'int64':  # Don't scale binary classification targets
                self.y_train = pd.Series(
                    self.target_scaler.fit_transform(self.y_train.values.reshape(-1, 1)).flatten(),
                    index=self.y_train.index
                )
                
                self.y_test = pd.Series(
                    self.target_scaler.transform(self.y_test.values.reshape(-1, 1)).flatten(),
                    index=self.y_test.index
                )
            
            print("✅ Data scaling completed")
        
        # Save train/test splits to CSV
        if self.save_csv:
            # Combine features and targets for complete datasets
            train_data = self.X_train.copy()
            train_data['target'] = self.y_train
            train_data.to_csv(self.train_csv_path)
            
            test_data = self.X_test.copy()
            test_data['target'] = self.y_test
            test_data.to_csv(self.test_csv_path)
            
            print(f"💾 Training data saved to: {self.train_csv_path}")
            print(f"💾 Test data saved to: {self.test_csv_path}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the dataset
        
        Returns:
            dict: Dataset summary statistics
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        summary = {
            "symbol": self.symbol,
            "date_range": f"{self.processed_data.index.min().strftime('%Y-%m-%d')} to {self.processed_data.index.max().strftime('%Y-%m-%d')}",
            "total_samples": len(self.processed_data),
            "features_count": len([col for col in self.processed_data.columns if not col.startswith('target_')]),
            "targets_count": len([col for col in self.processed_data.columns if col.startswith('target_')]),
            "price_stats": {
                "min": self.processed_data['close'].min(),
                "max": self.processed_data['close'].max(),
                "mean": self.processed_data['close'].mean(),
                "std": self.processed_data['close'].std()
            }
        }
        
        if self.X_train is not None:
            summary.update({
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "train_period": f"{self.X_train.index.min().strftime('%Y-%m-%d')} to {self.X_train.index.max().strftime('%Y-%m-%d')}",
                "test_period": f"{self.X_test.index.min().strftime('%Y-%m-%d')} to {self.X_test.index.max().strftime('%Y-%m-%d')}"
            })
        
        return summary
    
    def get_csv_files(self) -> dict:
        """
        Get paths to all CSV files created by the dataset
        
        Returns:
            dict: Dictionary with CSV file paths and their descriptions
        """
        csv_files = {
            "raw_data": {
                "path": self.raw_csv_path,
                "description": "Raw OHLCV data from yfinance",
                "exists": os.path.exists(self.raw_csv_path)
            },
            "processed_data": {
                "path": self.processed_csv_path,
                "description": "Processed data with technical indicators and targets",
                "exists": os.path.exists(self.processed_csv_path)
            },
            "train_data": {
                "path": self.train_csv_path,
                "description": "Training dataset (features + target)",
                "exists": os.path.exists(self.train_csv_path)
            },
            "test_data": {
                "path": self.test_csv_path,
                "description": "Test dataset (features + target)",
                "exists": os.path.exists(self.test_csv_path)
            }
        }
        
        return csv_files
    
    def save_processed_data(self, filepath: str = None):
        """
        Save processed data to CSV file
        
        Args:
            filepath (str): Path to save the CSV file (optional, uses default if not provided)
        """
        if self.processed_data is None:
            raise ValueError("No processed data to save")
        
        if filepath is None:
            filepath = self.processed_csv_path
        
        self.processed_data.to_csv(filepath)
        print(f"💾 Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str = None):
        """
        Load processed data from CSV file
        
        Args:
            filepath (str): Path to the CSV file (optional, uses default if not provided)
        """
        if filepath is None:
            filepath = self.processed_csv_path
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
            
        self.processed_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"📁 Processed data loaded from {filepath}")
    
    @staticmethod
    def load_csv_data(filepath: str) -> pd.DataFrame:
        """
        Static method to load any CSV file created by the dataset
        
        Args:
            filepath (str): Path to the CSV file
        
        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
            
        return pd.read_csv(filepath, index_col=0, parse_dates=True)


def create_sp500_training_dataset(symbol: str = '^GSPC', target_type: str = 'return_1d', 
                                save_csv: bool = True) -> SP500Dataset:
    """
    Convenience function to create a complete S&P 500 training dataset
    
    Args:
        symbol (str): Stock symbol to use
        target_type (str): Target variable type
        save_csv (bool): Whether to save data as CSV files
    
    Returns:
        SP500Dataset: Fully prepared dataset ready for ML training
    """
    print("🚀 Creating complete S&P 500 training dataset...")
    print("=" * 60)
    
    # Initialize dataset
    dataset = SP500Dataset(symbol=symbol, start_date='2020-01-01', save_csv=save_csv)
    
    # Full pipeline
    dataset.fetch_data()
    dataset.preprocess_data(prediction_days=[1, 5, 10])
    dataset.prepare_features_and_targets(target_type=target_type)
    dataset.create_train_test_split(scale_data=True, scaler_type='standard')
    
    # Print summary
    print("\n📋 Dataset Summary:")
    print("=" * 30)
    summary = dataset.get_data_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Print CSV files information
    if save_csv:
        print("\n📁 CSV Files Created:")
        print("=" * 25)
        csv_files = dataset.get_csv_files()
        for name, info in csv_files.items():
            status = "✅ Created" if info['exists'] else "❌ Missing"
            print(f"{status} {name}: {info['description']}")
            print(f"        Path: {info['path']}")
    
    print("\n✅ Dataset ready for machine learning!")
    print("🎯 Access training data: dataset.X_train, dataset.y_train")
    print("🧪 Access test data: dataset.X_test, dataset.y_test")
    if save_csv:
        print("💾 CSV files available for external use")
    
    return dataset


if __name__ == "__main__":
    # Example usage
    print("📊 SP500 Dataset Creation Example")
    print("=" * 40)
    
    # Create a complete dataset
    dataset = create_sp500_training_dataset(
        symbol='^GSPC',
        target_type='return_1d'  # Predict 1-day returns
    )
    
    print(f"\n🎉 Dataset created successfully!")
    print(f"Ready to train ML models with {len(dataset.X_train)} training samples!")
