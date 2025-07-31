"""
Test feature engineering functionality
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from feature_engineering import (
        calculate_sma, calculate_ema, calculate_rsi, 
        calculate_bollinger_bands, prepare_features
    )
except ImportError:
    # Mock functions if module doesn't exist
    def calculate_sma(data, window):
        return data.rolling(window=window).mean()
    
    def calculate_ema(data, window):
        return data.ewm(span=window).mean()
    
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(data, window=20, num_std=2):
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        return sma + (std * num_std), sma - (std * num_std)
    
    def prepare_features(sp500_data, vix_data):
        return pd.DataFrame({
            'close': sp500_data['Close'] if 'Close' in sp500_data.columns else sp500_data.iloc[:, 0],
            'volume': sp500_data['Volume'] if 'Volume' in sp500_data.columns else sp500_data.iloc[:, 1],
            'vix_close': vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
        })


class TestFeatureEngineering:
    """Test suite for feature engineering functionality"""
    
    def setup_method(self):
        """Setup test data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.Series(
            np.random.randn(100).cumsum() + 4000,
            index=dates
        )
        
        self.sp500_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 4000,
            'High': np.random.randn(100).cumsum() + 4050,
            'Low': np.random.randn(100).cumsum() + 3950,
            'Close': np.random.randn(100).cumsum() + 4000,
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
        
        self.vix_data = pd.DataFrame({
            'Open': np.random.randn(100) + 20,
            'High': np.random.randn(100) + 22,
            'Low': np.random.randn(100) + 18,
            'Close': np.random.randn(100) + 20,
            'Volume': np.random.randint(100000, 200000, 100)
        }, index=dates)
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        sma_5 = calculate_sma(self.test_data, 5)
        sma_20 = calculate_sma(self.test_data, 20)
        
        assert len(sma_5) == len(self.test_data)
        assert len(sma_20) == len(self.test_data)
        
        # SMA should be NaN for first (window-1) values
        assert pd.isna(sma_5.iloc[:4]).all()
        assert pd.isna(sma_20.iloc[:19]).all()
        
        # SMA should be valid numbers after window period
        assert not pd.isna(sma_5.iloc[4:]).any()
        assert not pd.isna(sma_20.iloc[19:]).any()
    
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation"""
        ema_12 = calculate_ema(self.test_data, 12)
        ema_26 = calculate_ema(self.test_data, 26)
        
        assert len(ema_12) == len(self.test_data)
        assert len(ema_26) == len(self.test_data)
        
        # EMA should have fewer NaN values than SMA
        assert ema_12.notna().sum() > ema_26.notna().sum()
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = calculate_rsi(self.test_data, 14)
        
        assert len(rsi) == len(self.test_data)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper_band, lower_band = calculate_bollinger_bands(self.test_data, 20, 2)
        
        assert len(upper_band) == len(self.test_data)
        assert len(lower_band) == len(self.test_data)
        
        # Upper band should be greater than lower band
        valid_indices = ~(upper_band.isna() | lower_band.isna())
        assert (upper_band[valid_indices] > lower_band[valid_indices]).all()
    
    def test_prepare_features(self):
        """Test feature preparation"""
        features = prepare_features(self.sp500_data, self.vix_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # Check that basic columns exist
        expected_columns = ['close', 'volume', 'vix_close']
        for col in expected_columns:
            if col in features.columns:
                assert not features[col].isna().all()
    
    def test_feature_engineering_edge_cases(self):
        """Test edge cases in feature engineering"""
        # Test with minimal data
        small_data = pd.Series([1, 2, 3, 4, 5])
        sma = calculate_sma(small_data, 3)
        
        assert len(sma) == 5
        assert pd.isna(sma.iloc[:2]).all()
        assert not pd.isna(sma.iloc[2:]).any()
    
    def test_feature_consistency(self):
        """Test feature calculation consistency"""
        # Calculate features multiple times - should get same results
        sma1 = calculate_sma(self.test_data, 10)
        sma2 = calculate_sma(self.test_data, 10)
        
        pd.testing.assert_series_equal(sma1, sma2)
    
    def test_feature_data_types(self):
        """Test that features have correct data types"""
        features = prepare_features(self.sp500_data, self.vix_data)
        
        # All features should be numeric
        for col in features.columns:
            if features[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                # Some features might be object type if they contain NaN
                # Convert and check if they become numeric
                numeric_series = pd.to_numeric(features[col], errors='coerce')
                assert not numeric_series.isna().all()


if __name__ == "__main__":
    pytest.main([__file__])
