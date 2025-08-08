"""
Data fetching module for S&P 500 and other financial instruments
Uses yfinance to fetch real-time and historical data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class DataFetcher:
    """
    A class to fetch financial data using yfinance
    """
    
    def __init__(self):
        self.default_symbols = {
            'SP500': '^GSPC',  # S&P 500 Index
            'SPY': 'SPY',      # S&P 500 ETF
            'VOO': 'VOO',      # Vanguard S&P 500 ETF
            'NASDAQ': '^IXIC', # NASDAQ Composite
            'DOW': '^DJI'      # Dow Jones Industrial Average
        }
    
    def get_historical_data(
        self, 
        symbol: str = '^GSPC', 
        period: str = '2y',
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical data for a given symbol
        
        Args:
            symbol (str): Stock symbol (default: ^GSPC for S&P 500)
            period (str): Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            interval (str): Data interval (1m,2m,5m,15m,30m,60m,90m,1h,4h,1d,5d,1wk,1mo,3mo)
            Note: 1y interval is not supported by yfinance, use get_yearly_data() method instead
        
        Returns:
            pd.DataFrame: Historical data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Add additional columns
            data['symbol'] = symbol
            data['date'] = data.index
            
            print(f"✅ Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str = '^GSPC') -> Dict:
        """
        Fetch real-time data for a given symbol
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Dict: Real-time data including current price, change, volume, etc.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period='1d', interval='1m')
            
            if history.empty:
                raise ValueError(f"No real-time data available for {symbol}")
            
            latest = history.iloc[-1]
            
            real_time_data = {
                'symbol': symbol,
                'current_price': latest['Close'],
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'volume': latest['Volume'],
                'timestamp': latest.name,
                'previous_close': info.get('previousClose', None),
                'change': latest['Close'] - info.get('previousClose', latest['Close']),
                'change_percent': ((latest['Close'] - info.get('previousClose', latest['Close'])) / 
                                 info.get('previousClose', latest['Close'])) * 100 if info.get('previousClose') else 0,
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None)
            }
            
            print(f"✅ Successfully fetched real-time data for {symbol}")
            return real_time_data
            
        except Exception as e:
            print(f"❌ Error fetching real-time data for {symbol}: {str(e)}")
            return {}
    
    def get_multiple_symbols(
        self, 
        symbols: List[str], 
        period: str = '1y',
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data_dict[symbol] = self.get_historical_data(symbol, period, interval)
        
        return data_dict
    
    def get_sp500_components(self, top_n: int = 50) -> List[str]:
        """
        Get S&P 500 component symbols
        
        Args:
            top_n (int): Number of top components to return
        
        Returns:
            List[str]: List of S&P 500 component symbols
        """
        try:
            # This is a simplified list of major S&P 500 components
            # In a real application, you might want to scrape this from Wikipedia or other sources
            sp500_major_components = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'PFE', 'ABBV',
                'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'MRK', 'WMT', 'DHR',
                'VZ', 'ABT', 'ADBE', 'ACN', 'TXN', 'LLY', 'NEE', 'BMY', 'PM',
                'T', 'MDT', 'HON', 'ORCL', 'UPS', 'LOW', 'AMT', 'QCOM', 'IBM',
                'NKE', 'CVS', 'MMM', 'BA'
            ]
            
            return sp500_major_components[:top_n]
            
        except Exception as e:
            print(f"❌ Error getting S&P 500 components: {str(e)}")
            return []
    
    def get_intraday_data(
        self, 
        symbol: str = '^GSPC', 
        interval: str = '5m',
        days_back: int = 1
    ) -> pd.DataFrame:
        """
        Fetch intraday data for real-time analysis
        
        Args:
            symbol (str): Stock symbol
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m)
            days_back (int): Number of days to go back
        
        Returns:
            pd.DataFrame: Intraday data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Calculate the period string
            if days_back <= 7:
                period = f"{days_back}d"
            else:
                period = "1mo"
            
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No intraday data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            data['symbol'] = symbol
            
            print(f"✅ Successfully fetched {len(data)} intraday records for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data
        
        Args:
            symbol (str): Stock symbol to validate
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d')
            return not data.empty
        except:
            return False
    
    def get_yearly_data(
        self, 
        symbol: str = '^GSPC', 
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get yearly aggregated data (since yfinance doesn't support 1y interval)
        Aggregates from monthly data to create yearly OHLCV data
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
        
        Returns:
            pd.DataFrame: Yearly aggregated OHLCV data
        """
        try:
            # Default to 20 years of data if no dates provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=7300)  # ~20 years
            if end_date is None:
                end_date = datetime.now()
            
            # Get monthly data first
            monthly_data = self.get_historical_data_by_dates(symbol, start_date, end_date, '1mo')
            
            if monthly_data.empty:
                return pd.DataFrame()
            
            # Resample to yearly data
            yearly_data = monthly_data.resample('1YE').agg({
                'open': 'first',      # First open of the year
                'high': 'max',        # Highest high of the year
                'low': 'min',         # Lowest low of the year
                'close': 'last',      # Last close of the year
                'volume': 'sum',      # Total volume for the year
                'symbol': 'first'
            }).dropna()
            
            print(f"✅ Successfully created {len(yearly_data)} yearly records for {symbol}")
            return yearly_data
            
        except Exception as e:
            print(f"❌ Error creating yearly data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_data_by_dates(
        self, 
        symbol: str = '^GSPC', 
        start_date: datetime = None,
        end_date: datetime = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical data by specific date range
        
        Args:
            symbol (str): Stock symbol
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: Historical data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Default to 5 years if no dates provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=1825)  # ~5 years
            if end_date is None:
                end_date = datetime.now()
            
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            
            # Add additional columns
            data['symbol'] = symbol
            data['date'] = data.index
            
            print(f"✅ Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_basic_info(self, symbol: str = '^GSPC') -> Dict:
        """
        Get basic information about a symbol
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            Dict: Basic information about the symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            basic_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'currency': info.get('currency', 'USD')
            }
            
            return basic_info
            
        except Exception as e:
            print(f"❌ Error getting basic info for {symbol}: {str(e)}")
            return {}


def main():
    """
    Example usage of the DataFetcher class
    """
    fetcher = DataFetcher()
    
    print("🚀 Testing S&P 500 Data Fetcher")
    print("=" * 50)
    
    # Test historical data
    print("\n📊 Fetching S&P 500 historical data...")
    sp500_data = fetcher.get_historical_data('^GSPC', period='1y')
    if not sp500_data.empty:
        print(f"Data shape: {sp500_data.shape}")
        print(f"Date range: {sp500_data.index.min()} to {sp500_data.index.max()}")
        print(f"Latest close price: ${sp500_data['close'].iloc[-1]:.2f}")
    
    # Test real-time data
    print("\n⏱️ Fetching real-time data...")
    real_time = fetcher.get_real_time_data('^GSPC')
    if real_time:
        print(f"Current price: ${real_time['current_price']:.2f}")
        print(f"Change: {real_time['change']:+.2f} ({real_time['change_percent']:+.2f}%)")
    
    # Test basic info
    print("\n📋 Getting basic info...")
    info = fetcher.get_basic_info('^GSPC')
    if info:
        print(f"Name: {info['name']}")
        print(f"Market Cap: {info['market_cap']}")
    
    print("\n✅ Data fetcher testing completed!")


if __name__ == "__main__":
    main()
