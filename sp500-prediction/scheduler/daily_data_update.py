#!/usr/bin/env python3
"""
Daily Data Update Scheduler for S&P 500 Prediction System
Automates daily data collection, validation, and processing

Features:
- Fetches latest S&P 500 and VIX data from Yahoo Finance
- Validates data quality and completeness
- Updates processed features dataset
- Handles errors with proper logging and notifications
- Supports retry mechanisms for robustness
"""

import os
import sys
import logging
import smtplib
import traceback
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import yfinance as yf
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

try:
    from data_collection import SP500DataCollector
    from feature_engineering import FeatureEngineer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataUpdateError(Exception):
    """Custom exception for data update errors"""
    pass

class DailyDataUpdater:
    """Handles daily data collection and processing automation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Daily Data Updater
        
        Args:
            config: Configuration dictionary with settings
        """
        self.config = config or self._load_default_config()
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Initialize components
        self.data_collector = SP500DataCollector()
        self.feature_engineer = FeatureEngineer()
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Daily Data Updater initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            'max_retries': 3,
            'retry_delay': 60,  # seconds
            'data_validation_threshold': 0.95,  # 95% data completeness required
            'lookback_days': 5,  # Days to look back for validation
            'notification_email': os.getenv('NOTIFICATION_EMAIL'),
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_username': os.getenv('SMTP_USERNAME'),
            'smtp_password': os.getenv('SMTP_PASSWORD'),
            'symbols': {
                'sp500': '^GSPC',
                'vix': '^VIX'
            }
        }
    
    def validate_market_hours(self) -> bool:
        """
        Check if current time is appropriate for data updates
        
        Returns:
            bool: True if it's a good time to update data
        """
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            logger.info("Weekend detected, skipping data update")
            return False
        
        # Check if it's after market close (4 PM ET = 21:00 UTC approx)
        # This is a simplified check - in production, consider timezone handling
        if now.hour < 21:
            logger.info("Market may still be open, consider running after market close")
            # Don't skip, but log the warning
        
        return True
    
    def fetch_latest_data(self, symbol: str, days_back: int = 5) -> Optional[pd.DataFrame]:
        """
        Fetch latest data for a given symbol with retry mechanism
        
        Args:
            symbol: Yahoo Finance symbol (e.g., '^GSPC', '^VIX')
            days_back: Number of days to fetch (to ensure we get latest data)
            
        Returns:
            DataFrame with latest data or None if failed
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for attempt in range(self.config['max_retries']):
            try:
                logger.info(f"Fetching data for {symbol} (attempt {attempt + 1})")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )
                
                if data.empty:
                    raise DataUpdateError(f"No data returned for {symbol}")
                
                # Reset index to make date a column
                data.reset_index(inplace=True)
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt < self.config['max_retries'] - 1:
                    import time
                    time.sleep(self.config['retry_delay'])
                else:
                    logger.error(f"All attempts failed for {symbol}")
                    return None
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, str]:
        """
        Validate the quality of fetched data
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        try:
            # Check for required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            # Check for data completeness
            total_cells = len(data) * len(required_columns)
            non_null_cells = data[required_columns].count().sum()
            completeness = non_null_cells / total_cells
            
            if completeness < self.config['data_validation_threshold']:
                return False, f"Data completeness {completeness:.2%} below threshold {self.config['data_validation_threshold']:.2%}"
            
            # Check for reasonable price ranges (basic sanity check)
            if symbol == '^GSPC':  # S&P 500
                if data['close'].min() < 1000 or data['close'].max() > 10000:
                    return False, f"S&P 500 prices outside reasonable range: {data['close'].min():.2f} - {data['close'].max():.2f}"
            
            elif symbol == '^VIX':  # VIX
                if data['close'].min() < 0 or data['close'].max() > 100:
                    return False, f"VIX values outside reasonable range: {data['close'].min():.2f} - {data['close'].max():.2f}"
            
            # Check for duplicate dates
            if data['date'].duplicated().any():
                return False, "Duplicate dates found in data"
            
            # Check if we have recent data (within last 3 business days)
            latest_date = pd.to_datetime(data['date']).max().date()
            days_ago = (date.today() - latest_date).days
            
            if days_ago > 5:  # Allow some buffer for weekends/holidays
                return False, f"Latest data is {days_ago} days old (date: {latest_date})"
            
            logger.info(f"Data validation passed for {symbol}: {len(data)} rows, {completeness:.2%} complete")
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def update_raw_data(self) -> Dict[str, bool]:
        """
        Update raw data files with latest data
        
        Returns:
            Dictionary with update status for each symbol
        """
        results = {}
        
        for name, symbol in self.config['symbols'].items():
            logger.info(f"Updating raw data for {name} ({symbol})")
            
            try:
                # Fetch latest data
                new_data = self.fetch_latest_data(symbol)
                if new_data is None:
                    results[name] = False
                    continue
                
                # Validate data quality
                is_valid, message = self.validate_data_quality(new_data, symbol)
                if not is_valid:
                    logger.error(f"Data validation failed for {name}: {message}")
                    results[name] = False
                    continue
                
                # Load existing data if it exists
                raw_file = self.raw_dir / f"{name}_historical.csv"
                
                if raw_file.exists():
                    existing_data = pd.read_csv(raw_file)
                    existing_data['date'] = pd.to_datetime(existing_data['date'])
                    
                    # Remove any overlapping dates and append new data
                    new_data['date'] = pd.to_datetime(new_data['date'])
                    latest_existing_date = existing_data['date'].max()
                    
                    # Only add truly new data
                    new_rows = new_data[new_data['date'] > latest_existing_date]
                    
                    if len(new_rows) > 0:
                        updated_data = pd.concat([existing_data, new_rows], ignore_index=True)
                        logger.info(f"Added {len(new_rows)} new rows for {name}")
                    else:
                        updated_data = existing_data
                        logger.info(f"No new data to add for {name}")
                else:
                    updated_data = new_data
                    logger.info(f"Creating new raw data file for {name}")
                
                # Save updated data
                updated_data.to_csv(raw_file, index=False)
                logger.info(f"Successfully updated raw data for {name}")
                results[name] = True
                
            except Exception as e:
                logger.error(f"Error updating raw data for {name}: {str(e)}")
                results[name] = False
        
        return results
    
    def update_processed_features(self) -> bool:
        """
        Update processed features dataset using the feature engineering pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting feature engineering update")
            
            # Run the feature engineering pipeline
            engineered_data = self.feature_engineer.engineer_features()
            
            if engineered_data is None:
                logger.error("Feature engineering returned None")
                return False
            
            # Validate the engineered features
            if len(engineered_data) == 0:
                logger.error("Feature engineering returned empty dataset")
                return False
            
            # Check for required columns
            required_columns = ['date', 'target', 'close']
            missing_columns = [col for col in required_columns if col not in engineered_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns in engineered data: {missing_columns}")
                return False
            
            logger.info(f"Successfully updated processed features: {engineered_data.shape}")
            logger.info(f"Latest date in processed data: {engineered_data['date'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating processed features: {str(e)}")
            traceback.print_exc()
            return False
    
    def send_notification(self, subject: str, message: str, is_error: bool = False):
        """
        Send email notification about update status
        
        Args:
            subject: Email subject
            message: Email message body
            is_error: Whether this is an error notification
        """
        if not self.config.get('notification_email'):
            logger.info("No notification email configured, skipping notification")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['smtp_username']
            msg['To'] = self.config['notification_email']
            msg['Subject'] = f"[S&P500 Predictor] {subject}"
            
            # Add timestamp and system info
            full_message = f"""
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'ERROR' if is_error else 'SUCCESS'}

{message}

---
S&P 500 Prediction System
Daily Data Update Service
"""
            
            msg.attach(MIMEText(full_message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_username'], self.config['smtp_password'])
            text = msg.as_string()
            server.sendmail(self.config['smtp_username'], self.config['notification_email'], text)
            server.quit()
            
            logger.info("Notification email sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification email: {str(e)}")
    
    def run_daily_update(self) -> bool:
        """
        Run the complete daily update process
        
        Returns:
            bool: True if successful, False if any critical errors occurred
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting daily data update at {start_time}")
        logger.info("=" * 60)
        
        success = True
        error_messages = []
        
        try:
            # Check if it's appropriate time to run
            if not self.validate_market_hours():
                logger.info("Skipping update due to market hours/weekend")
                return True
            
            # Step 1: Update raw data
            logger.info("Step 1: Updating raw data")
            raw_results = self.update_raw_data()
            
            failed_symbols = [name for name, result in raw_results.items() if not result]
            if failed_symbols:
                error_msg = f"Failed to update raw data for: {', '.join(failed_symbols)}"
                logger.error(error_msg)
                error_messages.append(error_msg)
                success = False
            
            # Step 2: Update processed features (only if raw data update was successful)
            if success or len(failed_symbols) == 0:  # Allow partial success
                logger.info("Step 2: Updating processed features")
                if not self.update_processed_features():
                    error_msg = "Failed to update processed features"
                    logger.error(error_msg)
                    error_messages.append(error_msg)
                    success = False
            else:
                logger.warning("Skipping feature engineering due to raw data failures")
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Send notification
            if success:
                message = f"""Daily data update completed successfully!

Execution time: {execution_time:.1f} seconds
Raw data updates: {raw_results}

All systems operational.
"""
                self.send_notification("Daily Update Successful", message, is_error=False)
                logger.info(f"Daily update completed successfully in {execution_time:.1f} seconds")
            else:
                message = f"""Daily data update completed with errors!

Execution time: {execution_time:.1f} seconds
Raw data updates: {raw_results}

Errors encountered:
{chr(10).join(error_messages)}

Please check the logs for more details.
"""
                self.send_notification("Daily Update Failed", message, is_error=True)
                logger.error(f"Daily update completed with errors in {execution_time:.1f} seconds")
            
            return success
            
        except Exception as e:
            error_msg = f"Critical error in daily update: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            
            self.send_notification(
                "Daily Update Critical Error",
                f"{error_msg}\n\nStacktrace:\n{traceback.format_exc()}",
                is_error=True
            )
            
            return False
        
        finally:
            logger.info("=" * 60)
            logger.info(f"Daily data update finished at {datetime.now()}")
            logger.info("=" * 60)

def main():
    """Main function for command line execution"""
    
    # Parse command line arguments (simple implementation)
    import argparse
    parser = argparse.ArgumentParser(description='Daily Data Update for S&P 500 Prediction System')
    parser.add_argument('--force', action='store_true', help='Force update regardless of market hours')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config if provided
    config = None
    if args.config:
        import json
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)
    
    # Create updater and run
    try:
        updater = DailyDataUpdater(config=config)
        
        # Override market hours check if forced
        if args.force:
            logger.info("Forcing update regardless of market hours")
            updater.validate_market_hours = lambda: True
        
        success = updater.run_daily_update()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run daily update: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()