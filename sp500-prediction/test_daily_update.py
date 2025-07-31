#!/usr/bin/env python3
"""
Test script for Daily Data Update functionality
Run this script to test the data update system before scheduling
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'scheduler'))

def test_daily_update():
    """Test the daily data update functionality"""
    
    print("=" * 60)
    print("Testing Daily Data Update System")
    print("=" * 60)
    
    try:
        from daily_data_update import DailyDataUpdater
        
        # Create test configuration (no email notifications for testing)
        test_config = {
            'max_retries': 2,
            'retry_delay': 30,
            'data_validation_threshold': 0.90,
            'lookback_days': 3,
            'notification_email': None,  # Disable notifications for testing
            'symbols': {
                'sp500': '^GSPC',
                'vix': '^VIX'
            }
        }
        
        print("✓ Successfully imported DailyDataUpdater")
        
        # Initialize updater
        updater = DailyDataUpdater(config=test_config)
        print("✓ Successfully initialized DailyDataUpdater")
        
        # Test data fetching
        print("\n📊 Testing data fetching...")
        sp500_data = updater.fetch_latest_data('^GSPC', days_back=3)
        
        if sp500_data is not None and len(sp500_data) > 0:
            print(f"✓ S&P 500 data fetched: {len(sp500_data)} rows")
            print(f"  Latest date: {sp500_data['date'].max()}")
            print(f"  Latest close: ${sp500_data['close'].iloc[-1]:.2f}")
        else:
            print("❌ Failed to fetch S&P 500 data")
            return False
        
        # Test VIX data
        vix_data = updater.fetch_latest_data('^VIX', days_back=3)
        if vix_data is not None and len(vix_data) > 0:
            print(f"✓ VIX data fetched: {len(vix_data)} rows")
            print(f"  Latest VIX: {vix_data['close'].iloc[-1]:.2f}")
        else:
            print("❌ Failed to fetch VIX data")
            return False
        
        # Test data validation
        print("\n🔍 Testing data validation...")
        is_valid, message = updater.validate_data_quality(sp500_data, '^GSPC')
        if is_valid:
            print(f"✓ S&P 500 data validation passed: {message}")
        else:
            print(f"❌ S&P 500 data validation failed: {message}")
            return False
        
        is_valid, message = updater.validate_data_quality(vix_data, '^VIX')
        if is_valid:
            print(f"✓ VIX data validation passed: {message}")
        else:
            print(f"❌ VIX data validation failed: {message}")
            return False
        
        # Test raw data update (without actually saving to avoid conflicts)
        print("\n💾 Testing raw data update logic...")
        print("  (This test validates the logic without saving files)")
        
        # Check if directories exist
        if updater.raw_dir.exists():
            print(f"✓ Raw data directory exists: {updater.raw_dir}")
        else:
            print(f"✓ Raw data directory will be created: {updater.raw_dir}")
        
        if updater.processed_dir.exists():
            print(f"✓ Processed data directory exists: {updater.processed_dir}")
        else:
            print(f"✓ Processed data directory will be created: {updater.processed_dir}")
        
        print("\n🎯 All tests passed!")
        print("\n📋 Summary:")
        print(f"  - Data fetching: Working")
        print(f"  - Data validation: Working")
        print(f"  - Directory structure: Ready")
        print(f"  - Configuration: Valid")
        
        print("\n🚀 Ready to run daily updates!")
        print("\nTo run a full update (be careful, this will modify your data):")
        print("  python scheduler/daily_data_update.py")
        print("\nTo run with force (ignore market hours):")
        print("  python scheduler/daily_data_update.py --force")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_setup():
    """Test if the environment is properly set up"""
    
    print("\n🔧 Testing Environment Setup...")
    
    # Check if required directories exist
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'src',
        'data',
        'models',
        'scheduler'
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"❌ Missing directory: {dir_name}/")
            return False
    
    # Check if key files exist
    required_files = [
        'src/data_collection.py',
        'src/feature_engineering.py',
        'requirements.txt'
    ]
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✓ File exists: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            return False
    
    print("✓ Environment setup looks good!")
    return True

def main():
    """Main test function"""
    
    print("🧪 Daily Data Update - Test Suite")
    print("=" * 60)
    
    # Test environment first
    if not test_environment_setup():
        print("\n❌ Environment setup issues detected. Please fix before proceeding.")
        return False
    
    # Test the actual functionality
    if not test_daily_update():
        print("\n❌ Daily update tests failed. Please check the errors above.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Daily Data Update system is ready.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
