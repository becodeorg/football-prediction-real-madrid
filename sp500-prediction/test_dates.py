#!/usr/bin/env python3
"""
Test the future predictions date handling
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append('src')

def test_future_predictions_dates():
    """Test if future predictions are generating correct 2025 dates"""
    try:
        from predict import SP500Predictor
        
        print("Testing future predictions date handling...")
        
        base_dir = os.getcwd()
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        
        # Load models and get latest data
        if not predictor.load_models():
            print("❌ Failed to load models")
            return False
            
        sp500_data, vix_data = predictor.get_latest_market_data()
        if sp500_data is None or vix_data is None:
            print("❌ Failed to get market data")
            return False
        
        print(f"✅ Latest market data fetched")
        print(f"   Latest S&P 500 date: {sp500_data['date'].iloc[-1]}")
        print(f"   Latest S&P 500 price: ${sp500_data['close'].iloc[-1]:.2f}")
        
        # Test the date calculation logic used in the app
        current_data = sp500_data.copy()
        predictions_list = []
        
        for day in range(1, 6):  # 5 days
            # This mimics the logic in get_future_predictions
            last_date = pd.to_datetime(current_data['date'].iloc[-1])
            
            # Ensure we're working with a naive datetime (remove timezone if present)
            if hasattr(last_date, 'tz') and last_date.tz is not None:
                last_date = last_date.tz_localize(None)
            
            # Calculate future date
            future_date = last_date + pd.Timedelta(days=day)
            
            # Skip weekends
            while future_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                future_date += pd.Timedelta(days=1)
            
            prediction_result = {
                'day': day,
                'date': future_date,
                'direction': 'UP',  # Dummy data for test
                'ensemble_confidence': 0.6  # Dummy data for test
            }
            
            predictions_list.append(prediction_result)
            print(f"   Day {day}: {future_date.strftime('%Y-%m-%d')} (weekday: {future_date.weekday()})")
        
        # Check if all dates are in 2025
        all_2025 = all(pred['date'].year == 2025 for pred in predictions_list)
        
        if all_2025:
            print("✅ All future prediction dates are correctly in 2025!")
            return True
        else:
            print("❌ Some future prediction dates are not in 2025!")
            for pred in predictions_list:
                print(f"   Day {pred['day']}: {pred['date']} (year: {pred['date'].year})")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    print("=" * 60)
    print("Future Predictions Date Test")
    print("=" * 60)
    
    success = test_future_predictions_dates()
    
    if success:
        print("\n🎉 Date handling is working correctly!")
        print("The 5-day prediction timeline should now show 2025 dates.")
    else:
        print("\n❌ Date handling issue detected.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
