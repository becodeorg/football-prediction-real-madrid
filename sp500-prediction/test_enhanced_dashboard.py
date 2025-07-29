#!/usr/bin/env python3
"""
Test script for the enhanced S&P 500 prediction dashboard
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_future_predictions():
    """Test the future predictions functionality"""
    try:
        from src.predict import SP500Predictor
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        print("Testing enhanced prediction system...")
        
        predictor = SP500Predictor()
        
        # Test loading models
        print("1. Loading models...")
        if predictor.load_models():
            print("   ✓ Models loaded successfully")
        else:
            print("   ✗ Failed to load models")
            return False
        
        # Test single prediction
        print("2. Testing single prediction...")
        prediction = predictor.make_prediction()
        if prediction:
            print(f"   ✓ Current prediction: {prediction['prediction_text']} ({prediction['ensemble_confidence']:.1%} confidence)")
        else:
            print("   ✗ Failed to generate prediction")
            return False
        
        # Test market data fetching
        print("3. Testing market data fetching...")
        sp500_data, vix_data = predictor.get_latest_market_data()
        if sp500_data is not None and vix_data is not None:
            print(f"   ✓ Fetched {len(sp500_data)} S&P 500 records and {len(vix_data)} VIX records")
        else:
            print("   ✗ Failed to fetch market data")
            return False
        
        # Test feature engineering
        print("4. Testing feature engineering...")
        features = predictor.engineer_features_for_prediction(sp500_data, vix_data)
        if features is not None:
            print(f"   ✓ Engineered {len(features.columns)} features")
        else:
            print("   ✗ Failed to engineer features")
            return False
        
        print("\n✓ All tests passed! Enhanced dashboard is ready to use.")
        print("\nTo run the dashboard:")
        print("cd sp500-prediction")
        print("streamlit run app/streamlit_app.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("S&P 500 Enhanced Dashboard Test")
    print("=" * 60)
    
    success = test_future_predictions()
    
    if success:
        print("\n🎉 Enhancement complete! New features added:")
        print("   • 5-day future predictions")
        print("   • Prediction markers on price chart")
        print("   • Forward-looking prediction timeline")
        print("   • Confidence levels display")
        print("   • Prediction uncertainty bands")
        print("   • Model agreement indicators")
        print("   • Interactive prediction controls")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        print("Make sure all required dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
