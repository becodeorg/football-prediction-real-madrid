#!/usr/bin/env python3
"""
Test script for the enhanced S&P 500 prediction dashboard
This script tests all the core functionality before running the Streamlit app
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

def test_model_loading():
    """Test model loading with correct paths"""
    print("1. Testing model loading...")
    
    try:
        from predict import SP500Predictor
        
        base_dir = os.getcwd()
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        success = predictor.load_models()
        
        if success:
            print(f"   ✓ Models loaded successfully")
            print(f"   ✓ {len(predictor.models)} models: {list(predictor.models.keys())}")
            print(f"   ✓ {len(predictor.feature_names)} features")
            return True
        else:
            print("   ✗ Failed to load models")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_data_collection():
    """Test data collection functionality"""
    print("2. Testing data collection...")
    
    try:
        from data_collection import SP500DataCollector
        
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, 'data')
        
        collector = SP500DataCollector(data_dir=data_dir)
        sp500_data = collector.fetch_sp500_data(period='1mo')
        vix_data = collector.fetch_vix_data(period='1mo')
        
        if sp500_data is not None and vix_data is not None:
            print(f"   ✓ Data fetched successfully")
            print(f"   ✓ S&P 500: {len(sp500_data)} records")
            print(f"   ✓ VIX: {len(vix_data)} records")
            print(f"   ✓ Latest price: ${sp500_data['close'].iloc[-1]:.2f}")
            return True, sp500_data, vix_data
        else:
            print("   ✗ Failed to fetch data")
            return False, None, None
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False, None, None

def test_prediction():
    """Test prediction functionality"""
    print("3. Testing prediction generation...")
    
    try:
        from predict import SP500Predictor
        
        base_dir = os.getcwd()
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        prediction = predictor.make_prediction()
        
        if prediction:
            print(f"   ✓ Prediction generated successfully")
            print(f"   ✓ Direction: {prediction['prediction_text']}")
            print(f"   ✓ Confidence: {prediction['ensemble_confidence']:.1%}")
            print(f"   ✓ Current Price: ${prediction['current_price']:.2f}")
            print(f"   ✓ Date: {prediction['date']}")
            return True, prediction
        else:
            print("   ✗ Failed to generate prediction")
            return False, None
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False, None

def test_future_predictions():
    """Test future predictions functionality"""
    print("4. Testing future predictions...")
    
    try:
        # Simulate the same logic as in streamlit app
        from predict import SP500Predictor
        import pandas as pd
        import numpy as np
        
        base_dir = os.getcwd()
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        
        predictor = SP500Predictor(models_dir=models_dir, data_dir=data_dir)
        
        # Load models and get latest data
        if not predictor.load_models():
            print("   ✗ Failed to load models for future predictions")
            return False
            
        sp500_data, vix_data = predictor.get_latest_market_data()
        if sp500_data is None or vix_data is None:
            print("   ✗ Failed to get market data for future predictions")
            return False
        
        print("   ✓ Models and data loaded for future predictions")
        
        # Test generating 3 days of predictions
        predictions_list = []
        current_data = sp500_data.copy()
        
        for day in range(1, 4):  # Test 3 days
            features_data = predictor.engineer_features_for_prediction(current_data, vix_data)
            if features_data is None:
                break
                
            latest_data = features_data.iloc[-1:].copy()
            
            # Prepare features
            feature_columns = []
            for feature in predictor.feature_names:
                if feature in latest_data.columns:
                    feature_columns.append(feature)
            
            X = latest_data[feature_columns].copy()
            
            # Handle missing features
            if len(feature_columns) < len(predictor.feature_names):
                missing_features = set(predictor.feature_names) - set(feature_columns)
                for feature in missing_features:
                    X[feature] = 0
            
            X = X[predictor.feature_names]
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Scale and predict
            X_scaled = predictor.scaler.transform(X.values)
            
            predictions = {}
            for name, model in predictor.models.items():
                pred_proba = model.predict_proba(X_scaled)[0]
                pred_class = model.predict(X_scaled)[0]
                
                predictions[name] = {
                    'prediction': int(pred_class),
                    'confidence': float(max(pred_proba))
                }
            
            ensemble_prediction = int(np.round(np.mean([pred['prediction'] for pred in predictions.values()])))
            ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            
            # Create future date
            last_date = pd.to_datetime(current_data['date'].iloc[-1])
            future_date = last_date + pd.Timedelta(days=day)
            
            # Skip weekends
            while future_date.weekday() >= 5:
                future_date += pd.Timedelta(days=1)
            
            # Estimate price
            current_price = float(current_data['close'].iloc[-1])
            volatility = current_data['close'].pct_change().std() * np.sqrt(252)
            
            if ensemble_prediction == 1:
                price_change = np.random.normal(0.001, volatility/252)
            else:
                price_change = np.random.normal(-0.001, volatility/252)
            
            estimated_price = current_price * (1 + price_change)
            
            prediction_result = {
                'day': day,
                'date': future_date,
                'estimated_price': estimated_price,
                'direction': 'UP' if ensemble_prediction == 1 else 'DOWN',
                'ensemble_confidence': float(ensemble_confidence)
            }
            
            predictions_list.append(prediction_result)
            
            # Update data for next iteration
            new_row = current_data.iloc[-1:].copy()
            new_row['date'] = future_date
            new_row['close'] = estimated_price
            new_row['open'] = current_price
            new_row['high'] = max(current_price, estimated_price) * 1.01
            new_row['low'] = min(current_price, estimated_price) * 0.99
            new_row['volume'] = current_data['volume'].iloc[-1]
            
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        if predictions_list:
            print(f"   ✓ Generated {len(predictions_list)} future predictions")
            for pred in predictions_list:
                print(f"     Day {pred['day']}: {pred['direction']} ({pred['ensemble_confidence']:.1%}) - ${pred['estimated_price']:.2f}")
            return True, predictions_list
        else:
            print("   ✗ Failed to generate future predictions")
            return False, None
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False, None

def test_streamlit_compatibility():
    """Test if required modules can be imported for Streamlit"""
    print("5. Testing Streamlit compatibility...")
    
    try:
        import streamlit
        import plotly.graph_objects
        import plotly.subplots
        print("   ✓ Streamlit and Plotly available")
        return True
    except ImportError as e:
        print(f"   ✗ Missing required package: {e}")
        print("   → Run: pip install streamlit plotly")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("S&P 500 Dashboard Functionality Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Model loading
    if not test_model_loading():
        all_tests_passed = False
    
    # Test 2: Data collection
    data_success, sp500_data, vix_data = test_data_collection()
    if not data_success:
        all_tests_passed = False
    
    # Test 3: Prediction
    pred_success, prediction = test_prediction()
    if not pred_success:
        all_tests_passed = False
    
    # Test 4: Future predictions
    future_success, future_preds = test_future_predictions()
    if not future_success:
        all_tests_passed = False
    
    # Test 5: Streamlit compatibility
    if not test_streamlit_compatibility():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Dashboard is ready to run.")
        print("\nTo start the dashboard:")
        print("streamlit run app/streamlit_app.py")
        print("\nThe dashboard will be available at: http://localhost:8501")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Make sure you're in the sp500-prediction directory")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check that models are trained: python src/train_model.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
