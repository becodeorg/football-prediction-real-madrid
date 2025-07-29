"""
Prediction Module for S&P 500 Prediction System
Responsible for making predictions using trained models
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import yfinance as yf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500Predictor:
    def __init__(self, models_dir="../models", data_dir="../data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.models = {}
        self.scaler = None
        self.feature_names = []
        
    def load_models(self):
        """
        Load trained models and scaler
        
        Returns:
            bool: Success status
        """
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            self.scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
            
            # Load feature names
            feature_names_path = os.path.join(self.models_dir, "feature_names.pkl")
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            # Load models
            model_files = ['random_forest.pkl', 'gradient_boosting.pkl', 'logistic_regression.pkl']
            
            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                if os.path.exists(model_path):
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_latest_market_data(self):
        """
        Get the latest market data for prediction
        
        Returns:
            pd.DataFrame: Latest market data
        """
        try:
            logger.info("Fetching latest market data...")
            
            # Get S&P 500 data
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1mo", interval="1d")
            
            # Get VIX data
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1mo", interval="1d")
            
            # Clean S&P 500 data
            sp500_data.columns = sp500_data.columns.str.lower()
            sp500_data['date'] = sp500_data.index
            sp500_data.reset_index(drop=True, inplace=True)
            
            # Clean VIX data
            vix_data.columns = ['vix_' + col.lower() for col in vix_data.columns]
            vix_data['date'] = vix_data.index
            vix_data.reset_index(drop=True, inplace=True)
            
            logger.info(f"Retrieved {len(sp500_data)} days of S&P 500 data")
            logger.info(f"Retrieved {len(vix_data)} days of VIX data")
            
            return sp500_data, vix_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None, None
    
    def engineer_features_for_prediction(self, sp500_data, vix_data):
        """
        Engineer features for the latest data (similar to training pipeline)
        
        Args:
            sp500_data (pd.DataFrame): S&P 500 data
            vix_data (pd.DataFrame): VIX data
            
        Returns:
            pd.DataFrame: Engineered features
        """
        try:
            data = sp500_data.copy()
            
            # Basic price data
            high = data['high']
            low = data['low']
            close = data['close']
            volume = data['volume']
            
            # Moving Averages
            data['sma_5'] = close.rolling(window=5).mean()
            data['sma_10'] = close.rolling(window=10).mean()
            data['sma_20'] = close.rolling(window=20).mean()
            data['sma_50'] = close.rolling(window=50, min_periods=1).mean()
            data['sma_200'] = close.rolling(window=200, min_periods=1).mean()
            
            # Exponential Moving Averages
            data['ema_12'] = close.ewm(span=12).mean()
            data['ema_26'] = close.ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_diff'] = data['macd'] - data['macd_signal']
            
            # RSI
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
            
            # Volatility
            data['volatility_5d'] = data['return_1d'].rolling(window=5).std()
            data['volatility_20d'] = data['return_1d'].rolling(window=20).std()
            
            # Lag features
            lag_periods = [1, 2, 3, 5]
            features_to_lag = ['close', 'volume', 'rsi', 'macd', 'return_1d', 'volatility_5d']
            
            for feature in features_to_lag:
                if feature in data.columns:
                    for lag in lag_periods:
                        data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
            
            # Merge with VIX data
            vix_features = vix_data[['date', 'vix_close', 'vix_volume']].copy()
            vix_features['vix_change'] = vix_features['vix_close'].pct_change()
            
            merged_data = pd.merge(data, vix_features, on='date', how='left')
            
            # Fill NaN values
            merged_data = merged_data.ffill().bfill()
            
            logger.info(f"Engineered features for {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return None
    
    def make_prediction(self, model_name='logistic_regression'):
        """
        Make prediction for the next trading day
        
        Args:
            model_name (str): Name of the model to use
            
        Returns:
            dict: Prediction results
        """
        try:
            # Load models if not already loaded
            if not self.models:
                success = self.load_models()
                if not success:
                    return None
            
            # Get latest market data
            sp500_data, vix_data = self.get_latest_market_data()
            if sp500_data is None or vix_data is None:
                return None
            
            # Engineer features
            features_data = self.engineer_features_for_prediction(sp500_data, vix_data)
            if features_data is None:
                return None
            
            # Get the latest record for prediction
            latest_data = features_data.iloc[-1:].copy()
            
            # Select only the features used in training
            feature_columns = []
            for feature in self.feature_names:
                if feature in latest_data.columns:
                    feature_columns.append(feature)
            
            # Prepare feature vector
            X = latest_data[feature_columns].copy()
            
            # Handle any missing features by filling with 0
            if len(feature_columns) < len(self.feature_names):
                missing_features = set(self.feature_names) - set(feature_columns)
                logger.warning(f"Missing features: {missing_features}")
                
                # Add missing features with 0 values
                for feature in missing_features:
                    X[feature] = 0
            
            # Reorder columns to match training order
            X = X[self.feature_names]
            
            # Fill any remaining NaN values with 0
            X = X.fillna(0)
            
            # Replace infinite values with 0
            X = X.replace([np.inf, -np.inf], 0)
            
            # Convert to numpy array
            X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions with all models
            predictions = {}
            
            for name, model in self.models.items():
                pred_proba = model.predict_proba(X_scaled)[0]
                pred_class = model.predict(X_scaled)[0]
                
                predictions[name] = {
                    'prediction': int(pred_class),
                    'probability_down': float(pred_proba[0]),
                    'probability_up': float(pred_proba[1]),
                    'confidence': float(max(pred_proba))
                }
            
            # Ensemble prediction (majority vote)
            ensemble_votes = [pred['prediction'] for pred in predictions.values()]
            ensemble_prediction = int(np.round(np.mean(ensemble_votes)))
            ensemble_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            
            # Current market data
            current_price = float(sp500_data['close'].iloc[-1])
            current_date = sp500_data['date'].iloc[-1]
            
            result = {
                'date': current_date.strftime('%Y-%m-%d'),
                'current_price': current_price,
                'individual_predictions': predictions,
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': float(ensemble_confidence),
                'prediction_text': 'UP' if ensemble_prediction == 1 else 'DOWN',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Prediction completed: {result['prediction_text']} with {ensemble_confidence:.2%} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_prediction_summary(self):
        """
        Get a formatted prediction summary
        
        Returns:
            str: Formatted prediction summary
        """
        prediction = self.make_prediction()
        
        if prediction is None:
            return "Error: Could not generate prediction"
        
        summary = f"""
S&P 500 Prediction Summary
========================
Date: {prediction['date']}
Current Price: ${prediction['current_price']:.2f}

PREDICTION: {prediction['prediction_text']}
Ensemble Confidence: {prediction['ensemble_confidence']:.1%}

Individual Model Predictions:
"""
        
        for model_name, pred in prediction['individual_predictions'].items():
            direction = "UP" if pred['prediction'] == 1 else "DOWN"
            summary += f"  {model_name}: {direction} ({pred['confidence']:.1%} confidence)\n"
        
        summary += f"\nGenerated: {prediction['timestamp']}"
        
        return summary

def main():
    """
    Main function to test prediction
    """
    predictor = SP500Predictor()
    
    # Get prediction summary
    summary = predictor.get_prediction_summary()
    print(summary)
    
    # Get detailed prediction
    prediction = predictor.make_prediction()
    if prediction:
        print(f"\nDetailed prediction data:")
        for key, value in prediction.items():
            if key != 'individual_predictions':
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()
