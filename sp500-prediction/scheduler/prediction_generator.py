#!/usr/bin/env python3
"""
Prediction Generator for S&P 500 Prediction System
Automates daily prediction generation and dashboard data preparation

Features:
- Generates daily S&P 500 predictions using trained models
- Maintains prediction history and cache
- Prepares data for dashboard consumption
- Validates prediction quality and consistency
- Handles model loading and prediction pipeline
- Supports multiple prediction horizons
"""

import os
import sys
import json
import logging
import traceback
import sqlite3
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

try:
    from predict import SP500Predictor
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
        logging.FileHandler(log_dir / f"prediction_generator_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class PredictionDatabase:
    """Handles prediction storage and retrieval"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the prediction database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_date TEXT NOT NULL,
                        target_date TEXT NOT NULL,
                        prediction_type TEXT NOT NULL,
                        prediction_value REAL NOT NULL,
                        confidence REAL,
                        model_used TEXT,
                        actual_value REAL,
                        created_at TEXT NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                # Create model performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Create indices for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_date 
                    ON predictions(prediction_date, target_date)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_performance_date 
                    ON model_performance(date, model_name)
                ''')
                
                conn.commit()
                logger.info("Prediction database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise PredictionError(f"Database initialization failed: {str(e)}")
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save a prediction to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO predictions 
                    (prediction_date, target_date, prediction_type, prediction_value, 
                     confidence, model_used, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_data['prediction_date'],
                    prediction_data['target_date'],
                    prediction_data['prediction_type'],
                    prediction_data['prediction_value'],
                    prediction_data.get('confidence'),
                    prediction_data.get('model_used'),
                    datetime.now().isoformat(),
                    json.dumps(prediction_data.get('metadata', {}))
                ))
                
                conn.commit()
                logger.debug(f"Saved prediction for {prediction_data['target_date']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save prediction: {str(e)}")
            return False
    
    def get_recent_predictions(self, days: int = 30) -> pd.DataFrame:
        """Get recent predictions from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM predictions 
                    WHERE prediction_date >= date('now', '-{} days')
                    ORDER BY prediction_date DESC, created_at DESC
                '''.format(days)
                
                df = pd.read_sql_query(query, conn)
                logger.debug(f"Retrieved {len(df)} recent predictions")
                return df
                
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {str(e)}")
            return pd.DataFrame()
    
    def update_actual_values(self, date_str: str, actual_value: float) -> bool:
        """Update actual values for predictions when data becomes available"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_value = ?
                    WHERE target_date = ? AND actual_value IS NULL
                ''', (actual_value, date_str))
                
                conn.commit()
                rows_updated = cursor.rowcount
                logger.debug(f"Updated {rows_updated} predictions with actual value for {date_str}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update actual values: {str(e)}")
            return False

class DashboardDataPreparation:
    """Prepares data for dashboard consumption"""
    
    def __init__(self, data_dir: Path, prediction_db: PredictionDatabase):
        self.data_dir = data_dir
        self.prediction_db = prediction_db
        self.cache_dir = data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prepare comprehensive data for dashboard"""
        try:
            logger.info("Preparing dashboard data...")
            
            # Load processed features
            features_file = self.data_dir / "processed" / "sp500_features.csv"
            if not features_file.exists():
                raise PredictionError(f"Features file not found: {features_file}")
            
            features_df = pd.read_csv(features_file)
            features_df['date'] = pd.to_datetime(features_df['date'])
            
            # Get recent predictions
            predictions_df = self.prediction_db.get_recent_predictions(90)
            
            # Prepare historical data (last 90 days)
            recent_data = features_df.tail(90).copy()
            
            # Calculate additional metrics for dashboard
            dashboard_data = {
                'last_updated': datetime.now().isoformat(),
                'current_price': float(recent_data['close'].iloc[-1]) if len(recent_data) > 0 else None,
                'latest_date': recent_data['date'].max().isoformat() if len(recent_data) > 0 else None,
                'historical_data': self._prepare_historical_data(recent_data),
                'predictions': self._prepare_predictions_data(predictions_df),
                'performance_metrics': self._calculate_performance_metrics(predictions_df),
                'feature_importance': self._get_feature_importance(),
                'market_indicators': self._calculate_market_indicators(recent_data)
            }
            
            # Cache the prepared data
            cache_file = self.cache_dir / "dashboard_data.json"
            with open(cache_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info("Dashboard data prepared and cached successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to prepare dashboard data: {str(e)}")
            traceback.print_exc()
            return {}
    
    def _prepare_historical_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare historical price and volume data"""
        if data.empty:
            return []
        
        historical = []
        for _, row in data.iterrows():
            historical.append({
                'date': row['date'].isoformat(),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'sma_20': float(row.get('sma_20', 0)),
                'sma_50': float(row.get('sma_50', 0)),
                'rsi': float(row.get('rsi', 0)),
                'target': int(row.get('target', 0)) if pd.notna(row.get('target')) else None
            })
        
        return historical
    
    def _prepare_predictions_data(self, predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare predictions data for dashboard"""
        if predictions_df.empty:
            return []
        
        predictions = []
        for _, row in predictions_df.iterrows():
            prediction = {
                'prediction_date': row['prediction_date'],
                'target_date': row['target_date'],
                'prediction_type': row['prediction_type'],
                'prediction_value': float(row['prediction_value']),
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else None,
                'model_used': row['model_used'],
                'actual_value': float(row['actual_value']) if pd.notna(row['actual_value']) else None
            }
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_performance_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate prediction performance metrics"""
        if predictions_df.empty:
            return {}
        
        # Filter predictions with actual values
        completed_predictions = predictions_df[pd.notna(predictions_df['actual_value'])]
        
        if completed_predictions.empty:
            return {'total_predictions': len(predictions_df), 'completed_predictions': 0}
        
        # Calculate accuracy for binary predictions
        if 'prediction_value' in completed_predictions.columns:
            # Assume binary classification (0/1)
            binary_predictions = completed_predictions[
                completed_predictions['prediction_value'].isin([0, 1])
            ]
            
            if not binary_predictions.empty:
                accuracy = (binary_predictions['prediction_value'] == binary_predictions['actual_value']).mean()
            else:
                accuracy = None
        else:
            accuracy = None
        
        # Calculate recent performance (last 30 days)
        recent_date = datetime.now() - timedelta(days=30)
        recent_predictions = completed_predictions[
            pd.to_datetime(completed_predictions['prediction_date']) >= recent_date
        ]
        
        recent_accuracy = None
        if not recent_predictions.empty and 'prediction_value' in recent_predictions.columns:
            recent_binary = recent_predictions[recent_predictions['prediction_value'].isin([0, 1])]
            if not recent_binary.empty:
                recent_accuracy = (recent_binary['prediction_value'] == recent_binary['actual_value']).mean()
        
        return {
            'total_predictions': len(predictions_df),
            'completed_predictions': len(completed_predictions),
            'overall_accuracy': float(accuracy) if accuracy is not None else None,
            'recent_accuracy': float(recent_accuracy) if recent_accuracy is not None else None,
            'last_prediction_date': predictions_df['prediction_date'].max(),
            'prediction_completion_rate': len(completed_predictions) / len(predictions_df) if len(predictions_df) > 0 else 0
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            models_dir = project_root / "models"
            
            # Try to load Random Forest for feature importance
            rf_path = models_dir / "random_forest.pkl"
            if rf_path.exists():
                rf_model = joblib.load(rf_path)
                
                # Load feature names
                feature_names_path = models_dir / "feature_names.pkl"
                if feature_names_path.exists():
                    feature_names = joblib.load(feature_names_path)
                    
                    if hasattr(rf_model, 'feature_importances_'):
                        importance_dict = dict(zip(feature_names, rf_model.feature_importances_))
                        # Sort by importance and take top 10
                        top_features = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
                        return {k: float(v) for k, v in top_features.items()}
            
            return {}
            
        except Exception as e:
            logger.warning(f"Could not get feature importance: {str(e)}")
            return {}
    
    def _calculate_market_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate current market indicators"""
        if data.empty:
            return {}
        
        latest = data.iloc[-1]
        
        indicators = {
            'current_rsi': float(latest.get('rsi', 0)),
            'current_macd': float(latest.get('macd', 0)),
            'volatility_20d': float(data['close'].tail(20).std()) if len(data) >= 20 else None,
            'price_change_5d': float((latest['close'] - data['close'].iloc[-6]) / data['close'].iloc[-6] * 100) if len(data) >= 6 else None,
            'volume_avg_20d': float(data.get('volume', pd.Series([0])).tail(20).mean()) if len(data) >= 20 else None,
            'trend_direction': self._determine_trend(data)
        }
        
        return indicators
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine current trend direction"""
        if len(data) < 10:
            return "insufficient_data"
        
        recent_prices = data['close'].tail(10)
        if recent_prices.iloc[-1] > recent_prices.iloc[0]:
            return "upward"
        elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
            return "downward"
        else:
            return "sideways"

class PredictionGenerator:
    """Main prediction generation and management class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Prediction Generator
        
        Args:
            config: Configuration dictionary with settings
        """
        self.config = config or self._load_default_config()
        self.project_root = project_root
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # Initialize components
        self.predictor = SP500Predictor()
        self.prediction_db = PredictionDatabase(self.data_dir / "predictions.db")
        self.dashboard_prep = DashboardDataPreparation(self.data_dir, self.prediction_db)
        
        logger.info("Prediction Generator initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            'prediction_horizons': [1],  # Days ahead to predict
            'prediction_types': ['direction'],  # 'direction', 'price', 'volatility'
            'confidence_threshold': 0.6,  # Minimum confidence for predictions
            'max_prediction_age_days': 90,  # How long to keep predictions
            'enable_dashboard_update': True,
            'notification_email': os.getenv('NOTIFICATION_EMAIL'),
            'validate_predictions': True,
            'update_actual_values': True
        }
    
    def validate_models(self) -> Tuple[bool, str]:
        """Validate that required models are available and working"""
        try:
            # Check if model files exist
            required_models = ['random_forest.pkl', 'gradient_boosting.pkl', 'logistic_regression.pkl']
            missing_models = []
            
            for model_file in required_models:
                model_path = self.models_dir / model_file
                if not model_path.exists():
                    missing_models.append(model_file)
            
            if missing_models:
                return False, f"Missing model files: {missing_models}"
            
            # Test prediction functionality
            try:
                test_prediction = self.predictor.predict_next_day()
                if test_prediction is None:
                    return False, "Predictor returned None for test prediction"
                
                logger.info(f"Model validation passed - test prediction: {test_prediction}")
                return True, "Model validation successful"
                
            except Exception as e:
                return False, f"Prediction test failed: {str(e)}"
            
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for configured horizons and types"""
        try:
            logger.info("Starting prediction generation...")
            
            predictions = {}
            prediction_date = date.today().isoformat()
            
            for horizon in self.config['prediction_horizons']:
                target_date = (date.today() + timedelta(days=horizon)).isoformat()
                
                for pred_type in self.config['prediction_types']:
                    try:
                        if pred_type == 'direction':
                            # Predict market direction (up/down)
                            prediction_result = self.predictor.predict_next_day()
                            
                            if prediction_result is not None:
                                # Extract prediction value and confidence if available
                                if isinstance(prediction_result, dict):
                                    pred_value = prediction_result.get('prediction', prediction_result.get('direction', 0))
                                    confidence = prediction_result.get('confidence', 0.5)
                                    model_used = prediction_result.get('model', 'ensemble')
                                else:
                                    pred_value = prediction_result
                                    confidence = 0.5
                                    model_used = 'ensemble'
                                
                                # Store prediction
                                prediction_data = {
                                    'prediction_date': prediction_date,
                                    'target_date': target_date,
                                    'prediction_type': pred_type,
                                    'prediction_value': float(pred_value),
                                    'confidence': float(confidence),
                                    'model_used': model_used,
                                    'metadata': {
                                        'horizon_days': horizon,
                                        'generation_time': datetime.now().isoformat()
                                    }
                                }
                                
                                # Validate prediction if enabled
                                if self.config['validate_predictions']:
                                    if not self._validate_prediction(prediction_data):
                                        logger.warning(f"Prediction validation failed for {pred_type}, horizon {horizon}")
                                        continue
                                
                                # Save to database
                                if self.prediction_db.save_prediction(prediction_data):
                                    predictions[f"{pred_type}_{horizon}d"] = prediction_data
                                    logger.info(f"Generated {pred_type} prediction for {horizon} days: {pred_value}")
                                else:
                                    logger.error(f"Failed to save {pred_type} prediction for {horizon} days")
                            else:
                                logger.warning(f"No prediction returned for {pred_type}, horizon {horizon}")
                        
                        # Add other prediction types here in the future
                        # elif pred_type == 'price':
                        #     # Implement price prediction
                        # elif pred_type == 'volatility':
                        #     # Implement volatility prediction
                        
                    except Exception as e:
                        logger.error(f"Failed to generate {pred_type} prediction for horizon {horizon}: {str(e)}")
            
            logger.info(f"Generated {len(predictions)} predictions successfully")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {str(e)}")
            traceback.print_exc()
            return {}
    
    def _validate_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Validate a prediction before saving"""
        try:
            # Check required fields
            required_fields = ['prediction_date', 'target_date', 'prediction_type', 'prediction_value']
            for field in required_fields:
                if field not in prediction_data:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate prediction value range for direction predictions
            if prediction_data['prediction_type'] == 'direction':
                pred_value = prediction_data['prediction_value']
                if not (pred_value == 0 or pred_value == 1 or (0 <= pred_value <= 1)):
                    logger.warning(f"Invalid direction prediction value: {pred_value}")
                    return False
            
            # Check confidence threshold
            confidence = prediction_data.get('confidence', 0.5)
            if confidence < self.config['confidence_threshold']:
                logger.warning(f"Prediction confidence {confidence} below threshold {self.config['confidence_threshold']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Prediction validation error: {str(e)}")
            return False
    
    def update_actual_values(self) -> bool:
        """Update actual values for past predictions"""
        try:
            if not self.config['update_actual_values']:
                logger.info("Actual value updates disabled")
                return True
            
            logger.info("Updating actual values for past predictions...")
            
            # Load recent processed data
            features_file = self.data_dir / "processed" / "sp500_features.csv"
            if not features_file.exists():
                logger.warning("Features file not found, skipping actual value updates")
                return False
            
            features_df = pd.read_csv(features_file)
            features_df['date'] = pd.to_datetime(features_df['date'])
            
            # Get recent predictions that need actual values
            recent_predictions = self.prediction_db.get_recent_predictions(30)
            pending_predictions = recent_predictions[pd.isna(recent_predictions['actual_value'])]
            
            updates_count = 0
            for _, pred_row in pending_predictions.iterrows():
                target_date = pd.to_datetime(pred_row['target_date']).date()
                
                # Find matching data
                matching_data = features_df[features_df['date'].dt.date == target_date]
                
                if not matching_data.empty:
                    if pred_row['prediction_type'] == 'direction':
                        # Use target column for direction
                        actual_value = matching_data['target'].iloc[0]
                        if pd.notna(actual_value):
                            if self.prediction_db.update_actual_values(target_date.isoformat(), actual_value):
                                updates_count += 1
            
            logger.info(f"Updated actual values for {updates_count} predictions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update actual values: {str(e)}")
            return False
    
    def cleanup_old_predictions(self) -> bool:
        """Remove old predictions based on retention policy"""
        try:
            max_age = self.config['max_prediction_age_days']
            cutoff_date = (datetime.now() - timedelta(days=max_age)).isoformat()
            
            with sqlite3.connect(self.prediction_db.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE prediction_date < ?
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old predictions")
                return True
                
        except Exception as e:
            logger.error(f"Failed to cleanup old predictions: {str(e)}")
            return False
    
    def send_notification(self, subject: str, message: str, is_error: bool = False):
        """Send email notification about prediction status"""
        if not self.config.get('notification_email'):
            logger.info("No notification email configured, skipping notification")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = os.getenv('SMTP_USERNAME')
            msg['To'] = self.config['notification_email']
            msg['Subject'] = f"[S&P500 Predictor] {subject}"
            
            # Add timestamp and system info
            full_message = f"""
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: {'ERROR' if is_error else 'SUCCESS'}

{message}

---
S&P 500 Prediction System
Daily Prediction Generator
"""
            
            msg.attach(MIMEText(full_message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'), 
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(os.getenv('SMTP_USERNAME'), os.getenv('SMTP_PASSWORD'))
            text = msg.as_string()
            server.sendmail(os.getenv('SMTP_USERNAME'), self.config['notification_email'], text)
            server.quit()
            
            logger.info("Notification email sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification email: {str(e)}")
    
    def run_prediction_pipeline(self) -> bool:
        """Run the complete prediction generation pipeline"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting prediction generation pipeline at {start_time}")
        logger.info("=" * 60)
        
        success = True
        error_messages = []
        
        try:
            # Step 1: Validate models
            logger.info("Step 1: Validating models")
            is_valid, message = self.validate_models()
            
            if not is_valid:
                raise PredictionError(f"Model validation failed: {message}")
            
            logger.info(f"Model validation passed: {message}")
            
            # Step 2: Generate predictions
            logger.info("Step 2: Generating predictions")
            predictions = self.generate_predictions()
            
            if not predictions:
                error_msg = "No predictions were generated"
                logger.error(error_msg)
                error_messages.append(error_msg)
                success = False
            
            # Step 3: Update actual values for past predictions
            logger.info("Step 3: Updating actual values")
            if not self.update_actual_values():
                error_msg = "Failed to update actual values"
                logger.warning(error_msg)  # Non-critical error
                error_messages.append(error_msg)
            
            # Step 4: Prepare dashboard data
            if self.config['enable_dashboard_update']:
                logger.info("Step 4: Preparing dashboard data")
                dashboard_data = self.dashboard_prep.prepare_dashboard_data()
                
                if not dashboard_data:
                    error_msg = "Failed to prepare dashboard data"
                    logger.warning(error_msg)  # Non-critical error
                    error_messages.append(error_msg)
            
            # Step 5: Cleanup old predictions
            logger.info("Step 5: Cleaning up old predictions")
            if not self.cleanup_old_predictions():
                error_msg = "Failed to cleanup old predictions"
                logger.warning(error_msg)  # Non-critical error
                error_messages.append(error_msg)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Send notification
            if success:
                prediction_summary = {k: v['prediction_value'] for k, v in predictions.items()}
                message = f"""Prediction generation completed successfully!

Execution time: {execution_time:.1f} seconds
Predictions generated: {len(predictions)}
Prediction summary: {prediction_summary}

Dashboard data updated: {'Yes' if self.config['enable_dashboard_update'] else 'No'}
All systems operational.
"""
                self.send_notification("Prediction Generation Successful", message, is_error=False)
                logger.info(f"Prediction pipeline completed successfully in {execution_time:.1f} seconds")
            else:
                message = f"""Prediction generation completed with errors!

Execution time: {execution_time:.1f} seconds
Predictions generated: {len(predictions)}

Errors encountered:
{chr(10).join(error_messages)}

Please check the logs for more details.
"""
                self.send_notification("Prediction Generation Failed", message, is_error=True)
                logger.error(f"Prediction pipeline completed with errors in {execution_time:.1f} seconds")
            
            return success
            
        except Exception as e:
            error_msg = f"Critical error in prediction pipeline: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            notification_message = f"""Prediction generation failed!

Execution time: {execution_time:.1f} seconds
Error: {error_msg}

Stacktrace:
{traceback.format_exc()}
"""
            
            self.send_notification(
                "Prediction Generation Critical Error",
                notification_message,
                is_error=True
            )
            
            return False
        
        finally:
            logger.info("=" * 60)
            logger.info(f"Prediction generation pipeline finished at {datetime.now()}")
            logger.info("=" * 60)

def main():
    """Main function for command line execution"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Prediction Generator for S&P 500 Prediction System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard data preparation')
    parser.add_argument('--test-only', action='store_true', help='Test prediction generation without saving')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)
    
    # Create generator and run
    try:
        generator = PredictionGenerator(config=config)
        
        # Modify configuration based on arguments
        if args.no_dashboard:
            generator.config['enable_dashboard_update'] = False
            logger.info("Dashboard data preparation disabled")
        
        if args.test_only:
            logger.info("Test mode - validating models only")
            is_valid, message = generator.validate_models()
            if is_valid:
                logger.info(f"✓ Models validation successful: {message}")
                test_prediction = generator.predictor.predict_next_day()
                logger.info(f"✓ Test prediction: {test_prediction}")
                sys.exit(0)
            else:
                logger.error(f"✗ Models validation failed: {message}")
                sys.exit(1)
        
        success = generator.run_prediction_pipeline()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run prediction generator: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
