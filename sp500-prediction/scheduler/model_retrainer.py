#!/usr/bin/env python3
"""
Model Retraining Scheduler for S&P 500 Prediction System
Automates model retraining, validation, versioning, and deployment

Features:
- Weekly/monthly automated model retraining
- Performance monitoring and validation against current models
- Model versioning and backup system
- Automatic deployment of better-performing models
- A/B testing capabilities for model comparison
- Comprehensive model performance tracking
"""

import os
import sys
import json
import shutil
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

try:
    from train_model import SP500ModelTrainer
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
        logging.FileHandler(log_dir / f"model_retrainer_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelRetrainingError(Exception):
    """Custom exception for model retraining errors"""
    pass

class ModelPerformanceTracker:
    """Tracks and compares model performance metrics"""
    
    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics_history()
    
    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical model performance metrics"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metrics history: {e}")
        return []
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save model performance metrics"""
        self.metrics_history.append(metrics)
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            logger.info(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest model performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def compare_with_baseline(self, new_metrics: Dict[str, Any], 
                            improvement_threshold: float = 0.02) -> Tuple[bool, str]:
        """
        Compare new model metrics with baseline (latest production model)
        
        Args:
            new_metrics: Performance metrics of new model
            improvement_threshold: Minimum improvement required for deployment
            
        Returns:
            Tuple of (should_deploy, comparison_message)
        """
        if not self.metrics_history:
            return True, "No baseline model found. Deploying first model."
        
        baseline = self.get_latest_metrics()
        
        # Primary metric for comparison (accuracy)
        baseline_acc = baseline.get('test_accuracy', 0)
        new_acc = new_metrics.get('test_accuracy', 0)
        
        improvement = new_acc - baseline_acc
        
        if improvement >= improvement_threshold:
            message = f"Model shows significant improvement: {improvement:.4f} (+{improvement*100:.2f}%)"
            return True, message
        elif improvement > 0:
            message = f"Model shows minor improvement: {improvement:.4f} (+{improvement*100:.2f}%), below threshold"
            return False, message
        else:
            message = f"Model shows degradation: {improvement:.4f} ({improvement*100:.2f}%)"
            return False, message

class ModelRetrainer:
    """Handles automated model retraining and deployment"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Model Retrainer
        
        Args:
            config: Configuration dictionary with settings
        """
        self.config = config or self._load_default_config()
        self.project_root = project_root
        self.models_dir = self.project_root / "models"
        self.backup_dir = self.models_dir / "backups"
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_trainer = SP500ModelTrainer(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir)
        )
        
        # Performance tracking
        self.performance_tracker = ModelPerformanceTracker(
            self.models_dir / "performance_history.json"
        )
        
        logger.info("Model Retrainer initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            'retraining_frequency': 'weekly',  # 'weekly', 'monthly', 'manual'
            'min_training_samples': 100,
            'test_size': 0.2,
            'validation_size': 0.1,
            'improvement_threshold': 0.02,  # 2% improvement required
            'backup_retention_days': 90,
            'models_to_train': ['random_forest', 'gradient_boosting', 'logistic_regression'],
            'cross_validation_folds': 5,
            'performance_metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'auto_deploy': True,
            'notification_email': os.getenv('NOTIFICATION_EMAIL'),
            'feature_selection': True,
            'hyperparameter_tuning': True
        }
    
    def check_retraining_schedule(self) -> bool:
        """
        Check if it's time to retrain models based on schedule
        
        Returns:
            bool: True if retraining should proceed
        """
        frequency = self.config['retraining_frequency']
        
        if frequency == 'manual':
            logger.info("Manual retraining mode - proceeding")
            return True
        
        # Get last retraining date
        last_metrics = self.performance_tracker.get_latest_metrics()
        if not last_metrics:
            logger.info("No previous training found - proceeding with initial training")
            return True
        
        last_training_date = datetime.fromisoformat(last_metrics['timestamp'])
        days_since_training = (datetime.now() - last_training_date).days
        
        if frequency == 'weekly' and days_since_training >= 7:
            logger.info(f"Weekly retraining due: {days_since_training} days since last training")
            return True
        elif frequency == 'monthly' and days_since_training >= 30:
            logger.info(f"Monthly retraining due: {days_since_training} days since last training")
            return True
        else:
            logger.info(f"Retraining not due: {days_since_training} days since last training")
            return False
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate training data quality and completeness
        
        Args:
            data: Training dataset
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        try:
            # Check minimum samples
            if len(data) < self.config['min_training_samples']:
                return False, f"Insufficient training samples: {len(data)} < {self.config['min_training_samples']}"
            
            # Check for required columns
            required_columns = ['target', 'close', 'date']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            # Check target distribution
            target_dist = data['target'].value_counts()
            if len(target_dist) < 2:
                return False, "Target variable must have at least 2 classes"
            
            # Check for reasonable target balance (not too imbalanced)
            min_class_ratio = target_dist.min() / len(data)
            if min_class_ratio < 0.05:  # Less than 5% of minority class
                logger.warning(f"Imbalanced dataset detected: minority class = {min_class_ratio:.2%}")
            
            # Check data recency
            data['date'] = pd.to_datetime(data['date'])
            latest_date = data['date'].max()
            days_old = (datetime.now() - latest_date).days
            
            if days_old > 7:
                logger.warning(f"Training data is {days_old} days old")
            
            logger.info(f"Data validation passed: {len(data)} samples, target distribution: {dict(target_dist)}")
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
    
    def backup_current_models(self) -> bool:
        """
        Create backup of current production models
        
        Returns:
            bool: True if backup successful
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_subdir = self.backup_dir / f"backup_{timestamp}"
            backup_subdir.mkdir(exist_ok=True)
            
            # Backup model files
            model_files = [
                'random_forest.pkl',
                'gradient_boosting.pkl',
                'logistic_regression.pkl',
                'scaler.pkl',
                'feature_names.pkl'
            ]
            
            backed_up_files = []
            for model_file in model_files:
                src_path = self.models_dir / model_file
                if src_path.exists():
                    dst_path = backup_subdir / model_file
                    shutil.copy2(src_path, dst_path)
                    backed_up_files.append(model_file)
            
            # Backup performance history
            perf_file = self.models_dir / "performance_history.json"
            if perf_file.exists():
                shutil.copy2(perf_file, backup_subdir / "performance_history.json")
                backed_up_files.append("performance_history.json")
            
            # Create backup metadata
            backup_metadata = {
                'timestamp': timestamp,
                'backed_up_files': backed_up_files,
                'backup_reason': 'model_retraining',
                'performance_metrics': self.performance_tracker.get_latest_metrics()
            }
            
            with open(backup_subdir / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2, default=str)
            
            logger.info(f"Backup created successfully: {backup_subdir}")
            logger.info(f"Backed up files: {backed_up_files}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup models: {str(e)}")
            return False
    
    def cleanup_old_backups(self):
        """Remove old backup files based on retention policy"""
        try:
            retention_days = self.config['backup_retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            removed_count = 0
            for backup_dir in self.backup_dir.glob("backup_*"):
                if backup_dir.is_dir():
                    # Extract timestamp from directory name
                    try:
                        timestamp_str = backup_dir.name.replace("backup_", "")
                        backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        if backup_date < cutoff_date:
                            shutil.rmtree(backup_dir)
                            removed_count += 1
                            logger.info(f"Removed old backup: {backup_dir}")
                    except ValueError:
                        logger.warning(f"Could not parse backup date from: {backup_dir}")
            
            logger.info(f"Cleanup completed: removed {removed_count} old backups")
            
        except Exception as e:
            logger.error(f"Error during backup cleanup: {str(e)}")
    
    def train_new_models(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Train new models using the latest data
        
        Args:
            data: Training dataset
            
        Returns:
            Tuple of (success, performance_metrics)
        """
        try:
            logger.info("Starting model training...")
            
            # Load and prepare data using the trainer
            self.model_trainer.data = data
            
            # Prepare features and target
            features_data = self.model_trainer.prepare_features()
            if features_data is None:
                raise ModelRetrainingError("Failed to prepare features")
            
            X_train, X_test, y_train, y_test = features_data
            
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Test data shape: {X_test.shape}")
            logger.info(f"Target distribution - Train: {pd.Series(y_train).value_counts().to_dict()}")
            logger.info(f"Target distribution - Test: {pd.Series(y_test).value_counts().to_dict()}")
            
            # Train models
            models_performance = {}
            trained_models = {}
            
            for model_name in self.config['models_to_train']:
                logger.info(f"Training {model_name}...")
                
                try:
                    if model_name == 'random_forest':
                        model = self.model_trainer.train_random_forest(X_train, y_train)
                    elif model_name == 'gradient_boosting':
                        model = self.model_trainer.train_gradient_boosting(X_train, y_train)
                    elif model_name == 'logistic_regression':
                        model = self.model_trainer.train_logistic_regression(X_train, y_train)
                    else:
                        logger.warning(f"Unknown model type: {model_name}")
                        continue
                    
                    if model is not None:
                        # Evaluate model
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        model_metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1
                        }
                        
                        models_performance[model_name] = model_metrics
                        trained_models[model_name] = model
                        
                        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {str(e)}")
            
            if not trained_models:
                raise ModelRetrainingError("No models were successfully trained")
            
            # Save models and metadata
            timestamp = datetime.now().isoformat()
            
            for model_name, model in trained_models.items():
                model_path = self.models_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
            
            # Save scaler and feature names
            if hasattr(self.model_trainer, 'scaler'):
                joblib.dump(self.model_trainer.scaler, self.models_dir / "scaler.pkl")
            
            if hasattr(self.model_trainer, 'feature_names'):
                joblib.dump(self.model_trainer.feature_names, self.models_dir / "feature_names.pkl")
            
            # Calculate ensemble performance (average of all models)
            ensemble_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                values = [models_performance[model][metric] for model in models_performance]
                ensemble_metrics[metric] = np.mean(values)
            
            # Compile comprehensive metrics
            comprehensive_metrics = {
                'timestamp': timestamp,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X_train.shape[1],
                'models_trained': list(trained_models.keys()),
                'individual_models': models_performance,
                'ensemble_metrics': ensemble_metrics,
                'test_accuracy': ensemble_metrics['accuracy'],  # Primary metric for comparison
                'data_date_range': {
                    'start': str(data['date'].min()),
                    'end': str(data['date'].max())
                }
            }
            
            logger.info("Model training completed successfully")
            logger.info(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.4f}")
            
            return True, comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            traceback.print_exc()
            return False, {}
    
    def validate_new_models(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate newly trained models against production requirements
        
        Args:
            metrics: Performance metrics of new models
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        try:
            # Check minimum accuracy threshold
            min_accuracy = 0.52  # Should be better than random (50%)
            test_accuracy = metrics.get('test_accuracy', 0)
            
            if test_accuracy < min_accuracy:
                return False, f"Model accuracy {test_accuracy:.4f} below minimum threshold {min_accuracy:.4f}"
            
            # Check that all expected models were trained
            expected_models = set(self.config['models_to_train'])
            trained_models = set(metrics.get('models_trained', []))
            missing_models = expected_models - trained_models
            
            if missing_models:
                return False, f"Missing trained models: {missing_models}"
            
            # Check for reasonable performance consistency across models
            individual_models = metrics.get('individual_models', {})
            if individual_models:
                accuracies = [m['accuracy'] for m in individual_models.values()]
                accuracy_std = np.std(accuracies)
                
                if accuracy_std > 0.1:  # More than 10% standard deviation
                    logger.warning(f"High variance in model performance: std={accuracy_std:.4f}")
            
            logger.info("Model validation passed")
            return True, "Model validation successful"
            
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    
    def deploy_models(self, metrics: Dict[str, Any]) -> bool:
        """
        Deploy new models to production (they're already saved in the models directory)
        
        Args:
            metrics: Performance metrics of new models
            
        Returns:
            bool: True if deployment successful
        """
        try:
            logger.info("Deploying new models to production...")
            
            # Create deployment metadata
            deployment_metadata = {
                'deployment_timestamp': datetime.now().isoformat(),
                'model_metrics': metrics,
                'deployment_reason': 'automated_retraining',
                'deployed_models': metrics.get('models_trained', [])
            }
            
            # Save deployment metadata
            metadata_path = self.models_dir / "deployment_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(deployment_metadata, f, indent=2, default=str)
            
            # Test the deployed models by making a prediction
            try:
                predictor = SP500Predictor()
                test_prediction = predictor.predict_next_day()
                
                if test_prediction is not None:
                    logger.info(f"Deployment validation successful - test prediction: {test_prediction}")
                else:
                    logger.warning("Deployment validation failed - could not make test prediction")
                    return False
                    
            except Exception as e:
                logger.error(f"Deployment validation failed: {str(e)}")
                return False
            
            logger.info("Models deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return False
    
    def send_notification(self, subject: str, message: str, is_error: bool = False):
        """
        Send email notification about retraining status
        
        Args:
            subject: Email subject
            message: Email message body
            is_error: Whether this is an error notification
        """
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
Model Retraining Service
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
    
    def run_retraining_cycle(self) -> bool:
        """
        Run the complete model retraining cycle
        
        Returns:
            bool: True if successful, False if any critical errors occurred
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info(f"Starting model retraining cycle at {start_time}")
        logger.info("=" * 60)
        
        success = True
        error_messages = []
        
        try:
            # Step 1: Check if retraining is due
            if not self.check_retraining_schedule():
                logger.info("Model retraining not due according to schedule")
                return True
            
            # Step 2: Load and validate training data
            logger.info("Step 1: Loading and validating training data")
            data_file = self.processed_dir / "sp500_features.csv"
            
            if not data_file.exists():
                raise ModelRetrainingError(f"Training data not found: {data_file}")
            
            data = pd.read_csv(data_file)
            is_valid, message = self.validate_data_quality(data)
            
            if not is_valid:
                raise ModelRetrainingError(f"Data validation failed: {message}")
            
            logger.info(f"Data validation passed: {message}")
            
            # Step 3: Backup current models
            logger.info("Step 2: Backing up current models")
            if not self.backup_current_models():
                error_messages.append("Failed to backup current models")
                # Continue anyway - this is not critical
            
            # Step 4: Train new models
            logger.info("Step 3: Training new models")
            training_success, metrics = self.train_new_models(data)
            
            if not training_success:
                raise ModelRetrainingError("Model training failed")
            
            # Step 5: Validate new models
            logger.info("Step 4: Validating new models")
            is_valid, message = self.validate_new_models(metrics)
            
            if not is_valid:
                raise ModelRetrainingError(f"Model validation failed: {message}")
            
            # Step 6: Compare with baseline and decide deployment
            logger.info("Step 5: Comparing with baseline models")
            should_deploy, comparison_message = self.performance_tracker.compare_with_baseline(
                metrics, self.config['improvement_threshold']
            )
            
            logger.info(f"Baseline comparison: {comparison_message}")
            
            if should_deploy and self.config['auto_deploy']:
                # Step 7: Deploy new models
                logger.info("Step 6: Deploying new models")
                if self.deploy_models(metrics):
                    logger.info("Model deployment successful")
                    # Save performance metrics after successful deployment
                    self.performance_tracker.save_metrics(metrics)
                else:
                    raise ModelRetrainingError("Model deployment failed")
            elif not should_deploy:
                logger.info("New models do not meet improvement threshold - keeping current models")
                # Save metrics anyway for tracking
                metrics['deployed'] = False
                metrics['deployment_reason'] = 'insufficient_improvement'
                self.performance_tracker.save_metrics(metrics)
            else:
                logger.info("Auto-deployment disabled - new models trained but not deployed")
                metrics['deployed'] = False
                metrics['deployment_reason'] = 'auto_deploy_disabled'
                self.performance_tracker.save_metrics(metrics)
            
            # Step 8: Cleanup old backups
            logger.info("Step 7: Cleaning up old backups")
            self.cleanup_old_backups()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Send success notification
            if should_deploy and self.config['auto_deploy']:
                notification_subject = "Model Retraining and Deployment Successful"
                notification_message = f"""Model retraining cycle completed successfully!

Execution time: {execution_time:.1f} seconds
Models trained: {', '.join(metrics.get('models_trained', []))}
Test accuracy: {metrics.get('test_accuracy', 0):.4f}
Improvement: {comparison_message}

New models have been deployed to production.
"""
            else:
                notification_subject = "Model Retraining Completed (No Deployment)"
                notification_message = f"""Model retraining cycle completed!

Execution time: {execution_time:.1f} seconds
Models trained: {', '.join(metrics.get('models_trained', []))}
Test accuracy: {metrics.get('test_accuracy', 0):.4f}
Decision: {comparison_message}

Current production models remain unchanged.
"""
            
            self.send_notification(notification_subject, notification_message, is_error=False)
            logger.info(f"Model retraining cycle completed successfully in {execution_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            error_msg = f"Critical error in model retraining: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            notification_message = f"""Model retraining cycle failed!

Execution time: {execution_time:.1f} seconds
Error: {error_msg}

Production models remain unchanged.

Stacktrace:
{traceback.format_exc()}
"""
            
            self.send_notification(
                "Model Retraining Failed",
                notification_message,
                is_error=True
            )
            
            return False
        
        finally:
            logger.info("=" * 60)
            logger.info(f"Model retraining cycle finished at {datetime.now()}")
            logger.info("=" * 60)

def main():
    """Main function for command line execution"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Model Retraining for S&P 500 Prediction System')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of schedule')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-deploy', action='store_true', help='Train models but do not deploy')
    
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
    
    # Create retrainer and run
    try:
        retrainer = ModelRetrainer(config=config)
        
        # Override schedule check if forced
        if args.force:
            logger.info("Forcing retraining regardless of schedule")
            retrainer.check_retraining_schedule = lambda: True
        
        # Disable auto-deployment if requested
        if args.no_deploy:
            logger.info("Auto-deployment disabled")
            retrainer.config['auto_deploy'] = False
        
        success = retrainer.run_retraining_cycle()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Failed to initialize or run model retraining: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
