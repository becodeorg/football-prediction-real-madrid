"""
Model Training Module for S&P 500 Prediction System
Responsible for training machine learning models to predict stock movements
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SP500ModelTrainer:
    def __init__(self, data_dir="../data", models_dir="../models"):
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_processed_data(self):
        """
        Load processed data for training
        
        Returns:
            pd.DataFrame: Processed data
        """
        try:
            filepath = os.path.join(self.processed_dir, "sp500_features.csv")
            data = pd.read_csv(filepath)
            
            # Convert date column
            data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"Loaded processed data: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None
    
    def prepare_features_and_targets(self, data, target_column='target'):
        """
        Prepare features and targets for training
        
        Args:
            data (pd.DataFrame): Processed data
            target_column (str): Target column name
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Define columns to exclude from features
        exclude_columns = [
            'date', 'target', 'target_multiclass', 'next_close', 'price_change_pct'
        ]
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Prepare features and target
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        self.feature_names = feature_columns
        
        logger.info(f"Prepared features: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            dict: Trained models with their performance
        """
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 6]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            }
        }
        
        trained_models = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Grid search for best parameters
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=3,  # Reduced for faster training
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Predictions
                train_pred = best_model.predict(X_train)
                val_pred = best_model.predict(X_val)
                
                # Calculate performance
                train_accuracy = accuracy_score(y_train, train_pred)
                val_accuracy = accuracy_score(y_val, val_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
                
                model_info = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_predictions': val_pred
                }
                
                trained_models[model_name] = model_info
                
                logger.info(f"{model_name} - Val Accuracy: {val_accuracy:.4f}, CV: {cv_scores.mean():.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        return trained_models
    
    def evaluate_models(self, models, X_val, y_val):
        """
        Evaluate trained models
        
        Args:
            models (dict): Trained models
            X_val, y_val: Validation data
        """
        logger.info("=== Model Evaluation ===")
        
        best_model = None
        best_score = 0
        
        for model_name, model_info in models.items():
            model = model_info['model']
            val_pred = model.predict(X_val)
            
            logger.info(f"--- {model_name.upper()} ---")
            logger.info(f"Best parameters: {model_info['best_params']}")
            logger.info(f"Validation Accuracy: {model_info['val_accuracy']:.4f}")
            logger.info(f"Cross-validation: {model_info['cv_mean']:.4f}")
            
            # Classification report
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_val, val_pred))
            
            # Confusion matrix
            print(f"\nConfusion Matrix for {model_name}:")
            print(confusion_matrix(y_val, val_pred))
            print("\n" + "="*50 + "\n")
            
            # Track best model
            if model_info['val_accuracy'] > best_score:
                best_score = model_info['val_accuracy']
                best_model = (model_name, model)
        
        logger.info(f"Best model: {best_model[0]} with accuracy: {best_score:.4f}")
        return best_model
    
    def save_models(self, models, scaler):
        """
        Save trained models and scaler
        
        Args:
            models (dict): Trained models
            scaler: Fitted scaler
        """
        try:
            # Save scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
            
            # Save models
            for model_name, model_info in models.items():
                model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
                joblib.dump(model_info['model'], model_path)
                logger.info(f"{model_name} saved to {model_path}")
            
            # Save feature names
            feature_names_path = os.path.join(self.models_dir, "feature_names.pkl")
            joblib.dump(self.feature_names, feature_names_path)
            logger.info(f"Feature names saved to {feature_names_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def train_pipeline(self, target_column='target', test_size=0.2, val_size=0.2):
        """
        Complete training pipeline
        
        Args:
            target_column (str): Target column to predict
            test_size (float): Test set size
            val_size (float): Validation set size
            
        Returns:
            dict: Trained models
        """
        logger.info("Starting model training pipeline...")
        
        # Load data
        data = self.load_processed_data()
        if data is None:
            return None
        
        # Prepare features and targets
        X, y, feature_names = self.prepare_features_and_targets(data, target_column)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        trained_models = self.train_models(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate models
        best_model = self.evaluate_models(trained_models, X_val_scaled, y_val)
        
        # Save models
        self.save_models(trained_models, self.scaler)
        
        logger.info("Model training pipeline completed!")
        return trained_models, best_model

def main():
    """
    Main function to run model training
    """
    trainer = SP500ModelTrainer()
    
    # Train binary classification models
    print("\n=== TRAINING BINARY CLASSIFICATION MODELS ===\n")
    models, best_model = trainer.train_pipeline(target_column='target')
    
    print(f"\nBest performing model: {best_model[0]}")

if __name__ == "__main__":
    main()
