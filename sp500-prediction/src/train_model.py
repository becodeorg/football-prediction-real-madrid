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
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
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
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
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
                    cv=5,
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
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
                
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
                
                logger.info(f"{model_name} - Val Accuracy: {val_accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")\n        \n        return trained_models\n    \n    def evaluate_models(self, models, X_val, y_val):\n        \"\"\"\n        Evaluate trained models\n        \n        Args:\n            models (dict): Trained models\n            X_val, y_val: Validation data\n        \"\"\"\n        logger.info(\"\\n=== Model Evaluation ===\\n\")\n        \n        best_model = None\n        best_score = 0\n        \n        for model_name, model_info in models.items():\n            model = model_info['model']\n            val_pred = model.predict(X_val)\n            \n            logger.info(f\"--- {model_name.upper()} ---\")\n            logger.info(f\"Best parameters: {model_info['best_params']}\")\n            logger.info(f\"Validation Accuracy: {model_info['val_accuracy']:.4f}\")\n            logger.info(f\"Cross-validation: {model_info['cv_mean']:.4f} (+/- {model_info['cv_std'] * 2:.4f})\")\n            \n            # Classification report\n            print(f\"\\nClassification Report for {model_name}:\")\n            print(classification_report(y_val, val_pred))\n            \n            # Confusion matrix\n            print(f\"\\nConfusion Matrix for {model_name}:\")\n            print(confusion_matrix(y_val, val_pred))\n            print(\"\\n\" + \"=\"*50 + \"\\n\")\n            \n            # Track best model\n            if model_info['val_accuracy'] > best_score:\n                best_score = model_info['val_accuracy']\n                best_model = (model_name, model)\n        \n        logger.info(f\"Best model: {best_model[0]} with accuracy: {best_score:.4f}\")\n        return best_model\n    \n    def save_models(self, models, scaler):\n        \"\"\"\n        Save trained models and scaler\n        \n        Args:\n            models (dict): Trained models\n            scaler: Fitted scaler\n        \"\"\"\n        try:\n            # Save scaler\n            scaler_path = os.path.join(self.models_dir, \"scaler.pkl\")\n            joblib.dump(scaler, scaler_path)\n            logger.info(f\"Scaler saved to {scaler_path}\")\n            \n            # Save models\n            for model_name, model_info in models.items():\n                model_path = os.path.join(self.models_dir, f\"{model_name}.pkl\")\n                joblib.dump(model_info['model'], model_path)\n                logger.info(f\"{model_name} saved to {model_path}\")\n            \n            # Save feature names\n            feature_names_path = os.path.join(self.models_dir, \"feature_names.pkl\")\n            joblib.dump(self.feature_names, feature_names_path)\n            logger.info(f\"Feature names saved to {feature_names_path}\")\n            \n            # Save model performance summary\n            performance_summary = {}\n            for model_name, model_info in models.items():\n                performance_summary[model_name] = {\n                    'best_params': model_info['best_params'],\n                    'train_accuracy': model_info['train_accuracy'],\n                    'val_accuracy': model_info['val_accuracy'],\n                    'cv_mean': model_info['cv_mean'],\n                    'cv_std': model_info['cv_std']\n                }\n            \n            summary_path = os.path.join(self.models_dir, \"model_performance.pkl\")\n            joblib.dump(performance_summary, summary_path)\n            logger.info(f\"Performance summary saved to {summary_path}\")\n            \n        except Exception as e:\n            logger.error(f\"Error saving models: {e}\")\n    \n    def train_pipeline(self, target_column='target', test_size=0.2, val_size=0.2):\n        \"\"\"\n        Complete training pipeline\n        \n        Args:\n            target_column (str): Target column to predict\n            test_size (float): Test set size\n            val_size (float): Validation set size\n            \n        Returns:\n            dict: Trained models\n        \"\"\"\n        logger.info(\"Starting model training pipeline...\")\n        \n        # Load data\n        data = self.load_processed_data()\n        if data is None:\n            return None\n        \n        # Prepare features and targets\n        X, y, feature_names = self.prepare_features_and_targets(data, target_column)\n        \n        # Split data\n        X_temp, X_test, y_temp, y_test = train_test_split(\n            X, y, test_size=test_size, random_state=42, stratify=y\n        )\n        \n        X_train, X_val, y_train, y_val = train_test_split(\n            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp\n        )\n        \n        logger.info(f\"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\")\n        \n        # Scale features\n        X_train_scaled = self.scaler.fit_transform(X_train)\n        X_val_scaled = self.scaler.transform(X_val)\n        X_test_scaled = self.scaler.transform(X_test)\n        \n        # Train models\n        trained_models = self.train_models(X_train_scaled, y_train, X_val_scaled, y_val)\n        \n        # Evaluate models\n        best_model = self.evaluate_models(trained_models, X_val_scaled, y_val)\n        \n        # Save models\n        self.save_models(trained_models, self.scaler)\n        \n        logger.info(\"Model training pipeline completed!\")\n        return trained_models, best_model\n\ndef main():\n    \"\"\"\n    Main function to run model training\n    \"\"\"\n    trainer = SP500ModelTrainer()\n    \n    # Train binary classification models\n    print(\"\\n=== TRAINING BINARY CLASSIFICATION MODELS ===\\n\")\n    models, best_model = trainer.train_pipeline(target_column='target')\n    \n    print(f\"\\nBest performing model: {best_model[0]}\")\n    \n    # Train multi-class models\n    print(\"\\n=== TRAINING MULTI-CLASS CLASSIFICATION MODELS ===\\n\")\n    trainer_multiclass = SP500ModelTrainer(models_dir=\"../models/multiclass\")\n    models_mc, best_model_mc = trainer_multiclass.train_pipeline(target_column='target_multiclass')\n    \n    print(f\"\\nBest performing multi-class model: {best_model_mc[0]}\")\n\nif __name__ == \"__main__\":\n    main()
