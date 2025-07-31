#!/usr/bin/env python3
"""
Test Script for Prediction Generator
Validates prediction generation functionality before scheduling

Usage:
    python test_prediction_generator.py
"""

import os
import sys
import json
import tempfile
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'scheduler'))

def test_environment_setup():
    """Test if the environment is properly set up"""
    print("=" * 60)
    print("TESTING ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check project structure
    required_dirs = [
        project_root / "data" / "processed",
        project_root / "models",
        project_root / "src",
        project_root / "scheduler"
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            return False
    
    # Check required files
    required_files = [
        project_root / "data" / "processed" / "sp500_features.csv",
        project_root / "src" / "predict.py",
        project_root / "scheduler" / "prediction_generator.py"
    ]
    
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False
    
    # Check for model files
    model_files = [
        "random_forest.pkl",
        "gradient_boosting.pkl", 
        "logistic_regression.pkl",
        "scaler.pkl",
        "feature_names.pkl"
    ]
    
    models_dir = project_root / "models"
    existing_models = []
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            existing_models.append(model_file)
            print(f"✓ Existing model: {model_file}")
    
    if len(existing_models) >= 3:  # At least 3 models should exist
        print(f"✓ Found {len(existing_models)} model files")
    else:
        print(f"⚠️  Only found {len(existing_models)} model files - train models first")
        return False
    
    print("\n✓ Environment setup validation passed!")
    return True

def test_prediction_generator_import():
    """Test if PredictionGenerator can be imported"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION GENERATOR IMPORT")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionGenerator, PredictionDatabase, DashboardDataPreparation
        print("✓ Successfully imported PredictionGenerator")
        print("✓ Successfully imported PredictionDatabase")
        print("✓ Successfully imported DashboardDataPreparation")
        return True
    except ImportError as e:
        print(f"✗ Failed to import PredictionGenerator: {e}")
        return False

def test_prediction_database():
    """Test prediction database functionality"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION DATABASE")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionDatabase
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_db_path = Path(tmp.name)
        
        try:
            # Test database initialization
            db = PredictionDatabase(tmp_db_path)
            print("✓ Database initialization successful")
            
            # Test saving prediction
            test_prediction = {
                'prediction_date': '2025-07-30',
                'target_date': '2025-07-31',
                'prediction_type': 'direction',
                'prediction_value': 1.0,
                'confidence': 0.75,
                'model_used': 'test_model',
                'metadata': {'test': True}
            }
            
            if db.save_prediction(test_prediction):
                print("✓ Prediction saving successful")
            else:
                print("✗ Prediction saving failed")
                return False
            
            # Test retrieving predictions
            recent_predictions = db.get_recent_predictions(30)
            if len(recent_predictions) > 0:
                print(f"✓ Retrieved {len(recent_predictions)} predictions")
            else:
                print("✗ No predictions retrieved")
                return False
            
            # Test updating actual values
            if db.update_actual_values('2025-07-31', 1.0):
                print("✓ Actual value update successful")
            else:
                print("✗ Actual value update failed")
                return False
            
            print("\n✓ Prediction database tests passed!")
            return True
            
        finally:
            # Cleanup
            if tmp_db_path.exists():
                tmp_db_path.unlink()
        
    except Exception as e:
        print(f"✗ Prediction database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_functionality():
    """Test the underlying predictor functionality"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTOR FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Import predictor directly
        sys.path.append(str(project_root / 'src'))
        from predict import SP500Predictor
        
        predictor = SP500Predictor()
        print("✓ SP500Predictor initialized")
        
        # Test prediction generation
        prediction = predictor.predict_next_day()
        
        if prediction is not None:
            print(f"✓ Prediction generated: {prediction}")
            
            # Validate prediction format
            if isinstance(prediction, (int, float)):
                if prediction in [0, 1] or (0 <= prediction <= 1):
                    print("✓ Prediction value is valid")
                else:
                    print(f"⚠️  Prediction value {prediction} outside expected range [0,1]")
            elif isinstance(prediction, dict):
                if 'prediction' in prediction or 'direction' in prediction:
                    print("✓ Prediction format is valid (dictionary)")
                else:
                    print("⚠️  Prediction dictionary missing expected keys")
            else:
                print(f"⚠️  Unexpected prediction type: {type(prediction)}")
            
            return True
        else:
            print("✗ Predictor returned None")
            return False
            
    except Exception as e:
        print(f"✗ Predictor functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test PredictionGenerator configuration"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionGenerator
        
        # Test default configuration
        generator = PredictionGenerator()
        config = generator.config
        
        print("Default configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Validate required config keys
        required_keys = [
            'prediction_horizons', 'prediction_types', 'confidence_threshold',
            'max_prediction_age_days', 'enable_dashboard_update'
        ]
        
        for key in required_keys:
            if key in config:
                print(f"✓ Required config key present: {key}")
            else:
                print(f"✗ Missing required config key: {key}")
                return False
        
        # Test custom configuration
        custom_config = {
            'prediction_horizons': [1, 2],
            'prediction_types': ['direction'],
            'confidence_threshold': 0.8,
            'enable_dashboard_update': False
        }
        
        custom_generator = PredictionGenerator(config=custom_config)
        if custom_generator.config['confidence_threshold'] == 0.8:
            print("✓ Custom configuration loaded successfully")
        else:
            print("✗ Custom configuration not applied correctly")
            return False
        
        print("\n✓ Configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_validation():
    """Test model validation functionality"""
    print("\n" + "=" * 60)
    print("TESTING MODEL VALIDATION")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionGenerator
        
        generator = PredictionGenerator()
        
        # Test model validation
        is_valid, message = generator.validate_models()
        
        if is_valid:
            print(f"✓ Model validation passed: {message}")
        else:
            print(f"✗ Model validation failed: {message}")
            return False
        
        print("\n✓ Model validation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_generation():
    """Test prediction generation functionality"""
    print("\n" + "=" * 60)
    print("TESTING PREDICTION GENERATION")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionGenerator
        
        # Create generator with test configuration
        config = {
            'prediction_horizons': [1],
            'prediction_types': ['direction'],
            'confidence_threshold': 0.0,  # Lower threshold for testing
            'validate_predictions': True,
            'update_actual_values': False,  # Skip for testing
            'enable_dashboard_update': False  # Skip for testing
        }
        
        generator = PredictionGenerator(config=config)
        
        # Test prediction generation
        predictions = generator.generate_predictions()
        
        if predictions:
            print(f"✓ Generated {len(predictions)} predictions")
            
            for key, prediction in predictions.items():
                print(f"  {key}: {prediction['prediction_value']} (confidence: {prediction.get('confidence', 'N/A')})")
            
            # Validate prediction structure
            first_prediction = list(predictions.values())[0]
            required_fields = ['prediction_date', 'target_date', 'prediction_type', 'prediction_value']
            
            for field in required_fields:
                if field in first_prediction:
                    print(f"✓ Prediction contains required field: {field}")
                else:
                    print(f"✗ Prediction missing required field: {field}")
                    return False
            
        else:
            print("✗ No predictions generated")
            return False
        
        print("\n✓ Prediction generation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Prediction generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_data_preparation():
    """Test dashboard data preparation"""
    print("\n" + "=" * 60)
    print("TESTING DASHBOARD DATA PREPARATION")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionDatabase, DashboardDataPreparation
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_db_path = Path(tmp.name)
        
        try:
            # Create test database with some data
            db = PredictionDatabase(tmp_db_path)
            
            # Add test prediction
            test_prediction = {
                'prediction_date': '2025-07-30',
                'target_date': '2025-07-31',
                'prediction_type': 'direction',
                'prediction_value': 1.0,
                'confidence': 0.75,
                'model_used': 'test_model'
            }
            db.save_prediction(test_prediction)
            
            # Test dashboard data preparation
            dashboard_prep = DashboardDataPreparation(project_root / "data", db)
            
            dashboard_data = dashboard_prep.prepare_dashboard_data()
            
            if dashboard_data:
                print("✓ Dashboard data preparation successful")
                
                # Check required sections
                required_sections = ['last_updated', 'historical_data', 'predictions', 'performance_metrics']
                for section in required_sections:
                    if section in dashboard_data:
                        print(f"✓ Dashboard contains section: {section}")
                    else:
                        print(f"⚠️  Dashboard missing section: {section}")
                
                # Check if cache file was created
                cache_file = project_root / "data" / "cache" / "dashboard_data.json"
                if cache_file.exists():
                    print("✓ Dashboard cache file created")
                else:
                    print("⚠️  Dashboard cache file not created")
            else:
                print("✗ Dashboard data preparation failed")
                return False
        
        finally:
            # Cleanup
            if tmp_db_path.exists():
                tmp_db_path.unlink()
        
        print("\n✓ Dashboard data preparation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dashboard data preparation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dry_run():
    """Test a dry run of the prediction pipeline"""
    print("\n" + "=" * 60)
    print("TESTING DRY RUN")
    print("=" * 60)
    
    try:
        from prediction_generator import PredictionGenerator
        
        # Configure for dry run
        config = {
            'prediction_horizons': [1],
            'prediction_types': ['direction'],
            'confidence_threshold': 0.0,  # Lower threshold for testing
            'max_prediction_age_days': 90,
            'enable_dashboard_update': False,  # Disable for testing
            'notification_email': None,  # Disable notifications
            'validate_predictions': True,
            'update_actual_values': False  # Disable for testing
        }
        
        generator = PredictionGenerator(config=config)
        
        print("Dry run configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nThis would perform the following steps:")
        print("1. Validate models")
        print("2. Generate predictions")
        print("3. Update actual values (disabled)")
        print("4. Prepare dashboard data (disabled)")
        print("5. Cleanup old predictions")
        
        print("\n✓ Dry run configuration completed!")
        print("\nTo run actual prediction generation:")
        print("  python scheduler/prediction_generator.py")
        print("To test without dashboard updates:")
        print("  python scheduler/prediction_generator.py --no-dashboard")
        print("To test models only:")
        print("  python scheduler/prediction_generator.py --test-only")
        
        return True
        
    except Exception as e:
        print(f"✗ Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_integration():
    """Test integration with existing data"""
    print("\n" + "=" * 60)
    print("TESTING DATA INTEGRATION")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        # Check if processed data exists and is valid
        features_file = project_root / "data" / "processed" / "sp500_features.csv"
        
        if not features_file.exists():
            print("✗ Features file not found")
            return False
        
        features_df = pd.read_csv(features_file)
        print(f"✓ Loaded features data: {features_df.shape}")
        
        # Check required columns
        required_columns = ['date', 'close', 'target']
        missing_columns = [col for col in required_columns if col not in features_df.columns]
        
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        else:
            print("✓ All required columns present")
        
        # Check data recency
        features_df['date'] = pd.to_datetime(features_df['date'])
        latest_date = features_df['date'].max()
        days_old = (datetime.now() - latest_date).days
        
        print(f"Latest data date: {latest_date.date()}")
        print(f"Data age: {days_old} days")
        
        if days_old <= 7:
            print("✓ Data is recent")
        else:
            print("⚠️  Data is older than 7 days")
        
        # Check target distribution
        target_dist = features_df['target'].value_counts()
        print(f"Target distribution: {dict(target_dist)}")
        
        if len(target_dist) >= 2:
            print("✓ Target has multiple classes")
        else:
            print("⚠️  Target has only one class")
        
        print("\n✓ Data integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("PREDICTION GENERATOR TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Project root: {project_root}")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Prediction Generator Import", test_prediction_generator_import),
        ("Prediction Database", test_prediction_database),
        ("Predictor Functionality", test_predictor_functionality),
        ("Configuration", test_configuration),
        ("Model Validation", test_model_validation),
        ("Prediction Generation", test_prediction_generation),
        ("Dashboard Data Preparation", test_dashboard_data_preparation),
        ("Data Integration", test_data_integration),
        ("Dry Run", test_dry_run)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n🎉 {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Prediction generator is ready to use.")
        print("\nNext steps:")
        print("1. Test with: python scheduler/prediction_generator.py --test-only")
        print("2. Generate predictions: python scheduler/prediction_generator.py")
        print("3. Schedule regular predictions with cron")
        print("4. Monitor prediction database and dashboard cache")
    elif failed <= 2:
        print(f"\n⚠️  {failed} test(s) failed, but system may still be usable.")
        print("Check failed tests and address any critical issues.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please address issues before using prediction generator.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
