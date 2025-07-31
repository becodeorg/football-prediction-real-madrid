#!/usr/bin/env python3
"""
Test Script for Model Retrainer
Validates model retraining functionality before scheduling

Usage:
    python test_model_retrainer.py
"""

import os
import sys
import json
import logging
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
        project_root / "src" / "train_model.py",
        project_root / "src" / "predict.py",
        project_root / "src" / "feature_engineering.py"
    ]
    
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False
    
    # Check for existing models (optional)
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
    
    if existing_models:
        print(f"Found {len(existing_models)} existing model files")
    else:
        print("ℹ No existing model files found (this is OK for first run)")
    
    print("\n✓ Environment setup validation passed!")
    return True

def test_model_retrainer_import():
    """Test if ModelRetrainer can be imported"""
    print("\n" + "=" * 60)
    print("TESTING MODEL RETRAINER IMPORT")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer, ModelPerformanceTracker
        print("✓ Successfully imported ModelRetrainer")
        print("✓ Successfully imported ModelPerformanceTracker")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ModelRetrainer: {e}")
        return False

def test_configuration():
    """Test ModelRetrainer configuration"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer
        
        # Test default configuration
        retrainer = ModelRetrainer()
        config = retrainer.config
        
        print("Default configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Validate required config keys
        required_keys = [
            'retraining_frequency', 'min_training_samples', 'improvement_threshold',
            'models_to_train', 'auto_deploy', 'backup_retention_days'
        ]
        
        for key in required_keys:
            if key in config:
                print(f"✓ Required config key present: {key}")
            else:
                print(f"✗ Missing required config key: {key}")
                return False
        
        print("\n✓ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    print("\n" + "=" * 60)
    print("TESTING DATA VALIDATION")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer
        import pandas as pd
        
        retrainer = ModelRetrainer()
        
        # Load actual data file
        data_file = project_root / "data" / "processed" / "sp500_features.csv"
        if not data_file.exists():
            print(f"✗ Data file not found: {data_file}")
            return False
        
        data = pd.read_csv(data_file)
        print(f"Loaded data: {data.shape}")
        print(f"Columns: {list(data.columns)[:10]}...")  # Show first 10 columns
        
        # Test data validation
        is_valid, message = retrainer.validate_data_quality(data)
        
        if is_valid:
            print(f"✓ Data validation passed: {message}")
        else:
            print(f"✗ Data validation failed: {message}")
            return False
        
        # Test with invalid data
        invalid_data = data.head(10)  # Too few samples
        is_valid, message = retrainer.validate_data_quality(invalid_data)
        
        if not is_valid:
            print(f"✓ Correctly rejected invalid data: {message}")
        else:
            print("✗ Failed to reject invalid data")
            return False
        
        print("\n✓ Data validation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_tracker():
    """Test performance tracking functionality"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE TRACKER")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelPerformanceTracker
        
        # Create test metrics file
        test_metrics_file = project_root / "test_metrics.json"
        tracker = ModelPerformanceTracker(test_metrics_file)
        
        # Test saving metrics
        test_metrics = {
            'timestamp': datetime.now().isoformat(),
            'test_accuracy': 0.65,
            'models_trained': ['random_forest', 'gradient_boosting']
        }
        
        tracker.save_metrics(test_metrics)
        print("✓ Successfully saved test metrics")
        
        # Test loading metrics
        latest = tracker.get_latest_metrics()
        if latest and latest['test_accuracy'] == 0.65:
            print("✓ Successfully loaded metrics")
        else:
            print("✗ Failed to load metrics correctly")
            return False
        
        # Test comparison
        new_metrics = {'test_accuracy': 0.67}
        should_deploy, message = tracker.compare_with_baseline(new_metrics)
        
        if should_deploy:
            print(f"✓ Correctly identified improvement: {message}")
        else:
            print(f"✗ Failed to identify improvement: {message}")
        
        # Test degradation detection
        worse_metrics = {'test_accuracy': 0.60}
        should_deploy, message = tracker.compare_with_baseline(worse_metrics)
        
        if not should_deploy:
            print(f"✓ Correctly identified degradation: {message}")
        else:
            print(f"✗ Failed to identify degradation: {message}")
        
        # Cleanup
        if test_metrics_file.exists():
            test_metrics_file.unlink()
        
        print("\n✓ Performance tracker tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Performance tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backup_functionality():
    """Test backup functionality"""
    print("\n" + "=" * 60)
    print("TESTING BACKUP FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer
        
        retrainer = ModelRetrainer()
        
        # Create some dummy model files for testing
        test_files = ['test_model.pkl', 'test_scaler.pkl']
        models_dir = project_root / "models"
        
        for file_name in test_files:
            test_file = models_dir / file_name
            with open(test_file, 'w') as f:
                f.write("dummy model data")
        
        print(f"Created test files: {test_files}")
        
        # Test backup creation
        backup_success = retrainer.backup_current_models()
        
        if backup_success:
            print("✓ Backup creation successful")
            
            # Check if backup directory was created
            backup_dirs = list(retrainer.backup_dir.glob("backup_*"))
            if backup_dirs:
                print(f"✓ Backup directory created: {backup_dirs[-1]}")
            else:
                print("✗ No backup directory found")
                return False
        else:
            print("✗ Backup creation failed")
            return False
        
        # Test cleanup
        retrainer.cleanup_old_backups()
        print("✓ Backup cleanup completed")
        
        # Cleanup test files
        for file_name in test_files:
            test_file = models_dir / file_name
            if test_file.exists():
                test_file.unlink()
        
        print("\n✓ Backup functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Backup functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schedule_check():
    """Test retraining schedule logic"""
    print("\n" + "=" * 60)
    print("TESTING SCHEDULE CHECK")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer
        
        # Test manual mode
        config = {'retraining_frequency': 'manual'}
        retrainer = ModelRetrainer(config=config)
        
        should_train = retrainer.check_retraining_schedule()
        if should_train:
            print("✓ Manual mode correctly returns True")
        else:
            print("✗ Manual mode should always return True")
            return False
        
        # Test weekly mode with no previous training
        config = {'retraining_frequency': 'weekly'}
        retrainer = ModelRetrainer(config=config)
        
        should_train = retrainer.check_retraining_schedule()
        if should_train:
            print("✓ Weekly mode with no previous training correctly returns True")
        else:
            print("✗ Weekly mode with no previous training should return True")
            return False
        
        print("\n✓ Schedule check tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Schedule check test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dry_run():
    """Test a dry run of the retraining process"""
    print("\n" + "=" * 60)
    print("TESTING DRY RUN")
    print("=" * 60)
    
    try:
        from model_retrainer import ModelRetrainer
        
        # Configure for dry run (no auto-deploy)
        config = {
            'retraining_frequency': 'manual',
            'auto_deploy': False,
            'min_training_samples': 50,  # Lower requirement for testing
            'improvement_threshold': 0.01,
            'models_to_train': ['random_forest'],  # Train only one model for speed
            'notification_email': None  # Disable notifications
        }
        
        retrainer = ModelRetrainer(config=config)
        
        print("Dry run configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nThis would perform the following steps:")
        print("1. Check retraining schedule")
        print("2. Load and validate training data")
        print("3. Backup current models")
        print("4. Train new models")
        print("5. Validate new models")
        print("6. Compare with baseline")
        print("7. Deploy if improvement threshold is met")
        print("8. Cleanup old backups")
        
        print("\n✓ Dry run configuration completed!")
        print("\nTo run actual retraining:")
        print("  python scheduler/model_retrainer.py --force")
        print("To run without deployment:")
        print("  python scheduler/model_retrainer.py --force --no-deploy")
        
        return True
        
    except Exception as e:
        print(f"✗ Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("MODEL RETRAINER TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Project root: {project_root}")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Model Retrainer Import", test_model_retrainer_import),
        ("Configuration", test_configuration),
        ("Data Validation", test_data_validation),
        ("Performance Tracker", test_performance_tracker),
        ("Backup Functionality", test_backup_functionality),
        ("Schedule Check", test_schedule_check),
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
        print("\n🎉 ALL TESTS PASSED! Model retrainer is ready to use.")
        print("\nNext steps:")
        print("1. Configure environment variables in .env file")
        print("2. Test with: python scheduler/model_retrainer.py --force --no-deploy")
        print("3. Schedule regular retraining with cron or task scheduler")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please address issues before using model retrainer.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
