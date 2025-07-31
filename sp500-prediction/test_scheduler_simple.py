#!/usr/bin/env python3
"""
Simplified Test Script for Scheduler System
Tests core functionality without full imports

Usage:
    python test_scheduler_simple.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'scheduler'))
sys.path.append(str(project_root / 'src'))

def test_environment_setup():
    """Test if the environment is properly set up"""
    print("=" * 60)
    print("TESTING ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check project structure
    required_dirs = [
        project_root / "data",
        project_root / "models", 
        project_root / "src",
        project_root / "scheduler",
        project_root / "logs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_path.name}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            missing_dirs.append(dir_path)
    
    # Check required files
    required_files = [
        project_root / "scheduler" / "daily_data_update.py",
        project_root / "scheduler" / "model_retrainer.py", 
        project_root / "scheduler" / "prediction_generator.py",
        project_root / "scheduler" / "scheduler_main.py",
        project_root / "scheduler" / "scheduler_config.template.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ File exists: {file_path.name}")
        else:
            print(f"✗ Missing file: {file_path}")
            missing_files.append(file_path)
    
    # Check dependencies
    dependencies = {
        'apscheduler': 'APScheduler',
        'sqlalchemy': 'SQLAlchemy', 
        'psutil': 'psutil',
        'yfinance': 'yfinance',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }
    
    missing_deps = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not installed")
            missing_deps.append(name)
    
    # Summary
    if missing_dirs or missing_files or missing_deps:
        print(f"\n❌ Environment setup issues found:")
        if missing_dirs:
            print(f"  Missing directories: {[d.name for d in missing_dirs]}")
        if missing_files: 
            print(f"  Missing files: {[f.name for f in missing_files]}")
        if missing_deps:
            print(f"  Missing dependencies: {missing_deps}")
        return False
    else:
        print("\n✓ Environment setup validation passed!")
        return True

def test_configuration_template():
    """Test configuration template"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION TEMPLATE")
    print("=" * 60)
    
    template_file = project_root / "scheduler" / "scheduler_config.template.json"
    
    if not template_file.exists():
        print("✗ Template configuration file not found")
        return False
    
    try:
        with open(template_file, 'r') as f:
            config = json.load(f)
        
        print("✓ Template configuration loaded")
        print(f"  Scheduler type: {config.get('scheduler_type', 'Not set')}")
        print(f"  Timezone: {config.get('timezone', 'Not set')}")
        print(f"  Max workers: {config.get('max_workers', 'Not set')}")
        
        # Check required sections
        required_sections = ['scheduler_type', 'timezone', 'max_workers', 'tasks', 'notifications', 'persistence']
        missing_sections = []
        
        for section in required_sections:
            if section in config:
                print(f"✓ Configuration section present: {section}")
            else:
                print(f"✗ Missing configuration section: {section}")
                missing_sections.append(section)
        
        # Check tasks
        if 'tasks' in config:
            tasks = config['tasks']
            print(f"  Tasks configured: {len(tasks)}")
            
            for task_name, task_config in tasks.items():
                if 'enabled' in task_config and 'schedule' in task_config:
                    print(f"    ✓ {task_name}: properly configured")
                else:
                    print(f"    ✗ {task_name}: missing required fields")
        
        if missing_sections:
            print(f"\n❌ Missing configuration sections: {missing_sections}")
            return False
        else:
            print("\n✓ Configuration template validation passed!")
            return True
            
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in template: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading template: {e}")
        return False

def test_file_syntax():
    """Test if Python files have valid syntax"""
    print("\n" + "=" * 60)
    print("TESTING FILE SYNTAX")
    print("=" * 60)
    
    scheduler_files = [
        "daily_data_update.py",
        "model_retrainer.py", 
        "prediction_generator.py",
        "scheduler_main.py"
    ]
    
    syntax_errors = []
    
    for filename in scheduler_files:
        filepath = project_root / "scheduler" / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    source = f.read()
                compile(source, str(filepath), 'exec')
                print(f"✓ {filename}: syntax OK")
            except SyntaxError as e:
                print(f"✗ {filename}: syntax error at line {e.lineno}: {e.msg}")
                syntax_errors.append((filename, e.lineno, e.msg))
            except Exception as e:
                print(f"✗ {filename}: error reading file: {e}")
                syntax_errors.append((filename, 0, str(e)))
        else:
            print(f"✗ {filename}: file not found")
            syntax_errors.append((filename, 0, "File not found"))
    
    if syntax_errors:
        print(f"\n❌ Syntax errors found:")
        for filename, lineno, msg in syntax_errors:
            print(f"  {filename}:{lineno} - {msg}")
        return False
    else:
        print("\n✓ All Python files have valid syntax!")
        return True

def test_directory_structure():
    """Test directory structure and permissions"""
    print("\n" + "=" * 60)
    print("TESTING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    # Check data directories
    data_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "processed"
    ]
    
    for dir_path in data_dirs:
        if dir_path.exists():
            print(f"✓ Data directory exists: {dir_path.relative_to(project_root)}")
            
            # Check write permissions
            if os.access(dir_path, os.W_OK):
                print(f"  ✓ Writable")
            else:
                print(f"  ✗ Not writable")
        else:
            print(f"✗ Missing data directory: {dir_path.relative_to(project_root)}")
            
    # Check logs directory
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        print(f"✓ Logs directory exists")
        if os.access(logs_dir, os.W_OK):
            print(f"  ✓ Writable")
        else:
            print(f"  ✗ Not writable")
    else:
        print(f"✗ Missing logs directory")
        
    # Check models directory
    models_dir = project_root / "models"
    if models_dir.exists():
        print(f"✓ Models directory exists")
        model_files = list(models_dir.glob("*.pkl"))
        print(f"  Model files found: {len(model_files)}")
        for model_file in model_files:
            print(f"    {model_file.name}")
    else:
        print(f"✗ Missing models directory")
    
    print("\n✓ Directory structure check completed!")
    return True

def test_scheduler_config():
    """Test creating a configuration file"""
    print("\n" + "=" * 60)
    print("TESTING SCHEDULER CONFIGURATION")
    print("=" * 60)
    
    # Check if config file exists
    config_file = project_root / "scheduler" / "scheduler_config.json"
    template_file = project_root / "scheduler" / "scheduler_config.template.json"
    
    if config_file.exists():
        print("✓ scheduler_config.json exists")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("✓ Configuration file is valid JSON")
            print(f"  Scheduler type: {config.get('scheduler_type')}")
            print(f"  Tasks enabled: {len([t for t in config.get('tasks', {}).values() if t.get('enabled', True)])}")
        except Exception as e:
            print(f"✗ Error reading config file: {e}")
            return False
    else:
        print("⚠️  scheduler_config.json does not exist")
        if template_file.exists():
            print("✓ Template file available for copying")
            print("  Run: cp scheduler/scheduler_config.template.json scheduler/scheduler_config.json")
        else:
            print("✗ Template file also missing")
            return False
    
    print("\n✓ Configuration check completed!")
    return True

def test_installation_readiness():
    """Test if system is ready for scheduler installation"""
    print("\n" + "=" * 60)
    print("TESTING INSTALLATION READINESS")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 8:
        print("✓ Python version compatible")
    else:
        print("✗ Python version too old (requires 3.8+)")
        return False
    
    # Check basic imports that scheduler needs
    critical_imports = {
        'datetime': 'datetime',
        'json': 'json',
        'logging': 'logging', 
        'os': 'os',
        'sys': 'sys',
        'pathlib': 'pathlib',
        'threading': 'threading',
        'time': 'time'
    }
    
    for module, name in critical_imports.items():
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not available")
            return False
    
    # Check if we can create test files
    try:
        test_file = project_root / "logs" / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("✓ Can write to logs directory")
    except Exception as e:
        print(f"✗ Cannot write to logs directory: {e}")
        return False
    
    print("\n✓ System is ready for scheduler installation!")
    return True

def main():
    """Run all tests"""
    print("SCHEDULER SYSTEM VALIDATION")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Configuration Template", test_configuration_template),
        ("File Syntax", test_file_syntax),
        ("Directory Structure", test_directory_structure),
        ("Scheduler Configuration", test_scheduler_config),
        ("Installation Readiness", test_installation_readiness)
    ]
    
    passed = 0
    failed = 0
    warnings = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n🎉 {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nYour scheduler system is ready to use!")
        print("\nNext steps:")
        print("1. Configure: cp scheduler/scheduler_config.template.json scheduler/scheduler_config.json")
        print("2. Edit configuration: nano scheduler/scheduler_config.json")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Test run: python scheduler/scheduler_main.py --test-config")
        print("5. Start scheduler: python scheduler/scheduler_main.py")
    elif failed <= 2:
        print(f"\n⚠️  {failed} test(s) failed, but system may still be usable.")
        print("Review the failed tests and address critical issues.")
    else:
        print(f"\n❌ {failed} test(s) failed.")
        print("Please address the issues before using the scheduler system.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
