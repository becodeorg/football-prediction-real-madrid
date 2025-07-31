#!/usr/bin/env python3
"""
Docker Health Check Script for S&P 500 Prediction System
Monitors container health and service availability
"""

import sys
import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add application paths
sys.path.append('/app')
sys.path.append('/app/src')
sys.path.append('/app/scheduler')

def check_python_environment():
    """Check if Python environment is healthy"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            return False, "Python version too old"
        
        # Check required imports
        import pandas
        import numpy
        import sklearn
        import yfinance
        import apscheduler
        import sqlalchemy
        import psutil
        
        return True, "Python environment OK"
    except ImportError as e:
        return False, f"Missing package: {e}"
    except Exception as e:
        return False, f"Environment error: {e}"

def check_filesystem():
    """Check filesystem health"""
    try:
        # Check required directories
        required_dirs = [
            '/app/data',
            '/app/models', 
            '/app/logs',
            '/app/config'
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                return False, f"Missing directory: {dir_path}"
            
            if not os.access(dir_path, os.W_OK):
                return False, f"Directory not writable: {dir_path}"
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage('/app')
        usage_percent = (used / total) * 100
        
        if usage_percent > 95:
            return False, f"Disk usage critical: {usage_percent:.1f}%"
        
        return True, f"Filesystem OK (disk: {usage_percent:.1f}% used)"
    except Exception as e:
        return False, f"Filesystem error: {e}"

def check_memory():
    """Check memory usage"""
    try:
        import psutil
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            return False, f"Memory usage critical: {memory.percent:.1f}%"
        
        return True, f"Memory OK ({memory.percent:.1f}% used)"
    except Exception as e:
        return False, f"Memory check error: {e}"

def check_configuration():
    """Check configuration files"""
    try:
        config_paths = [
            '/app/config/scheduler_config.json',
            '/app/scheduler/scheduler_config.json'
        ]
        
        config_found = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Validate basic structure
                required_keys = ['scheduler_type', 'timezone', 'tasks']
                for key in required_keys:
                    if key not in config:
                        return False, f"Missing config key: {key}"
                
                config_found = True
                break
        
        if not config_found:
            return False, "No configuration file found"
        
        return True, "Configuration OK"
    except json.JSONDecodeError:
        return False, "Invalid JSON in configuration"
    except Exception as e:
        return False, f"Configuration error: {e}"

def check_database():
    """Check scheduler database"""
    try:
        db_paths = [
            '/app/data/scheduler_jobs.sqlite',
            '/app/scheduler_jobs.sqlite'
        ]
        
        for db_path in db_paths:
            if os.path.exists(db_path):
                # Try to connect to database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                
                return True, f"Database OK ({len(tables)} tables)"
        
        # Database doesn't exist yet, which is OK
        return True, "Database not initialized (OK for first run)"
    except Exception as e:
        return False, f"Database error: {e}"

def check_scheduler_process():
    """Check if scheduler process is running"""
    try:
        import psutil
        
        # Look for scheduler processes
        scheduler_found = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'scheduler_main.py' in cmdline:
                    scheduler_found = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if scheduler_found:
            return True, "Scheduler process running"
        else:
            return True, "Scheduler process not running (may be starting)"
    except Exception as e:
        return False, f"Process check error: {e}"

def check_recent_logs():
    """Check for recent log activity"""
    try:
        log_dir = Path('/app/logs')
        if not log_dir.exists():
            return True, "Logs directory not found (OK for first run)"
        
        # Look for recent log files
        recent_logs = []
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for log_file in log_dir.glob('*.log'):
            if log_file.stat().st_mtime > cutoff_time.timestamp():
                recent_logs.append(log_file.name)
        
        if recent_logs:
            return True, f"Recent log activity: {len(recent_logs)} files"
        else:
            return True, "No recent logs (OK for first run)"
    except Exception as e:
        return False, f"Log check error: {e}"

def check_data_files():
    """Check for required data files"""
    try:
        data_dir = Path('/app/data')
        if not data_dir.exists():
            return True, "Data directory not found (OK for first run)"
        
        # Check for data files
        raw_dir = data_dir / 'raw'
        processed_dir = data_dir / 'processed'
        
        raw_files = list(raw_dir.glob('*.csv')) if raw_dir.exists() else []
        processed_files = list(processed_dir.glob('*.csv')) if processed_dir.exists() else []
        
        return True, f"Data files: {len(raw_files)} raw, {len(processed_files)} processed"
    except Exception as e:
        return False, f"Data check error: {e}"

def run_health_checks():
    """Run all health checks"""
    checks = [
        ("Python Environment", check_python_environment),
        ("Filesystem", check_filesystem),
        ("Memory", check_memory),
        ("Configuration", check_configuration),
        ("Database", check_database),
        ("Scheduler Process", check_scheduler_process),
        ("Recent Logs", check_recent_logs),
        ("Data Files", check_data_files)
    ]
    
    results = []
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            results.append({
                'check': check_name,
                'status': 'PASS' if success else 'FAIL',
                'message': message
            })
            
            if not success:
                all_passed = False
                
        except Exception as e:
            results.append({
                'check': check_name,
                'status': 'ERROR',
                'message': str(e)
            })
            all_passed = False
    
    return all_passed, results

def main():
    """Main health check function"""
    try:
        # Run health checks
        all_passed, results = run_health_checks()
        
        # Print results
        print("S&P 500 Prediction System - Health Check")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Overall Status: {'HEALTHY' if all_passed else 'UNHEALTHY'}")
        print()
        
        for result in results:
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            print(f"{status_symbol} {result['check']}: {result['message']}")
        
        # Summary
        passed_count = sum(1 for r in results if r['status'] == 'PASS')
        total_count = len(results)
        
        print()
        print(f"Health Score: {passed_count}/{total_count} checks passed")
        
        # Exit with appropriate code
        if all_passed:
            print("Container is healthy")
            sys.exit(0)
        else:
            print("Container has health issues")
            sys.exit(1)
            
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
