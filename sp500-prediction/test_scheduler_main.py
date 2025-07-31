#!/usr/bin/env python3
"""
Test Script for Scheduler Orchestrator
Validates scheduler functionality and configuration

Usage:
    python test_scheduler_main.py
"""

import os
import sys
import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent  # This file is in sp500-prediction/
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
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            return False
    
    # Check required files
    required_files = [
        project_root / "scheduler" / "daily_data_update.py",
        project_root / "scheduler" / "model_retrainer.py",
        project_root / "scheduler" / "prediction_generator.py",
        project_root / "scheduler" / "scheduler_main.py"
    ]
    
    for file_path in required_files:
        if file_path.exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False
    
    # Check APScheduler dependency
    try:
        import apscheduler
        print(f"✓ APScheduler available: {apscheduler.__version__}")
    except ImportError:
        print("✗ APScheduler not installed")
        print("Install with: pip install apscheduler")
        return False
    
    print("\n✓ Environment setup validation passed!")
    return True

def test_scheduler_imports():
    """Test if scheduler components can be imported"""
    print("\n" + "=" * 60)
    print("TESTING SCHEDULER IMPORTS")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator, TaskDependencyManager, TaskStatus, TaskResult
        print("✓ Successfully imported SchedulerOrchestrator")
        print("✓ Successfully imported TaskDependencyManager")
        print("✓ Successfully imported TaskStatus")
        print("✓ Successfully imported TaskResult")
        return True
    except ImportError as e:
        print(f"✗ Failed to import scheduler components: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading and validation"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION LOADING")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator
        
        # Test default configuration
        orchestrator = SchedulerOrchestrator()
        config = orchestrator.config
        
        print("Default configuration loaded:")
        print(f"  Scheduler type: {config['scheduler_type']}")
        print(f"  Timezone: {config['timezone']}")
        print(f"  Max workers: {config['max_workers']}")
        print(f"  Enabled tasks: {len([t for t in config['tasks'].values() if t.get('enabled', True)])}")
        
        # Test required configuration keys
        required_keys = [
            'scheduler_type', 'timezone', 'max_workers', 'tasks', 
            'notifications', 'persistence'
        ]
        
        for key in required_keys:
            if key in config:
                print(f"✓ Required config key present: {key}")
            else:
                print(f"✗ Missing required config key: {key}")
                return False
        
        # Test task configuration
        required_tasks = ['daily_data_update', 'model_retraining', 'prediction_generation', 'health_check']
        
        for task in required_tasks:
            if task in config['tasks']:
                task_config = config['tasks'][task]
                if 'schedule' in task_config and 'enabled' in task_config:
                    print(f"✓ Task configured correctly: {task}")
                else:
                    print(f"✗ Task missing required fields: {task}")
                    return False
            else:
                print(f"✗ Missing task configuration: {task}")
                return False
        
        # Test custom configuration
        custom_config = {
            'scheduler_type': 'background',
            'timezone': 'US/Eastern',
            'tasks': {
                'daily_data_update': {
                    'enabled': False,
                    'schedule': {'type': 'cron', 'hour': 18}
                }
            }
        }
        
        custom_orchestrator = SchedulerOrchestrator(config=custom_config)
        if custom_orchestrator.config['timezone'] == 'US/Eastern':
            print("✓ Custom configuration loaded successfully")
        else:
            print("✗ Custom configuration not applied correctly")
            return False
        
        print("\n✓ Configuration loading tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_dependency_manager():
    """Test task dependency management"""
    print("\n" + "=" * 60)
    print("TESTING TASK DEPENDENCY MANAGER")
    print("=" * 60)
    
    try:
        from scheduler_main import TaskDependencyManager, TaskResult, TaskStatus
        
        # Create dependency manager
        dep_manager = TaskDependencyManager()
        
        # Test adding dependencies
        dep_manager.add_dependency('task_b', ['task_a'])
        dep_manager.add_dependency('task_c', ['task_a', 'task_b'])
        print("✓ Dependencies added successfully")
        
        # Test dependency checking - should fail initially
        if not dep_manager.can_run_task('task_b'):
            print("✓ Correctly blocked task_b (waiting for task_a)")
        else:
            print("✗ Failed to block task_b")
            return False
        
        # Record successful task_a result
        result_a = TaskResult(
            task_id='task_a',
            status=TaskStatus.SUCCESS,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        dep_manager.record_task_result(result_a)
        
        # Now task_b should be able to run
        if dep_manager.can_run_task('task_b'):
            print("✓ Correctly allowed task_b (task_a completed)")
        else:
            print("✗ Failed to allow task_b")
            return False
        
        # But task_c should still be blocked
        if not dep_manager.can_run_task('task_c'):
            print("✓ Correctly blocked task_c (waiting for task_b)")
        else:
            print("✗ Failed to block task_c")
            return False
        
        # Test status retrieval
        status = dep_manager.get_task_status('task_a')
        if status == TaskStatus.SUCCESS:
            print("✓ Task status retrieval working")
        else:
            print("✗ Task status retrieval failed")
            return False
        
        # Test reset
        dep_manager.reset_daily_results()
        if dep_manager.get_task_status('task_a') is None:
            print("✓ Daily reset working")
        else:
            print("✗ Daily reset failed")
            return False
        
        print("\n✓ Task dependency manager tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Task dependency manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler_setup():
    """Test scheduler setup and initialization"""
    print("\n" + "=" * 60)
    print("TESTING SCHEDULER SETUP")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator
        
        # Create test configuration
        test_config = {
            'scheduler_type': 'background',  # Use background for testing
            'timezone': 'UTC',
            'max_workers': 2,
            'tasks': {
                'test_task': {
                    'enabled': True,
                    'schedule': {
                        'type': 'interval',
                        'seconds': 30
                    },
                    'max_retries': 1
                }
            },
            'persistence': {
                'enabled': False  # Disable persistence for testing
            },
            'notifications': {
                'email_enabled': False  # Disable notifications for testing
            }
        }
        
        orchestrator = SchedulerOrchestrator(config=test_config)
        print("✓ Orchestrator created successfully")
        
        # Test scheduler setup (without starting)
        try:
            orchestrator._setup_scheduler()
            print("✓ Scheduler setup completed")
        except Exception as e:
            print(f"✗ Scheduler setup failed: {e}")
            return False
        
        # Test task dependency setup
        orchestrator._setup_task_dependencies()
        print("✓ Task dependencies setup completed")
        
        # Test component initialization (may fail if modules not available)
        try:
            orchestrator._initialize_task_components()
            print("✓ Task components initialized")
        except Exception as e:
            print(f"⚠️  Task components initialization failed (expected if src modules not available): {e}")
        
        print("\n✓ Scheduler setup tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Scheduler setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_job_scheduling():
    """Test job scheduling functionality"""
    print("\n" + "=" * 60)
    print("TESTING JOB SCHEDULING")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator
        from apscheduler.triggers.interval import IntervalTrigger
        
        # Create minimal test configuration
        test_config = {
            'scheduler_type': 'background',
            'timezone': 'UTC',
            'max_workers': 1,
            'job_defaults': {
                'coalesce': True,
                'max_instances': 1
            },
            'tasks': {
                'test_interval_task': {
                    'enabled': True,
                    'schedule': {
                        'type': 'interval',
                        'seconds': 5
                    },
                    'max_retries': 0
                }
            },
            'persistence': {'enabled': False},
            'notifications': {'email_enabled': False}
        }
        
        orchestrator = SchedulerOrchestrator(config=test_config)
        
        # Setup scheduler
        orchestrator._setup_scheduler()
        
        # Test adding a simple job
        def test_job():
            print(f"Test job executed at {datetime.now()}")
            return True
        
        # Add job manually for testing
        trigger = IntervalTrigger(seconds=10)
        orchestrator.scheduler.add_job(
            func=test_job,
            trigger=trigger,
            id='manual_test_job',
            name='Manual Test Job'
        )
        
        # Check if job was added
        jobs = orchestrator.scheduler.get_jobs()
        if len(jobs) > 0:
            print(f"✓ Job added successfully: {jobs[0].name}")
            print(f"  Next run: {jobs[0].next_run_time}")
        else:
            print("✗ No jobs found")
            return False
        
        # Test job removal
        orchestrator.scheduler.remove_job('manual_test_job')
        jobs_after_removal = orchestrator.scheduler.get_jobs()
        
        if len(jobs_after_removal) == 0:
            print("✓ Job removed successfully")
        else:
            print("✗ Job removal failed")
            return False
        
        print("\n✓ Job scheduling tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Job scheduling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_templates():
    """Test configuration template loading"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION TEMPLATES")
    print("=" * 60)
    
    try:
        # Test loading template configuration
        template_file = project_root / "scheduler" / "scheduler_config.template.json"
        
        if template_file.exists():
            with open(template_file, 'r') as f:
                template_config = json.load(f)
            
            print("✓ Template configuration loaded")
            print(f"  Scheduler type: {template_config.get('scheduler_type')}")
            print(f"  Tasks defined: {len(template_config.get('tasks', {}))}")
            
            # Test creating orchestrator with template
            from scheduler_main import SchedulerOrchestrator
            
            # Modify template for testing
            test_template = template_config.copy()
            test_template['scheduler_type'] = 'background'
            test_template['persistence']['enabled'] = False
            test_template['notifications']['email_enabled'] = False
            
            orchestrator = SchedulerOrchestrator(config=test_template)
            print("✓ Orchestrator created with template configuration")
            
        else:
            print("✗ Template configuration file not found")
            return False
        
        print("\n✓ Configuration template tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_status_reporting():
    """Test status reporting functionality"""
    print("\n" + "=" * 60)
    print("TESTING STATUS REPORTING")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator
        
        # Create test orchestrator
        test_config = {
            'scheduler_type': 'background',
            'timezone': 'UTC',
            'tasks': {
                'test_task': {
                    'enabled': True,
                    'schedule': {'type': 'interval', 'minutes': 10}
                }
            },
            'persistence': {'enabled': False},
            'notifications': {'email_enabled': False}
        }
        
        orchestrator = SchedulerOrchestrator(config=test_config)
        orchestrator._setup_scheduler()
        
        # Test status reporting
        status = orchestrator.get_status()
        
        required_status_keys = ['is_running', 'scheduler_type', 'timezone', 'total_jobs', 'jobs', 'task_stats']
        
        for key in required_status_keys:
            if key in status:
                print(f"✓ Status contains key: {key}")
            else:
                print(f"✗ Status missing key: {key}")
                return False
        
        print(f"Status details:")
        print(f"  Running: {status['is_running']}")
        print(f"  Type: {status['scheduler_type']}")
        print(f"  Timezone: {status['timezone']}")
        print(f"  Total jobs: {status['total_jobs']}")
        
        print("\n✓ Status reporting tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Status reporting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dry_run():
    """Test dry run functionality"""
    print("\n" + "=" * 60)
    print("TESTING DRY RUN")
    print("=" * 60)
    
    try:
        from scheduler_main import SchedulerOrchestrator
        
        # Create comprehensive test configuration
        test_config = {
            'scheduler_type': 'background',
            'timezone': 'UTC',
            'max_workers': 2,
            'tasks': {
                'daily_data_update': {
                    'enabled': True,
                    'schedule': {
                        'type': 'cron',
                        'hour': 22,
                        'minute': 0,
                        'day_of_week': 'mon-fri'
                    },
                    'max_retries': 2
                },
                'prediction_generation': {
                    'enabled': True,
                    'schedule': {
                        'type': 'cron',
                        'hour': 23,
                        'minute': 0,
                        'day_of_week': 'mon-fri'
                    },
                    'depends_on': ['daily_data_update'],
                    'max_retries': 1
                }
            },
            'persistence': {'enabled': False},
            'notifications': {'email_enabled': False}
        }
        
        orchestrator = SchedulerOrchestrator(config=test_config)
        
        print("Dry run configuration:")
        print(f"  Scheduler type: {test_config['scheduler_type']}")
        print(f"  Max workers: {test_config['max_workers']}")
        print(f"  Tasks: {list(test_config['tasks'].keys())}")
        
        print("\nThis configuration would schedule:")
        for task_name, task_config in test_config['tasks'].items():
            schedule = task_config['schedule']
            if schedule['type'] == 'cron':
                print(f"  {task_name}: {schedule.get('hour', '*')}:{schedule.get('minute', 0):02d} on {schedule.get('day_of_week', 'daily')}")
            elif schedule['type'] == 'interval':
                interval_parts = []
                if schedule.get('hours'): interval_parts.append(f"{schedule['hours']}h")
                if schedule.get('minutes'): interval_parts.append(f"{schedule['minutes']}m")
                if schedule.get('seconds'): interval_parts.append(f"{schedule['seconds']}s")
                print(f"  {task_name}: every {' '.join(interval_parts)}")
        
        print("\n✓ Dry run completed successfully!")
        print("\nTo run the actual scheduler:")
        print("  python scheduler/scheduler_main.py")
        print("To test configuration:")
        print("  python scheduler/scheduler_main.py --test-config")
        print("To run in daemon mode:")
        print("  python scheduler/scheduler_main.py --daemon")
        
        return True
        
    except Exception as e:
        print(f"✗ Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("SCHEDULER ORCHESTRATOR TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Project root: {project_root}")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Scheduler Imports", test_scheduler_imports),
        ("Configuration Loading", test_configuration_loading),
        ("Task Dependency Manager", test_task_dependency_manager),
        ("Scheduler Setup", test_scheduler_setup),
        ("Job Scheduling", test_job_scheduling),
        ("Configuration Templates", test_configuration_templates),
        ("Status Reporting", test_status_reporting),
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
        print("\n🎉 ALL TESTS PASSED! Scheduler orchestrator is ready to use.")
        print("\nNext steps:")
        print("1. Configure scheduler: cp scheduler/scheduler_config.template.json scheduler/scheduler_config.json")
        print("2. Test configuration: python scheduler/scheduler_main.py --test-config")
        print("3. Start scheduler: python scheduler/scheduler_main.py")
        print("4. Monitor with: tail -f logs/scheduler_main_*.log")
    elif failed <= 2:
        print(f"\n⚠️  {failed} test(s) failed, but system may still be usable.")
        print("Check failed tests and address any critical issues.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please address issues before using scheduler orchestrator.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
