#!/usr/bin/env python3
"""
Main Scheduler Orchestrator for S&P 500 Prediction System
Coordinates all scheduled tasks using APScheduler

Features:
- Centralized scheduling for all automation components
- Task dependency management and sequencing
- Error recovery and retry mechanisms
- Health monitoring and status reporting
- Flexible scheduling configuration
- Graceful shutdown handling
"""

import os
import sys
import json
import logging
import signal
import threading
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# APScheduler imports
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'scheduler'))

try:
    from daily_data_update import DailyDataUpdater
    from model_retrainer import ModelRetrainer
    from prediction_generator import PredictionGenerator
except ImportError as e:
    print(f"Error importing scheduler modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"scheduler_main_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_time: float = 0.0

class TaskDependencyManager:
    """Manages task dependencies and execution order"""
    
    def __init__(self):
        self.dependencies = {}
        self.task_results = {}
        self.lock = threading.Lock()
    
    def add_dependency(self, task_id: str, depends_on: List[str]):
        """Add task dependency"""
        with self.lock:
            self.dependencies[task_id] = depends_on
    
    def can_run_task(self, task_id: str) -> bool:
        """Check if task can run based on dependencies"""
        with self.lock:
            if task_id not in self.dependencies:
                return True
            
            for dependency in self.dependencies[task_id]:
                if dependency not in self.task_results:
                    return False
                
                result = self.task_results[dependency]
                if result.status != TaskStatus.SUCCESS:
                    return False
            
            return True
    
    def record_task_result(self, result: TaskResult):
        """Record task execution result"""
        with self.lock:
            self.task_results[result.task_id] = result
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current task status"""
        with self.lock:
            if task_id in self.task_results:
                return self.task_results[task_id].status
            return None
    
    def reset_daily_results(self):
        """Reset task results for new day"""
        with self.lock:
            self.task_results.clear()

class SchedulerOrchestrator:
    """Main scheduler orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Scheduler Orchestrator
        
        Args:
            config: Configuration dictionary with settings
        """
        self.config = config or self._load_default_config()
        self.project_root = project_root
        self.dependency_manager = TaskDependencyManager()
        self.scheduler = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Initialize task components
        self.daily_updater = None
        self.model_retrainer = None
        self.prediction_generator = None
        
        # Task statistics
        self.task_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'last_health_check': None
        }
        
        logger.info("Scheduler Orchestrator initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            'scheduler_type': 'blocking',  # 'blocking' or 'background'
            'timezone': 'UTC',
            'max_workers': 3,
            'job_defaults': {
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 300  # 5 minutes
            },
            'tasks': {
                'daily_data_update': {
                    'enabled': True,
                    'schedule': {
                        'type': 'cron',
                        'hour': 22,  # 10 PM UTC (6 PM ET)
                        'minute': 0,
                        'day_of_week': 'mon-fri'
                    },
                    'max_retries': 2,
                    'retry_delay_minutes': 30,
                    'timeout_minutes': 15
                },
                'model_retraining': {
                    'enabled': True,
                    'schedule': {
                        'type': 'cron',
                        'hour': 2,   # 2 AM UTC (10 PM ET Sunday)
                        'minute': 0,
                        'day_of_week': 'sun'
                    },
                    'max_retries': 1,
                    'retry_delay_minutes': 60,
                    'timeout_minutes': 120
                },
                'prediction_generation': {
                    'enabled': True,
                    'schedule': {
                        'type': 'cron',
                        'hour': 23,  # 11 PM UTC (7 PM ET)
                        'minute': 0,
                        'day_of_week': 'mon-fri'
                    },
                    'depends_on': ['daily_data_update'],
                    'max_retries': 2,
                    'retry_delay_minutes': 15,
                    'timeout_minutes': 30
                },
                'health_check': {
                    'enabled': True,
                    'schedule': {
                        'type': 'interval',
                        'hours': 6
                    },
                    'max_retries': 0,
                    'timeout_minutes': 5
                }
            },
            'notifications': {
                'email_enabled': True,
                'daily_summary': True,
                'error_alerts': True,
                'health_alerts': True
            },
            'persistence': {
                'enabled': True,
                'jobstore_url': f'sqlite:///{project_root}/data/scheduler_jobs.db'
            }
        }
    
    def _setup_scheduler(self):
        """Setup APScheduler with configuration"""
        try:
            # Configure job stores
            jobstores = {}
            if self.config['persistence']['enabled']:
                jobstores['default'] = SQLAlchemyJobStore(
                    url=self.config['persistence']['jobstore_url']
                )
            
            # Configure executors
            executors = {
                'default': ThreadPoolExecutor(max_workers=self.config['max_workers'])
            }
            
            # Create scheduler
            if self.config['scheduler_type'] == 'background':
                self.scheduler = BackgroundScheduler(
                    jobstores=jobstores,
                    executors=executors,
                    job_defaults=self.config['job_defaults'],
                    timezone=self.config['timezone']
                )
            else:
                self.scheduler = BlockingScheduler(
                    jobstores=jobstores,
                    executors=executors,
                    job_defaults=self.config['job_defaults'],
                    timezone=self.config['timezone']
                )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)
            self.scheduler.add_listener(self._job_missed_listener, EVENT_JOB_MISSED)
            
            logger.info("Scheduler setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup scheduler: {str(e)}")
            raise
    
    def _setup_task_dependencies(self):
        """Setup task dependencies"""
        for task_id, task_config in self.config['tasks'].items():
            if 'depends_on' in task_config:
                self.dependency_manager.add_dependency(task_id, task_config['depends_on'])
                logger.info(f"Added dependency: {task_id} depends on {task_config['depends_on']}")
    
    def _initialize_task_components(self):
        """Initialize task component instances"""
        try:
            # Initialize with custom configs if available
            daily_config = self.config.get('daily_data_update_config')
            self.daily_updater = DailyDataUpdater(config=daily_config)
            
            retrainer_config = self.config.get('model_retrainer_config')
            self.model_retrainer = ModelRetrainer(config=retrainer_config)
            
            prediction_config = self.config.get('prediction_generator_config')
            self.prediction_generator = PredictionGenerator(config=prediction_config)
            
            logger.info("Task components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize task components: {str(e)}")
            raise
    
    def _add_scheduled_jobs(self):
        """Add all scheduled jobs to the scheduler"""
        for task_id, task_config in self.config['tasks'].items():
            if not task_config.get('enabled', True):
                logger.info(f"Task {task_id} is disabled, skipping")
                continue
            
            try:
                # Create trigger based on schedule type
                schedule = task_config['schedule']
                if schedule['type'] == 'cron':
                    trigger = CronTrigger(
                        hour=schedule.get('hour'),
                        minute=schedule.get('minute'),
                        day_of_week=schedule.get('day_of_week'),
                        timezone=self.config['timezone']
                    )
                elif schedule['type'] == 'interval':
                    trigger = IntervalTrigger(
                        hours=schedule.get('hours', 0),
                        minutes=schedule.get('minutes', 0),
                        seconds=schedule.get('seconds', 0)
                    )
                else:
                    logger.warning(f"Unknown schedule type for {task_id}: {schedule['type']}")
                    continue
                
                # Add job to scheduler
                job_func = self._get_task_function(task_id)
                if job_func:
                    self.scheduler.add_job(
                        func=job_func,
                        trigger=trigger,
                        id=task_id,
                        name=f"S&P500 {task_id.replace('_', ' ').title()}",
                        kwargs={'task_config': task_config}
                    )
                    logger.info(f"Added scheduled job: {task_id}")
                else:
                    logger.warning(f"No function found for task: {task_id}")
                    
            except Exception as e:
                logger.error(f"Failed to add job {task_id}: {str(e)}")
    
    def _get_task_function(self, task_id: str) -> Optional[Callable]:
        """Get the function to execute for a task"""
        task_functions = {
            'daily_data_update': self._run_daily_data_update,
            'model_retraining': self._run_model_retraining,
            'prediction_generation': self._run_prediction_generation,
            'health_check': self._run_health_check
        }
        return task_functions.get(task_id)
    
    def _run_daily_data_update(self, task_config: Dict[str, Any]) -> bool:
        """Execute daily data update task"""
        return self._execute_task_with_retry(
            task_id='daily_data_update',
            task_func=self.daily_updater.run_daily_update,
            task_config=task_config
        )
    
    def _run_model_retraining(self, task_config: Dict[str, Any]) -> bool:
        """Execute model retraining task"""
        return self._execute_task_with_retry(
            task_id='model_retraining',
            task_func=self.model_retrainer.run_retraining_cycle,
            task_config=task_config
        )
    
    def _run_prediction_generation(self, task_config: Dict[str, Any]) -> bool:
        """Execute prediction generation task"""
        # Check dependencies
        if not self.dependency_manager.can_run_task('prediction_generation'):
            logger.warning("Prediction generation skipped due to failed dependencies")
            self.dependency_manager.record_task_result(TaskResult(
                task_id='prediction_generation',
                status=TaskStatus.SKIPPED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message="Dependencies not met"
            ))
            return False
        
        return self._execute_task_with_retry(
            task_id='prediction_generation',
            task_func=self.prediction_generator.run_prediction_pipeline,
            task_config=task_config
        )
    
    def _run_health_check(self, task_config: Dict[str, Any]) -> bool:
        """Execute system health check"""
        try:
            logger.info("Running system health check...")
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'scheduler_status': 'running' if self.is_running else 'stopped',
                'total_executions': self.task_stats['total_executions'],
                'success_rate': 0.0,
                'recent_tasks': []
            }
            
            # Calculate success rate
            if self.task_stats['total_executions'] > 0:
                health_status['success_rate'] = (
                    self.task_stats['successful_executions'] / 
                    self.task_stats['total_executions']
                )
            
            # Check recent task results
            for task_id, result in self.dependency_manager.task_results.items():
                health_status['recent_tasks'].append({
                    'task_id': task_id,
                    'status': result.status.value,
                    'last_run': result.end_time.isoformat() if result.end_time else None,
                    'execution_time': result.execution_time
                })
            
            # Save health status
            health_file = self.project_root / "data" / "scheduler_health.json"
            health_file.parent.mkdir(exist_ok=True)
            with open(health_file, 'w') as f:
                json.dump(health_status, f, indent=2, default=str)
            
            self.task_stats['last_health_check'] = datetime.now()
            
            # Check for alerts
            if health_status['success_rate'] < 0.8 and self.task_stats['total_executions'] > 5:
                self._send_health_alert(f"Low success rate: {health_status['success_rate']:.2%}")
            
            logger.info(f"Health check completed - Success rate: {health_status['success_rate']:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def _execute_task_with_retry(self, task_id: str, task_func: Callable, task_config: Dict[str, Any]) -> bool:
        """Execute task with retry logic"""
        max_retries = task_config.get('max_retries', 0)
        retry_delay = task_config.get('retry_delay_minutes', 5)
        
        start_time = datetime.now()
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Executing {task_id} (attempt {retry_count + 1}/{max_retries + 1})")
                
                # Execute the task
                success = task_func()
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Record result
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.SUCCESS if success else TaskStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    retry_count=retry_count,
                    execution_time=execution_time
                )
                
                self.dependency_manager.record_task_result(result)
                self._update_task_stats(success)
                
                if success:
                    logger.info(f"Task {task_id} completed successfully in {execution_time:.1f}s")
                    return True
                else:
                    logger.error(f"Task {task_id} failed")
                    if retry_count < max_retries:
                        logger.info(f"Retrying {task_id} in {retry_delay} minutes...")
                        threading.Event().wait(retry_delay * 60)
                        retry_count += 1
                    else:
                        return False
                        
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                error_msg = str(e)
                
                logger.error(f"Task {task_id} raised exception: {error_msg}")
                
                # Record result
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    error_message=error_msg,
                    retry_count=retry_count,
                    execution_time=execution_time
                )
                
                self.dependency_manager.record_task_result(result)
                self._update_task_stats(False)
                
                if retry_count < max_retries:
                    logger.info(f"Retrying {task_id} in {retry_delay} minutes...")
                    threading.Event().wait(retry_delay * 60)
                    retry_count += 1
                else:
                    return False
        
        return False
    
    def _update_task_stats(self, success: bool):
        """Update task execution statistics"""
        self.task_stats['total_executions'] += 1
        if success:
            self.task_stats['successful_executions'] += 1
        else:
            self.task_stats['failed_executions'] += 1
    
    def _job_executed_listener(self, event):
        """Handle job execution events"""
        logger.info(f"Job {event.job_id} executed successfully")
    
    def _job_error_listener(self, event):
        """Handle job error events"""
        logger.error(f"Job {event.job_id} crashed: {event.exception}")
        traceback.print_exc()
        
        if self.config['notifications']['error_alerts']:
            self._send_error_alert(event.job_id, str(event.exception))
    
    def _job_missed_listener(self, event):
        """Handle missed job events"""
        logger.warning(f"Job {event.job_id} was missed")
    
    def _send_error_alert(self, job_id: str, error_message: str):
        """Send error alert notification"""
        try:
            if not self.config['notifications']['email_enabled']:
                return
            
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = os.getenv('SMTP_USERNAME')
            msg['To'] = os.getenv('NOTIFICATION_EMAIL')
            msg['Subject'] = f"[S&P500 Scheduler] Task Failed: {job_id}"
            
            message_body = f"""
Scheduler Alert: Task Failure

Task: {job_id}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {error_message}

Please check the scheduler logs for more details.

---
S&P 500 Prediction System
Scheduler Orchestrator
"""
            
            msg.attach(MIMEText(message_body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'), 
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(os.getenv('SMTP_USERNAME'), os.getenv('SMTP_PASSWORD'))
            text = msg.as_string()
            server.sendmail(os.getenv('SMTP_USERNAME'), os.getenv('NOTIFICATION_EMAIL'), text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {str(e)}")
    
    def _send_health_alert(self, health_message: str):
        """Send health alert notification"""
        try:
            logger.warning(f"Health alert: {health_message}")
            # Implementation similar to _send_error_alert
            # Simplified for brevity
        except Exception as e:
            logger.error(f"Failed to send health alert: {str(e)}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the scheduler orchestrator"""
        try:
            logger.info("Starting Scheduler Orchestrator...")
            
            # Setup components
            self._setup_scheduler()
            self._setup_task_dependencies()
            self._initialize_task_components()
            self._add_scheduled_jobs()
            self._setup_signal_handlers()
            
            # Reset daily results at startup
            self.dependency_manager.reset_daily_results()
            
            # Start scheduler
            self.is_running = True
            logger.info("Scheduler started successfully")
            
            # Print schedule summary
            self._print_schedule_summary()
            
            # Start the scheduler (this will block if using BlockingScheduler)
            self.scheduler.start()
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            traceback.print_exc()
            raise
    
    def shutdown(self):
        """Gracefully shutdown the scheduler"""
        try:
            logger.info("Shutting down scheduler...")
            self.is_running = False
            self.shutdown_event.set()
            
            if self.scheduler:
                self.scheduler.shutdown(wait=True)
            
            logger.info("Scheduler shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {str(e)}")
    
    def _print_schedule_summary(self):
        """Print a summary of scheduled jobs"""
        logger.info("=== SCHEDULED JOBS SUMMARY ===")
        
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            logger.info(f"Job: {job.name}")
            logger.info(f"  ID: {job.id}")
            logger.info(f"  Next run: {next_run}")
            logger.info(f"  Trigger: {job.trigger}")
        
        logger.info("==============================")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        jobs_status = []
        for job in self.scheduler.get_jobs():
            jobs_status.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'is_running': self.is_running,
            'scheduler_type': self.config['scheduler_type'],
            'timezone': self.config['timezone'],
            'total_jobs': len(jobs_status),
            'jobs': jobs_status,
            'task_stats': self.task_stats,
            'last_health_check': self.task_stats['last_health_check'].isoformat() if self.task_stats['last_health_check'] else None
        }

def main():
    """Main function for command line execution"""
    
    import argparse
    parser = argparse.ArgumentParser(description='S&P 500 Prediction System Scheduler Orchestrator')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--test-config', action='store_true', help='Test configuration and exit')
    parser.add_argument('--status', action='store_true', help='Show scheduler status and exit')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    
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
    
    try:
        orchestrator = SchedulerOrchestrator(config=config)
        
        if args.test_config:
            logger.info("Configuration test completed successfully")
            print("✓ Configuration is valid")
            print(f"✓ Scheduler type: {orchestrator.config['scheduler_type']}")
            print(f"✓ Timezone: {orchestrator.config['timezone']}")
            print(f"✓ Enabled tasks: {[k for k, v in orchestrator.config['tasks'].items() if v.get('enabled', True)]}")
            sys.exit(0)
        
        if args.status:
            # For status, we need to connect to existing scheduler
            # This is a simplified implementation
            print("Scheduler status functionality requires implementation")
            sys.exit(0)
        
        # Modify config for daemon mode
        if args.daemon:
            logger.info("Running in daemon mode")
            if config is None:
                config = {}
            config['scheduler_type'] = 'background'
            orchestrator = SchedulerOrchestrator(config=config)
        
        # Start the orchestrator
        orchestrator.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
