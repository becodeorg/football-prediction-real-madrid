# S&P 500 Prediction Project - Automation System Setup Guide

## Overview

This guide provides comprehensive instructions for setting up and using the automated S&P 500 prediction system. The automation infrastructure includes daily data collection, model retraining, prediction generation, and centralized orchestration.

## System Architecture

```
sp500-prediction/
├── app/                    # Streamlit dashboard
├── data/                   # Data storage
├── models/                 # Trained models
├── src/                    # Core modules
├── scheduler/              # Automation system ⭐
│   ├── daily_data_update.py      # Daily data collection
│   ├── model_retrainer.py        # Automated model retraining
│   ├── prediction_generator.py   # Daily prediction pipeline
│   ├── scheduler_main.py         # Main orchestrator
│   └── scheduler_config.json     # Configuration
├── logs/                   # System logs
└── requirements.txt        # Dependencies
```

## Prerequisites

### 1. Python Environment Setup

```bash
# Create virtual environment (if not already done)
python3 -m venv Madrid
source Madrid/bin/activate  # On macOS/Linux
# or
Madrid\Scripts\activate     # On Windows

# Install required packages
pip install -r requirements.txt

# Install additional automation dependencies
pip install apscheduler>=3.10.0
pip install sqlalchemy>=1.4.0
pip install psutil>=5.9.0
```

### 2. Required Dependencies

The automation system requires these additional packages:

```txt
# Add to requirements.txt
apscheduler>=3.10.0    # Task scheduling
sqlalchemy>=1.4.0      # Database persistence
psutil>=5.9.0          # System monitoring
smtplib                # Email notifications (built-in)
logging                # Comprehensive logging (built-in)
```

### 3. Directory Structure

Ensure these directories exist:

```bash
mkdir -p logs
mkdir -p data/processed
mkdir -p data/raw
mkdir -p models
```

## Configuration

### 1. Create Configuration File

```bash
# Copy template configuration
cp scheduler/scheduler_config.template.json scheduler/scheduler_config.json

# Edit configuration
nano scheduler/scheduler_config.json  # or your preferred editor
```

### 2. Configuration Options

Key configuration sections:

#### Basic Settings
```json
{
  "scheduler_type": "blocking",      # "blocking" or "background"
  "timezone": "America/New_York",    # Your timezone
  "max_workers": 4,                  # Concurrent task limit
  "log_level": "INFO"                # Logging verbosity
}
```

#### Task Scheduling
```json
{
  "tasks": {
    "daily_data_update": {
      "enabled": true,
      "schedule": {
        "type": "cron",
        "hour": 22,           # 10 PM (after market close)
        "minute": 0,
        "day_of_week": "mon-fri"
      },
      "max_retries": 3,
      "retry_delay": 300
    },
    "prediction_generation": {
      "enabled": true,
      "schedule": {
        "type": "cron",
        "hour": 23,           # 11 PM (after data update)
        "minute": 0,
        "day_of_week": "mon-fri"
      },
      "depends_on": ["daily_data_update"],
      "max_retries": 2
    },
    "model_retraining": {
      "enabled": true,
      "schedule": {
        "type": "cron",
        "hour": 2,            # 2 AM Saturday
        "minute": 0,
        "day_of_week": "sat"
      },
      "max_retries": 1
    }
  }
}
```

#### Email Notifications
```json
{
  "notifications": {
    "email_enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "app-password",
    "recipient_emails": ["admin@yourcompany.com"]
  }
}
```

## Testing the System

### 1. Environment Validation

```bash
# Run comprehensive test suite
python test_scheduler_main.py

# Expected output:
# ✓ Environment setup validation passed!
# ✓ Configuration loading tests passed!
# ✓ Task dependency manager tests passed!
# ... etc
```

### 2. Configuration Testing

```bash
# Test your configuration
python scheduler/scheduler_main.py --test-config

# Validate specific tasks
python scheduler/scheduler_main.py --validate-tasks
```

### 3. Individual Component Testing

```bash
# Test data collection
python scheduler/daily_data_update.py --test

# Test model retraining
python scheduler/model_retrainer.py --test

# Test prediction generation
python scheduler/prediction_generator.py --test
```

## Running the Automation System

### 1. Interactive Mode (Development)

```bash
# Start scheduler in foreground
python scheduler/scheduler_main.py

# With custom configuration
python scheduler/scheduler_main.py --config scheduler/custom_config.json

# With verbose logging
python scheduler/scheduler_main.py --verbose
```

### 2. Daemon Mode (Production)

```bash
# Start as background daemon
python scheduler/scheduler_main.py --daemon

# Check if running
ps aux | grep scheduler_main

# Stop daemon
python scheduler/scheduler_main.py --stop
```

### 3. Systemd Service (Linux Production)

Create `/etc/systemd/system/sp500-scheduler.service`:

```ini
[Unit]
Description=S&P 500 Prediction Scheduler
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/sp500-prediction
Environment=PATH=/path/to/Madrid/bin
ExecStart=/path/to/Madrid/bin/python scheduler/scheduler_main.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable sp500-scheduler
sudo systemctl start sp500-scheduler
sudo systemctl status sp500-scheduler
```

## Monitoring and Logs

### 1. Log Files

```bash
# Main scheduler logs
tail -f logs/scheduler_main_*.log

# Task-specific logs
tail -f logs/daily_data_update_*.log
tail -f logs/model_retrainer_*.log
tail -f logs/prediction_generator_*.log

# Error logs
tail -f logs/scheduler_errors_*.log
```

### 2. System Status

```bash
# Get current status
python scheduler/scheduler_main.py --status

# View active jobs
python scheduler/scheduler_main.py --list-jobs

# View task history
python scheduler/scheduler_main.py --task-history
```

### 3. Health Monitoring

The system includes built-in health checks:

- **Daily Health Check**: Runs every hour to verify system status
- **Resource Monitoring**: Tracks CPU, memory, and disk usage
- **Task Success Rates**: Monitors task completion rates
- **Automated Alerts**: Sends notifications for failures

## Manual Task Execution

### 1. Run Individual Tasks

```bash
# Force data update
python scheduler/scheduler_main.py --run-task daily_data_update

# Force model retraining
python scheduler/scheduler_main.py --run-task model_retraining

# Force prediction generation
python scheduler/scheduler_main.py --run-task prediction_generation
```

### 2. Batch Operations

```bash
# Run complete daily pipeline
python scheduler/scheduler_main.py --run-pipeline daily

# Run complete weekly pipeline
python scheduler/scheduler_main.py --run-pipeline weekly

# Emergency model retrain
python scheduler/scheduler_main.py --emergency-retrain
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure APScheduler is installed
pip install apscheduler sqlalchemy psutil

# Check Python path
export PYTHONPATH=$PYTHONPATH:/path/to/sp500-prediction
```

#### 2. Permission Errors
```bash
# Ensure log directory is writable
chmod 755 logs/
chmod 644 logs/*.log
```

#### 3. Data Access Issues
```bash
# Test Yahoo Finance connectivity
python -c "import yfinance as yf; print(yf.download('SPY', period='1d'))"

# Check data directories
ls -la data/raw/
ls -la data/processed/
```

#### 4. Task Failures
```bash
# Check task logs
grep "ERROR\|FAILED" logs/scheduler_main_*.log

# View task dependencies
python scheduler/scheduler_main.py --show-dependencies

# Reset task states
python scheduler/scheduler_main.py --reset-tasks
```

### Recovery Procedures

#### 1. Restart Scheduler
```bash
# Stop current instance
python scheduler/scheduler_main.py --stop

# Clear job store (if needed)
rm -f scheduler_jobs.sqlite

# Restart
python scheduler/scheduler_main.py --daemon
```

#### 2. Emergency Model Recovery
```bash
# Restore backup models
python scheduler/model_retrainer.py --restore-backup

# Force model validation
python scheduler/model_retrainer.py --validate-models
```

#### 3. Data Recovery
```bash
# Re-download missing data
python scheduler/daily_data_update.py --backfill --days 30

# Rebuild processed features
python src/feature_engineering.py
```

## Performance Optimization

### 1. Resource Management

- **CPU Usage**: Monitor with `htop` or `top`
- **Memory Usage**: Adjust `max_workers` based on available RAM
- **Disk I/O**: Use SSD storage for better performance
- **Network**: Ensure stable internet for data downloads

### 2. Task Timing

- **Data Collection**: Schedule after market close (4 PM ET)
- **Predictions**: Generate after data is processed
- **Model Training**: Run during low-usage periods (weekends)
- **Health Checks**: Every hour during business hours

### 3. Configuration Tuning

```json
{
  "performance": {
    "data_batch_size": 1000,
    "model_cache_size": 100,
    "concurrent_downloads": 5,
    "retry_exponential_backoff": true
  }
}
```

## Security Considerations

### 1. Credential Management

```bash
# Use environment variables for sensitive data
export YAHOO_API_KEY="your_api_key"
export EMAIL_PASSWORD="app_password"

# Secure configuration file
chmod 600 scheduler/scheduler_config.json
```

### 2. Network Security

- Use HTTPS for all external API calls
- Implement rate limiting for Yahoo Finance API
- Monitor for unusual network activity

### 3. Log Security

```bash
# Rotate logs regularly
logrotate /etc/logrotate.d/sp500-scheduler

# Secure log directory
chmod 750 logs/
chown your-user:your-group logs/
```

## Integration with Dashboard

The automation system integrates seamlessly with the Streamlit dashboard:

1. **Automated Data Updates**: Dashboard shows latest predictions
2. **Real-time Status**: Dashboard displays scheduler status
3. **Performance Metrics**: Dashboard shows model performance trends
4. **Alert Integration**: Dashboard shows system health status

To view dashboard:

```bash
# Start dashboard
streamlit run app/streamlit_app.py

# Access at: http://localhost:8501
```

## Maintenance Schedule

### Daily
- Monitor logs for errors
- Check prediction generation success
- Verify data update completion

### Weekly
- Review model performance metrics
- Check disk space usage
- Validate backup systems

### Monthly
- Update dependencies
- Review and optimize configuration
- Perform full system health check
- Test disaster recovery procedures

## Getting Help

### Log Analysis
```bash
# Search for specific errors
grep -r "ERROR" logs/

# View recent failures
tail -n 100 logs/scheduler_errors_*.log

# Monitor in real-time
watch 'tail -n 20 logs/scheduler_main_*.log'
```

### System Diagnostics
```bash
# Run full diagnostics
python test_scheduler_main.py

# Check configuration validity
python scheduler/scheduler_main.py --validate-config

# Test all components
python scheduler/scheduler_main.py --test-all
```

For additional support, check the troubleshooting logs and system status outputs.

---

## Next Steps

After setting up the automation system:

1. **Configure Production Settings**: Adjust schedules for your timezone
2. **Set Up Monitoring**: Implement log rotation and alerting
3. **Test Failure Scenarios**: Validate error recovery procedures
4. **Optimize Performance**: Tune settings based on your hardware
5. **Deploy Dashboard**: Make predictions accessible via web interface

The automation system is now complete and ready for production use!
