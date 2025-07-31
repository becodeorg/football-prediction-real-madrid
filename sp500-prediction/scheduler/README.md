# S&P 500 Prediction System - Scheduler

This directory contains automation scripts for the S&P 500 prediction system.

## Components

### 1. Daily Data Update (`daily_data_update.py`)
Automates daily data collection and processing.

**Features:**
- Fetches latest S&P 500 and VIX data from Yahoo Finance
- Validates data quality and completeness
- Updates raw and processed datasets
- Error handling with retry mechanisms
- Email notifications
- Market hours awareness

**Usage:**
```bash
# Run daily update
python scheduler/daily_data_update.py

# Force update regardless of market hours
python scheduler/daily_data_update.py --force

# Use custom configuration
python scheduler/daily_data_update.py --config /path/to/config.json

# Enable verbose logging
python scheduler/daily_data_update.py --verbose
```

### 2. Model Retrainer (`model_retrainer.py`)
Automates model retraining, validation, and deployment.

**Features:**
- Weekly/monthly automated model retraining
- Performance monitoring and validation against current models
- Model versioning and backup system
- Automatic deployment of better-performing models
- A/B testing capabilities for model comparison
- Comprehensive model performance tracking
- Rollback functionality if deployment fails

**Usage:**
```bash
# Run scheduled retraining
python scheduler/model_retrainer.py

# Force retraining regardless of schedule
python scheduler/model_retrainer.py --force

# Train models but don't deploy automatically
python scheduler/model_retrainer.py --no-deploy

# Use custom configuration
python scheduler/model_retrainer.py --config scheduler/retrainer_config.json

# Enable verbose logging
python scheduler/model_retrainer.py --verbose
```

**Configuration:**
The model retrainer supports configuration via:
- Environment variables (see `.env.retrainer.template`)
- JSON configuration file (see `retrainer_config.template.json`)
- Command line arguments

**Key Configuration Options:**
- `retraining_frequency`: 'weekly', 'monthly', or 'manual'
- `improvement_threshold`: Minimum improvement required for deployment (default: 2%)
- `models_to_train`: List of models to train ['random_forest', 'gradient_boosting', 'logistic_regression']
- `auto_deploy`: Whether to automatically deploy better models
- `backup_retention_days`: How long to keep model backups

**Performance Tracking:**
- Maintains history of model performance in `models/performance_history.json`
- Compares new models against baseline before deployment
- Tracks accuracy, precision, recall, and F1 scores
- Provides detailed comparison reports

### 3. Prediction Generator (`prediction_generator.py`)
Automates daily prediction generation and dashboard data preparation.

**Features:**
- Generates daily S&P 500 predictions using trained models
- Maintains prediction history in SQLite database
- Prepares data for dashboard consumption
- Validates prediction quality and consistency
- Updates actual values when data becomes available
- Supports multiple prediction horizons and types
- Performance tracking and accuracy monitoring

**Usage:**
```bash
# Run prediction generation
python scheduler/prediction_generator.py

# Test models only (no predictions saved)
python scheduler/prediction_generator.py --test-only

# Skip dashboard data preparation
python scheduler/prediction_generator.py --no-dashboard

# Use custom configuration
python scheduler/prediction_generator.py --config scheduler/prediction_config.json

# Enable verbose logging
python scheduler/prediction_generator.py --verbose
```

**Configuration:**
The prediction generator supports configuration via:
- Environment variables (see `.env.prediction.template`)
- JSON configuration file (see `prediction_config.template.json`)
- Command line arguments

**Key Configuration Options:**
- `prediction_horizons`: Days ahead to predict [1, 2, 5]
- `prediction_types`: Types of predictions ['direction', 'price', 'volatility']
- `confidence_threshold`: Minimum confidence for saving predictions (default: 0.6)
- `enable_dashboard_update`: Whether to update dashboard data
- `max_prediction_age_days`: How long to keep predictions in database

**Database Structure:**
- Predictions stored in SQLite database (`data/predictions.db`)
- Tracks prediction date, target date, prediction value, confidence, actual values
- Automatic cleanup of old predictions
- Performance metrics tracking

**Dashboard Integration:**
- Prepares cached data for dashboard in `data/cache/dashboard_data.json`
- Includes historical prices, predictions, performance metrics
- Feature importance from trained models
- Market indicators and trend analysis

## Configuration

### Environment Variables

The system can be configured using environment variables in `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `NOTIFICATION_EMAIL` | Email for notifications | None |
| `SMTP_SERVER` | SMTP server for emails | smtp.gmail.com |
| `SMTP_PORT` | SMTP port | 587 |
| `SMTP_USERNAME` | SMTP username | None |
| `SMTP_PASSWORD` | SMTP password/app password | None |
| `MAX_RETRIES` | Number of retry attempts | 3 |
| `RETRY_DELAY` | Delay between retries (seconds) | 60 |
| `DATA_VALIDATION_THRESHOLD` | Required data completeness | 0.95 |
| `LOOKBACK_DAYS` | Days to fetch for updates | 5 |

### Email Setup

To enable email notifications:

1. **Gmail Users**: 
   - Enable 2-factor authentication
   - Generate an App Password
   - Use the App Password in `SMTP_PASSWORD`

2. **Other Providers**: 
   - Configure the appropriate SMTP settings
   - Ensure less secure app access is enabled if required

## Scheduling

### Using Cron (Linux/macOS)

Add to your crontab (`crontab -e`):

```bash
# Run daily at 6 PM ET (after market close)
0 22 * * 1-5 cd /path/to/sp500-prediction && python scheduler/daily_data_update.py

# Run with force on weekends for testing (optional)
0 10 * * 6,0 cd /path/to/sp500-prediction && python scheduler/daily_data_update.py --force
```

### Using APScheduler (Python)

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from scheduler.daily_data_update import DailyDataUpdater

scheduler = BlockingScheduler()

def run_update():
    updater = DailyDataUpdater()
    updater.run_daily_update()

# Schedule for weekdays at 6 PM ET
scheduler.add_job(
    run_update,
    trigger=CronTrigger(hour=22, minute=0, day_of_week='mon-fri'),
    id='daily_data_update'
)

scheduler.start()
```

### Using Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily, after market close
4. Action: Start a program
5. Program: `python`
6. Arguments: `scheduler/daily_data_update.py`
7. Start in: `/path/to/sp500-prediction`

## Data Validation

The system performs comprehensive data validation:

### Raw Data Checks
- ✅ Required columns present (`date`, `open`, `high`, `low`, `close`, `volume`)
- ✅ Data completeness above threshold (default: 95%)
- ✅ Reasonable price ranges (S&P 500: 1000-10000, VIX: 0-100)
- ✅ No duplicate dates
- ✅ Recent data availability (within 5 business days)

### Feature Engineering Validation
- ✅ Successful pipeline execution
- ✅ Non-empty processed dataset
- ✅ Required target columns present
- ✅ Feature completeness checks

## Logging

Logs are automatically created in the `logs/` directory:

- **File**: `logs/daily_update_YYYYMMDD.log`
- **Console**: Real-time output
- **Format**: Timestamp - Logger - Level - Message

### Log Levels
- `INFO`: Normal operations and status updates
- `WARNING`: Non-critical issues (e.g., market closed)
- `ERROR`: Failures that don't stop execution
- `CRITICAL`: System failures requiring attention

## Error Handling

The system includes robust error handling:

1. **Network Issues**: Automatic retries with exponential backoff
2. **Data Quality Issues**: Validation with clear error messages
3. **Processing Errors**: Graceful failure with detailed logging
4. **Email Failures**: Logged but don't stop the update process

## Monitoring

### Success Indicators
- ✅ Email notification: "Daily Update Successful"
- ✅ Log message: "Daily update completed successfully"
- ✅ Updated files in `data/processed/`

### Failure Indicators
- ❌ Email notification: "Daily Update Failed"
- ❌ Log errors in daily log file
- ❌ Exit code 1 from script

### Health Checks

You can monitor the system health by:

```bash
# Check last update log
tail -50 logs/daily_update_$(date +%Y%m%d).log

# Verify data freshness
python -c "
import pandas as pd
df = pd.read_csv('data/processed/sp500_features.csv')
print(f'Latest data: {df[\"date\"].max()}')
"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root
   cd /path/to/sp500-prediction
   # Check Python path
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

2. **Yahoo Finance Connection Issues**
   - Check internet connection
   - Verify Yahoo Finance is accessible
   - Try running with `--verbose` for detailed logs

3. **Email Notification Failures**
   - Verify SMTP settings in `.env`
   - Check firewall/antivirus blocking SMTP
   - Test with Gmail App Password

4. **Permission Issues**
   ```bash
   # Make script executable
   chmod +x scheduler/daily_data_update.py
   # Check directory permissions
   ls -la data/
   ```

5. **Data Validation Failures**
   - Check Yahoo Finance data availability
   - Review validation thresholds in configuration
   - Examine the specific validation error in logs

### Debug Mode

Run with verbose logging for detailed debugging:

```bash
python scheduler/daily_data_update.py --verbose --force
```

## Integration

The daily data updater integrates with:

- **Data Collection** (`src/data_collection.py`): Uses existing data collection classes
- **Feature Engineering** (`src/feature_engineering.py`): Automatically updates processed features
- **Model Training**: Provides fresh data for model retraining
- **Dashboard**: Ensures dashboard has latest data

## Security Considerations

- 🔐 Store sensitive credentials in `.env` file (not in code)
- 🔐 Use App Passwords for email authentication
- 🔐 Restrict file permissions on configuration files
- 🔐 Consider using environment variables in production
- 🔐 Regular security updates for dependencies

## Performance

- ⚡ Typical execution time: 30-60 seconds
- ⚡ Network-dependent (Yahoo Finance API response)
- ⚡ Minimal resource usage
- ⚡ Efficient data processing with pandas

## Model Retrainer Setup

### Configuration Templates

Copy and configure the model retrainer templates:

```bash
# Environment configuration
cp scheduler/.env.retrainer.template scheduler/.env.retrainer

# Advanced JSON configuration (optional)
cp scheduler/retrainer_config.template.json scheduler/retrainer_config.json
cp scheduler/prediction_config.template.json scheduler/prediction_config.json
```

### Testing Model Retrainer

Before scheduling, test the model retrainer:

```bash
# Run comprehensive test suite
python test_model_retrainer.py

# Test training without deployment
python scheduler/model_retrainer.py --force --no-deploy --verbose
```

### Scheduling Model Retraining

**Weekly Retraining (Recommended):**
```bash
# Add to crontab for Sunday 2 AM
0 2 * * 0 cd /path/to/sp500-prediction && python scheduler/model_retrainer.py >> logs/cron.log 2>&1
```

**Monthly Retraining:**
```bash
# First Sunday of each month at 2 AM
0 2 1-7 * 0 cd /path/to/sp500-prediction && python scheduler/model_retrainer.py >> logs/cron.log 2>&1
```

### Model Performance Monitoring

Monitor model performance and deployment:

```bash
# Check latest model metrics
python -c "
import json
with open('models/performance_history.json') as f:
    history = json.load(f)
    latest = history[-1] if history else {}
    print(f'Latest model accuracy: {latest.get(\"test_accuracy\", \"N/A\")}')
    print(f'Training date: {latest.get(\"timestamp\", \"N/A\")}')
"

# Check model backups
ls -la models/backups/
```

## Complete System Setup

### 1. Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environments
cp scheduler/.env.template scheduler/.env
cp scheduler/.env.retrainer.template scheduler/.env.retrainer

# Edit configuration files with your settings
```

### 2. Test Everything
```bash
# Test data updates
python test_daily_update.py

# Test model retraining
python test_model_retrainer.py

# Test prediction generation
python test_prediction_generator.py

# Test manual runs
python scheduler/daily_data_update.py --force
python scheduler/model_retrainer.py --force --no-deploy
python scheduler/prediction_generator.py --test-only
```

### 3. Schedule Automation
```bash
# Edit crontab
crontab -e

# Add all three jobs:
# Daily data update (6 PM ET weekdays)
0 22 * * 1-5 cd /path/to/sp500-prediction && python scheduler/daily_data_update.py

# Weekly model retraining (Sunday 2 AM)
0 2 * * 0 cd /path/to/sp500-prediction && python scheduler/model_retrainer.py

# Daily prediction generation (7 PM ET weekdays, after data update)
0 23 * * 1-5 cd /path/to/sp500-prediction && python scheduler/prediction_generator.py
```

### 4. Monitor Operations
```bash
# Check logs regularly
tail -f logs/daily_update_$(date +%Y%m%d).log
tail -f logs/model_retrainer_$(date +%Y%m%d).log
tail -f logs/prediction_generator_$(date +%Y%m%d).log

# Set up log rotation (optional)
# Add to /etc/logrotate.d/sp500-prediction
```

## Advanced Model Retraining Features

### A/B Testing
The system supports A/B testing of models:
- Trains multiple models simultaneously
- Compares performance against baseline
- Only deploys if improvement exceeds threshold
- Maintains performance history for analysis

### Model Versioning
- Automatic backup before retraining
- Timestamped model versions
- Rollback capability if needed
- Performance metadata tracking

### Hyperparameter Tuning
Configure automatic hyperparameter optimization:
```json
{
  "hyperparameter_tuning": true,
  "training_parameters": {
    "random_forest": {
      "n_estimators": [100, 200, 300],
      "max_depth": [10, 20, 30, null]
    }
  }
}
```

## Next Steps

After setting up the complete automation system, consider:

1. **System Monitoring Dashboard** - Real-time monitoring of all automation components
2. **Cloud Deployment** - Deploy to AWS, Azure, or GCP with managed scheduling
3. **Real-time Alerting** - SMS/Slack alerts for critical failures
4. **Performance Analytics Dashboard** - Advanced visualization of prediction performance
5. **API Development** - REST API for accessing predictions and system status
6. **Mobile App** - Mobile interface for monitoring and alerts
7. **Advanced ML Features** - Ensemble methods, deep learning models, sentiment analysis

## Support

For issues or questions:

1. Check the logs in `logs/` directory
2. Run the test suites: `python test_daily_update.py` and `python test_model_retrainer.py`
3. Review this documentation
4. Check the main project README.md
5. Use `--verbose` flag for detailed debugging
