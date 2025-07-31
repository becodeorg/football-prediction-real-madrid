#!/bin/bash
# Docker Entrypoint Script for S&P 500 Prediction System
# Handles different runtime modes and configurations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  S&P 500 Prediction System${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to initialize configuration
init_config() {
    print_status "Initializing configuration..."
    
    # Create configuration directory if not exists
    mkdir -p /app/config
    
    # Copy default configuration if not exists
    if [ ! -f "/app/config/scheduler_config.json" ]; then
        if [ -f "/app/config/scheduler_config.template.json" ]; then
            cp /app/config/scheduler_config.template.json /app/config/scheduler_config.json
            print_status "Created default scheduler configuration"
        else
            print_warning "No configuration template found, using built-in defaults"
        fi
    fi
    
    # Set configuration path
    export SCHEDULER_CONFIG_PATH="/app/config/scheduler_config.json"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    python --version || {
        print_error "Python not found"
        exit 1
    }
    
    # Check required Python packages
    python -c "
import sys
required_packages = [
    'pandas', 'numpy', 'sklearn', 'yfinance',
    'apscheduler', 'sqlalchemy', 'psutil'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages available')
" || {
        print_error "Required Python packages missing"
        exit 1
    }
    
    print_status "All dependencies satisfied"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    # Create required directories
    mkdir -p /app/data/raw
    mkdir -p /app/data/processed
    mkdir -p /app/models
    mkdir -p /app/logs
    mkdir -p /app/config
    
    # Set permissions
    chmod 755 /app/data /app/models /app/logs /app/config
    chmod 755 /app/data/raw /app/data/processed
    
    print_status "Directories created and configured"
}

# Function to run database migrations (if needed)
setup_database() {
    print_status "Setting up database..."
    
    # Initialize scheduler database if needed
    python -c "
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

db_path = '/app/data/scheduler_jobs.sqlite'
engine = create_engine(f'sqlite:///{db_path}')

try:
    with engine.connect() as conn:
        # Test connection
        conn.execute(text('SELECT 1'))
    print('Database connection successful')
except Exception as e:
    print(f'Database setup: {e}')
"
    
    print_status "Database setup complete"
}

# Function to run pre-flight checks
preflight_checks() {
    print_status "Running pre-flight checks..."
    
    # Check disk space
    df -h /app | tail -1 | awk '{
        if (int($5) > 90) {
            print "WARNING: Disk usage high: " $5
            exit 1
        } else {
            print "Disk usage OK: " $5
        }
    }'
    
    # Check memory
    free -m | awk 'NR==2{
        if (int($4/$2 * 100) < 10) {
            print "WARNING: Low memory available"
            exit 1
        } else {
            print "Memory OK"
        }
    }'
    
    print_status "Pre-flight checks passed"
}

# Function to start scheduler
start_scheduler() {
    print_status "Starting S&P 500 Prediction Scheduler..."
    
    # Set PYTHONPATH to include application directories
    export PYTHONPATH="/app:/app/src:/app/scheduler:$PYTHONPATH"
    
    # Change to app directory
    cd /app
    
    # Start scheduler with appropriate configuration
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Running in development mode"
        exec python scheduler/scheduler_main.py --verbose
    else
        print_status "Running in production mode"
        exec python scheduler/scheduler_main.py --daemon
    fi
}

# Function to start dashboard
start_dashboard() {
    print_status "Starting S&P 500 Prediction Dashboard..."
    
    # Set PYTHONPATH
    export PYTHONPATH="/app:/app/src:/app/scheduler:$PYTHONPATH"
    
    # Change to app directory
    cd /app
    
    # Start Streamlit dashboard
    exec streamlit run app/streamlit_app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false
}

# Function to run tests
run_tests() {
    print_status "Running system tests..."
    
    # Set PYTHONPATH
    export PYTHONPATH="/app:/app/src:/app/scheduler:$PYTHONPATH"
    
    # Change to app directory
    cd /app
    
    # Run validation tests
    python test_scheduler_simple.py
    
    print_status "Tests completed"
}

# Function to run one-time tasks
run_task() {
    local task_name="$1"
    print_status "Running task: $task_name"
    
    # Set PYTHONPATH
    export PYTHONPATH="/app:/app/src:/app/scheduler:$PYTHONPATH"
    
    # Change to app directory
    cd /app
    
    case "$task_name" in
        "data-update")
            python scheduler/daily_data_update.py --manual
            ;;
        "model-train")
            python src/train_model.py
            ;;
        "prediction")
            python scheduler/prediction_generator.py --manual
            ;;
        "health-check")
            python scheduler/scheduler_main.py --health-check
            ;;
        *)
            print_error "Unknown task: $task_name"
            echo "Available tasks: data-update, model-train, prediction, health-check"
            exit 1
            ;;
    esac
    
    print_status "Task completed: $task_name"
}

# Function to show help
show_help() {
    echo "S&P 500 Prediction System - Docker Entrypoint"
    echo ""
    echo "Usage: docker run [options] sp500-prediction [command]"
    echo ""
    echo "Commands:"
    echo "  scheduler    Start the automated scheduler (default)"
    echo "  dashboard    Start the Streamlit dashboard"
    echo "  test         Run system validation tests"
    echo "  task <name>  Run a specific task"
    echo "  shell        Start an interactive shell"
    echo "  help         Show this help message"
    echo ""
    echo "Available tasks:"
    echo "  data-update  Download latest market data"
    echo "  model-train  Train prediction models"
    echo "  prediction   Generate predictions"
    echo "  health-check Check system health"
    echo ""
    echo "Environment variables:"
    echo "  ENVIRONMENT     development|production (default: production)"
    echo "  LOG_LEVEL       DEBUG|INFO|WARNING|ERROR (default: INFO)"
    echo "  TZ              Timezone (default: UTC)"
    echo ""
}

# Main execution
main() {
    print_header
    
    # Set default environment
    export ENVIRONMENT="${ENVIRONMENT:-production}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    # Initialize system
    init_config
    check_dependencies
    setup_directories
    setup_database
    
    # Parse command
    local command="${1:-scheduler}"
    
    case "$command" in
        "scheduler")
            preflight_checks
            start_scheduler
            ;;
        "dashboard")
            preflight_checks
            start_dashboard
            ;;
        "test")
            run_tests
            ;;
        "task")
            if [ -z "$2" ]; then
                print_error "Task name required"
                show_help
                exit 1
            fi
            run_task "$2"
            ;;
        "shell")
            print_status "Starting interactive shell..."
            exec /bin/bash
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
