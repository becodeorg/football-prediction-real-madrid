# S&P 500 Prediction System - Docker Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the S&P 500 prediction system using Docker containers in production environments.

## Prerequisites

### System Requirements

- **Docker**: Version 20.0+ with Docker Compose
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 10GB free space minimum
- **CPU**: 2+ cores recommended
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Network Requirements

- Internet access for data downloads
- SMTP access for email notifications (optional)
- Port 8501 available for dashboard (customizable)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd sp500-prediction

# Create data directories
mkdir -p docker_data/{data,models,logs,config}

# Set permissions (Linux/macOS)
chmod -R 755 docker_data/
```

### 2. Configuration

```bash
# Copy environment template
cp .env.production .env

# Edit configuration
nano .env  # or your preferred editor

# Copy scheduler configuration
cp scheduler/scheduler_config.template.json docker_data/config/scheduler_config.json
```

### 3. Build and Deploy

```bash
# Build the Docker image
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Access Services

- **Dashboard**: http://localhost:8501
- **Logs**: `docker-compose logs -f scheduler`
- **Health**: `docker-compose exec scheduler python healthcheck.py`

## Detailed Configuration

### Environment Variables

Edit the `.env` file to customize your deployment:

#### Essential Settings

```bash
# Environment
ENVIRONMENT=production
TZ=America/New_York  # Your timezone

# Email notifications
EMAIL_NOTIFICATIONS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
NOTIFICATION_RECIPIENTS=admin@yourcompany.com

# Scheduling (adjust for your timezone)
DATA_UPDATE_SCHEDULE=0 22 * * MON-FRI  # 10 PM weekdays
PREDICTION_SCHEDULE=0 23 * * MON-FRI   # 11 PM weekdays
MODEL_RETRAIN_SCHEDULE=0 2 * * SUN     # 2 AM Sunday
```

#### Performance Tuning

```bash
# Resource allocation
MAX_WORKERS=3
N_JOBS=-1  # Use all CPU cores
PANDAS_MEMORY_LIMIT=2000  # MB
```

### Scheduler Configuration

Edit `docker_data/config/scheduler_config.json`:

```json
{
  "scheduler_type": "blocking",
  "timezone": "America/New_York",
  "max_workers": 3,
  "tasks": {
    "daily_data_update": {
      "enabled": true,
      "schedule": {
        "type": "cron",
        "hour": 22,
        "minute": 0,
        "day_of_week": "mon-fri"
      }
    }
  }
}
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Production deployment
docker-compose up -d

# Scale services
docker-compose up -d --scale scheduler=1 --scale dashboard=2

# Update services
docker-compose pull
docker-compose up -d
```

### Option 2: Docker Run

```bash
# Build image
docker build -t sp500-prediction .

# Run scheduler
docker run -d \
  --name sp500-scheduler \
  --restart unless-stopped \
  -v $(pwd)/docker_data/data:/app/data \
  -v $(pwd)/docker_data/models:/app/models \
  -v $(pwd)/docker_data/logs:/app/logs \
  --env-file .env \
  sp500-prediction:latest scheduler

# Run dashboard
docker run -d \
  --name sp500-dashboard \
  --restart unless-stopped \
  -p 8501:8501 \
  -v $(pwd)/docker_data/data:/app/data:ro \
  -v $(pwd)/docker_data/models:/app/models:ro \
  --env-file .env \
  sp500-prediction:latest dashboard
```

### Option 3: Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sp500-scheduler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sp500-scheduler
  template:
    metadata:
      labels:
        app: sp500-scheduler
    spec:
      containers:
      - name: scheduler
        image: sp500-prediction:latest
        command: ["scheduler"]
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
        envFrom:
        - configMapRef:
            name: sp500-config
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: sp500-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: sp500-models-pvc
```

## Operations

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d scheduler

# Start with monitoring
docker-compose --profile monitoring up -d
```

### Monitoring

```bash
# View logs
docker-compose logs -f scheduler
docker-compose logs -f dashboard

# Check resource usage
docker stats

# Health checks
docker-compose exec scheduler python healthcheck.py

# Service status
docker-compose ps
```

### Maintenance

```bash
# Update configuration
docker-compose restart scheduler

# Backup data
docker run --rm \
  -v sp500_data:/backup-source:ro \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/sp500-data-$(date +%Y%m%d).tar.gz -C /backup-source .

# Update images
docker-compose pull
docker-compose up -d

# Clean unused resources
docker system prune -f
```

### Scaling

```bash
# Scale dashboard for high availability
docker-compose up -d --scale dashboard=3

# Load balancer configuration (nginx)
upstream sp500_dashboard {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}
```

## Security

### Container Security

```bash
# Run as non-root user (already configured)
USER sp500

# Read-only root filesystem (add to docker-compose.yml)
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
```

### Network Security

```yaml
# docker-compose.yml
networks:
  sp500_network:
    driver: bridge
    internal: true  # No external access
```

### Secrets Management

```bash
# Use Docker secrets
docker secret create smtp_password smtp_password.txt

# Reference in compose file
secrets:
  - smtp_password
```

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker-compose logs scheduler

# Check configuration
docker-compose config

# Validate environment
docker-compose exec scheduler env | grep -E "ENVIRONMENT|LOG_LEVEL"
```

#### Import Errors

```bash
# Check Python environment
docker-compose exec scheduler python --version

# Test imports
docker-compose exec scheduler python -c "import pandas, numpy, sklearn, yfinance"

# Check PYTHONPATH
docker-compose exec scheduler python -c "import sys; print(sys.path)"
```

#### Data Issues

```bash
# Check data directories
docker-compose exec scheduler ls -la /app/data

# Test data download
docker-compose exec scheduler python scheduler/daily_data_update.py --test

# Check permissions
docker-compose exec scheduler stat /app/data
```

#### Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Adjust memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

### Performance Optimization

#### CPU Optimization

```bash
# Set CPU affinity
docker-compose exec scheduler taskset -c 0-3 python scheduler/scheduler_main.py
```

#### Memory Optimization

```bash
# Reduce pandas memory usage
PANDAS_MEMORY_LIMIT=1000

# Limit scikit-learn threads
SKLEARN_THREAD_LIMIT=2
```

#### Storage Optimization

```bash
# Clean old logs
find docker_data/logs -name "*.log" -mtime +7 -delete

# Compress models
docker-compose exec scheduler gzip /app/models/*.pkl
```

## Production Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Email notifications tested
- [ ] Data directories created with correct permissions
- [ ] Firewall rules configured
- [ ] SSL certificates installed (if using HTTPS)
- [ ] Backup strategy implemented

### Post-Deployment

- [ ] Services started successfully
- [ ] Health checks passing
- [ ] Dashboard accessible
- [ ] Data updates working
- [ ] Predictions being generated
- [ ] Email notifications working
- [ ] Log rotation configured
- [ ] Monitoring alerts set up

### Ongoing Maintenance

- [ ] Regular backups
- [ ] Log monitoring
- [ ] Performance monitoring
- [ ] Security updates
- [ ] Configuration updates
- [ ] Data cleanup

## Integration Examples

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name sp500.yourcompany.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Monitoring (Prometheus)

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sp500-prediction'
    static_configs:
      - targets: ['localhost:8502']  # Metrics endpoint
```

### Alerting (Grafana)

```json
{
  "alert": {
    "name": "SP500 Scheduler Down",
    "message": "S&P 500 prediction scheduler is not responding",
    "frequency": "30s",
    "conditions": [
      {
        "query": {
          "queryType": "",
          "refId": "A"
        },
        "reducer": {
          "type": "last",
          "params": []
        },
        "evaluator": {
          "params": [1],
          "type": "lt"
        }
      }
    ]
  }
}
```

## Support

For issues and questions:

1. Check logs: `docker-compose logs -f`
2. Run health check: `docker-compose exec scheduler python healthcheck.py`
3. Validate configuration: `docker-compose config`
4. Review documentation: `AUTOMATION_SETUP_GUIDE.md`

Your S&P 500 prediction system is now ready for production deployment!
