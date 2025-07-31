# Azure Cloud Deployment Guide
# S&P 500 Prediction System on Microsoft Azure

## Overview

This guide provides comprehensive instructions for deploying the S&P 500 Prediction System to Microsoft Azure using multiple deployment options.

## Deployment Architecture

```
Azure Cloud Deployment Architecture
├── Resource Group (sp500-prediction-rg)
├── Azure Container Registry (ACR)
├── Azure Container Instances (ACI) / App Service
├── Azure Storage Account (File Share)
├── Azure Key Vault (Secrets Management)
├── Log Analytics Workspace (Monitoring)
├── Application Insights (APM)
└── Azure Load Balancer (Optional)
```

## Prerequisites

### 1. Azure Account Setup
- Active Azure subscription
- Azure CLI installed (`az --version`)
- PowerShell with Az modules (for Windows)
- Docker Desktop installed
- Git repository access

### 2. Required Tools
```bash
# Install Azure CLI (macOS)
brew install azure-cli

# Install Azure CLI (Linux)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Azure CLI (Windows)
# Download from: https://aka.ms/installazurecliwindows
```

### 3. Azure PowerShell Modules
```powershell
# Install Azure PowerShell modules
Install-Module -Name Az -Force -AllowClobber
Import-Module Az
```

## Deployment Options

### Option 1: Azure Container Instances (Recommended)

#### Step 1: Quick Deployment with Bash Script
```bash
# Navigate to deploy directory
cd deploy

# Make script executable
chmod +x azure_deploy.sh

# Deploy to Azure
./azure_deploy.sh deploy

# Check status
./azure_deploy.sh status

# Cleanup (when needed)
./azure_deploy.sh cleanup
```

#### Step 2: Manual Deployment Steps
```bash
# 1. Login to Azure
az login
az account set --subscription "your-subscription-id"

# 2. Create resource group
RESOURCE_GROUP="sp500-prediction-rg"
LOCATION="eastus"
az group create --name $RESOURCE_GROUP --location $LOCATION

# 3. Create Azure Container Registry
ACR_NAME="sp500predictionacr"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Standard --admin-enabled true

# 4. Build and push Docker image
az acr login --name $ACR_NAME
cd ..
docker build -t sp500-prediction .
docker tag sp500-prediction $ACR_NAME.azurecr.io/sp500-prediction:latest
docker push $ACR_NAME.azurecr.io/sp500-prediction:latest

# 5. Create storage account
STORAGE_ACCOUNT="sp500storage$(date +%s)"
az storage account create --resource-group $RESOURCE_GROUP --name $STORAGE_ACCOUNT --location $LOCATION --sku Standard_LRS

# 6. Create file share
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT --query '[0].value' --output tsv)
az storage share create --name sp500data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY

# 7. Deploy container group
az container create \
  --resource-group $RESOURCE_GROUP \
  --name sp500-prediction \
  --image $ACR_NAME.azurecr.io/sp500-prediction:latest \
  --registry-login-server $ACR_NAME.azurecr.io \
  --registry-username $(az acr credential show --name $ACR_NAME --query username --output tsv) \
  --registry-password $(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv) \
  --dns-name-label sp500-prediction-$(date +%s) \
  --ports 8501 \
  --cpu 2 \
  --memory 4 \
  --environment-variables ENVIRONMENT=production TZ=America/New_York \
  --azure-file-volume-share-name sp500data \
  --azure-file-volume-account-name $STORAGE_ACCOUNT \
  --azure-file-volume-account-key $STORAGE_KEY \
  --azure-file-volume-mount-path /app/data
```

### Option 2: Azure App Service

#### Step 1: PowerShell Deployment
```powershell
# Navigate to deploy directory
cd deploy

# Run PowerShell deployment script
./azure_deploy.ps1 -Action Deploy -ResourceGroupName "sp500-prediction-rg" -Location "East US"

# Check status
./azure_deploy.ps1 -Action Status

# Update deployment
./azure_deploy.ps1 -Action Update

# Cleanup
./azure_deploy.ps1 -Action Cleanup
```

#### Step 2: ARM Template Deployment
```bash
# Deploy using ARM template
az deployment group create \
  --resource-group sp500-prediction-rg \
  --template-file azure_app_service_template.json \
  --parameters appName=sp500-prediction \
               registryUsername=$(az acr credential show --name $ACR_NAME --query username --output tsv) \
               registryPassword=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)
```

### Option 3: Azure Kubernetes Service (AKS)

#### Step 1: Create AKS Cluster
```bash
# Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name sp500-aks \
  --node-count 2 \
  --node-vm-size Standard_B2s \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name sp500-aks

# Attach ACR to AKS
az aks update --resource-group $RESOURCE_GROUP --name sp500-aks --attach-acr $ACR_NAME
```

#### Step 2: Deploy to Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sp500-prediction
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sp500-prediction
  template:
    metadata:
      labels:
        app: sp500-prediction
    spec:
      containers:
      - name: sp500-app
        image: sp500predictionacr.azurecr.io/sp500-prediction:latest
        ports:
        - containerPort: 8501
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: TZ
          value: "America/New_York"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: sp500-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: sp500-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: sp500-prediction
```

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# Get service IP
kubectl get service sp500-service
```

## Configuration Management

### Environment Variables
```bash
# Essential production settings
ENVIRONMENT=production
TZ=America/New_York
LOG_LEVEL=INFO

# Data and model paths
DATA_PATH=/app/data
MODELS_PATH=/app/models
LOGS_PATH=/app/logs

# Scheduling
DATA_UPDATE_SCHEDULE="0 22 * * MON-FRI"
PREDICTION_SCHEDULE="0 23 * * MON-FRI"
MODEL_RETRAIN_SCHEDULE="0 2 * * SUN"

# Azure-specific
AZURE_STORAGE_ACCOUNT=your_storage_account
AZURE_STORAGE_KEY=your_storage_key
AZURE_FILE_SHARE=sp500data

# Email notifications
EMAIL_NOTIFICATIONS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### Azure Key Vault Integration
```bash
# Store secrets in Key Vault
VAULT_NAME="sp500-kv-$(date +%s)"
az keyvault create --name $VAULT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION

# Add secrets
az keyvault secret set --vault-name $VAULT_NAME --name "smtp-password" --value "your-app-password"
az keyvault secret set --vault-name $VAULT_NAME --name "storage-key" --value "$STORAGE_KEY"
az keyvault secret set --vault-name $VAULT_NAME --name "acr-password" --value "$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)"
```

## Monitoring and Logging

### Azure Monitor Setup
```bash
# Create Log Analytics workspace
WORKSPACE_NAME="sp500-logs"
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $WORKSPACE_NAME \
  --location $LOCATION

# Create Application Insights
APPINSIGHTS_NAME="sp500-insights"
az monitor app-insights component create \
  --app $APPINSIGHTS_NAME \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --workspace $(az monitor log-analytics workspace show --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --query id --output tsv)
```

### Monitoring Queries (KQL)
```kql
-- Container performance
ContainerInstanceLog_CL
| where TimeGenerated > ago(1h)
| where ContainerGroup_s == "sp500-prediction"
| project TimeGenerated, ContainerName_s, Message

-- Application errors
traces
| where timestamp > ago(1h)
| where severityLevel >= 3
| project timestamp, message, severityLevel

-- Resource utilization
Perf
| where TimeGenerated > ago(1h)
| where CounterName == "% Processor Time"
| summarize avg(CounterValue) by bin(TimeGenerated, 5m)
```

## Scaling and Performance

### Auto-scaling Configuration
```yaml
# For App Service
az appservice plan update \
  --name sp500-plan \
  --resource-group $RESOURCE_GROUP \
  --sku P1v2

# Enable autoscaling
az monitor autoscale create \
  --resource-group $RESOURCE_GROUP \
  --resource sp500-prediction \
  --resource-type Microsoft.Web/sites \
  --name sp500-autoscale \
  --min-count 1 \
  --max-count 5 \
  --count 2
```

### Performance Optimization
- **CPU**: 2-4 cores for scheduler, 1-2 for dashboard
- **Memory**: 2-4GB for data processing, 1-2GB for dashboard
- **Storage**: Premium SSD for faster data access
- **Network**: CDN for static assets (optional)

## Security Best Practices

### 1. Network Security
```bash
# Create private endpoints (premium feature)
az network private-endpoint create \
  --resource-group $RESOURCE_GROUP \
  --name sp500-storage-endpoint \
  --location $LOCATION \
  --subnet subnet-id \
  --private-connection-resource-id $(az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --query id --output tsv) \
  --connection-name sp500-storage-connection \
  --group-id file
```

### 2. Identity and Access Management
```bash
# Create managed identity
az identity create --resource-group $RESOURCE_GROUP --name sp500-identity

# Assign roles
az role assignment create \
  --assignee $(az identity show --resource-group $RESOURCE_GROUP --name sp500-identity --query principalId --output tsv) \
  --role "Storage Blob Data Contributor" \
  --scope $(az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --query id --output tsv)
```

### 3. SSL/TLS Configuration
```bash
# For App Service - enable HTTPS only
az webapp update --resource-group $RESOURCE_GROUP --name sp500-prediction --https-only true

# Custom domain and SSL (optional)
az webapp config hostname add --webapp-name sp500-prediction --resource-group $RESOURCE_GROUP --hostname yourdomain.com
```

## Backup and Disaster Recovery

### 1. Data Backup
```bash
# Automated backup using Azure Backup
az backup vault create \
  --resource-group $RESOURCE_GROUP \
  --name sp500-backup-vault \
  --location $LOCATION

# Backup file share
az backup protection enable-for-azurefileshare \
  --vault-name sp500-backup-vault \
  --resource-group $RESOURCE_GROUP \
  --policy-name DefaultPolicy \
  --storage-account $STORAGE_ACCOUNT \
  --azure-file-share sp500data
```

### 2. Multi-Region Deployment
```bash
# Deploy to secondary region
SECONDARY_REGION="westus2"
az group create --name "${RESOURCE_GROUP}-dr" --location $SECONDARY_REGION

# Replicate storage
az storage account create \
  --name "${STORAGE_ACCOUNT}dr" \
  --resource-group "${RESOURCE_GROUP}-dr" \
  --location $SECONDARY_REGION \
  --sku Standard_LRS \
  --kind StorageV2
```

## Cost Optimization

### 1. Resource Sizing
```bash
# Monitor costs
az consumption usage list --start-date 2024-01-01 --end-date 2024-01-31

# Right-size resources based on usage
az container show --resource-group $RESOURCE_GROUP --name sp500-prediction --query "containers[0].resources"
```

### 2. Scheduled Start/Stop
```bash
# Schedule container start/stop for cost savings
# Create Azure Automation runbook for scheduling
az automation runbook create \
  --automation-account-name sp500-automation \
  --resource-group $RESOURCE_GROUP \
  --name start-stop-containers \
  --type PowerShell
```

## Troubleshooting

### Common Issues

#### 1. Container Startup Issues
```bash
# Check container logs
az container logs --resource-group $RESOURCE_GROUP --name sp500-prediction --container-name sp500-scheduler

# Check container events
az container show --resource-group $RESOURCE_GROUP --name sp500-prediction --query "containers[0].instanceView"
```

#### 2. Storage Access Issues
```bash
# Test storage connectivity
az storage file list --share-name sp500data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY

# Check file share permissions
az storage share show --name sp500data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
```

#### 3. Image Pull Errors
```bash
# Check ACR permissions
az acr repository show --name $ACR_NAME --repository sp500-prediction

# Test ACR login
az acr login --name $ACR_NAME
docker pull $ACR_NAME.azurecr.io/sp500-prediction:latest
```

### Diagnostic Commands
```bash
# Resource group overview
az group show --name $RESOURCE_GROUP --output table

# Container group status
az container show --resource-group $RESOURCE_GROUP --name sp500-prediction --output table

# Storage account status
az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --output table

# ACR repositories
az acr repository list --name $ACR_NAME --output table
```

## Cleanup and Resource Management

### Complete Cleanup
```bash
# Delete entire resource group (careful!)
az group delete --name $RESOURCE_GROUP --yes --no-wait

# Or use the provided scripts
./azure_deploy.sh cleanup
./azure_deploy.ps1 -Action Cleanup
```

### Selective Cleanup
```bash
# Delete specific resources
az container delete --resource-group $RESOURCE_GROUP --name sp500-prediction --yes
az acr delete --resource-group $RESOURCE_GROUP --name $ACR_NAME --yes
az storage account delete --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --yes
```

## Next Steps

1. **Production Readiness**
   - Configure SSL certificates
   - Set up monitoring alerts
   - Implement backup strategies
   - Configure auto-scaling

2. **Integration**
   - Connect to external data sources
   - Integrate with CI/CD pipelines
   - Set up notification systems
   - Configure logging aggregation

3. **Optimization**
   - Monitor performance metrics
   - Optimize resource allocation
   - Implement caching strategies
   - Configure CDN for static assets

## Support and Resources

- **Azure Documentation**: https://docs.microsoft.com/en-us/azure/
- **Azure CLI Reference**: https://docs.microsoft.com/en-us/cli/azure/
- **Container Instances Docs**: https://docs.microsoft.com/en-us/azure/container-instances/
- **App Service Docs**: https://docs.microsoft.com/en-us/azure/app-service/

---

**Your S&P 500 prediction system is now ready for enterprise-scale deployment on Microsoft Azure!** 🚀
