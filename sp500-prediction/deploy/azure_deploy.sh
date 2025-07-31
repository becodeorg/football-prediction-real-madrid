#!/bin/bash
# Azure Container Instances Deployment Script
# Deploy S&P 500 Prediction System to Azure

set -e

# Configuration
RESOURCE_GROUP="sp500-prediction-rg"
LOCATION="eastus"
CONTAINER_GROUP_NAME="sp500-prediction"
ACR_NAME="sp500predictionacr"
IMAGE_NAME="sp500-prediction"
IMAGE_TAG="latest"
STORAGE_ACCOUNT="sp500storage"
FILE_SHARE_NAME="sp500data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Azure CLI is installed
check_azure_cli() {
    log "Checking Azure CLI installation..."
    if ! command -v az &> /dev/null; then
        error "Azure CLI is not installed. Please install it first."
        echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    success "Azure CLI is installed"
}

# Login to Azure
azure_login() {
    log "Checking Azure login status..."
    if ! az account show &> /dev/null; then
        log "Please login to Azure..."
        az login
    fi
    
    SUBSCRIPTION_ID=$(az account show --query id --output tsv)
    SUBSCRIPTION_NAME=$(az account show --query name --output tsv)
    success "Logged in to Azure subscription: $SUBSCRIPTION_NAME ($SUBSCRIPTION_ID)"
}

# Create resource group
create_resource_group() {
    log "Creating resource group: $RESOURCE_GROUP"
    
    if az group show --name $RESOURCE_GROUP &> /dev/null; then
        warning "Resource group $RESOURCE_GROUP already exists"
    else
        az group create \
            --name $RESOURCE_GROUP \
            --location $LOCATION \
            --tags project=sp500-prediction environment=production
        success "Created resource group: $RESOURCE_GROUP"
    fi
}

# Create Azure Container Registry
create_acr() {
    log "Creating Azure Container Registry: $ACR_NAME"
    
    if az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        warning "ACR $ACR_NAME already exists"
    else
        az acr create \
            --resource-group $RESOURCE_GROUP \
            --name $ACR_NAME \
            --sku Standard \
            --admin-enabled true \
            --tags project=sp500-prediction
        success "Created ACR: $ACR_NAME"
    fi
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
    log "ACR Login Server: $ACR_LOGIN_SERVER"
}

# Build and push Docker image
build_and_push_image() {
    log "Building and pushing Docker image..."
    
    # Login to ACR
    az acr login --name $ACR_NAME
    
    # Build image locally (from project root)
    cd ..
    log "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    docker build -t $IMAGE_NAME:$IMAGE_TAG .
    
    # Tag for ACR
    docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
    
    # Push to ACR
    log "Pushing image to ACR..."
    docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
    
    success "Image pushed to ACR: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
    cd deploy
}

# Create storage account and file share
create_storage() {
    log "Creating storage account: $STORAGE_ACCOUNT"
    
    if az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP &> /dev/null; then
        warning "Storage account $STORAGE_ACCOUNT already exists"
    else
        az storage account create \
            --resource-group $RESOURCE_GROUP \
            --name $STORAGE_ACCOUNT \
            --location $LOCATION \
            --sku Standard_LRS \
            --kind StorageV2 \
            --tags project=sp500-prediction
        success "Created storage account: $STORAGE_ACCOUNT"
    fi
    
    # Get storage key
    STORAGE_KEY=$(az storage account keys list \
        --resource-group $RESOURCE_GROUP \
        --account-name $STORAGE_ACCOUNT \
        --query '[0].value' \
        --output tsv)
    
    # Create file share
    log "Creating file share: $FILE_SHARE_NAME"
    if az storage share exists --name $FILE_SHARE_NAME --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY --output tsv | grep -q "True"; then
        warning "File share $FILE_SHARE_NAME already exists"
    else
        az storage share create \
            --name $FILE_SHARE_NAME \
            --account-name $STORAGE_ACCOUNT \
            --account-key $STORAGE_KEY \
            --quota 100
        success "Created file share: $FILE_SHARE_NAME"
    fi
}

# Create Key Vault for secrets
create_key_vault() {
    VAULT_NAME="sp500-kv-$(date +%s)"
    log "Creating Key Vault: $VAULT_NAME"
    
    az keyvault create \
        --name $VAULT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku standard \
        --tags project=sp500-prediction
    
    # Store ACR credentials
    ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)
    
    az keyvault secret set --vault-name $VAULT_NAME --name "acr-username" --value $ACR_USERNAME
    az keyvault secret set --vault-name $VAULT_NAME --name "acr-password" --value $ACR_PASSWORD
    az keyvault secret set --vault-name $VAULT_NAME --name "storage-key" --value $STORAGE_KEY
    
    success "Created Key Vault: $VAULT_NAME"
    echo "VAULT_NAME=$VAULT_NAME" >> azure_deployment.env
}

# Deploy container group
deploy_container_group() {
    log "Deploying container group: $CONTAINER_GROUP_NAME"
    
    # Get ACR credentials
    ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)
    
    # Deploy using YAML configuration
    cat > container-group.yaml << EOF
apiVersion: 2021-03-01
location: $LOCATION
name: $CONTAINER_GROUP_NAME
properties:
  containers:
  - name: sp500-scheduler
    properties:
      image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 2.0
      command:
        - /app/docker/entrypoint.sh
        - scheduler
      environmentVariables:
        - name: ENVIRONMENT
          value: production
        - name: TZ
          value: America/New_York
        - name: LOG_LEVEL
          value: INFO
      volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
        - name: logs-volume
          mountPath: /app/logs
  - name: sp500-dashboard
    properties:
      image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
      resources:
        requests:
          cpu: 0.5
          memoryInGb: 1.0
      command:
        - /app/docker/entrypoint.sh
        - dashboard
      ports:
        - port: 8501
          protocol: TCP
      environmentVariables:
        - name: ENVIRONMENT
          value: production
        - name: TZ
          value: America/New_York
      volumeMounts:
        - name: data-volume
          mountPath: /app/data
          readOnly: true
        - name: models-volume
          mountPath: /app/models
          readOnly: true
  imageRegistryCredentials:
  - server: $ACR_LOGIN_SERVER
    username: $ACR_USERNAME
    password: $ACR_PASSWORD
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8501
    dnsNameLabel: sp500-prediction-$RANDOM
  osType: Linux
  restartPolicy: Always
  volumes:
  - name: data-volume
    azureFile:
      shareName: $FILE_SHARE_NAME
      storageAccountName: $STORAGE_ACCOUNT
      storageAccountKey: $STORAGE_KEY
  - name: models-volume
    azureFile:
      shareName: $FILE_SHARE_NAME
      storageAccountName: $STORAGE_ACCOUNT
      storageAccountKey: $STORAGE_KEY
      readOnly: false
  - name: logs-volume
    azureFile:
      shareName: $FILE_SHARE_NAME
      storageAccountName: $STORAGE_ACCOUNT
      storageAccountKey: $STORAGE_KEY
tags:
  project: sp500-prediction
  environment: production
EOF

    # Deploy container group
    az container create \
        --resource-group $RESOURCE_GROUP \
        --file container-group.yaml
    
    success "Container group deployed successfully"
    
    # Get public IP
    PUBLIC_IP=$(az container show \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_GROUP_NAME \
        --query ipAddress.ip \
        --output tsv)
    
    FQDN=$(az container show \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_GROUP_NAME \
        --query ipAddress.fqdn \
        --output tsv)
    
    success "Deployment complete!"
    echo ""
    echo "===== DEPLOYMENT INFORMATION ====="
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Container Group: $CONTAINER_GROUP_NAME"
    echo "Public IP: $PUBLIC_IP"
    echo "FQDN: $FQDN"
    echo "Dashboard URL: http://$FQDN:8501"
    echo "=================================="
    
    # Save deployment info
    cat > azure_deployment.env << EOF
RESOURCE_GROUP=$RESOURCE_GROUP
CONTAINER_GROUP_NAME=$CONTAINER_GROUP_NAME
ACR_NAME=$ACR_NAME
STORAGE_ACCOUNT=$STORAGE_ACCOUNT
PUBLIC_IP=$PUBLIC_IP
FQDN=$FQDN
DASHBOARD_URL=http://$FQDN:8501
EOF
}

# Setup monitoring
setup_monitoring() {
    log "Setting up Azure Monitor..."
    
    # Create Log Analytics workspace
    WORKSPACE_NAME="sp500-logs"
    az monitor log-analytics workspace create \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE_NAME \
        --location $LOCATION \
        --tags project=sp500-prediction
    
    # Get workspace ID and key
    WORKSPACE_ID=$(az monitor log-analytics workspace show \
        --resource-group $RESOURCE_GROUP \
        --workspace-name $WORKSPACE_NAME \
        --query customerId \
        --output tsv)
    
    success "Log Analytics workspace created: $WORKSPACE_NAME"
    echo "Workspace ID: $WORKSPACE_ID"
}

# Main deployment function
main() {
    log "Starting Azure deployment for S&P 500 Prediction System"
    echo "=============================================="
    
    check_azure_cli
    azure_login
    create_resource_group
    create_acr
    create_storage
    build_and_push_image
    create_key_vault
    deploy_container_group
    setup_monitoring
    
    success "Azure deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Configure environment variables in Azure portal"
    echo "2. Set up SSL certificate for production"
    echo "3. Configure monitoring alerts"
    echo "4. Set up backup schedules"
    echo ""
    echo "Access your application at: $(cat azure_deployment.env | grep DASHBOARD_URL | cut -d'=' -f2)"
}

# Cleanup function
cleanup() {
    warning "Cleaning up Azure resources..."
    read -p "Are you sure you want to delete all resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        success "Resource group deletion initiated"
    else
        log "Cleanup cancelled"
    fi
}

# Script options
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        if [ -f azure_deployment.env ]; then
            source azure_deployment.env
            log "Checking deployment status..."
            az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_GROUP_NAME --output table
        else
            error "No deployment found. Run './azure_deploy.sh deploy' first."
        fi
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|status}"
        echo "  deploy  - Deploy the application to Azure"
        echo "  cleanup - Delete all Azure resources"
        echo "  status  - Check deployment status"
        exit 1
        ;;
esac
