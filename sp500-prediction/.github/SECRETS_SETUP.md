# GitHub Repository Secrets Configuration Guide
# Required secrets for CI/CD pipeline

## Azure Deployment Secrets

### AZURE_CREDENTIALS
# Azure Service Principal credentials for deployment
# Create with: az ad sp create-for-rbac --name "sp500-github-actions" --role contributor --scopes /subscriptions/{subscription-id} --sdk-auth
# Format: JSON object containing clientId, clientSecret, subscriptionId, tenantId
{
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "subscriptionId": "your-subscription-id",
  "tenantId": "your-tenant-id"
}

### ACR_USERNAME
# Azure Container Registry username
# Get with: az acr credential show --name sp500predictionacr --query username --output tsv

### ACR_PASSWORD
# Azure Container Registry password
# Get with: az acr credential show --name sp500predictionacr --query passwords[0].value --output tsv

### AZURE_STORAGE_KEY
# Azure Storage Account key for persistent data
# Get with: az storage account keys list --resource-group sp500-prediction-rg --account-name sp500storage --query [0].value --output tsv

## Application Secrets

### SMTP_PASSWORD
# Email notification password (app password for Gmail)
# Create app password in Gmail security settings

### DATABASE_URL (Optional)
# PostgreSQL connection string for production
# Format: postgresql://username:password@hostname:port/database

### REDIS_URL (Optional)
# Redis connection string for caching
# Format: redis://username:password@hostname:port/database

### SENTRY_DSN (Optional)
# Sentry error tracking DSN
# Get from Sentry project settings

## API Keys (Optional)

### ALPHA_VANTAGE_API_KEY
# Alternative data source API key

### QUANDL_API_KEY
# Financial data API key

### SLACK_WEBHOOK_URL
# Slack notifications webhook URL

### DISCORD_WEBHOOK_URL
# Discord notifications webhook URL

## Security Secrets

### ENCRYPTION_KEY
# Application encryption key (32 bytes base64 encoded)
# Generate with: python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"

### JWT_SECRET_KEY
# JWT token signing key
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"

## Instructions for Setting Up Secrets

### 1. Azure Setup
```bash
# Login to Azure
az login

# Create service principal for GitHub Actions
az ad sp create-for-rbac \
  --name "sp500-github-actions" \
  --role contributor \
  --scopes /subscriptions/{your-subscription-id} \
  --sdk-auth

# Create Azure Container Registry
az acr create \
  --resource-group sp500-prediction-rg \
  --name sp500predictionacr \
  --sku Standard \
  --admin-enabled true

# Get ACR credentials
az acr credential show --name sp500predictionacr
```

### 2. GitHub Repository Setup
1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Add each secret using the "New repository secret" button
4. Use the exact names listed above (case-sensitive)

### 3. Environment-specific Secrets
- **Production**: All secrets required
- **Staging**: Subset of secrets for testing
- **Development**: Local environment variables

### 4. Secret Rotation
- Azure credentials: Rotate every 90 days
- API keys: Monitor usage and rotate as needed
- Database passwords: Rotate every 6 months
- Encryption keys: Only rotate when necessary (breaks existing data)

### 5. Security Best Practices
- Use least privilege principle for Azure credentials
- Enable secret scanning in GitHub repository
- Monitor secret usage in GitHub Actions logs
- Use environment-specific secrets where possible
- Regularly audit and remove unused secrets

## Testing Secrets Setup

### Validate Azure Credentials
```bash
# Test Azure CLI login with service principal
az login --service-principal \
  --username $CLIENT_ID \
  --password $CLIENT_SECRET \
  --tenant $TENANT_ID

# Test ACR access
docker login sp500predictionacr.azurecr.io \
  --username $ACR_USERNAME \
  --password $ACR_PASSWORD
```

### Validate SMTP Configuration
```python
import smtplib
from email.mime.text import MimeText

def test_smtp():
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your-email@gmail.com', 'your-app-password')
    print("✅ SMTP configuration valid")
    server.quit()
```

## Troubleshooting

### Common Issues
1. **Azure login fails**: Check service principal credentials and permissions
2. **ACR push fails**: Verify ACR credentials and repository exists
3. **Storage access fails**: Check storage account key and permissions
4. **SMTP fails**: Verify app password and 2FA settings

### Debug Commands
```bash
# Test Azure authentication
az account show

# Test ACR connectivity
az acr check-health --name sp500predictionacr

# Test storage account access
az storage account show --name sp500storage --resource-group sp500-prediction-rg
```
