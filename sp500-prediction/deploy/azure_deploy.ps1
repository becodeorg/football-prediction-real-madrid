# Azure PowerShell Deployment Script for S&P 500 Prediction System
# Deploy using Azure Resource Manager templates and PowerShell

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "sp500-prediction-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "East US",
    
    [Parameter(Mandatory=$false)]
    [string]$AppName = "sp500-prediction",
    
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("Deploy", "Cleanup", "Status", "Update")]
    [string]$Action = "Deploy"
)

# Import required modules
Import-Module Az.Accounts -Force
Import-Module Az.Resources -Force
Import-Module Az.Storage -Force
Import-Module Az.KeyVault -Force
Import-Module Az.ContainerRegistry -Force
Import-Module Az.ContainerInstance -Force
Import-Module Az.OperationalInsights -Force
Import-Module Az.ApplicationInsights -Force

# Global variables
$Global:DeploymentConfig = @{
    ResourceGroupName = $ResourceGroupName
    Location = $Location
    AppName = $AppName
    ACRName = "$AppName-acr-$(Get-Random -Minimum 1000 -Maximum 9999)"
    StorageAccountName = "$($AppName.Replace('-',''))storage$(Get-Random -Minimum 1000 -Maximum 9999)"
    KeyVaultName = "$AppName-kv-$(Get-Random -Minimum 1000 -Maximum 9999)"
    ContainerGroupName = "$AppName-containers"
    LogWorkspaceName = "$AppName-logs"
    AppInsightsName = "$AppName-insights"
}

# Logging functions
function Write-DeploymentLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARNING" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-AzurePowerShell {
    Write-DeploymentLog "Checking Azure PowerShell modules..."
    
    $requiredModules = @("Az.Accounts", "Az.Resources", "Az.Storage", "Az.KeyVault")
    foreach ($module in $requiredModules) {
        if (!(Get-Module -ListAvailable -Name $module)) {
            Write-DeploymentLog "Installing module: $module" -Level "WARNING"
            Install-Module -Name $module -Force -AllowClobber
        }
    }
    Write-DeploymentLog "Azure PowerShell modules ready" -Level "SUCCESS"
}

function Connect-AzureAccount {
    Write-DeploymentLog "Connecting to Azure..."
    
    try {
        $context = Get-AzContext
        if (!$context) {
            Connect-AzAccount
        }
        
        if ($SubscriptionId) {
            Set-AzContext -SubscriptionId $SubscriptionId
        }
        
        $subscription = Get-AzContext
        Write-DeploymentLog "Connected to subscription: $($subscription.Subscription.Name)" -Level "SUCCESS"
        return $subscription
    }
    catch {
        Write-DeploymentLog "Failed to connect to Azure: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-ResourceGroup {
    Write-DeploymentLog "Creating resource group: $($Global:DeploymentConfig.ResourceGroupName)"
    
    try {
        $rg = Get-AzResourceGroup -Name $Global:DeploymentConfig.ResourceGroupName -ErrorAction SilentlyContinue
        if (!$rg) {
            $rg = New-AzResourceGroup -Name $Global:DeploymentConfig.ResourceGroupName -Location $Global:DeploymentConfig.Location -Tag @{
                Project = "sp500-prediction"
                Environment = "production"
                CreatedBy = "PowerShell"
                CreatedDate = (Get-Date).ToString("yyyy-MM-dd")
            }
            Write-DeploymentLog "Created resource group: $($rg.ResourceGroupName)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "Resource group already exists: $($rg.ResourceGroupName)" -Level "WARNING"
        }
        return $rg
    }
    catch {
        Write-DeploymentLog "Failed to create resource group: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-ContainerRegistry {
    Write-DeploymentLog "Creating Azure Container Registry: $($Global:DeploymentConfig.ACRName)"
    
    try {
        $acr = Get-AzContainerRegistry -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.ACRName -ErrorAction SilentlyContinue
        if (!$acr) {
            $acr = New-AzContainerRegistry -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.ACRName -Location $Global:DeploymentConfig.Location -Sku Standard -EnableAdminUser
            Write-DeploymentLog "Created ACR: $($acr.Name)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "ACR already exists: $($acr.Name)" -Level "WARNING"
        }
        
        # Get credentials
        $acrCredentials = Get-AzContainerRegistryCredential -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.ACRName
        return @{
            Registry = $acr
            Credentials = $acrCredentials
        }
    }
    catch {
        Write-DeploymentLog "Failed to create ACR: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-StorageAccount {
    Write-DeploymentLog "Creating storage account: $($Global:DeploymentConfig.StorageAccountName)"
    
    try {
        $storage = Get-AzStorageAccount -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.StorageAccountName -ErrorAction SilentlyContinue
        if (!$storage) {
            $storage = New-AzStorageAccount -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.StorageAccountName -Location $Global:DeploymentConfig.Location -SkuName Standard_LRS -Kind StorageV2 -Tag @{
                Project = "sp500-prediction"
                Environment = "production"
            }
            Write-DeploymentLog "Created storage account: $($storage.StorageAccountName)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "Storage account already exists: $($storage.StorageAccountName)" -Level "WARNING"
        }
        
        # Create file share
        $ctx = $storage.Context
        $fileShare = Get-AzStorageShare -Name "sp500data" -Context $ctx -ErrorAction SilentlyContinue
        if (!$fileShare) {
            $fileShare = New-AzStorageShare -Name "sp500data" -Context $ctx -Quota 100
            Write-DeploymentLog "Created file share: sp500data" -Level "SUCCESS"
        }
        
        return $storage
    }
    catch {
        Write-DeploymentLog "Failed to create storage account: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-KeyVault {
    Write-DeploymentLog "Creating Key Vault: $($Global:DeploymentConfig.KeyVaultName)"
    
    try {
        $kv = Get-AzKeyVault -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -VaultName $Global:DeploymentConfig.KeyVaultName -ErrorAction SilentlyContinue
        if (!$kv) {
            $kv = New-AzKeyVault -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -VaultName $Global:DeploymentConfig.KeyVaultName -Location $Global:DeploymentConfig.Location -Tag @{
                Project = "sp500-prediction"
                Environment = "production"
            }
            Write-DeploymentLog "Created Key Vault: $($kv.VaultName)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "Key Vault already exists: $($kv.VaultName)" -Level "WARNING"
        }
        
        return $kv
    }
    catch {
        Write-DeploymentLog "Failed to create Key Vault: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-LogAnalyticsWorkspace {
    Write-DeploymentLog "Creating Log Analytics workspace: $($Global:DeploymentConfig.LogWorkspaceName)"
    
    try {
        $workspace = Get-AzOperationalInsightsWorkspace -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.LogWorkspaceName -ErrorAction SilentlyContinue
        if (!$workspace) {
            $workspace = New-AzOperationalInsightsWorkspace -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.LogWorkspaceName -Location $Global:DeploymentConfig.Location -Sku PerGB2018 -Tag @{
                Project = "sp500-prediction"
                Environment = "production"
            }
            Write-DeploymentLog "Created Log Analytics workspace: $($workspace.Name)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "Log Analytics workspace already exists: $($workspace.Name)" -Level "WARNING"
        }
        
        return $workspace
    }
    catch {
        Write-DeploymentLog "Failed to create Log Analytics workspace: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function New-ApplicationInsights {
    param($WorkspaceResourceId)
    
    Write-DeploymentLog "Creating Application Insights: $($Global:DeploymentConfig.AppInsightsName)"
    
    try {
        $appInsights = Get-AzApplicationInsights -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.AppInsightsName -ErrorAction SilentlyContinue
        if (!$appInsights) {
            $appInsights = New-AzApplicationInsights -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.AppInsightsName -Location $Global:DeploymentConfig.Location -WorkspaceResourceId $WorkspaceResourceId -Tag @{
                Project = "sp500-prediction"
                Environment = "production"
            }
            Write-DeploymentLog "Created Application Insights: $($appInsights.Name)" -Level "SUCCESS"
        } else {
            Write-DeploymentLog "Application Insights already exists: $($appInsights.Name)" -Level "WARNING"
        }
        
        return $appInsights
    }
    catch {
        Write-DeploymentLog "Failed to create Application Insights: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function Build-DockerImage {
    param($ACRLoginServer, $ACRCredentials)
    
    Write-DeploymentLog "Building and pushing Docker image..."
    
    try {
        # Login to ACR
        $acrPassword = $ACRCredentials.Password | ConvertTo-SecureString -AsPlainText -Force
        $acrCredential = New-Object System.Management.Automation.PSCredential($ACRCredentials.Username, $acrPassword)
        
        # Build and push image (requires Docker CLI)
        $imageName = "sp500-prediction"
        $imageTag = "latest"
        $fullImageName = "$ACRLoginServer/$imageName`:$imageTag"
        
        # Change to project root
        Set-Location ..
        
        # Build image
        Write-DeploymentLog "Building Docker image: $fullImageName"
        & docker build -t $imageName`:$imageTag .
        if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
        
        # Tag for ACR
        & docker tag $imageName`:$imageTag $fullImageName
        if ($LASTEXITCODE -ne 0) { throw "Docker tag failed" }
        
        # Login to ACR
        & docker login $ACRLoginServer -u $ACRCredentials.Username -p $ACRCredentials.Password
        if ($LASTEXITCODE -ne 0) { throw "Docker login failed" }
        
        # Push image
        Write-DeploymentLog "Pushing image to ACR: $fullImageName"
        & docker push $fullImageName
        if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }
        
        # Return to deploy directory
        Set-Location deploy
        
        Write-DeploymentLog "Image pushed successfully: $fullImageName" -Level "SUCCESS"
        return $fullImageName
    }
    catch {
        Write-DeploymentLog "Failed to build/push Docker image: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function Deploy-ContainerGroup {
    param($ImageName, $StorageAccount, $ACRCredentials)
    
    Write-DeploymentLog "Deploying container group: $($Global:DeploymentConfig.ContainerGroupName)"
    
    try {
        # Get storage key
        $storageKey = (Get-AzStorageAccountKey -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $StorageAccount.StorageAccountName)[0].Value
        
        # Create container group using ARM template
        $templateParams = @{
            containerGroupName = $Global:DeploymentConfig.ContainerGroupName
            imageName = $ImageName
            acrLoginServer = $ACRCredentials.Registry.LoginServer
            acrUsername = $ACRCredentials.Credentials.Username
            acrPassword = $ACRCredentials.Credentials.Password
            storageAccountName = $StorageAccount.StorageAccountName
            storageAccountKey = $storageKey
            location = $Global:DeploymentConfig.Location
        }
        
        # Deploy using template file (you'll need to create this template)
        $deployment = New-AzResourceGroupDeployment -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -TemplateFile "azure_container_template.json" -TemplateParameterObject $templateParams
        
        if ($deployment.ProvisioningState -eq "Succeeded") {
            Write-DeploymentLog "Container group deployed successfully" -Level "SUCCESS"
            
            # Get container group details
            $containerGroup = Get-AzContainerGroup -ResourceGroupName $Global:DeploymentConfig.ResourceGroupName -Name $Global:DeploymentConfig.ContainerGroupName
            
            return @{
                ContainerGroup = $containerGroup
                PublicIP = $containerGroup.IpAddress
                FQDN = $containerGroup.Fqdn
                DashboardUrl = "http://$($containerGroup.Fqdn):8501"
            }
        } else {
            throw "Deployment failed with state: $($deployment.ProvisioningState)"
        }
    }
    catch {
        Write-DeploymentLog "Failed to deploy container group: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function Save-DeploymentInfo {
    param($DeploymentResult)
    
    $deploymentInfo = @{
        ResourceGroupName = $Global:DeploymentConfig.ResourceGroupName
        Location = $Global:DeploymentConfig.Location
        AppName = $Global:DeploymentConfig.AppName
        ACRName = $Global:DeploymentConfig.ACRName
        StorageAccountName = $Global:DeploymentConfig.StorageAccountName
        KeyVaultName = $Global:DeploymentConfig.KeyVaultName
        ContainerGroupName = $Global:DeploymentConfig.ContainerGroupName
        PublicIP = $DeploymentResult.PublicIP
        FQDN = $DeploymentResult.FQDN
        DashboardUrl = $DeploymentResult.DashboardUrl
        DeploymentDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    
    $deploymentInfo | ConvertTo-Json | Out-File "azure_deployment_info.json" -Encoding UTF8
    Write-DeploymentLog "Deployment information saved to azure_deployment_info.json" -Level "SUCCESS"
    
    return $deploymentInfo
}

function Start-Deployment {
    Write-DeploymentLog "Starting Azure deployment for S&P 500 Prediction System" -Level "SUCCESS"
    Write-Host "=============================================================" -ForegroundColor Blue
    
    try {
        # Pre-deployment checks
        Test-AzurePowerShell
        $subscription = Connect-AzureAccount
        
        # Core infrastructure
        $resourceGroup = New-ResourceGroup
        $acrResult = New-ContainerRegistry
        $storageAccount = New-StorageAccount
        $keyVault = New-KeyVault
        $logWorkspace = New-LogAnalyticsWorkspace
        $appInsights = New-ApplicationInsights -WorkspaceResourceId $logWorkspace.ResourceId
        
        # Build and deploy application
        $imageName = Build-DockerImage -ACRLoginServer $acrResult.Registry.LoginServer -ACRCredentials $acrResult.Credentials
        $containerDeployment = Deploy-ContainerGroup -ImageName $imageName -StorageAccount $storageAccount -ACRCredentials $acrResult
        
        # Save deployment information
        $deploymentInfo = Save-DeploymentInfo -DeploymentResult $containerDeployment
        
        # Display results
        Write-Host ""
        Write-Host "================= DEPLOYMENT COMPLETE =================" -ForegroundColor Green
        Write-Host "Resource Group: $($deploymentInfo.ResourceGroupName)" -ForegroundColor White
        Write-Host "Dashboard URL: $($deploymentInfo.DashboardUrl)" -ForegroundColor Cyan
        Write-Host "Public IP: $($deploymentInfo.PublicIP)" -ForegroundColor White
        Write-Host "FQDN: $($deploymentInfo.FQDN)" -ForegroundColor White
        Write-Host "=======================================================" -ForegroundColor Green
        
        return $deploymentInfo
    }
    catch {
        Write-DeploymentLog "Deployment failed: $($_.Exception.Message)" -Level "ERROR"
        throw
    }
}

function Remove-Deployment {
    Write-DeploymentLog "Removing Azure deployment..." -Level "WARNING"
    
    $confirmation = Read-Host "Are you sure you want to delete all resources in $($Global:DeploymentConfig.ResourceGroupName)? (y/N)"
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        try {
            Remove-AzResourceGroup -Name $Global:DeploymentConfig.ResourceGroupName -Force
            Write-DeploymentLog "Resource group deletion initiated" -Level "SUCCESS"
        }
        catch {
            Write-DeploymentLog "Failed to delete resource group: $($_.Exception.Message)" -Level "ERROR"
            throw
        }
    } else {
        Write-DeploymentLog "Cleanup cancelled" -Level "WARNING"
    }
}

function Get-DeploymentStatus {
    if (Test-Path "azure_deployment_info.json") {
        $deploymentInfo = Get-Content "azure_deployment_info.json" | ConvertFrom-Json
        
        Write-Host "================= DEPLOYMENT STATUS =================" -ForegroundColor Blue
        Write-Host "Resource Group: $($deploymentInfo.ResourceGroupName)" -ForegroundColor White
        Write-Host "Container Group: $($deploymentInfo.ContainerGroupName)" -ForegroundColor White
        Write-Host "Dashboard URL: $($deploymentInfo.DashboardUrl)" -ForegroundColor Cyan
        Write-Host "Deployment Date: $($deploymentInfo.DeploymentDate)" -ForegroundColor White
        Write-Host "====================================================" -ForegroundColor Blue
        
        # Check container status
        try {
            $containerGroup = Get-AzContainerGroup -ResourceGroupName $deploymentInfo.ResourceGroupName -Name $deploymentInfo.ContainerGroupName
            Write-Host "Container Status: $($containerGroup.State)" -ForegroundColor Green
        }
        catch {
            Write-Host "Container Status: Not Found" -ForegroundColor Red
        }
    } else {
        Write-DeploymentLog "No deployment information found. Run deployment first." -Level "WARNING"
    }
}

function Update-Deployment {
    Write-DeploymentLog "Updating Azure deployment..." -Level "WARNING"
    
    if (Test-Path "azure_deployment_info.json") {
        $deploymentInfo = Get-Content "azure_deployment_info.json" | ConvertFrom-Json
        
        # Update container image
        try {
            $acrResult = Get-AzContainerRegistry -ResourceGroupName $deploymentInfo.ResourceGroupName -Name $deploymentInfo.ACRName
            $acrCredentials = Get-AzContainerRegistryCredential -ResourceGroupName $deploymentInfo.ResourceGroupName -Name $deploymentInfo.ACRName
            
            $imageName = Build-DockerImage -ACRLoginServer $acrResult.LoginServer -ACRCredentials $acrCredentials
            
            # Restart container group to pull new image
            Restart-AzContainerGroup -ResourceGroupName $deploymentInfo.ResourceGroupName -Name $deploymentInfo.ContainerGroupName
            
            Write-DeploymentLog "Deployment updated successfully" -Level "SUCCESS"
        }
        catch {
            Write-DeploymentLog "Failed to update deployment: $($_.Exception.Message)" -Level "ERROR"
            throw
        }
    } else {
        Write-DeploymentLog "No existing deployment found. Run deployment first." -Level "ERROR"
    }
}

# Main execution
switch ($Action) {
    "Deploy" {
        Start-Deployment
    }
    "Cleanup" {
        Remove-Deployment
    }
    "Status" {
        Get-DeploymentStatus
    }
    "Update" {
        Update-Deployment
    }
    default {
        Write-Host "Usage: ./azure_deploy.ps1 -Action [Deploy|Cleanup|Status|Update]" -ForegroundColor Yellow
        Write-Host "  Deploy  - Deploy the application to Azure"
        Write-Host "  Cleanup - Delete all Azure resources"
        Write-Host "  Status  - Check deployment status"
        Write-Host "  Update  - Update existing deployment"
    }
}
