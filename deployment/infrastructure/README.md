# Azure Infrastructure for LangChain Platform

This directory contains Azure Bicep templates for deploying the LangChain Platform to Azure Container Apps.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure Resource Group                          │
│                                                                       │
│  ┌─────────────────┐     ┌─────────────────────────────────────┐    │
│  │ Container       │     │ Container Apps Environment           │    │
│  │ Registry (ACR)  │────▶│ ┌─────────────────────────────────┐ │    │
│  │                 │     │ │   LangChain Platform            │ │    │
│  └─────────────────┘     │ │   Container App                 │ │    │
│                          │ │   - FastAPI Server              │ │    │
│  ┌─────────────────┐     │ │   - LangChain Chains            │ │    │
│  │ Log Analytics   │◀────│ │   - IT Support Agents           │ │    │
│  │ Workspace       │     │ │   - Health Endpoints            │ │    │
│  └─────────────────┘     │ └─────────────────────────────────┘ │    │
│           │              └─────────────────────────────────────┘    │
│           ▼                                                          │
│  ┌─────────────────┐                                                 │
│  │ Application     │                                                 │
│  │ Insights        │                                                 │
│  └─────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Azure CLI** installed and logged in
2. **Bicep CLI** (included with Azure CLI 2.20.0+)
3. **Azure subscription** with Contributor access
4. **Key Vault** with API keys stored as secrets

## Quick Start

### 1. Create Resource Group

```bash
az group create \
  --name rg-langchain-dev \
  --location westeurope
```

### 2. Create Key Vault and Store Secrets

```bash
# Create Key Vault
az keyvault create \
  --name kv-langchain-dev \
  --resource-group rg-langchain-dev \
  --location westeurope

# Store secrets
az keyvault secret set --vault-name kv-langchain-dev --name openai-api-key --value "sk-..."
az keyvault secret set --vault-name kv-langchain-dev --name anthropic-api-key --value "sk-ant-..."
az keyvault secret set --vault-name kv-langchain-dev --name langsmith-api-key --value "lsv2_..."
az keyvault secret set --vault-name kv-langchain-dev --name tavily-api-key --value "tvly-..."
```

### 3. Update Parameters File

Edit `parameters.dev.json` and replace:
- `{subscription-id}` with your Azure subscription ID
- `{rg-name}` with your resource group name
- `{vault-name}` with your Key Vault name

### 4. Deploy Infrastructure

```bash
# Validate the template
az deployment group validate \
  --resource-group rg-langchain-dev \
  --template-file main.bicep \
  --parameters @parameters.dev.json

# Deploy
az deployment group create \
  --resource-group rg-langchain-dev \
  --template-file main.bicep \
  --parameters @parameters.dev.json \
  --name langchain-$(date +%Y%m%d%H%M%S)
```

### 5. Build and Push Docker Image

```bash
# Get ACR login server
ACR_NAME=$(az deployment group show \
  --resource-group rg-langchain-dev \
  --name langchain-* \
  --query "properties.outputs.containerRegistryName.value" -o tsv)

# Login to ACR
az acr login --name $ACR_NAME

# Build and push from deployment directory
cd ../
docker build -t $ACR_NAME.azurecr.io/langchain-platform:latest .
docker push $ACR_NAME.azurecr.io/langchain-platform:latest
```

## Module Reference

### main.bicep

Main orchestration template that deploys all resources.

**Required Parameters:**
| Parameter | Description |
|-----------|-------------|
| `openAIApiKey` | OpenAI API key (from Key Vault) |

**Optional Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `environment` | `dev` | Environment (dev/staging/prod) |
| `baseName` | `langchain` | Base name for resources |
| `imageTag` | `latest` | Docker image tag |
| `minReplicas` | `1` | Minimum container replicas |
| `maxReplicas` | `3` | Maximum container replicas |
| `cpu` | `0.5` | CPU cores per container |
| `memory` | `1Gi` | Memory per container |

### Modules

| Module | Description |
|--------|-------------|
| `containerRegistry.bicep` | Azure Container Registry |
| `logAnalytics.bicep` | Log Analytics Workspace |
| `applicationInsights.bicep` | Application Insights |
| `containerAppsEnvironment.bicep` | Container Apps Environment |
| `containerApp.bicep` | LangChain Platform Container App |

## CI/CD Integration

The GitHub Actions workflow (`.github/workflows/deploy-platform.yml`) automates:

1. **Test**: Run pytest and linting
2. **Build**: Build Docker image and push to ACR
3. **Deploy**: Update Container App with new image
4. **Infrastructure**: Deploy Bicep templates (manual trigger)

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AZURE_CLIENT_ID` | Azure AD app client ID |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `OPENAI_API_KEY_TEST` | OpenAI key for tests |

### Required GitHub Variables

| Variable | Description |
|----------|-------------|
| `AZURE_RESOURCE_GROUP` | Target resource group |
| `AZURE_CONTAINER_REGISTRY` | ACR name (without .azurecr.io) |

## Environment Configurations

### Development (parameters.dev.json)
- Single replica
- In-memory storage
- No authentication
- Full CORS access

### Production (parameters.prod.json)
- 2-10 replicas with autoscaling
- Redis memory backend
- Azure AD authentication enabled
- Restricted CORS origins
- API key authentication

## Monitoring

After deployment, access monitoring:

1. **Application Insights**: Azure Portal > Application Insights > langchain-{env}-insights
2. **Container Logs**: Azure Portal > Container Apps > langchain-{env}-app > Logs
3. **Metrics**: Azure Portal > Container Apps > Metrics

## Troubleshooting

### Container not starting

```bash
# Check container logs
az containerapp logs show \
  --name langchain-dev-app \
  --resource-group rg-langchain-dev \
  --type console
```

### Health check failing

```bash
# Check revision status
az containerapp revision list \
  --name langchain-dev-app \
  --resource-group rg-langchain-dev \
  --output table
```

### Image pull errors

```bash
# Verify ACR access
az acr login --name langchaindevacr

# List images
az acr repository list --name langchaindevacr
```
