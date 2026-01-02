// LangChain Platform - Azure Infrastructure
// Main deployment template for Container Apps deployment
//
// Usage:
//   az deployment group create \
//     --resource-group <resource-group> \
//     --template-file main.bicep \
//     --parameters @parameters.json
//
// Required parameters:
//   - openAIApiKey: OpenAI API key
//
// Optional but recommended:
//   - anthropicApiKey: Anthropic API key
//   - langsmithApiKey: LangSmith API key for tracing

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Base name for all resources')
param baseName string = 'langchain'

@description('Container image tag')
param imageTag string = 'latest'

// LLM API Keys
@description('OpenAI API Key')
@secure()
param openAIApiKey string

@description('Anthropic API Key')
@secure()
param anthropicApiKey string = ''

@description('LangSmith API Key')
@secure()
param langsmithApiKey string = ''

@description('LangSmith Project Name')
param langsmithProject string = 'langchain-platform'

@description('Enable LangSmith Tracing')
param enableTracing bool = true

@description('Tavily API Key for web search')
@secure()
param tavilyApiKey string = ''

// Security settings
@description('API Key for webhook authentication')
@secure()
param apiKey string = ''

@description('Enable API Key authentication')
param apiKeyEnabled bool = false

@description('CORS Origins')
param corsOrigins string = '*'

// Azure AD settings
@description('Azure AD Tenant ID')
param azureTenantId string = ''

@description('Azure AD Client ID')
param azureClientId string = ''

@description('Azure AD Client Secret')
@secure()
param azureClientSecret string = ''

@description('Enable Azure AD authentication')
param authEnabled bool = false

// Memory settings
@description('Memory backend type')
@allowed(['memory', 'redis', 'sqlite'])
param memoryBackend string = 'memory'

// Scaling settings
@description('Minimum replicas')
@minValue(0)
@maxValue(30)
param minReplicas int = 1

@description('Maximum replicas')
@minValue(1)
@maxValue(30)
param maxReplicas int = 3

@description('CPU cores')
param cpu string = '0.5'

@description('Memory')
param memory string = '1Gi'

// Tags
var tags = {
  environment: environment
  application: 'langchain-platform'
  managedBy: 'bicep'
}

// Resource naming
var resourcePrefix = '${baseName}-${environment}'
var containerRegistryName = replace('${baseName}${environment}acr', '-', '')
var logAnalyticsName = '${resourcePrefix}-logs'
var appInsightsName = '${resourcePrefix}-insights'
var containerAppsEnvName = '${resourcePrefix}-env'
var containerAppName = '${resourcePrefix}-app'

// Container Registry
module containerRegistry 'modules/containerRegistry.bicep' = {
  name: 'containerRegistry'
  params: {
    name: containerRegistryName
    location: location
    sku: environment == 'prod' ? 'Standard' : 'Basic'
    adminUserEnabled: true
    tags: tags
  }
}

// Log Analytics Workspace
module logAnalytics 'modules/logAnalytics.bicep' = {
  name: 'logAnalytics'
  params: {
    name: logAnalyticsName
    location: location
    sku: 'PerGB2018'
    retentionInDays: environment == 'prod' ? 90 : 30
    tags: tags
  }
}

// Application Insights
module appInsights 'modules/applicationInsights.bicep' = {
  name: 'appInsights'
  params: {
    name: appInsightsName
    location: location
    logAnalyticsWorkspaceId: logAnalytics.outputs.id
    applicationType: 'web'
    tags: tags
  }
}

// Container Apps Environment
module containerAppsEnvironment 'modules/containerAppsEnvironment.bicep' = {
  name: 'containerAppsEnvironment'
  params: {
    name: containerAppsEnvName
    location: location
    logAnalyticsWorkspaceId: logAnalytics.outputs.id
    logAnalyticsCustomerId: logAnalytics.outputs.customerId
    logAnalyticsPrimarySharedKey: logAnalytics.outputs.primarySharedKey
    internalOnly: false
    tags: tags
  }
}

// Container App - LangChain Platform
module containerApp 'modules/containerApp.bicep' = {
  name: 'containerApp'
  params: {
    name: containerAppName
    location: location
    containerAppsEnvironmentId: containerAppsEnvironment.outputs.id
    containerImage: '${containerRegistry.outputs.loginServer}/langchain-platform:${imageTag}'
    containerRegistryLoginServer: containerRegistry.outputs.loginServer
    containerRegistryUsername: containerRegistry.outputs.adminUsername
    containerRegistryPassword: listCredentials(containerRegistry.outputs.id, '2023-07-01').passwords[0].value
    cpu: cpu
    memory: memory
    minReplicas: minReplicas
    maxReplicas: maxReplicas
    externalIngress: true
    targetPort: 8000
    openAIApiKey: openAIApiKey
    anthropicApiKey: anthropicApiKey
    langsmithApiKey: langsmithApiKey
    langsmithProject: langsmithProject
    enableTracing: enableTracing
    tavilyApiKey: tavilyApiKey
    apiKey: apiKey
    apiKeyEnabled: apiKeyEnabled
    corsOrigins: corsOrigins
    memoryBackend: memoryBackend
    azureTenantId: azureTenantId
    azureClientId: azureClientId
    azureClientSecret: azureClientSecret
    authEnabled: authEnabled
    appInsightsConnectionString: appInsights.outputs.connectionString
    tags: tags
  }
}

// Outputs
@description('Container Registry login server')
output containerRegistryLoginServer string = containerRegistry.outputs.loginServer

@description('Container Registry name')
output containerRegistryName string = containerRegistry.outputs.name

@description('Container App URL')
output containerAppUrl string = containerApp.outputs.url

@description('Container App FQDN')
output containerAppFqdn string = containerApp.outputs.fqdn

@description('Application Insights connection string')
output appInsightsConnectionString string = appInsights.outputs.connectionString

@description('Application Insights instrumentation key')
output appInsightsInstrumentationKey string = appInsights.outputs.instrumentationKey

@description('Log Analytics Workspace ID')
output logAnalyticsWorkspaceId string = logAnalytics.outputs.id

@description('Container Apps Environment name')
output containerAppsEnvironmentName string = containerAppsEnvironment.outputs.name
