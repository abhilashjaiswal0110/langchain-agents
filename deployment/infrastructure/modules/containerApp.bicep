// Container App Module
// LangChain Platform Container App deployment

@description('Name of the Container App')
param name string

@description('Location for the Container App')
param location string = resourceGroup().location

@description('Resource ID of the Container Apps Environment')
param containerAppsEnvironmentId string

@description('Container image to deploy')
param containerImage string

@description('Container Registry login server')
param containerRegistryLoginServer string

@description('Container Registry username')
param containerRegistryUsername string

@description('Container Registry password')
@secure()
param containerRegistryPassword string

@description('CPU cores for the container (e.g., 0.5, 1, 2)')
param cpu string = '0.5'

@description('Memory for the container (e.g., 1Gi, 2Gi)')
param memory string = '1Gi'

@description('Minimum number of replicas')
@minValue(0)
@maxValue(30)
param minReplicas int = 1

@description('Maximum number of replicas')
@minValue(1)
@maxValue(30)
param maxReplicas int = 3

@description('Enable external ingress')
param externalIngress bool = true

@description('Target port for the container')
param targetPort int = 8000

@description('OpenAI API Key')
@secure()
param openAIApiKey string = ''

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

@description('API Key for webhook authentication')
@secure()
param apiKey string = ''

@description('Enable API Key authentication')
param apiKeyEnabled bool = false

@description('CORS Origins')
param corsOrigins string = '*'

@description('Memory backend type')
@allowed(['memory', 'redis', 'sqlite'])
param memoryBackend string = 'memory'

@description('Azure AD Tenant ID')
param azureTenantId string = ''

@description('Azure AD Client ID')
param azureClientId string = ''

@description('Azure AD Client Secret')
@secure()
param azureClientSecret string = ''

@description('Enable Azure AD authentication')
param authEnabled bool = false

@description('Application Insights connection string')
param appInsightsConnectionString string = ''

@description('Tags to apply to the resource')
param tags object = {}

// Build environment variables array
var baseEnvVars = [
  { name: 'PORT', value: '${targetPort}' }
  { name: 'LANGCHAIN_TRACING_V2', value: enableTracing ? 'true' : 'false' }
  { name: 'LANGCHAIN_PROJECT', value: langsmithProject }
  { name: 'LANGCHAIN_ENDPOINT', value: 'https://api.smith.langchain.com' }
  { name: 'API_KEY_ENABLED', value: apiKeyEnabled ? 'true' : 'false' }
  { name: 'CORS_ORIGINS', value: corsOrigins }
  { name: 'MEMORY_BACKEND', value: memoryBackend }
  { name: 'AUTH_ENABLED', value: authEnabled ? 'true' : 'false' }
  { name: 'AUTH_BYPASS_DEV', value: 'false' }
]

var secretEnvVars = [
  { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
  { name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }
  { name: 'LANGCHAIN_API_KEY', secretRef: 'langsmith-api-key' }
  { name: 'TAVILY_API_KEY', secretRef: 'tavily-api-key' }
  { name: 'API_KEY', secretRef: 'api-key' }
]

var azureAdEnvVars = authEnabled ? [
  { name: 'AZURE_TENANT_ID', value: azureTenantId }
  { name: 'AZURE_CLIENT_ID', value: azureClientId }
  { name: 'AZURE_CLIENT_SECRET', secretRef: 'azure-client-secret' }
] : []

var appInsightsEnvVars = !empty(appInsightsConnectionString) ? [
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsightsConnectionString }
] : []

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: externalIngress ? {
        external: true
        targetPort: targetPort
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
        corsPolicy: {
          allowedOrigins: corsOrigins == '*' ? ['*'] : split(corsOrigins, ',')
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
          allowCredentials: corsOrigins != '*'
        }
      } : null
      secrets: [
        { name: 'registry-password', value: containerRegistryPassword }
        { name: 'openai-api-key', value: openAIApiKey }
        { name: 'anthropic-api-key', value: anthropicApiKey }
        { name: 'langsmith-api-key', value: langsmithApiKey }
        { name: 'tavily-api-key', value: tavilyApiKey }
        { name: 'api-key', value: apiKey }
        { name: 'azure-client-secret', value: azureClientSecret }
      ]
      registries: [
        {
          server: containerRegistryLoginServer
          username: containerRegistryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'langchain-platform'
          image: containerImage
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: concat(baseEnvVars, secretEnvVars, azureAdEnvVars, appInsightsEnvVars)
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: targetPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: targetPort
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
  tags: tags
}

@description('The resource ID of the Container App')
output id string = containerApp.id

@description('The name of the Container App')
output name string = containerApp.name

@description('The FQDN of the Container App')
output fqdn string = externalIngress ? containerApp.properties.configuration.ingress.fqdn : ''

@description('The URL of the Container App')
output url string = externalIngress ? 'https://${containerApp.properties.configuration.ingress.fqdn}' : ''

@description('The latest revision name')
output latestRevisionName string = containerApp.properties.latestRevisionName
