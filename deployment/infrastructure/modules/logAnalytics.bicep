// Log Analytics Workspace Module
// Centralized logging for Container Apps and Application Insights

@description('Name of the Log Analytics Workspace')
param name string

@description('Location for the Log Analytics Workspace')
param location string = resourceGroup().location

@description('SKU for the Log Analytics Workspace')
@allowed(['Free', 'PerGB2018', 'PerNode', 'Premium', 'Standalone', 'Standard'])
param sku string = 'PerGB2018'

@description('Retention period in days')
@minValue(30)
@maxValue(730)
param retentionInDays int = 30

@description('Tags to apply to the resource')
param tags object = {}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: name
  location: location
  properties: {
    sku: {
      name: sku
    }
    retentionInDays: retentionInDays
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
    workspaceCapping: {
      dailyQuotaGb: -1 // Unlimited
    }
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: tags
}

@description('The resource ID of the Log Analytics Workspace')
output id string = logAnalytics.id

@description('The name of the Log Analytics Workspace')
output name string = logAnalytics.name

@description('The customer ID (workspace ID) of the Log Analytics Workspace')
output customerId string = logAnalytics.properties.customerId

@description('The primary shared key of the Log Analytics Workspace')
output primarySharedKey string = logAnalytics.listKeys().primarySharedKey
