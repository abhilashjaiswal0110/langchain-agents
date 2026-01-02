// Container Apps Environment Module
// Managed environment for running container apps

@description('Name of the Container Apps Environment')
param name string

@description('Location for the Container Apps Environment')
param location string = resourceGroup().location

@description('Resource ID of the Log Analytics Workspace')
param logAnalyticsWorkspaceId string

@description('Customer ID of the Log Analytics Workspace')
param logAnalyticsCustomerId string

@description('Primary shared key of the Log Analytics Workspace')
@secure()
param logAnalyticsPrimarySharedKey string

@description('Enable internal load balancer (for VNet integration)')
param internalOnly bool = false

@description('Tags to apply to the resource')
param tags object = {}

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: name
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsCustomerId
        sharedKey: logAnalyticsPrimarySharedKey
      }
    }
    zoneRedundant: false
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
  tags: tags
}

@description('The resource ID of the Container Apps Environment')
output id string = containerAppsEnvironment.id

@description('The name of the Container Apps Environment')
output name string = containerAppsEnvironment.name

@description('The default domain of the Container Apps Environment')
output defaultDomain string = containerAppsEnvironment.properties.defaultDomain

@description('The static IP of the Container Apps Environment')
output staticIp string = containerAppsEnvironment.properties.staticIp
