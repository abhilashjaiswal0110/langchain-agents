// Application Insights Module
// APM and monitoring for the LangChain platform

@description('Name of the Application Insights resource')
param name string

@description('Location for the Application Insights resource')
param location string = resourceGroup().location

@description('Resource ID of the Log Analytics Workspace')
param logAnalyticsWorkspaceId string

@description('Application type')
@allowed(['web', 'other'])
param applicationType string = 'web'

@description('Tags to apply to the resource')
param tags object = {}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: name
  location: location
  kind: applicationType
  properties: {
    Application_Type: applicationType
    WorkspaceResourceId: logAnalyticsWorkspaceId
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    RetentionInDays: 90
  }
  tags: tags
}

@description('The resource ID of the Application Insights')
output id string = applicationInsights.id

@description('The name of the Application Insights')
output name string = applicationInsights.name

@description('The instrumentation key')
output instrumentationKey string = applicationInsights.properties.InstrumentationKey

@description('The connection string')
output connectionString string = applicationInsights.properties.ConnectionString
