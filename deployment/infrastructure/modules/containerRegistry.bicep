// Container Registry Module
// Azure Container Registry for storing Docker images

@description('Name of the Container Registry')
param name string

@description('Location for the Container Registry')
param location string = resourceGroup().location

@description('SKU for the Container Registry')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Basic'

@description('Enable admin user for ACR')
param adminUserEnabled bool = true

@description('Tags to apply to the resource')
param tags object = {}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: adminUserEnabled
    policies: {
      retentionPolicy: {
        status: sku == 'Premium' ? 'enabled' : 'disabled'
        days: 30
      }
      trustPolicy: {
        type: 'Notary'
        status: 'disabled'
      }
    }
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
  }
  tags: tags
}

@description('The resource ID of the Container Registry')
output id string = containerRegistry.id

@description('The name of the Container Registry')
output name string = containerRegistry.name

@description('The login server URL of the Container Registry')
output loginServer string = containerRegistry.properties.loginServer

@description('The admin username (if enabled)')
output adminUsername string = adminUserEnabled ? containerRegistry.listCredentials().username : ''
