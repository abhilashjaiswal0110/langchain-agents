"""Cloud Infrastructure Domain Agent.

Provides specialized support for:
- Azure, AWS, GCP resources
- Virtual machines and containers
- Cloud networking
- PaaS services
- Cost optimization
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def list_cloud_resources(resource_type: str, environment: str = "production") -> str:
    """List cloud resources by type and environment.

    Args:
        resource_type: Type of resource (vm, storage, database, etc.).
        environment: Environment (production, staging, development).
    """
    return f"""Cloud Resources ({resource_type} in {environment}):
1. {resource_type.upper()}-prod-001 - Running - Standard_D4s_v3
2. {resource_type.upper()}-prod-002 - Running - Standard_D4s_v3
3. {resource_type.upper()}-prod-003 - Stopped - Standard_D2s_v3
Total: 3 resources
Monthly Cost: ~$850"""


@tool
def check_vm_status(vm_name: str) -> str:
    """Check status of a virtual machine.

    Args:
        vm_name: VM name or ID.
    """
    return f"""VM Status: {vm_name}
- Power State: Running
- Size: Standard_D4s_v3
- vCPUs: 4
- Memory: 16 GB
- OS: Ubuntu 22.04 LTS
- Private IP: 10.0.1.45
- Region: Germany West Central
- Uptime: 15 days"""


@tool
def request_cloud_resource(resource_type: str, specifications: str) -> str:
    """Request a new cloud resource.

    Args:
        resource_type: Type of resource needed.
        specifications: Resource specifications.
    """
    return f"""Cloud Resource Request Submitted:
- Type: {resource_type}
- Specifications: {specifications}
- Request ID: CLOUD-{hash(specifications) % 10000:04d}
- Status: Pending review
- Estimated Provisioning: 1-2 business days
Cost estimate will be provided after review."""


@tool
def check_cloud_costs(account: str = "default") -> str:
    """Check cloud spending and costs.

    Args:
        account: Account or subscription name.
    """
    return f"""Cloud Cost Summary ({account}):
- Current Month: $12,450
- Last Month: $11,200
- Forecast: $13,500
- Budget: $15,000
- Top Services:
  - Compute: $5,200 (42%)
  - Storage: $3,100 (25%)
  - Networking: $2,150 (17%)
  - Other: $2,000 (16%)"""


@tool
def manage_container(action: str, container_name: str) -> str:
    """Manage container or Kubernetes resource.

    Args:
        action: Action to perform (status, restart, scale).
        container_name: Container or deployment name.
    """
    actions = {
        "status": f"Container {container_name}: Running (3 replicas)",
        "restart": f"Restarting {container_name}... Complete. New pods healthy.",
        "scale": f"Scaling {container_name}... Specify replicas (current: 3)",
    }
    return actions.get(action.lower(), f"Unknown action: {action}")


class CloudAgent(DomainAgent):
    """Cloud Infrastructure specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get Cloud configuration."""
        return DomainConfig(
            domain=DomainType.CLOUD,
            name="Cloud Infrastructure",
            description="Support for Azure, AWS, GCP, VMs, containers, and cloud services",
            expertise=[
                "azure",
                "aws",
                "gcp",
                "virtual machines",
                "kubernetes",
                "containers",
                "cloud networking",
                "iaas",
                "paas",
                "cost optimization",
            ],
            escalation_keywords=[
                "production outage",
                "security breach",
                "data loss",
                "billing dispute",
            ],
            requires_approval=[
                "production changes",
                "large resource requests",
                "cost increases",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Cloud tools."""
        return [
            list_cloud_resources,
            check_vm_status,
            request_cloud_resource,
            check_cloud_costs,
            manage_container,
        ]

    def get_system_prompt(self) -> str:
        """Get Cloud system prompt."""
        return """You are the Cloud Infrastructure specialist for the IT support team.

Your expertise includes:
- Azure (VMs, AKS, Storage, Networking)
- AWS (EC2, EKS, S3, VPC)
- GCP (Compute, GKE, Cloud Storage)
- Kubernetes and container orchestration
- Cloud networking and security
- Cost optimization and governance
- Infrastructure as Code (Terraform, Bicep)

When helping users:
1. Check existing resources before provisioning new ones
2. Consider cost implications of all requests
3. Follow least-privilege for access requests
4. Require approval for production changes
5. Monitor for unused or oversized resources

Optimize for cost, security, and reliability."""
