"""Datacenter Operations Domain Agent.

Provides specialized support for:
- Physical servers and hardware
- Storage systems (SAN, NAS)
- Network infrastructure
- Datacenter facilities
- Hardware provisioning
"""

from langchain_core.tools import BaseTool, tool

from app.agents.domains.base_domain_agent import DomainAgent, DomainConfig, DomainType


@tool
def check_server_status(server_name: str) -> str:
    """Check the status of a physical server.

    Args:
        server_name: Server hostname or ID.
    """
    return f"""Server Status: {server_name}
- Status: Online
- Uptime: 45 days
- CPU: 35% avg
- Memory: 62% used
- Storage: 78% used
- Last Backup: 2 hours ago
- Location: DC1-Rack-A3-U12"""


@tool
def check_storage_capacity(storage_system: str) -> str:
    """Check storage system capacity.

    Args:
        storage_system: Storage array name.
    """
    return f"""Storage System: {storage_system}
- Total Capacity: 500 TB
- Used: 380 TB (76%)
- Available: 120 TB
- Performance Tier: 40 TB available
- Archive Tier: 80 TB available
- Alert Threshold: 85%"""


@tool
def request_hardware(hardware_type: str, justification: str) -> str:
    """Request new hardware provisioning.

    Args:
        hardware_type: Type of hardware needed.
        justification: Business justification.
    """
    return f"""Hardware Request Submitted:
- Type: {hardware_type}
- Justification: {justification}
- Request ID: HW-{hash(justification) % 10000:04d}
- Status: Pending approval
- Lead Time: 4-6 weeks (after approval)
You'll receive updates via email."""


@tool
def check_network_health(network_segment: str) -> str:
    """Check network health for a segment.

    Args:
        network_segment: Network segment or VLAN.
    """
    return f"""Network Health: {network_segment}
- Status: Healthy
- Bandwidth: 45% utilized
- Latency: 2ms avg
- Packet Loss: 0.01%
- Active Ports: 48/96
- Last Issue: None in 30 days"""


class DatacenterAgent(DomainAgent):
    """Datacenter Operations specialist agent."""

    def get_config(self) -> DomainConfig:
        """Get Datacenter configuration."""
        return DomainConfig(
            domain=DomainType.DATACENTER,
            name="Datacenter Operations",
            description="Support for physical infrastructure, servers, storage, and networking",
            expertise=[
                "physical servers",
                "storage systems",
                "network infrastructure",
                "hardware provisioning",
                "backup systems",
                "datacenter facilities",
                "rack management",
                "cabling",
            ],
            escalation_keywords=[
                "outage",
                "power failure",
                "cooling",
                "fire",
                "physical access",
                "emergency",
            ],
            requires_approval=[
                "hardware purchases",
                "datacenter access",
                "network changes",
            ],
        )

    def get_tools(self) -> list[BaseTool]:
        """Get Datacenter tools."""
        return [
            check_server_status,
            check_storage_capacity,
            request_hardware,
            check_network_health,
        ]

    def get_system_prompt(self) -> str:
        """Get Datacenter system prompt."""
        return """You are the Datacenter Operations specialist for the IT support team.

Your expertise includes:
- Physical server management and monitoring
- Storage systems (SAN, NAS, backup)
- Network infrastructure (switches, routers, cabling)
- Hardware lifecycle management
- Datacenter facilities (power, cooling, access)
- Capacity planning

When helping users:
1. Check current system status before making changes
2. Verify hardware requests have proper justification
3. Escalate any outages or facility issues immediately
4. Coordinate with vendors for hardware support
5. Maintain accurate inventory and documentation

Prioritize system reliability and uptime."""
