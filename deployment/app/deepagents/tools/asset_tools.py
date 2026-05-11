"""Asset/CMDB Management Tools for Deep Agents.

Tools for querying and managing Configuration Items.
"""

from langchain_core.tools import tool

from app.agents.servicenow_agent import (
    CMDB_DB,
    get_api_client,
    is_live_mode,
)

# Extended CMDB data for relationships
CMDB_RELATIONSHIPS = {
    "PROD-WEB-01": {
        "runs_on": "VMW-HOST-01",
        "depends_on": ["PROD-DB-01", "PROD-LB-01"],
        "used_by": ["SAP-ERP", "CRM-APP"],
    },
    "PROD-DB-01": {
        "runs_on": "VMW-HOST-02",
        "depends_on": ["SAN-STORAGE-01"],
        "used_by": ["SAP-ERP", "PROD-WEB-01"],
    },
    "SAP-ERP": {
        "runs_on": "PROD-APP-01",
        "depends_on": ["PROD-DB-01", "PROD-WEB-01"],
        "used_by": ["Finance", "HR", "Operations"],
    },
}

# Service mapping
SERVICE_MAP = {
    "PROD-WEB-01": ["Web Portal", "Customer Portal"],
    "PROD-DB-01": ["ERP Database", "CRM Database"],
    "SAP-ERP": ["Finance Services", "HR Services", "Supply Chain"],
    "VPN-GATEWAY-01": ["Remote Access", "Partner VPN"],
    "EXCHANGE-01": ["Email Services", "Calendar", "Contacts"],
}


@tool
def search_cmdb(
    query: str | None = None,
    ci_class: str | None = None,
    status: str | None = None,
    location: str | None = None,
    limit: int = 10,
) -> str:
    """Search the Configuration Management Database (CMDB).

    Use this to find CIs for impact analysis, change planning,
    or incident investigation.

    Args:
        query: Search query for CI name or description.
        ci_class: Filter by class (Server, Application, Network Device, Database).
        status: Filter by status (Operational, Maintenance, Retired).
        location: Filter by location/datacenter.
        limit: Maximum number of results.

    Returns:
        List of matching configuration items.
    """
    if is_live_mode():
        try:
            api = get_api_client()
            cis = api.search_cmdb(
                query=query,
                ci_class=ci_class or "cmdb_ci",
                status=status,
                limit=limit,
            )

            if not cis:
                return "No configuration items found matching the criteria."

            output = [f"**CMDB Search Results ({len(cis)} items) [LIVE]:**\n"]
            for ci in cis:
                output.append(f"""
**{ci.get("name", "N/A")}**
- Class: {ci.get("sys_class_name", "Unknown")}
- Status: {ci.get("install_status", "Unknown")}
- Location: {ci.get("location", "N/A")}
""")
            return "\n".join(output)
        except Exception as e:
            return f"Error searching CMDB: {e}"

    # Simulation mode
    results = []
    for ci_id, ci in CMDB_DB.items():
        if ci_class and ci["class"].lower() != ci_class.lower():
            continue
        if status and ci["status"].lower() != status.lower():
            continue
        if location and ci.get("location", "").lower() != location.lower():
            continue
        if query and query.lower() not in ci["name"].lower():
            continue
        results.append((ci_id, ci))
        if len(results) >= limit:
            break

    if not results:
        return "No configuration items found matching the criteria. [SIMULATION]"

    output = [f"**CMDB Search Results ({len(results)} items) [SIMULATION]:**\n"]
    for ci_id, ci in results:
        output.append(f"""
**{ci["name"]}** ({ci_id})
- Class: {ci["class"]}
- Status: {ci["status"]}
- Owner: {ci["owner"]}
""")
    return "\n".join(output)


@tool
def get_ci_details(ci_name: str) -> str:
    """Get comprehensive details about a Configuration Item.

    Use this for detailed information about a specific CI including
    technical specifications and ownership.

    Args:
        ci_name: Name of the configuration item.

    Returns:
        Detailed CI information.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    # Find CI by name
    ci_data = None
    ci_id = None
    for cid, ci in CMDB_DB.items():
        if ci["name"].lower() == ci_name.lower():
            ci_data = ci
            ci_id = cid
            break

    if not ci_data:
        return f"Configuration Item '{ci_name}' not found. [{mode}]"

    if ci_data["class"] == "Server":
        return f"""**Configuration Item: {ci_data["name"]}** [{mode}]

**Identity:**
- CI ID: {ci_id}
- Class: {ci_data["class"]}
- Status: {ci_data["status"]}

**Technical Details:**
- Operating System: {ci_data.get("os", "N/A")}
- IP Address: {ci_data.get("ip", "N/A")}
- Location: {ci_data.get("location", "N/A")}

**Ownership:**
- Owner: {ci_data["owner"]}
- Support Group: {ci_data["owner"]}

**Services:** {", ".join(SERVICE_MAP.get(ci_data["name"], ["Unknown"]))}"""

    else:
        deps = ", ".join(ci_data.get("dependencies", [])) or "None"
        return f"""**Configuration Item: {ci_data["name"]}** [{mode}]

**Identity:**
- CI ID: {ci_id}
- Class: {ci_data["class"]}
- Status: {ci_data["status"]}

**Technical Details:**
- Version: {ci_data.get("version", "N/A")}

**Ownership:**
- Owner: {ci_data["owner"]}

**Dependencies:** {deps}
**Services:** {", ".join(SERVICE_MAP.get(ci_data["name"], ["Unknown"]))}"""


@tool
def get_ci_relationships(ci_name: str) -> str:
    """Get relationship information for a Configuration Item.

    Use this to understand CI dependencies and impact scope
    for change planning or incident investigation.

    Args:
        ci_name: Name of the configuration item.

    Returns:
        CI relationship map showing dependencies and dependents.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    relationships = CMDB_RELATIONSHIPS.get(ci_name)

    if not relationships:
        return f"""**CI Relationships: {ci_name}** [{mode}]

No relationship data available for this CI.
Consider updating CMDB with dependency mapping."""

    runs_on = relationships.get("runs_on", "N/A")
    depends_on = ", ".join(relationships.get("depends_on", [])) or "None"
    used_by = ", ".join(relationships.get("used_by", [])) or "None"

    return f"""**CI Relationships: {ci_name}** [{mode}]

**Infrastructure:**
- Runs On: {runs_on}

**Dependencies (Downstream):**
{ci_name} depends on: {depends_on}

**Dependents (Upstream):**
Used by: {used_by}

**Impact Analysis:**
Changes to {ci_name} may affect: {used_by}
{ci_name} may be affected by changes to: {depends_on}"""


@tool
def get_affected_services(ci_name: str) -> str:
    """Get business services affected by a Configuration Item.

    Use this for business impact analysis during incidents
    or change planning.

    Args:
        ci_name: Name of the configuration item.

    Returns:
        List of affected business services with criticality.
    """
    mode = "LIVE" if is_live_mode() else "SIMULATION"

    services = SERVICE_MAP.get(ci_name, [])

    if not services:
        # Check if CI exists at all
        ci_exists = any(ci["name"].lower() == ci_name.lower() for ci in CMDB_DB.values())
        if ci_exists:
            return f"""**Affected Services: {ci_name}** [{mode}]

No service mappings found for this CI.
This may indicate a non-business-critical infrastructure component."""
        else:
            return f"Configuration Item '{ci_name}' not found. [{mode}]"

    output = [f"**Affected Services: {ci_name}** [{mode}]\n"]
    output.append("**Business Services:**\n")

    for service in services:
        # Simulate criticality based on service type
        if "ERP" in ci_name or "Finance" in service:
            criticality = "Critical"
        elif "Portal" in service or "HR" in service:
            criticality = "High"
        else:
            criticality = "Medium"

        output.append(f"- **{service}** (Criticality: {criticality})")

    output.append(f"\n**Total Services Affected:** {len(services)}")

    return "\n".join(output)
