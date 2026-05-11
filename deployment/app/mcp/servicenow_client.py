"""ServiceNow REST API client.

Provides async interface to ServiceNow Table API for:
- Incident management (create, read, update, search)
- CMDB queries
- Change request management
- Knowledge base access

Supports both live and simulation modes for development/testing.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class ServiceNowConfig:
    """Configuration for ServiceNow client.

    Attributes:
        instance: ServiceNow instance name (e.g., "dev12345")
        username: ServiceNow username
        password: ServiceNow password
        mode: Operation mode ("live" or "simulation")
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    """

    instance: str = ""
    username: str = ""
    password: str = ""
    mode: Literal["live", "simulation"] = "simulation"
    timeout: int = 30
    verify_ssl: bool = True

    @classmethod
    def from_env(cls) -> "ServiceNowConfig":
        """Create config from environment variables."""
        return cls(
            instance=os.getenv("SERVICENOW_INSTANCE", ""),
            username=os.getenv("SERVICENOW_USERNAME", ""),
            password=os.getenv("SERVICENOW_PASSWORD", ""),
            mode=os.getenv("SERVICENOW_MODE", "simulation"),  # type: ignore[arg-type]
            timeout=int(os.getenv("SERVICENOW_TIMEOUT", "30")),
            verify_ssl=os.getenv("SERVICENOW_VERIFY_SSL", "true").lower() == "true",
        )

    @property
    def base_url(self) -> str:
        """Get base URL for ServiceNow API."""
        return f"https://{self.instance}.service-now.com"

    @property
    def is_configured(self) -> bool:
        """Check if ServiceNow is properly configured."""
        return bool(self.instance and self.username and self.password)


class ServiceNowSimulator:
    """Simulated ServiceNow for development and testing."""

    def __init__(self) -> None:
        """Initialize simulator with sample data."""
        self._incidents: dict[str, dict[str, Any]] = {}
        self._changes: dict[str, dict[str, Any]] = {}
        self._cmdb: dict[str, list[dict[str, Any]]] = {
            "cmdb_ci_server": [
                {"sys_id": "srv001", "name": "web-server-01", "ip_address": "10.0.0.1", "status": "operational"},
                {"sys_id": "srv002", "name": "db-server-01", "ip_address": "10.0.0.2", "status": "operational"},
                {"sys_id": "srv003", "name": "app-server-01", "ip_address": "10.0.0.3", "status": "maintenance"},
            ],
            "cmdb_ci_computer": [
                {"sys_id": "pc001", "name": "DESKTOP-001", "assigned_to": "John Smith", "status": "in_use"},
                {"sys_id": "pc002", "name": "LAPTOP-001", "assigned_to": "Jane Doe", "status": "in_use"},
            ],
        }
        self._incident_counter = 10000

    def _generate_incident_number(self) -> str:
        """Generate a new incident number."""
        self._incident_counter += 1
        return f"INC{self._incident_counter:07d}"

    async def create_incident(
        self,
        short_description: str,
        description: str,
        priority: str = "3",
        category: str = "inquiry",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a simulated incident."""
        incident_number = self._generate_incident_number()
        sys_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        incident = {
            "sys_id": sys_id,
            "number": incident_number,
            "short_description": short_description,
            "description": description,
            "priority": priority,
            "category": category,
            "state": "1",  # New
            "impact": "3",
            "urgency": "3",
            "opened_at": now,
            "opened_by": "system",
            "sys_created_on": now,
            "sys_updated_on": now,
            **kwargs,
        }

        self._incidents[incident_number] = incident
        return {"result": incident}

    async def get_incident(self, incident_number: str) -> dict[str, Any]:
        """Get a simulated incident."""
        incident = self._incidents.get(incident_number)
        if not incident:
            return {"result": []}
        return {"result": [incident]}

    async def search_incidents(
        self,
        query: str,
        status: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search simulated incidents."""
        results = []
        for incident in self._incidents.values():
            # Simple text search
            if query.lower() in incident.get("short_description", "").lower():
                if status is None or incident.get("state") == status:
                    results.append(incident)
                    if len(results) >= limit:
                        break
        return {"result": results}

    async def update_incident(
        self,
        incident_number: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a simulated incident."""
        if incident_number not in self._incidents:
            return {"error": {"message": "Incident not found"}}

        self._incidents[incident_number].update(updates)
        self._incidents[incident_number]["sys_updated_on"] = datetime.now(timezone.utc).isoformat()
        return {"result": self._incidents[incident_number]}

    async def query_cmdb(
        self,
        ci_type: str,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Query simulated CMDB."""
        items = self._cmdb.get(ci_type, [])
        results = []

        for item in items:
            if query.lower() in item.get("name", "").lower():
                results.append(item)
                if len(results) >= limit:
                    break

        return {"result": results}


class ServiceNowClient:
    """Async client for ServiceNow REST API.

    Supports both live API calls and simulation mode for development.
    """

    def __init__(self, config: ServiceNowConfig | None = None) -> None:
        """Initialize ServiceNow client.

        Args:
            config: ServiceNow configuration.
        """
        self.config = config or ServiceNowConfig.from_env()
        self._simulator = ServiceNowSimulator()
        self._http_client: Any = None

    async def _get_http_client(self) -> Any:
        """Get or create HTTP client."""
        if self._http_client is None:
            if not HTTPX_AVAILABLE:
                msg = "httpx package not installed. Install with: pip install httpx"
                raise RuntimeError(msg)
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
        return self._http_client

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request to ServiceNow API.

        Args:
            method: HTTP method.
            endpoint: API endpoint (e.g., "/api/now/table/incident").
            json: Request body for POST/PUT.
            params: Query parameters.

        Returns:
            Response JSON.
        """
        client = await self._get_http_client()
        url = f"{self.config.base_url}{endpoint}"

        response = await client.request(
            method=method,
            url=url,
            json=json,
            params=params,
            auth=(self.config.username, self.config.password),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

        response.raise_for_status()
        return response.json()

    async def create_incident(
        self,
        short_description: str,
        description: str,
        priority: str = "3",
        category: str = "inquiry",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a ServiceNow incident.

        Args:
            short_description: Brief summary of the issue.
            description: Detailed description.
            priority: Priority level (1-4).
            category: Incident category.
            **kwargs: Additional incident fields.

        Returns:
            Created incident details.
        """
        if self.config.mode == "simulation":
            return await self._simulator.create_incident(short_description, description, priority, category, **kwargs)

        if not self.config.is_configured:
            return {"error": "ServiceNow not configured", "result": None}

        data = {
            "short_description": short_description,
            "description": description,
            "priority": priority,
            "category": category,
            **kwargs,
        }

        return await self._request("POST", "/api/now/table/incident", json=data)

    async def get_incident(self, incident_number: str) -> dict[str, Any]:
        """Get an incident by number.

        Args:
            incident_number: Incident number (e.g., "INC0010001").

        Returns:
            Incident details.
        """
        if self.config.mode == "simulation":
            result = await self._simulator.get_incident(incident_number)
            return result.get("result", [{}])[0] if result.get("result") else {}

        if not self.config.is_configured:
            return {"error": "ServiceNow not configured"}

        params = {
            "sysparm_query": f"number={incident_number}",
            "sysparm_limit": 1,
        }

        result = await self._request("GET", "/api/now/table/incident", params=params)
        return result.get("result", [{}])[0] if result.get("result") else {}

    async def search_incidents(
        self,
        query: str,
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for incidents.

        Args:
            query: Search query.
            status: Filter by status.
            limit: Maximum results.

        Returns:
            List of matching incidents.
        """
        if self.config.mode == "simulation":
            result = await self._simulator.search_incidents(query, status, limit)
            return result.get("result", [])

        if not self.config.is_configured:
            return [{"error": "ServiceNow not configured"}]

        # Build query
        query_parts = [f"short_descriptionLIKE{query}^ORdescriptionLIKE{query}"]
        if status:
            query_parts.append(f"state={status}")

        params = {
            "sysparm_query": "^".join(query_parts),
            "sysparm_limit": limit,
        }

        result = await self._request("GET", "/api/now/table/incident", params=params)
        return result.get("result", [])

    async def update_incident(
        self,
        incident_number: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an incident.

        Args:
            incident_number: Incident number to update.
            updates: Fields to update.

        Returns:
            Updated incident details.
        """
        if self.config.mode == "simulation":
            result = await self._simulator.update_incident(incident_number, updates)
            return result.get("result", {})

        if not self.config.is_configured:
            return {"error": "ServiceNow not configured"}

        # First get the sys_id
        incident = await self.get_incident(incident_number)
        if not incident or "sys_id" not in incident:
            return {"error": f"Incident {incident_number} not found"}

        sys_id = incident["sys_id"]
        result = await self._request(
            "PATCH",
            f"/api/now/table/incident/{sys_id}",
            json=updates,
        )
        return result.get("result", {})

    async def query_cmdb(
        self,
        ci_type: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the CMDB for configuration items.

        Args:
            ci_type: CI type (e.g., "cmdb_ci_server").
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching CIs.
        """
        if self.config.mode == "simulation":
            result = await self._simulator.query_cmdb(ci_type, query, limit)
            return result.get("result", [])

        if not self.config.is_configured:
            return [{"error": "ServiceNow not configured"}]

        params = {
            "sysparm_query": f"nameLIKE{query}",
            "sysparm_limit": limit,
        }

        result = await self._request("GET", f"/api/now/table/{ci_type}", params=params)
        return result.get("result", [])

    async def create_change_request(
        self,
        short_description: str,
        description: str,
        type: str = "normal",
        risk: str = "moderate",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a change request.

        Args:
            short_description: Brief summary.
            description: Detailed description.
            type: Change type (normal, standard, emergency).
            risk: Risk level.
            **kwargs: Additional fields.

        Returns:
            Created change request.
        """
        if self.config.mode == "simulation":
            # Simplified simulation
            return {
                "result": {
                    "number": f"CHG{int(datetime.now().timestamp()) % 1000000:07d}",
                    "short_description": short_description,
                    "description": description,
                    "type": type,
                    "risk": risk,
                    "state": "new",
                    **kwargs,
                }
            }

        if not self.config.is_configured:
            return {"error": "ServiceNow not configured", "result": None}

        data = {
            "short_description": short_description,
            "description": description,
            "type": type,
            "risk": risk,
            **kwargs,
        }

        return await self._request("POST", "/api/now/table/change_request", json=data)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Global client instance
_servicenow_client: ServiceNowClient | None = None


def get_servicenow_client() -> ServiceNowClient:
    """Get or create the global ServiceNow client.

    Returns:
        ServiceNowClient instance.
    """
    global _servicenow_client
    if _servicenow_client is None:
        _servicenow_client = ServiceNowClient()
    return _servicenow_client


def reset_servicenow_client() -> None:
    """Reset the global ServiceNow client."""
    global _servicenow_client
    _servicenow_client = None


# Convenience functions


async def create_incident(
    short_description: str,
    description: str,
    priority: str = "3",
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a ServiceNow incident.

    Args:
        short_description: Brief summary.
        description: Detailed description.
        priority: Priority level.
        **kwargs: Additional fields.

    Returns:
        Created incident.
    """
    client = get_servicenow_client()
    return await client.create_incident(short_description, description, priority, **kwargs)


async def get_incident(incident_number: str) -> dict[str, Any]:
    """Get an incident by number.

    Args:
        incident_number: Incident number.

    Returns:
        Incident details.
    """
    client = get_servicenow_client()
    return await client.get_incident(incident_number)


async def search_incidents(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Search for incidents.

    Args:
        query: Search query.
        **kwargs: Additional filters.

    Returns:
        List of incidents.
    """
    client = get_servicenow_client()
    return await client.search_incidents(query, **kwargs)
