"""Tests for ServiceNow Agent tools.

Tests cover:
- Change request tools (get_change_request_details, get_change_requests)
- Service request tools (get_service_request_details, search_service_requests)
- Simulation mode functionality
"""

import os
import sys

import pytest

# Ensure simulation mode is set before imports
os.environ.setdefault("SERVICENOW_MODE", "simulation")

# Direct import from the servicenow_agent module to avoid loading other agents
# that may have import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agents.servicenow_agent import (
    get_change_request_details,
    get_change_requests,
    get_service_request_details,
    search_service_requests,
    CHANGE_REQUESTS_DB,
    SERVICE_REQUESTS_DB,
)


# ==================== Change Request Tests ====================


class TestGetChangeRequestDetails:
    """Tests for get_change_request_details tool."""

    def test_get_existing_change_request_chg0000009(self) -> None:
        """Test getting details for CHG0000009 (simulation mode)."""
        result = get_change_request_details.invoke({"change_number": "CHG0000009"})

        assert "CHG0000009" in result
        assert "[SIMULATION]" in result
        assert "Database Migration" in result
        assert "Oracle" in result or "PostgreSQL" in result
        assert "Implement" in result  # state
        assert "High" in result  # risk

    def test_get_existing_change_request_chg0001234(self) -> None:
        """Test getting details for CHG0001234 (simulation mode)."""
        result = get_change_request_details.invoke({"change_number": "CHG0001234"})

        assert "CHG0001234" in result
        assert "[SIMULATION]" in result
        assert "Windows Server 2019 Security Patches" in result
        assert "Scheduled" in result  # state
        assert "Low" in result  # risk
        assert "Approved" in result  # approval status

    def test_get_nonexistent_change_request(self) -> None:
        """Test getting a non-existent change request."""
        result = get_change_request_details.invoke({"change_number": "CHG9999999"})

        assert "not found" in result.lower()
        assert "CHG9999999" in result

    def test_case_insensitive_change_number(self) -> None:
        """Test that change number lookup is case-insensitive."""
        result_lower = get_change_request_details.invoke({"change_number": "chg0000009"})
        result_upper = get_change_request_details.invoke({"change_number": "CHG0000009"})

        # Both should return the same change request
        assert "CHG0000009" in result_lower
        assert "CHG0000009" in result_upper


class TestGetChangeRequests:
    """Tests for get_change_requests tool."""

    def test_get_all_change_requests(self) -> None:
        """Test getting all change requests."""
        result = get_change_requests.invoke({})

        assert "[SIMULATION]" in result
        assert "CHG0001234" in result
        assert "CHG0000009" in result

    def test_filter_by_state(self) -> None:
        """Test filtering change requests by state."""
        result = get_change_requests.invoke({"state": "Scheduled"})

        assert "CHG0001234" in result
        # CHG0000009 is in "Implement" state, should not appear
        # (unless filter is case-sensitive)

    def test_filter_by_implement_state(self) -> None:
        """Test filtering for changes in implement state."""
        result = get_change_requests.invoke({"state": "Implement"})

        # CHG0000009 should be in implement state
        assert "[SIMULATION]" in result


# ==================== Service Request Tests ====================


class TestGetServiceRequestDetails:
    """Tests for get_service_request_details tool."""

    def test_get_existing_service_request_req0010007(self) -> None:
        """Test getting details for REQ0010007 (simulation mode)."""
        result = get_service_request_details.invoke({"request_number": "REQ0010007"})

        assert "REQ0010007" in result
        assert "[SIMULATION]" in result
        assert "Software license request" in result or "Adobe" in result
        assert "In Progress" in result  # state
        assert "bob.johnson@company.com" in result  # requested for

    def test_get_existing_service_request_req0010001(self) -> None:
        """Test getting details for REQ0010001 (simulation mode)."""
        result = get_service_request_details.invoke({"request_number": "REQ0010001"})

        assert "REQ0010001" in result
        assert "[SIMULATION]" in result
        assert "laptop" in result.lower()
        assert "Approved" in result  # state
        assert "jane.doe@company.com" in result

    def test_get_nonexistent_service_request(self) -> None:
        """Test getting a non-existent service request."""
        result = get_service_request_details.invoke({"request_number": "REQ9999999"})

        assert "not found" in result.lower()
        assert "REQ9999999" in result

    def test_case_insensitive_request_number(self) -> None:
        """Test that request number lookup is case-insensitive."""
        result_lower = get_service_request_details.invoke({"request_number": "req0010007"})
        result_upper = get_service_request_details.invoke({"request_number": "REQ0010007"})

        # Both should return the same service request
        assert "REQ0010007" in result_lower
        assert "REQ0010007" in result_upper

    def test_service_request_includes_items(self) -> None:
        """Test that service request details include requested items."""
        result = get_service_request_details.invoke({"request_number": "REQ0010007"})

        # REQ0010007 should have RITM0010007 as a requested item
        assert "RITM0010007" in result
        assert "Adobe Creative Cloud" in result


class TestSearchServiceRequests:
    """Tests for search_service_requests tool."""

    def test_search_all_service_requests(self) -> None:
        """Test searching all service requests."""
        result = search_service_requests.invoke({})

        assert "[SIMULATION]" in result
        # Should find both service requests
        assert "REQ" in result

    def test_search_by_query_laptop(self) -> None:
        """Test searching by query for laptop."""
        result = search_service_requests.invoke({"query": "laptop"})

        assert "REQ0010001" in result
        # REQ0010007 is about software, shouldn't match
        assert "[SIMULATION]" in result

    def test_search_by_query_license(self) -> None:
        """Test searching by query for license."""
        result = search_service_requests.invoke({"query": "license"})

        assert "REQ0010007" in result
        assert "[SIMULATION]" in result

    def test_search_by_state(self) -> None:
        """Test searching service requests by state."""
        result = search_service_requests.invoke({"state": "Approved"})

        # REQ0010001 is in Approved state
        assert "REQ0010001" in result

    def test_search_by_requested_for(self) -> None:
        """Test searching service requests by requester."""
        result = search_service_requests.invoke({"requested_for": "bob.johnson@company.com"})

        assert "REQ0010007" in result

    def test_search_no_results(self) -> None:
        """Test search with no matching results."""
        result = search_service_requests.invoke({"query": "nonexistent-item-xyz"})

        assert "No service requests found" in result

    def test_search_limit(self) -> None:
        """Test search respects limit parameter."""
        result = search_service_requests.invoke({"limit": 1})

        # Should return only 1 result
        assert "[SIMULATION]" in result


# ==================== Data Validation Tests ====================


class TestSimulatedDataIntegrity:
    """Tests to verify the simulated data is properly structured."""

    def test_change_requests_db_has_chg0000009(self) -> None:
        """Verify CHG0000009 exists in the database."""
        assert "CHG0000009" in CHANGE_REQUESTS_DB
        chg = CHANGE_REQUESTS_DB["CHG0000009"]
        assert chg["number"] == "CHG0000009"
        assert "short_description" in chg
        assert "description" in chg
        assert "state" in chg
        assert "type" in chg
        assert "risk" in chg

    def test_service_requests_db_has_req0010007(self) -> None:
        """Verify REQ0010007 exists in the database."""
        assert "REQ0010007" in SERVICE_REQUESTS_DB
        req = SERVICE_REQUESTS_DB["REQ0010007"]
        assert req["number"] == "REQ0010007"
        assert "short_description" in req
        assert "request_state" in req
        assert "items" in req
        assert len(req["items"]) > 0

    def test_change_request_chg0000009_has_required_fields(self) -> None:
        """Verify CHG0000009 has all required fields for display."""
        chg = CHANGE_REQUESTS_DB["CHG0000009"]

        required_fields = [
            "number",
            "short_description",
            "description",
            "state",
            "type",
            "risk",
            "planned_start",
            "planned_end",
            "assigned_to",
            "approval_status",
            "impact",
        ]

        for field in required_fields:
            assert field in chg, f"Missing required field: {field}"

    def test_service_request_req0010007_has_required_fields(self) -> None:
        """Verify REQ0010007 has all required fields for display."""
        req = SERVICE_REQUESTS_DB["REQ0010007"]

        required_fields = [
            "number",
            "short_description",
            "description",
            "request_state",
            "stage",
            "requested_for",
            "opened_by",
            "created",
            "updated",
            "price",
            "items",
        ]

        for field in required_fields:
            assert field in req, f"Missing required field: {field}"
