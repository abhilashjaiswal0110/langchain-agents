"""User Context for Azure AD authenticated users.

Provides structured user information extracted from Azure AD tokens:
- User identity (ID, email, name)
- Role assignments based on Azure AD groups
- Department and organization info
- Token metadata for audit trails

Following Enterprise Development Standards:
- Security Architect: Minimal data exposure, secure defaults
- Data Architect: Structured user representation
- Software Engineer: Immutable dataclasses, type safety
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class UserRole(str, Enum):
    """User roles mapped from Azure AD groups.

    These roles integrate with the existing RBAC system in governance/rbac.py
    """

    ADMIN = "admin"  # Full access, can approve any action
    OPERATOR = "operator"  # Can use all agents, approve L1 actions
    USER = "user"  # Can use agents, cannot approve
    VIEWER = "viewer"  # Read-only access
    ANONYMOUS = "anonymous"  # Unauthenticated (for public endpoints)


# Mapping from Azure AD group names to roles
# Configure via environment or settings
DEFAULT_GROUP_ROLE_MAPPING: dict[str, UserRole] = {
    "LangChain-Admins": UserRole.ADMIN,
    "LangChain-Operators": UserRole.OPERATOR,
    "LangChain-Users": UserRole.USER,
    "LangChain-Viewers": UserRole.VIEWER,
}


@dataclass(frozen=True)
class UserContext:
    """Immutable user context extracted from Azure AD token.

    This context is injected into all agent invocations for:
    - Audit logging (who did what)
    - RBAC enforcement (permission checks)
    - Personalization (user preferences)
    - Cost tracking (per-user usage)

    Attributes:
        user_id: Azure AD object ID (unique identifier)
        email: User's email address
        display_name: User's display name
        roles: List of assigned roles (mapped from groups)
        groups: Raw Azure AD group names
        department: User's department (if available)
        job_title: User's job title (if available)
        tenant_id: Azure AD tenant ID
        token_exp: Token expiration timestamp
        issued_at: When the context was created
        raw_claims: Original token claims (for debugging)

    Example:
        >>> user = UserContext(
        ...     user_id="abc-123",
        ...     email="john.doe@company.com",
        ...     display_name="John Doe",
        ...     roles=[UserRole.USER],
        ...     groups=["LangChain-Users"],
        ... )
        >>> user.has_role(UserRole.ADMIN)
        False
        >>> user.is_authenticated
        True
    """

    user_id: str
    email: str
    display_name: str
    roles: list[UserRole] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    department: str | None = None
    job_title: str | None = None
    tenant_id: str | None = None
    token_exp: datetime | None = None
    issued_at: datetime = field(default_factory=datetime.utcnow)
    raw_claims: dict[str, Any] = field(default_factory=dict)

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated (not anonymous)."""
        return self.user_id != "anonymous" and UserRole.ANONYMOUS not in self.roles

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return UserRole.ADMIN in self.roles

    @property
    def is_operator(self) -> bool:
        """Check if user has operator or higher role."""
        return UserRole.ADMIN in self.roles or UserRole.OPERATOR in self.roles

    @property
    def primary_role(self) -> UserRole:
        """Get the highest priority role.

        Role priority: ADMIN > OPERATOR > USER > VIEWER > ANONYMOUS
        """
        priority = [
            UserRole.ADMIN,
            UserRole.OPERATOR,
            UserRole.USER,
            UserRole.VIEWER,
            UserRole.ANONYMOUS,
        ]
        for role in priority:
            if role in self.roles:
                return role
        return UserRole.ANONYMOUS

    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role.

        Args:
            role: Role to check for.

        Returns:
            True if user has the role.
        """
        return role in self.roles

    def has_any_role(self, roles: list[UserRole]) -> bool:
        """Check if user has any of the specified roles.

        Args:
            roles: List of roles to check.

        Returns:
            True if user has at least one of the roles.
        """
        return any(role in self.roles for role in roles)

    def has_all_roles(self, roles: list[UserRole]) -> bool:
        """Check if user has all of the specified roles.

        Args:
            roles: List of roles to check.

        Returns:
            True if user has all the roles.
        """
        return all(role in self.roles for role in roles)

    def in_group(self, group_name: str) -> bool:
        """Check if user is in a specific Azure AD group.

        Args:
            group_name: Name of the Azure AD group.

        Returns:
            True if user is in the group.
        """
        return group_name in self.groups

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/logging.

        Note: Excludes raw_claims and token_exp for security.

        Returns:
            Dictionary representation of user context.
        """
        return {
            "user_id": self.user_id,
            "email": self.email,
            "display_name": self.display_name,
            "roles": [r.value for r in self.roles],
            "groups": self.groups,
            "department": self.department,
            "job_title": self.job_title,
            "tenant_id": self.tenant_id,
            "is_authenticated": self.is_authenticated,
            "primary_role": self.primary_role.value,
        }

    def to_audit_dict(self) -> dict[str, Any]:
        """Convert to minimal dictionary for audit logging.

        Returns:
            Minimal dictionary with just identity info.
        """
        return {
            "user_id": self.user_id,
            "email": self.email,
            "primary_role": self.primary_role.value,
        }

    @classmethod
    def anonymous(cls) -> "UserContext":
        """Create an anonymous user context.

        Used for unauthenticated requests to public endpoints.

        Returns:
            Anonymous UserContext instance.
        """
        return cls(
            user_id="anonymous",
            email="anonymous@system",
            display_name="Anonymous User",
            roles=[UserRole.ANONYMOUS],
            groups=[],
        )

    @classmethod
    def from_claims(
        cls,
        claims: dict[str, Any],
        group_role_mapping: dict[str, UserRole] | None = None,
    ) -> "UserContext":
        """Create UserContext from JWT claims.

        Args:
            claims: Decoded JWT claims from Azure AD token.
            group_role_mapping: Optional custom group-to-role mapping.

        Returns:
            UserContext populated from claims.

        Example:
            >>> claims = {
            ...     "oid": "user-123",
            ...     "email": "user@company.com",
            ...     "name": "User Name",
            ...     "groups": ["LangChain-Users"],
            ... }
            >>> user = UserContext.from_claims(claims)
        """
        mapping = group_role_mapping or DEFAULT_GROUP_ROLE_MAPPING

        # Extract groups from claims
        groups = claims.get("groups", [])
        if isinstance(groups, str):
            groups = [groups]

        # Map groups to roles
        roles: list[UserRole] = []
        for group in groups:
            if group in mapping:
                role = mapping[group]
                if role not in roles:
                    roles.append(role)

        # Default to USER role if authenticated but no specific role
        if not roles:
            roles = [UserRole.USER]

        # Parse expiration if present
        token_exp = None
        if "exp" in claims:
            try:
                token_exp = datetime.fromtimestamp(claims["exp"])
            except (ValueError, TypeError):
                pass

        return cls(
            user_id=claims.get("oid", claims.get("sub", "unknown")),
            email=claims.get("email", claims.get("preferred_username", "")),
            display_name=claims.get("name", claims.get("given_name", "Unknown")),
            roles=roles,
            groups=groups,
            department=claims.get("department"),
            job_title=claims.get("jobTitle"),
            tenant_id=claims.get("tid"),
            token_exp=token_exp,
            raw_claims=claims,
        )


def map_groups_to_roles(
    groups: list[str],
    mapping: dict[str, UserRole] | None = None,
) -> list[UserRole]:
    """Map Azure AD groups to application roles.

    Args:
        groups: List of Azure AD group names.
        mapping: Custom group-to-role mapping (uses default if None).

    Returns:
        List of UserRole values.
    """
    mapping = mapping or DEFAULT_GROUP_ROLE_MAPPING
    roles: list[UserRole] = []

    for group in groups:
        if group in mapping:
            role = mapping[group]
            if role not in roles:
                roles.append(role)

    return roles if roles else [UserRole.USER]
