"""Role-Based Access Control (RBAC) for enterprise IT agents.

Provides:
- Role definitions and permission mapping
- User role management
- Permission checking for agent operations
- API key to role association
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    OPERATOR = "operator"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"  # For automated/API integrations


class Permission(str, Enum):
    """Permissions for agent operations."""

    # Agent permissions
    AGENT_INVOKE = "agent:invoke"
    AGENT_READ = "agent:read"
    AGENT_LIST = "agent:list"

    # Approval permissions
    APPROVE_L1 = "agent:approve:l1"
    APPROVE_L2 = "agent:approve:l2"
    APPROVE_L3 = "agent:approve:l3"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # Admin permissions
    USER_MANAGE = "user:manage"
    CONFIG_MANAGE = "config:manage"
    SYSTEM_ADMIN = "system:admin"

    # Wildcard
    ALL = "*"


@dataclass
class UserContext:
    """Context for an authenticated user.

    Attributes:
        user_id: Unique user identifier
        role: User's role
        permissions: Explicit permissions (in addition to role)
        api_key_id: API key identifier used for auth
        metadata: Additional user metadata
        tenant_id: Tenant identifier for multi-tenancy isolation
    """

    user_id: str
    role: Role = Role.USER
    permissions: set[Permission] = field(default_factory=set)
    api_key_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission.

        Args:
            permission: Permission to check.

        Returns:
            True if user has the permission.
        """
        # Check explicit permissions
        if Permission.ALL in self.permissions:
            return True
        if permission in self.permissions:
            return True

        # Check role-based permissions
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        if Permission.ALL in role_perms:
            return True
        return permission in role_perms


# Role to permission mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {Permission.ALL},
    Role.OPERATOR: {
        Permission.AGENT_INVOKE,
        Permission.AGENT_READ,
        Permission.AGENT_LIST,
        Permission.APPROVE_L1,
        Permission.APPROVE_L2,
        Permission.AUDIT_READ,
    },
    Role.USER: {
        Permission.AGENT_INVOKE,
        Permission.AGENT_READ,
        Permission.AGENT_LIST,
    },
    Role.VIEWER: {
        Permission.AGENT_READ,
        Permission.AGENT_LIST,
        Permission.AUDIT_READ,
    },
    Role.SERVICE: {
        Permission.AGENT_INVOKE,
        Permission.AGENT_READ,
        Permission.AGENT_LIST,
    },
}


@dataclass
class RBACConfig:
    """Configuration for RBAC.

    Attributes:
        enabled: Whether RBAC is enabled
        default_role: Default role for unauthenticated requests
        api_key_roles: Mapping of API key prefixes to roles
        strict_mode: Whether to deny all unauthenticated requests
    """

    enabled: bool = True
    default_role: Role = Role.VIEWER
    api_key_roles: dict[str, Role] = field(default_factory=dict)
    strict_mode: bool = False

    @classmethod
    def from_env(cls) -> "RBACConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("RBAC_ENABLED", "true").lower() == "true",
            default_role=Role(os.getenv("RBAC_DEFAULT_ROLE", "viewer")),
            strict_mode=os.getenv("RBAC_STRICT_MODE", "false").lower() == "true",
        )


class RBACManager:
    """Manages role-based access control.

    Provides:
    - User context creation from API keys
    - Permission checking
    - Role management
    """

    def __init__(self, config: RBACConfig | None = None) -> None:
        """Initialize RBAC manager.

        Args:
            config: RBAC configuration.
        """
        self.config = config or RBACConfig.from_env()
        self._api_key_cache: dict[str, UserContext] = {}
        self._user_roles: dict[str, Role] = {}

    def get_user_context(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> UserContext:
        """Get user context from API key or user ID.

        Args:
            api_key: API key for authentication.
            user_id: Optional user ID override.

        Returns:
            UserContext for the request.
        """
        if not self.config.enabled:
            # RBAC disabled - return admin context
            return UserContext(
                user_id=user_id or "system",
                role=Role.ADMIN,
            )

        # Check API key cache
        if api_key and api_key in self._api_key_cache:
            return self._api_key_cache[api_key]

        # Determine role from API key
        role = self._get_role_from_api_key(api_key)

        # Create user context
        context = UserContext(
            user_id=user_id or self._extract_user_id(api_key),
            role=role,
            api_key_id=api_key[:8] + "..." if api_key else None,
        )

        # Cache for subsequent requests
        if api_key:
            self._api_key_cache[api_key] = context

        return context

    def _get_role_from_api_key(self, api_key: str | None) -> Role:
        """Determine role based on API key.

        Args:
            api_key: API key to check.

        Returns:
            Role for the API key.
        """
        if not api_key:
            if self.config.strict_mode:
                return Role.VIEWER  # Minimal permissions
            return self.config.default_role

        # Check configured API key roles
        for prefix, role in self.config.api_key_roles.items():
            if api_key.startswith(prefix):
                return role

        # Check environment-based admin keys
        admin_keys = os.getenv("RBAC_ADMIN_API_KEYS", "").split(",")
        if api_key in admin_keys:
            return Role.ADMIN

        operator_keys = os.getenv("RBAC_OPERATOR_API_KEYS", "").split(",")
        if api_key in operator_keys:
            return Role.OPERATOR

        # Default based on key pattern
        if api_key.startswith("sk-admin-"):
            return Role.ADMIN
        if api_key.startswith("sk-operator-"):
            return Role.OPERATOR
        if api_key.startswith("sk-service-"):
            return Role.SERVICE

        return self.config.default_role

    def _extract_user_id(self, api_key: str | None) -> str:
        """Extract user ID from API key.

        Args:
            api_key: API key to extract from.

        Returns:
            User ID string.
        """
        if not api_key:
            return "anonymous"

        # API key format: sk-{type}-{user_id}-{random}
        parts = api_key.split("-")
        if len(parts) >= 3:
            return parts[2]

        return f"user-{api_key[:8]}"

    def check_permission(
        self,
        user_context: UserContext,
        permission: Permission,
    ) -> bool:
        """Check if user has permission.

        Args:
            user_context: User context to check.
            permission: Permission required.

        Returns:
            True if user has permission.
        """
        if not self.config.enabled:
            return True

        return user_context.has_permission(permission)

    def require_permission(
        self,
        user_context: UserContext,
        permission: Permission,
    ) -> None:
        """Require a permission, raising exception if not met.

        Args:
            user_context: User context to check.
            permission: Permission required.

        Raises:
            PermissionDeniedError: If permission is not granted.
        """
        if not self.check_permission(user_context, permission):
            raise PermissionDeniedError(
                f"Permission denied: {permission.value} required",
                user_id=user_context.user_id,
                permission=permission,
            )

    def set_user_role(self, user_id: str, role: Role) -> None:
        """Set role for a user.

        Args:
            user_id: User identifier.
            role: Role to assign.
        """
        self._user_roles[user_id] = role

    def get_user_role(self, user_id: str) -> Role:
        """Get role for a user.

        Args:
            user_id: User identifier.

        Returns:
            User's role.
        """
        return self._user_roles.get(user_id, self.config.default_role)

    def add_api_key_role(self, api_key_prefix: str, role: Role) -> None:
        """Add API key prefix to role mapping.

        Args:
            api_key_prefix: API key prefix.
            role: Role to assign.
        """
        self.config.api_key_roles[api_key_prefix] = role

    def clear_cache(self) -> None:
        """Clear the API key cache."""
        self._api_key_cache.clear()


class PermissionDeniedError(Exception):
    """Raised when permission check fails."""

    def __init__(
        self,
        message: str,
        user_id: str | None = None,
        permission: Permission | None = None,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            user_id: User who was denied.
            permission: Permission that was denied.
        """
        super().__init__(message)
        self.user_id = user_id
        self.permission = permission


# Global RBAC manager instance
_rbac_manager: RBACManager | None = None


def get_rbac_manager() -> RBACManager:
    """Get or create the global RBAC manager."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def reset_rbac_manager() -> None:
    """Reset the global RBAC manager."""
    global _rbac_manager
    _rbac_manager = None


# Convenience functions


def check_permission(
    api_key: str | None,
    permission: Permission,
    user_id: str | None = None,
) -> bool:
    """Check if API key has permission.

    Args:
        api_key: API key for authentication.
        permission: Permission to check.
        user_id: Optional user ID.

    Returns:
        True if permission is granted.
    """
    manager = get_rbac_manager()
    context = manager.get_user_context(api_key, user_id)
    return manager.check_permission(context, permission)


def require_permission(
    api_key: str | None,
    permission: Permission,
    user_id: str | None = None,
) -> UserContext:
    """Require permission, returning user context if granted.

    Args:
        api_key: API key for authentication.
        permission: Permission required.
        user_id: Optional user ID.

    Returns:
        UserContext if permission is granted.

    Raises:
        PermissionDeniedError: If permission is denied.
    """
    manager = get_rbac_manager()
    context = manager.get_user_context(api_key, user_id)
    manager.require_permission(context, permission)
    return context


def get_permissions_for_role(role: Role) -> set[Permission]:
    """Get all permissions for a role.

    Args:
        role: Role to get permissions for.

    Returns:
        Set of permissions.
    """
    return ROLE_PERMISSIONS.get(role, set())
