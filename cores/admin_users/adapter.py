"""
Admin Users Adapter - UBP Framework Bridge Layer

This module provides the UBP framework integration for user management.
Acts as a bridge between UBP's module system and technical provider implementations.

Separation of Concerns:
- adapter.py: UBP framework bridge (this file)
- providers.py: Pure technical logic (ZERO UBP dependencies)
- __init__.py: Factory entry point
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import uuid
import json
from datetime import datetime

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    from _shared.operation_context import OperationContext

from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event
from .providers import PasswordHasher, UserManagementProvider, InvitationProvider

logger = logging.getLogger(__name__)


class AdminUsersAdapter(BaseHybridModule):
    """
    UBP Adapter for user management module.

    This adapter:
    - Integrates with UBP lifecycle (initialize, shutdown, health_check)
    - Delegates all business logic to UserManagementProvider
    - Publishes events to event bus
    - Provides request tracking
    - Handles statistics and monitoring

    NO business logic here - all in providers.py
    """

    def __init__(self, module_path: Path, **kwargs):
        """Initialize the adapter."""
        super().__init__(module_path, **kwargs)

        # Store kwargs for fallback Redis access
        self.kwargs = kwargs

        # Technical providers (initialized in initialize())
        self.password_hasher: Optional[PasswordHasher] = None
        self.user_provider: Optional[UserManagementProvider] = None
        self.invitation_provider: Optional[InvitationProvider] = None
        self._initialized = False

    # ========================================================================
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # ========================================================================

    def _build_context_from_di(self) -> OperationContext:
        """
        Build OperationContext from DI container — backward compatibility for REST path.
        
        MCP-COMPAT: When ctx is not provided (REST path), this method constructs
        an OperationContext from the DI container state.
        
        Returns:
            OperationContext with default values
        """
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """
        Normalize any context format to OperationContext.
        
        MCP-COMPAT: Handles both legacy security context (ctx.user.user_id) 
        and new OperationContext format for backward compatibility.
        
        Args:
            ctx: Either OperationContext, legacy security context, or None
            
        Returns:
            OperationContext instance
        """
        if ctx is None:
            return self._build_context_from_di()
        
        # Already an OperationContext
        if isinstance(ctx, OperationContext):
            return ctx
        
        # Legacy security context format (ctx.user.user_id, ctx.user.roles)
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return OperationContext(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        
        # Fallback
        return self._build_context_from_di()

    def _require_initialized(self) -> UserManagementProvider:
        """
        Ensure module is initialized and return the user provider.

        Returns:
            UserManagementProvider instance

        Raises:
            RuntimeError: If module not initialized
        """
        if not self._initialized or not self.user_provider:
            raise RuntimeError("admin_users module not initialized")
        return self.user_provider

    # ========================================================================
    # Security Helpers (SECURITY PATCH P0)
    # ========================================================================

    def _require_ctx(self, ctx: Any) -> Any:
        """
        Validate and return security context.

        Args:
            ctx: Security context object

        Returns:
            Validated ctx

        Raises:
            ValueError: If ctx is None or missing user info
        """
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            raise ValueError("Security context required for this operation")
        if not hasattr(ctx.user, "user_id"):
            raise ValueError("Security context must contain user_id")
        return ctx

    def _is_admin(self, ctx: Any) -> bool:
        """
        Check if the current user is a platform administrator.

        Args:
            ctx: Security context with user info

        Returns:
            True if user has admin role, False otherwise
        """
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            return False
        if not hasattr(ctx.user, "roles"):
            return False

        roles = ctx.user.roles
        # Ensure roles is iterable
        if not isinstance(roles, (list, set, tuple)):
            return False

        return "admin" in roles

    def _is_client_admin(self, ctx: Any) -> bool:
        """
        Check if the current user is a client-level administrator.

        client_admin can manage users within their own client only.

        Args:
            ctx: Security context with user info

        Returns:
            True if user has client_admin role, False otherwise
        """
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            return False
        if not hasattr(ctx.user, "roles"):
            return False

        roles = ctx.user.roles
        if not isinstance(roles, (list, set, tuple)):
            return False

        return "client_admin" in roles

    def _require_admin(self, ctx: Any, operation: str) -> Any:
        """
        Require platform admin privileges for an operation.

        SECURITY PATCH P0: Centralized admin check for all sensitive operations.

        Args:
            ctx: Security context
            operation: Name of the operation for error logging

        Returns:
            The validated ctx

        Raises:
            PermissionError: If user is not admin
            ValueError: If ctx is invalid
        """
        ctx = self._require_ctx(ctx)
        if not self._is_admin(ctx):
            logger.warning(
                f"[SECURITY] Unauthorized {operation} attempt by user {ctx.user.user_id}",
                extra={"user_id": ctx.user.user_id, "operation": operation},
            )
            raise PermissionError(f"Only administrators can perform: {operation}")
        return ctx

    def _require_admin_or_client_admin(
        self,
        ctx: Any,
        operation: str,
        target_client_id: Optional[str] = None,
    ) -> Any:
        """
        Require platform admin OR client_admin privilege.

        client_admin is restricted to managing users within their own client.

        Args:
            ctx: Security context
            operation: Name of the operation for error logging
            target_client_id: Client scope for client_admin restriction

        Returns:
            The validated ctx

        Raises:
            PermissionError: If user lacks sufficient privilege
        """
        ctx = self._require_ctx(ctx)

        if self._is_admin(ctx):
            return ctx  # Platform admin: unrestricted

        if self._is_client_admin(ctx):
            # client_admin can only manage their own client's users
            if target_client_id is not None:
                caller_client_id = getattr(ctx.user, "client_id", None)
                if caller_client_id != target_client_id:
                    logger.warning(
                        f"[SECURITY] client_admin {ctx.user.user_id} attempted cross-client "
                        f"{operation} (own={caller_client_id}, target={target_client_id})",
                        extra={
                            "user_id": ctx.user.user_id,
                            "operation": operation,
                            "target_client_id": target_client_id,
                        },
                    )
                    raise PermissionError(
                        "client_admin can only manage users in their own client"
                    )
            return ctx

        logger.warning(
            f"[SECURITY] Unauthorized {operation} attempt by user {ctx.user.user_id}",
            extra={"user_id": ctx.user.user_id, "operation": operation},
        )
        raise PermissionError(
            f"Admin or client_admin role required to perform: {operation}"
        )

    async def _check_last_client_admin(
        self,
        user_id: str,
        client_id: str,
        provider: Any,
        *,
        active_only: bool = True,
    ) -> None:
        """
        Raise PermissionError if `user_id` is the last (active) client_admin for `client_id`.

        Shared guard used by both delete_user() and update_user() to prevent
        leaving a client without any client_admin (orphan tenant).

        Special case (orphan admin): if the referenced `client_id` no longer
        exists in the registry (client deleted, but auto-admin survived),
        the guard is skipped — the admin IS orphan, deletion is the right
        action and platform admin must be allowed to clean it up.

        Args:
            user_id: The user about to be removed/demoted/deactivated.
            client_id: The client that must keep at least one active client_admin.
            provider: Initialized user provider (for list_users query).
            active_only: If True (default), only count active client_admins.

        Raises:
            PermissionError: If this user is the last (active) client_admin AND
                the client still exists.
        """
        # Orphan-skip: if the parent client no longer exists, the guard is
        # meaningless — allow cleanup of leftover client_admin.
        if self.di_container:
            try:
                admin_clients = await self.di_container.resolve("admin_clients")
                if admin_clients is not None:
                    try:
                        await admin_clients.get_client_internal(client_id)
                    except ValueError:
                        logger.info(
                            "[ADMIN_USERS] Skipping last-client_admin guard for user "
                            f"{user_id}: parent client {client_id} no longer exists "
                            "(orphan admin cleanup)."
                        )
                        return
            except Exception as e:
                # Don't block on resolution errors — fall through to normal guard.
                logger.debug(
                    f"[ADMIN_USERS] Could not verify client {client_id} existence "
                    f"for orphan check: {e}"
                )

        all_users = await provider.list_users(
            filter_params={"client_id": client_id}
        )
        remaining = [
            u for u in all_users
            if "client_admin" in u.get("roles", [])
            and u.get("user_id") != user_id
            and (not active_only or u.get("is_active", True))
        ]
        if not remaining:
            raise PermissionError(
                f"Cannot remove the last active client_admin for client {client_id}. "
                "Create or promote another client_admin first."
            )

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize user management module.

        Sets up:
        - Password hasher (bcrypt)
        - User management provider
        - Redis connection

        Returns:
            Initialization status with module info

        Raises:
            RuntimeError: If Redis not available or passlib not installed
        """
        logger.info(
            f"Initializing {self.manifest.name} module",
            extra={"mod_name": self.manifest.name},
        )

        # Initialize password hasher
        rounds = (
            self.config.get("security", {})
            .get("password_hashing", {})
            .get("rounds", 12)
        )
        self.password_hasher = PasswordHasher(rounds=rounds)

        # Get Redis client - try DI container first, then fall back to event_bus/kwargs
        redis_client = None

        # Try DI container (recommended approach)
        if self.di_container:
            try:
                # Import Redis type
                import redis.asyncio as aioredis

                redis_client = await self.di_container.resolve(aioredis.Redis)
                logger.debug("Redis client resolved from DI container")
            except Exception as e:
                logger.warning(f"Failed to resolve Redis from DI container: {e}")

        # Fall back to event_bus or kwargs (legacy pattern)
        if not redis_client:
            # Try event_bus.redis_client (legacy pattern - may not exist on all EventBus implementations)
            if self.event_bus:
                redis_client = getattr(self.event_bus, "redis_client", None)
                if redis_client:
                    logger.debug("Redis client obtained from event_bus")
            if not redis_client and "redis_client" in self.kwargs:
                redis_client = self.kwargs["redis_client"]
                logger.debug("Redis client obtained from kwargs")

        if not redis_client:
            logger.error(
                "Redis client not available via DI container, event_bus, or kwargs"
            )
            raise RuntimeError("Redis client not available for admin_users module")

        # Initialize user management provider
        self.user_provider = UserManagementProvider(
            redis_client=redis_client,
            password_hasher=self.password_hasher,
            config=self.config,
        )

        # Initialize invitation provider
        self.invitation_provider = InvitationProvider(redis_client=redis_client)

        self._initialized = True

        logger.info(f"✅ {self.manifest.name} initialized successfully")

        return {
            "status": "initialized",
            "module": self.manifest.name,
            "storage": "redis",
            "security": {
                "password_hashing": "bcrypt",
                "bcrypt_rounds": rounds,
                "mfa_support": self.config.get("security", {}).get(
                    "mfa_enabled", False
                ),
            },
            "features": {
                "username_index": "O(1) lookup",
                "event_bus": "enabled" if self.publisher else "disabled",
                "multi_tenancy": "supported",
                "audit_logging": "enabled",
            },
        }

    async def shutdown(self) -> None:
        """Shutdown and cleanup."""
        logger.info(f"Shutting down {self.manifest.name} module")
        self._initialized = False
        self.password_hasher = None
        self.user_provider = None
        logger.info(f"✅ {self.manifest.name} shutdown successfully")

    # ========================================================================
    # Public Registration (MVP+ Task #28)
    # ========================================================================

    async def _validate_client_id(self, client_id: str) -> bool:
        """
        Validate that client_id exists in Redis.

        This is a security measure to prevent arbitrary registrations.
        Only users with a valid client_id can register.

        Args:
            client_id: The client ID to validate

        Returns:
            True if client exists and is active, False otherwise
        """
        if not client_id or not client_id.strip():
            logger.warning("[SECURITY] Empty client_id in registration attempt")
            return False

        # Get Redis client
        redis_client = None
        provider = self._require_initialized()
        redis_client = provider.redis_client

        if not redis_client:
            # Enterprise security hardening:
            # - In production (or unknown env), fail closed to prevent arbitrary client_id usage.
            # - In dev/test, allow to keep local development usable.
            env = str(self.config.get("env", "")).lower()
            is_dev_or_test = env in {"dev", "test"}

            if is_dev_or_test:
                logger.warning(
                    "[SECURITY WARNING] Redis unavailable for client_id validation in DEV/TEST. "
                    "Allowing registration with unverified client_id."
                )
                return True

            logger.error(
                "[SECURITY] Redis unavailable for client_id validation in PROD. "
                "Denying operation (fail closed)."
            )
            return False

        try:
            # Check if client exists using admin_clients key pattern
            # Key pattern: ubp:admin:client:{client_id}
            client_key = f"ubp:admin:client:{client_id}"
            client_data = await redis_client.get(client_key)

            if not client_data:
                logger.warning(
                    f"[SECURITY] Registration attempt with non-existent client_id: {client_id}"
                )
                return False

            # Parse client data to check if active
            import json

            client = json.loads(client_data)

            if not client.get("is_active", False):
                logger.warning(
                    f"[SECURITY] Registration attempt with inactive client_id: {client_id}"
                )
                return False

            logger.info(f"[OK] client_id validated: {client_id}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to validate client_id: {e}")
            # Fail closed on errors - deny registration
            return False

    async def _check_client_user_limit(self, client_id: str) -> Dict[str, Any]:
        """
        Check if the client has available user slots.

        Enterprise v2.0: Enforces max_users limit per client.

        Args:
            client_id: The client ID to check

        Returns:
            Dict with can_register (bool), max_users, current_users, available_slots
        """
        provider = self._require_initialized()
        redis_client = provider.redis_client

        try:
            client_key = f"ubp:admin:client:{client_id}"
            client_data = await redis_client.get(client_key)

            if not client_data:
                return {
                    "can_register": False,
                    "error": f"Client not found: {client_id}",
                }

            client = json.loads(client_data)
            user_limits = client.get("user_limits", {})

            max_users = user_limits.get("max_users", 100)
            current_users = user_limits.get("current_users", 0)
            available = max_users - current_users

            return {
                "can_register": available > 0,
                "max_users": max_users,
                "current_users": current_users,
                "available_slots": available,
                "client_id": client_id,
            }

        except Exception as e:
            logger.error(f"[ERROR] Failed to check user limit: {e}")
            return {
                "can_register": False,
                "error": f"Failed to check user limit: {str(e)}",
            }

    async def _track_user_registration(
        self,
        client_id: str,
        user_id: str,
        username: str,
    ) -> Dict[str, Any]:
        """
        Track user registration with client and increment user count.

        Enterprise v2.0: Updates client's current_users count and adds user to client_users index.

        Args:
            client_id: Client ID the user registered with
            user_id: Newly created user ID
            username: Username for logging

        Returns:
            Dict with tracking status
        """
        provider = self._require_initialized()
        redis_client = provider.redis_client

        try:
            # 1. Update client's user count
            client_key = f"ubp:admin:client:{client_id}"
            client_data = await redis_client.get(client_key)

            if client_data:
                client = json.loads(client_data)
                user_limits = client.get("user_limits", {})
                current = user_limits.get("current_users", 0)
                client["user_limits"]["current_users"] = current + 1
                client["updated_at"] = datetime.utcnow().isoformat()

                # Save updated client
                await redis_client.set(client_key, json.dumps(client))

                # Update clients index
                await redis_client.hset(
                    "ubp:admin:clients", client_id, json.dumps(client)
                )

            # 2. Add user to client_users set (for efficient lookup)
            client_users_key = f"ubp:client_users:{client_id}"
            await redis_client.sadd(client_users_key, user_id)

            logger.info(
                f"[OK] User {username} ({user_id}) tracked for client {client_id}",
                extra={"user_id": user_id, "client_id": client_id},
            )

            return {
                "status": "success",
                "user_id": user_id,
                "client_id": client_id,
                "message": "User registration tracked successfully",
            }

        except Exception as e:
            logger.error(f"[ERROR] Failed to track user registration: {e}")
            return {
                "status": "error",
                "error": f"Failed to track registration: {str(e)}",
            }

    async def _create_personal_kb(
        self,
        user_id: str,
        username: str,
        client_id: str,
    ) -> Dict[str, Any]:
        """
        Create a personal knowledge base for a newly registered user.

        Enterprise v2.0: Auto-creates personal KB based on client's kb_config.

        Args:
            user_id: User ID
            username: Username
            client_id: Client ID (to check kb_config)

        Returns:
            Dict with KB creation status
        """
        provider = self._require_initialized()
        redis_client = provider.redis_client

        try:
            # 1. Check client's KB configuration
            client_key = f"ubp:admin:client:{client_id}"
            client_data = await redis_client.get(client_key)

            if not client_data:
                logger.warning(f"Client not found for KB creation: {client_id}")
                return {"status": "skipped", "reason": "client_not_found"}

            client = json.loads(client_data)
            kb_config = client.get("kb_config", {})
            default_user_kb = kb_config.get("default_user_kb_config", {})

            # Check if personal KB is enabled for this client
            personal_kb_enabled = default_user_kb.get("personal_kb_enabled", True)

            if not personal_kb_enabled:
                logger.info(f"Personal KB disabled for client {client_id}")
                return {"status": "skipped", "reason": "personal_kb_disabled"}

            # 2. Create personal KB name (personal_{user_id[:8]})
            kb_name = f"personal_{user_id[:8]}"

            # 3. Try to get rag_orchestrator module via DI container
            rag_module = None
            if self.di_container:
                try:
                    rag_module = await self.di_container.resolve("rag_orchestrator")
                except Exception as e:
                    logger.warning(f"Could not resolve rag_orchestrator: {e}")

            if not rag_module:
                # WARNING: This fallback path should NOT be used in production!
                # If rag_module is not available, we store metadata only and set ACL directly.
                # This indicates a DI container initialization issue that should be investigated.
                logger.warning(
                    "[DI-WARNING] rag_orchestrator module not available during personal KB creation. "
                    "Using fallback path - this should NOT happen in production! "
                    "Check DI container initialization order.",
                    extra={
                        "user_id": user_id,
                        "username": username,
                        "client_id": client_id,
                        "di_container_available": self.di_container is not None,
                    },
                )

                # Fallback: Store KB metadata in Redis, actual collection will be created lazily
                kb_metadata = {
                    "name": kb_name,
                    "owner_user_id": user_id,
                    "owner_username": username,
                    "client_id": client_id,
                    "type": "personal",
                    "max_size_mb": default_user_kb.get("personal_kb_max_size_mb", 100),
                    "max_documents": default_user_kb.get(
                        "personal_kb_max_documents", 50
                    ),
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "pending_creation",
                }

                # Store KB metadata
                kb_key = f"rag:kb:{kb_name}:metadata"
                await redis_client.set(kb_key, json.dumps(kb_metadata))

                # Add to user's KB list
                user_kbs_key = f"ubp:user_kbs:{user_id}"
                await redis_client.sadd(user_kbs_key, kb_name)

                # SECURITY FIX P0: Set ACL directly in Redis (rag_module unavailable)
                # User must have write access to their own personal KB for queries to work
                # Key schema: ubp:{env}:rag:acl:{entity_type}:{entity_id}:{collection_id}
                from ubp_enterprise_hybrid.backend.app.infra.redis_keys import get_key_manager

                key_manager = get_key_manager()
                acl_key = key_manager.key("rag", "acl", "user", user_id, kb_name)

                await redis_client.set(acl_key, "write")
                logger.info(
                    f"[OK] ACL set for personal KB: user {user_id} -> {kb_name} (write)",
                    extra={"user_id": user_id, "kb_name": kb_name, "acl_key": acl_key},
                )

                logger.info(
                    f"[OK] Personal KB metadata stored for user {username}: {kb_name}",
                    extra={"user_id": user_id, "kb_name": kb_name},
                )

                return {
                    "status": "metadata_created",
                    "kb_name": kb_name,
                    "message": "KB metadata stored, collection will be created on first use",
                }

            # 4. Create actual collection via rag_orchestrator
            # Need to create a fake admin context for the operation
            from types import SimpleNamespace

            admin_ctx = SimpleNamespace(
                user=SimpleNamespace(
                    user_id="system",
                    roles=["admin"],
                )
            )

            result = await rag_module.create_knowledge_base(
                name=kb_name,
                description=f"Personal knowledge base for {username}",
                ctx=admin_ctx,
            )

            if result.get("status") == "created":
                # Store KB metadata
                kb_metadata = {
                    "name": kb_name,
                    "owner_user_id": user_id,
                    "owner_username": username,
                    "client_id": client_id,
                    "type": "personal",
                    "max_size_mb": default_user_kb.get("personal_kb_max_size_mb", 100),
                    "max_documents": default_user_kb.get(
                        "personal_kb_max_documents", 50
                    ),
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "active",
                }

                kb_key = f"rag:kb:{kb_name}:metadata"
                await redis_client.set(kb_key, json.dumps(kb_metadata))

                # Add to user's KB list
                user_kbs_key = f"ubp:user_kbs:{user_id}"
                await redis_client.sadd(user_kbs_key, kb_name)

                # Set ACL: owner has full access (read+write)
                if hasattr(rag_module, "set_permission"):
                    await rag_module.set_permission(
                        entity_type="user",
                        entity_id=user_id,
                        collection_id=kb_name,
                        access_level="write",
                        ctx=admin_ctx,
                    )

                logger.info(
                    f"[OK] Personal KB created for user {username}: {kb_name}",
                    extra={"user_id": user_id, "kb_name": kb_name},
                )

                return {
                    "status": "created",
                    "kb_name": kb_name,
                    "message": f"Personal KB '{kb_name}' created successfully",
                }
            else:
                logger.warning(f"Failed to create personal KB: {result.get('message')}")
                return {
                    "status": "error",
                    "kb_name": kb_name,
                    "error": result.get("message", "Unknown error"),
                }

        except Exception as e:
            logger.error(f"[ERROR] Failed to create personal KB: {e}")
            return {
                "status": "error",
                "error": f"Failed to create personal KB: {str(e)}",
            }

    async def register(
        self,
        username: str,
        password: str,
        email: str,
        client_id: str,
        full_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Public self-registration endpoint.

        This is a PUBLIC endpoint (no auth required) with security measures:
        1. client_id validation against Redis
        2. User limit check (Enterprise v2.0)
        3. Rate limiting (declared in manifest, enforced by router)
        4. Forced safe defaults (roles=["user"], is_active=True)
        5. Audit trail (registered_via_client field)
        6. Client-user tracking (Enterprise v2.0)
        7. Personal KB creation (Enterprise v2.0)

        Args:
            username: Unique username (3-50 chars)
            password: Password (min 6 chars)
            email: User email (required for self-registration)
            client_id: ID of the client application (must exist in Redis)
            full_name: Optional full name

        Returns:
            Success response with user_id

        Raises:
            ValueError: If validation fails or client_id invalid
        """
        request_id = str(uuid.uuid4())

        logger.info(
            f"[REGISTER] Self-registration attempt",
            extra={
                "username": username,
                "email": email,
                "client_id": client_id,
                "request_id": request_id,
            },
        )

        # 1. Validate client_id exists and is active
        if not await self._validate_client_id(client_id):
            raise ValueError(f"Invalid or inactive client_id: {client_id}")

        # 2. Check user limit (Enterprise v2.0)
        limit_check = await self._check_client_user_limit(client_id)
        if not limit_check.get("can_register"):
            error_msg = limit_check.get(
                "error",
                f"User limit reached for this client. Max: {limit_check.get('max_users', 'N/A')}, "
                f"Current: {limit_check.get('current_users', 'N/A')}",
            )
            logger.warning(
                f"[LIMIT] User registration blocked: {error_msg}",
                extra={"client_id": client_id, "request_id": request_id},
            )
            raise ValueError(error_msg)

        # 3. Validate input
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(username) > 50:
            raise ValueError("Username must be at most 50 characters")

        if not password or len(password) < 6:
            raise ValueError("Password must be at least 6 characters")

        if not email or "@" not in email:
            raise ValueError("Valid email is required for registration")

        # 4. Check username availability
        provider = self._require_initialized()
        username_key = (
            f"{provider.config['redis']['keys']['username_index_prefix']}{username}"
        )
        if await provider.redis_client.exists(username_key):
            raise ValueError(f"Username '{username}' is already taken")

        # 5. Create user with FORCED safe defaults
        # This ensures self-registered users cannot elevate privileges
        user = await provider.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name or username,
            roles=["user"],  # FORCED: cannot be overridden by request
            is_active=True,  # FORCED: new users are active
            tenant_id=None,  # No tenant for self-registration (single-tenant MVP)
            client_id=client_id,  # ISSUE B FIX: Pass client_id for token generation
        )

        # 6. Store audit trail - which client was used to register
        user_id = user["user_id"]
        user_key = f"{provider.config['redis']['keys']['user_prefix']}{user_id}"

        user_data = await provider.redis_client.get(user_key)
        if user_data:
            user_record = json.loads(user_data)
            user_record["registered_via_client"] = client_id
            user_record["registration_ip"] = kwargs.get("_request_ip", "unknown")
            await provider.redis_client.set(user_key, json.dumps(user_record))

            # Update index too
            users_index = provider.config["redis"]["keys"]["users_index"]
            await provider.redis_client.hset(
                users_index, user_id, json.dumps(user_record)
            )

        # 7. Track user registration with client (Enterprise v2.0)
        tracking_result = await self._track_user_registration(
            client_id=client_id,
            user_id=user_id,
            username=username,
        )

        # 8. Create personal KB (Enterprise v2.0)
        kb_result = await self._create_personal_kb(
            user_id=user_id,
            username=username,
            client_id=client_id,
        )

        # 8.1 CRITICAL: Update user record with personal_kb field
        # Without this, ingest_to_personal_kb will fail with "Personal KB not enabled"
        if kb_result.get("kb_name"):
            user_data = await provider.redis_client.get(user_key)
            if user_data:
                user_record = json.loads(user_data)
                user_record["personal_kb"] = kb_result.get("kb_name")
                await provider.redis_client.set(user_key, json.dumps(user_record))

                # Update index too
                users_index = provider.config["redis"]["keys"]["users_index"]
                await provider.redis_client.hset(
                    users_index, user_id, json.dumps(user_record)
                )

                logger.debug(
                    f"[OK] User record updated with personal_kb: {kb_result.get('kb_name')}",
                    extra={"user_id": user_id, "personal_kb": kb_result.get("kb_name")},
                )

        # 9. Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.registered",
                {
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "client_id": client_id,
                    "personal_kb": kb_result.get("kb_name"),
                    "timestamp": user["created_at"],
                    "request_id": request_id,
                },
            )

        logger.info(
            f"[OK] User registered successfully",
            extra={
                "user_id": user_id,
                "username": username,
                "client_id": client_id,
                "personal_kb": kb_result.get("kb_name"),
                "request_id": request_id,
            },
        )

        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "message": "Registration successful. Please login with your credentials.",
            "personal_kb": kb_result.get("kb_name"),
            "request_id": request_id,
        }

    # ========================================================================
    # CRUD Operations (delegated to provider)
    # ========================================================================

    async def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        is_active: bool = True,
        target_client_id: Optional[str] = None,
        target_user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Create a new user account.

        Admin can create users for any client.
        client_admin can create users only for their own client, and cannot
        assign admin or client_admin roles.
        Public registration uses the separate `register()` endpoint.

        Args:
            username: Unique username
            password: Password (will be hashed)
            email: User email (optional)
            full_name: Full name (optional)
            roles: List of role names (optional, "admin"/"client_admin" requires admin caller)
            tenant_id: Tenant ID for multi-tenancy (optional)
            is_active: Active status (default True)
            target_client_id: Target client (for creating users in specific clients)
            request_id: Request tracking ID (optional)
            ctx: Request context with caller identity (REQUIRED)
        """
        # Allow platform admin OR client_admin (scoped to own client)
        ctx = self._require_admin_or_client_admin(
            ctx, "create_user", target_client_id
        )

        # SECURITY: client_admin cannot assign privileged roles
        if self._is_client_admin(ctx) and not self._is_admin(ctx):
            privileged = {"admin", "client_admin"}
            if roles and privileged.intersection(set(roles)):
                raise PermissionError(
                    "client_admin cannot assign admin or client_admin roles"
                )
            # client_admin can only create users in their own client
            caller_client_id = getattr(ctx.user, "client_id", None)
            if target_client_id and target_client_id != caller_client_id:
                raise PermissionError(
                    "client_admin can only create users in their own client"
                )
            # Force target to their own client if not specified
            if not target_client_id:
                target_client_id = caller_client_id

        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        caller_user_id = ctx.user.user_id
        caller_client_id = getattr(ctx.user, "client_id", None)

        # ====================================================================
        # SECURITY RULE-001 & RULE-004: Admin can specify target_client_id
        # ====================================================================
        # Admin can specify target_client_id to create users for any client
        # If not specified, use caller's client_id (may be None for system admin)
        if target_client_id:
            effective_client_id = target_client_id
            logger.info(
                f"[SECURITY] {ctx.user.user_id} creating user for client {target_client_id}",
                extra={"request_id": request_id},
            )
        else:
            effective_client_id = caller_client_id

        # Validate effective_client_id exists (if provided)
        if effective_client_id:
            if not await self._validate_client_id(effective_client_id):
                raise ValueError(
                    f"Invalid or inactive client_id: {effective_client_id}"
                )

            # Check user limit for the client
            limit_check = await self._check_client_user_limit(effective_client_id)
            if not limit_check.get("can_register"):
                raise ValueError(
                    f"User limit reached for client. Max: {limit_check.get('max_users', 'N/A')}, "
                    f"Current: {limit_check.get('current_users', 'N/A')}"
                )

        logger.info(
            "[SECURITY] Creating user with validated context",
            extra={
                "username": username,
                "request_id": request_id,
                "caller_user_id": caller_user_id,
                "caller_client_id": caller_client_id,
                "caller_is_admin": True,  # Guaranteed by _require_admin
                "effective_client_id": effective_client_id,
                "roles_requested": roles,
            },
        )

        # Validate target_user_id (only admins can force a specific UUID; client_admin
        # is rejected to prevent privilege/identity hijack scenarios)
        forced_user_id: Optional[str] = None
        if target_user_id:
            if not self._is_admin(ctx):
                raise PermissionError(
                    "target_user_id forcing requires platform admin caller"
                )
            try:
                uuid.UUID(target_user_id)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(
                    f"target_user_id must be a valid UUID: {target_user_id!r}"
                )
            forced_user_id = target_user_id
            logger.warning(
                "[SECURITY] target_user_id forcing requested by admin %s for username '%s' → %s",
                caller_user_id, username, forced_user_id,
            )

        # Delegate to provider
        provider = self._require_initialized()
        user = await provider.create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name,
            roles=roles,
            tenant_id=tenant_id,
            is_active=is_active,
            client_id=effective_client_id,
            forced_user_id=forced_user_id,
        )

        # Track user with client if client_id provided
        if effective_client_id:
            await self._track_user_registration(
                client_id=effective_client_id,
                user_id=user["user_id"],
                username=username,
            )

        # ====================================================================
        # BUG-004 FIX: Create Personal KB for user (like register_user does)
        # ====================================================================
        # This ensures parity between admin-created and self-registered users
        kb_result = None
        if effective_client_id:
            kb_result = await self._create_personal_kb(
                user_id=user["user_id"],
                username=username,
                client_id=effective_client_id,
            )

            # Update user record with personal_kb field (same as register_user)
            if kb_result and kb_result.get("kb_name"):
                user_key = f"{provider.config['redis']['keys']['user_prefix']}{user['user_id']}"
                user_data = await provider.redis_client.get(user_key)
                if user_data:
                    user_record = json.loads(user_data)
                    user_record["personal_kb"] = kb_result.get("kb_name")
                    await provider.redis_client.set(user_key, json.dumps(user_record))

                    # Update index too
                    users_index = provider.config["redis"]["keys"]["users_index"]
                    await provider.redis_client.hset(
                        users_index, user["user_id"], json.dumps(user_record)
                    )

                    logger.info(
                        f"[BUG-004-FIX] User record updated with personal_kb: {kb_result.get('kb_name')}",
                        extra={
                            "user_id": user["user_id"],
                            "personal_kb": kb_result.get("kb_name"),
                        },
                    )

                # Add to response
                user["personal_kb"] = kb_result.get("kb_name")

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.created",
                {
                    "user_id": user["user_id"],
                    "username": username,
                    "roles": roles or [],
                    "tenant_id": tenant_id,
                    "client_id": effective_client_id,
                    "personal_kb": kb_result.get("kb_name") if kb_result else None,
                    "created_by_user_id": caller_user_id,
                    "created_by_client_id": caller_client_id,
                    "timestamp": user["created_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        user["request_id"] = request_id
        return user

    async def list_users(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        ctx: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        List users with filtering and pagination.

        Admin can list all users; client_admin can only list users in their own client.
        client_admin listings are automatically scoped to their client_id.

        Args:
            filter: Filter parameters
            limit: Max results
            offset: Skip results
            ctx: Request context with caller identity (REQUIRED)
        """
        # Allow platform admin or client_admin (client_admin is auto-scoped below)
        filter_client_id = (filter or {}).get("client_id") if filter else None
        ctx = self._require_admin_or_client_admin(ctx, "list_users", filter_client_id)

        # client_admin: force-scope results to their own client
        if self._is_client_admin(ctx) and not self._is_admin(ctx):
            caller_client_id = getattr(ctx.user, "client_id", None)
            if caller_client_id:
                filter = dict(filter or {})
                filter["client_id"] = caller_client_id

        logger.debug(
            "Listing users",
            extra={
                "filter": filter,
                "limit": limit,
                "offset": offset,
                "admin_user_id": ctx.user.user_id,
            },
        )

        provider = self._require_initialized()
        all_users = await provider.list_users(
            filter_params=filter,
            limit=None,
            offset=None,  # Get all first, then filter
        )

        # Apply pagination
        if offset:
            all_users = all_users[offset:]
        if limit:
            all_users = all_users[:limit]

        return all_users

    async def get_user(self, user_id: str, ctx: Any = None) -> Dict[str, Any]:
        """
        Get user by ID.

        SECURITY PATCH P0 - IDOR FIX:
        - Platform admin can access any user
        - client_admin can access users within their own client
        - Non-admin users can ONLY access their own data

        Args:
            user_id: User UUID
            ctx: Request context with caller identity (REQUIRED)
        """
        ctx = self._require_ctx(ctx)
        caller_user_id = ctx.user.user_id
        caller_is_admin = self._is_admin(ctx)
        caller_is_client_admin = self._is_client_admin(ctx)

        # Platform admin has unrestricted access
        if caller_is_admin:
            pass
        elif caller_is_client_admin:
            # client_admin can only access own-client users — enforce after fetch
            pass
        elif caller_user_id != user_id:
            # Regular users can only access their own data (IDOR protection)
            logger.warning(
                f"[SECURITY] IDOR attempt blocked: user {caller_user_id} tried to access user {user_id}",
                extra={
                    "caller_user_id": caller_user_id,
                    "target_user_id": user_id,
                },
            )
            raise PermissionError("Access denied to other user data")

        logger.debug("Getting user", extra={"user_id": user_id})
        provider = self._require_initialized()
        user = await provider.get_user(user_id)

        # client_admin cross-client probe prevention:
        # client_admin can only see users within their own client
        if caller_is_client_admin and not caller_is_admin and caller_user_id != user_id:
            caller_client_id = getattr(ctx.user, "client_id", None)
            target_client_id = user.get("client_id")
            if target_client_id and caller_client_id != target_client_id:
                logger.warning(
                    f"[SECURITY] client_admin {caller_user_id} attempted cross-client user probe "
                    f"(own={caller_client_id}, target_user_client={target_client_id})",
                    extra={"caller_user_id": caller_user_id, "target_user_id": user_id},
                )
                raise PermissionError("Access denied to user in different client")

        return user

    async def get_user_internal(self, user_id: str) -> Dict[str, Any]:
        """
        Get user by ID - INTERNAL USE ONLY.

        This method bypasses security checks for module-to-module calls.
        Use this ONLY for internal operations like:
        - Checking user client membership for KB operations (GAP-003/004)
        - Internal validation operations
        - Cross-module lookups

        DO NOT expose this method via API routes.

        Args:
            user_id: User UUID

        Returns:
            User object (without password_hash)
        """
        logger.debug(
            "Getting user (internal)",
            extra={"user_id": user_id, "caller": "internal"},
        )
        provider = self._require_initialized()
        return await provider.get_user(user_id)

    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with username and password.

        Delegated to UserManagementProvider for bcrypt verification.

        Args:
            username: Username
            password: Plaintext password

        Returns:
            User object (without password_hash) if authenticated

        Raises:
            ValueError: If user not found or password invalid
        """
        logger.debug("Authenticating user", extra={"username": username})
        provider = self._require_initialized()
        return await provider.authenticate_user(username, password)

    async def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        mfa_enabled: Optional[bool] = None,
        client_id: Optional[str] = None,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Update user information.

        Admin can update any user.
        client_admin can update users within their own client only, and cannot
        assign admin or client_admin roles.

        Args:
            user_id: User UUID
            username: New username (optional)
            email: New email (optional)
            full_name: New full name (optional)
            roles: New roles (optional)
            tenant_id: New tenant ID (optional)
            is_active: New active status (optional)
            mfa_enabled: MFA enabled status (optional)
            request_id: Request tracking ID
            ctx: Request context with caller identity (REQUIRED)
        """
        # Generate request ID first (needed for logging)
        if not request_id:
            request_id = str(uuid.uuid4())

        # Get target user to determine their client_id for scoping
        provider = self._require_initialized()
        target_user = await provider.get_user(user_id)
        target_user_client_id = target_user.get("client_id")
        target_user_roles = set(target_user.get("roles", []))

        # Allow platform admin or client_admin (scoped to own client)
        ctx = self._require_admin_or_client_admin(
            ctx, "update_user", target_user_client_id
        )

        # SECURITY: client_admin cannot assign privileged roles
        if self._is_client_admin(ctx) and not self._is_admin(ctx):
            privileged = {"admin", "client_admin"}
            if roles and privileged.intersection(set(roles)):
                raise PermissionError(
                    "client_admin cannot assign admin or client_admin roles"
                )

        # ----------------------------------------------------------------
        # ORPHAN PROTECTION: Prevent demotion / deactivation / client-move
        # that would leave a client with no active client_admin.
        #
        # Triggers when ALL of the following are true:
        #   1. The target user currently has client_admin role
        #   2. This update would strip that role, deactivate them,
        #      or move them to a different client
        #   3. They are the last active client_admin for their current client
        # ----------------------------------------------------------------
        if target_user_client_id and "client_admin" in target_user_roles:
            would_demote = roles is not None and "client_admin" not in roles
            would_deactivate = is_active is False
            would_move = client_id is not None and client_id != target_user_client_id

            if would_demote or would_deactivate or would_move:
                reason = (
                    "role removal" if would_demote
                    else "deactivation" if would_deactivate
                    else "client reassignment"
                )
                try:
                    await self._check_last_client_admin(
                        user_id, target_user_client_id, provider
                    )
                except PermissionError:
                    raise PermissionError(
                        f"Cannot perform {reason} on the last active client_admin "
                        f"for client {target_user_client_id}. "
                        "Create or promote another client_admin first."
                    )

        caller_user_id = ctx.user.user_id
        caller_client_id = getattr(ctx.user, "client_id", None)

        logger.info(
            "Updating user",
            extra={
                "user_id": user_id,
                "request_id": request_id,
                "caller_user_id": caller_user_id,
                "caller_is_admin": self._is_admin(ctx),
            },
        )

        # Delegate to provider
        user = await provider.update_user(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            roles=roles,
            tenant_id=tenant_id,
            is_active=is_active,
            mfa_enabled=mfa_enabled,
            client_id=client_id,
        )

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.updated",
                {
                    "user_id": user_id,
                    "username": user.get("username"),
                    "updated_by_user_id": caller_user_id,
                    "updated_by_client_id": caller_client_id,
                    "changes": {
                        "username": username is not None,
                        "roles": roles is not None,
                        "is_active": is_active is not None,
                        "client_id": client_id is not None,
                    },
                    "timestamp": user["updated_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        user["request_id"] = request_id
        return user

    async def delete_user(
        self,
        user_id: str,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, str]:
        """
        Delete a user account.

        Admin can delete any user.
        client_admin can delete users within their own client only,
        but cannot delete other client_admin or admin users.

        Args:
            user_id: User UUID
            request_id: Request tracking ID
            ctx: Request context with caller identity (REQUIRED)
        """
        # Generate request ID first
        if not request_id:
            request_id = str(uuid.uuid4())

        # Get target user to determine their client_id for scoping
        provider = self._require_initialized()
        target_user = await provider.get_user(user_id)
        target_user_client_id = target_user.get("client_id")
        target_user_roles = target_user.get("roles", [])

        # Allow platform admin or client_admin (scoped to own client)
        ctx = self._require_admin_or_client_admin(
            ctx, "delete_user", target_user_client_id
        )

        # SECURITY: client_admin cannot delete privileged users
        if self._is_client_admin(ctx) and not self._is_admin(ctx):
            privileged = {"admin", "client_admin"}
            if privileged.intersection(set(target_user_roles)):
                raise PermissionError(
                    "client_admin cannot delete admin or client_admin users"
                )

        # ORPHAN PROTECTION: Cannot delete the last client_admin for a client.
        # At least one active client_admin must remain so the client is manageable.
        if target_user_client_id and "client_admin" in target_user_roles:
            await self._check_last_client_admin(
                user_id, target_user_client_id, provider
            )

        caller_user_id = ctx.user.user_id
        caller_client_id = getattr(ctx.user, "client_id", None)

        logger.info(
            "Deleting user",
            extra={
                "user_id": user_id,
                "request_id": request_id,
                "caller_user_id": caller_user_id,
                "caller_is_admin": self._is_admin(ctx),
            },
        )

        # Delegate to provider
        result = await provider.delete_user(user_id)

        # BUG-DEL-001: Decrement user count and remove from client_users set
        target_client_id = target_user.get("client_id")
        if target_client_id and self.di_container:
            try:
                admin_clients = await self.di_container.resolve("admin_clients")
                if admin_clients:
                    await admin_clients.decrement_user_count(
                        target_client_id, user_id
                    )
                    logger.info(
                        f"[ADMIN_USERS] Decremented user count for client {target_client_id}"
                    )
            except Exception as e:
                # Non-fatal: client count is cosmetic, don't block user deletion
                logger.warning(
                    f"[ADMIN_USERS] Failed to decrement user count for client "
                    f"{target_client_id}: {e}"
                )

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.deleted",
                {
                    "user_id": user_id,
                    "username": result["username"],
                    "deleted_by_user_id": caller_user_id,
                    "deleted_by_client_id": caller_client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        result["request_id"] = request_id
        return result

    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Change user password.

        Delegated to UserManagementProvider.
        Publishes event on success.
        """
        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Changing password", extra={"user_id": user_id, "request_id": request_id}
        )

        # Delegate to provider
        provider = self._require_initialized()
        result = await provider.change_password(
            user_id=user_id, old_password=old_password, new_password=new_password
        )

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.password_changed",
                {
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        result["request_id"] = request_id
        return result

    async def admin_reset_password(
        self,
        user_id: str,
        new_password: str,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, str]:
        """Admin password reset — bypasses old password verification.

        Security: requires admin (any user) or client_admin (own client only).
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        provider = self._require_initialized()

        # Get target user to determine client scope
        target_user = await provider.get_user(user_id)
        target_user_client_id = target_user.get("client_id")

        # Enforce admin or client_admin with domain check
        ctx = self._require_admin_or_client_admin(
            ctx, "admin_reset_password", target_user_client_id
        )

        logger.info(
            "Admin password reset",
            extra={
                "user_id": user_id,
                "caller": getattr(ctx.user, "user_id", "unknown"),
                "request_id": request_id,
            },
        )

        result = await provider.admin_reset_password(
            user_id=user_id, new_password=new_password
        )

        if self.publisher:
            await self.publisher.publish(
                "admin.user.password_reset",
                {
                    "user_id": user_id,
                    "reset_by": getattr(ctx.user, "user_id", "unknown"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        result["request_id"] = request_id
        return result

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get user management statistics.

        Delegated to UserManagementProvider.
        """
        provider = self._require_initialized()
        stats = await provider.get_stats()

        return {
            "module": self.manifest.name,
            **stats,
            "storage": "redis",
            "security": {"password_hashing": "bcrypt", "username_index": "O(1)"},
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for user management module.

        Tests Redis connectivity and module initialization status.
        """
        health = {
            "module": self.manifest.name,
            "status": "healthy",
            "initialized": self._initialized,
            "redis": {"connected": self.user_provider is not None, "status": "unknown"},
        }

        # Test Redis connection
        if self.user_provider and self.user_provider.redis_client:
            try:
                await self.user_provider.redis_client.ping()
                health["redis"]["status"] = "healthy"
            except Exception as e:
                health["redis"]["status"] = "unhealthy"
                health["redis"]["error"] = str(e)
                health["status"] = "degraded"
                logger.warning("Redis health check failed", extra={"error": str(e)})
        else:
            health["status"] = "unhealthy"
            health["redis"]["status"] = "not_connected"

        return health

    async def ingest_to_personal_kb(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document to user's personal knowledge base.

        This endpoint allows users to ingest documents to their Personal KB.
        Fixes GAP-INGEST-002: User-scoped document ingestion to Personal KB.

        Security:
        - Requires authenticated context
        - Verifies user is ingesting to their own Personal KB
        - Respects Personal KB limits from client config

        Args:
            user_id: User ID performing the ingestion
            text: Document text to ingest
            metadata: Optional metadata for the document
            ctx: Security context (required)
            request_id: Optional request tracking ID

        Returns:
            Ingestion result with document_id and status
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        metadata = metadata or {}

        # Verify context exists
        if not ctx or not hasattr(ctx, "user"):
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": "Authentication required",
                "request_id": request_id,
            }

        ctx_user_id = getattr(ctx.user, "user_id", None)
        client_id = getattr(ctx.user, "client_id", None)

        # Verify user is ingesting to their own Personal KB
        if ctx_user_id != user_id:
            logger.warning(
                "Unauthorized personal KB ingest attempt",
                extra={
                    "ctx_user_id": ctx_user_id,
                    "requested_user_id": user_id,
                    "request_id": request_id,
                },
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": "Can only ingest to your own Personal KB",
                "request_id": request_id,
            }

        # Get user data to verify Personal KB
        try:
            provider = self._require_initialized()
            user_data = await provider.get_user(user_id)

            if not user_data:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": f"User {user_id} not found",
                    "request_id": request_id,
                }

            # Check if user has a Personal KB
            personal_kb = user_data.get("personal_kb")
            if not personal_kb:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": "Personal KB not enabled for this user",
                    "request_id": request_id,
                }

            # SECURITY FIX P0: personal_kb is stored as a string (kb_name), not a dict
            # Handle both formats for backward compatibility
            if isinstance(personal_kb, str):
                collection_id = personal_kb
            elif isinstance(personal_kb, dict):
                collection_id = personal_kb.get("collection_id") or personal_kb.get(
                    "kb_name"
                )
            else:
                collection_id = None

            if not collection_id:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": "Personal KB collection_id not found",
                    "request_id": request_id,
                }

            # Get Personal KB limits from client config
            if client_id and self.di_container:
                try:
                    admin_clients = await self.di_container.resolve("admin_clients")
                    if admin_clients:
                        # Use internal method to bypass admin check for config lookup
                        client_data = await admin_clients.get_client_internal(client_id)
                        if client_data:
                            kb_config = client_data.get("kb_config", {})
                            user_kb_config = kb_config.get("default_user_kb_config", {})

                            # Check if Personal KB is enabled in client config
                            if not user_kb_config.get("personal_kb_enabled", True):
                                return {
                                    "document_id": "",
                                    "chunks_count": 0,
                                    "status": "error",
                                    "message": "Personal KB is disabled for this client",
                                    "request_id": request_id,
                                }

                            # Check document size limit
                            max_size_mb = user_kb_config.get(
                                "personal_kb_max_size_mb", 100
                            )
                            text_size_mb = len(text.encode("utf-8")) / (1024 * 1024)
                            if text_size_mb > max_size_mb:
                                return {
                                    "document_id": "",
                                    "chunks_count": 0,
                                    "status": "error",
                                    "message": f"Document size ({text_size_mb:.2f}MB) exceeds Personal KB limit ({max_size_mb}MB)",
                                    "request_id": request_id,
                                }
                except Exception as e:
                    logger.warning(
                        f"Could not check client KB limits: {e}",
                        extra={"client_id": client_id, "request_id": request_id},
                    )

        except Exception as e:
            logger.error(
                f"Error checking user configuration: {e}",
                extra={"user_id": user_id, "request_id": request_id},
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": f"Error checking user configuration: {str(e)}",
                "request_id": request_id,
            }

        # Resolve rag_orchestrator module
        if not self.di_container:
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": "DI container not available",
                "request_id": request_id,
            }

        try:
            rag_module = await self.di_container.resolve("rag_orchestrator")
            if not rag_module:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": "RAG orchestrator not available",
                    "request_id": request_id,
                }

            # Add user tracking to metadata
            metadata["user_id"] = user_id
            metadata["client_id"] = client_id if client_id else "none"
            metadata["uploader_id"] = user_id
            metadata["ingestion_source"] = "personal_kb_endpoint"

            # Call the authorized ingest endpoint
            result = await rag_module.ingest_document_authorized(
                collection_id=collection_id,
                text=text,
                metadata=metadata,
                ctx=ctx,
            )

            # Add request_id to result
            result["request_id"] = request_id

            logger.info(
                "Personal KB document ingestion completed",
                extra={
                    "user_id": user_id,
                    "collection_id": collection_id,
                    "status": result.get("status"),
                    "document_id": result.get("document_id"),
                    "request_id": request_id,
                },
            )

            return result

        except Exception as e:
            logger.error(
                f"Error during personal KB ingestion: {e}",
                extra={"user_id": user_id, "request_id": request_id},
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": f"Error during ingestion: {str(e)}",
                "request_id": request_id,
            }

    # ========================================================================
    # Invitation-based Registration & Email Verification
    # ========================================================================

    async def register_with_invitation(
        self,
        username: str,
        password: str,
        email: str,
        display_name: str,
        invitation_code: str,
        request_ip: str,
    ) -> Dict[str, Any]:
        """
        Register a new user using an invitation code.

        Flow:
        1. Rate limit check
        2. Validate invitation code
        3. Validate input fields
        4. Create user (is_active=True, email_verified=False)
        5. Create email verification token
        6. Consume invitation usage
        7. Publish event

        Returns:
            Dict with user_id, username, verification_token, message
        """
        provider = self._require_initialized()
        inv = self.invitation_provider

        # 1. Rate limit
        await inv.check_registration_rate(request_ip)

        # 2. Validate invitation
        await inv.validate_invitation(invitation_code)

        # 3. Input validation
        if not username or len(username) < 3 or len(username) > 50:
            raise ValueError("Username deve essere tra 3 e 50 caratteri")
        if not password or len(password) < 6:
            raise ValueError("Password deve essere almeno 6 caratteri")
        if not email or "@" not in email:
            raise ValueError("Email non valida")
        if not display_name or len(display_name) < 1 or len(display_name) > 100:
            raise ValueError("Display name deve essere tra 1 e 100 caratteri")

        # 4. Create user via existing provider
        user = await provider.create_user(
            username=username,
            password=password,
            email=email,
            full_name=display_name,
            roles=["user"],
            is_active=True,
        )
        user_id = user["user_id"]

        # 5. Patch user record with registration metadata
        redis_client = provider.redis_client
        user_key = f"{provider.config['redis']['keys']['user_prefix']}{user_id}"
        user_raw = await redis_client.get(user_key)
        if user_raw:
            user_data = json.loads(user_raw)
            user_data["email_verified"] = False
            user_data["invitation_code"] = invitation_code
            user_data["registered_at"] = datetime.utcnow().isoformat()
            user_data["registration_ip"] = request_ip
            await redis_client.set(user_key, json.dumps(user_data))
            # Update index
            users_index = provider.config["redis"]["keys"]["users_index"]
            await redis_client.hset(users_index, user_id, json.dumps(user_data))

        # 6. Create email verification token
        token = await inv.create_email_verification_token(user_id, email)

        # 6b. Send verification email (fire-and-forget, non-blocking)
        try:
            from ubp_enterprise_hybrid.backend.app.core.email_service import get_email_service
            email_svc = get_email_service()
            await email_svc.send_verification_email(email, username, token)
        except Exception as e:
            logger.warning(f"Verification email failed for {username}: {e}")

        # 7. Consume invitation
        await inv.consume_invitation(invitation_code, user_id)

        # 8. Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.registered_via_invitation",
                {
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "invitation_code": invitation_code,
                },
            )

        logger.info(
            f"User registered via invitation: {username} ({user_id})",
            extra={"user_id": user_id, "invitation_code": invitation_code},
        )

        return {
            "user_id": user_id,
            "username": username,
            "message": "Registrazione completata. Verifica la tua email per attivare l'account.",
        }

    async def verify_email(self, token: str) -> Dict[str, Any]:
        """
        Verify user email using a verification token.

        Returns:
            Dict with user_id, username, message
        """
        provider = self._require_initialized()
        inv = self.invitation_provider

        # 1. Consume token
        token_data = await inv.consume_email_verification_token(token)
        user_id = token_data["user_id"]

        # 2. Update user record
        redis_client = provider.redis_client
        user_key = f"{provider.config['redis']['keys']['user_prefix']}{user_id}"
        user_raw = await redis_client.get(user_key)
        if not user_raw:
            raise ValueError("Utente non trovato")

        user_data = json.loads(user_raw)
        user_data["email_verified"] = True
        user_data["email_verified_at"] = datetime.utcnow().isoformat()
        user_data["updated_at"] = datetime.utcnow().isoformat()
        await redis_client.set(user_key, json.dumps(user_data))

        # Update index
        users_index = provider.config["redis"]["keys"]["users_index"]
        await redis_client.hset(users_index, user_id, json.dumps(user_data))

        # 3. Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.user.email_verified",
                {
                    "user_id": user_id,
                    "username": user_data.get("username", ""),
                    "email": token_data.get("email", ""),
                },
            )

        logger.info(
            f"Email verified for user {user_data.get('username', '')} ({user_id})"
        )

        return {
            "user_id": user_id,
            "username": user_data.get("username", ""),
            "message": "Email verificata con successo. Ora puoi effettuare il login.",
        }

    # --- Admin invitation management ---

    async def create_invitation(
        self,
        max_uses: int = 10,
        expires_at: Optional[str] = None,
        code: Optional[str] = None,
        ctx: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create an invitation code (admin only)."""
        self._require_initialized()
        created_by = ctx.user.user_id if ctx else "system"
        return await self.invitation_provider.create_invitation(
            created_by=created_by,
            max_uses=max_uses,
            expires_at=expires_at,
            code=code,
        )

    async def list_invitations(self, ctx: Optional[Any] = None) -> List[Dict[str, Any]]:
        """List all invitation codes (admin only)."""
        self._require_initialized()
        return await self.invitation_provider.list_invitations()

    async def get_invitation(self, code: str, ctx: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Get a specific invitation (admin only)."""
        self._require_initialized()
        return await self.invitation_provider.get_invitation(code)

    async def revoke_invitation(self, code: str, ctx: Optional[Any] = None) -> Dict[str, Any]:
        """Revoke an invitation code (admin only, soft-delete)."""
        self._require_initialized()
        return await self.invitation_provider.revoke_invitation(code)
