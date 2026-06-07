"""
Admin Clients Adapter - UBP Framework Bridge Layer

This module provides the UBP framework integration for client management.
Acts as a bridge between UBP's module system and technical provider implementations.

Separation of Concerns:
- adapter.py: UBP framework bridge (this file)
- providers.py: Pure technical logic (ZERO UBP dependencies)
- __init__.py: Factory entry point
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import uuid
import re
import json
import secrets
import string
from datetime import datetime

from ubp_enterprise_hybrid.modules.cores._shared import (
    BaseHybridModule,
    PLATFORM_ADMIN_CLIENT_ID,
    is_platform_admin_client,
    EntityType,
    AccessLevel,
    ErrorMessages,
)

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    from _shared.operation_context import OperationContext

# SEC-HIGH-002 FIX: KB name validation pattern
# Allows: letters, numbers, underscores, hyphens
# Must start with letter or underscore (for system prefixes like client_, personal_)
KB_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$")
from ubp_enterprise_hybrid.backend.app.infra.event_bus import Event
from .providers import SecretGenerator, ClientManagementProvider

logger = logging.getLogger(__name__)


class AdminClientsAdapter(BaseHybridModule):
    """
    UBP Adapter for client management module.

    This adapter:
    - Integrates with UBP lifecycle (initialize, shutdown, health_check)
    - Delegates all business logic to ClientManagementProvider
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
        self.secret_generator: Optional[SecretGenerator] = None
        self.client_provider: Optional[ClientManagementProvider] = None
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

        client_admin can manage settings and users within their own client only.

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
        self, ctx: Any, operation: str, target_client_id: Optional[str] = None
    ) -> Any:
        """
        Require platform admin OR client_admin privilege for an operation.

        client_admin users are restricted to their own client.
        Platform admins can access any client.

        Args:
            ctx: Security context
            operation: Name of the operation for error logging
            target_client_id: Client ID being accessed (used to scope client_admin)

        Returns:
            The validated ctx

        Raises:
            PermissionError: If user lacks sufficient privilege
            ValueError: If ctx is invalid
        """
        ctx = self._require_ctx(ctx)

        if self._is_admin(ctx):
            return ctx  # Platform admin: full access

        if self._is_client_admin(ctx):
            # client_admin can only manage their own client
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
                        "client_admin can only manage their own client"
                    )
            return ctx

        logger.warning(
            f"[SECURITY] Unauthorized {operation} attempt by user {ctx.user.user_id}",
            extra={"user_id": ctx.user.user_id, "operation": operation},
        )
        raise PermissionError(
            f"Admin or client_admin role required to perform: {operation}"
        )

    async def _create_initial_client_admin(
        self,
        client_id: str,
        client_name: str,
        request_id: str,
        ctx: Any,
    ) -> Dict[str, Any]:
        """
        Auto-create an initial client_admin user for a newly created client.

        TRANSACTIONAL GUARANTEE:
        If user creation fails (for any reason), this method raises an exception.
        The caller (create_client) must then mark the client as bootstrap_status=failed
        and is_active=False so it is not considered operational.
        The caller is responsible for compensating (rollback or status update).

        The password is generated here and returned to the caller.
        It is NEVER stored in any log or Redis — only returned in the response once.

        Args:
            client_id: The newly created client's ID
            client_name: Client name (used to derive username)
            request_id: Request tracking ID
            ctx: Admin security context (required for create_user)

        Returns:
            Dict with initial admin credentials (username + password, for one-time display)

        Raises:
            RuntimeError: If admin_users module is unavailable
            ValueError: If user creation fails even after retry
        """
        if not self.di_container:
            raise RuntimeError(
                "[AUTO-ADMIN] DI container unavailable — cannot create initial client_admin"
            )

        try:
            admin_users = await self.di_container.resolve("admin_users")
        except Exception as e:
            raise RuntimeError(
                f"[AUTO-ADMIN] Could not resolve admin_users module: {e}"
            ) from e

        if not admin_users:
            raise RuntimeError(
                "[AUTO-ADMIN] admin_users module not registered"
            )

        # Derive a safe username from the client name
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", client_name.lower()).strip("_")
        if not safe_name:
            safe_name = "client"
        username = f"{safe_name[:40]}_admin"

        # Generate a secure random password (16 chars, alphanumeric + specials)
        # SECURITY: This value is NEVER logged — it is only returned to the caller once.
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        # Ensure complexity: at least one uppercase, lowercase, digit, special
        password = (
            secrets.choice(string.ascii_uppercase)
            + secrets.choice(string.ascii_lowercase)
            + secrets.choice(string.digits)
            + secrets.choice("!@#$%^&*")
            + "".join(secrets.choice(alphabet) for _ in range(12))
        )
        # Shuffle to avoid predictable prefix
        password_list = list(password)
        secrets.SystemRandom().shuffle(password_list)
        password = "".join(password_list)

        last_error: Optional[Exception] = None
        for attempt, uname in enumerate([username, f"{safe_name[:34]}_admin_{secrets.token_hex(3)}"]):
            try:
                user = await admin_users.create_user(
                    username=uname,
                    password=password,
                    email=None,
                    full_name=f"{client_name} Admin",
                    roles=["client_admin"],
                    target_client_id=client_id,
                    request_id=request_id,
                    ctx=ctx,
                )
                # SECURITY: password is intentionally omitted from log extra fields
                logger.info(
                    "[AUTO-ADMIN] Initial client_admin user created successfully",
                    extra={
                        "client_id": client_id,
                        "username": uname,
                        "user_id": user.get("user_id"),
                        "request_id": request_id,
                        "attempt": attempt + 1,
                    },
                )
                # Return credentials — password shown only once, never logged
                return {
                    "username": uname,
                    "password": password,  # PLAINTEXT — one-time, caller must not persist
                    "user_id": user.get("user_id"),
                    "roles": ["client_admin"],
                }
            except ValueError as e:
                if attempt == 0:
                    # Username collision → retry with unique suffix
                    logger.warning(
                        f"[AUTO-ADMIN] Attempt {attempt + 1} failed (collision): {e}. Retrying.",
                        extra={"client_id": client_id, "request_id": request_id},
                    )
                    last_error = e
                    continue
                else:
                    last_error = e
                    break
            except Exception as e:
                last_error = e
                break

        # All attempts exhausted — raise so the caller can compensate
        raise ValueError(
            f"[AUTO-ADMIN] All attempts to create initial client_admin failed: {last_error}"
        ) from last_error

    async def _resolve_redis_strict(self):
        """
        Resolve Redis client from DI container (Pure DI - No Fallback).

        Resolution strategy:
        1. String key "system_redis_client" (explicit, stable)
        2. Type key aioredis.Redis (standard DI)

        Returns:
            Redis client instance or None if not available

        Note:
            This method does NOT fall back to EventBus internals.
            If Redis is required and not available, caller should fail-fast.
        """
        if not self.di_container:
            logger.warning(
                f"[{self.manifest.name}] DI container not available - "
                "cannot resolve Redis"
            )
            return None

        # Strategy 1: String key (preferred - explicit and stable)
        try:
            client = await self.di_container.resolve("system_redis_client")
            logger.debug(f"[{self.manifest.name}] Redis resolved via string key")
            return client
        except (ValueError, KeyError):
            pass

        # Strategy 2: Type key (fallback - standard DI)
        try:
            import redis.asyncio as aioredis

            client = await self.di_container.resolve(aioredis.Redis)
            logger.debug(f"[{self.manifest.name}] Redis resolved via type key")
            return client
        except (ValueError, KeyError):
            pass

        logger.debug(f"[{self.manifest.name}] Redis not registered in DI container")
        return None

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize client management module (Pure DI - No Fallback).

        Sets up:
        - Secret generator
        - Client management provider
        - Redis connection via DI container

        Returns:
            Initialization status with module info

        Raises:
            RuntimeError: If Redis not available via DI container
        """
        logger.info(
            f"Initializing {self.manifest.name} module",
            extra={"mod_name": self.manifest.name},
        )

        # Initialize secret generator
        secret_config = self.config.get("security", {}).get("secret_generation", {})
        length = secret_config.get("length", 32)
        charset = secret_config.get("charset", "alphanumeric")
        self.secret_generator = SecretGenerator(length=length, charset=charset)

        # ========================================================
        # REDIS RESOLUTION (Pure DI - No Fallback on EventBus)
        # ========================================================
        redis_client = await self._resolve_redis_strict()

        # Redis is REQUIRED for admin_clients - fail fast if not available
        if not redis_client:
            logger.error(
                f"[{self.manifest.name}] FATAL: Redis client not available via DI container. "
                "Ensure Redis is running and registered with key 'system_redis_client'."
            )
            raise RuntimeError(
                f"Redis client not available for {self.manifest.name} module. "
                "Check DI container registration."
            )

        logger.info(f"[{self.manifest.name}] Redis resolved via DI container")

        # Initialize client management provider
        self.client_provider = ClientManagementProvider(
            redis_client=redis_client,
            secret_generator=self.secret_generator,
            config=self.config,
        )

        self._initialized = True

        logger.info(f"✅ {self.manifest.name} initialized successfully")

        return {
            "status": "initialized",
            "module": self.manifest.name,
            "storage": "redis",
            "security": {
                "secret_generation": charset,
                "secret_length": length,
                "secret_hashing": "bcrypt",
                "bcrypt_rounds": self.config.get("security", {})
                .get("secret_hashing", {})
                .get("rounds", 12),
            },
            "features": {
                "client_name_index": "O(1) lookup",
                "event_bus": "enabled" if self.publisher else "disabled",
                "multi_tenancy": "supported",
                "secret_rotation": "enabled",
                "soft_revocation": "enabled",
            },
        }

    async def shutdown(self) -> None:
        """Shutdown and cleanup."""
        logger.info(f"Shutting down {self.manifest.name} module")
        self._initialized = False
        self.secret_generator = None
        self.client_provider = None
        logger.info(f"✅ {self.manifest.name} shutdown successfully")

    # ========================================================================
    # CRUD Operations (delegated to provider)
    # ========================================================================

    # ARCH-BRIDGE-001: public entry point for bridge sync called by admin_packs_router.
    # Accepts a request-scoped redis_client to avoid DI race on self.client_provider.
    async def sync_pipeline_to_tenant(
        self,
        client_id: str,
        pipeline_config: Dict[str, Any],
        redis_client: Any,
    ) -> None:
        """Sync pipeline_config fields to tenant_pipeline:{client_id} Redis HASH.

        Uses the caller-provided redis_client (request-scoped, already connected)
        instead of the provider's internal connection, preventing bridge sync failures
        caused by DI race conditions at startup.
        """
        await ClientManagementProvider.sync_pipeline_to_tenant_with_redis(
            client_id, pipeline_config, redis_client
        )

    async def create_client(
        self,
        client_name: str,
        client_type: str,
        description: Optional[str] = None,
        redirect_uris: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        is_active: bool = True,
        expires_at: Optional[str] = None,
        # Enterprise configuration v2.0
        kb_config: Optional[Dict[str, Any]] = None,
        ingestion_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        user_limits: Optional[Dict[str, Any]] = None,
        rag_config: Optional[Dict[str, Any]] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
        authorization: Optional[Dict[str, Any]] = None,
        settings_permissions: Optional[Dict[str, Any]] = None,
        # v6.6: Per-client pipeline/feature/rate configuration
        pipeline_config: Optional[Dict[str, Any]] = None,
        feature_flags: Optional[Dict[str, Any]] = None,
        rate_limits: Optional[Dict[str, Any]] = None,
        # PRESET-001: Domain-specific configuration
        domain_config: Optional[Dict[str, Any]] = None,
        # v17.19 W1: Per-client policy
        client_policy: Optional[Dict[str, Any]] = None,
        # System client protection
        is_system: bool = False,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Create a new client with enterprise configuration.

        SECURITY PATCH P0: Admin-only operation.
        Non-admin users cannot create clients.

        All business logic delegated to ClientManagementProvider.
        This method only handles UBP integration (event publishing, request tracking).
        """
        # SECURITY PATCH P0: Require admin for create_client
        ctx = self._require_admin(ctx, "create_client")

        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        caller_user_id = ctx.user.user_id
        caller_client_id = getattr(ctx.user, "client_id", None)

        logger.info(
            "Creating client",
            extra={
                "client_name": client_name,
                "client_type": client_type,
                "request_id": request_id,
                "scopes_count": len(scopes) if scopes else 0,
                "has_kb_config": kb_config is not None,
                "has_model_config": model_config is not None,
                "created_by_user_id": caller_user_id,
            },
        )

        # Delegate to provider
        client = await self.client_provider.create_client(
            client_name=client_name,
            client_type=client_type,
            description=description,
            redirect_uris=redirect_uris,
            scopes=scopes,
            tenant_id=tenant_id,
            is_active=is_active,
            expires_at=expires_at,
            # Enterprise config
            kb_config=kb_config,
            ingestion_config=ingestion_config,
            model_config=model_config,
            user_limits=user_limits,
            rag_config=rag_config,
            prompt_config=prompt_config,
            authorization=authorization,
            settings_permissions=settings_permissions,
            # v6.6
            pipeline_config=pipeline_config,
            feature_flags=feature_flags,
            rate_limits=rate_limits,
            # PRESET-001
            domain_config=domain_config,
            # v17.19 W1
            client_policy=client_policy,
            # System client protection
            is_system=is_system,
        )
        # GAP-001 only handled update_client(), but create_client() also needs it
        if kb_config:
            initial_kbs = kb_config.get("universal_kbs_assigned", [])
            if isinstance(initial_kbs, list) and initial_kbs:
                logger.info(
                    "[BUG-006] Syncing initial KB ACLs for new client",
                    extra={
                        "client_id": client["client_id"],
                        "kbs_count": len(initial_kbs),
                        "kbs": initial_kbs,
                        "request_id": request_id,
                    },
                )
                try:
                    await self._sync_client_kb_acl(
                        client_id=client["client_id"],
                        kbs_added=initial_kbs,
                        kbs_removed=[],
                        ctx=ctx,
                        request_id=request_id,
                    )
                except Exception as e:
                    # Don't fail client creation - ACL can be synced via update_client
                    logger.error(
                        "[BUG-006] ACL sync failed for new client, client created but ACL not set",
                        extra={
                            "client_id": client["client_id"],
                            "error": str(e),
                            "request_id": request_id,
                        },
                    )

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.created",
                {
                    "client_id": client["client_id"],
                    "client_name": client_name,
                    "client_type": client_type,
                    "scopes": scopes or [],
                    "tenant_id": tenant_id,
                    "has_kb_config": kb_config is not None,
                    "has_model_config": model_config is not None,
                    "timestamp": client["created_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        client["request_id"] = request_id

        # AUTO-CREATE INITIAL CLIENT ADMIN USER (TRANSACTIONAL)
        # The new client starts with bootstrap_status="pending" (set in provider).
        # If user creation succeeds → mark bootstrap_status="ok".
        # If user creation fails → mark bootstrap_status="failed" + is_active=False
        # so the client is not considered operational and admin is alerted.
        try:
            initial_admin = await self._create_initial_client_admin(
                client_id=client["client_id"],
                client_name=client_name,
                request_id=request_id,
                ctx=ctx,
            )
            # Mark client as fully bootstrapped
            await self.client_provider.set_bootstrap_status(
                client["client_id"], "ok"
            )
            client["bootstrap_status"] = "ok"
            # Include initial admin credentials in response (only shown once)
            client["initial_admin_user"] = initial_admin

        except Exception as bootstrap_err:
            # COMPENSATE: mark client inactive so it is not accidentally used
            logger.error(
                "[AUTO-ADMIN] Bootstrap failed — marking client inactive. "
                "Admin must fix bootstrap via the admin panel.",
                extra={
                    "client_id": client["client_id"],
                    "request_id": request_id,
                    "error": str(bootstrap_err),
                },
            )
            try:
                await self.client_provider.set_bootstrap_status(
                    client["client_id"], "failed", is_active=False
                )
            except Exception as status_err:
                logger.error(
                    "[AUTO-ADMIN] Could not update bootstrap_status after failure",
                    extra={"client_id": client["client_id"], "error": str(status_err)},
                )

            # Re-raise so the API layer returns an error and the operator is informed
            raise RuntimeError(
                f"Client '{client_name}' was created but initial admin user could not be "
                f"provisioned. The client has been set inactive (bootstrap_status=failed). "
                f"Reason: {bootstrap_err}"
            ) from bootstrap_err

        return client

    async def list_clients(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        ctx: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        List all clients with filtering and pagination.

        SECURITY PATCH P0: Admin-only operation.
        Non-admin users cannot list clients.

        Args:
            filter: Filter parameters
            limit: Max results
            offset: Skip results
            ctx: Request context with caller identity (REQUIRED)

        Returns:
            List of client objects

        Raises:
            PermissionError: If user is not admin
            ValueError: If ctx is invalid
        """
        # SECURITY PATCH P0: Require admin for list_clients
        ctx = self._require_admin(ctx, "list_clients")

        logger.debug(
            "Listing clients",
            extra={
                "filter": filter,
                "limit": limit,
                "offset": offset,
                "admin_user_id": ctx.user.user_id,
            },
        )

        return await self.client_provider.list_clients(
            filter_params=filter, limit=limit, offset=offset
        )

    async def get_client(self, client_id: str, ctx: Any = None) -> Dict[str, Any]:
        """
        Get client by ID.

        Admin can view any client; client_admin can view their own client only.

        Args:
            client_id: Client UUID
            ctx: Request context with caller identity (REQUIRED)

        Returns:
            Client object

        Raises:
            PermissionError: If user lacks sufficient privilege
            ValueError: If ctx is invalid
        """
        # Allow platform admin OR client_admin for their own client
        ctx = self._require_admin_or_client_admin(ctx, "get_client", client_id)

        logger.debug(
            "Getting client",
            extra={"client_id": client_id, "admin_user_id": ctx.user.user_id},
        )
        return await self.client_provider.get_client(client_id)

    async def get_client_by_name(self, client_name: str, ctx: Any = None) -> Optional[Dict[str, Any]]:
        """Get client by name via O(1) name index lookup. Admin only."""
        self._require_admin(ctx, "get_client_by_name")
        return await self.client_provider.get_client_by_name(client_name)

    async def get_client_internal(self, client_id: str) -> Dict[str, Any]:
        """
        Get client by ID - INTERNAL USE ONLY (without client_secret_hash).

        This method bypasses security checks for module-to-module calls.
        Safe for read-only operations where the hash is not needed.

        DO NOT expose this method via API routes.
        """
        logger.debug(
            "Getting client (internal)",
            extra={"client_id": client_id, "caller": "internal"},
        )
        return await self.client_provider.get_client(client_id)

    async def get_client_internal_raw(self, client_id: str) -> Dict[str, Any]:
        """
        Get client by ID - INTERNAL USE ONLY (full object, includes client_secret_hash).

        Use this ONLY when the caller needs to read-modify-write the full client
        back to Redis (e.g. _save_local_config). Using get_client() for that
        strips client_secret_hash and corrupts the stored object on write-back.

        DO NOT expose this method via API routes.
        """
        logger.debug(
            "Getting client (internal raw)",
            extra={"client_id": client_id, "caller": "internal_raw"},
        )
        return await self.client_provider._get_client_raw(client_id)

    async def update_client(
        self,
        client_id: str,
        client_name: Optional[str] = None,
        description: Optional[str] = None,
        redirect_uris: Optional[List[str]] = None,
        scopes: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        expires_at: Optional[str] = None,
        # Enterprise configuration v2.0
        kb_config: Optional[Dict[str, Any]] = None,
        ingestion_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        user_limits: Optional[Dict[str, Any]] = None,
        rag_config: Optional[Dict[str, Any]] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
        authorization: Optional[Dict[str, Any]] = None,
        settings_permissions: Optional[Dict[str, Any]] = None,
        # v6.6: Per-client pipeline/feature/rate configuration
        pipeline_config: Optional[Dict[str, Any]] = None,
        feature_flags: Optional[Dict[str, Any]] = None,
        rate_limits: Optional[Dict[str, Any]] = None,
        # PRESET-001: Domain-specific configuration
        domain_config: Optional[Dict[str, Any]] = None,
        # v17.19 W1: Per-client policy
        client_policy: Optional[Dict[str, Any]] = None,
        # Pack authorization + system protection
        pack_authorization: Optional[Dict[str, Any]] = None,
        is_system: Optional[bool] = None,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Update client information and enterprise configuration.

        Admin can update any client; client_admin can update their own client only.
        Note: client_admin cannot modify pipeline_config, feature_flags, rate_limits
        (those are platform-admin-only fields).

        GAP-001 FIX (Task #55):
        - When kb_config.universal_kbs_assigned changes, automatically
          sync ACL permissions via rag_orchestrator.set_permission()
        - This enables inheritance: all users of this client get access

        Delegated to ClientManagementProvider.
        Publishes event on success.
        """
        # Allow platform admin OR client_admin for their own client
        ctx = self._require_admin_or_client_admin(ctx, "update_client", client_id)

        # client_admin cannot modify platform-level configuration fields
        if self._is_client_admin(ctx) and not self._is_admin(ctx):
            for restricted_field in ("pipeline_config", "feature_flags", "rate_limits"):
                local_val = locals().get(restricted_field)
                if local_val is not None:
                    logger.warning(
                        f"[SECURITY] client_admin {ctx.user.user_id} attempted to modify "
                        f"restricted field '{restricted_field}'",
                        extra={"client_id": client_id, "user_id": ctx.user.user_id},
                    )
                    raise PermissionError(
                        f"client_admin cannot modify '{restricted_field}'. "
                        "Platform admin required."
                    )

        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        caller_user_id = ctx.user.user_id

        logger.info(
            "Updating client",
            extra={
                "client_id": client_id,
                "request_id": request_id,
                "updated_by_user_id": caller_user_id,
            },
        )

        # ====================================================================
        # GAP-001 FIX: Capture old KB assignments BEFORE update
        # ====================================================================
        old_kbs_assigned: List[str] = []
        if kb_config is not None and "universal_kbs_assigned" in kb_config:
            try:
                old_client = await self.client_provider.get_client(client_id)
                old_kb_config = old_client.get("kb_config") or {}
                old_kbs_assigned = old_kb_config.get("universal_kbs_assigned") or []
                if not isinstance(old_kbs_assigned, list):
                    old_kbs_assigned = []
            except Exception as e:
                logger.warning(
                    f"Could not fetch old client state for ACL sync: {e}",
                    extra={"client_id": client_id, "request_id": request_id},
                )

        # Delegate to provider
        client = await self.client_provider.update_client(
            client_id=client_id,
            client_name=client_name,
            description=description,
            redirect_uris=redirect_uris,
            scopes=scopes,
            tenant_id=tenant_id,
            is_active=is_active,
            expires_at=expires_at,
            # Enterprise config
            kb_config=kb_config,
            ingestion_config=ingestion_config,
            model_config=model_config,
            user_limits=user_limits,
            rag_config=rag_config,
            prompt_config=prompt_config,
            authorization=authorization,
            settings_permissions=settings_permissions,
            # v6.6
            pipeline_config=pipeline_config,
            feature_flags=feature_flags,
            rate_limits=rate_limits,
            # PRESET-001
            domain_config=domain_config,
            # v17.19 W1
            client_policy=client_policy,
            # Pack authorization + system protection
            pack_authorization=pack_authorization,
            is_system=is_system,
        )

        # ====================================================================
        # GAP-001 FIX: Sync ACL permissions for KB assignment changes
        # ====================================================================
        if kb_config is not None and "universal_kbs_assigned" in kb_config:
            new_kbs_assigned = kb_config.get("universal_kbs_assigned") or []
            if not isinstance(new_kbs_assigned, list):
                new_kbs_assigned = []

            # Calculate diff
            kbs_added = set(new_kbs_assigned) - set(old_kbs_assigned)
            kbs_removed = set(old_kbs_assigned) - set(new_kbs_assigned)

            if kbs_added or kbs_removed:
                await self._sync_client_kb_acl(
                    client_id=client_id,
                    kbs_added=list(kbs_added),
                    kbs_removed=list(kbs_removed),
                    ctx=ctx,
                    request_id=request_id,
                )

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.updated",
                {
                    "client_id": client_id,
                    "client_name": client.get("client_name"),
                    "changes": {
                        "client_name": client_name is not None,
                        "scopes": scopes is not None,
                        "is_active": is_active is not None,
                        "expires_at": expires_at is not None,
                        "kb_config": kb_config is not None,
                        "model_config": model_config is not None,
                        "user_limits": user_limits is not None,
                    },
                    "timestamp": client["updated_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        client["request_id"] = request_id
        return client

    async def delete_client(
        self,
        client_id: str,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, str]:
        """
        Delete a client.

        SECURITY PATCH P0: Admin-only operation.
        Non-admin users cannot delete clients.

        Delegated to ClientManagementProvider.
        Publishes event on success.
        """
        # SECURITY PATCH P0: Require admin for delete_client
        ctx = self._require_admin(ctx, "delete_client")

        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        caller_user_id = ctx.user.user_id

        logger.info(
            "Deleting client",
            extra={
                "client_id": client_id,
                "request_id": request_id,
                "deleted_by_user_id": caller_user_id,
            },
        )

        # Delegate to provider
        result = await self.client_provider.delete_client(client_id)

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.deleted",
                {
                    "client_id": client_id,
                    "client_name": result["client_name"],
                    "deleted_by_user_id": caller_user_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        result["request_id"] = request_id
        return result

    async def rotate_secret(
        self, client_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Rotate client secret.

        Delegated to ClientManagementProvider.
        Publishes event on success.
        """
        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Rotating client secret",
            extra={"client_id": client_id, "request_id": request_id},
        )

        # Delegate to provider
        result = await self.client_provider.rotate_secret(client_id)

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.secret_rotated",
                {
                    "client_id": client_id,
                    "rotated_at": result["rotated_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        result["request_id"] = request_id
        return result

    async def retry_client_bootstrap(
        self,
        client_id: str,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Retry initial admin user creation for a client whose bootstrap failed.

        Platform admin only.

        IDEMPOTENCY / CONCURRENCY GUARD:
        Only allowed when bootstrap_status == "failed".
        Before creating the user the status is atomically moved to
        "pending_retry".  A concurrent retry will read "pending_retry" and be
        rejected with a 409-style ValueError, preventing duplicate bootstrap
        users.  On success the status becomes "ok"; on failure it reverts to
        "failed" so the admin can try again.

        Returns the new admin credentials (password shown once).

        Raises:
            PermissionError: If caller is not admin
            ValueError: If client not found, bootstrap_status is not 'failed',
                        or a concurrent retry is already in progress
            RuntimeError: If admin user creation fails again
        """
        ctx = self._require_admin(ctx, "retry_client_bootstrap")

        if not request_id:
            request_id = str(uuid.uuid4())

        client = await self.client_provider.get_client(client_id)
        if not client:
            raise ValueError(f"Client not found: {client_id}")

        bootstrap_status = client.get("bootstrap_status", "ok")

        # Only retry from 'failed' state — guard against:
        #   - redundant retries on an already-working client (bootstrap_status=ok)
        #   - concurrent retries (bootstrap_status=pending_retry)
        if bootstrap_status == "ok":
            raise ValueError(
                f"Client {client_id} bootstrap_status is 'ok' — no retry needed."
            )
        if bootstrap_status == "pending_retry":
            raise ValueError(
                f"Client {client_id} bootstrap retry is already in progress "
                "(bootstrap_status=pending_retry). Wait for it to complete."
            )
        if bootstrap_status != "failed":
            raise ValueError(
                f"Client {client_id} bootstrap_status is '{bootstrap_status}'. "
                "Retry is only allowed from 'failed' state."
            )

        client_name = client["client_name"]

        # --- ATOMIC LOCK: transition failed → pending_retry ---
        # Any concurrent request will now see pending_retry and be rejected above.
        await self.client_provider.set_bootstrap_status(client_id, "pending_retry")

        logger.info(
            "[BOOTSTRAP-RETRY] Retrying bootstrap for client",
            extra={"client_id": client_id, "client_name": client_name, "request_id": request_id},
        )

        try:
            # Attempt to create admin user (will raise on failure)
            initial_admin = await self._create_initial_client_admin(
                client_id=client_id,
                client_name=client_name,
                request_id=request_id,
                ctx=ctx,
            )
        except Exception as create_err:
            # Revert lock so admin can try again
            try:
                await self.client_provider.set_bootstrap_status(client_id, "failed")
            except Exception as revert_err:
                logger.error(
                    "[BOOTSTRAP-RETRY] Could not revert pending_retry → failed",
                    extra={"client_id": client_id, "error": str(revert_err)},
                )
            raise RuntimeError(
                f"[BOOTSTRAP-RETRY] Admin user creation failed: {create_err}"
            ) from create_err

        # --- SUCCESS: mark client as fully bootstrapped and active ---
        await self.client_provider.set_bootstrap_status(
            client_id, "ok", is_active=True
        )

        logger.info(
            "[BOOTSTRAP-RETRY] Bootstrap retry succeeded",
            extra={"client_id": client_id, "request_id": request_id},
        )

        return {
            "client_id": client_id,
            "client_name": client_name,
            "bootstrap_status": "ok",
            "initial_admin_user": initial_admin,
            "request_id": request_id,
        }

    async def revoke_client(
        self,
        client_id: str,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Revoke client access (soft delete).

        Delegated to ClientManagementProvider.
        Publishes event on success.
        """
        # Generate request ID
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Revoking client",
            extra={"client_id": client_id, "reason": reason, "request_id": request_id},
        )

        # Delegate to provider
        result = await self.client_provider.revoke_client(client_id, reason)

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.revoked",
                {
                    "client_id": client_id,
                    "client_name": result["client_name"],
                    "reason": reason,
                    "revoked_at": result["revoked_at"],
                    "request_id": request_id,
                },
            )

        # Add request_id to response
        result["request_id"] = request_id
        return result

    # ========================================================================
    # ACL Synchronization (GAP-001 Fix - Task #55)
    # ========================================================================

    async def _sync_client_kb_acl(
        self,
        client_id: str,
        kbs_added: List[str],
        kbs_removed: List[str],
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Synchronize ACL permissions when client KB assignments change.

        GAP-001 FIX: This method ensures that when a KB is assigned to a client
        via kb_config.universal_kbs_assigned, the corresponding ACL entry is
        created in Redis (rag:acl:client:{client_id}:{kb_name}).

        This enables the inheritance model:
        - ACL on client level grants access to ALL users of that client
        - rag_orchestrator.check_access() checks client ACL if user ACL not found

        Args:
            client_id: The client ID
            kbs_added: List of KB names being added
            kbs_removed: List of KB names being removed
            ctx: Security context (passed to rag_orchestrator)
            request_id: Request tracking ID
        """
        if not self.di_container:
            logger.warning(
                "[GAP-001] DI container not available, cannot sync ACL",
                extra={"client_id": client_id, "request_id": request_id},
            )
            return

        # Resolve rag_orchestrator module
        rag_module = None
        try:
            rag_module = await self.di_container.resolve("rag_orchestrator")
        except Exception as e:
            logger.warning(
                f"[GAP-001] Could not resolve rag_orchestrator: {e}",
                extra={"client_id": client_id, "request_id": request_id},
            )
            return

        if not rag_module:
            logger.warning(
                "[GAP-001] rag_orchestrator not available, ACL sync skipped",
                extra={"client_id": client_id, "request_id": request_id},
            )
            return

        # Grant ACL for added KBs
        # BUG-006 FIX: universal_kbs_assigned contains dicts {"kb_name": str, "access_level": str}
        # NOT plain strings. Extract kb_name and access_level from each entry.
        for kb_entry in kbs_added:
            # Handle both dict format and legacy string format
            if isinstance(kb_entry, dict):
                kb_name = kb_entry.get("kb_name")
                access_level = kb_entry.get("access_level", "read")
            else:
                # Legacy string format fallback
                kb_name = kb_entry
                access_level = "read"

            if not kb_name:
                logger.warning(
                    "[GAP-001] Skipping invalid KB entry (no kb_name)",
                    extra={"kb_entry": kb_entry, "client_id": client_id},
                )
                continue

            # GAP-ACL-002: Check if KB exists in Qdrant before creating ACL
            # This prevents orphan ACL entries in Redis for non-existent collections
            kb_exists = await self._check_kb_exists_in_qdrant(kb_name, ctx)
            if not kb_exists:
                logger.warning(
                    f"[GAP-ACL-002] Skipping ACL for non-existent KB '{kb_name}' - "
                    "preventing orphan ACL entry in Redis",
                    extra={
                        "client_id": client_id,
                        "kb_name": kb_name,
                        "action": "skip_acl",
                        "reason": "kb_not_found_in_qdrant",
                        "request_id": request_id,
                    },
                )
                continue

            try:
                await rag_module.set_permission(
                    entity_type="client",
                    entity_id=client_id,
                    collection_id=kb_name,
                    access_level=access_level,
                    ctx=ctx,
                )
                logger.info(
                    f"[GAP-001] ACL granted: client={client_id}, kb={kb_name}, access={access_level}",
                    extra={
                        "client_id": client_id,
                        "kb_name": kb_name,
                        "access_level": access_level,
                        "action": "grant",
                        "request_id": request_id,
                    },
                )
            except Exception as e:
                logger.error(
                    f"[GAP-001] Failed to grant ACL for KB '{kb_name}': {e}",
                    extra={
                        "client_id": client_id,
                        "kb_name": kb_name,
                        "request_id": request_id,
                        "error": str(e),
                    },
                )

        # Revoke ACL for removed KBs
        # BUG-006 FIX: Same handling for removed KBs
        for kb_entry in kbs_removed:
            # Handle both dict format and legacy string format
            if isinstance(kb_entry, dict):
                kb_name = kb_entry.get("kb_name")
            else:
                kb_name = kb_entry

            if not kb_name:
                logger.warning(
                    "[GAP-001] Skipping invalid KB entry for removal (no kb_name)",
                    extra={"kb_entry": kb_entry, "client_id": client_id},
                )
                continue

            try:
                await rag_module.set_permission(
                    entity_type="client",
                    entity_id=client_id,
                    collection_id=kb_name,
                    access_level="none",
                    ctx=ctx,
                )
                logger.info(
                    f"[GAP-001] ACL revoked: client={client_id}, kb={kb_name}",
                    extra={
                        "client_id": client_id,
                        "kb_name": kb_name,
                        "action": "revoke",
                        "request_id": request_id,
                    },
                )
            except Exception as e:
                logger.error(
                    f"[GAP-001] Failed to revoke ACL for KB '{kb_name}': {e}",
                    extra={
                        "client_id": client_id,
                        "kb_name": kb_name,
                        "request_id": request_id,
                        "error": str(e),
                    },
                )

        # Publish event for ACL sync
        if self.publisher and (kbs_added or kbs_removed):
            await self.publisher.publish(
                "admin.client.kb_acl_synced",
                {
                    "client_id": client_id,
                    "kbs_granted": kbs_added,
                    "kbs_revoked": kbs_removed,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    async def get_client_stats(self) -> Dict[str, Any]:
        """
        Get client management statistics.

        Delegated to ClientManagementProvider.
        """
        stats = await self.client_provider.get_client_stats()

        return {
            "module": self.manifest.name,
            **stats,
            "storage": "redis",
            "security": {"secret_hashing": "bcrypt", "client_name_index": "O(1)"},
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for client management module.

        Tests Redis connectivity and module initialization status.
        """
        health = {
            "module": self.manifest.name,
            "status": "healthy",
            "initialized": self._initialized,
            "redis": {
                "connected": self.client_provider is not None,
                "status": "unknown",
            },
        }

        # Test Redis connection
        if self.client_provider and self.client_provider.redis_client:
            try:
                await self.client_provider.redis_client.ping()
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

    # ========================================================================
    # User Management for Clients (v2.0)
    # ========================================================================

    async def get_client_users(
        self,
        client_id: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        request_id: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """
        Get all users registered via this client.

        Admin can list users for any client; client_admin can list users
        for their own client only.

        Delegates to admin_users module instead of accessing Redis directly.
        """
        # Require admin or client_admin (scoped to own client)
        ctx = self._require_admin_or_client_admin(ctx, "get_client_users", client_id)

        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Getting client users",
            extra={"client_id": client_id, "request_id": request_id},
        )

        result: Dict[str, Any] = {
            "client_id": client_id,
            "users": [],
            "total": 0,
            "request_id": request_id,
        }

        if not self.di_container:
            logger.error(
                "DI container unavailable while fetching client users",
                extra={"client_id": client_id, "request_id": request_id},
            )
            result["error"] = "User management module unavailable"
            return result

        try:
            admin_users = await self.di_container.resolve("admin_users")
        except Exception as resolve_error:
            logger.error(
                f"Failed to resolve admin_users module: {resolve_error}",
                extra={"client_id": client_id, "request_id": request_id},
                exc_info=True,
            )
            result["error"] = "User management module unavailable"
            return result

        if not admin_users:
            logger.error(
                "admin_users module not registered",
                extra={"client_id": client_id, "request_id": request_id},
            )
            result["error"] = "User management module unavailable"
            return result

        if not ctx:
            logger.warning(
                "Security context required to fetch client users",
                extra={"client_id": client_id, "request_id": request_id},
            )
            result["error"] = ErrorMessages.INVALID_CONTEXT
            return result

        try:
            filter_params = dict(filter or {})
            filter_params["client_id"] = client_id

            users = await admin_users.list_users(
                filter=filter_params,
                limit=limit,
                offset=offset,
                ctx=ctx,
            )

            result["users"] = users
            result["total"] = len(users)
            return result
        except PermissionError:
            # Bubble up so API layer can return 403
            raise
        except Exception as e:
            logger.error(
                f"Failed to fetch users for client {client_id}: {e}",
                extra={"client_id": client_id, "request_id": request_id},
                exc_info=True,
            )
            result["error"] = str(e)
            return result

    async def test_client(
        self, client_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test client configuration by verifying all settings are operational.

        Delegated to ClientManagementProvider.
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Testing client configuration",
            extra={"client_id": client_id, "request_id": request_id},
        )

        result = await self.client_provider.test_client(client_id)

        # Publish event if test failed
        if self.publisher and result["overall_status"] == "fail":
            await self.publisher.publish(
                "admin.client.test_failed",
                {
                    "client_id": client_id,
                    "client_name": result.get("client_name"),
                    "overall_status": result["overall_status"],
                    "tested_at": result["tested_at"],
                    "request_id": request_id,
                },
            )

        result["request_id"] = request_id
        return result

    # ========================================================================
    # User Count Management (v2.0)
    # ========================================================================

    async def increment_user_count(
        self, client_id: str, user_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Increment user count for a client.

        Called when a new user registers via this client.
        Delegated to ClientManagementProvider.
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Incrementing user count",
            extra={
                "client_id": client_id,
                "user_id": user_id,
                "request_id": request_id,
            },
        )

        result = await self.client_provider.increment_user_count(client_id, user_id)

        # Check if limit is nearly reached (90%)
        user_limits = result.get("user_limits", {})
        max_users = user_limits.get("max_users", 100)
        current_users = user_limits.get("current_users", 0)

        if current_users >= max_users:
            # Publish event for user limit reached
            if self.publisher:
                await self.publisher.publish(
                    "admin.client.user_limit_reached",
                    {
                        "client_id": client_id,
                        "max_users": max_users,
                        "current_users": current_users,
                        "timestamp": datetime.utcnow().isoformat(),
                        "request_id": request_id,
                    },
                )

        result["request_id"] = request_id
        return result

    async def decrement_user_count(
        self, client_id: str, user_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decrement user count for a client.

        Called when a user is deleted or deactivated.
        Delegated to ClientManagementProvider.
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.info(
            "Decrementing user count",
            extra={
                "client_id": client_id,
                "user_id": user_id,
                "request_id": request_id,
            },
        )

        result = await self.client_provider.decrement_user_count(client_id, user_id)

        result["request_id"] = request_id
        return result

    async def check_user_limit(
        self, client_id: str, request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if client has available user slots.

        Delegated to ClientManagementProvider.
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.debug(
            "Checking user limit",
            extra={"client_id": client_id, "request_id": request_id},
        )

        result = await self.client_provider.check_user_limit(client_id)

        result["request_id"] = request_id
        return result

    async def ingest_to_client_kb(
        self,
        client_id: str,
        collection_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        request_id: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document to a client's knowledge base with authorization checks.

        This endpoint allows clients to ingest documents to their assigned KBs.
        Fixes GAP-INGEST-001: Client-scoped document ingestion.

        Security:
        - Requires authenticated context
        - Verifies user belongs to the client
        - Checks client has write access to the collection
        - Respects ingestion_config limits

        Args:
            client_id: Client ID performing the ingestion
            collection_id: Target knowledge base
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

        user_id = getattr(ctx.user, "user_id", None)
        user_client_id = getattr(ctx.user, "client_id", None)
        user_roles = getattr(ctx.user, "roles", []) or []

        # ====================================================================
        # BUG-CLIENT-002 FIX: Allow admin users to ingest to any client
        # Admin users (platform admins) can manage any client's KB
        # Client admins can only manage their own client's KB
        # ====================================================================
        is_admin = "admin" in user_roles

        # Verify authorization: admin can access any, others need matching client_id
        if not is_admin and user_client_id != client_id:
            logger.warning(
                "Unauthorized client ingest attempt",
                extra={
                    "user_id": user_id,
                    "user_client_id": user_client_id,
                    "requested_client_id": client_id,
                    "is_admin": is_admin,
                    "request_id": request_id,
                },
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": "User does not belong to this client",
                "request_id": request_id,
            }

        # Log admin cross-client action for audit
        if is_admin and user_client_id != client_id:
            logger.info(
                "[AUDIT] Admin ingesting to different client's KB",
                extra={
                    "admin_user_id": user_id,
                    "admin_client_id": user_client_id,
                    "target_client_id": client_id,
                    "request_id": request_id,
                },
            )

        # Get client configuration
        try:
            client_data = await self.client_provider.get_client(client_id)
            if not client_data:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": f"Client {client_id} not found",
                    "request_id": request_id,
                }

            # Check if client is active
            if not client_data.get("is_active", False):
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": "Client is not active",
                    "request_id": request_id,
                }

            # Get ingestion config
            ingestion_config = client_data.get("ingestion_config", {})
            if not ingestion_config.get("enabled", True):
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": "Ingestion is disabled for this client",
                    "request_id": request_id,
                }

            # Check document size limit
            max_size_mb = ingestion_config.get("max_document_size_mb", 10)
            text_size_mb = len(text.encode("utf-8")) / (1024 * 1024)
            if text_size_mb > max_size_mb:
                return {
                    "document_id": "",
                    "chunks_count": 0,
                    "status": "error",
                    "message": f"Document size ({text_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)",
                    "request_id": request_id,
                }

        except Exception as e:
            logger.error(
                f"Error checking client configuration: {e}",
                extra={"client_id": client_id, "request_id": request_id},
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": f"Error checking client configuration: {str(e)}",
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

            # Add client tracking to metadata
            metadata["client_id"] = client_id
            metadata["uploader_id"] = user_id
            metadata["ingestion_source"] = "client_endpoint"

            # Call the authorized ingest endpoint
            result = await rag_module.ingest_document_authorized(
                collection_id=collection_id,
                text=text,
                metadata=metadata,
                ctx=ctx,
                tags=tags,
            )

            # Add request_id to result
            result["request_id"] = request_id

            logger.info(
                "Client document ingestion completed",
                extra={
                    "client_id": client_id,
                    "collection_id": collection_id,
                    "status": result.get("status"),
                    "document_id": result.get("document_id"),
                    "request_id": request_id,
                },
            )

            # Ensure KB ownership metadata exists for implicit client KBs
            try:
                metadata_record = (
                    await rag_module.get_collection_metadata_internal(collection_id)
                    if hasattr(rag_module, "get_collection_metadata_internal")
                    else None
                ) or {}

                if not metadata_record.get("owner_client_id"):
                    logger.info(
                        "Setting owner metadata for client KB",
                        extra={
                            "collection_id": collection_id,
                            "owner_client_id": client_id,
                            "request_id": request_id,
                        },
                    )
                    metadata_record["owner_client_id"] = client_id
                    if hasattr(rag_module, "update_collection_metadata_internal"):
                        await rag_module.update_collection_metadata_internal(
                            collection_id, metadata_record
                        )
            except Exception as metadata_error:
                logger.warning(
                    "Failed to set KB ownership metadata",
                    extra={
                        "collection_id": collection_id,
                        "error": str(metadata_error),
                        "request_id": request_id,
                    },
                )

            return result

        except Exception as e:
            logger.error(
                f"Error during client ingestion: {e}",
                extra={"client_id": client_id, "request_id": request_id},
            )
            return {
                "document_id": "",
                "chunks_count": 0,
                "status": "error",
                "message": f"Error during ingestion: {str(e)}",
                "request_id": request_id,
            }

    # ========================================================================
    # GAP-003 / GAP-004: Client KB-to-User Assignment (Task #55)
    # ========================================================================

    async def _check_kb_exists_in_qdrant(
        self,
        kb_name: str,
        ctx: Any = None,
    ) -> bool:
        """
        GAP-VAL-001: Check if a KB (collection) exists in Qdrant.

        This helper is used to distinguish between:
        - 404: KB does not exist in Qdrant
        - 403: KB exists but client is not authorized

        Strategy: Fail-Open
        - If Qdrant is unreachable or verification fails, returns True
        - This prevents blocking critical admin operations when Qdrant is down
        - A warning is logged for monitoring

        Args:
            kb_name: The knowledge base name to check
            ctx: Security context (passed for potential future use)

        Returns:
            True if KB exists (or cannot be verified), False if confirmed non-existent
        """
        if not self.di_container:
            logger.warning(
                "[GAP-VAL-001] DI container not available, assuming KB exists (fail-open)",
                extra={"kb_name": kb_name},
            )
            return True

        try:
            rag_module = await self.di_container.resolve("rag_orchestrator")
            if not rag_module:
                logger.warning(
                    "[GAP-VAL-001] rag_orchestrator not available, assuming KB exists (fail-open)",
                    extra={"kb_name": kb_name},
                )
                return True

            # Try to access qdrant_module from rag_orchestrator
            qdrant_module = getattr(rag_module, "qdrant_module", None)
            if not qdrant_module:
                # Fallback: try to resolve rag_qdrant directly
                try:
                    qdrant_module = await self.di_container.resolve("rag_qdrant")
                except Exception:
                    pass

            if not qdrant_module:
                logger.warning(
                    "[GAP-VAL-001] qdrant_module not available, assuming KB exists (fail-open)",
                    extra={"kb_name": kb_name},
                )
                return True

            # Strategy 1: qdrant_module.provider.qdrant_client.collection_exists()
            # This is the correct path: rag_qdrant adapter -> provider -> qdrant_client
            provider = getattr(qdrant_module, "provider", None)
            if provider:
                qdrant_client = getattr(provider, "qdrant_client", None)
                if qdrant_client and hasattr(qdrant_client, "collection_exists"):
                    exists = await qdrant_client.collection_exists(kb_name)
                    logger.debug(
                        f"[GAP-VAL-001] KB existence check via qdrant_client: {kb_name} -> {exists}",
                        extra={"kb_name": kb_name, "exists": exists},
                    )
                    return exists

            # Strategy 2: Direct method on qdrant_module (if exposed)
            if hasattr(qdrant_module, "collection_exists"):
                exists = await qdrant_module.collection_exists(kb_name)
                logger.debug(
                    f"[GAP-VAL-001] KB existence check via qdrant_module: {kb_name} -> {exists}",
                    extra={"kb_name": kb_name, "exists": exists},
                )
                return exists

            # Strategy 3: Try list_collections and check if kb_name is in the list
            if hasattr(qdrant_module, "list_collections"):
                try:
                    collections = await qdrant_module.list_collections()
                    # list_collections returns List[str]
                    if isinstance(collections, list):
                        exists = kb_name in collections
                        logger.debug(
                            f"[GAP-VAL-001] KB existence check via list_collections: {kb_name} -> {exists}",
                            extra={"kb_name": kb_name, "exists": exists},
                        )
                        return exists
                except Exception as e:
                    logger.warning(
                        f"[GAP-VAL-001] list_collections failed: {e}",
                        extra={"kb_name": kb_name},
                    )

            # Cannot verify - fail-open
            logger.warning(
                "[GAP-VAL-001] No method to verify KB existence, assuming exists (fail-open)",
                extra={"kb_name": kb_name},
            )
            return True

        except Exception as e:
            logger.warning(
                f"[GAP-VAL-001] Error checking KB existence: {e}, assuming exists (fail-open)",
                extra={"kb_name": kb_name, "error": str(e)},
            )
            return True

    async def _get_kb_metadata(self, kb_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve KB metadata stored in Redis, if available."""
        redis_client = None
        if self.client_provider and hasattr(self.client_provider, "redis_client"):
            redis_client = getattr(self.client_provider, "redis_client", None)
        if not redis_client:
            return None

        metadata_key = f"rag:kb:{kb_name}:metadata"
        try:
            raw_metadata = await redis_client.get(metadata_key)
            if not raw_metadata:
                return None
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            return json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid KB metadata format",
                extra={"kb_name": kb_name, "error": str(exc)},
            )
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Unable to fetch KB metadata",
                extra={"kb_name": kb_name, "error": str(exc)},
            )
            return None

    async def _validate_client_kb_user_operation(
        self,
        client_id: str,
        user_id: str,
        kb_name: str,
        ctx: Any,
        operation: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate preconditions for KB assignment/revocation operations.

        GAP-003/004 Security Validation:
        1. Validate KB name format (SEC-HIGH-002 fix)
        2. Caller must be admin OR belong to the same client
        3. Target user must belong to the client
        4. KB must be accessible to the client (in universal_kbs_assigned OR owned by client)

        Args:
            client_id: Target client ID
            user_id: Target user ID
            kb_name: KB to assign/revoke
            ctx: Security context
            operation: "assign" or "revoke"
            request_id: Request tracking ID

        Returns:
            Dict with:
                - valid: bool (True if all checks pass)
                - error: str (error message if invalid)
                - client_data: Dict (client data if valid)
                - access_level: str (default access level for this KB)
        """
        # SEC-HIGH-002 FIX: Validate KB name format to prevent injection
        if not kb_name or not KB_NAME_PATTERN.match(kb_name):
            logger.warning(
                f"[GAP-003/004] Invalid KB name format: {kb_name[:20] if kb_name else 'None'}...",
                extra={
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": False,
                "error": ErrorMessages.INVALID_KB_NAME.format(
                    kb_name=kb_name or "empty"
                ),
                "client_data": None,
                "access_level": None,
            }

        # Security context validation
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            return {
                "valid": False,
                "error": ErrorMessages.INVALID_CONTEXT,
                "client_data": None,
                "access_level": None,
            }

        caller_user_id = getattr(ctx.user, "user_id", None)
        caller_client_id = getattr(ctx.user, "client_id", None)
        caller_is_admin = self._is_admin(ctx)

        # Check 1: Caller authorization
        # Admin can operate on any client
        # Non-admin can only operate on their own client
        if not caller_is_admin:
            if caller_client_id != client_id:
                logger.warning(
                    f"[GAP-003/004] Unauthorized {operation} attempt: caller client mismatch",
                    extra={
                        "caller_user_id": caller_user_id,
                        "caller_client_id": caller_client_id,
                        "target_client_id": client_id,
                        "operation": operation,
                        "request_id": request_id,
                    },
                )
                return {
                    "valid": False,
                    "error": f"Access denied: you can only {operation} KB access for users in your own client",
                    "client_data": None,
                    "access_level": None,
                }

        # Check 2: Get client data
        try:
            client_data = await self.client_provider.get_client(client_id)
        except Exception as e:
            logger.warning(
                f"[GAP-003/004] Client not found for {operation}",
                extra={
                    "client_id": client_id,
                    "error": str(e),
                    "request_id": request_id,
                },
            )
            return {
                "valid": False,
                "error": ErrorMessages.CLIENT_NOT_FOUND.format(client_id=client_id),
                "client_data": None,
                "access_level": None,
            }

        if not client_data:
            return {
                "valid": False,
                "error": ErrorMessages.CLIENT_NOT_FOUND.format(client_id=client_id),
                "client_data": None,
                "access_level": None,
            }

        # Check 3: Verify user belongs to this client
        # Get user from admin_users module via DI
        user_belongs_to_client = False
        if self.di_container:
            try:
                admin_users = await self.di_container.resolve("admin_users")
                if admin_users:
                    user_data = await admin_users.get_user_internal(user_id)
                    if user_data and user_data.get("client_id") == client_id:
                        user_belongs_to_client = True
            except Exception as e:
                logger.warning(
                    f"[GAP-003/004] Could not verify user client membership: {e}",
                    extra={
                        "user_id": user_id,
                        "client_id": client_id,
                        "request_id": request_id,
                    },
                )
                # Fall through - will fail below

        if not user_belongs_to_client:
            logger.warning(
                f"[GAP-003/004] User does not belong to client",
                extra={
                    "user_id": user_id,
                    "client_id": client_id,
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": False,
                "error": f"User {user_id} does not belong to client {client_id}",
                "client_data": None,
                "access_level": None,
            }

        # Check 4: Verify KB accessibility (GAP-VAL-001: 404 vs 403 distinction)
        kb_config = client_data.get("kb_config", {}) or {}
        universal_kbs = kb_config.get("universal_kbs_assigned", []) or []
        default_user_kb_config = kb_config.get("default_user_kb_config", {}) or {}
        default_access_level = default_user_kb_config.get(
            "universal_kb_access_level", "read"
        )
        client_prefix_base = f"client_{client_id[:8]}" if client_id else None
        implicit_prefixes: Tuple[str, ...] = ()
        if client_prefix_base:
            implicit_prefixes = (
                f"{client_prefix_base}_",
                client_prefix_base,
            )

        # FIX: Implicit ownership - client-prefixed KBs are always authorized
        if implicit_prefixes and any(
            kb_name.startswith(prefix) for prefix in implicit_prefixes
        ):
            logger.info(
                "Allowing implicit client-owned KB access",
                extra={
                    "client_id": client_id,
                    "kb_name": kb_name,
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": True,
                "error": None,
                "client_data": client_data,
                "access_level": default_access_level,
            }

        # Metadata-based ownership: trust KB metadata if present
        try:
            metadata = await self._get_kb_metadata(kb_name)
        except Exception:
            metadata = None

        if metadata and metadata.get("owner_client_id") == client_id:
            logger.info(
                "Allowing KB access via metadata ownership",
                extra={
                    "client_id": client_id,
                    "kb_name": kb_name,
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": True,
                "error": None,
                "client_data": client_data,
                "access_level": default_access_level,
            }

        # Step 4a: Check if KB exists in Qdrant (GAP-VAL-001)
        # This distinguishes between "not found" (404) and "not authorized" (403)
        kb_exists = await self._check_kb_exists_in_qdrant(kb_name, ctx)

        if not kb_exists:
            logger.warning(
                f"[GAP-VAL-001] KB not found in Qdrant for {operation}",
                extra={
                    "kb_name": kb_name,
                    "client_id": client_id,
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": False,
                "error": f"Knowledge base '{kb_name}' not found",
                "error_code": 404,
                "client_data": None,
                "access_level": None,
            }

        # Step 4b: KB exists - now check if client has access
        # KB is accessible if it's in client's universal_kbs_assigned or implicitly owned
        kb_is_accessible = (kb_name in universal_kbs) or (
            implicit_prefixes
            and any(kb_name.startswith(prefix) for prefix in implicit_prefixes)
        )

        if not kb_is_accessible:
            logger.warning(
                f"[GAP-VAL-001] KB exists but client not authorized for {operation}",
                extra={
                    "kb_name": kb_name,
                    "client_id": client_id,
                    "universal_kbs": universal_kbs,
                    "operation": operation,
                    "request_id": request_id,
                },
            )
            return {
                "valid": False,
                "error": f"Access denied: client is not authorized to access knowledge base '{kb_name}'",
                "error_code": 403,
                "client_data": None,
                "access_level": None,
            }

        return {
            "valid": True,
            "error": None,
            "client_data": client_data,
            "access_level": default_access_level,
        }

    async def assign_kb_to_user(
        self,
        client_id: str,
        user_id: str,
        kb_name: str,
        access_level: Optional[str] = None,
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assign a KB to a specific user with custom access level.

        GAP-003 Implementation:
        Allows clients to grant KB access to individual users, overriding
        the default inheritance from client-level ACL.

        Security Requirements:
        - Caller must be admin OR belong to the same client
        - Target user must belong to the client
        - KB must be in client's universal_kbs_assigned OR owned by client

        Args:
            client_id: Client ID owning the user
            user_id: User ID to grant access
            kb_name: Knowledge base name
            access_level: Access level (read, write). Defaults to client's default.
            ctx: Security context (REQUIRED)
            request_id: Request tracking ID

        Returns:
            Dict with assignment result

        Raises:
            PermissionError: If caller lacks authorization
            ValueError: If validation fails
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        # GAP-AUDIT-001: Extract admin identity from context for audit trail
        admin_user_id = None
        admin_client_id = None
        if ctx and hasattr(ctx, "user") and ctx.user:
            admin_user_id = getattr(ctx.user, "user_id", None)
            admin_client_id = getattr(ctx.user, "client_id", None)

        # GAP-AUDIT-001: Use warning level to ensure visibility in default logging config
        logger.warning(
            f"[AUDIT][GAP-003] KB assignment initiated | audit_type=KB_ACCESS_CHANGE action=ASSIGN "
            f"admin_user_id={admin_user_id} target_client_id={client_id} target_user_id={user_id} "
            f"kb_name={kb_name} access_level={access_level} request_id={request_id}",
        )

        # Validate preconditions
        validation = await self._validate_client_kb_user_operation(
            client_id=client_id,
            user_id=user_id,
            kb_name=kb_name,
            ctx=ctx,
            operation="assign",
            request_id=request_id,
        )

        if not validation["valid"]:
            # GAP-VAL-001: Propagate error_code (404/403) to response
            response = {
                "success": False,
                "error": validation["error"],
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }
            if "error_code" in validation:
                response["error_code"] = validation["error_code"]
            return response

        client_data = validation["client_data"]

        # Use provided access level or default
        final_access_level = access_level or validation["access_level"]

        # Validate access level
        if not AccessLevel.is_valid(final_access_level):
            return {
                "success": False,
                "error": ErrorMessages.INVALID_ACCESS_LEVEL.format(
                    level=final_access_level
                ),
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Create ACL entry via rag_orchestrator
        rag_module = None
        if self.di_container:
            try:
                rag_module = await self.di_container.resolve("rag_orchestrator")
            except Exception as e:
                logger.error(
                    f"[GAP-003] Failed to resolve rag_orchestrator: {e}",
                    extra={"request_id": request_id},
                )

        if not rag_module:
            return {
                "success": False,
                "error": "RAG orchestrator not available",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Set user-level ACL
        acl_set = False
        try:
            await rag_module.set_permission_internal(
                entity_type=EntityType.USER.value,
                entity_id=user_id,
                collection_id=kb_name,
                access_level=final_access_level,
            )
            acl_set = True
        except Exception as e:
            logger.error(
                f"[GAP-003] Failed to set ACL: {e}",
                extra={
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "request_id": request_id,
                },
            )
            return {
                "success": False,
                "error": "Failed to set permission. Please try again.",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Update custom_user_kb_assignments in client config
        kb_config = client_data.get("kb_config", {}) or {}
        custom_assignments = kb_config.get("custom_user_kb_assignments", {}) or {}

        # Structure: { "user_id": { "kb_name": "access_level" } }
        if user_id not in custom_assignments:
            custom_assignments[user_id] = {}
        custom_assignments[user_id][kb_name] = final_access_level

        # Update client with new custom_user_kb_assignments
        kb_config["custom_user_kb_assignments"] = custom_assignments

        try:
            await self.client_provider.update_client(
                client_id=client_id,
                kb_config=kb_config,
            )
        except Exception as e:
            # SEC-HIGH-001 FIX: Rollback ACL if config update fails
            logger.error(
                f"[GAP-003] Config update failed, rolling back ACL: {e}",
                extra={
                    "client_id": client_id,
                    "request_id": request_id,
                },
            )
            if acl_set:
                try:
                    await rag_module.set_permission_internal(
                        entity_type=EntityType.USER.value,
                        entity_id=user_id,
                        collection_id=kb_name,
                        access_level=AccessLevel.NONE.value,
                    )
                    logger.info(
                        "[GAP-003] ACL rolled back successfully after config failure",
                        extra={
                            "user_id": user_id,
                            "kb_name": kb_name,
                            "request_id": request_id,
                        },
                    )
                except Exception as rollback_error:
                    logger.error(
                        f"[GAP-003] CRITICAL: ACL rollback also failed: {rollback_error}",
                        extra={
                            "user_id": user_id,
                            "kb_name": kb_name,
                            "request_id": request_id,
                        },
                    )
            return {
                "success": False,
                "error": "Failed to update configuration. Operation rolled back.",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.user_kb_assigned",
                {
                    "client_id": client_id,
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "access_level": final_access_level,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        # GAP-AUDIT-001: Enhanced success audit log (warning level for visibility)
        logger.warning(
            f"[AUDIT][GAP-003] KB successfully assigned | audit_type=KB_ACCESS_CHANGE action=ASSIGN "
            f"status=SUCCESS admin_user_id={admin_user_id} target_client_id={client_id} "
            f"target_user_id={user_id} kb_name={kb_name} access_level={final_access_level} "
            f"timestamp={datetime.utcnow().isoformat()} request_id={request_id}",
        )

        return {
            "success": True,
            "client_id": client_id,
            "user_id": user_id,
            "kb_name": kb_name,
            "access_level": final_access_level,
            "request_id": request_id,
        }

    async def revoke_kb_from_user(
        self,
        client_id: str,
        user_id: str,
        kb_name: str,
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Revoke a user's custom KB access, reverting to client-level inheritance.

        GAP-004 Implementation:
        Allows clients to remove custom KB access from individual users.
        After revocation, the user's access reverts to client-level ACL.

        Security Requirements:
        - Caller must be admin OR belong to the same client
        - Target user must belong to the client
        - KB must be in client's universal_kbs_assigned OR owned by client

        Args:
            client_id: Client ID owning the user
            user_id: User ID to revoke access
            kb_name: Knowledge base name
            ctx: Security context (REQUIRED)
            request_id: Request tracking ID

        Returns:
            Dict with revocation result

        Raises:
            PermissionError: If caller lacks authorization
            ValueError: If validation fails
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        # GAP-AUDIT-001: Extract admin identity from context for audit trail
        admin_user_id = None
        admin_client_id = None
        if ctx and hasattr(ctx, "user") and ctx.user:
            admin_user_id = getattr(ctx.user, "user_id", None)
            admin_client_id = getattr(ctx.user, "client_id", None)

        # GAP-AUDIT-001: Use warning level for visibility
        logger.warning(
            f"[AUDIT][GAP-004] KB revocation initiated | audit_type=KB_ACCESS_CHANGE action=REVOKE "
            f"admin_user_id={admin_user_id} target_client_id={client_id} target_user_id={user_id} "
            f"kb_name={kb_name} request_id={request_id}",
        )

        # Validate preconditions
        validation = await self._validate_client_kb_user_operation(
            client_id=client_id,
            user_id=user_id,
            kb_name=kb_name,
            ctx=ctx,
            operation="revoke",
            request_id=request_id,
        )

        if not validation["valid"]:
            # GAP-VAL-001: Propagate error_code (404/403) to response
            response = {
                "success": False,
                "error": validation["error"],
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }
            if "error_code" in validation:
                response["error_code"] = validation["error_code"]
            return response

        client_data = validation["client_data"]

        # Remove ACL entry via rag_orchestrator (set to "none")
        rag_module = None
        if self.di_container:
            try:
                rag_module = await self.di_container.resolve("rag_orchestrator")
            except Exception as e:
                logger.error(
                    f"[GAP-004] Failed to resolve rag_orchestrator: {e}",
                    extra={"request_id": request_id},
                )

        if not rag_module:
            return {
                "success": False,
                "error": "RAG orchestrator not available",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Get current assignment for potential rollback
        kb_config = client_data.get("kb_config", {}) or {}
        custom_assignments = kb_config.get("custom_user_kb_assignments", {}) or {}
        previous_access_level = None
        if user_id in custom_assignments and kb_name in custom_assignments[user_id]:
            previous_access_level = custom_assignments[user_id][kb_name]

        # Remove user-level ACL (set to "none" - will fallback to client ACL)
        acl_removed = False
        try:
            await rag_module.set_permission_internal(
                entity_type=EntityType.USER.value,
                entity_id=user_id,
                collection_id=kb_name,
                access_level=AccessLevel.NONE.value,
            )
            acl_removed = True
        except Exception as e:
            logger.error(
                f"[GAP-004] Failed to remove ACL: {e}",
                extra={
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "request_id": request_id,
                },
            )
            return {
                "success": False,
                "error": "Failed to remove permission. Please try again.",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Update custom_user_kb_assignments in client config
        # Remove the assignment if it exists
        if user_id in custom_assignments and kb_name in custom_assignments[user_id]:
            del custom_assignments[user_id][kb_name]
            # Clean up empty user entry
            if not custom_assignments[user_id]:
                del custom_assignments[user_id]

        # Update client with new custom_user_kb_assignments
        kb_config["custom_user_kb_assignments"] = custom_assignments

        try:
            await self.client_provider.update_client(
                client_id=client_id,
                kb_config=kb_config,
            )
        except Exception as e:
            # SEC-HIGH-001 FIX: Rollback ACL if config update fails
            logger.error(
                f"[GAP-004] Config update failed, rolling back ACL removal: {e}",
                extra={
                    "client_id": client_id,
                    "request_id": request_id,
                },
            )
            if acl_removed and previous_access_level:
                try:
                    await rag_module.set_permission_internal(
                        entity_type=EntityType.USER.value,
                        entity_id=user_id,
                        collection_id=kb_name,
                        access_level=previous_access_level,
                    )
                    logger.info(
                        "[GAP-004] ACL rolled back successfully after config failure",
                        extra={
                            "user_id": user_id,
                            "kb_name": kb_name,
                            "request_id": request_id,
                        },
                    )
                except Exception as rollback_error:
                    logger.error(
                        f"[GAP-004] CRITICAL: ACL rollback also failed: {rollback_error}",
                        extra={
                            "user_id": user_id,
                            "kb_name": kb_name,
                            "request_id": request_id,
                        },
                    )
            return {
                "success": False,
                "error": "Failed to update configuration. Operation rolled back.",
                "client_id": client_id,
                "user_id": user_id,
                "kb_name": kb_name,
                "request_id": request_id,
            }

        # Publish event
        if self.publisher:
            await self.publisher.publish(
                "admin.client.user_kb_revoked",
                {
                    "client_id": client_id,
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )

        # GAP-AUDIT-001: Enhanced success audit log (warning level for visibility)
        logger.warning(
            f"[AUDIT][GAP-004] KB successfully revoked | audit_type=KB_ACCESS_CHANGE action=REVOKE "
            f"status=SUCCESS admin_user_id={admin_user_id} target_client_id={client_id} "
            f"target_user_id={user_id} kb_name={kb_name} previous_access_level={previous_access_level} "
            f"timestamp={datetime.utcnow().isoformat()} request_id={request_id}",
        )

        return {
            "success": True,
            "client_id": client_id,
            "user_id": user_id,
            "kb_name": kb_name,
            "reverted_to": "client_level_acl",
            "request_id": request_id,
        }

    # ========================================================================
    # GAP-BULK-001: Bulk KB Operations
    # ========================================================================

    async def bulk_assign_kb_to_users(
        self,
        client_id: str,
        assignments: List[Dict[str, Any]],
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        GAP-BULK-001: Assign a KB to multiple users in a single operation.

        This method iterates over the assignments list and calls assign_kb_to_user
        for each entry, aggregating results.

        Args:
            client_id: Client ID owning the users
            assignments: List of assignment dicts, each containing:
                - user_id: User to assign KB to
                - kb_name: Knowledge base name
                - access_level: Optional access level (default: "read")
            ctx: Security context (REQUIRED)
            request_id: Request tracking ID

        Returns:
            Dict with aggregated results:
                - success: True if all assignments succeeded
                - success_count: Number of successful assignments
                - failure_count: Number of failed assignments
                - results: List of individual assignment results
                - request_id: Request tracking ID

        Example:
            await adapter.bulk_assign_kb_to_users(
                client_id="client-123",
                assignments=[
                    {"user_id": "user-1", "kb_name": "kb-1", "access_level": "read"},
                    {"user_id": "user-2", "kb_name": "kb-1", "access_level": "write"},
                    {"user_id": "user-3", "kb_name": "kb-1"},  # defaults to "read"
                ],
                ctx=ctx,
            )
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.warning(
            f"[AUDIT][GAP-BULK-001] Bulk KB assignment initiated | "
            f"client_id={client_id} assignment_count={len(assignments)} request_id={request_id}",
        )

        if not assignments:
            return {
                "success": True,
                "success_count": 0,
                "failure_count": 0,
                "results": [],
                "message": "No assignments provided",
                "request_id": request_id,
            }

        results = []
        success_count = 0
        failure_count = 0

        for assignment in assignments:
            user_id = assignment.get("user_id")
            kb_name = assignment.get("kb_name")
            access_level = assignment.get("access_level", "read")

            if not user_id or not kb_name:
                results.append({
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "success": False,
                    "error": "Missing required fields: user_id and kb_name",
                })
                failure_count += 1
                continue

            # Call single assign method
            result = await self.assign_kb_to_user(
                client_id=client_id,
                user_id=user_id,
                kb_name=kb_name,
                access_level=access_level,
                ctx=ctx,
                request_id=f"{request_id}_bulk_{user_id[:8]}",
            )

            results.append({
                "user_id": user_id,
                "kb_name": kb_name,
                "access_level": access_level,
                **result,
            })

            if result.get("success"):
                success_count += 1
            else:
                failure_count += 1

        overall_success = failure_count == 0

        logger.warning(
            f"[AUDIT][GAP-BULK-001] Bulk KB assignment completed | "
            f"client_id={client_id} success_count={success_count} failure_count={failure_count} "
            f"overall_success={overall_success} request_id={request_id}",
        )

        return {
            "success": overall_success,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(assignments),
            "results": results,
            "request_id": request_id,
        }

    async def bulk_revoke_kb_from_users(
        self,
        client_id: str,
        revocations: List[Dict[str, Any]],
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        GAP-BULK-001: Revoke KB access from multiple users in a single operation.

        This method iterates over the revocations list and calls revoke_kb_from_user
        for each entry, aggregating results.

        Args:
            client_id: Client ID owning the users
            revocations: List of revocation dicts, each containing:
                - user_id: User to revoke KB from
                - kb_name: Knowledge base name
            ctx: Security context (REQUIRED)
            request_id: Request tracking ID

        Returns:
            Dict with aggregated results:
                - success: True if all revocations succeeded
                - success_count: Number of successful revocations
                - failure_count: Number of failed revocations
                - results: List of individual revocation results
                - request_id: Request tracking ID

        Example:
            await adapter.bulk_revoke_kb_from_users(
                client_id="client-123",
                revocations=[
                    {"user_id": "user-1", "kb_name": "kb-1"},
                    {"user_id": "user-2", "kb_name": "kb-1"},
                    {"user_id": "user-3", "kb_name": "kb-1"},
                ],
                ctx=ctx,
            )
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        logger.warning(
            f"[AUDIT][GAP-BULK-001] Bulk KB revocation initiated | "
            f"client_id={client_id} revocation_count={len(revocations)} request_id={request_id}",
        )

        if not revocations:
            return {
                "success": True,
                "success_count": 0,
                "failure_count": 0,
                "results": [],
                "message": "No revocations provided",
                "request_id": request_id,
            }

        results = []
        success_count = 0
        failure_count = 0

        for revocation in revocations:
            user_id = revocation.get("user_id")
            kb_name = revocation.get("kb_name")

            if not user_id or not kb_name:
                results.append({
                    "user_id": user_id,
                    "kb_name": kb_name,
                    "success": False,
                    "error": "Missing required fields: user_id and kb_name",
                })
                failure_count += 1
                continue

            # Call single revoke method
            result = await self.revoke_kb_from_user(
                client_id=client_id,
                user_id=user_id,
                kb_name=kb_name,
                ctx=ctx,
                request_id=f"{request_id}_bulk_{user_id[:8]}",
            )

            results.append({
                "user_id": user_id,
                "kb_name": kb_name,
                **result,
            })

            if result.get("success"):
                success_count += 1
            else:
                failure_count += 1

        overall_success = failure_count == 0

        logger.warning(
            f"[AUDIT][GAP-BULK-001] Bulk KB revocation completed | "
            f"client_id={client_id} success_count={success_count} failure_count={failure_count} "
            f"overall_success={overall_success} request_id={request_id}",
        )

        return {
            "success": overall_success,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(revocations),
            "results": results,
            "request_id": request_id,
        }

    async def get_user_kb_assignments(
        self,
        client_id: str,
        user_id: str,
        ctx: Any = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all KB assignments for a user (both inherited and custom).

        Returns a comprehensive view of user's KB access including:
        - Client-level inherited KBs (from universal_kbs_assigned)
        - Custom user-level assignments (from custom_user_kb_assignments)
        - Personal KB (if enabled)

        Args:
            client_id: Client ID owning the user
            user_id: User ID to query
            ctx: Security context
            request_id: Request tracking ID

        Returns:
            Dict with KB assignments categorized by source
        """
        if not request_id:
            request_id = str(uuid.uuid4())

        # Security context validation
        if not ctx or not hasattr(ctx, "user") or not ctx.user:
            return {
                "success": False,
                "error": ErrorMessages.INVALID_CONTEXT,
                "request_id": request_id,
            }

        caller_user_id = getattr(ctx.user, "user_id", None)
        caller_client_id = getattr(ctx.user, "client_id", None)
        caller_is_admin = self._is_admin(ctx)

        # Authorization: admin can query any, non-admin only their client's users
        if not caller_is_admin and caller_client_id != client_id:
            return {
                "success": False,
                "error": "Access denied: can only query users in your own client",
                "request_id": request_id,
            }

        # Get client data
        try:
            client_data = await self.client_provider.get_client(client_id)
        except Exception:
            return {
                "success": False,
                "error": ErrorMessages.CLIENT_NOT_FOUND.format(client_id=client_id),
                "request_id": request_id,
            }

        if not client_data:
            return {
                "success": False,
                "error": ErrorMessages.CLIENT_NOT_FOUND.format(client_id=client_id),
                "request_id": request_id,
            }

        kb_config = client_data.get("kb_config", {}) or {}
        default_user_kb_config = kb_config.get("default_user_kb_config", {}) or {}
        custom_assignments = kb_config.get("custom_user_kb_assignments", {}) or {}

        # Get inherited KBs
        # BUG-006 FIX: universal_kbs_assigned contains dicts {"kb_name": str, "access_level": str}
        inherited_kbs_raw = kb_config.get("universal_kbs_assigned", []) or []
        default_access = default_user_kb_config.get("universal_kb_access_level", "read")

        # Get custom assignments for this user
        user_custom = custom_assignments.get(user_id, {}) or {}

        # Build response - handle both dict and legacy string formats
        inherited_kbs_normalized = []
        for kb_entry in inherited_kbs_raw:
            if isinstance(kb_entry, dict):
                kb_name = kb_entry.get("kb_name")
                access_level = kb_entry.get("access_level", default_access)
            else:
                kb_name = kb_entry
                access_level = default_access
            if kb_name:
                inherited_kbs_normalized.append(
                    {
                        "kb_name": kb_name,
                        "access_level": access_level,
                        "source": "client",
                    }
                )

        result = {
            "success": True,
            "client_id": client_id,
            "user_id": user_id,
            "inherited_kbs": inherited_kbs_normalized,
            "custom_kbs": [
                {"kb_name": kb, "access_level": level, "source": "custom"}
                for kb, level in user_custom.items()
            ],
            "personal_kb_enabled": default_user_kb_config.get(
                "personal_kb_enabled", True
            ),
            "request_id": request_id,
        }

        # Calculate effective access (custom overrides inherited)
        effective = {}
        for kb_info in inherited_kbs_normalized:
            kb_name = kb_info["kb_name"]
            effective[kb_name] = {
                "access_level": kb_info["access_level"],
                "source": "client",
            }
        for kb, level in user_custom.items():
            effective[kb] = {"access_level": level, "source": "custom"}

        result["effective_kbs"] = [
            {"kb_name": kb, **info} for kb, info in effective.items()
        ]

        return result
