"""
Admin Users Providers - Pure Technical Logic

This module contains pure technical implementations with ZERO UBP dependencies.
All business logic related to user management, password hashing, and Redis operations.

Separation of Concerns:
- providers.py: Pure technical logic (this file)
- adapter.py: UBP framework bridge
- __init__.py: Factory entry point
"""

from typing import Dict, Any, List, Optional
import json
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from passlib.context import CryptContext

    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False


class PasswordHasher:
    """
    Password hashing provider using bcrypt.

    Pure technical implementation with no framework dependencies.
    """

    def __init__(self, rounds: int = 12):
        """
        Initialize password hasher.

        Args:
            rounds: bcrypt rounds (default 12)

        Raises:
            RuntimeError: If passlib not available
        """
        if not PASSLIB_AVAILABLE:
            raise RuntimeError(
                "passlib[bcrypt] not installed. Run: pip install passlib[bcrypt]"
            )

        # Use PBKDF2-SHA256 (NIST-approved, pure-Python, no C++ dependencies)
        # This ensures portability across Windows/Linux/Mac without bcrypt DLL issues
        self.pwd_context = CryptContext(
            schemes=["pbkdf2_sha256", "bcrypt"],  # PBKDF2 primary, bcrypt fallback
            deprecated="auto",
            pbkdf2_sha256__default_rounds=rounds
            * 1000,  # PBKDF2 uses higher iteration count
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a plaintext password.

        Args:
            password: Plaintext password

        Returns:
            Bcrypt hash
        """
        # Ensure password is a string (not bytes)
        if isinstance(password, bytes):
            password = password.decode("utf-8")
        elif not isinstance(password, str):
            password = str(password)

        # Hash with PBKDF2-SHA256 (no length limits like bcrypt's 72 bytes)
        return self.pwd_context.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plaintext password to verify
            password_hash: Stored bcrypt hash

        Returns:
            True if password matches, False otherwise
        """
        # Ensure password is a string (not bytes)
        if isinstance(password, bytes):
            password = password.decode("utf-8")
        elif not isinstance(password, str):
            password = str(password)

        # Bcrypt has 72 byte limit - truncate if necessary
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            password = password_bytes[:72].decode("utf-8", errors="ignore")

        return self.pwd_context.verify(password, password_hash)


class UserManagementProvider:
    """
    User management provider with Redis storage.

    Pure technical implementation handling all user CRUD operations.
    No UBP framework dependencies.
    """

    def __init__(
        self, redis_client: Any, password_hasher: PasswordHasher, config: Dict[str, Any]
    ):
        """
        Initialize user management provider.

        Args:
            redis_client: Redis client instance (duck-typed)
            password_hasher: PasswordHasher instance
            config: Module configuration dict
        """
        self.redis_client = redis_client
        self.password_hasher = password_hasher
        self.config = config

    # ========================================================================
    # User CRUD Operations
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
        client_id: Optional[str] = None,
        forced_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new user with secure password hashing.

        Args:
            username: Unique username
            password: Plaintext password (will be hashed)
            email: User email (optional)
            full_name: Full name (optional)
            roles: List of role names (optional)
            tenant_id: Tenant ID for multi-tenancy (optional)
            is_active: Active status (default True)
            client_id: Client ID the user belongs to (optional)

        Returns:
            Created user object (without password_hash)

        Raises:
            ValueError: If validation fails or username exists
        """
        # Validation
        validation = self.config.get("validation", {})
        username_min = validation.get("username", {}).get("min_length", 3)
        username_max = validation.get("username", {}).get("max_length", 50)

        if not username or len(username) < username_min or len(username) > username_max:
            raise ValueError(
                f"Username must be {username_min}-{username_max} characters"
            )

        password_min = (
            self.config.get("security", {})
            .get("password_policy", {})
            .get("min_length", 6)
        )
        if not password or len(password) < password_min:
            raise ValueError(f"Password must be at least {password_min} characters")

        # Check username uniqueness (O(1) via Redis index)
        username_key = (
            f"{self.config['redis']['keys']['username_index_prefix']}{username}"
        )
        if await self.redis_client.exists(username_key):
            raise ValueError(f"Username '{username}' already exists")

        # Generate user ID (or accept caller-forced UUID for system bootstrap)
        if forced_user_id:
            try:
                uuid.UUID(forced_user_id)
            except (ValueError, TypeError, AttributeError):
                raise ValueError(
                    f"forced_user_id must be a valid UUID: {forced_user_id!r}"
                )
            # Reject if uid already mapped (prevents identity hijack)
            user_key_prefix = self.config["redis"]["keys"]["user_prefix"]
            if await self.redis_client.exists(f"{user_key_prefix}{forced_user_id}"):
                raise ValueError(
                    f"forced_user_id already in use: {forced_user_id}"
                )
            user_id = forced_user_id
        else:
            user_id = str(uuid.uuid4())

        # Hash password
        password_hash = self.password_hasher.hash_password(password)

        # Create user object
        now = datetime.utcnow().isoformat()
        user = {
            "user_id": user_id,
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "full_name": full_name,
            "roles": roles or [],
            "tenant_id": tenant_id,
            "is_active": is_active,
            "mfa_enabled": False,
            "client_id": client_id,
            "registered_via_client": client_id,
            "created_at": now,
            "updated_at": now,
        }

        # Store in Redis
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        await self.redis_client.set(user_key, json.dumps(user))

        # Add to users index
        users_index = self.config["redis"]["keys"]["users_index"]
        await self.redis_client.hset(users_index, user_id, json.dumps(user))

        # Create username → user_id index (O(1) lookup)
        await self.redis_client.set(username_key, user_id)

        # Return user without password_hash
        user_response = user.copy()
        user_response.pop("password_hash")
        return user_response

    async def list_users(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all users with optional filtering and pagination.

        Args:
            filter_params: Filters {is_active: bool, roles: [str], tenant_id: str}
            limit: Maximum results
            offset: Skip results

        Returns:
            List of user objects (without password_hash)
        """
        users_key = self.config["redis"]["keys"]["users_index"]
        user_data = await self.redis_client.hgetall(users_key)

        users = []
        for user_json in user_data.values():
            user = json.loads(user_json)
            user.pop("password_hash", None)
            users.append(user)

        # Apply filters
        if filter_params:
            if "is_active" in filter_params:
                users = [
                    u for u in users if u.get("is_active") == filter_params["is_active"]
                ]

            if "roles" in filter_params:
                filter_roles = set(filter_params["roles"])
                users = [u for u in users if set(u.get("roles", [])) & filter_roles]

            if "tenant_id" in filter_params:
                users = [
                    u for u in users if u.get("tenant_id") == filter_params["tenant_id"]
                ]

            if "client_id" in filter_params:
                users = [
                    u for u in users if u.get("client_id") == filter_params["client_id"]
                ]

        # Pagination
        if offset:
            users = users[offset:]
        if limit:
            users = users[:limit]

        return users

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get user by ID.

        Args:
            user_id: User UUID

        Returns:
            User object (without password_hash)

        Raises:
            ValueError: If user not found
        """
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)
        user.pop("password_hash", None)
        return user

    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with username and password.

        Args:
            username: Username
            password: Plaintext password

        Returns:
            User object (without password_hash) if authenticated

        Raises:
            ValueError: If user not found or password invalid
        """
        # Get user_id from username index
        username_key = (
            f"{self.config['redis']['keys']['username_index_prefix']}{username}"
        )
        user_id = await self.redis_client.get(username_key)

        if not user_id:
            raise ValueError(f"User not found: {username}")

        # Get full user object (including password_hash)
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)

        # Verify password
        password_hash = user.get("password_hash")
        if not password_hash:
            raise ValueError("User has no password hash")

        if not self.password_hasher.verify_password(password, password_hash):
            raise ValueError("Invalid password")

        # Check if user is active
        if not user.get("is_active", True):
            raise ValueError("User account is disabled")

        # Return user without password_hash
        user.pop("password_hash", None)
        return user

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
    ) -> Dict[str, Any]:
        """
        Update user information.

        Args:
            user_id: User UUID
            username: New username (optional, must be unique)
            email: New email (optional)
            full_name: New full name (optional)
            roles: New roles (optional)
            tenant_id: New tenant ID (optional)
            is_active: New active status (optional)
            mfa_enabled: MFA enabled status (optional)

        Returns:
            Updated user object (without password_hash)

        Raises:
            ValueError: If user not found or username already exists
        """
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)
        old_username = user["username"]

        # Update username if provided and different
        if username is not None and username != old_username:
            # Validate
            validation = self.config.get("validation", {})
            username_min = validation.get("username", {}).get("min_length", 3)
            username_max = validation.get("username", {}).get("max_length", 50)

            if len(username) < username_min or len(username) > username_max:
                raise ValueError(
                    f"Username must be {username_min}-{username_max} characters"
                )

            # Check uniqueness
            new_username_key = (
                f"{self.config['redis']['keys']['username_index_prefix']}{username}"
            )
            if await self.redis_client.exists(new_username_key):
                raise ValueError(f"Username '{username}' already exists")

            # Update username index
            old_username_key = (
                f"{self.config['redis']['keys']['username_index_prefix']}{old_username}"
            )
            await self.redis_client.delete(old_username_key)
            await self.redis_client.set(new_username_key, user_id)

            user["username"] = username

        # Update other fields
        if email is not None:
            user["email"] = email
        if full_name is not None:
            user["full_name"] = full_name
        if roles is not None:
            user["roles"] = roles
        if tenant_id is not None:
            user["tenant_id"] = tenant_id
        if is_active is not None:
            user["is_active"] = is_active
        if mfa_enabled is not None:
            user["mfa_enabled"] = mfa_enabled
        if client_id is not None:
            user["client_id"] = client_id

        user["updated_at"] = datetime.utcnow().isoformat()

        # Save
        await self.redis_client.set(user_key, json.dumps(user))

        # Update index
        users_index = self.config["redis"]["keys"]["users_index"]
        await self.redis_client.hset(users_index, user_id, json.dumps(user))

        # Return without password_hash
        user_response = user.copy()
        user_response.pop("password_hash", None)
        return user_response

    async def delete_user(self, user_id: str) -> Dict[str, str]:
        """
        Delete a user account and all associated Redis data.

        Args:
            user_id: User UUID

        Returns:
            Deletion confirmation with cleanup stats

        Raises:
            ValueError: If user not found
        """
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)
        username = user["username"]

        # Delete user record
        await self.redis_client.delete(user_key)

        # Remove from index
        users_index = self.config["redis"]["keys"]["users_index"]
        await self.redis_client.hdel(users_index, user_id)

        # Remove username index
        username_key = (
            f"{self.config['redis']['keys']['username_index_prefix']}{username}"
        )
        await self.redis_client.delete(username_key)

        # --- BUG-DEL-001: Cleanup individual sessions ---
        # The sorted set ubp:memory:user:{user_id}:sessions contains all session IDs.
        # Each session has 7+ Redis keys under ubp:memory:session:{session_id}:*
        # that are NOT caught by the ubp:memory:user:{user_id}:* pattern.
        sessions_key = f"ubp:memory:user:{user_id}:sessions"
        session_ids_raw = await self.redis_client.zrange(sessions_key, 0, -1)
        sessions_cleaned = 0
        for sid_raw in session_ids_raw:
            sid = sid_raw.decode() if isinstance(sid_raw, bytes) else sid_raw
            session_keys = [
                f"ubp:memory:session:{sid}:messages",
                f"ubp:memory:session:{sid}:metadata",
                f"ubp:memory:session:{sid}:state",
                f"ubp:memory:session:{sid}:pending",
                f"ubp:memory:session:{sid}:recompress_requested",
                f"ubp:memory:session:{sid}:retrieval_hints",
            ]
            # Also delete cached_context (legacy + client-scoped variants)
            cached_ctx_base = f"ubp:memory:session:{sid}:cached_context"
            session_keys.append(cached_ctx_base)
            async for k in self.redis_client.scan_iter(
                match=f"{cached_ctx_base}:*", count=10
            ):
                session_keys.append(k)
            sessions_cleaned += await self.redis_client.delete(*session_keys)

        if sessions_cleaned:
            logger.info(
                f"[ADMIN_USERS] Cleaned {sessions_cleaned} session keys "
                f"for {len(session_ids_raw)} sessions of user {username} ({user_id})"
            )

        # --- Cleanup all user-associated Redis data ---
        cleanup_patterns = [
            f"ubp:profile:user:{user_id}:*",
            f"ubp:memory:user:{user_id}:*",
            f"ubp:dev:rag:history:{user_id}:*",
            f"ubp:rag:history:{user_id}:*",
            f"ubp:dev:rag:config:user:{user_id}",
            f"ubp:rag:acl:user:{user_id}:*",
            f"ubp:user_kbs:{user_id}",
            # BUG-DEL-001: timeline events + seq counters
            f"ubp:timeline:{user_id}:*",
            f"ubp:timeline:seq:{user_id}:*",
        ]
        cleaned = 0
        for pattern in cleanup_patterns:
            if "*" in pattern:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    cleaned += await self.redis_client.delete(*keys)
            else:
                cleaned += await self.redis_client.delete(pattern)

        if cleaned:
            logger.info(
                f"[ADMIN_USERS] Cleaned {cleaned} associated Redis keys "
                f"for user {username} ({user_id})"
            )

        return {
            "message": "User deleted successfully",
            "user_id": user_id,
            "username": username,
            "redis_keys_cleaned": cleaned + sessions_cleaned,
        }

    async def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> Dict[str, str]:
        """
        Change user password with verification.

        Args:
            user_id: User UUID
            old_password: Current password (for verification)
            new_password: New password

        Returns:
            Success confirmation

        Raises:
            ValueError: If user not found, old password invalid, or new password invalid
        """
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)

        # Verify old password
        if not self.password_hasher.verify_password(
            old_password, user["password_hash"]
        ):
            raise ValueError("Invalid current password")

        # Validate new password
        password_min = (
            self.config.get("security", {})
            .get("password_policy", {})
            .get("min_length", 6)
        )
        if len(new_password) < password_min:
            raise ValueError(f"New password must be at least {password_min} characters")

        # Hash and save
        user["password_hash"] = self.password_hasher.hash_password(new_password)
        user["updated_at"] = datetime.utcnow().isoformat()

        await self.redis_client.set(user_key, json.dumps(user))

        # Update index
        users_index = self.config["redis"]["keys"]["users_index"]
        await self.redis_client.hset(users_index, user_id, json.dumps(user))

        return {"message": "Password changed successfully", "user_id": user_id}

    async def admin_reset_password(
        self, user_id: str, new_password: str
    ) -> Dict[str, str]:
        """Admin password reset — bypasses old password verification.

        Only to be called from adapter with admin/client_admin authorization.
        """
        user_key = f"{self.config['redis']['keys']['user_prefix']}{user_id}"
        user_json = await self.redis_client.get(user_key)

        if not user_json:
            raise ValueError(f"User not found: {user_id}")

        user = json.loads(user_json)

        # Validate new password
        password_min = (
            self.config.get("security", {})
            .get("password_policy", {})
            .get("min_length", 6)
        )
        if len(new_password) < password_min:
            raise ValueError(f"New password must be at least {password_min} characters")

        # Hash and save
        user["password_hash"] = self.password_hasher.hash_password(new_password)
        user["updated_at"] = datetime.utcnow().isoformat()

        await self.redis_client.set(user_key, json.dumps(user))

        # Update index
        users_index = self.config["redis"]["keys"]["users_index"]
        await self.redis_client.hset(users_index, user_id, json.dumps(user))

        return {"message": "Password reset successfully", "user_id": user_id}

    # ========================================================================
    # Statistics
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get user management statistics.

        Returns:
            Statistics with counts and distributions
        """
        users_key = self.config["redis"]["keys"]["users_index"]
        all_users = await self.redis_client.hgetall(users_key)

        total_users = len(all_users)
        active_users = 0
        users_by_role = {}
        users_by_tenant = {}

        for user_json in all_users.values():
            user = json.loads(user_json)

            if user.get("is_active"):
                active_users += 1

            for role in user.get("roles", []):
                users_by_role[role] = users_by_role.get(role, 0) + 1

            tenant = user.get("tenant_id", "default")
            users_by_tenant[tenant] = users_by_tenant.get(tenant, 0) + 1

        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "users_by_role": users_by_role,
            "users_by_tenant": users_by_tenant,
        }


# ============================================================================
# Invitation & Email Verification Provider
# ============================================================================


class InvitationProvider:
    """
    Manages invitation codes, email verification tokens, and registration rate limiting.

    All state is stored in Redis with the following key patterns:
    - ubp:invitation:{CODE}        → Hash (invitation data)
    - ubp:email_verify:{TOKEN}     → Hash (verification data, TTL 24h)
    - ubp:register:rate:{IP}       → String counter (TTL 60s)
    """

    INVITE_KEY_PREFIX = "ubp:invitation:"
    VERIFY_KEY_PREFIX = "ubp:email_verify:"
    REGISTER_RATE_PREFIX = "ubp:register:rate:"
    EMAIL_VERIFY_TTL = 86400   # 24 hours
    REGISTER_RATE_TTL = 60     # 1 minute window
    REGISTER_RATE_MAX = 5      # max attempts per window

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client

    # --- Invitation CRUD ---

    async def create_invitation(
        self,
        created_by: str,
        max_uses: int = 10,
        expires_at: Optional[str] = None,
        code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new invitation code and store it in Redis."""
        import random
        import string

        if code is None:
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

        key = f"{self.INVITE_KEY_PREFIX}{code}"

        # Check uniqueness
        if await self.redis_client.exists(key):
            raise ValueError(f"Invitation code '{code}' already exists")

        now = datetime.utcnow().isoformat()
        invitation = {
            "code": code,
            "created_by": created_by,
            "max_uses": max_uses,
            "current_uses": 0,
            "expires_at": expires_at or "",
            "created_at": now,
            "used_by": json.dumps([]),
            "is_active": True,
        }

        await self.redis_client.hset(key, mapping={
            k: str(v) if isinstance(v, (bool, int)) else v
            for k, v in invitation.items()
        })

        logger.info(f"Invitation created: {code} by {created_by} (max_uses={max_uses})")
        return invitation

    async def get_invitation(self, code: str) -> Optional[Dict[str, Any]]:
        """Get invitation data by code, or None if not found."""
        key = f"{self.INVITE_KEY_PREFIX}{code}"
        data = await self.redis_client.hgetall(key)
        if not data:
            return None
        return self._parse_invitation(data)

    async def list_invitations(self) -> List[Dict[str, Any]]:
        """List all invitations via SCAN."""
        results = []
        cursor = 0
        while True:
            cursor, keys = await self.redis_client.scan(
                cursor, match=f"{self.INVITE_KEY_PREFIX}*", count=100
            )
            for key in keys:
                data = await self.redis_client.hgetall(key)
                if data:
                    results.append(self._parse_invitation(data))
            if cursor == 0:
                break
        return results

    async def validate_invitation(self, code: str) -> Dict[str, Any]:
        """
        Validate that an invitation code is usable.

        Raises ValueError with specific message if invalid.
        Returns the invitation data if valid.
        """
        invitation = await self.get_invitation(code)
        if invitation is None:
            raise ValueError("Codice invito non valido")

        if not invitation.get("is_active", False):
            raise ValueError("Codice invito revocato")

        current = invitation.get("current_uses", 0)
        maximum = invitation.get("max_uses", 0)
        if current >= maximum:
            raise ValueError("Codice invito esaurito")

        expires_at = invitation.get("expires_at", "")
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at)
            except (ValueError, TypeError):
                exp = None  # malformed date → treat as no expiry
            if exp is not None and datetime.utcnow() > exp:
                raise ValueError("Codice invito scaduto")

        return invitation

    async def consume_invitation(self, code: str, user_id: str) -> None:
        """Increment usage counter and record user_id in used_by list."""
        key = f"{self.INVITE_KEY_PREFIX}{code}"
        await self.redis_client.hincrby(key, "current_uses", 1)

        # Append user_id to used_by list
        raw = await self.redis_client.hget(key, "used_by")
        used_by = json.loads(raw) if raw else []
        used_by.append(user_id)
        await self.redis_client.hset(key, "used_by", json.dumps(used_by))

    async def revoke_invitation(self, code: str) -> Dict[str, Any]:
        """Soft-delete: set is_active=False (preserves audit trail)."""
        invitation = await self.get_invitation(code)
        if invitation is None:
            raise ValueError(f"Invitation code '{code}' not found")

        key = f"{self.INVITE_KEY_PREFIX}{code}"
        await self.redis_client.hset(key, "is_active", "False")
        invitation["is_active"] = False
        logger.info(f"Invitation revoked: {code}")
        return invitation

    # --- Email verification ---

    async def create_email_verification_token(
        self, user_id: str, email: str
    ) -> str:
        """Create a single-use email verification token with 24h TTL."""
        token = str(uuid.uuid4())
        key = f"{self.VERIFY_KEY_PREFIX}{token}"
        now = datetime.utcnow().isoformat()

        await self.redis_client.hset(key, mapping={
            "user_id": user_id,
            "email": email,
            "created_at": now,
        })
        await self.redis_client.expire(key, self.EMAIL_VERIFY_TTL)

        logger.info(f"Email verification token created for user {user_id}")
        return token

    async def consume_email_verification_token(self, token: str) -> Dict[str, str]:
        """
        Validate and consume a verification token (single-use).

        Returns dict with user_id and email.
        Raises ValueError if token is invalid or expired.
        """
        key = f"{self.VERIFY_KEY_PREFIX}{token}"
        data = await self.redis_client.hgetall(key)
        if not data:
            raise ValueError("Token di verifica non valido o scaduto")

        # Decode bytes if needed
        result = {}
        for k, v in data.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            v_str = v.decode() if isinstance(v, bytes) else v
            result[k_str] = v_str

        # Delete token (single-use)
        await self.redis_client.delete(key)

        return result

    # --- Rate limiting ---

    async def check_registration_rate(self, client_ip: str) -> None:
        """
        Enforce registration rate limit: max 5 per minute per IP.

        Raises ValueError if limit exceeded.
        """
        key = f"{self.REGISTER_RATE_PREFIX}{client_ip}"
        current = await self.redis_client.get(key)

        if current is not None:
            count = int(current)
            if count >= self.REGISTER_RATE_MAX:
                raise ValueError("Troppi tentativi di registrazione. Riprova tra un minuto.")

        pipe = self.redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self.REGISTER_RATE_TTL)
        await pipe.execute()

    # --- Helpers ---

    @staticmethod
    def _parse_invitation(data: Dict) -> Dict[str, Any]:
        """Parse Redis hash data into a clean invitation dict."""
        parsed = {}
        for k, v in data.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            v_str = v.decode() if isinstance(v, bytes) else v
            parsed[k_str] = v_str

        # Type conversions
        parsed["max_uses"] = int(parsed.get("max_uses", 0))
        parsed["current_uses"] = int(parsed.get("current_uses", 0))
        parsed["is_active"] = parsed.get("is_active", "True") == "True"
        try:
            parsed["used_by"] = json.loads(parsed.get("used_by", "[]"))
        except (json.JSONDecodeError, TypeError):
            parsed["used_by"] = []

        return parsed
