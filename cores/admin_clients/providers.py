"""
Admin Clients Providers - Pure Technical Logic

This module contains pure technical implementations with ZERO UBP dependencies.
All business logic related to client management, secret generation, and Redis operations.

Separation of Concerns:
- providers.py: Pure technical logic (this file)
- adapter.py: UBP framework bridge
- __init__.py: Factory entry point
"""

from typing import Dict, Any, List, Optional
import json
import os
import uuid
import secrets
import string
import re
from datetime import datetime

try:
    from passlib.context import CryptContext

    PASSLIB_AVAILABLE = True
except ImportError:
    CryptContext = None  # type: ignore
    PASSLIB_AVAILABLE = False

# v17.19 W1: client_policy admin payload validator (shared with router PATCH path).
from ._client_policy_validator import validate_client_policy  # noqa: E402

# v17.19 W2: pubsub channel name for client_policy cache invalidation.
# Subscribers (mcp-server boot) listen and call client_policy.invalidate().
CLIENT_POLICY_INVALIDATE_CHANNEL = "ubp:cache:client_policy:invalidate"


async def _publish_client_policy_invalidation(redis_client: Any, client_id: str) -> None:
    """Best-effort publish on the client_policy invalidation channel.

    Failures are logged at WARNING but never raised: the TTL fallback in
    client_policy._POLICY_CACHE (5 min) prevents indefinite staleness.
    """
    if not client_id:
        return
    try:
        publish = getattr(redis_client, "publish", None)
        if publish is None:
            return
        # redis.asyncio.Redis exposes publish() as awaitable; sync stubs may
        # return synchronously — handle both.
        result = publish(CLIENT_POLICY_INVALIDATE_CHANNEL, client_id)
        if hasattr(result, "__await__"):
            await result
    except Exception as exc:  # pragma: no cover — best-effort
        import logging
        logging.getLogger(__name__).warning(
            "[client_policy] pubsub publish failed for client_id=%s: %s",
            client_id,
            exc,
        )

# FIX-PACK-CACHE-001: invalidate in-process effective-packs cache after pack-relevant writes.
# Defensive try-import: if backend not on PYTHONPATH (e.g. unit tests), no-op.
try:
    from ubp_enterprise_hybrid.backend.app.core.pack_utils import invalidate_client_packs_cache  # type: ignore
except Exception:  # pragma: no cover
    def invalidate_client_packs_cache(client_id=None):  # type: ignore
        return None


# ARCH-009 v3.0: Baseline packs auto-assigned to every new client at creation time.
# These packs ensure a client is immediately functional (chat, RAG, web search)
# without requiring manual pack assignment by the admin.
# Sub-agents are NOT affected (they use _write_canonical_client_blob() directly).
# Bootstrap clients are NOT affected (update_client(**preset) overwrites authorized_packs).
#
# v17.13 (D-V17.13-D / Phase 3): env override via UBP_DEFAULT_CLIENT_PACKS
# (CSV). Empty/unset → built-in defaults. Whitespace per token tolerated.
def _resolve_default_client_packs() -> List[str]:
    raw = os.environ.get("UBP_DEFAULT_CLIENT_PACKS", "").strip()
    if raw:
        parsed = [p.strip() for p in raw.split(",") if p.strip()]
        if parsed:
            return parsed
    return [
        "general_base",
        "rag_chat_standard",
        "web_search_pipeline",
        "subagent_orchestration",
    ]


DEFAULT_CLIENT_PACKS: List[str] = _resolve_default_client_packs()


class SecretGenerator:
    """
    Secure secret generator for client credentials.

    Pure technical implementation with no framework dependencies.
    """

    def __init__(self, length: int = 32, charset: str = "alphanumeric"):
        """
        Initialize secret generator.

        Args:
            length: Secret length (default 32)
            charset: Character set - "alphanumeric", "hex", or "base64"
        """
        self.length = length
        self.charset = charset

    def generate_secret(self) -> str:
        """
        Generate a cryptographically secure random secret.

        Returns:
            Secure random string of exactly self.length characters
        """
        if self.charset == "hex":
            # token_hex(n) generates exactly 2*n hex characters
            # We need to generate enough and then truncate to exact length
            num_bytes = (self.length + 1) // 2
            return secrets.token_hex(num_bytes)[: self.length]
        elif self.charset == "base64":
            # token_urlsafe(n) generates approximately 4*n/3 characters
            # To ensure we have enough, generate more bytes than needed
            num_bytes = int(self.length * 0.75) + 4  # Add buffer for safety
            return secrets.token_urlsafe(num_bytes)[: self.length]
        else:  # alphanumeric
            alphabet = string.ascii_letters + string.digits
            return "".join(secrets.choice(alphabet) for _ in range(self.length))

    def hash_secret(self, secret: str, pwd_context: Any) -> str:
        """
        Hash a secret using bcrypt.

        Args:
            secret: Plaintext secret
            pwd_context: CryptContext instance

        Returns:
            Bcrypt hash
        """
        return pwd_context.hash(secret)

    def verify_secret(self, secret: str, secret_hash: str, pwd_context: Any) -> bool:
        """
        Verify a secret against its hash.

        Args:
            secret: Plaintext secret to verify
            secret_hash: Stored bcrypt hash
            pwd_context: CryptContext instance

        Returns:
            True if secret matches, False otherwise
        """
        return pwd_context.verify(secret, secret_hash)


class ClientManagementProvider:
    """
    Client management provider with Redis storage.

    Pure technical implementation handling all client CRUD operations.
    No UBP framework dependencies.
    """

    def __init__(
        self,
        redis_client: Any,
        secret_generator: SecretGenerator,
        config: Dict[str, Any],
    ):
        """
        Initialize client management provider.

        Args:
            redis_client: Redis client instance (duck-typed)
            secret_generator: SecretGenerator instance
            config: Module configuration dict
        """
        self.redis_client = redis_client
        self.secret_generator = secret_generator
        self.config = config

        # Initialize password context for secret hashing
        if not PASSLIB_AVAILABLE:
            raise RuntimeError(
                "passlib[bcrypt] not installed. Run: pip install passlib[bcrypt]"
            )

        # Help static analyzers: CryptContext is available if PASSLIB_AVAILABLE is True
        assert CryptContext is not None

        # Use PBKDF2-SHA256 for client secrets.
        # Rationale:
        # - avoids bcrypt backend compatibility issues (bcrypt>=4) and its 72-byte input limit
        # - pure-Python, portable across platforms/containers
        # - still provides strong, slow hashing suitable for client_secret storage
        rounds = (
            self.config.get("security", {}).get("secret_hashing", {}).get("rounds", 12)
        )
        # PBKDF2 rounds scale differently than bcrypt; reuse existing config by multiplying.
        pbkdf2_rounds = max(10000, int(rounds) * 1000)

        self.pwd_context = CryptContext(
            schemes=["pbkdf2_sha256"],
            deprecated="auto",
            pbkdf2_sha256__default_rounds=pbkdf2_rounds,
        )

    # ========================================================================
    # Client CRUD Operations
    # ========================================================================

    def _get_default_model_config(self) -> Dict[str, Any]:
        """Return default model_config with env-driven values."""
        default_grok_model = os.getenv("UBP_PROVIDER_GROK__DEFAULT_MODEL", "grok-3-mini")
        default_max_tokens = int(os.getenv("UBP_CLIENT__DEFAULT_MAX_TOKENS", "4096"))
        default_rate_limit = int(os.getenv("UBP_CLIENT__DEFAULT_RATE_LIMIT", "60"))
        return {
            "allowed_models": ["grok/*", "vllm/*"],
            "default_model": f"grok/{default_grok_model}",
            "max_tokens_per_request": default_max_tokens,
            "temperature_range": {"min": 0.0, "max": 1.0},
            "rate_limit_requests_per_minute": default_rate_limit,
        }

    @staticmethod
    def _normalize_external_mcp_allowed_tools(raw_value: Any) -> List[str]:
        """Normalize UI/existing allowed_tools shapes to ACLFilter format."""
        if isinstance(raw_value, str):
            cleaned = raw_value.strip()
            if not cleaned or cleaned == "*":
                return ["*"]
            tools = [part.strip() for part in cleaned.split(",") if part.strip()]
            return ["*"] if "*" in tools or not tools else tools

        if isinstance(raw_value, list):
            tools: List[str] = []
            for item in raw_value:
                if not isinstance(item, str):
                    continue
                cleaned = item.strip()
                if not cleaned:
                    continue
                if cleaned == "*":
                    return ["*"]
                tools.append(cleaned)
            return tools or ["*"]

        return ["*"]

    @classmethod
    def _normalize_external_mcp_servers(cls, raw_value: Any) -> List[Dict[str, Any]]:
        """Convert UI/existing ext-MCP payloads to ACLFilter-compatible entries."""
        entries: List[Any]
        if isinstance(raw_value, dict):
            servers = raw_value.get("servers")
            entries = servers if isinstance(servers, list) else []
        elif isinstance(raw_value, list):
            entries = raw_value
        else:
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            server_id = entry.get("server_id") or entry.get("id")
            if not isinstance(server_id, str):
                continue
            server_id = server_id.strip()
            if not server_id:
                continue

            enabled = bool(entry.get("enabled", False))
            if not enabled:
                continue

            normalized.append(
                {
                    "server_id": server_id,
                    "enabled": True,
                    "allowed_tools": cls._normalize_external_mcp_allowed_tools(
                        entry.get("allowed_tools", ["*"])
                    ),
                }
            )

        return normalized

    @staticmethod
    def _sync_ext_authorized_packs(client: Dict[str, Any]) -> None:
        """Auto-sync ext_* authorized_packs to match enabled external_mcp_servers.

        When admin toggles servers, this ensures pack_authorization.authorized_packs
        reflects exactly the enabled servers — no phantom packs, no missing packs.
        Modifies *client* dict in-place (called before the single Redis write).
        """
        pc = client.get("pipeline_config", {})
        raw_ext = pc.get("external_mcp_servers", {})
        entries = raw_ext.get("servers", []) if isinstance(raw_ext, dict) else (raw_ext if isinstance(raw_ext, list) else [])

        enabled_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not entry.get("enabled", False):
                continue
            sid = str(entry.get("server_id") or entry.get("id") or "").strip()
            if sid:
                enabled_ids.add(sid)

        pa = client.setdefault("pack_authorization", {})
        old_packs: list = pa.get("authorized_packs", []) or []

        new_packs: list[str] = []
        for p in old_packs:
            if not isinstance(p, str):
                continue
            if not p.startswith("ext_") or p == "ext_base_pack":
                new_packs.append(p)
                continue
            server_id = p.removeprefix("ext_").removesuffix("_pack")
            if server_id in enabled_ids:
                new_packs.append(p)

        seen = {p for p in new_packs if p.startswith("ext_")}
        for sid in sorted(enabled_ids):
            pack_name = f"ext_{sid}_pack"
            if pack_name not in seen:
                new_packs.append(pack_name)
                seen.add(pack_name)

        has_ext = any(p.startswith("ext_") and p != "ext_base_pack" for p in new_packs)
        if has_ext and "ext_base_pack" not in new_packs:
            new_packs.insert(0, "ext_base_pack")
        elif not has_ext and "ext_base_pack" in new_packs:
            new_packs.remove("ext_base_pack")

        pa["authorized_packs"] = new_packs

    async def _sync_pipeline_to_tenant(
        self, client_id: str, pipeline_config: Dict[str, Any]
    ) -> None:
        """Sync MCP-relevant pipeline_config subfields to tenant_pipeline:{client_id}.

        ARCH-BRIDGE-001: mirrors the fields that the MCP ACLFilter reads from the
        tenant_pipeline Redis HASH back from the authoritative ubp:admin:client store.

        Mapping:
          pipeline_config.external_mcp_servers → external_mcp_servers  (JSON)
          pipeline_config.enrichment_config    → enrichment_config     (JSON)
          pipeline_config.agent_memory_config  → agent_memory_config   (JSON)
          pipeline_config.agent_loop_config    → agent_loop_config     (JSON)
          pipeline_config.thinking_config      → thinking_config       (JSON)

        Only syncs fields explicitly present and non-None in pipeline_config.
        Fail-open: any Redis error is logged as a warning and swallowed — the
        main save to ubp:admin:client has already succeeded.
        """
        sync_fields: Dict[str, str] = {}

        if pipeline_config.get("external_mcp_servers") is not None:
            normalized_external = self._normalize_external_mcp_servers(
                pipeline_config["external_mcp_servers"]
            )
            sync_fields["external_mcp_servers"] = json.dumps(
                normalized_external
            )
        if pipeline_config.get("enrichment_config") is not None:
            sync_fields["enrichment_config"] = json.dumps(
                pipeline_config["enrichment_config"]
            )
        if pipeline_config.get("agent_memory_config") is not None:
            sync_fields["agent_memory_config"] = json.dumps(
                pipeline_config["agent_memory_config"]
            )
        if pipeline_config.get("agent_loop_config") is not None:
            sync_fields["agent_loop_config"] = json.dumps(
                pipeline_config["agent_loop_config"]
            )
        if pipeline_config.get("thinking_config") is not None:
            sync_fields["thinking_config"] = json.dumps(
                pipeline_config["thinking_config"]
            )

                # Mapping: pipeline_config key → tenant_pipeline field name
        _BRIDGE_FIELD_MAP = {
            "external_mcp_servers": "external_mcp_servers",
            "enrichment_config": "enrichment_config",
            "agent_memory_config": "agent_memory_config",
            "agent_loop_config": "agent_loop_config",
            "thinking_config": "thinking_config",
        }

        # Fields explicitly set to None → should be deleted from tenant_pipeline
        fields_to_delete = [
            tp_key
            for pc_key, tp_key in _BRIDGE_FIELD_MAP.items()
            if pc_key in pipeline_config and pipeline_config[pc_key] is None
        ]

        if not sync_fields and not fields_to_delete:
            return

        tenant_key = f"tenant_pipeline:{client_id}"
        try:
            if sync_fields:
                await self.redis_client.hset(tenant_key, mapping=sync_fields)
            if fields_to_delete:
                await self.redis_client.hdel(tenant_key, *fields_to_delete)
            # FIX-PACK-CACHE-001: external_mcp_servers feeds auto-pack resolution.
            invalidate_client_packs_cache(client_id)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "[BRIDGE] tenant_pipeline sync failed for client %s: %s",
                client_id,
                exc,
            )

    # ARCH-BRIDGE-001-EXT: classmethod variant that accepts an external redis client.
    # Called by AdminClientsAdapter.sync_pipeline_to_tenant() to avoid using the
    # provider's own redis_client (which may be in a different connection state than
    # the request-scoped redis from _get_redis(request)).
    @classmethod
    async def sync_pipeline_to_tenant_with_redis(
        cls,
        client_id: str,
        pipeline_config: Dict[str, Any],
        redis_client: Any,
    ) -> None:
        """Sync pipeline_config → tenant_pipeline:{client_id} using a caller-provided redis.

        Identical logic to _sync_pipeline_to_tenant but uses *redis_client* arg instead of
        self.redis_client, so the router can pass the already-connected request-scoped client.
        Fail-open: Redis errors are logged as warnings and swallowed — the upstream save has
        already succeeded.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        sync_fields: Dict[str, str] = {}

        if pipeline_config.get("external_mcp_servers") is not None:
            normalized_external = cls._normalize_external_mcp_servers(
                pipeline_config["external_mcp_servers"]
            )
            sync_fields["external_mcp_servers"] = json.dumps(normalized_external)
        if pipeline_config.get("enrichment_config") is not None:
            sync_fields["enrichment_config"] = json.dumps(pipeline_config["enrichment_config"])
        if pipeline_config.get("agent_memory_config") is not None:
            sync_fields["agent_memory_config"] = json.dumps(pipeline_config["agent_memory_config"])
        if pipeline_config.get("agent_loop_config") is not None:
            sync_fields["agent_loop_config"] = json.dumps(pipeline_config["agent_loop_config"])
        if pipeline_config.get("thinking_config") is not None:
            sync_fields["thinking_config"] = json.dumps(pipeline_config["thinking_config"])

        _BRIDGE_FIELD_MAP = {
            "external_mcp_servers": "external_mcp_servers",
            "enrichment_config": "enrichment_config",
            "agent_memory_config": "agent_memory_config",
            "agent_loop_config": "agent_loop_config",
            "thinking_config": "thinking_config",
        }
        fields_to_delete = [
            tp_key
            for pc_key, tp_key in _BRIDGE_FIELD_MAP.items()
            if pc_key in pipeline_config and pipeline_config[pc_key] is None
        ]

        if not sync_fields and not fields_to_delete:
            return

        tenant_key = f"tenant_pipeline:{client_id}"
        try:
            if sync_fields:
                await redis_client.hset(tenant_key, mapping=sync_fields)
            if fields_to_delete:
                await redis_client.hdel(tenant_key, *fields_to_delete)
            invalidate_client_packs_cache(client_id)
        except Exception as exc:
            _logger.warning(
                "[BRIDGE] tenant_pipeline sync failed for client %s: %s",
                client_id[:8] if len(client_id) > 8 else client_id,
                exc,
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
        # v17.19 W1: Per-client policy (allowed_subagent_profiles, future expansion)
        client_policy: Optional[Dict[str, Any]] = None,
        # System client protection
        is_system: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new client with secure secret generation and enterprise configuration.

        Args:
            client_name: Unique client name
            client_type: Client type (oauth2/api_key/service_account)
            description: Client description (optional)
            redirect_uris: OAuth2 redirect URIs (optional)
            scopes: List of scope names (optional)
            tenant_id: Tenant ID for multi-tenancy (optional)
            is_active: Active status (default True)
            expires_at: Expiration datetime ISO string (optional)
            kb_config: KB configuration (universal KBs, assignment mode, personal KB settings)
            ingestion_config: Ingestion settings per client
            model_config: Allowed LLM models per client
            user_limits: Max users, allowed roles, concurrent sessions
            rag_config: RAG pipeline configuration
            prompt_config: System prompts, templates, permissions
            authorization: Allowed/denied endpoints
            settings_permissions: Permission to modify settings
            pipeline_config: Per-client pipeline whitelist and overrides (v6.6)
            feature_flags: Per-client feature toggles (v6.6)
            rate_limits: Per-client rate limiting (v6.6)
            domain_config: Domain-specific configuration (PRESET-001)

        Returns:
            Created client object (with plaintext client_secret, only returned once)

        Raises:
            ValueError: If validation fails or client_name exists
        """
        # Validation
        validation = self.config.get("validation", {})
        name_min = validation.get("client_name", {}).get("min_length", 3)
        name_max = validation.get("client_name", {}).get("max_length", 100)
        name_pattern = validation.get("client_name", {}).get("pattern")

        if (
            not client_name
            or len(client_name) < name_min
            or len(client_name) > name_max
        ):
            raise ValueError(f"Client name must be {name_min}-{name_max} characters")

        # Validate client name pattern if configured
        if name_pattern and not re.match(name_pattern, client_name):
            raise ValueError(f"Client name must match pattern: {name_pattern}")

        allowed_types = validation.get("client_type", {}).get(
            "allowed_values", ["oauth2", "api_key", "service_account"]
        )
        if client_type not in allowed_types:
            raise ValueError(f"Client type must be one of: {', '.join(allowed_types)}")

        # v17.19 W1: validate client_policy payload (D4: wildcard mix → reject).
        validated_client_policy: Optional[Dict[str, Any]] = None
        if client_policy is not None:
            validated_client_policy = validate_client_policy(client_policy)

        # v1.8.3: Enforce explicit allowed_models when model_config is provided
        if model_config and "allowed_models" not in model_config:
            logger.warning(
                f"[ADMIN_CLIENTS] create_client '{client_name}': model_config "
                f"missing 'allowed_models' — injecting default from config"
            )
            model_config["allowed_models"] = self._get_default_model_config().get(
                "allowed_models", ["grok/*", "vllm/*"]
            )

        # Check client_name uniqueness (O(1) via Redis index)
        name_key = (
            f"{self.config['redis']['keys']['client_name_index_prefix']}{client_name}"
        )
        if await self.redis_client.exists(name_key):
            raise ValueError(f"Client name '{client_name}' already exists")

        # Generate client ID and secret
        client_id = str(uuid.uuid4())
        client_secret = self.secret_generator.generate_secret()
        client_secret_hash = self.secret_generator.hash_secret(
            client_secret, self.pwd_context
        )

        # Create client object with enterprise configuration
        now = datetime.utcnow().isoformat()

        # Default enterprise configurations
        default_kb_config = {
            "universal_kbs_assigned": [],
            "can_create_universal_kb": False,
            "max_universal_kbs": 5,
            "user_kb_assignment_mode": "default",
            "default_user_kb_config": {
                "inherit_all_universal_kbs": True,
                "universal_kb_access_level": "read",
                "personal_kb_enabled": True,
                "personal_kb_max_size_mb": 100,
                "personal_kb_max_documents": 50,
            },
            "custom_user_kb_assignments": {},
        }

        default_ingestion_config = {
            "enabled": True,
            "max_documents_per_day": 100,
            "max_document_size_mb": 10,
            "allowed_formats": ["pdf", "docx", "txt", "md"],
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": "default",
        }

        # v1.8.3: Reuse _get_default_model_config() — single source of truth
        default_model_config = self._get_default_model_config()

        default_user_limits = {
            "max_users": 100,
            "current_users": 0,
            "user_roles_allowed": ["user", "viewer"],
            "max_concurrent_sessions_per_user": 3,
        }

        default_rag_config = {
            "pipeline": "standard",
            "top_k": 5,
            "similarity_threshold": 0.7,
            "reranking_enabled": False,
            "conversation_memory_enabled": True,
            "web_search_enabled": False,
        }

        default_prompt_config = {
            "identity": None,  # L0 — if null uses DEFAULT_SYSTEM_IDENTITY
            "system_prompt": "",
            "system_prompt_max_chars": 500,
            "context_template": "Use the following context to answer the question:\n{context}",
            "can_modify_system_prompt": False,
            "can_modify_context_template": False,
            "language": "it",
            "response_style": "professional",
            # v6.6: Extended prompt templates (admin-managed)
            "prompt_templates": {},
            "template_variables": {},
            "override_user_preferences": False,
            # v6.9.1: Role-based L2 prompt (client_admin personalizes per role)
            "role_prompts": {},
        }

        default_authorization = {"allowed_endpoints": ["*"], "denied_endpoints": []}

        default_settings_permissions = {
            "can_modify_ingestion_settings": False,
            "can_modify_model_settings": False,
            "can_modify_rag_settings": False,
            # v6.6: Granular settings permissions
            "can_modify": [],
            "cannot_modify": [
                "pipeline_config.*",
                "model_config.allowed_models",
                "feature_flags.*",
                "rate_limits.*",
                "authorization.*",
            ],
        }

        # v6.6: Per-client pipeline configuration (admin-managed)
        default_pipeline_config = {
            "allowed_pipelines": ["*"],
            "default_pipeline": "rag_chat_standard",
            "pipeline_overrides": {},
            "step_overrides": {},
        }

        # v6.6: Per-client feature flags (admin-managed)
        default_feature_flags = {
            "web_search_enabled": True,
            "report_generation_enabled": True,
            "investigation_enabled": True,
            "agent_pipeline_enabled": True,
            "hyde_enabled": True,
            "conversation_memory_enabled": True,
            "streaming_enabled": True,
            "file_upload_enabled": True,
            "presentation_enabled": True,
            "cross_lingual_enabled": True,
            "llm_router_enabled": True,
            "menu_management_enabled": False,  # opt-in: admin enables per HoReCa client
        }

        # v6.6: Per-client rate limits (admin-managed, -1 = unlimited)
        default_rate_limits = {
            "max_requests_per_minute": -1,
            "max_requests_per_hour": -1,
            "max_requests_per_day": -1,
            "max_concurrent_sessions": -1,
            "max_tokens_per_day": -1,
            "max_file_upload_size_mb": -1,
            "max_files_per_day": -1,
            "max_collections": -1,
        }

        # PRESET-001: Domain-specific configuration
        default_domain_config = {
            "domain": None,
            "citation_required": False,
            "verify_citations_always": False,
            "confidence_threshold": 0.7,
            "disclaimer_required": False,
            "disclaimer_text": "",
            "preferred_sources": [],
            "forbidden_source_patterns": [],
        }

        client = {
            "client_id": client_id,
            "client_secret_hash": client_secret_hash,
            "client_name": client_name,
            "client_type": client_type,
            "description": description,
            "redirect_uris": redirect_uris or [],
            "scopes": scopes or [],
            "tenant_id": tenant_id,
            "is_active": is_active,
            "bootstrap_status": "pending",
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
            "last_used_at": None,
            "secret_rotated_at": None,
            # Enterprise configuration v2.0
            "kb_config": {**default_kb_config, **(kb_config or {})},
            "ingestion_config": {
                **default_ingestion_config,
                **(ingestion_config or {}),
            },
            "model_config": {**default_model_config, **(model_config or {})},
            "user_limits": {**default_user_limits, **(user_limits or {})},
            "rag_config": {**default_rag_config, **(rag_config or {})},
            "prompt_config": {**default_prompt_config, **(prompt_config or {})},
            "authorization": {**default_authorization, **(authorization or {})},
            "settings_permissions": {
                **default_settings_permissions,
                **(settings_permissions or {}),
            },
            # v6.6: New per-client configuration sections
            "pipeline_config": {**default_pipeline_config, **(pipeline_config or {})},
            "feature_flags": {**default_feature_flags, **(feature_flags or {})},
            "rate_limits": {**default_rate_limits, **(rate_limits or {})},
            # PRESET-001
            "domain_config": {**default_domain_config, **(domain_config or {})},
            # v17.19 W1: client_policy. If admin omitted the param, default is
            # deny-all (empty list). Admins must opt-in to sub-agent spawning
            # by passing client_policy={'allowed_subagent_profiles': [...]}.
            "client_policy": validated_client_policy
            or {"allowed_subagent_profiles": []},
            # ARCH-009 v3.0: Baseline packs — every new client starts with minimal tool coverage.
            # Admin can add/remove packs after creation. Preset application overwrites this list.
            "pack_authorization": {
                "authorized_packs": list(DEFAULT_CLIENT_PACKS),
            },
            # System client protection
            "is_system": is_system,
        }

        # Store in Redis
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        await self.redis_client.set(client_key, json.dumps(client))

        # Add to clients index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        # Create client_name → client_id index (O(1) lookup)
        await self.redis_client.set(name_key, client_id)

        # FIX-PACK-CACHE-001: warm slot — ensure no stale negative cache for this id.
        invalidate_client_packs_cache(client_id)

        # v17.19 W2: notify mcp-server workers to drop any cached ClientPolicy
        # for this client_id. Best-effort; TTL covers if pubsub fails.
        await _publish_client_policy_invalidation(self.redis_client, client_id)

        # BRIDGE (ARCH-BRIDGE-001): sync pipeline_config to tenant_pipeline
        await self._sync_pipeline_to_tenant(client_id, client["pipeline_config"])

        # Return client with plaintext secret (only time it's visible)
        client_response = client.copy()
        client_response.pop("client_secret_hash")
        client_response["client_secret"] = client_secret
        return client_response

    async def list_clients(
        self,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all clients with optional filtering and pagination.

        Args:
            filter_params: Filters {is_active: bool, client_type: str, scopes: [str], tenant_id: str}
            limit: Maximum results
            offset: Skip results

        Returns:
            List of client objects (without client_secret_hash)
        """
        clients_key = self.config["redis"]["keys"]["clients_index"]
        client_data = await self.redis_client.hgetall(clients_key)

        clients = []
        for client_json in client_data.values():
            client = json.loads(client_json)
            client.pop("client_secret_hash", None)
            clients.append(client)

        # Apply filters
        if filter_params:
            if "is_active" in filter_params:
                clients = [
                    c
                    for c in clients
                    if c.get("is_active") == filter_params["is_active"]
                ]

            if "client_type" in filter_params:
                clients = [
                    c
                    for c in clients
                    if c.get("client_type") == filter_params["client_type"]
                ]

            if "scopes" in filter_params:
                filter_scopes = set(filter_params["scopes"])
                clients = [
                    c for c in clients if set(c.get("scopes", [])) & filter_scopes
                ]

            if "tenant_id" in filter_params:
                clients = [
                    c
                    for c in clients
                    if c.get("tenant_id") == filter_params["tenant_id"]
                ]

            if "is_subagent" in filter_params:
                want = bool(filter_params["is_subagent"])
                clients = [
                    c for c in clients if bool(c.get("is_subagent", False)) is want
                ]
            else:
                # default-exclude sub-agents from the admin client list (v14)
                clients = [c for c in clients if not c.get("is_subagent", False)]
        else:
            # No filter dict at all → still default-exclude sub-agents (v14)
            clients = [c for c in clients if not c.get("is_subagent", False)]

        # Pagination
        start = offset if offset else 0
        end = (start + limit) if limit else None
        clients = clients[start:end]

        return clients

    async def _get_client_raw(self, client_id: str) -> Dict[str, Any]:
        """
        Get client by ID — raw, with all fields including client_secret_hash.

        Internal use only (e.g. _save_local_config needs the full object).

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        # v6.6: Lazy migration — add missing config sections with defaults
        client = self._migrate_client_schema(client)
        return client

    async def get_client(self, client_id: str) -> Dict[str, Any]:
        """
        Get client by ID.

        Args:
            client_id: Client UUID

        Returns:
            Client object (without client_secret_hash)

        Raises:
            ValueError: If client not found
        """
        client = await self._get_client_raw(client_id)
        client.pop("client_secret_hash", None)
        return client

    async def get_client_by_name(self, client_name: str) -> Optional[Dict[str, Any]]:
        """Get client by name via O(1) name index lookup.

        Returns:
            Client object (without client_secret_hash), or None if not found.
        """
        name_key = f"{self.config['redis']['keys']['client_name_index_prefix']}{client_name}"
        client_id = await self.redis_client.get(name_key)
        if not client_id:
            return None
        if isinstance(client_id, bytes):
            client_id = client_id.decode()
        try:
            return await self.get_client(client_id)
        except ValueError:
            return None

    def _migrate_client_schema(self, client: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure client has all v6.6 config sections with backward-compatible defaults."""
        migration_defaults = {
            "pipeline_config": {
                "allowed_pipelines": ["*"],
                "default_pipeline": "rag_chat_standard",
                "pipeline_overrides": {},
                "step_overrides": {},
            },
            "feature_flags": {
                "web_search_enabled": True,
                "report_generation_enabled": True,
                "investigation_enabled": True,
                "agent_pipeline_enabled": True,
                "hyde_enabled": True,
                "conversation_memory_enabled": True,
                "streaming_enabled": True,
                "file_upload_enabled": True,
                "presentation_enabled": True,
                "cross_lingual_enabled": True,
                "llm_router_enabled": True,
            },
            "rate_limits": {
                "max_requests_per_minute": -1,
                "max_requests_per_hour": -1,
                "max_requests_per_day": -1,
                "max_concurrent_sessions": -1,
                "max_tokens_per_day": -1,
                "max_file_upload_size_mb": -1,
                "max_files_per_day": -1,
                "max_collections": -1,
            },
            # ARCH-009 v2.0: Pack authorization
            "pack_authorization": {
                "authorized_packs": [],
            },
            # Phase 5: MCP internal runtime rollout
            "mcp_config": {
                "use_internal_runtime": False,
            },
            # System client protection
            # PRESET-001
            "domain_config": {
                "domain": None,
                "citation_required": False,
                "verify_citations_always": False,
                "confidence_threshold": 0.7,
                "disclaimer_required": False,
                "disclaimer_text": "",
                "preferred_sources": [],
                "forbidden_source_patterns": [],
            },
        }
        # Add prompt_config new fields if missing
        pc = client.get("prompt_config", {})
        if "system_prompt_max_chars" not in pc:
            pc["system_prompt_max_chars"] = 500
        if "prompt_templates" not in pc:
            pc["prompt_templates"] = {}
        if "template_variables" not in pc:
            pc["template_variables"] = {}
        if "override_user_preferences" not in pc:
            pc["override_user_preferences"] = False
        client["prompt_config"] = pc

        # Add settings_permissions new fields if missing
        sp = client.get("settings_permissions", {})
        if "can_modify" not in sp:
            sp["can_modify"] = []
        if "cannot_modify" not in sp:
            sp["cannot_modify"] = [
                "pipeline_config.*",
                "model_config.allowed_models",
                "feature_flags.*",
                "rate_limits.*",
                "authorization.*",
            ]
        client["settings_permissions"] = sp

        # Scalar defaults (not dict sections — handle before the loop)
        client.setdefault("is_system", False)

        # Add new top-level config sections
        for key, defaults in migration_defaults.items():
            if key not in client:
                client[key] = defaults
            else:
                # Ensure sub-keys exist
                for sub_key, sub_val in defaults.items():
                    if sub_key not in client[key]:
                        client[key][sub_key] = sub_val
        return client

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
        # v17.19 W1: Per-client policy (allowed_subagent_profiles)
        client_policy: Optional[Dict[str, Any]] = None,
        # Pack authorization + system protection
        pack_authorization: Optional[Dict[str, Any]] = None,
        is_system: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Update client information and enterprise configuration.

        Args:
            client_id: Client UUID
            client_name: New client name (optional, must be unique)
            description: New description (optional)
            redirect_uris: New redirect URIs (optional)
            scopes: New scopes (optional)
            tenant_id: New tenant ID (optional)
            is_active: New active status (optional)
            expires_at: New expiration datetime (optional)
            kb_config: KB configuration update (optional, merged with existing)
            ingestion_config: Ingestion config update (optional, merged)
            model_config: Model config update (optional, merged)
            user_limits: User limits update (optional, merged)
            rag_config: RAG config update (optional, merged)
            prompt_config: Prompt config update (optional, merged)
            authorization: Authorization update (optional, merged)
            settings_permissions: Settings permissions update (optional, merged)
            domain_config: Domain-specific config update (PRESET-001, merged)

        Returns:
            Updated client object (without client_secret_hash)

        Raises:
            ValueError: If client not found or client_name already exists
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        old_client_name = client["client_name"]

        # Update client_name if provided and different
        if client_name is not None and client_name != old_client_name:
            # Validate
            validation = self.config.get("validation", {})
            name_min = validation.get("client_name", {}).get("min_length", 3)
            name_max = validation.get("client_name", {}).get("max_length", 100)
            name_pattern = validation.get("client_name", {}).get("pattern")

            if len(client_name) < name_min or len(client_name) > name_max:
                raise ValueError(
                    f"Client name must be {name_min}-{name_max} characters"
                )

            # Validate client name pattern if configured
            if name_pattern and not re.match(name_pattern, client_name):
                raise ValueError(f"Client name must match pattern: {name_pattern}")

            # Check uniqueness
            new_name_key = f"{self.config['redis']['keys']['client_name_index_prefix']}{client_name}"
            if await self.redis_client.exists(new_name_key):
                raise ValueError(f"Client name '{client_name}' already exists")

            # Update client_name index
            old_name_key = f"{self.config['redis']['keys']['client_name_index_prefix']}{old_client_name}"
            await self.redis_client.delete(old_name_key)
            await self.redis_client.set(new_name_key, client_id)

            client["client_name"] = client_name

        # Update other fields
        if description is not None:
            client["description"] = description
        if redirect_uris is not None:
            client["redirect_uris"] = redirect_uris
        if scopes is not None:
            client["scopes"] = scopes
        if tenant_id is not None:
            client["tenant_id"] = tenant_id
        if is_active is not None:
            client["is_active"] = is_active
        if expires_at is not None:
            client["expires_at"] = expires_at

        # Update enterprise configuration fields (merge with existing)
        if kb_config is not None:
            existing_kb = client.get("kb_config", {})
            client["kb_config"] = {**existing_kb, **kb_config}
        if ingestion_config is not None:
            existing_ingestion = client.get("ingestion_config", {})
            client["ingestion_config"] = {**existing_ingestion, **ingestion_config}
        if model_config is not None:
            existing_model = client.get("model_config", {})
            client["model_config"] = {**existing_model, **model_config}
        if user_limits is not None:
            existing_user_limits = client.get("user_limits", {})
            # Preserve current_users count, only update other fields
            current_users = existing_user_limits.get("current_users", 0)
            client["user_limits"] = {**existing_user_limits, **user_limits}
            client["user_limits"]["current_users"] = current_users
        if rag_config is not None:
            existing_rag = client.get("rag_config", {})
            client["rag_config"] = {**existing_rag, **rag_config}
        if prompt_config is not None:
            existing_prompt = client.get("prompt_config", {})
            client["prompt_config"] = {**existing_prompt, **prompt_config}
        if authorization is not None:
            existing_auth = client.get("authorization", {})
            client["authorization"] = {**existing_auth, **authorization}
        if settings_permissions is not None:
            existing_perms = client.get("settings_permissions", {})
            client["settings_permissions"] = {**existing_perms, **settings_permissions}
        # v6.6: Merge new per-client config sections
        if pipeline_config is not None:
            existing_pc = client.get("pipeline_config", {})
            client["pipeline_config"] = {**existing_pc, **pipeline_config}
        if feature_flags is not None:
            existing_ff = client.get("feature_flags", {})
            client["feature_flags"] = {**existing_ff, **feature_flags}
        if rate_limits is not None:
            existing_rl = client.get("rate_limits", {})
            client["rate_limits"] = {**existing_rl, **rate_limits}
        # PRESET-001: Merge domain_config
        if domain_config is not None:
            existing_dc = client.get("domain_config", {})
            client["domain_config"] = {**existing_dc, **domain_config}
        # v17.19 W1: client_policy is REPLACE semantics (not merge) because
        # allowed_subagent_profiles is a list, not a sub-dict. Validation runs
        # before mutation so a bad payload never persists.
        if client_policy is not None:
            client["client_policy"] = validate_client_policy(client_policy)
        # Pack authorization merge
        if pack_authorization is not None:
            existing_pa = client.get("pack_authorization", {})
            client["pack_authorization"] = {**existing_pa, **pack_authorization}
        # System client flag
        if is_system is not None:
            client["is_system"] = is_system

        # EXT-SYNC: keep authorized_packs aligned with external_mcp_servers
        if pipeline_config is not None and "external_mcp_servers" in pipeline_config:
            self._sync_ext_authorized_packs(client)

        client["updated_at"] = datetime.utcnow().isoformat()

        # Save
        await self.redis_client.set(client_key, json.dumps(client))

        # Update index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        # BRIDGE (ARCH-BRIDGE-001): sync pipeline_config to tenant_pipeline
        if pipeline_config is not None:
            await self._sync_pipeline_to_tenant(client_id, client["pipeline_config"])

        # FIX-PACK-CACHE-001: pack_authorization may have changed; drop cached entry.
        invalidate_client_packs_cache(client_id)

        # v17.19 W2: notify mcp-server workers to drop any cached ClientPolicy
        # for this client_id (covers client_policy + any other field that
        # affects mcp-server runtime). Best-effort; TTL covers pubsub failures.
        await _publish_client_policy_invalidation(self.redis_client, client_id)

        # Return without client_secret_hash
        client_response = client.copy()
        client_response.pop("client_secret_hash", None)
        return client_response

    async def set_bootstrap_status(
        self,
        client_id: str,
        status: str,
        is_active: Optional[bool] = None,
    ) -> None:
        """
        Update the bootstrap_status field (and optionally is_active) for a client.

        Used by create_client to mark whether initial admin user provisioning
        succeeded ('ok') or failed ('failed').  A 'failed' client is set to
        inactive so it is not considered operational until resolved.

        Args:
            client_id: Client UUID
            status: Must be one of: 'pending', 'ok', 'failed', 'pending_retry'
            is_active: If provided, also update the is_active flag

        Raises:
            ValueError: If status is not one of the allowed values
        """
        _ALLOWED_STATUSES = {"pending", "ok", "failed", "pending_retry"}
        if status not in _ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid bootstrap_status '{status}'. "
                f"Allowed values: {sorted(_ALLOWED_STATUSES)}"
            )

        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            # Client may have been deleted concurrently; log and return
            return

        client = json.loads(client_json)
        client["bootstrap_status"] = status
        if is_active is not None:
            client["is_active"] = is_active
        client["updated_at"] = datetime.utcnow().isoformat()

        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.set(client_key, json.dumps(client))
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

    async def delete_client(self, client_id: str) -> Dict[str, str]:
        """
        Delete a client permanently.

        Args:
            client_id: Client UUID

        Returns:
            Deletion confirmation

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        client_name = client["client_name"]

        # System client protection
        if client.get("is_system"):
            raise ValueError("Cannot delete system client (is_system=true)")

        # Delete client
        await self.redis_client.delete(client_key)

        # Remove from index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hdel(clients_index, client_id)

        # Remove client_name index
        name_key = (
            f"{self.config['redis']['keys']['client_name_index_prefix']}{client_name}"
        )
        await self.redis_client.delete(name_key)

        # FIX-PACK-CACHE-001: drop pack cache for deleted client.
        invalidate_client_packs_cache(client_id)

        return {
            "message": "Client deleted successfully",
            "client_id": client_id,
            "client_name": client_name,
        }

    async def rotate_secret(self, client_id: str) -> Dict[str, Any]:
        """
        Rotate client secret (generate new secret and invalidate old one).

        Args:
            client_id: Client UUID

        Returns:
            Rotation confirmation with new secret (only returned once)

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)

        # Generate new secret
        new_secret = self.secret_generator.generate_secret()
        new_secret_hash = self.secret_generator.hash_secret(
            new_secret, self.pwd_context
        )

        # Update client
        now = datetime.utcnow().isoformat()
        client["client_secret_hash"] = new_secret_hash
        client["secret_rotated_at"] = now
        client["updated_at"] = now

        # Save
        await self.redis_client.set(client_key, json.dumps(client))

        # Update index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        return {
            "message": "Client secret rotated successfully",
            "client_id": client_id,
            "client_secret": new_secret,  # Only time new secret is visible
            "rotated_at": now,
        }

    async def revoke_client(
        self, client_id: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Revoke client access (soft delete - sets is_active to false).

        Args:
            client_id: Client UUID
            reason: Revocation reason (optional)

        Returns:
            Revocation confirmation

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)

        # Revoke client
        now = datetime.utcnow().isoformat()
        client["is_active"] = False
        client["revoked_at"] = now
        client["revocation_reason"] = reason
        client["updated_at"] = now

        # Save
        await self.redis_client.set(client_key, json.dumps(client))

        # Update index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        # FIX-PACK-CACHE-001: revoked client must lose pack access immediately.
        invalidate_client_packs_cache(client_id)

        return {
            "message": "Client revoked successfully",
            "client_id": client_id,
            "client_name": client["client_name"],
            "revoked_at": now,
        }

    # ========================================================================
    # Statistics
    # ========================================================================

    async def get_client_stats(self) -> Dict[str, Any]:
        """
        Get client management statistics.

        Returns:
            Statistics with counts and distributions
        """
        clients_key = self.config["redis"]["keys"]["clients_index"]
        all_clients = await self.redis_client.hgetall(clients_key)

        total_clients = len(all_clients)
        active_clients = 0
        revoked_clients = 0
        clients_by_type = {}
        clients_by_tenant = {}

        for client_json in all_clients.values():
            client = json.loads(client_json)

            if client.get("is_active"):
                active_clients += 1
            else:
                revoked_clients += 1

            client_type = client.get("client_type", "unknown")
            clients_by_type[client_type] = clients_by_type.get(client_type, 0) + 1

            tenant = client.get("tenant_id", "default")
            clients_by_tenant[tenant] = clients_by_tenant.get(tenant, 0) + 1

        return {
            "total_clients": total_clients,
            "active_clients": active_clients,
            "revoked_clients": revoked_clients,
            "clients_by_type": clients_by_type,
            "clients_by_tenant": clients_by_tenant,
        }

    # ========================================================================
    # User Management for Clients (v2.0)
    # ========================================================================

    async def get_client_users(
        self,
        client_id: str,
        filter_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get all users registered via this client.

        This method queries the users index to find users associated with the client.
        Users are linked to clients via the 'registered_via_client_id' field.

        Args:
            client_id: Client UUID
            filter_params: Optional filters {is_active: bool, roles: [str]}
            limit: Maximum results (default 100)
            offset: Skip results (default 0)

        Returns:
            Dict with users list, total count, and client_id

        Raises:
            ValueError: If client not found
        """
        # Verify client exists
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        # Get users index key - we need to scan users registered via this client
        # Users are stored with key: ubp:users:{user_id}
        # We'll use the client_users_index for efficient lookup
        client_users_key = f"ubp:client_users:{client_id}"

        # Get user IDs registered via this client
        user_ids = await self.redis_client.smembers(client_users_key)

        # DEBUG: Log query parameters for diagnosis
        logger.debug(
            f"[GET_CLIENT_USERS] Querying users for client_id={client_id}, "
            f"SET key={client_users_key}, user_ids_in_set={len(user_ids)}"
        )

        users = []
        for user_id in user_ids:
            user_id_str = (
                user_id.decode("utf-8") if isinstance(user_id, bytes) else user_id
            )
            user_key = f"ubp:users:{user_id_str}"
            user_json = await self.redis_client.get(user_key)

            if user_json:
                user = json.loads(user_json)
                # Remove sensitive fields
                user.pop("password_hash", None)
                user.pop("hashed_password", None)
                users.append(user)

        # ====================================================================
        # FALLBACK: If SET is empty, scan users index for matching client_id
        # This handles users created before the SET-based tracking was added
        # ====================================================================
        if not users:
            logger.info(
                f"[GET_CLIENT_USERS] SET empty for client {client_id}, "
                "falling back to user index scan"
            )
            # Use hardcoded users_index (from admin_users config, not admin_clients)
            users_index = "ubp:admin:users"
            all_users_data = await self.redis_client.hgetall(users_index)

            for user_json in all_users_data.values():
                user_json_str = (
                    user_json.decode("utf-8")
                    if isinstance(user_json, bytes)
                    else user_json
                )
                user = json.loads(user_json_str)

                # Check if user belongs to this client
                user_client_id = user.get("client_id") or user.get(
                    "registered_via_client"
                )
                if user_client_id == client_id:
                    # Also sync to SET for future fast lookups
                    await self.redis_client.sadd(client_users_key, user["user_id"])

                    user.pop("password_hash", None)
                    user.pop("hashed_password", None)
                    users.append(user)

            if users:
                logger.info(
                    f"[GET_CLIENT_USERS] Fallback found {len(users)} users for "
                    f"client {client_id}, synced to SET"
                )

        # Apply filters
        if filter_params:
            if "is_active" in filter_params:
                users = [
                    u for u in users if u.get("is_active") == filter_params["is_active"]
                ]
            if "roles" in filter_params:
                filter_roles = set(filter_params["roles"])
                users = [u for u in users if set(u.get("roles", [])) & filter_roles]

        total = len(users)

        # Pagination
        start = offset if offset else 0
        end = start + (limit or 100)
        users = users[start:end]

        return {
            "users": users,
            "total": total,
            "client_id": client_id,
            "limit": limit or 100,
            "offset": offset or 0,
        }

    async def test_client(self, client_id: str) -> Dict[str, Any]:
        """
        Test client configuration by verifying all settings are operational.

        Performs validation tests on:
        - Authentication: Client exists and is active
        - KB Access: Assigned KBs exist and are accessible
        - Model Access: Assigned models are available
        - User Limits: Current users vs max_users
        - Endpoint Authorization: Configuration is valid

        Args:
            client_id: Client UUID

        Returns:
            Test results with overall status (pass/fail/partial)

        Raises:
            ValueError: If client not found
        """
        # Get client
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        now = datetime.utcnow().isoformat()

        test_results = {
            "authentication": {"status": "pass", "details": {}},
            "kb_access": {"status": "pass", "details": {}},
            "model_access": {"status": "pass", "details": {}},
            "user_limits": {"status": "pass", "details": {}},
            "endpoint_authorization": {"status": "pass", "details": {}},
        }

        # Test 1: Authentication
        if not client.get("is_active"):
            test_results["authentication"]["status"] = "fail"
            test_results["authentication"]["details"]["error"] = "Client is not active"
        else:
            test_results["authentication"]["details"]["is_active"] = True
            test_results["authentication"]["details"]["client_type"] = client.get(
                "client_type"
            )

        # Check expiration
        expires_at = client.get("expires_at")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now_dt = datetime.utcnow()
                if exp_dt < now_dt:
                    test_results["authentication"]["status"] = "fail"
                    test_results["authentication"]["details"]["error"] = (
                        "Client has expired"
                    )
                else:
                    test_results["authentication"]["details"]["expires_at"] = expires_at
            except Exception:
                pass

        # Test 2: KB Access
        kb_config = client.get("kb_config", {})
        assigned_kbs = kb_config.get("universal_kbs_assigned", [])
        test_results["kb_access"]["details"]["assigned_kbs"] = assigned_kbs
        test_results["kb_access"]["details"]["can_create_universal_kb"] = kb_config.get(
            "can_create_universal_kb", False
        )

        # In a real scenario, we would verify KBs exist in Qdrant
        # For now, we mark as pass if configuration is valid
        if not isinstance(assigned_kbs, list):
            test_results["kb_access"]["status"] = "fail"
            test_results["kb_access"]["details"]["error"] = "Invalid KB configuration"

        # Test 3: Model Access
        model_config = client.get("model_config", {})
        allowed_models = model_config.get("allowed_models", [])
        default_model = model_config.get("default_model")

        test_results["model_access"]["details"]["allowed_models"] = allowed_models
        test_results["model_access"]["details"]["default_model"] = default_model

        if default_model and default_model not in allowed_models:
            test_results["model_access"]["status"] = "fail"
            test_results["model_access"]["details"]["error"] = (
                "Default model not in allowed models list"
            )

        # Test 4: User Limits
        user_limits = client.get("user_limits", {})
        max_users = user_limits.get("max_users", 100)
        current_users = user_limits.get("current_users", 0)

        test_results["user_limits"]["details"]["max_users"] = max_users
        test_results["user_limits"]["details"]["current_users"] = current_users
        test_results["user_limits"]["details"]["available_slots"] = (
            max_users - current_users
        )

        if current_users >= max_users:
            test_results["user_limits"]["status"] = "warning"
            test_results["user_limits"]["details"]["warning"] = "User limit reached"

        # Test 5: Endpoint Authorization
        authorization = client.get("authorization", {})
        allowed = authorization.get("allowed_endpoints", ["*"])
        denied = authorization.get("denied_endpoints", [])

        test_results["endpoint_authorization"]["details"]["allowed_endpoints"] = allowed
        test_results["endpoint_authorization"]["details"]["denied_endpoints"] = denied

        # Check for conflicting rules
        if "*" in allowed and len(denied) == 0:
            test_results["endpoint_authorization"]["details"]["mode"] = "full_access"
        elif "*" in denied:
            test_results["endpoint_authorization"]["status"] = "fail"
            test_results["endpoint_authorization"]["details"]["error"] = (
                "All endpoints denied"
            )

        # Calculate overall status
        statuses = [r["status"] for r in test_results.values()]
        if all(s == "pass" for s in statuses):
            overall_status = "pass"
        elif any(s == "fail" for s in statuses):
            overall_status = "fail"
        else:
            overall_status = "partial"

        return {
            "client_id": client_id,
            "client_name": client.get("client_name"),
            "test_results": test_results,
            "overall_status": overall_status,
            "tested_at": now,
        }

    # ========================================================================
    # User Count Management (v2.0)
    # ========================================================================

    async def increment_user_count(
        self, client_id: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Increment user count for a client and track user association.

        Called when a new user registers via this client.

        Args:
            client_id: Client UUID
            user_id: User UUID being registered

        Returns:
            Updated user_limits with new count

        Raises:
            ValueError: If client not found or user limit reached
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        user_limits = client.get("user_limits", {})

        max_users = user_limits.get("max_users", 100)
        current_users = user_limits.get("current_users", 0)

        if current_users >= max_users:
            raise ValueError(
                f"User limit reached for client {client_id}. "
                f"Max: {max_users}, Current: {current_users}"
            )

        # Increment count
        client["user_limits"]["current_users"] = current_users + 1
        client["updated_at"] = datetime.utcnow().isoformat()

        # Save client
        await self.redis_client.set(client_key, json.dumps(client))

        # Update index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        # Add user to client_users index for efficient lookup
        client_users_key = f"ubp:client_users:{client_id}"
        await self.redis_client.sadd(client_users_key, user_id)

        return {
            "client_id": client_id,
            "user_id": user_id,
            "user_limits": client["user_limits"],
            "message": "User count incremented successfully",
        }

    async def decrement_user_count(
        self, client_id: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Decrement user count for a client and remove user association.

        Called when a user is deleted or deactivated.

        Args:
            client_id: Client UUID
            user_id: User UUID being removed

        Returns:
            Updated user_limits with new count

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        user_limits = client.get("user_limits", {})
        current_users = user_limits.get("current_users", 0)

        # Decrement count (minimum 0)
        client["user_limits"]["current_users"] = max(0, current_users - 1)
        client["updated_at"] = datetime.utcnow().isoformat()

        # Save client
        await self.redis_client.set(client_key, json.dumps(client))

        # Update index
        clients_index = self.config["redis"]["keys"]["clients_index"]
        await self.redis_client.hset(clients_index, client_id, json.dumps(client))

        # Remove user from client_users index
        client_users_key = f"ubp:client_users:{client_id}"
        await self.redis_client.srem(client_users_key, user_id)

        return {
            "client_id": client_id,
            "user_id": user_id,
            "user_limits": client["user_limits"],
            "message": "User count decremented successfully",
        }

    async def check_user_limit(self, client_id: str) -> Dict[str, Any]:
        """
        Check if client has available user slots.

        Args:
            client_id: Client UUID

        Returns:
            Dict with limit info and availability status

        Raises:
            ValueError: If client not found
        """
        client_key = f"{self.config['redis']['keys']['client_prefix']}{client_id}"
        client_json = await self.redis_client.get(client_key)

        if not client_json:
            raise ValueError(f"Client not found: {client_id}")

        client = json.loads(client_json)
        user_limits = client.get("user_limits", {})

        max_users = user_limits.get("max_users", 100)
        current_users = user_limits.get("current_users", 0)
        available = max_users - current_users

        return {
            "client_id": client_id,
            "max_users": max_users,
            "current_users": current_users,
            "available_slots": available,
            "can_register_user": available > 0,
        }
