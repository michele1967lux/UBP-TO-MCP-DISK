"""
Standalone tests for context_gate module.

Tests auth (deny-by-default) and enrichment (degrade with defaults)
without requiring live Redis or module infrastructure.
"""

import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Ensure module is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ubp_enterprise_hybrid.modules.cores.context_gate.adapter import (
    ContextGateAdapter,
    GateError,
)
from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_ctx(client_id="client_001", user_id="user_abc"):
    """Create a minimal UBPContext-like object."""
    return SimpleNamespace(
        user=SimpleNamespace(client_id=client_id, user_id=user_id)
    )


def _mock_resolver(
    allowed=None,
    feature_flags=None,
    default_pipeline="rag_chat_standard",
    domain_config=None,
    pipeline_overrides=None,
    step_overrides=None,
    client_config=None,
    fail_auth=False,
    fail_enrichment=False,
):
    """Create a mock ClientConfigResolver."""
    resolver = MagicMock()

    if fail_auth:
        resolver.get_allowed_pipelines = AsyncMock(
            side_effect=ConnectionError("Redis unreachable")
        )
        resolver.get_feature_flags = AsyncMock(
            side_effect=ConnectionError("Redis unreachable")
        )
        resolver.get_default_pipeline = AsyncMock(
            side_effect=ConnectionError("Redis unreachable")
        )
    else:
        resolver.get_allowed_pipelines = AsyncMock(
            return_value=allowed or ["*"]
        )
        resolver.get_feature_flags = AsyncMock(
            return_value=feature_flags or {}
        )
        resolver.get_default_pipeline = AsyncMock(
            return_value=default_pipeline
        )

    if fail_enrichment:
        resolver.get_domain_config = AsyncMock(
            side_effect=Exception("enrichment error")
        )
        resolver.resolve_pipeline_config = AsyncMock(
            side_effect=Exception("enrichment error")
        )
        resolver.get_step_overrides = AsyncMock(
            side_effect=Exception("enrichment error")
        )
        resolver._get_client_config = AsyncMock(
            side_effect=Exception("enrichment error")
        )
        resolver.get_dedicated_pipeline = AsyncMock(
            side_effect=Exception("enrichment error")
        )
    else:
        resolver.get_domain_config = AsyncMock(
            return_value=domain_config or {"preset": "generic"}
        )
        resolver.resolve_pipeline_config = AsyncMock(
            return_value=pipeline_overrides or {}
        )
        resolver.get_step_overrides = AsyncMock(
            return_value=step_overrides or {}
        )
        resolver._get_client_config = AsyncMock(
            return_value=client_config or {"pipeline_config": {}}
        )
        resolver.get_dedicated_pipeline = AsyncMock(return_value=None)

    return resolver


# ── Test Classes ──────────────────────────────────────────────────────────


class TestA_AuthDenyByDefault:
    """Gate MUST raise GateError when auth context is missing."""

    @pytest.mark.asyncio
    async def test_ctx_none_raises(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        with pytest.raises(GateError, match="Missing security context"):
            await gate.resolve(ctx=None)

    @pytest.mark.asyncio
    async def test_no_user_raises(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        ctx = SimpleNamespace()  # no .user
        with pytest.raises(GateError, match="Missing client_id"):
            await gate.resolve(ctx=ctx)

    @pytest.mark.asyncio
    async def test_flat_operation_context_fallback_resolves(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver()
        ctx = OperationContext(
            client_id="client_001",
            user_id="user_abc",
            roles=["user"],
            source="mcp",
        )

        result = await gate.resolve(ctx=ctx)
        assert result["client_id"] == "client_001"
        assert result["user_id"] == "user_abc"

    @pytest.mark.asyncio
    async def test_no_client_id_raises(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        ctx = SimpleNamespace(user=SimpleNamespace(client_id=None, user_id="u1"))
        with pytest.raises(GateError, match="Missing client_id"):
            await gate.resolve(ctx=ctx)

    @pytest.mark.asyncio
    async def test_empty_client_id_raises(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        ctx = SimpleNamespace(user=SimpleNamespace(client_id="", user_id="u1"))
        with pytest.raises(GateError, match="Missing client_id"):
            await gate.resolve(ctx=ctx)

    @pytest.mark.asyncio
    async def test_error_count_increments(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        assert gate._error_count == 0
        with pytest.raises(GateError):
            await gate.resolve(ctx=None)
        assert gate._error_count == 1
        with pytest.raises(GateError):
            await gate.resolve(ctx=None)
        assert gate._error_count == 2


class TestB_RedisUnreachable:
    """Gate MUST raise GateError when auth data can't be resolved."""

    @pytest.mark.asyncio
    async def test_redis_down_raises_gate_error(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(fail_auth=True)

        with pytest.raises(GateError, match="Cannot resolve authorization"):
            await gate.resolve(ctx=_make_ctx())

    @pytest.mark.asyncio
    async def test_error_message_includes_client_id(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(fail_auth=True)

        with pytest.raises(GateError, match="client_001"):
            await gate.resolve(ctx=_make_ctx(client_id="client_001"))


class TestC_WhitelistFiltering:
    """Allowed pipelines must be correctly resolved and filtered."""

    @pytest.mark.asyncio
    async def test_wildcard_returns_all(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(allowed=["*"])

        result = await gate.resolve(ctx=_make_ctx())
        assert result["allowed_pipelines"] == ["*"]

    @pytest.mark.asyncio
    async def test_specific_whitelist(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=["rag_chat_standard", "web_search_v2"]
        )

        result = await gate.resolve(ctx=_make_ctx())
        assert result["allowed_pipelines"] == ["rag_chat_standard", "web_search_v2"]

    @pytest.mark.asyncio
    async def test_feature_flag_exclusion(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=["rag_chat_standard", "web_search_pipeline", "report_full"],
            feature_flags={"web_search_enabled": False},
        )

        result = await gate.resolve(ctx=_make_ctx())
        assert "web_search_pipeline" not in result["allowed_pipelines"]
        assert "rag_chat_standard" in result["allowed_pipelines"]
        assert "report_full" in result["allowed_pipelines"]

    @pytest.mark.asyncio
    async def test_multiple_feature_flag_exclusions(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=[
                "rag_chat_standard", "web_search_pipeline",
                "report_full", "research_pipeline",
            ],
            feature_flags={
                "web_search_enabled": False,
                "report_generation_enabled": False,
                "investigation_enabled": False,
            },
        )

        result = await gate.resolve(ctx=_make_ctx())
        assert result["allowed_pipelines"] == ["rag_chat_standard"]

    @pytest.mark.asyncio
    async def test_wildcard_not_filtered_by_features(self):
        """Wildcard can't be enumerated — exclusions handled by execute() safety net."""
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=["*"],
            feature_flags={"web_search_enabled": False},
        )

        result = await gate.resolve(ctx=_make_ctx())
        assert result["allowed_pipelines"] == ["*"]


class TestD_SuccessfulResolve:
    """Full successful resolve with all data."""

    @pytest.mark.asyncio
    async def test_full_resolve(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=["rag_chat_v2", "web_search_v2"],
            feature_flags={"rag_chat_v2_enabled": True, "web_search_v2_enabled": True},
            default_pipeline="rag_chat_v2",
            domain_config={"domain": "medical"},
            pipeline_overrides={"top_k": 5, "temperature": 0.3},
            client_config={
                "pipeline_config": {
                    "global_settings": {"max_context_tokens": 4096}
                }
            },
        )

        result = await gate.resolve(
            ctx=_make_ctx(client_id="c1", user_id="u1"),
            pipeline_name="rag_chat_v2",
        )

        assert result["client_id"] == "c1"
        assert result["user_id"] == "u1"
        assert result["allowed_pipelines"] == ["rag_chat_v2", "web_search_v2"]
        assert result["feature_flags"]["rag_chat_v2_enabled"] is True
        assert result["default_pipeline"] == "rag_chat_v2"
        assert result["domain_preset"] == "medical"
        assert result["settings"]["top_k"] == 5
        assert result["settings"]["max_context_tokens"] == 4096
        assert result["enrichment_degraded"] is False
        assert result["resolved_at_ms"] >= 0

    @pytest.mark.asyncio
    async def test_resolve_count_increments(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver()

        assert gate._resolve_count == 0
        await gate.resolve(ctx=_make_ctx())
        assert gate._resolve_count == 1
        await gate.resolve(ctx=_make_ctx())
        assert gate._resolve_count == 2


class TestE_EnrichmentDegradation:
    """Enrichment failure must NOT block the pipeline — return defaults."""

    @pytest.mark.asyncio
    async def test_enrichment_fails_auth_succeeds(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            allowed=["rag_chat_standard"],
            feature_flags={"web_search_enabled": True},
            fail_enrichment=True,
        )

        result = await gate.resolve(ctx=_make_ctx())
        # Auth data must be present
        assert result["allowed_pipelines"] == ["rag_chat_standard"]
        assert result["feature_flags"]["web_search_enabled"] is True
        # Enrichment degraded
        assert result["enrichment_degraded"] is True
        assert result["domain_preset"] == "generic"
        assert result["settings"] == {}
        assert result["pipeline_overrides"] == {}

    @pytest.mark.asyncio
    async def test_user_preferences_unavailable(self):
        """Missing user_profile_memory → no crash, empty prefs."""
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver()
        # No user_profile_memory set

        result = await gate.resolve(ctx=_make_ctx())
        assert result["user_preferences"] == {}
        assert result["enrichment_degraded"] is False


class TestF_UserPreferenceMerge:
    """User preferences override client settings."""

    @pytest.mark.asyncio
    async def test_user_prefs_override_client(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            pipeline_overrides={"top_k": 10, "temperature": 0.5},
            client_config={
                "pipeline_config": {"global_settings": {"max_results": 20}}
            },
        )

        # Mock user_profile_memory
        mock_upm = MagicMock()
        mock_upm.call_operation = AsyncMock(
            return_value={"top_k": 3, "preferred_language": "it"}
        )
        gate.set_user_profile_memory(mock_upm)

        result = await gate.resolve(
            ctx=_make_ctx(), pipeline_name="rag_chat_v2"
        )

        # top_k: user(3) overrides client(10)
        assert result["settings"]["top_k"] == 3
        # preferred_language: from user only
        assert result["settings"]["preferred_language"] == "it"
        # max_results: from client (user didn't override)
        assert result["settings"]["max_results"] == 20
        # User prefs captured
        assert result["user_preferences"]["top_k"] == 3


class TestG_DomainPreset:
    """Domain preset resolution."""

    @pytest.mark.asyncio
    async def test_medical_domain(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(
            domain_config={"domain": "medical", "citation_required": True}
        )

        result = await gate.resolve(ctx=_make_ctx())
        assert result["domain_preset"] == "medical"

    @pytest.mark.asyncio
    async def test_no_domain_defaults_generic(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(domain_config={})

        result = await gate.resolve(ctx=_make_ctx())
        assert result["domain_preset"] == "generic"

    @pytest.mark.asyncio
    async def test_none_domain_defaults_generic(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver(domain_config=None)

        result = await gate.resolve(ctx=_make_ctx())
        assert result["domain_preset"] == "generic"


class TestH_CallOperation:
    """Dispatch via call_operation (ModuleLoader interface)."""

    @pytest.mark.asyncio
    async def test_dispatch_resolve(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver()

        result = await gate.call_operation("resolve", ctx=_make_ctx())
        assert "allowed_pipelines" in result

    @pytest.mark.asyncio
    async def test_dispatch_initialize(self):
        gate = ContextGateAdapter()
        result = await gate.call_operation("initialize")
        assert result["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_dispatch_health_check(self):
        gate = ContextGateAdapter()
        result = await gate.call_operation("health_check")
        assert result["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_raises(self):
        gate = ContextGateAdapter()
        with pytest.raises(ValueError, match="Unknown operation"):
            await gate.call_operation("nonexistent")


class TestI_HealthCheck:
    """Health check returns accurate metrics."""

    @pytest.mark.asyncio
    async def test_health_after_resolves(self):
        gate = ContextGateAdapter()
        await gate.initialize()
        gate._resolver = _mock_resolver()

        await gate.resolve(ctx=_make_ctx())
        await gate.resolve(ctx=_make_ctx())

        with pytest.raises(GateError):
            await gate.resolve(ctx=None)

        health = await gate.health_check()
        assert health["status"] == "healthy"
        assert health["resolve_count"] == 2
        assert health["error_count"] == 1


class TestJ_AdminClientsInjection:
    """set_admin_clients propagates to resolver."""

    @pytest.mark.asyncio
    async def test_set_admin_clients(self):
        gate = ContextGateAdapter()
        gate._resolver = _mock_resolver()

        mock_admin = MagicMock()
        gate.set_admin_clients(mock_admin)

        assert gate._admin_clients is mock_admin
        gate._resolver.set_admin_clients.assert_called_once_with(mock_admin)
