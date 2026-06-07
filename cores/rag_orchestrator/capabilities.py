"""Aggregates dynamic capability data for the user facade."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ubp_enterprise_hybrid.backend.app.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover - only for static typing
    from .adapter import RagOrchestratorAdapter


class CapabilityManager:
    """Compute feature availability and defaults for user bootstrap."""

    def __init__(
        self,
        adapter: "RagOrchestratorAdapter",
        settings: Optional[Settings] = None,
    ) -> None:
        self._adapter = adapter
        if settings:
            self._settings = settings
        else:
            from ubp_enterprise_hybrid.backend.app.api.admin_settings_routes import settings_manager
            self._settings = settings_manager.get_settings()
        self._logger = logging.getLogger(__name__)
        self._admin_clients = None

    async def _resolve_admin_clients(self):
        if self._admin_clients is not None:
            return self._admin_clients
        di_container = getattr(self._adapter, "di_container", None)
        if not di_container:
            return None
        try:
            self._admin_clients = await di_container.resolve("admin_clients")
        except Exception as exc:  # pragma: no cover - diagnostic only
            self._logger.warning(
                "Unable to resolve admin_clients for capability lookup: %s", exc
            )
            self._admin_clients = None
        return self._admin_clients

    async def _load_client_config(self, client_id: Optional[str]) -> Dict[str, Any]:
        if not client_id:
            return {}
        admin_clients = await self._resolve_admin_clients()
        if not admin_clients:
            return {}
        try:
            data = await admin_clients.get_client_internal(client_id)
            return data or {}
        except Exception as exc:  # pragma: no cover - diagnostic only
            self._logger.warning(
                "Failed to fetch client config for %s: %s", client_id, exc
            )
            return {}

    @staticmethod
    def _as_bool(store: Dict[str, Any], key: str, default: bool = True) -> bool:
        value = store.get(key)
        return value if isinstance(value, bool) else default

    def _build_enrichment_modes(self) -> List[str]:
        modes: List[str] = []
        enrichment = self._settings.enrichment
        if enrichment.enabled:
            modes.append("standard")
            if getattr(enrichment, "investigative_enabled", False):
                modes.append("investigative")
        return modes

    def _artifact_enabled(self) -> bool:
        artifact_settings = getattr(self._settings, "artifact", None)
        if artifact_settings is None:
            return True
        return getattr(artifact_settings, "enabled", True)

    def _get_report_templates(self) -> List[Dict[str, str]]:
        """Get available report templates from session manager."""
        report_manager = getattr(self._adapter, "report_session_manager", None)
        if not report_manager:
            return []

        try:
            # get_available_templates returns list of template info
            templates = report_manager.get_available_templates()
            if isinstance(templates, list):
                # Extract ID and name for each template
                return [
                    {"id": t.get("id", t.get("template_id", "")), "name": t.get("name", t.get("title", ""))}
                    for t in templates
                    if isinstance(t, dict)
                ]
            return []
        except Exception as exc:
            self._logger.warning("Failed to load report templates: %s", exc)
            return []

    async def get_full_user_context(
        self,
        user_id: str,
        client_id: Optional[str],
        ctx: Any,
    ) -> Dict[str, Any]:
        """Return feature flags and defaults for the calling user."""

        client_config = await self._load_client_config(client_id)
        client_features = client_config.get("features")
        if not isinstance(client_features, dict):
            client_features = {}

        kb_config = client_config.get("kb_config") or {}
        default_user_kb = kb_config.get("default_user_kb_config") or {}

        # System/module availability
        system_web_enabled = getattr(self._settings.web_search, "enabled", True)
        module_web_enabled = bool(getattr(self._adapter, "web_search_module", None))
        system_report_enabled = self._artifact_enabled()
        module_report_enabled = bool(
            getattr(self._adapter, "report_session_manager", None)
        )

        web_allowed = (
            system_web_enabled
            and module_web_enabled
            and self._as_bool(client_features, "web_search", True)
        )
        report_allowed = (
            system_report_enabled
            and module_report_enabled
            and self._as_bool(client_features, "reporting", True)
        )

        enrichment_settings = self._settings.enrichment
        enrichment_caps = {
            "available": bool(enrichment_settings.enabled),
            "steps": {
                # GPU-accelerated (no LLM)
                "rerank": bool(enrichment_settings.rerank_enabled),
                # LLM-dependent
                "hyde": bool(enrichment_settings.hyde_enabled),
                "query_expansion": bool(enrichment_settings.query_expansion_enabled),
                "investigative": bool(enrichment_settings.investigative_enabled),
                # Post-processing
                "query_filters": bool(getattr(enrichment_settings, "query_filters_enabled", True)),
                "fusion": bool(enrichment_settings.fusion_enabled),
                "dedup": bool(enrichment_settings.dedup_enabled),
                "compression": bool(enrichment_settings.compression_enabled),
            },
            "modes": self._build_enrichment_modes(),
        }

        memory_settings = self._settings.memory
        features = {
            "web_search": web_allowed,
            "deep_research": report_allowed,
            "enrichment": enrichment_caps,
            "memory": {
                "type": memory_settings.strategy,
                "enabled": bool(memory_settings.structured_enabled),
            },
            # BUG-USER-003 FIX: Default to True for consistency with _create_personal_kb
            "personal_kb": bool(default_user_kb.get("personal_kb_enabled", True)),
        }

        rag_defaults = self._settings.rag
        defaults = {
            "temperature": rag_defaults.temperature,
            "max_tokens": rag_defaults.max_tokens,
            "top_k": rag_defaults.top_k,
            "model": getattr(self._settings.roles, "rag_provider", None),
        }

        # Build reports capability info (v2.6 UX enhancement)
        template_list = self._get_report_templates() if report_allowed else []
        reports_cap = {
            "enabled": report_allowed,
            "templates": template_list,
            "dynamic_planning": True,  # v2.4+ feature
        }

        # Build artifacts capability info (v2.6 UX enhancement)
        artifacts_cap = {
            "enabled": self._artifact_enabled(),
            "list": True,      # GET /artifacts endpoint available
            "download": True,  # GET /artifacts/{id} endpoint available
            "formats": ["docx", "md", "json", "xlsx", "csv", "pptx", "pdf"],  # All supported formats
        }

        # Add reports and artifacts to features dict
        features["reports"] = reports_cap
        features["artifacts"] = artifacts_cap

        return {
            "features": features,
            "defaults": defaults,
            "client": {
                "features": client_features,
                "kb": default_user_kb,
            },
        }
