"""
citations_verifier/adapter.py

UBP framework bridge — DI resolution, EventBus, lifecycle, Redis, security.

Operations:
- health_check: Module status
- verify_document: Full verification pipeline
- verify_claim: Single claim verification
- get_trusted_sources: Trust list access (used by other modules)
- manage_trust_list: CRUD on trust lists
- discover_trusted_sources: Auto-discovery of new sources
- get_search_filter: Quick search filter for web_search integration

v1.0.0 — 2026-02-15
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .providers import (
    # Config
    VerificationConfig,
    TrustListConfig,
    DiscoveryConfig,
    SearchIntegrationConfig,
    # Components
    ClaimExtractor,
    RAGVerifier,
    WebVerifier,
    TrustListManager,
    TrustListDiscovery,
    SearchFilterBuilder,
    VerificationOrchestrator,
    # Models
    TrustedSource,
    ClaimStatus,
)

# CV-HIGH-001 fix: ProviderMapper for LLM delegation fallback
try:
    from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper, ProviderConfigurationError
    PROVIDER_MAPPER_AVAILABLE = True
except ImportError:
    PROVIDER_MAPPER_AVAILABLE = False
    ProviderMapper = None
    ProviderConfigurationError = Exception

logger = logging.getLogger(__name__)


# ============================================================================
# ENV resolver
# ============================================================================


def _resolve_env(value: Any) -> Any:
    """Resolve ${VAR:-default} patterns."""
    if not isinstance(value, str):
        return value
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    def _replace(m):
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
    resolved = re.sub(pattern, _replace, value)
    if resolved.lower() in ("true", "yes", "1"):
        return True
    if resolved.lower() in ("false", "no", "0"):
        return False
    try:
        return int(resolved) if "." not in resolved else float(resolved)
    except (ValueError, TypeError):
        return resolved


def _load_config(module_path: Path) -> Dict[str, Any]:
    """Load and resolve config.json."""
    config_path = module_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = json.load(f)

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        return _resolve_env(obj)

    return _walk(raw)


# ============================================================================
# Adapter
# ============================================================================


class CitationsVerifierAdapter:
    """
    UBP adapter for the citations_verifier module.

    Lifecycle:
    - initialize(): Load config, create components, seed trust lists
    - verify_document(): Full pipeline
    - get_trusted_sources(): Used by other modules as source filter
    - discover_trusted_sources(): Auto-populate trust lists
    - shutdown(): Clean up
    """

    def __init__(
        self,
        module_path: Path,
        di_container: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        self.module_path = Path(module_path)
        self.di_container = di_container
        self.event_bus = event_bus
        self._initialized = False

        # Config
        self._config: Dict[str, Any] = {}
        self._verification_config = VerificationConfig()
        self._trust_config = TrustListConfig()
        self._discovery_config = DiscoveryConfig()
        self._search_config = SearchIntegrationConfig()

        # Components
        self._extractor: Optional[ClaimExtractor] = None
        self._rag_verifier: Optional[RAGVerifier] = None
        self._web_verifier: Optional[WebVerifier] = None
        self._trust_manager: Optional[TrustListManager] = None
        self._discovery: Optional[TrustListDiscovery] = None
        self._filter_builder: Optional[SearchFilterBuilder] = None
        self._orchestrator: Optional[VerificationOrchestrator] = None

        # DI dependencies (lazy)
        self._web_search: Optional[Any] = None
        self._llm_module: Optional[Any] = None
        self._redis: Optional[Any] = None

    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    def _build_context_from_di(self) -> OperationContext:
        """Build OperationContext from DI — backward compatibility for REST path."""
        return OperationContext(
            client_id="default",
            user_id=None,
            session_id=None,
            source="rest",
        )

    def _normalize_ctx(self, ctx: Any) -> OperationContext:
        """Normalize any context format to OperationContext."""
        if ctx is None:
            return self._build_context_from_di()
        if isinstance(ctx, OperationContext):
            return ctx
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
        return self._build_context_from_di()

    # ===================================================================
    # Lifecycle
    # ===================================================================

    async def initialize(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Initialize module: load config, create components, seed trust lists."""
        if self._initialized:
            return {"status": "already_initialized"}

        # Load config
        self._config = _load_config(self.module_path)
        self._build_configs()

        # Create components
        self._extractor = ClaimExtractor(self._verification_config)
        self._rag_verifier = RAGVerifier(self._verification_config)
        self._web_verifier = WebVerifier(self._verification_config)
        self._trust_manager = TrustListManager(self._trust_config)
        self._discovery = TrustListDiscovery(self._discovery_config)
        self._filter_builder = SearchFilterBuilder(self._search_config)
        self._orchestrator = VerificationOrchestrator(
            config=self._verification_config,
            claim_extractor=self._extractor,
            rag_verifier=self._rag_verifier,
            web_verifier=self._web_verifier,
            trust_manager=self._trust_manager,
        )

        # Resolve DI
        await self._resolve_dependencies()

        # Seed trust lists
        predefined = self._config.get("predefined_lists", {})
        total = await self._trust_manager.initialize(self._redis, predefined)

        self._initialized = True
        logger.info(f"[CITATIONS] Initialized with {total} trusted domains")
        return {
            "status": "initialized",
            "trusted_domains_loaded": total,
            "trust_lists": self._trust_manager.get_all_domain_names(),
        }

    async def shutdown(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Clean shutdown."""
        self._initialized = False
        logger.info("[CITATIONS] Shut down")
        return {"status": "shutdown"}

    async def health_check(self, ctx=None, **kwargs) -> Dict[str, Any]:
        """Module health status."""
        if not self._initialized:
            return {"module": "citations_verifier", "status": "not_initialized"}

        ws_ok = self._web_search is not None
        redis_ok = self._redis is not None
        llm_ok = self._llm_module is not None
        lists = len(self._trust_manager.get_all_domain_names()) if self._trust_manager else 0
        total = self._trust_manager.get_total_domains() if self._trust_manager else 0

        status = "healthy" if ws_ok else "degraded"

        return {
            "module": "citations_verifier",
            "status": status,
            "web_search_available": ws_ok,
            "redis_connected": redis_ok,
            "llm_available": llm_ok,
            "trust_lists_loaded": lists,
            "total_trusted_domains": total,
            "config": {
                "default_depth": self._verification_config.default_depth,
                "filter_mode": self._search_config.filter_mode,
            },
        }

    # ===================================================================
    # Core Operations
    # ===================================================================

    async def verify_document(
        self,
        text: str,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None,
        verification_depth: Optional[str] = None,
        language: str = "it",
        max_claims: int = 0,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full document verification pipeline.

        Extracts claims, verifies against RAG chunks and/or web,
        produces trust score and per-claim evidence.
        """
        if not self._initialized:
            await self.initialize(ctx)

        depth = verification_depth or self._verification_config.default_depth
        ws = await self._get_web_search()
        llm = await self._get_llm_module()

        report = await self._orchestrator.verify_document(
            text=text,
            rag_chunks=rag_chunks,
            domain=domain,
            depth=depth,
            language=language,
            web_search=ws,
            llm=llm,
            max_claims=max_claims,
        )

        result = report.to_dict()

        # Publish events
        if self.event_bus:
            try:
                await self.event_bus.publish(
                    "citations.verification_completed",
                    {
                        "trust_score": report.trust_score,
                        "claims_total": report.claims_total,
                        "claims_contradicted": report.claims_contradicted,
                        "depth": depth,
                    },
                )

                if report.claims_contradicted > 0:
                    contradicted = [c for c in report.claims if c.status == ClaimStatus.CONTRADICTED]
                    for c in contradicted:
                        await self.event_bus.publish(
                            "citations.claim_contradicted",
                            c.to_dict(),
                        )
            except Exception:
                pass

        logger.info(
            f"[CITATIONS] Document verified: score={report.trust_score:.2f} "
            f"claims={report.claims_total} verified={report.claims_verified} "
            f"contradicted={report.claims_contradicted}"
        )

        return result

    async def verify_claim(
        self,
        claim: str,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
        domain: Optional[str] = None,
        web_verify: bool = True,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Verify a single claim."""
        if not self._initialized:
            await self.initialize(ctx)

        from .providers import Claim
        claim_obj = Claim(text=claim)

        ws = await self._get_web_search() if web_verify else None
        llm = await self._get_llm_module()

        trusted = []
        if domain and self._trust_manager:
            trusted = self._trust_manager.get_domains(domain, min_score=0.7)

        result = await self._orchestrator._verify_single_claim(
            claim=claim_obj,
            rag_chunks=rag_chunks,
            trusted_domains=trusted,
            depth="standard" if web_verify else "quick",
            language="it",
            web_search=ws,
            llm=llm,
        )

        return result.to_dict()

    # ===================================================================
    # Trust List Operations
    # ===================================================================

    async def get_trusted_sources(
        self,
        domain: str,
        min_trust_score: float = 0.7,
        format: str = "domains",
        exclusive: bool = False,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get trust list for a domain.

        This is the PRIMARY interface for other modules to access trust lists.

        Formats:
        - domains: List of URL strings
        - full: Full TrustedSource objects with metadata
        - searxng: "site:x OR site:y" filter for SearXNG queries
        """
        if not self._initialized:
            await self.initialize(ctx)

        sources = await self._trust_manager.get_list(domain, min_score=min_trust_score)

        if format == "domains":
            sites = [s.url for s in sources]
            search_filter = " OR ".join(f"site:{s}" for s in sites[:10])
            return {
                "domain": domain,
                "sources": sites,
                "count": len(sites),
                "exclusive": exclusive,
                "search_filter": search_filter,
            }
        elif format == "searxng":
            filter_str = self._trust_manager.build_search_filter(
                domain, min_score=min_trust_score,
                max_sites=self._search_config.max_sites_in_filter,
            )
            return {
                "domain": domain,
                "search_filter": filter_str,
                "exclusive": exclusive,
                "count": len(sources),
            }
        else:  # full
            return {
                "domain": domain,
                "sources": [s.to_dict() for s in sources],
                "count": len(sources),
                "exclusive": exclusive,
            }

    async def manage_trust_list(
        self,
        action: str,
        domain: Optional[str] = None,
        entries: Optional[List[Dict[str, Any]]] = None,
        target_url: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """CRUD operations on trust lists."""
        if not self._initialized:
            await self.initialize(ctx)

        if action == "list_all":
            all_domains = self._trust_manager.get_all_domain_names()
            result = {}
            for d in all_domains:
                sources = self._trust_manager.get_list_sync(d)
                result[d] = {
                    "count": len(sources),
                    "top_score": sources[0].trust_score if sources else 0,
                    "sample": [s.url for s in sources[:3]],
                }
            return {"action": "list_all", "success": True, "domains": result}

        if not domain:
            return {"action": action, "success": False, "error": "domain required"}

        if action == "add" and entries:
            sources = [
                TrustedSource(
                    url=e.get("url", ""),
                    trust_score=float(e.get("trust_score", e.get("score", 0.7))),
                    notes=e.get("notes", ""),
                )
                for e in entries if e.get("url")
            ]
            count = await self._trust_manager.add_entries(domain, sources, self._redis)
            self._publish_trust_update(domain, "add", count)
            return {
                "action": "add", "success": True, "affected_count": count,
                "domain": domain, "current_count": len(self._trust_manager.get_list_sync(domain)),
            }

        elif action == "remove" and target_url:
            removed = await self._trust_manager.remove_entry(domain, target_url, self._redis)
            if removed:
                self._publish_trust_update(domain, "remove", 1)
            return {
                "action": "remove", "success": removed,
                "affected_count": 1 if removed else 0, "domain": domain,
                "current_count": len(self._trust_manager.get_list_sync(domain)),
            }

        elif action == "update" and target_url:
            score = kwargs.get("trust_score")
            notes = kwargs.get("notes")
            updated = await self._trust_manager.update_entry(
                domain, target_url, trust_score=score, notes=notes, redis=self._redis,
            )
            return {
                "action": "update", "success": updated,
                "affected_count": 1 if updated else 0, "domain": domain,
            }

        elif action == "create_list":
            created = await self._trust_manager.create_list(domain, redis=self._redis)
            return {"action": "create_list", "success": created, "domain": domain}

        elif action == "delete_list":
            deleted = await self._trust_manager.delete_list(domain, self._redis)
            return {"action": "delete_list", "success": deleted, "domain": domain}

        return {"action": action, "success": False, "error": f"Unknown action: {action}"}

    async def discover_trusted_sources(
        self,
        domain: str,
        seed_queries: Optional[List[str]] = None,
        min_score_to_add: float = 0.8,
        max_discoveries: int = 20,
        auto_add: bool = False,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Discover and score new trusted sources via web analysis.
        """
        if not self._initialized:
            await self.initialize(ctx)

        ws = await self._get_web_search()
        if not ws:
            return {"error": "web_search not available", "discovered": []}

        existing = set(self._trust_manager.get_domains(domain))

        discoveries = await self._discovery.discover(
            domain=domain,
            web_search=ws,
            existing_urls=existing,
            seed_queries=seed_queries,
            max_discoveries=max_discoveries,
        )

        # Auto-add if enabled
        auto_added = 0
        proposed = 0
        already_known = 0

        for d in discoveries:
            if d.status == "already_known":
                already_known += 1
            elif auto_add and d.trust_score >= min_score_to_add:
                source = TrustedSource(
                    url=d.url,
                    trust_score=d.trust_score,
                    notes=f"Auto-discovered: {json.dumps(d.authority_signals)}",
                    auto_discovered=True,
                )
                added = await self._trust_manager.add_entries(domain, [source], self._redis)
                if added > 0:
                    d.status = "auto_added"
                    auto_added += 1
            else:
                proposed += 1

        # Publish event
        if self.event_bus and discoveries:
            try:
                await self.event_bus.publish(
                    "citations.sources_discovered",
                    {
                        "domain": domain,
                        "total": len(discoveries),
                        "auto_added": auto_added,
                        "proposed": proposed,
                    },
                )
            except Exception:
                pass

        logger.info(
            f"[CITATIONS] Discovery for '{domain}': "
            f"{len(discoveries)} found, {auto_added} auto-added, {proposed} proposed"
        )

        return {
            "discovered": [d.to_dict() for d in discoveries],
            "auto_added": auto_added,
            "proposed": proposed,
            "already_known": already_known,
            "search_queries_used": seed_queries or self._discovery.DOMAIN_SEED_QUERIES.get(domain, []),
        }

    async def get_search_filter(
        self,
        domain: str,
        exclusive: bool = False,
        max_sites: int = 0,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Quick method: returns search filter for web_search module.

        Designed to be called inline during search query construction:
        filter = await citations.get_search_filter("medical", exclusive=True)
        query = f"{user_query} ({filter['filter_query']})"
        """
        if not self._initialized:
            await self.initialize(ctx)

        return self._filter_builder.build_filter(
            trust_manager=self._trust_manager,
            domain=domain,
            exclusive=exclusive,
            max_sites=max_sites,
        )

    # ===================================================================
    # Event Handlers
    # ===================================================================

    async def on_report_generated(self, event: Any) -> None:
        """Auto-verify reports when report.generated event is received."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            text = payload.get("text", payload.get("content", payload.get("full_draft", "")))
            chunks = payload.get("rag_chunks", payload.get("chunks", []))
            domain = payload.get("domain")

            if text and len(text) > 100:
                await self.verify_document(
                    text=text,
                    rag_chunks=chunks,
                    domain=domain,
                    verification_depth="standard",
                )
        except Exception as e:
            logger.warning(f"[CITATIONS] Auto-verify on report.generated failed: {e}")

    async def on_rag_response_ready(self, event: Any) -> None:
        """Auto-verify RAG responses."""
        try:
            payload = event.payload if hasattr(event, "payload") else event
            text = payload.get("response", payload.get("text", ""))
            chunks = payload.get("chunks", [])

            if text and len(text) > 200 and chunks:
                await self.verify_document(
                    text=text,
                    rag_chunks=chunks,
                    verification_depth="quick",
                )
        except Exception as e:
            logger.warning(f"[CITATIONS] Auto-verify on rag.response_ready failed: {e}")

    # ===================================================================
    # DI Resolution
    # ===================================================================

    async def _resolve_dependencies(self) -> None:
        """Resolve DI dependencies."""
        await self._get_web_search()
        await self._get_llm_module()
        await self._get_redis()

    async def _get_web_search(self) -> Optional[Any]:
        if self._web_search:
            return self._web_search
        if self.di_container:
            try:
                self._web_search = await self.di_container.resolve("web_search")
                if self._web_search:
                    logger.info("[CITATIONS] web_search resolved from DI")
            except Exception as e:
                logger.warning(f"[CITATIONS] web_search not available: {e}")
        return self._web_search

    async def _get_llm_module(self) -> Optional[Any]:
        if self._llm_module:
            return self._llm_module
        if not self.di_container:
            return None

        # CV-HIGH-001: ProviderMapper fallback chain (same pattern as BUG-001→005)
        if PROVIDER_MAPPER_AVAILABLE and ProviderMapper:
            try:
                provider_chain = ProviderMapper.resolve_chain("enrichment")
            except ProviderConfigurationError as exc:
                logger.warning(f"[CITATIONS] ProviderMapper config error: {exc}")
                provider_chain = None

            if provider_chain:
                for module_name, provider_name in provider_chain:
                    try:
                        resolved = await self.di_container.resolve(module_name)
                        if resolved:
                            self._llm_module = resolved
                            logger.info(
                                f"[CITATIONS] LLM resolved via ProviderMapper: "
                                f"module='{module_name}', provider='{provider_name}'"
                            )
                            return self._llm_module
                        logger.warning(
                            f"[CITATIONS] FALLBACK: module '{module_name}' "
                            f"(provider '{provider_name}') not ready, trying next"
                        )
                    except Exception as e:
                        logger.warning(f"[CITATIONS] FALLBACK: module '{module_name}' error: {e}")
                        continue
                logger.warning("[CITATIONS] FALLBACK EXHAUSTED: no LLM from ProviderMapper chain")

        # Fallback to hardcoded defaults (original behavior)
        try:
            self._llm_module = await self.di_container.resolve("inference_vllm")
            if not self._llm_module:
                self._llm_module = await self.di_container.resolve("inference_grok")
            if self._llm_module:
                logger.info("[CITATIONS] LLM module resolved from DI (legacy fallback)")
        except Exception as e:
            logger.warning(f"[CITATIONS] LLM module not available: {e}")
        return self._llm_module

    async def _get_redis(self) -> Optional[Any]:
        if self._redis:
            return self._redis
        if self.di_container:
            try:
                import redis.asyncio as aioredis
                self._redis = await self.di_container.resolve(aioredis.Redis)
                if self._redis:
                    logger.info("[CITATIONS] Redis resolved from DI")
            except Exception as e:
                logger.warning(f"[CITATIONS] Redis not available: {e}")
        return self._redis

    # ===================================================================
    # Internal helpers
    # ===================================================================

    def _publish_trust_update(self, domain: str, action: str, count: int) -> None:
        """Fire-and-forget trust list update event."""
        if self.event_bus:
            try:
                asyncio.create_task(self.event_bus.publish(
                    "citations.trust_list_updated",
                    {"domain": domain, "action": action, "affected": count},
                ))
            except Exception:
                pass

    def _build_configs(self) -> None:
        """Build typed configs from raw dict."""
        v = self._config.get("verification", {})
        t = self._config.get("trust_lists", {})
        d = self._config.get("discovery", {})
        s = self._config.get("search_integration", {})

        self._verification_config = VerificationConfig(
            enabled=v.get("enabled", True),
            default_depth=str(v.get("default_depth", "standard")),
            max_claims_per_document=int(v.get("max_claims_per_document", 50)),
            max_parallel_verifications=int(v.get("max_parallel_verifications", 5)),
            claim_min_length=int(v.get("claim_min_length", 15)),
            confidence_threshold_supported=float(v.get("confidence_threshold_supported", 0.75)),
            confidence_threshold_partial=float(v.get("confidence_threshold_partial", 0.45)),
            web_search_per_claim=int(v.get("web_search_per_claim", 2)),
            timeout_per_claim_s=int(v.get("timeout_per_claim_s", 30)),
        )

        self._trust_config = TrustListConfig(
            storage_backend=str(t.get("storage_backend", "redis")),
            redis_prefix=str(t.get("redis_prefix", "ubp:trust_list")),
            json_backup_path=str(t.get("json_backup_path", "data/trust_lists")),
            auto_backup_on_change=t.get("auto_backup_on_change", True),
            default_trust_score=float(t.get("default_trust_score", 0.7)),
            score_decay_days=int(t.get("score_decay_days", 90)),
            max_domains_per_list=int(t.get("max_domains_per_list", 200)),
        )

        signals_raw = d.get("authority_signals", "tld,https,domain_age,academic_refs")
        if isinstance(signals_raw, str):
            signals_list = [s.strip() for s in signals_raw.split(",")]
        else:
            signals_list = list(signals_raw)

        self._discovery_config = DiscoveryConfig(
            enabled=d.get("enabled", True),
            searches_per_domain=int(d.get("searches_per_domain", 5)),
            min_appearances=int(d.get("min_appearances", 3)),
            authority_signals=signals_list,
            auto_add_threshold=float(d.get("auto_add_threshold", 0.85)),
            cooldown_hours=int(d.get("cooldown_hours", 24)),
        )

        exclusive_raw = s.get("exclusive_domains", "")
        if isinstance(exclusive_raw, str) and exclusive_raw:
            exclusive_list = [d.strip() for d in exclusive_raw.split(",") if d.strip()]
        elif isinstance(exclusive_raw, list):
            exclusive_list = exclusive_raw
        else:
            exclusive_list = []

        self._search_config = SearchIntegrationConfig(
            filter_mode=str(s.get("filter_mode", "prioritize")),
            max_sites_in_filter=int(s.get("max_sites_in_filter", 10)),
            fallback_to_generic=s.get("fallback_to_generic", True),
            exclusive_domains=exclusive_list,
        )
