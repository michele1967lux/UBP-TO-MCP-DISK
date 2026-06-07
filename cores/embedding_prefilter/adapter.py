"""
embedding_prefilter/adapter.py — UBP Framework Bridge (Security + DI + Operations).

Follows the UBP 3-file pattern:
- adapter.py: security, DI, lifecycle, event publishing, R1 Safe Mode
- providers.py: pure routing logic (EmbeddingPrefilterEngine)
- schemas.py: data models

R1 Safe Mode Fallback:
  - asyncio.wait_for wrapper with configurable timeout (default 500ms)
  - On timeout/exception: returns LLM_ROUTER deferral
  - This is the ONLY reinforcement active in Phase 1

v1.0.0: Initial release (Phase 1 — Stabilization)

MCP-COMPAT (ARCH-008): Added OperationContext support for dual REST/MCP compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

from .schemas import PreRouteDecision

logger = logging.getLogger("ubp.embedding_prefilter")


def _load_config(module_path: Path) -> dict:
    cfg_path = module_path / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


class EmbeddingPrefilterAdapter:
    """Main adapter for embedding_prefilter module.

    Follows LLMRouterAdapter pattern (ARCH-005).
    Manages lifecycle, DI resolution, R1 safe mode, and event publishing.
    """

    def __init__(self, module_path, di_container=None, event_bus=None, **kwargs):
        if isinstance(module_path, str):
            module_path = Path(module_path)
        self.module_path = module_path
        self.di_container = di_container
        self.event_bus = event_bus
        self.config = _load_config(module_path)

        # Set during initialize()
        self._engine = None           # EmbeddingPrefilterEngine (providers.py)
        self._rag_qdrant = None       # rag_qdrant module (for embed_fn)
        self._profile_memory = None   # user_profile_memory (optional, for R3/R4 future)
        self._inference_vllm = None   # inference_vllm module (optional, for R2 vLLM disambiguation)
        self._redis = None
        self._initialized = False
        self._dedicated_model = None  # Standalone SentenceTransformer (avoids rag_qdrant model-swap)

        # R1 Safe Mode timeout (configurable via env UBP_PREFILTER__TIMEOUT_MS or config)
        r1_cfg = self.config.get("reinforcements", {}).get("r1_safe_mode", {})
        self._timeout_ms = r1_cfg.get("timeout_ms", 500)
        self._timeout_sec = self._timeout_ms / 1000.0

    # ------------------------------------------------------------------
    # MCP-COMPAT: OperationContext helpers (ARCH-008)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, **kwargs):
        """Resolve dependencies via DI container and initialize engine."""
        if self._initialized:
            return

        from .providers import EmbeddingPrefilterEngine

        # Resolve rag_qdrant (required — provides embed_fn)
        try:
            self._rag_qdrant = await self.di_container.resolve("rag_qdrant")
        except Exception as e:
            logger.error(f"[PREFILTER] Cannot resolve rag_qdrant: {e}")
            raise

        # Resolve Redis (optional)
        try:
            import redis.asyncio as aioredis
            self._redis = await self.di_container.resolve(aioredis.Redis)
        except Exception:
            logger.info("[PREFILTER] Redis not available — stability features disabled")

        # Resolve optional dependencies
        try:
            self._profile_memory = await self.di_container.resolve("user_profile_memory")
        except Exception:
            logger.info("[PREFILTER] user_profile_memory not available")

        # Resolve inference_vllm (optional — for R2 vLLM disambiguation)
        try:
            self._inference_vllm = await self.di_container.resolve("inference_vllm")
            logger.info("[PREFILTER] inference_vllm resolved for R2 disambiguation")
        except Exception:
            logger.info("[PREFILTER] inference_vllm not available — R2 uses static options")

        # Get embed_fn from rag_qdrant
        embed_fn = self._get_embed_fn()

        # Create engine with optional Qdrant loader and search fn
        qdrant_loader = self._create_qdrant_loader()
        qdrant_search_fn = self._create_qdrant_search_fn()
        vllm_fn = self._create_vllm_fn()
        self._engine = EmbeddingPrefilterEngine(
            embed_fn=embed_fn,
            config=self.config,
            qdrant_loader=qdrant_loader,
            qdrant_search_fn=qdrant_search_fn,
            vllm_fn=vllm_fn,
        )

        # Bootstrap: idempotent seed upsert (before centroid init so centroids see fresh data)
        await self._maybe_bootstrap()

        # Initialize centroids
        init_result = await self._engine.initialize()

        self._initialized = True
        logger.info(
            f"[PREFILTER] Initialized: clusters={init_result.get('clusters')}, "
            f"timeout={self._timeout_ms}ms, "
            f"elapsed={init_result.get('elapsed_ms', 0)}ms"
        )

    # ------------------------------------------------------------------
    # Bootstrap: idempotent seed → Qdrant upsert
    # ------------------------------------------------------------------

    async def _maybe_bootstrap(self):
        """Idempotent seed upsert into Qdrant on startup.

        Reads config from layer1_knn.bootstrap (config.json) with env overrides
        from PrefilterSettings. Does nothing if bootstrap is disabled.

        Safety:
        - NO delete, recreate_collection, or forced reload.
        - Uses deterministic SHA1→uint64 point IDs from seed file.
        - Repeated runs produce identical state (idempotent upsert).
        - Hash-based change detection skips upsert if seed unchanged.
        """
        bootstrap_cfg = self.config.get("layer1_knn", {}).get("bootstrap", {})
        enabled = bootstrap_cfg.get("enabled", False)

        # Env override takes precedence
        import os
        env_enabled = os.getenv("UBP_PREFILTER__BOOTSTRAP_ENABLED")
        if env_enabled is not None:
            enabled = env_enabled.lower() in ("true", "1", "yes")

        if not enabled:
            logger.info("[PREFILTER-BOOTSTRAP] bootstrap_enabled=false | action=DISABLED")
            return

        seed_dir = os.getenv(
            "UBP_PREFILTER__BOOTSTRAP_SEED_DIR",
            bootstrap_cfg.get("seed_dir", "data/seeds"),
        )
        batch_size = bootstrap_cfg.get("embedding_batch_size", 64)
        auto_rebootstrap = bootstrap_cfg.get("auto_rebootstrap_on_seed_change", False)
        collection = self.config.get("layer1_knn", {}).get("collection", "routing_prototypes_v2")

        # Locate seed file
        seed_path = Path(seed_dir) / "routing_prototypes_v2.json"
        if not seed_path.is_absolute():
            # Resolve relative to repo root (container: /app/)
            for base in [Path("/app"), Path.cwd(), self.module_path.parent.parent.parent.parent]:
                candidate = base / seed_path
                if candidate.exists():
                    seed_path = candidate
                    break

        if not seed_path.exists():
            logger.warning(f"[PREFILTER-BOOTSTRAP] Seed file not found: {seed_path}")
            return

        # Load seed
        import hashlib
        seed_data = seed_path.read_text(encoding="utf-8")
        seed_hash = hashlib.sha256(seed_data.encode("utf-8")).hexdigest()[:16]

        seed_json = json.loads(seed_data)
        prototypes = seed_json.get("prototypes", [])
        if not prototypes:
            logger.warning("[PREFILTER-BOOTSTRAP] Seed file has no prototypes")
            return

        # Hash-based skip (ALWAYS check when Redis is available)
        # auto_rebootstrap only controls behavior on hash MISMATCH, not match
        redis_hash_key = f"ubp:prefilter:bootstrap_hash:{collection}"
        if self._redis:
            try:
                stored_hash = await self._redis.get(redis_hash_key)
                if stored_hash:
                    stored = stored_hash.decode("utf-8") if isinstance(stored_hash, bytes) else str(stored_hash)
                    if stored == seed_hash:
                        # Same seed → always SKIP
                        logger.info(
                            f"[PREFILTER-BOOTSTRAP] bootstrap_enabled=true | "
                            f"seed_path={seed_path} | seed_hash={seed_hash} | "
                            f"action=SKIP | reason=hash_match"
                        )
                        return
                    elif not auto_rebootstrap:
                        # Seed changed but flag off → SKIP + WARNING
                        logger.warning(
                            f"[PREFILTER-BOOTSTRAP] bootstrap_enabled=true | "
                            f"seed_path={seed_path} | seed_hash={seed_hash} | "
                            f"stored_hash={stored} | action=SKIP | "
                            f"reason=seed_changed_but_auto_rebootstrap=false"
                        )
                        return
                    # else: hash mismatch + auto_rebootstrap=true → fall through to APPLY
            except Exception as exc:
                logger.warning(f"[PREFILTER-BOOTSTRAP] Redis hash check failed: {exc}")
                # Redis unavailable — proceed with upsert

        # Perform upsert
        t0 = time.time()
        try:
            n_upserted = await self._bootstrap_upsert(
                prototypes=prototypes,
                collection=collection,
                batch_size=batch_size,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            # Store hash in Redis for change detection
            if self._redis:
                try:
                    await self._redis.set(redis_hash_key, seed_hash, ex=86400 * 30)  # 30d TTL
                except Exception:
                    pass

            logger.info(
                f"[PREFILTER-BOOTSTRAP] bootstrap_enabled=true | "
                f"seed_path={seed_path} | seed_hash={seed_hash} | "
                f"action=APPLY | points={n_upserted}/{len(prototypes)} | "
                f"collection={collection} | elapsed={elapsed_ms}ms"
            )
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.error(
                f"[PREFILTER-BOOTSTRAP] Upsert failed after {elapsed_ms}ms: {e}"
            )

    async def _bootstrap_upsert(
        self,
        prototypes: list,
        collection: str,
        batch_size: int = 64,
    ) -> int:
        """Embed prototypes and upsert to Qdrant with deterministic IDs.

        Returns number of points upserted.
        """
        import asyncio
        import hashlib
        import numpy as np
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        import os

        host = os.getenv("UBP_QDRANT__HOST", "ubp-qdrant")
        port = int(os.getenv("UBP_QDRANT__PORT", "6333"))

        # Deterministic ID: SHA1(cluster|type|language|text) → uint64
        def _det_id(row):
            cluster = str(row.get("cluster", "")).strip().lower()
            typ = str(row.get("type", "")).strip().lower()
            lang = str(row.get("language", "it")).strip().lower() or "it"
            text = str(row.get("text", "")).strip().lower()
            source = f"{cluster}|{typ}|{lang}|{text}"
            sha1_hex = hashlib.sha1(source.encode("utf-8")).hexdigest()
            return int(sha1_hex[:16], 16)

        # Get embed function (already initialized)
        embed_fn = self._engine._embed if self._engine else self._get_embed_fn()
        mdim = self.config.get("layer1_knn", {}).get("matryoshka_dim", 128)

        # Batch embed
        points = []
        for i in range(0, len(prototypes), batch_size):
            batch = prototypes[i : i + batch_size]
            embeddings = await asyncio.gather(
                *[embed_fn(p["text"]) for p in batch]
            )
            for proto, vec in zip(batch, embeddings):
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                if mdim > 0 and len(vec) > mdim:
                    vec = vec[:mdim]
                point_id = _det_id(proto)
                payload = {
                    "cluster": proto.get("cluster"),
                    "type": proto.get("type"),
                    "text": proto.get("text"),
                    "weight": float(proto.get("weight", 1.0)),
                    "subcategory": proto.get("subcategory", "general"),
                    "language": proto.get("language", "it"),
                    "active": bool(proto.get("active", True)),
                    "created_at": proto.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                    "source": "routing_prototypes_v2",
                }
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        # Upsert to Qdrant (sync client in thread)
        def _sync_upsert():
            client = QdrantClient(host=host, port=port)
            # Ensure collection exists (DO NOT recreate or delete)
            if not client.collection_exists(collection):
                from qdrant_client.models import VectorParams, Distance
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=mdim, distance=Distance.COSINE),
                )
                logger.info(f"[PREFILTER-BOOTSTRAP] Created collection '{collection}' (dim={mdim}, Cosine)")

            # Upsert in batches (Qdrant max ~1000 per call)
            upserted = 0
            for j in range(0, len(points), 256):
                client.upsert(
                    collection_name=collection,
                    points=points[j : j + 256],
                )
                upserted += len(points[j : j + 256])

            # Create payload indexes (idempotent — Qdrant skips if already exists)
            from qdrant_client.models import PayloadSchemaType
            for field, schema_type in [
                ("cluster", PayloadSchemaType.KEYWORD),
                ("type", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
                ("active", PayloadSchemaType.BOOL),
            ]:
                try:
                    client.create_payload_index(
                        collection_name=collection,
                        field_name=field,
                        field_schema=schema_type,
                    )
                except Exception:
                    pass  # Already exists — ok

            client.close()
            return upserted

        return await asyncio.to_thread(_sync_upsert)

    def _get_embed_fn(self):
        """Create embedding function for prefilter routing.

        If dedicated_model=true (default), loads a standalone SentenceTransformer
        in GPU memory, fully independent from rag_qdrant. This prevents model-swap
        contention when rag_qdrant switches between models for different collections.

        If dedicated_model=false, falls back to rag_qdrant.embedding_manager.embed
        (shared instance — susceptible to model-swapping timeouts).
        """
        l1_cfg = self.config.get("layer1_embedding", {})
        use_dedicated = l1_cfg.get("dedicated_model", True)

        if use_dedicated:
            return self._create_dedicated_embed_fn(l1_cfg)

        return self._get_shared_embed_fn()

    def _create_dedicated_embed_fn(self, l1_cfg: dict):
        """Load embedder via SharedModelPool (GPU dedup with rag_qdrant)."""
        # Resolve model name: config override > rag_qdrant auto-detect
        model_name = l1_cfg.get("model")
        if not model_name:
            provider = getattr(self._rag_qdrant, 'provider', None)
            if provider:
                emb_mgr = getattr(provider, 'embedding_manager', None)
                if emb_mgr:
                    model_name = emb_mgr.config.model
        if not model_name:
            raise ValueError("[PREFILTER] No embedding model: set layer1_embedding.model in config.json")

        device = l1_cfg.get("device", "auto")

        from ubp_enterprise_hybrid.modules.cores._shared.model_pool import SharedModelPool
        self._dedicated_model = SharedModelPool.get_sentence_transformer(
            model_name=model_name,
            device=device,
            trust_remote_code=True,
        )
        dim = self._dedicated_model.get_sentence_embedding_dimension()
        logger.info(
            f"[PREFILTER] Embedder loaded via SharedModelPool: {model_name}, dim={dim}"
        )

        model_ref = self._dedicated_model

        async def _embed_single(text: str) -> list:
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: model_ref.encode(
                    text, normalize_embeddings=True
                ).tolist(),
            )

        return _embed_single

    def _get_shared_embed_fn(self):
        """Fallback: resolve embed_fn from rag_qdrant (shared instance)."""
        # Canonical path: adapter.provider.embedding_manager.embed
        provider = getattr(self._rag_qdrant, 'provider', None)
        if provider:
            emb_mgr = getattr(provider, 'embedding_manager', None)
            if emb_mgr and hasattr(emb_mgr, 'embed'):
                logger.info("[PREFILTER] embed_fn resolved: provider.embedding_manager.embed (SHARED)")
                raw_fn = emb_mgr.embed

                async def _embed_single(text: str):
                    result = raw_fn(text)
                    if hasattr(result, '__await__'):
                        result = await result
                    if isinstance(result, list) and result and isinstance(result[0], list):
                        return result[0]
                    return result

                return _embed_single

        # Fallback: wrap call_operation
        logger.info("[PREFILTER] embed_fn resolved: call_operation('embed_text') fallback")

        async def _embed_via_operation(text: str):
            result = await self._rag_qdrant.call_operation(
                "embed_text", text=text
            )
            return result

        return _embed_via_operation

    def _create_qdrant_loader(self):
        """Create a callable that loads centroids from Qdrant routing_prototypes.

        Returns None if prototype_collection is not configured.
        The returned callable fetches all active prototypes, groups by cluster,
        computes weighted centroids, and returns (centroids_dict, stats_dict).
        """
        l1_cfg = self.config.get("layer1_embedding", {})
        collection = l1_cfg.get("prototype_collection")
        if not collection:
            logger.info("[PREFILTER] No prototype_collection configured — using hardcoded exemplars")
            return None

        mdim = l1_cfg.get("matryoshka_dim", 0)

        async def _load_from_qdrant():
            import numpy as np
            from qdrant_client import QdrantClient
            import os

            host = os.getenv("UBP_QDRANT__HOST", "ubp-qdrant")
            port = int(os.getenv("UBP_QDRANT__PORT", "6333"))
            client = QdrantClient(host=host, port=port)

            if not client.collection_exists(collection):
                logger.warning(f"[PREFILTER] Collection '{collection}' not found in Qdrant")
                return None, {}

            # Scroll all active points
            all_points = []
            offset = None
            while True:
                result = client.scroll(
                    collection_name=collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                    scroll_filter={
                        "must": [{"key": "active", "match": {"value": True}}]
                    } if True else None,
                )
                points, next_offset = result
                all_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            if not all_points:
                logger.warning(f"[PREFILTER] No active points in '{collection}'")
                return None, {}

            # Group by cluster, compute weighted centroids
            from collections import defaultdict
            cluster_vecs = defaultdict(list)
            cluster_weights = defaultdict(list)
            for pt in all_points:
                cluster = pt.payload.get("cluster", "unknown")
                weight = pt.payload.get("weight", 1.0)
                vec = np.array(pt.vector, dtype=np.float64)
                if mdim > 0 and len(vec) > mdim:
                    vec = vec[:mdim]
                cluster_vecs[cluster].append(vec)
                cluster_weights[cluster].append(weight)

            centroids = {}
            stats = {}
            for cluster, vecs in cluster_vecs.items():
                weights = np.array(cluster_weights[cluster], dtype=np.float64)
                vecs_arr = np.array(vecs)
                # Weighted mean
                weighted_sum = np.average(vecs_arr, axis=0, weights=weights)
                # L2-normalize
                norm = np.linalg.norm(weighted_sum)
                if norm > 1e-9:
                    weighted_sum = weighted_sum / norm
                centroids[cluster] = weighted_sum
                stats[cluster] = len(vecs)

            client.close()
            return centroids, stats

        logger.info(f"[PREFILTER] Qdrant loader configured: collection='{collection}'")
        return _load_from_qdrant

    def _create_qdrant_search_fn(self):
        """Create async callable for Qdrant vector search across collections.

        Used by Layer 3 RAG Preview and Web Signal evidence sources.
        Returns list of dicts with 'score' key, sorted by score desc.
        Silently ignores non-existent collections.

        P0-FIX: Uses AsyncQdrantClient (no threads → no GIL contention).
        Persistent client + cached collection dimensions avoid per-query overhead.
        """
        import os
        host = os.getenv("UBP_QDRANT__HOST", "ubp-qdrant")
        port = int(os.getenv("UBP_QDRANT__PORT", "6333"))
        _client_holder: dict = {"client": None}
        _dim_cache: dict = {}  # collection_name → dimension (or None if not exists)

        async def _get_client():
            if _client_holder["client"] is None:
                from qdrant_client import AsyncQdrantClient
                _client_holder["client"] = AsyncQdrantClient(host=host, port=port)
            return _client_holder["client"]

        async def _reset_client():
            """Reconnect after Qdrant restart or network error."""
            old = _client_holder.get("client")
            if old:
                try:
                    await old.close()
                except Exception:
                    pass
            _client_holder["client"] = None
            _dim_cache.clear()

        async def _qdrant_search(collections, vector, limit=3,
                                 score_threshold=0.55, filter_payload=None):
            try:
                client = await _get_client()
            except Exception:
                await _reset_client()
                return []
            merged = []
            for coll_name in collections:
                try:
                    # Check dimension cache first
                    if coll_name not in _dim_cache:
                        if not await client.collection_exists(coll_name):
                            _dim_cache[coll_name] = None
                            continue
                        coll_info = await client.get_collection(coll_name)
                        _dim_cache[coll_name] = coll_info.config.params.vectors.size

                    coll_dim = _dim_cache[coll_name]
                    if coll_dim is None:
                        continue

                    search_vec = vector[:coll_dim] if len(vector) > coll_dim else vector

                    search_kwargs = {
                        "collection_name": coll_name,
                        "query_vector": search_vec,
                        "limit": limit,
                        "score_threshold": score_threshold,
                    }
                    if filter_payload:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        conditions = []
                        for key, val in filter_payload.items():
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=val))
                            )
                        search_kwargs["query_filter"] = Filter(must=conditions)

                    results = await client.search(**search_kwargs)
                    for r in results:
                        merged.append({"score": r.score, "collection": coll_name})
                except Exception:
                    # Invalidate caches on error — Qdrant may have restarted
                    _dim_cache.pop(coll_name, None)
                    await _reset_client()
                    continue
            merged.sort(key=lambda x: x["score"], reverse=True)
            return merged[:limit]

        return _qdrant_search

    def _create_vllm_fn(self):
        """Create async callable for vLLM chat (R2 disambiguation).

        Returns None if inference_vllm is not available.
        The wrapper calls inference_vllm.chat() with system+user messages,
        temperature 0.3, max_tokens 150 (short JSON response).
        """
        if not self._inference_vllm:
            return None

        vllm_ref = self._inference_vllm

        async def _vllm_chat(system_prompt: str, user_prompt: str) -> Optional[str]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = await vllm_ref.chat(
                messages=messages,
                max_tokens=150,
                temperature=0.3,
            )
            # Extract text from response
            if isinstance(result, dict):
                return result.get("response", result.get("content", ""))
            return str(result) if result else None

        return _vllm_chat

    async def resolve_r2_response(
        self,
        original_query: str,
        user_response: str,
        ranked_scores: Dict[str, float],
    ) -> Optional[Dict[str, str]]:
        """Resolve R2 second turn via engine.

        Delegates to engine.r2_resolve_response() which calls vLLM.
        Returns {"route": "...", "reasoning": "..."} or None.
        """
        if not self._engine:
            return None
        return await self._engine.r2_resolve_response(
            original_query=original_query,
            user_response=user_response,
            ranked_scores=ranked_scores,
        )

    async def shutdown(self, **kwargs):
        """Release model reference. SharedModelPool owns actual lifecycle."""
        self._initialized = False
        self._engine = None
        self._dedicated_model = None  # Release reference, pool owns model
        logger.info("[PREFILTER] Shutdown complete (reference released)")

    async def health_check(self, **kwargs) -> Dict[str, Any]:
        """Return health status."""
        source = getattr(self._engine, '_centroid_source', 'unknown') if self._engine else 'none'
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "centroid_source": source,
            "timeout_ms": self._timeout_ms,
            "has_redis": self._redis is not None,
            "has_profile_memory": self._profile_memory is not None,
            "has_vllm": self._inference_vllm is not None,
        }

    async def reload_centroids(self, **kwargs) -> Dict[str, Any]:
        """Hot-reload centroids from Qdrant without restart."""
        if not self._engine:
            return {"status": "error", "reason": "engine not initialized"}
        return await self._engine.reload_centroids()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def call_operation(self, operation: str, **kwargs):
        """Dispatch operation by name (ModuleLoader interface)."""
        if operation == "initialize":
            return await self.initialize(**kwargs)
        if operation == "pre_route":
            return await self.pre_route(**kwargs)
        if operation == "get_routing_profile":
            return await self.get_routing_profile(**kwargs)
        if operation == "health_check":
            return await self.health_check(**kwargs)
        if operation == "shutdown":
            return await self.shutdown(**kwargs)
        if operation == "reload_centroids":
            return await self.reload_centroids(**kwargs)
        # Admin CRUD for routing prototypes
        if operation == "list_prototypes":
            return await self._list_prototypes(**kwargs)
        if operation == "add_prototype":
            return await self._add_prototype(**kwargs)
        if operation == "update_prototype":
            return await self._update_prototype(**kwargs)
        if operation == "delete_prototype":
            return await self._delete_prototype(**kwargs)
        if operation == "prototype_stats":
            return await self._prototype_stats(**kwargs)
        if operation == "get_metrics":
            return self._engine.get_metrics()
        if operation == "reset_metrics":
            self._engine.reset_metrics()
            return {"status": "metrics_reset"}
        if operation == "get_prototype_candidates":
            return self._engine.get_prototype_candidates()
        if operation == "resolve_r2_response":
            return await self.resolve_r2_response(**kwargs)
        raise ValueError(f"Unknown operation: {operation}")

    async def pre_route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        ctx: Any = None,
        **kwargs,
    ) -> PreRouteDecision:
        """Route a query using the 4-layer embedding prefilter.

        R1 Safe Mode: wraps engine call with asyncio.wait_for timeout.
        On timeout or exception, returns safe fallback to LLM_ROUTER.

        Args:
            query: User query text
            context: Optional context dict
            ctx: UBPContext

        Returns:
            PreRouteDecision
        """
        if not self._initialized:
            await self.initialize()

        def _log_r1_fallback_structured(decision: PreRouteDecision) -> None:
            l1_cfg = self.config.get("layer1_knn", self.config.get("layer1_embedding", {}))
            # P1-FIX: derive metadata from scores when available
            scores = decision.scores or {}
            sorted_scores = sorted(scores.values(), reverse=True)
            top2_delta = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
            score_range = (max(sorted_scores) - min(sorted_scores)) if sorted_scores else 0.0
            uncertainty = 1.0 - decision.confidence if decision.confidence > 0 else 0.0
            payload = {
                "query_id": decision.decision_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "selected_lane": decision.route,
                "confidence": float(decision.confidence),
                "uncertainty": round(uncertainty, 6),
                "top2_delta": round(top2_delta, 6),
                "score_range": round(score_range, 6),
                "absolute_floor_triggered": False,
                "low_range_triggered": False,
                "anti_proto_triggered": False,
                "deferred": bool(decision.deferred_to_llm_router),
                "k_value": int(l1_cfg.get("top_k", 0)),
                "matches_per_lane": {"chat": 0, "web": 0, "rag": 0, "report": 0},
            }
            logger.info(f"[PREFILTER-DECISION] {json.dumps(payload, ensure_ascii=False)}")

        # Extract user_id from context
        user_id = None
        if ctx:
            user_id = getattr(ctx, "user_id", None)
            if not user_id and hasattr(ctx, "user"):
                user_id = getattr(ctx.user, "user_id", None)

        # Apply timeout from env override if available
        timeout_sec = self._timeout_sec
        if ctx and hasattr(ctx, 'request') and hasattr(ctx.request, 'app'):
            try:
                settings = getattr(ctx.request.app.state, 'settings', None)
                pf_settings = getattr(settings, 'prefilter', None)
                if pf_settings:
                    timeout_sec = getattr(pf_settings, 'timeout_ms', self._timeout_ms) / 1000.0
            except Exception:
                pass

        # R1 Safe Mode: asyncio.wait_for wrapper
        # decision_id generated here and propagated to engine for correlation.
        # NOTE: Timeout recovery (L2.5 scores) is now handled INSIDE
        # providers.py pre_route() — the engine returns a valid PreRouteDecision
        # even on timeout. This adapter is a simple passthrough with a safety net.
        decision_id = str(uuid.uuid4())[:8]
        try:
            decision = await asyncio.wait_for(
                self._engine.pre_route(
                    query=query,
                    context=context or {},
                    user_id=user_id,
                    decision_id=decision_id,
                ),
                timeout=timeout_sec,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Safety net: if the timeout fires BEFORE L3 (unlikely but possible
            # during embedding), the engine may not have had a chance to recover.
            # Use pre_l3 scores if available; otherwise defer.
            pre_l3 = getattr(self._engine, "_last_pre_l3_scores", None)
            if pre_l3 and max(pre_l3.values()) > 0.35:
                best_route = max(pre_l3, key=pre_l3.get)
                best_conf = pre_l3[best_route]
                decision = PreRouteDecision(
                    decision_id=decision_id,
                    route=best_route,
                    confidence=best_conf,
                    raw_confidence=best_conf,
                    reasoning=f"R1_timeout_recovery:adapter_safety|{best_route}={best_conf:.3f}",
                    layer_trace=["R1:timeout", "adapter:safety"],
                    severity_level="medium",
                    time_ms=self._timeout_ms,
                    deferred_to_llm_router=False,
                    scores=pre_l3,
                )
            else:
                decision = PreRouteDecision(
                    decision_id=decision_id,
                    route="LLM_ROUTER",
                    confidence=0.0,
                    raw_confidence=0.0,
                    reasoning=f"safe_fallback:TimeoutError({self._timeout_ms}ms)",
                    layer_trace=["R1:timeout"],
                    severity_level="low",
                    time_ms=self._timeout_ms,
                    deferred_to_llm_router=True,
                )
            logger.warning(
                "[PREFILTER] R1 adapter safety net: timeout=%dms, "
                "route=%s, recovered=%s",
                self._timeout_ms, decision.route,
                not decision.deferred_to_llm_router,
            )
            _log_r1_fallback_structured(decision)
            await self._publish_event(
                "embedding_prefilter.fallback_activated",
                {"reason": "timeout", "timeout_ms": self._timeout_ms,
                 "recovered": not decision.deferred_to_llm_router},
            )
        except Exception as e:
            decision = PreRouteDecision(
                decision_id=decision_id,
                route="LLM_ROUTER",
                confidence=0.0,
                raw_confidence=0.0,
                reasoning=f"safe_fallback:{type(e).__name__}:{str(e)[:100]}",
                layer_trace=["R1:exception"],
                severity_level="low",
                time_ms=0.0,
                deferred_to_llm_router=True,
            )
            logger.warning(
                f"[PREFILTER] R1 Safe Fallback: {type(e).__name__} -> LLM_ROUTER: {e}"
            )
            _log_r1_fallback_structured(decision)
            await self._publish_event(
                "embedding_prefilter.fallback_activated",
                {"reason": type(e).__name__, "error": str(e)[:200]},
            )

        # Publish route_decided or deferred event
        if decision.deferred_to_llm_router:
            await self._publish_event(
                "embedding_prefilter.deferred_to_llm_router",
                decision.to_dict(),
            )
        else:
            await self._publish_event(
                "embedding_prefilter.route_decided",
                decision.to_dict(),
            )

        return decision

    async def get_routing_profile(
        self,
        user_id: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Get per-user routing profile from Redis (R3, future).

        Phase 1: Returns None (R3 not active).
        """
        return None

    # ------------------------------------------------------------------
    # Admin CRUD — Routing Prototypes
    # ------------------------------------------------------------------

    def _get_qdrant_client(self):
        """Get a Qdrant client for admin operations."""
        import os
        from qdrant_client import QdrantClient
        host = os.getenv("UBP_QDRANT__HOST", "ubp-qdrant")
        port = int(os.getenv("UBP_QDRANT__PORT", "6333"))
        return QdrantClient(host=host, port=port)

    def _get_collection_name(self) -> str:
        return self.config.get("layer1_embedding", {}).get(
            "prototype_collection", "routing_prototypes"
        )

    async def _list_prototypes(self, cluster: str = None, active: bool = None,
                                limit: int = 100, offset: int = 0, **kwargs):
        """List prototypes with optional filters."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = self._get_qdrant_client()
        collection = self._get_collection_name()

        must_conditions = []
        if cluster:
            must_conditions.append(FieldCondition(key="cluster", match=MatchValue(value=cluster)))
        if active is not None:
            must_conditions.append(FieldCondition(key="active", match=MatchValue(value=active)))

        scroll_filter = Filter(must=must_conditions) if must_conditions else None

        points, _next = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset if offset else None,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )
        client.close()

        return {
            "prototypes": [
                {"id": str(pt.id), **pt.payload} for pt in points
            ],
            "count": len(points),
        }

    async def _add_prototype(self, text: str, cluster: str, weight: float = 1.0,
                              language: str = "it", domain: str = "general", **kwargs):
        """Add a new routing prototype: embed text → upsert to Qdrant."""
        import numpy as np
        from qdrant_client.models import PointStruct
        from datetime import datetime, timezone

        valid_clusters = {"chat", "rag", "report", "web_search"}
        if cluster not in valid_clusters:
            return {"status": "error", "reason": f"Invalid cluster. Must be one of: {valid_clusters}"}
        if not text or len(text.strip()) < 3:
            return {"status": "error", "reason": "Text must be at least 3 characters"}
        if not (0.0 < weight <= 5.0):
            return {"status": "error", "reason": "Weight must be in (0, 5]"}

        # Embed with dedicated model
        vec = await self._engine._safe_embed(text)
        mdim = self.config.get("layer1_embedding", {}).get("matryoshka_dim", 0)
        if mdim > 0 and len(vec) > mdim:
            vec = vec[:mdim]
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 1e-9:
            arr = arr / norm

        point_id = str(uuid.uuid4())
        client = self._get_qdrant_client()
        collection = self._get_collection_name()

        client.upsert(
            collection_name=collection,
            points=[PointStruct(
                id=point_id,
                vector=arr.tolist(),
                payload={
                    "cluster": cluster,
                    "text": text,
                    "weight": weight,
                    "language": language,
                    "domain": domain,
                    "source": "admin",
                    "active": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )],
        )
        client.close()

        logger.info(f"[PREFILTER] Prototype added: id={point_id}, cluster={cluster}, text='{text[:40]}'")
        return {
            "status": "created",
            "id": point_id,
            "cluster": cluster,
            "note": "Call reload_centroids to apply changes",
        }

    async def _update_prototype(self, id: str, weight: float = None,
                                 cluster: str = None, active: bool = None, **kwargs):
        """Update prototype metadata. Does NOT re-embed text."""
        from qdrant_client.models import PointStruct
        client = self._get_qdrant_client()
        collection = self._get_collection_name()

        # Fetch existing point
        points = client.retrieve(collection_name=collection, ids=[id], with_payload=True, with_vectors=True)
        if not points:
            client.close()
            return {"status": "error", "reason": f"Prototype {id} not found"}

        pt = points[0]
        payload = dict(pt.payload)
        if weight is not None:
            payload["weight"] = weight
        if cluster is not None:
            payload["cluster"] = cluster
        if active is not None:
            payload["active"] = active

        client.upsert(
            collection_name=collection,
            points=[PointStruct(id=id, vector=pt.vector, payload=payload)],
        )
        client.close()

        logger.info(f"[PREFILTER] Prototype updated: id={id}")
        return {"status": "updated", "id": id, "payload": payload}

    async def _delete_prototype(self, id: str, hard: bool = False, **kwargs):
        """Soft-delete (active=false) or hard-delete a prototype."""
        client = self._get_qdrant_client()
        collection = self._get_collection_name()

        if hard:
            from qdrant_client.models import PointIdsList
            client.delete(collection_name=collection, points_selector=PointIdsList(points=[id]))
            client.close()
            logger.info(f"[PREFILTER] Prototype hard-deleted: id={id}")
            return {"status": "deleted", "id": id, "mode": "hard"}

        # Soft delete: set active=false
        result = await self._update_prototype(id=id, active=False)
        if result.get("status") == "updated":
            result["mode"] = "soft"
            result["status"] = "deleted"
        return result

    async def _prototype_stats(self, **kwargs):
        """Return per-cluster statistics."""
        client = self._get_qdrant_client()
        collection = self._get_collection_name()

        all_points, _ = client.scroll(
            collection_name=collection, limit=10000,
            with_payload=True, with_vectors=False,
        )
        client.close()

        from collections import defaultdict
        stats = defaultdict(lambda: {"total": 0, "active": 0, "avg_weight": 0.0, "weights": []})
        for pt in all_points:
            c = pt.payload.get("cluster", "unknown")
            stats[c]["total"] += 1
            if pt.payload.get("active", True):
                stats[c]["active"] += 1
            stats[c]["weights"].append(pt.payload.get("weight", 1.0))

        result = {}
        for c, s in stats.items():
            result[c] = {
                "total": s["total"],
                "active": s["active"],
                "avg_weight": round(sum(s["weights"]) / len(s["weights"]), 3) if s["weights"] else 0,
            }

        return {
            "collection": collection,
            "clusters": result,
            "total_points": len(all_points),
            "centroid_source": getattr(self._engine, '_centroid_source', 'unknown') if self._engine else 'none',
        }

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    async def _publish_event(self, event_name: str, data: Any) -> None:
        """Publish event to event bus (best-effort, sync/async safe)."""
        if self.event_bus:
            try:
                result = self.event_bus.publish(event_name, data)
                # Handle both sync and async event bus implementations
                if hasattr(result, '__await__'):
                    await result
            except Exception:
                pass
