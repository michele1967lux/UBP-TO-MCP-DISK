"""
UBP Framework Bridge for Reranker Module
Integrates RerankerProvider with UBP module system.

Provides document reranking using cross-encoder models.

Operations:
- rerank: Rerank documents by relevance to query
- rerank_search_results: Search + rerank in one operation
- get_reranker_info: Get backend information
- health_check: Module health status
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import uuid
import os

from ubp_enterprise_hybrid.modules.cores._shared import BaseHybridModule
from .providers import RerankerProvider, RerankerBackend, RerankResult

# MCP-COMPAT (ARCH-008): Import OperationContext for dual path support
try:
    from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
except ModuleNotFoundError:
    try:
        from _shared.operation_context import OperationContext
    except ModuleNotFoundError:
        from ..._shared.operation_context import OperationContext

logger = logging.getLogger(__name__)


class RerankerAdapter(BaseHybridModule):
    """
    UBP adapter for document reranking.

    Provides cross-encoder reranking with multiple backend options.
    Follows 3-file pattern: no business logic here, only UBP integration.

    Security:
    - User ID always from ctx, never from payload
    - Admin-only operations protected
    """

    def __init__(self, module_path: Path, **kwargs):
        super().__init__(module_path, **kwargs)
        self.provider: Optional[RerankerProvider] = None
        self._total_rerank_calls = 0
        self._total_documents_reranked = 0

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

    async def initialize(self) -> None:
        """Initialize module and provider."""
        logger.info(f"Initializing {self.manifest.name}")

        # Parse backend from config
        backend_str = self.config.get("backend", "local_cross_encoder")
        try:
            backend = RerankerBackend(backend_str)
        except ValueError:
            backend = RerankerBackend.LOCAL_CROSS_ENCODER
            logger.warning(
                f"Invalid backend '{backend_str}', using local_cross_encoder"
            )

        # Get API key from config or environment
        api_key = self.config.get("api_key")
        if not api_key:
            if backend == RerankerBackend.COHERE:
                api_key = os.getenv("COHERE_API_KEY")
            elif backend == RerankerBackend.JINA:
                api_key = os.getenv("JINA_API_KEY")

        # Get effective settings (ENV/override > config.json)
        effective = self._get_effective_settings()

        # Initialize provider — model loaded via SharedModelPool for GPU dedup
        logger.info(f"[RERANKER] Initializing provider...")
        self.provider = RerankerProvider(
            backend=backend,
            model_name=effective["model_name"],
            api_key=api_key,
            device=effective["device"],
            batch_size=effective["batch_size"],
            preload=False,  # Model injected from pool below
        )

        # Inject shared model for local backend (GPU dedup)
        if backend == RerankerBackend.LOCAL_CROSS_ENCODER:
            from ubp_enterprise_hybrid.modules.cores._shared.model_pool import SharedModelPool
            shared_model = SharedModelPool.get_cross_encoder(
                model_name=effective["model_name"],
                device=effective["device"],
                max_length=effective.get("max_length", 512),
            )
            self.provider.inject_shared_model(shared_model)

        logger.info(
            f"Reranker config source: {effective.get('_source', 'unknown')}, "
            f"model={effective['model_name']}"
        )

        logger.info(f"✅ {self.manifest.name} initialized with {backend.value} backend (model preloaded)")

    async def shutdown(self) -> None:
        """Shutdown module. Releases reference to shared model (pool owns lifecycle)."""
        logger.info(f"Shutting down {self.manifest.name}")
        self.provider = None
        logger.info(f"{self.manifest.name} shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        provider_health = self.provider.health_check() if self.provider else None
        effective = self._get_effective_settings()

        return {
            "module": self.manifest.name,
            "version": self.manifest.version,
            "status": "healthy" if self.provider else "unhealthy",
            "total_rerank_calls": self._total_rerank_calls,
            "total_documents_reranked": self._total_documents_reranked,
            "provider": provider_health,
            "config_source": effective.get("_source", "unknown"),
            "effective_model": effective.get("model_name", "unknown"),
        }

    # === SETTINGS & HOT-RELOAD ===

    def _get_effective_settings(self) -> Dict[str, Any]:
        """Get reranker settings with hot-reload support.

        Priority:
        1. Redis override (settings_manager) - if override enabled
        2. ENV variables (UBP_ENRICHMENT__RERANK_*) - via EnrichmentSettings
        3. Module config.json - fallback
        """
        try:
            from ubp_enterprise_hybrid.backend.app.api.admin_settings_routes import settings_manager

            settings = settings_manager.get_settings()
            enrichment = settings.enrichment

            fields = {
                "model_name":  ("enrichment.rerank_model", enrichment.rerank_model),
                "device":      ("enrichment.rerank_device", enrichment.rerank_device),
                "batch_size":  ("enrichment.rerank_batch_size", enrichment.rerank_batch_size),
                "max_length":  ("enrichment.rerank_max_length", enrichment.rerank_max_length),
            }

            result = {}
            for key, (redis_key, base_value) in fields.items():
                value, source = settings_manager.get_effective_value(redis_key, base_value)
                result[key] = value
                if source == "redis_override":
                    logger.info(f"[HotReload] reranker {key}={value} (override)")

            # Resolve "auto" device to actual PyTorch device
            result["device"] = self._resolve_device(result.get("device", "cpu"))
            result["_source"] = "settings/env"
            return result

        except Exception as e:
            logger.debug(f"Settings not available, using config.json: {e}")
            return {
                "model_name":  self.config.get("model_name", "cross-encoder/stsb-roberta-large"),
                "device":      self._resolve_device(self.config.get("device", "cpu")),
                "batch_size":  self.config.get("batch_size", 32),
                "max_length":  self.config.get("max_length", 512),
                "_source": "config.json",
            }

    def _check_model_change(self) -> None:
        """Reinitialize provider if model changed via hot-reload."""
        if not self.provider:
            return
        effective = self._get_effective_settings()
        current_model = self.provider.get_model_name()
        if effective["model_name"] != current_model:
            logger.info(f"[HotReload] Reranker model changed: {current_model} -> {effective['model_name']}")
            backend = self.provider.backend
            api_key = self.provider.get_api_key()
            self.provider = RerankerProvider(
                backend=backend,
                model_name=effective["model_name"],
                api_key=api_key,
                device=effective["device"],
                batch_size=effective["batch_size"],
                preload=True,  # Preload new model immediately
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to actual PyTorch device string."""
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # === OPERATIONS ===
    
    async def rerank_internal(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Internal rerank method compatible with IReranker protocol.
        
        This method is used when rag_reranker is called from other modules
        (like rag_orchestrator) instead of via API endpoint. It bypasses
        authentication checks since internal calls are already authenticated.
        
        Args:
            query: Search query
            chunks: List of documents/chunks with 'content' or 'text' field
            top_k: Number of top results to return (None = all)
            return_scores: Whether to include scores
            **kwargs: Additional parameters
        
        Returns:
            Dict compatible with enrichment_pipeline's rerank format:
            {
                "reranked_chunks": [...],
                "model_used": "...",
                "time_ms": ...,
                "original_count": ...
            }
        """
        import time

        self._check_model_change()

        if not self.provider:
            return {
                "reranked_chunks": chunks,
                "model_used": "none",
                "time_ms": 0,
                "original_count": len(chunks),
                "error": "Provider not initialized"
            }
        
        if not chunks:
            return {
                "reranked_chunks": [],
                "model_used": self.provider.get_info().get("model", "unknown"),
                "time_ms": 0,
                "original_count": 0,
            }
        
        # Normalize field names: chunks from enrichment use 'text', we use 'content'
        normalized_docs = []
        for chunk in chunks:
            # Always create a copy to avoid mutating input
            doc = chunk.copy()
            if "content" not in doc and "text" in doc:
                doc["content"] = doc.get("text", "")
            elif "text" not in doc and "content" in doc:
                doc["text"] = doc.get("content", "")
            normalized_docs.append(doc)
        
        start_time = time.perf_counter()
        
        try:
            # Perform reranking using provider
            actual_top_k = top_k if top_k is not None else len(normalized_docs)
            results = await self.provider.rerank(
                query=query,
                documents=normalized_docs,
                top_k=min(actual_top_k, len(normalized_docs)),
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Build lookup for collection field from input chunks
            _chunk_collections = {}
            for c in chunks:
                cid = c.get("doc_id") or c.get("id", "")
                if cid and c.get("collection"):
                    _chunk_collections[cid] = c["collection"]

            # Convert to enrichment_pipeline format
            reranked_chunks = []
            for r in results:
                chunk = {
                    "text": r.content,  # enrichment uses 'text'
                    "content": r.content,  # retrieval_strategy uses 'content'
                    "doc_id": r.doc_id,
                    "metadata": r.metadata.copy(),
                }
                # Preserve collection from input chunk
                coll = _chunk_collections.get(r.doc_id, "")
                if coll:
                    chunk["kb_id"] = coll
                    chunk["collection"] = coll
                if return_scores:
                    chunk["rerank_score"] = r.rerank_score
                    chunk["original_score"] = r.original_score
                    chunk["score"] = r.rerank_score  # Primary score field
                reranked_chunks.append(chunk)
            
            return {
                "reranked_chunks": reranked_chunks,
                "model_used": self.provider.get_info().get("model", "unknown"),
                "time_ms": elapsed_ms,
                "original_count": len(chunks),
            }
            
        except Exception as e:
            logger.error(f"Internal rerank failed: {e}", exc_info=True)
            # Return original chunks on error
            return {
                "reranked_chunks": chunks,
                "model_used": self.provider.get_info().get("model", "unknown"),
                "time_ms": (time.perf_counter() - start_time) * 1000,
                "original_count": len(chunks),
                "error": str(e)
            }

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of documents with 'content' field
                      (optionally 'doc_id', 'score', 'metadata')
            top_k: Number of top results to return
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Reranked documents with original and new scores/ranks
        """
        request_id = request_id or str(uuid.uuid4())

        self._check_model_change()

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        if not query or not query.strip():
            return {"error": "Query is required", "request_id": request_id}

        if not documents:
            return {
                "query": query,
                "results": [],
                "count": 0,
                "request_id": request_id,
            }

        try:
            # Perform reranking
            results = await self.provider.rerank(
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents)),
            )

            self._total_rerank_calls += 1
            self._total_documents_reranked += len(documents)

            return {
                "query": query,
                "results": [r.to_dict() for r in results],
                "count": len(results),
                "original_count": len(documents),
                "backend": self.provider.get_info().get("backend", "unknown"),
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Rerank failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def rerank_search_results(
        self,
        query: str,
        collection_name: str,
        initial_top_k: int = 50,
        final_top_k: int = 10,
        search_type: str = "hybrid",
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform search and rerank in one operation.

        This is a convenience method that:
        1. Calls hybrid_search, qdrant search, or sparse search
        2. Reranks the results using configured backend
        3. Returns final top_k

        Args:
            query: Search query
            collection_name: Target collection
            initial_top_k: Number of results to retrieve initially
            final_top_k: Number of results after reranking
            search_type: 'hybrid', 'dense', or 'sparse'
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Search results reranked by relevance
        """
        request_id = request_id or str(uuid.uuid4())

        self._check_model_change()

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        user_id = self._get_user_id_from_ctx(ctx)
        if not user_id:
            return {"error": "User not authenticated", "request_id": request_id}

        if not query or not query.strip():
            return {"error": "Query is required", "request_id": request_id}

        if not collection_name:
            return {"error": "Collection name is required", "request_id": request_id}

        try:
            # Step 1: Get initial results
            search_results: List[Dict[str, Any]] = []

            if search_type == "hybrid":
                hybrid_module = self._get_module("rag_hybrid_search")
                if hybrid_module:
                    response = await hybrid_module.hybrid_search(
                        query=query,
                        collection_name=collection_name,
                        top_k=initial_top_k,
                        ctx=ctx,
                    )
                    if "error" not in response:
                        search_results = response.get("results", [])
                else:
                    # Fallback to dense search
                    search_type = "dense"
                    logger.warning(
                        "rag_hybrid_search not available, falling back to dense"
                    )

            if search_type in ("dense", "sparse") or not search_results:
                qdrant_module = self._get_module("rag_qdrant")
                if qdrant_module:
                    response = await qdrant_module.search(
                        query=query,
                        collection_name=collection_name,
                        top_k=initial_top_k,
                        ctx=ctx,
                    )
                    if "error" not in response:
                        # Convert to standard format
                        for hit in response.get("results", []):
                            search_results.append(
                                {
                                    "doc_id": hit.get("id", hit.get("doc_id", "")),
                                    "content": hit.get("content", ""),
                                    "score": hit.get("score", 0.0),
                                    "metadata": hit.get("metadata", {}),
                                }
                            )
                else:
                    return {
                        "error": "No search module available",
                        "request_id": request_id,
                    }

            if not search_results:
                return {
                    "query": query,
                    "collection": collection_name,
                    "results": [],
                    "count": 0,
                    "search_type": search_type,
                    "reranked": False,
                    "request_id": request_id,
                }

            # Step 2: Rerank results
            reranked = await self.provider.rerank(
                query=query,
                documents=search_results,
                top_k=final_top_k,
            )

            self._total_rerank_calls += 1
            self._total_documents_reranked += len(search_results)

            return {
                "query": query,
                "collection": collection_name,
                "results": [r.to_dict() for r in reranked],
                "count": len(reranked),
                "initial_count": len(search_results),
                "search_type": search_type,
                "reranked": True,
                "backend": self.provider.get_info().get("backend", "unknown"),
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Rerank search failed: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def get_reranker_info(
        self,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get information about the configured reranker.

        Args:
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Reranker backend information and statistics
        """
        request_id = request_id or str(uuid.uuid4())

        if not self.provider:
            return {"error": "Provider not initialized", "request_id": request_id}

        info = self.provider.get_info()

        return {
            **info,
            "total_rerank_calls": self._total_rerank_calls,
            "total_documents_reranked": self._total_documents_reranked,
            "request_id": request_id,
        }

    async def list_reranker_models(
        self,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        List all available reranker models across all providers.

        Returns models from:
        - sentence-transformers (local CrossEncoder models)
        - ollama (dynamic discovery)
        - cohere (if API key configured)
        - jina (if API key configured)

        Args:
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Dict with models list, providers status, and counts
        """
        from .providers import list_all_reranker_models

        request_id = request_id or str(uuid.uuid4())

        # Get Ollama URL from config or environment
        ollama_base_url = self.config.get(
            "ollama_base_url",
            os.getenv("OLLAMA_BASE_URL", "http://ubp-ollama:11434")
        )

        # Get API keys from config or environment
        cohere_api_key = self.config.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
        jina_api_key = self.config.get("jina_api_key") or os.getenv("JINA_API_KEY")

        try:
            result = await list_all_reranker_models(
                ollama_base_url=ollama_base_url,
                cohere_api_key=cohere_api_key,
                jina_api_key=jina_api_key,
            )

            return {
                **result,
                "request_id": request_id,
            }

        except Exception as e:
            logger.error(f"Failed to list reranker models: {e}", exc_info=True)
            return {"error": str(e), "request_id": request_id}

    async def compare_backends(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        request_id: Optional[str] = None,
        ctx=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare reranking results across available backends.

        Admin-only operation for evaluation purposes.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of results per backend
            request_id: Optional request identifier
            ctx: Security context

        Returns:
            Results from each available backend for comparison
        """
        request_id = request_id or str(uuid.uuid4())

        self._check_model_change()

        if not self._is_admin(ctx):
            return {"error": "Admin access required", "request_id": request_id}

        if not documents:
            return {"error": "Documents required", "request_id": request_id}

        results = {}

        # Test local cross-encoder
        try:
            local_provider = RerankerProvider(
                backend=RerankerBackend.LOCAL_CROSS_ENCODER,
                device=self.config.get("device", "cpu"),
            )
            local_results = await local_provider.rerank(query, documents, top_k)
            results["local_cross_encoder"] = {
                "results": [r.to_dict() for r in local_results],
                "status": "success",
            }
        except Exception as e:
            results["local_cross_encoder"] = {
                "status": "error",
                "error": str(e),
            }

        # Test Cohere if API key available
        cohere_key = self.config.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
        if cohere_key:
            try:
                cohere_provider = RerankerProvider(
                    backend=RerankerBackend.COHERE,
                    api_key=cohere_key,
                )
                cohere_results = await cohere_provider.rerank(query, documents, top_k)
                results["cohere"] = {
                    "results": [r.to_dict() for r in cohere_results],
                    "status": "success",
                }
            except Exception as e:
                results["cohere"] = {
                    "status": "error",
                    "error": str(e),
                }

        # Test Jina if API key available
        jina_key = self.config.get("jina_api_key") or os.getenv("JINA_API_KEY")
        if jina_key:
            try:
                jina_provider = RerankerProvider(
                    backend=RerankerBackend.JINA,
                    api_key=jina_key,
                )
                jina_results = await jina_provider.rerank(query, documents, top_k)
                results["jina"] = {
                    "results": [r.to_dict() for r in jina_results],
                    "status": "success",
                }
            except Exception as e:
                results["jina"] = {
                    "status": "error",
                    "error": str(e),
                }

        return {
            "query": query,
            "document_count": len(documents),
            "backends_tested": list(results.keys()),
            "results": results,
            "request_id": request_id,
        }

    # === HELPER METHODS ===

    def _get_user_id_from_ctx(self, ctx) -> Optional[str]:
        """Extract user_id from security context."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "user_id", None)
        return None

    def _is_admin(self, ctx) -> bool:
        """Check if user is admin."""
        if ctx and hasattr(ctx, "user") and ctx.user:
            return getattr(ctx.user, "is_admin", False)
        return False

    def _get_module(self, module_name: str):
        """Get another module from the system."""
        if hasattr(self, "module_manager") and self.module_manager:
            return self.module_manager.get_module(module_name)
        return None
