"""
query_expansion_pipeline/delegation.py

LLM delegation layer for query expansion.

Provides:
- LLMDelegator: Main LLM interface
- Module resolution via DI container
- Provider chain support
- Timeout and retry handling

Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from .prompts import PromptTemplates

# BUG-005 fix: ProviderMapper for LLM delegation fallback
try:
    from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper, ProviderConfigurationError
    PROVIDER_MAPPER_AVAILABLE = True
except ImportError:
    PROVIDER_MAPPER_AVAILABLE = False
    ProviderMapper = None
    ProviderConfigurationError = Exception

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...
    async def resolve_module(self, module_name: str) -> Optional[Any]: ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LLMConfig:
    """LLM delegation configuration."""
    module: str = "inference_ollama_grok"
    operation: str = "generate"
    provider: str = "grok"
    timeout_seconds: int = 30
    max_retries: int = 2
    temperature: float = 0.7
    max_tokens: int = 300


# ============================================================================
# DI Container Registry Adapter
# ============================================================================


class DIContainerModuleRegistry:
    """Adapter for DI container as module registry."""
    
    def __init__(self, di_container: Any):
        self._di_container = di_container
        self._cache: Dict[str, Any] = {}
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get cached module."""
        return self._cache.get(module_name)
    
    async def resolve_module(self, module_name: str) -> Optional[Any]:
        """Resolve and cache module."""
        if module_name in self._cache:
            return self._cache[module_name]
        
        try:
            module = await self._di_container.resolve(module_name)
            self._cache[module_name] = module
            return module
        except Exception as e:
            logger.warning(f"Failed to resolve module '{module_name}': {e}")
            return None


# ============================================================================
# LLM Delegator
# ============================================================================


class LLMDelegator:
    """
    Delegates LLM operations for query expansion.
    
    Handles:
    - Module resolution
    - Prompt building
    - Response parsing
    - Timeout and retries
    """
    
    def __init__(
        self,
        config: LLMConfig,
        module_registry: Optional[IModuleRegistry] = None,
        event_publisher: Optional[Callable] = None,
    ):
        self.config = config
        self._registry = module_registry
        self._publisher = event_publisher
        self._llm_module: Optional[Any] = None
        self._prompts = PromptTemplates()
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._llm_module is not None
    
    async def ensure_llm(self) -> Optional[Any]:
        """Ensure LLM module is resolved. BUG-005: ProviderMapper fallback."""
        if self._llm_module:
            return self._llm_module

        if not self._registry:
            logger.warning("[QUERY_EXPANSION] Module registry not available")
            return None

        # v6.2.1: Try ProviderMapper first (role-based provider chain)
        if PROVIDER_MAPPER_AVAILABLE and ProviderMapper:
            try:
                provider_chain = ProviderMapper.resolve_chain("enrichment")
            except ProviderConfigurationError as exc:
                logger.warning(f"[QUERY_EXPANSION] ProviderMapper config error: {exc}")
                provider_chain = None

            if provider_chain:
                for module_name, provider_name in provider_chain:
                    try:
                        resolved = await self._registry.resolve_module(module_name)
                    except Exception as e:
                        logger.warning(
                            f"[QUERY_EXPANSION] FALLBACK: module '{module_name}' "
                            f"resolution failed: {e}, trying next"
                        )
                        continue
                    if not resolved:
                        logger.warning(
                            f"[QUERY_EXPANSION] FALLBACK: module '{module_name}' "
                            f"(provider '{provider_name}') not ready, trying next"
                        )
                        continue
                    # Update mutable LLMConfig to match resolved provider
                    self.config.module = module_name
                    self.config.provider = provider_name
                    self._llm_module = resolved
                    logger.info(
                        f"[QUERY_EXPANSION] LLM resolved via ProviderMapper: "
                        f"module='{module_name}', provider='{provider_name}'"
                    )
                    return self._llm_module
                else:
                    logger.warning(
                        "[QUERY_EXPANSION] FALLBACK EXHAUSTED: no LLM from "
                        "ProviderMapper chain"
                    )

        # Fallback to config defaults (original behavior)
        try:
            self._llm_module = await self._registry.resolve_module(self.config.module)
            if self._llm_module:
                logger.info(
                    f"[QUERY_EXPANSION] LLM resolved via config: "
                    f"module='{self.config.module}', provider='{self.config.provider}'"
                )
            return self._llm_module
        except Exception as e:
            logger.error(f"[QUERY_EXPANSION] Failed to resolve LLM module: {e}")
            return None
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using LLM."""
        llm = await self.ensure_llm()
        
        if not llm:
            raise RuntimeError("LLM module not available")
        
        operation = getattr(llm, self.config.operation, None)
        if not operation:
            raise RuntimeError(f"LLM operation '{self.config.operation}' not found")
        
        # Build parameters
        params = {
            "prompt": prompt,
        }

        if self.config.provider:
            params["provider"] = self.config.provider

        if temperature is not None:
            params["temperature"] = temperature
        elif self.config.temperature:
            params["temperature"] = self.config.temperature

        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        elif self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        
        # Call with timeout and retries
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    operation(**params),
                    timeout=self.config.timeout_seconds,
                )
                
                # Extract text from result
                if isinstance(result, dict):
                    return result.get("text", result.get("response", ""))
                return str(result)
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout (attempt {attempt + 1})")
                if attempt == self.config.max_retries:
                    raise
            except Exception as e:
                logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries:
                    raise
        
        return ""
    
    async def expand_semantic(
        self,
        query: str,
        num_variants: int = 3,
        language: str = "en",
    ) -> List[str]:
        """Generate semantic expansions."""
        prompt = self._prompts.semantic_expansion(
            query=query,
            num_variants=num_variants,
            language=language,
        )
        
        response = await self.generate(prompt)
        return self._parse_list_response(response, query)
    
    async def decompose_query(
        self,
        query: str,
        max_subqueries: int = 5,
        language: str = "en",
    ) -> List[str]:
        """Decompose complex query."""
        prompt = self._prompts.decomposition(
            query=query,
            max_subqueries=max_subqueries,
            language=language,
        )
        
        response = await self.generate(prompt)
        return self._parse_list_response(response, query)
    
    async def expand_contextual(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        language: str = "en",
    ) -> str:
        """Expand with conversation context."""
        prompt = self._prompts.contextual(
            query=query,
            chat_history=chat_history,
            language=language,
        )
        
        response = await self.generate(prompt)
        
        # Return first line only
        lines = response.strip().split('\n')
        return lines[0].strip() if lines else query
    
    async def extract_keywords(
        self,
        query: str,
        max_keywords: int = 5,
        language: str = "en",
    ) -> List[str]:
        """Extract keywords from query."""
        prompt = self._prompts.keyword_extraction(
            query=query,
            max_keywords=max_keywords,
            language=language,
        )
        
        response = await self.generate(prompt)
        return self._parse_list_response(response, query)
    
    async def reformulate(
        self,
        query: str,
        language: str = "en",
    ) -> List[str]:
        """Reformulate query."""
        prompt = self._prompts.reformulation(
            query=query,
            language=language,
        )
        
        response = await self.generate(prompt)
        return self._parse_list_response(response, query)
    
    def _parse_list_response(
        self,
        response: str,
        original: str,
    ) -> List[str]:
        """Parse LLM response into list."""
        lines = response.strip().split('\n')
        results = []
        
        for line in lines:
            text = line.strip()
            
            # Remove common prefixes
            if text and text[0].isdigit():
                import re
                text = re.sub(r'^[\d]+[\.\)\-\s]+', '', text)
            
            if text.startswith(('-', '•', '*', '>')):
                text = text[1:].strip()
            
            # Skip empty or same as original
            if not text or len(text) < 3:
                continue
            if text.lower() == original.lower():
                continue
            
            results.append(text)
        
        return results
    
    def get_caller(self) -> Optional[Callable]:
        """Get a callable for strategies to use."""
        async def caller(prompt: str) -> str:
            return await self.generate(prompt)
        
        if self.is_available() or self._registry:
            return caller
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM delegation health."""
        if not self._llm_module:
            await self.ensure_llm()
        
        return {
            "status": "available" if self._llm_module else "unavailable",
            "module": self.config.module,
            "operation": self.config.operation,
        }
