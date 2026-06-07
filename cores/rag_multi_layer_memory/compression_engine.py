"""
Compression Engine — LLM-powered compression for Layer 0 → Layer 1 (→ Layer 2).

Uses ProviderMapper for role-based LLM resolution with fallback chain.
The compression provider is configured via UBP_MEMORY__LLM_PROVIDER env var.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import CompressionResult, Layer1Block, Layer2Memory
from .utils import (
    format_layer1_for_prompt,
    format_layer2_for_prompt,
    format_snapshots_for_prompt,
)

# ---------------------------------------------------------------------------
# ENV-based configuration (shared across both compression modules)
# ---------------------------------------------------------------------------
_COMPRESSION_PROVIDER = os.environ.get("UBP_COMPRESSION__PROVIDER", "")
_COMPRESSION_ROLE = os.environ.get("UBP_COMPRESSION__ROLE", "enrichment")
_COMPRESSION_TEMPERATURE = float(os.environ.get("UBP_COMPRESSION__TEMPERATURE", "0.3"))

logger = logging.getLogger(__name__)


class CompressionEngine:
    """
    LLM-powered compression engine.

    Compresses Layer 0 snapshots into Layer 1 blocks,
    and optionally updates Layer 2 long-term memory.

    Uses ProviderMapper to resolve the LLM module and provider,
    configured via UBP_MEMORY__LLM_PROVIDER environment variable.
    """

    def __init__(
        self,
        prompts_path: Path,
        config: Dict[str, Any],
        di_container: Optional[Any] = None,
    ):
        """
        Initialize compression engine.

        Args:
            prompts_path: Path to the prompts/ directory.
            config: Compression configuration dict.
            di_container: DI container for module resolution.
        """
        self._prompts_path = prompts_path
        self._config = config
        self._di_container = di_container
        self._llm_module = None
        self._provider_name: Optional[str] = None

        # Load prompt templates
        self._compress_prompt = self._load_prompt("compress_layer1.md")
        self._layer2_prompt = self._load_prompt("update_layer2.md")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from the prompts directory."""
        prompt_file = self._prompts_path / filename
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        logger.warning(f"[CompressionEngine] Prompt file not found: {prompt_file}")
        return ""

    async def _resolve_llm(self) -> Tuple[Optional[Any], Optional[str]]:
        """
        Resolve LLM module via ProviderMapper resolve_chain.

        Resolution strategy (ENV-driven, no hardcoded fallback):
        1. If UBP_COMPRESSION__PROVIDER is set, try that single provider first.
        2. Always fall through to resolve_chain(UBP_COMPRESSION__ROLE) which
           returns a health-filtered N-level chain managed by ProviderMapper.

        Returns:
            Tuple of (llm_module, provider_name) or (None, None) if unavailable.
        """
        if self._llm_module:
            return self._llm_module, self._provider_name

        if not self._di_container:
            logger.warning("[CompressionEngine] No DI container — cannot resolve LLM")
            return None, None

        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper

            # Build candidate chain: explicit provider (if set) + resolve_chain
            candidates: List[Tuple[str, str]] = []

            # 1. Explicit provider from ENV or config
            explicit = (
                _COMPRESSION_PROVIDER
                or self._config.get("compression", {}).get("llm_provider", "")
            )
            if explicit and explicit in ProviderMapper.PROVIDER_MAP:
                candidates.append(ProviderMapper.PROVIDER_MAP[explicit])

            # 2. Full resolve_chain for the configured role (health-aware)
            role = _COMPRESSION_ROLE
            chain = ProviderMapper.resolve_chain(role)
            for entry in chain:
                if entry not in candidates:
                    candidates.append(entry)

            # 3. Walk chain, resolve first available module
            for module_name, internal_provider in candidates:
                try:
                    module = await self._di_container.resolve(module_name)
                    if module:
                        self._llm_module = module
                        self._provider_name = internal_provider
                        logger.info(
                            "[CompressionEngine] LLM resolved: %s/%s (role=%s)",
                            module_name, internal_provider, role,
                        )
                        return module, internal_provider
                except (ValueError, Exception) as err:
                    logger.debug(
                        "[CompressionEngine] DI resolve failed for '%s': %s",
                        module_name, err,
                    )
                    continue

            logger.warning(
                "[CompressionEngine] No LLM resolved from chain (%d candidates, role=%s)",
                len(candidates), role,
            )

        except ImportError:
            logger.warning(
                "[CompressionEngine] ProviderMapper not available, "
                "compression will use fallback"
            )
        except Exception as e:
            logger.warning("[CompressionEngine] LLM resolution error: %s", e)

        return None, None

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the resolved LLM with a prompt and return the raw text response.

        Args:
            prompt: The full prompt string.

        Returns:
            Raw text response from the LLM, or None on failure.
        """
        module, provider = await self._resolve_llm()
        if not module:
            logger.warning("[CompressionEngine] No LLM module available for compression")
            return None

        temperature = _COMPRESSION_TEMPERATURE
        max_tokens = self._config.get("compression", {}).get("max_tokens", 1500)

        try:
            # Prefer chat() over generate() for thinking control
            if hasattr(module, "chat"):
                kwargs = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if provider:
                    kwargs["provider"] = provider

                result = await module.chat(**kwargs)
                msg = result.get("message", {})
                return msg.get("content", "") if isinstance(msg, dict) else ""
            elif hasattr(module, "generate"):
                kwargs = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if provider:
                    kwargs["provider"] = provider

                result = await module.generate(**kwargs)
                return result.get("text", "")
            else:
                logger.error("[CompressionEngine] LLM module has no chat/generate method")
                return None

        except Exception as e:
            logger.error(f"[CompressionEngine] LLM call failed: {e}")
            return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from LLM response text.

        Handles responses wrapped in markdown code blocks.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        if not text:
            return None

        # Strip markdown code blocks (only opening/closing fences)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove opening fence (first line) and closing fence (last line)
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                lines = lines[1:-1]
            elif len(lines) >= 1:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

            logger.warning(
                f"[CompressionEngine] Failed to parse JSON from LLM response: "
                f"{cleaned[:200]}..."
            )
            return None

    async def compress(
        self,
        snapshots: List[Dict[str, Any]],
        current_layer1: List[Dict[str, Any]],
        current_layer2: Dict[str, Any],
    ) -> CompressionResult:
        """
        Compress Layer 0 snapshots into a new Layer 1 block,
        and optionally update Layer 2.

        Args:
            snapshots: Layer 0 snapshots to compress.
            current_layer1: Existing Layer 1 blocks.
            current_layer2: Current Layer 2 memory dict.

        Returns:
            CompressionResult with new block and optional Layer 2 update.
        """
        if not snapshots:
            return CompressionResult()

        # Build the compression prompt (use replace to avoid conflicts with JSON braces)
        prompt = self._compress_prompt
        prompt = prompt.replace("{layer0_snapshots}", format_snapshots_for_prompt(snapshots))
        prompt = prompt.replace("{current_layer1}", format_layer1_for_prompt(current_layer1))
        prompt = prompt.replace("{current_layer2}", format_layer2_for_prompt(current_layer2))

        # Call LLM
        raw_response = await self._call_llm(prompt)

        if not raw_response:
            # Fallback: create a basic block without LLM
            logger.warning(
                "[CompressionEngine] LLM unavailable, using rule-based fallback"
            )
            return self._fallback_compress(snapshots)

        # Parse response
        parsed = self._extract_json(raw_response)
        if not parsed:
            logger.warning(
                "[CompressionEngine] Could not parse LLM compression response, "
                "using fallback"
            )
            return self._fallback_compress(snapshots)

        return self._parse_compression_result(parsed)

    def _parse_compression_result(
        self, parsed: Dict[str, Any]
    ) -> CompressionResult:
        """Parse the raw JSON response into a CompressionResult."""
        result = CompressionResult()

        # Parse new Layer 1 block
        raw_block = parsed.get("new_layer1_block")
        if raw_block and isinstance(raw_block, dict):
            try:
                result.new_layer1_block = Layer1Block(**raw_block)
            except Exception as e:
                logger.warning(f"[CompressionEngine] Invalid Layer 1 block: {e}")

        # Parse Layer 2 update
        result.layer2_updated = parsed.get("layer2_updated", False)
        raw_layer2 = parsed.get("updated_layer2")
        if result.layer2_updated and raw_layer2 and isinstance(raw_layer2, dict):
            try:
                result.updated_layer2 = Layer2Memory(**raw_layer2)
            except Exception as e:
                logger.warning(f"[CompressionEngine] Invalid Layer 2 update: {e}")
                result.layer2_updated = False

        return result

    def _fallback_compress(
        self, snapshots: List[Dict[str, Any]]
    ) -> CompressionResult:
        """
        Rule-based fallback compression when LLM is unavailable.

        Merges key_facts, preferences, and entities from snapshots
        into a basic Layer 1 block.
        """
        from .utils import build_turn_range

        all_facts = []
        all_explicit = []
        all_inferred = []
        all_choices = []
        all_pending = []
        focus_parts = []

        for snap in snapshots:
            all_facts.extend(snap.get("key_facts", []))
            prefs = snap.get("preferences", {})
            all_explicit.extend(prefs.get("explicit", []))
            all_inferred.extend(prefs.get("inferred", []))
            all_pending.extend(snap.get("pending", []))
            if snap.get("focus"):
                focus_parts.append(snap["focus"])

        # Deduplicate
        all_facts = list(dict.fromkeys(all_facts))
        all_explicit = list(dict.fromkeys(all_explicit))
        all_inferred = list(dict.fromkeys(all_inferred))

        turn_range = build_turn_range(snapshots)
        last_turn = max((s.get("turn", 0) for s in snapshots), default=0)

        block = Layer1Block(
            turn_range=turn_range,
            focus="; ".join(focus_parts[:3]) if focus_parts else "mixed topics",
            evolution_summary="Rule-based compression (LLM unavailable)",
            user_choices=all_choices,
            user_rules=[],
            specifications=[],
            key_facts=all_facts[:10],
            preferences={
                "explicit": all_explicit[:5],
                "inferred": all_inferred[:5],
            },
            dynamic_context={},
            importance=5,
            last_updated_turn=last_turn,
        )

        return CompressionResult(new_layer1_block=block)
