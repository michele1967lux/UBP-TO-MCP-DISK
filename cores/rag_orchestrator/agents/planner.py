"""
ARCHITECTURE v2.4: Dynamic Swarm Reporting - Dynamic Planner

The "Big Brain" that creates report plans dynamically using LLM.

Instead of relying on static templates, this planner:
1. Analyzes the user's request with a powerful LLM (Planner Model)
2. Generates a custom report structure with sections
3. Assigns source preferences (RAG/WEB/MIXED) per section
4. Returns a ReportPlan compatible with the Session Manager

Environment Configuration:
    UBP_REPORT__PLANNER_PROVIDER=  (auto via ProviderMapper, oppure grok/vllm_remote)
    UBP_REPORT__PLANNER_TEMPERATURE=0.7
    UBP_REPORT__PLANNER_MAX_TOKENS=2000
    UBP_REPORT__DYNAMIC_PLANNING=true
    UBP_REPORT__MAX_SECTIONS=10

Author: UBP Team
Version: 2.4.0
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .report_session import ReportPlan, SectionPlan, SourcePreference
from .report_utils import extract_subject

# BUG-2 fix: ProviderMapper for LLM delegation fallback
try:
    from ubp_enterprise_hybrid.modules.cores._shared import ProviderMapper
    _PROVIDER_MAPPER_OK = True
except ImportError:
    _PROVIDER_MAPPER_OK = False

logger = logging.getLogger(__name__)


# =============================================================================
# PLANNER PROMPTS
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a Senior Research Analyst and Report Architect.
Your job is to analyze user requests and create structured research plans.

You must return a valid JSON response with the following structure:
{
    "title": "Report title",
    "sections": [
        {
            "title": "Section title",
            "description": "What this section should cover",
            "source_preference": "rag_only|web_only|rag_first|mixed|llm_reasoning",
            "suggested_queries": ["query1", "query2"]
        }
    ]
}

Source preference guidelines:
- "rag_only": Use for internal/proprietary information (company docs, system info)
- "web_only": Use for external/current information (market data, news, competitors)
- "rag_first": Use when internal docs might help but external fallback is acceptable
- "mixed": Use when both internal and external sources are valuable
- "llm_reasoning": Use for analysis, recommendations, conclusions

Keep sections focused and actionable. Aim for 3-6 sections unless the topic requires more.
"""

PLANNER_USER_PROMPT = """Create a structured research plan for the following report request:

**User Request:** {query}

**Available Context/History:**
{context}

**Available Internal Knowledge Bases:**
{collections}

Return ONLY valid JSON, no explanation."""


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PlannerConfig:
    """Configuration for the Dynamic Planner."""

    # Provider settings (v6.0.1: model resolved by inference module)
    planner_provider: str = ""
    temperature: float = 0.4   # v6.8.5: lowered from 0.7 for structured plan output
    max_tokens: int = 2000

    # Planning settings
    dynamic_planning_enabled: bool = True
    fallback_to_static: bool = True
    max_sections: int = 10

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        """Load configuration from environment variables."""
        return cls(
            planner_provider=os.getenv("UBP_REPORT__PLANNER_PROVIDER", ""),
            temperature=float(os.getenv("UBP_REPORT__PLANNER_TEMPERATURE", "0.4")),
            max_tokens=int(os.getenv("UBP_REPORT__PLANNER_MAX_TOKENS", "2000")),
            dynamic_planning_enabled=os.getenv("UBP_REPORT__DYNAMIC_PLANNING", "true").lower() == "true",
            fallback_to_static=os.getenv("UBP_REPORT__FALLBACK_TO_STATIC", "true").lower() == "true",
            max_sections=int(os.getenv("UBP_REPORT__MAX_SECTIONS", "10")),
        )


# =============================================================================
# DYNAMIC PLANNER
# =============================================================================

class DynamicPlanner:
    """
    Dynamic Report Planner using LLM.

    The "Big Brain" that analyzes requests and generates custom report structures.
    Uses a powerful model (Planner Model) for intelligent plan generation.

    Usage:
        planner = DynamicPlanner(llm_module)
        plan = await planner.create_plan(
            query="Analisi comparativa tra Qdrant e Milvus",
            context="User is building a RAG system",
            collections=["ubp_system_docs"]
        )
    """

    def __init__(
        self,
        llm_module=None,
        config: Optional[PlannerConfig] = None,
    ):
        """
        Initialize DynamicPlanner.

        Args:
            llm_module: LLM module with generate() method
            config: Optional planner configuration
        """
        self.llm = llm_module
        self.config = config or PlannerConfig.from_env()

        # v6.4.2: Provider resolution with ProviderMapper fallback (BUG-2 fix)
        # Priority: env var explicit > ProviderMapper chain > config default
        env_explicit = os.getenv("UBP_REPORT__PLANNER_PROVIDER")
        if env_explicit:
            self._provider = env_explicit
        elif _PROVIDER_MAPPER_OK:
            try:
                chain = ProviderMapper.resolve_chain("enrichment")
                if chain:
                    self._provider = chain[0][1]  # (module_name, provider_name)
                    logger.info(f"[PLANNER] ProviderMapper resolved: {self._provider}")
                else:
                    self._provider = self.config.planner_provider
            except Exception as e:
                logger.warning(f"[PLANNER] ProviderMapper failed: {e}")
                self._provider = self.config.planner_provider
        else:
            self._provider = self.config.planner_provider

        logger.info(
            f"DynamicPlanner initialized: provider={self._provider}, "
            f"dynamic={self.config.dynamic_planning_enabled}"
        )

    async def create_plan(
        self,
        query: str,
        context: Optional[str] = None,
        collections: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> ReportPlan:
        """
        Create a dynamic report plan using LLM.

        Args:
            query: User's report request
            context: Optional conversation history or context
            collections: Optional list of available RAG collections
            conversation_id: Optional conversation ID for tracking

        Returns:
            ReportPlan with dynamically generated sections

        Raises:
            ValueError: If plan generation fails and no fallback
        """
        start_time = time.time()

        if not self.config.dynamic_planning_enabled:
            raise ValueError("Dynamic planning is disabled")

        logger.info(
            f"[PLANNER] Creating dynamic plan for: '{query[:50]}...'",
            extra={
                "provider": self._provider,
                "collections": collections,
            }
        )

        # Build prompt
        user_prompt = PLANNER_USER_PROMPT.format(
            query=query,
            context=context or "No previous context",
            collections=", ".join(collections) if collections else "None specified",
        )

        # Retry-eligible exceptions (network/timeout issues)
        _RETRY_EXCEPTIONS = (
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
        # Immediate fallback exceptions (parsing/data issues)
        _FALLBACK_EXCEPTIONS = (
            json.JSONDecodeError,
            KeyError,
        )

        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                # Call LLM
                response = await self._call_llm(user_prompt)

                # Parse JSON response
                plan_data = self._parse_llm_response(response)

                # Convert to ReportPlan
                plan = self._build_report_plan(
                    plan_data=plan_data,
                    query=query,
                    collections=collections or [],
                )

                elapsed_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"[PLANNER] Dynamic plan created: {len(plan.sections)} sections "
                    f"(attempt {attempt + 1})",
                    extra={
                        "sections": [s.title for s in plan.sections],
                        "time_ms": elapsed_ms,
                    }
                )

                return plan

            except _FALLBACK_EXCEPTIONS as e:
                # Parsing errors: immediate fallback, no retry
                logger.error(f"[PLANNER] Plan parsing failed: {e}")
                last_error = e
                break

            except _RETRY_EXCEPTIONS as e:
                # Network/timeout errors: retry with backoff
                last_error = e
                if attempt < 2:
                    delay = 2 ** attempt
                    logger.warning(
                        f"[PLANNER] Plan generation network error (attempt {attempt + 1}/3): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"[PLANNER] Plan generation failed after 3 attempts: {e}")

            except ValueError as e:
                # LLM response parsing issues: immediate fallback
                logger.error(f"[PLANNER] Plan value error: {e}")
                last_error = e
                break

            except Exception as e:
                # Unexpected errors: log and fallback
                logger.error(f"[PLANNER] Unexpected plan generation error: {type(e).__name__}: {e}")
                last_error = e
                break

        # All retries exhausted or non-retryable error
        if self.config.fallback_to_static:
            logger.info(
                f"[PLANNER] Falling back to static template (reason: {type(last_error).__name__})"
            )
            return self._create_fallback_plan(query, collections or [])

        raise ValueError(f"Failed to create plan: {last_error}")

    async def _call_llm(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with planner prompts."""
        if not self.llm:
            raise ValueError("LLM module not available")

        # Combine system prompt with user prompt (LLM module doesn't support system_prompt param)
        sys_prompt = system_prompt if system_prompt is not None else PLANNER_SYSTEM_PROMPT
        combined_prompt = f"""### SYSTEM INSTRUCTIONS ###
{sys_prompt}

### USER REQUEST ###
{user_prompt}"""

        # Build generation parameters
        params = {
            "prompt": combined_prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # v6.8.x: Re-resolve provider at call time (health may change)
        if _PROVIDER_MAPPER_OK:
            try:
                chain = ProviderMapper.resolve_chain("enrichment")
                if chain:
                    resolved = chain[0][1]
                    if resolved != self._provider:
                        logger.info("[PLANNER] Provider re-resolved: %s → %s", self._provider, resolved)
                        self._provider = resolved
            except Exception:
                pass  # keep existing self._provider

        if self._provider:
            params["provider"] = self._provider

        # Call LLM
        result = await self.llm.generate(**params)

        # Extract response text
        if isinstance(result, dict):
            return result.get("response", result.get("text", ""))
        return str(result)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles common issues like markdown code blocks, extra text,
        and models that echo prompt templates before the JSON output.
        v6.0.1: Brace-counting extraction for robustness with smaller models.
        """
        if not response:
            raise ValueError("Empty LLM response")

        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if json_match:
            response = json_match.group(1)

        # Brace-counting: find first { and its matching }
        json_start = response.find("{")
        if json_start >= 0:
            depth = 0
            in_string = False
            escape_next = False
            for i in range(json_start, len(response)):
                ch = response[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    if in_string:
                        escape_next = True
                        continue  # v6.4.1: Skip only when inside a string
                    # Outside strings: backslash is invalid JSON, but don't skip
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = response[json_start:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON parse error (brace-matched): {e}")
                        break

        # Fallback: first { to last }
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (first-last): {e}")

        # Attempt direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")

    def _build_report_plan(
        self,
        plan_data: Dict[str, Any],
        query: str,
        collections: List[str],
    ) -> ReportPlan:
        """Convert parsed JSON to ReportPlan object."""
        sections = []

        for i, section_data in enumerate(plan_data.get("sections", [])):
            if i >= self.config.max_sections:
                logger.warning(
                    f"[PLANNER] Max sections ({self.config.max_sections}) reached, truncating"
                )
                break

            # Parse source preference
            source_pref_str = section_data.get("source_preference", "rag_first")
            try:
                source_pref = SourcePreference(source_pref_str)
            except ValueError:
                source_pref = SourcePreference.RAG_FIRST

            section = SectionPlan(
                title=section_data.get("title", f"Section {i + 1}"),
                description=section_data.get("description", ""),
                source_preference=source_pref,
                required=section_data.get("required", True),
                max_tokens=section_data.get("max_tokens", 1000),
                suggested_queries=section_data.get("suggested_queries", []),
                depends_on=section_data.get("depends_on", []),
            )
            sections.append(section)

        # Extract subject from query
        subject = extract_subject(query)

        return ReportPlan(
            template_id="dynamic",
            template_name=plan_data.get("title", "Dynamic Report"),
            subject=subject,
            sections=sections,
            collections=collections,
            user_modifications=[],
        )

    def _create_fallback_plan(
        self,
        query: str,
        collections: List[str],
    ) -> ReportPlan:
        """
        Create a generic fallback plan when dynamic planning fails.

        Uses a sensible default structure for any report request.
        """
        subject = extract_subject(query)

        sections = [
            SectionPlan(
                title="Executive Summary",
                description="Overview and key findings",
                source_preference=SourcePreference.MIXED,
                max_tokens=500,
            ),
            SectionPlan(
                title="Background",
                description="Context and relevant information",
                source_preference=SourcePreference.RAG_FIRST,
                max_tokens=800,
            ),
            SectionPlan(
                title="Analysis",
                description="Detailed analysis of the topic",
                source_preference=SourcePreference.RAG_FIRST,
                max_tokens=1200,
            ),
            SectionPlan(
                title="External Context",
                description="External information and market context",
                source_preference=SourcePreference.WEB_ONLY,
                max_tokens=800,
            ),
            SectionPlan(
                title="Conclusions & Recommendations",
                description="Summary and actionable recommendations",
                source_preference=SourcePreference.LLM_REASONING,
                max_tokens=600,
                depends_on=["Analysis", "External Context"],
            ),
        ]

        return ReportPlan(
            template_id="fallback_generic",
            template_name="Generic Research Report",
            subject=subject,
            sections=sections,
            collections=collections,
        )

    async def refine_plan(
        self,
        plan: ReportPlan,
        user_feedback: str,
    ) -> ReportPlan:
        """
        Refine an existing plan based on user feedback.

        Args:
            plan: Current report plan
            user_feedback: User's modification request

        Returns:
            Updated ReportPlan with tracked modifications
        """
        logger.info(f"[PLANNER] Refining plan: '{user_feedback[:50]}...'")

        # Serialize current sections with all structural fields
        current_sections_json = json.dumps([
            {
                "title": s.title,
                "description": s.description,
                "source_preference": s.source_preference.value,
                "required": s.required,
                "max_tokens": s.max_tokens,
                "suggested_queries": s.suggested_queries,
            }
            for s in plan.sections
        ], ensure_ascii=False, indent=2)

        refinement_prompt = f"""You are modifying an existing report plan.

**Current plan for:** "{plan.subject}"
**Current title:** "{plan.template_name}"
**Current sections (JSON):**
{current_sections_json}

**User modification request:** {user_feedback}

**Rules:**
- Apply the user's request precisely (add, remove, reorder, or modify sections).
- NEVER remove or merge existing sections unless the user EXPLICITLY asks to remove or merge them.
- Preserve all existing sections that the user did NOT ask to change.
- Preserve all fields for unchanged sections (title, description, source_preference, required, max_tokens, suggested_queries).
- For new sections, generate appropriate suggested_queries (2-3 per section), set required=true, max_tokens=1000.
- Keep "Analysis and Recommendations" or similar conclusion sections at the end.
- Return the COMPLETE updated plan with ALL sections (not just the changed ones).

Return ONLY valid JSON:
{{
    "title": "Updated report title",
    "sections": [
        {{
            "title": "Section title",
            "description": "What this section covers",
            "source_preference": "rag_only|web_only|rag_first|mixed|llm_reasoning",
            "required": true,
            "max_tokens": 1000,
            "suggested_queries": ["query1", "query2"]
        }}
    ]
}}"""

        # Use a dedicated system prompt for refinement — no section count limit
        refine_system = (
            "You are a Report Plan Editor. You modify existing report plans "
            "according to user requests. You must return valid JSON only. "
            "NEVER remove sections unless the user explicitly asks to remove them."
        )

        try:
            response = await self._call_llm(refinement_prompt, system_prompt=refine_system)
            plan_data = self._parse_llm_response(response)

            refined_plan = self._build_report_plan(
                plan_data=plan_data,
                query=plan.subject,
                collections=plan.collections,
            )

            # Validate: refined plan must have at least 1 section
            if not refined_plan.sections:
                logger.warning("[PLANNER] Refined plan has 0 sections, keeping original")
                plan.user_modifications.append(f"[FAILED] {user_feedback}")
                return plan

            # Track modification history
            refined_plan.user_modifications = plan.user_modifications + [user_feedback]

            logger.info(
                f"[PLANNER] Plan refined: {len(plan.sections)} -> {len(refined_plan.sections)} sections",
                extra={"feedback": user_feedback[:80]},
            )

            return refined_plan

        except Exception as e:
            logger.warning(f"[PLANNER] Refinement failed: {e}, keeping original plan")
            plan.user_modifications.append(f"[FAILED] {user_feedback}")
            return plan


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def should_use_dynamic_planning(query: str) -> bool:
    """
    Determine if a query should use dynamic planning vs static templates.

    Returns True for complex/unique requests, False for common patterns.
    """
    # Check for comparison queries (always dynamic)
    if re.search(r"(confronto|comparison|versus|vs\.?|differenz)", query, re.I):
        return True

    # Check for specific entity analysis
    if re.search(r"(analisi|analysis)\s+(di|of|su)\s+\w+", query, re.I):
        return True

    # Check query complexity (word count)
    word_count = len(query.split())
    if word_count > 15:
        return True

    return False
