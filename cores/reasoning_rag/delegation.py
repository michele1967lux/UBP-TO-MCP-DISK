"""
reasoning_rag/delegation.py

Delegation layer for LLM and retrieval operations.
Implements core reasoning strategies.

Strategies:
- Self-Ask: Iterative sub-question decomposition
- Chain-of-Thought: Interleaved reasoning and retrieval
- Evidence Attribution: Citation and source tracking
- Verification: Multi-source fact checking

ZERO direct imports from other modules - uses DI for resolution.

v1.0.0: Initial release
"""

from __future__ import annotations

# WARN-CV-001 fix: shared LLM response normalizer
try:
    from ubp_enterprise_hybrid.modules.cores._shared.utils import extract_llm_text as _extract_llm_text
except ImportError:
    _extract_llm_text = None  # type: ignore[assignment]

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .prompts import (
    get_template,
    detect_language,
    ReasoningStrategy,
    QueryComplexity,
    QueryIntent,
)
from .providers import (
    ReasoningResult,
    ReasoningTrace,
    ReasoningTraceEntry,
    SubQuestion,
    ReasoningStep,
    Claim,
    Evidence,
    Citation,
    Verification,
    Contradiction,
    RetrievedDocument,
    QueryAnalysis,
    VerificationStatus,
    AttributionType,
    SelfAskConfig,
    ChainOfThoughtConfig,
    EvidenceConfig,
    VerificationConfig,
    RetrievalConfig,
    DebugConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class IModuleRegistry(Protocol):
    """Protocol for module registry."""
    def get_module(self, module_name: str) -> Optional[Any]: ...
    def is_module_loaded(self, module_name: str) -> bool: ...


class IEventPublisher(Protocol):
    """Protocol for event publishing."""
    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None: ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LLMDelegationConfig:
    """Configuration for LLM delegation."""
    llm_module: str = "inference_ollama_grok"
    llm_operation: str = "generate"
    timeout_seconds: int = 30
    max_retries: int = 2
    fallback_enabled: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ["chain_of_thought", "self_ask", "direct"])


# ============================================================================
# Reasoning Delegator
# ============================================================================


class ReasoningDelegator:
    """
    Handles LLM and retrieval delegation for reasoning strategies.
    
    Features:
    - Self-Ask iterative decomposition
    - Chain-of-Thought interleaved reasoning
    - Evidence attribution with citations
    - Multi-source verification
    - Retrieval integration
    """
    
    def __init__(
        self,
        llm_config: LLMDelegationConfig,
        retrieval_config: RetrievalConfig,
        self_ask_config: SelfAskConfig,
        cot_config: ChainOfThoughtConfig,
        evidence_config: EvidenceConfig,
        verification_config: VerificationConfig,
        module_registry: IModuleRegistry,
        event_publisher: Optional[IEventPublisher] = None,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.llm_config = llm_config
        self.retrieval_config = retrieval_config
        self.self_ask_config = self_ask_config
        self.cot_config = cot_config
        self.evidence_config = evidence_config
        self.verification_config = verification_config
        self._module_registry = module_registry
        self._event_publisher = event_publisher
        self._debug = debug_config or DebugConfig()
        
        self._llm_module: Optional[Any] = None
        self._retrieval_module: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Check if LLM module is available."""
        module = self._module_registry.get_module(self.llm_config.llm_module)
        return module is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of delegation."""
        try:
            llm_module = await self._get_llm_module()
            retrieval_module = await self._get_retrieval_module()
            
            return {
                "status": "available" if llm_module else "degraded",
                "llm_module": self.llm_config.llm_module,
                "llm_available": llm_module is not None,
                "retrieval_module": self.retrieval_config.module,
                "retrieval_available": retrieval_module is not None,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    async def _get_llm_module(self) -> Optional[Any]:
        """Get or resolve the LLM module."""
        if self._llm_module:
            return self._llm_module

        # Try cache first, then async resolution
        module = self._module_registry.get_module(self.llm_config.llm_module)
        if not module and hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module(self.llm_config.llm_module)
        if module:
            self._llm_module = module
        return module
    
    async def _get_retrieval_module(self) -> Optional[Any]:
        """Get or resolve the retrieval module."""
        if self._retrieval_module:
            return self._retrieval_module

        module = self._module_registry.get_module(self.retrieval_config.module)
        if not module and hasattr(self._module_registry, "resolve_module"):
            module = await self._module_registry.resolve_module(self.retrieval_config.module)
        if module:
            self._retrieval_module = module
        return module
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        """Call the LLM module with ProviderMapper fallback chain."""
        # Try primary module first (cached)
        module = await self._get_llm_module()
        if module:
            try:
                return await self._execute_llm_call(module, prompt, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"[REASONING] Primary LLM failed: {e}, trying fallback chain")
                self._llm_module = None  # Clear cache to force re-resolution

        # Fallback: walk ProviderMapper chain
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.provider_mapper import ProviderMapper
            chain = ProviderMapper.resolve_chain("rag")
            for module_name, provider_name in chain:
                if module_name == self.llm_config.llm_module:
                    continue  # Skip already-failed primary
                fallback_module = self._module_registry.get_module(module_name)
                if not fallback_module and hasattr(self._module_registry, "resolve_module"):
                    fallback_module = await self._module_registry.resolve_module(module_name)
                if fallback_module:
                    try:
                        result = await self._execute_llm_call(fallback_module, prompt, temperature, max_tokens)
                        logger.info(f"[REASONING] Fallback succeeded with {module_name}")
                        return result
                    except Exception as fb_err:
                        logger.warning(f"[REASONING] Fallback {module_name} also failed: {fb_err}")
        except Exception as ie:
            logger.warning(
                f"[REASONING] ProviderMapper NOT AVAILABLE during fallback chain walk. "
                f"Cannot attempt alternative providers. Cause: {ie}"
            )

        raise RuntimeError(f"LLM module '{self.llm_config.llm_module}' not available (all fallbacks exhausted)")

    async def _execute_llm_call(self, module: Any, prompt: str, temperature: float, max_tokens: int) -> str:
        """Execute a single LLM call on given module."""
        if self._debug.log_prompts:
            logger.debug(f"[REASONING] Prompt:\n{prompt[:500]}...")

        operation = getattr(module, self.llm_config.llm_operation, None)
        if not operation:
            raise RuntimeError(f"Operation '{self.llm_config.llm_operation}' not found")

        result = await asyncio.wait_for(
            operation(prompt=prompt, temperature=temperature, max_tokens=max_tokens),
            timeout=self.llm_config.timeout_seconds,
        )

        # WARN-CV-001: shared normalizer
        if _extract_llm_text is not None:
            response = _extract_llm_text(result)
        elif isinstance(result, dict):
            response = result.get("text") or result.get("response") or result.get("content", "")
        else:
            response = str(result)

        if self._debug.log_responses:
            logger.debug(f"[REASONING] Response:\n{response[:500]}...")

        return response
    
    async def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve documents for a query."""
        module = await self._get_retrieval_module()
        if not module:
            logger.warning(f"Retrieval module '{self.retrieval_config.module}' not available")
            return []
        
        top_k = top_k or self.retrieval_config.default_top_k
        
        try:
            operation = getattr(module, self.retrieval_config.operation, None)
            if not operation:
                logger.warning(f"Retrieval operation '{self.retrieval_config.operation}' not found")
                return []
            
            result = await asyncio.wait_for(
                operation(query=query, top_k=top_k),
                timeout=self.retrieval_config.timeout_seconds,
            )
            
            # Parse result into RetrievedDocument objects
            docs = []
            if isinstance(result, dict) and "results" in result:
                for i, item in enumerate(result["results"]):
                    docs.append(RetrievedDocument(
                        doc_id=item.get("id", str(uuid.uuid4())),
                        content=item.get("content", item.get("text", "")),
                        score=float(item.get("score", 0.0)),
                        source=item.get("source", item.get("metadata", {}).get("source", "")),
                        page=item.get("page"),
                        metadata=item.get("metadata", {}),
                    ))
            elif isinstance(result, list):
                for i, item in enumerate(result):
                    if isinstance(item, dict):
                        docs.append(RetrievedDocument(
                            doc_id=item.get("id", str(uuid.uuid4())),
                            content=item.get("content", item.get("text", "")),
                            score=float(item.get("score", 0.0)),
                            source=item.get("source", ""),
                            metadata=item.get("metadata", {}),
                        ))
            
            if self._debug.log_retrievals:
                logger.debug(f"[REASONING] Retrieved {len(docs)} documents for: {query[:50]}...")
            
            return docs
            
        except asyncio.TimeoutError:
            logger.warning(f"Retrieval timeout for: {query[:50]}...")
            return []
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try full response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"raw_response": response}
    
    # ========================================================================
    # Self-Ask Strategy
    # ========================================================================
    
    async def execute_self_ask(
        self,
        query: str,
        session_id: str,
        language: str = "en",
    ) -> ReasoningResult:
        """
        Execute Self-Ask reasoning strategy.
        
        Iteratively:
        1. Generate sub-questions
        2. Retrieve for each sub-question
        3. Integrate answers
        4. Repeat until convergence or max iterations
        """
        start_time = time.perf_counter()
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.SELF_ASK,
        )
        
        all_sub_questions: List[SubQuestion] = []
        all_retrieved_docs: List[RetrievedDocument] = []
        accumulated_context = ""
        previous_questions: List[str] = []
        
        final_answer = ""
        final_confidence = 0.0
        
        for iteration in range(1, self.self_ask_config.max_iterations + 1):
            iter_start = time.perf_counter()
            
            # Generate sub-questions
            decomposition_prompt = get_template("self_ask_decomposition", language).format(
                query=query,
                context=accumulated_context or "No context gathered yet.",
                iteration=iteration,
                max_iterations=self.self_ask_config.max_iterations,
                max_sub_questions=self.self_ask_config.max_sub_questions_per_iteration,
                previous_questions=", ".join(previous_questions) if previous_questions else "None",
            )
            
            decomposition_response = await self._call_llm(
                prompt=decomposition_prompt,
                temperature=self.self_ask_config.sub_question_temperature,
            )
            
            decomposition = self._parse_json_response(decomposition_response)
            
            # Check if we can answer now
            can_answer = decomposition.get("can_answer_now", False)
            current_confidence = float(decomposition.get("confidence", 0.0))
            
            if can_answer or (
                self.self_ask_config.early_stop_on_confidence and
                current_confidence >= self.self_ask_config.confidence_threshold
            ):
                if self._debug.log_sub_questions:
                    logger.info(f"[SELF-ASK] Early stop at iteration {iteration}, confidence: {current_confidence}")
                break
            
            # Process sub-questions
            sub_questions = decomposition.get("sub_questions", [])
            
            for sq_text in sub_questions[:self.self_ask_config.max_sub_questions_per_iteration]:
                sq = SubQuestion(
                    question_id=str(uuid.uuid4()),
                    question=sq_text,
                    iteration=iteration,
                )
                
                # Retrieve for sub-question
                docs = await self._retrieve(sq_text, self.self_ask_config.retrieval_top_k)
                sq.retrieved_docs = docs
                all_retrieved_docs.extend(docs)
                
                # Generate answer for sub-question if we have docs
                if docs:
                    context_text = "\n\n".join([d.content for d in docs[:3]])
                    sq_answer_prompt = f"""Answer this sub-question based on the context:
                    
Sub-question: {sq_text}

Context:
{context_text}

Provide a concise answer:"""
                    
                    sq_answer = await self._call_llm(
                        prompt=sq_answer_prompt,
                        temperature=0.2,
                        max_tokens=300,
                    )
                    sq.answer = sq_answer.strip()
                    sq.answered = True
                    sq.confidence = 0.7  # Base confidence for retrieved answer
                
                all_sub_questions.append(sq)
                previous_questions.append(sq_text)
                
                # Add to accumulated context
                if sq.answer:
                    accumulated_context += f"\nQ: {sq_text}\nA: {sq.answer}\n"
                
                # Log
                trace.add_entry(
                    entry_type="sub_question",
                    content=sq.to_dict(),
                    duration_ms=(time.perf_counter() - iter_start) * 1000,
                )
            
            if self._debug.log_sub_questions:
                logger.info(f"[SELF-ASK] Iteration {iteration}: {len(sub_questions)} sub-questions")
        
        # Integration: Synthesize final answer
        if all_sub_questions:
            sub_qa_pairs = "\n\n".join([
                f"Q{i+1}: {sq.question}\nA{i+1}: {sq.answer or 'No answer found'}"
                for i, sq in enumerate(all_sub_questions)
            ])
            
            integration_prompt = get_template("self_ask_integration", language).format(
                query=query,
                sub_qa_pairs=sub_qa_pairs,
            )
            
            integration_response = await self._call_llm(
                prompt=integration_prompt,
                temperature=self.self_ask_config.integration_temperature,
            )
            
            integration = self._parse_json_response(integration_response)
            final_answer = integration.get("answer", "Unable to synthesize answer.")
            final_confidence = float(integration.get("confidence", 0.5))
            
            trace.add_entry(
                entry_type="integration",
                content={
                    "answer": final_answer,
                    "confidence": final_confidence,
                    "key_points": integration.get("key_points", []),
                },
            )
        else:
            final_answer = "Unable to decompose query into sub-questions."
            final_confidence = 0.0
        
        trace.completed_at = datetime.utcnow()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish event
        if self._event_publisher:
            await self._event_publisher.publish(
                "reasoning.self_ask.completed",
                {
                    "session_id": session_id,
                    "iterations": len(set(sq.iteration for sq in all_sub_questions)),
                    "sub_questions": len(all_sub_questions),
                    "confidence": final_confidence,
                    "time_ms": elapsed_ms,
                },
            )
        
        return ReasoningResult(
            session_id=session_id,
            query=query,
            strategy_used=ReasoningStrategy.SELF_ASK,
            answer=final_answer,
            confidence=final_confidence,
            reasoning_trace=trace,
            sub_questions=all_sub_questions,
            retrieved_docs=all_retrieved_docs,
            time_ms=elapsed_ms,
            iteration_count=max(sq.iteration for sq in all_sub_questions) if all_sub_questions else 0,
        )
    
    # ========================================================================
    # Chain-of-Thought Strategy
    # ========================================================================
    
    async def execute_chain_of_thought(
        self,
        query: str,
        session_id: str,
        language: str = "en",
    ) -> ReasoningResult:
        """
        Execute Chain-of-Thought reasoning with interleaved retrieval.
        
        Iteratively:
        1. Generate a reasoning step
        2. If retrieval needed, fetch relevant documents
        3. Continue reasoning with new context
        4. Synthesize final answer
        """
        start_time = time.perf_counter()
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )
        
        reasoning_steps: List[ReasoningStep] = []
        all_retrieved_docs: List[RetrievedDocument] = []
        accumulated_context = ""
        previous_thoughts: List[str] = []
        
        for step_num in range(1, self.cot_config.max_reasoning_steps + 1):
            step_start = time.perf_counter()
            
            # Generate reasoning step
            reasoning_prompt = get_template("cot_reasoning", language).format(
                query=query,
                context=accumulated_context or "No additional context yet.",
                step=step_num,
                max_steps=self.cot_config.max_reasoning_steps,
                previous_thoughts="\n".join([f"- {t}" for t in previous_thoughts]) if previous_thoughts else "None",
            )
            
            reasoning_response = await self._call_llm(
                prompt=reasoning_prompt,
                temperature=self.cot_config.reasoning_temperature,
            )
            
            step_data = self._parse_json_response(reasoning_response)
            
            step = ReasoningStep(
                step_id=str(uuid.uuid4()),
                step_number=step_num,
                thought=step_data.get("thought", ""),
                needs_retrieval=step_data.get("needs_retrieval", False),
                retrieval_query=step_data.get("retrieval_query"),
                intermediate_conclusion=step_data.get("intermediate_conclusion"),
                confidence=float(step_data.get("confidence", 0.5)),
            )
            
            previous_thoughts.append(step.thought)
            
            # Retrieval if needed
            if step.needs_retrieval and step.retrieval_query:
                docs = await self._retrieve(step.retrieval_query, self.cot_config.retrieval_top_k)
                step.retrieved_docs = docs
                all_retrieved_docs.extend(docs)
                
                # Add to context
                if docs:
                    context_addition = "\n\nRetrieved information:\n" + "\n".join([
                        f"[{i+1}] {d.content[:300]}..." if len(d.content) > 300 else f"[{i+1}] {d.content}"
                        for i, d in enumerate(docs[:3])
                    ])
                    accumulated_context += context_addition
            
            reasoning_steps.append(step)
            
            trace.add_entry(
                entry_type="reasoning_step",
                content=step.to_dict(),
                duration_ms=(time.perf_counter() - step_start) * 1000,
            )
            
            if self._debug.log_reasoning_steps:
                logger.info(f"[COT] Step {step_num}: {step.thought[:100]}...")
            
            # Check if ready for final answer
            if step_data.get("ready_for_final_answer", False):
                break
        
        # Synthesis: Generate final answer
        reasoning_chain = "\n\n".join([
            f"Step {s.step_number}: {s.thought}" +
            (f"\nConclusion: {s.intermediate_conclusion}" if s.intermediate_conclusion else "")
            for s in reasoning_steps
        ])
        
        synthesis_prompt = get_template("cot_synthesis", language).format(
            query=query,
            reasoning_chain=reasoning_chain,
            all_context=accumulated_context or "No additional context retrieved.",
        )
        
        synthesis_response = await self._call_llm(
            prompt=synthesis_prompt,
            temperature=self.cot_config.synthesis_temperature,
        )
        
        synthesis = self._parse_json_response(synthesis_response)
        final_answer = synthesis.get("answer", "Unable to synthesize answer.")
        final_confidence = float(synthesis.get("confidence", 0.5))
        
        trace.add_entry(
            entry_type="synthesis",
            content={
                "answer": final_answer,
                "confidence": final_confidence,
                "reasoning_summary": synthesis.get("reasoning_summary", ""),
            },
        )
        
        trace.completed_at = datetime.utcnow()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish event
        if self._event_publisher:
            await self._event_publisher.publish(
                "reasoning.cot.completed",
                {
                    "session_id": session_id,
                    "steps": len(reasoning_steps),
                    "retrievals": sum(1 for s in reasoning_steps if s.needs_retrieval),
                    "confidence": final_confidence,
                    "time_ms": elapsed_ms,
                },
            )
        
        return ReasoningResult(
            session_id=session_id,
            query=query,
            strategy_used=ReasoningStrategy.CHAIN_OF_THOUGHT,
            answer=final_answer,
            confidence=final_confidence,
            reasoning_trace=trace,
            reasoning_steps=reasoning_steps,
            retrieved_docs=all_retrieved_docs,
            time_ms=elapsed_ms,
            step_count=len(reasoning_steps),
        )
    
    # ========================================================================
    # Evidence Attribution Strategy
    # ========================================================================
    
    async def execute_evidence_attribution(
        self,
        query: str,
        session_id: str,
        language: str = "en",
    ) -> ReasoningResult:
        """
        Execute Evidence Attribution strategy.
        
        1. Retrieve relevant documents
        2. Generate answer with inline citations
        3. Extract and attribute claims
        """
        start_time = time.perf_counter()
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.EVIDENCE_ATTRIBUTION,
        )
        
        # Retrieve documents
        docs = await self._retrieve(query, self.retrieval_config.default_top_k)
        
        if not docs:
            return ReasoningResult(
                session_id=session_id,
                query=query,
                strategy_used=ReasoningStrategy.EVIDENCE_ATTRIBUTION,
                answer="Unable to find relevant sources to answer this question.",
                confidence=0.0,
                reasoning_trace=trace,
                time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        trace.add_entry(
            entry_type="retrieval",
            content={"doc_count": len(docs), "queries": [query]},
        )
        
        # Format sources for prompt
        sources_text = "\n\n".join([
            f"[Source {i+1}] (Score: {d.score:.2f})\n{d.content}"
            for i, d in enumerate(docs)
        ])
        
        # Generate answer with citations
        citation_prompt = get_template("evidence_answer_with_citations", language).format(
            query=query,
            sources=sources_text,
        )
        
        citation_response = await self._call_llm(
            prompt=citation_prompt,
            temperature=0.2,
        )
        
        citation_data = self._parse_json_response(citation_response)
        
        answer_with_citations = citation_data.get("answer_with_citations", "")
        citations_data = citation_data.get("citations", [])
        final_confidence = float(citation_data.get("confidence", 0.5))
        
        # Build citations
        citations: List[Citation] = []
        for cit in citations_data:
            citations.append(Citation(
                citation_id=cit.get("id", len(citations) + 1),
                source_doc_id=docs[cit.get("source_id", 0)].doc_id if cit.get("source_id", 0) < len(docs) else "",
                cited_text=cit.get("text", ""),
                page=cit.get("page"),
                source_title=docs[cit.get("source_id", 0)].source if cit.get("source_id", 0) < len(docs) else "",
            ))
        
        # Extract claims if enabled
        claims: List[Claim] = []
        evidence: List[Evidence] = []
        
        if self.evidence_config.claim_extraction_enabled:
            claim_prompt = get_template("evidence_claim_extraction", language).format(
                text=answer_with_citations,
            )
            
            claim_response = await self._call_llm(
                prompt=claim_prompt,
                temperature=0.1,
            )
            
            claim_data = self._parse_json_response(claim_response)
            
            for claim_item in claim_data.get("claims", []):
                claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    text=claim_item.get("claim", ""),
                    claim_type=claim_item.get("type", "fact"),
                    importance=claim_item.get("importance", "medium"),
                    needs_verification=claim_item.get("needs_verification", True),
                )
                claims.append(claim)
            
            trace.add_entry(
                entry_type="claim_extraction",
                content={"claim_count": len(claims)},
            )
            
            # Attribute claims to sources
            if claims:
                claims_text = "\n".join([f"- {c.text}" for c in claims])
                
                attribution_prompt = get_template("evidence_attribution", language).format(
                    claims=claims_text,
                    sources=sources_text,
                )
                
                attribution_response = await self._call_llm(
                    prompt=attribution_prompt,
                    temperature=0.1,
                )
                
                attribution_data = self._parse_json_response(attribution_response)
                
                for attr in attribution_data.get("attributions", []):
                    # Find matching claim
                    claim_text = attr.get("claim", "")
                    matching_claim = next((c for c in claims if c.text == claim_text), None)
                    
                    if matching_claim:
                        evidence.append(Evidence(
                            evidence_id=str(uuid.uuid4()),
                            claim_id=matching_claim.claim_id,
                            source_doc_id=docs[attr.get("source_ids", [0])[0]].doc_id if attr.get("source_ids") else "",
                            supporting_text=attr.get("supporting_text", ""),
                            confidence=float(attr.get("confidence", 0.5)),
                            attribution_type=AttributionType(attr.get("attribution_type", "direct")),
                        ))
        
        trace.add_entry(
            entry_type="evidence_attribution",
            content={
                "answer": answer_with_citations[:200] + "...",
                "citation_count": len(citations),
                "claim_count": len(claims),
                "evidence_count": len(evidence),
            },
        )
        
        trace.completed_at = datetime.utcnow()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish event
        if self._event_publisher:
            await self._event_publisher.publish(
                "reasoning.evidence.completed",
                {
                    "session_id": session_id,
                    "citations": len(citations),
                    "claims": len(claims),
                    "confidence": final_confidence,
                    "time_ms": elapsed_ms,
                },
            )
        
        return ReasoningResult(
            session_id=session_id,
            query=query,
            strategy_used=ReasoningStrategy.EVIDENCE_ATTRIBUTION,
            answer=answer_with_citations,
            confidence=final_confidence,
            reasoning_trace=trace,
            claims=claims,
            evidence=evidence,
            citations=citations,
            retrieved_docs=docs,
            time_ms=elapsed_ms,
        )
    
    # ========================================================================
    # Verification Strategy
    # ========================================================================
    
    async def execute_verification(
        self,
        query: str,
        session_id: str,
        claims_to_verify: Optional[List[str]] = None,
        language: str = "en",
    ) -> ReasoningResult:
        """
        Execute Verification strategy.
        
        1. Extract claims (if not provided)
        2. Retrieve from multiple sources
        3. Verify each claim
        4. Detect contradictions
        """
        start_time = time.perf_counter()
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.VERIFICATION,
        )
        
        # Extract claims if not provided
        claims: List[Claim] = []
        
        if claims_to_verify:
            for i, claim_text in enumerate(claims_to_verify):
                claims.append(Claim(
                    claim_id=str(uuid.uuid4()),
                    text=claim_text,
                    claim_type="fact",
                    importance="high",
                    needs_verification=True,
                ))
        else:
            # Generate a preliminary answer and extract claims
            docs = await self._retrieve(query, self.retrieval_config.default_top_k)
            
            if docs:
                context = "\n".join([d.content[:500] for d in docs[:3]])
                preliminary_prompt = f"""Answer this question based on the context:

Question: {query}

Context:
{context}

Provide a factual answer:"""
                
                preliminary_answer = await self._call_llm(preliminary_prompt, temperature=0.2)
                
                # Extract claims from preliminary answer
                claim_prompt = get_template("evidence_claim_extraction", language).format(
                    text=preliminary_answer,
                )
                
                claim_response = await self._call_llm(claim_prompt, temperature=0.1)
                claim_data = self._parse_json_response(claim_response)
                
                for claim_item in claim_data.get("claims", []):
                    if claim_item.get("needs_verification", True):
                        claims.append(Claim(
                            claim_id=str(uuid.uuid4()),
                            text=claim_item.get("claim", ""),
                            claim_type=claim_item.get("type", "fact"),
                            importance=claim_item.get("importance", "medium"),
                            needs_verification=True,
                        ))
        
        if not claims:
            return ReasoningResult(
                session_id=session_id,
                query=query,
                strategy_used=ReasoningStrategy.VERIFICATION,
                answer="No claims to verify.",
                confidence=0.0,
                reasoning_trace=trace,
                time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        trace.add_entry(
            entry_type="claim_extraction",
            content={"claim_count": len(claims)},
        )
        
        # Retrieve from multiple sources for each claim
        all_docs: List[RetrievedDocument] = []
        verifications: List[Verification] = []
        contradictions: List[Contradiction] = []
        
        for claim in claims:
            # Retrieve for this claim
            claim_docs = await self._retrieve(claim.text, self.verification_config.min_sources_for_verification + 2)
            all_docs.extend(claim_docs)
            
            if len(claim_docs) < self.verification_config.min_sources_for_verification:
                verifications.append(Verification(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    notes="Insufficient sources for verification",
                ))
                continue
            
            # Verify claim against sources
            sources_text = "\n\n".join([
                f"[Source {i+1}]\n{d.content}"
                for i, d in enumerate(claim_docs)
            ])
            
            verify_prompt = get_template("verification_fact_check", language).format(
                claims=f"- {claim.text}",
                sources=sources_text,
            )
            
            verify_response = await self._call_llm(verify_prompt, temperature=self.verification_config.fact_check_temperature)
            verify_data = self._parse_json_response(verify_response)
            
            # Process verification result
            for ver in verify_data.get("verifications", []):
                status_str = ver.get("status", "unverified")
                status = VerificationStatus(status_str) if status_str in [s.value for s in VerificationStatus] else VerificationStatus.UNVERIFIED
                
                verifications.append(Verification(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status=status,
                    supporting_sources=[claim_docs[i].doc_id for i in ver.get("supporting_sources", []) if i < len(claim_docs)],
                    contradicting_sources=[claim_docs[i].doc_id for i in ver.get("contradicting_sources", []) if i < len(claim_docs)],
                    confidence=float(ver.get("confidence", 0.5)),
                    notes=ver.get("notes", ""),
                ))
            
            # Check for contradictions
            if self.verification_config.contradiction_detection:
                for contr in verify_data.get("contradictions_found", []):
                    contradictions.append(Contradiction(
                        claim=contr.get("claim", claim.text),
                        source_a_id=claim_docs[0].doc_id if claim_docs else "",
                        source_a_says=contr.get("source_a", ""),
                        source_b_id=claim_docs[1].doc_id if len(claim_docs) > 1 else "",
                        source_b_says=contr.get("source_b", ""),
                        severity=contr.get("severity", "medium"),
                    ))
        
        trace.add_entry(
            entry_type="verification",
            content={
                "verified": sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED),
                "partially": sum(1 for v in verifications if v.status == VerificationStatus.PARTIALLY_VERIFIED),
                "unverified": sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIED),
                "contradicted": sum(1 for v in verifications if v.status == VerificationStatus.CONTRADICTED),
                "contradictions": len(contradictions),
            },
        )
        
        # Generate summary answer
        verification_summary = "\n".join([
            f"- {v.claim_text}: {v.status.value} (confidence: {v.confidence:.2f})"
            for v in verifications
        ])
        
        final_answer = f"""Verification Results:

{verification_summary}

{"Contradictions detected: " + str(len(contradictions)) if contradictions else "No contradictions detected."}"""
        
        # Calculate overall confidence
        if verifications:
            verified_count = sum(1 for v in verifications if v.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED))
            final_confidence = verified_count / len(verifications)
        else:
            final_confidence = 0.0
        
        trace.completed_at = datetime.utcnow()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish event
        if self._event_publisher:
            await self._event_publisher.publish(
                "reasoning.verification.completed",
                {
                    "session_id": session_id,
                    "claims_verified": len(verifications),
                    "contradictions": len(contradictions),
                    "confidence": final_confidence,
                    "time_ms": elapsed_ms,
                },
            )
        
        return ReasoningResult(
            session_id=session_id,
            query=query,
            strategy_used=ReasoningStrategy.VERIFICATION,
            answer=final_answer,
            confidence=final_confidence,
            reasoning_trace=trace,
            claims=claims,
            verifications=verifications,
            contradictions=contradictions,
            retrieved_docs=all_docs,
            time_ms=elapsed_ms,
        )
    
    # ========================================================================
    # Direct Strategy
    # ========================================================================
    
    async def execute_direct(
        self,
        query: str,
        session_id: str,
        language: str = "en",
    ) -> ReasoningResult:
        """
        Execute Direct strategy - simple retrieval and answer.
        """
        start_time = time.perf_counter()
        
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.DIRECT,
        )
        
        # Retrieve
        docs = await self._retrieve(query, self.retrieval_config.default_top_k)
        
        trace.add_entry(
            entry_type="retrieval",
            content={"doc_count": len(docs)},
        )
        
        # Generate answer
        context = "\n\n".join([d.content for d in docs[:5]]) if docs else "No relevant context found."
        
        answer_prompt = get_template("direct_answer", language).format(
            query=query,
            context=context,
        )
        
        answer_response = await self._call_llm(answer_prompt, temperature=0.3)
        answer_data = self._parse_json_response(answer_response)
        
        final_answer = answer_data.get("answer", answer_response)
        final_confidence = float(answer_data.get("confidence", 0.5 if docs else 0.2))
        
        trace.completed_at = datetime.utcnow()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return ReasoningResult(
            session_id=session_id,
            query=query,
            strategy_used=ReasoningStrategy.DIRECT,
            answer=final_answer,
            confidence=final_confidence,
            reasoning_trace=trace,
            retrieved_docs=docs,
            time_ms=elapsed_ms,
        )


# Import datetime for trace timestamps
from datetime import datetime
