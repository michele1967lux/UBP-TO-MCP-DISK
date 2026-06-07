"""
v5.1.2: Evidence Abstraction (M1) and Reasoning Pass (M2)

M1 - EvidenceAbstractor:
    Extracts structured evidence from research documents using the worker LLM (vLLM).
    Produces per-section EvidenceMatrix with entries for each source.
    Uses pipe-delimited format to avoid Qwen3-4B-AWQ JSON echo issues.

M2 - ReasoningPass:
    Synthesizes evidence matrices into diagnostic patterns, emerging approaches,
    and evidence gaps using the planner LLM (Grok).
    Single-call reasoning, no recommendations.

Author: UBP Team
Version: 5.1.2
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class EvidenceEntry:
    """Single evidence item extracted from a source document."""

    source_index: int           # [1], [2]...
    condition: str              # Condition/topic addressed
    intervention: str           # Study/approach analyzed
    outcomes: str               # Key results
    evidence_strength: str      # strong/moderate/weak/anecdotal
    study_type: str             # RCT/cohort/review/expert_opinion/case_report
    limitations: str            # Known limitations
    source_ref: str             # Original source reference


@dataclass
class EvidenceMatrix:
    """Evidence matrix for a single report section."""

    section_title: str
    entries: List[EvidenceEntry]
    extraction_time_ms: float
    total_sources_processed: int
    status: str                 # success/partial/error
    error_message: Optional[str] = None


@dataclass
class ReasoningOutput:
    """Output of the reasoning pass across all evidence."""

    diagnostic_patterns: List[str]
    emerging_approaches: List[str]
    evidence_gaps: List[str]
    confidence: str             # always "exploratory"
    reasoning_time_ms: float
    status: str                 # success/error
    error_message: Optional[str] = None


# =============================================================================
# M1: EVIDENCE ABSTRACTOR
# =============================================================================

# Pipe-delimited prompt format — avoids Qwen3-4B-AWQ JSON echo
_EVIDENCE_EXTRACTION_PROMPT = """\
Sei un estrattore di evidenze. Analizza i documenti e per ogni fonte estrai le informazioni chiave.

Sezione: {section_title}

Documenti:
{documents_text}

Per ogni fonte, rispondi con UNA riga nel formato:
INDEX | CONDIZIONE | INTERVENTO | RISULTATI | FORZA | TIPO_STUDIO | LIMITAZIONI | RIFERIMENTO

Dove:
- INDEX: numero fonte [1], [2]...
- CONDIZIONE: argomento trattato
- INTERVENTO: studio/approccio analizzato
- RISULTATI: risultati chiave (max 30 parole)
- FORZA: strong/moderate/weak/anecdotal
- TIPO_STUDIO: RCT/cohort/review/expert_opinion/case_report
- LIMITAZIONI: limitazioni note (max 15 parole)
- RIFERIMENTO: riferimento fonte originale

Se un documento non contiene evidenze estraibili, omettilo.
Rispondi SOLO con le righe formattate, nessun testo aggiuntivo."""


class EvidenceAbstractor:
    """
    M1: Extracts structured evidence from research documents.

    Uses the worker LLM (vLLM local) for parallel, low-latency extraction.
    """

    def __init__(self, llm_module):
        """
        Args:
            llm_module: Worker LLM module with generate() method (vLLM)
        """
        self._llm = llm_module

    async def extract_evidence(
        self,
        section_title: str,
        documents: List[Dict[str, Any]],
    ) -> EvidenceMatrix:
        """
        Extract evidence entries from documents for a single section.

        Args:
            section_title: Title of the report section
            documents: List of research documents with 'text' field

        Returns:
            EvidenceMatrix with parsed entries
        """
        start_time = time.time()

        if not documents:
            return EvidenceMatrix(
                section_title=section_title,
                entries=[],
                extraction_time_ms=0.0,
                total_sources_processed=0,
                status="success",
            )

        # Build documents text for prompt (cap at 10 docs)
        doc_parts = []
        for i, doc in enumerate(documents[:10], 1):
            text = doc.get("text", "")[:800]  # Cap per-doc length
            source = doc.get("source", "unknown")
            doc_parts.append(f"[{i}] ({source}): {text}")

        documents_text = "\n\n".join(doc_parts)

        prompt = _EVIDENCE_EXTRACTION_PROMPT.format(
            section_title=section_title,
            documents_text=documents_text,
        )

        try:
            result = await self._llm.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1200,
            )

            if isinstance(result, dict):
                text = result.get("response", result.get("text", ""))
            else:
                text = str(result)

            entries = self._parse_entries(text, len(documents))
            extraction_time_ms = (time.time() - start_time) * 1000

            return EvidenceMatrix(
                section_title=section_title,
                entries=entries,
                extraction_time_ms=extraction_time_ms,
                total_sources_processed=min(len(documents), 10),
                status="success" if entries else "partial",
            )

        except Exception as e:
            logger.warning(f"[M1] Evidence extraction failed for '{section_title}': {e}")
            return EvidenceMatrix(
                section_title=section_title,
                entries=[],
                extraction_time_ms=(time.time() - start_time) * 1000,
                total_sources_processed=0,
                status="error",
                error_message=str(e),
            )

    async def extract_all_sections(
        self,
        research_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, EvidenceMatrix]:
        """
        Extract evidence for all sections in parallel.

        Args:
            research_results: Dict mapping section_title -> research result

        Returns:
            Dict mapping section_title -> EvidenceMatrix
        """
        semaphore = asyncio.Semaphore(4)  # Limit parallel LLM calls

        async def _extract_with_semaphore(title: str, docs: List[Dict]) -> tuple:
            async with semaphore:
                matrix = await self.extract_evidence(title, docs)
                return title, matrix

        tasks = []
        for section_title, research in research_results.items():
            docs = research.get("documents", [])
            tasks.append(_extract_with_semaphore(section_title, docs))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        matrices = {}
        for item in results:
            if isinstance(item, Exception):
                logger.error(f"[M1] Parallel extraction error: {item}")
                continue
            title, matrix = item
            matrices[title] = matrix

        logger.info(
            f"[M1] Evidence extraction complete: {len(matrices)} sections, "
            f"{sum(len(m.entries) for m in matrices.values())} total entries"
        )
        return matrices

    @staticmethod
    def _parse_entries(text: str, num_documents: int) -> List[EvidenceEntry]:
        """Parse pipe-delimited LLM output into EvidenceEntry objects."""
        entries = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 7:
                continue

            try:
                # Parse index — handle formats like "1", "[1]", "1."
                idx_str = parts[0].strip("[] .")
                source_index = int(idx_str) if idx_str.isdigit() else 0

                # Validate evidence strength
                strength = parts[4].lower().strip()
                if strength not in ("strong", "moderate", "weak", "anecdotal"):
                    strength = "weak"

                # Validate study type
                study_type = parts[5].strip().lower()
                valid_types = {"rct", "cohort", "review", "expert_opinion", "case_report"}
                if study_type not in valid_types:
                    study_type = "review"

                entries.append(EvidenceEntry(
                    source_index=source_index,
                    condition=parts[1][:200],
                    intervention=parts[2][:200],
                    outcomes=parts[3][:300],
                    evidence_strength=strength,
                    study_type=study_type,
                    limitations=parts[6][:200] if len(parts) > 6 else "",
                    source_ref=parts[7][:200] if len(parts) > 7 else "",
                ))
            except (ValueError, IndexError) as e:
                logger.debug(f"[M1] Skipping unparseable line: {line[:80]}... ({e})")
                continue

        return entries


# =============================================================================
# M2: REASONING PASS
# =============================================================================

_REASONING_PROMPT = """\
Sei un Clinical Research Analyst. Analizza la matrice di evidenze e identifica pattern diagnostici, approcci emergenti e lacune nelle evidenze.

Argomento del report: {plan_subject}

Matrice evidenze per sezione:
{evidence_summary}

Rispondi in formato pipe-delimited con 3 blocchi:

DIAGNOSTIC_PATTERNS:
- Un pattern per riga (max 8 righe)

EMERGING_APPROACHES:
- Un approccio per riga (max 6 righe)

EVIDENCE_GAPS:
- Una lacuna per riga (max 6 righe)

Regole:
- Basa l'analisi SOLO sulle evidenze fornite
- NON fare raccomandazioni cliniche
- Usa linguaggio esplorativo: "i dati suggeriscono", "emerge un pattern"
- Identifica contraddizioni tra sezioni se presenti"""


class ReasoningPass:
    """
    M2: Synthesizes evidence matrices into analytical reasoning.

    Uses the planner LLM (Grok cloud) for high-quality single-call reasoning.
    """

    def __init__(self, llm_module):
        """
        Args:
            llm_module: Planner LLM module with generate() method (Grok)
        """
        self._llm = llm_module

    async def reason(
        self,
        evidence_matrices: Dict[str, EvidenceMatrix],
        plan_subject: str,
    ) -> ReasoningOutput:
        """
        Perform reasoning across all evidence matrices.

        Args:
            evidence_matrices: Dict mapping section_title -> EvidenceMatrix
            plan_subject: Overall report subject for context

        Returns:
            ReasoningOutput with patterns, approaches, and gaps
        """
        start_time = time.time()

        # Build evidence summary for prompt
        summary_parts = []
        for title, matrix in evidence_matrices.items():
            if not matrix.entries:
                summary_parts.append(f"## {title}\n[Nessuna evidenza estratta]")
                continue

            section_lines = [f"## {title}"]
            for entry in matrix.entries:
                section_lines.append(
                    f"- [{entry.source_index}] {entry.condition}: "
                    f"{entry.outcomes} (forza: {entry.evidence_strength}, "
                    f"tipo: {entry.study_type})"
                )
            summary_parts.append("\n".join(section_lines))

        evidence_summary = "\n\n".join(summary_parts)

        prompt = _REASONING_PROMPT.format(
            plan_subject=plan_subject,
            evidence_summary=evidence_summary,
        )

        try:
            result = await self._llm.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=1500,
            )

            if isinstance(result, dict):
                text = result.get("response", result.get("text", ""))
            else:
                text = str(result)

            parsed = self._parse_reasoning(text)
            reasoning_time_ms = (time.time() - start_time) * 1000

            return ReasoningOutput(
                diagnostic_patterns=parsed.get("patterns", []),
                emerging_approaches=parsed.get("approaches", []),
                evidence_gaps=parsed.get("gaps", []),
                confidence="exploratory",
                reasoning_time_ms=reasoning_time_ms,
                status="success",
            )

        except Exception as e:
            logger.warning(f"[M2] Reasoning pass failed: {e}")
            return ReasoningOutput(
                diagnostic_patterns=[],
                emerging_approaches=[],
                evidence_gaps=[],
                confidence="exploratory",
                reasoning_time_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=str(e),
            )

    @staticmethod
    def _parse_reasoning(text: str) -> Dict[str, List[str]]:
        """Parse reasoning output into structured lists."""
        result = {"patterns": [], "approaches": [], "gaps": []}
        current_section = None

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            if "DIAGNOSTIC_PATTERN" in upper:
                current_section = "patterns"
                continue
            elif "EMERGING_APPROACH" in upper:
                current_section = "approaches"
                continue
            elif "EVIDENCE_GAP" in upper:
                current_section = "gaps"
                continue

            if current_section and line.startswith("- "):
                item = line[2:].strip()
                if item:
                    result[current_section].append(item)

        return result
