"""
Text Chunking - Enterprise Grade

Production-ready text chunking with:
- Multiple splitting strategies (sentence, paragraph, semantic, recursive)
- Overlap management
- Metadata preservation
- Token-aware splitting
- Language detection
"""

from __future__ import annotations

import logging
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import unicodedata

logger = logging.getLogger(__name__)


# ============================================================================
# Chunk Data Structure
# ============================================================================


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # v6.4.1: Validate position metadata consistency
        if self.end_char < self.start_char:
            self.end_char = self.start_char + len(self.text)
        if self.end_char - self.start_char != len(self.text):
            self.end_char = self.start_char + len(self.text)

    @property
    def id(self) -> str:
        """Generate deterministic chunk ID."""
        doc_id = self.metadata.get("doc_id", "")
        content = f"{doc_id}:{self.text}:{self.index}:{self.start_char}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def length(self) -> int:
        """Text length in characters."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "length": self.length,
            "word_count": self.word_count,
            "metadata": self.metadata,
        }


# ============================================================================
# Chunking Strategies
# ============================================================================


class SplitStrategy(Enum):
    """Text splitting strategies."""

    FIXED = "fixed"  # Fixed character count
    SENTENCE = "sentence"  # By sentence boundaries
    PARAGRAPH = "paragraph"  # By paragraph boundaries
    RECURSIVE = "recursive"  # Recursive splitting with multiple separators
    SEMANTIC = "semantic"  # Semantic boundaries (requires model)
    TOKEN = "token"  # By token count


@dataclass
class ChunkingConfig:
    """Chunking configuration.
    
    FIX-CHUNK-DOC-003 (v1.8.5): Added validation for chunk_size bounds
    to prevent OOM/DoS from extreme values.
    
    Valid ranges:
    - chunk_size: 50-4000 characters
    - chunk_overlap: 0 to chunk_size/2
    - min_chunk_size: 10-500 characters
    """

    strategy: str = "sentence"
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    preserve_sentences: bool = True
    preserve_paragraphs: bool = False
    include_metadata: bool = True
    strip_whitespace: bool = True
    normalize_whitespace: bool = True
    sentence_endings: str = ".!?"
    paragraph_separator: str = "\n\n"
    recursive_separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )
    
    def __post_init__(self):
        """Validate configuration values after initialization.
        
        FIX-CHUNK-DOC-003: Enforce bounds to prevent resource exhaustion.
        """
        # Chunk size validation (50-4000 chars)
        MIN_CHUNK = 50
        MAX_CHUNK = 4000
        
        if self.chunk_size < MIN_CHUNK:
            logger.warning(
                f"[FIX-CHUNK-DOC-003] chunk_size={self.chunk_size} below minimum, "
                f"adjusting to {MIN_CHUNK}"
            )
            self.chunk_size = MIN_CHUNK
        elif self.chunk_size > MAX_CHUNK:
            logger.warning(
                f"[FIX-CHUNK-DOC-003] chunk_size={self.chunk_size} above maximum, "
                f"capping at {MAX_CHUNK}"
            )
            self.chunk_size = MAX_CHUNK
        
        # Chunk overlap validation (0 to chunk_size/2)
        max_overlap = self.chunk_size // 2
        if self.chunk_overlap < 0:
            logger.warning(
                f"[FIX-CHUNK-DOC-003] chunk_overlap={self.chunk_overlap} negative, "
                f"adjusting to 0"
            )
            self.chunk_overlap = 0
        elif self.chunk_overlap > max_overlap:
            logger.warning(
                f"[FIX-CHUNK-DOC-003] chunk_overlap={self.chunk_overlap} exceeds 50% of chunk_size, "
                f"capping at {max_overlap}"
            )
            self.chunk_overlap = max_overlap
        
        # Min chunk size validation
        if self.min_chunk_size < 10:
            self.min_chunk_size = 10
        elif self.min_chunk_size > 500:
            self.min_chunk_size = 500
        
        # Max chunk size should be >= chunk_size
        if self.max_chunk_size < self.chunk_size:
            _old = self.max_chunk_size
            _new = self.chunk_size * 2
            self.max_chunk_size = _new
            import logging
            logging.getLogger(__name__).warning(
                f"[CHUNKER] max_chunk_size auto-corrected: {_old} -> {_new} "
                f"(was less than chunk_size={self.chunk_size})"
            )


# ============================================================================
# Text Preprocessor
# ============================================================================


class TextPreprocessor:
    """Text preprocessing utilities."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace with single space
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        # Normalize multiple spaces to single
        text = re.sub(r" +", " ", text)
        # Normalize multiple newlines to double
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove control characters and invisible Unicode characters.

        v6.4.2: Extended to handle Cf (Format) and Zs (NBSP) categories.
        Fixes S5-01: BOM, ZWSP, ZWNJ, ZWJ, NBSP now cleaned during chunking.
        """
        # NBSP -> regular space (conversion, not removal)
        text = text.replace('\u00a0', ' ')

        # Known invisible Cf (Format) characters to strip
        _INVISIBLE_CF = {
            '\ufeff',   # BOM (Byte Order Mark)
            '\u200b',   # ZWSP (Zero Width Space)
            '\u200c',   # ZWNJ (Zero Width Non-Joiner)
            '\u200d',   # ZWJ (Zero Width Joiner)
            '\u200e',   # LRM (Left-to-Right Mark)
            '\u200f',   # RLM (Right-to-Left Mark)
            '\u2060',   # Word Joiner
            '\ufffe',   # Noncharacter
        }

        return "".join(
            char
            for char in text
            if (unicodedata.category(char) != "Cc" or char in "\n\t")
            and char not in _INVISIBLE_CF
        )

    @staticmethod
    def clean_text(
        text: str,
        normalize_ws: bool = True,
        normalize_uni: bool = True,
        remove_ctrl: bool = True,
    ) -> str:
        """Apply all text cleaning operations."""
        # v6.4.1: Correct order — NFKC first (may produce ctrl chars), then remove ctrl
        if normalize_uni:
            text = TextPreprocessor.normalize_unicode(text)
        if remove_ctrl:
            text = TextPreprocessor.remove_control_chars(text)
        if normalize_ws:
            text = TextPreprocessor.normalize_whitespace(text)
        return text


# ============================================================================
# Sentence Splitter
# ============================================================================


class SentenceSplitter:
    """
    Intelligent sentence splitting.

    Handles:
    - Standard sentence endings
    - Abbreviations
    - Quotes and parentheses
    - Lists and bullets
    """

    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "al",
        "fig",
        "vol",
        "no",
        "pp",
        "inc",
        "corp",
        "ltd",
        "co",
        "dept",
        "div",
        "est",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
        "st",
        "nd",
        "rd",
        "th",
        "ave",
        "blvd",
        # Italian abbreviations
        "dott", "sig", "avv", "ing", "arch", "geom", "rag",
        "ecc", "pag", "cap", "sez", "art", "tel", "gen",
        "amm", "dir", "on", "sen", "comm",
    }

    def __init__(self, endings: str = ".!?"):
        self.endings = endings
        self._sentence_pattern = self._build_pattern()

    def _build_pattern(self) -> re.Pattern:
        """Build sentence splitting pattern."""
        # Match sentence endings followed by space and capital letter
        endings_escaped = re.escape(self.endings)
        pattern = rf"([{endings_escaped}])(\s+)(?=[A-ZÀ-ÖØ-Þ0-9\[\(\"\'])"
        return re.compile(pattern)

    def _is_abbreviation(self, text: str) -> bool:
        """Check if text ends with an abbreviation."""
        words = text.lower().split()
        if not words:
            return False

        last_word = words[-1].rstrip(".").lower()
        if last_word in self.ABBREVIATIONS:
            return True
        # Strip Italian article elisions (l'Art., dell'Ing., etc.)
        apo = last_word.find("'")
        if apo >= 0:
            last_word = last_word[apo + 1:]
            if last_word in self.ABBREVIATIONS:
                return True
        # v6.4.0: Multi-part abbreviations (U.S.A., Ph.D.)
        if re.match(r'^([a-z]\.){2,}$', words[-1].lower()):
            return True
        return False

    def split(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []

        # Preprocess
        text = text.strip()

        # Simple split first
        parts = self._sentence_pattern.split(text)

        sentences = []
        current = ""

        i = 0
        while i < len(parts):
            current += parts[i]

            # Check if this is a sentence ending
            if i + 1 < len(parts) and parts[i + 1] in self.endings:
                current += parts[i + 1]

                # Check for abbreviation
                if self._is_abbreviation(current):
                    # Continue to next part
                    if i + 2 < len(parts):
                        current += parts[i + 2]
                    i += 3
                    continue

                # Add whitespace if present
                if i + 2 < len(parts):
                    # This is a sentence break
                    sentences.append(current.strip())
                    current = ""
                    i += 3
                    continue
                else:
                    i += 2
                    continue

            i += 1

        # Add remaining text
        if current.strip():
            sentences.append(current.strip())

        # Fallback: if no sentences found, try simpler split
        if not sentences:
            raw_parts = re.split(r"(?<=[.!?])\s+", text)
            sentences = []
            for part in raw_parts:
                if sentences and self._is_abbreviation(sentences[-1]):
                    sentences[-1] = sentences[-1] + " " + part
                else:
                    sentences.append(part)

        return [s for s in sentences if s.strip()]


# ============================================================================
# Base Chunker
# ============================================================================


class Chunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self._sentence_splitter = SentenceSplitter(config.sentence_endings)

    @abstractmethod
    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into chunks."""
        pass

    def _preprocess(self, text: str) -> str:
        """Preprocess text before chunking."""
        if self.config.strip_whitespace:
            text = text.strip()
        # v6.4.0: Full preprocessing (unicode + ctrl chars + whitespace)
        if self.config.normalize_whitespace:
            text = self.preprocessor.clean_text(text)
        return text

    def _create_chunk(
        self,
        text: str,
        index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Create a chunk with metadata."""
        chunk_metadata = {}

        if self.config.include_metadata and metadata:
            chunk_metadata.update(metadata)

        chunk_metadata.update(
            {"chunk_index": index, "chunk_strategy": self.config.strategy}
        )

        return Chunk(
            text=text,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata,
        )

    # Semantic separators hierarchy (most to least semantic)
    # Used by recursive split to preserve meaning
    RECURSIVE_SEPARATORS = [
        "\n\n",  # Paragraph break (strongest semantic boundary)
        "\n",  # Line break
        ". ",  # Sentence end
        "? ",  # Question end
        "! ",  # Exclamation end
        "; ",  # Semicolon (clause boundary - common in legal text)
        ", ",  # Comma (phrase boundary)
        " ",  # Word boundary
        "",  # Character level (last resort)
    ]

    def _split_oversized_text(self, text: str) -> List[str]:
        """Split text that exceeds max_chunk_size using recursive hierarchical splitting.

        UPGRADE v1.8.2: Recursive semantic split for better readability.
        Preserves meaning by splitting at semantic boundaries in priority order:
        paragraphs > sentences > clauses (;) > phrases (,) > words > characters

        This ensures legal/technical text is split at commas or semicolons
        rather than arbitrary positions, maintaining sense in both halves.

        Args:
            text: Text to split

        Returns:
            List of text pieces, each <= max_chunk_size
        """
        max_size = self.config.max_chunk_size

        # If text is within limits, return as-is
        if len(text) <= max_size:
            return [text]

        logger.info(
            f"[RECURSIVE-SPLIT] Text exceeds max_chunk_size ({len(text)} > {max_size}), "
            f"applying recursive semantic split"
        )

        return self._recursive_split(text, self.RECURSIVE_SEPARATORS, max_size)

    def _recursive_split(
        self, text: str, separators: List[str], max_size: int, depth: int = 0
    ) -> List[str]:
        """Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: Remaining separators to try (most to least semantic)
            max_size: Maximum chunk size
            depth: Recursion depth (for logging)

        Returns:
            List of text pieces, each <= max_size
        """
        # Base case: text fits
        if len(text) <= max_size:
            return [text.strip()] if text.strip() else []

        # v6.4.1: Depth limit to prevent excessive recursion on pathological text
        MAX_DEPTH = 20
        if depth >= MAX_DEPTH:
            logger.warning(f"[CHUNKER] _recursive_split hit depth limit ({MAX_DEPTH}), force splitting")
            return self._force_char_split(text, max_size)

        # Base case: no more separators, force character split
        if not separators:
            logger.warning(
                f"[RECURSIVE-SPLIT] No separators left, forcing character split "
                f"at depth {depth} for {len(text)} chars"
            )
            return self._force_char_split(text, max_size)

        # Get current separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Empty separator means character-level split
        if separator == "":
            return self._force_char_split(text, max_size)

        # Try to split on this separator
        if separator in text:
            parts = text.split(separator)

            # Reassemble parts into chunks that fit max_size
            result = []
            current_chunk = ""

            for i, part in enumerate(parts):
                # Add separator back (except for last part)
                part_with_sep = part + separator if i < len(parts) - 1 else part

                # Check if adding this part would exceed max_size
                test_chunk = current_chunk + part_with_sep

                if len(test_chunk) <= max_size:
                    # Fits, accumulate
                    current_chunk = test_chunk
                else:
                    # Doesn't fit
                    if current_chunk:
                        # Save current chunk
                        result.append(current_chunk.strip())
                        current_chunk = ""

                    # Check if this single part fits
                    if len(part_with_sep) <= max_size:
                        current_chunk = part_with_sep
                    else:
                        # Part itself is too large, recurse with next separator
                        sub_parts = self._recursive_split(
                            part_with_sep, remaining_separators, max_size, depth + 1
                        )
                        result.extend(sub_parts)

            # Don't forget remaining chunk
            if current_chunk.strip():
                result.append(current_chunk.strip())

            # Filter empty strings and return
            return [r for r in result if r]

        # Separator not found in text, try next one
        return self._recursive_split(text, remaining_separators, max_size, depth + 1)

    def _force_char_split(self, text: str, max_size: int) -> List[str]:
        """Last resort: split at character boundaries.

        Tries to split at word boundaries within max_size, but will
        hard-cut if no space is found (e.g., very long URLs or codes).

        Args:
            text: Text to split
            max_size: Maximum chunk size

        Returns:
            List of text pieces, each <= max_size
        """
        pieces = []
        remaining = text

        while remaining:
            if len(remaining) <= max_size:
                if remaining.strip():
                    pieces.append(remaining.strip())
                break

            # Try to find last space within limit
            split_at = max_size
            last_space = remaining.rfind(" ", 0, max_size)

            # v6.4.1: Less conservative — cut at word boundary only if reasonably close
            if last_space > max_size * 3 // 4:
                # Found a reasonable space, use it
                split_at = last_space
            else:
                # No good space, hard cut at max_size
                logger.debug(
                    f"[FORCE-CHAR-SPLIT] Hard cut at {max_size} chars (no space found)"
                )

            piece = remaining[:split_at].strip()
            if piece:
                pieces.append(piece)
            remaining = remaining[split_at:].strip()

        return pieces

    def _merge_small_chunks(
        self, chunks: List[Chunk], metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Merge chunks smaller than min_chunk_size.

        FIX GAP-CHUNKING-001: Use chunk_size as merge limit, not max_chunk_size.
        This ensures that merged chunks respect the configured chunk_size target.

        FIX GAP-METADATA-001: Accept and propagate metadata to merged chunks.
        Without this, filename and other critical metadata is lost during merge.
        """
        if not chunks:
            return chunks

        merged = []
        current_text = ""
        current_start = 0

        # Use chunk_size as the merge target limit, not max_chunk_size
        merge_limit = self.config.chunk_size

        for chunk in chunks:
            if not current_text:
                current_text = chunk.text
                current_start = chunk.start_char
            elif len(current_text) + len(chunk.text) + 1 < merge_limit:
                # Only merge if result stays under chunk_size
                current_text = current_text + " " + chunk.text
            else:
                if len(current_text) >= self.config.min_chunk_size:
                    merged.append(
                        self._create_chunk(
                            current_text,
                            len(merged),
                            current_start,
                            current_start + len(current_text),
                            metadata,
                        )
                    )
                elif merged:
                    # v6.3.3: Append to previous chunk instead of discarding
                    _prev = merged[-1]
                    _new_text = _prev.text + "\n" + current_text
                    merged[-1] = self._create_chunk(
                        _new_text, _prev.index, _prev.start_char,
                        _prev.start_char + len(_new_text), metadata,
                    )
                    logger.debug(f"[CHUNKER] Appended {len(current_text)} chars to previous chunk (below min_chunk_size)")
                else:
                    # First chunk and undersized — keep it anyway
                    merged.append(
                        self._create_chunk(
                            current_text, len(merged), current_start,
                            current_start + len(current_text), metadata,
                        )
                    )
                    logger.debug(f"[CHUNKER] Kept undersized first chunk ({len(current_text)} chars)")
                current_text = chunk.text
                current_start = chunk.start_char

        # Add remaining (tail)
        if current_text:
            if len(current_text) >= self.config.min_chunk_size:
                merged.append(
                    self._create_chunk(
                        current_text,
                        len(merged),
                        current_start,
                        current_start + len(current_text),
                        metadata,
                    )
                )
            elif merged:
                # v6.3.3: Append tail to last chunk instead of discarding
                _prev = merged[-1]
                _new_text = _prev.text + "\n" + current_text
                merged[-1] = self._create_chunk(
                    _new_text, _prev.index, _prev.start_char,
                    _prev.start_char + len(_new_text), metadata,
                )
                logger.debug(f"[CHUNKER] Appended tail ({len(current_text)} chars) to last chunk")
            else:
                # Only fragment in document — keep it
                merged.append(
                    self._create_chunk(
                        current_text, len(merged), current_start,
                        current_start + len(current_text), metadata,
                    )
                )
                logger.debug(f"[CHUNKER] Kept undersized single chunk ({len(current_text)} chars)")

        return merged


# ============================================================================
# Fixed Size Chunker
# ============================================================================


class FixedChunker(Chunker):
    """Split text into fixed-size chunks with overlap."""

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        text = self._preprocess(text)

        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Adjust end to not cut words if possible
            if end < len(text) and self.config.preserve_sentences:
                # Find last space before end
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(chunk_text, chunk_index, start, end, metadata)
                )
                chunk_index += 1

            # Calculate next start with overlap
            start = end - self.config.chunk_overlap
            _last_start = chunks[-1].start_char if chunks else -1
            if start <= _last_start:
                start = end

        # v6.3.3: Safety — if chunking produced 0 results, return full text as single chunk
        if not chunks:
            chunks = [self._create_chunk(text, 0, 0, len(text), metadata)]
            logger.warning(
                f"[CHUNKER] FixedChunker produced 0 chunks for text of {len(text)} chars — "
                f"returning full text as single chunk"
            )

        return chunks


# ============================================================================
# Sentence Chunker
# ============================================================================


class SentenceChunker(Chunker):
    """Split text into chunks based on sentence boundaries."""

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into sentence-based chunks."""
        text = self._preprocess(text)

        if not text:
            return []

        # Split into sentences
        sentences = self._sentence_splitter.split(text)

        if not sentences:
            return [self._create_chunk(text, 0, 0, len(text), metadata)]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        # Track position in original text
        pos = 0

        for sentence in sentences:
            # Find sentence position in original text
            # v6.4.1: Track position cumulatively to avoid wrong matches on duplicates
            sentence_start = text.find(sentence, pos)
            if sentence_start == -1:
                sentence_start = pos
                logger.debug(f"[CHUNKER] Sentence not found from pos={pos}, using fallback")
            else:
                pos = sentence_start + len(sentence)

            # FIX-CHUNK-LIMIT-001: Handle oversized sentences
            # If a single sentence exceeds max_chunk_size, force-split it
            if len(sentence) > self.config.max_chunk_size:
                # First, save any current chunk
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            chunk_index,
                            current_start,
                            current_start + len(current_chunk),
                            metadata,
                        )
                    )
                    chunk_index += 1
                    current_chunk = ""

                # Split the oversized sentence and add as separate chunks
                sentence_pieces = self._split_oversized_text(sentence)
                piece_start = sentence_start
                for piece in sentence_pieces:
                    if len(piece) >= self.config.min_chunk_size:
                        chunks.append(
                            self._create_chunk(
                                piece,
                                chunk_index,
                                piece_start,
                                piece_start + len(piece),
                                metadata,
                            )
                        )
                        chunk_index += 1
                    piece_start += len(piece) + 1  # +1 for space

                current_start = piece_start
                pos = sentence_start + len(sentence)
                continue

            # Check if adding this sentence exceeds chunk size
            potential_chunk = (
                (current_chunk + " " + sentence).strip() if current_chunk else sentence
            )
            if len(potential_chunk) > self.config.chunk_size and current_chunk:
                # Save current chunk
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            chunk_index,
                            current_start,
                            current_start + len(current_chunk),
                            metadata,
                        )
                    )
                    chunk_index += 1

                # Start new chunk
                # Include overlap from previous sentences
                if self.config.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.config.chunk_overlap :]
                    # Find sentence boundary in overlap
                    last_sentence_end = max(
                        overlap_text.rfind("."),
                        overlap_text.rfind("!"),
                        overlap_text.rfind("?"),
                    )
                    if last_sentence_end > 0:
                        overlap_text = overlap_text[last_sentence_end + 1 :].strip()

                    current_chunk = (
                        (overlap_text + " " + sentence).strip()
                        if overlap_text
                        else sentence
                    )
                else:
                    current_chunk = sentence

                current_start = sentence_start
            else:
                if not current_chunk:
                    current_start = sentence_start
                current_chunk = potential_chunk

            pos = sentence_start + len(sentence)

        # Add final chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(
                self._create_chunk(
                    current_chunk,
                    chunk_index,
                    current_start,
                    current_start + len(current_chunk),
                    metadata,
                )
            )

        return self._merge_small_chunks(chunks, metadata)


# ============================================================================
# Paragraph Chunker
# ============================================================================


class ParagraphChunker(Chunker):
    """Split text into chunks based on paragraph boundaries."""

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Split text into paragraph-based chunks."""
        text = self._preprocess(text)

        if not text:
            return []

        # Split into paragraphs
        # v6.4.1: Support regex separators (detect by presence of regex metacharacters)
        import re as _re
        _sep = self.config.paragraph_separator
        if any(c in _sep for c in r'.*+?[](){}|\^$'):
            try:
                paragraphs = _re.split(_sep, text)
            except _re.error:
                paragraphs = text.split(_sep)  # Fallback to literal
        else:
            paragraphs = text.split(_sep)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [self._create_chunk(text, 0, 0, len(text), metadata)]

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        pos = 0

        for paragraph in paragraphs:
            # Find paragraph position
            para_start = text.find(paragraph, pos)
            if para_start == -1:
                para_start = pos

            # v6.4.0: Split oversized paragraphs
            if len(paragraph) > self.config.max_chunk_size:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk, chunk_index, current_start,
                            current_start + len(current_chunk), metadata,
                        )
                    )
                    chunk_index += 1
                    current_chunk = ""
                for oc in self._split_oversized_text(paragraph):
                    chunks.append(
                        self._create_chunk(
                            oc.text, chunk_index, para_start,
                            para_start + len(oc.text), metadata,
                        )
                    )
                    chunk_index += 1
                pos = para_start + len(paragraph)
                current_start = pos
                continue

            # Check if adding this paragraph exceeds chunk size
            potential_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )

            if len(potential_chunk) > self.config.chunk_size and current_chunk:
                # Save current chunk
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            chunk_index,
                            current_start,
                            current_start + len(current_chunk),
                            metadata,
                        )
                    )
                    chunk_index += 1

                current_chunk = paragraph
                current_start = para_start
            else:
                if not current_chunk:
                    current_start = para_start
                current_chunk = potential_chunk

            pos = para_start + len(paragraph)

        # Add final chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(
                self._create_chunk(
                    current_chunk,
                    chunk_index,
                    current_start,
                    current_start + len(current_chunk),
                    metadata,
                )
            )

        return self._merge_small_chunks(chunks, metadata)


# ============================================================================
# Recursive Chunker
# ============================================================================


class RecursiveChunker(Chunker):
    """
    Recursive text splitting with multiple separators.

    Tries separators in order until chunks are small enough.
    """

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """Recursively split text using multiple separators."""
        text = self._preprocess(text)

        if not text:
            return []

        raw_chunks = self._recursive_split(text, self.config.recursive_separators)

        # Convert to Chunk objects
        chunks = []
        pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos

            chunks.append(
                self._create_chunk(
                    chunk_text, i, start, start + len(chunk_text), metadata
                )
            )

            pos = start + len(chunk_text)

        return self._merge_small_chunks(chunks, metadata)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # No more separators, force split
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            # Empty separator means character-by-character
            parts = list(text)

        # Merge parts into chunks
        chunks = []
        current = ""

        for idx, part in enumerate(parts):
            # Add separator back (except for last part)
            is_last = (idx == len(parts) - 1)
            part_with_sep = part + separator if (separator and not is_last) else part

            if not current:
                current = part_with_sep
            elif len(current) + len(part_with_sep) <= self.config.chunk_size:
                current += part_with_sep
            else:
                # Current chunk is large enough, save it
                if current.strip():
                    if len(current) > self.config.chunk_size:
                        # Still too large, recurse
                        chunks.extend(
                            self._recursive_split(current, remaining_separators)
                        )
                    else:
                        chunks.append(current.rstrip(separator))
                current = part_with_sep

        # Handle remaining
        if current.strip():
            if len(current) > self.config.chunk_size:
                chunks.extend(self._recursive_split(current, remaining_separators))
            else:
                chunks.append(current.rstrip(separator))

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """Force split text when no separators work."""
        chunks = []
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)

        for i in range(0, len(text), step):
            chunk = text[i : i + self.config.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        # v6.4.0: Merge last chunk if too small
        if len(chunks) > 1 and len(chunks[-1]) < self.config.min_chunk_size:
            chunks[-2] = chunks[-2] + chunks[-1]
            chunks.pop()

        return chunks


# ============================================================================
# Chunking Manager
# ============================================================================


class ChunkingManager:
    """
    Chunking manager with strategy selection and monitoring.

    Features:
    - Multiple chunking strategies
    - Automatic strategy selection
    - Chunk statistics
    - Overlap management
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._chunkers: Dict[str, Chunker] = {}
        self._init_chunkers()

        # Statistics
        import threading
        self._stats_lock = threading.Lock()
        self._total_chunks = 0
        self._total_chars = 0
        self._total_documents = 0

    def _init_chunkers(self) -> None:
        """Initialize chunker instances."""
        self._chunkers = {
            "fixed": FixedChunker(self.config),
            "sentence": SentenceChunker(self.config),
            "paragraph": ParagraphChunker(self.config),
            "recursive": RecursiveChunker(self.config),
        }

    def chunk(
        self,
        text: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk text using specified strategy.

        Args:
            text: Input text
            strategy: Chunking strategy (defaults to config)
            metadata: Additional metadata for chunks

        Returns:
            List of chunks
        """
        if not text or not text.strip():
            return []

        strategy = strategy or self.config.strategy

        if strategy not in self._chunkers:
            logger.warning(f"Unknown strategy '{strategy}', using 'sentence'")
            strategy = "sentence"

        chunker = self._chunkers[strategy]
        chunks = chunker.chunk(text, metadata)

        # Update statistics
        with self._stats_lock:
            self._total_documents += 1
            self._total_chunks += len(chunks)
            self._total_chars += len(text)

        logger.debug(
            f"Chunked document into {len(chunks)} chunks",
            extra={
                "strategy": strategy,
                "input_chars": len(text),
                "chunks": len(chunks),
            },
        )

        return chunks

    def chunk_with_overlap_ids(
        self,
        text: str,
        doc_id: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk text with document ID and overlap tracking.

        Args:
            text: Input text
            doc_id: Document identifier
            strategy: Chunking strategy
            metadata: Additional metadata

        Returns:
            List of chunks with proper IDs
        """
        base_metadata = {**(metadata or {})}  # v6.4.1: Shallow copy — don't mutate caller's dict
        base_metadata["doc_id"] = doc_id

        chunks = self.chunk(text, strategy, base_metadata)

        # Add chunk relationships
        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["chunk_id"] = f"{doc_id}:chunk:{i}"

            if i > 0:
                chunk.metadata["prev_chunk_id"] = f"{doc_id}:chunk:{i - 1}"
            if i < len(chunks) - 1:
                chunk.metadata["next_chunk_id"] = f"{doc_id}:chunk:{i + 1}"

        return chunks

    @property
    def stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        avg_chunks = 0.0
        if self._total_documents > 0:
            avg_chunks = self._total_chunks / self._total_documents

        return {
            "total_documents": self._total_documents,
            "total_chunks": self._total_chunks,
            "total_chars": self._total_chars,
            "average_chunks_per_doc": round(avg_chunks, 2),
            "config": {
                "strategy": self.config.strategy,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "min_chunk_size": self.config.min_chunk_size,
                "max_chunk_size": self.config.max_chunk_size,
            },
        }


# ============================================================================
# Factory Function
# ============================================================================


def create_chunking_manager(config: Dict[str, Any]) -> ChunkingManager:
    """
    Create a ChunkingManager from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured ChunkingManager instance
    """
    chunking_config = config.get("chunking", {})

    return ChunkingManager(
        ChunkingConfig(
            # v6.4.1: Accept both "split_by" (legacy) and "strategy" (direct) config keys
            strategy=chunking_config.get("strategy", chunking_config.get("split_by", "sentence")),
            chunk_size=chunking_config.get("chunk_size", 500),
            chunk_overlap=chunking_config.get("chunk_overlap", 50),
            min_chunk_size=chunking_config.get("min_chunk_size", 50),
            max_chunk_size=chunking_config.get("max_chunk_size", 2000),
            preserve_sentences=chunking_config.get("preserve_sentences", True),
            preserve_paragraphs=chunking_config.get("preserve_paragraphs", False),
            include_metadata=chunking_config.get(
                "include_metadata", True
            ),  # FIX: Explicitly enable metadata in chunks
        )
    )
