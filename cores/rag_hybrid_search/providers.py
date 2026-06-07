"""
Hybrid Search Provider - Pure Technical Logic
Zero UBP dependencies. Can be tested standalone.
Implements BM25 sparse search and score fusion algorithms.

Features:
- BM25 (Okapi BM25) for sparse/keyword search
- Multiple fusion strategies (RRF, Weighted, Max)
- Score normalization for weighted fusion
- Per-collection index management
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import math
import re
import logging

logger = logging.getLogger(__name__)


class FusionMethod(str, Enum):
    """Score fusion methods."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted score combination
    MAX = "max"  # Take max score


@dataclass
class SearchResult:
    """Represents a search result."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # 'dense', 'sparse', or 'hybrid'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": round(self.score, 6),
            "metadata": self.metadata,
            "source": self.source,
        }


class BM25Index:
    """
    In-memory BM25 index for sparse retrieval.

    BM25 (Okapi BM25) formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
    - f(qi,D) = term frequency of qi in document D
    - |D| = document length (number of tokens)
    - avgdl = average document length across collection
    - k1 = term frequency saturation parameter (default 1.5)
    - b = document length normalization parameter (default 0.75)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical)
            epsilon: Floor for IDF to handle rare terms
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Index structures
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # term -> doc count
        self.doc_lens: Dict[str, int] = {}  # doc_id -> length
        self.term_freqs: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: freq}
        self.documents: Dict[str, str] = {}  # doc_id -> content
        self.doc_metadata: Dict[str, Dict] = {}  # doc_id -> metadata

        self.total_docs = 0
        self.avgdl = 0.0
        self._idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Performs:
        - Lowercase conversion
        - Alphanumeric token extraction
        - Handles hyphenated terms (e.g., 'SKU-12345')
        """
        text = text.lower()
        # Extract alphanumeric tokens, including hyphenated terms
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)
        return tokens

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document to the index.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dictionary
        """
        # If document already exists, remove it first
        if doc_id in self.documents:
            self.remove_document(doc_id)

        tokens = self.tokenize(content)

        if not tokens:
            logger.debug(f"Document {doc_id} has no tokens, skipping")
            return

        # Store document
        self.documents[doc_id] = content
        self.doc_metadata[doc_id] = metadata or {}
        self.doc_lens[doc_id] = len(tokens)

        # Count term frequencies
        term_freq: Dict[str, int] = defaultdict(int)
        seen_terms: set = set()

        for token in tokens:
            term_freq[token] += 1
            if token not in seen_terms:
                self.doc_freqs[token] += 1
                seen_terms.add(token)

        self.term_freqs[doc_id] = dict(term_freq)
        self.total_docs += 1

        # Update average document length
        total_len = sum(self.doc_lens.values())
        self.avgdl = total_len / self.total_docs if self.total_docs > 0 else 0

        # Invalidate IDF cache (terms may have new document frequencies)
        self._idf_cache.clear()

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document identifier to remove

        Returns:
            True if document was removed, False if not found
        """
        if doc_id not in self.documents:
            return False

        # Update doc freqs (decrement count for each term)
        for term in self.term_freqs.get(doc_id, {}).keys():
            self.doc_freqs[term] -= 1
            if self.doc_freqs[term] <= 0:
                del self.doc_freqs[term]

        # Remove from structures
        del self.documents[doc_id]
        del self.doc_lens[doc_id]
        del self.term_freqs[doc_id]
        if doc_id in self.doc_metadata:
            del self.doc_metadata[doc_id]

        self.total_docs -= 1

        # Update avgdl
        if self.total_docs > 0:
            total_len = sum(self.doc_lens.values())
            self.avgdl = total_len / self.total_docs
        else:
            self.avgdl = 0

        self._idf_cache.clear()
        return True

    def _idf(self, term: str) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.

        Uses smoothed IDF formula:
        IDF = log((N - n + 0.5) / (n + 0.5) + 1)

        Where:
        - N = total documents
        - n = documents containing term
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        doc_freq = self.doc_freqs.get(term, 0)

        if doc_freq == 0:
            idf = 0.0
        else:
            # Smoothed IDF with +1 to avoid negative values
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            idf = max(idf, self.epsilon)

        self._idf_cache[term] = idf
        return idf

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search the index using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by BM25 score descending
        """
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            idf = self._idf(token)

            for doc_id, term_freqs in self.term_freqs.items():
                if token not in term_freqs:
                    continue

                tf = term_freqs[token]
                doc_len = self.doc_lens[doc_id]

                # BM25 score component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score and return top_k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    content=self.documents[doc_id],
                    score=score,
                    metadata=self.doc_metadata.get(doc_id, {}),
                    source="sparse",
                )
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": self.total_docs,
            "unique_terms": len(self.doc_freqs),
            "average_doc_length": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }

    def clear(self) -> None:
        """Clear the entire index."""
        self.doc_freqs.clear()
        self.doc_lens.clear()
        self.term_freqs.clear()
        self.documents.clear()
        self.doc_metadata.clear()
        self._idf_cache.clear()
        self.total_docs = 0
        self.avgdl = 0.0


class HybridSearchProvider:
    """
    Hybrid search combining dense (vector) and sparse (BM25) retrieval.

    Features:
    - BM25 for sparse/keyword search
    - Integration with vector search (results passed in)
    - Multiple fusion strategies (RRF, Weighted, Max)
    - Score normalization for fair combination
    - Per-collection index management
    """

    def __init__(
        self,
        fusion_method: FusionMethod = FusionMethod.RRF,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize provider.

        Args:
            fusion_method: Method for combining dense and sparse scores
            dense_weight: Weight for dense search (0-1) in weighted fusion
            rrf_k: K parameter for RRF (typically 60)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

        # Collection-specific BM25 indexes
        self._indexes: Dict[str, BM25Index] = {}
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

    def get_or_create_index(self, collection_name: str) -> BM25Index:
        """Get or create a BM25 index for a collection."""
        if collection_name not in self._indexes:
            self._indexes[collection_name] = BM25Index(
                k1=self._bm25_k1,
                b=self._bm25_b,
            )
            logger.debug(f"Created BM25 index for collection: {collection_name}")
        return self._indexes[collection_name]

    def index_document(
        self,
        collection_name: str,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Index a document for sparse search.

        Args:
            collection_name: Target collection
            doc_id: Document identifier
            content: Document text content
            metadata: Optional metadata
        """
        index = self.get_or_create_index(collection_name)
        index.add_document(doc_id, content, metadata)
        logger.debug(f"Indexed document {doc_id} in collection {collection_name}")

    def remove_from_index(self, collection_name: str, doc_id: str) -> bool:
        """
        Remove a document from the sparse index.

        Args:
            collection_name: Target collection
            doc_id: Document identifier

        Returns:
            True if removed, False if not found
        """
        if collection_name not in self._indexes:
            return False
        return self._indexes[collection_name].remove_document(doc_id)

    def sparse_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform sparse (BM25) search.

        Args:
            collection_name: Target collection
            query: Search query
            top_k: Number of results

        Returns:
            List of SearchResult from BM25 search
        """
        if collection_name not in self._indexes:
            logger.debug(f"No BM25 index for collection: {collection_name}")
            return []

        return self._indexes[collection_name].search(query, top_k)

    def hybrid_search(
        self,
        query: str,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int = 10,
        fusion_method: Optional[FusionMethod] = None,
        dense_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Combine dense and sparse results using fusion.

        Args:
            query: Original query (for context/logging)
            dense_results: Results from vector search
            sparse_results: Results from BM25 search
            top_k: Number of final results
            fusion_method: Override default fusion method
            dense_weight: Override default dense weight

        Returns:
            Fused results sorted by combined score
        """
        method = fusion_method or self.fusion_method
        weight = dense_weight if dense_weight is not None else self.dense_weight

        if method == FusionMethod.RRF:
            return self._rrf_fusion(dense_results, sparse_results, top_k)
        elif method == FusionMethod.WEIGHTED:
            return self._weighted_fusion(dense_results, sparse_results, top_k, weight)
        elif method == FusionMethod.MAX:
            return self._max_fusion(dense_results, sparse_results, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _rrf_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1 / (k + rank_i(d))

        Where:
        - k = constant (typically 60) to prevent high scores for top ranks
        - rank_i(d) = rank of document d in result set i (1-indexed)
        """
        scores: Dict[str, float] = defaultdict(float)
        docs: Dict[str, SearchResult] = {}

        # Score from dense results
        for rank, result in enumerate(dense_results, start=1):
            scores[result.doc_id] += 1.0 / (self.rrf_k + rank)
            if result.doc_id not in docs:
                docs[result.doc_id] = result

        # Score from sparse results
        for rank, result in enumerate(sparse_results, start=1):
            scores[result.doc_id] += 1.0 / (self.rrf_k + rank)
            if result.doc_id not in docs:
                docs[result.doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_ids:
            result = docs[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                    source="hybrid",
                )
            )

        return results

    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
        dense_weight: float,
    ) -> List[SearchResult]:
        """
        Weighted score fusion with normalization.

        hybrid_score(d) = α * norm_dense(d) + (1-α) * norm_sparse(d)

        Where α is the dense weight.
        """
        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        scores: Dict[str, float] = defaultdict(float)
        docs: Dict[str, SearchResult] = {}

        # Weighted dense scores
        for result in dense_results:
            norm_score = dense_scores.get(result.doc_id, 0)
            scores[result.doc_id] += dense_weight * norm_score
            if result.doc_id not in docs:
                docs[result.doc_id] = result

        # Weighted sparse scores
        sparse_weight = 1.0 - dense_weight
        for result in sparse_results:
            norm_score = sparse_scores.get(result.doc_id, 0)
            scores[result.doc_id] += sparse_weight * norm_score
            if result.doc_id not in docs:
                docs[result.doc_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_ids:
            result = docs[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                    source="hybrid",
                )
            )

        return results

    def _max_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Take maximum normalized score from either source.

        Useful when one retrieval method may be significantly
        better for certain queries.
        """
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        scores: Dict[str, float] = {}
        docs: Dict[str, SearchResult] = {}

        for result in dense_results:
            docs[result.doc_id] = result
            scores[result.doc_id] = dense_scores.get(result.doc_id, 0)

        for result in sparse_results:
            if result.doc_id not in docs:
                docs[result.doc_id] = result
            sparse_score = sparse_scores.get(result.doc_id, 0)
            scores[result.doc_id] = max(scores.get(result.doc_id, 0), sparse_score)

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_ids:
            result = docs[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    content=result.content,
                    score=score,
                    metadata=result.metadata,
                    source="hybrid",
                )
            )

        return results

    def _normalize_scores(
        self,
        results: List[SearchResult],
    ) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: Search results with scores

        Returns:
            Dictionary mapping doc_id to normalized score
        """
        if not results:
            return {}

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Handle case where all scores are equal
        if max_score == min_score:
            return {r.doc_id: 1.0 for r in results}

        normalized = {}
        for result in results:
            normalized[result.doc_id] = (result.score - min_score) / (
                max_score - min_score
            )

        return normalized

    def get_index_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a collection's BM25 index."""
        if collection_name not in self._indexes:
            return None
        return self._indexes[collection_name].get_stats()

    def list_indexed_collections(self) -> List[str]:
        """List all collections with BM25 indexes."""
        return list(self._indexes.keys())

    def clear_index(self, collection_name: str) -> bool:
        """
        Clear the BM25 index for a collection.

        Returns:
            True if cleared, False if collection not found
        """
        if collection_name not in self._indexes:
            return False
        self._indexes[collection_name].clear()
        return True

    def delete_index(self, collection_name: str) -> bool:
        """
        Delete the BM25 index for a collection.

        Returns:
            True if deleted, False if not found
        """
        if collection_name in self._indexes:
            del self._indexes[collection_name]
            return True
        return False

    def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        total_docs = sum(idx.total_docs for idx in self._indexes.values())

        return {
            "status": "healthy",
            "fusion_method": self.fusion_method.value,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "indexed_collections": list(self._indexes.keys()),
            "total_indexed_docs": total_docs,
        }
