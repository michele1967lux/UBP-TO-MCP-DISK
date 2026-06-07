"""
RAG Simple Memory Providers - Pure Technical Logic

Pure provider implementations with zero framework dependencies.
Can be tested independently and reused in other contexts.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStoreProvider(Protocol):
    """
    Protocol defining interface contract for vector store providers.

    Benefits:
    - Type safety at development time
    - Clear contract for implementations
    - Runtime validation with isinstance()
    - IDE autocomplete support
    """

    def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a document to the store."""
        ...

    def query(
        self, query_text: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query the store and return top-k similar documents."""
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        ...

    def clear(self) -> None:
        """Clear all documents."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        ...


class SimpleVectorStore:
    """Simple in-memory vector store using TF-IDF and cosine similarity."""

    # Memory estimation constants (in bytes)
    AVG_DOCUMENT_OVERHEAD = 1000  # Average overhead per document
    AVG_VOCABULARY_OVERHEAD = 100  # Average overhead per vocabulary term
    BYTES_PER_FLOAT64 = 8  # Size of a float64 in bytes
    BYTES_TO_MB = 1024 * 1024  # Conversion factor

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store."""
        self.config = config
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        # Memory protection limits
        self.max_documents = config.get("limits", {}).get("max_documents", 10000)
        self.max_document_size = config.get("limits", {}).get(
            "max_document_size", 1000000
        )  # 1MB
        self.max_memory_mb = config.get("limits", {}).get("max_memory_mb", 500)  # 500MB

        logger.info(
            "SimpleVectorStore initialized",
            extra={
                "max_documents": self.max_documents,
                "max_memory_mb": self.max_memory_mb,
            },
        )

    def _check_memory_limits(self) -> None:
        """Check if memory limits are exceeded."""
        # Check document count
        if len(self.documents) >= self.max_documents:
            raise MemoryError(f"Maximum document limit reached: {self.max_documents}")

        # Check approximate memory usage
        estimated_memory_mb = (
            len(self.documents) * self.AVG_DOCUMENT_OVERHEAD  # Document overhead
            + len(self.vocabulary) * self.AVG_VOCABULARY_OVERHEAD  # Vocabulary overhead
            + len(self.vectors)
            * len(self.vocabulary)
            * self.BYTES_PER_FLOAT64  # Vector storage (float64)
        ) / self.BYTES_TO_MB

        if estimated_memory_mb > self.max_memory_mb:
            logger.warning(
                "Memory limit approaching",
                extra={
                    "estimated_mb": estimated_memory_mb,
                    "limit_mb": self.max_memory_mb,
                },
            )
            raise MemoryError(
                f"Memory limit exceeded: {estimated_memory_mb:.2f}MB / {self.max_memory_mb}MB"
            )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        # Simple tokenization
        text = text.lower() if self.config["preprocessing"]["lowercase"] else text
        tokens = text.split()

        # Remove stopwords if enabled
        if self.config["preprocessing"]["remove_stopwords"]:
            stopwords = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
            }
            tokens = [t for t in tokens if t not in stopwords]

        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        tf = {}
        total = len(tokens)
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Normalize
        for token in tf:
            tf[token] = tf[token] / total

        return tf

    def _vectorize(self, text: str) -> np.ndarray:
        """Convert text to vector using TF-IDF."""
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # Create vector
        vector = np.zeros(len(self.vocabulary))
        for token, freq in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf.get(token, 1.0)
                vector[idx] = freq * idf

        # Normalize if enabled
        if self.config["embedding"]["normalize"]:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector

    def add_document(
        self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document to the store.

        Raises:
            ValueError: If document is invalid
            MemoryError: If memory limits exceeded
        """
        # Validate inputs
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string")

        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        # Check document size
        doc_size = sys.getsizeof(text)
        if doc_size > self.max_document_size:
            logger.warning(
                "Document size exceeds limit",
                extra={
                    "doc_id": doc_id,
                    "size_bytes": doc_size,
                    "limit_bytes": self.max_document_size,
                },
            )
            raise ValueError(
                f"Document size {doc_size} exceeds limit of {self.max_document_size} bytes"
            )

        # Check memory limits before adding
        self._check_memory_limits()

        logger.debug(
            "Adding document",
            extra={
                "doc_id": doc_id,
                "text_length": len(text),
                "has_metadata": metadata is not None,
            },
        )

        # Update vocabulary
        tokens = self._tokenize(text)
        vocab_changed = False
        for token in set(tokens):
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
                vocab_changed = True

        # Update IDF (simplified - would need full corpus for accurate IDF)
        for token in set(tokens):
            self.idf[token] = self.idf.get(token, 1.0)

        # Store document
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens,
        }

        # If vocabulary changed, rebuild all vectors
        if vocab_changed and len(self.documents) > 1:
            logger.debug(
                "Rebuilding vectors due to vocabulary change",
                extra={
                    "vocab_size": len(self.vocabulary),
                    "num_documents": len(self.documents),
                },
            )
            # Rebuild all document vectors with new vocabulary
            for doc_id_rebuild, doc_data in self.documents.items():
                if doc_id_rebuild != doc_id:  # Skip current doc, we'll compute it next
                    self.vectors[doc_id_rebuild] = self._vectorize(doc_data["text"])

        # Compute and store vector for current document
        self.vectors[doc_id] = self._vectorize(text)

        logger.info(
            "Document added successfully",
            extra={
                "doc_id": doc_id,
                "total_documents": len(self.documents),
                "vocabulary_size": len(self.vocabulary),
            },
        )

    def query(
        self, query_text: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query the store and return top-k similar documents."""
        if not self.documents:
            return []

        # Vectorize query
        query_vector = self._vectorize(query_text)

        # Compute similarities
        similarities = []
        for doc_id, doc_vector in self.vectors.items():
            # Cosine similarity
            similarity = np.dot(query_vector, doc_vector)

            if similarity >= threshold:
                similarities.append(
                    {
                        "doc_id": doc_id,
                        "score": float(similarity),
                        "text": self.documents[doc_id]["text"],
                        "metadata": self.documents[doc_id]["metadata"],
                    }
                )

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        Supports deleting parent doc by removing its chunk documents.
        """
        # Direct delete if exists
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            return True

        # If not found, attempt to delete chunks with parent_doc_id == doc_id
        to_delete = [
            cid
            for cid, doc in self.documents.items()
            if isinstance(doc.get("metadata"), dict)
            and doc["metadata"].get("parent_doc_id") == doc_id
        ]
        if to_delete:
            for cid in to_delete:
                del self.documents[cid]
                if cid in self.vectors:
                    del self.vectors[cid]
            return True

        return False

    def clear(self):
        """Clear all documents."""
        self.documents.clear()
        self.vectors.clear()
        self.vocabulary.clear()
        self.idf.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "vector_dimension": len(self.vocabulary),
        }
