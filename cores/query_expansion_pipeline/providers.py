"""
query_expansion_pipeline/providers.py

Core providers and data classes for query expansion.

Provides:
- Data classes (ExpandedQuery, ExpansionResult, Intent, Entity)
- QueryNormalizer: Query cleaning and normalization
- SynonymProvider: Synonym/hypernym expansion
- EntityExtractor: Named entity extraction
- IntentClassifier: Query intent detection
- QualityScorer: Expansion quality scoring
- CacheProvider: Redis caching

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class ExpansionStrategy(str, Enum):
    """Query expansion strategies."""
    SEMANTIC = "semantic"
    SYNONYM = "synonym"
    DECOMPOSE = "decompose"
    REFORMULATE = "reformulate"
    KEYWORDS = "keywords"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"


class QueryIntent(str, Enum):
    """Query intent types."""
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    PROCEDURAL = "procedural"
    FACTUAL = "factual"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Named entity types."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECH"
    LOCATION = "LOCATION"
    DATE = "DATE"
    NUMBER = "NUMBER"
    OTHER = "OTHER"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ExpandedQuery:
    """A single expanded query variant."""
    text: str
    strategy: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "score": round(self.score, 3),
            "metadata": self.metadata,
        }


@dataclass
class DetectedIntent:
    """Detected query intent."""
    intent: QueryIntent
    confidence: float
    signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
        }


@dataclass
class ExtractedEntity:
    """Extracted named entity."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            "normalized": self.normalized,
        }


@dataclass
class ExpansionResult:
    """Complete expansion result."""
    original_query: str
    expanded_queries: List[ExpandedQuery]
    combined_query: str
    strategy_used: str
    intent: Optional[DetectedIntent] = None
    entities: List[ExtractedEntity] = field(default_factory=list)
    language: str = "en"
    time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "expanded_queries": [q.to_dict() for q in self.expanded_queries],
            "combined_query": self.combined_query,
            "strategy_used": self.strategy_used,
            "intent": self.intent.to_dict() if self.intent else None,
            "entities": [e.to_dict() for e in self.entities],
            "language": self.language,
            "time_ms": round(self.time_ms, 2),
            "expansion_count": len(self.expanded_queries),
            "metadata": self.metadata,
        }
    
    def get_all_queries(self, include_original: bool = True) -> List[str]:
        """Get all query texts."""
        queries = [q.text for q in self.expanded_queries]
        if include_original and self.original_query not in queries:
            queries.insert(0, self.original_query)
        return queries


@dataclass
class DecomposedQuery:
    """Decomposed complex query."""
    original: str
    subqueries: List[str]
    dependencies: Dict[int, List[int]] = field(default_factory=dict)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "subqueries": self.subqueries,
            "dependencies": self.dependencies,
            "reasoning": self.reasoning,
        }


# ============================================================================
# Query Normalizer
# ============================================================================


class QueryNormalizer:
    """
    Normalizes and cleans queries.
    
    Operations:
    - Whitespace normalization
    - Punctuation handling
    - Abbreviation expansion
    - Optional lowercase
    - Optional stopword removal
    """
    
    # Common abbreviations
    ABBREVIATIONS = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "api": "application programming interface",
        "db": "database",
        "sql": "structured query language",
        "ui": "user interface",
        "ux": "user experience",
        "llm": "large language model",
        "rag": "retrieval augmented generation",
        "gpu": "graphics processing unit",
        "cpu": "central processing unit",
        "os": "operating system",
        "vs": "versus",
        "ie": "that is",
        "eg": "for example",
    }
    
    # Common stopwords
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "of", "to", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "and", "but", "if", "or", "because", "until",
        "while", "about", "against", "this", "that", "these", "those",
    }
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        expand_abbreviations: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.expand_abbreviations = expand_abbreviations
    
    def normalize(self, query: str) -> str:
        """Normalize a query."""
        # Basic cleaning
        text = query.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Expand abbreviations
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w.lower() not in self.STOPWORDS]
            text = ' '.join(words)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations."""
        words = text.split()
        result = []
        
        for word in words:
            lower = word.lower().strip('.,!?')
            if lower in self.ABBREVIATIONS:
                # Preserve case of first letter
                expanded = self.ABBREVIATIONS[lower]
                if word[0].isupper():
                    expanded = expanded.capitalize()
                result.append(expanded)
            else:
                result.append(word)
        
        return ' '.join(result)


# ============================================================================
# Intent Classifier
# ============================================================================


class IntentClassifier:
    """
    Classifies query intent using rule-based patterns.
    
    Intents:
    - informational: Seeking information
    - definition: Asking for definition
    - procedural: How to do something
    - comparison: Comparing things
    - factual: Specific facts
    - opinion: Seeking opinions
    """
    
    # Intent patterns
    PATTERNS = {
        QueryIntent.DEFINITION: [
            r"^what (?:is|are) (?:a |an |the )?",
            r"^define ",
            r"^meaning of ",
            r"^definition of ",
            r" meaning$",
            r" definition$",
        ],
        QueryIntent.PROCEDURAL: [
            r"^how (?:do|can|to|should) ",
            r"^how (?:is|are) .+ (?:done|made|created)",
            r"^steps to ",
            r"^guide (?:to|for) ",
            r"^tutorial ",
            r"^instructions for ",
        ],
        QueryIntent.COMPARISON: [
            r" vs\.? ",
            r" versus ",
            r" compared to ",
            r" difference between ",
            r" better than ",
            r" or ",
            r"^compare ",
            r"^which is better",
        ],
        QueryIntent.FACTUAL: [
            r"^when (?:did|was|is|will)",
            r"^where (?:is|are|was|were)",
            r"^who (?:is|was|are|were)",
            r"^how (?:many|much|long|old|far)",
            r" in (?:what year|which year)",
        ],
        QueryIntent.OPINION: [
            r"^(?:do you|should i|is it) (?:think|recommend|suggest)",
            r" opinion ",
            r" review ",
            r"^best ",
            r"^worst ",
            r" recommend ",
        ],
        QueryIntent.NAVIGATIONAL: [
            r" official ",
            r" website ",
            r" login ",
            r" sign in ",
            r" download ",
            r"\.com$",
            r"\.org$",
        ],
    }
    
    def classify(self, query: str) -> DetectedIntent:
        """Classify query intent."""
        query_lower = query.lower().strip()
        
        best_intent = QueryIntent.INFORMATIONAL
        best_confidence = 0.3
        signals = []
        
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    signals.append(f"matched: {pattern[:30]}...")
                    
                    # Calculate confidence based on match strength
                    confidence = 0.7 + (0.1 * len(signals))
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        # Cap confidence
        best_confidence = min(best_confidence, 0.95)
        
        return DetectedIntent(
            intent=best_intent,
            confidence=best_confidence,
            signals=signals[:3],  # Limit signals
        )


# ============================================================================
# Entity Extractor
# ============================================================================


class EntityExtractor:
    """
    Extracts named entities from queries.
    
    Uses rule-based patterns for common entity types.
    """
    
    # Entity patterns
    PATTERNS = {
        EntityType.TECHNOLOGY: [
            r"\b(Python|Java|JavaScript|TypeScript|Rust|Go|C\+\+|Ruby|PHP|Swift|Kotlin)\b",
            r"\b(React|Vue|Angular|Django|FastAPI|Flask|Spring|Node\.js|Express)\b",
            r"\b(Docker|Kubernetes|AWS|Azure|GCP|PostgreSQL|MongoDB|Redis|Elasticsearch)\b",
            r"\b(TensorFlow|PyTorch|Keras|scikit-learn|Pandas|NumPy)\b",
            r"\b(GPT|BERT|LLaMA|Claude|Gemini|ChatGPT)\b",
        ],
        EntityType.PRODUCT: [
            r"\b(iPhone|iPad|MacBook|Windows|Linux|Ubuntu|Android|iOS)\b",
            r"\b(Chrome|Firefox|Safari|Edge|Opera)\b",
            r"\b(VS Code|Visual Studio|IntelliJ|PyCharm|Sublime)\b",
        ],
        EntityType.ORGANIZATION: [
            r"\b(Google|Microsoft|Apple|Amazon|Meta|OpenAI|Anthropic)\b",
            r"\b(Netflix|Spotify|Tesla|SpaceX|NVIDIA)\b",
            r"\b(IBM|Oracle|SAP|Salesforce|Adobe)\b",
        ],
        EntityType.DATE: [
            r"\b(\d{4})\b",  # Years
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
            r"\b(today|yesterday|tomorrow|last week|next month)\b",
        ],
    }
    
    def extract(self, query: str) -> List[ExtractedEntity]:
        """Extract entities from query."""
        entities = []
        
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                    ))
        
        # Remove duplicates
        seen = set()
        unique = []
        for e in entities:
            key = (e.text.lower(), e.start)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        return unique


# ============================================================================
# Synonym Provider
# ============================================================================


class SynonymProvider:
    """
    Provides synonyms and related terms.
    
    Uses a built-in dictionary for common terms.
    """
    
    # Basic synonym dictionary
    SYNONYMS = {
        # Tech terms
        "create": ["make", "build", "generate", "develop", "construct"],
        "delete": ["remove", "erase", "eliminate", "clear", "destroy"],
        "update": ["modify", "change", "edit", "revise", "alter"],
        "get": ["retrieve", "fetch", "obtain", "acquire", "access"],
        "search": ["find", "look for", "query", "seek", "locate"],
        "fast": ["quick", "rapid", "speedy", "swift", "efficient"],
        "slow": ["sluggish", "delayed", "lagging", "inefficient"],
        "error": ["bug", "issue", "problem", "fault", "defect"],
        "fix": ["repair", "resolve", "correct", "patch", "debug"],
        
        # Common verbs
        "use": ["utilize", "employ", "apply", "leverage"],
        "show": ["display", "present", "demonstrate", "reveal"],
        "help": ["assist", "aid", "support", "guide"],
        "learn": ["understand", "study", "master", "grasp"],
        "work": ["function", "operate", "perform", "run"],
        
        # Common nouns
        "way": ["method", "approach", "technique", "manner"],
        "example": ["instance", "sample", "case", "illustration"],
        "problem": ["issue", "challenge", "difficulty", "obstacle"],
        "solution": ["answer", "resolution", "fix", "remedy"],
        "difference": ["distinction", "variation", "contrast", "disparity"],
        
        # Question words (for reformulation)
        "what": ["which", "what kind of"],
        "how": ["in what way", "by what means"],
        "why": ["for what reason", "what causes"],

        # ── HoReCa / Food & Beverage domain ────────────────────────
        # Spirits
        "gin": ["dry gin", "london dry gin", "gin botanico", "botanical gin"],
        "rum": ["rhum", "ron", "dark rum", "white rum"],
        "whiskey": ["whisky", "bourbon", "scotch", "single malt"],
        "whisky": ["whiskey", "bourbon", "scotch", "single malt"],
        "vodka": ["vodka premium", "flavored vodka"],
        "tequila": ["mezcal", "tequila reposado", "tequila blanco"],
        "grappa": ["distillato", "acquavite", "brandy"],
        # Wine & sparkling
        "vino": ["wine", "rosso", "bianco", "rosé"],
        "prosecco": ["spumante", "bollicine", "champagne", "cava"],
        "champagne": ["spumante", "bollicine", "prosecco", "brut"],
        "bollicine": ["prosecco", "spumante", "champagne"],
        # Beer
        "birra": ["beer", "ale", "lager", "pilsner", "ipa"],
        "beer": ["birra", "ale", "lager", "pilsner", "ipa"],
        # Cocktails
        "cocktail": ["drink", "mixed drink", "aperitivo"],
        "aperitivo": ["aperitif", "cocktail", "spritz"],
        "spritz": ["aperol spritz", "hugo", "campari spritz"],
        "negroni": ["sbagliato", "boulevardier", "americano"],
        # Flavour descriptors (IT ↔ EN)
        "secco": ["dry", "brut", "crisp", "asciutto"],
        "dolce": ["sweet", "zuccherato", "morbido", "abboccato"],
        "amaro": ["bitter", "amaricante", "intenso"],
        "profumato": ["aromatico", "fragrante", "botanical", "aromatic"],
        "aromatico": ["profumato", "fragrante", "botanical", "aromatic"],
        "fruttato": ["fruity", "fruit-forward", "fresco"],
        "speziato": ["spicy", "pepato", "piccante"],
        "leggero": ["light", "delicato", "soft", "smooth"],
        "forte": ["strong", "intenso", "robusto", "full-bodied"],
        "fresco": ["fresh", "crisp", "rinfrescante"],
        # Food
        "cibo": ["food", "piatto", "portata"],
        "piatto": ["dish", "portata", "pietanza"],
        "antipasto": ["starter", "appetizer", "stuzzichino"],
        "primo": ["pasta", "risotto", "zuppa", "minestra"],
        "secondo": ["main course", "carne", "pesce"],
        "dessert": ["dolce", "torta", "gelato", "tiramisù"],
        "contorno": ["side dish", "insalata", "verdure"],
        "panino": ["sandwich", "toast", "burger", "hamburger"],
        "burger": ["hamburger", "panino", "smash burger"],
        # Dietary
        "vegano": ["vegan", "plant-based", "senza derivati animali"],
        "vegetariano": ["vegetarian", "senza carne"],
        "senza glutine": ["gluten-free", "celiaco", "no glutine"],
        "analcolico": ["non-alcoholic", "zero alcol", "mocktail"],
    }
    
    def get_synonyms(
        self,
        word: str,
        max_synonyms: int = 3,
    ) -> List[str]:
        """Get synonyms for a word."""
        lower = word.lower()
        
        if lower in self.SYNONYMS:
            return self.SYNONYMS[lower][:max_synonyms]
        
        return []
    
    def expand_query_with_synonyms(
        self,
        query: str,
        max_synonyms_per_term: int = 2,
    ) -> List[str]:
        """Expand query by replacing words with synonyms."""
        words = query.split()
        expansions = []
        
        for i, word in enumerate(words):
            synonyms = self.get_synonyms(word, max_synonyms_per_term)
            
            for syn in synonyms:
                new_words = words[:i] + [syn] + words[i+1:]
                expansions.append(' '.join(new_words))
        
        return expansions


# ============================================================================
# Quality Scorer
# ============================================================================


class QualityScorer:
    """
    Scores expansion quality.
    
    Criteria:
    - Relevance to original
    - Diversity from other expansions
    - Length appropriateness
    - Grammar/structure
    """
    
    def __init__(
        self,
        min_length: int = 3,
        max_length: int = 200,
        similarity_threshold: float = 0.9,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.similarity_threshold = similarity_threshold
    
    def score_expansion(
        self,
        original: str,
        expansion: str,
        existing: List[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a single expansion."""
        existing = existing or []
        scores = {}
        
        # Length score
        length = len(expansion)
        if length < self.min_length:
            scores["length"] = 0.0
        elif length > self.max_length:
            scores["length"] = 0.5
        else:
            scores["length"] = 1.0
        
        # Relevance score (word overlap)
        orig_words = set(original.lower().split())
        exp_words = set(expansion.lower().split())
        
        if orig_words:
            overlap = len(orig_words & exp_words) / len(orig_words)
            scores["relevance"] = min(overlap + 0.3, 1.0)
        else:
            scores["relevance"] = 0.5
        
        # Diversity score
        if existing:
            max_similarity = 0
            for ex in existing:
                sim = self._word_similarity(expansion, ex)
                max_similarity = max(max_similarity, sim)
            
            scores["diversity"] = 1.0 - max_similarity
        else:
            scores["diversity"] = 1.0
        
        # Overall score
        total = (
            scores["length"] * 0.2 +
            scores["relevance"] * 0.4 +
            scores["diversity"] * 0.4
        )
        
        return total, scores
    
    def _word_similarity(self, text1: str, text2: str) -> float:
        """Compute word-based similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union
    
    def filter_expansions(
        self,
        original: str,
        expansions: List[ExpandedQuery],
        min_score: float = 0.3,
    ) -> List[ExpandedQuery]:
        """Filter expansions by quality."""
        filtered = []
        seen_texts = []
        
        for exp in expansions:
            # Check duplicate
            is_dup = any(
                self._word_similarity(exp.text, seen) > self.similarity_threshold
                for seen in seen_texts
            )
            
            if is_dup:
                continue
            
            # Score
            score, _ = self.score_expansion(original, exp.text, seen_texts)
            
            if score >= min_score:
                exp.score = score
                filtered.append(exp)
                seen_texts.append(exp.text)
        
        return filtered


# ============================================================================
# Cache Provider
# ============================================================================


class CacheProvider:
    """Redis cache for expansion results."""
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        prefix: str = "ubp:query_expansion",
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self.prefix = prefix
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._stats = {"hits": 0, "misses": 0}
    
    def _make_key(self, query: str, strategy: str) -> str:
        """Generate cache key."""
        hash_input = f"{strategy}:{query}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()
        return f"{self.prefix}:{hash_val}"
    
    async def get(
        self,
        query: str,
        strategy: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached expansions."""
        if not self.enabled or not self._redis:
            return None
        
        try:
            import json
            key = self._make_key(query, strategy)
            value = await self._redis.get(key)
            
            if value:
                self._stats["hits"] += 1
                return json.loads(value)
            
            self._stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        query: str,
        strategy: str,
        expansions: List[Dict[str, Any]],
    ) -> bool:
        """Cache expansions."""
        if not self.enabled or not self._redis:
            return False
        
        try:
            import json
            key = self._make_key(query, strategy)
            await self._redis.setex(key, self.ttl, json.dumps(expansions))
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / max(total, 1)
        
        return {
            "enabled": self.enabled and self._redis is not None,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 3),
        }


# ============================================================================
# Language Detector
# ============================================================================


class LanguageDetector:
    """Simple language detection."""
    
    # Language indicators
    INDICATORS = {
        "it": {
            "il", "la", "di", "che", "è", "un", "per", "con", "non",
            "sono", "come", "cosa", "questo", "quello", "anche", "più",
            "della", "delle", "degli", "nei", "nella", "quando", "perché",
        },
        "de": {
            "der", "die", "das", "und", "ist", "ein", "eine", "für",
            "mit", "auf", "den", "dem", "nicht", "sich", "von", "sind",
        },
        "fr": {
            "le", "la", "les", "de", "du", "des", "est", "une", "un",
            "pour", "avec", "dans", "sur", "sont", "pas", "que", "qui",
        },
        "es": {
            "el", "la", "de", "que", "es", "un", "una", "para", "con",
            "no", "en", "los", "las", "del", "por", "son", "más",
        },
    }
    
    def detect(self, text: str) -> str:
        """Detect language of text."""
        words = set(text.lower().split())
        
        best_lang = "en"
        best_score = 0
        
        for lang, indicators in self.INDICATORS.items():
            score = len(words & indicators)
            if score > best_score:
                best_score = score
                best_lang = lang
        
        # Need at least 2 matches to be confident
        if best_score < 2:
            return "en"
        
        return best_lang
