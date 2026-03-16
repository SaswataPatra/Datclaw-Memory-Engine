"""
DAPPY Relation Classifier

Classifies relations between entities using the same pattern as label classification:
1. DeBERTa zero-shot (fast, primary)
2. LLM fallback (when uncertain)
3. Training data collection

Key features:
- Dynamic relation types (base + discovered)
- Reuses HuggingFace API integration
- Collects training data for future model improvement

Phase 1B Implementation
"""

import logging
import json
import httpx
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Base relation types organized by category
RELATION_CATEGORIES = {
    "family": [
        "sister_of", "brother_of", "parent_of", "child_of",
        "spouse_of", "cousin_of", "grandparent_of", "relative_of"
    ],
    "professional": [
        "works_at", "works_with", "colleague_of", "manages",
        "employed_by", "founded", "member_of", "partner_of"
    ],
    "personal": [
        "likes", "dislikes", "prefers", "loves", "hates",
        "interested_in", "friend_of", "knows"
    ],
    "temporal": [
        "evolves_to", "supersedes", "contradicts", "preceded_by",
        "followed_by", "during", "after", "before"
    ],
    "factual": [
        "located_at", "lives_in", "from", "owns", "has",
        "part_of", "instance_of", "type_of"
    ],
    "other": []  # Catch-all for discovered types
}


def get_all_relation_types(include_discovered: bool = True) -> List[str]:
    """Get all relation types including discovered ones."""
    types = []
    for category, relations in RELATION_CATEGORIES.items():
        types.extend(relations)
    return types


def get_relation_category(relation_type: str) -> str:
    """Get category for a relation type."""
    for category, relations in RELATION_CATEGORIES.items():
        if relation_type in relations:
            return category
    return "other"


@dataclass
class RelationResult:
    """Result of relation classification."""
    subject: str
    object: str
    relation: str
    category: str
    confidence: float
    is_discovered: bool = False
    source: str = "deberta"  # deberta, llm, heuristic
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "object": self.object,
            "relation": self.relation,
            "category": self.category,
            "confidence": self.confidence,
            "is_discovered": self.is_discovered,
            "source": self.source,
            "metadata": self.metadata
        }


class RelationClassifier:
    """
    Relation classification using the same pattern as label classification:
    1. DeBERTa zero-shot (fast)
    2. LLM fallback (when uncertain)
    3. Training data collection
    
    Reuses HuggingFace API integration from the label classifier.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize relation classifier.
        
        Args:
            config: Configuration dict with:
                - hf_api.api_key: HuggingFace API key
                - hf_api.model: Model name for zero-shot
                - relation_classification.confidence_threshold: LLM fallback threshold
                - relation_classification.llm_model: LLM model for fallback
        """
        self.config = config or {}
        
        # HuggingFace API settings
        hf_config = self.config.get('hf_api', {})
        self.hf_api_key = hf_config.get('api_key', '')
        self.hf_model = hf_config.get('model', 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
        # Updated to new HuggingFace router endpoint (Dec 2024)
        self.hf_api_url = f"https://router.huggingface.co/hf-inference/models/{self.hf_model}"
        
        # Relation classification settings
        rel_config = self.config.get('relation_classification', {})
        self.confidence_threshold = rel_config.get('confidence_threshold', 0.5)
        self.llm_model = rel_config.get('llm_model', 'gpt-4o-mini')
        
        # All relation types
        self.relation_types = get_all_relation_types()
        
        # Discovered relation types (loaded from store)
        self.discovered_types: List[str] = []
        
        # LLM client (lazy loaded)
        self._llm_client = None
        
        logger.info(f"✅ RelationClassifier initialized")
        logger.info(f"   Model: {self.hf_model}")
        logger.info(f"   Confidence threshold: {self.confidence_threshold}")
        logger.info(f"   Base relation types: {len(self.relation_types)}")
    
    @property
    def llm_client(self):
        """Lazy load OpenAI client."""
        if self._llm_client is None:
            try:
                from openai import AsyncOpenAI
                self._llm_client = AsyncOpenAI()
            except ImportError:
                logger.warning("OpenAI not installed, LLM fallback disabled")
        return self._llm_client
    
    @property
    def all_relation_types(self) -> List[str]:
        """Get all relation types including discovered ones."""
        return self.relation_types + self.discovered_types
    
    def add_discovered_type(self, relation_type: str, category: str = "other"):
        """Add a newly discovered relation type."""
        if relation_type not in self.all_relation_types:
            self.discovered_types.append(relation_type)
            if category in RELATION_CATEGORIES:
                RELATION_CATEGORIES[category].append(relation_type)
            logger.info(f"Added discovered relation type: {relation_type} ({category})")
    
    async def classify(
        self,
        subject: str,
        object_text: str,
        context: str,
        user_id: str = None
    ) -> RelationResult:
        """
        Classify the relation between subject and object.
        
        Args:
            subject: Subject entity text
            object_text: Object entity text
            context: Full sentence/context containing both entities
            user_id: Optional user ID for personalization
        
        Returns:
            RelationResult with relation type, confidence, and category
        """
        logger.debug(f"Classifying relation: {subject} → ? → {object_text}")
        
        # Step 1: Try DeBERTa zero-shot
        relation, confidence = await self._classify_with_deberta(
            subject, object_text, context
        )
        
        if confidence >= self.confidence_threshold:
            category = get_relation_category(relation)
            logger.info(f"🔍 RelationClassifier: {subject} → {relation} → {object_text} (method=DeBERTa, conf={confidence:.2f}, category={category})")
            return RelationResult(
                subject=subject,
                object=object_text,
                relation=relation,
                category=category,
                confidence=confidence,
                source="deberta"
            )
        
        # Step 2: LLM fallback
        logger.debug(f"DeBERTa confidence {confidence:.2f} < {self.confidence_threshold}, using LLM fallback")
        
        result = await self._classify_with_llm(
            subject, object_text, context
        )
        
        # Step 3: If new relation discovered, add to types
        if result.is_discovered:
            self.add_discovered_type(result.relation, result.category)
            logger.info(f"🆕 RelationClassifier: Discovered NEW relation type '{result.relation}' (category={result.category})")
        
        logger.info(f"🔍 RelationClassifier: {subject} → {result.relation} → {object_text} (method={result.source}, conf={result.confidence:.2f}, category={result.category})")
        return result
    
    async def _classify_with_deberta(
        self,
        subject: str,
        object_text: str,
        context: str
    ) -> Tuple[str, float]:
        """
        Zero-shot classification using DeBERTa via HuggingFace API.
        
        Uses natural language inference (NLI) to classify relations.
        Includes exponential backoff retry logic for API reliability.
        """
        if not self.hf_api_key:
            logger.debug("HuggingFace API key not configured, skipping DeBERTa")
            return "unknown", 0.0
        
        # Build the premise and hypotheses for NLI
        premise = f"Context: {context}. The entities mentioned are '{subject}' and '{object_text}'."
        
        # Create hypotheses for each relation type
        hypotheses = []
        for rel_type in self.all_relation_types:
            # Convert snake_case to natural language
            rel_natural = rel_type.replace("_", " ")
            hypothesis = f"{subject} {rel_natural} {object_text}"
            hypotheses.append((rel_type, hypothesis))
        
        # Exponential backoff retry logic
        max_retries = 3
        base_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Use multi-label classification
                    response = await client.post(
                        self.hf_api_url,
                        headers={"Authorization": f"Bearer {self.hf_api_key}"},
                        json={
                            "inputs": premise,
                            "parameters": {
                                "candidate_labels": [h[1] for h in hypotheses],
                                "multi_label": False
                            }
                        }
                    )
                    
                    # Handle different status codes
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Handle different response formats
                        if isinstance(result, dict) and "labels" in result:
                            top_hypothesis = result["labels"][0]
                            top_score = result["scores"][0]
                            
                            # Map back to relation type
                            for rel_type, hypothesis in hypotheses:
                                if hypothesis == top_hypothesis:
                                    logger.debug(f"DeBERTa classified: {rel_type} (conf={top_score:.2f})")
                                    return rel_type, top_score
                            
                            # If no exact match, use the first relation type
                            return self.all_relation_types[0], top_score
                        
                        logger.warning(f"Unexpected HuggingFace response format: {result}")
                        return "unknown", 0.0
                    
                    elif response.status_code == 503:
                        # Model loading, retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.info(f"HuggingFace model loading (503), retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(f"HuggingFace API still loading after {max_retries} attempts, falling back to LLM")
                            return "unknown", 0.0
                    
                    elif response.status_code in [410, 404]:
                        # Model deprecated or not found, don't retry
                        logger.warning(f"HuggingFace model unavailable ({response.status_code}), falling back to LLM")
                        return "unknown", 0.0
                    
                    elif response.status_code == 429:
                        # Rate limited, retry with backoff
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"HuggingFace rate limited (429), retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.warning(f"HuggingFace rate limit persists, falling back to LLM")
                            return "unknown", 0.0
                    
                    else:
                        # Other error, log and fall back
                        logger.warning(f"HuggingFace API error {response.status_code}, falling back to LLM")
                        return "unknown", 0.0
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"HuggingFace timeout, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(f"HuggingFace timeout after {max_retries} attempts, falling back to LLM")
                    return "unknown", 0.0
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"DeBERTa error: {e}, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(f"DeBERTa classification failed after {max_retries} attempts: {e}, falling back to LLM")
                    return "unknown", 0.0
        
        # Should never reach here, but just in case
        return "unknown", 0.0
    
    async def _classify_with_llm(
        self,
        subject: str,
        object_text: str,
        context: str
    ) -> RelationResult:
        """
        LLM fallback for relation classification.
        Can discover new relation types.
        """
        if self.llm_client is None:
            logger.warning("LLM client not available")
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="unknown",
                category="other",
                confidence=0.3,
                source="heuristic"
            )
        
        prompt = f"""Analyze the relationship between two entities in this text.

Text: "{context}"
Entity 1 (Subject): {subject}
Entity 2 (Object): {object_text}

Known relation types: {', '.join(self.all_relation_types)}

Relation categories:
- family: sister_of, brother_of, parent_of, spouse_of, etc.
- professional: works_at, colleague_of, manages, employed_by, etc.
- personal: likes, dislikes, prefers, friend_of, knows, etc.
- temporal: evolves_to, supersedes, contradicts, etc.
- factual: located_at, lives_in, owns, has, part_of, etc.

What is the relationship from Entity 1 to Entity 2?

If the relationship matches a known type, use that exact type.
If not, suggest a new relation type in snake_case format.

Respond in JSON format only:
{{
    "relation": "relation_type",
    "confidence": 0.0-1.0,
    "is_new": true/false,
    "category": "family|professional|personal|temporal|factual|other",
    "reasoning": "brief explanation"
}}
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return RelationResult(
                subject=subject,
                object=object_text,
                relation=result.get("relation", "unknown"),
                category=result.get("category", "other"),
                confidence=result.get("confidence", 0.7),
                is_discovered=result.get("is_new", False),
                source="llm",
                metadata={"reasoning": result.get("reasoning", "")}
            )
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="unknown",
                category="other",
                confidence=0.3,
                source="heuristic"
            )
    
    async def classify_all_pairs(
        self,
        entity_pairs: List[Tuple[Any, Any]],
        context: str,
        user_id: str = None
    ) -> List[RelationResult]:
        """
        Classify relations for all entity pairs.
        
        Args:
            entity_pairs: List of (entity1, entity2) tuples
            context: Full text context
            user_id: Optional user ID
        
        Returns:
            List of RelationResult for each pair
        """
        results = []
        
        for e1, e2 in entity_pairs:
            # Get text from entity (handle both ExtractedEntity and dict)
            subject = e1.text if hasattr(e1, 'text') else e1.get('text', str(e1))
            object_text = e2.text if hasattr(e2, 'text') else e2.get('text', str(e2))
            
            result = await self.classify(
                subject=subject,
                object_text=object_text,
                context=context,
                user_id=user_id
            )
            results.append(result)
        
        return results
    
    def classify_with_heuristics(
        self,
        subject: str,
        object_text: str,
        context: str
    ) -> RelationResult:
        """
        Fast heuristic-based classification (no API calls).
        Used as fallback when APIs are unavailable.
        """
        context_lower = context.lower()
        subject_lower = subject.lower()
        object_lower = object_text.lower()
        
        # Family relations
        family_patterns = {
            "sister_of": ["sister", "sis"],
            "brother_of": ["brother", "bro"],
            "parent_of": ["parent", "father", "mother", "dad", "mom"],
            "spouse_of": ["husband", "wife", "spouse", "married"],
            "child_of": ["son", "daughter", "child"],
        }
        
        for relation, patterns in family_patterns.items():
            for pattern in patterns:
                if pattern in context_lower:
                    return RelationResult(
                        subject=subject,
                        object=object_text,
                        relation=relation,
                        category="family",
                        confidence=0.6,
                        source="heuristic"
                    )
        
        # Professional relations
        if "works at" in context_lower or "work at" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="works_at",
                category="professional",
                confidence=0.7,
                source="heuristic"
            )
        
        if "colleague" in context_lower or "coworker" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="colleague_of",
                category="professional",
                confidence=0.6,
                source="heuristic"
            )
        
        # Personal relations
        if "friend" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="friend_of",
                category="personal",
                confidence=0.6,
                source="heuristic"
            )
        
        if "like" in context_lower or "love" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="likes",
                category="personal",
                confidence=0.5,
                source="heuristic"
            )
        
        if "hate" in context_lower or "dislike" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="dislikes",
                category="personal",
                confidence=0.5,
                source="heuristic"
            )
        
        # Location relations
        if "live" in context_lower or "from" in context_lower:
            return RelationResult(
                subject=subject,
                object=object_text,
                relation="lives_in",
                category="factual",
                confidence=0.5,
                source="heuristic"
            )
        
        # Default: knows
        return RelationResult(
            subject=subject,
            object=object_text,
            relation="knows",
            category="personal",
            confidence=0.3,
            source="heuristic"
        )

