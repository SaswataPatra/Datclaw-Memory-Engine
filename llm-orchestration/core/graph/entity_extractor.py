"""
DAPPY Entity Extractor
Extracts entities from text using spaCy NER.

Key features:
- Uses spaCy for fast NER (no LLM calls for basic entity detection)
- Computes confidence scores (not hardcoded!)
- Maps spaCy entity types to DAPPY categories
- Supports both transformer (en_core_web_trf) and medium (en_core_web_md) models

Phase 1A Implementation
"""

import logging
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text."""
    text: str
    type: str  # DAPPY type (person, organization, location, etc.)
    original_type: str  # spaCy type (PERSON, ORG, GPE, etc.)
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "original_type": self.original_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class EntityExtractor:
    """
    Entity extraction using spaCy NER.
    
    Uses en_core_web_md by default (fast, good accuracy).
    Can use en_core_web_trf for higher accuracy (if available).
    
    Computes confidence from:
    1. Entity type certainty (some types are more reliable)
    2. Context clarity (longer entities are usually clearer)
    3. Token probability (if using transformer model)
    """
    
    # Entity type mapping: spaCy type → DAPPY type
    TYPE_MAPPING = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",      # Countries, cities, states
        "LOC": "location",      # Non-GPE locations
        "FAC": "location",      # Facilities
        "PRODUCT": "concept",
        "EVENT": "event",
        "WORK_OF_ART": "concept",
        "LAW": "concept",
        "LANGUAGE": "concept",
        "DATE": "temporal",
        "TIME": "temporal",
        "MONEY": "value",
        "QUANTITY": "quantity",
        "ORDINAL": "quantity",
        "CARDINAL": "quantity",
        "PERCENT": "value",
        "NORP": "group",        # Nationalities, religions, political groups
    }
    
    # Type certainty scores (some entity types are more reliable)
    TYPE_CERTAINTY = {
        "PERSON": 0.90,
        "ORG": 0.85,
        "GPE": 0.90,
        "LOC": 0.85,
        "DATE": 0.95,
        "TIME": 0.95,
        "MONEY": 0.95,
        "PERCENT": 0.95,
        "CARDINAL": 0.90,
        "EVENT": 0.75,
        "PRODUCT": 0.70,
        "WORK_OF_ART": 0.70,
        "NORP": 0.80,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize entity extractor.
        
        Args:
            config: Configuration dict with optional keys:
                - model: spaCy model name (default: en_core_web_md)
                - fallback_model: Fallback model if primary fails
        """
        self.config = config or {}
        self.nlp = None
        self.model_name = None
        self.has_transformer = False
        
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model with fallback."""
        import spacy
        
        primary_model = self.config.get('model', 'en_core_web_md')
        fallback_model = self.config.get('fallback_model', 'en_core_web_sm')
        
        # Try primary model first
        try:
            self.nlp = spacy.load(primary_model)
            self.model_name = primary_model
            self.has_transformer = 'trf' in primary_model
            logger.info(f"✅ Loaded spaCy model: {primary_model}")
        except OSError:
            logger.warning(f"Primary model {primary_model} not found, trying fallback")
            try:
                self.nlp = spacy.load(fallback_model)
                self.model_name = fallback_model
                self.has_transformer = False
                logger.info(f"✅ Loaded fallback spaCy model: {fallback_model}")
            except OSError:
                # Last resort: download and load en_core_web_sm
                logger.warning("Downloading en_core_web_sm as last resort")
                import subprocess
                # Check if we've already attempted to load en_core_web_sm to avoid repeated downloads
                if not getattr(self, "_en_core_web_sm_loaded", False):
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self._en_core_web_sm_loaded = True
                self.nlp = spacy.load("en_core_web_sm")
                self.model_name = "en_core_web_sm"
                self.has_transformer = False
        
        logger.info(f"  Model type: {'Transformer' if self.has_transformer else 'Standard'}")
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of ExtractedEntity objects with computed confidence
        """
        if not text or not text.strip():
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Compute confidence (not hardcoded!)
            confidence = self._compute_confidence(ent, doc)
            
            # Map type
            dappy_type = self.TYPE_MAPPING.get(ent.label_, "unknown")
            
            entity = ExtractedEntity(
                text=ent.text,
                type=dappy_type,
                original_type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=confidence,
                metadata={
                    "model": self.model_name,
                    "has_transformer": self.has_transformer
                }
            )
            entities.append(entity)
        
        logger.info(f"📍 EntityExtractor: Extracted {len(entities)} entities: {[f'{e.text}({e.type}, conf={e.confidence:.2f})' for e in entities]}")
        return entities
    
    def _compute_confidence(self, ent, doc) -> float:
        """
        Compute entity extraction confidence.
        
        Improved version based on empirically valid signals:
        1. Type certainty: Entity type reliability (from validation data)
        2. Span quality: Proper noun density (strong NER signal)
        3. Span cleanliness: No stopwords/determiners (quality indicator)
        4. Model confidence: Use spaCy's built-in score if available
        
        REMOVED (invalid assumptions):
        - Position in sentence (no empirical support)
        - token.prob (not NER confidence, just LM probability)
        - Word count alone (weak signal)
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Factor 1: Type certainty (entity type reliability)
        # These should ideally be tuned on validation data
        type_certainty = self.TYPE_CERTAINTY.get(ent.label_, 0.70)
        
        # Factor 2: Span quality (proper noun density)
        # Entities with more proper nouns are typically clearer
        # "Sarah" (PROPN) = 1.0, "the company" (DET + NOUN) = 0.0
        proper_noun_count = sum(1 for token in ent if token.pos_ == "PROPN")
        span_quality = proper_noun_count / len(ent) if len(ent) > 0 else 0.0
        
        # Factor 3: Span cleanliness (no noise tokens)
        # Penalize entities with determiners, stopwords, or punctuation
        noise_tokens = sum(
            1 for token in ent 
            if token.pos_ in ("DET", "PUNCT") or token.is_stop
        )
        span_cleanliness = 1.0 - (noise_tokens / len(ent)) if len(ent) > 0 else 1.0
        
        # Factor 4: Model confidence (if available)
        # Some spaCy models provide entity-level confidence scores
        model_confidence = 0.80  # Default baseline
        if hasattr(ent._, "confidence"):
            model_confidence = ent._.confidence
        elif self.has_transformer:
            # For transformer models, we can use a slightly higher baseline
            model_confidence = 0.85
        
        # Weighted combination (empirically more reasonable than previous version)
        # Weights prioritize:
        # - Model confidence (40%) - if model provides it, trust it most
        # - Type certainty (30%) - entity type is a strong signal
        # - Span quality (20%) - proper nouns indicate clear entities
        # - Span cleanliness (10%) - penalize noisy spans
        confidence = (
            0.40 * model_confidence +
            0.30 * type_certainty +
            0.20 * span_quality +
            0.10 * span_cleanliness
        )
        
        # Clamp to [0.0, 1.0] and round
        final_confidence = round(max(0.0, min(1.0, confidence)), 3)
        
        # ========== CONFIDENCE LOGGING (Phase 1E) ==========
        logger.debug(
            f"📊 Entity confidence for '{ent.text}' ({ent.label_}): "
            f"model={model_confidence:.2f}, "
            f"type={type_certainty:.2f}, "
            f"span_qual={span_quality:.2f}, "
            f"span_clean={span_cleanliness:.2f} "
            f"→ FINAL={final_confidence:.3f}"
        )
        
        return final_confidence
        
   #HELPER FUNCTIONS FOR EXTRACTION 
    def extract_with_context(
        self,
        text: str,
        context_window: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: Input text
            context_window: Characters of context to include
            
        Returns:
            List of entities with context
        """
        entities = self.extract(text)
        
        results = []
        for ent in entities:
            # Get surrounding context
            start = max(0, ent.start - context_window)
            end = min(len(text), ent.end + context_window)
            context = text[start:end]
            
            result = ent.to_dict()
            result["context"] = context
            result["context_start"] = start
            result["context_end"] = end
            results.append(result)
        
        return results
    
    def extract_persons(self, text: str) -> List[ExtractedEntity]:
        """Extract only person entities."""
        return [e for e in self.extract(text) if e.type == "person"]
    
    def extract_organizations(self, text: str) -> List[ExtractedEntity]:
        """Extract only organization entities."""
        return [e for e in self.extract(text) if e.type == "organization"]
    
    def extract_locations(self, text: str) -> List[ExtractedEntity]:
        """Extract only location entities."""
        return [e for e in self.extract(text) if e.type == "location"]
    
    def get_entity_pairs(self, text: str) -> List[tuple]:
        """
        Get pairs of entities for potential relation extraction.
        
        Returns:
            List of (entity1, entity2) tuples
        """
        entities = self.extract(text)
        
        pairs = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Only pair different entities
                if e1.text.lower() != e2.text.lower():
                    pairs.append((e1, e2))
        
        return pairs

 