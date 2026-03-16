"""
Speech Act Extractor

Extracts ephemeral speech acts like vocatives, gratitude, greetings.

Examples:
    "Thanks, Mel" → thanks(User, Mel)
    "Hello, John" → greets(User, John)
    "I agree, Mel" → addresses(User, Mel)
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .relation_extractor import ExtractedRelation

logger = logging.getLogger(__name__)


@dataclass
class SpeechActTriple:
    """A speech act extracted from text."""
    subject: str  # The speaker
    predicate: str  # thanks, greets, addresses
    object: str  # The addressee
    confidence: float


class SpeechActExtractor:
    """
    Extracts speech act relations (vocatives, gratitude, greetings) from sentences.
    
    Patterns:
    1. Vocative: ", Name" or "Name,"
    2. Gratitude: "Thanks/Thank you, Name"
    3. Greeting: "Hello/Hi, Name"
    
    Note: These are typically Tier 4 (ephemeral) with short TTL.
    """
    
    # Gratitude patterns
    GRATITUDE_PATTERNS = [
        r'\b(thanks?|thank\s+you)\b',
        r'\b(appreciate\s+it)\b',
    ]
    
    # Greeting patterns
    GREETING_PATTERNS = [
        r'\b(hello|hi|hey|good\s+morning|good\s+evening|good\s+afternoon)\b',
    ]
    
    def __init__(self, nlp, entity_resolver, config: Dict = None):
        """
        Initialize the speech act extractor.
        
        Args:
            nlp: spaCy language model
            entity_resolver: Entity resolver for canonicalizing entities
            config: Configuration dict
        """
        self.nlp = nlp
        self.entity_resolver = entity_resolver
        self.config = config or {}
        
        # Compile regex patterns
        self.gratitude_regex = re.compile('|'.join(self.GRATITUDE_PATTERNS), re.IGNORECASE)
        self.greeting_regex = re.compile('|'.join(self.GREETING_PATTERNS), re.IGNORECASE)
        
        logger.info("✅ SpeechActExtractor initialized")
    
    async def extract(
        self,
        text: str,
        resolved_text: str,
        doc,
        resolved_entities: Dict,
        user_id: str,
        memory_id: str,
        ego_score: float,
        metadata: Dict[str, Any],
        entity_resolver,
        relation_normalizer
    ) -> List[ExtractedRelation]:
        """
        Extract speech act relations from text.
        
        Args:
            text: Original text
            resolved_text: Text after coreference resolution
            doc: spaCy Doc object
            resolved_entities: Dict of resolved entities
            user_id: User ID
            memory_id: Memory ID
            ego_score: Ego score
            metadata: Metadata dict
            entity_resolver: Entity resolver instance
            relation_normalizer: Relation normalizer instance
        
        Returns:
            List of ExtractedRelation objects
        """
        logger.info(f"🗣️  SpeechActExtractor: Extracting from '{text[:50]}...'")
        
        # Parse text if doc not provided
        if doc is None:
            doc = self.nlp(resolved_text or text)
        
        triples = []
        
        # Pattern 1: Vocative (", Name" or "Name,")
        triples.extend(self._extract_vocative(doc, text))
        
        # Pattern 2: Gratitude
        triples.extend(self._extract_gratitude(doc, text))
        
        # Pattern 3: Greeting
        triples.extend(self._extract_greeting(doc, text))
        
        if not triples:
            logger.info("   → No speech act patterns found")
            return []
        
        logger.info(f"   → Found {len(triples)} speech act triples")
        
        # Get speaker from metadata
        speaker = metadata.get("speaker", "User")
        
        # Convert triples to ExtractedRelation objects
        relations = []
        for triple in triples:
            try:
                # Resolve entities
                subject_entity = await entity_resolver.resolve(
                    speaker, user_id, entity_type="person"
                )
                
                object_entity = await entity_resolver.resolve(
                    triple.object, user_id, entity_type="person"
                )
                
                # Create relation
                relation = ExtractedRelation(
                    subject_entity_id=subject_entity.entity_id,
                    subject_text=speaker,
                    object_entity_id=object_entity.entity_id,
                    object_text=triple.object,
                    relation=triple.predicate,
                    category="social",
                    confidence=triple.confidence,
                    context=text,
                    source="speech_act_extractor",
                    memory_id=memory_id,
                    metadata={
                        **metadata,
                        "pattern": "speech_act",
                        "tier": 4,  # Ephemeral
                        "ttl_days": 7
                    }
                )
                
                relations.append(relation)
                logger.info(f"   ✅ {speaker} --[{triple.predicate}]--> {triple.object} (conf={triple.confidence:.2f})")
            
            except Exception as e:
                logger.warning(f"Failed to create speech act relation: {e}")
                continue
        
        logger.info(f"🗣️  SpeechActExtractor: Extracted {len(relations)} relations")
        return relations
    
    def _extract_vocative(self, doc, text: str) -> List[SpeechActTriple]:
        """
        Extract: ", Mel" or "Mel,"
        Pattern: Comma + PROPN or PROPN + Comma
        """
        triples = []
        
        # Look for proper nouns adjacent to commas
        for i, token in enumerate(doc):
            if token.pos_ == "PROPN":
                # Check if preceded or followed by comma
                has_comma_before = i > 0 and doc[i-1].text == ","
                has_comma_after = i < len(doc) - 1 and doc[i+1].text == ","
                
                if has_comma_before or has_comma_after:
                    # This is likely a vocative
                    triple = SpeechActTriple(
                        subject="User",  # Will be replaced with speaker
                        predicate="addresses",
                        object=token.text,
                        confidence=0.75
                    )
                    triples.append(triple)
                    logger.debug(f"   Vocative pattern: addressing '{token.text}'")
        
        return triples
    
    def _extract_gratitude(self, doc, text: str) -> List[SpeechActTriple]:
        """
        Extract: "Thanks, Mel"
        Pattern: Gratitude word + comma + PROPN
        """
        triples = []
        
        # Check if text contains gratitude pattern
        if self.gratitude_regex.search(text):
            # Find proper nouns after the gratitude word
            for i, token in enumerate(doc):
                if token.pos_ == "PROPN":
                    # Check if there's a gratitude word before it
                    for j in range(max(0, i-5), i):
                        if self.gratitude_regex.search(doc[j].text.lower()):
                            triple = SpeechActTriple(
                                subject="User",  # Will be replaced with speaker
                                predicate="thanks",
                                object=token.text,
                                confidence=0.85
                            )
                            triples.append(triple)
                            logger.debug(f"   Gratitude pattern: thanking '{token.text}'")
                            break
        
        return triples
    
    def _extract_greeting(self, doc, text: str) -> List[SpeechActTriple]:
        """
        Extract: "Hello, John"
        Pattern: Greeting word + comma + PROPN
        """
        triples = []
        
        # Check if text contains greeting pattern
        if self.greeting_regex.search(text):
            # Find proper nouns after the greeting word
            for i, token in enumerate(doc):
                if token.pos_ == "PROPN":
                    # Check if there's a greeting word before it
                    for j in range(max(0, i-5), i):
                        if self.greeting_regex.search(doc[j].text.lower()):
                            triple = SpeechActTriple(
                                subject="User",  # Will be replaced with speaker
                                predicate="greets",
                                object=token.text,
                                confidence=0.80
                            )
                            triples.append(triple)
                            logger.debug(f"   Greeting pattern: greeting '{token.text}'")
                            break
        
        return triples

