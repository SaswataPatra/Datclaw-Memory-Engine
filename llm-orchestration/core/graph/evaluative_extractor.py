"""
Evaluative Extractor

Extracts value judgments and preferences from evaluative sentences.

Examples:
    "Hanging with loved ones is amazing" → values(User, loved ones)
    "Music is great" → values(User, music)
    "I love pizza" → values(User, pizza)
    "I hate traffic" → dislikes(User, traffic)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .relation_extractor import ExtractedRelation

logger = logging.getLogger(__name__)


@dataclass
class EvaluativeTriple:
    """A value judgment extracted from text."""
    subject: str  # The thing being evaluated
    predicate: str  # values, dislikes, prefers
    sentiment: str  # positive, negative, neutral
    confidence: float


class EvaluativeExtractor:
    """
    Extracts evaluative relations (values, preferences) from sentences.
    
    Patterns:
    1. Gerund + Copula + Adjective: "Hanging with X is amazing"
    2. Noun + Copula + Adjective: "Music is great"
    3. Emotion Verb: "I love X", "I hate Y"
    """
    
    # Positive adjectives indicating value
    POSITIVE_ADJECTIVES = {
        "amazing", "great", "wonderful", "fantastic", "excellent", "awesome",
        "beautiful", "lovely", "perfect", "incredible", "outstanding",
        "good", "nice", "cool", "fun", "enjoyable", "pleasant"
    }
    
    # Negative adjectives indicating dislike
    NEGATIVE_ADJECTIVES = {
        "terrible", "awful", "horrible", "bad", "poor", "disappointing",
        "unpleasant", "annoying", "frustrating", "boring", "dull"
    }
    
    # Emotion verbs
    POSITIVE_VERBS = {"love", "adore", "enjoy", "like", "appreciate", "cherish"}
    NEGATIVE_VERBS = {"hate", "despise", "dislike", "detest", "loathe"}
    
    def __init__(self, nlp, entity_resolver, config: Dict = None):
        """
        Initialize the evaluative extractor.
        
        Args:
            nlp: spaCy language model
            entity_resolver: Entity resolver for canonicalizing entities
            config: Configuration dict
        """
        self.nlp = nlp
        self.entity_resolver = entity_resolver
        self.config = config or {}
        
        logger.info("✅ EvaluativeExtractor initialized")
    
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
        Extract evaluative relations from text.
        
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
        logger.info(f"🎨 EvaluativeExtractor: Extracting from '{text[:50]}...'")
        
        # Parse text if doc not provided
        if doc is None:
            doc = self.nlp(resolved_text or text)
        
        triples = []
        
        # Pattern 1: Gerund + Copula + Adjective
        triples.extend(self._extract_gerund_evaluation(doc))
        
        # Pattern 2: Noun + Copula + Adjective
        triples.extend(self._extract_noun_evaluation(doc))
        
        # Pattern 3: Emotion Verb
        triples.extend(self._extract_emotion_verb(doc))
        
        if not triples:
            logger.info("   → No evaluative patterns found")
            return []
        
        logger.info(f"   → Found {len(triples)} evaluative triples")
        
        # Convert triples to ExtractedRelation objects
        relations = []
        for triple in triples:
            # Resolve the subject entity
            try:
                subject_entity = await entity_resolver.resolve(
                    triple.subject, user_id, entity_type=None
                )
                
                # Get speaker from metadata (for "I love X" → "User values X")
                speaker = metadata.get("speaker", "User")
                speaker_entity = await entity_resolver.resolve(
                    speaker, user_id, entity_type="person"
                )
                
                # Determine relation type based on sentiment
                if triple.sentiment == "positive":
                    relation_type = "values"
                    category = "affective"
                elif triple.sentiment == "negative":
                    relation_type = "dislikes"
                    category = "affective"
                else:
                    relation_type = "evaluates"
                    category = "cognitive"
                
                # Create relation
                relation = ExtractedRelation(
                    subject_entity_id=speaker_entity.entity_id,
                    subject_text=speaker,
                    object_entity_id=subject_entity.entity_id,
                    object_text=triple.subject,
                    relation=relation_type,
                    category=category,
                    confidence=triple.confidence,
                    context=text,
                    source="evaluative_extractor",
                    memory_id=memory_id,
                    metadata={
                        **metadata,
                        "sentiment": triple.sentiment,
                        "pattern": "evaluative"
                    }
                )
                
                relations.append(relation)
                logger.info(f"   ✅ {speaker} --[{relation_type}]--> {triple.subject} (sentiment={triple.sentiment}, conf={triple.confidence:.2f})")
            
            except Exception as e:
                logger.warning(f"Failed to create evaluative relation for '{triple.subject}': {e}")
                continue
        
        logger.info(f"🎨 EvaluativeExtractor: Extracted {len(relations)} relations")
        return relations
    
    def _extract_gerund_evaluation(self, doc) -> List[EvaluativeTriple]:
        """
        Extract: "Hanging with loved ones is amazing"
        Pattern: VERB-ing + with/of + NOUN + is/are + ADJ
        """
        triples = []
        
        for token in doc:
            # Look for gerunds (VBG) as subjects OR in the sentence
            if token.tag_ == "VBG":
                # Check if it's a subject of a copula construction
                # Pattern 1: "Hanging ... is amazing" (gerund is nsubj)
                if token.dep_ in ("nsubj", "csubj"):
                    head = token.head
                elif token.head.pos_ == "AUX" or token.head.lemma_ == "be":
                    # Pattern 2: "Hanging ... is ..." (gerund's head is copula)
                    head = token.head
                else:
                    continue
                
                # Find the copula (is/are)
                copula = None
                if head.pos_ == "AUX" or head.lemma_ == "be":
                    copula = head
                else:
                    for child in head.children:
                        if child.dep_ == "cop":
                            copula = child
                            break
                
                if copula:
                    # Find the adjective (predicate)
                    adj = None
                    # Check if head itself is an adjective
                    if head.pos_ == "ADJ":
                        adj = head
                    else:
                        # Look for adjective in children
                        for child in head.children:
                            if child.pos_ == "ADJ" and child.dep_ in ("acomp", "attr"):
                                adj = child
                                break
                    
                    if adj:
                        # Extract the full gerund phrase
                        gerund_phrase = self._get_gerund_phrase(token)
                        
                        # Determine sentiment
                        sentiment = self._get_sentiment(adj.text.lower())
                        
                        if sentiment:
                            triple = EvaluativeTriple(
                                subject=gerund_phrase,
                                predicate="values" if sentiment == "positive" else "dislikes",
                                sentiment=sentiment,
                                confidence=0.85
                            )
                            triples.append(triple)
                            logger.debug(f"   Gerund pattern: '{gerund_phrase}' is {adj.text} → {sentiment}")
        
        return triples
    
    def _extract_noun_evaluation(self, doc) -> List[EvaluativeTriple]:
        """
        Extract: "Music is great"
        Pattern: NOUN + is/are + ADJ
        """
        triples = []
        
        for token in doc:
            # Look for nouns as subjects
            if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("nsubj", "nsubjpass"):
                # Find the copula
                copula = None
                for child in token.head.children:
                    if child.dep_ == "cop":
                        copula = child
                        break
                
                if copula:
                    # Find the adjective
                    adj = None
                    for child in token.head.children:
                        if child.pos_ == "ADJ":
                            adj = child
                            break
                    
                    if adj:
                        # Extract the noun phrase
                        noun_phrase = self._get_noun_phrase(token)
                        
                        # Determine sentiment
                        sentiment = self._get_sentiment(adj.text.lower())
                        
                        if sentiment:
                            triple = EvaluativeTriple(
                                subject=noun_phrase,
                                predicate="values" if sentiment == "positive" else "dislikes",
                                sentiment=sentiment,
                                confidence=0.80
                            )
                            triples.append(triple)
                            logger.debug(f"   Noun pattern: '{noun_phrase}' is {adj.text} → {sentiment}")
        
        return triples
    
    def _extract_emotion_verb(self, doc) -> List[EvaluativeTriple]:
        """
        Extract: "I love music", "I hate traffic"
        Pattern: SUBJ + VERB + OBJ
        """
        triples = []
        
        for token in doc:
            verb_lemma = token.lemma_.lower()
            
            # Check if it's an emotion verb
            if verb_lemma in self.POSITIVE_VERBS or verb_lemma in self.NEGATIVE_VERBS:
                # Find the object
                obj = None
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj"):
                        obj = child
                        break
                
                if obj:
                    # Extract the object phrase
                    obj_phrase = self._get_noun_phrase(obj)
                    
                    # Determine sentiment
                    sentiment = "positive" if verb_lemma in self.POSITIVE_VERBS else "negative"
                    
                    triple = EvaluativeTriple(
                        subject=obj_phrase,
                        predicate="values" if sentiment == "positive" else "dislikes",
                        sentiment=sentiment,
                        confidence=0.90
                    )
                    triples.append(triple)
                    logger.debug(f"   Emotion verb pattern: {verb_lemma} '{obj_phrase}' → {sentiment}")
        
        return triples
    
    def _get_gerund_phrase(self, token) -> str:
        """Extract the full gerund phrase including prepositional phrases."""
        words = [token.text]
        
        # Add children (prepositions, objects)
        for child in token.children:
            if child.dep_ in ("prep", "pobj", "dobj", "compound", "amod"):
                # Recursively get the subtree
                words.extend([t.text for t in child.subtree])
        
        return " ".join(words)
    
    def _get_noun_phrase(self, token) -> str:
        """Extract the noun phrase including modifiers."""
        # Get the noun chunk if available
        if token.doc.noun_chunks:
            for chunk in token.doc.noun_chunks:
                if token in chunk:
                    return chunk.text
        
        # Fallback: just the token and its compounds
        words = []
        for child in token.children:
            if child.dep_ in ("compound", "amod"):
                words.append(child.text)
        words.append(token.text)
        
        return " ".join(words)
    
    def _get_sentiment(self, adj: str) -> Optional[str]:
        """Determine sentiment from adjective."""
        if adj in self.POSITIVE_ADJECTIVES:
            return "positive"
        elif adj in self.NEGATIVE_ADJECTIVES:
            return "negative"
        return None

