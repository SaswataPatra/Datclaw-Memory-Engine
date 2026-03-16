"""
Opinion Extractor

Extracts beliefs, stances, and agreements from opinion sentences.

Examples:
    "I agree with Mel" → aligns_with(User, Mel)
    "I think family is important" → believes(User, family is important)
    "I support renewable energy" → supports(User, renewable energy)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .relation_extractor import ExtractedRelation

logger = logging.getLogger(__name__)


@dataclass
class OpinionTriple:
    """An opinion extracted from text."""
    subject: str  # The person holding the opinion
    predicate: str  # believes, aligns_with, supports, opposes
    object: str  # The target of the opinion
    confidence: float


class OpinionExtractor:
    """
    Extracts opinion relations (beliefs, agreements, stances) from sentences.
    
    Patterns:
    1. Agreement: "I agree/disagree with X"
    2. Belief: "I think/believe X"
    3. Stance: "I support/oppose X"
    """
    
    # Agreement verbs
    AGREEMENT_VERBS = {"agree", "concur", "align"}
    DISAGREEMENT_VERBS = {"disagree", "differ", "oppose"}
    
    # Belief verbs
    BELIEF_VERBS = {"think", "believe", "feel", "consider", "reckon", "suppose"}
    
    # Stance verbs
    SUPPORT_VERBS = {"support", "endorse", "advocate", "champion", "back"}
    OPPOSE_VERBS = {"oppose", "reject", "resist", "fight"}
    
    def __init__(self, nlp, entity_resolver, config: Dict = None):
        """
        Initialize the opinion extractor.
        
        Args:
            nlp: spaCy language model
            entity_resolver: Entity resolver for canonicalizing entities
            config: Configuration dict
        """
        self.nlp = nlp
        self.entity_resolver = entity_resolver
        self.config = config or {}
        
        logger.info("✅ OpinionExtractor initialized")
    
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
        Extract opinion relations from text.
        
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
        logger.info(f"💭 OpinionExtractor: Extracting from '{text[:50]}...'")
        
        # Parse text if doc not provided
        if doc is None:
            doc = self.nlp(resolved_text or text)
        
        triples = []
        
        # Pattern 1: Agreement/Disagreement (transitive: "I agree with X")
        triples.extend(self._extract_agreement(doc))
        
        # Pattern 2: Intransitive opinion verbs ("I agree", "I support")
        triples.extend(self._extract_intransitive_opinion(doc, resolved_entities, metadata))
        
        # Pattern 3: Belief
        triples.extend(self._extract_belief(doc))
        
        # Pattern 4: Stance (support/oppose)
        triples.extend(self._extract_stance(doc))
        
        if not triples:
            logger.info("   → No opinion patterns found")
            return []
        
        logger.info(f"   → Found {len(triples)} opinion triples")
        
        # Convert triples to ExtractedRelation objects
        relations = []
        for triple in triples:
            try:
                # Resolve entities
                subject_entity = await entity_resolver.resolve(
                    triple.subject, user_id, entity_type="person"
                )
                
                object_entity = await entity_resolver.resolve(
                    triple.object, user_id, entity_type=None
                )
                
                # Create relation
                relation = ExtractedRelation(
                    subject_entity_id=subject_entity.entity_id,
                    subject_text=triple.subject,
                    object_entity_id=object_entity.entity_id,
                    object_text=triple.object,
                    relation=triple.predicate,
                    category="cognitive",
                    confidence=triple.confidence,
                    context=text,
                    source="opinion_extractor",
                    memory_id=memory_id,
                    metadata={
                        **metadata,
                        "pattern": "opinion"
                    }
                )
                
                relations.append(relation)
                logger.info(f"   ✅ {triple.subject} --[{triple.predicate}]--> {triple.object} (conf={triple.confidence:.2f})")
            
            except Exception as e:
                logger.warning(f"Failed to create opinion relation: {e}")
                continue
        
        logger.info(f"💭 OpinionExtractor: Extracted {len(relations)} relations")
        return relations
    
    def _extract_intransitive_opinion(self, doc, resolved_entities: Dict, metadata: Dict) -> List[OpinionTriple]:
        """
        Extract intransitive opinion verbs: "I agree", "I support"
        
        For sentences like "I 100% agree, Mel", we extract:
        - The opinion verb (agree)
        - The subject (I)
        - The implicit object (look for nearby entities or vocatives)
        
        Pattern: SUBJ + OPINION_VERB (no explicit object/prep phrase)
        """
        triples = []
        
        # Combined set of all opinion verbs
        ALL_OPINION_VERBS = (
            self.AGREEMENT_VERBS | self.DISAGREEMENT_VERBS | 
            self.SUPPORT_VERBS | self.OPPOSE_VERBS
        )
        
        for token in doc:
            verb_lemma = token.lemma_.lower()
            
            if verb_lemma in ALL_OPINION_VERBS:
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                        break
                
                if not subject:
                    continue
                
                # Check if there's already an explicit object or prep phrase
                has_explicit_object = any(
                    child.dep_ in ("dobj", "pobj", "prep", "ccomp") 
                    for child in token.children
                )
                
                if has_explicit_object:
                    # Already handled by transitive patterns
                    continue
                
                # This is an intransitive use - find implicit object
                implicit_object = self._find_implicit_object(doc, token, resolved_entities, metadata)
                
                if implicit_object:
                    # Determine predicate based on verb type
                    if verb_lemma in self.AGREEMENT_VERBS:
                        predicate = "aligns_with"
                    elif verb_lemma in self.DISAGREEMENT_VERBS:
                        predicate = "disagrees_with"
                    elif verb_lemma in self.SUPPORT_VERBS:
                        predicate = "supports"
                    elif verb_lemma in self.OPPOSE_VERBS:
                        predicate = "opposes"
                    else:
                        predicate = "aligns_with"  # Default
                    
                    triple = OpinionTriple(
                        subject=subject,
                        predicate=predicate,
                        object=implicit_object,
                        confidence=0.75  # Lower confidence for implicit objects
                    )
                    triples.append(triple)
                    logger.debug(f"   Intransitive opinion: {subject} {verb_lemma} (implicit: {implicit_object}) → {predicate}")
        
        return triples
    
    def _find_implicit_object(self, doc, verb_token, resolved_entities: Dict, metadata: Dict) -> Optional[str]:
        """
        Find the implicit object for an intransitive opinion verb.
        
        Strategies:
        1. Look for vocative (direct address): "I agree, Mel" → Mel
        2. Look for nearby entities in the same sentence
        3. Look for previous speaker in dialogue context
        4. Fall back to generic "previous statement"
        
        Args:
            doc: spaCy Doc
            verb_token: The opinion verb token
            resolved_entities: Dict of resolved entities
            metadata: Metadata dict (may contain dialogue context)
        
        Returns:
            Implicit object text, or None if not found
        """
        sent = verb_token.sent
        
        # Strategy 1: Vocative (direct address)
        # Look for tokens with dep="vocative" or npadvmod near the verb
        for token in sent:
            if token.dep_ == "vocative" or (token.dep_ == "npadvmod" and token.pos_ == "PROPN"):
                logger.debug(f"      Found vocative: {token.text}")
                return token.text
        
        # Strategy 2: Nearby named entities (within same sentence)
        # Prioritize entities that appear AFTER the verb (e.g., "I agree, Mel" → Mel comes after "agree")
        nearby_entities = []
        for ent_match in resolved_entities.values():
            ent_name = ent_match.get("name", "")
            # Check if entity appears in the same sentence and after the verb
            if ent_name and ent_name in sent.text:
                # Find position relative to verb
                for token in sent:
                    if ent_name.lower() in token.text.lower() and token.i > verb_token.i:
                        nearby_entities.append(ent_name)
                        break
        
        if nearby_entities:
            logger.debug(f"      Found nearby entity: {nearby_entities[0]}")
            return nearby_entities[0]
        
        # Strategy 3: Previous speaker in dialogue (from metadata)
        # If this is a dialogue turn, the implicit object might be the previous speaker
        if "speaker" in metadata and "turn_index" in metadata and metadata.get("turn_index", 0) > 0:
            # Look for other speakers in resolved_entities
            current_speaker = metadata.get("speaker")
            other_speakers = [
                ent_match.get("name") for ent_match in resolved_entities.values()
                if ent_match.get("name") != current_speaker and ent_match.get("entity", {}).type == "person"
            ]
            if other_speakers:
                logger.debug(f"      Found previous speaker: {other_speakers[0]}")
                return other_speakers[0]
        
        # Strategy 4: Fall back to "previous statement" (generic implicit object)
        # This is a low-confidence fallback
        logger.debug("      No explicit implicit object found, using 'previous statement'")
        return None  # Return None to skip this relation (too generic)
    
    def _extract_agreement(self, doc) -> List[OpinionTriple]:
        """
        Extract: "I agree with Mel"
        Pattern: SUBJ + agree/disagree + with + OBJ
        """
        triples = []
        
        for token in doc:
            verb_lemma = token.lemma_.lower()
            
            if verb_lemma in self.AGREEMENT_VERBS or verb_lemma in self.DISAGREEMENT_VERBS:
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                        break
                
                # Find object (after "with")
                obj = None
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "with":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = grandchild.text
                                break
                
                if subject and obj:
                    predicate = "aligns_with" if verb_lemma in self.AGREEMENT_VERBS else "disagrees_with"
                    
                    triple = OpinionTriple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.90
                    )
                    triples.append(triple)
                    logger.debug(f"   Agreement pattern: {subject} {verb_lemma} with {obj} → {predicate}")
        
        return triples
    
    def _extract_belief(self, doc) -> List[OpinionTriple]:
        """
        Extract: "I think family is important"
        Pattern: SUBJ + think/believe + CLAUSE
        """
        triples = []
        
        for token in doc:
            verb_lemma = token.lemma_.lower()
            
            if verb_lemma in self.BELIEF_VERBS:
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                        break
                
                # Find the complement clause (ccomp)
                clause = None
                for child in token.children:
                    if child.dep_ == "ccomp":
                        # Extract the full clause
                        clause = " ".join([t.text for t in child.subtree])
                        break
                
                if subject and clause:
                    triple = OpinionTriple(
                        subject=subject,
                        predicate="believes",
                        object=clause,
                        confidence=0.85
                    )
                    triples.append(triple)
                    logger.debug(f"   Belief pattern: {subject} {verb_lemma} '{clause}' → believes")
        
        return triples
    
    def _extract_stance(self, doc) -> List[OpinionTriple]:
        """
        Extract: "I support renewable energy"
        Pattern: SUBJ + support/oppose + OBJ
        """
        triples = []
        
        for token in doc:
            verb_lemma = token.lemma_.lower()
            
            if verb_lemma in self.SUPPORT_VERBS or verb_lemma in self.OPPOSE_VERBS:
                # Find subject
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child.text
                        break
                
                # Find object
                obj = None
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj"):
                        # Get the full noun phrase
                        obj = " ".join([t.text for t in child.subtree])
                        break
                
                if subject and obj:
                    predicate = "supports" if verb_lemma in self.SUPPORT_VERBS else "opposes"
                    
                    triple = OpinionTriple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.88
                    )
                    triples.append(triple)
                    logger.debug(f"   Stance pattern: {subject} {verb_lemma} {obj} → {predicate}")
        
        return triples

