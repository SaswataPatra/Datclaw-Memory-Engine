"""
DAPPY Relation Extractor

Integrates EntityExtractor + EntityResolver + RelationClassifier
to extract structured relations from text.

Phase 1C: Entity-pair based extraction
Phase 1D: Dependency-based extraction (NEW)

This is the main entry point for relation extraction in the memory pipeline.

Phase 1D Implementation
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_resolver import EntityResolver
from .relation_classifier import RelationClassifier, RelationResult
from .relation_training_collector import RelationTrainingCollector
from .schemas import CandidateEdge, SupportingMention

# Phase 1D: Dependency-based extraction
from .coref_resolver import CorefResolver
from .dependency_extractor import DependencyExtractor
from .relation_normalizer import RelationNormalizer

# Phase 1F: Dialogue processing
from .dialogue_processor import DialogueProcessor

# Phase 1F.3: Relation validation
from .relation_validator import RelationValidator

logger = logging.getLogger(__name__)


# ========== NP TYPE INFERENCE (Phase 1E) ==========
# Lexicon-based type inference for noun phrases

NP_TYPE_LEXICONS = {
    "ROLE": {
        "manager", "owner", "director", "ceo", "cto", "cfo", "president",
        "engineer", "developer", "designer", "architect", "consultant",
        "doctor", "nurse", "teacher", "professor", "instructor",
        "employee", "worker", "staff", "member", "volunteer",
        "artist", "musician", "writer", "author", "creator",
        "chef", "cook", "waiter", "bartender", "server"
    },
    "ORG_LIKE": {
        "shop", "store", "company", "firm", "corporation", "business",
        "restaurant", "cafe", "bar", "pub", "diner", "bakery",
        "university", "school", "college", "academy", "institute",
        "bank", "hospital", "clinic", "pharmacy", "library",
        "office", "factory", "warehouse", "lab", "laboratory",
        "agency", "department", "division", "team", "group"
    },
    "FACILITY": {
        "building", "mall", "center", "complex", "plaza",
        "museum", "gallery", "theater", "cinema", "stadium",
        "studio", "workshop", "garage", "hangar", "terminal",
        "park", "garden", "zoo", "aquarium", "resort"
    },
    "ARTWORK": {
        "painting", "sculpture", "drawing", "sketch", "artwork",
        "piece", "creation", "work", "masterpiece", "portrait",
        "landscape", "mural", "installation", "print", "photograph"
    },
    "MATERIAL": {
        "oil", "acrylic", "watercolor", "wood", "metal", "steel",
        "canvas", "paper", "clay", "stone", "marble", "bronze",
        "glass", "plastic", "fabric", "leather", "silk"
    },
    "ATTRIBUTE": set(),  # Handled by pattern matching
}


def infer_np_type(span_text: str, doc, entity_type: Optional[str]) -> str:
    """
    Infer coarse type for a noun phrase.
    
    Priority:
    1. If spaCy NER assigned a type → use it (mapped to our taxonomy)
    2. Check head lemma against lexicons
    3. Check for patterns (e.g., "X year old")
    4. POS-based heuristics
    5. Default to None
    
    Args:
        span_text: The text of the noun phrase
        doc: spaCy Doc object
        entity_type: Existing entity type from NER (if any)
    
    Returns:
        Inferred type (ROLE, ORG_LIKE, FACILITY, ARTWORK, MATERIAL, ATTRIBUTE, or original NER type)
    """
    # 1. Use existing NER type if available (map to our taxonomy)
    if entity_type:
        # Map spaCy NER types to our taxonomy
        ner_type_map = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "FAC": "FACILITY",
            "PRODUCT": "product",
            "EVENT": "event",
            "DATE": "temporal",
            "TIME": "temporal",
            "MONEY": "attribute",
            "PERCENT": "attribute",
            "QUANTITY": "attribute"
        }
        mapped_type = ner_type_map.get(entity_type.upper(), entity_type)
        if mapped_type != entity_type:
            return mapped_type
        return entity_type
    
    # 2. Extract head lemma from span text
    # For multi-word phrases, the last word is usually the head
    span_lower = span_text.lower().strip()
    words = span_text.split()
    
    if not words:
        return None
    
    # Get the head word (last word in the phrase)
    head_word = words[-1].lower()
    
    # Try to find the head word in the doc to get its lemma
    head_lemma = None
    head_pos = None
    
    for token in doc:
        if token.text.lower() == head_word:
            head_lemma = token.lemma_.lower()
            head_pos = token.pos_
            break
    
    # If we couldn't find it in doc, use the head word as lemma (and strip plural 's')
    if not head_lemma:
        head_lemma = head_word.rstrip('s')
    
    if not head_lemma:
        return None
    
    # 3. Check lexicons
    for np_type, lexicon in NP_TYPE_LEXICONS.items():
        if head_lemma in lexicon:
            logger.debug(f"   NP type inference: '{span_text}' → {np_type} (lexicon match on '{head_lemma}')")
            return np_type
    
    # 4. Pattern matching
    # "X year old" → ATTRIBUTE
    if "year old" in span_lower or "years old" in span_lower:
        logger.debug(f"   NP type inference: '{span_text}' → ATTRIBUTE (age pattern)")
        return "ATTRIBUTE"
    
    # 5. POS-based heuristics
    if head_pos == "ADJ":
        logger.debug(f"   NP type inference: '{span_text}' → ATTRIBUTE (adjective)")
        return "ATTRIBUTE"
    
    # 6. Default
    return None


@dataclass
class ExtractedRelation:
    """A fully extracted relation with resolved entities."""
    subject_entity_id: str
    subject_text: str
    object_entity_id: str
    object_text: str
    relation: str
    category: str
    confidence: float
    context: str
    source: str = "pipeline"
    memory_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_candidate_edge(
        self,
        user_id: str,
        ego_score: float = 0.5
    ) -> CandidateEdge:
        """Convert to CandidateEdge for storage."""
        return CandidateEdge(
            user_id=user_id,
            subject_entity_id=self.subject_entity_id,
            subject_span={"text": self.subject_text},
            predicate=self.relation,
            object_entity_id=self.object_entity_id,
            object_span={"text": self.object_text},
            supporting_mentions=[
                SupportingMention(
                    mem_id=self.memory_id or "",
                    srl_conf=self.confidence,
                    ego=ego_score
                )
            ],
            aggregated_features={
                "category": self.category,
                "source": self.source,
                **self.metadata
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_entity_id": self.subject_entity_id,
            "subject_text": self.subject_text,
            "object_entity_id": self.object_entity_id,
            "object_text": self.object_text,
            "relation": self.relation,
            "category": self.category,
            "confidence": self.confidence,
            "context": self.context,
            "source": self.source,
            "memory_id": self.memory_id,
            "metadata": self.metadata
        }


class RelationExtractor:
    """
    Main entry point for relation extraction.
    
    NEW (Phase 1D): Dependency-based extraction with 92-95% coverage
    1. CorefResolver - Resolve pronouns before extraction
    2. DependencyExtractor - Extract relations using 10 dependency patterns
    3. RelationNormalizer - Normalize predicates to canonical forms
    4. EntityResolver - Resolve entities to canonical IDs
    5. Confidence composition - Weighted combination of 4 factors
    6. Fallback to zero-shot/LLM for edge cases
    
    LEGACY (Phase 1C): Entity-pair based extraction (fallback)
    1. EntityExtractor - Extract entities using spaCy NER
    2. EntityResolver - Resolve entities to canonical forms
    3. RelationClassifier - Classify relations between entities
    4. RelationTrainingCollector - Collect training data
    
    Usage:
        extractor = RelationExtractor(db, config)
        relations = await extractor.extract(text, user_id)
        for rel in relations:
            candidate_edge = rel.to_candidate_edge(user_id, ego_score)
    """
    
    def __init__(
        self,
        db,
        config: Optional[Dict[str, Any]] = None,
        embedding_service = None,
        llm_provider = None,
        collect_training_data: bool = True
    ):
        """
        Initialize relation extractor.
        
        Args:
            db: ArangoDB database connection
            config: Configuration dict
            embedding_service: Optional embedding service for entity resolution
            collect_training_data: Whether to collect training data
        """
        self.config = config or {}
        self.embedding_service = embedding_service
        self.llm_provider = llm_provider
        
        # Check if dependency extraction is enabled
        self.use_dependency_extraction = self.config.get('dependency_extraction', {}).get('enabled', True)
        
        # Phase 1D: Dependency-based components (NEW)
        if self.use_dependency_extraction:
            try:
                self.coref_resolver = CorefResolver(
                    provider=self.config.get('coref', {}).get('provider', 'simple'),
                    model=self.config.get('coref', {}).get('model', 'en_core_web_md'),
                    config=self.config.get('coref', {})
                )
                
                self.dependency_extractor = DependencyExtractor(
                    nlp=self.coref_resolver.nlp,  # Reuse spaCy model
                    config=self.config.get('dependency_extraction', {})
                )
                
                self.relation_normalizer = RelationNormalizer(
                    embedding_service=embedding_service,
                    config=self.config.get('dependency_extraction', {})
                )
                
                # Phase 1F: Dialogue processor
                self.dialogue_processor = DialogueProcessor(
                    config=self.config.get('dialogue_processing', {})
                )
                
                # Phase 1F.3: Relation validator
                self.relation_validator = RelationValidator()
                
                logger.info("✅ Dependency-based extraction enabled")
                logger.info("✅ Dialogue processor initialized")
                logger.info("✅ Relation validator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize dependency extraction: {e}, falling back to entity-pair method")
                self.use_dependency_extraction = False
        
        # Phase 1C: Entity-pair components (LEGACY/FALLBACK)
        # NOTE: These MUST be initialized before Phase 1G extractors!
        self.entity_extractor = EntityExtractor(
            config=self.config.get('entity_extraction', {})
        )
        
        self.entity_resolver = EntityResolver(
            db=db,
            config=self.config,
            embedding_service=embedding_service
        )
        
        self.relation_classifier = RelationClassifier(
            config=self.config
        )
        
        # Phase 1G: Sentence intent classification and routing
        # NOTE: This requires entity_resolver to be initialized first!
        if self.use_dependency_extraction:
            try:
                from .sentence_intent_classifier import SentenceIntentClassifier
                from .intent_router import IntentRouter
                from .evaluative_extractor import EvaluativeExtractor
                from .opinion_extractor import OpinionExtractor
                from .speech_act_extractor import SpeechActExtractor
                
                self.intent_classifier = SentenceIntentClassifier(
                    config=self.config,
                    llm_client=self.llm_provider  # LLMProvider from ChatbotService (fallback when HF times out)
                )
                
                # Initialize specialized extractors
                self.evaluative_extractor = EvaluativeExtractor(
                    nlp=self.coref_resolver.nlp,
                    entity_resolver=self.entity_resolver,
                    config=self.config.get('sentence_intent', {}).get('extractors', {}).get('evaluative', {})
                )
                
                self.opinion_extractor = OpinionExtractor(
                    nlp=self.coref_resolver.nlp,
                    entity_resolver=self.entity_resolver,
                    config=self.config.get('sentence_intent', {}).get('extractors', {}).get('opinion', {})
                )
                
                self.speech_act_extractor = SpeechActExtractor(
                    nlp=self.coref_resolver.nlp,
                    entity_resolver=self.entity_resolver,
                    config=self.config.get('sentence_intent', {}).get('extractors', {}).get('speech_act', {})
                )
                
                self.intent_router = IntentRouter(
                    dependency_extractor=self.dependency_extractor,
                    evaluative_extractor=self.evaluative_extractor,
                    opinion_extractor=self.opinion_extractor,
                    speech_act_extractor=self.speech_act_extractor,
                    directive_extractor=None,  # TODO: Implement DirectiveExtractor
                    config=self.config
                )
                
                logger.info("✅ Intent classifier initialized")
                logger.info("✅ Evaluative extractor initialized")
                logger.info("✅ Opinion extractor initialized")
                logger.info("✅ Speech act extractor initialized")
                logger.info("✅ Intent router initialized")
            except Exception as e:
                logger.error(f"Failed to initialize intent system: {e}", exc_info=True)
                # Don't disable dependency extraction, just the intent routing
                self.intent_classifier = None
                self.intent_router = None
        
        # Training data collector
        self.collect_training_data = collect_training_data
        if collect_training_data:
            self.training_collector = RelationTrainingCollector(
                db_path=self.config.get('relation_training', {}).get(
                    'db_path', 'data/relation_training.db'
                )
            )
        else:
            self.training_collector = None
        
        logger.info(f"✅ RelationExtractor initialized (dependency_extraction={self.use_dependency_extraction})")
    
    async def extract(
        self,
        text: str,
        user_id: str,
        memory_id: str = None,
        ego_score: float = 0.5,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> List[ExtractedRelation]:
        """
        Extract relations from text.
        
        NEW (Phase 1D): Uses dependency-based extraction for 92-95% coverage
        FALLBACK (Phase 1C): Uses entity-pair method if dependency extraction fails
        
        Args:
            text: Input text to extract relations from
            user_id: User ID for entity resolution
            memory_id: Optional memory ID for linking
            ego_score: Ego score of the memory
            session_id: Optional session ID for context
            metadata: Optional metadata (e.g. triggers)
        
        Returns:
            List of ExtractedRelation objects
        """
        if not text or not text.strip():
            return []
        
        # Merge session_id into metadata for tracking
        metadata = metadata or {}
        if session_id:
            metadata["session_id"] = session_id
        
        logger.debug(f"Extracting relations from: '{text[:50]}...'")
        
        # Phase 1F: Pre-process dialogue format (if detected)
        processed_dialogue = None
        if self.use_dependency_extraction and hasattr(self, 'dialogue_processor'):
            is_dialogue, format_type = self.dialogue_processor.is_dialogue_format(text)
            if is_dialogue:
                logger.info(f"📋 Dialogue detected (format={format_type}), pre-processing...")
                processed_dialogue = self.dialogue_processor.process(text)
                
                # Store dialogue metadata
                metadata["dialogue_format"] = format_type
                metadata["dialogue_turns"] = len(processed_dialogue.turns)
                metadata["dialogue_dates"] = [d.isoformat() for d in processed_dialogue.dates]
                
                # For now, extract from each turn separately
                # TODO: Cross-turn relation extraction (e.g., "I" in turn 2 refers to speaker from turn 1)
                all_relations = []
                for turn in processed_dialogue.turns:
                    turn_metadata = {
                        **metadata,
                        "speaker": turn.speaker,
                        "turn_index": turn.turn_index,
                        "turn_date": turn.date.isoformat() if turn.date else None
                    }
                    
                    logger.info(f"💬 Extracting from turn {turn.turn_index} ({turn.speaker}): '{turn.text[:50]}...'")
                    
                    # Convert to speaker-attributed text for better entity extraction
                    # "I agree" → "Caroline agrees"
                    speaker_text = self.dialogue_processor.to_speaker_attributed_text(turn)
                    logger.info(f"   → Speaker-attributed: '{speaker_text[:100]}...'")
                    
                    turn_relations = await self._extract_dependency_based(
                        speaker_text, user_id, memory_id, ego_score, turn_metadata
                    )
                    
                    # Add speaker attribution to entities
                    for rel in turn_relations:
                        rel.metadata["speaker"] = turn.speaker
                        if turn.date:
                            rel.metadata["turn_date"] = turn.date.isoformat()
                    
                    all_relations.extend(turn_relations)
                
                # Extract aliases (nicknames) and create alias relations
                aliases = self.dialogue_processor.extract_aliases(processed_dialogue)
                if aliases:
                    logger.info(f"🏷️  Detected {len(aliases)} alias mappings")
                    for canonical_name, alias_list in aliases.items():
                        for alias in alias_list:
                            # Create an "also_known_as" relation
                            # We'll need to resolve both entities first
                            try:
                                canonical_entity = await self.entity_resolver.resolve(
                                    canonical_name, user_id, entity_type="person"
                                )
                                alias_entity = await self.entity_resolver.resolve(
                                    alias, user_id, entity_type="person"
                                )
                                
                                # Create alias relation
                                alias_relation = ExtractedRelation(
                                    subject_entity_id=canonical_entity.entity_id,
                                    subject_text=canonical_name,
                                    object_entity_id=alias_entity.entity_id,
                                    object_text=alias,
                                    relation="also_known_as",
                                    category="identity",
                                    confidence=0.85,  # High confidence for detected aliases
                                    context=f"{canonical_name} is also known as {alias}",
                                    source="dialogue_alias_detection",
                                    metadata={}
                                )
                                all_relations.append(alias_relation)
                                logger.info(f"   → Created alias: {canonical_name} --[also_known_as]--> {alias}")
                            except Exception as e:
                                logger.warning(f"Failed to create alias relation {canonical_name} → {alias}: {e}")
                
                if all_relations:
                    logger.info(f"✅ Dialogue extraction: {len(all_relations)} relations from {len(processed_dialogue.turns)} turns")
                    
                    # Log final summary of ALL relations across ALL turns
                    logger.info("\n" + "="*80)
                    logger.info(f"📋 DIALOGUE FINAL SUMMARY ({len(all_relations)} total relations)")
                    logger.info("="*80)
                    for idx, rel in enumerate(all_relations, 1):
                        speaker = rel.metadata.get("speaker", "unknown")
                        logger.info(f" {idx}. [{speaker:12}] {rel.subject_text:20} --[{rel.relation:15}]--> {rel.object_text:20} (conf={rel.confidence:.3f})")
                    logger.info("="*80 + "\n")
                    
                    return all_relations
                else:
                    logger.debug("No relations from dialogue extraction, falling back to standard extraction")
        
        # Try dependency-based extraction first (Phase 1D)
        if self.use_dependency_extraction:
            try:
                relations = await self._extract_dependency_based(
                    text, user_id, memory_id, ego_score, metadata
                )
                
                if relations:
                    logger.info(f"✅ Dependency extraction: {len(relations)} relations")
                    return relations
                else:
                    logger.debug("No relations from dependency extraction, falling back to entity-pair")
            except Exception as e:
                logger.warning(f"Dependency extraction failed: {e}, falling back to entity-pair method")
        
        # Fallback to entity-pair method (Phase 1C)
        return await self._extract_entity_pairs(text, user_id, memory_id, ego_score, metadata)
    
    async def _extract_dependency_based(
        self,
        text: str,
        user_id: str,
        memory_id: str,
        ego_score: float,
        metadata: Dict[str, Any]
    ) -> List[ExtractedRelation]:
        """
        NEW: Dependency-based extraction (Phase 1D + 1G).
        
        Flow:
        0. Intent classification (Phase 1G) - NEW!
        1. Coref resolution
        2. Dependency parsing
        3. Entity resolution for each triple
        4. Relation normalization
        5. Confidence composition
        6. Fallback to zero-shot/LLM if needed
        """
        # Step 0: Intent classification (Phase 1G)
        if self.intent_classifier and self.intent_router:
            logger.info(f"🎯 Step 0: Classify sentence intent")
            intent_result = await self.intent_classifier.classify(text, user_id)
            logger.info(f"   → Intent: {intent_result.intent} (conf={intent_result.confidence:.2f}, method={intent_result.method})")
        else:
            # Intent system not available, treat as 'fact'
            logger.debug("Intent system not initialized, treating as 'fact'")
            from .sentence_intent_classifier import IntentResult
            intent_result = IntentResult(intent="fact", confidence=1.0, method="default")
        
        # Step 1: Coref resolution (needed by all extractors)
        logger.debug(f"🔗 Step 1: Coreference resolution")
        resolved_text, coref_clusters = self.coref_resolver.resolve(text)
        logger.debug(f"   → Resolved {len(coref_clusters)} coref clusters")
        
        # Parse text (needed by all extractors)
        doc = self.coref_resolver.nlp(resolved_text)
        
        # If intent is NOT 'fact', route to specialized extractor
        if intent_result.intent != "fact" and self.intent_router:
            logger.info(f"   → Routing to '{intent_result.intent}' extractor")
            
            try:
                # Route through IntentRouter
                routed_relations = await self.intent_router.route_and_extract(
                    text=text,
                    intent_result=intent_result,
                    user_id=user_id,
                    memory_id=memory_id,
                    ego_score=ego_score,
                    metadata=metadata,
                    resolved_text=resolved_text,
                    doc=doc,
                    resolved_entities={},  # Will be populated by extractors
                    entity_resolver=self.entity_resolver,
                    relation_normalizer=self.relation_normalizer,
                    relation_classifier=self.relation_classifier,
                    relation_validator=self.relation_validator
                )
                
                if routed_relations:
                    logger.info(f"✅ Intent-specific extraction: {len(routed_relations)} relations")
                    return routed_relations
                else:
                    logger.warning(f"   ⚠️  '{intent_result.intent}' extractor returned no relations, falling back to fact extraction")
            
            except Exception as e:
                logger.warning(f"   ⚠️  '{intent_result.intent}' extractor failed: {e}, falling back to fact extraction")
        
        # Continue with fact extraction (for 'fact' intent or fallback)
        
        # Step 2: Extract entities (NER + important noun phrases)
        logger.info(f"📍 Step 2: Extract entities (NER + noun phrases)")
        # doc already created above for intent routing
        
        # 2a. Extract NER entities (PERSON, ORG, LOC, etc.)
        extracted_entities = self.entity_extractor.extract(resolved_text)
        logger.info(f"   → Found {len(extracted_entities)} NER entities: {[e.text for e in extracted_entities]}")
        
        # 2a.1. If this is a dialogue turn, ensure the speaker is extracted as an entity
        speaker_name = metadata.get("speaker") if metadata else None
        if speaker_name:
            # Check if speaker is already in extracted entities
            speaker_entity_idx = None
            for idx, e in enumerate(extracted_entities):
                if e.text.lower() == speaker_name.lower():
                    speaker_entity_idx = idx
                    break
            
            if speaker_entity_idx is not None:
                # Speaker found but might have wrong type (e.g., "organization")
                # Force correct it to "person"
                existing_entity = extracted_entities[speaker_entity_idx]
                if existing_entity.type != "person":
                    logger.info(f"   → Correcting speaker '{speaker_name}' type: {existing_entity.type} → person")
                    extracted_entities[speaker_entity_idx] = ExtractedEntity(
                        text=speaker_name,
                        type="person",
                        start=existing_entity.start,
                        end=existing_entity.end,
                        confidence=0.95,  # High confidence for explicit speaker
                        original_type="PERSON"
                    )
            else:
                logger.info(f"   → Adding speaker '{speaker_name}' as entity (not found in NER)")
                # Create a synthetic entity for the speaker
                speaker_entity = ExtractedEntity(
                    text=speaker_name,
                    type="person",
                    start=0,
                    end=len(speaker_name),
                    confidence=0.95,  # High confidence for explicit speaker
                    original_type="PERSON"
                )
                extracted_entities.insert(0, speaker_entity)  # Prepend speaker
            
            # 2a.2. Detect first-person pronouns and add speaker as entity
            # This handles cases like "I agree" where "I" should resolve to the speaker
            pronoun_pattern = r'\b(I|me|my|mine|myself)\b'
            if re.search(pronoun_pattern, resolved_text, re.IGNORECASE):
                logger.info(f"   → Detected first-person pronouns, will map to speaker '{speaker_name}'")
                # The speaker is already added above, so pronouns will resolve to them
        
        # 2b. Special handling for copula patterns (e.g., "my name is John")
        # Extract proper nouns from copula constructions that NER might have missed
        copula_entities = self._extract_copula_entities(doc, resolved_text)
        if copula_entities:
            logger.info(f"   → Found {len(copula_entities)} entities from copula patterns: {[e.text for e in copula_entities]}")
            extracted_entities.extend(copula_entities)
        
        # 2c. Extract important noun phrases (for common nouns like "oil paintings", "manager")
        noun_phrases = self._extract_noun_phrases(doc)
        logger.info(f"   → Found {len(noun_phrases)} noun phrases: {noun_phrases}")
        
        # Combine NER entities and noun phrases
        all_entities = list(extracted_entities)
        for np_text in noun_phrases:
            # Infer type for noun phrase using lexicon
            inferred_type = infer_np_type(np_text, doc, None)
            
            if inferred_type:
                logger.debug(f"   NP type inferred: '{np_text}' → {inferred_type}")
            
            # Create a pseudo-entity for noun phrases
            all_entities.append(ExtractedEntity(
                text=np_text,
                type=inferred_type,  # Use inferred type (ROLE, ORG_LIKE, etc.)
                original_type="NOUN_PHRASE",  # Mark as noun phrase
                start=0,  # Position not important for noun phrases
                end=0,
                confidence=0.75  # Lower confidence than NER
            ))
        
        logger.info(f"   → Total {len(all_entities)} entities (NER + noun phrases)")
        
        if not all_entities:
            logger.debug("No entities found, falling back to entity-pair method")
            return []
        
        # Resolve all entities
        resolved_entities = {}
        for entity in all_entities:
            # Infer type for all entities (NER + noun phrases)
            # For noun phrases, entity.type might be None, so we need to infer
            inferred_type = infer_np_type(entity.text, doc, entity.type)
            final_type = inferred_type if inferred_type else entity.type
            
            # Update the original entity's type for later use
            if final_type and not entity.type:
                entity.type = final_type
            
            resolved = await self.entity_resolver.resolve(
                text=entity.text,
                user_id=user_id,
                context=resolved_text,
                entity_type=final_type
            )
            if resolved:
                # Update the resolved entity's type with inferred type
                if final_type and not resolved.type:
                    resolved.type = final_type
                
                resolved_entities[entity.text.lower()] = {
                    "entity": resolved,
                    "original": entity
                }
        
        logger.info(f"   → Resolved {len(resolved_entities)} entities")
        
        # Step 3: Dependency parsing to find connections
        # Constrain patterns based on intent (Phase 1G)
        intent = metadata.get("intent", "fact")
        enabled_patterns = None  # None = all patterns
        
        # Disable apposition pattern for non-fact intents to avoid "holds_title 100%" junk
        if intent != "fact":
            enabled_patterns = ["svo", "prepositional", "copula", "passive", "coordination", "nested", "relative"]
            logger.debug(f"   → Intent is '{intent}', disabling 'apposition' pattern")
        
        logger.info(f"🔍 Step 3: Dependency parsing to find entity connections (intent={intent})")
        dep_triples = self.dependency_extractor.extract_from_doc(doc, enabled_patterns=enabled_patterns)
        
        if not dep_triples:
            logger.debug("No dependency triples extracted")
            return []
        
        logger.info(f"   → Extracted {len(dep_triples)} dependency triples")
        
        # Log all raw triples for debugging
        logger.info(f"🔍 Raw triples from dependency parser:")
        for idx, triple in enumerate(dep_triples, 1):
            logger.info(f"   {idx}. {triple.subject_text} --[{triple.predicate_lemma}]--> {triple.object_text} (pattern={triple.pattern})")
        
        # Step 4: Match dependency triples to NER entities
        logger.info(f"🔗 Step 4: Matching triples to NER entities")
        
        # ========== PHASE 1: COLLECT ALL PREDICATE CANDIDATES (NO NORMALIZATION YET) ==========
        logger.info(f"📦 Phase 1: Collecting ALL predicate candidates from ALL spans")
        all_predicate_candidates = []
        processed_spans = set()  # Track processed spans to avoid duplicates
        
        for idx, triple in enumerate(dep_triples, 1):
            try:
                # Resolve relative pronouns before matching
                subj_text = self._resolve_relative_pronoun(triple.subject_text, doc)
                obj_text = self._resolve_relative_pronoun(triple.object_text, doc)
                
                # Find subject entity (pass metadata for pronoun resolution)
                subj_match = self._find_entity_match(subj_text, resolved_entities, doc, metadata)
                
                # IMPORTANT: If subject was a relative pronoun (who/which), find the ORIGINAL entity it refers to
                # Example: "who works" where "who" refers to "Angela" → use Angela as subject, not "woman"
                if triple.subject_text.lower() in ("who", "which", "that"):
                    # Find the antecedent (the entity this pronoun modifies)
                    logger.info(f"   🔍 Finding antecedent for relative pronoun '{triple.subject_text}'")
                    antecedent_match = self._find_relative_pronoun_antecedent(triple.subject_text, doc, resolved_entities)
                    if antecedent_match:
                        logger.info(f"   ✅ Using antecedent '{antecedent_match['original'].text}' instead of '{subj_match['original'].text}'")
                        subj_match = antecedent_match
                    else:
                        logger.info(f"   ⚠️  No antecedent found, using resolved pronoun '{subj_match['original'].text}'")
                
                if not subj_match:
                    logger.debug(f"   ⏭️  Triple {idx}: No subject match for '{subj_text}'")
                    continue
                
                # Check if we've already processed this span
                span_key = f"{subj_match['entity'].entity_id}:{obj_text.lower()}"
                if span_key in processed_spans:
                    logger.info(f"   ⏭️  Triple {idx}: Skipping duplicate span '{obj_text[:50]}...'")
                    continue
                
                processed_spans.add(span_key)
                
                # COLLECT predicate candidates from this span (NO normalization yet!)
                candidates = self._collect_predicate_candidates_from_span(
                    span_text=obj_text,
                    subject_match=subj_match,
                    predicate=triple.predicate_lemma,
                    pattern_conf=triple.confidence,
                    modality_score=triple.modality_score,
                    pattern=triple.pattern,
                    is_negated=triple.is_negated,
                    is_passive=triple.is_passive,
                    is_question=triple.is_question,
                    modality=triple.modality,
                    resolved_entities=resolved_entities,
                    resolved_text=resolved_text,
                    doc=doc,
                    user_id=user_id,
                    memory_id=memory_id,
                    metadata=metadata,
                    depth=0
                )
                
                if not candidates:
                    logger.debug(f"   ⏭️  Triple {idx}: No object entities found in '{obj_text}'")
                    continue
                
                # Add all candidates to the global list
                all_predicate_candidates.extend(candidates)
                
            except Exception as e:
                logger.warning(f"Failed to process triple {idx}: {e}", exc_info=True)
                continue
        
        if not all_predicate_candidates:
            logger.info("No predicate candidates collected")
            return []
        
        logger.info(f"📦 Collected {len(all_predicate_candidates)} predicate candidates from ALL spans")
        
        # ========== PHASE 2: ONE BATCH NORMALIZATION FOR ALL PREDICATES ==========
        logger.info(f"🤖 Phase 2: Batch normalizing ALL {len(all_predicate_candidates)} predicates in ONE call")
        
        # Extract just the normalization inputs
        normalization_inputs = []
        for candidate in all_predicate_candidates:
            normalization_inputs.append({
                "predicate_lemma": candidate["inferred_predicate"],
                "subject_type": candidate["subject_match"]["original"].type,
                "object_type": candidate["obj_match"]["original"].type,
                "context": candidate["resolved_text"],
                "user_id": candidate["user_id"],
                "memory_id": candidate["memory_id"],
                "session_id": candidate["metadata"].get("session_id")
            })
        
        # ONE batch normalization call for ALL predicates!
        normalized_results = await self.relation_normalizer.normalize_batch(normalization_inputs)
        
        # ========== PHASE 3: CREATE RELATIONS USING NORMALIZED RESULTS ==========
        logger.info(f"🔨 Phase 3: Creating relations with normalized predicates")
        relations = []
        
        for i, (canonical_relation, norm_conf) in enumerate(normalized_results):
            candidate = all_predicate_candidates[i]
            
            # Compose confidence
            final_confidence = self._compose_confidence(
                pattern_conf=candidate["pattern_conf"],
                norm_conf=norm_conf,
                modality_score=candidate["modality_score"],
                resolver_conf_subj=candidate["subject_match"]["original"].confidence,
                resolver_conf_obj=candidate["obj_match"]["original"].confidence,
                subject_text=candidate["subject_match"]["original"].text,
                object_text=candidate["obj_match"]["original"].text,
                relation=canonical_relation
            )
            
            # Phase 1F.3: Semantic validation
            subject_type = candidate["subject_match"]["entity"].type
            object_type = candidate["obj_match"]["entity"].type
            
            validation_result = self.relation_validator.validate(
                relation=canonical_relation,
                subject_text=candidate["subject_match"]["original"].text,
                subject_type=subject_type,
                object_text=candidate["obj_match"]["original"].text,
                object_type=object_type
            )
            
            if not validation_result.is_valid:
                logger.warning(
                    f"🚫 VALIDATION FAILED: {candidate['subject_match']['original'].text} --[{canonical_relation}]--> {candidate['obj_match']['original'].text}"
                )
                logger.warning(f"   Reason: {validation_result.reason}")
                logger.warning(f"   Confidence penalty: {validation_result.confidence_penalty:.2f}")
                
                # Apply confidence penalty
                final_confidence *= validation_result.confidence_penalty
                logger.info(f"   → Adjusted confidence: {final_confidence:.3f}")
            
            # DeBERTa/LLM fallback (if needed)
            CONFIDENCE_THRESHOLD_FOR_FALLBACK = 0.60
            
            # Check if we should trigger LLM despite high confidence (semantic issues)
            force_llm_fallback = self.relation_validator.should_trigger_llm_fallback(
                relation=canonical_relation,
                subject_type=subject_type,
                object_type=object_type,
                base_confidence=final_confidence
            )
            
            logger.info(
                f"📊 Fallback check: final_confidence={final_confidence:.3f} "
                f"{'< BELOW' if final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK else '>= ABOVE'} "
                f"threshold={CONFIDENCE_THRESHOLD_FOR_FALLBACK:.2f}"
                f"{' (FORCED by validator)' if force_llm_fallback else ''} → "
                f"{'TRIGGER DeBERTa/LLM' if (final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK or force_llm_fallback) else 'NO FALLBACK NEEDED'}"
            )
            
            if final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK or force_llm_fallback:
                try:
                    classifier_result = await self.relation_classifier.classify(
                        subject=candidate["subject_match"]["original"].text,
                        object_text=candidate["obj_match"]["original"].text,
                        context=candidate["resolved_text"],
                        user_id=candidate["user_id"]
                    )
                    
                    if classifier_result.confidence > final_confidence:
                        logger.info(f"   ✅ DeBERTa/LLM improved: {canonical_relation} → {classifier_result.relation} (conf={classifier_result.confidence:.2f})")
                        canonical_relation = classifier_result.relation
                        final_confidence = classifier_result.confidence
                except Exception as e:
                    logger.warning(f"   DeBERTa/LLM fallback failed: {e}")
            
            # Infer category
            category = self._infer_category(canonical_relation)
            
            # IS_A edge filtering
            if canonical_relation == "is_a":
                resolved_obj_type = candidate["obj_match"]["entity"].type
                if not self._should_keep_is_a_edge(resolved_obj_type):
                    logger.info(f"🚫 FILTERED invalid is_a: {candidate['subject_match']['original'].text} --[is_a]--> {candidate['obj_match']['original'].text} (resolved_type={resolved_obj_type})")
                    continue
            
            # Create relation
            relation = ExtractedRelation(
                subject_entity_id=candidate["subject_match"]["entity"].entity_id,
                subject_text=candidate["subject_match"]["original"].text,
                object_entity_id=candidate["obj_match"]["entity"].entity_id,
                object_text=candidate["obj_match"]["original"].text,
                relation=canonical_relation,
                category=category,
                confidence=final_confidence,
                context=candidate["resolved_text"],
                source="dependency",
                memory_id=candidate["memory_id"],
                metadata=candidate["metadata"]
            )
            
            relations.append(relation)
            
            # Log for training
            if self.training_collector:
                self.training_collector.log_extraction(
                    subject=relation.subject_text,
                    object_text=relation.object_text,
                    context=candidate["resolved_text"],
                    predicted_relation=relation.relation,
                    confidence=relation.confidence,
                    category=relation.category,
                    source="dependency",
                    user_id=candidate["user_id"],
                    memory_id=candidate["memory_id"]
                )
        
        logger.info(f"✅ Dependency extraction: {len(relations)} relations from NER entities")
        
        # ========== RELATION DEDUPLICATION (Phase 1E) ==========
        logger.info(f"🔄 Phase 4: Deduplication")
        deduplicated_relations = self._deduplicate_relations(relations)
        
        if len(deduplicated_relations) < len(relations):
            logger.info(f"   → Deduplicated: {len(relations)} → {len(deduplicated_relations)} relations")
        else:
            logger.info(f"   → No duplicates found")
        
        # ========== FINAL SUMMARY ==========
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 FINAL EXTRACTED RELATIONS ({len(deduplicated_relations)} total)")
        logger.info(f"{'='*80}")
        for idx, rel in enumerate(deduplicated_relations, 1):
            logger.info(
                f"{idx:2d}. {rel.subject_text:20s} --[{rel.relation:15s}]--> {rel.object_text:20s} "
                f"(conf={rel.confidence:.3f}, cat={rel.category})"
            )
        logger.info(f"{'='*80}\n")
        
        return deduplicated_relations
    
    def _collect_predicate_candidates_from_span(
        self,
        span_text: str,
        subject_match: Dict,
        predicate: str,
        pattern_conf: float,
        modality_score: float,
        pattern: str,
        is_negated: bool,
        is_passive: bool,
        is_question: bool,
        modality: Optional[str],
        resolved_entities: Dict,
        resolved_text: str,
        doc,
        user_id: str,
        memory_id: str,
        metadata: Dict,
        depth: int = 0,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Collect predicate candidates from a span WITHOUT normalizing them yet.
        
        This is Phase 1 of the new 3-phase approach:
        1. Collect ALL candidates
        2. Batch normalize ALL at once
        3. Create relations
        
        Returns:
            List of candidate dicts with all info needed for later normalization and relation creation
        """
        if depth > max_depth:
            logger.debug(f"      Max recursion depth reached at {depth}")
            return []
        
        candidates = []
        indent = "   " * (depth + 1)
        
        logger.info(f"{indent}🔄 Collecting candidates from span (depth={depth}): '{span_text[:80]}...'")
        logger.info(f"{indent}   Subject: {subject_match['original'].text}, Predicate: {predicate}")
        
        # Step 1: Find ALL entities within this span
        entities_in_span = []
        for entity_text, entity_data in resolved_entities.items():
            if entity_text in span_text.lower():
                entities_in_span.append(entity_data)
        
        if not entities_in_span:
            logger.info(f"{indent}   ❌ No entities found in span")
            return []
        
        logger.info(f"{indent}   ✅ Found {len(entities_in_span)} entities: {[e['entity'].canonical_name for e in entities_in_span]}")
        
        # Step 2: Collect candidates for each entity in span
        for obj_match in entities_in_span:
            # Skip self-referential
            if subject_match['entity'].entity_id == obj_match['entity'].entity_id:
                continue
            
            # Infer predicate
            inferred_predicate = self._infer_predicate_for_nested_entity(
                base_predicate=predicate,
                subject_type=subject_match['original'].type,
                object_type=obj_match['original'].type,
                object_text=obj_match['original'].text,
                span_text=span_text,
                doc=doc
            )
            
            if inferred_predicate != predicate:
                logger.debug(f"{indent}   Inferred predicate: {predicate} → {inferred_predicate} (for {obj_match['original'].text})")
            
            # Store candidate (NO normalization yet!)
            candidates.append({
                "subject_match": subject_match,
                "obj_match": obj_match,
                "inferred_predicate": inferred_predicate,
                "pattern_conf": pattern_conf,
                "modality_score": modality_score,
                "pattern": pattern,
                "is_negated": is_negated,
                "is_passive": is_passive,
                "is_question": is_question,
                "modality": modality,
                "resolved_text": resolved_text,
                "user_id": user_id,
                "memory_id": memory_id,
                "metadata": metadata
            })
        
        logger.info(f"{indent}   📦 Collected {len(candidates)} candidates from this span")
        return candidates
    
    async def _extract_relations_from_span(
        self,
        span_text: str,
        subject_match: Dict,
        predicate: str,
        pattern_conf: float,
        modality_score: float,
        pattern: str,
        is_negated: bool,
        is_passive: bool,
        is_question: bool,
        modality: Optional[str],
        resolved_entities: Dict,
        resolved_text: str,
        doc,
        user_id: str,
        memory_id: str,
        metadata: Dict,
        depth: int = 0,
        max_depth: int = 3
    ) -> List[ExtractedRelation]:
        """
        Recursively extract relations from a span that may contain nested entities.
        
        Example:
            span: "the manager of a gift shop in Chapel Hill"
            
            Extracts:
            1. subject → works_as → manager (main relation)
            2. manager → works_at → gift shop (nested)
            3. gift shop → located_in → Chapel Hill (nested)
        
        Args:
            span_text: The object span to parse
            subject_match: The subject entity
            predicate: The predicate connecting subject to objects
            pattern_conf: Confidence from dependency pattern
            modality_score: Modality score
            pattern: Pattern name
            is_negated: Whether negated
            is_passive: Whether passive voice
            is_question: Whether question
            modality: Modality type
            resolved_entities: Dict of all resolved entities
            resolved_text: Full resolved text
            doc: spaCy Doc
            user_id: User ID
            memory_id: Memory ID
            metadata: Additional metadata
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            List of ExtractedRelation objects
        """
        if depth > max_depth:
            logger.debug(f"      Max recursion depth reached at {depth}")
            return []
        
        relations = []
        indent = "   " * (depth + 1)
        
        logger.info(f"{indent}🔄 Parsing span (depth={depth}): '{span_text[:80]}...'")
        logger.info(f"{indent}   Subject: {subject_match['original'].text}, Predicate: {predicate}")
        
        # Step 1: Find ALL entities within this span
        entities_in_span = []
        for entity_text, entity_data in resolved_entities.items():
            if entity_text in span_text.lower():
                entities_in_span.append(entity_data)
        
        if not entities_in_span:
            logger.info(f"{indent}   ❌ No entities found in span")
            return []
        
        logger.info(f"{indent}   ✅ Found {len(entities_in_span)} entities: {[e['entity'].canonical_name for e in entities_in_span]}")
        
        # Step 2: Collect all predicates for batch normalization
        predicate_batch = []
        for obj_match in entities_in_span:
            # Skip self-referential
            if subject_match['entity'].entity_id == obj_match['entity'].entity_id:
                continue
            
            # SMART PREDICATE INFERENCE: Infer better predicates for nested entities
            # Example: "manager of gift shop in Chapel Hill"
            #   - manager → works_as (base predicate)
            #   - gift shop → works_at (inferred from "of")
            #   - Chapel Hill → located_in (inferred from "in" + location type)
            inferred_predicate = self._infer_predicate_for_nested_entity(
                base_predicate=predicate,
                subject_type=subject_match['original'].type,
                object_type=obj_match['original'].type,
                object_text=obj_match['original'].text,
                span_text=span_text,
                doc=doc
            )
            
            if inferred_predicate != predicate:
                logger.debug(f"{indent}   Inferred predicate: {predicate} → {inferred_predicate} (for {obj_match['original'].text})")
            
            predicate_batch.append({
                "predicate_lemma": inferred_predicate,
                "subject_type": subject_match['original'].type,
                "object_type": obj_match['original'].type,
                "context": resolved_text,
                "user_id": user_id,
                "memory_id": memory_id,
                "session_id": metadata.get("session_id"),
                "obj_match": obj_match  # Store for later use
            })
        
        # Batch normalize all predicates at once (HUGE PERFORMANCE WIN!)
        if not predicate_batch:
            return []
        
        normalized_results = await self.relation_normalizer.normalize_batch(predicate_batch)
        
        # Step 3: Create relations using normalized results
        for i, (canonical_relation, norm_conf) in enumerate(normalized_results):
            obj_match = predicate_batch[i]["obj_match"]
            
            # Compose confidence
            final_confidence = self._compose_confidence(
                pattern_conf=pattern_conf,
                norm_conf=norm_conf,
                modality_score=modality_score,
                resolver_conf_subj=subject_match['original'].confidence,
                resolver_conf_obj=obj_match['original'].confidence,
                subject_text=subject_match['original'].text,
                object_text=obj_match['original'].text,
                relation=canonical_relation
            )
            
            # ========== DEBERTA/LLM FALLBACK (Phase 1E) ==========
            # If confidence is low, use DeBERTa/LLM to re-classify the relation
            CONFIDENCE_THRESHOLD_FOR_FALLBACK = 0.60  # Lower threshold to trigger fallback
            
            # Log fallback decision
            logger.info(
                f"📊 Fallback check: final_confidence={final_confidence:.3f} "
                f"{'< BELOW' if final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK else '>= ABOVE'} "
                f"threshold={CONFIDENCE_THRESHOLD_FOR_FALLBACK:.2f} → "
                f"{'TRIGGER DeBERTa/LLM' if final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK else 'NO FALLBACK NEEDED'}"
            )
            
            if final_confidence < CONFIDENCE_THRESHOLD_FOR_FALLBACK:
                logger.debug(f"{indent}   Low confidence ({final_confidence:.2f}), using DeBERTa/LLM fallback")
                try:
                    classifier_result = await self.relation_classifier.classify(
                        subject=subject_match['original'].text,
                        object_text=obj_match['original'].text,
                        context=resolved_text,
                        user_id=user_id
                    )
                    
                    # Use classifier result if confidence is higher
                    if classifier_result.confidence > final_confidence:
                        logger.info(f"{indent}   ✅ DeBERTa/LLM improved: {canonical_relation} → {classifier_result.relation} (conf={classifier_result.confidence:.2f})")
                        canonical_relation = classifier_result.relation
                        final_confidence = classifier_result.confidence
                        category = classifier_result.category
                except Exception as e:
                    logger.warning(f"{indent}   DeBERTa/LLM fallback failed: {e}")
            
            # Infer category
            category = self._infer_category(canonical_relation)
            
            # ========== IS_A EDGE FILTERING (Phase 1E) ==========
            # Filter out invalid is_a edges (e.g., person is_a organization)
            # Use the RESOLVED entity's type, not the extracted entity's type
            if canonical_relation == "is_a":
                resolved_obj_type = obj_match['entity'].type
                if not self._should_keep_is_a_edge(resolved_obj_type):
                    logger.debug(f"{indent}⏭️  Filtered invalid is_a: {subject_match['original'].text} --[is_a]--> {obj_match['original'].text} (resolved_type={resolved_obj_type})")
                    continue
            
            # Create relation
            relation = ExtractedRelation(
                subject_entity_id=subject_match['entity'].entity_id,
                subject_text=subject_match['original'].text,
                object_entity_id=obj_match['entity'].entity_id,
                object_text=obj_match['original'].text,
                relation=canonical_relation,
                category=category,
                confidence=final_confidence,
                context=resolved_text,
                source="dependency",
                memory_id=memory_id,
                metadata={
                    "pattern": pattern,
                    "is_negated": is_negated,
                    "is_passive": is_passive,
                    "is_question": is_question,
                    "modality": modality,
                    "modality_score": modality_score,
                    "depth": depth,
                    **metadata
                }
            )
            
            relations.append(relation)
            logger.debug(f"{indent}✅ Created relation: {subject_match['original'].text} --[{canonical_relation}]--> {obj_match['original'].text}")
        
        # Step 3: Recursively parse nested structures within the span
        # DISABLED FOR NOW - causing infinite recursion
        # TODO: Implement smarter nested parsing with cycle detection
        # Parse the span itself to find nested relations (e.g., "manager of gift shop" → "gift shop in Chapel Hill")
        # span_doc = self.entity_extractor.nlp(span_text)
        # nested_triples = self.dependency_extractor.extract_from_doc(span_doc)
        # 
        # if nested_triples:
        #     logger.debug(f"{indent}Found {len(nested_triples)} nested triples in span")
        #     ...
        
        logger.debug(f"{indent}Skipping nested recursion (depth={depth}) to avoid infinite loops")
        
        return relations
    
    def _extract_noun_phrases(self, doc) -> List[str]:
        """
        Extract important noun phrases from the document.
        
        Focuses on:
        - Compound nouns (e.g., "oil paintings", "gift shop")
        - Head nouns with modifiers (e.g., "local artists")
        - Excludes pronouns, determiners, very common words
        
        Returns:
            List of noun phrase strings
        """
        noun_phrases = []
        seen = set()
        
        # Extract noun chunks (spaCy's built-in noun phrase detection)
        for chunk in doc.noun_chunks:
            # Skip if it's just a pronoun or determiner
            if chunk.root.pos_ in ("PRON", "DET"):
                continue
            
            # Skip very common/generic words
            if chunk.text.lower() in ("it", "this", "that", "these", "those", "something", "anything"):
                continue
            
            # Check if chunk overlaps with NER entities
            overlapping_ents = [ent for ent in doc.ents 
                              if (chunk.start <= ent.start < chunk.end or 
                                  ent.start <= chunk.start < ent.end)]
            
            # If chunk is EXACTLY a NER entity, skip it (we already have it)
            if overlapping_ents and len(overlapping_ents) == 1:
                ent = overlapping_ents[0]
                if chunk.start == ent.start and chunk.end == ent.end:
                    continue
            
            # If chunk contains NER but has more content, extract the root noun
            # Example: "a 31 year old woman" contains "31 year old" (DATE) but root is "woman"
            if overlapping_ents and chunk.root.pos_ == "NOUN":
                # Check if the root noun is outside the NER entity
                root_is_ner = any(ent.start <= chunk.root.i < ent.end for ent in overlapping_ents)
                if not root_is_ner:
                    # Extract just the root noun (the main entity)
                    text = chunk.root.text
                    if text.lower() not in seen and text.lower() not in ("it", "this", "that"):
                        noun_phrases.append(text)
                        seen.add(text.lower())
                    continue
                else:
                    # Root is part of NER, skip this chunk
                    continue
            
            # Clean the chunk text (remove leading articles)
            text = chunk.text
            for article in ["the ", "a ", "an "]:
                if text.lower().startswith(article):
                    text = text[len(article):]
            
            text = text.strip()
            
            # Only keep if it has at least one noun and isn't too long
            words = text.split()
            if len(words) > 0 and len(words) <= 4 and text.lower() not in seen:
                noun_phrases.append(text)
                seen.add(text.lower())
        
        return noun_phrases
    
    def _infer_predicate_for_nested_entity(
        self,
        base_predicate: str,
        subject_type: Optional[str],
        object_type: Optional[str],
        object_text: str,
        span_text: str,
        doc
    ) -> str:
        """
        Infer a more specific predicate for nested entities using preposition analysis.
        
        Strategy:
        1. Parse the span to find the preposition connecting to the object entity
        2. Map preposition + entity type → predicate
        3. Fall back to base predicate if no clear pattern
        
        Examples:
            - "manager of gift shop in Chapel Hill"
              - "of gift shop" → part_of/works_at
              - "in Chapel Hill" → located_in
        
        Args:
            base_predicate: The original predicate from dependency parsing
            subject_type: Type of subject entity
            object_type: Type of object entity
            object_text: The object entity text
            span_text: The full span text for context
            doc: spaCy Doc for the full sentence
            
        Returns:
            Inferred predicate
        """
        # Parse the span to find prepositions
        span_doc = self.entity_extractor.nlp(span_text)
        
        # Find the object entity in the span
        object_lower = object_text.lower()
        prep_before_object = None
        
        # Strategy: Find the entity and look for prepositions in a window around it
        entity_token_idx = None
        for i, token in enumerate(span_doc):
            # Check if this token or nearby tokens match the entity
            if object_lower in span_text[token.idx:token.idx+len(object_text)+10].lower():
                entity_token_idx = i
                break
        
        if entity_token_idx is not None:
            # Look for preposition in a 3-token window before the entity
            for j in range(max(0, entity_token_idx-3), entity_token_idx):
                if span_doc[j].pos_ == "ADP":  # ADP = adposition (preposition)
                    prep_before_object = span_doc[j].text.lower()
                    logger.debug(f"      Found preposition '{prep_before_object}' before '{object_text}'")
                    # Keep the closest preposition (last one found)
        
        # If still not found, try a simple string search
        if not prep_before_object:
            # Common prepositions to look for
            common_preps = [" in ", " at ", " of ", " from ", " with ", " by ", " for ", " on "]
            for prep in common_preps:
                # Find the entity position in the span
                entity_pos = span_text.lower().find(object_lower)
                if entity_pos > 0:
                    # Look at the text before the entity
                    before_entity = span_text[:entity_pos].lower()
                    # Check if any preposition appears in the last 20 characters
                    for p in common_preps:
                        if p in before_entity[-20:]:
                            prep_before_object = p.strip()
                            logger.debug(f"      Found preposition '{prep_before_object}' via string search for '{object_text}'")
                            break
                    if prep_before_object:
                        break
        
        # If no preposition found, use base predicate
        if not prep_before_object:
            return base_predicate
        
        # ========== TYPE-AWARE "OF" INFERENCE (Phase 1E) ==========
        # Use smarter type-aware mapping for prepositions
        
        # Normalize entity types
        normalized_obj_type = self._normalize_entity_type(object_type)
        normalized_subj_type = self._normalize_entity_type(subject_type)
        
        # Type-aware inference for "of" preposition
        if prep_before_object == "of":
            inferred = self._infer_of_relation(
                subject_type=normalized_subj_type,
                object_type=normalized_obj_type,
                object_text=object_text
            )
            if inferred:
                logger.debug(f"      Type-aware 'of' inference: {inferred} (subj={normalized_subj_type}, obj={normalized_obj_type})")
                return inferred
        
        # Standard preposition mapping (for non-"of" prepositions)
        predicate_map = {
            # Location prepositions
            ("in", "location"): "located_in",
            ("at", "location"): "located_at",
            ("near", "location"): "near",
            ("from", "location"): "from",
            
            # Organization/workplace prepositions
            ("at", "organization"): "works_at",
            ("at", "ORG_LIKE"): "works_at",
            ("at", "FACILITY"): "works_at",
            ("for", "organization"): "works_for",
            ("for", "ORG_LIKE"): "works_for",
            
            # Association
            ("with", None): "associated_with",
            
            # Temporal
            ("on", "temporal"): "occurred_on",
            ("at", "temporal"): "occurred_at",
            ("in", "temporal"): "occurred_in",
            
            # Source/origin
            ("from", "person"): "from",
            ("by", "person"): "created_by",
            ("by", "ROLE"): "created_by",
        }
        
        # Try exact match first
        key = (prep_before_object, normalized_obj_type)
        if key in predicate_map:
            logger.debug(f"      Inferred from preposition '{prep_before_object}' + type '{normalized_obj_type}': {predicate_map[key]}")
            return predicate_map[key]
        
        # Try generic preposition match (type=None)
        key = (prep_before_object, None)
        if key in predicate_map:
            logger.debug(f"      Inferred from preposition '{prep_before_object}': {predicate_map[key]}")
            return predicate_map[key]
        
        # Fallback: use base predicate
        logger.debug(f"      No inference for prep='{prep_before_object}', type='{normalized_obj_type}', using base: {base_predicate}")
        return base_predicate
    
    def _normalize_entity_type(self, entity_type: Optional[str]) -> Optional[str]:
        """Normalize entity types to a consistent taxonomy."""
        if not entity_type:
            return None
        
        type_map = {
            "PERSON": "person",
            "ORG": "organization",
            "ORGANIZATION": "organization",
            "GPE": "location",
            "LOC": "location",
            "FAC": "FACILITY",
            "DATE": "temporal",
            "TIME": "temporal",
            "temporal": "temporal",
        }
        
        return type_map.get(entity_type, entity_type)
    
    def _infer_of_relation(
        self,
        subject_type: Optional[str],
        object_type: Optional[str],
        object_text: str
    ) -> Optional[str]:
        """
        Type-aware inference for 'of' preposition.
        
        Examples:
            - Person of Location → located_in
            - Person (ROLE) of Organization → works_at/manages/owns
            - Artwork of Material → made_of
        """
        # Get head lemma from object text
        object_head = object_text.split()[-1].lower().rstrip('s') if object_text else ""
        
        # Person of Location → located_in
        if subject_type == "person" and object_type == "location":
            return "located_in"
        
        # Person (with role) of Organization → works_at/manages/owns
        if subject_type == "person" and object_type in ("organization", "ORG_LIKE", "FACILITY"):
            # Check if subject has role indicators
            if object_head in ("manager", "director", "head", "chief"):
                return "manages"
            if object_head in ("owner", "founder"):
                return "owns"
            return "works_at"
        
        # Role of Organization → works_at (when subject is role, not person)
        if subject_type == "ROLE" and object_type in ("organization", "ORG_LIKE", "FACILITY"):
            return "works_at"
        
        # Artwork of Material → made_of
        if subject_type == "ARTWORK" and object_type in ("MATERIAL", "ARTWORK"):
            return "made_of"
        
        # Object of Material → made_of
        if object_type == "MATERIAL":
            return "made_of"
        
        # Generic fallback for "of"
        return "part_of"
    
    def _find_relative_pronoun_antecedent(self, pronoun_text: str, doc, resolved_entities: Dict) -> Optional[Dict]:
        """
        Find the antecedent entity that a relative pronoun refers to.
        
        Example:
            "Angela is a woman who works..." 
            → "who" refers to "Angela" (not "woman")
        
        Args:
            pronoun_text: The relative pronoun ("who", "which", "that")
            doc: spaCy Doc object
            resolved_entities: Dict of resolved entities
            
        Returns:
            Entity dict for the antecedent, or None
        """
        pronoun_lower = pronoun_text.lower().strip()
        
        if pronoun_lower not in ("who", "which", "that"):
            return None
        
        # Find the pronoun token in the doc
        for token in doc:
            if token.text.lower() == pronoun_lower and token.pos_ == "PRON":
                # Walk up the dependency tree to find what this relative clause modifies
                head = token.head
                
                # The head of a relative pronoun is usually the verb in the relative clause
                # We need to find what noun this relative clause modifies
                while head and head.dep_ in ("relcl", "acl"):
                    head = head.head
                
                # Now head should be the noun being modified
                if head and head.pos_ in ("NOUN", "PROPN"):
                    # But we want the MAIN subject of the sentence, not just the immediate noun
                    # Walk further up to find the root subject
                    root_subj = head
                    while root_subj.head and root_subj.head != root_subj:
                        if root_subj.dep_ in ("nsubj", "nsubjpass"):
                            # Found the main subject
                            break
                        if root_subj.head.dep_ == "ROOT":
                            # Check if there's a subject of the root
                            for child in root_subj.head.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    root_subj = child
                                    break
                            break
                        root_subj = root_subj.head
                    
                    # Try to match this to a resolved entity (no metadata needed here - not a pronoun)
                    antecedent_match = self._find_entity_match(root_subj.text, resolved_entities, doc, None)
                    if antecedent_match:
                        logger.debug(f"      Found antecedent for '{pronoun_text}': {root_subj.text}")
                        return antecedent_match
        
        return None
    
    def _resolve_relative_pronoun(self, span_text: str, doc) -> str:
        """
        Resolve relative pronouns (who, which, that) to their antecedents.
        
        For example:
        - "who" in "woman who works" → "woman"
        - "which" in "company which hired" → "company"
        
        Args:
            span_text: The span text (might be a relative pronoun)
            doc: spaCy Doc object
            
        Returns:
            Resolved text (antecedent if pronoun, otherwise original)
        """
        span_lower = span_text.lower().strip()
        
        # Check if it's a relative pronoun
        if span_lower not in ("who", "which", "that"):
            return span_text
        
        # Find the token in the doc
        for token in doc:
            if token.text.lower() == span_lower and token.pos_ == "PRON":
                # This is a relative pronoun
                # Find what it refers to by walking up the dependency tree
                head = token.head
                
                # Walk up to find the noun being modified
                while head and head.dep_ in ("relcl", "acl"):
                    head = head.head
                
                # If we found a noun, return it
                if head and head.pos_ in ("NOUN", "PROPN"):
                    logger.debug(f"      Resolved relative pronoun '{span_text}' → '{head.text}'")
                    return head.text
        
        # If we couldn't resolve it, return original
        return span_text
    
    def _find_entity_match(self, span_text: str, resolved_entities: Dict, doc, metadata: Dict = None) -> Optional[Dict]:
        """
        Find if a dependency span matches any NER entity.
        
        Matching strategies:
        0. Pronoun match (I/me/my → speaker from metadata)
        1. Exact match (case-insensitive)
        2. Check if span is contained in entity (entity is longer)
        3. Extract core noun from span and match
        4. Contains match (entity contained in span) - prioritize longer matches
        
        Returns:
            Dict with 'entity' and 'original' keys, or None
        """
        span_lower = span_text.lower().strip()
        
        # Strategy 0: First-person pronoun → speaker
        # This is the key to Option C!
        if metadata and metadata.get("speaker"):
            speaker_name = metadata["speaker"]
            pronouns = ["i", "me", "my", "mine", "myself"]
            
            if span_lower in pronouns:
                # Map pronoun to speaker
                speaker_lower = speaker_name.lower()
                if speaker_lower in resolved_entities:
                    logger.debug(f"         Mapped pronoun '{span_text}' → speaker '{speaker_name}'")
                    return resolved_entities[speaker_lower]
        
        # Strategy 1: Exact match
        if span_lower in resolved_entities:
            return resolved_entities[span_lower]
        
        # Strategy 2: Span contained in entity (entity is longer/more specific)
        for entity_text, entity_data in resolved_entities.items():
            if span_lower in entity_text:
                return entity_data
        
        # Strategy 3: For long spans (>5 words), try to extract core noun first
        # This handles cases like "a 31 year old woman who works..." → should match "woman"
        if len(span_text.split()) > 5:
            # Parse the span to find the main noun
            span_doc = self.entity_extractor.nlp(span_text)
            
            # Skip temporal/age descriptors
            skip_words = {"year", "years", "old", "day", "days", "month", "months"}
            
            # Find significant nouns (not time/age words)
            for token in span_doc:
                if token.pos_ == "NOUN" and token.text.lower() not in skip_words:
                    # Check if this noun (or its lemma) is in our resolved entities
                    noun_lower = token.text.lower()
                    if noun_lower in resolved_entities:
                        logger.debug(f"         Matched core noun '{token.text}' from span '{span_text[:50]}...'")
                        return resolved_entities[noun_lower]
                    
                    # Also check lemma
                    lemma_lower = token.lemma_.lower()
                    if lemma_lower in resolved_entities:
                        logger.debug(f"         Matched core noun lemma '{token.lemma_}' from span '{span_text[:50]}...'")
                        return resolved_entities[lemma_lower]
        
        # Strategy 4: Check if any entity is contained in the span
        # Prioritize longer matches (more specific entities)
        matches = []
        for entity_text, entity_data in resolved_entities.items():
            if entity_text in span_lower:
                matches.append((len(entity_text), entity_data))
        
        if matches:
            # Sort by length descending and return the longest match
            matches.sort(key=lambda x: x[0], reverse=True)
            return matches[0][1]
        
        # Strategy 5: Check if span contains any NER entity from doc
        for ent in doc.ents:
            ent_lower = ent.text.lower()
            if ent_lower in span_lower and ent_lower in resolved_entities:
                return resolved_entities[ent_lower]
        
        return None
    
    def _extract_core_entity(self, span_text: str, doc) -> Optional[str]:
        """
        Extract core entity from a dependency span.
        
        Filters out:
        - Pronouns (who, which, that, etc.)
        - Very long clauses (>15 words) - increased from 10 to handle complex copula
        - Spans without any nouns
        
        Returns:
            Core entity text or None if invalid
        """
        # Skip pronouns
        if span_text.lower() in ("who", "which", "that", "what", "where", "when", "how", "why"):
            logger.debug(f"         Filtered: pronoun '{span_text}'")
            return None
        
        # Skip very long spans (likely full clauses)
        # Increased to 20 to allow complex copula like "a 31 year old woman who works..."
        words = span_text.split()
        if len(words) > 20:
            logger.info(f"         Filtered: too long ({len(words)} words) '{span_text[:50]}...'")
            return None
        
        # Try to find a named entity within the span
        for ent in doc.ents:
            if ent.text in span_text:
                # Return the named entity
                return ent.text
        
        # Parse the span to find nouns
        span_doc = self.entity_extractor.nlp(span_text)
        
        # Check for proper nouns first (highest priority)
        proper_nouns = [token.text for token in span_doc if token.pos_ == "PROPN"]
        if proper_nouns:
            return proper_nouns[0]
        
        # For complex spans (>5 words), try to extract the main noun (not time/age descriptors)
        # This handles cases like "a 31 year old woman who works as..."
        if len(words) > 5:
            # Find significant nouns (not time/age words like "year", "old", "day")
            skip_words = {"year", "years", "old", "day", "days", "month", "months"}
            
            for token in span_doc:
                if (token.pos_ in ("NOUN", "PROPN") and 
                    token.text.lower() not in ("who", "which", "that") and
                    token.text.lower() not in skip_words):
                    
                    # For person/entity nouns, don't include age modifiers
                    # Just return the noun itself (e.g., "woman" not "31 year old woman")
                    return token.text
        
        # Extract head noun phrase (the main noun with its modifiers)
        # Look for the root noun and its direct children
        nouns = [token for token in span_doc if token.pos_ in ("NOUN", "PROPN")]
        if nouns:
            # Get the first significant noun (for complex spans) or last noun (for simple spans)
            head_noun = nouns[0] if len(words) > 5 else nouns[-1]
            
            # Include modifiers (adjectives, compounds, numbers)
            modifiers = []
            for child in head_noun.children:
                if child.dep_ in ("amod", "compound", "nummod") and child.i < head_noun.i:
                    modifiers.append(child.text)
            
            # Build noun phrase
            if modifiers:
                return " ".join(modifiers + [head_noun.text])
            else:
                return head_noun.text
        
        # If no nouns found, skip
        logger.debug(f"         Filtered: no nouns found in '{span_text}'")
        return None
    
    def _compose_confidence(
        self,
        pattern_conf: float,
        norm_conf: float,
        modality_score: float,
        resolver_conf_subj: float = 0.80,
        resolver_conf_obj: float = 0.80,
        subject_text: str = "",
        object_text: str = "",
        relation: str = ""
    ) -> float:
        """
        Compose final confidence from 4 factors (weighted).
        
        Weights:
        - Pattern confidence: 40%
        - Entity resolution: 30% (average of subject + object)
        - Normalization: 20%
        - Modality: 10%
        """
        entity_conf = (resolver_conf_subj + resolver_conf_obj) / 2
        
        final = (
            0.40 * pattern_conf +
            0.30 * entity_conf +
            0.20 * norm_conf +
            0.10 * modality_score
        )
        
        final_confidence = round(max(0.0, min(1.0, final)), 3)
        
        # ========== DETAILED COMPOSITION LOGGING (Phase 1E) ==========
        logger.info(
            f"🎯 CONFIDENCE COMPOSITION for '{subject_text} --[{relation}]--> {object_text}':\n"
            f"   Pattern:       {pattern_conf:.3f} × 0.40 = {0.40 * pattern_conf:.3f}\n"
            f"   Entity (avg):  {entity_conf:.3f} × 0.30 = {0.30 * entity_conf:.3f}\n"
            f"     ├─ Subject:  {resolver_conf_subj:.3f}\n"
            f"     └─ Object:   {resolver_conf_obj:.3f}\n"
            f"   Normalization: {norm_conf:.3f} × 0.20 = {0.20 * norm_conf:.3f}\n"
            f"   Modality:      {modality_score:.3f} × 0.10 = {0.10 * modality_score:.3f}\n"
            f"   ════════════════════════════════════════════════\n"
            f"   FINAL CONFIDENCE: {final_confidence:.3f}"
        )
        
        return final_confidence
    
    def _infer_category(self, relation: str) -> str:
        """Infer category from relation type."""
        # Simple mapping - could be enhanced
        category_map = {
            "employed_by": "professional",
            "resides_in": "spatial",
            "studies_subject": "educational",
            "enrolled_at": "educational",
            "sibling_of": "familial",
            "parent_of": "familial",
            "child_of": "familial",
            "spouse_of": "familial",
            "owns": "possessive",
            "created": "creative",
            "founded": "professional",
            "leads": "professional",
            "knows": "social",
            "likes": "affective",
            "loves": "affective",
            "hates": "affective",
        }
        
        return category_map.get(relation, "factual")
    
    def _extract_copula_entities(self, doc, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from copula patterns that NER might miss.
        
        Handles patterns like:
        - "my name is John" → extracts "John" as PERSON
        - "I am Sarah" → extracts "Sarah" as PERSON
        - "this is Alex" → extracts "Alex" as PERSON
        
        Args:
            doc: spaCy Doc object
            text: Original text
            
        Returns:
            List of ExtractedEntity objects
        """
        entities = []
        
        # Pattern 1: "my name is X" or "name is X" (case-insensitive, captures any word)
        name_pattern = r'\b(?:my\s+)?name\s+is\s+(\w+(?:\s+\w+)*)\b'
        for match in re.finditer(name_pattern, text, re.IGNORECASE):
            name = match.group(1).strip()
            # Capitalize first letter of each word for consistency
            name = ' '.join(word.capitalize() for word in name.split())
            logger.info(f"   🎯 Copula pattern detected: 'name is {name}' → extracting as PERSON")
            entities.append(ExtractedEntity(
                text=name,
                type="person",
                start=match.start(1),
                end=match.end(1),
                confidence=0.95,
                original_type="PERSON"
            ))
        
        # Pattern 2: "I am X" (case-insensitive)
        i_am_pattern = r'\bI\s+am\s+(\w+(?:\s+\w+)*)\b'
        for match in re.finditer(i_am_pattern, text, re.IGNORECASE):
            name = match.group(1).strip()
            # Skip if it's a common adjective/role (not a name)
            if name.lower() not in ('a', 'an', 'the', 'happy', 'sad', 'good', 'bad'):
                name = ' '.join(word.capitalize() for word in name.split())
                logger.info(f"   🎯 Copula pattern detected: 'I am {name}' → extracting as PERSON")
                entities.append(ExtractedEntity(
                    text=name,
                    type="person",
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.95,
                    original_type="PERSON"
                ))
        
        # Pattern 3: Use spaCy dependency parsing to find copula constructions
        for token in doc:
            # Look for copula verbs (is, am, are, was, were)
            if token.lemma_ == "be" and token.dep_ == "ROOT":
                # Find the subject (nsubj)
                subject = None
                attr = None
                
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child
                    elif child.dep_ == "attr":  # Attribute (predicate nominative)
                        attr = child
                
                # Check if subject is "name" or "I" and attr is a proper noun
                if subject and attr:
                    subj_text = subject.text.lower()
                    attr_text = attr.text
                    
                    # Check if this is an identity statement
                    is_identity = (
                        subj_text in ("name", "i") or
                        "name" in subj_text
                    )
                    
                    # Check if attr looks like a proper noun (starts with capital)
                    is_proper_noun = attr_text and attr_text[0].isupper()
                    
                    if is_identity and is_proper_noun:
                        logger.info(f"   🎯 Copula pattern detected via dependency: '{subject.text} {token.text} {attr.text}' → extracting '{attr.text}' as PERSON")
                        entities.append(ExtractedEntity(
                            text=attr.text,
                            type="person",
                            start=attr.idx,
                            end=attr.idx + len(attr.text),
                            confidence=0.90,
                            original_type="PERSON"
                        ))
        
        return entities
    
    def _should_keep_is_a_edge(self, object_type: Optional[str]) -> bool:
        """
        Filter out invalid is_a edges.
        
        is_a should only be used for:
        - ROLE/TITLE (manager, CEO)
        - ATTRIBUTE (31 year old, tall)
        - person types (woman, man, artist)
        - CATEGORY (employee, resident)
        
        NOT for:
        - organization/ORG_LIKE/FACILITY (gift shop, company)
        - location (Chapel Hill)
        - product/ARTWORK
        """
        if not object_type:
            # Allow if type is unknown (conservative)
            return True
        
        # Allowed types for is_a
        ALLOWED_IS_A_TYPES = {
            "ROLE", "ATTRIBUTE", "person", "CATEGORY",
            "TITLE", "JOB", "OCCUPATION"
        }
        
        # Rejected types for is_a
        REJECTED_IS_A_TYPES = {
            "organization", "ORG_LIKE", "FACILITY",
            "location", "GPE", "LOC",
            "product", "ARTWORK", "MATERIAL"
        }
        
        # Normalize type
        normalized_type = self._normalize_entity_type(object_type)
        
        # Check allowed
        if normalized_type in ALLOWED_IS_A_TYPES or object_type in ALLOWED_IS_A_TYPES:
            return True
        
        # Check rejected
        if normalized_type in REJECTED_IS_A_TYPES or object_type in REJECTED_IS_A_TYPES:
            return False
        
        # Default: allow (conservative)
        return True
    
    def _deduplicate_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """
        Remove duplicate/redundant relations for same (subject, object) pair.
        
        Priority:
        1. Specific predicates (works_at, manages, located_in) > Generic (part_of, is_a, related_to)
        2. Higher confidence > Lower confidence
        
        Example:
            - Angela --[work_as]--> manager (conf=0.85)
            - Angela --[is_a]--> manager (conf=0.80)
            → Keep work_as, drop is_a
        """
        if not relations:
            return []
        
        # Group by (subject_id, object_id)
        grouped = {}
        for rel in relations:
            key = (rel.subject_entity_id, rel.object_entity_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rel)
        
        # For each group, pick the best relation
        deduplicated = []
        GENERIC_PREDICATES = {"part_of", "is_a", "related_to", "associated_with"}
        
        for key, rels in grouped.items():
            if len(rels) == 1:
                deduplicated.append(rels[0])
                continue
            
            # Sort by: specific predicates first, then confidence
            sorted_rels = sorted(
                rels,
                key=lambda r: (
                    r.relation not in GENERIC_PREDICATES,  # Specific first (True > False)
                    r.confidence  # Higher confidence first
                ),
                reverse=True
            )
            
            # Keep the best one
            best_rel = sorted_rels[0]
            deduplicated.append(best_rel)
            
            # Log dropped duplicates
            for dropped in sorted_rels[1:]:
                logger.info(
                    f"   🗑️  REMOVED duplicate: {dropped.subject_text} --[{dropped.relation}]--> {dropped.object_text} "
                    f"(conf={dropped.confidence:.3f}) → Kept: {best_rel.relation} (conf={best_rel.confidence:.3f})"
                )
        
        return deduplicated
    
    async def _extract_entity_pairs(
        self,
        text: str,
        user_id: str,
        memory_id: str,
        ego_score: float,
        metadata: Dict[str, Any]
    ) -> List[ExtractedRelation]:
        """
        LEGACY: Entity-pair based extraction (Phase 1C).
        
        Used as fallback when dependency extraction fails or is disabled.
        """
        # Step 1: Extract entities using spaCy NER
        entities = self.entity_extractor.extract(text)
        
        if len(entities) < 2:
            logger.debug(f"Not enough entities for relation extraction: {len(entities)}")
            return []
        
        logger.debug(f"Extracted {len(entities)} entities")
        
        # Step 2: Resolve entities to canonical forms
        resolved_entities = []
        for entity in entities:
            resolved = await self.entity_resolver.resolve(
                text=entity.text,
                user_id=user_id,
                context=text,
                entity_type=entity.type
            )
            if resolved:
                resolved_entities.append({
                    "extracted": entity,
                    "resolved": resolved
                })
        
        logger.debug(f"Resolved {len(resolved_entities)} entities")
        
        # Step 3: Get entity pairs for relation classification
        relations = []
        
        for i, e1 in enumerate(resolved_entities):
            for e2 in resolved_entities[i+1:]:
                # Skip if same entity
                if e1["resolved"].entity_id == e2["resolved"].entity_id:
                    continue
                
                # Classify relation
                result = await self._classify_relation(
                    e1, e2, text, user_id, memory_id, metadata
                )
                
                if result:
                    relations.append(result)
        
        logger.info(f"📊 RelationExtractor: Extracted {len(relations)} relations from text")
        for rel in relations:
            logger.info(f"   → {rel.subject_text} ({rel.subject_entity_id}) --[{rel.relation}]--> {rel.object_text} ({rel.object_entity_id})")
        return relations
    
    async def _classify_relation(
        self,
        e1: Dict,
        e2: Dict,
        context: str,
        user_id: str,
        memory_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[ExtractedRelation]:
        """Classify relation between two entities."""
        try:
            # Try DeBERTa/LLM classification first
            result = await self.relation_classifier.classify(
                subject=e1["extracted"].text,
                object_text=e2["extracted"].text,
                context=context,
                user_id=user_id
            )
            
            # Fall back to heuristics if confidence is too low
            if result.confidence < 0.3:
                result = self.relation_classifier.classify_with_heuristics(
                    subject=e1["extracted"].text,
                    object_text=e2["extracted"].text,
                    context=context
                )
            
            # Merge metadata
            final_metadata = {
                "is_discovered": result.is_discovered,
                "subject_type": e1["extracted"].type,
                "object_type": e2["extracted"].type
            }
            if metadata:
                final_metadata.update(metadata)
            
            # Create extracted relation
            extracted = ExtractedRelation(
                subject_entity_id=e1["resolved"].entity_id,
                subject_text=e1["extracted"].text,
                object_entity_id=e2["resolved"].entity_id,
                object_text=e2["extracted"].text,
                relation=result.relation,
                category=result.category,
                confidence=result.confidence,
                context=context,
                source=result.source,
                memory_id=memory_id,
                metadata=final_metadata
            )
            
            # Log for training
            if self.training_collector:
                self.training_collector.log_extraction(
                    subject=e1["extracted"].text,
                    object_text=e2["extracted"].text,
                    context=context,
                    predicted_relation=result.relation,
                    confidence=result.confidence,
                    category=result.category,
                    source=result.source,
                    user_id=user_id,
                    memory_id=memory_id
                )
                
                # Log discovered relation types
                if result.is_discovered:
                    self.training_collector.log_discovered_relation(
                        relation_name=result.relation,
                        category=result.category,
                        example_context=context,
                        example_subject=e1["extracted"].text,
                        example_object=e2["extracted"].text
                    )
            
            return extracted
            
        except Exception as e:
            logger.warning(f"Relation classification failed: {e}")
            return None
    
    async def extract_with_heuristics_only(
        self,
        text: str,
        user_id: str,
        memory_id: str = None
    ) -> List[ExtractedRelation]:
        """
        Extract relations using only heuristics (no API calls).
        Useful for batch processing or when APIs are unavailable.
        """
        if not text or not text.strip():
            return []
        
        # Extract entities
        entities = self.entity_extractor.extract(text)
        
        if len(entities) < 2:
            return []
        
        # Resolve entities
        resolved_entities = []
        for entity in entities:
            resolved = await self.entity_resolver.resolve(
                text=entity.text,
                user_id=user_id,
                context=text,
                entity_type=entity.type,
                create_if_missing=True
            )
            if resolved:
                resolved_entities.append({
                    "extracted": entity,
                    "resolved": resolved
                })
        
        # Classify with heuristics only
        relations = []
        
        for i, e1 in enumerate(resolved_entities):
            for e2 in resolved_entities[i+1:]:
                if e1["resolved"].entity_id == e2["resolved"].entity_id:
                    continue
                
                result = self.relation_classifier.classify_with_heuristics(
                    subject=e1["extracted"].text,
                    object_text=e2["extracted"].text,
                    context=text
                )
                
                extracted = ExtractedRelation(
                    subject_entity_id=e1["resolved"].entity_id,
                    subject_text=e1["extracted"].text,
                    object_entity_id=e2["resolved"].entity_id,
                    object_text=e2["extracted"].text,
                    relation=result.relation,
                    category=result.category,
                    confidence=result.confidence,
                    context=text,
                    source="heuristic",
                    memory_id=memory_id
                )
                relations.append(extracted)
        
        return relations
    
    def get_training_stats(self) -> Dict:
        """Get training data collection statistics."""
        if self.training_collector:
            return self.training_collector.get_stats()
        return {}

