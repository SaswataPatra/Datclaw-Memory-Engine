"""
Intent Router

Routes sentences to appropriate extractors based on classified intent.
This is the orchestration layer that connects intent classification to extraction.
"""

import logging
from typing import Dict, List, Any, Optional

from .sentence_intent_classifier import IntentResult
from .relation_extractor import ExtractedRelation

logger = logging.getLogger(__name__)


class IntentRouter:
    """
    Routes sentences to appropriate extractors based on intent.
    
    Flow:
        Intent Classification → Route to Extractor → Merge Results → Add Metadata
    """
    
    def __init__(
        self,
        dependency_extractor,
        evaluative_extractor=None,
        opinion_extractor=None,
        speech_act_extractor=None,
        directive_extractor=None,
        config: Dict = None
    ):
        """
        Initialize the intent router.
        
        Args:
            dependency_extractor: Standard dependency-based extractor (for facts)
            evaluative_extractor: Extractor for evaluations/preferences
            opinion_extractor: Extractor for opinions/beliefs
            speech_act_extractor: Extractor for speech acts
            directive_extractor: Extractor for directives/intents
            config: Configuration dict
        """
        self.dependency_extractor = dependency_extractor
        self.evaluative_extractor = evaluative_extractor
        self.opinion_extractor = opinion_extractor
        self.speech_act_extractor = speech_act_extractor
        self.directive_extractor = directive_extractor
        self.config = config or {}
        
        # Check which extractors are enabled
        extractors_config = self.config.get("sentence_intent", {}).get("extractors", {})
        self.evaluative_enabled = extractors_config.get("evaluative", {}).get("enabled", True)
        self.opinion_enabled = extractors_config.get("opinion", {}).get("enabled", True)
        self.speech_act_enabled = extractors_config.get("speech_act", {}).get("enabled", True)
        self.directive_enabled = extractors_config.get("directive", {}).get("enabled", True)
        
        logger.info(f"✅ IntentRouter initialized")
        logger.info(f"   → Evaluative: {self.evaluative_enabled}")
        logger.info(f"   → Opinion: {self.opinion_enabled}")
        logger.info(f"   → SpeechAct: {self.speech_act_enabled}")
        logger.info(f"   → Directive: {self.directive_enabled}")
    
    async def route_and_extract(
        self,
        text: str,
        intent_result: IntentResult,
        user_id: str,
        memory_id: str,
        ego_score: float,
        metadata: Dict[str, Any],
        # Additional context needed by extractors
        resolved_text: str = None,
        doc = None,
        resolved_entities: Dict = None,
        entity_resolver = None,
        relation_normalizer = None,
        relation_classifier = None,
        relation_validator = None
    ) -> List[ExtractedRelation]:
        """
        Route sentence to appropriate extractor(s) based on intent.
        
        Args:
            text: Original text
            intent_result: Classification result
            user_id: User ID
            memory_id: Memory ID
            ego_score: Ego score
            metadata: Metadata dict
            resolved_text: Text after coreference resolution
            doc: spaCy Doc object
            resolved_entities: Dict of resolved entities
            entity_resolver: Entity resolver instance
            relation_normalizer: Relation normalizer instance
            relation_classifier: Relation classifier instance
            relation_validator: Relation validator instance
        
        Returns:
            List of ExtractedRelation objects
        """
        intent = intent_result.intent
        confidence = intent_result.confidence
        
        logger.info(f"🎯 Routing to '{intent}' extractor (conf={confidence:.2f})")
        
        # Add intent metadata
        metadata = {**metadata, "intent": intent, "intent_confidence": confidence, "intent_method": intent_result.method}
        
        relations = []
        
        # Route based on intent
        if intent == "fact":
            # Use standard dependency extraction
            relations = await self._extract_fact(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver, relation_normalizer, relation_classifier, relation_validator
            )
        
        elif intent == "evaluation" and self.evaluative_enabled and self.evaluative_extractor:
            # Use evaluative extractor
            relations = await self._extract_evaluation(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver, relation_normalizer
            )
        
        elif intent == "opinion" and self.opinion_enabled and self.opinion_extractor:
            # Use opinion extractor
            relations = await self._extract_opinion(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver, relation_normalizer
            )
        
        elif intent == "speech_act" and self.speech_act_enabled and self.speech_act_extractor:
            # Use speech act extractor
            relations = await self._extract_speech_act(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver
            )
        
        elif intent == "directive" and self.directive_enabled and self.directive_extractor:
            # Use directive extractor
            relations = await self._extract_directive(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver, relation_normalizer
            )
        
        else:
            # Fallback to fact extraction if extractor not available
            logger.warning(f"Extractor for '{intent}' not available or disabled, falling back to fact extraction")
            relations = await self._extract_fact(
                text, resolved_text, doc, resolved_entities,
                user_id, memory_id, ego_score, metadata,
                entity_resolver, relation_normalizer, relation_classifier, relation_validator
            )
        
        # Add intent metadata to all relations
        for rel in relations:
            rel.metadata["intent"] = intent
            rel.metadata["intent_confidence"] = confidence
        
        logger.info(f"✅ Extracted {len(relations)} relations via '{intent}' extractor")
        
        return relations
    
    async def _extract_fact(
        self, text, resolved_text, doc, resolved_entities,
        user_id, memory_id, ego_score, metadata,
        entity_resolver, relation_normalizer, relation_classifier, relation_validator
    ) -> List[ExtractedRelation]:
        """Extract factual relations using dependency parser."""
        # This will be called from RelationExtractor._extract_dependency_based
        # For now, return empty list - the actual extraction happens in the parent method
        logger.debug("   → Using standard dependency extraction")
        return []
    
    async def _extract_evaluation(
        self, text, resolved_text, doc, resolved_entities,
        user_id, memory_id, ego_score, metadata,
        entity_resolver, relation_normalizer
    ) -> List[ExtractedRelation]:
        """Extract evaluative relations (values, preferences)."""
        if not self.evaluative_extractor:
            return []
        
        logger.debug("   → Using evaluative extractor")
        return await self.evaluative_extractor.extract(
            text, resolved_text, doc, resolved_entities,
            user_id, memory_id, ego_score, metadata,
            entity_resolver, relation_normalizer
        )
    
    async def _extract_opinion(
        self, text, resolved_text, doc, resolved_entities,
        user_id, memory_id, ego_score, metadata,
        entity_resolver, relation_normalizer
    ) -> List[ExtractedRelation]:
        """Extract opinion relations (beliefs, stances)."""
        if not self.opinion_extractor:
            return []
        
        logger.debug("   → Using opinion extractor")
        return await self.opinion_extractor.extract(
            text, resolved_text, doc, resolved_entities,
            user_id, memory_id, ego_score, metadata,
            entity_resolver, relation_normalizer
        )
    
    async def _extract_speech_act(
        self, text, resolved_text, doc, resolved_entities,
        user_id, memory_id, ego_score, metadata,
        entity_resolver
    ) -> List[ExtractedRelation]:
        """Extract speech act relations (addresses, thanks)."""
        if not self.speech_act_extractor:
            return []
        
        logger.debug("   → Using speech act extractor")
        return await self.speech_act_extractor.extract(
            text, resolved_text, doc, resolved_entities,
            user_id, memory_id, ego_score, metadata,
            entity_resolver
        )
    
    async def _extract_directive(
        self, text, resolved_text, doc, resolved_entities,
        user_id, memory_id, ego_score, metadata,
        entity_resolver, relation_normalizer
    ) -> List[ExtractedRelation]:
        """Extract directive relations (wants, intends)."""
        if not self.directive_extractor:
            return []
        
        logger.debug("   → Using directive extractor")
        return await self.directive_extractor.extract(
            text, resolved_text, doc, resolved_entities,
            user_id, memory_id, ego_score, metadata,
            entity_resolver, relation_normalizer
        )

