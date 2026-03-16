"""
DAPPY Graph-of-Thoughts Integration Module

Integrates the Graph-of-Thoughts components into the main memory pipeline:
1. CommandParser - Parse explicit user commands
2. EntityExtractor - Extract entities from text
3. EntityResolver - Resolve entities to canonical forms
4. RelationExtractor - Extract relations between entities
5. CandidateEdgeStore - Store candidate edges
6. CommandTrainingCollector - Collect training data

This module provides a unified interface for the chatbot service.

Phase 1C Implementation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_resolver import EntityResolver
from .relation_extractor import RelationExtractor, ExtractedRelation
from .relation_classifier import RelationClassifier
from .relation_training_collector import RelationTrainingCollector
from .edge_store import CandidateEdgeStore, ThoughtEdgeStore
from .schemas import CandidateEdge, SupportingMention
from ..command_parser import CommandParser, UserCommand, CommandCategory
from ..command_training_collector import CommandTrainingCollector

logger = logging.getLogger(__name__)


class GraphIntegration:
    """
    Unified interface for Graph-of-Thoughts integration.
    
    This class orchestrates all graph components and provides
    a simple API for the chatbot service to use.
    
    Usage:
        graph = GraphIntegration(db, config)
        
        # Process a message
        result = await graph.process_message(
            user_id="user123",
            session_id="session456",
            message="My sister Sarah works at Google",
            ego_score=0.75
        )
        
        # Result contains:
        # - commands: Parsed user commands
        # - entities: Extracted entities
        # - relations: Extracted relations
        # - candidate_edges: Created candidate edges
    """
    
    def __init__(
        self,
        db,
        config: Optional[Dict[str, Any]] = None,
        embedding_service = None,
        enabled: bool = True
    ):
        """
        Initialize graph integration.
        
        Args:
            db: ArangoDB database connection
            config: Configuration dict
            embedding_service: Optional embedding service
            enabled: Whether graph extraction is enabled
        """
        self.db = db
        self.config = config or {}
        self.embedding_service = embedding_service
        self.enabled = enabled
        
        if not enabled:
            logger.info("Graph integration disabled")
            return
        
        # Initialize components
        self._init_components()
        
        logger.info("✅ GraphIntegration initialized")
    
    def _init_components(self):
        """Initialize all graph components."""
        # Command parser
        self.command_parser = CommandParser(config=self.config)
        
        # Command training collector
        self.command_collector = CommandTrainingCollector(
            db_path=self.config.get('command_training', {}).get(
                'db_path', 'data/command_training.db'
            )
        )
        
        # Entity extractor
        self.entity_extractor = EntityExtractor(
            config=self.config.get('entity_extraction', {})
        )
        
        # Entity resolver
        self.entity_resolver = EntityResolver(
            db=self.db,
            config=self.config,
            embedding_service=self.embedding_service
        )
        
        # Relation classifier
        self.relation_classifier = RelationClassifier(
            config=self.config
        )
        
        # Relation training collector
        self.relation_collector = RelationTrainingCollector(
            db_path=self.config.get('relation_training', {}).get(
                'db_path', 'data/relation_training.db'
            )
        )
        
        # Candidate edge store
        self.candidate_edge_store = CandidateEdgeStore(db=self.db)
        
        # Thought edge store (for promoted edges)
        self.thought_edge_store = ThoughtEdgeStore(db=self.db)
        
        logger.info("  All graph components initialized")
    
    async def process_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        memory_id: str = None,
        ego_score: float = 0.5,
        system_tier: int = None
    ) -> Dict[str, Any]:
        """
        Process a message through the graph pipeline.
        
        This is the main entry point for graph extraction.
        
        Args:
            user_id: User ID
            session_id: Session ID
            message: User message
            memory_id: Optional memory ID (for linking)
            ego_score: Ego score of the memory
            system_tier: System-assigned tier (for training data)
        
        Returns:
            Dict containing:
            - commands: List of parsed commands
            - remaining_text: Text after commands
            - entities: List of extracted entities
            - relations: List of extracted relations
            - candidate_edges: List of created candidate edges
            - command_config: Aggregated command configuration
        """
        if not self.enabled:
            return {
                "commands": [],
                "remaining_text": message,
                "entities": [],
                "relations": [],
                "candidate_edges": [],
                "command_config": {}
            }
        
        result = {
            "commands": [],
            "remaining_text": message,
            "entities": [],
            "relations": [],
            "candidate_edges": [],
            "command_config": {}
        }
        
        try:
            # Step 1: Parse commands
            commands, remaining_text = self.command_parser.parse(message)
            result["commands"] = commands
            result["remaining_text"] = remaining_text
            
            if commands:
                # Log commands for training
                for cmd in commands:
                    self.command_collector.log_command(
                        user_id=user_id,
                        message=message,
                        command=cmd,
                        session_id=session_id,
                        system_tier=system_tier,
                        system_ego_score=ego_score,
                        memory_id=memory_id
                    )
                
                # Aggregate command config
                result["command_config"] = self._aggregate_command_config(commands)
                
                logger.info(f"Parsed {len(commands)} commands: {[c.command for c in commands]}")
            
            # Step 2: Check if this is a retrieval-only command
            if commands and self.command_parser.is_retrieval_only(commands):
                logger.debug("Retrieval-only command, skipping extraction")
                return result
            
            # Step 3: Extract entities from remaining text
            text_to_process = remaining_text if remaining_text else message
            entities = self.entity_extractor.extract(text_to_process)
            result["entities"] = [e.to_dict() for e in entities]
            
            if len(entities) < 2:
                logger.debug(f"Only {len(entities)} entities found, skipping relation extraction")
                return result
            
            # Step 4: Resolve entities and extract relations
            relations = await self._extract_relations(
                entities=entities,
                text=text_to_process,
                user_id=user_id,
                memory_id=memory_id,
                ego_score=ego_score
            )
            result["relations"] = [r.to_dict() for r in relations]
            
            # Step 5: Create candidate edges
            candidate_edges = await self._create_candidate_edges(
                relations=relations,
                user_id=user_id,
                ego_score=ego_score
            )
            result["candidate_edges"] = [
                {"candidate_id": ce.candidate_id, "predicate": ce.predicate}
                for ce in candidate_edges
            ]
            
            logger.info(
                f"Graph extraction complete: "
                f"{len(entities)} entities, "
                f"{len(relations)} relations, "
                f"{len(candidate_edges)} candidate edges"
            )
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}", exc_info=True)
        
        return result
    
    def _aggregate_command_config(self, commands: List[UserCommand]) -> Dict[str, Any]:
        """Aggregate configuration from multiple commands."""
        config = {
            "importance_weight": self.command_parser.get_importance_weight(commands),
            "target_tier": self.command_parser.get_target_tier(commands),
            "requires_confirmation": self.command_parser.requires_confirmation(commands),
            "is_retrieval_only": self.command_parser.is_retrieval_only(commands),
            "should_encrypt": self.command_parser.should_encrypt(commands),
            "creates_edge": self.command_parser.creates_edge(commands),
            "actions": [cmd.config.get("action") for cmd in commands]
        }
        return config
    
    async def _extract_relations(
        self,
        entities: List[ExtractedEntity],
        text: str,
        user_id: str,
        memory_id: str = None,
        ego_score: float = 0.5
    ) -> List[ExtractedRelation]:
        """Extract relations between entities."""
        relations = []
        
        # Resolve entities first
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
        
        # Get entity pairs
        for i, e1 in enumerate(resolved_entities):
            for e2 in resolved_entities[i+1:]:
                # Skip if same entity
                if e1["resolved"].entity_id == e2["resolved"].entity_id:
                    continue
                
                # Classify relation (use heuristics for speed)
                result = self.relation_classifier.classify_with_heuristics(
                    subject=e1["extracted"].text,
                    object_text=e2["extracted"].text,
                    context=text
                )
                
                # Create extracted relation
                relation = ExtractedRelation(
                    subject_entity_id=e1["resolved"].entity_id,
                    subject_text=e1["extracted"].text,
                    object_entity_id=e2["resolved"].entity_id,
                    object_text=e2["extracted"].text,
                    relation=result.relation,
                    category=result.category,
                    confidence=result.confidence,
                    context=text,
                    source=result.source,
                    memory_id=memory_id
                )
                relations.append(relation)
                
                # Log for training
                self.relation_collector.log_extraction(
                    subject=e1["extracted"].text,
                    object_text=e2["extracted"].text,
                    context=text,
                    predicted_relation=result.relation,
                    confidence=result.confidence,
                    category=result.category,
                    source=result.source,
                    user_id=user_id,
                    memory_id=memory_id
                )
        
        return relations
    
    async def _create_candidate_edges(
        self,
        relations: List[ExtractedRelation],
        user_id: str,
        ego_score: float = 0.5
    ) -> List[CandidateEdge]:
        """Create candidate edges from relations."""
        candidate_edges = []
        
        for relation in relations:
            candidate = relation.to_candidate_edge(
                user_id=user_id,
                ego_score=ego_score
            )
            
            # Store in candidate edge store
            stored = await self.candidate_edge_store.create_or_update(candidate)
            candidate_edges.append(stored)
        
        return candidate_edges
    
    def get_command_help(self) -> str:
        """Get help text for available commands."""
        return self.command_parser.get_command_help()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training data collection statistics."""
        return {
            "commands": self.command_collector.get_stats(),
            "relations": self.relation_collector.get_stats()
        }
    
    async def handle_command_action(
        self,
        command: UserCommand,
        user_id: str,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Handle a specific command action.
        
        This is called by the chatbot service to execute
        command-specific actions.
        
        Args:
            command: Parsed command
            user_id: User ID
            session_id: Session ID
            message: Original message
        
        Returns:
            Dict with action result
        """
        action = command.config.get("action")
        
        if action == "search":
            # Handle /search command
            return await self._handle_search(command, user_id)
        
        elif action == "show_context":
            # Handle /context command
            return await self._handle_context(user_id, session_id)
        
        elif action == "delete":
            # Handle /forget command
            return {"action": "delete", "requires_confirmation": True}
        
        elif action == "create_edge":
            # Handle /link command
            return await self._handle_link(command, user_id)
        
        elif action == "delete_edge":
            # Handle /unlink command
            return await self._handle_unlink(command, user_id)
        
        elif action == "trigger_correction":
            # Handle /correct command
            return {"action": "correction", "edge_type": "contradicts"}
        
        elif action == "create_supersedes_edge":
            # Handle /evolve command
            return {"action": "evolve", "edge_type": "supersedes"}
        
        else:
            # Default: return command config
            return {"action": action, "config": command.config}
    
    async def _handle_search(
        self,
        command: UserCommand,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle /search command."""
        query = command.target_text
        
        # TODO: Implement actual search using Qdrant
        # For now, return placeholder
        return {
            "action": "search",
            "query": query,
            "results": []  # Will be populated by retrieval system
        }
    
    async def _handle_context(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Handle /context command."""
        # TODO: Implement context retrieval using session_id
        return {
            "action": "show_context",
            "session_id": session_id,
            "context": []  # Will be populated by context manager
        }
    
    async def _handle_link(
        self,
        command: UserCommand,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle /link command."""
        if not command.args:
            return {"action": "link", "error": "No entity specified"}
        
        entity_name = command.args[0]
        
        # Resolve entity
        entity = await self.entity_resolver.resolve(
            text=entity_name,
            user_id=user_id,
            create_if_missing=False
        )
        
        if entity:
            return {
                "action": "link",
                "entity_id": entity.entity_id,
                "entity_name": entity.canonical_name
            }
        else:
            return {
                "action": "link",
                "error": f"Entity '{entity_name}' not found"
            }
    
    async def _handle_unlink(
        self,
        command: UserCommand,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle /unlink command."""
        if not command.args:
            return {"action": "unlink", "error": "No entity specified"}
        
        entity_name = command.args[0]
        
        return {
            "action": "unlink",
            "entity_name": entity_name,
            "requires_confirmation": True
        }





