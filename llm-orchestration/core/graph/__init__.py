# DAPPY Graph-of-Thoughts Module
# Phase 1A: Entity Extraction & Resolution (Revised)
# Phase 1B: Relation Classification & Training Data
# Phase 1C: Integrated Relation Extraction Pipeline + Graph Integration + ArangoDB
# Phase 1D: Dependency-Based Extraction (Coref + Dependency Parsing)
# Phase 2: Activation Scorer (12 features)
# Phase 3: PPR-based Graph Retrieval

from .schemas import (
    ThoughtEdge,
    CandidateEdge,
    Entity,
    EdgeType,
    RelationCategory,
    SupportingMention,
    EDGE_TYPE_TAXONOMY,
    HIGH_IMPACT_RELATIONS,
    get_relation_category,
)
from .edge_store import CandidateEdgeStore, ThoughtEdgeStore
from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_resolver import EntityResolver
from .relation_classifier import (
    RelationClassifier,
    RelationResult,
    RELATION_CATEGORIES,
    get_all_relation_types,
    get_relation_category as get_relation_category_v2,
)
from .relation_training_collector import RelationTrainingCollector
from .normalization_training_collector import NormalizationTrainingCollector
from .relation_extractor import RelationExtractor, ExtractedRelation
from .graph_integration import GraphIntegration

# Phase 1D: Dependency-based extraction
from .coref_resolver import CorefResolver, CorefCluster
from .dependency_extractor import DependencyExtractor, DependencyTriple
from .relation_normalizer import RelationNormalizer

# Phase 1F: Dialogue processing
from .dialogue_processor import DialogueProcessor, DialogueTurn, ProcessedDialogue

# Phase 1F.3: Relation validation
from .relation_validator import RelationValidator, ValidationResult

# Phase 1G: Sentence intent classification and routing
from .sentence_intent_classifier import SentenceIntentClassifier, IntentResult
from .intent_router import IntentRouter
from .evaluative_extractor import EvaluativeExtractor
from .opinion_extractor import OpinionExtractor
from .speech_act_extractor import SpeechActExtractor

from .activation_scorer import (
    ActivationScorer, 
    ActivationResult, 
    HeuristicCombiner,
)
from .relation_importance_scorer import (
    RelationImportanceScorer,
    RelationImportanceTrainingCollector,
    RelationScorerResult,
)
from .ppr_retrieval import PPRRetrieval, PPRResult
from .arango_integration import (
    ArangoGraphManager,
    GraphPipeline,
    create_graph_pipeline,
)
from .query_helpers import GraphQueryHelper, apply_temporal_filter

__all__ = [
    # Phase 1D: Dependency-based extraction
    "CorefResolver",
    "CorefCluster",
    "DependencyExtractor",
    "DependencyTriple",
    "RelationNormalizer",
    # Phase 1F: Dialogue processing
    "DialogueProcessor",
    "DialogueTurn",
    "ProcessedDialogue",
    # Phase 1F.3: Relation validation
    "RelationValidator",
    "ValidationResult",
    # Phase 1G: Intent classification and routing
    "SentenceIntentClassifier",
    "IntentResult",
    "IntentRouter",
    "EvaluativeExtractor",
    "OpinionExtractor",
    "SpeechActExtractor",
    # Original exports
    # Schemas
    'ThoughtEdge',
    'CandidateEdge', 
    'Entity',
    'EdgeType',
    'RelationCategory',
    'SupportingMention',
    'EDGE_TYPE_TAXONOMY',
    'HIGH_IMPACT_RELATIONS',
    'get_relation_category',
    # Entity Extraction & Resolution
    'EntityExtractor',
    'ExtractedEntity',
    'EntityResolver',
    # Relation Classification
    'RelationClassifier',
    'RelationResult',
    'RELATION_CATEGORIES',
    'get_all_relation_types',
    'RelationTrainingCollector',
    'NormalizationTrainingCollector',
    # Integrated Relation Extraction
    'RelationExtractor',
    'ExtractedRelation',
    # Graph Integration
    'GraphIntegration',
    # ArangoDB Integration (Main Entry Point for Chatbot)
    'ArangoGraphManager',
    'GraphPipeline',
    'create_graph_pipeline',
    # Activation Scoring (Phase 2)
    'ActivationScorer',
    'ActivationResult',
    'HeuristicCombiner',
    # Relation Importance (ComponentScorer pattern)
    'RelationImportanceScorer',
    'RelationImportanceTrainingCollector',
    'RelationScorerResult',
    # PPR Retrieval (Phase 3)
    'PPRRetrieval',
    'PPRResult',
    # Stores
    'CandidateEdgeStore',
    'ThoughtEdgeStore',
    # Query Helpers
    'GraphQueryHelper',
    'apply_temporal_filter',
]

