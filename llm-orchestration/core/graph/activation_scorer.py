"""
DAPPY Activation Scorer

Scores candidate edges for promotion to the canonical Knowledge Graph.
Uses 12 features with a heuristic combiner (LightGBM can be added later).

Features (12 total):
1. ego_score - Max ego score of supporting mentions
2. edge_evidence_count - Count of supporting memories
3. recency_weight - exp(-λ * time_since_last_mention)
4. frequency_rate - Mentions per week
5. session_diversity - Count of distinct sessions
6. avg_sentiment - Average sentiment of mentions
7. relation_importance - Via RelationImportanceScorer (ComponentScorer pattern)
8. contradiction_score - Penalty from contradictions
9. node_graph_proximity - Distance to existing KG nodes
10. promote_count - Historical promotion attempts
11. demote_count - Historical demotion count
12. edge_novelty - Similarity to existing edges

Phase 2 Implementation

NOTE: relation_importance uses RelationImportanceScorer which:
      1. Has heuristic defaults in code
      2. Can be overridden via config
      3. Collects training data for learning (Phase 2)
      4. Will be learned per-user (Phase 3)
"""

import logging
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .schemas import CandidateEdge, ThoughtEdge
from .relation_importance_scorer import (
    RelationImportanceScorer,
    RelationImportanceTrainingCollector,
    RelationScorerResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    """Result of activation scoring."""
    activation_score: float
    component_scores: Dict[str, float]
    decision: str  # "promote", "keep", "demote"
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HeuristicCombiner:
    """
    Heuristic combiner for activation scoring.
    Used until enough training data is collected for LightGBM.
    
    Weights are configurable hyperparameters.
    """
    
    DEFAULT_WEIGHTS = {
        "ego_score": 0.15,
        "edge_evidence_count": 0.15,
        "recency_weight": 0.10,
        "frequency_rate": 0.10,
        "session_diversity": 0.08,
        "avg_sentiment": 0.05,
        "relation_importance": 0.12,
        "contradiction_score": -0.10,  # Penalty
        "node_graph_proximity": 0.10,
        "promote_count": 0.05,
        "demote_count": -0.05,  # Penalty
        "edge_novelty": 0.05,
    }
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def combine(self, scores: Dict[str, float]) -> float:
        """
        Combine component scores using weighted sum.
        Handles negative weights for penalty features.
        """
        total = 0.0
        
        for name, weight in self.weights.items():
            # For negative weights (penalties), default to 0 if missing
            default = 0.5 if weight >= 0 else 0.0
            score = scores.get(name, default)
            total += weight * score
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, total))


class ActivationScorer:
    """
    Scores candidate edges for promotion to the Knowledge Graph.
    
    Uses 12 features organized by source:
    - 5 reused from ego scoring (adapted)
    - 4 computed from CandidateEdge
    - 3 new (graph proximity, promote/demote counts)
    
    Uses RelationImportanceScorer (ComponentScorer pattern) for relation importance.
    This allows importance to be learned from promotion/demotion outcomes.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db = None,
        embedding_service = None,
        enable_importance_training: bool = True
    ):
        """
        Initialize activation scorer.
        
        Args:
            config: Configuration dict with:
                - activation.promotion_threshold: Score needed to promote
                - activation.heuristic_weights: Custom weights
            db: ArangoDB database for graph queries
            embedding_service: For edge novelty computation
            enable_importance_training: Enable training data collection
        """
        self.config = config or {}
        self.db = db
        self.embedding_service = embedding_service
        
        # Activation settings
        activation_config = self.config.get('activation', {})
        self.promotion_threshold = activation_config.get('promotion_threshold', 0.7)
        self.demotion_threshold = activation_config.get('demotion_threshold', 0.3)
        
        # Decay parameter
        self.decay_lambda = self.config.get('graph_of_thoughts', {}).get(
            'decay_lambda', 0.000012
        )
        
        # Initialize RelationImportanceScorer (ComponentScorer pattern)
        # Defaults are in the scorer, config can OVERRIDE
        self.relation_importance_scorer = RelationImportanceScorer(config=self.config)
        
        # Enable training data collection for learning importance
        if enable_importance_training:
            training_config = self.config.get('relation_training', {})
            if training_config.get('enabled', True):
                db_path = training_config.get(
                    'importance_db_path', 
                    'data/relation_importance_training.db'
                )
                self.relation_importance_scorer.enable_training_collection(db_path)
        
        # Initialize combiner
        weights = activation_config.get('heuristic_weights')
        self.combiner = HeuristicCombiner(weights=weights)
        
        # Intent-based activation weights (Phase 1G)
        # Higher weights for valuable intents (evaluation, directive), lower for ephemeral (speech_act)
        intent_config = self.config.get('sentence_intent', {})
        self.intent_weights = intent_config.get('activation_weights', {
            'fact': 1.0,
            'evaluation': 1.2,
            'opinion': 0.9,
            'speech_act': 0.3,
            'directive': 1.1
        })
        
        logger.info(f"✅ ActivationScorer initialized")
        logger.info(f"   Promotion threshold: {self.promotion_threshold}")
        logger.info(f"   Demotion threshold: {self.demotion_threshold}")
        logger.info(f"   RelationImportanceScorer: enabled")
        logger.info(f"   Intent weights: {self.intent_weights}")
    
    async def score(self, candidate: CandidateEdge) -> ActivationResult:
        """
        Score a candidate edge for promotion.
        
        Args:
            candidate: CandidateEdge to score
        
        Returns:
            ActivationResult with score and decision
        """
        scores = {}
        
        # Compute all 12 features
        scores["ego_score"] = self._compute_ego_score(candidate)
        scores["edge_evidence_count"] = self._compute_evidence_count(candidate)
        scores["recency_weight"] = self._compute_recency(candidate)
        scores["frequency_rate"] = self._compute_frequency_rate(candidate)
        scores["session_diversity"] = self._compute_session_diversity(candidate)
        scores["avg_sentiment"] = self._compute_avg_sentiment(candidate)
        scores["relation_importance"] = self._compute_relation_importance(candidate)
        scores["contradiction_score"] = self._compute_contradiction_score(candidate)
        scores["node_graph_proximity"] = await self._compute_graph_proximity(candidate)
        scores["promote_count"] = self._compute_promote_count(candidate)
        scores["demote_count"] = self._compute_demote_count(candidate)
        scores["edge_novelty"] = await self._compute_edge_novelty(candidate)
        
        # Combine scores
        activation_score = self.combiner.combine(scores)
        
        # Apply intent-based weight (Phase 1G)
        # Extract intent from aggregated_features (set by RelationExtractor)
        intent = candidate.aggregated_features.get('intent', 'fact') if candidate.aggregated_features else 'fact'
        intent_weight = self.intent_weights.get(intent, 1.0)
        
        if intent != 'fact':
            logger.debug(f"   Applying intent weight: {intent} → {intent_weight}x (base_score={activation_score:.3f})")
        
        activation_score = activation_score * intent_weight
        
        # Clamp to [0, 1] after applying intent weight
        activation_score = max(0.0, min(1.0, activation_score))
        
        # Determine decision
        decision = self._get_decision(activation_score)
        
        # Compute confidence
        confidence = self._compute_confidence(scores, activation_score)
        
        result = ActivationResult(
            activation_score=activation_score,
            component_scores=scores,
            decision=decision,
            confidence=confidence,
            metadata={
                "candidate_id": candidate.candidate_id,
                "predicate": candidate.predicate
            }
        )
        
        # Log promotion outcome for training (learns relation importance)
        self._log_promotion_outcome(candidate, result)
        
        # Log scoring details
        logger.info(f"⚡ ActivationScorer: Scored edge {candidate.candidate_id}")
        logger.info(f"   Relation: {candidate.subject_entity_id} --[{candidate.predicate}]--> {candidate.object_entity_id}")
        logger.info(f"   Activation Score: {activation_score:.3f} → Decision: {decision.upper()}")
        logger.info(f"   Top Features: ego={scores['ego_score']:.2f}, evidence={scores['edge_evidence_count']:.2f}, importance={scores['relation_importance']:.2f}")
        
        return result
    
    def _log_promotion_outcome(self, candidate: CandidateEdge, result: ActivationResult):
        """Log promotion/demotion outcome for training relation importance."""
        try:
            self.relation_importance_scorer.log_promotion_outcome(
                relation_type=candidate.predicate,
                was_promoted=(result.decision == "promote"),
                activation_score=result.activation_score,
                user_id=candidate.user_id,
                edge_evidence_count=len(candidate.supporting_mentions)
            )
        except Exception as e:
            # Non-fatal - training data collection shouldn't break scoring
            logger.debug(f"Failed to log promotion outcome: {e}")
    
    def _compute_ego_score(self, candidate: CandidateEdge) -> float:
        """Feature 1: Max ego score of supporting mentions."""
        if not candidate.supporting_mentions:
            return 0.5
        ego_scores = [m.ego for m in candidate.supporting_mentions if m.ego > 0]
        if not ego_scores:
            return 0.5
        max_ego = max(ego_scores)
        avg_ego = sum(ego_scores) / len(ego_scores)
        # Blend: prioritize max but consider average
        # 70% max (captures peak importance)
        # 30% avg (captures consistency)
        return 0.7*max_ego + 0.3*avg_ego

        
    def _compute_evidence_count(self, candidate: CandidateEdge) -> float:
        """Feature 2: Normalized evidence count."""
        count = len(candidate.supporting_mentions)
        # Normalize: 1 mention = 0.2, 5+ mentions = 1.0
        return min(1.0, count / 5.0)
    
    def _compute_recency(self, candidate: CandidateEdge) -> float:
        """Feature 3: Recency weight using exponential decay."""
        # Use last_fired_at from CandidateEdge schema
        last_seen = candidate.last_fired_at
        if not last_seen:
            return 0.5
        
        now = datetime.utcnow()
        if isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
            last_seen = last_seen.replace(tzinfo=None)
        
        hours_ago = (now - last_seen).total_seconds() / 3600
        
        # Exponential decay: exp(-λ * hours)
        return math.exp(-self.decay_lambda * hours_ago)
    
    def _compute_frequency_rate(self, candidate: CandidateEdge) -> float:
        """Feature 4: Mentions per week."""
        # Use first_seen from CandidateEdge schema
        first_seen = candidate.first_seen
        if not candidate.supporting_mentions or not first_seen:
            return 0.0
        
        now = datetime.utcnow()
        if isinstance(first_seen, str):
            first_seen = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
            first_seen = first_seen.replace(tzinfo=None)
        
        weeks = max(1, (now - first_seen).days / 7)
        mentions_per_week = len(candidate.supporting_mentions) / weeks
        
        # Normalize: 1/week = 0.5, 5+/week = 1.0
        return min(1.0, mentions_per_week / 5.0)
    
    def _compute_session_diversity(self, candidate: CandidateEdge) -> float:
        """Feature 5: Count of distinct sessions."""
        # Use distinct_session_count from CandidateEdge schema
        distinct_sessions = candidate.distinct_session_count
        if not distinct_sessions:
            features = candidate.aggregated_features or {}
            distinct_sessions = features.get("distinct_session_count", 1)
        
        # Normalize: 1 session = 0.2, 5+ sessions = 1.0
        return min(1.0, distinct_sessions / 5.0)
    
    def _compute_avg_sentiment(self, candidate: CandidateEdge) -> float:
        """Feature 6: Average sentiment of mentions."""
        features = candidate.aggregated_features or {}
        avg_sentiment = features.get("avg_sentiment", 0.5)
        
        # Sentiment is already 0-1, but we want intensity
        # Transform: 0.5 (neutral) = 0.3, 0 or 1 (extreme) = 1.0
        intensity = abs(avg_sentiment - 0.5) * 2
        return 0.3 + intensity * 0.7
    
    def _compute_relation_importance(self, candidate: CandidateEdge) -> float:
        """Feature 7: Importance based on relation type (via RelationImportanceScorer)."""
        result = self.relation_importance_scorer.score(candidate.predicate)
        return result.score
    
    def _compute_contradiction_score(self, candidate: CandidateEdge) -> float:
        """Feature 8: Penalty from contradictions."""
        features = candidate.aggregated_features or {}
        contradiction_count = features.get("contradiction_count", 0)
        
        # More contradictions = higher penalty
        # 0 contradictions = 0.0, 3+ = 1.0
        return min(1.0, contradiction_count / 3.0)
    
    async def _compute_graph_proximity(self, candidate: CandidateEdge) -> float:
        """Feature 9: Distance to existing KG nodes."""
        if not self.db:
            return 0.5
        
        try:
            # Check if subject and object entities exist in the KG
            subject_exists = await self._entity_in_kg(candidate.subject_entity_id)
            object_exists = await self._entity_in_kg(candidate.object_entity_id)
            
            if subject_exists and object_exists:
                return 1.0  # Both endpoints exist
            elif subject_exists or object_exists:
                return 0.7  # One endpoint exists
            else:
                return 0.3  # Neither exists (new subgraph)
                
        except Exception as e:
            logger.debug(f"Graph proximity check failed: {e}")
            return 0.5
    
    async def _entity_in_kg(self, entity_id: str) -> bool:
        """Check if entity has edges in the KG."""
        if not entity_id or not self.db:
            return False
        
        try:
            query = """
            FOR e IN thought_edges
            FILTER e._from == @entity_id OR e._to == @entity_id
            LIMIT 1
            RETURN 1
            """
            cursor = self.db.aql.execute(query, bind_vars={"entity_id": entity_id})
            return len(list(cursor)) > 0
        except:
            return False
    
    def _compute_promote_count(self, candidate: CandidateEdge) -> float:
        """Feature 10: Historical promotion attempts."""
        features = candidate.aggregated_features or {}
        promote_count = features.get("promote_count", 0)
        
        # Normalize: 0 = 0.0, 3+ = 1.0
        return min(1.0, promote_count / 3.0)
    
    def _compute_demote_count(self, candidate: CandidateEdge) -> float:
        """Feature 11: Historical demotion count (penalty)."""
        features = candidate.aggregated_features or {}
        demote_count = features.get("demote_count", 0)
        
        # Normalize: 0 = 0.0, 3+ = 1.0
        return min(1.0, demote_count / 3.0)
    
    async def _compute_edge_novelty(self, candidate: CandidateEdge) -> float:
        """Feature 12: Novelty compared to existing edges."""
        if not self.db:
            return 0.5
        
        try:
            # Check if similar edge already exists
            query = """
            FOR e IN thought_edges
            FILTER e._from == @from AND e._to == @to
            LIMIT 1
            RETURN e
            """
            cursor = self.db.aql.execute(query, bind_vars={
                "from": f"entities/{candidate.subject_entity_id}",
                "to": f"entities/{candidate.object_entity_id}"
            })
            
            existing = list(cursor)
            if existing:
                # Edge exists - low novelty
                return 0.2
            else:
                # New edge - high novelty
                return 0.8
                
        except Exception as e:
            logger.debug(f"Edge novelty check failed: {e}")
            return 0.5
    
    def _get_decision(self, activation_score: float) -> str:
        """Determine promotion decision based on score."""
        if activation_score >= self.promotion_threshold:
            return "promote"
        elif activation_score <= self.demotion_threshold:
            return "demote"
        else:
            return "keep"
    
    def _compute_confidence(
        self,
        scores: Dict[str, float],
        activation_score: float
    ) -> float:
        """
        Compute confidence in the decision.
        
        Based on:
        - Distance from thresholds
        - Variance of component scores
        """
        # Distance from nearest threshold
        if activation_score >= self.promotion_threshold:
            distance = activation_score - self.promotion_threshold
        elif activation_score <= self.demotion_threshold:
            distance = self.demotion_threshold - activation_score
        else:
            distance = min(
                activation_score - self.demotion_threshold,
                self.promotion_threshold - activation_score
            )
        
        # Normalize distance to confidence
        threshold_confidence = min(1.0, distance * 5)
        
        # Variance of scores (lower variance = higher confidence)
        score_values = list(scores.values())
        if score_values:
            mean = sum(score_values) / len(score_values)
            variance = sum((s - mean) ** 2 for s in score_values) / len(score_values)
            variance_confidence = 1.0 - min(1.0, variance * 4)
        else:
            variance_confidence = 0.5
        
        # Combine
        return 0.6 * threshold_confidence + 0.4 * variance_confidence
    
    async def batch_score(
        self,
        candidates: List[CandidateEdge]
    ) -> List[ActivationResult]:
        """Score multiple candidates."""
        results = []
        for candidate in candidates:
            result = await self.score(candidate)
            results.append(result)
        return results

