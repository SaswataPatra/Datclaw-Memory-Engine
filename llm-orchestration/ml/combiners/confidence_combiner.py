from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfidenceCombiner:
    """
    Combines various confidence signals and routing rules to determine
    the final action for a memory (auto-store, active learning, discard).
    Also applies penalties for PII or semantic inconsistencies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2 (per-user risk tolerance)
        # Thresholds for routing decisions (auto-store vs. ask user vs. discard)
        # Current: Conservative (high threshold for auto-store to avoid false positives)
        # Future: Learned per-user (some users trust AI more, others want more control)
        self.auto_store_threshold = config.get('ego_scoring', {}).get('auto_store_confidence_threshold', 0.85)
        self.active_learning_threshold = config.get('ego_scoring', {}).get('active_learning_confidence_threshold', 0.60)
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2 (per-user privacy/accuracy preference)
        # Penalties applied when PII or semantic inconsistencies are detected
        # Current: Moderate penalties (0.3 for PII, 0.2 for inconsistency)
        # Future: Learned per-user (privacy-conscious users: higher PII penalty)
        self.pii_penalty = config.get('ego_scoring', {}).get('pii_penalty', 0.3)
        self.inconsistency_penalty = config.get('ego_scoring', {}).get('inconsistency_penalty', 0.2)
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2
        # Weights for combining multiple confidence signals into a single score
        # Current: Prioritize extractor (0.5) > LLM (0.3) > semantic consistency (0.2)
        # Future: Learned from user feedback (which signal is most reliable?)
        self.confidence_weights = config.get('ego_scoring', {}).get('confidence_weights', {
            'extractor_confidence': 0.5,
            'llm_confidence': 0.3,
            'semantic_consistency_confidence': 0.2
        })
        
        logger.info("ConfidenceCombiner initialized.")
    
    def combine(
        self,
        ego_score: float,
        extractor_confidence: float,
        llm_confidence: float,
        is_semantically_consistent: bool,
        has_pii: bool,
        user_engagement_score: float = 0.0, # From EngagementScorer
        **kwargs
    ) -> Dict[str, Any]:
        """
        Combines confidence signals and applies routing rules.
        
        Args:
            ego_score: The ego score from the LightGBMCombiner.
            extractor_confidence: Confidence from the memory extraction model.
            llm_confidence: Confidence from the LLM (e.g., how sure it is about the memory).
            is_semantically_consistent: True if memory is consistent with existing knowledge.
            has_pii: True if PII was detected in the memory.
            user_engagement_score: Score from EngagementScorer.
        
        Returns:
            A dictionary with 'final_confidence', 'routing_decision', and 'breakdown'.
        """
        
        # 1. Calculate combined confidence
        combined_confidence = (
            self.confidence_weights.get('extractor_confidence', 0) * extractor_confidence +
            self.confidence_weights.get('llm_confidence', 0) * llm_confidence +
            self.confidence_weights.get('semantic_consistency_confidence', 0) * (1.0 if is_semantically_consistent else 0.0)
        )
        
        # Normalize combined confidence if weights don't sum to 1
        total_weights = sum(self.confidence_weights.values())
        if total_weights > 0:
            combined_confidence /= total_weights
        
        # Start with base confidence
        final_confidence = combined_confidence
        breakdown = {
            'base_confidence': combined_confidence,
            'penalties': {},
            'boosts': {}
        }
        
        # 2. Apply penalties
        if has_pii:
            final_confidence -= self.pii_penalty
            breakdown['penalties']['pii_detected'] = self.pii_penalty
            logger.warning(f"PII detected, applying penalty: {self.pii_penalty}")
        
        if not is_semantically_consistent:
            final_confidence -= self.inconsistency_penalty
            breakdown['penalties']['semantic_inconsistency'] = self.inconsistency_penalty
            logger.warning(f"Semantic inconsistency detected, applying penalty: {self.inconsistency_penalty}")
        
        # Clip final confidence to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # 3. Determine routing decision
        routing_decision = 'discard'
        if final_confidence >= self.auto_store_threshold and ego_score >= self.config.get('shadow_tier', {}).get('require_confirmation_threshold', 0.75):
            routing_decision = 'auto_store' # Auto-store to Tier 1
        elif final_confidence >= self.active_learning_threshold:
            routing_decision = 'active_learning' # Suggest to user for feedback (Shadow Tier)
        
        # If ego_score is low, even high confidence might not lead to auto_store
        if ego_score < self.config.get('ego_scoring', {}).get('thresholds', {}).get('tier3', 0.20):
            routing_decision = 'discard' # Too low ego score, discard regardless of confidence
        
        return {
            'final_confidence': final_confidence,
            'routing_decision': routing_decision,
            'breakdown': breakdown
        }
