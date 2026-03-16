from typing import Dict, Any
from ml.component_scorers.base import ComponentScorer, ScorerResult


class EngagementScorer(ComponentScorer):
    """
    Calculates an engagement score based on conversation dynamics,
    such as user response length, follow-up questions, and elaboration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        engagement_config = config.get('ego_scoring', {}).get('engagement', {})
        
        # 🔄 HYPERPARAMETER - Being LEARNED by LightGBM in Phase 1.5 (NOW!)
        # Current: Hardcoded internal weights for combining engagement signals
        # Future: LightGBM learns optimal weights from bootstrap training data
        # These weights determine how much each signal contributes to engagement score
        self.response_length_weight = engagement_config.get('response_length_weight', 0.4)
        self.followup_count_weight = engagement_config.get('followup_count_weight', 0.3)
        self.elaboration_score_weight = engagement_config.get('elaboration_score_weight', 0.3)
        
        # ⚙️ STAYS - Normalization constants for engagement signals
        # Used to scale raw values to [0, 1] range
        self.max_response_length = engagement_config.get('max_response_length', 200)
        self.max_followup_count = engagement_config.get('max_followup_count', 5)
    
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Score engagement for a memory.
        Requires 'user_response_length', 'followup_count', 'elaboration_score' in memory.
        
        OPTIMIZATION: Uses adaptive baseline for high-importance facts.
        Short factual statements ("I have 1 cat") shouldn't be penalized.
        """
        user_response_length = memory.get('user_response_length', 0)
        followup_count = memory.get('followup_count', 0)
        elaboration_score = memory.get('elaboration_score', 0.0) # LLM-based, 0-1
        
        # Get explicit importance if available (from kwargs)
        explicit_importance = kwargs.get('explicit_importance', 0.5)
        
        # ADAPTIVE BASELINE: High-importance facts get a baseline boost
        # This prevents short but important statements from being penalized
        if explicit_importance >= 0.7:
            # High-importance facts (identity, family, health, pets)
            # Baseline: 0.5 (instead of 0.0)
            # Rationale: "I have 1 cat" is important even if response is short
            baseline = 0.5
            response_weight = 0.5  # Response length matters less
        else:
            # Low-importance chat (casual conversation)
            # Baseline: 0.0
            # Rationale: Engagement matters more for casual chat
            baseline = 0.0
            response_weight = 1.0  # Response length matters more
        
        # Normalize response length with adaptive weight
        normalized_response_length = min(1.0, user_response_length / self.max_response_length)
        
        # Normalize followup count
        normalized_followup_count = min(1.0, followup_count / self.max_followup_count)
        
        # Combine with internal weights and adaptive baseline
        engagement_score = baseline + (
            self.response_length_weight * normalized_response_length * response_weight +
            self.followup_count_weight * normalized_followup_count +
            self.elaboration_score_weight * elaboration_score
        )
        
        # Ensure score is within [0, 1]
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        return ScorerResult(
            score=engagement_score,
            metadata={
                "normalized_response_length": normalized_response_length,
                "normalized_followup_count": normalized_followup_count,
                "elaboration_score": elaboration_score,
                "explicit_importance": explicit_importance,
                "baseline_applied": baseline,
                "response_weight": response_weight
            }
        )
