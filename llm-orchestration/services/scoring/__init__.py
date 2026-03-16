"""
Scoring Module

Handles all ML-based memory scoring operations:
- Component scorers (novelty, frequency, sentiment, importance, engagement)
- Score combination (LightGBM or weighted average)
- Confidence calculation
- Tier determination

Usage:
    from services.scoring import MLScorer
    
    scorer = MLScorer(
        ml_scorers=scorers_dict,
        ml_combiner=lightgbm_combiner,
        confidence_combiner=conf_combiner,
        classifier_manager=classifier_mgr,
        embedding_service=embedding_svc,
        ml_executor=executor,
        regex_fallback=fallback,
        classifier_type="hf_api"
    )
    
    ego_score, confidence, triggers = await scorer.score_memory(
        user_id="user123",
        user_message="I love Python",
        assistant_response="That's great!"
    )
"""

from .ml_scorer import MLScorer
from .pii_detector import PIIDetector

__all__ = ['MLScorer', 'PIIDetector']

