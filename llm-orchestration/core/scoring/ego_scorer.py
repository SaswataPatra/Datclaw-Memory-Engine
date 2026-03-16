"""
DAPPY - Time-Aware Ego Scoring Service
Calculates multi-dimensional importance score with temporal awareness
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class EgoScoreComponents:
    """
    Breakdown of ego score components for explainability
    """
    explicit_importance: float
    recency_decay: float
    frequency: float
    sentiment_intensity: float
    engagement: float
    reference_count: float
    confidence: float
    source_weight: float
    novelty: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "explicit_importance": self.explicit_importance,
            "recency_decay": self.recency_decay,
            "frequency": self.frequency,
            "sentiment_intensity": self.sentiment_intensity,
            "engagement": self.engagement,
            "reference_count": self.reference_count,
            "confidence": self.confidence,
            "source_weight": self.source_weight,
            "novelty": self.novelty
        }


@dataclass
class EgoScoreResult:
    """Result of ego scoring with breakdown"""
    ego_score: float
    tier: str  # 'tier1' | 'tier2' | 'tier3' | 'tier4'
    components: EgoScoreComponents
    timestamp: str


class RecencyCalculator:
    """
    Calculate tier-specific recency decay using exponential half-life
    
    Formula: exp(-λ * age) where λ = ln(2) / half_life
    """
    
    def __init__(self, config: Dict[str, Any]):
        recency_config = config.get('ego_scoring', {}).get('recency', {})
        
        # Tier-specific half-lives
        self.tier1_half_life_days = recency_config.get('tier1_half_life_days', 180)  # 6 months
        self.tier2_half_life_days = recency_config.get('tier2_half_life_days', 7)    # 1 week
        self.tier3_half_life_days = recency_config.get('tier3_half_life_days', 1)    # 1 day
        self.tier4_half_life_minutes = recency_config.get('tier4_half_life_minutes', 5)  # 5 minutes
        
        # Pre-compute decay constants (λ = ln(2) / half_life)
        self.tier1_lambda = math.log(2) / self.tier1_half_life_days
        self.tier2_lambda = math.log(2) / self.tier2_half_life_days
        self.tier3_lambda = math.log(2) / self.tier3_half_life_days
        self.tier4_lambda = math.log(2) / (self.tier4_half_life_minutes / (24 * 60))  # Convert to days
    
    def calculate(
        self,
        observed_at: datetime,
        current_time: Optional[datetime] = None,
        tier: str = 'tier2'
    ) -> float:
        """
        Calculate recency decay based on age and tier
        
        Returns: Value between 0 and 1 (1 = just now, 0 = very old)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Ensure both datetimes are timezone-aware for comparison
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        # Calculate age in days
        age_seconds = (current_time - observed_at).total_seconds()
        age_days = age_seconds / (24 * 3600)
        
        # Select decay constant based on tier
        if tier == 'tier1':
            decay_lambda = self.tier1_lambda
        elif tier == 'tier2':
            decay_lambda = self.tier2_lambda
        elif tier == 'tier3':
            decay_lambda = self.tier3_lambda
        elif tier == 'tier4':
            decay_lambda = self.tier4_lambda
        else:
            # Default to tier2
            decay_lambda = self.tier2_lambda
        
        # Exponential decay: exp(-λ * age)
        recency_score = math.exp(-decay_lambda * age_days)
        
        return max(0.0, min(1.0, recency_score))


class TemporalEgoScorer:
    """
    Time-aware ego scorer (Phase 1 - hand-tuned formula)
    
    In Phase 1.5, this will be replaced with ML-based classifier
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recency_calc = RecencyCalculator(config)
        
        # Load weights from config
        weights_config = config.get('ego_scoring', {}).get('weights', {})
        self.weights = {
            'explicit_importance': weights_config.get('explicit_importance', 0.20),
            'recency_decay': weights_config.get('recency_decay', 0.20),
            'frequency': weights_config.get('frequency', 0.10),
            'sentiment_intensity': weights_config.get('sentiment_intensity', 0.08),
            'engagement': weights_config.get('engagement', 0.12),
            'reference_count': weights_config.get('reference_count', 0.10),
            'confidence': weights_config.get('confidence', 0.10),
            'source_weight': weights_config.get('source_weight', 0.05),
            'novelty': weights_config.get('novelty', 0.05)
        }
        
        # Load tier thresholds
        thresholds_config = config.get('ego_scoring', {}).get('thresholds', {})
        self.thresholds = {
            'tier1': thresholds_config.get('tier1', 0.75),
            'tier2': thresholds_config.get('tier2', 0.45),
            'tier3': thresholds_config.get('tier3', 0.20)
        }
    
    def calculate(
        self,
        memory: Dict[str, Any],
        current_tier: Optional[str] = None
    ) -> EgoScoreResult:
        """
        Calculate ego score for a memory
        
        Args:
            memory: Memory dict with features
            current_tier: Current tier (for tier-specific recency decay)
        
        Returns:
            EgoScoreResult with score, tier, and component breakdown
        """
        
        # Extract features
        explicit_importance = memory.get('explicit_importance', 0.0)
        observed_at = memory.get('observed_at')
        frequency_7d = memory.get('frequency_7d', 0)
        sentiment_score = memory.get('sentiment_score', 0.0)
        user_response_length = memory.get('user_response_length', 0)
        followup_count = memory.get('followup_count', 0)
        reference_count = memory.get('reference_count', 0)
        llm_confidence = memory.get('llm_confidence', 0.5)
        source_weight = memory.get('source_weight', 0.7)
        novelty_score = memory.get('novelty_score', 0.5)
        
        # Calculate recency decay (tier-aware)
        if observed_at is None:
            observed_at = datetime.utcnow()
        elif isinstance(observed_at, str):
            observed_at = datetime.fromisoformat(observed_at)
        
        recency_decay = self.recency_calc.calculate(
            observed_at,
            tier=current_tier or 'tier2'
        )
        
        # Normalize features to [0, 1]
        frequency_norm = min(1.0, frequency_7d / 10.0)  # Cap at 10 references
        sentiment_intensity = abs(sentiment_score)
        
        # Normalize engagement (tokens + followups)
        engagement_raw = (user_response_length / 100.0) + (followup_count * 0.2)
        engagement_norm = min(1.0, engagement_raw)
        
        # Normalize reference count
        reference_norm = min(1.0, reference_count / 5.0)  # Cap at 5 references
        
        # Create components object
        components = EgoScoreComponents(
            explicit_importance=explicit_importance,
            recency_decay=recency_decay,
            frequency=frequency_norm,
            sentiment_intensity=sentiment_intensity,
            engagement=engagement_norm,
            reference_count=reference_norm,
            confidence=llm_confidence,
            source_weight=source_weight,
            novelty=novelty_score
        )
        
        # Weighted sum
        ego_score_raw = (
            self.weights['explicit_importance'] * explicit_importance +
            self.weights['recency_decay'] * recency_decay +
            self.weights['frequency'] * frequency_norm +
            self.weights['sentiment_intensity'] * sentiment_intensity +
            self.weights['engagement'] * engagement_norm +
            self.weights['reference_count'] * reference_norm +
            self.weights['confidence'] * llm_confidence +
            self.weights['source_weight'] * source_weight +
            self.weights['novelty'] * novelty_score
        )
        
        # Clip to [0, 1]
        ego_score = max(0.0, min(1.0, ego_score_raw))
        
        # Determine tier
        if ego_score >= self.thresholds['tier1']:
            tier = 'tier1'
        elif ego_score >= self.thresholds['tier2']:
            tier = 'tier2'
        elif ego_score >= self.thresholds['tier3']:
            tier = 'tier3'
        else:
            tier = 'tier4'
        
        result = EgoScoreResult(
            ego_score=ego_score,
            tier=tier,
            components=components,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.debug(
            f"Calculated ego score: {ego_score:.3f} → {tier}",
            extra={
                "memory_id": memory.get('memory_id'),
                "ego_score": ego_score,
                "tier": tier,
                "components": components.to_dict()
            }
        )
        
        return result
    
    def explain(self, result: EgoScoreResult) -> str:
        """Generate human-readable explanation of score"""
        
        lines = [
            f"Ego Score: {result.ego_score:.3f} → {result.tier.upper()}",
            "",
            "Component Breakdown:"
        ]
        
        components_dict = result.components.to_dict()
        for name, value in components_dict.items():
            weight = self.weights.get(name, 0.0)
            contribution = weight * value
            lines.append(
                f"  {name:25s}: {value:.3f} × {weight:.2f} = {contribution:.4f}"
            )
        
        return "\n".join(lines)

