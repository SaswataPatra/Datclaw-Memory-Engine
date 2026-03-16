"""
Unit tests for ActivationScorer.

Tests:
1. Feature computation (all 12 features)
2. Heuristic combiner
3. Decision logic
4. Confidence computation
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from core.graph.activation_scorer import (
    ActivationScorer,
    ActivationResult,
    HeuristicCombiner,
)
from core.graph.relation_importance_scorer import (
    RelationImportanceScorer,
    RelationImportanceTrainingCollector,
)
from core.graph.schemas import CandidateEdge, SupportingMention


class TestHeuristicCombiner:
    """Test HeuristicCombiner."""
    
    def test_init_default_weights(self):
        """Test initialization with default weights."""
        combiner = HeuristicCombiner()
        
        assert len(combiner.weights) == 12
        assert combiner.weights["ego_score"] == 0.15
        assert combiner.weights["contradiction_score"] == -0.10
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        custom = {"ego_score": 0.5, "edge_evidence_count": 0.5}
        combiner = HeuristicCombiner(weights=custom)
        
        assert combiner.weights["ego_score"] == 0.5
    
    def test_combine_all_high(self):
        """Test combining all high scores."""
        combiner = HeuristicCombiner()
        
        scores = {
            "ego_score": 1.0,
            "edge_evidence_count": 1.0,
            "recency_weight": 1.0,
            "frequency_rate": 1.0,
            "session_diversity": 1.0,
            "avg_sentiment": 1.0,
            "relation_importance": 1.0,
            "contradiction_score": 0.0,  # Low = good
            "node_graph_proximity": 1.0,
            "promote_count": 1.0,
            "demote_count": 0.0,  # Low = good
            "edge_novelty": 1.0,
        }
        
        result = combiner.combine(scores)
        
        # Should be high (close to 1.0)
        assert result > 0.8
    
    def test_combine_all_low(self):
        """Test combining all low scores."""
        combiner = HeuristicCombiner()
        
        scores = {
            "ego_score": 0.0,
            "edge_evidence_count": 0.0,
            "recency_weight": 0.0,
            "frequency_rate": 0.0,
            "session_diversity": 0.0,
            "avg_sentiment": 0.0,
            "relation_importance": 0.0,
            "contradiction_score": 1.0,  # High = bad
            "node_graph_proximity": 0.0,
            "promote_count": 0.0,
            "demote_count": 1.0,  # High = bad
            "edge_novelty": 0.0,
        }
        
        result = combiner.combine(scores)
        
        # Should be low (close to 0.0)
        assert result < 0.2
    
    def test_combine_with_penalties(self):
        """Test that negative weights apply penalties."""
        combiner = HeuristicCombiner()
        
        # Good scores but high contradictions
        scores = {
            "ego_score": 0.8,
            "edge_evidence_count": 0.8,
            "contradiction_score": 1.0,  # High penalty
            "demote_count": 1.0,  # High penalty
        }
        
        result = combiner.combine(scores)
        
        # Penalties should reduce the score
        assert result < 0.5


class TestActivationScorer:
    """Test ActivationScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create ActivationScorer instance."""
        return ActivationScorer(config={
            "activation": {
                "promotion_threshold": 0.7,
                "demotion_threshold": 0.3
            }
        })
    
    @pytest.fixture
    def sample_candidate(self):
        """Create a sample CandidateEdge."""
        now = datetime.utcnow()
        return CandidateEdge(
            user_id="user123",
            subject_entity_id="entity_sarah",
            subject_span={"text": "Sarah"},
            predicate="sister_of",
            object_entity_id="entity_user",
            object_span={"text": "me"},
            first_seen=now - timedelta(days=7),
            last_fired_at=now - timedelta(hours=1),
            edge_evidence_count=3,
            distinct_session_count=3,
            recency_weight=0.9,
            frequency_rate=0.6,
            supporting_mentions=[
                SupportingMention(
                    mem_id="mem1",
                    observed_at=now - timedelta(days=7),
                    ego=0.85
                ),
                SupportingMention(
                    mem_id="mem2",
                    observed_at=now - timedelta(days=3),
                    ego=0.75
                ),
                SupportingMention(
                    mem_id="mem3",
                    observed_at=now - timedelta(hours=1),
                    ego=0.60
                ),
            ],
            aggregated_features={
                "distinct_session_count": 3,
                "avg_sentiment": 0.7,
                "contradiction_count": 0,
                "promote_count": 1,
                "demote_count": 0
            }
        )
    
    def test_init(self, scorer):
        """Test scorer initialization."""
        assert scorer is not None
        assert scorer.promotion_threshold == 0.7
        assert scorer.demotion_threshold == 0.3
    
    @pytest.mark.asyncio
    async def test_score_high_quality_candidate(self, scorer, sample_candidate):
        """Test scoring a high-quality candidate."""
        result = await scorer.score(sample_candidate)
        
        assert isinstance(result, ActivationResult)
        assert result.activation_score > 0.5
        assert result.decision in ["promote", "keep", "demote"]
        assert len(result.component_scores) == 12
    
    def test_compute_ego_score(self, scorer, sample_candidate):
        """Test ego score computation."""
        score = scorer._compute_ego_score(sample_candidate)
        
        # Should be max of mentions (0.85)
        assert score == 0.85
    
    def test_compute_ego_score_empty(self, scorer):
        """Test ego score with no mentions."""
        candidate = CandidateEdge(
            user_id="user123",
            subject_entity_id="e1",
            subject_span="A",
            predicate="knows",
            object_entity_id="e2",
            object_span="B",
            supporting_mentions=[]
        )
        
        score = scorer._compute_ego_score(candidate)
        assert score == 0.5  # Default
    
    def test_compute_evidence_count(self, scorer, sample_candidate):
        """Test evidence count computation."""
        score = scorer._compute_evidence_count(sample_candidate)
        
        # 3 mentions / 5 = 0.6
        assert score == 0.6
    
    def test_compute_recency(self, scorer, sample_candidate):
        """Test recency computation."""
        score = scorer._compute_recency(sample_candidate)
        
        # Last seen 1 hour ago, should be high
        assert score > 0.9
    
    def test_compute_frequency_rate(self, scorer, sample_candidate):
        """Test frequency rate computation."""
        score = scorer._compute_frequency_rate(sample_candidate)
        
        # 3 mentions over 1 week = 3/week, normalized
        assert 0.5 < score < 0.7
    
    def test_compute_session_diversity(self, scorer, sample_candidate):
        """Test session diversity computation."""
        score = scorer._compute_session_diversity(sample_candidate)
        
        # 3 sessions / 5 = 0.6
        assert score == 0.6
    
    def test_compute_avg_sentiment(self, scorer, sample_candidate):
        """Test sentiment computation."""
        score = scorer._compute_avg_sentiment(sample_candidate)
        
        # 0.7 sentiment = 0.4 intensity, score = 0.3 + 0.4*0.7 = 0.58
        assert 0.5 < score < 0.7
    
    def test_compute_relation_importance(self, scorer, sample_candidate):
        """Test relation importance computation."""
        score = scorer._compute_relation_importance(sample_candidate)
        
        # "sister_of" = 0.95
        assert score == 0.95
    
    def test_compute_contradiction_score(self, scorer, sample_candidate):
        """Test contradiction score computation."""
        score = scorer._compute_contradiction_score(sample_candidate)
        
        # 0 contradictions = 0.0
        assert score == 0.0
    
    def test_compute_promote_count(self, scorer, sample_candidate):
        """Test promote count computation."""
        score = scorer._compute_promote_count(sample_candidate)
        
        # 1 promote / 3 = 0.33
        assert abs(score - 0.33) < 0.1
    
    def test_compute_demote_count(self, scorer, sample_candidate):
        """Test demote count computation."""
        score = scorer._compute_demote_count(sample_candidate)
        
        # 0 demotes = 0.0
        assert score == 0.0
    
    def test_get_decision_promote(self, scorer):
        """Test promotion decision."""
        decision = scorer._get_decision(0.8)
        assert decision == "promote"
    
    def test_get_decision_keep(self, scorer):
        """Test keep decision."""
        decision = scorer._get_decision(0.5)
        assert decision == "keep"
    
    def test_get_decision_demote(self, scorer):
        """Test demotion decision."""
        decision = scorer._get_decision(0.2)
        assert decision == "demote"


class TestRelationImportanceScorer:
    """Test RelationImportanceScorer (ComponentScorer pattern)."""
    
    @pytest.fixture
    def scorer(self):
        """Create a RelationImportanceScorer instance."""
        return RelationImportanceScorer()
    
    def test_family_relations_high(self, scorer):
        """Test that family relations have high importance."""
        family_relations = ["sister_of", "brother_of", "parent_of", "spouse_of"]
        
        for rel in family_relations:
            result = scorer.score(rel)
            assert result.score >= 0.85, f"{rel} should have high importance"
    
    def test_professional_relations_medium(self, scorer):
        """Test that professional relations have medium importance."""
        professional_relations = ["works_at", "colleague_of", "manages"]
        
        for rel in professional_relations:
            result = scorer.score(rel)
            assert 0.6 <= result.score <= 0.85, f"{rel} should have medium importance"
    
    def test_unknown_relation_default(self, scorer):
        """Test that unknown relations get default importance (0.5)."""
        result = scorer.score("unknown_relation")
        assert result.score == 0.5
        assert result.metadata.get("is_default") is True
    
    def test_config_override(self):
        """Test that config can override default importance."""
        config = {
            "activation": {
                "relation_importance_map": {
                    "custom_relation": 0.99
                }
            }
        }
        scorer = RelationImportanceScorer(config=config)
        
        result = scorer.score("custom_relation")
        assert result.score == 0.99
    
    def test_add_relation_type(self, scorer):
        """Test adding a new relation type dynamically."""
        scorer.add_relation_type("new_relation", 0.77)
        
        result = scorer.score("new_relation")
        assert result.score == 0.77


class TestRelationImportanceTraining:
    """Test training data collection for relation importance."""
    
    @pytest.fixture
    def collector(self, tmp_path):
        """Create a collector with temp DB."""
        db_path = str(tmp_path / "test_importance.db")
        return RelationImportanceTrainingCollector(db_path)
    
    def test_log_outcome(self, collector):
        """Test logging a promotion outcome."""
        collector.log_outcome(
            relation_type="works_at",
            was_promoted=True,
            activation_score=0.85,
            user_id="user123",
            edge_evidence_count=3
        )
        
        stats = collector.get_stats()
        assert stats["total_outcomes"] == 1
    
    def test_promotion_rates(self, collector):
        """Test computing promotion rates."""
        # Log multiple outcomes
        for i in range(10):
            collector.log_outcome(
                relation_type="sister_of",
                was_promoted=True,  # Always promoted
                activation_score=0.9,
                user_id="user123"
            )
        
        for i in range(5):
            collector.log_outcome(
                relation_type="knows",
                was_promoted=(i < 1),  # Only 1/5 promoted
                activation_score=0.5,
                user_id="user123"
            )
        
        rates = collector.get_promotion_rates()
        
        assert "sister_of" in rates
        assert rates["sister_of"]["rate"] == 1.0  # 100% promoted
        assert rates["knows"]["rate"] == 0.2  # 20% promoted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

