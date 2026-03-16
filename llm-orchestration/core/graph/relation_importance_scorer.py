"""
DAPPY Relation Importance Scorer

A ComponentScorer that maps relation types to importance values.
Follows the same pattern as ExplicitImportanceScorer for memory labels.

Phase 1: Heuristic defaults (configurable)
Phase 2: Learned from promotion/demotion outcomes via LightGBM

The importance values themselves can be learned over time by:
1. Collecting training data when edges are promoted/demoted
2. Training a model to predict which relations lead to promotions
3. Updating the importance map based on learned weights
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import sqlite3
import uuid

logger = logging.getLogger(__name__)


@dataclass
class RelationScorerResult:
    """Result from relation importance scoring."""
    score: float
    metadata: Dict[str, Any]


class RelationImportanceScorer:
    """
    Scores relation types by their importance to the user's knowledge graph.
    
    Like ExplicitImportanceScorer, this maps relation_type → importance.
    
    🔄 HYPERPARAMETERS - Will be LEARNED in Phase 2-3:
    - Current: Universal importance map based on relation type
    - Future: Learned per-user (some users prioritize work > family)
    - Phase 2: Learn from promotion/demotion feedback
    - Phase 3: Context-aware importance (work context: professional=1.0)
    """
    
    # Default importance map (used if config doesn't override)
    # These are heuristic starting points
    DEFAULT_IMPORTANCE_MAP = {
        # Family (high importance - core identity)
        "sister_of": 0.95,
        "brother_of": 0.95,
        "parent_of": 0.95,
        "child_of": 0.95,
        "spouse_of": 0.98,
        "partner_of": 0.95,
        "relative_of": 0.85,
        "family_of": 0.90,
        "sibling_of": 0.95,
        
        # Professional (medium-high)
        "works_at": 0.80,
        "works_with": 0.70,
        "colleague_of": 0.65,
        "manages": 0.75,
        "employed_by": 0.75,
        "reports_to": 0.70,
        "ceo_of": 0.85,
        "founded": 0.85,
        
        # Personal preferences (medium)
        "likes": 0.60,
        "dislikes": 0.60,
        "prefers": 0.65,
        "loves": 0.75,
        "hates": 0.70,
        "enjoys": 0.60,
        "avoids": 0.65,
        "allergic_to": 0.85,  # Health-critical
        
        # Social (medium)
        "friend_of": 0.70,
        "knows": 0.40,
        "met": 0.35,
        "dated": 0.65,
        
        # Temporal/Corrections (high - important for consistency)
        "contradicts": 0.85,
        "supersedes": 0.80,
        "evolves_to": 0.75,
        "replaces": 0.80,
        "updates": 0.75,
        
        # Factual (medium)
        "located_at": 0.55,
        "lives_in": 0.70,
        "born_in": 0.80,
        "from": 0.60,
        "owns": 0.60,
        "has": 0.50,
        "uses": 0.45,
        "studied_at": 0.70,
        "graduated_from": 0.75,
        
        # Actions (lower - often transient)
        "visited": 0.40,
        "mentioned": 0.30,
        "asked_about": 0.35,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scorer.
        
        Args:
            config: Configuration dict. The importance map can be overridden via:
                    config['activation']['relation_importance_map']
                    OR config['relation_classification']['relation_importance']
        """
        self.config = config or {}
        
        # Load importance map from config (config OVERRIDES defaults)
        self.importance_map = self._load_importance_map()
        
        # Default for unknown relations
        self.default_importance = self.importance_map.get('default', 0.5)
        
        # Training data collector (for learning)
        self._training_collector = None
        
        logger.info(
            f"✅ RelationImportanceScorer initialized with "
            f"{len(self.importance_map)} relation types"
        )
    
    def _load_importance_map(self) -> Dict[str, float]:
        """
        Load importance map from config, falling back to defaults.
        
        Config can override individual relations - we merge with defaults.
        """
        # Start with defaults
        importance_map = self.DEFAULT_IMPORTANCE_MAP.copy()
        
        # Try activation config first
        activation_config = self.config.get('activation', {})
        config_map = activation_config.get('relation_importance_map', {})
        
        # If not in activation, try relation_classification
        if not config_map:
            relation_config = self.config.get('relation_classification', {})
            config_map = relation_config.get('relation_importance', {})
        
        # Merge config overrides (config wins)
        if config_map:
            importance_map.update(config_map)
            logger.info(f"Loaded {len(config_map)} relation importance overrides from config")
        
        return importance_map
    
    def score(self, relation_type: str) -> RelationScorerResult:
        """
        Score the importance of a relation type.
        
        Args:
            relation_type: The relation type (e.g., "sister_of", "works_at")
        
        Returns:
            RelationScorerResult with importance score
        """
        # Normalize relation type
        relation_type = relation_type.lower().strip()
        
        # Lookup importance
        importance = self.importance_map.get(relation_type, self.default_importance)
        
        return RelationScorerResult(
            score=importance,
            metadata={
                "relation_type": relation_type,
                "mapped_importance": importance,
                "is_default": relation_type not in self.importance_map
            }
        )
    
    def get_category_importance(self, relation_type: str) -> float:
        """
        Get importance by inferring category from relation type.
        
        This is a fallback when exact relation isn't in the map.
        """
        relation_lower = relation_type.lower()
        
        # Infer category from relation name
        if any(fam in relation_lower for fam in ["sister", "brother", "parent", "child", "spouse", "family"]):
            return 0.90
        elif any(work in relation_lower for work in ["work", "employ", "manage", "colleague", "report"]):
            return 0.75
        elif any(pref in relation_lower for pref in ["like", "love", "hate", "prefer", "enjoy", "avoid"]):
            return 0.65
        elif any(temp in relation_lower for temp in ["contradict", "supersede", "evolve", "replace", "update"]):
            return 0.80
        else:
            return self.default_importance
    
    def add_relation_type(self, relation_type: str, importance: float):
        """
        Add or update a relation type's importance.
        
        This is called when:
        1. A new relation type is discovered
        2. Importance is learned from training data
        """
        self.importance_map[relation_type.lower()] = importance
        logger.info(f"Updated relation importance: {relation_type} = {importance}")
    
    def enable_training_collection(self, db_path: str = "data/relation_importance_training.db"):
        """Enable training data collection for learning importance values."""
        self._training_collector = RelationImportanceTrainingCollector(db_path)
        logger.info(f"Enabled relation importance training collection: {db_path}")
    
    def log_promotion_outcome(
        self,
        relation_type: str,
        was_promoted: bool,
        activation_score: float,
        user_id: str,
        edge_evidence_count: int = 1
    ):
        """
        Log a promotion/demotion outcome for training.
        
        This data is used to learn which relations are more important.
        """
        if self._training_collector:
            self._training_collector.log_outcome(
                relation_type=relation_type,
                was_promoted=was_promoted,
                activation_score=activation_score,
                user_id=user_id,
                edge_evidence_count=edge_evidence_count
            )


class RelationImportanceTrainingCollector:
    """
    Collects training data for learning relation importance.
    
    Data collected:
    - Which relation types lead to promotions (positive labels)
    - Which relation types lead to demotions/keep (negative labels)
    - Activation scores at promotion time
    
    This data can be used to train a model that predicts:
    - P(promotion | relation_type, edge_features)
    - Optimal importance weights per relation type
    """
    
    def __init__(self, db_path: str = "data/relation_importance_training.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for training data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Promotion outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS promotion_outcomes (
                id TEXT PRIMARY KEY,
                relation_type TEXT NOT NULL,
                was_promoted INTEGER NOT NULL,
                activation_score REAL NOT NULL,
                edge_evidence_count INTEGER NOT NULL,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                current_importance REAL
            )
        """)
        
        # Aggregated stats per relation type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_stats (
                relation_type TEXT PRIMARY KEY,
                total_attempts INTEGER DEFAULT 0,
                promoted_count INTEGER DEFAULT 0,
                avg_activation_when_promoted REAL,
                avg_activation_when_kept REAL,
                last_updated TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_relation 
            ON promotion_outcomes(relation_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_user 
            ON promotion_outcomes(user_id)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized relation importance training DB: {self.db_path}")
    
    def log_outcome(
        self,
        relation_type: str,
        was_promoted: bool,
        activation_score: float,
        user_id: str,
        edge_evidence_count: int = 1,
        current_importance: float = None
    ):
        """Log a promotion outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert outcome
            cursor.execute("""
                INSERT INTO promotion_outcomes 
                (id, relation_type, was_promoted, activation_score, 
                 edge_evidence_count, user_id, timestamp, current_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                relation_type.lower(),
                1 if was_promoted else 0,
                activation_score,
                edge_evidence_count,
                user_id,
                datetime.utcnow().isoformat(),
                current_importance
            ))
            
            # Update aggregated stats
            cursor.execute("""
                INSERT INTO relation_stats (relation_type, total_attempts, promoted_count, last_updated)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(relation_type) DO UPDATE SET
                    total_attempts = total_attempts + 1,
                    promoted_count = promoted_count + ?,
                    last_updated = ?
            """, (
                relation_type.lower(),
                1 if was_promoted else 0,
                datetime.utcnow().isoformat(),
                1 if was_promoted else 0,
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to log promotion outcome: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_promotion_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get promotion rates per relation type.
        
        Returns dict: relation_type → {total, promoted, rate}
        
        This can be used to derive learned importance:
        - Higher promotion rate → higher importance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT relation_type, total_attempts, promoted_count
            FROM relation_stats
            WHERE total_attempts >= 5  -- Minimum samples
            ORDER BY promoted_count * 1.0 / total_attempts DESC
        """)
        
        results = {}
        for row in cursor.fetchall():
            relation_type, total, promoted = row
            results[relation_type] = {
                "total": total,
                "promoted": promoted,
                "rate": promoted / total if total > 0 else 0
            }
        
        conn.close()
        return results
    
    def compute_learned_importance(self, min_samples: int = 10) -> Dict[str, float]:
        """
        Compute learned importance values from training data.
        
        Uses promotion rate as a proxy for importance:
        importance = base_weight * promotion_rate + (1 - base_weight) * default
        
        Args:
            min_samples: Minimum samples needed to trust the data
        
        Returns:
            Dict of relation_type → learned_importance
        """
        rates = self.get_promotion_rates()
        
        learned = {}
        for relation_type, stats in rates.items():
            if stats["total"] >= min_samples:
                # Blend promotion rate with prior
                # High promotion rate → high importance
                learned[relation_type] = 0.3 + 0.7 * stats["rate"]
        
        return learned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training collection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM promotion_outcomes")
        total_outcomes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM relation_stats")
        unique_relations = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT relation_type, promoted_count * 1.0 / total_attempts as rate
            FROM relation_stats
            WHERE total_attempts >= 5
            ORDER BY rate DESC
            LIMIT 5
        """)
        top_promoted = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_outcomes": total_outcomes,
            "unique_relations": unique_relations,
            "top_promoted_relations": [
                {"relation": r, "rate": round(rate, 3)} 
                for r, rate in top_promoted
            ]
        }

