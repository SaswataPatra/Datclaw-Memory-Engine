"""
DAPPY Normalization Training Data Collector

Collects normalization decisions (rule-based vs LLM) for training future ML models.
Similar to RelationTrainingCollector but focused on predicate normalization.

Phase 1E Implementation
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class NormalizationTrainingCollector:
    """
    Collects normalization training data for future ML model training.
    
    Logs:
    1. Rule-based normalization attempts
    2. LLM corrections/improvements
    3. User corrections (if available)
    4. Context and entity types
    
    This data will be used to train a specialized normalization model in Phase 2.
    """
    
    def __init__(self, db_path: str = "data/normalization_training.db"):
        """
        Initialize training data collector.
        
        Args:
            db_path: Path to SQLite database for training data
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"✅ NormalizationTrainingCollector initialized: {db_path}")
    
    def _init_db(self):
        """Initialize SQLite database schema."""
        # Create directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main normalization table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS normalizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicate_lemma TEXT NOT NULL,
                subject_type TEXT,
                object_type TEXT,
                context TEXT,
                rule_result TEXT NOT NULL,
                rule_confidence REAL NOT NULL,
                rule_method TEXT NOT NULL,
                llm_result TEXT,
                llm_confidence REAL,
                llm_reasoning TEXT,
                user_correction TEXT,
                user_id TEXT,
                memory_id TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT
            )
        """)
        
        # Index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicate 
            ON normalizations(predicate_lemma)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_types 
            ON normalizations(subject_type, object_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON normalizations(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def log_normalization(
        self,
        predicate_lemma: str,
        subject_type: Optional[str],
        object_type: Optional[str],
        context: str,
        rule_result: str,
        rule_confidence: float,
        rule_method: str,
        llm_result: Optional[str] = None,
        llm_confidence: Optional[float] = None,
        llm_reasoning: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Log a normalization decision.
        
        Args:
            predicate_lemma: Original predicate lemma
            subject_type: Entity type of subject
            object_type: Entity type of object
            context: Full sentence context
            rule_result: Canonical relation from rule-based normalization
            rule_confidence: Confidence of rule-based result
            rule_method: Method used (lexicon, type-aware, embedding, original)
            llm_result: Canonical relation from LLM (if fallback was triggered)
            llm_confidence: Confidence of LLM result
            llm_reasoning: LLM's reasoning for the normalization
            user_id: User ID for tracking
            memory_id: Memory ID for linking
            session_id: Session ID for context
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO normalizations (
                    predicate_lemma, subject_type, object_type, context,
                    rule_result, rule_confidence, rule_method,
                    llm_result, llm_confidence, llm_reasoning,
                    user_correction, user_id, memory_id, timestamp, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                predicate_lemma, subject_type, object_type, context,
                rule_result, rule_confidence, rule_method,
                llm_result, llm_confidence, llm_reasoning,
                None,  # user_correction (to be added later if user corrects)
                user_id, memory_id, datetime.utcnow().isoformat(), session_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(
                f"📝 Logged normalization: {predicate_lemma} → {rule_result} "
                f"(rule_conf={rule_confidence:.2f}, llm={'Yes' if llm_result else 'No'})"
            )
        
        except Exception as e:
            logger.error(f"Failed to log normalization: {e}")
    
    def update_user_correction(
        self,
        predicate_lemma: str,
        context: str,
        user_correction: str,
        user_id: str
    ):
        """
        Update with user correction (if user explicitly corrects a relation).
        
        This is high-value ground truth for training.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find the most recent normalization for this predicate + context
            cursor.execute("""
                UPDATE normalizations
                SET user_correction = ?
                WHERE predicate_lemma = ? 
                  AND context = ?
                  AND user_id = ?
                  AND id = (
                      SELECT id FROM normalizations
                      WHERE predicate_lemma = ? AND context = ? AND user_id = ?
                      ORDER BY timestamp DESC
                      LIMIT 1
                  )
            """, (user_correction, predicate_lemma, context, user_id,
                  predicate_lemma, context, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Updated user correction: {predicate_lemma} → {user_correction}")
        
        except Exception as e:
            logger.error(f"Failed to update user correction: {e}")
    
    def get_training_data(
        self,
        min_confidence: float = 0.7,
        include_llm_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get training data for ML model training.
        
        Args:
            min_confidence: Minimum confidence threshold
            include_llm_only: If True, only return cases where LLM was used
        
        Returns:
            List of training examples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    predicate_lemma, subject_type, object_type, context,
                    rule_result, rule_confidence, rule_method,
                    llm_result, llm_confidence, llm_reasoning,
                    user_correction
                FROM normalizations
                WHERE rule_confidence >= ?
            """
            
            if include_llm_only:
                query += " AND llm_result IS NOT NULL"
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, (min_confidence,))
            rows = cursor.fetchall()
            
            training_data = []
            for row in rows:
                training_data.append({
                    "predicate_lemma": row[0],
                    "subject_type": row[1],
                    "object_type": row[2],
                    "context": row[3],
                    "rule_result": row[4],
                    "rule_confidence": row[5],
                    "rule_method": row[6],
                    "llm_result": row[7],
                    "llm_confidence": row[8],
                    "llm_reasoning": row[9],
                    "user_correction": row[10],
                    # Ground truth: user_correction > llm_result > rule_result
                    "ground_truth": row[10] or row[7] or row[4]
                })
            
            conn.close()
            return training_data
        
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected training data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total normalizations
            cursor.execute("SELECT COUNT(*) FROM normalizations")
            total = cursor.fetchone()[0]
            
            # LLM fallback count
            cursor.execute("SELECT COUNT(*) FROM normalizations WHERE llm_result IS NOT NULL")
            llm_count = cursor.fetchone()[0]
            
            # User corrections count
            cursor.execute("SELECT COUNT(*) FROM normalizations WHERE user_correction IS NOT NULL")
            user_corrections = cursor.fetchone()[0]
            
            # Method distribution
            cursor.execute("""
                SELECT rule_method, COUNT(*) 
                FROM normalizations 
                GROUP BY rule_method
            """)
            method_dist = dict(cursor.fetchall())
            
            # Average confidences
            cursor.execute("SELECT AVG(rule_confidence) FROM normalizations")
            avg_rule_conf = cursor.fetchone()[0] or 0.0
            
            cursor.execute("SELECT AVG(llm_confidence) FROM normalizations WHERE llm_confidence IS NOT NULL")
            avg_llm_conf = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return {
                "total_normalizations": total,
                "llm_fallback_count": llm_count,
                "llm_fallback_rate": llm_count / total if total > 0 else 0.0,
                "user_corrections": user_corrections,
                "method_distribution": method_dist,
                "avg_rule_confidence": avg_rule_conf,
                "avg_llm_confidence": avg_llm_conf
            }
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

