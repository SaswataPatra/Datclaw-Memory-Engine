"""
DAPPY Relation Training Data Collector

Collects relation extraction data for model training.
Extends the pattern from TrainingDataCollector.

Key features:
- Logs all relation extractions
- Tracks discovered relation types
- Stores corrections for model improvement
- Exports training data in HuggingFace format

Phase 1B Implementation
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RelationTrainingCollector:
    """
    Collects relation extraction data for future model training.
    
    This service logs:
    - Relation extractions (with confidence and source)
    - Discovered relation types
    - Relation corrections (from user feedback or validation)
    
    Data can be exported for:
    - Fine-tuning relation extraction models
    - Training custom classifiers
    - Analyzing extraction patterns
    """
    
    def __init__(self, db_path: str = "data/relation_training.db"):
        """
        Initialize the relation training data collector.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"✅ RelationTrainingCollector initialized: {self.db_path}")
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Relation extractions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_extractions (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                object TEXT NOT NULL,
                context TEXT NOT NULL,
                predicted_relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                category TEXT,
                source TEXT NOT NULL,
                user_id TEXT,
                memory_id TEXT,
                timestamp TEXT NOT NULL,
                is_validated INTEGER DEFAULT 0,
                validated_relation TEXT,
                validation_source TEXT
            )
        """)
        
        # Table 2: Discovered relation types
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_relations (
                id TEXT PRIMARY KEY,
                relation_name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                example_context TEXT,
                example_subject TEXT,
                example_object TEXT,
                usage_count INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Table 3: Relation corrections (from user feedback)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_corrections (
                id TEXT PRIMARY KEY,
                extraction_id TEXT NOT NULL,
                original_relation TEXT NOT NULL,
                corrected_relation TEXT NOT NULL,
                correction_source TEXT NOT NULL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (extraction_id) REFERENCES relation_extractions(id)
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extractions_timestamp 
            ON relation_extractions(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extractions_user 
            ON relation_extractions(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_extractions_relation 
            ON relation_extractions(predicted_relation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_discovered_name 
            ON discovered_relations(relation_name)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Relation training database tables initialized")
    
    def log_extraction(
        self,
        subject: str,
        object_text: str,
        context: str,
        predicted_relation: str,
        confidence: float,
        category: str = None,
        source: str = "deberta",
        user_id: str = None,
        memory_id: str = None
    ) -> str:
        """
        Log a relation extraction for training.
        
        Args:
            subject: Subject entity text
            object_text: Object entity text
            context: Full context text
            predicted_relation: Predicted relation type
            confidence: Confidence score
            category: Relation category
            source: Extraction source (deberta, llm, heuristic)
            user_id: Optional user ID
            memory_id: Optional memory ID
        
        Returns:
            Extraction ID (UUID)
        """
        extraction_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO relation_extractions
            (id, subject, object, context, predicted_relation, confidence,
             category, source, user_id, memory_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            extraction_id,
            subject,
            object_text,
            context,
            predicted_relation,
            confidence,
            category,
            source,
            user_id,
            memory_id,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged relation extraction: {subject} → {predicted_relation} → {object_text}")
        return extraction_id
    
    def log_discovered_relation(
        self,
        relation_name: str,
        category: str,
        example_context: str = None,
        example_subject: str = None,
        example_object: str = None
    ) -> str:
        """
        Log a newly discovered relation type.
        
        Args:
            relation_name: Name of the new relation type
            category: Category for the relation
            example_context: Example context where it was discovered
            example_subject: Example subject entity
            example_object: Example object entity
        
        Returns:
            Discovery ID (UUID)
        """
        discovery_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO discovered_relations
                (id, relation_name, category, example_context, 
                 example_subject, example_object, usage_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                discovery_id,
                relation_name,
                category,
                example_context,
                example_subject,
                example_object,
                1,
                datetime.utcnow().isoformat()
            ))
            logger.info(f"Logged discovered relation type: {relation_name} ({category})")
        except sqlite3.IntegrityError:
            # Relation already exists, increment usage count
            cursor.execute("""
                UPDATE discovered_relations
                SET usage_count = usage_count + 1
                WHERE relation_name = ?
            """, (relation_name,))
            logger.debug(f"Incremented usage count for: {relation_name}")
        
        conn.commit()
        conn.close()
        
        return discovery_id
    
    def log_correction(
        self,
        extraction_id: str,
        original_relation: str,
        corrected_relation: str,
        correction_source: str = "user",
        reasoning: str = None
    ) -> str:
        """
        Log a relation correction.
        
        Args:
            extraction_id: ID of the original extraction
            original_relation: Original predicted relation
            corrected_relation: Corrected relation type
            correction_source: Source of correction (user, validator, etc.)
            reasoning: Optional reasoning for the correction
        
        Returns:
            Correction ID (UUID)
        """
        correction_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert correction
        cursor.execute("""
            INSERT INTO relation_corrections
            (id, extraction_id, original_relation, corrected_relation,
             correction_source, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            correction_id,
            extraction_id,
            original_relation,
            corrected_relation,
            correction_source,
            reasoning,
            datetime.utcnow().isoformat()
        ))
        
        # Update extraction with validation
        cursor.execute("""
            UPDATE relation_extractions
            SET is_validated = 1,
                validated_relation = ?,
                validation_source = ?
            WHERE id = ?
        """, (corrected_relation, correction_source, extraction_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged correction: {original_relation} → {corrected_relation}")
        return correction_id
    
    def get_training_dataset(
        self,
        min_confidence: float = 0.7,
        include_corrections: bool = True,
        limit: int = None
    ) -> List[Dict]:
        """
        Export training dataset for relation classifier.
        
        Args:
            min_confidence: Minimum confidence threshold
            include_corrections: Include corrected examples
            limit: Maximum number of examples
        
        Returns:
            List of training examples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT subject, object, context, 
                   COALESCE(validated_relation, predicted_relation) as relation,
                   category, confidence, source, is_validated
            FROM relation_extractions
            WHERE (confidence >= ? OR is_validated = 1)
        """
        
        if not include_corrections:
            query += " AND is_validated = 0"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (min_confidence,))
        rows = cursor.fetchall()
        conn.close()
        
        dataset = []
        for row in rows:
            dataset.append({
                "subject": row[0],
                "object": row[1],
                "context": row[2],
                "relation": row[3],
                "category": row[4],
                "confidence": row[5],
                "source": row[6],
                "is_validated": bool(row[7])
            })
        
        logger.info(f"Exported {len(dataset)} training examples")
        return dataset
    
    def get_discovered_relations(self) -> List[Dict]:
        """Get all discovered relation types."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT relation_name, category, usage_count, 
                   example_context, example_subject, example_object
            FROM discovered_relations
            ORDER BY usage_count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "relation_name": row[0],
                "category": row[1],
                "usage_count": row[2],
                "example_context": row[3],
                "example_subject": row[4],
                "example_object": row[5]
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total extractions
        cursor.execute("SELECT COUNT(*) FROM relation_extractions")
        total_extractions = cursor.fetchone()[0]
        
        # By source
        cursor.execute("""
            SELECT source, COUNT(*) 
            FROM relation_extractions 
            GROUP BY source
        """)
        by_source = dict(cursor.fetchall())
        
        # By category
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM relation_extractions 
            WHERE category IS NOT NULL
            GROUP BY category
        """)
        by_category = dict(cursor.fetchall())
        
        # Validated count
        cursor.execute("""
            SELECT COUNT(*) FROM relation_extractions 
            WHERE is_validated = 1
        """)
        validated_count = cursor.fetchone()[0]
        
        # Discovered relations count
        cursor.execute("SELECT COUNT(*) FROM discovered_relations")
        discovered_count = cursor.fetchone()[0]
        
        # Corrections count
        cursor.execute("SELECT COUNT(*) FROM relation_corrections")
        corrections_count = cursor.fetchone()[0]
        
        # Top relations
        cursor.execute("""
            SELECT predicted_relation, COUNT(*) as cnt
            FROM relation_extractions
            GROUP BY predicted_relation
            ORDER BY cnt DESC
            LIMIT 10
        """)
        top_relations = dict(cursor.fetchall())
        
        # Recent extractions (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM relation_extractions
            WHERE datetime(timestamp) >= datetime('now', '-1 day')
        """)
        recent_extractions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_extractions": total_extractions,
            "by_source": by_source,
            "by_category": by_category,
            "validated_count": validated_count,
            "discovered_relations": discovered_count,
            "corrections_count": corrections_count,
            "top_relations": top_relations,
            "recent_extractions_24h": recent_extractions
        }
    
    def export_to_huggingface_format(
        self,
        output_path: str,
        min_confidence: float = 0.7
    ):
        """
        Export training data in HuggingFace Datasets format.
        
        Format:
        {
            "text": "Context with subject and object",
            "subject": "Entity1",
            "object": "Entity2",
            "relation": "relation_type",
            "category": "category"
        }
        
        Args:
            output_path: Path to save JSONL file
            min_confidence: Minimum confidence threshold
        """
        dataset = self.get_training_dataset(min_confidence=min_confidence)
        
        with open(output_path, 'w') as f:
            for example in dataset:
                hf_example = {
                    "text": example["context"],
                    "subject": example["subject"],
                    "object": example["object"],
                    "relation": example["relation"],
                    "category": example["category"]
                }
                f.write(json.dumps(hf_example) + '\n')
        
        logger.info(f"Exported {len(dataset)} examples to {output_path}")

