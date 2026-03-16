"""
Training Data Collector Service

Collects classification corrections for future model training.
This data is used to fine-tune or train custom classifiers (DistilBERT/DeBERTa).

Sources:
1. Semantic validator corrections (LLM identifies incorrect labels)
2. Label discovery events (new labels are created)
3. Confidence distribution flags (suspicious classification patterns)

Storage: SQLite database for easy querying and export
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects classification corrections for future model training.
    
    This service logs:
    - Semantic validation corrections (high-value data)
    - Label discovery events (new label creation)
    - Confidence distribution anomalies (suspicious patterns)
    
    Data can be exported for:
    - Fine-tuning DistilBERT/DeBERTa
    - Training custom classifiers
    - Analyzing false positive patterns
    """
    
    def __init__(self, db_path: str = "data/training_corrections.db"):
        """
        Initialize the training data collector.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"✅ TrainingDataCollector initialized: {self.db_path}")
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Classification corrections
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS classification_corrections (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                predicted_labels TEXT NOT NULL,
                corrected_labels TEXT,
                invalid_labels TEXT,
                scores TEXT NOT NULL,
                reasoning TEXT,
                correction_source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                classifier_confidence REAL,
                routing_decision TEXT
            )
        """)
        
        # Table 2: Discovered labels
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_labels (
                id TEXT PRIMARY KEY,
                label_name TEXT NOT NULL,
                importance REAL NOT NULL,
                context_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0
            )
        """)
        
        # Table 3: Confidence distribution anomalies
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS confidence_anomalies (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                predicted_labels TEXT NOT NULL,
                scores TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                issue_description TEXT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT
            )
        """)
        
        # Indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_corrections_timestamp 
            ON classification_corrections(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_corrections_source 
            ON classification_corrections(correction_source)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_labels_name 
            ON discovered_labels(label_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp 
            ON confidence_anomalies(timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"  📊 Database tables initialized")
    
    def log_semantic_correction(
        self,
        text: str,
        predicted_labels: List[str],
        invalid_labels: List[str],
        scores: Dict[str, float],
        reasoning: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        classifier_confidence: Optional[float] = None,
        routing_decision: Optional[str] = None
    ) -> str:
        """
        Log when semantic validator identifies incorrect labels.
        
        This is HIGH VALUE data - LLM has explicitly said these labels are wrong.
        
        Args:
            text: Input text that was classified
            predicted_labels: Labels predicted by classifier
            invalid_labels: Labels identified as incorrect by LLM
            scores: Confidence scores for each label
            reasoning: LLM's reasoning for why labels are invalid
            user_id: Optional user identifier
            session_id: Optional session identifier
            classifier_confidence: Maximum confidence score from classifier
            routing_decision: How the classification was routed (auto_accept, semantic_validated, etc.)
        
        Returns:
            Correction ID (UUID)
        """
        correction_id = str(uuid.uuid4())
        valid_labels = [l for l in predicted_labels if l not in invalid_labels]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO classification_corrections
            (id, text, predicted_labels, corrected_labels, invalid_labels, 
             scores, reasoning, correction_source, timestamp, user_id, 
             session_id, classifier_confidence, routing_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            correction_id,
            text,
            json.dumps(predicted_labels),
            json.dumps(valid_labels),
            json.dumps(invalid_labels),
            json.dumps(scores),
            reasoning,
            'semantic_validator',
            datetime.utcnow().isoformat(),
            user_id,
            session_id,
            classifier_confidence,
            routing_decision
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged semantic correction: {correction_id}")
        logger.debug(f"   Text: {text[:50]}...")
        logger.debug(f"   Invalid: {invalid_labels}")
        logger.debug(f"   Valid: {valid_labels}")
        
        return correction_id
    
    def log_label_discovery(
        self,
        text: str,
        existing_labels: List[str],
        discovered_labels: List[Dict[str, any]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Log when new labels are discovered.
        
        This helps track label evolution and emerging concepts.
        
        Args:
            text: Input text that triggered label discovery
            existing_labels: Labels that existed before discovery
            discovered_labels: New labels discovered (list of dicts with 'name' and 'importance')
            user_id: Optional user identifier
            session_id: Optional session identifier
        
        Returns:
            Tuple of (discovery_id, list of label_ids)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Log the discovery event
        discovery_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO classification_corrections
            (id, text, predicted_labels, corrected_labels, invalid_labels,
             scores, reasoning, correction_source, timestamp, user_id, session_id, classifier_confidence, routing_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            discovery_id,
            text,
            json.dumps(existing_labels),
            json.dumps([l['name'] for l in discovered_labels]),
            json.dumps([]),
            json.dumps({l['name']: l['importance'] for l in discovered_labels}),
            f"Discovered {len(discovered_labels)} new labels",
            'label_discovery',
            datetime.utcnow().isoformat(),
            user_id,
            session_id,
            None,
            'label_discovery'
        ))
        
        # Log each discovered label
        label_ids = []
        for label_info in discovered_labels:
            label_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO discovered_labels
                (id, label_name, importance, context_text, timestamp, usage_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                label_id,
                label_info['name'],
                label_info['importance'],
                text,
                datetime.utcnow().isoformat(),
                0
            ))
            label_ids.append(label_id)
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged label discovery: {[l['name'] for l in discovered_labels]}")
        
        return discovery_id, label_ids
    
    def log_confidence_anomaly(
        self,
        text: str,
        predicted_labels: List[str],
        scores: Dict[str, float],
        issue_type: str,
        issue_description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Log confidence distribution anomalies.
        
        Examples:
        - Too many high-confidence labels
        - Multiple labels with similar mid-range scores
        - Suspiciously uniform distribution
        
        Args:
            text: Input text
            predicted_labels: Predicted labels
            scores: Confidence scores
            issue_type: Type of anomaly (e.g., 'too_many_labels', 'suspicious_distribution')
            issue_description: Human-readable description
            user_id: Optional user identifier
            session_id: Optional session identifier
        
        Returns:
            Anomaly ID (UUID)
        """
        anomaly_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO confidence_anomalies
            (id, text, predicted_labels, scores, issue_type, issue_description, timestamp, user_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            anomaly_id,
            text,
            json.dumps(predicted_labels),
            json.dumps(scores),
            issue_type,
            issue_description,
            datetime.utcnow().isoformat(),
            user_id,
            session_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Logged confidence anomaly: {issue_type}")
        logger.debug(f"   Text: {text[:50]}...")
        logger.debug(f"   Description: {issue_description}")
        
        return anomaly_id
    
    def get_training_dataset(
        self,
        min_confidence: float = 0.0,
        source_filter: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Export training dataset for model training.
        
        Args:
            min_confidence: Minimum classifier confidence to include
            source_filter: Filter by correction source (e.g., ['semantic_validator'])
            limit: Maximum number of examples to return
        
        Returns:
            List of training examples with corrected labels
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT text, corrected_labels, scores, correction_source, 
                   classifier_confidence, reasoning, routing_decision
            FROM classification_corrections
            WHERE corrected_labels IS NOT NULL
        """
        
        if min_confidence > 0:
            query += f" AND (classifier_confidence IS NULL OR classifier_confidence >= {min_confidence})"
        
        if source_filter:
            placeholders = ','.join('?' * len(source_filter))
            query += f" AND correction_source IN ({placeholders})"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        if source_filter:
            cursor.execute(query, source_filter)
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        conn.close()
        
        dataset = []
        for row in rows:
            dataset.append({
                'text': row[0],
                'labels': json.loads(row[1]) if row[1] else [],
                'scores': json.loads(row[2]),
                'source': row[3],
                'confidence': row[4],
                'reasoning': row[5],
                'routing_decision': row[6]
            })
        
        logger.info(f"📊 Exported {len(dataset)} training examples")
        return dataset
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with statistics about collected data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total corrections
        cursor.execute("SELECT COUNT(*) FROM classification_corrections")
        total_corrections = cursor.fetchone()[0]
        
        # By source
        cursor.execute("""
            SELECT correction_source, COUNT(*) 
            FROM classification_corrections 
            GROUP BY correction_source
        """)
        by_source = dict(cursor.fetchall())
        
        # By routing decision
        cursor.execute("""
            SELECT routing_decision, COUNT(*) 
            FROM classification_corrections 
            WHERE routing_decision IS NOT NULL
            GROUP BY routing_decision
        """)
        by_routing = dict(cursor.fetchall())
        
        # Total discovered labels
        cursor.execute("SELECT COUNT(*) FROM discovered_labels")
        total_labels = cursor.fetchone()[0]
        
        # Total anomalies
        cursor.execute("SELECT COUNT(*) FROM confidence_anomalies")
        total_anomalies = cursor.fetchone()[0]
        
        # Recent corrections (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM classification_corrections
            WHERE datetime(timestamp) >= datetime('now', '-1 day')
        """)
        recent_corrections = cursor.fetchone()[0]
        
        # Most common invalid labels
        cursor.execute("""
            SELECT invalid_labels FROM classification_corrections
            WHERE correction_source = 'semantic_validator'
            AND invalid_labels != '[]'
        """)
        invalid_label_rows = cursor.fetchall()
        
        conn.close()
        
        # Count invalid labels
        invalid_label_counts = {}
        for row in invalid_label_rows:
            labels = json.loads(row[0])
            for label in labels:
                invalid_label_counts[label] = invalid_label_counts.get(label, 0) + 1
        
        # Sort by count
        top_invalid_labels = sorted(
            invalid_label_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_corrections': total_corrections,
            'by_source': by_source,
            'by_routing': by_routing,
            'total_discovered_labels': total_labels,
            'total_anomalies': total_anomalies,
            'recent_corrections_24h': recent_corrections,
            'top_invalid_labels': dict(top_invalid_labels)
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
            "text": "I love Python programming",
            "labels": ["preference", "programming_affinity"],
            "label_scores": [0.95, 0.88]
        }
        
        Args:
            output_path: Path to save JSONL file
            min_confidence: Minimum confidence threshold
        """
        dataset = self.get_training_dataset(min_confidence=min_confidence)
        
        with open(output_path, 'w') as f:
            for example in dataset:
                # Convert to HuggingFace format
                hf_example = {
                    'text': example['text'],
                    'labels': example['labels'],
                    'label_scores': [
                        example['scores'].get(label, 0.0)
                        for label in example['labels']
                    ]
                }
                f.write(json.dumps(hf_example) + '\n')
        
        logger.info(f"📦 Exported {len(dataset)} examples to {output_path}")

