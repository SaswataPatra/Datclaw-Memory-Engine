"""
DAPPY Command Training Data Collector

Collects explicit user commands as training data.
These are HIGH VALUE signals because:
1. User explicitly stated their intent
2. No ambiguity - we know exactly what they want
3. Perfect for training tier prediction and importance models

Phase 1C Implementation
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from .command_parser import UserCommand, CommandCategory

logger = logging.getLogger(__name__)


class CommandTrainingCollector:
    """
    Collects explicit user commands as training data.
    
    This data is used to train:
    1. Tier prediction models (what tier should a memory be?)
    2. Importance scoring models (how important is this memory?)
    3. Classification models (what type of memory is this?)
    
    The key insight: Explicit commands are GROUND TRUTH.
    When a user says "/remember", they're telling us this is important.
    When they say "/tier3", they're telling us exactly where it belongs.
    """
    
    def __init__(self, db_path: str = "data/command_training.db"):
        """
        Initialize the command training data collector.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        logger.info(f"✅ CommandTrainingCollector initialized: {self.db_path}")
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table 1: Explicit commands
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS explicit_commands (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                message TEXT NOT NULL,
                command TEXT NOT NULL,
                command_category TEXT NOT NULL,
                args TEXT,
                target_text TEXT NOT NULL,
                
                -- What the user explicitly requested
                explicit_tier INTEGER,
                explicit_weight REAL,
                explicit_action TEXT,
                
                -- What the system assigned (for comparison)
                system_tier INTEGER,
                system_ego_score REAL,
                
                -- Context
                conversation_context TEXT,
                
                -- Metadata
                timestamp TEXT NOT NULL,
                memory_id TEXT,
                was_confirmed INTEGER DEFAULT 1,
                confirmation_timestamp TEXT
            )
        """)
        
        # Table 2: Command patterns (for learning common patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS command_patterns (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                command TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                avg_tier REAL,
                avg_weight REAL,
                last_seen TEXT NOT NULL
            )
        """)
        
        # Table 3: Tier corrections (when user overrides system tier)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tier_corrections (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                system_tier INTEGER NOT NULL,
                user_tier INTEGER NOT NULL,
                command TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_commands_user 
            ON explicit_commands(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_commands_command 
            ON explicit_commands(command)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_commands_tier 
            ON explicit_commands(explicit_tier)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_command 
            ON command_patterns(command)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Command training database tables initialized")
    
    def log_command(
        self,
        user_id: str,
        message: str,
        command: UserCommand,
        session_id: str = None,
        system_tier: int = None,
        system_ego_score: float = None,
        conversation_context: List[str] = None,
        memory_id: str = None
    ) -> str:
        """
        Log an explicit command for training.
        
        This is HIGH VALUE training data because the user
        explicitly told us what they want.
        
        Args:
            user_id: User ID
            message: Original message with command
            command: Parsed UserCommand object
            session_id: Optional session ID
            system_tier: What tier the system would have assigned
            system_ego_score: What ego score the system calculated
            conversation_context: Recent conversation for context
            memory_id: ID of the stored memory (if applicable)
        
        Returns:
            Command log ID
        """
        log_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO explicit_commands
            (id, user_id, session_id, message, command, command_category,
             args, target_text, explicit_tier, explicit_weight, explicit_action,
             system_tier, system_ego_score, conversation_context, timestamp, memory_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            user_id,
            session_id,
            message,
            command.command,
            command.category.value,
            json.dumps(command.args),
            command.target_text,
            command.config.get("tier"),
            command.config.get("weight"),
            command.config.get("action"),
            system_tier,
            system_ego_score,
            json.dumps(conversation_context) if conversation_context else None,
            datetime.utcnow().isoformat(),
            memory_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged command: /{command.command} for user {user_id}")
        
        # Also update patterns
        self._update_pattern(command)
        
        # Check for tier correction
        if system_tier and command.config.get("tier"):
            if system_tier != command.config.get("tier"):
                self._log_tier_correction(
                    user_id, message, system_tier, 
                    command.config.get("tier"), command.command
                )
        
        return log_id
    
    def _update_pattern(self, command: UserCommand):
        """Update command pattern statistics."""
        # Extract pattern from target text (simplified)
        pattern = self._extract_pattern(command.target_text)
        
        if not pattern:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if pattern exists
        cursor.execute("""
            SELECT id, frequency, avg_tier, avg_weight
            FROM command_patterns
            WHERE pattern = ? AND command = ?
        """, (pattern, command.command))
        
        row = cursor.fetchone()
        
        tier = command.config.get("tier", 0)
        weight = command.config.get("weight", 0.5)
        
        if row:
            # Update existing pattern
            new_freq = row[1] + 1
            new_avg_tier = (row[2] * row[1] + tier) / new_freq if tier else row[2]
            new_avg_weight = (row[3] * row[1] + weight) / new_freq
            
            cursor.execute("""
                UPDATE command_patterns
                SET frequency = ?, avg_tier = ?, avg_weight = ?, last_seen = ?
                WHERE id = ?
            """, (new_freq, new_avg_tier, new_avg_weight, 
                  datetime.utcnow().isoformat(), row[0]))
        else:
            # Create new pattern
            cursor.execute("""
                INSERT INTO command_patterns
                (id, pattern, command, frequency, avg_tier, avg_weight, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                pattern,
                command.command,
                1,
                tier if tier else None,
                weight,
                datetime.utcnow().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _extract_pattern(self, text: str) -> Optional[str]:
        """
        Extract a pattern from text for pattern matching.
        
        Simplifies text to identify common patterns:
        - "My sister Sarah" → "my [RELATION] [NAME]"
        - "I like pizza" → "i like [NOUN]"
        """
        if not text or len(text) < 5:
            return None
        
        # Simple pattern extraction (can be enhanced with NER)
        text_lower = text.lower().strip()
        
        # Common patterns
        patterns = [
            (r"my (sister|brother|mother|father|mom|dad|wife|husband|partner)", 
             "my [FAMILY_RELATION]"),
            (r"i (like|love|hate|prefer|enjoy)", 
             "i [PREFERENCE_VERB]"),
            (r"my name is", "my name is [NAME]"),
            (r"i am (\d+)", "i am [AGE]"),
            (r"i work at", "i work at [ORG]"),
            (r"i live in", "i live in [LOCATION]"),
        ]
        
        import re
        for regex, pattern in patterns:
            if re.search(regex, text_lower):
                return pattern
        
        # Default: first 3 words
        words = text_lower.split()[:3]
        return " ".join(words) if words else None
    
    def _log_tier_correction(
        self,
        user_id: str,
        message: str,
        system_tier: int,
        user_tier: int,
        command: str
    ):
        """Log when user's tier differs from system's tier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO tier_corrections
            (id, user_id, message, system_tier, user_tier, command, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            user_id,
            message,
            system_tier,
            user_tier,
            command,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Tier correction: system={system_tier} → user={user_tier}")
    
    def get_training_dataset(
        self,
        command_filter: List[str] = None,
        min_frequency: int = 1,
        limit: int = None
    ) -> List[Dict]:
        """
        Export training dataset for ML models.
        
        Args:
            command_filter: Only include these commands
            min_frequency: Minimum pattern frequency
            limit: Maximum examples
        
        Returns:
            List of training examples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT message, command, command_category, target_text,
                   explicit_tier, explicit_weight, system_tier, system_ego_score,
                   conversation_context
            FROM explicit_commands
            WHERE was_confirmed = 1
        """
        
        if command_filter:
            placeholders = ','.join('?' * len(command_filter))
            query += f" AND command IN ({placeholders})"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        if command_filter:
            cursor.execute(query, command_filter)
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        conn.close()
        
        dataset = []
        for row in rows:
            dataset.append({
                "message": row[0],
                "command": row[1],
                "category": row[2],
                "target_text": row[3],
                "explicit_tier": row[4],
                "explicit_weight": row[5],
                "system_tier": row[6],
                "system_ego_score": row[7],
                "context": json.loads(row[8]) if row[8] else None
            })
        
        logger.info(f"Exported {len(dataset)} training examples")
        return dataset
    
    def get_tier_corrections(self) -> List[Dict]:
        """Get all tier corrections for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message, system_tier, user_tier, command, timestamp
            FROM tier_corrections
            ORDER BY timestamp DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "message": row[0],
                "system_tier": row[1],
                "user_tier": row[2],
                "command": row[3],
                "timestamp": row[4]
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total commands
        cursor.execute("SELECT COUNT(*) FROM explicit_commands")
        total_commands = cursor.fetchone()[0]
        
        # By command
        cursor.execute("""
            SELECT command, COUNT(*) 
            FROM explicit_commands 
            GROUP BY command
            ORDER BY COUNT(*) DESC
        """)
        by_command = dict(cursor.fetchall())
        
        # By category
        cursor.execute("""
            SELECT command_category, COUNT(*) 
            FROM explicit_commands 
            GROUP BY command_category
        """)
        by_category = dict(cursor.fetchall())
        
        # By tier
        cursor.execute("""
            SELECT explicit_tier, COUNT(*) 
            FROM explicit_commands 
            WHERE explicit_tier IS NOT NULL
            GROUP BY explicit_tier
        """)
        by_tier = dict(cursor.fetchall())
        
        # Tier corrections
        cursor.execute("SELECT COUNT(*) FROM tier_corrections")
        tier_corrections = cursor.fetchone()[0]
        
        # Top patterns
        cursor.execute("""
            SELECT pattern, command, frequency
            FROM command_patterns
            ORDER BY frequency DESC
            LIMIT 10
        """)
        top_patterns = [
            {"pattern": row[0], "command": row[1], "frequency": row[2]}
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            "total_commands": total_commands,
            "by_command": by_command,
            "by_category": by_category,
            "by_tier": by_tier,
            "tier_corrections": tier_corrections,
            "top_patterns": top_patterns
        }
    
    def export_for_tier_model(self, output_path: str):
        """
        Export training data specifically for tier prediction model.
        
        Format:
        {
            "text": "My sister Sarah lives in NYC",
            "tier": 2,
            "command": "remember"
        }
        """
        dataset = self.get_training_dataset()
        
        with open(output_path, 'w') as f:
            for example in dataset:
                if example["explicit_tier"]:
                    tier_example = {
                        "text": example["target_text"],
                        "tier": example["explicit_tier"],
                        "command": example["command"]
                    }
                    f.write(json.dumps(tier_example) + '\n')
        
        logger.info(f"Exported tier training data to {output_path}")
    
    def export_for_importance_model(self, output_path: str):
        """
        Export training data for importance scoring model.
        
        Format:
        {
            "text": "My sister Sarah lives in NYC",
            "importance": 0.8,
            "command": "remember"
        }
        """
        dataset = self.get_training_dataset()
        
        with open(output_path, 'w') as f:
            for example in dataset:
                if example["explicit_weight"]:
                    importance_example = {
                        "text": example["target_text"],
                        "importance": example["explicit_weight"],
                        "command": example["command"]
                    }
                    f.write(json.dumps(importance_example) + '\n')
        
        logger.info(f"Exported importance training data to {output_path}")

