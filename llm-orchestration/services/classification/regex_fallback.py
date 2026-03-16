"""
Regex Fallback Classifier

Fast, reliable pattern-based classification for when ML models fail or timeout.
Uses regex patterns to detect memory-worthy content.

This is a fallback mechanism, not the primary classifier.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class RegexFallback:
    """
    Pattern-based memory classification using regex.
    
    Fast and reliable, but limited compared to ML models:
    - Doesn't understand context
    - Misses sarcasm and nuance
    - Can't handle novel patterns
    
    Use cases:
    - Fallback when ML models timeout
    - Fallback when ML models fail
    - Quick validation of ML results
    
    ENHANCEMENT: Can use discovered labels from previous classifications
    to dynamically expand pattern matching.
    """
    
    def __init__(self, label_store=None):
        """
        Initialize regex patterns for each memory type.
        
        Args:
            label_store: Optional LabelStore to access discovered labels
        """
        self.label_store = label_store
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        
        # Identity patterns
        self.identity_patterns = [
            re.compile(r"\bmy name is\b", re.IGNORECASE),
            re.compile(r"\bi am\b", re.IGNORECASE),
            re.compile(r"\bi'm\b", re.IGNORECASE),
            re.compile(r"\bcall me\b", re.IGNORECASE),
            re.compile(r"\bi identify as\b", re.IGNORECASE),
            re.compile(r"\bi go by\b", re.IGNORECASE),
            re.compile(r"\bthey call me\b", re.IGNORECASE),
            re.compile(r"\bpeople know me as\b", re.IGNORECASE),
        ]
        
        # Family patterns
        self.family_patterns = [
            re.compile(r"\bmy (mother|mom|mum|mommy|ma)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (father|dad|daddy|papa|pa)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (sister|brother|sibling)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (son|daughter|child|kid)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (wife|husband|spouse|partner)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (grandmother|grandma|grandfather|grandpa)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (aunt|uncle|cousin|niece|nephew)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy famil(y|ies)\b", re.IGNORECASE),
            re.compile(r"\bmy parents\b", re.IGNORECASE),
        ]
        
        # Preference patterns
        self.preference_patterns = [
            re.compile(r"\bi (love|like|enjoy|prefer|adore|appreciate)\b", re.IGNORECASE),
            re.compile(r"\bi (hate|dislike|despise|detest|can't stand)\b", re.IGNORECASE),
            re.compile(r"\bmy favorite\b", re.IGNORECASE),
            re.compile(r"\bmy favourite\b", re.IGNORECASE),
            re.compile(r"\bi'm (interested in|passionate about|fascinated by)\b", re.IGNORECASE),
            re.compile(r"\bi'm (not interested in|not into|not a fan of)\b", re.IGNORECASE),
            re.compile(r"\bmy (hobby|hobbies|interest|passion)\b", re.IGNORECASE),
        ]
        
        # Fact patterns (work, location, education)
        self.fact_patterns = [
            re.compile(r"\bi (work|study|live|reside) (at|in|for|near)\b", re.IGNORECASE),
            re.compile(r"\bmy (job|work|career|occupation|profession|role|position)\b", re.IGNORECASE),
            re.compile(r"\bi (graduated|studied) (from|at)\b", re.IGNORECASE),
            re.compile(r"\bmy (company|school|university|college)\b", re.IGNORECASE),
            re.compile(r"\bmy (home|house|apartment)\b", re.IGNORECASE),
            re.compile(r"\bi'm (employed|working|studying) (at|in|for)\b", re.IGNORECASE),
        ]
        
        # High-value patterns (financial, responsibility)
        self.high_value_patterns = [
            re.compile(r"(handling|managing|responsible for|overseeing).*(billion|million|\$\d+[bm])", re.IGNORECASE),
            re.compile(r"(assets|portfolio|budget|revenue).*(billion|million|\$\d+[bm])", re.IGNORECASE),
            re.compile(r"(\d+\s*(billion|million)).*(dollar|asset|portfolio)", re.IGNORECASE),
            re.compile(r"(ceo|cto|cfo|director|vp|vice president)", re.IGNORECASE),
            re.compile(r"(lead|leading|head of).*(team|department|division|company)", re.IGNORECASE),
        ]
        
        # Goal patterns
        self.goal_patterns = [
            re.compile(r"\bi (want to|plan to|hope to|aim to|intend to)\b", re.IGNORECASE),
            re.compile(r"\bmy (goal|dream|aspiration|ambition)\b", re.IGNORECASE),
            re.compile(r"\bi'm (planning|hoping|aiming) to\b", re.IGNORECASE),
            re.compile(r"\bin the future,? i\b", re.IGNORECASE),
        ]
        
        # Relationship patterns (non-family)
        self.relationship_patterns = [
            re.compile(r"\bmy (friend|buddy|pal|mate)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (colleague|coworker|teammate)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (boyfriend|girlfriend|partner)(s|'s)?\b", re.IGNORECASE),
            re.compile(r"\bmy (best friend|bff)\b", re.IGNORECASE),
        ]
        
        # Event patterns
        self.event_patterns = [
            re.compile(r"\b(yesterday|today|tomorrow|last (week|month|year))\b", re.IGNORECASE),
            re.compile(r"\bi (went|visited|attended|experienced)\b", re.IGNORECASE),
            re.compile(r"\bwhen i (was|went|visited)\b", re.IGNORECASE),
            re.compile(r"\b(happened|occurred|took place)\b", re.IGNORECASE),
        ]
        
        # Opinion patterns
        self.opinion_patterns = [
            re.compile(r"\bi (think|believe|feel|reckon|suppose)\b", re.IGNORECASE),
            re.compile(r"\bin my (opinion|view|perspective)\b", re.IGNORECASE),
            re.compile(r"\bi'm (convinced|certain|sure) that\b", re.IGNORECASE),
            re.compile(r"\bit seems to me\b", re.IGNORECASE),
        ]
    
    def detect_triggers(self, message: str, use_discovered_labels: bool = True) -> List[str]:
        """
        Detect memory triggers in message using regex patterns.
        
        ENHANCEMENT: Can also use discovered labels for better coverage.
        
        Args:
            message: User message
            use_discovered_labels: If True, also check against discovered labels
        
        Returns:
            List of detected triggers (e.g., ['identity', 'family', 'programming_affinity'])
        """
        triggers = []
        
        # Check each pattern category
        if self._match_any(message, self.identity_patterns):
            triggers.append("identity")
        
        if self._match_any(message, self.family_patterns):
            triggers.append("family")
        
        if self._match_any(message, self.preference_patterns):
            triggers.append("preference")
        
        if self._match_any(message, self.fact_patterns):
            triggers.append("fact")
        
        if self._match_any(message, self.high_value_patterns):
            triggers.append("high_value")
        
        if self._match_any(message, self.goal_patterns):
            triggers.append("goal")
        
        if self._match_any(message, self.relationship_patterns):
            triggers.append("relationship")
        
        if self._match_any(message, self.event_patterns):
            triggers.append("event")
        
        if self._match_any(message, self.opinion_patterns):
            triggers.append("opinion")
        
        # ENHANCEMENT: Use discovered labels for semantic matching
        if use_discovered_labels and self.label_store:
            discovered_triggers = self._match_discovered_labels(message)
            triggers.extend(discovered_triggers)
        
        return triggers
    
    def _match_discovered_labels(self, message: str) -> List[str]:
        """
        Match message against discovered labels using semantic keywords.
        
        For example:
        - "programming_affinity" → matches "python", "coding", "programming"
        - "pet_info" → matches "cat", "dog", "pet"
        - "technology_interest" → matches "tech", "software", "AI"
        
        Args:
            message: User message
            
        Returns:
            List of discovered labels that match
        """
        if not self.label_store:
            return []
        
        matched_labels = []
        message_lower = message.lower()
        
        # Get all discovered labels
        all_labels = self.label_store.get_all_labels()
        
        # Simple keyword matching based on label names
        for label in all_labels:
            # Extract keywords from label name (e.g., "programming_affinity" → ["programming", "affinity"])
            keywords = label.replace('_', ' ').split()
            
            # Check if any keyword appears in message
            for keyword in keywords:
                if len(keyword) > 3 and keyword in message_lower:  # Ignore short words
                    matched_labels.append(label)
                    logger.debug(f"   Discovered label matched: '{label}' (keyword: '{keyword}')")
                    break
        
        return matched_labels
    
    def _match_any(self, message: str, patterns: List[re.Pattern]) -> bool:
        """
        Check if message matches any pattern in the list.
        
        Args:
            message: Text to check
            patterns: List of compiled regex patterns
        
        Returns:
            True if any pattern matches
        """
        for pattern in patterns:
            if pattern.search(message):
                return True
        return False
    
    def get_pattern_count(self) -> int:
        """Get total number of patterns across all categories."""
        return sum([
            len(self.identity_patterns),
            len(self.family_patterns),
            len(self.preference_patterns),
            len(self.fact_patterns),
            len(self.high_value_patterns),
            len(self.goal_patterns),
            len(self.relationship_patterns),
            len(self.event_patterns),
            len(self.opinion_patterns),
        ])

