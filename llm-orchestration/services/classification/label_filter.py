"""
Label Filter - Post-Classification Rule-Based Filtering

Provides fast, deterministic filtering of classification results based on rules.
This is more scalable than adding examples to LLM prompts.

Usage:
    filter = LabelFilter()
    filtered_labels = filter.apply_rules(text, predicted_labels, scores)
"""

import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class LabelFilter:
    """
    Rule-based post-classification filter.
    
    Filters out labels that don't make semantic sense based on:
    - Text content patterns
    - Label-specific thresholds
    - Exclusion rules
    - Text length requirements
    """
    
    def __init__(self):
        """Initialize label filter with rules."""
        
        # Label-specific confidence thresholds
        # Labels prone to false positives require higher confidence
        self.label_thresholds = {
            'identity': 0.85,        # Strict - only high confidence
            'expert_opinion': 0.80,  # Strict - casual opinions trigger this
            'high_value': 0.85,      # Strict - sensitive info
            'medical_condition': 0.80,  # Strict - technical terms trigger this
            'treatment_implications': 0.80,  # Strict
            'personal_health': 0.75,  # Medium strict
            'event': 0.65,           # Medium - often over-triggered
            'fact': 0.60,            # Medium
            'opinion': 0.55,         # Lenient
            'preference': 0.55,      # Lenient
            'family': 0.60,          # Medium
            'relationship': 0.60,    # Medium
        }
        
        # Exclusion rules: patterns that invalidate a label
        self.exclusion_rules = {
            'identity': {
                'exclude_if_contains': [
                    r'\b(cat|dog|pet|animal|bird|fish)\b',  # Pet names
                    r'\b(car|house|city|country)\b',  # Object/place names
                ],
                'require_if_not_contains': [
                    r'\b(my name|i am|i identify|call me|known as)\b'
                ],
            },
            'expert_opinion': {
                'exclude_if_contains': [
                    r'\b(i love|i like|i hate|i think|i feel|i have)\b',  # Personal feelings
                    r'\b(my cat|my dog|my pet)\b',  # Personal anecdotes
                ],
                'min_length': 40,  # Expert opinions are usually longer
            },
            'high_value': {
                'exclude_if_contains': [
                    r'\b(cat|dog|pet)\b',  # Pet ownership is not high-value
                    r'\b(like|love|enjoy)\b',  # Casual mentions
                ],
                'require_if_not_contains': [
                    r'\b(password|ssn|credit card|bank|account|salary|income)\b'
                ],
            },
            'medical_condition': {
                'exclude_if_contains': [
                    r'\bmemory\s+(project|system|database|store|manager)\b',  # Technical context
                    r'\bhealth\s+(check|status|system)\b',  # System health
                ],
            },
            'treatment_implications': {
                'exclude_if_contains': [
                    r'\b(project|plan|strategy|approach|method)\b',  # Business context
                ],
            },
            'event': {
                'exclude_if_contains': [
                    r'\b(i have|i own|i am)\b',  # States, not events
                ],
                'min_length': 15,  # Events usually have more detail
            },
        }
    
    def apply_rules(
        self,
        text: str,
        predicted_labels: List[str],
        scores: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Apply rule-based filtering to classification results.
        
        Args:
            text: The text that was classified
            predicted_labels: Labels from classifier
            scores: Confidence scores for each label
        
        Returns:
            (filtered_labels, filtered_scores)
        """
        text_lower = text.lower()
        filtered_labels = []
        filtered_scores = {}
        removed_labels = []
        
        for label in predicted_labels:
            score = scores.get(label, 0.0)
            
            # Check 1: Label-specific threshold
            threshold = self.label_thresholds.get(label, 0.5)
            if score < threshold:
                logger.debug(f"   ❌ Filtered '{label}': score {score:.3f} < threshold {threshold:.3f}")
                removed_labels.append((label, f"score {score:.3f} < {threshold:.3f}"))
                continue
            
            # Check 2: Exclusion rules
            if label in self.exclusion_rules:
                rules = self.exclusion_rules[label]
                
                # Check exclusion patterns
                if 'exclude_if_contains' in rules:
                    excluded = False
                    for pattern in rules['exclude_if_contains']:
                        if re.search(pattern, text_lower):
                            logger.debug(f"   ❌ Filtered '{label}': matches exclusion pattern '{pattern}'")
                            removed_labels.append((label, f"matches exclusion: {pattern}"))
                            excluded = True
                            break
                    if excluded:
                        continue
                
                # Check required patterns
                if 'require_if_not_contains' in rules:
                    has_required = False
                    for pattern in rules['require_if_not_contains']:
                        if re.search(pattern, text_lower):
                            has_required = True
                            break
                    if not has_required:
                        logger.debug(f"   ❌ Filtered '{label}': missing required pattern")
                        removed_labels.append((label, "missing required pattern"))
                        continue
                
                # Check minimum length
                if 'min_length' in rules:
                    if len(text) < rules['min_length']:
                        logger.debug(f"   ❌ Filtered '{label}': text too short ({len(text)} < {rules['min_length']})")
                        removed_labels.append((label, f"text too short: {len(text)} < {rules['min_length']}"))
                        continue
            
            # Passed all checks
            filtered_labels.append(label)
            filtered_scores[label] = score
        
        if removed_labels:
            logger.info(f"🔧 Rule-based filter removed {len(removed_labels)} labels:")
            for label, reason in removed_labels:
                logger.info(f"   • {label}: {reason}")
        
        return filtered_labels, filtered_scores

