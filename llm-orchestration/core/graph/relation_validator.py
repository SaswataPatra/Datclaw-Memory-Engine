"""
DAPPY Relation Validator

Validates extracted relations for semantic coherence.
Prevents nonsense relations like "holds_title 100%".

Phase 1F.3 Implementation
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of relation validation."""
    is_valid: bool
    confidence_penalty: float  # 0.0 to 1.0 (multiply with original confidence)
    reason: str


class RelationValidator:
    """
    Validates relations for semantic coherence.
    
    Checks:
    1. Entity type compatibility with relation
    2. Semantic plausibility
    3. Common sense violations
    """
    
    # Relation → Required entity types
    RELATION_TYPE_CONSTRAINTS = {
        "holds_title": {
            "subject": ["person", "PERSON"],
            "object": ["ROLE", "role", "person"]  # Must be a role/title, not a value!
        },
        "employed_by": {
            "subject": ["person", "PERSON"],
            "object": ["organization", "ORG", "ORG_LIKE", "FACILITY"]
        },
        "located_in": {
            "object": ["location", "LOC", "GPE", "FACILITY"]
        },
        "is_a": {
            "object": ["ROLE", "CATEGORY", "ATTRIBUTE", "person", "temporal"]
        },
        "has_age": {
            "subject": ["person", "PERSON"],
            "object": ["temporal", "DATE", "value"]
        },
    }
    
    # Entity types that should NEVER be objects for certain relations
    INVALID_OBJECT_TYPES = {
        "holds_title": ["value", "temporal", "DATE", "location", "LOC"],
        "employed_by": ["value", "temporal", "DATE", "person", "PERSON"],
        "located_in": ["value", "temporal", "DATE", "person", "PERSON"],
    }
    
    # Patterns that indicate nonsense
    NONSENSE_PATTERNS = [
        (r'^\d+%$', "percentage value"),  # "100%"
        (r'^[<>]=?\d+', "comparison operator"),  # "<5", ">=10"
        (r'^\d+$', "bare number"),  # "42"
    ]
    
    def __init__(self):
        """Initialize relation validator."""
        logger.info("✅ RelationValidator initialized")
    
    def validate(
        self,
        relation: str,
        subject_text: str,
        subject_type: Optional[str],
        object_text: str,
        object_type: Optional[str]
    ) -> ValidationResult:
        """
        Validate a relation for semantic coherence.
        
        Args:
            relation: The relation type (e.g., "holds_title")
            subject_text: Subject entity text
            subject_type: Subject entity type
            object_text: Object entity text
            object_type: Object entity type
        
        Returns:
            ValidationResult with is_valid, confidence_penalty, and reason
        """
        # Check 1: Entity type constraints
        if relation in self.RELATION_TYPE_CONSTRAINTS:
            constraints = self.RELATION_TYPE_CONSTRAINTS[relation]
            
            # Check subject type
            if "subject" in constraints:
                if subject_type and subject_type not in constraints["subject"]:
                    return ValidationResult(
                        is_valid=False,
                        confidence_penalty=0.3,
                        reason=f"Invalid subject type '{subject_type}' for relation '{relation}' (expected: {constraints['subject']})"
                    )
            
            # Check object type
            if "object" in constraints:
                if object_type and object_type not in constraints["object"]:
                    return ValidationResult(
                        is_valid=False,
                        confidence_penalty=0.3,
                        reason=f"Invalid object type '{object_type}' for relation '{relation}' (expected: {constraints['object']})"
                    )
        
        # Check 2: Invalid object types for specific relations
        if relation in self.INVALID_OBJECT_TYPES:
            if object_type in self.INVALID_OBJECT_TYPES[relation]:
                return ValidationResult(
                    is_valid=False,
                    confidence_penalty=0.2,
                    reason=f"Invalid object type '{object_type}' for relation '{relation}'"
                )
        
        # Check 3: Nonsense patterns in object text
        import re
        for pattern, pattern_name in self.NONSENSE_PATTERNS:
            if re.match(pattern, object_text.strip()):
                return ValidationResult(
                    is_valid=False,
                    confidence_penalty=0.1,
                    reason=f"Object text '{object_text}' matches nonsense pattern: {pattern_name}"
                )
        
        # Check 4: Common sense violations
        # "holds_title" with non-role objects
        if relation == "holds_title":
            # Common non-role words that should be rejected
            non_roles = ["agree", "disagree", "yes", "no", "maybe", "ok", "okay"]
            if object_text.lower() in non_roles:
                return ValidationResult(
                    is_valid=False,
                    confidence_penalty=0.1,
                    reason=f"'{object_text}' is not a valid title/role"
                )
        
        # All checks passed
        return ValidationResult(
            is_valid=True,
            confidence_penalty=1.0,
            reason="Valid relation"
        )
    
    def should_trigger_llm_fallback(
        self,
        relation: str,
        subject_type: Optional[str],
        object_type: Optional[str],
        base_confidence: float
    ) -> bool:
        """
        Determine if LLM fallback should be triggered despite high confidence.
        
        This catches cases where dependency parsing is confident but semantically wrong.
        """
        # If types don't match constraints, trigger LLM even with high confidence
        if relation in self.RELATION_TYPE_CONSTRAINTS:
            constraints = self.RELATION_TYPE_CONSTRAINTS[relation]
            
            if "subject" in constraints and subject_type:
                if subject_type not in constraints["subject"]:
                    logger.debug(f"🚨 Triggering LLM fallback: subject type mismatch for '{relation}'")
                    return True
            
            if "object" in constraints and object_type:
                if object_type not in constraints["object"]:
                    logger.debug(f"🚨 Triggering LLM fallback: object type mismatch for '{relation}'")
                    return True
        
        # If object type is in invalid list, trigger LLM
        if relation in self.INVALID_OBJECT_TYPES and object_type:
            if object_type in self.INVALID_OBJECT_TYPES[relation]:
                logger.debug(f"🚨 Triggering LLM fallback: invalid object type for '{relation}'")
                return True
        
        return False

