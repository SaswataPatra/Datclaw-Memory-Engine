"""
Classification Module

Handles all memory classification logic including:
- ClassifierManager: Main orchestrator for all classification operations
- Zero-shot classification (via HF API or local models)
- Semantic coherence validation (LLM-based false positive detection)
- Confidence distribution analysis (statistical validation)
- Regex fallback (pattern-based classification when ML fails)

Usage:
    from services.classification import ClassifierManager, SemanticValidator, RegexFallback
    
    # Main classification (recommended)
    manager = ClassifierManager(
        classifier_type="hf_api",
        memory_classifier=hf_classifier,
        semantic_validator=validator,
        regex_fallback=fallback,
        label_discovery=discovery,
        label_store=store,
        ml_executor=executor
    )
    labels, scores = await manager.classify_memory("I love Python", user_id="user123")
    
    # Direct validation (for custom workflows)
    validator = SemanticValidator(llm_service)
    filtered_labels, filtered_scores, needs_discovery = await validator.validate_classification(
        text="I love Python",
        predicted_labels=["preference", "medical_condition"],
        scores={"preference": 0.95, "medical_condition": 0.58}
    )
    
    # Direct fallback (for testing)
    fallback = RegexFallback()
    triggers = fallback.detect_triggers("My name is John")
"""

from .classifier_manager import ClassifierManager
from .semantic_validator import SemanticValidator
from .regex_fallback import RegexFallback
from .label_filter import LabelFilter

__all__ = [
    'ClassifierManager',
    'SemanticValidator',
    'RegexFallback',
    'LabelFilter',
]

