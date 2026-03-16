"""
Semantic Validator

Validates classification results using:
1. Semantic Coherence: LLM-based validation of label appropriateness
2. Confidence Distribution Analysis: Statistical analysis of score patterns

Prevents false positives like "medical_condition" being triggered by "memory project".

ENHANCEMENT: Logs corrections to TrainingDataCollector for future model training.
"""

import logging
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from services.training_data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


class SemanticValidator:
    """
    Validates zero-shot classification results to prevent false positives.
    
    Uses two complementary approaches:
    1. LLM-based semantic coherence checking
    2. Statistical confidence distribution analysis
    
    ENHANCEMENT: Logs all corrections to TrainingDataCollector for dataset building.
    """
    
    def __init__(self, llm_service, training_collector: Optional['TrainingDataCollector'] = None):
        """
        Initialize semantic validator.
        
        Args:
            llm_service: LLM service for coherence checking
            training_collector: Optional TrainingDataCollector for logging corrections
        """
        self.llm_service = llm_service
        self.training_collector = training_collector
        self.validation_cache = {}  # Cache recent validations to save API calls
    
    async def validate_classification(
        self,
        text: str,
        predicted_labels: List[str],
        scores: Dict[str, float],
        threshold: float = 0.5
    ) -> Tuple[List[str], Dict[str, float], bool]:
        """
        Validate classification results and filter out false positives.
        
        Args:
            text: The text that was classified
            predicted_labels: Labels predicted by classifier
            scores: Confidence scores for each label
            threshold: Classification threshold
        
        Returns:
            (filtered_labels, filtered_scores, needs_discovery)
            - filtered_labels: Valid labels after filtering
            - filtered_scores: Scores for valid labels
            - needs_discovery: True if validation suggests new labels needed
        """
        if not predicted_labels:
            return [], {}, False
        
        # Step 1: Confidence distribution analysis (statistical - fast, always run)
        is_suspicious, reason = self.analyze_confidence_distribution(
            scores=scores,
            threshold=threshold,
            text=text
        )
        
        # Step 2: Semantic coherence check (LLM-based - expensive, only run if suspicious)
        is_coherent = True
        invalid_labels = []
        
        if is_suspicious:
            # Only call LLM if statistical analysis flags issues
            logger.info(f"🔍 Running LLM coherence check due to suspicious distribution")
            is_coherent, invalid_labels = await self.check_semantic_coherence(
                text=text,
                predicted_labels=predicted_labels,
                scores=scores
            )
        
        # Filter out invalid labels
        filtered_labels = [
            label for label in predicted_labels
            if label not in invalid_labels
        ]
        
        filtered_scores = {
            label: score for label, score in scores.items()
            if label in filtered_labels
        }
        
        # Determine if discovery is needed
        needs_discovery = (not is_coherent) or is_suspicious
        
        if invalid_labels:
            logger.info(f"🧹 Filtered out invalid labels: {invalid_labels}")
        
        if is_suspicious:
            logger.warning(f"⚠️  Suspicious confidence distribution: {reason}")
        
        return filtered_labels, filtered_scores, needs_discovery
    
    async def check_semantic_coherence(
        self,
        text: str,
        predicted_labels: List[str],
        scores: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Use LLM to validate if predicted labels actually make semantic sense.
        
        Common false positives:
        - "medical_condition" triggered by "memory" in technical contexts
        - "treatment_implications" triggered by "project" or "plan"
        - "health_info" triggered by "system health" or "app health"
        
        Args:
            text: The text that was classified
            predicted_labels: Labels predicted by zero-shot classifier
            scores: Confidence scores for each label
        
        Returns:
            (is_coherent, invalid_labels)
            - is_coherent: True if labels make sense overall
            - invalid_labels: List of labels that don't fit
        """
        if not predicted_labels:
            return True, []
        
        # Check cache to avoid redundant API calls
        cache_key = f"{text[:100]}:{','.join(sorted(predicted_labels))}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Build prompt for LLM validation
        labels_with_scores = [
            f"{label} ({scores.get(label, 0):.2f})"
            for label in predicted_labels
        ]
        
        prompt = f"""Analyze if these classification labels make semantic sense for the given text.

Text: "{text}"

Predicted Labels: {', '.join(labels_with_scores)}

Current Label Definitions:
- identity: USER'S OWN name, age, gender, pronouns, personal identity (NOT pet names, NOT other people)
- family: Family relationships, relatives
- preference: Likes, dislikes, favorites, tastes
- fact: Work, location, education, biographical facts
- high_value: Sensitive personal information (financial, passwords, SSN)
- goal: Personal goals, aspirations, plans
- relationship: Non-family relationships, friends
- event: Events, experiences, things that happened
- opinion: Opinions, beliefs, views, perspectives
- medical_condition: Diagnosed health conditions, illnesses, symptoms
- treatment_implications: Medical treatments, medications, therapies
- personal_health: General health information, wellness

Task: Identify which labels are FALSE POSITIVES (don't fit the text).

Common false positives:
- "medical_condition" triggered by "memory" in software/technical contexts
- "treatment_implications" triggered by "project", "plan", "strategy"  
- "personal_health" triggered by "system health", "app health"
- "identity" triggered by "me" in rhetorical questions ("is it just me?")
- "identity" triggered by PET NAMES (e.g., "my cat's name is Shero" - this is pet_info, NOT identity)
- "high_value" triggered by casual mentions of money/tech or pet ownership
- "expert_opinion" triggered by casual statements (e.g., "I love my cat" is NOT an expert opinion)
- "event" triggered by general statements (e.g., "I have a cat" is NOT an event, it's a fact/preference)

IMPORTANT: Only mark labels as invalid if they are CLEARLY wrong. If a label is somewhat relevant, keep it.

Respond ONLY in JSON format (no markdown):
{{
  "is_coherent": true/false,
  "invalid_labels": ["label1", "label2"],
  "reasoning": "brief explanation"
  "valid_labels": ["label1", "label2"],
  "valid_labels_reasoning": "brief explanation"
  "invalid_labels_reasoning": "brief explanation"
}}"""

        try:
            # Import LLMMessage for proper message formatting
            from llm.providers.base import LLMMessage
            
            response = await self.llm_service.chat(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,  # Deterministic
                max_tokens=250
            )
            
            # Extract text from LLMResponse object
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = response_text.strip()
            
            # Clean response (remove markdown code blocks if present)
            if response_text.startswith("```"):
                # Extract JSON from markdown code block
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Parse JSON response
            result = json.loads(response_text)
            
            is_coherent = result.get('is_coherent', True)
            invalid_labels = result.get('invalid_labels', [])
            reasoning = result.get('reasoning', '')
            
            if not is_coherent and invalid_labels:
                logger.info(f"✅ Semantic validation: Caught {len(invalid_labels)} false positive(s)")
                logger.info(f"   Text: {text[:100]}...")
                logger.info(f"   False positives: {invalid_labels}")
                logger.info(f"   Reasoning: {reasoning}")
                
                # LOG CORRECTION TO TRAINING DATA COLLECTOR
                if self.training_collector:
                    try:
                        max_confidence = max(scores.values()) if scores else None
                        self.training_collector.log_semantic_correction(
                            text=text,
                            predicted_labels=predicted_labels,
                            invalid_labels=invalid_labels,
                            scores=scores,
                            reasoning=reasoning,
                            classifier_confidence=max_confidence
                        )
                    except Exception as e:
                        logger.error(f"Failed to log semantic correction: {e}")
            
            # Cache result
            self.validation_cache[cache_key] = (is_coherent, invalid_labels)
            
            # Limit cache size
            if len(self.validation_cache) > 100:
                # Remove oldest entry
                self.validation_cache.pop(next(iter(self.validation_cache)))
            
            return is_coherent, invalid_labels
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response: {response[:200]}...")
            # On parse error, assume coherent (fail open)
            return True, []
        
        except Exception as e:
            logger.error(f"Semantic coherence check failed: {e}")
            # On error, assume coherent (fail open)
            return True, []
    
    def analyze_confidence_distribution(
        self,
        scores: Dict[str, float],
        threshold: float = 0.5,
        text: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Analyze confidence score distribution to detect suspicious patterns.
        
        Suspicious patterns indicate classifier uncertainty:
        1. Too many labels (>5) - classifier is confused/overfitting
        2. Multiple mid-range scores (0.5-0.7) - classifier is uncertain
        3. High variance - some very high, some barely above threshold
        
        Args:
            scores: Label confidence scores
            threshold: Classification threshold
        
        Returns:
            (is_suspicious, reason)
            - is_suspicious: True if distribution looks suspicious
            - reason: Explanation of why it's suspicious
        """
        if not scores:
            return False, ""
        
        above_threshold = {k: v for k, v in scores.items() if v >= threshold}
        
        if not above_threshold:
            return False, ""
        
        # Pattern 1: Too many labels (>5 suggests overfitting)
        if len(above_threshold) > 5:
            reason = f"Too many labels ({len(above_threshold)}) above threshold - classifier may be overfitting"
            
            # LOG ANOMALY
            if self.training_collector and text:
                try:
                    self.training_collector.log_confidence_anomaly(
                        text=text,
                        predicted_labels=list(above_threshold.keys()),
                        scores=scores,
                        issue_type='too_many_labels',
                        issue_description=reason
                    )
                except Exception as e:
                    logger.error(f"Failed to log confidence anomaly: {e}")
            
            return True, reason
        
        # Pattern 2: Multiple labels in mid-range (0.5-0.7) with similar scores
        # This suggests the classifier is uncertain and guessing
        mid_range_labels = {k: v for k, v in above_threshold.items() if 0.5 <= v <= 0.7}
        if len(mid_range_labels) >= 3:
            score_values = list(mid_range_labels.values())
            score_variance = max(score_values) - min(score_values)
            if score_variance < 0.15:  # Scores are very similar (within 0.15)
                labels_str = ', '.join([f"{k}:{v:.2f}" for k, v in mid_range_labels.items()])
                reason = f"Multiple mid-range labels with similar scores ({labels_str}) - uncertain classification"
                
                # LOG ANOMALY
                if self.training_collector and text:
                    try:
                        self.training_collector.log_confidence_anomaly(
                            text=text,
                            predicted_labels=list(above_threshold.keys()),
                            scores=scores,
                            issue_type='suspicious_distribution',
                            issue_description=reason
                        )
                    except Exception as e:
                        logger.error(f"Failed to log confidence anomaly: {e}")
                
                return True, reason
        
        # Pattern 3: High variance (one very high, others barely above threshold)
        # This suggests one label is correct, others are noise
        if len(above_threshold) >= 3:
            score_values = list(above_threshold.values())
            max_score = max(score_values)
            min_score = min(score_values)
            if max_score > 0.85 and min_score < 0.6:
                return True, f"High variance in scores (max={max_score:.2f}, min={min_score:.2f}) - mixed confidence"
        
        return False, ""
    
    def clear_cache(self):
        """Clear the validation cache."""
        self.validation_cache.clear()
        logger.info("Semantic validation cache cleared")

