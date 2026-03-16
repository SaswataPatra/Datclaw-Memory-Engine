"""
Classifier Manager

Orchestrates all memory classification logic:
- Zero-shot classification (HF API or local)
- DistilBERT classification (legacy)
- Sentence-level classification
- Label discovery integration
- Semantic validation
- Rule-based filtering (NEW)
- Regex fallback
- Confidence-based routing

This is the main entry point for all classification operations.
"""

import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from .label_filter import LabelFilter

logger = logging.getLogger(__name__)

# Confidence-based routing thresholds
CONFIDENCE_THRESHOLDS = {
    'auto_accept': 0.85,      # High confidence - accept automatically, skip LLM validation
    'semantic_check': 0.60,   # Medium confidence - run LLM semantic validation
    'force_fallback': 0.60    # Low confidence - use regex fallback
}


class ClassifierManager:
    """
    Manages all memory classification operations.
    
    Responsibilities:
    - Route to appropriate classifier (HF API, zero-shot, DistilBERT, regex)
    - Handle sentence-level classification
    - Integrate semantic validation
    - Trigger label discovery when needed
    - Coordinate fallback mechanisms
    """
    
    def __init__(
        self,
        classifier_type: str,
        memory_classifier,  # HF API or zero-shot classifier
        semantic_validator,  # SemanticValidator instance
        regex_fallback,  # RegexFallback instance
        label_discovery=None,  # DynamicLabelDiscovery instance
        label_store=None,  # LabelStore instance
        ml_executor=None,  # ThreadPoolExecutor for CPU-bound operations
        ml_scorers=None,  # Dict of ML scorers (for importance map updates)
        config: Optional[Dict[str, Any]] = None,
        use_rule_filter: bool = True  # Enable rule-based filtering
    ):
        """
        Initialize classifier manager.
        
        Args:
            classifier_type: Type of classifier ("hf_api", "zeroshot", "distilbert", "regex")
            memory_classifier: The main classifier instance
            semantic_validator: Validator for filtering false positives
            regex_fallback: Fallback pattern-based classifier
            label_discovery: Optional label discovery service
            label_store: Optional label persistence store
            ml_executor: ThreadPoolExecutor for running CPU-bound operations
            ml_scorers: Dict of ML scorers (for updating importance maps)
            config: Configuration dict
        """
        self.classifier_type = classifier_type
        self.memory_classifier = memory_classifier
        self.semantic_validator = semantic_validator
        self.regex_fallback = regex_fallback
        self.label_discovery = label_discovery
        self.label_store = label_store
        self.ml_executor = ml_executor
        self.ml_scorers = ml_scorers
        self.config = config or {}
        self.use_rule_filter = use_rule_filter
        
        # Initialize rule-based filter
        if use_rule_filter:
            self.label_filter = LabelFilter()
            logger.info("  ✅ Rule-based label filter enabled")
        else:
            self.label_filter = None
            logger.info("  ⚠️  Rule-based label filter disabled")
        
        logger.info(f"✅ ClassifierManager initialized with type: {classifier_type}")
    
    async def classify_memory(
        self,
        message: str,
        user_id: str,
        conversation_context: Optional[List[Dict]] = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Main entry point for memory classification.
        
        Routes to appropriate classifier based on type.
        
        Args:
            message: User message to classify
            user_id: User ID for label discovery
            conversation_context: Recent conversation history (reserved for future use)
        
        Returns:
            (predicted_labels, scores)
        """
        if self.classifier_type == "hf_api":
            return await self._classify_with_zeroshot(message, user_id, conversation_context)
        elif self.classifier_type == "zeroshot":
            return await self._classify_with_zeroshot(message, user_id, conversation_context)
        elif self.classifier_type == "llm":
            # Direct LLM classification (fast, no HF API)
            return await self._classify_with_llm(message, user_id)
        elif self.classifier_type == "distilbert":
            # TODO: Move DistilBERT classification here if needed
            raise NotImplementedError("DistilBERT classification should be handled in chatbot_service for now")
        elif self.classifier_type == "regex":
            # Regex fallback (fast, no API calls)
            triggers = self.regex_fallback.detect_triggers(message)
            label_scores = {label: 1.0 for label in triggers}
            # Return same format as zeroshot: (triggers, label_scores, classifier_confidence, needs_discovery)
            classifier_confidence = 0.8 if triggers else 0.3
            needs_discovery = False
            return triggers, label_scores, classifier_confidence, needs_discovery
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    async def _classify_with_zeroshot(
        self,
        message: str,
        user_id: str,
        conversation_context: Optional[List[Dict]] = None  # Reserved for future use
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Classify memory type using Zero-Shot classifier with sentence-level granularity.
        
        Uses environment-based model selection and OpenAI for label discovery.
        Includes semantic coherence checking and confidence distribution analysis.
        
        Args:
            message: User message (can be a long paragraph)
            user_id: User ID for label discovery
            conversation_context: Reserved for future context-aware classification
        
        Returns:
            (predicted_labels, scores)
        """
        _ = conversation_context  # Suppress unused parameter warning
        
        if not self.memory_classifier:
            # Fallback to regex if zero-shot failed to load
            logger.warning("Zero-shot classifier not available, falling back to regex")
            triggers = self.regex_fallback.detect_triggers(message)
            return triggers, {label: 1.0 for label in triggers}
        
        try:
            # Split message into sentences for fine-grained classification
            from ml.utils import split_for_memory
            sentences = split_for_memory(message, min_words=3, filter_questions=True)
        
            logger.info(f"Split message into {len(sentences)} sentence(s) for classification")
            
            # Classify each sentence independently
            all_labels = set()
            aggregated_scores = {}
            needs_discovery_count = 0
            
            for i, sentence in enumerate(sentences, 1):
                # Get prediction for this sentence
                try:
                    # HF API classifier is async, call directly with 3 retries
                    # Local zero-shot is sync, run in thread pool
                    if self.classifier_type == "hf_api":
                        predicted_labels, scores, needs_discovery = await self.memory_classifier.predict_single(
                            sentence,
                            threshold=0.5,
                            max_retries=3  # Try 3 times before giving up
                        )
                    else:
                        # Run local zero-shot in thread pool to avoid blocking
                        predicted_labels, scores, needs_discovery = await self.ml_executor.run(
                            self.memory_classifier.predict_single,
                            sentence,
                            threshold=0.5,
                            timeout=60.0  # Increased to 60s for complex sentences on CPU
                        )
                    
                    # Check if classifier returned empty results after all retries
                    # This means API failed 3 times - fall back to regex
                    if not predicted_labels and not scores:
                        logger.warning(f"⚠️  Classifier failed for sentence {i} after retries. Falling back to regex.")
                        logger.warning(f"   Sentence: {sentence[:100]}...")
                        regex_triggers = self.regex_fallback.detect_triggers(sentence)
                        if regex_triggers:
                            for trigger in regex_triggers:
                                all_labels.add(trigger)
                                aggregated_scores[trigger] = max(aggregated_scores.get(trigger, 0.0), 0.7)
                            logger.info(f"  Sentence {i} (regex fallback): {sentence[:50]}... → {regex_triggers}")
                        continue
                    
                    if needs_discovery:
                        needs_discovery_count += 1
                    
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️  Zero-shot prediction timed out for sentence {i} (60s). Falling back to regex.")
                    logger.warning(f"   Sentence: {sentence[:100]}...")
                    # Fallback to regex for this sentence
                    regex_triggers = self.regex_fallback.detect_triggers(sentence)
                    if regex_triggers:
                        for trigger in regex_triggers:
                            all_labels.add(trigger)
                            aggregated_scores[trigger] = max(aggregated_scores.get(trigger, 0.0), 0.7)
                        logger.info(f"  Sentence {i} (timeout fallback): {sentence[:50]}... → {regex_triggers}")
                    continue
                except Exception as e:
                    logger.error(f"⚠️  Zero-shot prediction failed for sentence {i}: {e}")
                    # Fall back to regex on exception
                    regex_triggers = self.regex_fallback.detect_triggers(sentence)
                    if regex_triggers:
                        for trigger in regex_triggers:
                            all_labels.add(trigger)
                            aggregated_scores[trigger] = max(aggregated_scores.get(trigger, 0.0), 0.7)
                        logger.info(f"  Sentence {i} (error fallback): {sentence[:50]}... → {regex_triggers}")
                    continue
                
                if predicted_labels:
                    logger.info(f"  Sentence {i}: {sentence[:50]}... → {predicted_labels}")
                    all_labels.update(predicted_labels)
                    
                    # Aggregate scores (take maximum score for each label across sentences)
                    for label, score in scores.items():
                        if label in predicted_labels:  # Only aggregate for predicted labels
                            aggregated_scores[label] = max(
                                aggregated_scores.get(label, 0.0),
                                score
                            )
                else:
                    # No labels predicted (timeout/failure) - fall back to regex
                    logger.warning(f"  ⚠️  Classifier returned empty for sentence {i}. Falling back to regex.")
                    logger.warning(f"     Sentence: {sentence[:50]}...")
                    regex_triggers = self.regex_fallback.detect_triggers(sentence, use_discovered_labels=True)
                    if regex_triggers:
                        for trigger in regex_triggers:
                            all_labels.add(trigger)
                            aggregated_scores[trigger] = max(aggregated_scores.get(trigger, 0.0), 0.7)
                        logger.info(f"  Sentence {i} (regex fallback): {sentence[:50]}... → {regex_triggers}")
            
            # Convert set to list
            final_labels = list(all_labels)
            
            if not final_labels:
                logger.info("No memory-worthy content detected in any sentence")
                return [], {}
            
            logger.info(f"Zero-shot aggregated labels: {final_labels}")
            
            # 🔧 RULE-BASED FILTERING (Fast, deterministic)
            # Apply before expensive LLM semantic validation
            if self.label_filter:
                logger.debug("🔧 Applying rule-based filter...")
                final_labels, aggregated_scores = self.label_filter.apply_rules(
                    text=message,
                    predicted_labels=final_labels,
                    scores=aggregated_scores
                )
            
            # 🎯 CONFIDENCE-BASED ROUTING
            # Determine if we need semantic validation based on confidence
            max_confidence = max(aggregated_scores.values()) if aggregated_scores else 0.0
            routing_decision = self._determine_routing(max_confidence)
            
            logger.info(f"🎯 Routing decision: {routing_decision} (max confidence: {max_confidence:.2f})")
            
            # 🔍 SEMANTIC VALIDATION (Coherence + Confidence Distribution)
            # ALWAYS run confidence distribution analysis (fast, statistical)
            # ONLY skip expensive LLM coherence check for auto-accept
            
            if routing_decision == 'auto_accept':
                # Step 1: Run confidence distribution analysis (fast, no LLM call)
                is_suspicious, reason = self.semantic_validator.analyze_confidence_distribution(
                    scores=aggregated_scores,
                    threshold=0.5,
                    text=message
                )
                
                if is_suspicious:
                    # Even with high confidence, if distribution is suspicious, validate
                    logger.warning(f"⚠️  High confidence but suspicious distribution: {reason}")
                    logger.info(f"🔍 Running full semantic validation despite auto-accept")
                    final_labels, aggregated_scores, validation_needs_discovery = await self.semantic_validator.validate_classification(
                        text=message,
                        predicted_labels=final_labels,
                        scores=aggregated_scores,
                        threshold=0.5
                    )
                else:
                    # Clean high-confidence prediction - skip LLM validation
                    logger.info(f"✅ Auto-accepting high-confidence classification (clean distribution, skipping LLM validation)")
                    validation_needs_discovery = False
            else:
                # Medium/low confidence - run full semantic validation
                final_labels, aggregated_scores, validation_needs_discovery = await self.semantic_validator.validate_classification(
                    text=message,
                    predicted_labels=final_labels,
                    scores=aggregated_scores,
                    threshold=0.5
                )
            
            # If all labels were filtered out, fall back to regex
            if not final_labels:
                logger.warning("⚠️  All labels filtered by semantic validation. Falling back to regex.")
                regex_triggers = self.regex_fallback.detect_triggers(message)
                return regex_triggers, {label: 0.7 for label in regex_triggers}
            
            # Update discovery flag if validation suggests it
            if validation_needs_discovery:
                needs_discovery_count += 1
            
            # Trigger label discovery if needed
            if self.label_discovery and needs_discovery_count > 0:
                logger.info(f"🔍 {needs_discovery_count}/{len(sentences)} sentences had low confidence")
                
                try:
                    # Discover new labels for the full message
                    new_labels_with_importance = await self.label_discovery.discover_labels(
                        text=message,
                        existing_labels=self.memory_classifier.current_labels,
                        current_scores=aggregated_scores
                    )
                    
                    if new_labels_with_importance:
                        label_names = [l['name'] for l in new_labels_with_importance]
                        logger.info(f"✨ Discovered new labels: {label_names}")
                        
                        # LOG LABEL DISCOVERY
                        if self.semantic_validator and self.semantic_validator.training_collector:
                            try:
                                self.semantic_validator.training_collector.log_label_discovery(
                                    text=message,
                                    existing_labels=list(self.memory_classifier.current_labels),
                                    discovered_labels=new_labels_with_importance,
                                    user_id=user_id
                                )
                            except Exception as e:
                                logger.error(f"Failed to log label discovery: {e}")
                        
                        # Add to classifier
                        self.memory_classifier.add_labels(label_names)
                        
                        # Add to importance map dynamically
                        if self.ml_scorers and 'explicit_importance' in self.ml_scorers:
                            for label_info in new_labels_with_importance:
                                label_name = label_info['name']
                                importance = label_info['importance']
                                self.ml_scorers['explicit_importance'].importance_map[label_name] = importance
                                logger.info(f"  📊 Set importance: {label_name} = {importance:.2f}")
                        
                        # Save to store with importance
                        for label_info in new_labels_with_importance:
                            self.label_store.add_label(
                                label=label_info['name'],
                                context=message[:200],
                                user_id=user_id,
                                importance=label_info['importance']
                            )
                        
                        # Re-classify with new labels AND re-validate
                        logger.info("🔄 Re-classifying with new labels...")
                        if self.classifier_type == "hf_api":
                            reclassified_labels, reclassified_scores, _ = await self.memory_classifier.predict_single(
                                message,
                                threshold=0.5
                            )
                        else:
                            reclassified_labels, reclassified_scores, _ = await self.ml_executor.run(
                                self.memory_classifier.predict_single,
                                message,
                                threshold=0.5,
                                timeout=60.0
                            )
                        
                        # IMPORTANT: Re-validate the new classification to filter false positives
                        logger.info("🔍 Re-validating re-classified labels...")
                        final_labels, aggregated_scores, _ = await self.semantic_validator.validate_classification(
                            text=message,
                            predicted_labels=reclassified_labels,
                            scores=reclassified_scores,
                            threshold=0.5
                        )
                        logger.info(f"  Updated labels (after validation): {final_labels}")
                
                except Exception as e:
                    logger.error(f"Label discovery failed: {e}", exc_info=True)
            
            return final_labels, aggregated_scores
            
        except Exception as e:
            logger.error(f"Error in zero-shot classification: {e}", exc_info=True)
            # Fallback to regex
            triggers = self.regex_fallback.detect_triggers(message)
            return triggers, {label: 1.0 for label in triggers}
    
    def _determine_routing(self, max_confidence: float) -> str:
        """
        Determine routing decision based on classifier confidence.
        
        Routing strategies:
        - auto_accept (≥0.85): High confidence - skip LLM validation, accept immediately
        - semantic_check (0.60-0.85): Medium confidence - run LLM validation
        - force_fallback (<0.60): Low confidence - use regex fallback
        
        Args:
            max_confidence: Maximum confidence score from classifier
        
        Returns:
            Routing decision: 'auto_accept', 'semantic_check', or 'force_fallback'
        """
        if max_confidence >= CONFIDENCE_THRESHOLDS['auto_accept']:
            return 'auto_accept'
        elif max_confidence >= CONFIDENCE_THRESHOLDS['semantic_check']:
            return 'semantic_check'
        else:
            return 'force_fallback'
    
    async def _classify_with_llm(
        self,
        message: str,
        user_id: str
    ) -> Tuple[List[str], Dict[str, float], float, bool]:
        """
        Classify memory using direct LLM call (OpenAI gpt-4o-mini).
        Fast alternative to HuggingFace API, no zero-shot model needed.
        
        Args:
            message: User message to classify
            user_id: User ID for label discovery
        
        Returns:
            (triggers, label_scores, classifier_confidence, needs_discovery)
        """
        try:
            from openai import AsyncOpenAI
            import json
            
            # Get available labels
            base_labels = self.memory_classifier.base_labels if hasattr(self.memory_classifier, 'base_labels') else []
            discovered_labels = []
            if self.label_discovery:
                discovered_labels = self.label_discovery.get_discovered_labels(user_id)
            
            all_labels = list(set(base_labels + discovered_labels))
            
            if not all_labels:
                # No labels available, use regex fallback
                logger.warning("No labels available for LLM classification, using regex")
                triggers = self.regex_fallback.detect_triggers(message)
                return triggers, {label: 0.7 for label in triggers}, 0.7, False
            
            # Create OpenAI client
            client = AsyncOpenAI()
            
            # Build prompt
            prompt = f"""Classify the following user message into one or more memory categories.

Available categories: {', '.join(all_labels)}

User message: "{message}"

Return a JSON object with:
- "labels": list of relevant category names (can be empty if none apply)
- "confidence": overall confidence (0.0-1.0)
- "scores": dict mapping each selected label to its confidence (0.0-1.0)

Example: {{"labels": ["health", "medication"], "confidence": 0.85, "scores": {{"health": 0.9, "medication": 0.8}}}}

Return ONLY the JSON object, no other text."""
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a memory classification assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            result = json.loads(result_text)
            triggers = result.get("labels", [])
            confidence = result.get("confidence", 0.7)
            label_scores = result.get("scores", {label: confidence for label in triggers})
            
            logger.info(f"  🤖 LLM classification: {triggers} (confidence: {confidence:.2f})")
            
            # Check if label discovery is needed (low confidence or new patterns)
            needs_discovery = confidence < 0.6 and len(triggers) == 0
            
            return triggers, label_scores, confidence, needs_discovery
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using regex fallback")
            triggers = self.regex_fallback.detect_triggers(message)
            return triggers, {label: 0.7 for label in triggers}, 0.7, False

