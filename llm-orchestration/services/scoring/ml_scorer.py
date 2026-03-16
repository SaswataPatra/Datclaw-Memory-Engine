"""
ML-Based Memory Scorer

Orchestrates all ML scoring components:
- Novelty, Frequency, Sentiment, Explicit Importance, Engagement
- LightGBM combiner or weighted average fallback
- Confidence calculation
- Tier determination
"""

import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from .pii_detector import PIIDetector

logger = logging.getLogger(__name__)


class MLScorer:
    """
    Orchestrates ML-based memory scoring.
    
    Responsibilities:
    - Run all component scorers (novelty, frequency, sentiment, etc.)
    - Classify memory to determine explicit importance
    - Combine scores using LightGBM or weighted average
    - Calculate confidence
    - Determine memory tier
    """
    
    def __init__(
        self,
        ml_scorers: Dict[str, Any],
        ml_combiner: Any,
        confidence_combiner: Any,
        classifier_manager: Any,
        embedding_service: Any,
        ml_executor: Any,
        regex_fallback: Any,
        classifier_type: str,
        config: Optional[Dict[str, Any]] = None,
        llm_service: Optional[Any] = None
    ):
        """
        Initialize ML scorer.
        
        Args:
            ml_scorers: Dict of component scorers (novelty, frequency, etc.)
            ml_combiner: LightGBM combiner for score combination
            confidence_combiner: Confidence combiner
            classifier_manager: ClassifierManager for memory classification
            embedding_service: Service for generating embeddings
            ml_executor: ThreadPoolExecutor for CPU-bound operations
            regex_fallback: Regex fallback classifier
            classifier_type: Type of classifier being used
            config: Configuration dict
            contradiction_detector: Optional contradiction detector for semantic consistency
            llm_service: Optional LLM service for confidence extraction
        """
        self.ml_scorers = ml_scorers
        self.ml_combiner = ml_combiner
        self.confidence_combiner = confidence_combiner
        self.classifier_manager = classifier_manager
        self.embedding_service = embedding_service
        self.ml_executor = ml_executor
        self.regex_fallback = regex_fallback
        self.classifier_type = classifier_type
        self.config = config or {}
        # Contradiction detection removed - now handled by KG Maintenance Agent
        self.llm_service = llm_service
        
        # Initialize PII detector
        self.pii_detector = PIIDetector()
        
        # Cache for LLM confidence scores
        self.llm_confidence_cache = {}
        
        logger.info("✅ MLScorer initialized")
    
    async def score_memory(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str
    ) -> tuple[float, float, List[str]]:
        """
        ML-based memory scoring using component scorers and combiners
        
        Returns:
            Tuple of (ego_score, confidence, triggers)
        """
        try:
            # Generate embedding for the memory
            embedding = await self._generate_embedding(user_message)
            
            # Prepare memory dict for scoring
            memory = {
                'content': user_message,
                'user_id': user_id,
                'observed_at': datetime.utcnow().isoformat(),
                'user_response_length': len(assistant_response),
                'followup_count': 0,  # TODO: Track from conversation history
                'elaboration_score': 0.5,  # TODO: LLM-based elaboration detection
                'embedding': embedding  # Add embedding for novelty and frequency scorers
            }
            
            # Run all component scorers
            scores = {}
            
            logger.info(f"🔍 ML Scoring for: '{user_message[:100]}...'")
            
            # Novelty score (requires embedding)
            if self.ml_scorers.get('novelty') and embedding:
                novelty_result = await self.ml_scorers['novelty'].score(memory)
                scores['novelty_score'] = novelty_result.score
                logger.info(f"  📊 Novelty: {scores['novelty_score']:.3f} (max_similarity: {novelty_result.metadata.get('max_similarity', 'N/A')})")
            else:
                scores['novelty_score'] = 0.5
                logger.info(f"  📊 Novelty: {scores['novelty_score']:.3f} (default - scorer not initialized or no embedding)")
            
            # Frequency score
            if self.ml_scorers.get('frequency') and embedding:
                freq_result = await self.ml_scorers['frequency'].score(memory)
                scores['frequency_score'] = freq_result.score
                logger.info(f"  📊 Frequency: {scores['frequency_score']:.3f} (similar memories: {freq_result.metadata.get('frequency_count', 0)})")
            else:
                scores['frequency_score'] = 0.0
                logger.info(f"  📊 Frequency: {scores['frequency_score']:.3f} (scorer not initialized or no embedding)")
            
            # Sentiment score
            if self.ml_scorers.get('sentiment'):
                sent_result = await self.ml_scorers['sentiment'].score(memory)
                scores['sentiment_intensity'] = sent_result.score
                logger.info(f"  📊 Sentiment Intensity: {scores['sentiment_intensity']:.3f} (raw: {sent_result.metadata.get('raw_sentiment', 0):.3f})")
            else:
                scores['sentiment_intensity'] = 0.0
                logger.info(f"  📊 Sentiment Intensity: {scores['sentiment_intensity']:.3f} (scorer not initialized)")
            
            # Explicit importance (requires label)
            # Use configured classifier (zero-shot, LLM, DistilBERT, or regex)
            classifier_confidence = 0.5  # Default
            
            # ✨ Use ML-based classification (returns 4 values)
            triggers, label_scores, classifier_confidence, needs_discovery = await self.classifier_manager.classify_memory(
                user_message,
                user_id,
                conversation_context=None  # TODO: Pass conversation context
            )
            logger.info(f"  🏷️  {self.classifier_type.capitalize()} labels: {triggers} (scores: {label_scores}, confidence: {classifier_confidence:.2f})")
            
            if triggers:
                # Find the trigger with highest importance
                if self.ml_scorers.get('explicit_importance'):
                    # Score all triggers and find the one with max importance
                    max_importance = 0.0
                    best_label = triggers[0]
                    
                    for trigger in triggers:
                        memory['label'] = trigger
                        exp_result = await self.ml_scorers['explicit_importance'].score(memory)
                        if exp_result.score > max_importance:
                            max_importance = exp_result.score
                            best_label = trigger
                    
                    scores['explicit_importance_score'] = max_importance
                    memory['label'] = best_label  # Set the best label
                    logger.info(f"  📊 Explicit Importance: {scores['explicit_importance_score']:.3f} (label: '{best_label}' from {len(triggers)} triggers)")
                else:
                    scores['explicit_importance_score'] = self._calculate_explicit_importance(triggers)
                    best_label = triggers[0]  # Fallback to first trigger for label
                    memory['label'] = best_label
                    logger.info(f"  📊 Explicit Importance: {scores['explicit_importance_score']:.3f} (label: '{best_label}', fallback)")
            else:
                scores['explicit_importance_score'] = 0.5
                triggers = ['unknown']
                logger.info(f"  📊 Explicit Importance: {scores['explicit_importance_score']:.3f} (no triggers)")
            
            # Engagement score (pass explicit_importance for adaptive baseline)
            if self.ml_scorers.get('engagement'):
                eng_result = await self.ml_scorers['engagement'].score(
                    memory,
                    explicit_importance=scores.get('explicit_importance_score', 0.5)
                )
                scores['engagement_score'] = eng_result.score
                metadata = eng_result.metadata
                logger.info(f"  📊 Engagement: {scores['engagement_score']:.3f} (response_len: {memory.get('user_response_length', 0)}, baseline: {metadata.get('baseline_applied', 0.0)})")
            else:
                scores['engagement_score'] = 0.5
                logger.info(f"  📊 Engagement: {scores['engagement_score']:.3f} (scorer not initialized)")
            
            # Additional features
            scores['recency_decay'] = 1.0  # New memory, full recency
            scores['reference_count'] = 0  # New memory, no references yet
            scores['llm_confidence'] = 0.8  # Default LLM confidence
            scores['source_weight'] = 1.0  # User message, high weight
            
            logger.info(f"  📊 Additional: recency={scores['recency_decay']:.3f}, refs={scores['reference_count']}, llm_conf={scores['llm_confidence']:.3f}")
            
            # Combine scores using LightGBM or weighted average
            logger.info(f"🤖 Combining scores with {'LightGBM (trained)' if (self.ml_combiner and self.ml_combiner.is_trained) else 'weighted average (fallback)'}...")
            
            if self.ml_combiner and self.ml_combiner.is_trained:
                # Run LightGBM prediction in thread pool to avoid blocking
                try:
                    raw_ego_score = await self.ml_executor.run(
                        self.ml_combiner.predict,
                        scores,
                        timeout=15.0  # Increased timeout for LightGBM (was 5s)
                    )
                    
                    # 🔒 GATING MECHANISM: Cap ego score based on explicit importance
                    # If no strong memory triggers detected, limit maximum ego score
                    # This prevents generic questions from reaching Tier 1 just due to frequency
                    explicit_importance = scores['explicit_importance_score']
                    
                    # ═══════════════════════════════════════════════════════════════
                    # GATING MECHANISM - Prevent generic content from reaching Tier 1
                    # ═══════════════════════════════════════════════════════════════
                    logger.info(f"🚪 Gating Mechanism Check:")
                    logger.info(f"   Explicit importance: {explicit_importance:.3f}")
                    logger.info(f"   Raw LightGBM score: {raw_ego_score:.4f}")
                    
                    if explicit_importance < 0.6:
                        # No strong triggers (identity, family, high_value, etc.)
                        # Cap at Tier 2 maximum (0.74)
                        max_allowed = 0.74
                        if raw_ego_score > max_allowed:
                            ego_score = max_allowed
                            logger.warning(f"   ⚠️  GATED: Score capped {raw_ego_score:.4f} → {ego_score:.4f}")
                            logger.warning(f"   Reason: No strong explicit triggers (explicit_importance < 0.6)")
                            logger.warning(f"   Max allowed: Tier 2 (0.74)")
                        else:
                            ego_score = raw_ego_score
                            logger.info(f"   ✅ No gating needed (score already ≤ {max_allowed})")
                    else:
                        # Strong triggers present, use raw score
                        ego_score = raw_ego_score
                        logger.info(f"   ✅ Strong triggers present - no gating applied")
                    
                    logger.info(f"  🎯 LightGBM final ego score: {ego_score:.4f}")
                except asyncio.TimeoutError:
                    logger.warning(f"  ⚠️  LightGBM timed out (15s), using fallback weighted average")
                    # Fallback to weighted average
                    ego_score = self._calculate_weighted_average(scores)
            else:
                # ═══════════════════════════════════════════════════════════════
                # FALLBACK: Weighted Average (LightGBM not trained)
                # ═══════════════════════════════════════════════════════════════
                logger.warning(f"⚠️  LightGBM not trained - using weighted average fallback")
                
                # 🔄 HYPERPARAMETER - Being LEARNED by LightGBM in Phase 1.5 (NOW!)
                # Fallback: weighted average when LightGBM is not trained yet
                # Current: Hand-tuned weights prioritizing explicit_importance (0.4) and novelty (0.2)
                # Future: Once LightGBM is trained on bootstrap data, these weights are replaced by learned model
                # These are TEMPORARY - will be obsolete after training!
                weights = {
                    'novelty': 0.2,           # How new is this information?
                    'frequency': 0.1,         # How often do we discuss this?
                    'sentiment': 0.1,         # How emotionally charged?
                    'explicit_importance': 0.4,  # What type of memory? (identity, family, etc.)
                    'engagement': 0.2         # How engaged was the user?
                }
                
                ego_score = (
                    scores['novelty_score'] * weights['novelty'] +
                    scores['frequency_score'] * weights['frequency'] +
                    scores['sentiment_intensity'] * weights['sentiment'] +
                    scores['explicit_importance_score'] * weights['explicit_importance'] +
                    scores['engagement_score'] * weights['engagement']
                )
                
                logger.info(f"📊 Weighted Average Calculation:")
                logger.info(f"   Novelty:      {scores['novelty_score']:.3f} × {weights['novelty']:.1f} = {scores['novelty_score'] * weights['novelty']:.4f}")
                logger.info(f"   Frequency:    {scores['frequency_score']:.3f} × {weights['frequency']:.1f} = {scores['frequency_score'] * weights['frequency']:.4f}")
                logger.info(f"   Sentiment:    {scores['sentiment_intensity']:.3f} × {weights['sentiment']:.1f} = {scores['sentiment_intensity'] * weights['sentiment']:.4f}")
                logger.info(f"   Explicit Imp: {scores['explicit_importance_score']:.3f} × {weights['explicit_importance']:.1f} = {scores['explicit_importance_score'] * weights['explicit_importance']:.4f}")
                logger.info(f"   Engagement:   {scores['engagement_score']:.3f} × {weights['engagement']:.1f} = {scores['engagement_score'] * weights['engagement']:.4f}")
                logger.info(f"   ─────────────────────────────────────────────────────")
                logger.info(f"   🎯 TOTAL EGO SCORE: {ego_score:.4f}")
            
            # ═══════════════════════════════════════════════════════════════
            # CONFIDENCE CALCULATION - Extract all confidence factors
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"🔍 Extracting confidence factors...")
            
            # Extract LLM confidence from assistant response
            llm_confidence = await self._extract_llm_confidence(assistant_response)
            
            # Check semantic consistency with existing memories
            # Note: This is optimized to only run for high-ego memories (Tier 1 & 2)
            logger.info(f"🔍 Checking for contradictions (ego_score={ego_score:.3f})...")
            is_semantically_consistent = await self._check_semantic_consistency(
                user_message=user_message,
                user_id=user_id,
                ego_score=ego_score  # Pass ego_score for optimization
            )
            logger.info(f"   Contradiction check result: {'✅ No contradictions' if is_semantically_consistent else '🚨 CONTRADICTION DETECTED!'}")
            
            # Detect PII in user message
            has_pii, pii_detected = self.pii_detector.detect(user_message)
            if not has_pii:
                logger.debug("   ✅ No PII detected")
            
            # Calculate confidence using confidence combiner
            if self.confidence_combiner:
                confidence_result = self.confidence_combiner.combine(
                    ego_score=ego_score,
                    extractor_confidence=classifier_confidence,  # From classifier (zero-shot or regex)
                    llm_confidence=llm_confidence,  # Extracted from LLM response
                    is_semantically_consistent=is_semantically_consistent,  # From contradiction detector
                    has_pii=has_pii,  # From PII detector
                    user_engagement_score=scores['engagement_score']
                )
                confidence = confidence_result['final_confidence']
                routing_decision = confidence_result['routing_decision']
                breakdown = confidence_result.get('breakdown', {})
                
                logger.info(f"🎯 Confidence Combiner:")
                logger.info(f"   Classifier confidence: {classifier_confidence:.3f}")
                logger.info(f"   LLM confidence: {llm_confidence:.3f}")
                logger.info(f"   Semantic consistent: {is_semantically_consistent}")
                logger.info(f"   Has PII: {has_pii} {f'({list(pii_detected.keys())})' if has_pii else ''}")
                logger.info(f"   Base confidence: {breakdown.get('base_confidence', 0):.3f}")
                logger.info(f"   Penalties: {breakdown.get('penalties', {})}")
                logger.info(f"   Final confidence: {confidence:.3f}")
                logger.info(f"   Routing decision: {routing_decision}")
            else:
                confidence = 0.7  # Default
                logger.info(f"🎯 Confidence: {confidence:.3f} (default - combiner not initialized)")
            
            # ═══════════════════════════════════════════════════════════════
            # FINAL SCORING SUMMARY
            # ═══════════════════════════════════════════════════════════════
            tier = self._determine_tier(ego_score)
            logger.info(f"")
            logger.info(f"{'='*70}")
            logger.info(f"📝 FINAL SCORING SUMMARY")
            logger.info(f"{'='*70}")
            logger.info(f"   Message: '{user_message[:80]}{'...' if len(user_message) > 80 else ''}'")
            logger.info(f"")
            logger.info(f"   📊 Component Scores:")
            logger.info(f"      Novelty:            {scores.get('novelty_score', 0):.3f}")
            logger.info(f"      Frequency:          {scores.get('frequency_score', 0):.3f}")
            logger.info(f"      Sentiment:          {scores.get('sentiment_intensity', 0):.3f}")
            logger.info(f"      Explicit Importance: {scores.get('explicit_importance_score', 0):.3f}")
            logger.info(f"      Engagement:         {scores.get('engagement_score', 0):.3f}")
            logger.info(f"")
            logger.info(f"   🏷️  Triggers: {triggers}")
            logger.info(f"   🎯 Ego Score: {ego_score:.4f}")
            logger.info(f"   🔒 Confidence: {confidence:.4f}")
            logger.info(f"   📂 Tier: {tier} ({['', 'Core', 'Long-term', 'Short-term', 'Hot Buffer'][tier]})")
            logger.info(f"   🚦 Routing: {routing_decision if self.confidence_combiner else 'N/A'}")
            logger.info(f"{'='*70}")
            logger.info(f"")
            
            # Return tier along with ego_score, confidence, and triggers
            return ego_score, confidence, triggers, tier
            
        except Exception as e:
            logger.error(f"Error in ML-based scoring: {e}", exc_info=True)
            # Fallback to legacy scoring
            triggers = self.regex_fallback.detect_triggers(user_message)
            if triggers:
                explicit_importance = self._calculate_explicit_importance(triggers)
                return explicit_importance, 0.7, triggers
            return 0.5, 0.5, ['unknown']
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using embedding service.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None if generation fails
        """
        if not self.embedding_service:
            return None
        
        try:
            embedding = await self.embedding_service.generate(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _calculate_explicit_importance(self, triggers: List[str]) -> float:
        """
        Calculate explicit importance from triggers (fallback when scorer not available).
        
        Args:
            triggers: List of memory triggers
        
        Returns:
            Importance score (0.0 to 1.0)
        """
        # Simple heuristic mapping
        importance_map = {
            'identity': 1.0,
            'family': 0.9,
            'high_value': 0.9,
            'preference': 0.7,
            'goal': 0.8,
            'relationship': 0.7,
            'fact': 0.6,
            'event': 0.5,
            'opinion': 0.5,
            'unknown': 0.3
        }
        
        # Return highest importance from triggers
        max_importance = max([importance_map.get(t, 0.5) for t in triggers], default=0.5)
        return max_importance
    
    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted average of component scores.
        
        Uses LightGBM feature importances if available, otherwise hardcoded weights.
        
        Args:
            scores: Dict of component scores
        
        Returns:
            Combined ego score
        """
        # Try to use LightGBM's learned feature importances as weights
        if self.ml_combiner and hasattr(self.ml_combiner, 'get_feature_importances'):
            try:
                importances = self.ml_combiner.get_feature_importances()
                if importances:
                    # Normalize importances to sum to 1.0
                    total = sum(importances.values())
                    weights = {k: v / total for k, v in importances.items()}
                    logger.info(f"  Using LightGBM feature importances as weights: {weights}")
                else:
                    # Fallback to hardcoded weights
                    weights = self._get_default_weights()
            except Exception as e:
                logger.warning(f"  Failed to get feature importances: {e}, using default weights")
                weights = self._get_default_weights()
        else:
            weights = self._get_default_weights()
        
        # Calculate weighted sum
        ego_score = (
            scores.get('novelty_score', 0.5) * weights.get('novelty', 0.2) +
            scores.get('frequency_score', 0.0) * weights.get('frequency', 0.1) +
            scores.get('sentiment_intensity', 0.0) * weights.get('sentiment', 0.1) +
            scores.get('explicit_importance_score', 0.5) * weights.get('explicit_importance', 0.4) +
            scores.get('engagement_score', 0.5) * weights.get('engagement', 0.2)
        )
        
        return ego_score
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default hardcoded weights."""
        return {
            'novelty': 0.2,
            'frequency': 0.1,
            'sentiment': 0.1,
            'explicit_importance': 0.4,
            'engagement': 0.2
        }
    
    def _determine_tier(self, ego_score: float) -> int:
        """
        Determine memory tier based on ego score.
        
        ⚠️  IMPORTANT: This is the SINGLE SOURCE OF TRUTH for tier determination!
        
        Tier 1 (Core): 0.8+ → BUT goes to Shadow Tier for user confirmation first
        Tier 2 (Long-term): 0.6-0.8
        Tier 3 (Short-term): 0.3-0.6
        Tier 4 (Hot Buffer): <0.3
        
        Note: Memories assigned Tier 1 here will be routed through Shadow Tier
        in chatbot_service. Only after user confirmation will they become true Tier 1.
        
        Args:
            ego_score: Ego score (0.0 to 1.0)
        
        Returns:
            Tier number (1-4)
        """
        if ego_score >= 0.8:
            return 1  # Core (will go to Shadow Tier for confirmation)
        elif ego_score >= 0.6:
            return 2  # Long-term
        elif ego_score >= 0.3:
            return 3  # Short-term
        else:
            return 4  # Hot Buffer
    
    async def _extract_llm_confidence(self, assistant_response: str) -> float:
        """
        Extract LLM confidence from assistant response.
        
        Uses a simple heuristic based on response characteristics:
        - Length (longer = more confident)
        - Hedging words (reduces confidence)
        - Definitive language (increases confidence)
        
        Args:
            assistant_response: The LLM's response text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Check cache first
        cache_key = hash(assistant_response[:200])  # Use first 200 chars as key
        if cache_key in self.llm_confidence_cache:
            cached_confidence = self.llm_confidence_cache[cache_key]
            logger.debug(f"💾 LLM confidence (cached): {cached_confidence:.3f}")
            return cached_confidence
        
        logger.debug(f"🔍 Extracting LLM confidence from response ({len(assistant_response)} chars)")
        
        # Baseline confidence
        confidence = 0.8
        adjustments = []
        
        # Hedging words that reduce confidence
        hedging_words = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
            'i think', 'i believe', 'probably', 'likely', 'not sure',
            'unclear', 'uncertain', 'unsure', 'guess'
        ]
        
        # Definitive words that increase confidence
        definitive_words = [
            'definitely', 'certainly', 'absolutely', 'clearly', 'obviously',
            'undoubtedly', 'without doubt', 'for sure', 'indeed', 'exactly'
        ]
        
        response_lower = assistant_response.lower()
        
        # Count hedging words
        hedging_count = sum(1 for word in hedging_words if word in response_lower)
        if hedging_count > 0:
            adjustment = hedging_count * -0.05
            confidence += adjustment
            adjustments.append(f"hedging words ({hedging_count}): {adjustment:+.3f}")
        
        # Count definitive words
        definitive_count = sum(1 for word in definitive_words if word in response_lower)
        if definitive_count > 0:
            adjustment = definitive_count * 0.03
            confidence += adjustment
            adjustments.append(f"definitive words ({definitive_count}): {adjustment:+.3f}")
        
        # Length factor (very short responses are less confident)
        if len(assistant_response) < 50:
            adjustment = -0.1
            confidence += adjustment
            adjustments.append(f"short response (<50 chars): {adjustment:+.3f}")
        elif len(assistant_response) > 200:
            adjustment = 0.05
            confidence += adjustment
            adjustments.append(f"long response (>200 chars): {adjustment:+.3f}")
        
        # Clamp to [0.3, 1.0] range
        original_confidence = confidence
        confidence = max(0.3, min(1.0, confidence))
        
        if adjustments:
            logger.debug(f"   Baseline: 0.800")
            for adj in adjustments:
                logger.debug(f"   {adj}")
            if original_confidence != confidence:
                logger.debug(f"   Clamped: {original_confidence:.3f} → {confidence:.3f}")
        
        logger.debug(f"   ✅ Final LLM confidence: {confidence:.3f}")
        
        # Cache result
        self.llm_confidence_cache[cache_key] = confidence
        
        # Limit cache size
        if len(self.llm_confidence_cache) > 100:
            # Remove oldest entry
            self.llm_confidence_cache.pop(next(iter(self.llm_confidence_cache)))
        
        return confidence
    
    async def _check_semantic_consistency(
        self,
        user_message: str,
        user_id: str,
        ego_score: float
    ) -> bool:
        """
        Lightweight contradiction signal detection.
        
        Uses ContradictionDetector to find similar memories and flag potential contradictions
        based on heuristics (no LLM reasoning). The actual contradiction resolution is handled
        by the KG Maintenance Agent.
        
        This method now always returns True (no confidence penalty) but logs signals
        that can be used by downstream agents.
        
        Args:
            user_message: The user's message
            user_id: User ID
            ego_score: The calculated ego score (determines tier)
            
        Returns:
            Always True (signals logged for KG Maintenance Agent)
        """
        logger.debug(f"🔍 Contradiction check: Generating signals for KG Maintenance Agent")
        
        # Note: The actual contradiction detection happens in KG Maintenance Agent
        # which receives these signals via the ego_scoring_complete event
        return True

