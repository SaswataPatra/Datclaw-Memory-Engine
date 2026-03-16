"""
Sentence Intent Classifier

Classifies sentences into semantic intent categories (fact, evaluation, opinion, etc.)
to route them to appropriate extractors.

This is a lightweight gate that prevents treating all sentences as world-facts.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str  # fact, evaluation, opinion, speech_act, directive
    confidence: float
    method: str  # heuristic, zero_shot, llm
    raw_scores: Dict[str, float] = None


class SentenceIntentClassifier:
    """
    Classifies sentences into intent categories using a multi-stage approach:
    1. Heuristics (fast path)
    2. Zero-shot classification (HuggingFace)
    3. LLM fallback (for edge cases)
    
    Reuses the zero-shot pattern from ZeroShotMemoryClassifier.
    """
    
    # Intent labels for classification
    INTENT_LABELS = [
        "fact",           # Objective world state: "Sarah works at Google"
        "evaluation",     # Value judgment: "Music is amazing"
        "opinion",        # Belief/stance: "I think X", "I agree with Y"
        "speech_act",     # Conversational action: "Thanks, John"
        "directive"       # Command/intent: "I want to learn Python"
    ]
    
    # Heuristic patterns for fast classification
    HEURISTIC_PATTERNS = {
        "opinion": [
            r'\b(I|we)\s+(think|believe|feel|suppose|guess|assume)\b',
            r'\b(I|we)\s+.*?\s+(agree|disagree)\b',  # Handles "I 100% agree"
            r'\b(agree|disagree)\s+with\b',
            r'\bin\s+my\s+(opinion|view|mind)\b',
        ],
        "evaluation": [
            r'\b(is|are|was|were)\s+(amazing|great|terrible|awful|wonderful|horrible|fantastic|bad|good|excellent|poor)\b',
            r'\b(love|hate|adore|despise)\s+',
            r'\b(so|very|really|extremely)\s+(good|bad|nice|cool|fun)\b',
        ],
        "speech_act": [
            r'\b(thanks|thank you|hello|hi|hey|bye|goodbye)\b',
            r',\s*[A-Z][a-z]+\s*[.!?]?$',  # Vocative at end
        ],
        "directive": [
            r'\b(I|we)\s+(want|need|would like|wish)\s+to\b',
            r'\b(tell|ask|remind|help)\s+\w+\s+to\b',
            r'^(please|kindly)\s+',
        ],
    }
    
    def __init__(
        self,
        config: Dict,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize the intent classifier.
        
        Args:
            config: Configuration dict with classifier settings
            llm_client: Optional LLM client for fallback
        """
        self.config = config.get("sentence_intent", {})
        self.llm_client = llm_client
        
        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.enable_heuristics = self.config.get("classifier", {}).get("enable_heuristics", True)
        self.enable_llm_fallback = self.config.get("classifier", {}).get("enable_llm_fallback", True)
        self.confidence_threshold = self.config.get("classifier", {}).get("confidence_threshold", 0.6)
        
        # HuggingFace API settings
        hf_config = self.config.get("classifier", {})
        self.hf_api_key = hf_config.get("api_key") or config.get("ml", {}).get("hf_api", {}).get("api_key")
        self.hf_model = hf_config.get("model", "facebook/bart-large-mnli")
        self.hf_timeout = hf_config.get("timeout", 10.0)
        
        logger.info(f"✅ SentenceIntentClassifier initialized (heuristics={self.enable_heuristics}, llm_fallback={self.enable_llm_fallback})")
    
    async def classify(self, text: str, user_id: str) -> IntentResult:
        """
        Classify a sentence into an intent category.
        
        Args:
            text: The sentence to classify
            user_id: User ID (for logging/training)
        
        Returns:
            IntentResult with intent, confidence, and method
        """
        if not self.enabled:
            return IntentResult(intent="fact", confidence=1.0, method="disabled")

        # Stage 1: Zero-shot classification (primary path; generalizes better than regex heuristics)
        try:
            intent, confidence, raw_scores = await self._classify_with_zero_shot(text)
            
            if confidence >= self.confidence_threshold:
                logger.debug(f"🎯 Intent (zero-shot): {intent} (conf={confidence:.2f}) for '{text[:50]}...'")
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    method="zero_shot",
                    raw_scores=raw_scores
                )
            else:
                logger.debug(f"⚠️  Low confidence ({confidence:.2f}) from zero-shot, trying LLM fallback")
        
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
        
        # Stage 2: LLM fallback (semantic; slower but robust)
        if self.enable_llm_fallback and self.llm_client:
            try:
                intent, confidence = await self._classify_with_llm(text)
                logger.debug(f"🎯 Intent (LLM): {intent} (conf={confidence:.2f}) for '{text[:50]}...'")
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    method="llm"
                )
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Stage 3: Heuristics LAST (only if enabled) — keeps us from “regex hacking” the main path.
        # This is mainly useful when HF/LLM are unavailable or time out.
        if self.enable_heuristics:
            heuristic_intent = self._classify_with_heuristics(text)
            if heuristic_intent:
                logger.debug(f"🎯 Intent (heuristic fallback): {heuristic_intent} for '{text[:50]}...'")
                return IntentResult(
                    intent=heuristic_intent,
                    confidence=0.80,
                    method="heuristic"
                )
        
        # Default: treat as fact
        logger.debug(f"🎯 Intent (default): fact for '{text[:50]}...'")
        return IntentResult(intent="fact", confidence=0.5, method="default")
    
    def _classify_with_heuristics(self, text: str) -> Optional[str]:
        """
        Fast heuristic-based classification using regex patterns.
        
        Returns:
            Intent label if matched, None otherwise
        """
        text_lower = text.lower()
        
        for intent, patterns in self.HEURISTIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent
        
        return None
    
    async def _classify_with_zero_shot(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify using HuggingFace zero-shot classification.
        
        Returns:
            (intent, confidence, raw_scores)
        """
        if not self.hf_api_key:
            raise ValueError("HuggingFace API key not configured")
        
        import httpx
        
        # Prepare API request (aligned with working HFApiClassifier)
        url = f"https://router.huggingface.co/hf-inference/models/{self.hf_model}"
        
        payload = {
            "inputs": text,
            "parameters": {
                "candidate_labels": self.INTENT_LABELS,
                "multi_label": False
            }
        }
        
        # Make API call with retry logic (aligned with working HFApiClassifier)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self.hf_api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=self.hf_timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # HF router can return either:
                    # - dict: {"labels":[...], "scores":[...], ...}
                    # - list[dict]: [{"labels":[...], "scores":[...], ...}]
                    if isinstance(result, list):
                        if not result:
                            raise ValueError("HuggingFace returned empty list response")
                        result = result[0]

                    if not isinstance(result, dict):
                        raise ValueError(f"Unexpected HuggingFace response type: {type(result)}")

                    # Some error shapes:
                    # {"error": "..."} or {"errors":[...]}
                    if "error" in result:
                        raise ValueError(f"HuggingFace error: {result.get('error')}")
                    if "errors" in result:
                        raise ValueError(f"HuggingFace errors: {result.get('errors')}")

                    # Parse result - handle both response formats:
                    # Format 1 (multi-label): {"labels": [...], "scores": [...]}
                    # Format 2 (single-label): {"label": "...", "score": ...}
                    
                    if "labels" in result and "scores" in result:
                        # Multi-label format (DeBERTa with multi_label=True)
                        labels = result.get("labels", []) or []
                        scores = result.get("scores", []) or []
                        
                        if labels and scores:
                            raw_scores = dict(zip(labels, scores))
                            top_intent = labels[0]
                            top_confidence = scores[0]
                            
                            logger.debug(f"   HF zero-shot raw_scores (multi): {raw_scores}")
                            return top_intent, top_confidence, raw_scores
                        else:
                            raise ValueError(f"Empty labels or scores in multi-label format: {result}")
                    
                    elif "label" in result and "score" in result:
                        # Single-label format (BART with multi_label=False)
                        top_intent = result["label"]
                        top_confidence = result["score"]
                        raw_scores = {top_intent: top_confidence}
                        
                        logger.debug(f"   HF zero-shot result (single): {top_intent} ({top_confidence:.3f})")
                        return top_intent, top_confidence, raw_scores
                    
                    else:
                        raise ValueError(f"Unexpected HuggingFace response format: {result}")
            
            except httpx.TimeoutException:
                if attempt < max_retries:
                    delay = 2 ** attempt
                    logger.warning(f"⚠️  HuggingFace API timeout (attempt {attempt}/{max_retries})")
                    logger.warning(f"   Text: {text[:100]}...")
                    logger.warning(f"   Retrying in {delay} seconds (exponential backoff)...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ HuggingFace API timeout after {max_retries} attempts")
                    logger.error(f"   Text: {text[:100]}...")
                    raise
            
            except httpx.HTTPStatusError as e:
                # For 503 (model loading), 502 (bad gateway), and 429 (rate limit), retry with exponential backoff
                if e.response.status_code in [502, 503, 429] and attempt < max_retries:
                    delay = 2 ** attempt
                    error_type = {502: "Bad Gateway", 503: "Model loading", 429: "Rate limit"}.get(
                        e.response.status_code, f"HTTP {e.response.status_code}"
                    )
                    logger.warning(f"⚠️  {error_type} - attempt {attempt}/{max_retries}")
                    logger.warning(f"   Retrying in {delay} seconds (exponential backoff)...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ HuggingFace API HTTP error: {e.response.status_code}")
                    logger.error(f"   Response: {e.response.text[:200]}")
                    raise
            
            except Exception as e:
                logger.error(f"❌ Unexpected error during HuggingFace API call (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}", exc_info=True)
                if attempt < max_retries:
                    delay = 2 ** attempt
                    logger.warning(f"   Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
        
        raise Exception(f"Max retries exceeded for HuggingFace API (all {max_retries} attempts failed)")
    
    async def _classify_with_llm(self, text: str) -> Tuple[str, float]:
        """
        Classify using LLM (OpenAI) as fallback.
        
        Returns:
            (intent, confidence)
        """
        if not self.llm_client:
            raise ValueError("LLM client not available")
        
        prompt = f"""Classify the following sentence into ONE of these intent categories:

- fact: Objective world state (e.g., "Sarah works at Google", "I am 27 years old")
- evaluation: Value judgment (e.g., "Music is amazing", "This is terrible")
- opinion: Belief or stance (e.g., "I think X", "I agree with Y")
- speech_act: Conversational action (e.g., "Thanks, John", "Hello")
- directive: Command or intent (e.g., "I want to learn Python", "Tell Sarah to call")

Sentence: "{text}"

Respond with ONLY the intent label (one word).
"""
        
        try:
            # Support either:
            # - OpenAI Async client (self.llm_client.chat.completions.create)
            # - Our internal LLMProvider interface (self.llm_client.chat([...]))
            intent_text: Optional[str] = None

            # LLMProvider path
            if hasattr(self.llm_client, "chat") and not hasattr(getattr(self.llm_client, "chat"), "completions"):
                try:
                    from llm.providers.base import LLMMessage  # local import to avoid circular deps
                except Exception:
                    # Fallback import path if running under different module root
                    from llm_orchestration.llm.providers.base import LLMMessage  # type: ignore

                llm_resp = await self.llm_client.chat(
                    messages=[LLMMessage(role="user", content=prompt)],
                    temperature=0.0,
                    max_tokens=10,
                )
                intent_text = (llm_resp.content or "").strip().lower()
            else:
                # OpenAI SDK path
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                intent_text = response.choices[0].message.content.strip().lower()

            intent = (intent_text or "").strip().lower()
            
            # Validate intent
            if intent in self.INTENT_LABELS:
                return intent, 0.8  # Moderate confidence for LLM
            else:
                logger.warning(f"LLM returned invalid intent: {intent}, defaulting to 'fact'")
                return "fact", 0.5
        
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise

