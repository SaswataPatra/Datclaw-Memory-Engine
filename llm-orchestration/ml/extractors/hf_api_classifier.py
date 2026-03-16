"""
HuggingFace Inference API Classifier

Fast, cost-effective zero-shot classification using HuggingFace's hosted inference API.
Replaces local CPU-bound models with cloud GPU inference.

Cost: ~$0.06 per 1,000 requests
Latency: 200-500ms (vs 30-60s local)
"""

import asyncio
import httpx
from typing import List, Dict, Tuple, Optional, Set
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class HuggingFaceAPIClassifier:
    """
    Zero-shot classification using HuggingFace Inference API.
    
    Provides fast, scalable classification without local model overhead.
    Supports dynamic label discovery and multi-label classification.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        base_labels: Optional[List[str]] = None,
        low_confidence_threshold: float = 0.3,
        timeout: float = 10.0
    ):
        """
        Initialize HuggingFace API classifier.
        
        Args:
            api_key: HuggingFace API key (get from https://huggingface.co/settings/tokens)
            model_name: Model to use for inference
            base_labels: Initial set of labels (defaults to standard memory labels)
            low_confidence_threshold: Threshold for triggering label discovery
            timeout: API request timeout in seconds
        """
        if not api_key or api_key == "your_key_here":
            raise ValueError("Valid HuggingFace API key required. Get one at https://huggingface.co/settings/tokens")
        
        self.api_key = api_key
        self.model_name = model_name
        # Updated to new HuggingFace Inference API endpoint (Nov 2024)
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.low_confidence_threshold = low_confidence_threshold
        self.timeout = timeout
        
        # Base labels (same as local zero-shot)
        self.base_labels = base_labels or [
            'identity',      # Name, age, gender, pronouns
            'family',        # Family relationships
            'preference',    # Likes, dislikes, favorites
            'fact',          # Work, location, education
            'high_value',    # Sensitive information
            'goal',          # Personal goals, aspirations
            'relationship',  # Non-family relationships
            'event',         # Events, experiences
            'opinion',       # Opinions, beliefs, views
            'unknown'        # Catch-all
        ]
        
        # Dynamic labels discovered at runtime
        self.discovered_labels: Set[str] = set()
        
        logger.info(f"HuggingFace API Classifier initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Base labels ({len(self.base_labels)}): {self.base_labels}")
        logger.info(f"  Timeout: {timeout}s")
        logger.info(f"  Low confidence threshold: {low_confidence_threshold}")
    
    @property
    def current_labels(self) -> List[str]:
        """Get current set of labels (base + discovered)"""
        return self.base_labels + list(self.discovered_labels)
    
    def add_labels(self, new_labels: List[str]):
        """
        Add new labels to the classifier.
        
        Args:
            new_labels: List of new label names to add
        """
        for label in new_labels:
            if label not in self.base_labels and label not in self.discovered_labels:
                self.discovered_labels.add(label)
                logger.info(f"  ➕ Added new label: '{label}'")
    
    async def predict_single(
        self,
        text: str,
        threshold: float = 0.5,
        labels: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> Tuple[List[str], Dict[str, float], bool]:
        """
        Predict memory types for a single text using HuggingFace API with retry logic.
        
        Args:
            text: Text to classify
            threshold: Minimum score to include a label
            labels: Override labels (uses current_labels if None)
            max_retries: Maximum number of retry attempts (default: 3)
        
        Returns:
            Tuple of (predicted_labels, scores, needs_discovery)
            - predicted_labels: List of labels above threshold
            - scores: Dict mapping all labels to their scores
            - needs_discovery: True if all scores are below low_confidence_threshold
        """
        candidate_labels = labels or self.current_labels
        
        if not text or not text.strip():
            logger.warning("Empty text provided for classification")
            return [], {}, False
        
        # Retry loop
        for attempt in range(1, max_retries + 1):
            try:
                # Call HuggingFace Inference API
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "inputs": text,
                            "parameters": {
                                "candidate_labels": candidate_labels,
                                "multi_label": True
                            }
                        },
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Success! Break out of retry loop
                    break
                    
            except httpx.TimeoutException:
                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s
                    delay = 2 ** attempt
                    logger.warning(f"⚠️  HuggingFace API timeout (attempt {attempt}/{max_retries})")
                    logger.warning(f"   Text: {text[:100]}...")
                    logger.warning(f"   Retrying in {delay} seconds (exponential backoff)...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ HuggingFace API timeout after {max_retries} attempts")
                    logger.error(f"   Text: {text[:100]}...")
                    return [], {}, False
            
            except httpx.HTTPStatusError as e:
                # For 503 (model loading) and 502 (bad gateway), retry with exponential backoff
                if e.response.status_code in [502, 503] and attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s
                    delay = 2 ** attempt
                    error_type = "Bad Gateway (502)" if e.response.status_code == 502 else "Model loading (503)"
                    logger.warning(f"⚠️  {error_type} - attempt {attempt}/{max_retries}")
                    logger.warning(f"   Retrying in {delay} seconds (exponential backoff)...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ HuggingFace API HTTP error: {e.response.status_code}")
                    logger.error(f"   Response: {e.response.text[:200]}")
                    
                    # Check for common errors
                    if e.response.status_code == 401:
                        logger.error("   Invalid API key! Get one at https://huggingface.co/settings/tokens")
                    elif e.response.status_code == 429:
                        logger.error("   Rate limit exceeded! Upgrade at https://huggingface.co/pricing")
                    elif e.response.status_code in [502, 503]:
                        logger.error("   Server error. All retries exhausted.")
                    
                    return [], {}, False
            
            except Exception as e:
                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s
                    delay = 2 ** attempt
                    logger.warning(f"⚠️  HuggingFace API error (attempt {attempt}/{max_retries}): {e}")
                    logger.warning(f"   Retrying in {delay} seconds (exponential backoff)...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ HuggingFace API error after {max_retries} attempts: {e}")
                    return [], {}, False
        
        # Parse response - HF API now returns list of dicts: [{'label': 'x', 'score': 0.9}, ...]
        if isinstance(result, list) and len(result) > 0 and 'label' in result[0] and 'score' in result[0]:
            # New format: [{'label': 'opinion', 'score': 0.998}, ...]
            scores = {item['label']: item['score'] for item in result}
            
            # Filter by threshold
            predicted_labels = [
                label for label, score in scores.items()
                if score >= threshold
            ]
            
            # Check if discovery is needed (all scores below low confidence threshold)
            max_score = max(scores.values()) if scores else 0.0
            needs_discovery = max_score < self.low_confidence_threshold
            
            logger.info(f"  ✅ Classified as: {predicted_labels} (threshold={threshold})")
            if needs_discovery:
                logger.info(f"  🔍 Low confidence (max={max_score:.2f}), may need discovery")
            
            return predicted_labels, scores, needs_discovery
        
        elif isinstance(result, dict) and 'labels' in result and 'scores' in result:
            # Old format (fallback): {'labels': [...], 'scores': [...]}
            labels_list = result['labels']
            scores_list = result['scores']
            scores = dict(zip(labels_list, scores_list))
            
            predicted_labels = [
                label for label, score in scores.items()
                if score >= threshold
            ]
            
            max_score = max(scores.values()) if scores else 0.0
            needs_discovery = max_score < self.low_confidence_threshold
            
            logger.info(f"  ✅ Classified as: {predicted_labels} (threshold={threshold})")
            if needs_discovery:
                logger.info(f"  🔍 Low confidence (max={max_score:.2f}), may need discovery")
            
            return predicted_labels, scores, needs_discovery
        
        else:
            logger.error(f"❌ Unexpected API response format: {result}")
            logger.error(f"   Type: {type(result)}")
            if isinstance(result, list) and len(result) > 0:
                logger.error(f"   First item: {result[0]}")
            return [], {}, False


class HuggingFaceAPIError(Exception):
    """Exception raised for HuggingFace API errors"""
    pass

