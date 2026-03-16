"""
Zero-Shot Memory Classifier with Dynamic Label Discovery

Uses facebook/bart-large-mnli for zero-shot classification with evolving labels.
No training required - works out of the box!

Advantages over DistilBERT:
- No training needed
- Faster on CPU (~2-5s vs 60s)
- Dynamic labels (can add new categories at runtime)
- High accuracy on diverse inputs
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class ZeroShotMemoryClassifier:
    """
    Zero-shot classifier that can classify text into ANY labels without training.
    
    Features:
    - Uses pre-trained BART model (facebook/bart-large-mnli)
    - Supports dynamic label discovery via LLM
    - Multi-label classification
    - Confidence-based label suggestions
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        environment: str = "local",
        base_labels: Optional[List[str]] = None,
        device: Optional[str] = None,
        low_confidence_threshold: float = 0.3
    ):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: HuggingFace model name (overrides environment default)
            environment: "local" or "production" (selects optimal model)
            base_labels: Initial set of labels (can be extended dynamically)
            device: Device to run on (-1 for CPU, 0 for GPU)
            low_confidence_threshold: Threshold for triggering label discovery
        """
        # Select model based on environment if not explicitly provided
        if model_name is None:
            if environment == "production":
                model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
                logger.info("🏭 Production environment: Using DeBERTa-v3-large (98% accuracy)")
            else:  # local/development
                model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
                logger.info("💻 Local environment: Using mDeBERTa-v3-base (92-95% accuracy)")
        else:
            logger.info(f"📦 Using custom model: {model_name}")
        # Device setup
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"Loading zero-shot classifier: {model_name}")
        logger.info(f"  Device: {'GPU' if device >= 0 else 'CPU'}")
        
        # Load pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        
        logger.info(f"✅ Zero-shot classifier loaded successfully")
        
        # Base labels (can be extended)
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
        
        # Confidence threshold for label discovery
        self.low_confidence_threshold = low_confidence_threshold
        
        logger.info(f"  Base labels ({len(self.base_labels)}): {self.base_labels}")
        logger.info(f"  Low confidence threshold: {self.low_confidence_threshold}")
    
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
    
    def predict_single(
        self,
        text: str,
        threshold: float = 0.5,
        labels: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict[str, float], bool]:
        """
        Predict memory types for a single text.
        
        Args:
            text: User message
            threshold: Confidence threshold for positive prediction
            labels: Optional custom labels (uses current_labels if None)
        
        Returns:
            Tuple of (predicted_labels, all_scores, needs_discovery)
            - predicted_labels: List of labels above threshold
            - all_scores: Dict mapping label -> confidence score
            - needs_discovery: True if max confidence < low_confidence_threshold
        """
        logger.debug(f"🔍 Zero-shot predicting for: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        
        # Use provided labels or current labels
        candidate_labels = labels or self.current_labels
        
        # Run zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=candidate_labels,
            multi_label=True  # Allow multiple labels
        )
        
        # Convert to dict
        scores = {
            label: score 
            for label, score in zip(result['labels'], result['scores'])
        }
        
        # Log top 3 predictions
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.debug(f"  📊 Top 3 predictions: {[(label, f'{score:.3f}') for label, score in top_3]}")
        
        # Filter by threshold
        predicted_labels = [
            label for label, score in scores.items()
            if score >= threshold
        ]
        
        # Check if we need label discovery
        max_score = max(scores.values()) if scores else 0.0
        needs_discovery = max_score < self.low_confidence_threshold
        
        if needs_discovery:
            logger.warning(
                f"  ⚠️  Low confidence (max={max_score:.3f} < {self.low_confidence_threshold}). "
                f"Label discovery recommended."
            )
        
        # If no labels above threshold, return 'unknown'
        if not predicted_labels:
            logger.debug(f"  ⚠️  No labels above threshold {threshold}, defaulting to 'unknown'")
            predicted_labels = ['unknown']
        else:
            logger.info(f"  ✅ Classified as: {predicted_labels} (threshold={threshold})")
        
        return predicted_labels, scores, needs_discovery
    
    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        labels: Optional[List[str]] = None
    ) -> List[Tuple[List[str], Dict[str, float], bool]]:
        """
        Predict memory types for multiple texts.
        
        Args:
            texts: List of user messages
            threshold: Confidence threshold
            labels: Optional custom labels
        
        Returns:
            List of (predicted_labels, scores, needs_discovery) tuples
        """
        return [
            self.predict_single(text, threshold, labels)
            for text in texts
        ]
    
    def get_label_distribution(self, text: str) -> Dict[str, float]:
        """
        Get probability distribution over all labels.
        
        Args:
            text: User message
        
        Returns:
            Dict mapping label -> probability
        """
        _, scores, _ = self.predict_single(text, threshold=0.0)
        return scores


class DynamicLabelDiscovery:
    """
    Discovers new memory labels using LLM when existing labels don't fit well.
    """
    
    def __init__(
        self,
        llm_client,
        config: Dict,
        enabled: bool = True,
        label_store: Optional['LabelStore'] = None
    ):
        """
        Initialize label discovery.
        
        Args:
            llm_client: OpenAI client for label generation
            config: Configuration dict
            enabled: Whether discovery is enabled
            label_store: Optional LabelStore to access previously discovered labels
        """
        self.llm_client = llm_client
        self.config = config
        self.enabled = enabled
        self.label_store = label_store #TO-DO later use sqlite3 database for persistent storage
        
        logger.info(f"DynamicLabelDiscovery initialized (enabled={enabled}, label_store={'provided' if label_store else 'not provided'})")
    
    async def discover_labels(
        self,
        text: str,
        existing_labels: List[str],
        current_scores: Dict[str, float]
    ) -> List[Dict[str, float]]:
        """
        Ask OpenAI to suggest new labels with importance scores for text that doesn't fit existing categories.
        
        Uses GPT-4o-mini for fast, accurate label generation.
        
        Args:
            text: User message that triggered discovery
            existing_labels: Current label set
            current_scores: Confidence scores for existing labels
        
        Returns:
            List of dicts with 'name' and 'importance' keys
            Example: [{'name': 'health_info', 'importance': 0.95}, ...]
        """
        if not self.enabled:
            logger.debug("Label discovery disabled, skipping")
            return []
        
        if not self.llm_client:
            logger.warning("LLM client not configured, skipping label discovery")
            return []
        
        logger.info(f"🔍 Discovering new labels for: '{text[:80]}...'")
        
        # Format current scores for context
        top_scores = sorted(
            current_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        scores_str = ", ".join([f"{label}: {score:.2f}" for label, score in top_scores])
        
        # Get previously discovered labels from LabelStore
        discovered_labels = []
        if self.label_store:
            discovered_labels = self.label_store.get_all_labels()
            logger.debug(f"  📚 Found {len(discovered_labels)} previously discovered labels")
        
        # Combine all labels (base + discovered)
        all_existing_labels = sorted(set(existing_labels + discovered_labels))
        
        # Format discovered labels with their importance for context
        discovered_labels_info = ""
        if discovered_labels and self.label_store:
            labels_with_importance = self.label_store.get_labels_with_importance()
            discovered_labels_formatted = [
                f"{label} (importance: {labels_with_importance.get(label, 0.7):.2f})"
                for label in sorted(discovered_labels)
            ]
            discovered_labels_info = f"\n\nPREVIOUSLY DISCOVERED labels ({len(discovered_labels)} total):\n{', '.join(discovered_labels_formatted)}"
        
        prompt = f"""Analyze this user message and determine if it fits existing categories OR needs new ones.

Message: "{text}"

BASE memory categories ({len(existing_labels)} total):
{', '.join(sorted(existing_labels))}{discovered_labels_info}

TOTAL categories available: {len(all_existing_labels)}

Current classification scores: {scores_str}

CRITICAL INSTRUCTIONS:
1. **FIRST**, check if the message fits ANY existing category (even if it's not in the top scores)
2. **ONLY** suggest NEW categories if NONE of the existing ones are appropriate
3. Prefer using existing categories over creating new ones to avoid redundancy

Examples of when NOT to create new labels:
- "my cat's name is Shero" → USE: pet_info, pet_affection, animal_relationship (DON'T create: pet_identity, cat_name)
- "I love playing with my cat" → USE: pet_affection, companion_relationship (DON'T create: pet_playtime, cat_interaction)
- "my mom's name is Sarah" → USE: family, identity (DON'T create: parent_info, family_member_name)

If you MUST suggest NEW categories (1-3 max), they should:
- Be generic enough to apply to similar messages
- Be specific enough to be meaningful
- Use lowercase, underscore-separated format (e.g., "health_info", "travel_plans")
- Be CLEARLY different from ALL existing categories

For each label, rate its importance (0.0 to 1.0) based on:
- How critical is this to the user's identity/well-being?
- Reference scores: identity=1.0, family=1.0, health=0.95, preference=0.9, work=0.7, opinion=0.5

Return a JSON object with a "labels" array. If existing categories are sufficient, return an empty array.
Example: {{"labels": [
  {{"name": "health_info", "importance": 0.95}},
  {{"name": "travel_plans", "importance": 0.7}}
]}}

Or if existing categories fit: {{"labels": []}}
"""
        
        try:
            # Call OpenAI GPT-4o-mini for label generation
            response = await self.llm_client.chat.completions.create(
                model=self.config.get('llm', {}).get('model', 'gpt-4o-mini'),
                messages=[{
                    "role": "system",
                    "content": "You are a memory categorization expert. Suggest concise, meaningful, and unique labels for memory classification."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=100,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            logger.debug(f"  LLM response: {content}")
            
            # Extract JSON (response_format ensures it's valid JSON)
            response_json = json.loads(content)
            
            # Handle different response formats (be flexible!)
            suggested_labels = []
            
            if isinstance(response_json, list):
                # Direct array: ["label1", "label2"]
                suggested_labels = response_json
            elif isinstance(response_json, dict):
                # Try common keys
                for key in ['labels', 'suggested_labels', 'new_labels', 'categories']:
                    if key in response_json:
                        suggested_labels = response_json[key]
                        break
                
                # If still empty, extract all values from dict
                if not suggested_labels:
                    # Flatten nested structures
                    for value in response_json.values():
                        if isinstance(value, list):
                            suggested_labels.extend(value)
                        elif isinstance(value, str):
                            suggested_labels.append(value)
                    
                    if suggested_labels:
                        logger.debug(f"  📦 Extracted labels from nested structure: {suggested_labels}")
            
            if not suggested_labels:
                # Empty labels array means LLM determined existing categories are sufficient
                if isinstance(response_json, dict) and 'labels' in response_json and response_json['labels'] == []:
                    logger.info(f"  ✅ LLM confirmed existing categories are sufficient (no new labels needed)")
                else:
                    logger.warning(f"  ⚠️  Could not extract labels from response: {response_json}")
                return []
            
            if not isinstance(suggested_labels, list):
                # Convert single string to list
                if isinstance(suggested_labels, str):
                    suggested_labels = [suggested_labels]
                else:
                    logger.warning(f"  ⚠️  Labels is not a list or string: {suggested_labels}")
                    return []
            
            # Validate and format labels
            valid_labels = []
            for item in suggested_labels:
                if isinstance(item, dict) and 'name' in item:
                    # New format: {"name": "label", "importance": 0.95}
                    label_name = item['name'].lower().strip()
                    importance = item.get('importance', 0.7)  # Default to 0.7 if missing
                    if len(label_name) > 0:
                        valid_labels.append({'name': label_name, 'importance': float(importance)})
                elif isinstance(item, str):
                    # Legacy format: "label" (assign default importance)
                    label_name = item.lower().strip()
                    if len(label_name) > 0:
                        valid_labels.append({'name': label_name, 'importance': 0.7})
            
            label_names = [l['name'] for l in valid_labels]
            logger.info(f"  ✅ Discovered {len(valid_labels)} new labels: {label_names}")
            return valid_labels
            
        except json.JSONDecodeError as e:
            logger.error(f"  ❌ Failed to parse LLM response as JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"  ❌ Label discovery failed: {e}", exc_info=True)
            return []


class LabelStore:
    """
    Persistent storage for discovered labels.
    """
    
    def __init__(self, storage_path: str = "data/discovered_labels.json"):
        """
        Initialize label store.
        
        Args:
            storage_path: Path to JSON file for storing labels
        """
        self.storage_path = storage_path
        self.labels: Dict[str, Dict] = {}  # label -> metadata
        
        self._load()
    
    def _load(self):
        """Load labels from disk"""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.labels = json.load(f)
                logger.info(f"Loaded {len(self.labels)} discovered labels from {self.storage_path}")
        except Exception as e:
            logger.warning(f"Could not load labels from {self.storage_path}: {e}")
            self.labels = {}
    
    def _save(self):
        """Save labels to disk"""
        try:
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.labels, f, indent=2)
            logger.debug(f"Saved {len(self.labels)} labels to {self.storage_path}")
        except Exception as e:
            logger.error(f"Could not save labels to {self.storage_path}: {e}")
    
    def add_label(self, label: str, context: str = "", user_id: str = "", importance: float = 0.7):
        """
        Add a discovered label.
        
        Args:
            label: Label name
            context: Example text that triggered discovery
            user_id: User who triggered discovery
            importance: Importance score (0.0-1.0) for ego scoring
        """
        if label not in self.labels:
            self.labels[label] = {
                "discovered_at": datetime.utcnow().isoformat(),
                "usage_count": 0,
                "context": context[:200],  # Store first 200 chars
                "user_id": user_id,
                "importance": importance
            }
            self._save()
            logger.info(f"  💾 Stored new label: '{label}' (importance={importance:.2f})")
    
    def increment_usage(self, label: str):
        """Increment usage count for a label"""
        if label in self.labels:
            self.labels[label]["usage_count"] += 1
            self._save()
    
    def get_all_labels(self) -> List[str]:
        """Get all discovered labels"""
        return list(self.labels.keys())
    
    def get_labels_with_importance(self) -> Dict[str, float]:
        """
        Get all discovered labels with their importance scores.
        
        Returns:
            Dict mapping label names to importance scores
        """
        return {
            label: metadata.get('importance', 0.7)
            for label, metadata in self.labels.items()
        }
    
    def get_label_metadata(self, label: str) -> Optional[Dict]:
        """Get metadata for a label"""
        return self.labels.get(label)

