"""
Adaptive Label Discovery for Memory Classification

Discovers new semantic categories when 'unknown' is predicted with high confidence.
Uses LLM to generate labels from conversation context, then appends to base label set.

Phase 2 Feature - Currently DISABLED by default for safe debugging.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
from difflib import SequenceMatcher
from openai import AsyncOpenAI
import json
import os

logger = logging.getLogger(__name__)


class AdaptiveLabelDiscovery:
    """
    Discovers new labels when 'unknown' is predicted with high confidence.
    Uses LLM to generate semantic labels from context.
    
    Features:
    - Progressive enhancement (start with base labels, expand as needed)
    - LLM-driven discovery (leverages GPT's semantic understanding)
    - Shared label space (all users benefit from discoveries)
    - Quality control (requires multiple examples before promoting)
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        config: Dict[str, Any],
        base_labels: List[str],
        enabled: bool = False  # 🚨 DISABLED by default
    ):
        self.llm = llm_client
        self.config = config
        self.enabled = enabled
        
        # Base labels (never removed)
        self.base_labels = base_labels
        
        # 🔄 HYPERPARAMETER - Dynamically discovered labels (Phase 2+)
        # These are appended as we discover new semantic categories
        self.discovered_labels: List[str] = []
        
        # Track discovery metadata
        self.label_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load persisted discovered labels
        self._load_discovered_labels()
        
        # 🔄 HYPERPARAMETER - Discovery thresholds (Phase 2 - will be learned)
        discovery_config = config.get('adaptive_discovery', {})
        self.unknown_threshold = discovery_config.get('unknown_threshold', 0.7)
        self.min_examples_to_add = discovery_config.get('min_examples_to_add', 3)
        self.label_confidence_threshold = discovery_config.get('label_confidence_threshold', 0.8)
        self.similarity_threshold = discovery_config.get('similarity_threshold', 0.8)
        
        logger.info(
            f"AdaptiveLabelDiscovery initialized (enabled={enabled})",
            extra={
                "base_labels_count": len(self.base_labels),
                "discovered_labels_count": len(self.discovered_labels),
                "unknown_threshold": self.unknown_threshold
            }
        )
    
    def get_current_labels(self) -> List[str]:
        """Get current label set (base + discovered)"""
        if not self.enabled:
            return self.base_labels
        return self.base_labels + self.discovered_labels
    
    async def classify_with_discovery(
        self,
        text: str,
        user_id: str,
        base_prediction: Tuple[List[str], Dict[str, float]],
        conversation_context: Optional[List[Dict]] = None
    ) -> Tuple[List[str], Dict[str, float], Optional[str]]:
        """
        Enhance classification with adaptive label discovery.
        
        Args:
            text: User message
            user_id: User ID for tracking
            base_prediction: (predicted_labels, all_scores) from base classifier
            conversation_context: Recent conversation history
        
        Returns:
            (predicted_labels, all_scores, discovered_label_candidate)
        """
        predicted_labels, scores = base_prediction
        
        # If discovery is disabled, return base prediction
        if not self.enabled:
            return predicted_labels, scores, None
        
        # Check if 'unknown' is dominant
        if 'unknown' not in scores or scores['unknown'] < self.unknown_threshold:
            return predicted_labels, scores, None
        
        logger.info(
            f"🔍 Unknown category detected (confidence={scores['unknown']:.2f}), triggering discovery...",
            extra={"text_preview": text[:100], "user_id": user_id}
        )
        
        # Discover label from context
        discovered_label = await self._discover_label_from_context(
            text=text,
            user_id=user_id,
            conversation_context=conversation_context
        )
        
        if not discovered_label:
            return predicted_labels, scores, None
        
        logger.info(f"✨ Discovered potential new label: '{discovered_label}'")
        
        # Add to candidate pool
        self._add_to_candidate_pool(discovered_label, text, user_id)
        
        # Check if we should promote to official label
        if self._should_promote_label(discovered_label):
            await self._promote_to_official_label(discovered_label)
            
            # Override 'unknown' with discovered label
            predicted_labels = [discovered_label]
            scores[discovered_label] = 0.85  # High confidence for discovered
            logger.info(f"🎉 Using newly promoted label: '{discovered_label}'")
        
        return predicted_labels, scores, discovered_label
    
    async def _discover_label_from_context(
        self,
        text: str,
        user_id: str,
        conversation_context: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Use LLM to discover a semantic label from user context.
        
        Args:
            text: The current message
            user_id: User ID for context retrieval
            conversation_context: Recent conversation history
        
        Returns:
            Discovered label (e.g., 'health', 'cooking', 'travel') or None
        """
        # Build context for LLM
        context_messages = []
        if conversation_context:
            context_messages = [
                f"User: {msg.get('user', msg.get('content', ''))}\nAssistant: {msg.get('assistant', msg.get('response', ''))}"
                for msg in conversation_context[-5:]  # Last 5 exchanges
            ]
        
        prompt = f"""You are analyzing a user's message to discover its semantic category.

Current message: "{text}"

Recent conversation context:
{chr(10).join(context_messages) if context_messages else "No prior context"}

Existing categories we already have:
{', '.join(self.base_labels)}

Discovered categories so far:
{', '.join(self.discovered_labels) if self.discovered_labels else "None yet"}

Task: If this message represents a NEW semantic category not covered by existing labels, suggest a single-word or two-word label for it.

Requirements:
1. Label must be a general semantic category (not too specific)
2. Label should be reusable for future similar messages
3. Label must be different from existing categories
4. Use lowercase, underscore-separated (e.g., 'health_wellness', 'cooking_recipes', 'travel_plans')

If this message fits an existing category, return "EXISTING".
If you're unsure, return "UNCLEAR".

Return ONLY the label or EXISTING or UNCLEAR, nothing else.

Label:"""

        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a semantic categorization expert. Be concise and precise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=20
            )
            
            discovered_label = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if discovered_label in ["existing", "unclear", ""]:
                logger.debug(f"LLM returned: {discovered_label} (no new label)")
                return None
            
            # Clean label (remove quotes, extra spaces)
            discovered_label = discovered_label.strip('"\'').replace(' ', '_')
            
            # Validate format (alphanumeric + underscore only)
            if not discovered_label.replace('_', '').isalnum():
                logger.warning(f"Invalid label format: {discovered_label}")
                return None
            
            logger.info(f"LLM suggested label: '{discovered_label}'")
            return discovered_label
            
        except Exception as e:
            logger.error(f"Error discovering label: {e}", exc_info=True)
            return None
    
    def _add_to_candidate_pool(self, label: str, example_text: str, user_id: str):
        """Add discovered label to candidate pool"""
        if label not in self.label_metadata:
            self.label_metadata[label] = {
                'discovered_at': datetime.utcnow().isoformat(),
                'example_texts': [],
                'user_ids': set(),
                'frequency': 0
            }
        
        self.label_metadata[label]['example_texts'].append(example_text)
        self.label_metadata[label]['user_ids'].add(user_id)
        self.label_metadata[label]['frequency'] += 1
        
        logger.info(
            f"Added to candidate pool: '{label}'",
            extra={
                "frequency": self.label_metadata[label]['frequency'],
                "unique_users": len(self.label_metadata[label]['user_ids'])
            }
        )
    
    def _should_promote_label(self, label: str) -> bool:
        """
        Decide if a candidate label should be promoted to official label set.
        
        Criteria:
        1. Seen at least N times (min_examples_to_add)
        2. Used by at least 1 user (can increase for multi-user deployments)
        3. Not too similar to existing labels
        """
        if label not in self.label_metadata:
            return False
        
        metadata = self.label_metadata[label]
        
        # Check frequency
        if metadata['frequency'] < self.min_examples_to_add:
            logger.debug(
                f"Label '{label}' needs more examples",
                extra={
                    "current": metadata['frequency'],
                    "required": self.min_examples_to_add
                }
            )
            return False
        
        # Check user diversity (at least 1 user for single-user testing)
        if len(metadata['user_ids']) < 1:
            return False
        
        # Check similarity to existing labels (prevent duplicates)
        if self._is_too_similar_to_existing(label):
            logger.warning(f"Label '{label}' too similar to existing labels")
            return False
        
        return True
    
    def _is_too_similar_to_existing(self, new_label: str) -> bool:
        """Check if new label is too similar to existing ones"""
        all_existing = self.base_labels + self.discovered_labels
        
        for existing in all_existing:
            similarity = SequenceMatcher(None, new_label, existing).ratio()
            if similarity > self.similarity_threshold:
                logger.debug(f"'{new_label}' is {similarity:.2f} similar to '{existing}'")
                return True
        
        return False
    
    async def _promote_to_official_label(self, label: str):
        """
        Promote a candidate label to official discovered labels.
        This triggers a classifier retraining (future work).
        """
        if label in self.discovered_labels:
            return
        
        logger.info(f"🎉 Promoting '{label}' to official label set!")
        
        # Add to discovered labels
        self.discovered_labels.append(label)
        
        # Save to persistent storage
        await self._save_discovered_labels()
        
        # TODO: Trigger classifier retraining with new label
        # This will be implemented in Phase 2 when we have incremental fine-tuning
        logger.info(f"TODO: Retrain classifier with new label '{label}'")
    
    async def _generate_synthetic_examples(
        self,
        label: str,
        seed_examples: List[str],
        num_variations: int = 10
    ) -> List[str]:
        """
        Generate synthetic examples for new label (for future retraining).
        Currently not used - will be needed for incremental fine-tuning.
        """
        prompt = f"""Generate {num_variations} diverse variations of user messages that belong to the category '{label}'.

Seed examples:
{chr(10).join(f"- {ex}" for ex in seed_examples[:3])}

Generate variations that:
1. Maintain the semantic category '{label}'
2. Vary in phrasing, detail level, and tone
3. Are realistic user messages

Return one variation per line, no numbering:
"""
        
        try:
            response = await self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate realistic user messages."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            variations = [
                line.strip().lstrip('0123456789.-) ')
                for line in response.choices[0].message.content.split('\n')
                if line.strip()
            ]
            
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Error generating synthetic examples: {e}")
            return []
    
    async def _save_discovered_labels(self):
        """Persist discovered labels to disk"""
        save_dir = "models/distilbert"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "discovered_labels.json")
        
        data = {
            'discovered_labels': self.discovered_labels,
            'label_metadata': {
                label: {
                    **meta,
                    'user_ids': list(meta['user_ids'])  # Convert set to list for JSON
                }
                for label, meta in self.label_metadata.items()
            },
            'last_updated': datetime.utcnow().isoformat()
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.discovered_labels)} discovered labels to {save_path}")
        except Exception as e:
            logger.error(f"Error saving discovered labels: {e}")
    
    def _load_discovered_labels(self):
        """Load persisted discovered labels from disk"""
        save_path = "models/distilbert/discovered_labels.json"
        
        if not os.path.exists(save_path):
            logger.debug("No persisted discovered labels found")
            return
        
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            self.discovered_labels = data.get('discovered_labels', [])
            
            # Restore metadata with set conversion
            loaded_metadata = data.get('label_metadata', {})
            for label, meta in loaded_metadata.items():
                meta['user_ids'] = set(meta.get('user_ids', []))
                self.label_metadata[label] = meta
            
            logger.info(
                f"Loaded {len(self.discovered_labels)} discovered labels from {save_path}",
                extra={"labels": self.discovered_labels}
            )
        except Exception as e:
            logger.error(f"Error loading discovered labels: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about label discovery"""
        return {
            'enabled': self.enabled,
            'base_labels_count': len(self.base_labels),
            'discovered_labels_count': len(self.discovered_labels),
            'discovered_labels': self.discovered_labels,
            'candidate_labels_count': len(self.label_metadata),
            'candidate_labels': list(self.label_metadata.keys()),
            'total_discoveries': sum(meta['frequency'] for meta in self.label_metadata.values()),
            'config': {
                'unknown_threshold': self.unknown_threshold,
                'min_examples_to_add': self.min_examples_to_add,
                'similarity_threshold': self.similarity_threshold
            }
        }

