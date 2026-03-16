"""
DistilBERT-based Memory Type Classifier

Replaces regex patterns with a fine-tuned transformer model for context-aware
memory type classification.

Memory Types:
- identity: Name, pronouns, self-description
- family: Family relationships and information
- preference: Likes, dislikes, favorites
- fact: Work, location, education, biographical info
- high_value: High-stakes work/financial information
- goal: Personal goals and aspirations
- relationship: Non-family relationships
- event: Events and experiences
- opinion: Opinions and views
- unknown: Unclassified
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DistilBERTMemoryClassifier(nn.Module):
    """
    Fine-tuned DistilBERT for multi-label memory type classification.
    
    Advantages over regex:
    - Understands context and semantics
    - Handles edge cases (sarcasm, negation, ambiguity)
    - Multi-label (a message can be both 'preference' and 'family')
    - Confidence scores for each label
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 10,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        super().__init__()
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load pre-trained DistilBERT
        logger.info(f"Loading DistilBERT base model '{model_name}' (this may take a minute on first run)...")
        self.config = DistilBertConfig.from_pretrained(model_name)
        logger.info("  ✓ Config loaded")
        self.distilbert = DistilBertModel.from_pretrained(model_name, config=self.config)
        logger.info("  ✓ Model loaded")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        logger.info("  ✓ Tokenizer loaded")
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Label mapping
        self.label_names = [
            'identity',
            'family',
            'preference',
            'fact',
            'high_value',
            'goal',
            'relationship',
            'event',
            'opinion',
            'unknown'
        ]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"DistilBERTMemoryClassifier initialized on {self.device}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            logits: Raw logits for each label [batch_size, num_labels]
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(
        self,
        texts: List[str],
        threshold: float = 0.5,
        max_length: int = 128
    ) -> List[Dict[str, float]]:
        """
        Predict memory types for a list of texts.
        
        Args:
            texts: List of user messages
            threshold: Confidence threshold for positive prediction (0.0-1.0)
            max_length: Maximum sequence length for tokenization
        
        Returns:
            List of dicts with label -> confidence mappings
            Example: [{'preference': 0.92, 'family': 0.15, ...}, ...]
        """
        self.eval()
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)  # Multi-label classification
        
        # Convert to list of dicts
        results = []
        for probs in probabilities.cpu().numpy():
            result = {
                label: float(prob)
                for label, prob in zip(self.label_names, probs)
            }
            results.append(result)
        
        return results
    
    def predict_single(
        self,
        text: str,
        threshold: float = 0.5,
        max_length: int = 128
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Predict memory types for a single text.
        
        Args:
            text: User message
            threshold: Confidence threshold for positive prediction
            max_length: Maximum sequence length
        
        Returns:
            Tuple of (predicted_labels, all_scores)
            Example: (['preference', 'family'], {'preference': 0.92, 'family': 0.78, ...})
        """
        logger.debug(f"🔍 DistilBERT predicting for text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        
        results = self.predict([text], threshold, max_length)
        scores = results[0]
        
        # Log top 3 predictions
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.debug(f"  📊 Top 3 predictions: {[(label, f'{score:.3f}') for label, score in top_3]}")
        
        # Filter by threshold
        predicted_labels = [
            label for label, score in scores.items()
            if score >= threshold
        ]
        
        # If no labels above threshold, return 'unknown'
        if not predicted_labels:
            logger.debug(f"  ⚠️  No labels above threshold {threshold}, defaulting to 'unknown'")
            predicted_labels = ['unknown']
        else:
            logger.info(f"  ✅ DistilBERT classified as: {predicted_labels} (threshold={threshold})")
        
        return predicted_labels, scores
    
    def save_model(self, path: str):
        """Save model weights and tokenizer"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'label_names': self.label_names,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        # Use weights_only=False for now since checkpoint contains config dict
        # TODO: In production, separate model weights from metadata for security
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def get_label_distribution(self, text: str) -> Dict[str, float]:
        """
        Get probability distribution over all labels for a single text.
        Useful for debugging and understanding model predictions.
        
        Args:
            text: User message
        
        Returns:
            Dict mapping label -> probability
        """
        _, scores = self.predict_single(text, threshold=0.0)
        return scores


def create_training_batch(
    texts: List[str],
    labels: List[List[str]],
    tokenizer: DistilBertTokenizer,
    label_to_idx: Dict[str, int],
    max_length: int = 128,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a training batch from texts and labels.
    
    Args:
        texts: List of user messages
        labels: List of label lists (multi-label)
        tokenizer: DistilBERT tokenizer
        label_to_idx: Mapping from label name to index
        max_length: Maximum sequence length
        device: Device to move tensors to
    
    Returns:
        Tuple of (input_ids, attention_mask, label_tensor)
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Create multi-hot label tensor
    num_labels = len(label_to_idx)
    label_tensor = torch.zeros(len(texts), num_labels, dtype=torch.float32)
    
    for i, text_labels in enumerate(labels):
        for label in text_labels:
            if label in label_to_idx:
                label_tensor[i, label_to_idx[label]] = 1.0
    
    label_tensor = label_tensor.to(device)
    
    return input_ids, attention_mask, label_tensor

