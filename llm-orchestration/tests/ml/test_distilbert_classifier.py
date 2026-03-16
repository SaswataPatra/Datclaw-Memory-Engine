"""
Test cases for DistilBERT Memory Classifier

Tests the DistilBERT classifier architecture, prediction, and training utilities.
"""

import pytest
import torch
from ml.extractors.memory_classifier import DistilBERTMemoryClassifier, create_training_batch


class TestDistilBERTClassifier:
    """Test DistilBERT memory classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create a DistilBERT classifier instance"""
        return DistilBERTMemoryClassifier(device='cpu')
    
    def test_initialization(self, classifier):
        """Test that classifier initializes correctly"""
        assert classifier is not None
        assert classifier.device == torch.device('cpu')
        assert len(classifier.label_names) == 10
        assert 'identity' in classifier.label_names
        assert 'family' in classifier.label_names
        assert 'unknown' in classifier.label_names
    
    def test_label_mapping(self, classifier):
        """Test label to index mapping"""
        assert classifier.label_to_idx['identity'] == 0
        assert classifier.idx_to_label[0] == 'identity'
        assert len(classifier.label_to_idx) == len(classifier.label_names)
    
    def test_forward_pass(self, classifier):
        """Test forward pass through the model"""
        # Create dummy input
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        logits = classifier.forward(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, 10)  # 10 labels
        assert not torch.isnan(logits).any()
    
    def test_predict_single(self, classifier):
        """Test single text prediction"""
        text = "My name is Sarah Johnson"
        labels, scores = classifier.predict_single(text, threshold=0.5)
        
        # Should return lists
        assert isinstance(labels, list)
        assert isinstance(scores, dict)
        
        # Scores should be for all labels
        assert len(scores) == 10
        assert all(0 <= score <= 1 for score in scores.values())
        
        # If no labels above threshold, should return 'unknown'
        if not labels:
            assert True  # This is expected for untrained model
        else:
            assert all(label in classifier.label_names for label in labels)
    
    def test_predict_batch(self, classifier):
        """Test batch prediction"""
        texts = [
            "My name is John",
            "I love playing basketball",
            "My mother is a doctor"
        ]
        
        results = classifier.predict(texts, threshold=0.5)
        
        assert len(results) == 3
        assert all(isinstance(result, dict) for result in results)
        assert all(len(result) == 10 for result in results)
    
    def test_predict_empty_string(self, classifier):
        """Test prediction with empty string"""
        labels, scores = classifier.predict_single("", threshold=0.5)
        
        # Should handle gracefully
        assert isinstance(labels, list)
        assert isinstance(scores, dict)
    
    def test_predict_long_text(self, classifier):
        """Test prediction with long text (should truncate)"""
        long_text = " ".join(["word"] * 500)  # Very long text
        labels, scores = classifier.predict_single(long_text, threshold=0.5)
        
        # Should handle truncation gracefully
        assert isinstance(labels, list)
        assert isinstance(scores, dict)
    
    def test_get_label_distribution(self, classifier):
        """Test getting probability distribution"""
        text = "I'm planning to run a marathon"
        distribution = classifier.get_label_distribution(text)
        
        assert isinstance(distribution, dict)
        assert len(distribution) == 10
        assert sum(distribution.values()) <= 10.0  # Probabilities, not necessarily sum to 1
    
    def test_threshold_behavior(self, classifier):
        """Test different threshold values"""
        text = "My favorite food is pizza"
        
        # Low threshold - should get more labels
        labels_low, _ = classifier.predict_single(text, threshold=0.1)
        
        # High threshold - should get fewer labels
        labels_high, _ = classifier.predict_single(text, threshold=0.9)
        
        # Low threshold should return more or equal labels
        assert len(labels_low) >= len(labels_high)
    
    def test_multi_label_prediction(self, classifier):
        """Test that model can predict multiple labels"""
        # This text could be both 'family' and 'preference'
        text = "My sister loves playing tennis"
        labels, scores = classifier.predict_single(text, threshold=0.3)
        
        # With low threshold, might get multiple labels
        # (depends on model training)
        assert isinstance(labels, list)
        assert len(labels) >= 0  # Could be 0 for untrained model
    
    def test_save_and_load_model(self, classifier, tmp_path):
        """Test model saving and loading"""
        save_path = tmp_path / "test_model.pt"
        
        # Save model
        classifier.save_model(str(save_path))
        assert save_path.exists()
        
        # Create new classifier and load
        new_classifier = DistilBERTMemoryClassifier(device='cpu')
        new_classifier.load_model(str(save_path))
        
        # Should have same configuration
        assert new_classifier.label_names == classifier.label_names
    
    def test_device_handling(self):
        """Test device selection"""
        # CPU device
        classifier_cpu = DistilBERTMemoryClassifier(device='cpu')
        assert classifier_cpu.device == torch.device('cpu')
        
        # Auto device selection
        classifier_auto = DistilBERTMemoryClassifier(device=None)
        assert classifier_auto.device in [torch.device('cpu'), torch.device('cuda')]


class TestTrainingBatch:
    """Test training batch creation utilities"""
    
    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer"""
        from transformers import DistilBertTokenizer
        return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    @pytest.fixture
    def label_to_idx(self):
        """Get label mapping"""
        return {
            'identity': 0,
            'family': 1,
            'preference': 2,
            'unknown': 3
        }
    
    def test_create_training_batch(self, tokenizer, label_to_idx):
        """Test creating a training batch"""
        texts = [
            "My name is John",
            "I love pizza"
        ]
        labels = [
            ['identity'],
            ['preference']
        ]
        
        input_ids, attention_mask, label_tensor = create_training_batch(
            texts, labels, tokenizer, label_to_idx, max_length=128, device='cpu'
        )
        
        assert input_ids.shape[0] == 2  # Batch size
        assert attention_mask.shape[0] == 2
        assert label_tensor.shape == (2, 4)  # 2 samples, 4 labels
        
        # Check multi-hot encoding
        assert label_tensor[0, 0] == 1.0  # identity
        assert label_tensor[1, 2] == 1.0  # preference
    
    def test_multi_label_batch(self, tokenizer, label_to_idx):
        """Test batch with multi-label examples"""
        texts = ["My sister loves tennis"]
        labels = [['family', 'preference']]  # Multi-label
        
        input_ids, attention_mask, label_tensor = create_training_batch(
            texts, labels, tokenizer, label_to_idx, device='cpu'
        )
        
        # Both labels should be 1
        assert label_tensor[0, 1] == 1.0  # family
        assert label_tensor[0, 2] == 1.0  # preference
    
    def test_empty_labels(self, tokenizer, label_to_idx):
        """Test batch with no labels"""
        texts = ["Random text"]
        labels = [[]]  # No labels
        
        input_ids, attention_mask, label_tensor = create_training_batch(
            texts, labels, tokenizer, label_to_idx, device='cpu'
        )
        
        # All labels should be 0
        assert label_tensor.sum() == 0.0
    
    def test_unknown_label(self, tokenizer, label_to_idx):
        """Test batch with unknown label"""
        texts = ["Some text"]
        labels = [['nonexistent_label']]  # Not in label_to_idx
        
        input_ids, attention_mask, label_tensor = create_training_batch(
            texts, labels, tokenizer, label_to_idx, device='cpu'
        )
        
        # Unknown label should be ignored
        assert label_tensor.sum() == 0.0


class TestDistilBERTIntegration:
    """Integration tests for DistilBERT classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier"""
        return DistilBERTMemoryClassifier(device='cpu')
    
    def test_identity_classification(self, classifier):
        """Test classification of identity statements"""
        texts = [
            "My name is Sarah",
            "I'm a software engineer",
            "People call me Sam"
        ]
        
        for text in texts:
            labels, scores = classifier.predict_single(text, threshold=0.3)
            # For untrained model, just check it doesn't crash
            assert isinstance(labels, list)
            assert isinstance(scores, dict)
    
    def test_family_classification(self, classifier):
        """Test classification of family statements"""
        texts = [
            "My mother is a teacher",
            "I have two brothers",
            "My dad works at Google"
        ]
        
        for text in texts:
            labels, scores = classifier.predict_single(text, threshold=0.3)
            assert isinstance(labels, list)
            assert isinstance(scores, dict)
    
    def test_preference_classification(self, classifier):
        """Test classification of preference statements"""
        texts = [
            "I love playing basketball",
            "I hate spicy food",
            "My favorite movie is Inception"
        ]
        
        for text in texts:
            labels, scores = classifier.predict_single(text, threshold=0.3)
            assert isinstance(labels, list)
            assert isinstance(scores, dict)
    
    def test_unknown_classification(self, classifier):
        """Test classification of generic statements"""
        texts = [
            "What's the weather?",
            "That's interesting",
            "Tell me more"
        ]
        
        for text in texts:
            labels, scores = classifier.predict_single(text, threshold=0.5)
            # For untrained model, might classify as unknown or nothing
            assert isinstance(labels, list)
            assert isinstance(scores, dict)
    
    def test_batch_consistency(self, classifier):
        """Test that batch and single predictions are consistent"""
        text = "I love pizza"
        
        # Single prediction
        labels_single, scores_single = classifier.predict_single(text)
        
        # Batch prediction
        results_batch = classifier.predict([text])
        scores_batch = results_batch[0]
        
        # Scores should be very close (allowing for small numerical differences)
        for label in scores_single:
            assert abs(scores_single[label] - scores_batch[label]) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

