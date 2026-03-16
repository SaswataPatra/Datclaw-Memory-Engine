"""
Tests for text splitting utilities
"""

import pytest
from ml.utils.text_splitter import (
    SentenceSplitter,
    MemoryTextSplitter,
    split_sentences,
    split_for_memory
)


class TestSentenceSplitter:
    """Tests for basic sentence splitter"""
    
    def test_single_sentence(self):
        """Test splitting a single sentence"""
        splitter = SentenceSplitter()
        text = "My name is John."
        sentences = splitter.split(text)
        
        assert len(sentences) == 1
        assert sentences[0] == "My name is John."
    
    def test_multiple_sentences(self):
        """Test splitting multiple sentences"""
        splitter = SentenceSplitter()
        text = "My name is John. I love pizza. I work at Google."
        sentences = splitter.split(text)
        
        assert len(sentences) == 3
        assert "My name is John." in sentences
        assert "I love pizza." in sentences
        assert "I work at Google." in sentences
    
    def test_abbreviations(self):
        """Test that abbreviations don't cause incorrect splits"""
        splitter = SentenceSplitter()
        text = "Dr. Smith works at U.S.A. Inc. in New York."
        sentences = splitter.split(text)
        
        # Should be treated as one sentence
        assert len(sentences) == 1
        assert sentences[0] == "Dr. Smith works at U.S.A. Inc. in New York."
    
    def test_min_words_filter(self):
        """Test filtering of short sentences"""
        splitter = SentenceSplitter(min_words=3)
        text = "Hi! My name is John. Bye."
        sentences = splitter.split(text)
        
        # "Hi!" and "Bye." should be filtered out (< 3 words)
        assert len(sentences) == 1
        assert sentences[0] == "My name is John."
    
    def test_empty_text(self):
        """Test handling of empty text"""
        splitter = SentenceSplitter()
        assert splitter.split("") == []
        assert splitter.split("   ") == []
    
    def test_very_short_text(self):
        """Test handling of very short text"""
        splitter = SentenceSplitter(min_words=3)
        text = "Hi there"  # Only 2 words
        sentences = splitter.split(text)
        
        # Should return original text even if below threshold
        assert len(sentences) == 1
        assert sentences[0] == "Hi there"
    
    def test_split_with_metadata(self):
        """Test splitting with metadata"""
        splitter = SentenceSplitter()
        text = "My name is John. I love pizza."
        results = splitter.split_with_metadata(text)
        
        assert len(results) == 2
        assert results[0]['text'] == "My name is John."
        assert results[0]['index'] == 0
        assert results[0]['word_count'] == 4
        assert results[1]['text'] == "I love pizza."
        assert results[1]['index'] == 1
        assert results[1]['word_count'] == 3
    
    def test_paragraph_splitting(self):
        """Test splitting a longer paragraph"""
        splitter = SentenceSplitter()
        text = (
            "Hi! My name is Sarah Johnson and I'm a software engineer. "
            "I love playing basketball on weekends. "
            "My father is a doctor and my sister just graduated from MIT. "
            "I'm currently working at Google in San Francisco."
        )
        sentences = splitter.split(text)
        
        # Should split into 5 sentences (Hi! is filtered due to min_words)
        assert len(sentences) == 4  # "Hi!" filtered out
        assert any("Sarah Johnson" in s for s in sentences)
        assert any("basketball" in s for s in sentences)
        assert any("father" in s for s in sentences)
        assert any("Google" in s for s in sentences)


class TestMemoryTextSplitter:
    """Tests for memory-specific text splitter"""
    
    def test_filter_questions(self):
        """Test filtering of questions"""
        splitter = MemoryTextSplitter(filter_questions=True)
        text = "My name is John. What is your name? I love pizza."
        sentences = splitter.split_for_memory_extraction(text)
        
        # Question should be filtered out
        assert len(sentences) == 2
        assert "My name is John." in sentences
        assert "I love pizza." in sentences
        assert not any("What is your name?" in s for s in sentences)
    
    def test_no_filter_questions(self):
        """Test keeping questions when filter is disabled"""
        splitter = MemoryTextSplitter(filter_questions=False)
        text = "My name is John. What is your name? I love pizza."
        sentences = splitter.split_for_memory_extraction(text)
        
        # All sentences should be kept
        assert len(sentences) == 3
        assert "What is your name?" in sentences
    
    def test_filter_generic_statements(self):
        """Test filtering of generic statements"""
        splitter = MemoryTextSplitter(filter_questions=True)
        text = "Tell me about yourself. My name is John. Can you help me?"
        sentences = splitter.split_for_memory_extraction(text)
        
        # Generic statements should be filtered
        assert len(sentences) == 1
        assert sentences[0] == "My name is John."
    
    def test_all_filtered_fallback(self):
        """Test fallback when all sentences are filtered"""
        splitter = MemoryTextSplitter(filter_questions=True)
        text = "What is your name? Can you help me? How are you?"
        sentences = splitter.split_for_memory_extraction(text)
        
        # Should return original sentences when all are filtered
        assert len(sentences) == 3
    
    def test_mixed_content(self):
        """Test mixed memory and non-memory content"""
        splitter = MemoryTextSplitter(filter_questions=True)
        text = (
            "Hi! My name is Sarah and I love basketball. "
            "What sports do you like? "
            "My father is a doctor. "
            "Can you tell me more about that?"
        )
        sentences = splitter.split_for_memory_extraction(text)
        
        # Should keep only memory-worthy sentences
        assert len(sentences) == 2
        assert any("Sarah" in s for s in sentences)
        assert any("father" in s for s in sentences)


class TestConvenienceFunctions:
    """Tests for convenience functions"""
    
    def test_split_sentences_function(self):
        """Test split_sentences convenience function"""
        text = "My name is John. I love pizza."
        sentences = split_sentences(text)
        
        assert len(sentences) == 2
        assert "My name is John." in sentences
    
    def test_split_for_memory_function(self):
        """Test split_for_memory convenience function"""
        text = "My name is John. What is your name? I love pizza."
        sentences = split_for_memory(text)
        
        # Question should be filtered
        assert len(sentences) == 2
        assert "My name is John." in sentences
        assert "I love pizza." in sentences
    
    def test_split_for_memory_custom_params(self):
        """Test split_for_memory with custom parameters"""
        text = "Hi! My name is John. What is your name?"
        
        # With question filtering
        sentences = split_for_memory(text, filter_questions=True)
        assert len(sentences) == 1
        
        # Without question filtering
        sentences = split_for_memory(text, filter_questions=False)
        assert len(sentences) == 2


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_unicode_text(self):
        """Test handling of unicode characters"""
        splitter = SentenceSplitter()
        text = "My name is José. I love café. I work at Zürich."
        sentences = splitter.split(text)
        
        assert len(sentences) == 3
        assert "José" in sentences[0]
        assert "café" in sentences[1]
        assert "Zürich" in sentences[2]
    
    def test_multiple_punctuation(self):
        """Test handling of multiple punctuation marks"""
        splitter = SentenceSplitter()
        text = "Really?! Yes!!! That's amazing..."
        sentences = splitter.split(text)
        
        # Should handle multiple punctuation correctly
        assert len(sentences) >= 1
    
    def test_no_punctuation(self):
        """Test handling of text without punctuation"""
        splitter = SentenceSplitter()
        text = "My name is John I love pizza"
        sentences = splitter.split(text)
        
        # Should treat as single sentence
        assert len(sentences) == 1
    
    def test_newlines_and_whitespace(self):
        """Test handling of newlines and extra whitespace"""
        splitter = SentenceSplitter()
        text = "My name is John.\n\nI love pizza.  \n  I work at Google."
        sentences = splitter.split(text)
        
        assert len(sentences) == 3
        # Whitespace should be stripped
        assert all(s.strip() == s for s in sentences)
    
    def test_very_long_sentence(self):
        """Test handling of very long sentences"""
        splitter = SentenceSplitter()
        # Create a very long sentence
        text = "My name is " + " and ".join([f"thing{i}" for i in range(100)]) + "."
        sentences = splitter.split(text)
        
        # Should still be treated as one sentence
        assert len(sentences) == 1
        assert len(sentences[0]) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

