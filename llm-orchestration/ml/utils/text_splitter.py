"""
Text Splitting Utilities for Memory Extraction

Provides sentence-level text splitting for fine-grained memory classification.
"""

import logging
from typing import List, Tuple, Dict, Any
from nltk.tokenize import sent_tokenize
import nltk

logger = logging.getLogger(__name__)

# Ensure punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class SentenceSplitter:
    """
    Splits text into sentences for fine-grained classification.
    
    Handles edge cases like:
    - Abbreviations (Dr., U.S.A., etc.)
    - Multiple punctuation marks
    - Very short sentences (filtered out)
    """
    
    def __init__(self, min_words: int = 3):
        """
        Initialize sentence splitter.
        
        Args:
            min_words: Minimum number of words for a sentence to be valid
        """
        self.min_words = min_words
    
    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text (can be a paragraph or single sentence)
        
        Returns:
            List of sentences (filtered for minimum word count)
        """
        if not text or not text.strip():
            return []
        
        # Use NLTK's sentence tokenizer (handles abbreviations well)
        sentences = sent_tokenize(text)
        
        # Filter out very short sentences (likely noise or incomplete)
        valid_sentences = [
            s.strip() for s in sentences 
            if len(s.split()) >= self.min_words
        ]
        
        # If no valid sentences, return original text as single sentence
        if not valid_sentences:
            valid_sentences = [text.strip()]
        
        return valid_sentences
    
    def split_with_metadata(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sentences with metadata.
        
        Args:
            text: Input text
        
        Returns:
            List of dicts with 'text', 'index', 'word_count' keys
        """
        sentences = self.split(text)
        
        return [
            {
                'text': sent,
                'index': i,
                'word_count': len(sent.split()),
                'char_count': len(sent)
            }
            for i, sent in enumerate(sentences)
        ]


class MemoryTextSplitter(SentenceSplitter):
    """
    Specialized text splitter for memory extraction.
    
    Extends SentenceSplitter with memory-specific logic:
    - Filters out questions (usually not memories)
    - Filters out very generic statements
    - Handles multi-sentence memories
    """
    
    def __init__(self, min_words: int = 3, filter_questions: bool = True):
        """
        Initialize memory text splitter.
        
        Args:
            min_words: Minimum words per sentence
            filter_questions: Whether to filter out questions
        """
        super().__init__(min_words)
        self.filter_questions = filter_questions
    
    def split_for_memory_extraction(self, text: str) -> List[str]:
        """
        Split text and filter for memory-worthy sentences.
        
        Args:
            text: Input text
        
        Returns:
            List of memory-worthy sentences
        """
        sentences = self.split(text)
        
        if not self.filter_questions:
            return sentences
        
        # Filter out questions (usually not memories)
        memory_sentences = []
        for sent in sentences:
            # Skip if it's clearly a question
            if sent.strip().endswith('?'):
                logger.debug(f"Filtered question: {sent[:50]}...")
                continue
            
            # Skip very generic statements
            generic_patterns = [
                'what', 'how', 'why', 'when', 'where',
                'tell me', 'can you', 'could you', 'would you'
            ]
            
            sent_lower = sent.lower()
            if any(pattern in sent_lower[:20] for pattern in generic_patterns):
                logger.debug(f"Filtered generic: {sent[:50]}...")
                continue
            
            memory_sentences.append(sent)
        
        # If we filtered everything out, return original sentences
        if not memory_sentences:
            return sentences
        
        return memory_sentences


# Singleton instances for easy import
default_splitter = SentenceSplitter()
memory_splitter = MemoryTextSplitter()


def split_sentences(text: str, min_words: int = 3) -> List[str]:
    """
    Convenience function to split text into sentences.
    
    Args:
        text: Input text
        min_words: Minimum words per sentence
    
    Returns:
        List of sentences
    """
    splitter = SentenceSplitter(min_words=min_words)
    return splitter.split(text)


def split_for_memory(text: str, min_words: int = 3, filter_questions: bool = True) -> List[str]:
    """
    Convenience function to split text for memory extraction.
    
    Args:
        text: Input text
        min_words: Minimum words per sentence
        filter_questions: Whether to filter out questions
    
    Returns:
        List of memory-worthy sentences
    """
    splitter = MemoryTextSplitter(min_words=min_words, filter_questions=filter_questions)
    return splitter.split_for_memory_extraction(text)

