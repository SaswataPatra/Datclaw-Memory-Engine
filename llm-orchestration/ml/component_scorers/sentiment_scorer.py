from typing import Dict, Any
from ml.component_scorers.base import ComponentScorer, ScorerResult
import re


class SentimentScorer(ComponentScorer):
    """
    Calculates sentiment score for a memory's content using a heuristic approach.
    Returns a score between -1.0 (very negative) and +1.0 (very positive).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        sentiment_config = config.get('ego_scoring', {}).get('sentiment', {})
        
        # 🔄 HYPERPARAMETER - Will be REPLACED by transformer model in Phase 2
        # Current: Heuristic word lists for basic sentiment analysis
        # Future: DistilBERT or RoBERTa for context-aware sentiment (understands sarcasm, negation, etc.)
        # These lists are Phase 1 placeholders - adequate for MVP but not production-grade
        
        self.positive_words = set(sentiment_config.get('positive_words', [
            'love', 'like', 'enjoy', 'great', 'awesome', 'amazing', 'excellent',
            'wonderful', 'fantastic', 'happy', 'glad', 'excited', 'thrilled',
            'perfect', 'brilliant', 'best', 'favorite', 'favourite', 'adore',
            'appreciate', 'delighted', 'pleased', 'satisfied', 'grateful',
            'good', 'positive', 'optimistic', 'hopeful', 'joy', 'bliss'
        ]))
        
        self.negative_words = set(sentiment_config.get('negative_words', [
            'hate', 'dislike', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'annoying', 'frustrating', 'angry', 'upset', 'sad', 'disappointed',
            'disgusting', 'despise', 'detest', 'unfortunate', 'regret', 'worried',
            'stressed', 'scared', 'afraid', 'anxious', 'concerned',
            'poor', 'negative', 'pessimistic', 'dread', 'misery', 'grief'
        ]))
        
        self.intensifiers = set(sentiment_config.get('intensifiers', [
            'very', 'extremely', 'really', 'super', 'incredibly', 'highly', 'deeply'
        ]))
        
        self.diminishers = set(sentiment_config.get('diminishers', [
            'a little', 'slightly', 'somewhat', 'barely', 'hardly'
        ]))
        
        self.negations = set(sentiment_config.get('negations', [
            'not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t', 'aren\'t'
        ]))
    
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Score sentiment of a memory's content.
        Requires 'content' in memory.
        """
        content = memory.get('content', '')
        if not content:
            return ScorerResult(score=0.0, metadata={"reason": "empty_content"})
        
        tokens = re.findall(r'\b\w+\b', content.lower())
        
        sentiment_score = 0
        for i, token in enumerate(tokens):
            current_score = 0
            
            if token in self.positive_words:
                current_score = 1
            elif token in self.negative_words:
                current_score = -1
            
            if current_score != 0:
                # Check for negations (e.g., "not good")
                if i > 0 and tokens[i-1] in self.negations:
                    current_score *= -1
                
                # Check for intensifiers/diminishers
                if i > 0:
                    if tokens[i-1] in self.intensifiers:
                        current_score *= 1.5
                    elif tokens[i-1] in self.diminishers:
                        current_score *= 0.5
                
                sentiment_score += current_score
        
        # Normalize to [-1, 1]
        if len(tokens) > 0:
            normalized_score = sentiment_score / len(tokens)
        else:
            normalized_score = 0.0
        
        # Convert to [0, 1] for ego scoring (intensity)
        # We want the absolute intensity for ego scoring, not valence
        # e.g., "I love it" (0.8) and "I hate it" (0.8) are both intense
        sentiment_intensity = abs(normalized_score)
        
        return ScorerResult(
            score=sentiment_intensity,
            metadata={"raw_sentiment": normalized_score, "sentiment_intensity": sentiment_intensity}
        )
