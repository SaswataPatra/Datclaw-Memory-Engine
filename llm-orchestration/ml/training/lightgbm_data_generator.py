"""
Bootstrap Dataset Generator for LightGBM Ego Score Combiner

Generates training data with:
1. Seed examples with hand-labeled target ego scores
2. GPT-4 generated variations
3. Component scores computed from actual scorers
4. Feature engineering for LightGBM input
"""

from typing import List, Dict, Tuple, Optional, Any
import random
import logging
import asyncio
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LightGBMDataGenerator:
    """
    Generate bootstrap training dataset for LightGBM ego score combiner.
    
    Strategy:
    1. Create seed examples with target ego scores (human-labeled)
    2. Generate variations using GPT-4
    3. Compute component scores using actual scorers
    4. Create feature vectors for LightGBM training
    """
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Seed examples with target ego scores
        # Format: (text, label, target_ego_score, reasoning)
        self.seed_examples = [
            # Tier 1: Core memories (ego_score >= 0.75)
            ("My name is Sarah Johnson", "identity", 0.95, "Core identity information"),
            ("My mother's name is Elizabeth Chen", "family", 0.90, "Close family member"),
            ("I'm the CEO of TechCorp", "high_value", 0.92, "High-stakes professional identity"),
            ("My father passed away last year", "family", 0.88, "Significant family event"),
            ("I have a twin sister named Emma", "family", 0.85, "Close family relationship"),
            ("I'm responsible for a $500M budget", "high_value", 0.90, "High-value responsibility"),
            ("My goal is to become a published author", "goal", 0.80, "Important personal goal"),
            ("I'm married to Alex for 10 years", "relationship", 0.82, "Long-term significant relationship"),
            ("I was diagnosed with diabetes", "medical_condition", 0.88, "Important health condition"),
            ("I'm on insulin therapy", "treatment_implications", 0.85, "Critical medical treatment"),
            ("I'm allergic to peanuts", "personal_health", 0.80, "Serious health information"),
            ("I'm non-binary and use they/them pronouns", "identity", 0.93, "Core identity"),
            ("My daughter was born in 2020", "family", 0.87, "Major life event"),
            
            # Tier 2: Long-term memories (0.50 <= ego_score < 0.75)
            ("I love playing basketball", "preference", 0.70, "Strong personal preference"),
            ("I work at Google as a software engineer", "fact", 0.68, "Important biographical fact"),
            ("I graduated from MIT in 2015", "fact", 0.65, "Educational background"),
            ("I live in San Francisco", "fact", 0.62, "Current location"),
            ("I hate spicy food", "preference", 0.60, "Clear preference"),
            ("My best friend is moving to New York", "relationship", 0.58, "Close friend event"),
            ("I'm learning Spanish", "goal", 0.55, "Active goal"),
            ("I enjoy reading science fiction", "preference", 0.52, "Hobby/interest"),
            ("I'm vegetarian for ethical reasons", "preference", 0.68, "Strong value-based preference"),
            ("I'm training for a marathon", "goal", 0.63, "Active fitness goal"),
            ("I grew up in a small town in Texas", "fact", 0.60, "Childhood background"),
            ("I have anxiety and see a therapist", "personal_health", 0.72, "Mental health information"),
            ("I'm planning to buy a house next year", "goal", 0.58, "Major life goal"),
            ("I speak three languages fluently", "fact", 0.64, "Significant skill"),
            
            # Tier 3: Short-term memories (0.20 <= ego_score < 0.50)
            ("I went to a concert last weekend", "event", 0.45, "Recent event"),
            ("I think remote work is productive", "opinion", 0.42, "Personal opinion"),
            ("I'm attending a conference next month", "event", 0.40, "Upcoming event"),
            ("I visited the Grand Canyon last summer", "event", 0.38, "Past experience"),
            ("I prefer tea over coffee", "preference", 0.35, "Minor preference"),
            ("My colleague helped me with a project", "relationship", 0.32, "Work relationship"),
            ("I'm reading a book about AI", "event", 0.28, "Current activity"),
            ("I like the color blue", "preference", 0.25, "Trivial preference"),
            ("I had a great workout this morning", "event", 0.33, "Recent activity"),
            ("I'm feeling stressed about deadlines", "opinion", 0.38, "Current emotional state"),
            ("I tried a new restaurant yesterday", "event", 0.30, "Recent experience"),
            ("I think AI will change everything", "opinion", 0.44, "Technology opinion"),
            ("I'm watching a new TV series", "event", 0.26, "Current entertainment"),
            ("I prefer working in the morning", "preference", 0.29, "Work preference"),
            
            # Tier 4: Hot buffer (ego_score < 0.20)
            ("What's the weather like today?", "unknown", 0.10, "Generic question"),
            ("That's interesting", "unknown", 0.08, "Generic response"),
            ("I see", "unknown", 0.05, "Acknowledgment"),
            ("Tell me more about that", "unknown", 0.07, "Generic prompt"),
            ("How does that work?", "unknown", 0.12, "Generic question"),
            ("Okay, got it", "unknown", 0.06, "Acknowledgment"),
            ("Thanks for the info", "unknown", 0.09, "Generic thanks"),
            ("Can you explain that?", "unknown", 0.11, "Generic request"),
            ("Hmm, interesting point", "unknown", 0.08, "Generic acknowledgment"),
            ("What do you think?", "unknown", 0.10, "Generic question"),
        ]
        
        logger.info(f"LightGBMDataGenerator initialized with {len(self.seed_examples)} seed examples")
    
    def get_seed_examples(self) -> List[Tuple[str, str, float, str]]:
        """Get all seed examples"""
        return self.seed_examples
    
    async def generate_variations(
        self,
        seed_text: str,
        label: str,
        target_ego_score: float,
        num_variations: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Generate variations of a seed example while maintaining similar ego score.
        
        Args:
            seed_text: Original seed example
            label: Memory type label
            target_ego_score: Target ego score for this example
            num_variations: Number of variations to generate
        
        Returns:
            List of (text, label, ego_score) tuples
        """
        # Determine tier for context
        if target_ego_score >= 0.75:
            tier_desc = "Tier 1 (Core memory - highly important, identity-defining)"
        elif target_ego_score >= 0.50:
            tier_desc = "Tier 2 (Long-term - important biographical information)"
        elif target_ego_score >= 0.20:
            tier_desc = "Tier 3 (Short-term - recent events, minor preferences)"
        else:
            tier_desc = "Tier 4 (Hot buffer - generic, low importance)"
        
        prompt = f"""Generate {num_variations} variations of the following user statement.
The original statement is a {tier_desc} memory with ego score {target_ego_score:.2f}.

Original: "{seed_text}"
Label: {label}

Generate variations that maintain similar importance level and meaning:
- Keep the same memory type ({label})
- Maintain similar ego score (~{target_ego_score:.2f})
- Vary phrasing, detail level, and tone
- Some variations can be slightly more/less important (±0.05 ego score)

Return one variation per line, no numbering:
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant generating realistic user statements with consistent importance levels."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            variations_text = response.choices[0].message.content.strip()
            variations = [
                line.strip().lstrip('0123456789.-) ')
                for line in variations_text.split('\n')
                if line.strip() and len(line.strip()) > 10
            ]
            
            # Assign ego scores with slight variation
            result = []
            for var in variations[:num_variations]:
                # Add small random variation to ego score (±0.05)
                varied_score = target_ego_score + random.uniform(-0.05, 0.05)
                varied_score = max(0.0, min(1.0, varied_score))  # Clip to [0, 1]
                result.append((var, label, varied_score))
            
            logger.debug(f"Generated {len(result)} variations for '{seed_text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return []
    
    async def generate_full_dataset(
        self,
        variations_per_seed: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Generate full bootstrap dataset.
        
        Args:
            variations_per_seed: Number of variations per seed example
        
        Returns:
            List of (text, label, target_ego_score) tuples
        """
        dataset = []
        
        # Add seed examples
        for text, label, ego_score, _ in self.seed_examples:
            dataset.append((text, label, ego_score))
        
        logger.info(f"Added {len(dataset)} seed examples")
        
        # Generate variations
        for text, label, ego_score, reasoning in self.seed_examples:
            logger.info(f"Generating variations for: '{text[:50]}...' (ego={ego_score:.2f})")
            
            variations = await self.generate_variations(
                text, label, ego_score, variations_per_seed
            )
            dataset.extend(variations)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        return dataset
    
    def save_dataset(self, dataset: List[Tuple[str, str, float]], filepath: str):
        """Save dataset to file (without features - those will be computed during training)"""
        import json
        
        with open(filepath, 'w') as f:
            for text, label, target_ego_score in dataset:
                f.write(json.dumps({
                    "text": text,
                    "label": label,
                    "target_ego_score": target_ego_score
                }) + '\n')
        
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Tuple[str, str, float]]:
        """Load dataset from file"""
        import json
        
        dataset = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                dataset.append((data['text'], data['label'], data['target_ego_score']))
        
        logger.info(f"Loaded {len(dataset)} examples from {filepath}")
        return dataset
    
    def get_ego_score_distribution(self, dataset: List[Tuple[str, str, float]]) -> Dict[str, int]:
        """Get distribution of ego scores by tier"""
        from collections import Counter
        
        tier_counts = Counter()
        for _, _, ego_score in dataset:
            if ego_score >= 0.75:
                tier_counts['Tier 1 (>= 0.75)'] += 1
            elif ego_score >= 0.50:
                tier_counts['Tier 2 (0.50-0.75)'] += 1
            elif ego_score >= 0.20:
                tier_counts['Tier 3 (0.20-0.50)'] += 1
            else:
                tier_counts['Tier 4 (< 0.20)'] += 1
        
        return dict(tier_counts)
    
    def get_label_distribution(self, dataset: List[Tuple[str, str, float]]) -> Dict[str, int]:
        """Get distribution of labels"""
        from collections import Counter
        
        label_counts = Counter()
        for _, label, _ in dataset:
            label_counts[label] += 1
        
        return dict(label_counts)


async def compute_component_scores(
    text: str,
    label: str,
    user_id: str,
    component_scorers: Dict[str, Any],
    embedding_service: Any,
    simulate_realistic_features: bool = True
) -> Dict[str, float]:
    """
    Compute component scores for a single text using actual scorers.
    
    Args:
        text: User message
        label: Memory type label
        user_id: User ID for frequency/novelty scoring
        component_scorers: Dict of initialized component scorers
        embedding_service: Embedding service for generating embeddings
        simulate_realistic_features: If True, add realistic variation to features
    
    Returns:
        Dict of component scores
    """
    # Generate embedding
    embedding = await embedding_service.generate(text)
    
    # Simulate realistic response length variation
    base_response_len = len(text) * 3  # Assume response is ~3x input length
    response_variation = random.uniform(0.5, 2.0)  # ±50% to +100%
    simulated_response_len = int(base_response_len * response_variation)
    
    # Simulate followup count based on label importance
    followup_weights = {
        'identity': (1, 4),
        'family': (1, 4),
        'high_value': (1, 3),
        'medical_condition': (1, 3),
        'treatment_implications': (1, 3),
        'goal': (0, 3),
        'fact': (0, 2),
        'preference': (0, 2),
        'relationship': (0, 2),
        'event': (0, 1),
        'opinion': (0, 1),
        'unknown': (0, 0)
    }
    followup_range = followup_weights.get(label, (0, 1))
    simulated_followup = random.randint(*followup_range)
    
    # Simulate elaboration score (higher for important memories)
    elaboration_weights = {
        'identity': (0.7, 1.0),
        'family': (0.7, 1.0),
        'high_value': (0.6, 0.9),
        'medical_condition': (0.6, 0.9),
        'goal': (0.5, 0.8),
        'fact': (0.4, 0.7),
        'preference': (0.3, 0.6),
        'event': (0.2, 0.5),
        'opinion': (0.2, 0.5),
        'unknown': (0.1, 0.3)
    }
    elaboration_range = elaboration_weights.get(label, (0.3, 0.6))
    simulated_elaboration = random.uniform(*elaboration_range)
    
    # Prepare memory dict
    memory = {
        'content': text,
        'user_id': user_id,
        'label': label,
        'embedding': embedding,
        'user_response_length': simulated_response_len,
        'followup_count': simulated_followup,
        'elaboration_score': simulated_elaboration
    }
    
    scores = {}
    
    # Novelty score - simulate realistic distribution
    if 'novelty' in component_scorers and embedding:
        result = await component_scorers['novelty'].score(memory)
        base_novelty = result.score
        
        if simulate_realistic_features:
            # Add realistic variation: most memories are somewhat novel (0.6-1.0)
            # but some are repeated (0.3-0.6)
            if random.random() < 0.7:  # 70% are novel
                scores['novelty_score'] = random.uniform(0.6, 1.0)
            else:  # 30% are repeated
                scores['novelty_score'] = random.uniform(0.3, 0.6)
        else:
            scores['novelty_score'] = base_novelty
    else:
        scores['novelty_score'] = random.uniform(0.5, 1.0) if simulate_realistic_features else 0.5
    
    # Frequency score - simulate realistic distribution
    if 'frequency' in component_scorers and embedding:
        result = await component_scorers['frequency'].score(memory)
        base_frequency = result.score
        
        if simulate_realistic_features:
            # Frequency depends on label type
            # Identity/family: often discussed (0.2-0.6)
            # Facts/preferences: sometimes discussed (0.1-0.4)
            # Events/opinions: rarely discussed (0.0-0.2)
            frequency_ranges = {
                'identity': (0.2, 0.6),
                'family': (0.2, 0.5),
                'high_value': (0.1, 0.4),
                'medical_condition': (0.2, 0.5),
                'preference': (0.1, 0.3),
                'fact': (0.1, 0.3),
                'goal': (0.1, 0.4),
                'relationship': (0.0, 0.3),
                'event': (0.0, 0.2),
                'opinion': (0.0, 0.2),
                'unknown': (0.0, 0.1)
            }
            freq_range = frequency_ranges.get(label, (0.0, 0.2))
            scores['frequency_score'] = random.uniform(*freq_range)
        else:
            scores['frequency_score'] = base_frequency
    else:
        if simulate_realistic_features:
            freq_range = (0.0, 0.3)
            scores['frequency_score'] = random.uniform(*freq_range)
        else:
            scores['frequency_score'] = 0.0
    
    # Sentiment score - use actual scorer
    if 'sentiment' in component_scorers:
        result = await component_scorers['sentiment'].score(memory)
        scores['sentiment_intensity'] = result.score
    else:
        scores['sentiment_intensity'] = 0.0
    
    # Explicit importance score - use actual scorer
    if 'explicit_importance' in component_scorers:
        result = await component_scorers['explicit_importance'].score(memory)
        scores['explicit_importance_score'] = result.score
    else:
        scores['explicit_importance_score'] = 0.5
    
    # Engagement score - use actual scorer with simulated features
    if 'engagement' in component_scorers:
        result = await component_scorers['engagement'].score(memory)
        scores['engagement_score'] = result.score
    else:
        scores['engagement_score'] = 0.5
    
    # Additional features with realistic variation
    if simulate_realistic_features:
        # Recency decay: most memories are recent (0.8-1.0), some are older (0.5-0.8)
        scores['recency_decay'] = random.uniform(0.8, 1.0) if random.random() < 0.7 else random.uniform(0.5, 0.8)
        
        # Reference count: most have 0-2 references
        scores['reference_count'] = random.choice([0, 0, 0, 1, 1, 2])
        
        # LLM confidence: usually high (0.7-0.9), sometimes uncertain (0.5-0.7)
        scores['llm_confidence'] = random.uniform(0.7, 0.9) if random.random() < 0.8 else random.uniform(0.5, 0.7)
        
        # Source weight: usually 1.0 (user message), sometimes 0.8 (inferred)
        scores['source_weight'] = 1.0 if random.random() < 0.9 else 0.8
    else:
        scores['recency_decay'] = 1.0
        scores['reference_count'] = 0
        scores['llm_confidence'] = 0.8
        scores['source_weight'] = 1.0
    
    return scores

