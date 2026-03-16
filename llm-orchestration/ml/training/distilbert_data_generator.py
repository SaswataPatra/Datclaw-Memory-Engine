"""
Training Data Generator for DistilBERT Memory Classifier

Generates synthetic training examples for each memory type using:
1. Seed examples (hand-crafted high-quality examples)
2. LLM-generated variations (GPT-4 for diversity)
3. Augmentation techniques (paraphrasing, context injection)
"""

from typing import List, Dict, Tuple, Optional
import random
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class DistilBERTDataGenerator:
    """
    Generate training data for DistilBERT memory classifier.
    
    Strategy:
    1. Start with seed examples for each memory type
    2. Use GPT-4 to generate variations
    3. Apply augmentation (negation, context, etc.)
    4. Balance dataset across all labels
    """
    
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Seed examples for each memory type
        self.seed_examples = {
            'identity': [
                "My name is Sarah Johnson",
                "I'm a software engineer",
                "People call me Sam",
                "I identify as non-binary",
                "I go by the name Alex",
                "I'm known as the tech guy in my friend group",
                "My full name is Michael Robert Thompson",
                "I prefer to be called Mike",
            ],
            'family': [
                "My mother's name is Elizabeth",
                "I have two younger brothers",
                "My dad works as a doctor",
                "My sister just graduated from college",
                "I'm married with three kids",
                "My grandmother raised me",
                "My parents divorced when I was young",
                "I have a twin brother named Jake",
            ],
            'preference': [
                "I love playing basketball",
                "I hate spicy food",
                "My favorite movie is The Shawshank Redemption",
                "I enjoy reading science fiction",
                "I can't stand horror movies",
                "I'm passionate about photography",
                "I adore Italian cuisine",
                "I prefer tea over coffee",
            ],
            'fact': [
                "I work at Google as a product manager",
                "I live in San Francisco",
                "I graduated from MIT in 2015",
                "I studied computer science",
                "My office is in downtown Seattle",
                "I went to Harvard for my MBA",
                "I'm currently working remotely from Bali",
                "I'm employed at a startup in Austin",
            ],
            'high_value': [
                "I'm responsible for a $50 million budget",
                "I manage a team of 200 engineers",
                "I'm the CTO of our company",
                "I handle portfolios worth over $2 billion",
                "I'm leading the merger with TechCorp",
                "I oversee the entire West Coast operations",
                "I'm the VP of Product at Amazon",
                "I'm managing the IPO process for our startup",
            ],
            'goal': [
                "I want to learn Spanish fluently",
                "My goal is to run a marathon next year",
                "I'm planning to start my own business",
                "I hope to travel to Japan someday",
                "I'm working towards becoming a senior engineer",
                "I want to write a book about my experiences",
                "My dream is to climb Mount Everest",
                "I'm aiming to get my PhD in AI",
            ],
            'relationship': [
                "My best friend is moving to New York",
                "I've been dating Sarah for two years",
                "My colleague John helped me with the project",
                "I met my mentor at a conference",
                "My roommate is from Australia",
                "I'm close with my neighbor Maria",
                "My boss is very supportive",
                "I have a study group with three classmates",
            ],
            'event': [
                "I went to a concert last weekend",
                "I'm attending a wedding next month",
                "I just came back from a trip to Paris",
                "I graduated last year",
                "I'm going to a conference in Boston",
                "I attended a hackathon yesterday",
                "I'm planning a surprise party for my friend",
                "I visited the Grand Canyon last summer",
            ],
            'opinion': [
                "I think remote work is more productive",
                "In my view, AI will change everything",
                "I believe climate change is the biggest threat",
                "I feel that education should be free",
                "I think Python is the best programming language",
                "In my opinion, the movie was overrated",
                "I believe in work-life balance",
                "I think social media is harmful",
            ],
            'unknown': [
                "What's the weather like today?",
                "Can you help me with this?",
                "That's interesting",
                "I see",
                "Tell me more about that",
                "How does that work?",
                "What do you think?",
                "Okay, got it",
            ]
        }
        
        logger.info("DistilBERTDataGenerator initialized with seed examples")
    
    def get_seed_examples(self) -> List[Tuple[str, List[str]]]:
        """
        Get all seed examples with their labels.
        
        Returns:
            List of (text, labels) tuples
        """
        examples = []
        for label, texts in self.seed_examples.items():
            for text in texts:
                examples.append((text, [label]))
        return examples
    
    async def generate_variations(
        self,
        seed_text: str,
        label: str,
        num_variations: int = 5
    ) -> List[str]:
        """
        Generate variations of a seed example using GPT-4.
        
        Args:
            seed_text: Original seed example
            label: Memory type label
            num_variations: Number of variations to generate
        
        Returns:
            List of generated variations
        """
        prompt = f"""Generate {num_variations} diverse variations of the following user statement.
Keep the core meaning and memory type ({label}), but vary:
- Phrasing and word choice
- Level of detail
- Tone (casual, formal, emotional)
- Context (add or remove background info)

Original statement: "{seed_text}"

Generate variations (one per line, no numbering):
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for data generation
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse text variations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # High temperature for diversity
                max_tokens=500
            )
            
            variations_text = response.choices[0].message.content.strip()
            variations = [
                line.strip()
                for line in variations_text.split('\n')
                if line.strip() and not line.strip().startswith('-')
            ]
            
            # Clean up variations (remove numbering, bullets, etc.)
            cleaned_variations = []
            for var in variations:
                # Remove common prefixes
                var = var.lstrip('0123456789.-) ')
                if len(var) > 10:  # Filter out very short variations
                    cleaned_variations.append(var)
            
            logger.debug(f"Generated {len(cleaned_variations)} variations for '{seed_text[:50]}...'")
            return cleaned_variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
            return []
    
    async def generate_full_dataset(
        self,
        variations_per_seed: int = 5,
        include_multi_label: bool = True
    ) -> List[Tuple[str, List[str]]]:
        """
        Generate full training dataset.
        
        Args:
            variations_per_seed: Number of variations to generate per seed
            include_multi_label: Whether to include multi-label examples
        
        Returns:
            List of (text, labels) tuples
        """
        dataset = []
        
        # Add seed examples
        seed_examples = self.get_seed_examples()
        dataset.extend(seed_examples)
        logger.info(f"Added {len(seed_examples)} seed examples")
        
        # Generate variations for each seed
        for label, seed_texts in self.seed_examples.items():
            logger.info(f"Generating variations for label: {label}")
            
            for seed_text in seed_texts:
                variations = await self.generate_variations(
                    seed_text, label, variations_per_seed
                )
                
                for variation in variations:
                    dataset.append((variation, [label]))
        
        logger.info(f"Generated {len(dataset)} single-label examples")
        
        # Generate multi-label examples
        if include_multi_label:
            multi_label_examples = await self._generate_multi_label_examples()
            dataset.extend(multi_label_examples)
            logger.info(f"Added {len(multi_label_examples)} multi-label examples")
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Final dataset size: {len(dataset)} examples")
        return dataset
    
    async def _generate_multi_label_examples(self) -> List[Tuple[str, List[str]]]:
        """
        Generate examples that belong to multiple categories.
        
        Examples:
        - "My sister loves playing tennis" -> ['family', 'preference']
        - "I work at Google and I love it" -> ['fact', 'preference']
        - "My goal is to become a CEO" -> ['goal', 'high_value']
        """
        multi_label_prompts = [
            ("family + preference", "Generate 5 statements about family members' preferences or hobbies"),
            ("fact + preference", "Generate 5 statements about work/education that express likes/dislikes"),
            ("goal + high_value", "Generate 5 statements about career goals involving leadership or high responsibility"),
            ("family + event", "Generate 5 statements about family events or gatherings"),
            ("relationship + preference", "Generate 5 statements about friends' or partners' preferences"),
            ("identity + fact", "Generate 5 statements combining self-description with work/education"),
        ]
        
        multi_label_examples = []
        
        for label_combo, prompt in multi_label_prompts:
            labels = label_combo.split(" + ")
            
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant generating realistic user statements."},
                        {"role": "user", "content": f"{prompt}. Return one statement per line, no numbering."}
                    ],
                    temperature=0.8,
                    max_tokens=300
                )
                
                statements_text = response.choices[0].message.content.strip()
                statements = [
                    line.strip().lstrip('0123456789.-) ')
                    for line in statements_text.split('\n')
                    if line.strip() and len(line.strip()) > 10
                ]
                
                for statement in statements[:5]:
                    multi_label_examples.append((statement, labels))
                
                logger.debug(f"Generated {len(statements)} examples for {label_combo}")
                
            except Exception as e:
                logger.error(f"Error generating multi-label examples for {label_combo}: {e}")
        
        return multi_label_examples
    
    def save_dataset(self, dataset: List[Tuple[str, List[str]]], filepath: str):
        """Save dataset to file"""
        import json
        
        with open(filepath, 'w') as f:
            for text, labels in dataset:
                f.write(json.dumps({"text": text, "labels": labels}) + '\n')
        
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Tuple[str, List[str]]]:
        """Load dataset from file"""
        import json
        
        dataset = []
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                dataset.append((data['text'], data['labels']))
        
        logger.info(f"Loaded {len(dataset)} examples from {filepath}")
        return dataset
    
    def get_label_distribution(self, dataset: List[Tuple[str, List[str]]]) -> Dict[str, int]:
        """Get distribution of labels in dataset"""
        from collections import Counter
        
        label_counts = Counter()
        for _, labels in dataset:
            for label in labels:
                label_counts[label] += 1
        
        return dict(label_counts)

