from typing import List, Dict, Any
import random
import uuid
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BootstrapDatasetGenerator:
    """
    Generates a synthetic bootstrap dataset for training the LightGBM ego scorer.
    Combines seed examples with GPT-generated variations.
    """
    
    def __init__(self, config: Dict[str, Any], llm_client: Any = None): # llm_client can be OpenAI, etc.
        self.config = config
        self.llm_client = llm_client
        self.seed_examples: List[Dict[str, Any]] = []
        self.synthetic_multiplier = config.get('training', {}).get('synthetic_multiplier', 50)
        self.max_synthetic_per_seed = config.get('training', {}).get('max_synthetic_per_seed', 50)
        
        logger.info("BootstrapDatasetGenerator initialized.")
    
    def add_seed_example(self, content: str, label: str, target_ego_score: float, user_id: str = "seed_user"):
        """
        Adds a human-labeled seed example to the dataset.
        """
        self.seed_examples.append({
            "content": content,
            "label": label,
            "user_id": user_id,
            "target_ego_score": target_ego_score,
            "source": "human_seed"
        })
        logger.debug(f"Added seed example: {content[:50]}... (score: {target_ego_score})")
    
    async def generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """
        Generates synthetic variations for each seed example using an LLM.
        """
        synthetic_data = []
        
        if not self.llm_client:
            logger.warning("LLM client not provided. Cannot generate synthetic data.")
            return []
        
        for seed in self.seed_examples:
            prompt = self._build_synthetic_prompt(seed)
            
            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.7,
                    n=1 # Request one response, then parse multiple variations
                )
                
                variations = self._parse_synthetic_response(response)
                
                for var_content in variations[:self.max_synthetic_per_seed]:
                    synthetic_data.append({
                        "content": var_content,
                        "label": seed['label'],
                        "user_id": seed['user_id'],
                        "target_ego_score": seed['target_ego_score'] + random.uniform(-0.05, 0.05), # Add slight noise
                        "source": "synthetic_llm",
                        "original_seed_content": seed['content']
                    })
                logger.debug(f"Generated {len(variations)} synthetic variations for seed: {seed['content'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error generating synthetic data for seed '{seed['content'][:50]}...': {e}")
        
        return synthetic_data
    
    def _build_synthetic_prompt(self, seed: Dict[str, Any]) -> str:
        """
        Builds the prompt for the LLM to generate synthetic variations.
        """
        return f"""Generate 5-10 diverse variations of the following user statement, keeping the core meaning and importance level.
The original statement is about '{seed['label']}' and has a target ego score of {seed['target_ego_score']:.2f}.

Original statement: "{seed['content']}"

Variations (one per line):
- """
    
    def _parse_synthetic_response(self, response: str) -> List[str]:
        """
        Parses the LLM's response to extract individual synthetic statements.
        """
        variations = [line.strip() for line in response.split('\n') if line.strip().startswith('- ')]
        return [v[2:] for v in variations if len(v) > 2] # Remove '- ' prefix
    
    async def generate_full_dataset(self) -> List[Dict[str, Any]]:
        """
        Generates the full bootstrap dataset including seed and synthetic data.
        """
        full_dataset = list(self.seed_examples) # Start with human-labeled seeds
        
        synthetic_data = await self.generate_synthetic_data()
        full_dataset.extend(synthetic_data)
        
        logger.info(f"Generated full bootstrap dataset with {len(full_dataset)} samples.")
        return full_dataset
    
    def _generate_mock_features(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates mock component scores for a data point.
        In a real scenario, this would come from actual component scorers.
        """
        # These are simplified for bootstrap generation
        # Actual values would come from running component scorers
        return {
            'novelty_score': random.uniform(0.0, 1.0),
            'frequency_score': random.uniform(0.0, 1.0),
            'sentiment_intensity': random.uniform(0.0, 1.0),
            'explicit_importance_score': data_point['target_ego_score'] + random.uniform(-0.1, 0.1),
            'engagement_score': random.uniform(0.0, 1.0),
            'recency_decay': random.uniform(0.0, 1.0),
            'reference_count': random.uniform(0, 5),
            'llm_confidence': random.uniform(0.7, 0.99),
            'source_weight': random.uniform(0.5, 1.0),
            'target_ego_score': data_point['target_ego_score']
        }
    
    async def prepare_training_data_with_features(self) -> List[Dict[str, Any]]:
        """
        Generates the full dataset and adds mock features for training.
        In a real pipeline, actual component scorers would be used here.
        """
        raw_dataset = await self.generate_full_dataset()
        
        training_data_with_features = []
        for data_point in raw_dataset:
            features = self._generate_mock_features(data_point)
            training_data_with_features.append(features)
        
        logger.info(f"Prepared {len(training_data_with_features)} samples with features for training.")
        return training_data_with_features
