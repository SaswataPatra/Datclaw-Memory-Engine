"""
Test suite for Bootstrap Dataset Generator.
Tests seed example management and synthetic data generation.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from ml.training.bootstrap_generator import BootstrapDatasetGenerator


@pytest.fixture
def base_config():
    """Base configuration for Bootstrap Generator"""
    return {
        'training': {
            'synthetic_multiplier': 50,
            'max_synthetic_per_seed': 50
        }
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = AsyncMock()
    return client


def test_bootstrap_generator_initialization(base_config, mock_llm_client):
    """Test Bootstrap Generator initialization"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    assert generator.llm_client == mock_llm_client
    assert generator.seed_examples == []
    assert generator.synthetic_multiplier == 50
    assert generator.max_synthetic_per_seed == 50


def test_bootstrap_generator_add_seed_example(base_config, mock_llm_client):
    """Test adding seed examples"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example(
        content="My name is Saswata",
        label="identity:name",
        target_ego_score=1.0,
        user_id="test_user"
    )
    
    assert len(generator.seed_examples) == 1
    assert generator.seed_examples[0]['content'] == "My name is Saswata"
    assert generator.seed_examples[0]['label'] == "identity:name"
    assert generator.seed_examples[0]['target_ego_score'] == 1.0
    assert generator.seed_examples[0]['source'] == "human_seed"


def test_bootstrap_generator_add_multiple_seed_examples(base_config, mock_llm_client):
    """Test adding multiple seed examples"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    generator.add_seed_example("I love Python", "preference:programming", 0.9)
    generator.add_seed_example("The sky is blue", "fact", 0.7)
    
    assert len(generator.seed_examples) == 3


@pytest.mark.asyncio
async def test_bootstrap_generator_synthetic_data_no_llm(base_config):
    """Test synthetic data generation without LLM client"""
    generator = BootstrapDatasetGenerator(base_config, None)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    
    synthetic_data = await generator.generate_synthetic_data()
    
    assert synthetic_data == []


@pytest.mark.asyncio
async def test_bootstrap_generator_synthetic_data_generation(base_config, mock_llm_client):
    """Test synthetic data generation with mock LLM"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    
    # Mock LLM response
    mock_llm_client.generate.return_value = """- My name is John
- I'm called Sarah
- They call me Alex
- My given name is Maria
- I go by the name of David"""
    
    synthetic_data = await generator.generate_synthetic_data()
    
    assert len(synthetic_data) == 5
    assert all(d['label'] == 'identity:name' for d in synthetic_data)
    assert all(d['source'] == 'synthetic_llm' for d in synthetic_data)
    assert all(0.95 <= d['target_ego_score'] <= 1.05 for d in synthetic_data)


@pytest.mark.asyncio
async def test_bootstrap_generator_synthetic_data_max_limit(base_config, mock_llm_client):
    """Test that synthetic data respects max_synthetic_per_seed limit"""
    config = base_config.copy()
    config['training']['max_synthetic_per_seed'] = 3
    
    generator = BootstrapDatasetGenerator(config, mock_llm_client)
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    
    # Mock LLM response with more than max_synthetic_per_seed variations
    mock_llm_client.generate.return_value = """- Variation 1
- Variation 2
- Variation 3
- Variation 4
- Variation 5"""
    
    synthetic_data = await generator.generate_synthetic_data()
    
    # Should only generate up to max_synthetic_per_seed
    assert len(synthetic_data) <= 3


@pytest.mark.asyncio
async def test_bootstrap_generator_full_dataset(base_config, mock_llm_client):
    """Test full dataset generation (seeds + synthetic)"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    generator.add_seed_example("I love Python", "preference:programming", 0.9)
    
    # Mock LLM response
    mock_llm_client.generate.return_value = """- Variation 1
- Variation 2
- Variation 3"""
    
    full_dataset = await generator.generate_full_dataset()
    
    # Should include 2 seed examples + 6 synthetic (3 per seed)
    assert len(full_dataset) == 8
    
    # Check that seed examples are included
    seed_examples = [d for d in full_dataset if d['source'] == 'human_seed']
    assert len(seed_examples) == 2
    
    # Check that synthetic examples are included
    synthetic_examples = [d for d in full_dataset if d['source'] == 'synthetic_llm']
    assert len(synthetic_examples) == 6


@pytest.mark.asyncio
async def test_bootstrap_generator_parse_synthetic_response(base_config, mock_llm_client):
    """Test parsing of LLM synthetic response"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    response = """- First variation
- Second variation
- Third variation
Some other text
- Fourth variation"""
    
    variations = generator._parse_synthetic_response(response)
    
    assert len(variations) == 4
    assert variations[0] == "First variation"
    assert variations[1] == "Second variation"
    assert variations[2] == "Third variation"
    assert variations[3] == "Fourth variation"


@pytest.mark.asyncio
async def test_bootstrap_generator_build_synthetic_prompt(base_config, mock_llm_client):
    """Test building of synthetic prompt"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    seed = {
        'content': 'My name is Saswata',
        'label': 'identity:name',
        'target_ego_score': 1.0
    }
    
    prompt = generator._build_synthetic_prompt(seed)
    
    assert 'My name is Saswata' in prompt
    assert 'identity:name' in prompt
    assert '1.00' in prompt


@pytest.mark.asyncio
async def test_bootstrap_generator_mock_features(base_config, mock_llm_client):
    """Test mock feature generation"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    data_point = {
        'content': 'My name is Saswata',
        'label': 'identity:name',
        'target_ego_score': 1.0
    }
    
    features = generator._generate_mock_features(data_point)
    
    assert 'novelty_score' in features
    assert 'frequency_score' in features
    assert 'sentiment_intensity' in features
    assert 'explicit_importance_score' in features
    assert 'engagement_score' in features
    assert 'recency_decay' in features
    assert 'reference_count' in features
    assert 'llm_confidence' in features
    assert 'source_weight' in features
    assert 'target_ego_score' in features
    
    # All scores should be between 0 and 1 (except reference_count)
    assert 0 <= features['novelty_score'] <= 1
    assert 0 <= features['frequency_score'] <= 1
    assert 0 <= features['sentiment_intensity'] <= 1
    assert features['target_ego_score'] == 1.0


@pytest.mark.asyncio
async def test_bootstrap_generator_prepare_training_data(base_config, mock_llm_client):
    """Test preparation of training data with features"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    generator.add_seed_example("I love Python", "preference:programming", 0.9)
    
    # Mock LLM response
    mock_llm_client.generate.return_value = """- Variation 1
- Variation 2"""
    
    training_data = await generator.prepare_training_data_with_features()
    
    # Should have features for all examples (2 seeds + 4 synthetic)
    assert len(training_data) == 6
    
    # Each should have all required features
    for data in training_data:
        assert 'novelty_score' in data
        assert 'target_ego_score' in data


@pytest.mark.asyncio
async def test_bootstrap_generator_llm_error_handling(base_config, mock_llm_client):
    """Test error handling when LLM fails"""
    generator = BootstrapDatasetGenerator(base_config, mock_llm_client)
    
    generator.add_seed_example("My name is Saswata", "identity:name", 1.0)
    
    # Mock LLM to raise an exception
    mock_llm_client.generate.side_effect = Exception("LLM API error")
    
    synthetic_data = await generator.generate_synthetic_data()
    
    # Should return empty list on error
    assert synthetic_data == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

