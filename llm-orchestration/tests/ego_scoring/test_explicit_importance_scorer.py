"""
Comprehensive test suite for ExplicitImportanceScorer
Tests importance mapping based on memory labels/types
"""

import pytest
from ml.component_scorers import ExplicitImportanceScorer, ScorerResult


@pytest.fixture
def config():
    """Configuration for ExplicitImportanceScorer"""
    return {
        'ego_scoring': {
            'explicit_importance_map': {
                'identity': 1.0,
                'family': 1.0,
                'high_value': 0.95,
                'preference': 0.9,
                'goal': 0.85,
                'relationship': 0.85,
                'fact': 0.7,
                'work': 0.7,
                'education': 0.7,
                'event': 0.6,
                'opinion': 0.5,
                'unknown': 0.5
            },
            'default_explicit_importance': 0.5
        }
    }


@pytest.fixture
def explicit_importance_scorer(config):
    """Create ExplicitImportanceScorer instance"""
    return ExplicitImportanceScorer(config)


@pytest.mark.asyncio
async def test_explicit_importance_identity(explicit_importance_scorer):
    """Test identity label gets highest importance"""
    memory = {
        'content': 'My name is Saswata',
        'label': 'identity'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 1.0
    assert result.metadata['label'] == 'identity'
    assert result.metadata['mapped_importance'] == 1.0


@pytest.mark.asyncio
async def test_explicit_importance_family(explicit_importance_scorer):
    """Test family label gets highest importance"""
    memory = {
        'content': 'My father is Arun Kumar',
        'label': 'family'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 1.0
    assert result.metadata['label'] == 'family'


@pytest.mark.asyncio
async def test_explicit_importance_high_value(explicit_importance_scorer):
    """Test high_value label gets very high importance"""
    memory = {
        'content': 'I manage $5 billion in assets',
        'label': 'high_value'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.95
    assert result.metadata['label'] == 'high_value'


@pytest.mark.asyncio
async def test_explicit_importance_preference(explicit_importance_scorer):
    """Test preference label gets high importance"""
    memory = {
        'content': 'I love steaks',
        'label': 'preference'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.9
    assert result.metadata['label'] == 'preference'


@pytest.mark.asyncio
async def test_explicit_importance_goal(explicit_importance_scorer):
    """Test goal label importance"""
    memory = {
        'content': 'I want to learn quantum physics',
        'label': 'goal'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.85
    assert result.metadata['label'] == 'goal'


@pytest.mark.asyncio
async def test_explicit_importance_relationship(explicit_importance_scorer):
    """Test relationship label importance"""
    memory = {
        'content': 'John is my best friend',
        'label': 'relationship'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.85
    assert result.metadata['label'] == 'relationship'


@pytest.mark.asyncio
async def test_explicit_importance_fact(explicit_importance_scorer):
    """Test fact label importance"""
    memory = {
        'content': 'I work at Nexqloud',
        'label': 'fact'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.7
    assert result.metadata['label'] == 'fact'


@pytest.mark.asyncio
async def test_explicit_importance_work(explicit_importance_scorer):
    """Test work label importance"""
    memory = {
        'content': 'I am a blockchain developer',
        'label': 'work'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.7
    assert result.metadata['label'] == 'work'


@pytest.mark.asyncio
async def test_explicit_importance_education(explicit_importance_scorer):
    """Test education label importance"""
    memory = {
        'content': 'I studied at MIT',
        'label': 'education'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.7
    assert result.metadata['label'] == 'education'


@pytest.mark.asyncio
async def test_explicit_importance_event(explicit_importance_scorer):
    """Test event label importance"""
    memory = {
        'content': 'I went to the concert yesterday',
        'label': 'event'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.6
    assert result.metadata['label'] == 'event'


@pytest.mark.asyncio
async def test_explicit_importance_opinion(explicit_importance_scorer):
    """Test opinion label importance"""
    memory = {
        'content': 'I think the weather is nice',
        'label': 'opinion'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.5
    assert result.metadata['label'] == 'opinion'


@pytest.mark.asyncio
async def test_explicit_importance_unknown(explicit_importance_scorer):
    """Test unknown label gets default importance"""
    memory = {
        'content': 'Some random statement',
        'label': 'unknown'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.5
    assert result.metadata['label'] == 'unknown'


@pytest.mark.asyncio
async def test_explicit_importance_missing_label(explicit_importance_scorer):
    """Test missing label defaults to unknown"""
    memory = {
        'content': 'Some statement without label'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.5
    assert result.metadata['label'] == 'unknown'


@pytest.mark.asyncio
async def test_explicit_importance_case_insensitive(explicit_importance_scorer):
    """Test that label matching is case insensitive"""
    memory = {
        'content': 'My name is Saswata',
        'label': 'IDENTITY'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_explicit_importance_compound_label_full_match(explicit_importance_scorer):
    """Test compound label with full match (identity:name)"""
    memory = {
        'content': 'My name is Saswata',
        'label': 'identity:name'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    # Should match primary category 'identity' since full label not in map
    assert result.score == 1.0
    assert result.metadata['label'] == 'identity:name'


@pytest.mark.asyncio
async def test_explicit_importance_compound_label_primary_match(explicit_importance_scorer):
    """Test compound label falls back to primary category"""
    memory = {
        'content': 'I like pizza',
        'label': 'preference:food'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    # Should match primary category 'preference'
    assert result.score == 0.9


@pytest.mark.asyncio
async def test_explicit_importance_unmapped_label(explicit_importance_scorer):
    """Test unmapped label uses default"""
    memory = {
        'content': 'Some content',
        'label': 'random_unmapped_label'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.5  # default_explicit_importance


@pytest.mark.asyncio
async def test_explicit_importance_real_world_family_example(explicit_importance_scorer):
    """Test real-world example: father's name"""
    memory = {
        'content': 'my fathers name is arun kumar patra',
        'label': 'family'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 1.0
    assert result.metadata['label'] == 'family'
    assert result.metadata['mapped_importance'] == 1.0


@pytest.mark.asyncio
async def test_explicit_importance_real_world_preference_example(explicit_importance_scorer):
    """Test real-world example: food preference"""
    memory = {
        'content': 'i love steaks cooked medium rare',
        'label': 'preference'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert result.score == 0.9
    assert result.metadata['label'] == 'preference'


@pytest.mark.asyncio
async def test_explicit_importance_hierarchy(explicit_importance_scorer):
    """Test that importance hierarchy is correct"""
    labels_by_importance = [
        ('identity', 1.0),
        ('family', 1.0),
        ('high_value', 0.95),
        ('preference', 0.9),
        ('goal', 0.85),
        ('relationship', 0.85),
        ('fact', 0.7),
        ('work', 0.7),
        ('education', 0.7),
        ('event', 0.6),
        ('opinion', 0.5),
        ('unknown', 0.5)
    ]
    
    for label, expected_score in labels_by_importance:
        memory = {'content': 'test', 'label': label}
        result = await explicit_importance_scorer.score(memory)
        assert result.score == expected_score, f"Label '{label}' should have score {expected_score}"


@pytest.mark.asyncio
async def test_explicit_importance_metadata_completeness(explicit_importance_scorer):
    """Test that metadata includes all necessary information"""
    memory = {
        'content': 'My name is Saswata',
        'label': 'identity'
    }
    
    result = await explicit_importance_scorer.score(memory)
    
    assert 'label' in result.metadata
    assert 'mapped_importance' in result.metadata
    assert result.metadata['label'] == 'identity'
    assert result.metadata['mapped_importance'] == 1.0

