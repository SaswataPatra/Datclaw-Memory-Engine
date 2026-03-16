"""
Test KG Maintenance Agent prompt formatting.

This test ensures the prompt can be formatted without errors.
"""

import pytest
from services.kg_maintenance_agent import CONTRADICTION_DETECTION_PROMPT


def test_prompt_formatting_basic():
    """Test that the prompt can be formatted with basic inputs."""
    try:
        formatted = CONTRADICTION_DETECTION_PROMPT.format(
            new_memory="Test memory content",
            new_relations="- user --works_at--> company",
            existing_relations="- user --has_name--> john",
            contradiction_signals="No signals",
            reinforcement_candidates="No candidates"
        )
        assert len(formatted) > 0
        assert "Test memory content" in formatted
        print("✅ Basic formatting test passed")
    except Exception as e:
        pytest.fail(f"Prompt formatting failed: {e}")


def test_prompt_formatting_empty_inputs():
    """Test that the prompt can be formatted with empty inputs."""
    try:
        formatted = CONTRADICTION_DETECTION_PROMPT.format(
            new_memory="",
            new_relations="",
            existing_relations="",
            contradiction_signals="",
            reinforcement_candidates=""
        )
        assert len(formatted) > 0
        print("✅ Empty inputs formatting test passed")
    except Exception as e:
        pytest.fail(f"Prompt formatting with empty inputs failed: {e}")


def test_prompt_contains_json_examples():
    """Test that the prompt contains properly formatted JSON examples."""
    formatted = CONTRADICTION_DETECTION_PROMPT.format(
        new_memory="test",
        new_relations="test",
        existing_relations="test",
        contradiction_signals="test",
        reinforcement_candidates="test"
    )
    
    # Check that JSON structure is present (after formatting, {{ becomes {)
    assert '"contradictions":' in formatted
    assert '"reinforcements":' in formatted
    assert '"new_relations_to_add":' in formatted
    assert '{"subject":' in formatted or '{{"subject":' in formatted
    print("✅ JSON examples test passed")


def test_prompt_formatting_with_special_chars():
    """Test that the prompt can handle special characters in inputs."""
    try:
        formatted = CONTRADICTION_DETECTION_PROMPT.format(
            new_memory="User said: \"My father's name is O'Brien\"",
            new_relations="- user --has_father--> o'brien",
            existing_relations="- user --works_at--> company (confidence: 0.80)",
            contradiction_signals="Similar memory: \"Father is John\" (90 days ago)",
            reinforcement_candidates="- user --works_at--> company (mentions: 3)"
        )
        assert len(formatted) > 0
        assert "O'Brien" in formatted or "o'brien" in formatted
        print("✅ Special characters test passed")
    except Exception as e:
        pytest.fail(f"Prompt formatting with special chars failed: {e}")


def test_prompt_formatting_realistic():
    """Test with realistic data similar to production."""
    try:
        new_relations = """- user --works_at--> nexqloud (confidence: 0.80)
- user --skilled_in--> blockchain (confidence: 0.75)"""
        
        existing_relations = """- user --works_at--> datclaw (confidence: 0.80)
- user --has_father--> john patra (confidence: 0.90)
- user --lives_in--> mumbai (confidence: 0.85)"""
        
        contradiction_signals = """Similar memories found:
1. "I work at Datclaw" (similarity: 0.85, 30 days ago)
2. "My father is John" (similarity: 0.75, 60 days ago)"""
        
        reinforcement_candidates = """- user --works_at--> nexqloud (mentions: 2, last: 2024-01-15, created: 2024-01-01)
- user --skilled_in--> blockchain (mentions: 1, last: 2024-01-10, created: 2024-01-10)"""
        
        formatted = CONTRADICTION_DETECTION_PROMPT.format(
            new_memory="I work at Nexqloud as a blockchain developer",
            new_relations=new_relations,
            existing_relations=existing_relations,
            contradiction_signals=contradiction_signals,
            reinforcement_candidates=reinforcement_candidates
        )
        
        assert len(formatted) > 0
        assert "Nexqloud" in formatted
        assert "blockchain" in formatted
        print("✅ Realistic data test passed")
    except Exception as e:
        pytest.fail(f"Prompt formatting with realistic data failed: {e}")


def test_prompt_no_unescaped_braces():
    """Test that there are no unescaped single braces that would cause format errors."""
    import re
    
    # Get the raw prompt string
    prompt = CONTRADICTION_DETECTION_PROMPT
    
    # Find all {something} patterns (format placeholders)
    placeholders = re.findall(r'\{([a-z_]+)\}', prompt)
    expected_placeholders = ['new_memory', 'new_relations', 'existing_relations', 
                             'contradiction_signals', 'reinforcement_candidates']
    
    # Check that all placeholders are expected
    for placeholder in placeholders:
        assert placeholder in expected_placeholders, f"Unexpected placeholder: {{{placeholder}}}"
    
    # Check that all expected placeholders are present
    for expected in expected_placeholders:
        assert expected in placeholders, f"Missing expected placeholder: {{{expected}}}"
    
    # Try to actually format the prompt - this is the real test
    try:
        formatted = prompt.format(
            new_memory="test",
            new_relations="test",
            existing_relations="test",
            contradiction_signals="test",
            reinforcement_candidates="test"
        )
        assert len(formatted) > 0
    except (IndexError, KeyError) as e:
        pytest.fail(f"Prompt has formatting errors: {e}")
    
    print("✅ No unescaped braces test passed")


if __name__ == "__main__":
    print("Running KG Maintenance Prompt Tests...")
    print("=" * 80)
    
    test_prompt_formatting_basic()
    test_prompt_formatting_empty_inputs()
    test_prompt_contains_json_examples()
    test_prompt_formatting_with_special_chars()
    test_prompt_formatting_realistic()
    test_prompt_no_unescaped_braces()
    
    print("=" * 80)
    print("✅ All tests passed!")
