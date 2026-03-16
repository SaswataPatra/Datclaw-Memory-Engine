"""
Acceptance tests for dependency-based extraction.
Tests with 30+ real-world examples covering all sentence types.
"""

import pytest
import asyncio
from unittest.mock import Mock
from core.graph.relation_extractor import RelationExtractor


# Test cases: (input_text, expected_relations)
# Each expected relation: (subject, relation_contains, object)
ACCEPTANCE_CASES = [
    # ===== SIMPLE SVO =====
    (
        "Sarah studies physics",
        [("Sarah", "stud", "physics")]
    ),
    (
        "John loves programming",
        [("John", "love", "programming")]
    ),
    (
        "The student reads books",
        [("student", "read", "books")]
    ),
    
    # ===== PREPOSITIONAL =====
    (
        "Sarah lives in Boston",
        [("Sarah", "live", "Boston")]
    ),
    (
        "John works at Google",
        [("John", "work", "Google")]
    ),
    (
        "The company is based in California",
        [("company", "based", "California")]
    ),
    (
        "She graduated from MIT",
        [("She", "graduat", "MIT")]
    ),
    
    # ===== COPULA/ATTRIBUTIVE =====
    (
        "Sarah is my sister",
        [("Sarah", "", "sister")]
    ),
    (
        "John is a doctor",
        [("John", "", "doctor")]
    ),
    (
        "The CEO is Mark",
        [("CEO", "", "Mark")]
    ),
    
    # ===== APPOSITION =====
    (
        "Mark, the CEO, spoke yesterday",
        [("Mark", "", "CEO")]
    ),
    (
        "Sarah, my sister, studies physics",
        [("Sarah", "", "sister"), ("Sarah", "stud", "physics")]
    ),
    (
        "John Smith, our founder, retired",
        [("John Smith", "", "founder")]
    ),
    
    # ===== PASSIVE VOICE =====
    (
        "Physics is studied by Sarah",
        [("Sarah", "stud", "Physics")]
    ),
    (
        "The book was written by the author",
        [("author", "writ", "book")]
    ),
    (
        "The company was founded by Mark",
        [("Mark", "found", "company")]
    ),
    
    # ===== COORDINATION =====
    (
        "Sarah and John study physics",
        [("Sarah", "stud", "physics"), ("John", "stud", "physics")]
    ),
    (
        "She studies physics and chemistry",
        [("She", "stud", "physics"), ("She", "stud", "chemistry")]
    ),
    (
        "Mark and Sarah work at Google and Apple",
        [("Mark", "work", "Google"), ("Mark", "work", "Apple"),
         ("Sarah", "work", "Google"), ("Sarah", "work", "Apple")]
    ),
    
    # ===== NESTED CLAUSES =====
    (
        "I know that Sarah studies physics",
        [("Sarah", "stud", "physics")]
    ),
    (
        "She said that John works at Google",
        [("John", "work", "Google")]
    ),
    (
        "I believe that Mark is the CEO",
        [("Mark", "", "CEO")]
    ),
    
    # ===== WITH MODALITY =====
    (
        "Sarah might study physics",
        [("Sarah", "stud", "physics")]
    ),
    (
        "John will work at Google",
        [("John", "work", "Google")]
    ),
    (
        "She should visit Boston",
        [("She", "visit", "Boston")]
    ),
    
    # ===== WITH NEGATION =====
    (
        "Sarah does not study physics",
        [("Sarah", "stud", "physics")]
    ),
    (
        "John never works on weekends",
        [("John", "work", "weekends")]
    ),
    
    # ===== COMPLEX SENTENCES =====
    (
        "Sarah, my sister, studies physics at MIT and works at Google.",
        [("Sarah", "", "sister"), ("Sarah", "stud", "physics"),
         ("Sarah", "stud", "MIT"), ("Sarah", "work", "Google")]
    ),
    (
        "John, who graduated from Stanford, works at Apple in California.",
        [("John", "graduat", "Stanford"), ("John", "work", "Apple")]
    ),
    (
        "The CEO announced that the company will expand to Europe.",
        [("company", "expand", "Europe")]
    ),
    
    # ===== TEMPORAL =====
    (
        "Sarah started working at Google in 2020",
        [("Sarah", "work", "Google")]
    ),
    (
        "John graduated from MIT last year",
        [("John", "graduat", "MIT")]
    ),
    
    # ===== POSSESSIVE =====
    (
        "Sarah's sister lives in Boston",
        [("sister", "live", "Boston")]
    ),
    (
        "John's company is based in California",
        [("company", "based", "California")]
    ),
    
    # ===== QUESTIONS =====
    (
        "Does Sarah study physics?",
        [("Sarah", "stud", "physics")]
    ),
    (
        "Where does John work?",
        []  # May not extract cleanly
    ),
]


@pytest.fixture
def mock_db():
    """Mock ArangoDB database."""
    db = Mock()
    db.collection = Mock(return_value=Mock())
    return db


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.generate = Mock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def config():
    """Test configuration."""
    return {
        "dependency_extraction": {
            "enabled": True
        },
        "coref": {
            "provider": "simple",
            "model": "en_core_web_sm"
        }
    }


@pytest.fixture
def relation_extractor(mock_db, mock_embedding_service, config):
    """Create RelationExtractor with mocked dependencies."""
    extractor = RelationExtractor(
        db=mock_db,
        config=config,
        embedding_service=mock_embedding_service,
        collect_training_data=False
    )
    
    # Mock entity resolver
    async def mock_resolve(text, user_id, context, entity_type=None):
        entity = Mock()
        entity.entity_id = f"e_{text.lower().replace(' ', '_')}"
        entity.canonical_name = text
        entity.type = entity_type or "unknown"
        return entity
    
    extractor.entity_resolver.resolve = mock_resolve
    
    return extractor


@pytest.mark.parametrize("text,expected", ACCEPTANCE_CASES)
@pytest.mark.asyncio
async def test_acceptance(text, expected, relation_extractor):
    """
    Acceptance test for each case.
    
    Args:
        text: Input text
        expected: List of (subject, relation_contains, object) tuples
        relation_extractor: Fixture
    """
    relations = await relation_extractor.extract(
        text=text,
        user_id="test_user",
        memory_id="mem_test"
    )
    
    # If no expected relations, just check it doesn't crash
    if not expected:
        assert isinstance(relations, list)
        return
    
    # Check that all expected relations are present
    for exp_subj, exp_rel, exp_obj in expected:
        found = False
        for rel in relations:
            subj_match = exp_subj.lower() in rel.subject_text.lower()
            obj_match = exp_obj.lower() in rel.object_text.lower()
            rel_match = (not exp_rel) or (exp_rel.lower() in rel.relation.lower())
            
            if subj_match and obj_match and rel_match:
                found = True
                break
        
        assert found, f"Missing relation: ({exp_subj}, {exp_rel}, {exp_obj}) in text: '{text}'"


class TestCoverageByType:
    """Test coverage for each sentence type."""
    
    @pytest.mark.asyncio
    async def test_simple_declarative_coverage(self, relation_extractor):
        """Test: Simple declarative sentences (40% of real text)"""
        test_cases = [
            "Sarah studies physics",
            "John loves programming",
            "The student reads books",
            "Mark writes code",
            "She teaches mathematics"
        ]
        
        total = len(test_cases)
        extracted = 0
        
        for text in test_cases:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            if len(relations) >= 1:
                extracted += 1
        
        coverage = extracted / total
        assert coverage >= 0.90, f"Simple declarative coverage: {coverage:.2%} (target: 95%)"
    
    @pytest.mark.asyncio
    async def test_prepositional_coverage(self, relation_extractor):
        """Test: Prepositional relations (20% of real text)"""
        test_cases = [
            "Sarah lives in Boston",
            "John works at Google",
            "She graduated from MIT",
            "The company is based in California",
            "He traveled to Europe"
        ]
        
        total = len(test_cases)
        extracted = 0
        
        for text in test_cases:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            if len(relations) >= 1:
                extracted += 1
        
        coverage = extracted / total
        assert coverage >= 0.85, f"Prepositional coverage: {coverage:.2%} (target: 90%)"
    
    @pytest.mark.asyncio
    async def test_passive_voice_coverage(self, relation_extractor):
        """Test: Passive voice (10% of real text)"""
        test_cases = [
            "Physics is studied by Sarah",
            "The book was written by the author",
            "The company was founded by Mark",
            "The project is managed by John",
            "The code was reviewed by the team"
        ]
        
        total = len(test_cases)
        extracted = 0
        
        for text in test_cases:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            if len(relations) >= 1:
                extracted += 1
        
        coverage = extracted / total
        assert coverage >= 0.80, f"Passive voice coverage: {coverage:.2%} (target: 85%)"
    
    @pytest.mark.asyncio
    async def test_coordination_coverage(self, relation_extractor):
        """Test: Coordination (10% of real text)"""
        test_cases = [
            "Sarah and John study physics",
            "She studies physics and chemistry",
            "Mark and Sarah work at Google",
            "John and Mary live in Boston",
            "The CEO and CFO resigned"
        ]
        
        total = len(test_cases)
        extracted = 0
        
        for text in test_cases:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            # Should extract multiple relations
            if len(relations) >= 2:
                extracted += 1
        
        coverage = extracted / total
        assert coverage >= 0.85, f"Coordination coverage: {coverage:.2%} (target: 90%)"
    
    @pytest.mark.asyncio
    async def test_complex_nested_coverage(self, relation_extractor):
        """Test: Complex/nested sentences (10% of real text)"""
        test_cases = [
            "I know that Sarah studies physics",
            "She said that John works at Google",
            "I believe that Mark is the CEO",
            "He thinks that she lives in Boston",
            "They announced that the company will expand"
        ]
        
        total = len(test_cases)
        extracted = 0
        
        for text in test_cases:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            if len(relations) >= 1:
                extracted += 1
        
        coverage = extracted / total
        assert coverage >= 0.80, f"Complex/nested coverage: {coverage:.2%} (target: 85%)"


class TestRealWorldExamples:
    """Test with real-world conversational examples."""
    
    @pytest.mark.asyncio
    async def test_personal_introduction(self, relation_extractor):
        """Test: Personal introduction"""
        text = "Hi, I'm Sarah. I'm a student at MIT studying computer science."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract: (Sarah, student_at, MIT), (Sarah, studies, computer science)
        assert len(relations) >= 1
    
    @pytest.mark.asyncio
    async def test_family_description(self, relation_extractor):
        """Test: Family description"""
        text = "My sister Sarah lives in Boston with her husband John."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract multiple family/location relations
        assert len(relations) >= 1
    
    @pytest.mark.asyncio
    async def test_work_history(self, relation_extractor):
        """Test: Work history"""
        text = "I worked at Google for 5 years before joining Apple."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract work relations
        assert len(relations) >= 1
    
    @pytest.mark.asyncio
    async def test_education_background(self, relation_extractor):
        """Test: Education background"""
        text = "I graduated from Stanford with a degree in physics."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract education relations
        assert len(relations) >= 1
    
    @pytest.mark.asyncio
    async def test_hobby_interests(self, relation_extractor):
        """Test: Hobbies and interests"""
        text = "I love playing guitar and reading science fiction books."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract preference relations
        assert len(relations) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_very_short_text(self, relation_extractor):
        """Test: Very short text"""
        text = "Sarah."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should handle gracefully
        assert isinstance(relations, list)
    
    @pytest.mark.asyncio
    async def test_very_long_sentence(self, relation_extractor):
        """Test: Very long sentence"""
        text = "Sarah, who graduated from MIT with honors in computer science, works at Google in California where she leads a team of engineers developing machine learning models for natural language processing."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract multiple relations
        assert len(relations) >= 2
    
    @pytest.mark.asyncio
    async def test_special_characters(self, relation_extractor):
        """Test: Special characters"""
        text = "Sarah works at Google! She loves it :)"
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should handle special characters
        assert isinstance(relations, list)
    
    @pytest.mark.asyncio
    async def test_numbers_and_dates(self, relation_extractor):
        """Test: Numbers and dates"""
        text = "Sarah started working at Google in 2020."
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Should extract work relation
        assert len(relations) >= 1


class TestConfidenceScores:
    """Test confidence scoring."""
    
    @pytest.mark.asyncio
    async def test_high_confidence_simple(self, relation_extractor):
        """Test: High confidence for simple sentences"""
        text = "Sarah studies physics"
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        assert len(relations) >= 1
        assert relations[0].confidence >= 0.70
    
    @pytest.mark.asyncio
    async def test_lower_confidence_complex(self, relation_extractor):
        """Test: Lower confidence for complex sentences"""
        text = "I think Sarah might study physics"
        relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # May have lower confidence due to nesting + modality
        if relations:
            # Just check it's a valid confidence score
            assert 0.0 <= relations[0].confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

