"""
Test KG Maintenance Agent with implicit corrections.
Tests that the agent can detect corrections even when no new relations are extracted.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-orchestration"))

from services.kg_maintenance_agent import KGMaintenanceAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class MockKGStore:
    """Mock KG store for testing."""
    
    def __init__(self):
        self.db = None
        self.relations = []
        self.added_relations = []
    
    def get_user_relations(self, user_id: str, limit: int = 200):
        return self.relations
    
    def store_relations(self, user_id: str, memory_id: str, relations: list):
        self.added_relations.extend(relations)
        return len(relations)


async def test_implicit_correction():
    """Test that the agent detects corrections even without explicit relations."""
    logger.info("="*80)
    logger.info("TEST: KG Maintenance - Implicit Correction Detection")
    logger.info("="*80)
    
    mock_kg = MockKGStore()
    agent = KGMaintenanceAgent(knowledge_graph_store=mock_kg, api_key=OPENAI_API_KEY)
    
    # Scenario: User corrects father's name with short statement
    logger.info("\n📝 Scenario: Implicit correction")
    logger.info("   Existing KG: user --has_father--> albida patra")
    logger.info("   User says: 'No it is Arun Kumar Patra'")
    logger.info("   Context: Previous conversation was about father's name")
    
    # Existing KG
    mock_kg.relations = [
        {
            "subject": "user",
            "predicate": "has_father",
            "object": "albida patra",
            "confidence": 0.80
        },
        {
            "subject": "user",
            "predicate": "has_mother",
            "object": "saswati patra",
            "confidence": 0.80
        }
    ]
    
    # New memory: correction without explicit relation
    new_memory = "No it is Arun Kumar Patra"
    new_relations = []  # Consolidation extracted no relations (incomplete context)
    
    # Analyze
    logger.info("\n🔍 Running KG maintenance...")
    result = await agent.process_memory(
        user_id="test_user",
        memory_id="mem_correction",
        memory_content=new_memory,
        new_relations=new_relations,
        ego_score=0.85
    )
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Contradictions: {result['contradictions_found']}")
    logger.info(f"   Relations added: {result['relations_added']}")
    logger.info(f"   Relations updated: {result['relations_updated']}")
    logger.info(f"   Relations removed: {result['relations_removed']}")
    
    # Validate
    if result['contradictions_found'] > 0 or result['relations_added'] > 0:
        logger.info("\n✅ PASS: Agent detected the correction and proposed changes")
        logger.info(f"   Added relations: {mock_kg.added_relations}")
        return True
    else:
        logger.warning("\n⚠️  Agent did not detect correction (may need more context)")
        return False


async def test_question_no_changes():
    """Test that questions don't trigger unnecessary changes."""
    logger.info("\n" + "="*80)
    logger.info("TEST: KG Maintenance - Question (No Changes)")
    logger.info("="*80)
    
    mock_kg = MockKGStore()
    agent = KGMaintenanceAgent(knowledge_graph_store=mock_kg, api_key=OPENAI_API_KEY)
    
    # Scenario: User asks question about existing knowledge
    logger.info("\n📝 Scenario: Question about existing knowledge")
    logger.info("   Existing KG: user --has_father--> arun patra")
    logger.info("   User asks: 'What is my father's name?'")
    
    mock_kg.relations = [
        {
            "subject": "user",
            "predicate": "has_father",
            "object": "arun patra",
            "confidence": 0.90
        }
    ]
    
    new_memory = "What is my father's name?"
    new_relations = []
    
    logger.info("\n🔍 Running KG maintenance...")
    result = await agent.process_memory(
        user_id="test_user",
        memory_id="mem_question",
        memory_content=new_memory,
        new_relations=new_relations,
        ego_score=0.70
    )
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Contradictions: {result['contradictions_found']}")
    logger.info(f"   Relations added: {result['relations_added']}")
    logger.info(f"   Relations updated: {result['relations_updated']}")
    logger.info(f"   Relations removed: {result['relations_removed']}")
    
    if result['contradictions_found'] == 0 and result['relations_added'] == 0:
        logger.info("\n✅ PASS: Agent correctly identified no changes needed for question")
        return True
    else:
        logger.warning("\n⚠️  Agent proposed changes for a simple question (may be overzealous)")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting KG Maintenance - Implicit Correction Tests...\n")
    
    test1 = await test_implicit_correction()
    test2 = await test_question_no_changes()
    
    logger.info("\n" + "="*80)
    if test1 and test2:
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.info("⚠️  SOME TESTS HAD UNEXPECTED RESULTS (review above)")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
