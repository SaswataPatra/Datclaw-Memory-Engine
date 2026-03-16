"""
Test KG Maintenance Agent
Verifies contradiction detection and relation updates.
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
    
    def get_user_relations(self, user_id: str, limit: int = 200):
        return self.relations


async def test_contradiction_detection():
    """Test that the agent detects contradictions."""
    logger.info("="*80)
    logger.info("TEST: KG Maintenance Agent - Contradiction Detection")
    logger.info("="*80)
    
    mock_kg = MockKGStore()
    agent = KGMaintenanceAgent(knowledge_graph_store=mock_kg, api_key=OPENAI_API_KEY)
    
    # Scenario: User corrects father's name
    logger.info("\n📝 Scenario: Father name correction")
    logger.info("   Old: 'my fathers name is arun kumar patra'")
    logger.info("   New: 'no my fathers name is just arun patra'")
    
    # Existing relations in KG
    mock_kg.relations = [
        {
            "subject": "arun kumar patra",
            "predicate": "is_a",
            "object": "father",
            "confidence": 0.80
        },
        {
            "subject": "sunita patra",
            "predicate": "is_a",
            "object": "mother",
            "confidence": 0.80
        }
    ]
    
    # New memory with correction
    new_memory = "No my fathers name is not arun kumar patra its just Arun Patra"
    new_relations = [
        {
            "subject": "arun patra",
            "predicate": "is_a",
            "object": "father",
            "confidence": 0.70
        }
    ]
    
    # Analyze
    logger.info("\n🔍 Running contradiction detection...")
    analysis = await agent._analyze_relations(
        memory_content=new_memory,
        new_relations=new_relations,
        existing_relations=mock_kg.relations
    )
    
    contradictions = analysis.get("contradictions", [])
    reinforcements = analysis.get("reinforcements", [])
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Contradictions: {len(contradictions)}")
    for c in contradictions:
        logger.info(f"      Type: {c.get('type')}")
        logger.info(f"      Old: {c['old_relation']['subject']} --{c['old_relation']['predicate']}--> {c['old_relation']['object']}")
        logger.info(f"      New: {c['new_relation']['subject']} --{c['new_relation']['predicate']}--> {c['new_relation']['object']}")
        logger.info(f"      Resolution: {c.get('resolution')}")
        logger.info(f"      Reasoning: {c.get('reasoning')}")
    
    logger.info(f"   Reinforcements: {len(reinforcements)}")
    for r in reinforcements:
        rel = r['relation']
        logger.info(f"      {rel['subject']} --{rel['predicate']}--> {rel['object']}")
        logger.info(f"      Reasoning: {r.get('reasoning')}")
    
    # Validate
    if len(contradictions) > 0:
        logger.info("\n✅ PASS: Detected contradiction in father name")
        
        # Check if mother relation is reinforced (not contradicted)
        mother_reinforced = any(
            r['relation']['subject'] == 'sunita patra' and r['relation']['object'] == 'mother'
            for r in reinforcements
        )
        
        if mother_reinforced:
            logger.info("✅ PASS: Mother relation correctly identified as reinforcement (not contradiction)")
        else:
            logger.info("⚠️  NOTE: Mother relation not reinforced (acceptable - no new mention)")
        
        return True
    else:
        logger.error("❌ FAIL: Should have detected father name contradiction")
        return False


async def test_reinforcement_detection():
    """Test that the agent detects reinforcements."""
    logger.info("\n" + "="*80)
    logger.info("TEST: KG Maintenance Agent - Reinforcement Detection")
    logger.info("="*80)
    
    mock_kg = MockKGStore()
    agent = KGMaintenanceAgent(knowledge_graph_store=mock_kg, api_key=OPENAI_API_KEY)
    
    # Scenario: User mentions existing project again
    logger.info("\n📝 Scenario: Re-mentioning existing project")
    logger.info("   Old: 'working on liquidation bot'")
    logger.info("   New: 'the liquidation bot project is going well'")
    
    mock_kg.relations = [
        {
            "subject": "user",
            "predicate": "works_on",
            "object": "liquidation bot",
            "confidence": 0.80
        }
    ]
    
    new_memory = "The liquidation bot project is going well, making good progress"
    new_relations = [
        {
            "subject": "user",
            "predicate": "works_on",
            "object": "liquidation bot",
            "confidence": 0.85
        }
    ]
    
    logger.info("\n🔍 Running reinforcement detection...")
    analysis = await agent._analyze_relations(
        memory_content=new_memory,
        new_relations=new_relations,
        existing_relations=mock_kg.relations
    )
    
    contradictions = analysis.get("contradictions", [])
    reinforcements = analysis.get("reinforcements", [])
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Contradictions: {len(contradictions)}")
    logger.info(f"   Reinforcements: {len(reinforcements)}")
    
    for r in reinforcements:
        rel = r['relation']
        logger.info(f"      {rel['subject']} --{rel['predicate']}--> {rel['object']}")
        logger.info(f"      Reasoning: {r.get('reasoning')}")
    
    if len(reinforcements) > 0 and len(contradictions) == 0:
        logger.info("\n✅ PASS: Detected reinforcement, no false contradiction")
        return True
    else:
        logger.error("❌ FAIL: Should have detected reinforcement")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting KG Maintenance Agent tests...\n")
    
    test1 = await test_contradiction_detection()
    test2 = await test_reinforcement_detection()
    
    logger.info("\n" + "="*80)
    if test1 and test2:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("   - Contradiction detection working")
        logger.info("   - Reinforcement detection working")
    else:
        logger.error("❌ SOME TESTS FAILED")
    logger.info("="*80)
    
    return test1 and test2


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
