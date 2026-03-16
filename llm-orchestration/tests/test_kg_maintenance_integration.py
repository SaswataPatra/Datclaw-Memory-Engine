"""
End-to-End Integration Test: KG Maintenance Agent
Tests the full flow from memory creation to KG correction.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-orchestration"))

from arango import ArangoClient
from services.consolidation_service import ConsolidationService
from core.knowledge_graph_store import KnowledgeGraphStore
from services.kg_maintenance_agent import KGMaintenanceAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ARANGO_HOST = os.getenv("ARANGODB_HOST", "http://localhost:8529")
ARANGO_DB = os.getenv("ARANGODB_DATABASE", "dappy_dev")
ARANGO_USER = os.getenv("ARANGODB_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGODB_PASSWORD", "dappy_dev_password")


async def test_full_flow():
    """
    Test the complete flow:
    1. User says father's name (incorrect)
    2. Relations stored in KG
    3. User corrects father's name
    4. KG Maintenance Agent detects contradiction
    5. Old relation removed, new relation kept
    6. Verify KG is clean
    """
    logger.info("="*80)
    logger.info("INTEGRATION TEST: KG Maintenance - Full Flow")
    logger.info("="*80)
    
    # Setup
    logger.info("\n🔧 Setting up services...")
    
    arango_client = ArangoClient(hosts=ARANGO_HOST)
    arango_db = arango_client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)
    
    kg_store = KnowledgeGraphStore(db=arango_db)
    consolidation_service = ConsolidationService(api_key=OPENAI_API_KEY)
    kg_maintenance_agent = KGMaintenanceAgent(knowledge_graph_store=kg_store, api_key=OPENAI_API_KEY)
    
    test_user_id = "test_user_kg_maintenance"
    
    # Clean up any existing test data
    logger.info(f"   Cleaning up test data for user: {test_user_id}")
    try:
        arango_db.aql.execute(
            "FOR rel IN entity_relations FILTER rel.user_id == @user_id REMOVE rel IN entity_relations",
            bind_vars={"user_id": test_user_id}
        )
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")
    
    logger.info("   ✅ Services ready\n")
    
    # Step 1: User provides incorrect father's name
    logger.info("="*80)
    logger.info("STEP 1: User provides father's name (incorrect)")
    logger.info("="*80)
    
    memory1_text = "My father's name is Arun Kumar Patra"
    memory1_id = "mem_test_001"
    
    logger.info(f"   Memory: {memory1_text}")
    
    result1 = await consolidation_service.consolidate(
        text=memory1_text,
        ego_score=0.8,
        tier=1
    )
    
    logger.info(f"   Extracted entities: {result1['entities']}")
    logger.info(f"   Extracted relations: {result1['relations']}")
    
    if result1['relations']:
        stored = kg_store.store_relations(
            user_id=test_user_id,
            memory_id=memory1_id,
            relations=result1['relations']
        )
        logger.info(f"   ✅ Stored {stored} relations in KG")
    
    # Verify KG state
    kg_relations = kg_store.get_user_relations(user_id=test_user_id, limit=10)
    logger.info(f"\n   📊 KG State: {len(kg_relations)} relations")
    for rel in kg_relations:
        logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0):.2f})")
    
    # Step 2: User corrects father's name
    logger.info("\n" + "="*80)
    logger.info("STEP 2: User corrects father's name")
    logger.info("="*80)
    
    memory2_text = "No, my father's name is just Arun Patra, not Arun Kumar Patra"
    memory2_id = "mem_test_002"
    
    logger.info(f"   Memory: {memory2_text}")
    
    result2 = await consolidation_service.consolidate(
        text=memory2_text,
        ego_score=0.85,
        tier=1
    )
    
    logger.info(f"   Extracted entities: {result2['entities']}")
    logger.info(f"   Extracted relations: {result2['relations']}")
    
    if result2['relations']:
        stored = kg_store.store_relations(
            user_id=test_user_id,
            memory_id=memory2_id,
            relations=result2['relations']
        )
        logger.info(f"   ✅ Stored {stored} relations in KG")
    
    # Verify KG state BEFORE maintenance
    kg_relations_before = kg_store.get_user_relations(user_id=test_user_id, limit=10)
    logger.info(f"\n   📊 KG State BEFORE Maintenance: {len(kg_relations_before)} relations")
    for rel in kg_relations_before:
        logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0):.2f})")
    
    # Step 3: Run KG Maintenance Agent
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Run KG Maintenance Agent")
    logger.info("="*80)
    
    maintenance_result = await kg_maintenance_agent.process_memory(
        user_id=test_user_id,
        memory_id=memory2_id,
        memory_content=memory2_text,
        new_relations=result2['relations'],
        ego_score=0.85
    )
    
    logger.info(f"\n   📈 Maintenance Result:")
    logger.info(f"      Contradictions found: {maintenance_result['contradictions_found']}")
    logger.info(f"      Relations updated: {maintenance_result['relations_updated']}")
    logger.info(f"      Relations removed: {maintenance_result['relations_removed']}")
    
    # Step 4: Verify KG is clean
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Verify KG is Clean")
    logger.info("="*80)
    
    kg_relations_after = kg_store.get_user_relations(user_id=test_user_id, limit=10)
    logger.info(f"\n   📊 KG State AFTER Maintenance: {len(kg_relations_after)} relations")
    for rel in kg_relations_after:
        logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel.get('confidence', 0):.2f})")
    
    # Validation
    logger.info("\n" + "="*80)
    logger.info("VALIDATION")
    logger.info("="*80)
    
    # Check that only correct father name remains
    father_relations = [
        rel for rel in kg_relations_after
        if rel['predicate'] in ['has_father', 'father_of']
    ]
    
    logger.info(f"\n   Father relations remaining: {len(father_relations)}")
    
    if len(father_relations) == 1:
        father_rel = father_relations[0]
        if 'arun patra' in father_rel['object'].lower() and 'kumar' not in father_rel['object'].lower():
            logger.info("   ✅ PASS: Only correct father name remains (Arun Patra)")
            logger.info(f"      Relation: {father_rel['subject']} --{father_rel['predicate']}--> {father_rel['object']}")
            return True
        elif 'arun patra' in father_rel['subject'].lower() and 'kumar' not in father_rel['subject'].lower():
            logger.info("   ✅ PASS: Only correct father name remains (Arun Patra)")
            logger.info(f"      Relation: {father_rel['subject']} --{father_rel['predicate']}--> {father_rel['object']}")
            return True
        else:
            logger.error(f"   ❌ FAIL: Wrong father name remains: {father_rel}")
            return False
    elif len(father_relations) == 0:
        logger.error("   ❌ FAIL: No father relation found (should have kept the correct one)")
        return False
    else:
        logger.error(f"   ❌ FAIL: Multiple father relations remain (should be cleaned up)")
        for rel in father_relations:
            logger.error(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']}")
        return False


async def main():
    """Run the integration test."""
    logger.info("🚀 Starting KG Maintenance Integration Test...\n")
    
    success = await test_full_flow()
    
    logger.info("\n" + "="*80)
    if success:
        logger.info("✅ INTEGRATION TEST PASSED")
        logger.info("   - Contradiction detected")
        logger.info("   - Old relation removed")
        logger.info("   - KG is clean and accurate")
    else:
        logger.error("❌ INTEGRATION TEST FAILED")
    logger.info("="*80)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
