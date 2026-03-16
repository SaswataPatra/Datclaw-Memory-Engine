"""
End-to-end test for stable architecture:
- Vector search (primary retrieval)
- Knowledge graph relations (LLM context)
- Single consolidation LLM call (entities + relations with confidence)
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-orchestration"))

from arango import ArangoClient
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from core.knowledge_graph_store import KnowledgeGraphStore
from services.consolidation_service import ConsolidationService
from services.embedding_service import EmbeddingService
from services.context_manager import ContextMemoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

TEST_USER_ID = "test_stable_user"
ARANGO_URL = "http://localhost:8529"
ARANGO_DB = "dappy"
ARANGO_USER = "root"
ARANGO_PASSWORD = "dappy_dev_password"  # From docker-compose
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = "http://localhost:6333"


async def cleanup(arango_db, qdrant_client):
    """Clean up test data."""
    logger.info("🧹 Cleaning up test data...")
    
    # Delete test memories from ArangoDB
    try:
        query = "FOR m IN memories FILTER m.user_id == @user_id REMOVE m IN memories"
        arango_db.aql.execute(query, bind_vars={"user_id": TEST_USER_ID})
        logger.info("   Deleted test memories from ArangoDB")
    except Exception as e:
        logger.warning(f"   ArangoDB cleanup warning: {e}")
    
    # Delete test relations from ArangoDB
    try:
        query = "FOR r IN entity_relations FILTER r.user_id == @user_id REMOVE r IN entity_relations"
        arango_db.aql.execute(query, bind_vars={"user_id": TEST_USER_ID})
        logger.info("   Deleted test relations from ArangoDB")
    except Exception as e:
        logger.warning(f"   ArangoDB relations cleanup warning: {e}")
    
    # Delete test vectors from Qdrant
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qdrant_client.delete(
            collection_name="memories",
            points_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=TEST_USER_ID))]
            )
        )
        logger.info("   Deleted test vectors from Qdrant")
    except Exception as e:
        logger.warning(f"   Qdrant cleanup warning: {e}")


async def test_ingestion(consolidation_service, kg_store, embedding_service, arango_db, qdrant_client):
    """Test memory ingestion with consolidation."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Memory Ingestion with Consolidation")
    logger.info("="*80)
    
    test_memory = """
    I'm working on a liquidation bot project. It's a trading bot that monitors 
    cryptocurrency markets for liquidation events and executes trades automatically.
    The bot is built with Python and uses the Binance API.
    """
    
    # Step 1: Consolidate (extract entities + relations)
    logger.info("\n📝 Step 1: Consolidating memory...")
    result = await consolidation_service.consolidate(
        text=test_memory,
        ego_score=0.8,
        tier=1
    )
    
    entities = result.get("entities", [])
    relations = result.get("relations", [])
    
    logger.info(f"   Extracted {len(entities)} entities:")
    for ent in entities:
        logger.info(f"      - {ent['name']} ({ent['type']})")
    
    logger.info(f"   Extracted {len(relations)} relations:")
    for rel in relations:
        logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel['confidence']:.2f})")
    
    # Step 2: Store memory in ArangoDB
    logger.info("\n💾 Step 2: Storing memory in ArangoDB...")
    memory_id = f"mem_test_{int(datetime.now().timestamp())}"
    
    # Create memories collection if it doesn't exist
    if not arango_db.has_collection("memories"):
        arango_db.create_collection("memories")
        logger.info("   Created 'memories' collection")
    
    memories_collection = arango_db.collection("memories")
    memories_collection.insert({
        "_key": memory_id,
        "user_id": TEST_USER_ID,
        "content": test_memory,
        "ego_score": 0.8,
        "tier": 1,
        "created_at": datetime.now().isoformat(),
        "metadata": {"source": "test"}
    })
    logger.info(f"   Stored memory: {memory_id}")
    
    # Step 3: Store relations in KG
    logger.info("\n🕸️  Step 3: Storing relations in Knowledge Graph...")
    logger.info(f"   KG store object: {kg_store}")
    logger.info(f"   KG store _relations_collection: {kg_store._relations_collection}")
    
    # Re-initialize if needed
    if kg_store._relations_collection is None:
        logger.warning("   KG store not initialized, re-initializing...")
        kg_store._init_collection()
    
    stored_count = kg_store.store_relations(
        user_id=TEST_USER_ID,
        memory_id=memory_id,
        relations=relations
    )
    logger.info(f"   Stored {stored_count}/{len(relations)} relations")
    
    # Step 4: Store embedding in Qdrant
    logger.info("\n🔢 Step 4: Storing embedding in Qdrant...")
    embedding = await embedding_service.generate(test_memory)
    from qdrant_client.models import PointStruct
    qdrant_client.upsert(
        collection_name="memories",
        points=[
            PointStruct(
                id=memory_id,
                vector=embedding,
                payload={
                    "user_id": TEST_USER_ID,
                    "content": test_memory,
                    "ego_score": 0.8,
                    "tier": 1
                }
            )
        ]
    )
    logger.info(f"   Stored embedding for {memory_id}")
    
    return memory_id


async def test_retrieval(kg_store, embedding_service, arango_db, qdrant_client):
    """Test retrieval with vector search + KG relations."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Retrieval with Vector Search + KG Relations")
    logger.info("="*80)
    
    query = "What projects am I working on?"
    
    # Step 1: Vector search
    logger.info(f"\n🔍 Step 1: Vector search for: '{query}'")
    query_embedding = await embedding_service.generate(query)
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    search_results = qdrant_client.search(
        collection_name="memories",
        query_vector=query_embedding,
        query_filter=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=TEST_USER_ID))]
        ),
        limit=5
    )
    
    logger.info(f"   Found {len(search_results)} memories:")
    for hit in search_results:
        content = hit.payload.get("content", "")[:100]
        logger.info(f"      - Score: {hit.score:.3f} | {content}...")
    
    # Step 2: Get KG relations
    logger.info("\n🕸️  Step 2: Fetching Knowledge Graph relations...")
    relations = kg_store.get_user_relations(
        user_id=TEST_USER_ID,
        entity_names=None,  # Get all relations
        limit=50
    )
    
    logger.info(f"   Found {len(relations)} relations:")
    for rel in relations[:10]:  # Show first 10
        logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel['confidence']:.2f})")
    
    # Step 3: Format relations for LLM context
    logger.info("\n📋 Step 3: Formatting relations for LLM context...")
    if relations:
        lines = ["Known relationships:"]
        for rel in relations:
            subject = rel['subject']
            predicate = rel['predicate'].replace('_', ' ')
            obj = rel['object']
            confidence = rel.get('confidence', 0.5)
            lines.append(f"  - {subject} {predicate} {obj} (confidence: {confidence:.2f})")
        
        relation_context = "\n".join(lines)
        logger.info(f"   Relation context:\n{relation_context}")
    else:
        logger.warning("   No relations found!")
    
    return len(search_results) > 0 and len(relations) > 0


async def main():
    """Run end-to-end test."""
    logger.info("🚀 Starting stable architecture test...\n")
    
    # Test ArangoDB connection
    logger.info("🔌 Testing ArangoDB connection...")
    logger.info(f"   URL: {ARANGO_URL}")
    logger.info(f"   Database: {ARANGO_DB}")
    logger.info(f"   Username: {ARANGO_USER}")
    logger.info(f"   Password: {'***' if ARANGO_PASSWORD else '(empty)'}")
    
    try:
        arango_client = ArangoClient(hosts=ARANGO_URL)
        
        # Try to connect to system database first
        sys_db = arango_client.db('_system', username=ARANGO_USER, password=ARANGO_PASSWORD)
        logger.info("   ✅ Connected to _system database")
        
        # List available databases
        databases = sys_db.databases()
        logger.info(f"   Available databases: {databases}")
        
        # Check if target database exists
        if ARANGO_DB not in databases:
            logger.error(f"   ❌ Database '{ARANGO_DB}' does not exist!")
            logger.info(f"   Creating database '{ARANGO_DB}'...")
            sys_db.create_database(ARANGO_DB)
            logger.info(f"   ✅ Database '{ARANGO_DB}' created")
        
        # Connect to target database
        arango_db = arango_client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASSWORD)
        logger.info(f"   ✅ Connected to '{ARANGO_DB}' database")
        
        # List collections
        collections = arango_db.collections()
        collection_names = [c['name'] for c in collections if not c['name'].startswith('_')]
        logger.info(f"   Collections: {collection_names}")
        
    except Exception as e:
        logger.error(f"   ❌ ArangoDB connection failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    # Test Qdrant connection
    logger.info("\n🔌 Testing Qdrant connection...")
    logger.info(f"   URL: {QDRANT_URL}")
    
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        collections = qdrant_client.get_collections()
        logger.info(f"   ✅ Connected to Qdrant")
        logger.info(f"   Collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"   ❌ Qdrant connection failed: {e}")
        return
    
    # Initialize services
    consolidation_service = ConsolidationService(api_key=OPENAI_API_KEY)
    embedding_service = EmbeddingService(api_key=OPENAI_API_KEY)
    
    # Initialize KG store (will create entity_relations collection if needed)
    kg_store = KnowledgeGraphStore(db=arango_db)
    
    try:
        # Cleanup before test
        await cleanup(arango_db, qdrant_client)
        
        # Test 1: Ingestion
        memory_id = await test_ingestion(
            consolidation_service=consolidation_service,
            kg_store=kg_store,
            embedding_service=embedding_service,
            arango_db=arango_db,
            qdrant_client=qdrant_client
        )
        
        # Test 2: Retrieval
        success = await test_retrieval(
            kg_store=kg_store,
            embedding_service=embedding_service,
            arango_db=arango_db,
            qdrant_client=qdrant_client
        )
        
        # Final verdict
        logger.info("\n" + "="*80)
        if success:
            logger.info("✅ TEST PASSED: Stable architecture working correctly!")
            logger.info("   - Consolidation: entities + relations extracted")
            logger.info("   - Storage: relations stored with confidence scores")
            logger.info("   - Retrieval: vector search + KG relations retrieved")
        else:
            logger.error("❌ TEST FAILED: Some components not working")
        logger.info("="*80)
        
    finally:
        # Cleanup after test
        await cleanup(arango_db, qdrant_client)
        logger.info("\n✅ Test cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
