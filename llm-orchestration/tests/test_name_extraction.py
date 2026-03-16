"""
Test script to verify "my name is X" extraction and retrieval works.
"""

import asyncio
import logging
from arango import ArangoClient
from qdrant_client import QdrantClient
from config import load_config
from core.graph.arango_integration import create_graph_pipeline
from services.embedding_service import EmbeddingService
from llm.providers.factory import LLMProviderFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_name_extraction():
    """Test that 'my name is X' properly extracts entities and creates relations."""
    
    print("\n" + "="*70)
    print("🧪 Testing Name Extraction & Retrieval")
    print("="*70 + "\n")
    
    # Load config
    config = load_config()
    
    # Initialize services
    arango_config = config.get_section('arangodb')
    arango_client = ArangoClient(hosts=arango_config['url'])
    arango_db = arango_client.db(
        arango_config['database'],
        username=arango_config['username'],
        password=arango_config['password']
    )
    
    qdrant_config = config.get_section('qdrant')
    qdrant_client = QdrantClient(url=qdrant_config['url'])
    
    llm_config = config.get_section('llm')
    llm_provider = LLMProviderFactory.from_config(llm_config)
    
    openai_api_key = llm_config.get('openai', {}).get('api_key')
    embedding_service = EmbeddingService(api_key=openai_api_key) if openai_api_key else None
    
    # Create graph pipeline
    graph_pipeline = create_graph_pipeline(
        config=config.all,
        embedding_service=embedding_service,
        llm_provider=llm_provider
    )
    
    # Test user
    user_id = "test_user_name_extraction"
    
    # Test 1: Extract from "my name is saswata"
    print("\n📝 Test 1: Processing 'my name is saswata'")
    print("-" * 70)
    
    result = await graph_pipeline.process_memory(
        user_id=user_id,
        memory_id="test_memory_name_1",
        content="my name is saswata",
        ego_score=0.8,
        tier=2,
        session_id="test_session_1",
        metadata={}
    )
    
    print(f"\n✅ Extraction complete:")
    print(f"   Entities: {len(result.get('entities', []))}")
    print(f"   Relations: {len(result.get('relations', []))}")
    print(f"   Edges: {len(result.get('edges', []))}")
    
    if result.get('entities'):
        print(f"\n   📍 Entities extracted:")
        for ent in result['entities']:
            print(f"      • {ent.get('canonical_name')} (type: {ent.get('type')})")
    
    if result.get('relations'):
        print(f"\n   🔗 Relations extracted:")
        for rel in result['relations']:
            print(f"      • {rel.get('subject_text')} --[{rel.get('predicate')}]--> {rel.get('object_text')}")
    
    # Test 2: Query entities
    print("\n\n📊 Test 2: Querying entities in graph")
    print("-" * 70)
    
    entities_query = """
    FOR entity IN entities
        FILTER entity.user_id == @user_id
        RETURN {
            id: entity._key,
            name: entity.canonical_name,
            type: entity.type,
            aliases: entity.aliases
        }
    """
    
    cursor = arango_db.aql.execute(entities_query, bind_vars={'user_id': user_id})
    entities = list(cursor)
    
    print(f"\n   Found {len(entities)} entities:")
    for ent in entities:
        print(f"      • {ent['name']} (type: {ent['type']}, id: {ent['id']})")
    
    # Test 3: Query edges
    print("\n\n🔗 Test 3: Querying edges in graph")
    print("-" * 70)
    
    edges_query = """
    FOR edge IN candidate_edges
        FILTER edge.user_id == @user_id
        LET subject = DOCUMENT(CONCAT('entities/', edge.subject_entity_id))
        LET object = DOCUMENT(CONCAT('entities/', edge.object_entity_id))
        RETURN {
            subject: subject.canonical_name,
            predicate: edge.predicate,
            object: object.canonical_name,
            confidence: edge.confidence
        }
    """
    
    cursor = arango_db.aql.execute(edges_query, bind_vars={'user_id': user_id})
    edges = list(cursor)
    
    print(f"\n   Found {len(edges)} edges:")
    for edge in edges:
        print(f"      • {edge['subject']} --[{edge['predicate']}]--> {edge['object']} (conf: {edge['confidence']:.2f})")
    
    # Test 4: Test retrieval
    print("\n\n🔍 Test 4: Testing retrieval with 'do you remember my name?'")
    print("-" * 70)
    
    from services.context_manager import ContextMemoryManager
    from core.scoring.ego_scorer import TemporalEgoScorer
    from core.event_bus import EventBusFactory
    import redis.asyncio as redis
    
    redis_config = config.get_section('redis')
    redis_client = redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        db=redis_config['db'],
        decode_responses=True
    )
    
    ego_scorer = TemporalEgoScorer(config.all)
    event_bus = EventBusFactory.create('redis', redis_client, redis_config)
    
    from adapters.redis_message_bus import RedisMessageBus
    message_store = RedisMessageBus(redis_client, config.get_section('message_storage')['hot'])
    
    context_manager = ContextMemoryManager(
        redis_client=redis_client,
        config=config.all,
        ego_scorer=ego_scorer,
        event_bus=event_bus,
        message_store=message_store,
        arango_db=arango_db,
        qdrant_client=qdrant_client
    )
    
    memories = await context_manager.retrieve_relevant_memories(
        user_id=user_id,
        query="do you remember my name?",
        max_memories=5
    )
    
    print(f"\n   Retrieved {len(memories)} memories:")
    for mem in memories:
        print(f"      • [{mem.get('source', 'unknown')}] {mem['content'][:60]}... (ego: {mem.get('ego_score', 0):.2f})")
    
    # Cleanup
    await redis_client.close()
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    asyncio.run(test_name_extraction())
