#!/usr/bin/env python3
"""
Test PPR Integration with Chatbot
Verifies that PPR retrieval works end-to-end in chat conversations
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noise
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from arango import ArangoClient
from qdrant_client import QdrantClient
import redis.asyncio as redis

from config import load_config
from core.event_bus import EventBusFactory
from core.scoring.ego_scorer import TemporalEgoScorer
from services.context_manager import ContextMemoryManager
from adapters.redis_message_bus import RedisMessageBus
from core.shadow_tier import ShadowTier
from llm.providers.factory import LLMProviderFactory
from services.chatbot_service import ChatbotService


async def test_ppr_chatbot_integration():
    """Test PPR retrieval in chatbot conversations"""
    
    print("=" * 100)
    print("🧪 PPR CHATBOT INTEGRATION TEST")
    print("=" * 100)
    print()
    
    # Load config
    config = load_config()
    
    # Initialize Redis
    print("📡 Connecting to Redis...")
    redis_config = config.get_section('redis')
    redis_client = redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        db=redis_config['db'],
        password=redis_config.get('password'),
        decode_responses=True
    )
    await redis_client.ping()
    print("✅ Redis connected")
    
    # Initialize ArangoDB
    print("📡 Connecting to ArangoDB...")
    arango_config = config.get_section('arangodb')
    arango_client = ArangoClient(hosts=arango_config['url'])
    arango_db = arango_client.db(
        arango_config['database'],
        username=arango_config['username'],
        password=arango_config['password']
    )
    print("✅ ArangoDB connected")
    
    # Initialize Qdrant
    print("📡 Connecting to Qdrant...")
    qdrant_config = config.get_section('qdrant')
    qdrant_client = QdrantClient(url=qdrant_config['url'])
    print("✅ Qdrant connected")
    
    # Initialize Event Bus
    print("🔧 Initializing Event Bus...")
    event_bus_config = config.get_section('event_bus')
    event_bus = EventBusFactory.create(
        provider=event_bus_config['provider'],
        redis_client=redis_client,
        config=event_bus_config.get('redis', {})
    )
    print("✅ Event Bus initialized")
    
    # Initialize Ego Scorer
    print("🔧 Initializing Ego Scorer...")
    ego_scorer = TemporalEgoScorer(config.all)
    print("✅ Ego Scorer initialized")
    
    # Initialize Message Store
    print("🔧 Initializing Message Store...")
    message_storage_config = config.get_section('message_storage')
    message_store = RedisMessageBus(redis_client, message_storage_config['hot'])
    print("✅ Message Store initialized")
    
    # Initialize Context Manager with PPR
    print("🔧 Initializing Context Manager with PPR support...")
    context_manager = ContextMemoryManager(
        redis_client=redis_client,
        config=config.all,
        ego_scorer=ego_scorer,
        event_bus=event_bus,
        message_store=message_store,
        arango_db=arango_db,
        qdrant_client=qdrant_client
    )
    print("✅ Context Manager initialized")
    
    if context_manager.ppr_retrieval:
        print("   ✅ PPR retrieval is available")
    else:
        print("   ⚠️  PPR retrieval is NOT available")
    
    if context_manager.nlp:
        print("   ✅ spaCy is loaded for entity extraction")
    else:
        print("   ⚠️  spaCy is NOT loaded")
    
    # Initialize Shadow Tier
    print("🔧 Initializing Shadow Tier...")
    shadow_tier = ShadowTier(
        redis_client=redis_client,
        config=config.all,
        event_bus=event_bus
    )
    print("✅ Shadow Tier initialized")
    
    # Initialize LLM Provider
    print("🔧 Initializing LLM Provider...")
    llm_config = config.get_section('llm')
    llm_provider = LLMProviderFactory.from_config(llm_config)
    print(f"✅ LLM Provider initialized: {llm_provider.name}")
    
    # Initialize Chatbot Service
    print("🔧 Initializing Chatbot Service...")
    chatbot_config = config.get_section('chatbot')
    chatbot_service = ChatbotService(
        llm_provider=llm_provider,
        context_manager=context_manager,
        ego_scorer=ego_scorer,
        event_bus=event_bus,
        shadow_tier=shadow_tier,
        system_prompt=chatbot_config.get('system_prompt'),
        config=config.all,
        qdrant_client=qdrant_client,
        use_ml_scoring=False,  # Disable for faster testing
        use_graph_pipeline=False,  # Disable graph extraction for this test
        arango_db=arango_db
    )
    print("✅ Chatbot Service initialized")
    
    print()
    print("=" * 100)
    print("STEP 1: Check Graph Data")
    print("=" * 100)
    print()
    
    # Check what entities exist
    entities_query = """
    FOR entity IN entities
        FILTER entity.user_id == @user_id
        LIMIT 10
        RETURN {
            entity_id: entity._key,
            name: entity.canonical_name,
            type: entity.type
        }
    """
    
    entities_cursor = arango_db.aql.execute(entities_query, bind_vars={'user_id': 'saswata'})
    entities = list(entities_cursor)
    
    print(f"Found {len(entities)} entities for user 'saswata':")
    for i, ent in enumerate(entities, 1):
        print(f"  {i}. {ent['name']} ({ent['entity_id']}) - {ent['type']}")
    print()
    
    if not entities:
        print("❌ No entities found! Load some data first using load_locomo_data.py")
        return
    
    print()
    print("=" * 100)
    print("STEP 2: Test Entity Extraction from Query")
    print("=" * 100)
    print()
    
    # Test queries
    test_queries = [
        "Tell me about Caroline",
        "What do you know about Melanie?",
        "Tell me about music and family",
        "What happened at the support group?"
    ]
    
    for query in test_queries:
        print(f"Query: \"{query}\"")
        extracted = await context_manager._extract_entities_from_query(query)
        print(f"  Extracted entities: {extracted}")
        
        # Resolve to entity IDs
        entity_ids = await context_manager._resolve_entity_ids('saswata', extracted)
        print(f"  Resolved to {len(entity_ids)} entity IDs: {entity_ids[:3]}")
        print()
    
    print()
    print("=" * 100)
    print("STEP 3: Test Memory Retrieval")
    print("=" * 100)
    print()
    
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"🔍 Retrieving memories for: \"{query}\"")
        print()
        
        memories = await context_manager.retrieve_relevant_memories(
            user_id='saswata',
            query=query,
            max_memories=3,
            use_ppr=True,
            use_vector=False
        )
        
        if memories:
            print(f"✅ Found {len(memories)} relevant memories:")
            for i, mem in enumerate(memories, 1):
                content_preview = mem['content'][:100] + "..." if len(mem['content']) > 100 else mem['content']
                print(f"\n  {i}. Memory ID: {mem['memory_id']}")
                print(f"     Ego Score: {mem.get('ego_score', 0):.2f}")
                print(f"     Tier: {mem.get('tier', 'N/A')}")
                print(f"     Source: {mem.get('source', 'N/A')}")
                print(f"     Content: {content_preview}")
        else:
            print("  ⚠️  No memories found")
        
        print()
        print("-" * 100)
        print()
    
    print()
    print("=" * 100)
    print("STEP 4: Test Full Chat with PPR")
    print("=" * 100)
    print()
    
    # Test a chat interaction
    test_message = "Tell me what you know about Caroline"
    print(f"💬 User: {test_message}")
    print()
    
    try:
        response = await chatbot_service.chat(
            user_id='saswata',
            session_id='test_ppr_session',
            user_message=test_message,
            conversation_history=[],
            debug=True
        )
        
        print(f"🤖 Assistant: {response['assistant_message']}")
        print()
        
        if 'debug' in response:
            debug = response['debug']
            print("📊 Debug Info:")
            print(f"  - Optimized history length: {debug.get('optimized_history_length', 0)}")
            print(f"  - LLM messages sent: {len(debug.get('llm_messages', []))}")
            
            # Check if memories were included
            llm_messages = debug.get('llm_messages', [])
            memory_context_found = False
            for msg in llm_messages:
                if 'Relevant Context from Memory' in msg.get('content', ''):
                    memory_context_found = True
                    print(f"  - ✅ Memory context was included in LLM prompt")
                    break
            
            if not memory_context_found:
                print(f"  - ⚠️  No memory context found in LLM prompt")
        
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 100)
    print("✅ PPR CHATBOT INTEGRATION TEST COMPLETE")
    print("=" * 100)
    print()
    
    # Cleanup
    await redis_client.close()
    await event_bus.close()


if __name__ == "__main__":
    asyncio.run(test_ppr_chatbot_integration())
