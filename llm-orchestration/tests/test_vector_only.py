"""
Test: Vector Search Only (No Entity/Relations)

Tests if pure vector search on memory content is sufficient.

Run: python test_vector_only.py
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test")


async def main():
    from arango import ArangoClient
    from qdrant_client import QdrantClient
    from config import load_config
    from services.embedding_service import EmbeddingService
    from services.context_manager import ContextMemoryManager
    from core.scoring.ego_scorer import TemporalEgoScorer
    from adapters.redis_message_bus import RedisMessageBus
    from core.event_bus import RedisEventBus
    import redis

    cfg = load_config()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    # Initialize services
    arango_config = cfg.get_section('arangodb')
    arango_client = ArangoClient(hosts=arango_config['url'])
    db = arango_client.db(
        arango_config['database'],
        username=arango_config['username'],
        password=arango_config['password']
    )

    qdrant_config = cfg.get_section('qdrant')
    qdrant_client = QdrantClient(url=qdrant_config['url'])

    redis_config = cfg.get_section('redis')
    redis_client = redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        decode_responses=True
    )

    embedding_service = EmbeddingService(api_key=openai_api_key)
    ego_scorer = TemporalEgoScorer(config=cfg.all)
    
    event_bus_config = cfg.get_section('event_bus')
    event_bus = RedisEventBus(
        redis_client=redis_client,
        config=event_bus_config.get('redis', {})
    )
    
    message_storage_config = cfg.get_section('message_storage')
    message_store = RedisMessageBus(redis_client, message_storage_config['hot'])

    # Context manager with ONLY vector search (no entity/relation stuff)
    context_manager = ContextMemoryManager(
        redis_client=redis_client,
        config=cfg.all,
        ego_scorer=ego_scorer,
        event_bus=event_bus,
        message_store=message_store,
        arango_db=db,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        query_understanding_service=None,  # Disabled
        entity_memory_linker=None,  # Disabled
        entity_store=None  # Disabled
    )

    user_id = "test_vector_only"

    # Test memories
    memories = [
        ("mem_v1", "I'm working on the DAPPY project, an AI memory system for personal knowledge", 0.85, 1),
        ("mem_v2", "I recently started building a liquidation BOT for crypto trading on Ethereum", 0.87, 1),
        ("mem_v3", "The liquidation bot is a side project I'm really passionate about", 0.80, 1),
        ("mem_v4", "My sister Sarah lives in Boston and works as a professional photographer", 0.75, 2),
        ("mem_v5", "I'm also maintaining an SDK for API integration with third-party services", 0.70, 2),
        ("mem_v6", "I love playing guitar in my free time", 0.60, 2),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: Ingest Memories (Vector Search Only - No Entity/Relation Extraction)")
    print("=" * 80)

    for mem_id, content, ego, tier in memories:
        # Store in ArangoDB
        mem_doc = {
            "_key": mem_id,
            "user_id": user_id,
            "content": content,
            "ego_score": ego,
            "tier": tier,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        db.collection("memories").insert(mem_doc, overwrite=True)

        # Generate embedding and store in Qdrant
        embedding = await embedding_service.generate(content)
        if embedding:
            from qdrant_client.models import PointStruct
            import uuid
            qdrant_client.upsert(
                collection_name="memories",
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "node_id": mem_id,
                            "user_id": user_id,
                            "tier": tier,
                            "ego_score": ego,
                        },
                    )
                ],
            )
        print(f"  {mem_id}: {content[:60]}...")

    print("\n" + "=" * 80)
    print("STEP 2: Test Retrieval with Various Queries")
    print("=" * 80)

    test_cases = [
        ("what projects am I working on", ["dappy", "liquidation bot", "sdk"]),
        ("what bots am I building", ["liquidation bot"]),
        ("tell me about my sister", ["sarah"]),
        ("what things am I working on", ["dappy", "liquidation bot", "sdk"]),
        ("what am I building", ["dappy", "liquidation bot"]),
        ("tell me about my hobbies", ["guitar"]),
    ]

    for query, expected_keywords in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Query: '{query}'")
        print(f"Expected keywords: {expected_keywords}")
        print('─' * 80)
        
        results = await context_manager.retrieve_relevant_memories(
            user_id=user_id,
            query=query,
            max_memories=5,
            use_ppr=False,
            use_vector=True
        )

        print(f"Retrieved: {len(results)} memories\n")
        
        found_keywords = []
        for i, mem in enumerate(results, 1):
            print(f"  {i}. [{mem['memory_id']}] ego={mem.get('ego_score', 0):.2f}, rel={mem.get('relevance_score', 0):.2f}")
            print(f"     {mem['content'][:70]}...")
            
            # Check if expected keywords are in content
            content_lower = mem['content'].lower()
            for keyword in expected_keywords:
                if keyword.lower() in content_lower:
                    found_keywords.append(keyword)
        
        # Check coverage
        found_unique = list(set(found_keywords))
        missing = [k for k in expected_keywords if k not in found_unique]
        
        if missing:
            print(f"\n  ⚠️  Missing expected keywords: {missing}")
        else:
            print(f"\n  ✅ All expected keywords found")

    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    db.aql.execute("FOR m IN memories FILTER m.user_id == @uid REMOVE m IN memories", bind_vars={"uid": user_id})
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_client.delete(
        collection_name="memories",
        points_selector=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    )
    print("  Cleaned up")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
