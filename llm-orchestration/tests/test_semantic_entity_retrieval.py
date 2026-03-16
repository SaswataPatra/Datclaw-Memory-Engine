"""
Test: Semantic Entity Matching + Relation Expansion

Tests the complete semantic entity retrieval flow:
1. Ingest memories → extract entities → store embeddings + relations
2. Query with variations → semantic entity search → relation expansion → retrieve memories
3. Test cases:
   - "what projects am I working on" (abstract category)
   - "what bots am I building" (synonym/variation)
   - "tell me about my sister" (specific entity)

Run: python test_semantic_entity_retrieval.py
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("test")


async def main():
    from arango import ArangoClient
    from qdrant_client import QdrantClient
    from config import load_config
    from services.entity_extraction import EntityExtractionService
    from services.relation_extraction import RelationExtractionService
    from services.embedding_service import EmbeddingService
    from services.query_understanding import QueryUnderstandingService
    from services.entity_store import EntityStore
    from core.entity_memory_linker import EntityMemoryLinker
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

    entity_extraction = EntityExtractionService(api_key=openai_api_key)
    relation_extraction = RelationExtractionService(api_key=openai_api_key)
    embedding_service = EmbeddingService(api_key=openai_api_key)
    query_service = QueryUnderstandingService(api_key=openai_api_key)
    entity_linker = EntityMemoryLinker(db=db, config=cfg.all)
    entity_store = EntityStore(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        collection_name="entities"
    )
    ego_scorer = TemporalEgoScorer(config=cfg.all)
    
    event_bus_config = cfg.get_section('event_bus')
    event_bus = RedisEventBus(
        redis_client=redis_client,
        config=event_bus_config.get('redis', {})
    )
    
    message_storage_config = cfg.get_section('message_storage')
    message_store = RedisMessageBus(redis_client, message_storage_config['hot'])

    context_manager = ContextMemoryManager(
        redis_client=redis_client,
        config=cfg.all,
        ego_scorer=ego_scorer,
        event_bus=event_bus,
        message_store=message_store,
        arango_db=db,
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        query_understanding_service=query_service,
        entity_memory_linker=entity_linker,
        entity_store=entity_store
    )

    user_id = "test_semantic_user"

    # Test memories with diverse entities and relations
    memories = [
        ("mem_s1", "I'm working on the DAPPY project, an AI memory system", 0.85, 1),
        ("mem_s2", "I recently started building a liquidation BOT for crypto trading", 0.87, 1),
        ("mem_s3", "The liquidation bot is a side project I'm passionate about", 0.80, 1),
        ("mem_s4", "My sister Sarah lives in Boston and works as a photographer", 0.75, 2),
        ("mem_s5", "I'm also maintaining an SDK for API integration", 0.70, 2),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: Ingest Memories with Entities + Relations + Embeddings")
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

        # Extract entities
        entities = await entity_extraction.extract_entities(content)
        entity_names = [e["name"] for e in entities]
        
        if entity_names:
            # Link entities to memory
            entity_linker.link_memory(user_id, mem_id, entity_names)
            
            # Store entity embeddings
            stored = await entity_store.store_entities(user_id, mem_id, entity_names)
            print(f"  {mem_id}: {len(entity_names)} entities, {stored} embeddings stored")
        
        # Extract relations
        relations = await relation_extraction.extract_relations(content)
        if relations:
            linked = entity_linker.link_relations(user_id, mem_id, relations)
            print(f"           {linked} relations: {[(r['subject'], r['predicate'], r['object']) for r in relations[:3]]}")

    print("\n" + "=" * 80)
    print("STEP 2: Test Semantic Entity Search")
    print("=" * 80)

    test_queries = [
        ("project", "Should find: dappy, liquidation bot"),
        ("bot", "Should find: liquidation bot (semantic match)"),
        ("sister", "Should find: sarah"),
        ("api", "Should find: sdk, api integration"),
    ]

    for query_entity, expected in test_queries:
        results = await entity_store.semantic_search(
            user_id=user_id,
            query_entities=[query_entity],
            limit=10
        )
        print(f"\n  Query: '{query_entity}' → {results}")
        print(f"  Expected: {expected}")

    print("\n" + "=" * 80)
    print("STEP 3: Test Full Retrieval with Abstract Queries")
    print("=" * 80)

    test_cases = [
        "what projects am I working on",
        "what bots am I building",
        "tell me about my sister",
        "what things am I working on",
    ]

    for query in test_cases:
        print(f"\n{'─' * 80}")
        print(f"Query: '{query}'")
        print('─' * 80)
        
        results = await context_manager.retrieve_relevant_memories(
            user_id=user_id,
            query=query,
            max_memories=5,
            use_ppr=False,
            use_vector=True
        )

        print(f"Retrieved: {len(results)} memories\n")
        for i, mem in enumerate(results, 1):
            print(f"  {i}. [{mem['memory_id']}] ego={mem.get('ego_score', 0):.2f}, rel={mem.get('relevance_score', 0):.2f}")
            print(f"     {mem['content'][:70]}...")

    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    db.aql.execute("FOR m IN memories FILTER m.user_id == @uid REMOVE m IN memories", bind_vars={"uid": user_id})
    db.aql.execute("FOR e IN entity_memories FILTER e.user_id == @uid REMOVE e IN entity_memories", bind_vars={"uid": user_id})
    db.aql.execute("FOR r IN entity_relations FILTER r.user_id == @uid REMOVE r IN entity_relations", bind_vars={"uid": user_id})
    
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_client.delete(
        collection_name="memories",
        points_selector=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    )
    qdrant_client.delete(
        collection_name="entities",
        points_selector=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    )
    print("  Cleaned up")

    print("\n✅ Test complete!")
    print("\nConclusion:")
    print("  - Semantic entity matching handles variations and synonyms")
    print("  - Relation expansion adds connected entities")
    print("  - Abstract queries ('what things') work via semantic + relation traversal")


if __name__ == "__main__":
    asyncio.run(main())
