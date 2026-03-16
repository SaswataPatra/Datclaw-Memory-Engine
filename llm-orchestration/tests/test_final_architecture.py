"""
Test: Final Architecture - Vector Search + Relation Context

Tests the complete flow:
1. Ingest memories → extract entities + relations
2. Query → vector search retrieves memories
3. Relations are fed to LLM as structured context
4. LLM uses both memories + relations to generate response

Run: python test_final_architecture.py
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
    from services.entity_extraction import EntityExtractionService
    from services.relation_extraction import RelationExtractionService
    from services.embedding_service import EmbeddingService
    from core.entity_memory_linker import EntityMemoryLinker
    from core.event_bus import Event

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

    entity_extraction = EntityExtractionService(api_key=openai_api_key)
    relation_extraction = RelationExtractionService(api_key=openai_api_key)
    embedding_service = EmbeddingService(api_key=openai_api_key)
    entity_linker = EntityMemoryLinker(db=db, config=cfg.all)

    user_id = "test_final_user"

    # Test memories
    memories = [
        ("mem_f1", "I'm working on the DAPPY project, an AI memory system", 0.85, 1),
        ("mem_f2", "I recently started building a liquidation BOT for crypto trading", 0.87, 1),
        ("mem_f3", "The liquidation bot is a side project I'm passionate about", 0.80, 1),
        ("mem_f4", "My sister Sarah lives in Boston and works as a photographer", 0.75, 2),
        ("mem_f5", "I'm maintaining an SDK for API integration", 0.70, 2),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: Ingest Memories with Entities + Relations")
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

        # Extract entities and relations
        entities = await entity_extraction.extract_entities(content)
        entity_names = [e["name"] for e in entities]
        
        if entity_names:
            entity_linker.link_memory(user_id, mem_id, entity_names)
        
        relations = await relation_extraction.extract_relations(content)
        if relations:
            linked = entity_linker.link_relations(user_id, mem_id, relations)
            print(f"  {mem_id}: {len(entity_names)} entities, {linked} relations")
            for rel in relations[:3]:
                print(f"           {rel['subject']} --{rel['predicate']}--> {rel['object']}")
        else:
            print(f"  {mem_id}: {len(entity_names)} entities, 0 relations")

    print("\n" + "=" * 80)
    print("STEP 2: Check Stored Relations")
    print("=" * 80)

    query = """
    FOR rel IN entity_relations
        FILTER rel.user_id == @user_id
        RETURN {subject: rel.subject, predicate: rel.predicate, object: rel.object}
    """
    relations = list(db.aql.execute(query, bind_vars={"user_id": user_id}))
    print(f"\nTotal relations stored: {len(relations)}")
    print("\nUser's knowledge graph:")
    for rel in relations:
        print(f"  - {rel['subject']} {rel['predicate'].replace('_', ' ')} {rel['object']}")

    print("\n" + "=" * 80)
    print("STEP 3: Test Relation Context Retrieval")
    print("=" * 80)

    from services.context_manager import ContextMemoryManager
    from core.scoring.ego_scorer import TemporalEgoScorer
    from adapters.redis_message_bus import RedisMessageBus
    from core.event_bus import RedisEventBus
    from services.query_understanding import QueryUnderstandingService
    import redis

    redis_config = cfg.get_section('redis')
    redis_client = redis.Redis(
        host=redis_config['host'],
        port=redis_config['port'],
        decode_responses=True
    )

    ego_scorer = TemporalEgoScorer(config=cfg.all)
    event_bus_config = cfg.get_section('event_bus')
    event_bus = RedisEventBus(
        redis_client=redis_client,
        config=event_bus_config.get('redis', {})
    )
    message_storage_config = cfg.get_section('message_storage')
    message_store = RedisMessageBus(redis_client, message_storage_config['hot'])
    query_service = QueryUnderstandingService(api_key=openai_api_key)

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
        entity_memory_linker=entity_linker
    )

    # Test getting relation context
    relation_context = context_manager.get_user_relations_context(user_id=user_id, limit=30)
    print("\nRelation context for LLM:")
    print(relation_context)

    print("\n" + "=" * 80)
    print("STEP 4: Test Full Retrieval with Relation Context")
    print("=" * 80)

    test_queries = [
        "what projects am I working on",
        "what things am I working on",
        "tell me about my sister",
    ]

    for query in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Query: '{query}'")
        print('─' * 80)
        
        # Retrieve memories
        memories = await context_manager.retrieve_relevant_memories(
            user_id=user_id,
            query=query,
            max_memories=5,
            use_ppr=False,
            use_vector=True
        )
        
        print(f"\nRetrieved {len(memories)} memories:")
        for i, mem in enumerate(memories, 1):
            print(f"  {i}. {mem['content'][:60]}...")
        
        # Get relation context
        relation_ctx = context_manager.get_user_relations_context(user_id=user_id, limit=30)
        print(f"\nRelation context would be fed to LLM:")
        print(relation_ctx[:200] + "..." if len(relation_ctx) > 200 else relation_ctx)

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
    print("  Cleaned up")

    print("\n✅ Test complete!")
    print("\n" + "=" * 80)
    print("ARCHITECTURE SUMMARY")
    print("=" * 80)
    print("""
Retrieval:
  1. Vector search on memory content (primary - handles all semantic matching)
  2. Entity expansion (secondary - for exact entity queries)

Response Generation:
  1. Retrieved memories fed to LLM
  2. User's relations fed as structured context
  3. LLM reasons with both to generate response

Future (side-channel):
  - LLM returns hidden payload with relation updates
  - Update supporting_mentions for existing relations
  - Create new candidate edges
  - Promote high-confidence relations to thought edges
    """)


if __name__ == "__main__":
    asyncio.run(main())
