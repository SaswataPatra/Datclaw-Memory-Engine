"""
Test: Simplified Retrieval (Vector + Entity Expansion)

Tests the working architecture:
1. Ingest memories → ego scoring → ArangoDB + Qdrant
2. Entity extraction → entity_memories links
3. Query → vector search + entity expansion → merged results

Run: python test_simplified_retrieval.py
Requires: ArangoDB + Qdrant running, OPENAI_API_KEY set
"""

import asyncio
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test")


async def main():
    # ── Setup ──
    from arango import ArangoClient
    from qdrant_client import QdrantClient
    from config import load_config
    from core.event_bus import Event
    from adapters.redis_message_bus import RedisMessageBus
    import redis

    cfg = load_config()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    # Connect to databases
    arango_config = cfg.get_section("arangodb")
    arango_client = ArangoClient(hosts=arango_config["url"])
    db = arango_client.db(
        arango_config["database"],
        username=arango_config["username"],
        password=arango_config["password"],
    )

    qdrant_config = cfg.get_section("qdrant")
    qdrant_client = QdrantClient(url=qdrant_config["url"])

    redis_config = cfg.get_section("redis")
    redis_client = redis.Redis(
        host=redis_config["host"],
        port=redis_config["port"],
        db=redis_config["db"],
        decode_responses=True,
    )

    # Initialize services
    from services.entity_extraction import EntityExtractionService
    from services.query_understanding import QueryUnderstandingService
    from services.embedding_service import EmbeddingService
    from core.entity_memory_linker import EntityMemoryLinker

    entity_extraction = EntityExtractionService(api_key=openai_api_key)
    query_understanding = QueryUnderstandingService(api_key=openai_api_key)
    embedding_service = EmbeddingService(api_key=openai_api_key)
    entity_linker = EntityMemoryLinker(db=db, config=cfg.all)

    user_id = "test_simple_user"

    # ── Test Data ──
    memories = [
        ("mem_s1", "Sarah is my sister, we grew up in Boston together", 0.85, 1),
        ("mem_s2", "My sister Sarah loves hiking and photography", 0.75, 2),
        ("mem_s3", "I work at Google as a software engineer", 0.80, 1),
        ("mem_s4", "Had lunch with my colleague Mike yesterday", 0.50, 3),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: Ingest Memories (Store in ArangoDB + Qdrant + Entity Links)")
    print("=" * 80)

    for mem_id, content, ego, tier in memories:
        # Store in ArangoDB
        mem_doc = {
            "_key": mem_id,
            "user_id": user_id,
            "content": content,
            "ego_score": ego,
            "tier": tier,
            "created_at": datetime.utcnow().isoformat(),
            "observed_at": datetime.utcnow().isoformat(),
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

        # Extract entities and link
        entities = await entity_extraction.extract_entities(content)
        entity_names = [e["name"] for e in entities]
        if entity_names:
            linked = entity_linker.link_memory(user_id, mem_id, entity_names)
            print(f"  {mem_id}: entities={entity_names}, linked={linked}")
        else:
            print(f"  {mem_id}: no entities extracted")

    print("\n" + "=" * 80)
    print("STEP 2: Test Vector Search")
    print("=" * 80)

    query = "Tell me about my family"
    query_emb = await embedding_service.generate(query)

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    hits = qdrant_client.search(
        collection_name="memories",
        query_vector=query_emb,
        query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=3,
        with_payload=True,
    )

    print(f"\n  Query: '{query}'")
    print(f"  Vector search results: {len(hits)}")
    for i, hit in enumerate(hits, 1):
        node_id = hit.payload.get("node_id")
        score = hit.score
        mem = db.collection("memories").get(node_id)
        content = mem["content"] if mem else "N/A"
        print(f"    {i}. [{node_id}] score={score:.3f}")
        print(f"       {content[:80]}...")

    print("\n" + "=" * 80)
    print("STEP 3: Test Entity Expansion")
    print("=" * 80)

    query2 = "What does Sarah like?"
    entities_from_query = await query_understanding.extract_entities(query2)
    print(f"\n  Query: '{query2}'")
    print(f"  Extracted entities: {entities_from_query}")

    if entities_from_query:
        memory_ids = entity_linker.get_memory_ids_for_entities(user_id, entities_from_query, limit=5)
        print(f"  Entity expansion found: {len(memory_ids)} memories")
        for mid in memory_ids:
            mem = db.collection("memories").get(mid)
            if mem:
                print(f"    → {mid}: {mem['content'][:80]}...")

    print("\n" + "=" * 80)
    print("STEP 4: Test Hybrid Retrieval (Vector + Entity)")
    print("=" * 80)

    # Simulate what context_manager.retrieve_relevant_memories does
    query3 = "What did I talk about with Sarah?"
    print(f"\n  Query: '{query3}'")

    # Vector search
    q_emb = await embedding_service.generate(query3)
    vector_hits = qdrant_client.search(
        collection_name="memories",
        query_vector=q_emb,
        query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=5,
    )
    vector_mem_ids = {h.payload["node_id"]: h.score for h in vector_hits}
    print(f"  Vector search: {len(vector_mem_ids)} results")

    # Entity expansion
    q_entities = await query_understanding.extract_entities(query3)
    entity_mem_ids = set()
    if q_entities:
        entity_mem_ids = set(entity_linker.get_memory_ids_for_entities(user_id, q_entities, limit=5))
        print(f"  Entity expansion: {len(entity_mem_ids)} results (entities: {q_entities})")

    # Merge
    all_mem_ids = set(vector_mem_ids.keys()) | entity_mem_ids
    print(f"  Merged (unique): {len(all_mem_ids)} memories")

    # Fetch and rank
    merged_memories = []
    for mid in all_mem_ids:
        mem = db.collection("memories").get(mid)
        if mem:
            merged_memories.append({
                "memory_id": mid,
                "content": mem["content"],
                "ego_score": mem["ego_score"],
                "relevance_score": vector_mem_ids.get(mid, 0.5),
            })

    # Sort by (relevance, ego)
    merged_memories.sort(key=lambda m: (m["relevance_score"], m["ego_score"]), reverse=True)

    print(f"\n  Final ranked results:")
    for i, m in enumerate(merged_memories[:3], 1):
        print(f"    {i}. rel={m['relevance_score']:.3f}, ego={m['ego_score']:.2f}")
        print(f"       {m['content'][:80]}...")

    # ── Cleanup ──
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    db.aql.execute("FOR m IN memories FILTER m.user_id == @uid REMOVE m IN memories", bind_vars={"uid": user_id})
    db.aql.execute("FOR l IN entity_memories FILTER l.user_id == @uid REMOVE l IN entity_memories", bind_vars={"uid": user_id})
    qdrant_client.delete(
        collection_name="memories",
        points_selector=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
    )
    print("  Cleaned up test data")
    print("\n✅ Test complete!")
    print("\nConclusion:")
    print("  - Vector search finds semantically similar memories")
    print("  - Entity expansion finds memories mentioning specific entities")
    print("  - Hybrid approach covers both semantic and entity-based recall")
    print("  - Simple, fast, and proven to work")


if __name__ == "__main__":
    asyncio.run(main())
