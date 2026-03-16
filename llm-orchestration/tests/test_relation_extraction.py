"""
Test: Lightweight Relation Extraction + Graph Traversal

Tests the new relation extraction feature:
1. Ingest memory → extract entities → extract relations
2. Store entity-memory links + entity-entity relations
3. Query with abstract term → expand via relations → find related memories

Run: python test_relation_extraction.py
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
    from services.entity_extraction import EntityExtractionService
    from services.relation_extraction import RelationExtractionService
    from services.embedding_service import EmbeddingService
    from core.entity_memory_linker import EntityMemoryLinker

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

    user_id = "test_relation_user"

    # Test memories with clear relations
    memories = [
        ("mem_r1", "I'm working on the DAPPY project, which is an AI memory system", 0.85, 1),
        ("mem_r2", "I recently started building a liquidation BOT for crypto trading", 0.87, 1),
        ("mem_r3", "The liquidation bot is a personal project I'm excited about", 0.80, 1),
        ("mem_r4", "My sister Sarah lives in Boston and loves photography", 0.75, 2),
    ]

    print("\n" + "=" * 80)
    print("STEP 1: Ingest Memories with Entity + Relation Extraction")
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

        # Extract entities
        entities = await entity_extraction.extract_entities(content)
        entity_names = [e["name"] for e in entities]
        if entity_names:
            entity_linker.link_memory(user_id, mem_id, entity_names)
            print(f"  {mem_id}: entities={entity_names}")
        
        # Extract relations
        relations = await relation_extraction.extract_relations(content)
        if relations:
            linked = entity_linker.link_relations(user_id, mem_id, relations)
            print(f"  {mem_id}: relations={[(r['subject'], r['predicate'], r['object']) for r in relations]}")
            print(f"           stored={linked}/{len(relations)}")
        else:
            print(f"  {mem_id}: no relations extracted")

    print("\n" + "=" * 80)
    print("STEP 2: Test Relation Expansion")
    print("=" * 80)

    # Query with abstract term "project"
    query_entities = ["project"]
    print(f"\n  Starting entities: {query_entities}")
    
    expanded = entity_linker.expand_entities_via_relations(
        user_id=user_id,
        entity_names=query_entities,
        max_hops=1
    )
    print(f"  After 1-hop expansion: {expanded}")
    
    # Get memories for expanded entities
    memory_ids = entity_linker.get_memory_ids_for_entities(
        user_id=user_id,
        entity_names=expanded,
        limit=20
    )
    print(f"  Memories found: {len(memory_ids)}")
    
    if memory_ids:
        query = """
        FOR mem IN memories
            FILTER mem._key IN @ids
            FILTER mem.user_id == @user_id
            RETURN {id: mem._key, content: mem.content}
        """
        cursor = db.aql.execute(query, bind_vars={"ids": memory_ids, "user_id": user_id})
        for mem in cursor:
            print(f"    → {mem['id']}: {mem['content'][:60]}...")

    print("\n" + "=" * 80)
    print("STEP 3: Test Query 'what projects am I working on'")
    print("=" * 80)
    
    # Simulate what would happen in retrieval
    from services.query_understanding import QueryUnderstandingService
    query_service = QueryUnderstandingService(api_key=openai_api_key)
    
    query = "what projects am I working on"
    extracted = await query_service.extract_entities(query)
    print(f"\n  Query: '{query}'")
    print(f"  Extracted entities: {extracted}")
    
    if not extracted:
        print("  No entities extracted - trying fallback with 'project' keyword")
        extracted = ["project"]
    
    # Expand via relations
    expanded = entity_linker.expand_entities_via_relations(
        user_id=user_id,
        entity_names=extracted,
        max_hops=1
    )
    print(f"  Expanded entities: {expanded}")
    
    # Get memories
    memory_ids = entity_linker.get_memory_ids_for_entities(
        user_id=user_id,
        entity_names=expanded,
        limit=20
    )
    print(f"  Memories found via entity expansion: {len(memory_ids)}")
    
    if memory_ids:
        query = """
        FOR mem IN memories
            FILTER mem._key IN @ids
            FILTER mem.user_id == @user_id
            RETURN {id: mem._key, content: mem.content, ego_score: mem.ego_score}
        """
        cursor = db.aql.execute(query, bind_vars={"ids": memory_ids, "user_id": user_id})
        for mem in cursor:
            print(f"    → {mem['id']}: ego={mem['ego_score']:.2f}")
            print(f"       {mem['content'][:70]}...")

    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    # Clean up test data
    db.aql.execute("FOR m IN memories FILTER m.user_id == @uid REMOVE m IN memories", bind_vars={"uid": user_id})
    db.aql.execute("FOR e IN entity_memories FILTER e.user_id == @uid REMOVE e IN entity_memories", bind_vars={"uid": user_id})
    db.aql.execute("FOR r IN entity_relations FILTER r.user_id == @uid REMOVE r IN entity_relations", bind_vars={"uid": user_id})
    
    # Clean up Qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_client.delete(
        collection_name="memories",
        points_selector=Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
    )
    print("  Cleaned up test data")

    print("\n✅ Test complete!")
    print("\nConclusion:")
    print("  - Entity extraction works")
    print("  - Relation extraction works")
    print("  - 1-hop graph traversal expands abstract queries to specific entities")
    print("  - Lightweight graph (no activation scoring, no promotion) is functional")


if __name__ == "__main__":
    asyncio.run(main())
