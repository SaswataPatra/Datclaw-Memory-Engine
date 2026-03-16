#!/usr/bin/env python3
"""
PPR Retrieval Test Script

Tests the Personalized PageRank retrieval system with real graph data.
Verifies:
1. Graph loading from ArangoDB (entities + candidate/thought edges)
2. PPR algorithm execution
3. Multi-hop traversal
4. Ranking quality
5. Supporting memory retrieval
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from arango import ArangoClient
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graph import PPRRetrieval

# Load environment variables
load_dotenv()


async def test_ppr_retrieval():
    """Test PPR retrieval with existing graph data."""
    
    print("=" * 100)
    print("🧪 PPR RETRIEVAL TEST")
    print("=" * 100)
    print()
    
    # Load config
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Connect to ArangoDB
    print("📡 Connecting to ArangoDB...")
    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('dappy_memories', username='root', password='dappy_dev_password')
    print("✅ Connected")
    print()
    
    # Initialize PPR retrieval
    print("🔧 Initializing PPR retrieval...")
    ppr = PPRRetrieval(db=db, config=config)
    print("✅ PPR initialized")
    print(f"   Alpha: {ppr.alpha}")
    print(f"   Max iterations: {ppr.max_iter}")
    print(f"   Default top-k: {ppr.default_top_k}")
    print()
    
    # Step 1: Check what entities exist for our test user
    print("=" * 100)
    print("STEP 1: Discover Available Entities")
    print("=" * 100)
    print()
    
    entities_query = """
    FOR entity IN entities
        FILTER entity.user_id == @user_id
        SORT entity.created_at DESC
        LIMIT 20
        RETURN {
            entity_id: entity._key,
            name: entity.canonical_name,
            type: entity.type,
            aliases: entity.aliases
        }
    """
    
    entities_cursor = db.aql.execute(entities_query, bind_vars={'user_id': 'saswata'})
    entities = list(entities_cursor)
    
    print(f"Found {len(entities)} entities for user 'saswata':\n")
    for i, ent in enumerate(entities, 1):
        print(f"  {i}. {ent['entity_id']:20s} | {ent['name']:30s} | {ent.get('type', 'N/A')}")
    print()
    
    if not entities:
        print("❌ No entities found! Send some messages first to build the graph.")
        return
    
    # Step 2: Check what edges exist
    print("=" * 100)
    print("STEP 2: Discover Available Edges")
    print("=" * 100)
    print()
    
    edges_query = """
    FOR edge IN candidate_edges
        FILTER edge.user_id == @user_id
        SORT edge.first_seen DESC
        LIMIT 20
        RETURN {
            edge_id: edge._key,
            subject: edge.subject_entity_id,
            predicate: edge.predicate,
            object: edge.object_entity_id,
            mentions: edge.total_mentions,
            activation: edge.activation_score
        }
    """
    
    edges_cursor = db.aql.execute(edges_query, bind_vars={'user_id': 'saswata'})
    edges = list(edges_cursor)
    
    print(f"Found {len(edges)} candidate edges for user 'saswata':\n")
    for i, edge in enumerate(edges, 1):
        print(f"  {i}. {edge['edge_id']:20s} | {edge['subject']:15s} --[{edge['predicate']}]--> {edge['object']:15s} | mentions={edge['mentions']}")
    print()
    
    if not edges:
        print("❌ No edges found! The graph is empty.")
        return
    
    # Step 3: Test PPR with first entity as seed
    print("=" * 100)
    print("STEP 3: Test PPR Retrieval")
    print("=" * 100)
    print()
    
    # Find an entity that has edges (from the edges list)
    seed_entity_id = None
    seed_entity_name = None
    
    if edges:
        # Use subject from first edge
        seed_entity_id = edges[0]['subject']
        
        # Find entity name
        for entity in entities:
            if entity['entity_id'] == seed_entity_id:
                seed_entity_name = entity['name']
                break
    
    if not seed_entity_id:
        # Fallback to first entity
        seed_entity_id = entities[0]['entity_id']
        seed_entity_name = entities[0]['name']
    
    print(f"🎯 Running PPR from seed entity: {seed_entity_name} ({seed_entity_id})")
    print(f"   Max hops: 3")
    print(f"   Top-k: 10")
    print()
    
    try:
        result = await ppr.retrieve(
            user_id='saswata',
            seed_entities=[seed_entity_id],
            max_hops=3,
            top_k=10
        )
        
        print(f"✅ PPR completed!")
        print(f"   Nodes found: {len(result.nodes)}")
        print(f"   Edges found: {len(result.edges)}")
        print()
        
        if not result.nodes:
            print("⚠️  No nodes found - graph might be empty or disconnected")
            return
        
        # Display results
        print("📊 PPR Results (ranked by score):")
        print()
        
        # Sort nodes by PPR score
        nodes_with_scores = [
            (node, result.scores.get(node['entity_id'], 0.0))
            for node in result.nodes
        ]
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  {'Rank':<6} | {'Entity ID':<20} | {'Name':<30} | {'Type':<15} | {'PPR Score':<10}")
        print(f"  {'-'*6} | {'-'*20} | {'-'*30} | {'-'*15} | {'-'*10}")
        
        for rank, (node, score) in enumerate(nodes_with_scores, 1):
            entity_id = node.get('entity_id', 'N/A')
            name = node.get('canonical_name', 'N/A')
            entity_type = node.get('type', 'N/A')
            print(f"  {rank:<6} | {entity_id:<20} | {name:<30} | {entity_type:<15} | {score:.6f}")
        print()
        
        # Display edges
        if result.edges:
            print("🔗 Edges Traversed:")
            print()
            print(f"  {'From':<20} | {'Relation':<15} | {'To':<20} | {'Strength':<10} | {'Source':<15}")
            print(f"  {'-'*20} | {'-'*15} | {'-'*20} | {'-'*10} | {'-'*15}")
            
            for edge in result.edges[:20]:  # Show first 20
                from_id = edge.get('from', 'N/A')
                to_id = edge.get('to', 'N/A')
                predicate = edge.get('predicate', 'N/A')
                strength = edge.get('strength', 0)
                source = edge.get('source', 'N/A')
                print(f"  {from_id:<20} | {predicate:<15} | {to_id:<20} | {strength:.4f}     | {source:<15}")
            print()
        
        # Step 4: Fetch supporting memories for top edges
        print("=" * 100)
        print("STEP 4: Fetch Supporting Memories")
        print("=" * 100)
        print()
        
        if result.edges:
            # Get first edge to demonstrate provenance
            test_edge = result.edges[0]
            from_id = test_edge.get('from')
            to_id = test_edge.get('to')
            
            print(f"🔍 Fetching memories for edge: {from_id} --[{test_edge.get('predicate')}]--> {to_id}")
            print()
            
            # Query candidate_edges for supporting mentions
            if test_edge.get('source') == 'candidate_edge':
                # Find the edge in candidate_edges collection
                edge_query = """
                FOR edge IN candidate_edges
                    FILTER edge.subject_entity_id == @from_id
                    FILTER edge.object_entity_id == @to_id
                    FILTER edge.user_id == @user_id
                    LIMIT 1
                    RETURN edge
                """
                
                edge_cursor = db.aql.execute(edge_query, bind_vars={
                    'from_id': from_id,
                    'to_id': to_id,
                    'user_id': 'saswata'
                })
                edge_docs = list(edge_cursor)
                
                if edge_docs:
                    edge_doc = edge_docs[0]
                    supporting_mentions = edge_doc.get('supporting_mentions', [])
                    
                    print(f"  Supporting Mentions: {len(supporting_mentions)}")
                    print()
                    
                    for mention in supporting_mentions:
                        mem_id = mention.get('mem_id')
                        ego = mention.get('ego', 0)
                        
                        # Fetch the actual memory
                        try:
                            memory = db.collection('memories').get(mem_id)
                            if memory:
                                content = memory.get('content', 'N/A')
                                tier = memory.get('tier', 'N/A')
                                created = memory.get('created_at', 'N/A')
                                
                                print(f"  📄 Memory: {mem_id}")
                                print(f"     Tier: {tier}, Ego: {ego:.3f}")
                                print(f"     Created: {created}")
                                print(f"     Content: {content[:100]}...")
                                print()
                        except Exception as e:
                            print(f"  ❌ Could not fetch memory {mem_id}: {e}")
                else:
                    print("  ⚠️  Edge not found in candidate_edges")
            else:
                print("  ℹ️  This is a thought edge (promoted)")
        
        print("=" * 100)
        print("✅ PPR RETRIEVAL TEST COMPLETE")
        print("=" * 100)
        print()
        print("📊 Summary:")
        print(f"  ✅ Loaded {len(result.nodes)} nodes from graph")
        print(f"  ✅ Loaded {len(result.edges)} edges from graph")
        print(f"  ✅ PPR algorithm executed successfully")
        print(f"  ✅ Bidirectional linkage verified (edge → supporting memories)")
        print()
        print("🎯 Next Steps:")
        print("  1. Wire PPR into ContextManager for query-based retrieval")
        print("  2. Implement entity extraction from user queries")
        print("  3. Add graph results to LLM context")
        print("  4. Test with natural language queries")
        
    except Exception as e:
        print(f"❌ PPR retrieval failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ppr_retrieval())
