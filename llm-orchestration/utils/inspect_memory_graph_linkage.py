#!/usr/bin/env python3
"""
Memory-Graph Linkage Inspector

Queries ArangoDB to show bidirectional linkage between memories and graph elements.
Useful for debugging and verifying the graph extraction pipeline.
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from arango import ArangoClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


class MemoryGraphInspector:
    """Inspector for memory-graph bidirectional linkage."""
    
    def __init__(self):
        """Initialize ArangoDB connection."""
        arango_host = os.getenv('ARANGO_HOST', 'localhost')
        arango_port = os.getenv('ARANGO_PORT', '8529')
        arango_user = os.getenv('ARANGODB_USERNAME', 'root')
        arango_password = os.getenv('ARANGODB_PASSWORD', 'dappy_dev_password')
        arango_db = os.getenv('ARANGODB_DATABASE', 'dappy_memories')
        
        # Connect to ArangoDB
        client = ArangoClient(hosts=f'http://{arango_host}:{arango_port}')
        self.db = client.db(arango_db, username=arango_user, password=arango_password)
        
        print(f"✅ Connected to ArangoDB: {arango_db}")
        print()
    
    def list_recent_memories_with_graph_data(self, limit: int = 10, user_id: Optional[str] = None):
        """
        List recent memories that have graph extraction data.
        
        Args:
            limit: Number of memories to show
            user_id: Filter by user_id (optional)
        """
        print("=" * 100)
        print(f"📋 RECENT MEMORIES WITH GRAPH EXTRACTION (limit={limit})")
        print("=" * 100)
        print()
        
        # Build query
        query = """
        FOR mem IN memories
            FILTER mem.graph_extraction != null
        """
        
        if user_id:
            query += f"    FILTER mem.user_id == @user_id\n"
        
        query += """
            SORT mem.created_at DESC
            LIMIT @limit
            RETURN {
                memory_id: mem._key,
                user_id: mem.user_id,
                content: mem.content,
                tier: mem.tier,
                ego_score: mem.ego_score,
                created_at: mem.created_at,
                entity_count: mem.graph_extraction.entity_count,
                edge_count: mem.graph_extraction.edge_count,
                entities: mem.graph_extraction.extracted_entities,
                edges: mem.graph_extraction.extracted_edges
            }
        """
        
        bind_vars = {'limit': limit}
        if user_id:
            bind_vars['user_id'] = user_id
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        memories = list(cursor)
        
        if not memories:
            print("❌ No memories with graph extraction found")
            return
        
        print(f"Found {len(memories)} memories with graph extraction:\n")
        
        for i, mem in enumerate(memories, 1):
            print(f"{'─' * 100}")
            print(f"Memory #{i}: {mem['memory_id']}")
            print(f"{'─' * 100}")
            print(f"  User:       {mem['user_id']}")
            print(f"  Tier:       {mem['tier']}")
            print(f"  Ego Score:  {mem['ego_score']:.3f}")
            print(f"  Created:    {mem['created_at']}")
            print(f"  Content:    {mem['content'][:100]}...")
            print()
            print(f"  📊 Graph Extraction:")
            print(f"     → {mem['entity_count']} entities")
            print(f"     → {mem['edge_count']} edges")
            print()
            
            if mem['entities']:
                print(f"  📍 Extracted Entities:")
                for e in mem['entities']:
                    print(f"     - {e.get('entity_id', 'N/A'):20s} | {e.get('text', 'N/A'):30s} | {e.get('type', 'N/A')}")
                print()
            
            if mem['edges']:
                print(f"  🔗 Extracted Edges:")
                for e in mem['edges']:
                    print(f"     - {e.get('edge_id', 'N/A'):20s} | {e.get('subject', 'N/A'):15s} --[{e.get('relation', 'N/A')}]--> {e.get('object', 'N/A')}")
                print()
            
            print()
    
    def inspect_memory(self, memory_id: str):
        """
        Deep inspection of a specific memory's graph linkage.
        
        Args:
            memory_id: Memory ID to inspect
        """
        print("=" * 100)
        print(f"🔍 DEEP INSPECTION: Memory {memory_id}")
        print("=" * 100)
        print()
        
        # Fetch memory
        try:
            mem = self.db.collection('memories').get(memory_id)
        except Exception as e:
            print(f"❌ Memory not found: {e}")
            return
        
        if not mem:
            print(f"❌ Memory {memory_id} not found")
            return
        
        # Display memory details
        print("📄 Memory Details:")
        print(f"  ID:         {mem['_key']}")
        print(f"  User:       {mem.get('user_id', 'N/A')}")
        print(f"  Tier:       {mem.get('tier', 'N/A')}")
        print(f"  Ego Score:  {mem.get('ego_score', 0):.3f}")
        print(f"  Created:    {mem.get('created_at', 'N/A')}")
        print(f"  Content:    {mem.get('content', 'N/A')}")
        print()
        
        # Check for graph extraction
        graph_extraction = mem.get('graph_extraction')
        if not graph_extraction:
            print("⚠️  No graph extraction data found for this memory")
            return
        
        print("📊 Graph Extraction Summary:")
        print(f"  Entities:   {graph_extraction.get('entity_count', 0)}")
        print(f"  Edges:      {graph_extraction.get('edge_count', 0)}")
        print(f"  Timestamp:  {graph_extraction.get('extraction_timestamp', 'N/A')}")
        print()
        
        # Display entities
        entities = graph_extraction.get('extracted_entities', [])
        if entities:
            print("📍 Extracted Entities:")
            print(f"  {'Entity ID':<25} | {'Text':<30} | {'Type':<15}")
            print(f"  {'-'*25} | {'-'*30} | {'-'*15}")
            for e in entities:
                entity_type = e.get('type') or 'N/A'
                print(f"  {e.get('entity_id', 'N/A'):<25} | {e.get('text', 'N/A'):<30} | {entity_type:<15}")
            print()
            
            # Verify entities exist in graph
            print("🔗 Entity Verification (checking if entities exist in graph):")
            for entity in entities:
                entity_id = entity.get('entity_id')
                try:
                    entity_doc = self.db.collection('entities').get(entity_id)
                    if entity_doc:
                        print(f"  ✅ {entity_id} ({entity.get('text')}) - EXISTS")
                    else:
                        print(f"  ❌ {entity_id} ({entity.get('text')}) - NOT FOUND")
                except Exception as e:
                    print(f"  ❌ {entity_id} ({entity.get('text')}) - ERROR: {e}")
            print()
        
        # Display edges
        edges = graph_extraction.get('extracted_edges', [])
        if edges:
            print("🔗 Extracted Edges:")
            print(f"  {'Edge ID':<25} | {'Subject':<20} | {'Relation':<15} | {'Object':<20}")
            print(f"  {'-'*25} | {'-'*20} | {'-'*15} | {'-'*20}")
            for e in edges:
                print(f"  {e.get('edge_id', 'N/A'):<25} | {e.get('subject', 'N/A'):<20} | {e.get('relation', 'N/A'):<15} | {e.get('object', 'N/A'):<20}")
            print()
            
            # Verify edges exist and show supporting mentions
            print("🔗 Edge Verification (checking supporting mentions):")
            for edge in edges:
                edge_id = edge.get('edge_id')
                try:
                    edge_doc = self.db.collection('candidate_edges').get(edge_id)
                    if edge_doc:
                        supporting_mentions = edge_doc.get('supporting_mentions', [])
                        print(f"  ✅ {edge_id} - EXISTS")
                        print(f"     Subject: {edge_doc.get('subject_entity_id', 'N/A')}")
                        print(f"     Predicate: {edge_doc.get('predicate', 'N/A')}")
                        print(f"     Object: {edge_doc.get('object_entity_id', 'N/A')}")
                        print(f"     Supporting Mentions: {len(supporting_mentions)}")
                        
                        # Show supporting mentions
                        if supporting_mentions:
                            for mention in supporting_mentions:
                                mem_id = mention.get('mem_id', 'N/A')
                                ego = mention.get('ego', 0)
                                is_current = "← THIS MEMORY" if mem_id == memory_id else ""
                                print(f"       - {mem_id} (ego={ego:.2f}) {is_current}")
                    else:
                        print(f"  ❌ {edge_id} - NOT FOUND")
                except Exception as e:
                    print(f"  ❌ {edge_id} - ERROR: {e}")
            print()
    
    def find_memories_by_entity(self, entity_text: str, user_id: Optional[str] = None):
        """
        Find all memories that extracted a specific entity.
        
        Args:
            entity_text: Entity text to search for
            user_id: Filter by user_id (optional)
        """
        print("=" * 100)
        print(f"🔍 MEMORIES THAT EXTRACTED ENTITY: '{entity_text}'")
        print("=" * 100)
        print()
        
        # Build query
        query = """
        FOR mem IN memories
            FILTER mem.graph_extraction != null
        """
        
        if user_id:
            query += f"    FILTER mem.user_id == @user_id\n"
        
        query += """
            FILTER LENGTH(
                FOR entity IN mem.graph_extraction.extracted_entities
                    FILTER LOWER(entity.text) == LOWER(@entity_text)
                    RETURN entity
            ) > 0
            SORT mem.created_at DESC
            RETURN {
                memory_id: mem._key,
                content: mem.content,
                tier: mem.tier,
                ego_score: mem.ego_score,
                created_at: mem.created_at,
                entities: (
                    FOR entity IN mem.graph_extraction.extracted_entities
                        FILTER LOWER(entity.text) == LOWER(@entity_text)
                        RETURN entity
                )
            }
        """
        
        bind_vars = {'entity_text': entity_text}
        if user_id:
            bind_vars['user_id'] = user_id
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        memories = list(cursor)
        
        if not memories:
            print(f"❌ No memories found that extracted entity '{entity_text}'")
            return
        
        print(f"Found {len(memories)} memories:\n")
        
        for i, mem in enumerate(memories, 1):
            print(f"{i}. Memory: {mem['memory_id']}")
            print(f"   Tier: {mem['tier']}, Ego: {mem['ego_score']:.3f}")
            print(f"   Created: {mem['created_at']}")
            print(f"   Content: {mem['content'][:80]}...")
            print(f"   Entities: {mem['entities']}")
            print()
    
    def find_memories_by_relation(self, relation: str, user_id: Optional[str] = None):
        """
        Find all memories that extracted a specific relation.
        
        Args:
            relation: Relation type to search for
            user_id: Filter by user_id (optional)
        """
        print("=" * 100)
        print(f"🔍 MEMORIES THAT EXTRACTED RELATION: '{relation}'")
        print("=" * 100)
        print()
        
        # Build query
        query = """
        FOR mem IN memories
            FILTER mem.graph_extraction != null
        """
        
        if user_id:
            query += f"    FILTER mem.user_id == @user_id\n"
        
        query += """
            FILTER LENGTH(
                FOR edge IN mem.graph_extraction.extracted_edges
                    FILTER LOWER(edge.relation) == LOWER(@relation)
                    RETURN edge
            ) > 0
            SORT mem.created_at DESC
            RETURN {
                memory_id: mem._key,
                content: mem.content,
                tier: mem.tier,
                ego_score: mem.ego_score,
                created_at: mem.created_at,
                edges: (
                    FOR edge IN mem.graph_extraction.extracted_edges
                        FILTER LOWER(edge.relation) == LOWER(@relation)
                        RETURN edge
                )
            }
        """
        
        bind_vars = {'relation': relation}
        if user_id:
            bind_vars['user_id'] = user_id
        
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        memories = list(cursor)
        
        if not memories:
            print(f"❌ No memories found that extracted relation '{relation}'")
            return
        
        print(f"Found {len(memories)} memories:\n")
        
        for i, mem in enumerate(memories, 1):
            print(f"{i}. Memory: {mem['memory_id']}")
            print(f"   Tier: {mem['tier']}, Ego: {mem['ego_score']:.3f}")
            print(f"   Created: {mem['created_at']}")
            print(f"   Content: {mem['content'][:80]}...")
            print(f"   Edges:")
            for edge in mem['edges']:
                print(f"     - {edge.get('subject')} --[{edge.get('relation')}]--> {edge.get('object')}")
            print()


def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect memory-graph bidirectional linkage')
    parser.add_argument('command', choices=['list', 'inspect', 'find-entity', 'find-relation'],
                       help='Command to run')
    parser.add_argument('--memory-id', help='Memory ID to inspect (for inspect command)')
    parser.add_argument('--entity', help='Entity text to search for (for find-entity command)')
    parser.add_argument('--relation', help='Relation type to search for (for find-relation command)')
    parser.add_argument('--user-id', help='Filter by user ID')
    parser.add_argument('--limit', type=int, default=10, help='Number of results to show (default: 10)')
    
    args = parser.parse_args()
    
    inspector = MemoryGraphInspector()
    
    if args.command == 'list':
        inspector.list_recent_memories_with_graph_data(limit=args.limit, user_id=args.user_id)
    
    elif args.command == 'inspect':
        if not args.memory_id:
            print("❌ --memory-id required for inspect command")
            return
        inspector.inspect_memory(args.memory_id)
    
    elif args.command == 'find-entity':
        if not args.entity:
            print("❌ --entity required for find-entity command")
            return
        inspector.find_memories_by_entity(args.entity, user_id=args.user_id)
    
    elif args.command == 'find-relation':
        if not args.relation:
            print("❌ --relation required for find-relation command")
            return
        inspector.find_memories_by_relation(args.relation, user_id=args.user_id)


if __name__ == "__main__":
    main()

