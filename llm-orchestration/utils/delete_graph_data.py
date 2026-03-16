#!/usr/bin/env python3
"""
Delete graph data (entities, edges) from ArangoDB.

Usage:
    # Delete specific entity
    python utils/delete_graph_data.py --entity e_abc123
    
    # Delete specific candidate edge
    python utils/delete_graph_data.py --candidate ce_abc123
    
    # Delete specific thought edge
    python utils/delete_graph_data.py --thought te_abc123
    
    # Delete all graph data for a user
    python utils/delete_graph_data.py --user user_123
    
    # Delete ALL graph data (DANGEROUS!)
    python utils/delete_graph_data.py --all --confirm
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arango import ArangoClient
from config import Config


def get_db():
    """Get ArangoDB connection."""
    config = Config()
    arango_config = config.get_section('arangodb')
    
    client = ArangoClient(hosts=arango_config['url'])
    db = client.db(
        name=arango_config['database'],
        username=arango_config['username'],
        password=arango_config['password']
    )
    return db


def delete_entity(db, entity_id: str):
    """Delete a specific entity."""
    try:
        # Delete entity
        entities = db.collection('entities')
        result = entities.delete(entity_id)
        print(f"✅ Deleted entity: {entity_id}")
        
        # Delete associated edges
        deleted_edges = 0
        for edge_collection in ['candidate_edges', 'thought_edges']:
            if db.has_collection(edge_collection):
                edges = db.collection(edge_collection)
                
                # Delete edges where entity is subject
                query = f"""
                FOR e IN {edge_collection}
                FILTER e.subject_entity_id == @entity_id
                REMOVE e IN {edge_collection}
                RETURN OLD
                """
                cursor = db.aql.execute(query, bind_vars={'entity_id': entity_id})
                deleted_edges += len(list(cursor))
                
                # Delete edges where entity is object
                query = f"""
                FOR e IN {edge_collection}
                FILTER e.object_entity_id == @entity_id
                REMOVE e IN {edge_collection}
                RETURN OLD
                """
                cursor = db.aql.execute(query, bind_vars={'entity_id': entity_id})
                deleted_edges += len(list(cursor))
        
        if deleted_edges > 0:
            print(f"✅ Deleted {deleted_edges} associated edges")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting entity {entity_id}: {e}")
        return False


def delete_candidate_edge(db, candidate_id: str):
    """Delete a specific candidate edge."""
    try:
        edges = db.collection('candidate_edges')
        result = edges.delete(candidate_id)
        print(f"✅ Deleted candidate edge: {candidate_id}")
        return True
    except Exception as e:
        print(f"❌ Error deleting candidate edge {candidate_id}: {e}")
        return False


def delete_thought_edge(db, edge_id: str):
    """Delete a specific thought edge."""
    try:
        edges = db.collection('thought_edges')
        result = edges.delete(edge_id)
        print(f"✅ Deleted thought edge: {edge_id}")
        
        # Also delete bidirectional pair if exists
        query = """
        FOR e IN thought_edges
        FILTER e.pair_edge_id == @edge_id
        REMOVE e IN thought_edges
        RETURN OLD
        """
        cursor = db.aql.execute(query, bind_vars={'edge_id': edge_id})
        pairs = list(cursor)
        if pairs:
            print(f"✅ Deleted {len(pairs)} bidirectional pair(s)")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting thought edge {edge_id}: {e}")
        return False


def delete_user_graph_data(db, user_id: str):
    """Delete all graph data for a specific user."""
    try:
        deleted_counts = {}
        
        # Delete entities
        if db.has_collection('entities'):
            query = """
            FOR e IN entities
            FILTER e.user_id == @user_id
            REMOVE e IN entities
            RETURN OLD
            """
            cursor = db.aql.execute(query, bind_vars={'user_id': user_id})
            deleted_counts['entities'] = len(list(cursor))
        
        # Delete candidate edges
        if db.has_collection('candidate_edges'):
            query = """
            FOR e IN candidate_edges
            FILTER e.user_id == @user_id
            REMOVE e IN candidate_edges
            RETURN OLD
            """
            cursor = db.aql.execute(query, bind_vars={'user_id': user_id})
            deleted_counts['candidate_edges'] = len(list(cursor))
        
        # Delete thought edges
        if db.has_collection('thought_edges'):
            query = """
            FOR e IN thought_edges
            FILTER e.user_id == @user_id
            REMOVE e IN thought_edges
            RETURN OLD
            """
            cursor = db.aql.execute(query, bind_vars={'user_id': user_id})
            deleted_counts['thought_edges'] = len(list(cursor))
        
        print(f"✅ Deleted graph data for user {user_id}:")
        for collection, count in deleted_counts.items():
            print(f"   - {collection}: {count} documents")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting user graph data: {e}")
        return False


def delete_all_graph_data(db, confirmed: bool = False):
    """Delete ALL graph data (DANGEROUS!)."""
    if not confirmed:
        print("❌ Must use --confirm flag to delete all graph data")
        return False
    
    try:
        deleted_counts = {}
        
        for collection_name in ['entities', 'candidate_edges', 'thought_edges']:
            if db.has_collection(collection_name):
                collection = db.collection(collection_name)
                count = collection.count()
                collection.truncate()
                deleted_counts[collection_name] = count
        
        print("✅ Deleted ALL graph data:")
        for collection, count in deleted_counts.items():
            print(f"   - {collection}: {count} documents")
        
        return True
    except Exception as e:
        print(f"❌ Error deleting all graph data: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Delete graph data from ArangoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--entity', help='Delete specific entity by ID')
    parser.add_argument('--candidate', help='Delete specific candidate edge by ID')
    parser.add_argument('--thought', help='Delete specific thought edge by ID')
    parser.add_argument('--user', help='Delete all graph data for a user')
    parser.add_argument('--all', action='store_true', help='Delete ALL graph data (requires --confirm)')
    parser.add_argument('--confirm', action='store_true', help='Confirm deletion of all data')
    
    args = parser.parse_args()
    
    # Check that at least one action is specified
    if not any([args.entity, args.candidate, args.thought, args.user, args.all]):
        parser.print_help()
        sys.exit(1)
    
    # Get database connection
    db = get_db()
    
    # Perform deletion
    success = False
    
    if args.entity:
        success = delete_entity(db, args.entity)
    elif args.candidate:
        success = delete_candidate_edge(db, args.candidate)
    elif args.thought:
        success = delete_thought_edge(db, args.thought)
    elif args.user:
        success = delete_user_graph_data(db, args.user)
    elif args.all:
        success = delete_all_graph_data(db, args.confirm)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

