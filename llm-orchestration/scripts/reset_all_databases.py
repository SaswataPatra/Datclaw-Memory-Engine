#!/usr/bin/env python3
"""
Reset All Databases - Clean slate for testing
Clears: ArangoDB, Qdrant, Redis
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from arango import ArangoClient
from qdrant_client import QdrantClient
import redis.asyncio as redis

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reset_all():
    """Reset all databases"""
    
    print("=" * 100)
    print("🧹 RESETTING ALL DATABASES")
    print("=" * 100)
    print()
    
    # 1. Reset ArangoDB
    print("🗄️  Resetting ArangoDB...")
    try:
        arango_client = ArangoClient(hosts='http://localhost:8529')
        db = arango_client.db('dappy_memories', username='root', password='dappy_dev_password')
        
        # Clear collections
        collections = ['entities', 'candidate_edges', 'memories']
        for coll_name in collections:
            try:
                coll = db.collection(coll_name)
                count = coll.count()
                coll.truncate()
                print(f"   ✅ Cleared {coll_name}: {count} documents deleted")
            except Exception as e:
                print(f"   ⚠️  {coll_name}: {e}")
        
        print("   ✅ ArangoDB reset complete")
    except Exception as e:
        print(f"   ❌ ArangoDB reset failed: {e}")
    
    print()
    
    # 2. Reset Qdrant
    print("🔍 Resetting Qdrant...")
    try:
        qdrant_client = QdrantClient(url="http://localhost:6333")
        
        # List all collections
        collections = qdrant_client.get_collections().collections
        
        for coll in collections:
            try:
                qdrant_client.delete_collection(coll.name)
                print(f"   ✅ Deleted collection: {coll.name}")
            except Exception as e:
                print(f"   ⚠️  {coll.name}: {e}")
        
        print("   ✅ Qdrant reset complete")
    except Exception as e:
        print(f"   ❌ Qdrant reset failed: {e}")
    
    print()
    
    # 3. Reset Redis
    print("💾 Resetting Redis...")
    try:
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        await redis_client.flushdb()
        print("   ✅ Redis reset complete")
        
        await redis_client.close()
    except Exception as e:
        print(f"   ❌ Redis reset failed: {e}")
    
    print()
    print("=" * 100)
    print("✅ ALL DATABASES RESET COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(reset_all())
