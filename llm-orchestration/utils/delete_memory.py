#!/usr/bin/env python3
"""
Delete Memory Utility

Deletes a specific memory from both ArangoDB and Qdrant.

Usage:
    python utils/delete_memory.py <memory_id>
    python utils/delete_memory.py 5c862b62-6671-42c4-aac8-c794cea25040
    
    # Delete multiple memories
    python utils/delete_memory.py <id1> <id2> <id3>
"""

import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arango import ArangoClient
from qdrant_client import QdrantClient
from config import load_config


async def delete_memory(memory_id: str, config: dict) -> tuple[bool, bool]:
    """
    Delete memory from both databases
    
    Args:
        memory_id: UUID of the memory to delete
        config: Configuration dict
        
    Returns:
        Tuple of (arango_success, qdrant_success)
    """
    
    # Connect to ArangoDB
    arango_config = config.get_section('arangodb')
    arango_client = ArangoClient(hosts=arango_config['url'])
    db = arango_client.db(
        arango_config['database'],
        username=arango_config['username'],
        password=arango_config['password']
    )
    
    # Connect to Qdrant
    qdrant_config = config.get_section('qdrant')
    qdrant_client = QdrantClient(url=qdrant_config['url'])
    
    arango_success = False
    qdrant_success = False
    
    # Delete from ArangoDB
    try:
        collection = db.collection(arango_config.get('memory_collection', 'memories'))
        result = collection.delete(memory_id)
        if result:
            print(f"  ✅ ArangoDB: Deleted")
            arango_success = True
        else:
            print(f"  ⚠️  ArangoDB: Not found")
    except Exception as e:
        print(f"  ❌ ArangoDB: Error - {e}")
    
    # Delete from Qdrant
    try:
        collection_name = qdrant_config.get('collection_name', 'memories')
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=[memory_id]
        )
        print(f"  ✅ Qdrant: Deleted")
        qdrant_success = True
    except Exception as e:
        print(f"  ❌ Qdrant: Error - {e}")
    
    return arango_success, qdrant_success


async def delete_memories(memory_ids: list[str]):
    """Delete multiple memories"""
    
    # Load config once
    config = load_config()
    
    print(f"\n🗑️  Deleting {len(memory_ids)} memory(ies)")
    print("=" * 70)
    
    results = []
    for i, memory_id in enumerate(memory_ids, 1):
        print(f"\n[{i}/{len(memory_ids)}] Memory: {memory_id}")
        arango_ok, qdrant_ok = await delete_memory(memory_id, config)
        results.append((memory_id, arango_ok, qdrant_ok))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print("=" * 70)
    
    total = len(results)
    both_success = sum(1 for _, a, q in results if a and q)
    partial_success = sum(1 for _, a, q in results if (a or q) and not (a and q))
    failed = sum(1 for _, a, q in results if not a and not q)
    
    print(f"  ✅ Fully deleted:     {both_success}/{total}")
    print(f"  ⚠️  Partially deleted: {partial_success}/{total}")
    print(f"  ❌ Failed:            {failed}/{total}")
    
    if partial_success > 0 or failed > 0:
        print("\n⚠️  Details:")
        for memory_id, arango_ok, qdrant_ok in results:
            if not (arango_ok and qdrant_ok):
                status = []
                if not arango_ok:
                    status.append("ArangoDB failed")
                if not qdrant_ok:
                    status.append("Qdrant failed")
                print(f"  - {memory_id}: {', '.join(status)}")
    
    print("=" * 70)
    print("✅ Deletion complete!\n")


def main():
    """Main CLI entry point"""
    
    # Check if IDs provided as arguments
    if len(sys.argv) >= 2:
        memory_ids = sys.argv[1:]
    else:
        # Interactive mode: prompt for IDs
        print("🗑️  Memory Deletion Utility")
        print("=" * 70)
        print("\nEnter memory ID(s) to delete (comma-separated for multiple):")
        print("Example: 5c862b62-6671-42c4-aac8-c794cea25040")
        print("         or: id1, id2, id3")
        print("\nPress Ctrl+C to cancel\n")
        
        try:
            user_input = input("Memory ID(s): ").strip()
        except KeyboardInterrupt:
            print("\n\n⚠️  Cancelled by user")
            sys.exit(0)
        
        if not user_input:
            print("❌ Error: No memory ID provided")
            sys.exit(1)
        
        # Split by comma and clean up whitespace
        memory_ids = [id.strip() for id in user_input.split(',') if id.strip()]
        
        if not memory_ids:
            print("❌ Error: No valid memory ID provided")
            sys.exit(1)
    
    # Validate UUIDs (basic check)
    invalid_ids = []
    for memory_id in memory_ids:
        # UUID format: 8-4-4-4-12 characters
        parts = memory_id.split('-')
        if len(parts) != 5 or len(memory_id) != 36:
            invalid_ids.append(memory_id)
    
    if invalid_ids:
        print("❌ Error: Invalid memory ID format")
        print("\nInvalid IDs:")
        for invalid_id in invalid_ids:
            print(f"  - {invalid_id}")
        print("\nExpected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        sys.exit(1)
    
    # Run deletion
    try:
        asyncio.run(delete_memories(memory_ids))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

