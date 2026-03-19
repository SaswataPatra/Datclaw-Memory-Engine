#!/usr/bin/env python3
"""
Quick test script for the Angela sentence to verify entity and relation extraction.
"""
import asyncio
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.graph.relation_extractor import RelationExtractor
from core.graph.entity_resolver import EntityResolver
from core.graph.entity_extractor import EntityExtractor
from core.graph.relation_classifier import RelationClassifier
from core.graph.coref_resolver import CorefResolver
from core.graph.dependency_extractor import DependencyExtractor
from core.graph.relation_normalizer import RelationNormalizer
from services.embedding_service import EmbeddingService
from arango import ArangoClient
import yaml


async def main():
    """Test the Angela sentence."""
    
    # Load config
    with open('config/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ArangoDB
    arango_config = config['arangodb']
    # Replace env vars with defaults
    url = arango_config['url'].replace('${ARANGO_HOST:localhost}', 'localhost').replace('${ARANGO_PORT:8529}', '8529')
    password = arango_config['password'].replace('${ARANGODB_PASSWORD}', os.getenv('ARANGODB_PASSWORD', 'dappy_dev_password'))
    
    client = ArangoClient(hosts=url)
    db = client.db(
        arango_config['database'],
        username=arango_config['username'],
        password=password
    )
    
    # Initialize embedding service
    embedding_service = EmbeddingService(config)
    
    # Initialize the relation extractor
    relation_extractor = RelationExtractor(
        db=db,
        config=config,
        embedding_service=embedding_service,
        collect_training_data=False
    )
    
    # Test sentence
    sentence = "Angela is a 31 year old woman who works as the manager of a gift shop in Chapel Hill and she sells interesting pieces from local artists including oil paintings"
    
    print("=" * 80)
    print("Testing Angela Sentence")
    print("=" * 80)
    print(f"\n📝 Input: {sentence}\n")
    
    # Extract relations
    relations = await relation_extractor.extract(
        text=sentence,
        user_id="test_user",
        memory_id="test_memory_123",
        ego_score=0.85,
        session_id="test_session",
        metadata={"source": "test"}
    )
    
    print("\n" + "=" * 80)
    print(f"✅ RESULTS: Extracted {len(relations)} relations")
    print("=" * 80)
    
    for idx, rel in enumerate(relations, 1):
        print(f"\n{idx}. {rel.subject_text} --[{rel.relation}]--> {rel.object_text}")
        print(f"   Category: {rel.category}")
        print(f"   Confidence: {rel.confidence:.2f}")
        print(f"   Source: {rel.source}")
        if rel.metadata:
            print(f"   Metadata: {rel.metadata}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

