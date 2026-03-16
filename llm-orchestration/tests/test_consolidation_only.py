"""
Simplified test for ConsolidationService only.
Tests the core functionality we built: single LLM call for entities + relations with confidence.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-orchestration"))

from services.consolidation_service import ConsolidationService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def test_consolidation():
    """Test consolidation service extracts entities and relations with confidence."""
    logger.info("="*80)
    logger.info("TEST: ConsolidationService - Single LLM Call")
    logger.info("="*80)
    
    consolidation_service = ConsolidationService(api_key=OPENAI_API_KEY)
    
    test_cases = [
        {
            "name": "Project Memory",
            "text": """
            I'm working on a liquidation bot project. It's a trading bot that monitors 
            cryptocurrency markets for liquidation events and executes trades automatically.
            The bot is built with Python and uses the Binance API.
            """,
            "ego_score": 0.8,
            "tier": 1
        },
        {
            "name": "Personal Relationship",
            "text": "My sister Sarah lives in San Francisco and works as a software engineer at Google.",
            "ego_score": 0.9,
            "tier": 1
        },
        {
            "name": "Temporary Thought (should extract nothing)",
            "text": "I wonder if I should get coffee today? Maybe later.",
            "ego_score": 0.2,
            "tier": 3
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Test Case {i}: {test_case['name']}")
        logger.info(f"{'='*80}")
        logger.info(f"Text: {test_case['text'].strip()}")
        logger.info(f"Ego Score: {test_case['ego_score']}, Tier: {test_case['tier']}")
        
        result = await consolidation_service.consolidate(
            text=test_case['text'],
            ego_score=test_case['ego_score'],
            tier=test_case['tier']
        )
        
        entities = result.get("entities", [])
        relations = result.get("relations", [])
        
        logger.info(f"\n📊 Results:")
        logger.info(f"   Entities: {len(entities)}")
        for ent in entities:
            logger.info(f"      - {ent['name']} ({ent['type']})")
        
        logger.info(f"   Relations: {len(relations)}")
        for rel in relations:
            logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel['confidence']:.2f})")
        
        # Validate
        if test_case['name'] == "Temporary Thought (should extract nothing)":
            if len(relations) == 0:
                logger.info("   ✅ PASS: Correctly extracted no stable relations from temporary thought")
            else:
                logger.error("   ❌ FAIL: Should not extract relations from temporary thoughts")
                all_passed = False
        else:
            if len(entities) > 0 and len(relations) > 0:
                logger.info("   ✅ PASS: Extracted entities and relations")
                
                # Check confidence scores
                for rel in relations:
                    if 'confidence' not in rel or not (0.0 <= rel['confidence'] <= 1.0):
                        logger.error(f"   ❌ FAIL: Invalid confidence score: {rel.get('confidence')}")
                        all_passed = False
                    else:
                        logger.info(f"   ✅ Confidence score valid: {rel['confidence']:.2f}")
            else:
                logger.error("   ❌ FAIL: Should extract entities and relations from stable memory")
                all_passed = False
    
    # Final verdict
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("   - ConsolidationService working correctly")
        logger.info("   - Single LLM call extracts entities + relations")
        logger.info("   - Confidence scores included and valid")
        logger.info("   - Filters out temporary/speculative content")
    else:
        logger.error("❌ SOME TESTS FAILED")
    logger.info("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(test_consolidation())
    sys.exit(0 if success else 1)
