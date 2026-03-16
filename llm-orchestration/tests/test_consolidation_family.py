"""
Test Consolidation Service with improved family relation predicates.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llm-orchestration"))

from services.consolidation_service import ConsolidationService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def test_family_relations():
    """Test that family relations use proper directional predicates."""
    logger.info("="*80)
    logger.info("TEST: Consolidation Service - Family Relations")
    logger.info("="*80)
    
    service = ConsolidationService(api_key=OPENAI_API_KEY)
    
    # Test 1: Father's name
    logger.info("\n📝 Test 1: Father's name")
    text1 = "My father's name is Arun Kumar Patra"
    result1 = await service.consolidate(text=text1, ego_score=0.8, tier=1)
    
    logger.info(f"   Entities: {result1['entities']}")
    logger.info(f"   Relations: {result1['relations']}")
    
    # Check for proper predicate
    has_proper_predicate = any(
        (r['subject'] == 'user' and r['predicate'] in ['has_father', 'father_of'] and 'arun' in r['object'].lower())
        or (r['subject'].lower().startswith('arun') and r['predicate'] in ['father_of'] and r['object'] == 'user')
        for r in result1['relations']
    )
    
    if has_proper_predicate:
        logger.info("   ✅ PASS: Using directional predicate (has_father or father_of)")
    else:
        logger.info("   ⚠️  Using different predicate structure")
    
    # Test 2: Mother's name
    logger.info("\n📝 Test 2: Mother's name")
    text2 = "My mother's name is Sunita Patra"
    result2 = await service.consolidate(text=text2, ego_score=0.8, tier=1)
    
    logger.info(f"   Entities: {result2['entities']}")
    logger.info(f"   Relations: {result2['relations']}")
    
    has_proper_predicate = any(
        (r['subject'] == 'user' and r['predicate'] in ['has_mother', 'mother_of'] and 'sunita' in r['object'].lower())
        or (r['subject'].lower().startswith('sunita') and r['predicate'] in ['mother_of'] and r['object'] == 'user')
        for r in result2['relations']
    )
    
    if has_proper_predicate:
        logger.info("   ✅ PASS: Using directional predicate (has_mother or mother_of)")
    else:
        logger.info("   ⚠️  Using different predicate structure")
    
    # Test 3: Sister's name
    logger.info("\n📝 Test 3: Sister's name")
    text3 = "My sister Priya lives in Mumbai"
    result3 = await service.consolidate(text=text3, ego_score=0.8, tier=1)
    
    logger.info(f"   Entities: {result3['entities']}")
    logger.info(f"   Relations: {result3['relations']}")
    
    has_sister_relation = any(
        r['subject'] == 'user' and r['predicate'] in ['has_sister', 'sister_of']
        for r in result3['relations']
    )
    
    has_location_relation = any(
        'priya' in r['subject'].lower() and r['predicate'] == 'lives_in' and 'mumbai' in r['object'].lower()
        for r in result3['relations']
    )
    
    if has_sister_relation:
        logger.info("   ✅ PASS: Extracted sister relation")
    else:
        logger.info("   ⚠️  Sister relation not extracted as expected")
    
    if has_location_relation:
        logger.info("   ✅ PASS: Extracted location relation")
    else:
        logger.info("   ⚠️  Location relation not extracted as expected")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TEST COMPLETE: Review predicate usage above")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(test_family_relations())
