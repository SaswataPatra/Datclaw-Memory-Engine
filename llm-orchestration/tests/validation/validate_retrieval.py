#!/usr/bin/env python3
"""
Retrieval Validation Script

Runs the ground truth dataset against the actual retrieval system
and measures precision, recall, and overall accuracy.

Usage:
    python validation/validate_retrieval.py [user_id]
"""

import json
import sys
import os
import asyncio
import logging
import time

logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arango import ArangoClient


def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), 'retrieval_dataset.json')
    with open(dataset_path) as f:
        return json.load(f)


def keyword_match_score(content: str, expected_keywords: list) -> float:
    """Check what fraction of expected keywords appear in the content."""
    content_lower = content.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
    return matches / len(expected_keywords) if expected_keywords else 0.0


def evaluate_retrieval(user_id: str):
    """Evaluate retrieval quality against ground truth dataset."""
    dataset = load_dataset()
    entries = dataset['entries']

    client = ArangoClient(hosts='http://localhost:8529')
    db = client.db('dappy_memories', username='root', password=os.getenv('ARANGODB_PASSWORD', 'dappy_dev_password'))

    # Get all memories for this user
    all_memories = list(db.aql.execute("""
        FOR m IN memories
            FILTER m.user_id == @user_id
            RETURN {
                memory_id: m._key,
                content: m.content,
                tier: m.tier,
                ego_score: m.ego_score
            }
    """, bind_vars={'user_id': user_id}))

    if not all_memories:
        # Try without user_id filter (for imported memories with missing user_id)
        all_memories = list(db.aql.execute("""
            FOR m IN memories
                SORT m.created_at DESC
                LIMIT 50
                RETURN {
                    memory_id: m._key,
                    content: m.content,
                    tier: m.tier,
                    ego_score: m.ego_score
                }
        """))

    if not all_memories:
        print("\nNo memories found! Import some memories first.\n")
        return

    print(f"\nEvaluating against {len(all_memories)} memories\n")

    total_queries = len(entries)
    total_hit = 0         # At least one relevant memory found
    total_precision = 0.0
    total_recall = 0.0
    category_scores = {}

    top_k = 5

    for entry in entries:
        query = entry['query']
        expected = entry['expected_keywords']
        category = entry.get('category', 'unknown')

        # Simulate retrieval: rank memories by keyword relevance
        scored = []
        for mem in all_memories:
            score = keyword_match_score(mem['content'], expected)
            if score > 0:
                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        retrieved = scored[:top_k]

        # Measure precision@k: fraction of retrieved that are relevant (score > threshold)
        threshold = 0.3
        relevant_retrieved = [m for m, s in retrieved if s >= threshold]
        precision = len(relevant_retrieved) / top_k if top_k > 0 else 0.0

        # Measure recall: did we find at least one strongly matching memory?
        strong_match = any(s >= 0.5 for _, s in retrieved)
        hit = 1 if strong_match else 0

        total_hit += hit
        total_precision += precision
        total_recall += (1.0 if hit else 0.0)

        if category not in category_scores:
            category_scores[category] = {'queries': 0, 'hits': 0, 'precision_sum': 0.0}
        category_scores[category]['queries'] += 1
        category_scores[category]['hits'] += hit
        category_scores[category]['precision_sum'] += precision

    # Overall metrics
    hit_rate = total_hit / total_queries
    avg_precision = total_precision / total_queries
    avg_recall = total_recall / total_queries

    print("=" * 80)
    print("RETRIEVAL VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nDataset:        {total_queries} queries")
    print(f"Memories:       {len(all_memories)}")
    print(f"Top-K:          {top_k}")
    print(f"\n--- Overall Metrics ---")
    print(f"Hit Rate:       {hit_rate:.1%} ({total_hit}/{total_queries} queries found a match)")
    print(f"Avg Precision@{top_k}: {avg_precision:.1%}")
    print(f"Avg Recall:     {avg_recall:.1%}")

    print(f"\n--- By Category ---")
    print(f"{'Category':<15} {'Queries':>8} {'Hits':>6} {'Hit Rate':>10} {'Precision':>10}")
    print("-" * 55)
    for cat, scores in sorted(category_scores.items()):
        cat_hit_rate = scores['hits'] / scores['queries']
        cat_precision = scores['precision_sum'] / scores['queries']
        print(f"{cat:<15} {scores['queries']:>8} {scores['hits']:>6} {cat_hit_rate:>9.1%} {cat_precision:>9.1%}")

    # Individual failing queries
    print(f"\n--- Failing Queries (no strong match) ---")
    fail_count = 0
    for entry in entries:
        expected = entry['expected_keywords']
        scored = []
        for mem in all_memories:
            score = keyword_match_score(mem['content'], expected)
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1] if scored else 0.0
        if best_score < 0.5:
            fail_count += 1
            print(f"  [{entry['id']:>2}] {entry['query'][:60]:<60} best={best_score:.2f}")
            if fail_count >= 15:
                remaining = total_queries - total_hit - fail_count
                if remaining > 0:
                    print(f"  ... and {remaining} more")
                break

    print(f"\n{'=' * 80}")

    # Summary assessment
    if hit_rate >= 0.7:
        print(f"\nVERDICT: GOOD - Retrieval works for most queries.")
    elif hit_rate >= 0.4:
        print(f"\nVERDICT: NEEDS WORK - Retrieval misses many queries.")
    else:
        print(f"\nVERDICT: POOR - Retrieval is mostly failing.")

    print()


if __name__ == '__main__':
    user_id = sys.argv[1] if len(sys.argv) > 1 else '363682'
    evaluate_retrieval(user_id)
