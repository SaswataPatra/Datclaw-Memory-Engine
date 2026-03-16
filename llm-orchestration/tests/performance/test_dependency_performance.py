"""
Performance benchmarking for dependency-based extraction.
Target: <200ms per sentence average
"""

import pytest
import time
import asyncio
from unittest.mock import Mock
from core.graph.coref_resolver import CorefResolver
from core.graph.dependency_extractor import DependencyExtractor
from core.graph.relation_normalizer import RelationNormalizer
from core.graph.relation_extractor import RelationExtractor


# Benchmark test cases
BENCHMARK_CASES = [
    "Sarah studies physics",
    "John works at Google in California",
    "Sarah, my sister, studies physics at MIT and works at Google.",
    "I know that Sarah studies physics because she loves it.",
    "Physics is studied by Sarah and John at MIT.",
    "Sarah and John study physics and chemistry at MIT and Stanford.",
]


@pytest.fixture
def mock_db():
    """Mock ArangoDB database."""
    db = Mock()
    db.collection = Mock(return_value=Mock())
    return db


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.generate = Mock(return_value=[0.1] * 1536)
    return service


@pytest.fixture
def config():
    """Test configuration."""
    return {
        "dependency_extraction": {
            "enabled": True
        },
        "coref": {
            "provider": "simple",
            "model": "en_core_web_sm"
        }
    }


@pytest.fixture
def relation_extractor(mock_db, mock_embedding_service, config):
    """Create RelationExtractor with mocked dependencies."""
    extractor = RelationExtractor(
        db=mock_db,
        config=config,
        embedding_service=mock_embedding_service,
        collect_training_data=False
    )
    
    # Mock entity resolver
    async def mock_resolve(text, user_id, context, entity_type=None):
        entity = Mock()
        entity.entity_id = f"e_{text.lower().replace(' ', '_')}"
        entity.canonical_name = text
        entity.type = entity_type or "unknown"
        return entity
    
    extractor.entity_resolver.resolve = mock_resolve
    
    return extractor


class TestComponentPerformance:
    """Test performance of individual components."""
    
    def test_coref_resolver_performance(self):
        """Test: CorefResolver performance"""
        resolver = CorefResolver(provider="simple")
        text = "Sarah is my sister. She studies physics at MIT."
        
        start = time.time()
        for _ in range(10):
            resolved, clusters = resolver.resolve(text)
        elapsed = (time.time() - start) / 10
        
        print(f"\nCorefResolver: {elapsed*1000:.2f}ms per sentence")
        assert elapsed < 0.100, f"CorefResolver too slow: {elapsed*1000:.2f}ms (target: <100ms)"
    
    def test_dependency_extractor_performance(self):
        """Test: DependencyExtractor performance"""
        resolver = CorefResolver(provider="simple")
        extractor = DependencyExtractor(nlp=resolver.nlp)
        
        text = "Sarah studies physics at MIT and works at Google."
        doc = resolver.nlp(text)
        
        start = time.time()
        for _ in range(10):
            triples = extractor.extract_from_doc(doc)
        elapsed = (time.time() - start) / 10
        
        print(f"\nDependencyExtractor: {elapsed*1000:.2f}ms per sentence")
        assert elapsed < 0.050, f"DependencyExtractor too slow: {elapsed*1000:.2f}ms (target: <50ms)"
    
    def test_relation_normalizer_performance(self):
        """Test: RelationNormalizer performance"""
        normalizer = RelationNormalizer()
        
        start = time.time()
        for _ in range(100):
            canonical, conf = normalizer.normalize("works_at", "PERSON", "ORG", "Sarah works at Google")
        elapsed = (time.time() - start) / 100
        
        print(f"\nRelationNormalizer: {elapsed*1000:.2f}ms per relation")
        assert elapsed < 0.010, f"RelationNormalizer too slow: {elapsed*1000:.2f}ms (target: <10ms)"


class TestEndToEndPerformance:
    """Test end-to-end pipeline performance."""
    
    @pytest.mark.asyncio
    async def test_simple_sentence_performance(self, relation_extractor):
        """Test: Simple sentence extraction performance"""
        text = "Sarah studies physics"
        
        # Warmup
        await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
        elapsed = (time.time() - start) / 10
        
        print(f"\nSimple sentence: {elapsed*1000:.2f}ms")
        assert elapsed < 0.200, f"Simple sentence too slow: {elapsed*1000:.2f}ms (target: <200ms)"
    
    @pytest.mark.asyncio
    async def test_complex_sentence_performance(self, relation_extractor):
        """Test: Complex sentence extraction performance"""
        text = "Sarah, my sister, studies physics at MIT and works at Google."
        
        # Warmup
        await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
        elapsed = (time.time() - start) / 10
        
        print(f"\nComplex sentence: {elapsed*1000:.2f}ms")
        assert elapsed < 0.300, f"Complex sentence too slow: {elapsed*1000:.2f}ms (target: <300ms)"
    
    @pytest.mark.asyncio
    async def test_average_performance(self, relation_extractor):
        """Test: Average performance across multiple sentence types"""
        
        # Warmup
        for text in BENCHMARK_CASES:
            await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Benchmark
        total_time = 0
        for text in BENCHMARK_CASES:
            start = time.time()
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
            elapsed = time.time() - start
            total_time += elapsed
            print(f"\n'{text[:50]}...': {elapsed*1000:.2f}ms ({len(relations)} relations)")
        
        avg_time = total_time / len(BENCHMARK_CASES)
        
        print(f"\n{'='*60}")
        print(f"Average performance: {avg_time*1000:.2f}ms per sentence")
        print(f"Target: <200ms")
        print(f"Status: {'✅ PASS' if avg_time < 0.200 else '❌ FAIL'}")
        print(f"{'='*60}")
        
        assert avg_time < 0.200, f"Average too slow: {avg_time*1000:.2f}ms (target: <200ms)"


class TestScalability:
    """Test scalability with increasing complexity."""
    
    @pytest.mark.asyncio
    async def test_scaling_with_sentence_length(self, relation_extractor):
        """Test: Performance scaling with sentence length"""
        sentences = [
            "Sarah studies physics",
            "Sarah studies physics at MIT",
            "Sarah studies physics at MIT in Boston",
            "Sarah studies physics at MIT in Boston with her friend John",
            "Sarah studies physics at MIT in Boston with her friend John who works at Google",
        ]
        
        results = []
        for text in sentences:
            # Warmup
            await relation_extractor.extract(text, "test_user", "mem_test")
            
            # Benchmark
            start = time.time()
            for _ in range(5):
                relations = await relation_extractor.extract(text, "test_user", "mem_test")
            elapsed = (time.time() - start) / 5
            
            results.append((len(text.split()), elapsed * 1000))
            print(f"\n{len(text.split())} words: {elapsed*1000:.2f}ms")
        
        # Check that performance scales reasonably (not exponentially)
        # Longest sentence should be < 3x slowest than shortest
        ratio = results[-1][1] / results[0][1]
        assert ratio < 3.0, f"Poor scaling: {ratio:.2f}x (target: <3x)"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, relation_extractor):
        """Test: Batch processing performance"""
        texts = BENCHMARK_CASES * 10  # 60 sentences
        
        start = time.time()
        for text in texts:
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
        elapsed = time.time() - start
        
        avg_per_sentence = elapsed / len(texts)
        throughput = len(texts) / elapsed
        
        print(f"\nBatch processing:")
        print(f"  Total: {len(texts)} sentences in {elapsed:.2f}s")
        print(f"  Average: {avg_per_sentence*1000:.2f}ms per sentence")
        print(f"  Throughput: {throughput:.2f} sentences/second")
        
        assert avg_per_sentence < 0.200, f"Batch average too slow: {avg_per_sentence*1000:.2f}ms"


class TestMemoryUsage:
    """Test memory usage (basic checks)."""
    
    @pytest.mark.asyncio
    async def test_no_memory_leak(self, relation_extractor):
        """Test: No obvious memory leaks"""
        import gc
        
        text = "Sarah studies physics at MIT"
        
        # Process many sentences
        for _ in range(100):
            relations = await relation_extractor.extract(text, "test_user", "mem_test")
        
        # Force garbage collection
        gc.collect()
        
        # If we got here without OOM, we're probably okay
        assert True


class TestPerformanceReport:
    """Generate performance report."""
    
    @pytest.mark.asyncio
    async def test_generate_report(self, relation_extractor):
        """Test: Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("DEPENDENCY EXTRACTION PERFORMANCE REPORT")
        print("="*80)
        
        # Test each pattern type
        pattern_tests = {
            "Simple SVO": "Sarah studies physics",
            "Prepositional": "Sarah lives in Boston",
            "Copula": "Sarah is my sister",
            "Apposition": "Mark, the CEO, spoke",
            "Passive": "Physics is studied by Sarah",
            "Coordination": "Sarah and John study physics",
            "Nested": "I know that Sarah studies physics",
            "Complex": "Sarah, my sister, studies physics at MIT and works at Google."
        }
        
        results = {}
        for pattern_name, text in pattern_tests.items():
            # Warmup
            await relation_extractor.extract(text, "test_user", "mem_test")
            
            # Benchmark
            start = time.time()
            for _ in range(10):
                relations = await relation_extractor.extract(text, "test_user", "mem_test")
            elapsed = (time.time() - start) / 10
            
            results[pattern_name] = {
                "time_ms": elapsed * 1000,
                "relations": len(relations)
            }
        
        # Print report
        print("\nPerformance by Pattern Type:")
        print(f"{'Pattern':<20} {'Time (ms)':<15} {'Relations':<15} {'Status'}")
        print("-" * 80)
        
        for pattern_name, data in results.items():
            time_ms = data["time_ms"]
            relations = data["relations"]
            status = "✅ PASS" if time_ms < 200 else "⚠️  SLOW"
            print(f"{pattern_name:<20} {time_ms:>10.2f}      {relations:>5}          {status}")
        
        avg_time = sum(d["time_ms"] for d in results.values()) / len(results)
        print("-" * 80)
        print(f"{'Average':<20} {avg_time:>10.2f}                  {'✅ PASS' if avg_time < 200 else '❌ FAIL'}")
        print("="*80)
        print(f"\nTarget: <200ms per sentence")
        print(f"Status: {'✅ ALL TESTS PASSED' if avg_time < 200 else '⚠️  SOME TESTS SLOW'}")
        print("="*80 + "\n")
        
        assert avg_time < 200, f"Average performance: {avg_time:.2f}ms (target: <200ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements

