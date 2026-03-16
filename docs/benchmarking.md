# Benchmarking with MemoryBench

Datclaw can be benchmarked using [MemoryBench](https://github.com/supermemoryai/memorybench), a standardized evaluation framework for memory systems.

## Setup

### 1. Install MemoryBench

```bash
cd /path/to/your/workspace
git clone https://github.com/supermemoryai/memorybench.git
cd memorybench
bun install
```

### 2. Configure DAPPY Provider

MemoryBench includes a DAPPY provider out of the box. Make sure your Datclaw instance is running:

```bash
# In Datclaw-Memory-Engine directory
docker-compose up -d
cd llm-orchestration && ./start_service.sh
```

### 3. Run Benchmark

```bash
cd memorybench

# Run with default concurrency
bun run --provider dappy

# Run with higher concurrency (faster)
bun run --provider dappy --concurrency-ingest 5 --concurrency-search 10 --concurrency-answer 10
```

## What Gets Tested

MemoryBench evaluates:

1. **Ingestion**: Can the system ingest conversation data?
2. **Search**: Can it retrieve relevant memories for a query?
3. **Answer**: Can it answer questions using retrieved memories?
4. **Evaluate**: Are the answers correct?

## Datasets

MemoryBench includes multiple datasets:

- **LoCoMo-10**: 10 questions, ~200 chunks per question
- **LoCoMo-25**: 25 questions (full dataset)
- **Custom datasets**: You can add your own

## Performance Tips

### Ingestion Speed

Datclaw uses batched consolidation for 10x faster ingestion:

- Default: ~10-15 minutes for 4000 chunks
- With `--concurrency-ingest 5`: Processes 5 questions in parallel

### Rate Limits

OpenAI API has rate limits:
- **TPM** (tokens per minute): ~200k for tier 1
- **RPD** (requests per day): 10,000 for tier 1

For large benchmarks (25 questions = ~10,000 LLM calls), you may need:
- Multiple API keys (rotate them)
- Higher tier OpenAI account
- Or run smaller datasets first

### Reducing LLM Calls

To avoid rate limits during testing, you can temporarily disable LLM-based classification:

```yaml
# llm-orchestration/config/base.yaml
ml:
  classifier_type: "regex"  # Uses pattern matching instead of LLM
```

This reduces LLM calls by ~50% during ingestion.

## Interpreting Results

MemoryBench outputs accuracy metrics:

```json
{
  "accuracy": 0.85,
  "questions_passed": 17,
  "questions_total": 20,
  "details": [...]
}
```

Good targets:
- **Single-hop questions** (direct facts): 90%+ accuracy
- **Multi-hop questions** (reasoning): 70%+ accuracy
- **Temporal questions** (time-based): 80%+ accuracy

## Troubleshooting

### "No results found" for queries

- Check ego scores: Low scores may filter out relevant memories
- Check tier thresholds: Adjust in `config/base.yaml`
- Check embeddings: Verify Qdrant has vectors

### Slow ingestion

- Increase `--concurrency-ingest`
- Check OpenAI rate limits
- Monitor logs for bottlenecks

### Rate limit errors

- Use multiple API keys
- Reduce concurrency
- Switch to `classifier_type: "regex"` temporarily

## More Information

See the [MemoryBench documentation](https://github.com/supermemoryai/memorybench) for full details on providers, datasets, and evaluation metrics.
