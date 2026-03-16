# Architecture Overview

## Core Design Principle

**Vector search handles semantic retrieval. Knowledge graph provides structured context for LLM reasoning.**

## System Components

### 1. Ingestion Pipeline

```
User Message
    ↓
Memory Creation + Ego Scoring
    ↓
Entity & Relation Extraction (LLM)
    ↓
Embedding Generation (OpenAI)
    ↓
Store in ArangoDB + Qdrant
    ↓
Background: KG Maintenance
```

### 2. Retrieval Pipeline

```
User Query
    ↓
Vector Search (Qdrant)
    ↓
Fetch Relations (ArangoDB)
    ↓
Combine: Memories + Relations
    ↓
Feed to LLM
    ↓
Generate Response
```

## Data Stores

- **Redis**: Event bus, hot tier (temporary storage)
- **ArangoDB**: Memories, entities, relations (canonical store)
- **Qdrant**: Memory embeddings (vector search)

## Key Services

- **ChatbotService**: User-facing chat interface
- **IngestionService**: Processes and stores memories
- **ConsolidationService**: Extracts entities, relations, context summaries
- **ContextManager**: Retrieves relevant memories for queries
- **KGMaintenanceAgent**: Detects contradictions, reinforces relations

## Workers (Background)

- **ConsolidationWorker**: Processes consolidation queue
- **KGMaintenanceWorker**: Maintains knowledge graph integrity
- **ArangoConsumer**: Persists memories to ArangoDB
- **QdrantConsumer**: Stores embeddings in Qdrant

## Why This Architecture?

1. **Vector search is semantic**: Full sentence embeddings capture meaning better than entity-level search
2. **Relations are context**: The knowledge graph gives LLM structured facts to reason with
3. **Background processing**: Heavy work (entity extraction, KG maintenance) doesn't block user responses
4. **Event-driven**: Redis Streams enable async, scalable processing
