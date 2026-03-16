# DAPPY LLM Orchestration Service

**Version:** 1.0.0  
**Phase:** 1 MVP  
**Status:** 🚀 Core Implementation Complete (75%)

Human-cognition-inspired cognitive memory system with intelligent context management, time-aware ego scoring, and event-driven architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)

---

## 🎯 Overview

The LLM Orchestration Service is the cognitive layer of DAPPY, responsible for:

- **Intelligent Context Management**: Monitors LLM context usage and performs ego-based flushing
- **Time-Aware Ego Scoring**: Calculates multi-dimensional importance scores with temporal awareness
- **Async Consolidation**: Background processing with priority queues (HIGH/MED/LOW)
- **Event-Sourced Consistency**: Decoupled architecture with ArangoDB (canonical) ↔ Qdrant (vector index)
- **Raw Message Storage**: Hot (Redis), Long-term (ArangoDB/MongoDB), Cold (S3)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│  FastAPI REST API (Port 8001)               │
└──────────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────────┐
│  Context Memory Manager                      │
│  - Token counting (tiktoken)                 │
│  - Intelligent flush (50% threshold)         │
│  - Emergency flush (80% threshold)           │
│  - Ego-based prioritization                  │
└──────────────────────────────────────────────┘
              ↓ (flush to Tier 4)
┌──────────────────────────────────────────────┐
│  Redis Tier 4 (Hot Buffer - 10 min TTL)     │
└──────────────────────────────────────────────┘
              ↓ (publish event)
┌──────────────────────────────────────────────┐
│  Event Bus (Redis Streams)                   │
│  - memory.upsert                             │
│  - tier4.flush                               │
│  - contradiction.detected                    │
└──────────────────────────────────────────────┘
       ↓                    ↓
┌──────────────┐   ┌──────────────┐
│ Consolidation│   │  Consumers   │
│   Worker     │   │ - ArangoDB   │
│ (HIGH/MED/   │   │ - Qdrant     │
│  LOW queues) │   └──────────────┘
└──────────────┘
       ↓
┌─────────────────────────────────────────────┐
│  ArangoDB (Canonical)  +  Qdrant (Vectors) │
└─────────────────────────────────────────────┘
```

---

## 📦 Components

### **1. Event Bus (`core/event_bus.py`)**
- Abstract `EventBus` protocol for decoupling
- `RedisEventBus` with Streams + Consumer Groups
- `KafkaEventBus` placeholder (Phase 3)
- Dead letter queue for failed messages
- Idempotent event processing

### **2. Context Memory Manager (`services/context_manager.py`)**
- Monitors token usage with tiktoken
- Intelligent flush at 50% capacity
- Emergency flush at 80% capacity
- Ego-based prioritization (keep/summarize/drop)
- Stores messages to hot tier

### **3. Time-Aware Ego Scoring (`core/scoring/ego_scorer.py`)**
- `RecencyCalculator` with tier-specific half-lives
- `TemporalEgoScorer` with 9-feature formula
- Config-driven weights and thresholds
- Explainable component breakdown

### **4. Raw Message Storage (`core/messages/`, `adapters/`)**
- Hot: Redis Streams (14-day retention)
- Long-term: ArangoDB (Phase 1) / MongoDB (Phase 1.5)
- Cold: S3 NDJSON archives (Phase 1)
- Provenance: message → memory linking

### **5. Consolidation Worker (`workers/consolidation_worker.py`)**
- Fast worker: HIGH priority, <100ms target latency
- Batch worker: MED/LOW priority, 128 items per batch
- Spaced repetition triggers (1d, 3d, 7d, 30d)
- Backpressure handling with TTL bump

### **6. Event Consumers**
- **ArangoDB Consumer**: Writes memory nodes to canonical store
- **Qdrant Consumer**: Generates embeddings and indexes vectors
- Versioned writes for idempotency
- Error tracking and statistics

---

## 🚀 Installation

### **Prerequisites**

- Python 3.10+
- Redis 7.0+
- ArangoDB 3.10+
- Qdrant 1.7+
- OpenAI API key

### **Install Dependencies**

```bash
cd llm-orchestration
pip install -r requirements.txt
```

### **Environment Variables**

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
ARANGO_PASSWORD=your-password
MONGO_URI=mongodb://localhost:27017  # Phase 1.5
S3_KMS_KEY=your-kms-key  # For cold archive
```

---

## ⚙️ Configuration

Edit `config/base.yaml` to customize:

- **Context Memory**: Max tokens, flush thresholds, keep/summarize thresholds
- **Ego Scoring**: Feature weights, tier thresholds, recency half-lives
- **Consolidation**: Priority thresholds, worker concurrency, batch size
- **Message Storage**: Retention days, providers (Redis/Arango/Mongo/S3)
- **Event Bus**: Provider (redis/kafka), stream prefix, consumer groups

**Example:**

```yaml
context_memory:
  max_tokens: 128000
  flush_threshold: 0.5
  emergency_threshold: 0.8

ego_scoring:
  weights:
    recency_decay: 0.20
    explicit_importance: 0.20
    engagement: 0.12
  thresholds:
    tier1: 0.75  # Core memory
    tier2: 0.45  # Long-term
```

---

## 🌐 API Endpoints

### **Health Check**
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "dappy-llm-orchestrator",
  "version": "1.0.0",
  "timestamp": "2025-10-10T12:00:00Z",
  "components": {
    "redis": "healthy",
    "event_bus": "healthy",
    "consolidation_worker": "running",
    "arango_consumer": "running",
    "qdrant_consumer": "running"
  }
}
```

### **Context Management**
```bash
POST /context/manage
```

Request:
```json
{
  "user_id": "user_123",
  "session_id": "session_456",
  "conversation_history": [
    {
      "message_id": "msg_1",
      "role": "user",
      "content": "Tell me about Python",
      "timestamp": "2025-10-10T12:00:00Z",
      "sequence": 1
    }
  ],
  "new_message": {
    "message_id": "msg_2",
    "role": "user",
    "content": "What are decorators?",
    "timestamp": "2025-10-10T12:05:00Z",
    "sequence": 2
  }
}
```

Response:
```json
{
  "optimized_history": [...],
  "metadata": {
    "current_tokens": 2500,
    "max_tokens": 128000,
    "usage_percent": 0.0195,
    "flushed": false,
    "emergency_flush": false
  }
}
```

### **Ego Scoring**
```bash
POST /scoring/ego
```

Request:
```json
{
  "memory": {
    "memory_id": "mem_123",
    "content": "User is allergic to peanuts",
    "observed_at": "2025-10-10T12:00:00Z",
    "explicit_importance": 1.0,
    "sentiment_score": -0.5,
    "user_response_length": 50
  },
  "current_tier": "tier2"
}
```

Response:
```json
{
  "ego_score": 0.82,
  "tier": "tier1",
  "components": {
    "explicit_importance": 1.0,
    "recency_decay": 0.95,
    "frequency": 0.1,
    "sentiment_intensity": 0.5,
    "engagement": 0.5,
    "reference_count": 0.0,
    "confidence": 0.9,
    "source_weight": 1.0,
    "novelty": 0.8
  },
  "timestamp": "2025-10-10T12:00:00Z"
}
```

### **Consolidation Stats**
```bash
GET /consolidation/stats
```

Response:
```json
{
  "queues": {
    "high_queue": {"length": 5, "groups": 1},
    "med_queue": {"length": 120, "groups": 1},
    "low_queue": {"length": 300, "groups": 1},
    "total_backlog": 425
  },
  "arango_consumer": {
    "processed_count": 1500,
    "error_count": 2,
    "error_rate": 0.0013,
    "total_memories": 1498
  },
  "qdrant_consumer": {
    "processed_count": 1498,
    "error_count": 3,
    "embedding_errors": 1,
    "total_points": 1495
  }
}
```

---

## 🛠️ Development

### **Run Locally**

```bash
cd llm-orchestration
python -m uvicorn api.main:app --reload --port 8001
```

### **Access Documentation**

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### **Logging**

Logs are structured JSON format by default. Configure in `config/base.yaml`:

```yaml
monitoring:
  logging:
    level: "INFO"  # DEBUG | INFO | WARNING | ERROR
    format: "json"
    output: "stdout"
```

---

## 🧪 Testing

### **Run Unit Tests**

```bash
pytest tests/unit -v
```

### **Run Integration Tests**

```bash
pytest tests/integration -v
```

### **Test Coverage**

```bash
pytest --cov=. --cov-report=html
```

---

## 🚢 Deployment

### **Docker**

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### **Docker Compose**

```yaml
version: '3.8'

services:
  llm-orchestrator:
    build: .
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ARANGO_PASSWORD=${ARANGO_PASSWORD}
    depends_on:
      - redis
      - arangodb
      - qdrant
```

### **Kubernetes**

See `k8s/` directory for Kubernetes manifests (Phase 3).

---

## 📊 Monitoring

### **Prometheus Metrics**

Exposed at `:9090/metrics`:

- `context_flush_total`: Total context flushes
- `ego_score_distribution`: Histogram of ego scores
- `consolidation_queue_depth`: Queue depths by priority
- `embedding_generation_latency`: P50/P95/P99 latencies
- `arango_write_errors`: ArangoDB write errors
- `qdrant_index_errors`: Qdrant indexing errors

### **Grafana Dashboard**

Import `grafana/dashboard.json` for pre-built dashboard (Phase 1.5).

---

## 🔮 Roadmap

### **Phase 1 (Current)** ✅ 75% Complete
- ✅ Event Bus architecture
- ✅ Raw message storage
- ✅ Time-aware ego scoring
- ✅ Context memory manager
- ✅ Async consolidation
- ✅ Event-sourced consistency
- 🚧 Temporal contradiction detection
- 🚧 Shadow tier (Tier 0.5)
- 🚧 Unit & integration tests

### **Phase 1.5 (Hardening)** 🔜
- ML-based ego scoring (LightGBM)
- MongoDB adapter for messages
- Load testing & benchmarking
- Reconciliation job (ArangoDB ↔ Qdrant)

### **Phase 2 (Advanced Features)** 📅
- Curiosity engine (Restless Node + CuriousLLM)
- Reflection mechanism
- Periodicity detection
- Timeline UI

### **Phase 3 (Scale)** 🚀
- Kafka event bus
- HippoRAG2 retrieval
- Multi-region deployment

---

## 📄 License

See root `LICENSE.md`

---

## 👥 Contributing

This is a private research project. For questions, contact the maintainer.

---

**Built with ❤️ for human-cognition-inspired AGI memory.**

