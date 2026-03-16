"""
DAPPY Prometheus Metrics

Centralized metrics for monitoring retrieval, ingestion, scoring, and storage.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# ── Retrieval ──
RETRIEVAL_LATENCY = Histogram(
    'dappy_retrieval_seconds',
    'Memory retrieval latency in seconds',
    ['method'],  # ppr, fallback, vector
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

RETRIEVAL_RESULTS = Counter(
    'dappy_retrieval_results_total',
    'Total retrieval results',
    ['method', 'outcome']  # outcome: hit, miss
)

RETRIEVAL_MEMORIES_RETURNED = Histogram(
    'dappy_retrieval_memories_count',
    'Number of memories returned per retrieval',
    buckets=[0, 1, 2, 3, 5, 10, 20]
)

# ── Ingestion ──
INGESTION_DURATION = Histogram(
    'dappy_ingestion_seconds',
    'Memory ingestion total duration',
    ['source_type'],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

INGESTION_CHUNKS = Counter(
    'dappy_ingestion_chunks_total',
    'Total chunks processed during ingestion',
    ['source_type', 'outcome']  # outcome: success, error
)

# ── Scoring ──
SCORING_LATENCY = Histogram(
    'dappy_scoring_seconds',
    'Memory scoring latency',
    ['scorer'],  # ml, trigger, ego
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

EGO_SCORE_DISTRIBUTION = Histogram(
    'dappy_ego_score',
    'Distribution of ego scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

TIER_ASSIGNMENT = Counter(
    'dappy_tier_assignments_total',
    'Memory tier assignments',
    ['tier']
)

# ── Storage ──
STORAGE_OPERATIONS = Counter(
    'dappy_storage_ops_total',
    'Storage operations',
    ['store', 'operation', 'outcome']  # store: arango/qdrant/redis, outcome: success/error
)

STORAGE_LATENCY = Histogram(
    'dappy_storage_seconds',
    'Storage operation latency',
    ['store', 'operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# ── Graph ──
GRAPH_ENTITIES = Gauge(
    'dappy_graph_entities',
    'Number of entities in the graph',
    ['user_id']
)

GRAPH_EDGES = Gauge(
    'dappy_graph_edges',
    'Number of edges in the graph',
    ['user_id']
)

GRAPH_EXTRACTION_LATENCY = Histogram(
    'dappy_graph_extraction_seconds',
    'Graph extraction latency per memory',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# ── Chat ──
CHAT_LATENCY = Histogram(
    'dappy_chat_seconds',
    'End-to-end chat response latency',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

CHAT_REQUESTS = Counter(
    'dappy_chat_requests_total',
    'Total chat requests',
    ['outcome']  # success, error
)

# ── System ──
ACTIVE_USERS = Gauge('dappy_active_users', 'Number of active users')
TOTAL_MEMORIES = Gauge('dappy_total_memories', 'Total memories stored')
SERVICE_INFO = Info('dappy_service', 'Service version and configuration')
