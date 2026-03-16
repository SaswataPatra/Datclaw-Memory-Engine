"""
DAPPY LLM Orchestration Service - FastAPI Application
Main API entry point
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import redis.asyncio as redis

from config import load_config
from core.event_bus import EventBusFactory
from core.scoring.ego_scorer import TemporalEgoScorer
from services.context_manager import ContextMemoryManager
from adapters.redis_message_bus import RedisMessageBus
from adapters.arango_message_store import ArangoMessageStore
from workers.consolidation_worker import ConsolidationWorker
from workers.arango_consumer import ArangoDBConsumer
from workers.qdrant_consumer import QdrantConsumer
# ContradictionDetector: LLM calls stripped, now only used by KG Maintenance Agent
# Initialized there, not here
# from core.contradiction_detector import TemporalContradictionDetector
from core.shadow_tier import ShadowTier
from llm.providers.factory import LLMProviderFactory
from services.chatbot_service import ChatbotService
from services.entity_extraction import EntityExtractionService
from api.auth import (
    AuthManager, set_auth_manager, get_current_user,
    SignupRequest, LoginRequest, TokenResponse, UserResponse,
)

# Configure logging: console + rotating file
import logging.handlers
import os

_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)

_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(level=_log_level, format=_log_format)

# Rotating file handler: 10MB per file, keep 5 backups
_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_log_dir, 'dappy.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
_file_handler.setLevel(_log_level)
_file_handler.setFormatter(logging.Formatter(_log_format))
logging.getLogger().addHandler(_file_handler)

# Separate error log for quick triage
_error_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_log_dir, 'error.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=3,
    encoding='utf-8'
)
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(logging.Formatter(_log_format))
logging.getLogger().addHandler(_error_handler)

# Reduce noise from httpx (OpenAI API calls)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized: level={_log_level}, dir={_log_dir}")

# Load configuration
config = load_config()

# Create FastAPI app
app = FastAPI(
    title="DAPPY LLM Orchestration Service",
    description="Cognitive memory system with intelligent context management",
    version="1.0.0"
)

# Prometheus metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialized on startup)
redis_client = None
event_bus = None
ego_scorer = None
context_manager = None
message_store = None
consolidation_worker = None
arango_consumer = None
kg_maintenance_worker = None
qdrant_consumer = None
shadow_tier = None
chatbot_service = None
auth_manager = None
ingestion_service = None
service_ready = False


# Pydantic models
class Message(BaseModel):
    message_id: str
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str
    timestamp: Optional[str] = None
    sequence: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ContextRequest(BaseModel):
    user_id: str
    session_id: str
    conversation_history: List[Message]
    new_message: Optional[Message] = None


class ContextResponse(BaseModel):
    optimized_history: List[Message]
    metadata: Dict[str, Any]


class EgoScoreRequest(BaseModel):
    memory: Dict[str, Any]
    current_tier: Optional[str] = None


class EgoScoreResponse(BaseModel):
    ego_score: float
    tier: str
    components: Dict[str, float]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    components: Dict[str, str]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    debug: bool = False


class ChatResponse(BaseModel):
    assistant_message: str
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    source_type: str = Field(..., description="Type of source: 'chatgpt', 'zip', 'text'")
    source: str = Field(..., description="Source identifier (URL, file path, etc.)")
    session_id: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    chunks_parsed: int
    memories_created: int
    errors: List[str]
    session_id: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup"""
    global redis_client, event_bus, ego_scorer, context_manager, message_store
    global consolidation_worker, arango_consumer, qdrant_consumer
    global shadow_tier, chatbot_service, auth_manager, ingestion_service
    
    logger.info("Starting DAPPY LLM Orchestration Service...")
    
    try:
        # Initialize Redis
        redis_config = config.get_section('redis')
        redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config.get('password'),
            decode_responses=True
        )
        
        # Test Redis connection
        await redis_client.ping()
        logger.info("Redis connected")
        
        # Initialize Event Bus
        event_bus_config = config.get_section('event_bus')
        event_bus = EventBusFactory.create(
            provider=event_bus_config['provider'],
            redis_client=redis_client,
            config=event_bus_config.get('redis', {})
        )
        logger.info("Event Bus initialized")
        
        # Initialize Ego Scorer
        ego_scorer = TemporalEgoScorer(config.all)
        logger.info("Ego Scorer initialized")
        
        # Initialize Message Store
        message_storage_config = config.get_section('message_storage')
        message_store = RedisMessageBus(redis_client, message_storage_config['hot'])
        logger.info("Message Store initialized")
        
        # Initialize database clients (needed by Context Manager for PPR)
        from arango import ArangoClient
        from qdrant_client import QdrantClient
        
        arango_config = config.get_section('arangodb')
        arango_client = ArangoClient(hosts=arango_config['url'])
        
        # Connect to _system database first to create database if needed
        sys_db = arango_client.db(
            '_system',
            username=arango_config['username'],
            password=arango_config['password']
        )
        
        # Create database if it doesn't exist
        if not sys_db.has_database(arango_config['database']):
            sys_db.create_database(arango_config['database'])
            logger.info(f"Created database: {arango_config['database']}")
        
        # Now connect to the application database
        arango_db = arango_client.db(
            arango_config['database'],
            username=arango_config['username'],
            password=arango_config['password']
        )
        
        qdrant_config = config.get_section('qdrant')
        qdrant_client = QdrantClient(url=qdrant_config['url'])
        
        # Initialize Auth Manager
        auth_config = config.get_section('auth')
        auth_manager = AuthManager(
            arango_db=arango_db,
            secret_key=auth_config.get('secret_key'),
            algorithm=auth_config.get('algorithm'),
            token_expire_minutes=auth_config.get('token_expire_minutes'),
        )
        set_auth_manager(auth_manager)
        logger.info("Auth Manager initialized")
        
        # Initialize Embedding Service (needed for context manager vector search)
        from services.embedding_service import EmbeddingService
        llm_config = config.get_section('llm')
        openai_api_key = llm_config.get('openai', {}).get('api_key')
        embedding_service = None
        if openai_api_key:
            embedding_service = EmbeddingService(api_key=openai_api_key)
            logger.info("Embedding Service initialized")
        else:
            logger.warning("OpenAI API key not found, embedding/vector search will be disabled")
        
        # Initialize Query Understanding (LLM-based entity extraction)
        from services.query_understanding import QueryUnderstandingService
        query_understanding_service = None
        if openai_api_key:
            query_understanding_service = QueryUnderstandingService(api_key=openai_api_key)
            logger.info("Query Understanding Service initialized")
        
        # Initialize Knowledge Graph Store
        from core.knowledge_graph_store import KnowledgeGraphStore
        knowledge_graph_store = KnowledgeGraphStore(db=arango_db, config=config.all)
        logger.info("Knowledge Graph Store initialized")
        
        # Initialize Consolidation Service
        from services.consolidation_service import ConsolidationService
        consolidation_service = ConsolidationService(api_key=openai_api_key)
        logger.info("Consolidation Service initialized")
        
        # Initialize Context Manager with vector search + KG relations
        context_manager = ContextMemoryManager(
            redis_client=redis_client,
            config=config.all,
            ego_scorer=ego_scorer,
            event_bus=event_bus,
            message_store=message_store,
            arango_db=arango_db,  # For memory storage
            qdrant_client=qdrant_client,  # For vector search
            embedding_service=embedding_service,  # For query embedding
            knowledge_graph_store=knowledge_graph_store,  # For relation context
            query_understanding_service=query_understanding_service  # For query entity extraction
        )
        logger.info("Context Manager initialized with vector search + KG boosting")
        
        # Initialize Consolidation Worker
        
        consolidation_worker = ConsolidationWorker(
            redis_client=redis_client,
            arango_client=arango_client,
            qdrant_client=qdrant_client,
            config=config.all,
            ego_scorer=ego_scorer,
            event_bus=event_bus
        )
        
        await consolidation_worker.start()
        logger.info("Consolidation Worker started")
        
        # Initialize ArangoDB Consumer
        arango_consumer = ArangoDBConsumer(config.all, event_bus)
        await arango_consumer.start()
        logger.info("ArangoDB Consumer started")
        
        # Initialize Qdrant Consumer
        qdrant_consumer = QdrantConsumer(config.all, event_bus)
        await qdrant_consumer.start()
        logger.info("Qdrant Consumer started")

        # ContradictionDetector stripped of LLM calls, now initialized inside KG Maintenance Agent only
        
        # Initialize Shadow Tier
        shadow_tier = ShadowTier(
            redis_client=redis_client,
            config=config.all,
            event_bus=event_bus
        )
        logger.info("Shadow Tier initialized")
        
        # Initialize LLM Provider and Chatbot Service
        llm_provider = LLMProviderFactory.from_config(llm_config)
        logger.info(f"LLM Provider initialized: {llm_provider.name}")
        
        # Entity extraction (kept for query understanding only, not used in ingestion)
        entity_extraction_service = None
        if openai_api_key:
            entity_extraction_service = EntityExtractionService(api_key=openai_api_key)
            logger.info("Entity Extraction Service initialized (for query understanding)")
        
        # Background consolidation: DISABLED (files kept for future reference)
        # Reason: Over-complex. Vector + entity expansion is sufficient for now.
        # To re-enable: uncomment below and wire into chatbot/ingestion services
        background_consolidation = None
        # from services.background_consolidation import BackgroundConsolidationService
        # from core.graph.edge_store import CandidateEdgeStore, ThoughtEdgeStore
        # from core.graph.activation_scorer import ActivationScorer
        # candidate_edge_store = CandidateEdgeStore(db=arango_db)
        # thought_edge_store = ThoughtEdgeStore(db=arango_db)
        # activation_scorer = ActivationScorer(config=config.all, db=arango_db, embedding_service=embedding_service)
        # background_consolidation = BackgroundConsolidationService(...)
        logger.info("Graph consolidation: DISABLED (vector + entity expansion only)")
        
        chatbot_config = config.get_section('chatbot')
        chatbot_service = ChatbotService(
            llm_provider=llm_provider,
            context_manager=context_manager,
            ego_scorer=ego_scorer,
            event_bus=event_bus,
            shadow_tier=shadow_tier,
            system_prompt=chatbot_config.get('system_prompt'),
            config=config.all,
            qdrant_client=qdrant_client,
            embedding_service=embedding_service,
            use_ml_scoring=True,
            use_distilbert=False,  # Use classifier_type from config instead
            consolidation_service=consolidation_service,
            knowledge_graph_store=knowledge_graph_store,
        )
        logger.info("Chatbot Service initialized with ML scoring + KG consolidation")
        
        # Load DistilBERT in thread pool if enabled (to avoid blocking event loop)
        if chatbot_service.classifier_type != "regex":
            logger.info(f"Loading {chatbot_service.classifier_type} classifier in background thread...")
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            
            try:
                await loop.run_in_executor(
                    executor,
                    chatbot_service._init_classifier_sync
                )
                logger.info(f"✅ {chatbot_service.classifier_type.capitalize()} classifier loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load {chatbot_service.classifier_type} classifier: {e}", exc_info=True)
            finally:
                executor.shutdown(wait=False)
        
        # Wire up chatbot_service to context_manager (for processing state tracking)
        context_manager.set_chatbot_service(chatbot_service)
        
        # Initialize Ingestion Service
        from services.ingestion import IngestionService
        from services.ingestion.chatgpt_parser import ChatGPTShareParser
        from services.ingestion.session_json_parser import SessionJsonParser
        from services.classification.regex_fallback import RegexFallback
        
        regex_fallback = RegexFallback()
        ingestion_service = IngestionService(
            event_bus=event_bus,
            ego_scorer=ego_scorer,
            ml_scorer=chatbot_service.ml_scorer if hasattr(chatbot_service, 'ml_scorer') else None,
            graph_pipeline=None,  # Disabled
            regex_fallback=regex_fallback,
            consolidation_service=consolidation_service,
            knowledge_graph_store=knowledge_graph_store,
            background_consolidation=None,  # Disabled
            shadow_tier=shadow_tier,
            config=config.all
        )
        
        # Register parsers
        ingestion_service.register_parser("chatgpt", ChatGPTShareParser())
        ingestion_service.register_parser("session_json", SessionJsonParser())
        logger.info("Ingestion Service initialized with ChatGPT + session_json parsers")
        
        # Initialize Contradiction Signal Detector (lightweight, no LLM)
        from core.contradiction_detector import ContradictionSignalDetector
        contradiction_signal_detector = ContradictionSignalDetector(
            qdrant_client=qdrant_client,
            config=config.all
        )
        logger.info("Contradiction Signal Detector initialized (signals only, no LLM)")

        # Initialize KG Maintenance Agent and Worker
        from services.kg_maintenance_agent import KGMaintenanceAgent
        from workers.kg_maintenance_worker import KGMaintenanceWorker

        kg_maintenance_agent = KGMaintenanceAgent(
            knowledge_graph_store=knowledge_graph_store,
            api_key=openai_api_key,
            contradiction_signal_detector=contradiction_signal_detector
        )
        logger.info("KG Maintenance Agent initialized (with contradiction signals)")
        
        kg_maintenance_worker = KGMaintenanceWorker(
            event_bus=event_bus,
            arango_db=arango_db,
            kg_maintenance_agent=kg_maintenance_agent,
            config=config.all
        )
        await kg_maintenance_worker.start()
        logger.info("KG Maintenance Worker started")
        
        # Initialize KG Re-consolidation Service and Worker
        from services.kg_reconsolidation_service import KGReconsolidationService
        from workers.kg_reconsolidation_worker import KGReconsolidationWorker
        
        global kg_reconsolidation_service, kg_reconsolidation_worker
        kg_reconsolidation_service = KGReconsolidationService(
            knowledge_graph_store=knowledge_graph_store,
            consolidation_service=consolidation_service,
            arango_db=arango_db
        )
        logger.info("KG Re-consolidation Service initialized")
        
        kg_reconsolidation_worker = KGReconsolidationWorker(
            reconsolidation_service=kg_reconsolidation_service,
            arango_db=arango_db,
            interval_hours=24  # Run daily
        )
        logger.info("KG Re-consolidation Worker initialized (manual trigger only)")
        
        # Mark service as ready
        global service_ready
        service_ready = True
        
        logger.info("✅ All services started successfully and ready to accept requests")
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}", exc_info=True)
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DAPPY LLM Orchestration Service...")
    
    try:
        # Shutdown ML executor (wait for pending predictions to complete)
        from ml.utils import shutdown_global_executor
        shutdown_global_executor(wait=True)
        logger.info("ML Executor stopped")
        
        if consolidation_worker:
            await consolidation_worker.stop()
        
        if arango_consumer:
            await arango_consumer.stop()
        
        if qdrant_consumer:
            await qdrant_consumer.stop()
        
        if kg_maintenance_worker:
            await kg_maintenance_worker.stop()
        
        if event_bus:
            await event_bus.close()
        
        if redis_client:
            await redis_client.close()
        
        logger.info("✅ All services stopped successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# ============================================================================
# AUTH ENDPOINTS (public – no token required)
# ============================================================================

@app.post("/auth/signup", response_model=TokenResponse)
async def signup(request: SignupRequest):
    """Register a new user and return a JWT token."""
    from api.auth import get_auth_manager
    mgr = get_auth_manager()
    result = mgr.signup(request)
    return TokenResponse(**result)


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate and return a JWT token."""
    from api.auth import get_auth_manager
    mgr = get_auth_manager()
    result = mgr.login(request)
    return TokenResponse(**result)


@app.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return UserResponse(**current_user)


@app.post("/user/ingest", response_model=IngestResponse)
async def ingest_memories(request: IngestRequest, current_user: dict = Depends(get_current_user)):
    """
    Ingest memories from external sources (requires authentication).
    
    Supports:
    - ChatGPT shared conversation links
    - More formats coming soon (ZIP archives, plain text, etc.)
    """
    import time as _time
    from core.metrics import INGESTION_DURATION, INGESTION_CHUNKS
    
    _start = _time.time()
    try:
        user_id = current_user["user_id"]
        
        result = await ingestion_service.ingest(
            user_id=user_id,
            source_type=request.source_type,
            source=request.source,
            session_id=request.session_id
        )
        
        INGESTION_DURATION.labels(source_type=request.source_type).observe(_time.time() - _start)
        INGESTION_CHUNKS.labels(source_type=request.source_type, outcome='success').inc(result.get('memories_created', 0))
        if result.get('errors'):
            INGESTION_CHUNKS.labels(source_type=request.source_type, outcome='error').inc(len(result['errors']))
        
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/user/memories")
async def delete_all_user_memories(current_user: dict = Depends(get_current_user)):
    """Delete all memories, entities, and edges for the current user (temporary dev feature)."""
    try:
        user_id = current_user["user_id"]
        
        from arango import ArangoClient
        from qdrant_client import QdrantClient
        
        arango_config = config.get_section('arangodb')
        arango_client = ArangoClient(hosts=arango_config['url'])
        arango_db = arango_client.db(
            arango_config['database'],
            username=arango_config['username'],
            password=arango_config['password']
        )
        
        qdrant_config = config.get_section('qdrant')
        qdrant_client = QdrantClient(url=qdrant_config['url'])
        
        # Delete from ArangoDB
        deleted_counts = {}
        
        # Delete memories
        mem_result = arango_db.aql.execute(
            "FOR m IN memories FILTER m.user_id == @user_id REMOVE m IN memories RETURN OLD",
            bind_vars={'user_id': user_id}
        )
        deleted_counts['memories'] = len(list(mem_result))
        
        # Delete entities (legacy collection)
        ent_result = arango_db.aql.execute(
            "FOR e IN entities FILTER e.user_id == @user_id REMOVE e IN entities RETURN OLD",
            bind_vars={'user_id': user_id}
        )
        deleted_counts['entities'] = len(list(ent_result))
        
        # Delete candidate edges (legacy collection)
        edge_result = arango_db.aql.execute(
            "FOR e IN candidate_edges FILTER e.user_id == @user_id REMOVE e IN candidate_edges RETURN OLD",
            bind_vars={'user_id': user_id}
        )
        deleted_counts['edges'] = len(list(edge_result))
        
        # Delete entity relations (new KG collection)
        rel_result = arango_db.aql.execute(
            "FOR r IN entity_relations FILTER r.user_id == @user_id REMOVE r IN entity_relations RETURN OLD",
            bind_vars={'user_id': user_id}
        )
        deleted_counts['kg_relations'] = len(list(rel_result))
        
        # Delete from Qdrant
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        try:
            qdrant_client.delete(
                collection_name="memories",
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                )
            )
            deleted_counts['qdrant_vectors'] = 'deleted'
        except Exception as e:
            logger.warning(f"Failed to delete from Qdrant: {e}")
            deleted_counts['qdrant_vectors'] = 'failed'
        
        # Delete from Redis (hot memories)
        try:
            pattern = f"*{user_id}*"
            cursor = 0
            redis_deleted = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis_client.delete(*keys)
                    redis_deleted += len(keys)
                if cursor == 0:
                    break
            deleted_counts['redis_keys'] = redis_deleted
        except Exception as e:
            logger.warning(f"Failed to delete from Redis: {e}")
            deleted_counts['redis_keys'] = 'failed'
        
        logger.info(f"🗑️  Deleted all data for user {user_id}: {deleted_counts}")
        
        return {
            "status": "success",
            "message": f"All memories deleted for user {user_id}",
            "deleted": deleted_counts
        }
        
    except Exception as e:
        logger.error(f"Error deleting user memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# KG RE-CONSOLIDATION ENDPOINTS
# ============================================================================

@app.post("/admin/kg/reconsolidate/{user_id}")
async def trigger_kg_reconsolidation(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Trigger KG re-consolidation for a specific user.
    
    This runs:
    1. Merge duplicate relations
    2. Re-process high-tier memories without relations
    3. Decay stale relations
    """
    try:
        if not service_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        result = await kg_reconsolidation_service.run_full_reconsolidation(user_id)
        
        return {
            "status": "success",
            "message": f"KG re-consolidation completed for user {user_id}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error running KG re-consolidation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/kg/reconsolidate/all")
async def trigger_kg_reconsolidation_all(current_user: dict = Depends(get_current_user)):
    """
    Trigger KG re-consolidation for all users.
    """
    try:
        if not service_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        results = await kg_reconsolidation_worker.run_for_all_users()
        
        return {
            "status": "success",
            "message": f"KG re-consolidation completed for {len(results)} users",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error running KG re-consolidation for all users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/kg/merge-duplicates/{user_id}")
async def merge_duplicate_relations(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Merge duplicate relations for a specific user.
    """
    try:
        if not service_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        result = await kg_reconsolidation_service.merge_duplicate_relations(user_id)
        
        return {
            "status": "success",
            "message": f"Merged duplicate relations for user {user_id}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error merging duplicate relations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/kg/reprocess-unextracted/{user_id}")
async def reprocess_unextracted_memories(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Re-process high-tier memories without relations for a specific user.
    """
    try:
        if not service_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        result = await kg_reconsolidation_service.reprocess_unextracted_memories(user_id)
        
        return {
            "status": "success",
            "message": f"Re-processed unextracted memories for user {user_id}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error re-processing unextracted memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/kg/decay-stale/{user_id}")
async def decay_stale_relations(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Decay stale relations for a specific user.
    """
    try:
        if not service_ready:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        result = await kg_reconsolidation_service.decay_stale_relations(user_id)
        
        return {
            "status": "success",
            "message": f"Decayed stale relations for user {user_id}",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error decaying stale relations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BENCHMARK ENDPOINTS (MemoryBench integration - no auth for local testing)
# ============================================================================

class BenchmarkIngestRequest(BaseModel):
    """Request body for benchmark ingest."""
    user_id: str = Field(..., description="User ID (containerTag from MemoryBench)")
    sessions: List[Dict[str, Any]] = Field(..., description="UnifiedSession format from MemoryBench")


class BenchmarkIngestResponse(BaseModel):
    """Response for benchmark ingest."""
    status: str
    chunks_parsed: int
    memories_created: int
    errors: List[str]
    document_ids: List[str]


class BenchmarkSearchRequest(BaseModel):
    """Request body for benchmark search."""
    user_id: str = Field(..., description="User ID (containerTag from MemoryBench)")
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Max results to return")


@app.post("/benchmark/ingest", response_model=BenchmarkIngestResponse)
async def benchmark_ingest(request: BenchmarkIngestRequest):
    """
    Ingest sessions for MemoryBench. No auth required (local benchmark use).
    """
    try:
        import json
        sessions_json = json.dumps(request.sessions)
        result = await ingestion_service.ingest(
            user_id=request.user_id,
            source_type="session_json",
            source=sessions_json,
            session_id=f"benchmark_{request.user_id}",
        )
        doc_ids = [f"mem_{i}" for i in range(result.get("memories_created", 0))]
        return BenchmarkIngestResponse(
            status=result.get("status", "success"),
            chunks_parsed=result.get("chunks_parsed", 0),
            memories_created=result.get("memories_created", 0),
            errors=result.get("errors", []),
            document_ids=doc_ids,
        )
    except Exception as e:
        logger.error(f"Benchmark ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark/search")
async def benchmark_search(request: BenchmarkSearchRequest):
    """
    Search memories for MemoryBench. No auth required (local benchmark use).
    Returns memories in format compatible with MemoryBench answer phase.
    """
    try:
        memories = await context_manager.retrieve_relevant_memories(
            user_id=request.user_id,
            query=request.query,
            max_memories=request.limit,
            use_ppr=True,
            use_vector=True,
        )
        def _extract_temporal_and_context(m):
            """Extract temporal metadata and context summary from memory."""
            meta = m.get("metadata", {})
            temporal = meta.get("temporal", {})
            
            # Document date (when user said it)
            document_date = m.get("observed_at", "")
            if not document_date:
                ingestion = meta.get("ingestion_metadata", {})
                document_date = ingestion.get("iso_date", "") or ingestion.get("formatted_date", "")
            
            # Event dates (resolved from temporal extraction)
            event_dates = temporal.get("event_dates", [])
            time_expressions = temporal.get("time_expressions", [])
            
            # Context summary (factual summary of assistant response)
            context_summary = meta.get("context_summary", "")
            
            return {
                "document_date": document_date,
                "event_dates": event_dates,
                "time_expressions": time_expressions,
                "context_summary": context_summary,
            }

        return {
            "results": [
                {
                    "content": m.get("content", ""),
                    "memory": m.get("content", ""),
                    "chunk": m.get("content", ""),
                    "metadata": {
                        "ego_score": m.get("ego_score"),
                        "tier": m.get("tier"),
                        "relevance_score": m.get("relevance_score"),
                        "source": m.get("source", "vector_search"),
                        **_extract_temporal_and_context(m),
                    },
                }
                for m in memories
            ],
        }
    except Exception as e:
        logger.error(f"Benchmark search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/benchmark/clear/{user_id}")
async def benchmark_clear(user_id: str):
    """
    Clear all memories for a user (benchmark cleanup). No auth required.
    """
    try:
        from arango import ArangoClient
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        arango_config = config.get_section('arangodb')
        arango_client = ArangoClient(hosts=arango_config['url'])
        arango_db = arango_client.db(
            arango_config['database'],
            username=arango_config['username'],
            password=arango_config['password'],
        )
        qdrant_config = config.get_section('qdrant')
        qdrant_client = QdrantClient(url=qdrant_config['url'])

        deleted = {}
        mem_result = list(arango_db.aql.execute(
            "FOR m IN memories FILTER m.user_id == @uid REMOVE m IN memories RETURN OLD",
            bind_vars={"uid": user_id},
        ))
        deleted["memories"] = len(mem_result)

        rel_result = list(arango_db.aql.execute(
            "FOR r IN entity_relations FILTER r.user_id == @uid REMOVE r IN entity_relations RETURN OLD",
            bind_vars={"uid": user_id},
        ))
        deleted["entity_relations"] = len(rel_result)

        try:
            qdrant_client.delete(
                collection_name="memories",
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
            deleted["qdrant"] = "deleted"
        except Exception as e:
            deleted["qdrant"] = str(e)

        return {"status": "success", "deleted": deleted}
    except Exception as e:
        logger.error(f"Benchmark clear error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    # Check if service is fully initialized
    if not service_ready:
        return HealthResponse(
            status="initializing",
            service="dappy-llm-orchestrator",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
            components={"initialization": "in_progress"}
        )
    
    components_status = {}
    
    # Check Redis
    try:
        await redis_client.ping()
        components_status['redis'] = 'healthy'
    except:
        components_status['redis'] = 'unhealthy'
    
    # Check Event Bus
    components_status['event_bus'] = 'healthy' if event_bus else 'not_initialized'
    
    # Check Workers
    components_status['consolidation_worker'] = 'running' if consolidation_worker and consolidation_worker._running else 'stopped'
    components_status['arango_consumer'] = 'running' if arango_consumer and arango_consumer._running else 'stopped'
    components_status['qdrant_consumer'] = 'running' if qdrant_consumer and qdrant_consumer._running else 'stopped'
    
    # Check Chatbot Service
    components_status['chatbot_service'] = 'ready' if chatbot_service else 'not_initialized'
    
    # Overall status
    all_healthy = all(
        status in ['healthy', 'running', 'ready']
        for status in components_status.values()
    )
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        service="dappy-llm-orchestrator",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        components=components_status
    )


# Context management endpoint
@app.post("/context/manage", response_model=ContextResponse)
async def manage_context(request: ContextRequest):
    """
    Manage LLM context with intelligent flushing
    
    - Monitors token usage
    - Performs intelligent/emergency flush
    - Returns optimized conversation history
    """
    
    try:
        # Convert Pydantic models to dicts
        conversation_history = [msg.dict() for msg in request.conversation_history]
        new_message = request.new_message.dict() if request.new_message else None
        
        # Manage context
        optimized_history, metadata = await context_manager.manage_context(
            user_id=request.user_id,
            session_id=request.session_id,
            conversation_history=conversation_history,
            new_message=new_message
        )
        
        # Convert back to Pydantic models
        optimized_messages = [Message(**msg) for msg in optimized_history]
        
        return ContextResponse(
            optimized_history=optimized_messages,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Context management error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Ego scoring endpoint
@app.post("/scoring/ego", response_model=EgoScoreResponse)
async def calculate_ego_score(request: EgoScoreRequest):
    """
    Calculate ego score for a memory
    
    Returns score, tier, and component breakdown
    """
    
    try:
        result = ego_scorer.calculate(
            memory=request.memory,
            current_tier=request.current_tier
        )
        
        return EgoScoreResponse(
            ego_score=result.ego_score,
            tier=result.tier,
            components=result.components.to_dict(),
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Ego scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Consolidation stats endpoint
@app.get("/consolidation/stats")
async def get_consolidation_stats():
    """Get consolidation queue statistics"""
    
    try:
        queue_stats = await consolidation_worker.get_queue_stats()
        arango_stats = arango_consumer.get_stats()
        qdrant_stats = qdrant_consumer.get_stats()
        
        return {
            "queues": queue_stats,
            "arango_consumer": arango_stats,
            "qdrant_consumer": qdrant_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get consolidation stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Training data collection endpoints
@app.get("/training/stats")
async def get_training_data_stats():
    """
    Get training data collection statistics.
    
    Returns stats about:
    - Total corrections collected
    - Corrections by source (semantic_validator, label_discovery, etc.)
    - Corrections by routing decision
    - Total discovered labels
    - Total confidence anomalies
    - Recent corrections (last 24h)
    - Top invalid labels (most common false positives)
    """
    try:
        if not hasattr(chatbot_service, 'training_collector'):
            raise HTTPException(
                status_code=503,
                detail="Training data collector not initialized"
            )
        
        stats = chatbot_service.training_collector.get_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training data stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/dataset")
async def export_training_dataset(
    min_confidence: float = 0.0,
    source: Optional[str] = None,
    limit: int = 100
):
    """
    Export training dataset for model training.
    
    Args:
        min_confidence: Minimum classifier confidence (0.0-1.0)
        source: Filter by source (semantic_validator, label_discovery)
        limit: Maximum number of examples to return
    
    Returns:
        List of training examples with corrected labels
    """
    try:
        if not hasattr(chatbot_service, 'training_collector'):
            raise HTTPException(
                status_code=503,
                detail="Training data collector not initialized"
            )
        
        source_filter = [source] if source else None
        dataset = chatbot_service.training_collector.get_training_dataset(
            min_confidence=min_confidence,
            source_filter=source_filter,
            limit=limit
        )
        
        return {
            "dataset": dataset,
            "count": len(dataset),
            "filters": {
                "min_confidence": min_confidence,
                "source": source,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export training dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/export")
async def export_training_data_to_file(
    output_path: str = "data/training_export.jsonl",
    min_confidence: float = 0.7
):
    """
    Export training data to HuggingFace format (JSONL file).
    
    Args:
        output_path: Path to save the JSONL file
        min_confidence: Minimum confidence threshold
    
    Returns:
        Export status and file path
    """
    try:
        if not hasattr(chatbot_service, 'training_collector'):
            raise HTTPException(
                status_code=503,
                detail="Training data collector not initialized"
            )
        
        chatbot_service.training_collector.export_to_huggingface_format(
            output_path=output_path,
            min_confidence=min_confidence
        )
        
        return {
            "status": "success",
            "output_path": output_path,
            "min_confidence": min_confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export training data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Shadow tier endpoints
@app.get("/shadow/pending/{user_id}")
async def get_pending_shadow_memories(user_id: str):
    """Get pending shadow memories for user"""
    try:
        pending = await shadow_tier.get_pending_for_user(user_id)
        return {
            "user_id": user_id,
            "pending_count": len(pending),
            "pending_memories": [m.to_dict() for m in pending]
        }
    except Exception as e:
        logger.error(f"Failed to get pending shadow memories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shadow/approve/{user_id}/{shadow_id}")
async def approve_shadow_memory(user_id: str, shadow_id: str):
    """Approve shadow memory → promote to Tier 1"""
    try:
        result = await shadow_tier.approve_shadow_memory(user_id, shadow_id)
        if result:
            return {"status": "approved", "shadow_memory": result.to_dict()}
        else:
            raise HTTPException(status_code=404, detail="Shadow memory not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve shadow memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shadow/reject/{user_id}/{shadow_id}")
async def reject_shadow_memory(user_id: str, shadow_id: str, reason: Optional[str] = None):
    """Reject shadow memory"""
    try:
        result = await shadow_tier.reject_shadow_memory(user_id, shadow_id, reason)
        if result:
            return {"status": "rejected", "shadow_memory": result.to_dict()}
        else:
            raise HTTPException(status_code=404, detail="Shadow memory not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject shadow memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Contradiction detection endpoint - DISABLED
# Contradiction detection is now handled by KG Maintenance Agent
# @app.post("/contradiction/check")
# async def check_contradictions(new_memory: Dict[str, Any], user_id: str):
#     """Check if new memory contradicts existing memories"""
#     try:
#         contradictions = await contradiction_detector.check_for_contradictions(
#             new_memory=new_memory,
#             user_id=user_id
#         )
#         
#         return {
#             "user_id": user_id,
#             "contradictions_found": len(contradictions),
#             "contradictions": [
#                 {
#                     "memory1_id": c.memory1_id,
#                     "memory2_id": c.memory2_id,
#                     "temporal_gap_days": c.temporal_gap_days,
#                     "is_temporal_change": c.is_temporal_change,
#                     "requires_clarification": c.requires_clarification,
#                     "clarification_question": c.clarification_question,
#                     "similarity_score": c.similarity_score
#                 }
#                 for c in contradictions
#             ]
#         }
#     except Exception as e:
#         logger.error(f"Contradiction detection error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# Chat endpoints (protected)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Chat with DAPPY (requires authentication).

    The user_id is derived from the JWT token automatically.
    """
    import time as _time
    from core.metrics import CHAT_LATENCY, CHAT_REQUESTS
    
    _start = _time.time()
    try:
        response = await chatbot_service.chat(
            user_id=current_user["user_id"],
            session_id=request.session_id,
            user_message=request.message,
            conversation_history=request.conversation_history,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            debug=request.debug
        )
        CHAT_REQUESTS.labels(outcome='success').inc()
        CHAT_LATENCY.observe(_time.time() - _start)
        return ChatResponse(**response)
    except Exception as e:
        CHAT_REQUESTS.labels(outcome='error').inc()
        CHAT_LATENCY.observe(_time.time() - _start)
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Stream chat response from DAPPY (requires authentication)."""
    from fastapi.responses import StreamingResponse
    
    try:
        async def generate():
            async for chunk in chatbot_service.chat_stream(
                user_id=current_user["user_id"],
                session_id=request.session_id,
                user_message=request.message,
                conversation_history=request.conversation_history,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Chat streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Shadow Tier endpoints
class ShadowConfirmationRequest(BaseModel):
    session_id: str
    user_id: str
    confirmed: bool


class ShadowConfirmationResponse(BaseModel):
    status: str
    message: str
    node_id: Optional[str] = None


@app.get("/shadow/pending/{session_id}")
async def get_pending_shadow(session_id: str):
    """
    Get pending shadow tier confirmation for a session
    
    Returns clarification question if there's a pending confirmation.
    """
    try:
        pending = chatbot_service.get_pending_confirmation(session_id)
        if not pending:
            return {
                "has_pending": False,
                "message": "No pending confirmation"
            }
        
        return {
            "has_pending": True,
            "clarification_id": pending['clarification_id'],
            "question": pending['question'],
            "content": pending['content'],
            "ego_score": pending['ego_score']
        }
    except Exception as e:
        logger.error(f"Error getting pending shadow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shadow/confirm", response_model=ShadowConfirmationResponse)
async def confirm_shadow_memory(request: ShadowConfirmationRequest):
    """
    Confirm or reject a shadow tier memory
    
    - **session_id**: Session identifier
    - **user_id**: User identifier
    - **confirmed**: True to promote to Tier 1, False to demote to Tier 2
    """
    try:
        result = await chatbot_service.handle_shadow_confirmation(
            session_id=request.session_id,
            user_id=request.user_id,
            confirmed=request.confirmed
        )
        return ShadowConfirmationResponse(**result)
    except Exception as e:
        logger.error(f"Error confirming shadow memory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "DAPPY LLM Orchestration Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "context_management": "/context/manage",
            "ego_scoring": "/scoring/ego",
            "consolidation_stats": "/consolidation/stats",
            "shadow_pending": "/shadow/pending/{user_id}",
            "shadow_approve": "/shadow/approve/{user_id}/{shadow_id}",
            "shadow_reject": "/shadow/reject/{user_id}/{shadow_id}",
            "contradiction_check": "/contradiction/check",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.get('service.port', 8001),
        reload=True,
        log_level="info"
    )

