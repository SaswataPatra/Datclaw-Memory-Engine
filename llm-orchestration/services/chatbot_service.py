"""
Chatbot Service
Orchestrates LLM conversations with memory integration

Integrates:
- LLM Provider (OpenAI, Ollama, etc.)
- Context Manager
- ML-based ego scoring
- Shadow Tier for Tier 1 confirmation
- Graph-of-Thoughts pipeline (Phase 1C)
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
from datetime import datetime
import logging
import re
import asyncio
import uuid

from llm.providers.base import LLMProvider, LLMMessage, LLMResponse
from services.context_manager import ContextMemoryManager
from services.embedding_service import EmbeddingService
from services.classification import ClassifierManager, SemanticValidator, RegexFallback
from services.scoring import MLScorer
from core.scoring.ego_scorer import TemporalEgoScorer
from core.event_bus import EventBus
from core.shadow_tier import ShadowTier

# ML-based ego scoring components
from ml.component_scorers import (
    NoveltyScorer,
    FrequencyScorer,
    SentimentScorer,
    ExplicitImportanceScorer,
    EngagementScorer
)
from ml.combiners import LightGBMCombiner, ConfidenceCombiner
from ml.extractors import DistilBERTMemoryClassifier, AdaptiveLabelDiscovery
from ml.extractors.zeroshot_memory_classifier import (
    ZeroShotMemoryClassifier,
    DynamicLabelDiscovery,
    LabelStore
)
from ml.utils import get_global_executor

# Graph-of-Thoughts integration (Phase 1C)
try:
    from core.graph.arango_integration import GraphPipeline, create_graph_pipeline
    from core.command_parser import CommandParser
    GRAPH_AVAILABLE = True
except ImportError as e:
    GRAPH_AVAILABLE = False
    GraphPipeline = None
    CommandParser = None
    logging.getLogger(__name__).warning(f"Graph components not available: {e}")

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Orchestrates chatbot conversations with LLM integration
    
    Responsibilities:
    - Manage conversation flow
    - Call LLM provider (pluggable)
    - Integrate with ContextManager
    - Extract memories from conversations
    - Score and publish memories to Event Bus
    - Route high-ego memories through Shadow Tier
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        context_manager: ContextMemoryManager,
        ego_scorer: TemporalEgoScorer,
        event_bus: EventBus,
        shadow_tier: Optional[ShadowTier] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        qdrant_client=None,  # For ML scorers
        embedding_service: Optional[EmbeddingService] = None,  # For embedding generation
        use_ml_scoring: bool = False,  # Flag to enable ML-based scoring
        use_distilbert: bool = False,  # Flag to enable DistilBERT classifier (Phase 2)
        classifier_type: str = None,  # Classifier type: "hf_api" | "zeroshot" | "distilbert" | "regex" (None = read from config)
        use_graph_pipeline: bool = True,  # Enable Graph-of-Thoughts (Phase 1C) - deprecated
        arango_db=None,  # ArangoDB connection for graph pipeline
        consolidation_service=None,  # Memory consolidation (entities + relations)
        knowledge_graph_store=None  # Knowledge graph storage
    ):
        """
        Initialize ChatbotService
        
        Args:
            llm_provider: LLM provider instance (OpenAI, Ollama, etc.)
            context_manager: Context memory manager
            ego_scorer: Ego scorer for memory importance (legacy)
            event_bus: Event bus for publishing memories
            shadow_tier: Shadow tier manager for Tier 1 confirmation
            system_prompt: Optional custom system prompt
            config: Configuration dict for ML components
            qdrant_client: Qdrant client for ML scorers
            use_ml_scoring: Enable ML-based scoring (Phase 1.5)
        """
        self.llm_provider = llm_provider
        self.context_manager = context_manager
        self.ego_scorer = ego_scorer  # Keep for backward compatibility
        self.event_bus = event_bus
        self.shadow_tier = shadow_tier
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.config = config or {}
        self.embedding_service = embedding_service
        self.use_ml_scoring = use_ml_scoring
        self.use_distilbert = use_distilbert

        # Knowledge graph consolidation
        self.consolidation_service = consolidation_service
        self.knowledge_graph_store = knowledge_graph_store
        self.use_consolidation = consolidation_service is not None and knowledge_graph_store is not None

        # Graph pipeline controls memory extraction and KG consolidation
        # When True: extracts memories, entities, and relations from conversations
        # When False: chat-only mode (no memory extraction)
        self.use_graph_pipeline = use_graph_pipeline
        self.graph_pipeline = None
        self.command_parser = None
        self.background_consolidation = None
        
        # Track pending shadow tier confirmations per session
        self.pending_confirmations: Dict[str, Dict] = {}
        
        # Track memory processing state per session
        # Format: {session_id: {message_id: 'pending'|'processing'|'completed'}}
        self.memory_processing_state: Dict[str, Dict[str, str]] = {}
        
        # Initialize ML components if enabled
        # Initialize async executor for CPU-intensive operations
        # This prevents blocking the event loop during ML inference
        # Using 2 workers for better throughput (zero-shot is more memory-efficient than DistilBERT)
        self.ml_executor = get_global_executor(max_workers=4)  # Increased for zero-shot + LightGBM parallelism
        
        if use_ml_scoring:
            self._init_ml_components(qdrant_client)
            logger.info("ML-based ego scoring enabled")
        else:
            self.ml_scorers = None
            self.ml_combiner = None
            self.confidence_combiner = None
            logger.info("Using legacy regex-based scoring")
        
        # Initialize memory classifier based on type
        self.classifier_type = classifier_type or self.config.get('ml', {}).get('classifier_type', 'zeroshot')
        
        # DEBUG: Print what we got from config
        print(f"🔍 DEBUG: classifier_type from config: {self.config.get('ml', {}).get('classifier_type', 'NOT_SET')}")
        print(f"🔍 DEBUG: classifier_type parameter: {classifier_type}")
        print(f"🔍 DEBUG: final classifier_type: {self.classifier_type}")
        self.memory_classifier = None
        self.label_discovery = None
        self.label_store = None
        self._classifier_initialized = False
        
        # Legacy DistilBERT support (backward compatibility)
        self.use_distilbert = use_distilbert
        self.distilbert_classifier = None
        self.adaptive_discovery = None
        
        if self.classifier_type == "hf_api":
            logger.info("HuggingFace API classifier enabled (will load during startup)")
        elif self.classifier_type == "zeroshot":
            logger.info("Zero-shot memory classifier enabled (will load during startup)")
        elif self.classifier_type == "llm":
            logger.info("LLM-based classifier enabled (OpenAI gpt-4o-mini, no HF API)")
        elif self.classifier_type == "distilbert" or use_distilbert:
            logger.info("DistilBERT memory classifier enabled (will load during startup)")
            self.classifier_type = "distilbert"
        elif self.classifier_type == "regex":
            logger.info("Using regex-based memory detection (no model loading)")
        else:
            logger.warning(f"Unknown classifier type '{self.classifier_type}', falling back to regex")
            self.classifier_type = "regex"
        
        logger.info(f"")
        logger.info(f"{'='*70}")
        logger.info(f"🤖 ChatbotService Initialized")
        logger.info(f"{'='*70}")
        logger.info(f"   LLM Provider: {llm_provider.name}")
        logger.info(f"   Shadow Tier: {'✅ enabled' if shadow_tier else '❌ disabled'}")
        logger.info(f"   ML Scoring: {'✅ enabled' if use_ml_scoring else '❌ disabled (legacy regex)'}")
        logger.info(f"   Classifier: {self.classifier_type} {'(will load at startup)' if self.classifier_type != 'regex' else ''}")
        if use_ml_scoring and self.ml_combiner:
            logger.info(f"   LightGBM: {'✅ trained model loaded' if self.ml_combiner.is_trained else '⚠️  fallback weights'}")
        logger.info(f"{'='*70}")
        logger.info(f"")
        
        # Initialize training data collector
        from services.training_data_collector import TrainingDataCollector
        training_db_path = config.get('training_data', {}).get('db_path', 'data/training_corrections.db')
        self.training_collector = TrainingDataCollector(db_path=training_db_path)
        
        # Initialize classification helpers
        self.semantic_validator = SemanticValidator(llm_provider, training_collector=self.training_collector)
        self.regex_fallback = RegexFallback()  # Will be updated with label_store later
        self.classifier_manager = None  # Will be initialized after classifier is loaded
        self.ml_scorer = None  # Will be initialized after ML components are loaded
        self.label_store = None  # Will be initialized in classifier init
        logger.info(f"✅ Training data collector, semantic validator and regex fallback initialized")
    
    def _init_ml_components(self, qdrant_client):
        """Initialize ML-based scoring components"""
        try:
            # Initialize component scorers
            self.ml_scorers = {
                'novelty': NoveltyScorer(self.config, qdrant_client) if qdrant_client else None,
                'frequency': FrequencyScorer(self.config, qdrant_client, event_bus=self.event_bus) if qdrant_client else None,
                'sentiment': SentimentScorer(self.config),
                'explicit_importance': ExplicitImportanceScorer(self.config),
                'engagement': EngagementScorer(self.config)
            }
            
            # Initialize LightGBM combiner (disabled for now - will be used in the future)
            use_lightgbm = self.config.get('ml', {}).get('use_lightgbm', False)
            if use_lightgbm:
                self.ml_combiner = LightGBMCombiner(self.config)
                model_path = "models/lightgbm/combiner.pkl"
                try:
                    self.ml_combiner.load_model(model_path)
                    logger.info(f"Loaded trained LightGBM model from {model_path}")
                except FileNotFoundError:
                    logger.warning(f"No trained LightGBM model found at {model_path}. Will use weighted average fallback.")
                except Exception as e:
                    logger.warning(f"Could not load LightGBM model: {e}. Will use weighted average fallback.")
            else:
                self.ml_combiner = None
                logger.info("LightGBM disabled - using weighted average for ego scoring")
            
            # Initialize confidence combiner (disabled for now - will be used in the future)
            use_confidence_combiner = self.config.get('ml', {}).get('use_confidence_combiner', False)
            if use_confidence_combiner:
                self.confidence_combiner = ConfidenceCombiner(self.config)
            else:
                self.confidence_combiner = None
                logger.info("Confidence combiner disabled - using default confidence (0.7)")
            
            logger.info("ML components initialized successfully")
            
            # Initialize MLScorer after all ML components are loaded
            # Note: classifier_manager will be initialized later in _init_classifier_sync
            # So we'll initialize ml_scorer there instead
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}", exc_info=True)
            self.ml_scorers = None
            self.ml_combiner = None
            self.confidence_combiner = None
    
    def _init_distilbert_classifier_sync(self):
        """
        Synchronous initialization of DistilBERT classifier during service startup.
        
        We load synchronously to avoid PyTorch threading issues with torch.load().
        This blocks startup but ensures the model loads correctly.
        """
        try:
            logger.info("⏳ Loading DistilBERT classifier (this will take ~5-10 seconds)...")
            logger.info("   💡 Loading synchronously to avoid PyTorch threading issues")
            
            # Load DistilBERT synchronously
            logger.info("  📦 Loading base DistilBERT model...")
            self.distilbert_classifier = DistilBERTMemoryClassifier()
            logger.info("  ✓ Base model loaded")
            
            # Try to load trained model if it exists
            model_path = "models/distilbert/best_model.pt"
            try:
                logger.info(f"  📥 Loading trained weights from {model_path} (253MB, ~5s)...")
                self.distilbert_classifier.load_model(model_path)
                logger.info(f"  ✅ Trained model loaded successfully")
            except FileNotFoundError:
                logger.warning(f"  ⚠️  No trained model found at {model_path}. Using untrained model (train first!)")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not load trained model: {e}")
            
            # Initialize adaptive label discovery (disabled by default)
            adaptive_config = self.config.get('adaptive_discovery', {})
            self.adaptive_discovery = AdaptiveLabelDiscovery(
                llm_client=self.llm_provider.client if hasattr(self.llm_provider, 'client') else None,
                config=self.config,
                base_labels=self.distilbert_classifier.label_names,
                enabled=adaptive_config.get('enabled', False)
            )
            
            self._distilbert_initialized = True
            logger.info(
                f"✅ DistilBERT classifier ready (adaptive_discovery={'enabled' if self.adaptive_discovery.enabled else 'disabled'})"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize DistilBERT classifier: {e}", exc_info=True)
            logger.warning("⚠️  Falling back to regex-based memory detection")
            self.distilbert_classifier = None
            self.adaptive_discovery = None
            self._distilbert_initialized = True
    
    def _init_hf_api_classifier_sync(self):
        """
        Synchronous initialization of HuggingFace API classifier during service startup.
        
        Fast, cost-effective cloud inference (~$0.06 per 1,000 requests).
        Latency: 200-500ms (vs 30-60s local CPU).
        """
        try:
            hf_api_config = self.config.get('ml', {}).get('hf_api', {})
            api_key = hf_api_config.get('api_key')
            
            if not api_key or api_key == "${HUGGINGFACE_API_KEY}":
                raise ValueError(
                    "HuggingFace API key not set! "
                    "Get one at https://huggingface.co/settings/tokens "
                    "and add to .env: HUGGINGFACE_API_KEY=hf_xxxxx"
                )
            
            logger.info(f"⏳ Initializing HuggingFace API classifier...")
            
            # Initialize classifier
            from ml.extractors.hf_api_classifier import HuggingFaceAPIClassifier
            self.memory_classifier = HuggingFaceAPIClassifier(
                api_key=api_key,
                model_name=hf_api_config.get('model_name', 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'),
                low_confidence_threshold=hf_api_config.get('low_confidence_threshold', 0.3),
                timeout=hf_api_config.get('timeout', 10.0)
            )
            
            logger.info(f"  ✅ HuggingFace API classifier ready")
            
            # Initialize label store FIRST (needed by label discovery)
            label_store_path = hf_api_config.get('label_store_path', 'data/discovered_labels.json')
            self.label_store = LabelStore(storage_path=label_store_path)
            
            # Initialize label discovery (uses OpenAI + LabelStore)
            if hf_api_config.get('enable_discovery', True):
                self.label_discovery = DynamicLabelDiscovery(
                    llm_client=self.llm_provider.client if hasattr(self.llm_provider, 'client') else None,
                    config=self.config,
                    enabled=True,
                    label_store=self.label_store  # Pass label store for context
                )
                logger.info(f"  ✅ Label discovery enabled (OpenAI GPT-4o-mini)")
            else:
                self.label_discovery = None
                logger.info(f"  ⚠️  Label discovery disabled")
            
            # Update regex fallback with label store for discovered label matching
            self.regex_fallback.label_store = self.label_store
            
            # Load previously discovered labels
            discovered_labels = self.label_store.get_all_labels()
            if discovered_labels:
                self.memory_classifier.add_labels(discovered_labels)
                logger.info(f"  📚 Loaded {len(discovered_labels)} previously discovered labels")
                
                # Load importance scores into explicit importance scorer
                if self.ml_scorers and 'explicit_importance' in self.ml_scorers:
                    labels_with_importance = self.label_store.get_labels_with_importance()
                    for label, importance in labels_with_importance.items():
                        self.ml_scorers['explicit_importance'].importance_map[label] = importance
                    logger.info(f"  📊 Loaded importance scores for {len(labels_with_importance)} labels")
            
            self._classifier_initialized = True
            logger.info(f"✅ HuggingFace API classifier ready (cloud inference)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace API classifier: {e}", exc_info=True)
            logger.warning("⚠️  Falling back to regex-based memory detection")
            self.memory_classifier = None
            self.label_discovery = None
            self.label_store = None
            self._classifier_initialized = True
    
    def _init_zeroshot_classifier_sync(self):
        """
        Synchronous initialization of Zero-Shot classifier during service startup.
        
        Uses environment-based model selection:
        - local: mDeBERTa-v3-base (92-95% accuracy, 1.5s CPU)
        - production: DeBERTa-v3-large (98% accuracy, 0.3s GPU)
        """
        try:
            zeroshot_config = self.config.get('ml', {}).get('zeroshot', {})
            environment = zeroshot_config.get('environment', 'local')
            
            logger.info(f"⏳ Loading Zero-Shot classifier (environment: {environment})...")
            
            # Initialize classifier
            self.memory_classifier = ZeroShotMemoryClassifier(
                environment=environment,
                model_name=zeroshot_config.get('model_name'),  # Optional override
                low_confidence_threshold=zeroshot_config.get('low_confidence_threshold', 0.3),
                device=zeroshot_config.get('device', -1)
            )
            
            logger.info(f"  ✅ Zero-shot classifier loaded")
            
            # Initialize label store FIRST (needed by label discovery)
            label_store_path = zeroshot_config.get('label_store_path', 'data/discovered_labels.json')
            self.label_store = LabelStore(storage_path=label_store_path)
            
            # Initialize label discovery (uses OpenAI + LabelStore)
            if zeroshot_config.get('enable_discovery', True):
                self.label_discovery = DynamicLabelDiscovery(
                    llm_client=self.llm_provider.client if hasattr(self.llm_provider, 'client') else None,
                    config=self.config,
                    enabled=True,
                    label_store=self.label_store  # Pass label store for context
                )
                logger.info(f"  ✅ Label discovery enabled (OpenAI GPT-4o-mini)")
            else:
                self.label_discovery = None
                logger.info(f"  ⚠️  Label discovery disabled")
            
            # Update regex fallback with label store for discovered label matching
            self.regex_fallback.label_store = self.label_store
            
            # Load previously discovered labels
            discovered_labels = self.label_store.get_all_labels()
            if discovered_labels:
                self.memory_classifier.add_labels(discovered_labels)
                logger.info(f"  📚 Loaded {len(discovered_labels)} previously discovered labels")
                
                # Load importance scores into explicit importance scorer
                if self.ml_scorers and 'explicit_importance' in self.ml_scorers:
                    labels_with_importance = self.label_store.get_labels_with_importance()
                    for label, importance in labels_with_importance.items():
                        self.ml_scorers['explicit_importance'].importance_map[label] = importance
                    logger.info(f"  📊 Loaded importance scores for {len(labels_with_importance)} labels")
            
            self._classifier_initialized = True
            logger.info(f"✅ Zero-shot classifier ready ({environment} environment)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize zero-shot classifier: {e}", exc_info=True)
            logger.warning("⚠️  Falling back to regex-based memory detection")
            self.memory_classifier = None
            self.label_discovery = None
            self.label_store = None
            self._classifier_initialized = True
    
    def _init_classifier_sync(self):
        """
        Initialize the appropriate classifier based on classifier_type.
        Called during startup in a thread pool.
        """
        if self.classifier_type == "hf_api":
            self._init_hf_api_classifier_sync()
        elif self.classifier_type == "zeroshot":
            self._init_zeroshot_classifier_sync()
        elif self.classifier_type == "distilbert":
            self._init_distilbert_classifier_sync()
        elif self.classifier_type == "llm":
            self._classifier_initialized = True
            logger.info("LLM classifier ready (uses OpenAI API at classification time)")
        else:
            self._classifier_initialized = True
            logger.info("Using regex-based memory detection (no model loading)")
        
        # Initialize ClassifierManager after classifier is loaded
        self.classifier_manager = ClassifierManager(
            classifier_type=self.classifier_type,
            memory_classifier=self.memory_classifier,
            semantic_validator=self.semantic_validator,
            regex_fallback=self.regex_fallback,
            label_discovery=self.label_discovery,
            label_store=self.label_store,
            ml_executor=self.ml_executor,
            ml_scorers=self.ml_scorers,
            config=self.config
        )
        logger.info("✅ ClassifierManager initialized")
        
        # Initialize MLScorer after both ML components and classifier are loaded
        if self.ml_scorers:
            self.ml_scorer = MLScorer(
                ml_scorers=self.ml_scorers,
                ml_combiner=self.ml_combiner,
                confidence_combiner=self.confidence_combiner,
                classifier_manager=self.classifier_manager,
                embedding_service=self.embedding_service,
                ml_executor=self.ml_executor,
                regex_fallback=self.regex_fallback,
                classifier_type=self.classifier_type,
                config=self.config,
                llm_service=self.llm_provider
            )
            logger.info("✅ MLScorer initialized")
    
    async def _ensure_distilbert_loaded(self):
        """
        No-op since we now load DistilBERT synchronously during startup.
        Kept for backward compatibility.
        """
        # Already loaded during __init__
        pass
    
    def _default_system_prompt(self) -> str:
        """Default system prompt for DAPPY"""
        return """You are DAPPY, an intelligent AI assistant with a sophisticated memory system.

You remember important information about users across conversations and can recall context from previous interactions.

Key capabilities:
- You have a multi-tier memory system that remembers what's important
- You can recall information from past conversations
- You understand context and can maintain long conversations
- You're helpful, concise, and personable

When users share personal information (preferences, facts about themselves, etc.), acknowledge it naturally and remember it for future conversations."""
    
    async def chat(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response with metadata
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            conversation_history: Optional conversation history
            temperature: LLM temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            debug: Whether to include debug information
            
        Returns:
            Dict with assistant_message, conversation_history, and metadata
        """
        try:
            logger.info(
                f"Processing chat for user={user_id}, session={session_id}, "
                f"message_length={len(user_message)}, "
                f"input=\"{user_message[:100]}{'...' if len(user_message) > 100 else ''}\""
            )
            
            # 1. Manage context (get optimized history)
            if conversation_history is None:
                conversation_history = []
            
            new_message = {
                "message_id": f"msg_{len(conversation_history) + 1}",
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "sequence": len(conversation_history) + 1
            }
            
            optimized_history, context_metadata = await self.context_manager.manage_context(
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                new_message=new_message
            )
            
            logger.debug(
                f"Context optimized: {len(optimized_history)} messages, "
                f"tokens={context_metadata.get('current_tokens', 0)}"
            )
            
            # 2. Retrieve relevant memories from graph (PPR)
            logger.info("="*80)
            logger.info(f"📝 USER INPUT: {user_message}")
            logger.info("="*80)
            
            relevant_memories = []
            try:
                relevant_memories = await self.context_manager.retrieve_relevant_memories(
                    user_id=user_id,
                    query=user_message,
                    max_memories=5,
                    use_ppr=True,
                    use_vector=True  # Vector search (primary path)
                )
                
                if relevant_memories:
                    logger.info(f"📚 Retrieved {len(relevant_memories)} relevant memories from graph")
                    logger.info("="*80)
                    logger.info("🔍 VECTOR SEARCH RESULTS:")
                    for i, mem in enumerate(relevant_memories, 1):
                        logger.info(f"   Memory {i}:")
                        logger.info(f"      Content: {mem['content'][:200]}{'...' if len(mem['content']) > 200 else ''}")
                        logger.info(f"      Ego Score: {mem.get('ego_score', 0):.2f}")
                        logger.info(f"      Relevance: {mem.get('relevance_score', 0):.2f}")
                        logger.info(f"      Source: {mem.get('source', 'unknown')}")
                    logger.info("="*80)
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
            
            # 3. Build LLM messages with retrieved memories + relation context
            llm_messages = [LLMMessage(role="system", content=self.system_prompt)]
            
            # Add retrieved memories as context (if any)
            if relevant_memories:
                memory_context = "## Relevant Context from Memory\n\n"
           
                for i, mem in enumerate(relevant_memories, 1):
                    memory_context += f"{i}. {mem['content']}\n"
                    memory_context += f"   (Importance: {mem.get('ego_score', 0):.2f}, Source: {mem.get('source', 'unknown')})\n\n"
                
                # Add relation context (localized knowledge graph subgraph)
                relation_context = self.context_manager.get_localized_relations_context(
                    user_id=user_id,
                    retrieved_memories=relevant_memories,
                    limit=30
                )
                if relation_context:
                    logger.info("🕸️  KNOWLEDGE GRAPH RELATIONS:")
                    logger.info(relation_context)
                    logger.info("="*80)
                    memory_context += "\n" + relation_context + "\n"
                else:
                    logger.info("🕸️  KNOWLEDGE GRAPH RELATIONS: None found")
                    logger.info("="*80)
                
                # Log the full context being sent to LLM
                logger.info("📋 FULL CONTEXT SENT TO LLM:")
                logger.info(memory_context)
                logger.info("="*80)
                
                # Insert memory context before conversation
                llm_messages.append(LLMMessage(
                    role="system",
                    content=memory_context
                ))
            
            for msg in optimized_history:
                llm_messages.append(LLMMessage(
                    role=msg["role"],
                    content=msg["content"]
                ))
            
            # 4. Call LLM
            logger.debug(f"Calling LLM provider: {self.llm_provider.name}")
            llm_response = await self.llm_provider.chat(
                messages=llm_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(
                f"LLM response received: {llm_response.usage['total_tokens']} tokens, "
                f"finish_reason={llm_response.finish_reason}, "
                f"DEBUGGING MODE---------- output=\"{user_message} -> {llm_response.content[:100]}{'...' if len(llm_response.content) > 100 else ''}\""
            )
            
            # 5. Extract memories from conversation (BACKGROUND - fire-and-forget)
            message_id = new_message["message_id"]
            self.mark_message_processing(session_id, message_id, 'pending')
            asyncio.create_task(
                self._extract_and_score_memories(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=llm_response.content,
                    message_id=message_id
                )
            )
            
            # 6. Build response
            assistant_message_dict = {
                "message_id": f"msg_{len(optimized_history) + 1}",
                "role": "assistant",
                "content": llm_response.content,
                "timestamp": datetime.utcnow().isoformat(),
                "sequence": len(optimized_history) + 1
            }
            
            response = {
                "assistant_message": llm_response.content,
                "conversation_history": optimized_history + [assistant_message_dict],
                "metadata": {
                    "context": context_metadata,
                    "llm": {
                        "provider": self.llm_provider.name,
                        "model": llm_response.model,
                        "usage": llm_response.usage,
                        "finish_reason": llm_response.finish_reason
                    }
                }
            }
            
            # 6. Add debug info if requested
            if debug:
                response["debug"] = {
                    "llm_messages": [
                        {"role": msg.role, "content": msg.content}
                        for msg in llm_messages
                    ],
                    "system_prompt": self.system_prompt,
                    "optimized_history_length": len(optimized_history),
                    "memory_extraction": {
                        "triggers_detected": self.regex_fallback.detect_triggers(user_message)
                    }
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise
    
    async def chat_stream(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Stream chat response
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            conversation_history: Optional conversation history
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            str: Chunks of the response
        """
        try:
            logger.info(
                f"Starting streaming chat for user={user_id}, session={session_id}"
            )
            
            # Manage context
            if conversation_history is None:
                conversation_history = []
            
            new_message = {
                "message_id": f"msg_{len(conversation_history) + 1}",
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "sequence": len(conversation_history) + 1
            }
            
            optimized_history, _ = await self.context_manager.manage_context(
                user_id=user_id,
                session_id=session_id,
                conversation_history=conversation_history,
                new_message=new_message
            )
            
            # Build LLM messages
            llm_messages = [LLMMessage(role="system", content=self.system_prompt)]
            for msg in optimized_history:
                llm_messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
            
            # Stream from LLM
            full_response = ""
            async for chunk in self.llm_provider.chat_stream(
                messages=llm_messages,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                full_response += chunk
                yield chunk
            
            logger.info(f"Streaming completed: {len(full_response)} chars")
            
            # Extract memories after streaming completes (BACKGROUND - fire-and-forget)
            asyncio.create_task(
                self._extract_and_score_memories(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=full_response
                )
            )
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}", exc_info=True)
            raise
    
    def get_processing_state(self, session_id: str) -> Dict[str, str]:
        """
        Get memory processing state for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict mapping message_id to processing state ('pending'|'processing'|'completed')
        """
        return self.memory_processing_state.get(session_id, {})
    
    def mark_message_processing(self, session_id: str, message_id: str, state: str):
        """
        Mark a message's processing state
        
        Args:
            session_id: Session identifier
            message_id: Message identifier
            state: 'pending'|'processing'|'completed'
        """
        if session_id not in self.memory_processing_state:
            self.memory_processing_state[session_id] = {}
        self.memory_processing_state[session_id][message_id] = state
    
    async def _extract_and_score_memories(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        message_id: Optional[str] = None
    ):
        """
        Extract important information and score as memories
        
        This is a simple heuristic-based approach for Phase 1.
        In Phase 2, we can add more sophisticated NER and entity extraction.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            message_id: Message identifier for tracking processing state
        """
        # Mark as processing
        if message_id:
            self.mark_message_processing(session_id, message_id, 'processing')
        
        try:
            # ========== DIALOGUE PREPROCESSING (Phase 1F) ==========
            # If dialogue format detected, extract only the meaningful content for classification
            content_for_classification = user_message
            if self.use_graph_pipeline and hasattr(self, 'graph_pipeline'):
                try:
                    from core.graph import DialogueProcessor
                    dialogue_processor = DialogueProcessor()
                    is_dialogue, format_type = dialogue_processor.is_dialogue_format(user_message)
                    
                    if is_dialogue:
                        logger.info(f"📋 Dialogue detected for ego scoring, extracting content...")
                        processed = dialogue_processor.process(user_message)
                        
                        # Convert to narrative for classification (removes DATE markers, speaker labels)
                        content_for_classification = dialogue_processor.to_narrative(processed)
                        logger.info(f"   → Converted dialogue to narrative for classification: '{content_for_classification[:100]}...'")
                except Exception as e:
                    logger.warning(f"⚠️  Dialogue preprocessing failed: {e}, using original text")
            
            # ========== ML-BASED SCORING (Phase 1.5) ==========
            if self.use_ml_scoring and self.ml_scorer:
                ego_score, confidence, triggers, tier = await self.ml_scorer.score_memory(
                    user_id, content_for_classification, assistant_response
                )
                ego_result = type('obj', (object,), {
                    'ego_score': ego_score,
                    'tier': tier  # Use tier from ML scorer (single source of truth)
                })()
            else:
                # ========== LEGACY REGEX-BASED SCORING (Phase 1) ==========
                # Detect memory triggers
                triggers = self.regex_fallback.detect_triggers(user_message)
                
                if not triggers:
                    logger.debug("No memory triggers detected")
                    return
                
                logger.info(f"Memory triggers detected: {triggers}")
                
                # Calculate dynamic explicit importance based on triggers
                # Inline calculation (same logic as MLScorer._calculate_explicit_importance)
                importance_map = {
                    'identity': 1.0,
                    'family': 1.0,
                    'high_value': 0.95,
                    'preference': 0.9,
                    'fact': 0.7
                }
                explicit_importance = max(importance_map.get(trigger, 0.5) for trigger in triggers) if triggers else 0.5
                
                # Analyze sentiment
                sentiment_score = self._analyze_sentiment(user_message)
                
                # Create a memory with enhanced features
                memory_data = {
                    "content": user_message,
                    "observed_at": datetime.utcnow().isoformat(),
                    "explicit_importance": explicit_importance,  # Dynamic based on triggers!
                    "sentiment_score": sentiment_score,  # Basic sentiment analysis
                    "user_response_length": len(assistant_response)
                }
                
                # Score the memory
                ego_result = self.ego_scorer.calculate(memory=memory_data)
                
                # Use hardcoded confidence for now (Phase 1)
                confidence = 0.7
            
            logger.info(
                f"Memory scored: ego_score={ego_result.ego_score:.2f}, "
                f"tier={ego_result.tier}, confidence={confidence:.2f}"
            )
            
            # Check if should use shadow tier (Tier 1 safety)
            
            node_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat() + 'Z'  # RFC3339 format for Go
            
            use_shadow = False
            if self.shadow_tier:
                logger.debug(
                    f"🔍 Checking Shadow Tier: ego_score={ego_result.ego_score:.4f}, "
                    f"tier={ego_result.tier}, confidence={confidence:.4f}"
                )
                use_shadow = await self.shadow_tier.should_use_shadow_tier(
                    ego_score=ego_result.ego_score,
                    confidence=confidence
                )
                logger.info(
                    f"🚪 Shadow Tier Decision: {'ROUTE TO SHADOW TIER' if use_shadow else 'SKIP (direct store)'}"
                )
            
            if use_shadow:
                # Route through shadow tier (store as Tier 2, promote to Tier 1 after confirmation)
                logger.info(
                    f"Routing to shadow tier: ego_score={ego_result.ego_score:.2f}, "
                    f"confidence={confidence:.2f}"
                )
                
                from core.event_bus import Event
                
                # Store as Tier 2 initially (Shadow Tier candidate)
                # Will be promoted to Tier 1 after user confirmation
                event = Event(
                    topic="memory.upsert",
                    event_type="memory.created",
                    payload={
                        "node_id": node_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "content": user_message,
                        "ego_score": ego_result.ego_score,
                        "tier": 2,  # Store as Tier 2 initially
                        "confidence": confidence,
                        "source": "chat",
                        "observed_at": timestamp,
                        "created_at": timestamp,
                        "last_accessed_at": timestamp,
                        "version": 1,
                        "metadata": {
                            "triggers": triggers,
                            "assistant_response_length": len(assistant_response),
                            "shadow_tier_candidate": True,
                            "original_tier": 1
                        }
                    }
                )
                await self.event_bus.publish(event)
                
                logger.debug(f"Memory published to event bus as Tier 2 (Shadow Tier candidate)")
                
                clarification_id, question = await self.shadow_tier.propose_core_memory({
                    'node_id': node_id,
                    'user_id': user_id,
                    'content': user_message,
                    'summary': user_message[:200],  # First 200 chars as  TO-DO very bad logic write it later
                    'ego_score': ego_result.ego_score,
                    'confidence': confidence,
                    'sources': [session_id],
                    'model_version': 'v1'
                })
                
                # Store pending confirmation for this session
                self.pending_confirmations[session_id] = {
                    'clarification_id': clarification_id,
                    'question': question,
                    'node_id': node_id,
                    'content': user_message,
                    'ego_score': ego_result.ego_score
                }
                
                logger.info(
                    f"Shadow tier memory created: {node_id}, "
                    f"clarification_id={clarification_id}"
                )
                
                # ========== KNOWLEDGE GRAPH CONSOLIDATION (Shadow Tier) ==========
                if self.use_consolidation:
                    try:
                        logger.info("="*80)
                        logger.info(f"🧠 CONSOLIDATION (Shadow Tier): Extracting entities & relations")
                        logger.info(f"   Text: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
                        logger.info(f"   Ego Score: {ego_result.ego_score:.2f}, Tier: 2 (Shadow)")
                        
                        result = await self.consolidation_service.consolidate(
                            text=user_message,
                            ego_score=ego_result.ego_score,
                            tier=2  # Shadow tier stores as Tier 2
                        )
                        
                        entities = result.get("entities", [])
                        relations = result.get("relations", [])
                        
                        logger.info(f"   ✅ Extracted {len(entities)} entities, {len(relations)} relations")
                        if relations:
                            logger.info("   Relations:")
                            for rel in relations[:10]:
                                logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel['confidence']:.2f})")
                        
                        if relations:
                            stored = self.knowledge_graph_store.store_relations(
                                user_id=user_id,
                                memory_id=node_id,
                                relations=relations
                            )
                            logger.info(f"   💾 Stored {stored}/{len(relations)} relations in KG")
                        
                        # Publish v2 memory.upsert with entity tags for Qdrant
                        entity_names = [e.get('name', e) if isinstance(e, dict) else e for e in entities]
                        entity_names = [str(e).lower() for e in entity_names if e]  # Normalize to lowercase
                        
                        if entity_names:
                            entity_update_event = Event(
                                topic="memory.upsert",
                                event_type="memory.entities_extracted",
                                payload={
                                    "node_id": node_id,
                                    "user_id": user_id,
                                    "content": user_message,
                                    "ego_score": ego_result.ego_score,
                                    "tier": 2,
                                    "confidence": confidence,
                                    "source": "chat",
                                    "observed_at": timestamp,
                                    "created_at": timestamp,
                                    "last_accessed_at": timestamp,
                                    "entities": entity_names,  # Entity tags for localized KG
                                    "version": 2,
                                    "metadata": {
                                        "triggers": triggers,
                                        "assistant_response_length": len(assistant_response),
                                        "shadow_tier_candidate": True,
                                        "original_tier": 1
                                    }
                                }
                            )
                            await self.event_bus.publish(entity_update_event)
                            logger.info(f"   📡 Published v2 memory.upsert with {len(entity_names)} entity tags: {entity_names}")
                        else:
                            logger.info(f"   ℹ️  No entities extracted from shadow tier memory")
                        
                        # ALWAYS emit event for KG maintenance (even if no new relations)
                        # The agent will analyze the entire KG against this memory
                        kg_event = Event(
                            topic="events:ego_scoring_complete",
                            event_type="ego_scoring_complete",
                            payload={
                                "user_id": user_id,
                                "memory_id": node_id,
                                "memory_content": user_message,
                                "ego_score": ego_result.ego_score,
                                "tier": 2,  # Shadow tier stores as Tier 2
                                "confidence": confidence,
                                "new_relations": relations
                            }
                        )
                        await self.event_bus.publish(kg_event)
                        logger.debug(f"   📡 Published ego_scoring_complete event for KG maintenance")
                        
                        logger.info("="*80)
                    except Exception as e:
                        logger.warning(f"KG consolidation failed (non-fatal): {e}")
                elif self.use_graph_pipeline and self.graph_pipeline:
                    try:
                        graph_result = await self.graph_pipeline.process_memory(
                            user_id=user_id, memory_id=node_id, content=user_message,
                            ego_score=ego_result.ego_score, tier=2, session_id=session_id,
                            metadata={"triggers": triggers, "shadow_tier_candidate": True}
                        )
                        if graph_result.get("candidate_edges"):
                            logger.info(f"📊 Graph extraction (Shadow Tier): {len(graph_result.get('candidate_edges', []))} edges")
                    except Exception as graph_error:
                        logger.warning(f"Graph processing failed (non-fatal): {graph_error}")
                
            else:
                # Safe to store directly (Tier 2/3/4)
                # NOTE: If ML scorer assigned Tier 1, it should have been caught by Shadow Tier above
                from core.event_bus import Event
                
                # Use the tier determined by ML scorer (single source of truth)
                final_tier = ego_result.tier
                
                # Safety check: Tier 1 should have been routed to Shadow Tier
                if final_tier == 1:
                    logger.error(
                        f"🚨 CRITICAL: Tier 1 memory bypassed Shadow Tier! "
                        f"ego_score={ego_result.ego_score:.2f}, confidence={confidence:.2f}, "
                        f"use_shadow={use_shadow}, shadow_tier_enabled={self.shadow_tier is not None}"
                    )
                    # Downgrade to Tier 2 as safety measure
                    final_tier = 2
                
                event = Event(
                    topic="memory.upsert",
                    event_type="memory.created",
                    payload={
                        "node_id": node_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "content": user_message,
                        "ego_score": ego_result.ego_score,
                        "tier": final_tier,
                        "confidence": confidence,
                        "source": "chat",
                        "observed_at": timestamp,
                        "created_at": timestamp,
                        "last_accessed_at": timestamp,
                        "version": 1,
                        "metadata": {
                            "triggers": triggers,
                            "assistant_response_length": len(assistant_response)
                        }
                    }
                )
                await self.event_bus.publish(event)
                
                logger.debug(f"Memory published to event bus: tier={final_tier}")
                
                # ========== KNOWLEDGE GRAPH CONSOLIDATION ==========
                if self.use_consolidation:
                    try:
                        logger.info("="*80)
                        logger.info(f"🧠 CONSOLIDATION: Extracting entities & relations from user message")
                        logger.info(f"   Text: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
                        logger.info(f"   Ego Score: {ego_result.ego_score:.2f}, Tier: {final_tier}")
                        
                        result = await self.consolidation_service.consolidate(
                            text=user_message,
                            ego_score=ego_result.ego_score,
                            tier=final_tier
                        )
                        
                        entities = result.get("entities", [])
                        relations = result.get("relations", [])
                        
                        logger.info(f"   ✅ Extracted {len(entities)} entities, {len(relations)} relations")
                        if entities:
                            logger.info("   Entities:")
                            for ent in entities[:10]:  # Show first 10
                                logger.info(f"      - {ent['name']} ({ent['type']})")
                        if relations:
                            logger.info("   Relations:")
                            for rel in relations[:10]:  # Show first 10
                                logger.info(f"      - {rel['subject']} --{rel['predicate']}--> {rel['object']} (confidence: {rel['confidence']:.2f})")
                        
                        if relations:
                            stored = self.knowledge_graph_store.store_relations(
                                user_id=user_id,
                                memory_id=node_id,
                                relations=relations
                            )
                            logger.info(f"   💾 Stored {stored}/{len(relations)} relations in KG")
                        
                        # Publish v2 memory.upsert with entity tags for Qdrant
                        entity_names = [e.get('name', e) if isinstance(e, dict) else e for e in entities]
                        entity_names = [str(e).lower() for e in entity_names if e]  # Normalize to lowercase
                        
                        if entity_names:
                            entity_update_event = Event(
                                topic="memory.upsert",
                                event_type="memory.entities_extracted",
                                payload={
                                    "node_id": node_id,
                                    "user_id": user_id,
                                    "content": user_message,
                                    "ego_score": ego_result.ego_score,
                                    "tier": final_tier,
                                    "confidence": confidence,
                                    "source": "chat",
                                    "observed_at": timestamp,
                                    "created_at": timestamp,
                                    "last_accessed_at": timestamp,
                                    "entities": entity_names,  # Entity tags for localized KG
                                    "version": 2,
                                    "metadata": {
                                        "triggers": triggers,
                                        "assistant_response_length": len(assistant_response)
                                    }
                                }
                            )
                            await self.event_bus.publish(entity_update_event)
                            logger.info(f"   📡 Published v2 memory.upsert with {len(entity_names)} entity tags: {entity_names}")
                        else:
                            logger.info(f"   ℹ️  No entities extracted from direct store memory")
                        
                        # ALWAYS emit event for KG maintenance (even if no new relations)
                        # The agent will analyze the entire KG against this memory
                        kg_event = Event(
                            topic="events:ego_scoring_complete",
                            event_type="ego_scoring_complete",
                            payload={
                                "user_id": user_id,
                                "memory_id": node_id,
                                "memory_content": user_message,
                                "ego_score": ego_result.ego_score,
                                "tier": final_tier,
                                "confidence": confidence,
                                "new_relations": relations
                            }
                        )
                        await self.event_bus.publish(kg_event)
                        logger.debug(f"   📡 Published ego_scoring_complete event for KG maintenance")
                        
                        logger.info("="*80)
                    except Exception as e:
                        logger.warning(f"KG consolidation failed (non-fatal): {e}")
                
                if self.use_graph_pipeline and self.graph_pipeline:
                    try:
                        graph_result = await self.graph_pipeline.process_memory(
                            user_id=user_id, memory_id=node_id, content=user_message,
                            ego_score=ego_result.ego_score, tier=final_tier, session_id=session_id,
                            metadata={"triggers": triggers}
                        )
                        if graph_result.get("candidate_edges"):
                            logger.info(f"📊 Graph extraction: {len(graph_result.get('candidate_edges', []))} edges")
                    except Exception as graph_error:
                        logger.warning(f"Graph processing failed (non-fatal): {graph_error}")
            
            # Mark as completed
            if message_id:
                self.mark_message_processing(session_id, message_id, 'completed')
            
        except Exception as e:
            logger.error(f"Error extracting memories: {e}", exc_info=True)
            # Mark as failed (treat as completed for flushing purposes)
            if message_id:
                self.mark_message_processing(session_id, message_id, 'completed')
            # Don't fail the chat if memory extraction fails
    
    def get_pending_confirmation(self, session_id: str) -> Optional[Dict]:
        """
        Get pending shadow tier confirmation for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Pending confirmation dict or None
        """
        return self.pending_confirmations.get(session_id)
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using EmbeddingService
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not self.embedding_service:
            logger.warning("EmbeddingService not initialized, cannot generate embedding")
            return None
        
        return await self.embedding_service.generate(text)
    
    async def handle_shadow_confirmation(
        self,
        session_id: str,
        user_id: str,
        confirmed: bool
    ) -> Dict[str, Any]:
        """
        Handle user's response to shadow tier confirmation
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            confirmed: True if user confirmed, False if rejected
            
        Returns:
            Result dict with status
        """
        pending = self.pending_confirmations.get(session_id)
        if not pending:
            return {
                'status': 'error',
                'message': 'No pending confirmation for this session'
            }
        
        clarification_id = pending['clarification_id']
        node_id = pending['node_id']
        
        try:
            if confirmed:
                # User confirmed - promote to Tier 1
                if self.shadow_tier:
                    await self.shadow_tier.handle_user_confirmation(
                        clarification_id=clarification_id,
                        confirmed=True
                    )
                
                logger.info(f"Shadow memory approved: {node_id}")
                
                # Clear pending confirmation
                del self.pending_confirmations[session_id]
                
                return {
                    'status': 'approved',
                    'message': 'Memory promoted to core memory (Tier 1)',
                    'node_id': node_id
                }
            else:
                # User rejected - demote to Tier 2
                if self.shadow_tier:
                    await self.shadow_tier.handle_user_confirmation(
                        clarification_id=clarification_id,
                        confirmed=False
                    )
                
                logger.info(f"Shadow memory rejected: {node_id}")
                
                # Clear pending confirmation
                del self.pending_confirmations[session_id]
                
                return {
                    'status': 'rejected',
                    'message': 'Memory stored as long-term memory (Tier 2)',
                    'node_id': node_id
                }
                
        except Exception as e:
            logger.error(f"Error handling shadow confirmation: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    

    # NOTE: _determine_tier() has been REMOVED!
    # Tier determination is now handled ONLY by MLScorer._determine_tier()
    # This ensures a single source of truth and prevents inconsistencies.
    
    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """
        Calculate ego score using weighted average (fallback when LightGBM fails/times out).
        
        Uses LightGBM's learned feature importances as weights if available,
        otherwise falls back to hardcoded defaults.
        
        Args:
            scores: Dictionary of component scores
            
        Returns:
            Ego score (0.0-1.0)
        """
        # Get weights from trained LightGBM model (normalized feature importances)
        if self.ml_combiner and self.ml_combiner.is_trained:
            weights = self.ml_combiner.get_normalized_weights()
            logger.info(f"🎯 Using LightGBM-learned weights for fallback")
        else:
            # Hardcoded fallback weights (used when LightGBM not trained)
            weights = {
                'novelty_score': 0.2,
                'frequency_score': 0.1,
                'sentiment_intensity': 0.1,
                'explicit_importance_score': 0.4,
                'engagement_score': 0.2,
                'recency_decay': 0.0,
                'reference_count': 0.0,
                'llm_confidence': 0.0,
                'source_weight': 0.0
            }
            logger.info(f"🎯 Using hardcoded default weights (LightGBM not trained)")
        
        # Calculate weighted sum
        ego_score = sum(
            scores.get(feature, 0.0) * weight
            for feature, weight in weights.items()
        )
        
        logger.info(f"🎯 Weighted average fallback calculation:")
        for feature, weight in weights.items():
            if weight > 0.01:  # Only log significant weights
                score_val = scores.get(feature, 0.0)
                logger.info(f"   {feature:30s}: {score_val:.3f} × {weight:.3f} = {score_val * weight:.3f}")
        logger.info(f"   {'─' * 60}")
        logger.info(f"   TOTAL EGO SCORE: {ego_score:.3f}")
        
        return ego_score
    
    async def _classify_memory(
        self,
        message: str,
        user_id: str,
        conversation_context: Optional[List[Dict]] = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Classify memory type using configured classifier.
        
        Delegates to ClassifierManager for all classification logic.
        
        Args:
            message: User message
            user_id: User ID for adaptive discovery
            conversation_context: Recent conversation history
        
        Returns:
            (predicted_labels, scores)
        """
        if self.classifier_manager:
            return await self.classifier_manager.classify_memory(message, user_id, conversation_context)
        else:
            # Fallback if manager not initialized
            triggers = self.regex_fallback.detect_triggers(message)
            return triggers, {label: 1.0 for label in triggers}
    
