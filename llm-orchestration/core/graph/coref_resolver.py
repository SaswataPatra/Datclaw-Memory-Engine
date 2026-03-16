"""
DAPPY Coreference Resolution
Resolves pronouns and references before dependency parsing.

Key features:
- Resolves pronouns: "She studies" → "Sarah studies"
- Resolves demonstratives: "That company" → "Google"
- Maintains coref clusters for entity linking
- Supports neuralcoref and allennlp backends

Phase 1D Implementation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CorefCluster:
    """Represents a coreference cluster (group of mentions referring to same entity)."""
    cluster_id: int
    mentions: List[Tuple[int, int]]  # List of (start_char, end_char) spans
    main_mention: str  # The canonical/main mention text
    main_span: Tuple[int, int]  # Span of the main mention
    
    def contains_span(self, start: int, end: int) -> bool:
        """Check if this cluster contains a given span."""
        return any(s == start and e == end for s, e in self.mentions)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "mentions": self.mentions,
            "main_mention": self.main_mention,
            "main_span": self.main_span
        }


class CorefResolver:
    """
    Coreference resolution using neuralcoref or allennlp.
    
    Resolves pronouns and references to their canonical mentions:
    - "Sarah is my sister. She studies physics." 
      → "Sarah is my sister. Sarah studies physics."
    
    Usage:
        resolver = CorefResolver(provider="neuralcoref")
        resolved_text, clusters = resolver.resolve(text)
    """
    
    def __init__(
        self,
        provider: str = "neuralcoref",
        model: str = "en_core_web_lg",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize coreference resolver.
        
        Args:
            provider: "neuralcoref" or "allennlp" or "simple" (fallback)
            model: spaCy model name for neuralcoref
            config: Optional configuration dict
        """
        self.provider = provider
        self.model_name = model
        self.config = config or {}
        self.nlp = None
        self.coref_model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load coreference resolution model."""
        import spacy
        
        try:
            logger.info(f"🔗 CorefResolver: Loading '{self.provider}' provider with model '{self.model_name}'")
            if self.provider == "neuralcoref":
                self._load_neuralcoref()
            elif self.provider == "allennlp":
                self._load_allennlp()
            elif self.provider == "simple":
                self._load_simple()
            else:
                logger.warning(f"Unknown provider '{self.provider}', falling back to simple")
                self._load_simple()
            logger.info(f"✅ CorefResolver: Successfully loaded '{self.provider}' provider")
        except Exception as e:
            logger.warning(f"Failed to load {self.provider}: {e}, falling back to simple resolver")
            self._load_simple()
    
    def _load_neuralcoref(self):
        """Load neuralcoref model."""
        import spacy
        
        try:
            import neuralcoref
            
            # Load spaCy model
            self.nlp = spacy.load(self.model_name)
            
            # Add neuralcoref to pipeline
            neuralcoref.add_to_pipe(self.nlp)
            
            logger.info(f"✅ Loaded neuralcoref with {self.model_name}")
            self.provider = "neuralcoref"
            
        except ImportError:
            logger.warning("neuralcoref not installed. Install with: pip install neuralcoref")
            raise
        except OSError:
            logger.warning(f"spaCy model {self.model_name} not found. Install with: python -m spacy download {self.model_name}")
            raise
    
    def _load_allennlp(self):
        """Load allennlp coreference model."""
        try:
            from allennlp.predictors.predictor import Predictor
            import spacy
            
            # Load allennlp coref model
            self.coref_model = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
            )
            
            # Also load spaCy for tokenization
            self.nlp = spacy.load(self.model_name)
            
            logger.info("✅ Loaded allennlp coreference model")
            self.provider = "allennlp"
            
        except ImportError:
            logger.warning("allennlp not installed. Install with: pip install allennlp allennlp-models")
            raise
    
    def _load_simple(self):
        """Load simple rule-based resolver (fallback)."""
        import spacy
        
        # Just load spaCy, we'll use simple pronoun rules
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            # Try smaller model
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info("✅ Loaded simple rule-based coref resolver (fallback)")
        self.provider = "simple"
    
    def resolve(self, text: str) -> Tuple[str, List[CorefCluster]]:
        """
        Resolve coreferences in text.
        
        Args:
            text: Input text with pronouns/references
            
        Returns:
            Tuple of (resolved_text, coref_clusters)
            - resolved_text: Text with pronouns replaced by canonical mentions
            - coref_clusters: List of CorefCluster objects
        """
        if not text or not text.strip():
            return text, []
        
        if self.provider == "neuralcoref":
            return self._resolve_neuralcoref(text)
        elif self.provider == "allennlp":
            return self._resolve_allennlp(text)
        else:
            return self._resolve_simple(text)
    
    def _resolve_neuralcoref(self, text: str) -> Tuple[str, List[CorefCluster]]:
        """Resolve using neuralcoref."""
        doc = self.nlp(text)
        
        if not doc._.has_coref:
            logger.debug("No coreferences found")
            return text, []
        
        # Extract clusters
        clusters = []
        for cluster_id, cluster in enumerate(doc._.coref_clusters):
            # Main mention is the first/most representative mention
            main = cluster.main
            main_span = (main.start_char, main.end_char)
            main_text = main.text
            
            # All mentions in this cluster
            mentions = [(m.start_char, m.end_char) for m in cluster.mentions]
            
            clusters.append(CorefCluster(
                cluster_id=cluster_id,
                mentions=mentions,
                main_mention=main_text,
                main_span=main_span
            ))
        
        # Get resolved text (pronouns replaced)
        resolved_text = doc._.coref_resolved
        
        logger.info(f"🔗 CorefResolver: Resolved {len(clusters)} clusters")
        return resolved_text, clusters
    
    def _resolve_allennlp(self, text: str) -> Tuple[str, List[CorefCluster]]:
        """Resolve using allennlp."""
        # Run allennlp prediction
        prediction = self.coref_model.predict(document=text)
        
        # Extract clusters from prediction
        clusters = []
        document = prediction["document"]
        
        for cluster_id, cluster_spans in enumerate(prediction["clusters"]):
            # cluster_spans is list of [start_token, end_token] pairs
            mentions = []
            main_mention_text = None
            main_span = None
            
            for start_tok, end_tok in cluster_spans:
                # Convert token indices to character offsets
                # This is approximate - allennlp uses token indices
                start_char = len(" ".join(document[:start_tok]))
                end_char = start_char + len(" ".join(document[start_tok:end_tok+1]))
                
                mentions.append((start_char, end_char))
                
                # Use first mention as main
                if main_mention_text is None:
                    main_mention_text = " ".join(document[start_tok:end_tok+1])
                    main_span = (start_char, end_char)
            
            clusters.append(CorefCluster(
                cluster_id=cluster_id,
                mentions=mentions,
                main_mention=main_mention_text,
                main_span=main_span
            ))
        
        # Build resolved text by replacing pronouns
        resolved_text = self._build_resolved_text(text, clusters)
        
        logger.info(f"🔗 CorefResolver: Resolved {len(clusters)} clusters")
        return resolved_text, clusters
    
    def _resolve_simple(self, text: str) -> Tuple[str, List[CorefCluster]]:
        """
        Simple rule-based resolver (fallback).
        
        Handles:
        - Personal pronouns (he, she, him, her, his, hers)
        - Relative pronouns (who, which, that) in relative clauses
        - Subject pronouns in same/adjacent sentences
        
        Not as accurate as neuralcoref/allennlp but works without dependencies.
        """
        logger.debug(f"🔗 CorefResolver (simple): Processing text of length {len(text)}")
        doc = self.nlp(text)
        
        clusters = []
        replacements = []  # List of (start_char, end_char, replacement_text)
        
        # Track entities by type for better resolution
        entities_by_type = {}
        for ent in doc.ents:
            if ent.label_ not in entities_by_type:
                entities_by_type[ent.label_] = []
            entities_by_type[ent.label_].append(ent)
        
        # Process each sentence
        for sent_idx, sent in enumerate(doc.sents):
            # Find the main subject of this sentence (if any)
            main_subject = None
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    main_subject = token
                    break
            
            # Find PERSON entities in current and previous sentences
            persons = []
            for ent in doc.ents:
                if ent.label_ == "PERSON" and ent.start_char <= sent.end_char:
                    persons.append(ent)
            
            # Most recent person is likely antecedent
            antecedent = persons[-1] if persons else None
            
            # Process each token in the sentence
            for token in sent:
                replacement = None
                
                # 1. Handle relative pronouns (who, which, that) in relative clauses
                # NOTE: We DON'T replace relative pronouns in the text (they're grammatically necessary)
                # Instead, we just track them in clusters for entity resolution
                if token.lower_ in ("who", "which", "that") and token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj"):
                    # Find the head noun this relative clause modifies
                    head = token.head
                    
                    # Walk up to find the noun being modified
                    while head and head.dep_ in ("relcl", "acl"):
                        head = head.head
                    
                    # Track the coreference but don't replace
                    if head and head.pos_ in ("NOUN", "PROPN"):
                        logger.debug(f"   Tracked relative pronoun '{token.text}' → '{head.text}' (not replacing)")
                        # Don't set replacement - relative pronouns stay in text
                    elif antecedent:
                        logger.debug(f"   Tracked relative pronoun '{token.text}' → '{antecedent.text}' (not replacing)")
                        # Don't set replacement
                
                # 2. Handle personal pronouns (he, she, him, her, his, hers, they, them)
                elif token.pos_ == "PRON" and token.lower_ in ("he", "she", "him", "her", "his", "hers", "they", "them", "their"):
                    if antecedent:
                        replacement = antecedent.text
                        logger.debug(f"   Resolved personal pronoun '{token.text}' → '{replacement}'")
                
                # 3. Handle demonstratives (this, that, these, those) when they refer to entities
                elif token.lower_ in ("this", "that", "these", "those") and token.dep_ in ("nsubj", "dobj", "pobj"):
                    # Try to find a nearby entity
                    if antecedent:
                        replacement = antecedent.text
                        logger.debug(f"   Resolved demonstrative '{token.text}' → '{replacement}'")
                
                # Add replacement if found
                if replacement:
                    replacements.append((token.idx, token.idx + len(token.text), replacement))
        
        # Apply replacements (in reverse order to maintain offsets)
        replacements.sort(key=lambda x: x[0], reverse=True)
        resolved_text = text
        
        for start, end, replacement in replacements:
            resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
        
        if replacements:
            logger.info(f"🔗 CorefResolver (simple): Resolved {len(replacements)} pronouns")
        else:
            logger.info("🔗 CorefResolver (simple): No pronouns to resolve")
        
        return resolved_text, clusters
    
    def _build_resolved_text(self, text: str, clusters: List[CorefCluster]) -> str:
        """Build resolved text by replacing mentions with main mention."""
        resolved = text
        
        # Sort mentions by position (reverse order to maintain offsets)
        replacements = []
        for cluster in clusters:
            main = cluster.main_mention
            for start, end in cluster.mentions:
                if (start, end) != cluster.main_span:
                    replacements.append((start, end, main))
        
        # Sort by start position (descending) to maintain offsets
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply replacements
        for start, end, replacement in replacements:
            resolved = resolved[:start] + replacement + resolved[end:]
        
        return resolved
    
    def get_cluster_for_span(
        self,
        span: Tuple[int, int],
        clusters: List[CorefCluster]
    ) -> Optional[CorefCluster]:
        """
        Get the coreference cluster for a given span.
        
        Args:
            span: (start_char, end_char) tuple
            clusters: List of CorefCluster objects
            
        Returns:
            CorefCluster if found, None otherwise
        """
        start, end = span
        for cluster in clusters:
            if cluster.contains_span(start, end):
                return cluster
        return None
    
    def get_canonical_mention(
        self,
        span: Tuple[int, int],
        clusters: List[CorefCluster]
    ) -> Optional[str]:
        """
        Get the canonical mention for a given span.
        
        Args:
            span: (start_char, end_char) tuple
            clusters: List of CorefCluster objects
            
        Returns:
            Canonical mention text if found, None otherwise
        """
        cluster = self.get_cluster_for_span(span, clusters)
        if cluster:
            return cluster.main_mention
        return None

