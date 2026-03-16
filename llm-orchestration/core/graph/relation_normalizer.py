"""
DAPPY Relation Normalizer
Maps predicate lemmas to canonical relation types.

Key features:
- Lexicon-based mapping for common predicates
- Embedding similarity for unknown predicates
- Type-aware normalization (considers subject/object types)
- Confidence scoring

Phase 1D Implementation
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class RelationNormalizer:
    """
    Normalize predicates to canonical relation types.
    
    Examples:
    - "works" + "at" → "works_at" → "employed_by"
    - "lives" + "in" → "lives_in" → "resides_in"
    - "is" + possessive → "sister_of" → "sibling_of"
    
    Strategy:
    1. Exact lexicon match
    2. Embedding similarity to known relations (if available)
    3. Keep original if no match
    
    Usage:
        normalizer = RelationNormalizer(embedding_service=embedding_service)
        canonical, conf = normalizer.normalize("works_at", "PERSON", "ORG", context)
    """
    
    # Lexicon of common predicate mappings
    PREDICATE_MAPPING = {
        # Employment
        "works_at": "employed_by",
        "work_at": "employed_by",
        "work_for": "employed_by",
        "works_for": "employed_by",
        "employed_by": "employed_by",
        "employee_of": "employed_by",
        
        # Location
        "lives_in": "resides_in",
        "live_in": "resides_in",
        "resides_in": "resides_in",
        "located_in": "located_in",
        "based_in": "located_in",
        
        # Education
        "studies_at": "enrolled_at",
        "study_at": "enrolled_at",
        "attends": "enrolled_at",
        "attend": "enrolled_at",
        "enrolled_at": "enrolled_at",
        "student_at": "enrolled_at",
        
        "studies": "studies_subject",
        "study": "studies_subject",
        "learns": "studies_subject",
        "learn": "studies_subject",
        
        "graduated_from": "graduated_from",
        "graduate_from": "graduated_from",
        "alumnus_of": "graduated_from",
        
        # Family relations
        "sister_of": "sibling_of",
        "brother_of": "sibling_of",
        "sibling_of": "sibling_of",
        
        "mother_of": "parent_of",
        "father_of": "parent_of",
        "parent_of": "parent_of",
        
        "daughter_of": "child_of",
        "son_of": "child_of",
        "child_of": "child_of",
        
        "wife_of": "spouse_of",
        "husband_of": "spouse_of",
        "spouse_of": "spouse_of",
        "married_to": "spouse_of",
        
        # Ownership
        "owns": "owns",
        "own": "owns",
        "possesses": "owns",
        "possess": "owns",
        "has": "has",
        "have": "has",
        
        # Creation
        "creates": "created",
        "create": "created",
        "created": "created",
        "made": "created",
        "make": "created",
        "built": "created",
        "build": "created",
        "founded": "founded",
        "found": "founded",
        
        # Leadership
        "leads": "leads",
        "lead": "leads",
        "manages": "manages",
        "manage": "manages",
        "directs": "directs",
        "direct": "directs",
        "ceo_of": "leads",
        "founder_of": "founded",
        "holds_title": "holds_title",
        
        # Membership
        "member_of": "member_of",
        "part_of": "part_of",
        "belongs_to": "member_of",
        "belong_to": "member_of",
        
        # Interaction
        "knows": "knows",
        "know": "knows",
        "met": "met",
        "meet": "met",
        "contacted": "contacted",
        "contact": "contacted",
        
        # Preference
        "likes": "likes",
        "like": "likes",
        "loves": "loves",
        "love": "loves",
        "prefers": "prefers",
        "prefer": "prefers",
        "dislikes": "dislikes",
        "dislike": "dislikes",
        "hates": "hates",
        "hate": "hates",
        
        # Action
        "uses": "uses",
        "use": "uses",
        "utilizes": "uses",
        "utilize": "uses",
        "applies": "uses",
        "apply": "uses",
        
        # Temporal
        "started": "started",
        "start": "started",
        "began": "started",
        "begin": "started",
        "ended": "ended",
        "end": "ended",
        "finished": "ended",
        "finish": "ended",
        
        # Attribution
        "is": "is_a",
        "are": "is_a",
        "be": "is_a",
        "was": "is_a",
        "were": "is_a",
    }
    
    # Type-specific relation preferences
    TYPE_PREFERENCES = {
        ("PERSON", "ORG"): ["employed_by", "member_of", "founded", "leads"],
        ("PERSON", "GPE"): ["resides_in", "born_in", "citizen_of"],
        ("PERSON", "LOC"): ["resides_in", "visited", "located_in"],
        ("PERSON", "PERSON"): ["sibling_of", "parent_of", "child_of", "spouse_of", "knows", "met"],
        ("ORG", "GPE"): ["located_in", "based_in", "headquartered_in"],
        ("ORG", "ORG"): ["part_of", "subsidiary_of", "partner_of"],
    }
    
    def __init__(
        self,
        embedding_service=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize relation normalizer.
        
        Args:
            embedding_service: Optional embedding service for similarity matching
            config: Optional configuration dict
        """
        self.embedding_service = embedding_service
        self.config = config or {}
        
        # Load custom mappings from config if provided
        custom_mappings = self.config.get("custom_predicate_mappings", {})
        if custom_mappings:
            self.PREDICATE_MAPPING.update(custom_mappings)
        
        # LLM client (lazy loaded)
        self._llm_client = None
        
        # Training data collector (lazy loaded)
        self._training_collector = None
        
        logger.info(f"✅ RelationNormalizer initialized with {len(self.PREDICATE_MAPPING)} mappings")
    
    @property
    def llm_client(self):
        """Lazy load OpenAI client."""
        if self._llm_client is None:
            api_key = os.getenv("OPENAI_API_KEY") or self.config.get("llm_api", {}).get("api_key")
            if api_key:
                self._llm_client = AsyncOpenAI(api_key=api_key)
            else:
                logger.warning("OpenAI API key not found, LLM fallback will not work")
        return self._llm_client
    
    @property
    def training_collector(self):
        """Lazy load training collector."""
        if self._training_collector is None:
            try:
                from .normalization_training_collector import NormalizationTrainingCollector
                db_path = self.config.get("normalization_training", {}).get(
                    "db_path", "data/normalization_training.db"
                )
                self._training_collector = NormalizationTrainingCollector(db_path=db_path)
            except Exception as e:
                logger.warning(f"Failed to initialize training collector: {e}")
        return self._training_collector
    
    async def normalize(
        self,
        predicate_lemma: str,
        subject_type: Optional[str] = None,
        object_type: Optional[str] = None,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Normalize predicate to canonical relation type with LLM fallback.
        
        Args:
            predicate_lemma: Lemmatized predicate (e.g., "works_at", "study")
            subject_type: Entity type of subject (e.g., "PERSON", "ORG")
            object_type: Entity type of object
            context: Full sentence context (for LLM fallback)
            user_id: User ID for training data collection
            memory_id: Memory ID for training data collection
            session_id: Session ID for training data collection
            
        Returns:
            Tuple of (canonical_relation, confidence)
        """
        # Strategy 1: Exact lexicon match
        if predicate_lemma in self.PREDICATE_MAPPING:
            canonical = self.PREDICATE_MAPPING[predicate_lemma]
            confidence = 0.95
            source = "lexicon"
            
            logger.debug(
                f"📊 Normalization for '{predicate_lemma}' ({subject_type} → {object_type}): "
                f"method={source}, canonical='{canonical}' → CONFIDENCE={confidence:.3f}"
            )
            return (canonical, confidence)
        
        # Strategy 2: Type-aware matching
        if subject_type and object_type:
            type_pair = (subject_type, object_type)
            if type_pair in self.TYPE_PREFERENCES:
                # Check if predicate matches any preferred relation for this type pair
                for preferred in self.TYPE_PREFERENCES[type_pair]:
                    if predicate_lemma in preferred or preferred in predicate_lemma:
                        canonical = preferred
                        confidence = 0.85
                        source = "type-aware"
                        
                        logger.debug(
                            f"📊 Normalization for '{predicate_lemma}' ({subject_type} → {object_type}): "
                            f"method={source}, canonical='{canonical}' → CONFIDENCE={confidence:.3f}"
                        )
                        
                        # Check if LLM fallback is needed
                        if confidence < 0.75 and context:
                            canonical, confidence = await self._try_llm_fallback(
                                predicate_lemma, subject_type, object_type, context,
                                canonical, confidence, source, user_id, memory_id, session_id
                            )
                        
                        return (canonical, confidence)
        
        # Strategy 3: Embedding similarity (if available)
        if self.embedding_service and context:
            similar = await self._find_similar_relation(predicate_lemma, context)
            if similar:
                canonical, similarity = similar
                confidence = 0.70 + (similarity * 0.20)  # 0.70-0.90 range
                source = "embedding"
                
                logger.debug(
                    f"📊 Normalization for '{predicate_lemma}' ({subject_type} → {object_type}): "
                    f"method={source}, canonical='{canonical}', similarity={similarity:.2f} → CONFIDENCE={confidence:.3f}"
                )
                
                # Check if LLM fallback is needed
                if confidence < 0.75 and context:
                    canonical, confidence = await self._try_llm_fallback(
                        predicate_lemma, subject_type, object_type, context,
                        canonical, confidence, source, user_id, memory_id, session_id
                    )
                
                return (canonical, confidence)
        
        # Strategy 4: Keep original (no match found) - ALWAYS trigger LLM fallback
        canonical = predicate_lemma
        confidence = 0.70
        source = "original"
        
        logger.debug(
            f"📊 Normalization for '{predicate_lemma}' ({subject_type} → {object_type}): "
            f"method={source}, canonical='{canonical}' (no match) → CONFIDENCE={confidence:.3f}"
        )
        
        # Always try LLM for unknown predicates
        if context:
            canonical, confidence = await self._try_llm_fallback(
                predicate_lemma, subject_type, object_type, context,
                canonical, confidence, source, user_id, memory_id, session_id
            )
        
        return (canonical, confidence)
    
    async def normalize_batch(
        self,
        predicates: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Batch normalize multiple predicates with a single LLM call.
        
        This is MUCH more efficient than calling normalize() individually for each predicate.
        
        Args:
            predicates: List of dicts with keys:
                - predicate_lemma: str
                - subject_type: Optional[str]
                - object_type: Optional[str]
                - context: str
                - user_id: Optional[str]
                - memory_id: Optional[str]
                - session_id: Optional[str]
        
        Returns:
            List of (canonical_relation, confidence) tuples, in the same order as input
        """
        if not predicates:
            return []
        
        logger.info(f"🔄 Batch normalizing {len(predicates)} predicates")
        
        # First, try rule-based normalization for all
        results = []
        needs_llm = []  # Track which ones need LLM fallback
        
        for i, pred in enumerate(predicates):
            predicate_lemma = pred["predicate_lemma"]
            subject_type = pred.get("subject_type")
            object_type = pred.get("object_type")
            
            # Try lexicon match
            if predicate_lemma in self.PREDICATE_MAPPING:
                canonical = self.PREDICATE_MAPPING[predicate_lemma]
                confidence = 0.95
                results.append({
                    "index": i,
                    "canonical": canonical,
                    "confidence": confidence,
                    "method": "lexicon",
                    "needs_llm": False
                })
                continue
            
            # Try type-aware matching
            if subject_type and object_type:
                type_pair = (subject_type, object_type)
                if type_pair in self.TYPE_PREFERENCES:
                    for preferred in self.TYPE_PREFERENCES[type_pair]:
                        if predicate_lemma in preferred or preferred in predicate_lemma:
                            canonical = preferred
                            confidence = 0.85
                            results.append({
                                "index": i,
                                "canonical": canonical,
                                "confidence": confidence,
                                "method": "type-aware",
                                "needs_llm": confidence < 0.75
                            })
                            if confidence < 0.75:
                                needs_llm.append(i)
                            break
                    else:
                        # No type-aware match, keep original and mark for LLM
                        results.append({
                            "index": i,
                            "canonical": predicate_lemma,
                            "confidence": 0.70,
                            "method": "original",
                            "needs_llm": True
                        })
                        needs_llm.append(i)
                else:
                    # No type-aware match, keep original and mark for LLM
                    results.append({
                        "index": i,
                        "canonical": predicate_lemma,
                        "confidence": 0.70,
                        "method": "original",
                        "needs_llm": True
                    })
                    needs_llm.append(i)
            else:
                # No types, keep original and mark for LLM
                results.append({
                    "index": i,
                    "canonical": predicate_lemma,
                    "confidence": 0.70,
                    "method": "original",
                    "needs_llm": True
                })
                needs_llm.append(i)
        
        # If no predicates need LLM, return early
        if not needs_llm:
            logger.info(f"✅ All {len(predicates)} predicates normalized via rules (no LLM needed)")
            return [(r["canonical"], r["confidence"]) for r in sorted(results, key=lambda x: x["index"])]
        
        # Batch LLM call for all low-confidence predicates
        logger.info(f"🤖 Sending {len(needs_llm)} predicates to LLM for batch normalization")
        
        llm_results = await self._normalize_batch_with_llm(
            predicates=[predicates[i] for i in needs_llm]
        )
        
        # Update results with LLM improvements
        for idx, llm_result in zip(needs_llm, llm_results):
            original_result = results[idx]
            
            # Log for training
            if self.training_collector:
                pred = predicates[idx]
                self.training_collector.log_normalization(
                    predicate_lemma=pred["predicate_lemma"],
                    subject_type=pred.get("subject_type"),
                    object_type=pred.get("object_type"),
                    context=pred.get("context", ""),
                    rule_result=original_result["canonical"],
                    rule_confidence=original_result["confidence"],
                    rule_method=original_result["method"],
                    llm_result=llm_result["canonical_relation"],
                    llm_confidence=llm_result["confidence"],
                    llm_reasoning=llm_result.get("reasoning", ""),
                    user_id=pred.get("user_id"),
                    memory_id=pred.get("memory_id"),
                    session_id=pred.get("session_id")
                )
            
            # Use LLM result if confidence is higher
            if llm_result["confidence"] > original_result["confidence"]:
                logger.info(
                    f"✅ LLM improved normalization [{idx}]: {original_result['canonical']} → {llm_result['canonical_relation']} "
                    f"(conf={original_result['confidence']:.2f} → {llm_result['confidence']:.2f})"
                )
                results[idx]["canonical"] = llm_result["canonical_relation"]
                results[idx]["confidence"] = llm_result["confidence"]
        
        # Return in original order
        return [(r["canonical"], r["confidence"]) for r in sorted(results, key=lambda x: x["index"])]
    
    async def _try_llm_fallback(
        self,
        predicate_lemma: str,
        subject_type: Optional[str],
        object_type: Optional[str],
        context: str,
        rule_result: str,
        rule_confidence: float,
        rule_method: str,
        user_id: Optional[str],
        memory_id: Optional[str],
        session_id: Optional[str]
    ) -> Tuple[str, float]:
        """
        Try LLM fallback for normalization.
        
        Returns updated (canonical_relation, confidence) if LLM improves it.
        """
        NORMALIZATION_CONFIDENCE_THRESHOLD = 0.75
        
        logger.debug(
            f"🤖 Normalization confidence ({rule_confidence:.2f}) < {NORMALIZATION_CONFIDENCE_THRESHOLD:.2f}, "
            f"trying LLM fallback"
        )
        
        llm_result = await self._normalize_with_llm(
            predicate_lemma=predicate_lemma,
            subject_type=subject_type,
            object_type=object_type,
            context=context,
            rule_suggestion=rule_result
        )
        
        # Log for training
        if self.training_collector:
            self.training_collector.log_normalization(
                predicate_lemma=predicate_lemma,
                subject_type=subject_type,
                object_type=object_type,
                context=context,
                rule_result=rule_result,
                rule_confidence=rule_confidence,
                rule_method=rule_method,
                llm_result=llm_result["canonical_relation"],
                llm_confidence=llm_result["confidence"],
                llm_reasoning=llm_result["reasoning"],
                user_id=user_id,
                memory_id=memory_id,
                session_id=session_id
            )
        
        # Use LLM result if confidence is higher
        if llm_result["confidence"] > rule_confidence:
            logger.info(
                f"✅ LLM improved normalization: {rule_result} → {llm_result['canonical_relation']} "
                f"(conf={rule_confidence:.2f} → {llm_result['confidence']:.2f})"
            )
            return (llm_result["canonical_relation"], llm_result["confidence"])
        else:
            logger.debug(f"LLM did not improve confidence, keeping rule result")
            return (rule_result, rule_confidence)
    
    async def _normalize_with_llm(
        self,
        predicate_lemma: str,
        subject_type: Optional[str],
        object_type: Optional[str],
        context: str,
        rule_suggestion: str
    ) -> Dict[str, Any]:
        """
        Use LLM to normalize predicate when rule-based approach is uncertain.
        
        Args:
            predicate_lemma: Original predicate lemma
            subject_type: Entity type of subject
            object_type: Entity type of object
            context: Full sentence context
            rule_suggestion: Best guess from rule-based normalization
        
        Returns:
            Dict with canonical_relation, confidence, reasoning
        """
        if not self.llm_client:
            logger.warning("LLM client not available for normalization fallback")
            return {"canonical_relation": rule_suggestion, "confidence": 0.70, "reasoning": "LLM unavailable"}
        
        # Available canonical relations (from PREDICATE_MAPPING values)
        canonical_relations = sorted(set(self.PREDICATE_MAPPING.values()))
        
        prompt = f"""Given the following context, normalize the predicate to a canonical relation type.

Context: "{context}"
Predicate: "{predicate_lemma}"
Subject Type: {subject_type or "unknown"}
Object Type: {object_type or "unknown"}

Rule-based suggestion: "{rule_suggestion}"

Available canonical relations:
{', '.join(canonical_relations[:50])}  # Top 50 most common

Task: Choose the BEST canonical relation that captures the semantic meaning of the predicate in this context.

Return JSON:
{{
  "canonical_relation": "<best canonical relation>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation why this relation fits best>"
}}
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get("llm_api", {}).get("model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for more consistent results
            )
            
            result = json.loads(response.choices[0].message.content)
            
            logger.debug(
                f"🤖 LLM normalization: {predicate_lemma} → {result['canonical_relation']} "
                f"(conf={result['confidence']:.2f}, reasoning: {result['reasoning'][:50]}...)"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"LLM normalization failed: {e}")
            return {"canonical_relation": rule_suggestion, "confidence": 0.70, "reasoning": f"LLM error: {e}"}
    
    async def _find_similar_relation(
        self,
        predicate: str,
        context: str
    ) -> Optional[Tuple[str, float]]:
        """
        Find similar relation using embedding similarity.
        
        Args:
            predicate: Predicate to match
            context: Sentence context
            
        Returns:
            Tuple of (canonical_relation, similarity_score) or None
        """
        if not self.embedding_service:
            return None
        
        try:
            # Get embedding for predicate in context
            query_text = f"{predicate} in context: {context}"
            query_emb = await self.embedding_service.generate(query_text)
            
            # Compare to known relation descriptions
            best_match = None
            best_score = 0.0
            
            # Sample canonical relations with descriptions
            relation_descriptions = {
                "employed_by": "works at a company or organization",
                "resides_in": "lives in a location or city",
                "studies_subject": "studies or learns a subject",
                "enrolled_at": "attends or is enrolled at an institution",
                "sibling_of": "is a brother or sister of",
                "parent_of": "is the mother or father of",
                "spouse_of": "is married to",
                "owns": "possesses or owns something",
                "created": "made or built something",
                "founded": "started or established an organization",
                "leads": "manages or directs",
                "knows": "is acquainted with",
                "likes": "has a preference for",
            }
            
            for relation, description in relation_descriptions.items():
                desc_emb = await self.embedding_service.generate(description)
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_emb, desc_emb)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = relation
            
            # Only return if similarity is above threshold
            if best_score > 0.60:
                return (best_match, best_score)
            
        except Exception as e:
            logger.warning(f"Embedding-based normalization failed: {e}")
        
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _normalize_batch_with_llm(
        self,
        predicates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch normalize multiple predicates with a SINGLE LLM call.
        
        This is the key optimization: instead of N separate API calls, we make ONE call.
        
        Args:
            predicates: List of dicts with predicate_lemma, subject_type, object_type, context
        
        Returns:
            List of dicts with canonical_relation, confidence, reasoning
        """
        if not self.llm_client:
            logger.warning("LLM client not available for batch normalization")
            return [
                {"canonical_relation": p["predicate_lemma"], "confidence": 0.70, "reasoning": "LLM unavailable"}
                for p in predicates
            ]
        
        # Available canonical relations (from PREDICATE_MAPPING values)
        canonical_relations = sorted(set(self.PREDICATE_MAPPING.values()))
        
        # Build batch prompt
        predicates_json = []
        for i, pred in enumerate(predicates):
            predicates_json.append({
                "id": i,
                "context": pred.get("context", ""),
                "predicate": pred["predicate_lemma"],
                "subject_type": pred.get("subject_type", "unknown"),
                "object_type": pred.get("object_type", "unknown")
            })
        
        prompt = f"""Given the following predicates extracted from text, normalize each to a canonical relation type.

Available canonical relations:
{', '.join(canonical_relations[:50])}  # Top 50 most common

Predicates to normalize:
{json.dumps(predicates_json, indent=2)}

Task: For EACH predicate, choose the BEST canonical relation that captures its semantic meaning in context.

Return JSON array (one object per predicate, in the same order):
[
  {{
    "id": 0,
    "canonical_relation": "<best canonical relation>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
  }},
  ...
]
"""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.config.get("normalization_llm_fallback", {}).get("llm_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a semantic relation normalizer. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            results = json.loads(result_text)
            
            # Validate and sort by ID
            if not isinstance(results, list):
                raise ValueError("LLM did not return a list")
            
            # Sort by ID to ensure correct order
            results = sorted(results, key=lambda x: x.get("id", 0))
            
            # Validate each result
            validated_results = []
            for i, result in enumerate(results):
                if "canonical_relation" not in result:
                    logger.warning(f"LLM result {i} missing canonical_relation, using original")
                    validated_results.append({
                        "canonical_relation": predicates[i]["predicate_lemma"],
                        "confidence": 0.70,
                        "reasoning": "LLM parse error"
                    })
                else:
                    validated_results.append({
                        "canonical_relation": result["canonical_relation"],
                        "confidence": result.get("confidence", 0.85),
                        "reasoning": result.get("reasoning", "")
                    })
            
            return validated_results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM batch normalization response: {e}")
            return [
                {"canonical_relation": p["predicate_lemma"], "confidence": 0.70, "reasoning": "Parse error"}
                for p in predicates
            ]
        except Exception as e:
            logger.error(f"LLM batch normalization failed: {e}")
            return [
                {"canonical_relation": p["predicate_lemma"], "confidence": 0.70, "reasoning": f"Error: {e}"}
                for p in predicates
            ]
    
    def add_mapping(self, predicate: str, canonical: str):
        """
        Add a custom predicate mapping.
        
        Args:
            predicate: Predicate lemma
            canonical: Canonical relation type
        """
        self.PREDICATE_MAPPING[predicate] = canonical
        logger.info(f"Added custom mapping: {predicate} → {canonical}")
    
    def get_all_canonical_relations(self) -> List[str]:
        """Get list of all canonical relation types."""
        return list(set(self.PREDICATE_MAPPING.values()))

