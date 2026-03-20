"""
Extraction pipeline — staged flow (validate → extract → ready).

Downstream embedding / indexing stays in your existing services (Qdrant, Arango, etc.).
"""

from __future__ import annotations

import time
from typing import Any, List, Tuple

from .contracts import (
    IngestionJobSpec,
    PipelineRunResult,
    ProcessingStage,
    StageRecord,
)
from .registry import AdapterRegistry, resolve_source_type

from services.ingestion.models import ConversationChunk


class IngestionPipeline:
    """
    Runs the **extract** phase using a pluggable :class:`AdapterRegistry`.

    Use with :func:`ingestion.registry.build_default_registry` or a custom registry
    in tests.
    """

    def __init__(self, registry: AdapterRegistry) -> None:
        self._registry = registry

    async def run_extract(
        self, spec: IngestionJobSpec
    ) -> Tuple[List[ConversationChunk], PipelineRunResult]:
        """
        Validate job, run adapter.extract, return chunks + stage metadata.

        Does not write to databases or publish events.
        """
        result = PipelineRunResult()
        t0 = time.perf_counter()

        # validating
        if not (spec.source and str(spec.source).strip()):
            result.error = "Empty source"
            result.stages.append(
                StageRecord(name=ProcessingStage.VALIDATING, ok=False, detail=result.error)
            )
            result.stages.append(StageRecord(name=ProcessingStage.FAILED, ok=False))
            return [], result

        result.stages.append(
            StageRecord(
                name=ProcessingStage.VALIDATING,
                ok=True,
                detail="ok",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        key = resolve_source_type(spec, self._registry)
        try:
            adapter = self._registry.require(key)
        except KeyError as e:
            result.error = str(e)
            result.stages.append(
                StageRecord(name=ProcessingStage.EXTRACTING, ok=False, detail=result.error)
            )
            result.stages.append(StageRecord(name=ProcessingStage.FAILED, ok=False))
            return [], result

        t1 = time.perf_counter()
        try:
            chunks = await adapter.extract(spec.source, **spec.parser_kwargs)
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.stages.append(
                StageRecord(
                    name=ProcessingStage.EXTRACTING,
                    ok=False,
                    detail=result.error,
                    duration_ms=(time.perf_counter() - t1) * 1000,
                )
            )
            result.stages.append(StageRecord(name=ProcessingStage.FAILED, ok=False))
            return [], result

        # Normalize source_type on chunks for downstream metadata
        st = adapter.source_type
        for c in chunks:
            if not getattr(c, "source_type", None) or c.source_type == "unknown":
                c.source_type = st

        result.stages.append(
            StageRecord(
                name=ProcessingStage.EXTRACTING,
                ok=True,
                detail=f"{len(chunks)} chunks",
                duration_ms=(time.perf_counter() - t1) * 1000,
            )
        )
        result.stages.append(StageRecord(name=ProcessingStage.READY, ok=True))
        result.chunk_count = len(chunks)
        return chunks, result

    async def run_extract_only(
        self,
        source_type_key: str,
        source: str,
        **parser_kwargs: Any,
    ) -> Tuple[List[ConversationChunk], PipelineRunResult]:
        """Convenience: build a minimal spec from a legacy source_type string."""
        from .contracts import IngestionJobSpec

        spec = IngestionJobSpec(
            source=source,
            source_type=source_type_key,
            parser_kwargs=dict(parser_kwargs),
        )
        return await self.run_extract(spec)
