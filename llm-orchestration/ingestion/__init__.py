"""
Datclaw ingestion module — unified extraction pipeline and source adapters.

Use this package for:
  - Typed ingestion jobs (category / subtype)
  - Staged processing metadata (Supermemory-style flow)
  - Pluggable adapters (ChatGPT, files, paste, PDF, etc.)

The existing ``services.ingestion.IngestionService`` still orchestrates scoring,
events, and KG; this module focuses on **extract** and **normalize** into
``ConversationChunk`` lists.
"""

from .contracts import (
    IngestionJobSpec,
    PipelineRunResult,
    ProcessingStage,
    SourceCategory,
    SourceSubtype,
)
from .pipeline import IngestionPipeline
from .registry import AdapterRegistry, build_default_registry

__all__ = [
    "AdapterRegistry",
    "IngestionJobSpec",
    "IngestionPipeline",
    "PipelineRunResult",
    "ProcessingStage",
    "SourceCategory",
    "SourceSubtype",
    "build_default_registry",
]
