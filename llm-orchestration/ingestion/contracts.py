"""
Ingestion contracts — job specs and pipeline results (Supermemory-style stages).

Aligned with a single internal representation: ``services.ingestion.models.ConversationChunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceCategory(str, Enum):
    """High-level source family."""

    LLM_CHAT = "llm_chat"  # share links, exports, API-backed convos
    FILE = "file"
    WEB = "web"
    API = "api"


class SourceSubtype(str, Enum):
    """
    Concrete source; extend as you add adapters.

    Note: Adapters auto-register via ``build_default_registry()``.
    Add new enum values here for API validation and type safety, but the
    registry will work with any string subtype via adapter class attributes.
    """

    CHATGPT = "chatgpt"
    SESSION_JSON = "session_json"
    TEXT_PLAIN = "text_plain"
    MARKDOWN = "markdown"
    PDF = "pdf"
    PASTE = "paste"  # raw text in request body (browser paste)
    UNKNOWN = "unknown"


class ProcessingStage(str, Enum):
    """Stages for observability (mirrors common SaaS doc pipelines)."""

    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    READY = "ready"  # chunks produced; downstream: embed/index in your stack
    FAILED = "failed"


@dataclass
class IngestionJobSpec:
    """
    What to ingest. Maps to one adapter.

    - ``category`` + ``subtype`` select the adapter when ``source_type`` is omitted.
    - ``source`` is URL, filesystem path, or raw string (e.g. session JSON).
    """

    source: str
    category: SourceCategory = SourceCategory.LLM_CHAT
    subtype: SourceSubtype = SourceSubtype.CHATGPT
    # Optional explicit key (matches legacy register_parser names: "chatgpt", "session_json", ...)
    source_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parser_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageRecord:
    """One stage timing / status for observability."""

    name: ProcessingStage
    ok: bool
    detail: str = ""
    duration_ms: float = 0.0


@dataclass
class PipelineRunResult:
    """Output of the extraction pipeline (no scoring / no DB writes)."""

    stages: List[StageRecord] = field(default_factory=list)
    # ConversationChunk list is attached separately to avoid circular imports
    chunk_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": [
                {
                    "name": s.name.value,
                    "ok": s.ok,
                    "detail": s.detail,
                    "duration_ms": s.duration_ms,
                }
                for s in self.stages
            ],
            "chunk_count": self.chunk_count,
            "error": self.error,
        }
