# Ingestion architecture (Datclaw)

This document describes the **separate ingestion module** (`llm-orchestration/ingestion/`) and how it connects to the existing memory pipeline (`services.ingestion.IngestionService`).

## Goals

- **One extraction pipeline** with observable stages (similar in spirit to Supermemory-style `queued → extracting → chunking → embedding → …`).
- **Many sources**, same internal shape: `ConversationChunk` (defined in `services/ingestion/models.py`).
- **Pluggable adapters** without bloating `chatbot_service` or `api/main.py`.

## Layout

| Path | Role |
|------|------|
| `ingestion/contracts.py` | `IngestionJobSpec`, `SourceCategory`, `SourceSubtype`, `ProcessingStage`, `PipelineRunResult` |
| `ingestion/registry.py` | `AdapterRegistry`, `resolve_source_type()`, `build_default_registry()` |
| `ingestion/pipeline.py` | `IngestionPipeline.run_extract()` — validate → extract → ready |
| `ingestion/adapters/` | Per-source adapters (`PlainTextFileAdapter`, `PdfFileAdapter`, `text_paste`, etc.) |
| `services/ingestion/parser_adapter_bridge.py` | `SourceAdapterParser` — wraps a `SourceAdapter` as legacy `BaseParser` for `IngestionService.register_parser` |

## End-to-end flow

1. **Extract** — An adapter reads a URL, path, or raw string and returns `List[ConversationChunk]`.
2. **Memory pipeline** — `IngestionService.ingest_parsed_chunks()` (or `ingest()` which parses first) runs ego scoring, `memory.upsert` events, optional ML rescoring, consolidation, KG, etc.

Embedding and vector indexing remain in your **existing consumers** (Qdrant, Arango); this module does not replace them.

## API

- **`GET /user/ingest/adapters`** — Lists registered adapters (metadata for the import UI): `source_type`, `category`, `subtype`, `description`, `input_mode` (`file` | `url` | `text`).
- **`POST /user/ingest`** — `source_type` is a registry key: `chatgpt`, `session_json`, `text_file`, `pdf_file`, `text_paste`, …
- **`POST /user/ingest/spec`** — Structured `category` + `subtype` (see `IngestSpecRequest` in `api/main.py`); runs `IngestionPipeline` then `ingest_parsed_chunks`. Response may include `pipeline_stages`.
- **`POST /user/ingest/upload`** — Multipart `file` upload (browser); server writes a temp file and runs `text_file` or `pdf_file` ingest. Use this when the client cannot send a server filesystem path.
- **`text_paste` adapter** — `source` is raw body text (no path); use with `/user/ingest/spec` (`category=file`, `subtype=paste`) or `source_type=text_paste` on `/user/ingest`.

## Adding a new source

**Zero-touch registration** via auto-discovery:

1. **Write the adapter** in `ingestion/adapters/your_new_source.py`:
   ```python
   from ingestion.adapters.base import SourceAdapter
   from services.ingestion.models import ConversationChunk

   class YourNewSourceAdapter(SourceAdapter):
       category = "file"  # or "llm_chat", "web", "api"
       subtype = "your_subtype"  # e.g. "docx", "slack_export"
       enabled = True  # default; set False to disable

       @property
       def source_type(self) -> str:
           return "your_source_key"  # registry key

       async def extract(self, source: str, **kwargs) -> List[ConversationChunk]:
           # your logic here
           return [ConversationChunk(...)]
   ```

2. **That's it.** `build_default_registry()` auto-discovers all `SourceAdapter` subclasses in `ingestion/adapters/` and registers them. No changes to `main.py` or `registry.py` needed.

3. **(Optional)** Add your `subtype` to `SourceSubtype` enum in `contracts.py` for type safety in the API, and update the fallback mapping in `resolve_source_type()` if you want `/user/ingest/spec` to work without an explicit `source_type`.

## LLM share links (Gemini, Claude, …)

Public share URLs often rely on **HTML or session-specific APIs** and may violate provider ToS if scraped. The repo includes **stub** adapters that fail with a clear message until you add a supported flow (export file, browser extension, official API).

## PDF

`PdfFileAdapter` uses **`pypdf`** (`pip install pypdf`). Optional; core install stays lightweight.

## Relationship to `services/ingestion/`

- **`services/ingestion/`** — Legacy `BaseParser` implementations and **`IngestionService`** (orchestration, scoring, events).
- **`ingestion/`** — New **module boundary** for extraction + job/spec typing. Adapters can wrap legacy parsers (`LegacyParserAdapter`) or stand alone.

## See also

- `docs/ARCHITECTURE_TODO.md` — broader refactors.
- Supermemory-style document states (reference only): external product schemas, not copied code.
