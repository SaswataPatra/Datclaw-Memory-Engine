"""Unit tests for the top-level ``ingestion`` package (adapters + pipeline)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestion.contracts import IngestionJobSpec, SourceCategory, SourceSubtype
from ingestion.pipeline import IngestionPipeline
from ingestion.registry import build_default_registry, resolve_source_type


def test_resolve_source_type_explicit_key():
    spec = IngestionJobSpec(
        source="/tmp/x",
        source_type="session_json",
        category=SourceCategory.LLM_CHAT,
        subtype=SourceSubtype.CHATGPT,
    )
    assert resolve_source_type(spec) == "session_json"


@pytest.mark.asyncio
async def test_plain_text_file_extract():
    reg = build_default_registry()
    pipe = IngestionPipeline(reg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Hello\n\nWorld")
        path = f.name
    try:
        spec = IngestionJobSpec(
            source=path,
            category=SourceCategory.FILE,
            subtype=SourceSubtype.MARKDOWN,
        )
        chunks, pr = await pipe.run_extract(spec)
        assert pr.error is None
        assert pr.chunk_count == 1
        assert len(chunks) == 1
        assert "Hello" in chunks[0].content
        assert chunks[0].source_type == "text_file"
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_text_paste_extract():
    reg = build_default_registry()
    pipe = IngestionPipeline(reg)
    spec = IngestionJobSpec(
        source="  hello paste world  ",
        category=SourceCategory.FILE,
        subtype=SourceSubtype.PASTE,
    )
    chunks, pr = await pipe.run_extract(spec)
    assert pr.error is None
    assert len(chunks) == 1
    assert "hello paste" in chunks[0].content
    assert chunks[0].source_type == "text_paste"


@pytest.mark.asyncio
async def test_run_extract_only_helper():
    reg = build_default_registry()
    pipe = IngestionPipeline(reg)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("plain content")
        path = f.name
    try:
        chunks, pr = await pipe.run_extract_only("text_file", path)
        assert pr.error is None
        assert len(chunks) == 1
    finally:
        Path(path).unlink(missing_ok=True)
