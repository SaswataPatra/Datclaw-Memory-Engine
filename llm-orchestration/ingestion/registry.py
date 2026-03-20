"""
Adapter registry — maps job specs / source_type keys to ``SourceAdapter`` instances.

Auto-discovery: all ``SourceAdapter`` subclasses with ``enabled=True`` are registered.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Optional

from .adapters.base import SourceAdapter
from .contracts import IngestionJobSpec, SourceCategory, SourceSubtype

logger = logging.getLogger(__name__)


def resolve_source_type(spec: IngestionJobSpec, registry: Optional[AdapterRegistry] = None) -> str:
    """
    Determine the registry key for an ingestion job.

    If ``spec.source_type`` is set, it wins. Otherwise:
    1. Try to find an adapter in the registry that matches category + subtype.
    2. Fall back to hardcoded mapping (for legacy parsers without category/subtype).
    """
    if spec.source_type:
        return spec.source_type

    # Try registry lookup by category + subtype
    if registry:
        for key, adapter in registry._adapters.items():
            cat = getattr(adapter, "category", None) or getattr(type(adapter), "category", None)
            sub = getattr(adapter, "subtype", None) or getattr(type(adapter), "subtype", None)
            if cat == spec.category.value and sub == spec.subtype.value:
                return key

    # Fallback: hardcoded mapping for legacy parsers
    if spec.category == SourceCategory.LLM_CHAT:
        mapping = {
            SourceSubtype.CHATGPT: "chatgpt",
            SourceSubtype.SESSION_JSON: "session_json",
        }
        key = mapping.get(spec.subtype)
        if key:
            return key

    if spec.category == SourceCategory.FILE:
        mapping = {
            SourceSubtype.TEXT_PLAIN: "text_file",
            SourceSubtype.MARKDOWN: "text_file",
            SourceSubtype.PDF: "pdf_file",
            SourceSubtype.PASTE: "text_paste",
        }
        key = mapping.get(spec.subtype)
        if key:
            return key

    # Last resort
    return "chatgpt"


class AdapterRegistry:
    """Lookup adapters by source_type key (``chatgpt``, ``text_file``, ...)."""

    def __init__(self) -> None:
        self._adapters: Dict[str, SourceAdapter] = {}

    def register(self, source_type: str, adapter: SourceAdapter) -> None:
        self._adapters[source_type] = adapter
        logger.debug("Registered ingestion adapter: %s -> %s", source_type, type(adapter).__name__)

    def get(self, source_type: str) -> Optional[SourceAdapter]:
        return self._adapters.get(source_type)

    def require(self, source_type: str) -> SourceAdapter:
        a = self.get(source_type)
        if not a:
            raise KeyError(f"No ingestion adapter registered for source_type={source_type!r}")
        return a


def _discover_adapters() -> list[type[SourceAdapter]]:
    """
    Auto-discover all ``SourceAdapter`` subclasses in ``ingestion.adapters``.

    Returns concrete classes (not abstract, not disabled).
    """
    import ingestion.adapters as adapters_pkg

    discovered = []
    pkg_path = Path(adapters_pkg.__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        if module_name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"ingestion.adapters.{module_name}")
        except Exception as e:
            logger.warning("Failed to import ingestion.adapters.%s: %s", module_name, e)
            continue

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, SourceAdapter)
                and obj is not SourceAdapter
                and not inspect.isabstract(obj)
                and getattr(obj, "enabled", True)
            ):
                discovered.append(obj)

    return discovered


def build_default_registry() -> AdapterRegistry:
    """
    Built-in registry: auto-discovers all enabled adapters + legacy parsers.

    Legacy parsers (ChatGPT, session_json) are registered explicitly for backward compat.
    """
    from .adapters import LegacyParserAdapter
    from services.ingestion.chatgpt_parser import ChatGPTShareParser
    from services.ingestion.session_json_parser import SessionJsonParser

    reg = AdapterRegistry()

    # Legacy parsers (explicit)
    reg.register("chatgpt", LegacyParserAdapter(ChatGPTShareParser()))
    reg.register("session_json", LegacyParserAdapter(SessionJsonParser()))

    # Auto-discover new adapters
    for adapter_cls in _discover_adapters():
        try:
            instance = adapter_cls()
            key = instance.source_type
            reg.register(key, instance)
            logger.info(
                "Auto-registered adapter: %s (%s, category=%s, subtype=%s)",
                key,
                adapter_cls.__name__,
                getattr(adapter_cls, "category", None),
                getattr(adapter_cls, "subtype", None),
            )
        except Exception as e:
            logger.warning("Failed to instantiate %s: %s", adapter_cls.__name__, e)

    return reg
