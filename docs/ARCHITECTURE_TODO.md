# Datclaw Memory Engine – Architecture Improvement Backlog

This document captures the architectural improvements identified in the codebase review. Use it as a prioritized backlog for refactoring.

---

## High Priority (Maintainability)

### 4. Split `main.py` (God File)
- **Current:** ~1,474 lines, 15+ globals, many responsibilities.
- **Target structure:**
  - `api/routers/auth.py` – Auth endpoints
  - `api/routers/chat.py` – Chat endpoints
  - `api/routers/admin.py` – KG reconsolidation, delete memories
  - `api/routers/benchmark.py` – Benchmark endpoints
  - `api/routers/training.py` – Training endpoints
  - `api/schemas.py` – Pydantic models (10+ models)
  - `api/deps.py` – Dependency injection
  - `api/startup.py` or `core/bootstrap.py` – Initialization logic (~190 lines)

### 5. Extract DB Logic from Route Handlers
- **Issue:** `delete_all_user_memories`, `benchmark_clear` contain inline ArangoDB/Qdrant/Redis logic.
- **Fix:** Move to dedicated services (e.g. `MemoryDeletionService`, `BenchmarkService`).

### 6. Introduce Dependency Injection
- **Issue:** 15+ globals (`redis_client`, `event_bus`, `ego_scorer`, etc.) make testing and reuse harder.
- **Fix:** Use FastAPI `Depends()` with a DI container or factory functions.

---

## Medium Priority (Code Quality)

### 7. Split `chatbot_service.py` (~1,433 lines)
- **Issue:** Very large class, many constructor params (15+), mixed sync/async.
- **Fix:**
  - Extract config/builder for constructor params
  - Split into smaller services (e.g. `ChatOrchestrator`, `ContextBuilder`, `ResponseGenerator`)
  - Consider async-first design

### 8. Extract Logging Setup
- **Issue:** Logging setup (lines 36–74) lives in `main.py`.
- **Fix:** Move to `core/logging.py` or `config/logging.py`.

---

## Lower Priority (Nice to Have)

### 9. Split `base.yaml` (754 lines)
- **Issue:** Long config file; some sections could be split.
- **Fix:** Consider `config/base.yaml`, `config/llm.yaml`, `config/databases.yaml` with includes if supported.

### 10. Add Frontend Tests
- **Issue:** No unit or integration tests for React components.
- **Fix:** Add Vitest + React Testing Library; start with `ChatPage`, `AuthPage`, `ProtectedRoute`.

### 11. Reorganize `core/graph/`
- **Issue:** Large directory mixing many concerns.
- **Fix:** Sub-packages by responsibility (e.g. `core/graph/extraction/`, `core/graph/resolution/`, `core/graph/validation/`).

### 12. Clean Up `utils/`
- **Issue:** Mixes scripts and test helpers (e.g. `test_hf_api_key.py`).
- **Fix:** Move test helpers to `tests/` or `tests/utils/`.

### 13. Add Streaming UI to Frontend
- **Issue:** API supports streaming but frontend doesn't use it.
- **Fix:** Implement streaming response handling in `ChatPage.tsx`.

### 14. Replace `confirm()` / `alert()` in Frontend
- **Issue:** Native dialogs for destructive actions.
- **Fix:** Use modal components (e.g. Radix UI, Headless UI).

---

## Already Addressed

- ✅ **Duplicate shadow route** – Split into `/shadow/pending/user/{user_id}` and `/shadow/pending/session/{session_id}`
- ✅ **Print statements** – Replaced `print()` with `logger.debug()` in `chatbot_service.py`
- ✅ **CORS tightened** – Env-configurable via `CORS_ORIGINS`, defaults to permissive only for dev
- ✅ **Lightweight installation** – Split `requirements.txt` / `requirements-ml.txt`, lazy-load ML
- ✅ **Spacy optional** – Made optional in `context_manager.py` with fallback
- ✅ **ArangoDB auto-create** – Database created on startup if missing
- ✅ **Setup script** – Unified `setup.sh` / `setup.bat` for user-friendly install

---

## Summary Table

| # | Item                          | Priority   | Effort | Status |
|---|-------------------------------|------------|--------|--------|
| 1 | Fix duplicate shadow route   | Critical   | Small  | ✅ Done |
| 2 | Replace print with logging   | Critical   | Small  | ✅ Done |
| 3 | Tighten CORS                 | Critical   | Small  | ✅ Done |
| 4 | Split main.py                | High       | Large  | |
| 5 | Extract DB logic to services | High       | Medium | |
| 6 | Dependency injection         | High       | Large  | |
| 7 | Split chatbot_service.py     | Medium     | Large  | |
| 8 | Extract logging setup        | Medium     | Small  | |
| 9 | Split base.yaml              | Low        | Medium | |
| 10| Frontend tests               | Low        | Medium | |
| 11| Reorganize core/graph        | Low        | Large  | |
| 12| Clean up utils               | Low        | Small  | |
| 13| Streaming UI                 | Low        | Medium | |
| 14| Replace confirm/alert        | Low        | Small  | |

---

*Last updated: March 2026*
