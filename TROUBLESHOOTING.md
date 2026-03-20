# Troubleshooting Guide

Common issues and solutions when setting up or running Datclaw Memory Engine.

## Setup Issues

### "Docker not found"

**Problem:** Docker is not installed or not in PATH.

**Solution:**
- **macOS/Windows:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux:** Follow [official Docker installation guide](https://docs.docker.com/engine/install/)

After installation, verify: `docker --version`

---

### "Python 3.9+ not found"

**Problem:** Python is not installed or version is too old.

**Solution:**
- **macOS:** `brew install python@3.12`
- **Ubuntu/Debian:** `sudo apt update && sudo apt install python3.12 python3.12-venv`
- **Fedora/RHEL:** `sudo dnf install python3.12`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

Verify: `python3 --version` or `python --version`

---

### "Python venv module not available"

**Problem:** The `venv` module is not installed (common on Linux).

**Solution:**
- **Ubuntu/Debian:** `sudo apt install python3-venv`
- **Fedora/RHEL:** `sudo dnf install python3-virtualenv`

---

### ".env still has placeholder values"

**Problem:** You haven't configured your API keys and passwords.

**Solution:**
1. Open `.env` in your editor
2. Set `OPENAI_API_KEY=sk-your-actual-key`
3. Set `ARANGODB_PASSWORD=your-secure-password`
4. Save and re-run setup

---

## Docker Issues

### "Failed to start Docker services"

**Problem:** Docker daemon not running or configuration error.

**Solutions:**
1. Check Docker is running: `docker ps`
2. Check `.env` has `ARANGODB_PASSWORD` set
3. View logs: `docker-compose logs`
4. Restart services: `docker-compose down && docker-compose up -d`
5. Check port conflicts: `lsof -i :6379,8529,6333` (macOS/Linux)

---

### "Port already in use"

**Problem:** Another service is using Redis (6379), ArangoDB (8529), or Qdrant (6333).

**Solution:**
1. Find what's using the port: `lsof -i :6379` (macOS/Linux) or `netstat -ano | findstr :6379` (Windows)
2. Stop the conflicting service
3. Or modify `docker-compose.yml` to use different ports

---

### "Services not healthy after 30s"

**Problem:** Docker containers started but health checks failing.

**Solutions:**
1. Check logs: `docker-compose logs arangodb` (or redis, qdrant)
2. Check disk space: `df -h`
3. Restart services: `docker-compose restart`
4. Check `.env` password matches what ArangoDB expects

---

## Python/Backend Issues

### "ModuleNotFoundError: No module named 'X'"

**Problem:** Required Python package not installed.

**Solution:**
1. Activate virtual environment: `source .venv/bin/activate`
2. Reinstall dependencies: `pip install -r requirements.txt`
3. If it's an ML package (torch, spacy, etc.): `pip install -r requirements-ml.txt`

---

### "No module named 'spacy'" or "No module named 'torch'"

**Problem:** You're using core installation but code expects ML packages.

**Solution:**
- **Option 1:** Install ML dependencies: `pip install -r requirements-ml.txt`
- **Option 2:** Disable ML features in `config/base.yaml`:
  ```yaml
  chatbot:
    use_ml_scoring: false
    classifier_type: "hf_api"  # or "regex"
  ```

The system should auto-fallback, but explicit config is cleaner.

---

### "database not found" (ArangoDB)

**Problem:** ArangoDB database not created.

**Solution:**
The application creates the database automatically on startup. If this fails:
1. Check ArangoDB is running: `docker ps | grep arangodb`
2. Check logs: `docker-compose logs arangodb`
3. Verify `ARANGODB_PASSWORD` in `.env` matches `docker-compose.yml`
4. Manually create database:
   ```bash
   docker exec -it datclaw-arangodb arangosh --server.password YOUR_PASSWORD
   # In arangosh: db._createDatabase("dappy");
   ```

---

### "Connection refused" to Redis/ArangoDB/Qdrant

**Problem:** Backend can't connect to databases.

**Solutions:**
1. Check services are running: `docker-compose ps`
2. Check health: `docker ps` (should show "healthy")
3. Restart services: `docker-compose restart`
4. Check `.env` has correct connection details

---

### "Address already in use" (port 8000)

**Problem:** Another process is using port 8000.

**Solution:**
1. Find process: `lsof -i :8000` (macOS/Linux) or `netstat -ano | findstr :8000` (Windows)
2. Kill it: `kill -9 PID`
3. Or change port: `uvicorn api.main:app --port 8001`

---

## Frontend Issues

### "npm: command not found"

**Problem:** Node.js not installed.

**Solution:**
- **macOS:** `brew install node`
- **Ubuntu/Debian:** `sudo apt install nodejs npm`
- **Windows:** Download from [nodejs.org](https://nodejs.org/)

Verify: `node --version` (should be 18+)

---

### "Failed to fetch" or "Network Error"

**Problem:** Frontend can't connect to backend.

**Solutions:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Check frontend `.env` has correct `VITE_API_URL`
3. Check CORS settings in backend `api/main.py`

---

## Runtime Issues

### Chat responses are empty or error

**Problem:** LLM provider issue or configuration error.

**Solutions:**
1. Check API key is valid: `echo $OPENAI_API_KEY`
2. Check OpenAI API status: [status.openai.com](https://status.openai.com)
3. Check backend logs for errors
4. Try a different model in `config/base.yaml`

---

### "Out of memory" errors

**Problem:** ML models consuming too much RAM.

**Solutions:**
1. Use core installation (no ML): `pip uninstall torch transformers spacy`
2. Reduce batch sizes in `config/base.yaml`
3. Disable ML scoring: `use_ml_scoring: false`
4. Use a machine with more RAM (ML needs ~4GB)

---

### Slow response times

**Problem:** First request is slow (model loading).

**Solutions:**
- This is normal - first request loads models (~5-10s)
- Subsequent requests should be fast (<1s)
- For faster startup, use core installation (no ML)

---

## Database Issues

### Reset all databases

**Problem:** Need to start fresh.

**Solution:**
```bash
cd llm-orchestration
python scripts/reset_all_databases.py
```

This will:
- Clear Redis
- Drop and recreate ArangoDB collections
- Clear Qdrant collections

---

### ArangoDB authentication failed

**Problem:** Password mismatch.

**Solutions:**
1. Check `.env` has `ARANGODB_PASSWORD=your-password`
2. Check `docker-compose.yml` uses `${ARANGODB_PASSWORD}`
3. Restart ArangoDB: `docker-compose restart arangodb`
4. Or recreate: `docker-compose down && docker-compose up -d`

---

## Testing Issues

### Tests fail with "connection refused"

**Problem:** Tests expect databases to be running.

**Solution:**
```bash
# Start databases first
docker-compose up -d

# Wait for health
sleep 10

# Run tests
pytest tests/ -v
```

---

### Tests fail with import errors

**Problem:** Test dependencies not installed.

**Solution:**
```bash
pip install -r requirements.txt
# For ML tests:
pip install -r requirements-ml.txt
```

---

## Performance Issues

### High CPU usage

**Problem:** Background workers or ML models consuming CPU.

**Solutions:**
1. Check which workers are enabled in `config/base.yaml`
2. Disable unused workers
3. Reduce worker concurrency
4. Use core installation (no ML)

---

### High memory usage

**Problem:** ML models loaded in memory.

**Solutions:**
1. Use core installation (no ML models)
2. Reduce `max_workers` in config
3. Disable ML scoring: `use_ml_scoring: false`

---

## Still Stuck?

1. **Check logs:**
   - Backend: Check terminal where `start_service.sh` is running
   - Docker: `docker-compose logs -f`
   - Specific service: `docker-compose logs arangodb`

2. **Verify setup:**
   ```bash
   # Check Docker services
   docker-compose ps
   
   # Check backend health
   curl http://localhost:8000/health
   
   # Check Python packages
   pip list | grep -i "fastapi\|openai\|redis"
   ```

3. **Clean restart:**
   ```bash
   # Stop everything
   docker-compose down
   cd llm-orchestration
   deactivate  # Exit venv
   
   # Remove virtual environment
   rm -rf .venv
   
   # Run setup again
   cd ..
   ./setup.sh
   ```

4. **Get help:**
   - Open an issue on GitHub with:
     - Your OS and Python version
     - Error messages and logs
     - Steps to reproduce
   - Check existing issues for similar problems

---

## Ingestion

### ChatGPT share import returns 403 Forbidden

**Problem:** The backend calls `https://chatgpt.com/backend-api/share/{id}`. OpenAI often blocks non-browser / datacenter clients with **403** (Cloudflare and bot protection). This is **not** your API key — share links are fetched over HTTPS like a browser, not via the OpenAI API.

**What we do in code:** The parser first loads the public share page (to pick up cookies), then calls the JSON API with browser-like headers.

**If it still fails:**

1. **Optional cookie (local dev only)** — In `llm-orchestration/.env` set:
   ```bash
   CHATGPT_SHARE_COOKIE='__cf_bm=...; oai-did=...'
   ```
   Copy the **Cookie** header value from your browser’s devtools on `chatgpt.com` while logged in (same session as viewing the share). **Do not commit this.** Rotate cookies if leaked. This may still fail from some networks.

2. **Avoid share URL import** — Use **Session JSON** or **paste text** in the import UI, or export the conversation from ChatGPT and ingest the file.

3. **Network** — Try from a residential IP / different network if 403 persists.

---

## Common Gotchas

1. **Virtual environment not activated:** Always run `source .venv/bin/activate` before Python commands
2. **Wrong directory:** Scripts expect to be run from their directory (e.g., `./start_service.sh` from `llm-orchestration/`)
3. **Environment variables not loaded:** Use `./start_service.sh` which loads `.env` automatically
4. **Docker not running:** Start Docker Desktop (macOS/Windows) or Docker daemon (Linux)
5. **Firewall blocking ports:** Allow ports 6379, 8529, 6333, 8000, 3000

---

**Pro tip:** Run `./setup.sh` again if you're unsure about your setup state. It's idempotent and will detect what's already configured.
