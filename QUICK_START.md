# Quick Start - Datclaw Memory Engine

## 🚀 Get Running in 5 Minutes

### 1. Configure Secrets (2 minutes)

```bash
# Edit .env file
nano .env

# Update these two lines:
OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY
ARANGODB_PASSWORD=your-strong-password-here
```

### 2. Start Databases (1 minute)

```bash
docker-compose up -d
```

Wait 30 seconds for services to be healthy.

### 3. Start Backend (2 minutes)

```bash
cd llm-orchestration
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# Install dependencies (includes spacy model download)
./setup_dependencies.sh

# Start the service (DB/collections created on first run if needed)
./start_service.sh
```

### 4. Test It! (30 seconds)

```bash
# In a new terminal
curl http://localhost:8000/health
```

Should return: `{"status":"healthy"}`

### 5. Chat! (Optional)

```bash
cd llm-orchestration
./chat.sh
```

---

## 📋 Quick Commands

### Check Status
```bash
docker-compose ps                    # Check containers
curl http://localhost:8000/health    # Check backend
```

### View Logs
```bash
docker-compose logs -f               # Database logs
cd llm-orchestration && tail -f logs/app.log  # Backend logs
```

### Stop Everything
```bash
docker-compose down                  # Stop databases
# Ctrl+C in backend terminal         # Stop backend
```

### Reset Everything (⚠️ Deletes Data!)
```bash
docker-compose down -v
cd llm-orchestration
python scripts/reset_all_databases.py
```

---

## 🔧 Common Issues

**Can't connect to ArangoDB?**
- Check password in `.env` matches
- Verify container is running: `docker ps`

**Port already in use?**
- Check: `lsof -i :6379` (Redis), `:8529` (Arango), `:6333` (Qdrant)
- Change ports in `docker-compose.yml`

**Backend won't start?**
- Make sure virtual environment is activated
- Check `.env` file exists in project root
- Verify databases are healthy: `docker-compose ps`

---

## 📚 More Info

- Full guide: [STARTUP_GUIDE.md](STARTUP_GUIDE.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- API docs: [README.md](README.md#api-endpoints)
