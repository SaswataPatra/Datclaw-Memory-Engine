# Quick Start - Datclaw Memory Engine

## 🚀 Get Running in 5 Minutes

### One-Command Setup

```bash
# Clone the repo
git clone https://github.com/SaswataPatra/Datclaw-Memory-Engine.git
cd Datclaw-Memory-Engine

# Run setup script (handles everything)
./setup.sh        # macOS/Linux
# OR
setup.bat         # Windows
```

The script will:
- Check Docker, Python, Node.js
- Create `.env` (you'll need to add your OpenAI API key)
- Start Docker services
- Install dependencies (you choose: core only or core + ML)
- Set up frontend (optional)

### After Setup

**1. Edit `.env` file:**
```bash
nano .env  # or use your editor

# Set these:
OPENAI_API_KEY=sk-your-actual-key
ARANGODB_PASSWORD=your-secure-password
```

**2. Start the backend:**
```bash
cd llm-orchestration
./start_service.sh
```

**3. Test it:**
```bash
curl http://localhost:8000/health
```

**4. Chat:**
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
