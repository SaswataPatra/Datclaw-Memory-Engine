# Datclaw Memory Engine - Development Setup Guide

This guide will help you get the Datclaw Memory Engine up and running for local development and testing.

**Looking for production hosting?** Check out our [hosted service](https://datclaw.io) for a fully managed solution Coming Soon.

## Prerequisites

- Docker & Docker Compose installed
- Python 3.9+ installed
- Node.js 18+ installed (for frontend)
- OpenAI API key

## Step 1: Configure Environment Variables

1. The `.env` file has been created for you. Open it and update the following:

```bash
# Edit .env file
nano .env  # or use your preferred editor
```

2. Update these critical values:

```bash
# Set your OpenAI API key
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Set a strong ArangoDB password (use a password generator!)
ARANGODB_PASSWORD=your_strong_password_here_min_8_chars

# Optional: Set HuggingFace API key if using ML classifiers
HUGGINGFACE_API_KEY=hf_your_key_here
```

**Security Note:** The `.env` file is already in `.gitignore` and will NOT be committed to git. Use development API keys, not production keys.

## Step 2: Start Infrastructure Services

Start Redis, ArangoDB, and Qdrant in Docker:

```bash
# Make sure you're in the project root
cd /Users/saswatapatra/work/Datclaw-Memory-Engine

# Start all database services
docker-compose up -d

# Wait for services to be healthy (30-60 seconds)
# Check status
docker-compose ps

# View logs if needed
docker-compose logs -f
```

Expected output:
```
✓ Container datclaw-redis      Started
✓ Container datclaw-arangodb   Started
✓ Container datclaw-qdrant     Started
```

### Verify Services are Running

```bash
# Check all containers are healthy
docker ps

# Test Redis
docker exec datclaw-redis redis-cli ping
# Should return: PONG

# Test ArangoDB (replace <password> with your ARANGODB_PASSWORD)
curl -u root:<password> http://localhost:8529/_api/version
# Should return JSON with version info

# Test Qdrant
curl http://localhost:6333/
# Should return JSON with Qdrant info
```

## Step 3: Start Backend Service

```bash
# Navigate to backend directory
cd llm-orchestration

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies (includes spacy model download)
./setup_dependencies.sh

# Start the backend service (creates DB/collections on first run if needed)
./start_service.sh
```

The backend will start at `http://localhost:8000`

### Verify Backend is Running

```bash
# In a new terminal, test the health endpoint
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","version":"1.0.0"}
```

## Step 4: Start Frontend (Optional)

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start at `http://localhost:3000` (or the port shown in terminal)

## Step 5: Test the System

### Option A: Use the CLI Chat

```bash
cd llm-orchestration
./chat.sh
```

### Option B: Use the Web Interface

Open `http://localhost:3000` in your browser and start chatting!

### Option C: Use the API Directly

```bash
# Send a test message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "message": "Hello! My name is John and I love pizza."
  }'
```

## Troubleshooting

### Issue: Docker containers won't start

```bash
# Check if ports are already in use
lsof -i :6379  # Redis
lsof -i :8529  # ArangoDB
lsof -i :6333  # Qdrant

# Stop any conflicting services or change ports in docker-compose.yml
```

### Issue: ArangoDB authentication fails

Make sure the `ARANGODB_PASSWORD` in your `.env` file matches what you're using in the backend.

```bash
# Check the password is set correctly
grep ARANGODB_PASSWORD .env

# Restart ArangoDB with new password
docker-compose down
docker volume rm datclaw-memory-engine_arangodb_data  # WARNING: Deletes data!
docker-compose up -d arangodb
```

### Issue: Backend can't connect to databases

```bash
# Check if services are running
docker-compose ps

# Check backend logs
cd llm-orchestration
tail -f logs/app.log

# Verify environment variables are loaded
cd llm-orchestration
python -c "from dotenv import load_dotenv; import os; load_dotenv('../.env'); print(os.getenv('ARANGODB_PASSWORD'))"
```

### Issue: Python dependencies fail to install

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing again
pip install -r requirements.txt

# If specific packages fail, install them individually
pip install <package-name>
```

## Production Deployment

**For production use, we recommend our( Coming Soon ) [hosted service](https://datclaw.io)** which includes:
- Managed infrastructure
- Automatic scaling
- Enterprise support
- 99.9% uptime SLA
- Security and compliance

Self-hosting in production requires significant infrastructure management, security hardening, monitoring, and ongoing maintenance.

## Stopping Services

### Stop Backend (Development)

```bash
# Press Ctrl+C in the terminal running start_service.sh
# Or find and kill the process
ps aux | grep uvicorn
kill <PID>
```

### Stop Frontend

```bash
# Press Ctrl+C in the terminal running npm run dev
```

### Stop Docker Services

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (WARNING: Deletes all data!)
docker-compose down -v
```

## Next Steps

- Read the [Architecture Documentation](docs/architecture.md)
- Explore the [API Endpoints](README.md#api-endpoints)
- Run the [Test Suite](README.md#testing)
- Check out [KG Maintenance](docs/kg-maintenance.md) to understand how the knowledge graph works
- Consider [Contributing](CONTRIBUTING.md) to the project
- For production use, check out our [hosted service](https://datclaw.io)

## Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Review the [README.md](README.md)
3. Open an issue on GitHub

---

**Happy Memory Engineering! 🧠✨**
