# Datclaw Memory Engine - Dependencies

## Installation Options

Datclaw offers two installation modes to balance functionality with resource usage:

### Core Installation (Recommended)

**Size:** ~200MB  
**Install time:** 1-2 minutes  
**What you get:**
- Full memory engine functionality
- LLM-based entity/relation extraction (via OpenAI)
- Vector search (Qdrant)
- Knowledge graph (ArangoDB)
- Ego scoring
- Background workers
- All API endpoints

```bash
pip install -r requirements.txt
```

### Core + ML Installation (Optional)

**Size:** ~2-3GB  
**Install time:** 5-10 minutes  
**Additional features:**
- Local ML classifiers (DistilBERT, zero-shot)
- LightGBM ego score combiner
- spaCy entity extraction from queries
- Training utilities
- S3 cold storage

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm
```

## What's in Each File

### requirements.txt (Core)

| Category | Packages | Purpose |
|----------|----------|---------|
| **Web** | fastapi, uvicorn, pydantic | API server |
| **Databases** | redis, python-arango, qdrant-client | Data storage |
| **LLM** | openai, anthropic, tiktoken | AI providers |
| **Auth** | python-jose, bcrypt | Authentication (passwords hashed with bcrypt directly) |
| **Config** | pyyaml, python-dotenv | Configuration |
| **Monitoring** | prometheus-client, structlog | Logging/metrics |
| **Testing** | pytest, pytest-asyncio, httpx | Test suite |
| **Utilities** | numpy, nltk, rich, tenacity | Helpers |

**Optional (not pinned in `requirements.txt`):** `pypdf` — only needed for PDF ingestion (`source_type` / `pdf_file`). Plain text and Markdown files use the standard library.

### requirements-ml.txt (Optional)

| Category | Packages | Purpose |
|----------|----------|---------|
| **NLP** | spacy | Entity extraction from queries |
| **Deep Learning** | torch, transformers | Local classifiers |
| **ML Models** | lightgbm, scikit-learn, shap | Ego scoring combiner |
| **Training** | pandas, matplotlib, numba | Model training |
| **Storage** | boto3 | S3 cold archive |

## When to Use ML Installation

### Use Core Only If:
- You want fast setup
- You're okay with LLM-based extraction (OpenAI API calls)
- You don't need local ML classifiers
- You're developing/testing
- You want minimal resource usage

### Use Core + ML If:
- You want to train custom classifiers
- You need offline entity extraction (spaCy)
- You want LightGBM-based ego scoring
- You're doing ML research/tuning
- You have the disk space and time

## Configuration

The system automatically detects which packages are installed and adjusts behavior:

**In `config/base.yaml`:**

```yaml
chatbot:
  use_ml_scoring: false          # Set to true to use LightGBM combiner
  classifier_type: "hf_api"      # Options: "hf_api" | "regex" | "zeroshot" | "distilbert"
  use_distilbert: false          # Requires torch/transformers
```

**Fallback behavior:**
- If ML packages not installed, `use_ml_scoring` is ignored
- If spaCy not installed, entity extraction uses LLM fallback
- If classifiers not available, falls back to regex patterns

## Dependency Tree

### Core Dependencies

```
fastapi
├── pydantic
├── starlette
└── typing-extensions

openai
├── httpx
├── tiktoken
└── pydantic

redis
└── hiredis (optional, for speed)

python-arango
└── requests

qdrant-client
├── grpcio
└── httpx
```

### ML Dependencies

```
torch (~2GB)
├── numpy
├── sympy
└── networkx

transformers (~500MB)
├── torch
├── tokenizers
├── huggingface-hub
└── safetensors

spacy (~200MB)
├── thinc
├── cymem
├── preshed
└── en_core_web_sm model (~13MB)

lightgbm
└── scikit-learn
    └── scipy
```

## Troubleshooting

### Issue: "No module named 'spacy'"

**Solution:** You're using core installation. Either:
1. Install ML dependencies: `pip install -r requirements-ml.txt`
2. Or set `use_ml_scoring: false` in config (already default)

### Issue: "No module named 'torch'"

**Solution:** Same as above. The system will fall back to LLM-based classification.

### Issue: "Failed to load spaCy model"

**Solution:** Download the model:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Installation takes too long

**Solution:** Use core installation only:
```bash
pip install -r requirements.txt  # Skip requirements-ml.txt
```

### Issue: Out of disk space

**Solution:** Core installation uses ~200MB. ML installation needs ~3GB free space.

## Updating Dependencies

### Update Core

```bash
pip install --upgrade -r requirements.txt
```

### Update ML

```bash
pip install --upgrade -r requirements-ml.txt
```

### Update All

```bash
pip install --upgrade -r requirements.txt -r requirements-ml.txt
```

## Development

### Adding New Dependencies

**For core features:**
- Add to `requirements.txt`
- Keep it lightweight

**For ML/training features:**
- Add to `requirements-ml.txt`
- Make sure code has fallback when not installed

### Testing Without ML

```bash
# Install core only
pip install -r requirements.txt

# Run tests (ML tests will be skipped)
pytest tests/ -v
```

## Production Considerations

For production deployment:
- Use core installation for faster container builds
- Add ML dependencies only if needed
- Consider pre-built Docker images
- Pin all versions (already done in requirements files)

For hosted service, we handle all dependencies and optimization.

---

**Questions?** See [STARTUP_GUIDE.md](../STARTUP_GUIDE.md) or open an issue.
