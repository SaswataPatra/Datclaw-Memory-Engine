# DAPPY ML Training

Training scripts for Phase 1.5+ ML components.

## Components

### 1. DistilBERT Memory Classifier
**Purpose:** Replace regex patterns with context-aware memory type classification  
**File:** `train_distilbert.py`  
**Status:** ✅ Ready to train

### 2. LightGBM Ego Score Combiner
**Purpose:** Learn optimal weights for combining component scores  
**File:** `train_lightgbm.py` (coming next)  
**Status:** 🔜 In progress

---

## Training DistilBERT Memory Classifier

### Step 1: Install Dependencies

```bash
cd llm-orchestration
pip install -r requirements.txt
```

### Step 2: Generate Training Data

```bash
export OPENAI_API_KEY='sk-...'

python -m ml.training.train_distilbert \
  --generate-data \
  --dataset-path data/distilbert_dataset.jsonl \
  --variations-per-seed 5
```

**What this does:**
- Uses seed examples for 10 memory types (identity, family, preference, etc.)
- Generates 5 variations per seed using GPT-4o-mini
- Creates multi-label examples (e.g., "My sister loves tennis" → family + preference)
- Saves to `data/distilbert_dataset.jsonl`

**Expected output:**
```
Generated 500+ training examples
Label distribution:
  preference: 80 examples
  identity: 75 examples
  family: 70 examples
  ...
```

### Step 3: Train the Model

```bash
python -m ml.training.train_distilbert \
  --train \
  --dataset-path data/distilbert_dataset.jsonl \
  --output-dir models/distilbert \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 2e-5
```

**Training parameters:**
- **Epochs:** 5 (usually sufficient for fine-tuning)
- **Batch size:** 16 (adjust based on GPU memory)
- **Learning rate:** 2e-5 (standard for DistilBERT fine-tuning)
- **Device:** Auto-detects CUDA/CPU

**Expected training time:**
- CPU: ~30-45 minutes
- GPU (T4): ~5-10 minutes

**Expected metrics:**
```
Epoch 5/5
Training loss: 0.0523
Validation metrics:
  f1_micro: 0.8542
  f1_macro: 0.8123
  precision_micro: 0.8621
  recall_micro: 0.8467
  hamming_loss: 0.0312
✅ New best model saved! F1: 0.8123
```

### Step 4: Test the Model

```python
from ml.extractors.memory_classifier import DistilBERTMemoryClassifier

# Load trained model
model = DistilBERTMemoryClassifier()
model.load_model('models/distilbert/best_model.pt')

# Test predictions
text = "My mother loves gardening and I do too"
labels, scores = model.predict_single(text, threshold=0.5)

print(f"Predicted labels: {labels}")
print(f"All scores: {scores}")

# Expected output:
# Predicted labels: ['family', 'preference']
# All scores: {
#   'family': 0.92,
#   'preference': 0.87,
#   'identity': 0.12,
#   ...
# }
```

### Step 5: Integrate into Chatbot (Optional for Phase 2)

See `distilbert_4` TODO - integration will replace regex patterns in `chatbot_service.py`.

---

## Training LightGBM Combiner

The LightGBM combiner learns optimal weights for combining component scores:

```
ego_score = f(novelty, frequency, sentiment, explicit_importance, engagement)
```

### Step 1: Ensure Services are Running

```bash
# Start Qdrant (required for novelty/frequency scoring)
docker-compose -f docker-compose.integration.yml up -d qdrant

# Verify Qdrant is running
curl http://localhost:6333/health
```

### Step 2: Generate Bootstrap Dataset

```bash
export OPENAI_API_KEY='sk-...'

python -m ml.training.train_lightgbm \
  --generate-data \
  --dataset-path data/lightgbm_dataset.jsonl \
  --variations-per-seed 10
```

**What this does:**
- Uses 30+ seed examples with hand-labeled ego scores (0.05 to 0.95)
- Generates 10 variations per seed using GPT-4o-mini
- Covers all 4 tiers (Tier 1: Core, Tier 2: Long-term, Tier 3: Short-term, Tier 4: Hot buffer)
- Saves to `data/lightgbm_dataset.jsonl`

**Expected output:**
```
Generated 330+ training examples
Ego score distribution:
  Tier 1 (>= 0.75): 80 examples
  Tier 2 (0.50-0.75): 96 examples
  Tier 3 (0.20-0.50): 88 examples
  Tier 4 (< 0.20): 66 examples
```

### Step 3: Train the Model

```bash
python -m ml.training.train_lightgbm \
  --train \
  --dataset-path data/lightgbm_dataset.jsonl \
  --output-dir models/lightgbm \
  --qdrant-url http://localhost:6333
```

**What this does:**
1. Loads bootstrap dataset
2. Computes component scores for each example using actual scorers
3. Trains LightGBM regressor to predict target ego scores
4. Evaluates on test set (20% holdout)
5. Compares with fallback weights
6. Generates SHAP plots for feature importance
7. Saves trained model to `models/lightgbm/combiner.pkl`

**Expected training time:**
- With 330 examples: ~5-10 minutes (includes embedding generation)

**Expected metrics:**
```
Test Set Performance:
  MAE:  0.0423
  RMSE: 0.0567
  R²:   0.9234

Feature Importance:
  explicit_importance_score: 0.4521
  novelty_score: 0.2134
  engagement_score: 0.1823
  sentiment_intensity: 0.0912
  frequency_score: 0.0610

Fallback (hardcoded weights) performance:
  MAE: 0.0612
  R²:  0.8745

Improvement:
  MAE: 30.88% better
  R²:  5.59% better
```

### Step 4: Hyperparameter Tuning (Optional)

```bash
python -m ml.training.train_lightgbm \
  --train \
  --tune \
  --dataset-path data/lightgbm_dataset.jsonl \
  --output-dir models/lightgbm
```

**What this does:**
- Runs 5-fold cross-validation grid search
- Tests combinations of:
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `num_leaves`: [15, 31, 63]
  - `max_depth`: [-1, 5, 10]
- Finds optimal hyperparameters
- Trains final model with best params

**Expected tuning time:**
- ~15-30 minutes (45 combinations × 5 folds)

### Step 5: Load and Use Trained Model

```python
import pickle
from ml.combiners.lightgbm_combiner import LightGBMCombiner

# Load trained model
with open('models/lightgbm/combiner.pkl', 'rb') as f:
    combiner = pickle.load(f)

# Predict ego score
scores = {
    'novelty_score': 0.85,
    'frequency_score': 0.12,
    'sentiment_intensity': 0.67,
    'explicit_importance_score': 0.90,
    'engagement_score': 0.72,
    'recency_decay': 1.0,
    'reference_count': 0,
    'llm_confidence': 0.8,
    'source_weight': 1.0
}

ego_score = combiner.predict(scores)
print(f"Predicted ego score: {ego_score:.3f}")

# Get feature importance
importance = combiner.get_feature_importance()
print(f"Feature importance: {importance}")
```

### Step 6: Integrate into Production

The trained model will automatically be used in `chatbot_service.py` if:
1. Model file exists at `models/lightgbm/combiner.pkl`
2. `use_ml_scoring=True` in `ChatbotService` initialization
3. Model is loaded during service startup

**See:** `lightgbm_3` TODO for integration testing

---

## Directory Structure

```
ml/training/
├── README.md                          # This file
├── train_distilbert.py               # ✅ DistilBERT training script
├── distilbert_data_generator.py      # ✅ Data generation for DistilBERT
├── train_lightgbm.py                 # 🔜 LightGBM training script
└── lightgbm_data_generator.py        # 🔜 Data generation for LightGBM

data/
├── distilbert_dataset.jsonl          # Generated training data
└── lightgbm_dataset.jsonl            # 🔜 Bootstrap dataset

models/
├── distilbert/
│   └── best_model.pt                 # Trained DistilBERT model
└── lightgbm/
    └── combiner.pkl                  # 🔜 Trained LightGBM model
```

---

## Troubleshooting

### Out of Memory (GPU)
```bash
# Reduce batch size
python -m ml.training.train_distilbert --train --batch-size 8
```

### Slow Training (CPU)
```bash
# Use fewer variations per seed
python -m ml.training.train_distilbert --generate-data --variations-per-seed 3
```

### OpenAI API Rate Limits
```bash
# Generate data in smaller batches (edit distilbert_data_generator.py)
# Or use cached dataset if already generated
```

---

## Next Steps

1. ✅ Train DistilBERT classifier
2. 🔜 Create LightGBM bootstrap dataset generator
3. 🔜 Train LightGBM combiner
4. 🔜 Integrate both models into production
5. 🔜 Collect real user feedback for Phase 2 active learning

---

**Questions?** See `HYPERPARAMETERS_ROADMAP.md` for the full ML roadmap.

