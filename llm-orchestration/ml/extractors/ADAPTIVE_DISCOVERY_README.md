# Adaptive Label Discovery

**Status:** ✅ Implemented, 🚨 **DISABLED by default**  
**Phase:** 2 (Future feature)

---

## 🎯 What It Does

Automatically discovers new semantic categories when the base classifier predicts `unknown` with high confidence. Uses GPT-4o-mini to generate meaningful labels from conversation context.

### Example Flow

```
User: "I'm planning to run a marathon next year"
Base Classifier: unknown (0.85 confidence)

↓ Trigger Discovery

LLM Analysis: "This is about fitness/training goals"
Discovered Label: "fitness_training"

↓ Add to Candidate Pool

After 3 similar messages:
✅ Promote "fitness_training" to official label set

Next time:
User: "I need to improve my running stamina"
Classifier: fitness_training (0.87 confidence)
```

---

## 🔧 Configuration

### Enable/Disable

```yaml
# config/base.yaml
adaptive_discovery:
  enabled: false  # 🚨 Set to true to enable
  unknown_threshold: 0.7  # Trigger when unknown > 70%
  min_examples_to_add: 3  # Need 3 examples before promoting
  label_confidence_threshold: 0.8  # LLM must be 80%+ confident
  similarity_threshold: 0.8  # Prevent duplicates (80% similar)
```

### Hyperparameters

| Parameter | Default | Phase | Description |
|-----------|---------|-------|-------------|
| `enabled` | `false` | 2 | Master switch for feature |
| `unknown_threshold` | 0.7 | 2 | Confidence threshold to trigger discovery |
| `min_examples_to_add` | 3 | 2 | Examples needed before promoting label |
| `label_confidence_threshold` | 0.8 | 2 | LLM confidence required |
| `similarity_threshold` | 0.8 | 2 | Prevent duplicate labels |

---

## 💻 Usage Example

### Basic Integration

```python
from ml.extractors import DistilBERTMemoryClassifier, AdaptiveLabelDiscovery
from openai import AsyncOpenAI

# Initialize
llm_client = AsyncOpenAI(api_key="sk-...")
classifier = DistilBERTMemoryClassifier()

# Create adaptive discovery (disabled by default)
discovery = AdaptiveLabelDiscovery(
    llm_client=llm_client,
    config=config,
    base_labels=classifier.label_names,
    enabled=False  # 🚨 Disabled for safe debugging
)

# Classify with discovery
text = "I'm training for a triathlon"
base_prediction = classifier.predict_single(text)

# Enhance with discovery (only if enabled)
labels, scores, discovered = await discovery.classify_with_discovery(
    text=text,
    user_id="user123",
    base_prediction=base_prediction,
    conversation_context=recent_messages
)

if discovered:
    print(f"✨ Discovered new category: {discovered}")
```

### Get Current Labels

```python
# Get all labels (base + discovered)
all_labels = discovery.get_current_labels()
print(f"Total labels: {len(all_labels)}")
print(f"Base: {len(discovery.base_labels)}")
print(f"Discovered: {len(discovery.discovered_labels)}")
```

### Get Statistics

```python
stats = discovery.get_statistics()
print(stats)
# {
#   'enabled': False,
#   'base_labels_count': 10,
#   'discovered_labels_count': 0,
#   'candidate_labels_count': 0,
#   'total_discoveries': 0,
#   'config': {...}
# }
```

---

## 🔍 How It Works

### 1. Detection Phase
```
User message → Base Classifier → Prediction

If 'unknown' confidence > threshold:
  → Trigger Discovery
```

### 2. Discovery Phase
```
Extract context:
  - Current message
  - Last 5 conversation exchanges
  - Existing labels (base + discovered)

LLM Prompt:
  "Analyze this message and suggest a NEW semantic category"

LLM Response:
  - New label (e.g., "fitness_training")
  - "EXISTING" (fits existing category)
  - "UNCLEAR" (not confident)
```

### 3. Candidate Pool
```
New label → Add to candidate pool

Track:
  - Example texts
  - User IDs
  - Frequency count
  - Discovery timestamp
```

### 4. Promotion Decision
```
Check criteria:
  ✓ Frequency >= min_examples_to_add (3)
  ✓ Used by >= 1 user
  ✓ Not too similar to existing labels (< 80%)

If all pass:
  → Promote to official label set
  → Save to disk (models/distilbert/discovered_labels.json)
  → TODO: Trigger classifier retraining
```

---

## 📊 Label Quality Control

### Similarity Check
Prevents duplicate labels like:
- "health" vs "health_wellness" (85% similar) ❌
- "cooking" vs "travel" (20% similar) ✅

### Frequency Threshold
Requires multiple occurrences:
- 1 occurrence: Candidate
- 2 occurrences: Still candidate
- 3 occurrences: **Promoted** ✅

### User Diversity (Optional)
Can require multiple users:
- Single-user testing: `min_users = 1`
- Multi-user deployment: `min_users = 3`

---

## 🗂️ Persistence

### Storage Location
```
models/distilbert/discovered_labels.json
```

### Format
```json
{
  "discovered_labels": [
    "fitness_training",
    "cooking_recipes",
    "travel_plans"
  ],
  "label_metadata": {
    "fitness_training": {
      "discovered_at": "2025-11-14T10:30:00Z",
      "example_texts": [
        "I'm training for a marathon",
        "Need to improve my running stamina",
        "Started going to the gym 3x per week"
      ],
      "user_ids": ["user123", "user456"],
      "frequency": 5
    }
  },
  "last_updated": "2025-11-14T12:00:00Z"
}
```

### Auto-Loading
Discovered labels are automatically loaded on service restart.

---

## 🚀 Enabling in Production

### Step 1: Test Locally
```bash
# Edit config/base.yaml
adaptive_discovery:
  enabled: true  # Enable for testing

# Restart service
./start_service.sh
```

### Step 2: Monitor Discoveries
```bash
# Check discovered labels
cat models/distilbert/discovered_labels.json

# Check logs
tail -f logs/dappy.log | grep "Discovered"
```

### Step 3: Review & Curate
```python
# Review candidate labels
stats = discovery.get_statistics()
print(f"Candidates: {stats['candidate_labels']}")

# Manually promote/reject if needed
# (Admin dashboard - Phase 2 feature)
```

### Step 4: Retrain Classifier
```bash
# TODO: Implement incremental fine-tuning
# For now, manually retrain with new labels

python -m ml.training.train_distilbert \
  --train \
  --additional-labels fitness_training,cooking_recipes
```

---

## ⚠️ Known Limitations

### 1. No Automatic Retraining
- **Issue:** Classifier doesn't automatically retrain with new labels
- **Workaround:** Manual retraining or use zero-shot temporarily
- **Fix:** Phase 2 - Incremental fine-tuning with LoRA

### 2. Label Explosion Risk
- **Issue:** Too many niche labels over time
- **Mitigation:** Periodic pruning of low-frequency labels
- **Fix:** Phase 2 - Admin dashboard for label management

### 3. LLM Cost
- **Issue:** Each discovery calls GPT-4o-mini
- **Mitigation:** Only triggers when unknown > threshold
- **Cost:** ~$0.0001 per discovery (very cheap)

### 4. Context Window
- **Issue:** Only uses last 5 messages for context
- **Mitigation:** Usually sufficient for semantic understanding
- **Enhancement:** Phase 2 - Use full conversation summary

---

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ml/test_adaptive_discovery.py -v
```

### Integration Test
```python
# Test discovery flow
async def test_discovery():
    discovery = AdaptiveLabelDiscovery(
        llm_client=llm,
        config={'adaptive_discovery': {'enabled': True}},
        base_labels=['identity', 'family', ...],
        enabled=True
    )
    
    # Simulate unknown prediction
    text = "I'm training for a marathon"
    base_pred = (['unknown'], {'unknown': 0.85})
    
    labels, scores, discovered = await discovery.classify_with_discovery(
        text=text,
        user_id="test_user",
        base_prediction=base_pred
    )
    
    assert discovered is not None
    print(f"Discovered: {discovered}")
```

---

## 📈 Future Enhancements

### Phase 2
- [ ] Incremental fine-tuning with LoRA
- [ ] Admin dashboard for label management
- [ ] Label hierarchy (parent-child relationships)
- [ ] Automatic label merging (similar labels)

### Phase 3
- [ ] Per-user label preferences
- [ ] Context-aware label importance
- [ ] Semantic label relationships
- [ ] Multi-language label support

---

## 🐛 Troubleshooting

### Discovery Not Triggering
```python
# Check if enabled
print(discovery.enabled)  # Should be True

# Check unknown threshold
print(discovery.unknown_threshold)  # Default: 0.7

# Check prediction
labels, scores = classifier.predict_single(text)
print(f"Unknown score: {scores.get('unknown', 0)}")
```

### Labels Not Promoting
```python
# Check frequency
stats = discovery.get_statistics()
print(stats['candidate_labels'])

# Check metadata
for label, meta in discovery.label_metadata.items():
    print(f"{label}: {meta['frequency']} examples")
```

### LLM Errors
```python
# Check API key
print(os.getenv('OPENAI_API_KEY'))

# Check LLM client
response = await llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "test"}]
)
print(response)
```

---

**Ready to use!** 🎯

Enable in `config/base.yaml` when you're ready to test in Phase 2.

