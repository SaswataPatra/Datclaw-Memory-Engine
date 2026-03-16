# Ego Scoring Algorithm

## What Is Ego Scoring?

Ego scoring automatically determines how important a memory is to remember, scoring it from 0 (forget immediately) to 1 (remember forever).

## Why "Ego"?

The score represents importance from the **user's perspective** (their "ego network"). It's not about objective importance, but about what matters to *this specific user*.

## Components

Ego score is calculated from 6 components:

### 1. Explicit Importance (0-1)

Detects explicit signals:
- "This is important"
- "Remember this"
- "Don't forget"
- User corrections or emphasis

### 2. Temporal Recency (0-1)

Recent memories score higher:
- Formula: `recency = exp(-λ * age_days)`
- Half-life varies by tier (Tier 1: 180 days, Tier 4: 5 minutes)

### 3. Novelty (0-1)

New information scores higher than repeated facts:
- Compares embedding to existing memories
- High novelty = new concept
- Low novelty = already known

### 4. Frequency (0-1)

Repeated concepts indicate importance:
- Counts how often similar content appears
- Normalized: `min(1, count / 10)`

### 5. Sentiment Intensity (0-1)

Strong emotions indicate importance:
- Positive or negative intensity
- Neutral statements score lower

### 6. Engagement (0-1)

User behavior signals:
- Long messages
- Follow-up questions
- Corrections
- Explicit save actions

## Combining Components

All components are combined using a **LightGBM model** that learns optimal weights:

```
ego_score = LightGBM(
    explicit_importance,
    recency,
    novelty,
    frequency,
    sentiment_intensity,
    engagement,
    extractor_confidence
)
```

The model is trained on labeled examples and can be personalized per-user.

## Memory Tiers

Based on ego score, memories are assigned to tiers:

- **Tier 1** (ego ≥ 0.8): Long-term storage, 180-day half-life
- **Tier 2** (ego ≥ 0.6): Medium-term, 7-day half-life
- **Tier 3** (ego ≥ 0.4): Short-term, 1-day half-life
- **Tier 4** (ego < 0.4): Ephemeral, 5-minute half-life

Memories age out based on their tier, freeing up storage for what matters.

## Configuration

Tune ego scoring in `llm-orchestration/config/base.yaml`:

```yaml
ego_scoring:
  weights:
    explicit_importance: 0.25
    recency: 0.20
    novelty: 0.15
    frequency: 0.15
    sentiment: 0.15
    engagement: 0.10
  
  tier_thresholds:
    tier1: 0.8
    tier2: 0.6
    tier3: 0.4
```
