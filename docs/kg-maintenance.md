# Knowledge Graph Maintenance

## What It Does

The KG Maintenance Agent runs in the background and automatically:

1. **Detects contradictions** - When new facts conflict with existing ones
2. **Resolves conflicts** - Removes outdated information
3. **Reinforces relations** - Increments confidence when facts are repeated
4. **Cleans up** - Removes low-confidence relations that were never reinforced

## How It Works

### Trigger

After each memory is stored and scored, an event is emitted:

```python
Event(topic="ego_scoring_complete", payload={
    "user_id": "...",
    "memory_id": "...",
    "ego_score": 0.85,
    "relations_count": 3
})
```

### Processing

The KG Maintenance Worker:
1. Receives the event
2. Fetches the new memory and its relations
3. Compares with existing relations in the graph
4. Uses an LLM to detect contradictions and reinforcements
5. Updates the knowledge graph accordingly

### Example: Contradiction

**User says:** "My father's name is John Smith"
- Relation stored: `(user, father_name, John Smith)`

**User corrects:** "Actually, my father's name is James Smith"
- New relation: `(user, father_name, James Smith)`
- Agent detects contradiction
- Resolution: Remove old relation, keep new one
- Result: Clean, accurate graph

### Example: Reinforcement

**User says:** "I work on the liquidation bot"
- Relation: `(user, works_on, liquidation bot)` - confidence: 0.80, mentions: 0

**User mentions again:** "The liquidation bot is going well"
- Agent detects reinforcement
- Updates: confidence → 0.85, mentions → 1
- Result: Stronger, more confident relation

## Benefits

- **Self-correcting**: Graph stays accurate as users correct mistakes
- **Usage-driven**: Frequently mentioned facts gain confidence
- **Non-blocking**: Runs in background, doesn't slow down chat
- **LLM-powered**: Understands semantic contradictions, not just exact text matches

## Configuration

Configured in `llm-orchestration/config/base.yaml`:

```yaml
kg_maintenance:
  enabled: true
  batch_size: 5  # Process 5 events in parallel
  contradiction_threshold: 0.7
  cleanup_interval: 86400  # Daily cleanup of low-confidence relations
```
