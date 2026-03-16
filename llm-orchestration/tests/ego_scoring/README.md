# Ego Scoring Test Suite

This directory contains comprehensive tests for the ego scoring system components.

## Test Files

### 1. `test_component_scorers.py`
Tests all component scorers that extract features from memories:
- **NoveltyScorer**: Tests novelty calculation based on semantic similarity
- **FrequencyScorer**: Tests frequency calculation based on repeated patterns
- **SentimentScorer**: Tests sentiment intensity extraction
- **ExplicitImportanceScorer**: Tests label-based importance mapping
- **EngagementScorer**: Tests user engagement metrics
- **Integration Tests**: Tests all scorers working together

### 2. `test_lightgbm_combiner.py`
Tests the LightGBM Combiner that learns optimal feature weights:
- Model initialization and training
- Prediction accuracy for high/medium/low ego scores
- Feature importance extraction
- Boundary value handling
- Prediction consistency
- SHAP integration (optional)

### 3. `test_confidence_combiner.py`
Tests the Confidence Combiner for routing decisions:
- Confidence combination from multiple sources
- Routing decisions (auto_store, active_learning, discard)
- PII penalty application
- Semantic inconsistency penalty
- Multiple penalty handling
- Threshold boundary testing

### 4. `test_bootstrap_generator.py`
Tests the Bootstrap Dataset Generator for training data:
- Seed example management
- Synthetic data generation using LLM
- Full dataset generation (seeds + synthetic)
- Mock feature generation
- Training data preparation
- Error handling

## Running Tests

### Run all ego scoring tests:
```bash
cd llm-orchestration
pytest tests/ego_scoring/ -v
```

### Run specific test file:
```bash
pytest tests/ego_scoring/test_component_scorers.py -v
pytest tests/ego_scoring/test_lightgbm_combiner.py -v
pytest tests/ego_scoring/test_confidence_combiner.py -v
pytest tests/ego_scoring/test_bootstrap_generator.py -v
```

### Run with coverage:
```bash
pytest tests/ego_scoring/ --cov=ml --cov-report=html
```

### Run specific test:
```bash
pytest tests/ego_scoring/test_component_scorers.py::test_novelty_scorer_high_novelty -v
```

## Test Coverage

The test suite covers:
- ✅ All component scorers (5 scorers)
- ✅ LightGBM Combiner (training, prediction, feature importance)
- ✅ Confidence Combiner (routing, penalties, thresholds)
- ✅ Bootstrap Generator (seed management, synthetic generation)
- ✅ Edge cases and boundary values
- ✅ Error handling
- ✅ Integration scenarios

## Dependencies

Required packages for testing:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

## Test Data

Tests use:
- Mock Qdrant client for vector search
- Mock LLM client for synthetic data generation
- Predefined seed examples for training
- Synthetic feature generation for bootstrap testing

## Expected Results

All tests should pass with:
- Component scorers returning scores in [0, 1] range
- LightGBM predictions matching expected ego score ranges
- Confidence Combiner routing correctly based on thresholds
- Bootstrap Generator creating diverse training data

## Notes

- Tests use `pytest-asyncio` for async test support
- Mock objects are used to avoid external dependencies (Qdrant, LLM APIs)
- Tests are designed to run quickly without requiring actual database connections
- SHAP tests are optional and can be disabled for faster testing

