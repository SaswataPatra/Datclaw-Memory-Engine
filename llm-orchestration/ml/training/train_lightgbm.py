"""
Training Script for LightGBM Ego Score Combiner

Usage:
    python -m ml.training.train_lightgbm --generate-data --train --tune
"""

import argparse
import asyncio
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from ml.training.lightgbm_data_generator import LightGBMDataGenerator, compute_component_scores
from ml.combiners.lightgbm_combiner import LightGBMCombiner
from ml.component_scorers import (
    NoveltyScorer, FrequencyScorer, SentimentScorer,
    ExplicitImportanceScorer, EngagementScorer
)
from services.embedding_service import EmbeddingService
from qdrant_client import QdrantClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_dataset(api_key: str, output_path: str, variations_per_seed: int = 10):
    """Generate bootstrap training dataset"""
    logger.info("Generating bootstrap training dataset...")
    
    generator = LightGBMDataGenerator(api_key)
    dataset = await generator.generate_full_dataset(variations_per_seed=variations_per_seed)
    
    # Save dataset
    generator.save_dataset(dataset, output_path)
    
    # Print statistics
    ego_dist = generator.get_ego_score_distribution(dataset)
    label_dist = generator.get_label_distribution(dataset)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"Ego score distribution:")
    for tier, count in sorted(ego_dist.items()):
        logger.info(f"  {tier}: {count} examples")
    
    logger.info(f"Label distribution:")
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label}: {count} examples")
    
    return dataset


async def compute_features(
    dataset: List[Tuple[str, str, float]],
    config: Dict[str, Any],
    qdrant_url: str,
    openai_api_key: str
) -> pd.DataFrame:
    """
    Compute component scores for all examples in dataset.
    
    Args:
        dataset: List of (text, label, target_ego_score) tuples
        config: Configuration dict
        qdrant_url: Qdrant server URL
        openai_api_key: OpenAI API key
    
    Returns:
        DataFrame with features and target ego scores
    """
    logger.info("Computing component scores for all examples...")
    
    # Initialize services
    qdrant_client = QdrantClient(url=qdrant_url)
    embedding_service = EmbeddingService(api_key=openai_api_key)
    
    # Initialize component scorers
    component_scorers = {
        'novelty': NoveltyScorer(config, qdrant_client),
        'frequency': FrequencyScorer(config, qdrant_client),
        'sentiment': SentimentScorer(config),
        'explicit_importance': ExplicitImportanceScorer(config),
        'engagement': EngagementScorer(config)
    }
    
    # Compute scores for each example
    data = []
    user_id = "bootstrap_user"  # Use consistent user_id for training
    
    for idx, (text, label, target_ego_score) in enumerate(dataset):
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing example {idx + 1}/{len(dataset)}...")
        
        try:
            scores = await compute_component_scores(
                text, label, user_id, component_scorers, embedding_service,
                simulate_realistic_features=True  # Enable realistic feature simulation
            )
            scores['target_ego_score'] = target_ego_score
            scores['text'] = text
            scores['label'] = label
            data.append(scores)
        except Exception as e:
            logger.error(f"Error processing example {idx}: {e}")
            continue
    
    df = pd.DataFrame(data)
    logger.info(f"Computed features for {len(df)} examples")
    
    return df


def train_model(
    dataset_path: str,
    output_dir: str,
    config: Dict[str, Any],
    qdrant_url: str,
    openai_api_key: str,
    tune_hyperparameters: bool = False
):
    """
    Train LightGBM combiner model.
    
    Args:
        dataset_path: Path to dataset file
        output_dir: Output directory for model
        config: Configuration dict
        qdrant_url: Qdrant server URL
        openai_api_key: OpenAI API key
        tune_hyperparameters: Whether to run grid search
    """
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    generator = LightGBMDataGenerator(openai_api_key="dummy")
    dataset = generator.load_dataset(dataset_path)
    
    # Compute features
    df = asyncio.run(compute_features(dataset, config, qdrant_url, openai_api_key))
    
    # Save features for future use
    features_path = os.path.join(output_dir, 'training_features.csv')
    df.to_csv(features_path, index=False)
    logger.info(f"Features saved to {features_path}")
    
    # Prepare training data
    feature_cols = [
        'novelty_score', 'frequency_score', 'sentiment_intensity',
        'explicit_importance_score', 'engagement_score',
        'recency_decay', 'reference_count', 'llm_confidence', 'source_weight'
    ]
    
    X = df[feature_cols]
    y = df['target_ego_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Initialize model
    combiner = LightGBMCombiner(config)
    
    if tune_hyperparameters:
        logger.info("Running hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'max_depth': [-1, 5, 10]
        }
        
        grid_search = GridSearchCV(
            lgb.LGBMRegressor(objective='mae', metric='mae', verbose=-1, seed=42),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV MAE: {-grid_search.best_score_:.4f}")
        
        # Update combiner with best params
        combiner.model = grid_search.best_estimator_
        combiner.is_trained = True
        combiner.feature_names = feature_cols
    else:
        # Train with default parameters
        logger.info("Training with default parameters...")
        training_data = X_train.copy()
        training_data['target_ego_score'] = y_train
        combiner.train(training_data.to_dict('records'))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = combiner.model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Set Performance:")
    logger.info(f"{'='*60}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"{'='*60}\n")
    
    # Feature importance
    feature_importance = combiner.get_feature_importance()
    logger.info("Feature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {feature}: {importance:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'combiner.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(combiner, f)
    logger.info(f"Model saved to {model_path}")
    
    # Generate SHAP plots (if enabled)
    try:
        logger.info("Generating SHAP plots...")
        explainer = shap.TreeExplainer(combiner.model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
        logger.info(f"SHAP summary plot saved to {output_dir}/shap_summary.png")
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
        logger.info(f"SHAP importance plot saved to {output_dir}/shap_importance.png")
        
    except Exception as e:
        logger.warning(f"Could not generate SHAP plots: {e}")
    
    # Compare with fallback weights
    logger.info("\nComparing with fallback weights...")
    fallback_weights = {
        'novelty': 0.2,
        'frequency': 0.1,
        'sentiment': 0.1,
        'explicit_importance': 0.4,
        'engagement': 0.2
    }
    
    y_fallback = (
        X_test['novelty_score'] * fallback_weights['novelty'] +
        X_test['frequency_score'] * fallback_weights['frequency'] +
        X_test['sentiment_intensity'] * fallback_weights['sentiment'] +
        X_test['explicit_importance_score'] * fallback_weights['explicit_importance'] +
        X_test['engagement_score'] * fallback_weights['engagement']
    )
    
    fallback_mae = mean_absolute_error(y_test, y_fallback)
    fallback_r2 = r2_score(y_test, y_fallback)
    
    logger.info(f"Fallback (hardcoded weights) performance:")
    logger.info(f"  MAE: {fallback_mae:.4f}")
    logger.info(f"  R²:  {fallback_r2:.4f}")
    logger.info(f"\nImprovement:")
    logger.info(f"  MAE: {((fallback_mae - mae) / fallback_mae * 100):.2f}% better")
    logger.info(f"  R²:  {((r2 - fallback_r2) / abs(fallback_r2) * 100):.2f}% better")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Ego Score Combiner")
    
    parser.add_argument('--generate-data', action='store_true', help='Generate bootstrap dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--dataset-path', type=str, default='data/lightgbm_dataset.jsonl', help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='models/lightgbm', help='Output directory for model')
    parser.add_argument('--openai-api-key', type=str, help='OpenAI API key')
    parser.add_argument('--qdrant-url', type=str, default='http://localhost:6333', help='Qdrant server URL')
    parser.add_argument('--variations-per-seed', type=int, default=20, help='Variations per seed example (default: 20 for 1000+ examples)')
    parser.add_argument('--config-path', type=str, default='config/base.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    import yaml
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate dataset
    if args.generate_data:
        if not args.openai_api_key:
            args.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not args.openai_api_key:
                raise ValueError("OpenAI API key required. Set --openai-api-key or OPENAI_API_KEY env var.")
        
        asyncio.run(generate_dataset(
            args.openai_api_key,
            args.dataset_path,
            args.variations_per_seed
        ))
    
    # Train model
    if args.train:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}. Run with --generate-data first.")
        
        if not args.openai_api_key:
            args.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not args.openai_api_key:
                raise ValueError("OpenAI API key required. Set --openai-api-key or OPENAI_API_KEY env var.")
        
        train_model(
            args.dataset_path,
            args.output_dir,
            config,
            args.qdrant_url,
            args.openai_api_key,
            tune_hyperparameters=args.tune
        )


if __name__ == '__main__':
    main()

