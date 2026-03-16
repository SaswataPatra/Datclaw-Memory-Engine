from typing import Dict, Any, List, Optional
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import shap
import logging

from ml.component_scorers.base import ScorerResult

logger = logging.getLogger(__name__)


class LightGBMCombiner:
    """
    Combines various component scores into a final ego score using a LightGBM model.
    This model learns the optimal weights for each feature from a training dataset.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Load LightGBM parameters from config
        lgbm_params = config.get('ego_scoring', {}).get('lightgbm_params', {
            # ⚙️ STAYS - Task definition (regression for ego score prediction)
            'objective': 'mae',
            'metric': 'mae',
            
            # 🔧 HYPERPARAMETER - Will be TUNED via grid search in Phase 1.5
            # Current: Default values that work reasonably well
            # Future: Optimized via cross-validation on bootstrap dataset
            'n_estimators': 100,      # Number of boosting rounds
            'learning_rate': 0.1,     # Step size shrinkage
            'num_leaves': 31,         # Max number of leaves per tree
            
            # ⚙️ STAYS - Infrastructure/reproducibility settings
            'verbose': -1,            # Suppress training logs
            'n_jobs': -1,             # Use all CPU cores
            'seed': 42                # Random seed for reproducibility
        })
        self.model = lgb.LGBMRegressor(**lgbm_params)
        
        logger.info("LightGBMCombiner initialized with default parameters.")
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Trains the LightGBM model on a list of feature dictionaries.
        Each dictionary should contain component scores and a 'target_ego_score'.
        """
        if not training_data:
            logger.warning("No training data provided for LightGBMCombiner.")
            return
        
        df = pd.DataFrame(training_data)
        
        # Define feature columns (ensure these match the component scorer outputs)
        # These are the inputs to the LightGBM model
        feature_cols = [
            'novelty_score', 'frequency_score', 'sentiment_intensity',
            'explicit_importance_score', 'engagement_score',
            'recency_decay', 'reference_count', 'llm_confidence', 'source_weight'
        ]
        
        # Ensure all feature columns exist, fill missing with 0 or a sensible default
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0 # Or a more appropriate default
        
        self.feature_names = feature_cols
        
        # Prepare data
        X = df[feature_cols]
        y = df['target_ego_score'] # The ground truth ego score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training LightGBM model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"LightGBM training complete. Test MAE: {mae:.4f}")
        
        self.is_trained = True
        
        # Optional: SHAP for feature importance (can be resource intensive)
        if self.config.get('ego_scoring', {}).get('enable_shap', False):
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)
                logger.info("SHAP values calculated for LightGBM model.")
                # Store or visualize shap_values as needed
            except Exception as e:
                logger.warning(f"Failed to calculate SHAP values: {e}")
    
    def predict(self, features: Dict[str, Any]) -> float:
        """
        Predicts the ego score for a given set of features.
        Features should be the output of the component scorers.
        """
        if not self.is_trained:
            logger.warning("⚠️  LightGBMCombiner is not trained. Returning default score 0.5")
            return 0.5 # Default score if not trained
        
        logger.info(f"🤖 LightGBM Prediction - Input Features:")
        
        # Ensure features are in the correct order and format for the model
        input_features = pd.DataFrame([features])[self.feature_names]
        
        # Log each feature value
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            logger.info(f"   {feature_name:30s} = {value:.4f}")
        
        # Get feature importances
        importances = self.get_feature_importance()
        
        logger.info(f"🎯 LightGBM Feature Importances (learned weights):")
        for feature_name in self.feature_names:
            importance = importances.get(feature_name, 0.0)
            logger.info(f"   {feature_name:30s} → {importance:.4f}")
        
        # Make prediction
        prediction = self.model.predict(input_features)[0]
        
        # Clip prediction to [0, 1]
        clipped_prediction = max(0.0, min(1.0, float(prediction)))
        
        # Calculate weighted contribution of each feature
        logger.info(f"📊 Feature Contributions to Final Score:")
        normalized_weights = self.get_normalized_weights()
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)
            weight = normalized_weights.get(feature_name, 0.0)
            contribution = value * weight
            logger.info(f"   {feature_name:30s}: {value:.4f} × {weight:.4f} = {contribution:.4f}")
        
        logger.info(f"  ✅ LightGBM Final Score: {clipped_prediction:.4f} (raw: {prediction:.4f})")
        
        if clipped_prediction != prediction:
            logger.warning(f"  ⚠️  Score was clipped from {prediction:.4f} to {clipped_prediction:.4f}")
        
        return clipped_prediction
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Returns feature importance from the trained LightGBM model."""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def get_normalized_weights(self) -> Dict[str, float]:
        """
        Returns normalized feature importances as weights (sum to 1.0).
        Can be used as fallback weights when prediction times out.
        
        Returns:
            Dict mapping feature names to normalized weights (0.0-1.0, sum=1.0)
        """
        if not self.is_trained:
            # Return default weights if model not trained
            return {
                'novelty_score': 0.2,
                'frequency_score': 0.1,
                'sentiment_intensity': 0.1,
                'explicit_importance_score': 0.4,
                'engagement_score': 0.2,
                'recency_decay': 0.0,
                'reference_count': 0.0,
                'llm_confidence': 0.0,
                'source_weight': 0.0
            }
        
        importances = self.model.feature_importances_
        total = sum(importances)
        
        if total == 0:
            # Fallback to uniform weights
            return {name: 1.0 / len(self.feature_names) for name in self.feature_names}
        
        # Normalize to sum to 1.0
        normalized = {
            name: importance / total
            for name, importance in zip(self.feature_names, importances)
        }
        
        return normalized
    
    def save_model(self, path: str):
        """
        Save the trained LightGBM model to disk.
        
        Args:
            path: Path to save the model (pickle format)
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import pickle
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LightGBM model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained LightGBM model from disk.
        
        Args:
            path: Path to the saved model (pickle format)
        """
        import pickle
        
        logger.info(f"📥 Loading LightGBM model from {path}...")
        
        with open(path, 'rb') as f:
            loaded_combiner = pickle.load(f)
        
        # The training script saved the entire LightGBMCombiner object
        # Copy its attributes to this instance
        if isinstance(loaded_combiner, LightGBMCombiner):
            self.model = loaded_combiner.model
            self.feature_names = loaded_combiner.feature_names
            self.is_trained = loaded_combiner.is_trained
        else:
            # Fallback: assume it's a dict (from save_model method)
            self.model = loaded_combiner['model']
            self.feature_names = loaded_combiner['feature_names']
            self.is_trained = loaded_combiner['is_trained']
        
        logger.info(f"  ✅ LightGBM model loaded successfully")
        logger.info(f"  📊 Features ({len(self.feature_names)}): {self.feature_names}")
        logger.info(f"  🎯 Model trained: {self.is_trained}")
