from typing import Dict, Any
from ml.component_scorers.base import ComponentScorer, ScorerResult


class ExplicitImportanceScorer(ComponentScorer):
    """
    Calculates explicit importance based on the memory's label (type).
    This is a lookup-based scorer, mapping predefined labels to importance values.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2-3 (per-user preferences)
        # Current: Universal importance map based on memory type
        # Future: Learned per-user (e.g., some users prioritize work > family, others vice versa)
        # Phase 2: Active learning from user confirmations/rejections
        # Phase 3: MetaNet learns context-aware importance (work context: work=1.0, personal context: family=1.0)
        self.importance_map = config.get('ego_scoring', {}).get('explicit_importance_map', {
            'identity': 1.0,      # Core self-concept (name, pronouns, identity)
            'family': 1.0,        # Family relationships and information
            'high_value': 0.95,   # High-stakes work/financial information
            'preference': 0.9,    # Personal likes/dislikes
            'goal': 0.85,         # Personal goals and aspirations
            'relationship': 0.85, # Non-family relationships
            'fact': 0.7,          # Biographical facts (work, education, location)
            'work': 0.7,          # Work-related information
            'education': 0.7,     # Educational background
            'event': 0.6,         # Events and experiences
            'opinion': 0.5,       # Opinions and views
            'unknown': 0.5        # Unclassified memories
        })
        
        # 🔄 HYPERPARAMETER - Will be LEARNED in Phase 2-3
        # Fallback importance for memories without a recognized label
        self.default_importance = config.get('ego_scoring', {}).get('default_explicit_importance', 0.5)
    
    async def score(self, memory: Dict[str, Any], **kwargs) -> ScorerResult:
        """
        Score explicit importance of a memory.
        Requires 'label' in memory (e.g., 'identity:name', 'preference:food').
        """
        label = memory.get('label', 'unknown').lower()
        
        # Try to match full label first, then primary category
        importance = self.importance_map.get(label)
        if importance is None:
            primary_label = label.split(':')[0]
            importance = self.importance_map.get(primary_label, self.default_importance)
        
        return ScorerResult(
            score=importance,
            metadata={"label": label, "mapped_importance": importance}
        )
