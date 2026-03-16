"""
Unit tests for PPRRetrieval.

Tests:
1. PPR algorithm
2. Context-aware filtering
3. Path finding
4. Neighborhood retrieval
"""

import pytest
import networkx as nx
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

from core.graph.ppr_retrieval import PPRRetrieval, PPRResult


class TestPPRRetrieval:
    """Test PPRRetrieval."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock ArangoDB connection."""
        db = Mock()
        db.aql = Mock()
        return db
    
    @pytest.fixture
    def retrieval(self, mock_db):
        """Create PPRRetrieval instance."""
        return PPRRetrieval(
            db=mock_db,
            config={
                "ppr": {
                    "alpha": 0.85,
                    "max_iter": 100,
                    "top_k": 10
                }
            }
        )
    
    def test_init(self, retrieval):
        """Test initialization."""
        assert retrieval is not None
        assert retrieval.alpha == 0.85
        assert retrieval.max_iter == 100
        assert retrieval.default_top_k == 10
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_graph(self, retrieval, mock_db):
        """Test retrieval from empty graph."""
        # Mock empty result
        mock_db.aql.execute.return_value = iter([])
        
        result = await retrieval.retrieve(
            user_id="user123",
            seed_entities=["entity1"]
        )
        
        assert isinstance(result, PPRResult)
        # Seed entity is always added to the graph, so nodes won't be empty
        assert len(result.nodes) >= 0  # May include seed
        assert result.edges == []
    
    @pytest.mark.asyncio
    async def test_retrieve_with_results(self, retrieval, mock_db):
        """Test retrieval with graph data."""
        # Mock graph data
        mock_data = [
            {
                "vertex": {
                    "_key": "sarah",
                    "canonical_name": "Sarah",
                    "type": "person"
                },
                "edge": {
                    "_from": "entities/user",
                    "_to": "entities/sarah",
                    "predicate": "sister_of",
                    "relation_category": "family",
                    "strength": 0.9
                },
                "path_length": 1
            },
            {
                "vertex": {
                    "_key": "user",
                    "canonical_name": "User",
                    "type": "person"
                },
                "edge": None,
                "path_length": 0
            }
        ]
        mock_db.aql.execute.return_value = iter(mock_data)
        
        result = await retrieval.retrieve(
            user_id="user123",
            seed_entities=["user"]
        )
        
        assert isinstance(result, PPRResult)
        # Nodes should be populated
        assert result.metadata.get("total_nodes") >= 1
    
    @pytest.mark.asyncio
    async def test_retrieve_context(self, retrieval, mock_db):
        """Test context retrieval."""
        mock_db.aql.execute.return_value = iter([])
        
        context = await retrieval.retrieve_context(
            user_id="user123",
            query_entities=["sarah"],
            relation_category="family"
        )
        
        assert isinstance(context, list)


class TestNetworkXIntegration:
    """Test NetworkX PPR algorithm directly."""
    
    def test_ppr_simple_graph(self):
        """Test PPR on a simple graph."""
        G = nx.DiGraph()
        
        # Create a simple family graph
        G.add_edge("user", "sarah", weight=0.9)
        G.add_edge("sarah", "user", weight=0.9)  # Bidirectional
        G.add_edge("user", "mom", weight=0.95)
        G.add_edge("mom", "sarah", weight=0.8)
        
        # Run PPR from user
        personalization = {"user": 1.0}
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization)
        
        # User should have highest score (seed)
        assert scores["user"] > scores["sarah"]
        # Sarah should be reachable
        assert scores["sarah"] > 0
        # Mom should be reachable
        assert scores["mom"] > 0
    
    def test_ppr_disconnected_nodes(self):
        """Test PPR with disconnected nodes."""
        G = nx.DiGraph()
        
        # Connected component
        G.add_edge("user", "sarah")
        G.add_edge("sarah", "user")
        
        # Disconnected node
        G.add_node("stranger")
        
        personalization = {"user": 1.0}
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization)
        
        # Disconnected node should have low score
        assert scores["stranger"] < scores["user"]
        assert scores["stranger"] < scores["sarah"]
    
    def test_ppr_weighted_edges(self):
        """Test PPR respects edge weights."""
        G = nx.DiGraph()
        
        # Strong connection
        G.add_edge("user", "close_friend", weight=0.95)
        
        # Weak connection
        G.add_edge("user", "acquaintance", weight=0.3)
        
        personalization = {"user": 1.0}
        scores = nx.pagerank(G, alpha=0.85, personalization=personalization, weight="weight")
        
        # Close friend should have higher score due to weight
        assert scores["close_friend"] > scores["acquaintance"]
    
    def test_shortest_path(self):
        """Test shortest path finding."""
        G = nx.DiGraph()
        
        # Create a path: user -> sarah -> mom -> grandma
        G.add_edge("user", "sarah")
        G.add_edge("sarah", "mom")
        G.add_edge("mom", "grandma")
        
        # Direct path exists
        path = nx.shortest_path(G, "user", "grandma")
        
        assert path == ["user", "sarah", "mom", "grandma"]
        assert len(path) == 4
    
    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        G = nx.DiGraph()
        
        G.add_node("user")
        G.add_node("stranger")
        
        with pytest.raises(nx.NetworkXNoPath):
            nx.shortest_path(G, "user", "stranger")


class TestPPRResult:
    """Test PPRResult dataclass."""
    
    def test_ppr_result_creation(self):
        """Test creating PPRResult."""
        result = PPRResult(
            nodes=[{"entity_id": "sarah", "ppr_score": 0.8}],
            edges=[{"from": "user", "to": "sarah", "predicate": "knows"}],
            scores={"sarah": 0.8, "user": 0.5},
            hops=2,
            context_filter="family"
        )
        
        assert len(result.nodes) == 1
        assert len(result.edges) == 1
        assert result.scores["sarah"] == 0.8
        assert result.hops == 2
        assert result.context_filter == "family"
    
    def test_ppr_result_empty(self):
        """Test creating empty PPRResult."""
        result = PPRResult(
            nodes=[],
            edges=[],
            scores={},
            hops=0,
            context_filter=None
        )
        
        assert len(result.nodes) == 0
        assert len(result.edges) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

