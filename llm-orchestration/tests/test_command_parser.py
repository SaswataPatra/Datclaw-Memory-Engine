"""
Unit tests for CommandParser and CommandTrainingCollector.

Tests:
1. Command parsing
2. Alias resolution
3. System-only command handling
4. Multiple commands
5. Training data collection
"""

import pytest
import tempfile
import os
from core.command_parser import (
    CommandParser,
    UserCommand,
    CommandCategory,
    COMMAND_CONFIG,
    COMMAND_ALIASES,
    SYSTEM_ONLY_COMMANDS,
    parse_command,
)
from core.command_training_collector import CommandTrainingCollector


class TestCommandParser:
    """Test CommandParser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a CommandParser instance."""
        return CommandParser()
    
    def test_init(self, parser):
        """Test parser initialization."""
        assert parser is not None
        assert len(parser.commands) > 0
        assert len(parser.aliases) > 0
    
    # ==================== MEMORY CONTROL ====================
    
    def test_parse_remember(self, parser):
        """Test /remember command."""
        commands, remaining = parser.parse("/remember My sister Sarah lives in NYC")
        
        assert len(commands) == 1
        assert commands[0].command == "remember"
        assert commands[0].category == CommandCategory.MEMORY_CONTROL
        assert commands[0].target_text == "My sister Sarah lives in NYC"
        assert remaining == "My sister Sarah lives in NYC"
    
    def test_parse_forget(self, parser):
        """Test /forget command."""
        commands, remaining = parser.parse("/forget this memory")
        
        assert len(commands) == 1
        assert commands[0].command == "forget"
        assert commands[0].config.get("requires_confirmation") is True
    
    def test_parse_remember_all(self, parser):
        """Test /remember-all command."""
        commands, remaining = parser.parse("/remember-all")
        
        assert len(commands) == 1
        assert commands[0].command == "remember-all"
        assert commands[0].config.get("action") == "bulk_store"
    
    # ==================== TIER CONTROL ====================
    
    def test_parse_tier2(self, parser):
        """Test /tier2 command."""
        commands, remaining = parser.parse("/tier2 Important work info")
        
        assert len(commands) == 1
        assert commands[0].command == "tier2"
        assert commands[0].category == CommandCategory.TIER_CONTROL
        assert commands[0].config.get("tier") == 2
    
    def test_parse_tier3(self, parser):
        """Test /tier3 command."""
        commands, remaining = parser.parse("/tier3 Short term note")
        
        assert len(commands) == 1
        assert commands[0].command == "tier3"
        assert commands[0].config.get("tier") == 3
    
    def test_parse_tier4(self, parser):
        """Test /tier4 command."""
        commands, remaining = parser.parse("/tier4 Temporary info")
        
        assert len(commands) == 1
        assert commands[0].command == "tier4"
        assert commands[0].config.get("tier") == 4
    
    def test_tier1_blocked(self, parser):
        """Test that /tier1 is converted to /remember."""
        commands, remaining = parser.parse("/tier1 My core identity")
        
        # Should be converted to /remember, not /tier1
        assert len(commands) == 1
        assert commands[0].command == "remember"
        assert commands[0].config.get("tier") is None  # No tier forcing
    
    # ==================== PRIVACY ====================
    
    def test_parse_private(self, parser):
        """Test /private command."""
        commands, remaining = parser.parse("/private My SSN is 123-45-6789")
        
        assert len(commands) == 1
        assert commands[0].command == "private"
        assert commands[0].category == CommandCategory.PRIVACY
        assert commands[0].config.get("encrypt") is True
    
    def test_parse_sensitive(self, parser):
        """Test /sensitive command (alias for /private)."""
        commands, remaining = parser.parse("/sensitive My password")
        
        assert len(commands) == 1
        assert commands[0].command == "sensitive"
        assert commands[0].config.get("encrypt") is True
    
    # ==================== CORRECTIONS ====================
    
    def test_parse_correct(self, parser):
        """Test /correct command."""
        commands, remaining = parser.parse("/correct I actually prefer pasta")
        
        assert len(commands) == 1
        assert commands[0].command == "correct"
        assert commands[0].category == CommandCategory.CORRECTIONS
        assert commands[0].config.get("edge_type") == "contradicts"
    
    def test_parse_evolve(self, parser):
        """Test /evolve command."""
        commands, remaining = parser.parse("/evolve My taste has changed")
        
        assert len(commands) == 1
        assert commands[0].command == "evolve"
        assert commands[0].config.get("edge_type") == "supersedes"
    
    def test_parse_update(self, parser):
        """Test /update command."""
        commands, remaining = parser.parse("/update New information")
        
        assert len(commands) == 1
        assert commands[0].command == "update"
    
    # ==================== GRAPH CONTROL ====================
    
    def test_parse_link(self, parser):
        """Test /link command with argument."""
        commands, remaining = parser.parse("/link Sarah She is my sister")
        
        assert len(commands) == 1
        assert commands[0].command == "link"
        assert commands[0].category == CommandCategory.GRAPH
        assert "Sarah" in commands[0].args
        assert remaining == "She is my sister"
    
    def test_parse_unlink(self, parser):
        """Test /unlink command."""
        commands, remaining = parser.parse("/unlink OldFriend")
        
        assert len(commands) == 1
        assert commands[0].command == "unlink"
        assert commands[0].config.get("requires_confirmation") is True
    
    # ==================== RETRIEVAL ====================
    
    def test_parse_search(self, parser):
        """Test /search command."""
        commands, remaining = parser.parse("/search pizza preferences")
        
        assert len(commands) == 1
        assert commands[0].command == "search"
        assert commands[0].category == CommandCategory.RETRIEVAL
        assert commands[0].config.get("is_query") is True
    
    def test_parse_context(self, parser):
        """Test /context command."""
        commands, remaining = parser.parse("/context")
        
        assert len(commands) == 1
        assert commands[0].command == "context"
        assert commands[0].config.get("is_query") is True
    
    def test_parse_recall(self, parser):
        """Test /recall command."""
        commands, remaining = parser.parse("/recall sister")
        
        assert len(commands) == 1
        assert commands[0].command == "recall"
    
    # ==================== ALIASES ====================
    
    def test_alias_r(self, parser):
        """Test /r alias for /remember."""
        commands, remaining = parser.parse("/r My note")
        
        assert len(commands) == 1
        assert commands[0].command == "remember"
    
    def test_alias_f(self, parser):
        """Test /f alias for /forget."""
        commands, remaining = parser.parse("/f this")
        
        assert len(commands) == 1
        assert commands[0].command == "forget"
    
    def test_alias_t2(self, parser):
        """Test /t2 alias for /tier2."""
        commands, remaining = parser.parse("/t2 Important")
        
        assert len(commands) == 1
        assert commands[0].command == "tier2"
    
    def test_alias_p(self, parser):
        """Test /p alias for /private."""
        commands, remaining = parser.parse("/p Secret")
        
        assert len(commands) == 1
        assert commands[0].command == "private"
    
    def test_alias_s(self, parser):
        """Test /s alias for /search."""
        commands, remaining = parser.parse("/s query")
        
        assert len(commands) == 1
        assert commands[0].command == "search"
    
    # ==================== MULTIPLE COMMANDS ====================
    
    def test_multiple_commands(self, parser):
        """Test parsing multiple commands."""
        commands, remaining = parser.parse("/remember /private My SSN")
        
        assert len(commands) == 2
        assert commands[0].command == "remember"
        assert commands[1].command == "private"
        assert remaining == "My SSN"
    
    def test_multiple_commands_tier_privacy(self, parser):
        """Test combining tier and privacy commands."""
        commands, remaining = parser.parse("/tier2 /private Work password")
        
        assert len(commands) == 2
        assert commands[0].command == "tier2"
        assert commands[1].command == "private"
    
    # ==================== EDGE CASES ====================
    
    def test_no_command(self, parser):
        """Test message without command."""
        commands, remaining = parser.parse("Hello, how are you?")
        
        assert len(commands) == 0
        assert remaining == "Hello, how are you?"
    
    def test_empty_message(self, parser):
        """Test empty message."""
        commands, remaining = parser.parse("")
        
        assert len(commands) == 0
        assert remaining == ""
    
    def test_none_message(self, parser):
        """Test None message."""
        commands, remaining = parser.parse(None)
        
        assert len(commands) == 0
    
    def test_unknown_command(self, parser):
        """Test unknown command (should be ignored)."""
        commands, remaining = parser.parse("/unknown My message")
        
        assert len(commands) == 0
        assert remaining == "/unknown My message"
    
    def test_case_insensitive(self, parser):
        """Test that commands are case-insensitive."""
        commands1, _ = parser.parse("/REMEMBER Test")
        commands2, _ = parser.parse("/Remember Test")
        commands3, _ = parser.parse("/remember Test")
        
        assert len(commands1) == 1
        assert len(commands2) == 1
        assert len(commands3) == 1
        assert commands1[0].command == commands2[0].command == commands3[0].command
    
    # ==================== HELPER METHODS ====================
    
    def test_has_command(self, parser):
        """Test has_command method."""
        assert parser.has_command("/remember Test") is True
        assert parser.has_command("/r Test") is True
        assert parser.has_command("Hello") is False
        assert parser.has_command("") is False
    
    def test_get_importance_weight(self, parser):
        """Test get_importance_weight method."""
        commands, _ = parser.parse("/remember Test")
        weight = parser.get_importance_weight(commands)
        
        assert weight == 0.8  # /remember weight
    
    def test_get_target_tier(self, parser):
        """Test get_target_tier method."""
        commands, _ = parser.parse("/tier2 Test")
        tier = parser.get_target_tier(commands)
        
        assert tier == 2
    
    def test_requires_confirmation(self, parser):
        """Test requires_confirmation method."""
        forget_cmds, _ = parser.parse("/forget Test")
        remember_cmds, _ = parser.parse("/remember Test")
        
        assert parser.requires_confirmation(forget_cmds) is True
        assert parser.requires_confirmation(remember_cmds) is False
    
    def test_is_retrieval_only(self, parser):
        """Test is_retrieval_only method."""
        search_cmds, _ = parser.parse("/search query")
        remember_cmds, _ = parser.parse("/remember Test")
        
        assert parser.is_retrieval_only(search_cmds) is True
        assert parser.is_retrieval_only(remember_cmds) is False
    
    def test_should_encrypt(self, parser):
        """Test should_encrypt method."""
        private_cmds, _ = parser.parse("/private Secret")
        remember_cmds, _ = parser.parse("/remember Test")
        
        assert parser.should_encrypt(private_cmds) is True
        assert parser.should_encrypt(remember_cmds) is False
    
    def test_creates_edge(self, parser):
        """Test creates_edge method."""
        correct_cmds, _ = parser.parse("/correct New info")
        remember_cmds, _ = parser.parse("/remember Test")
        
        assert parser.creates_edge(correct_cmds) == "contradicts"
        assert parser.creates_edge(remember_cmds) is None


class TestConvenienceFunction:
    """Test parse_command convenience function."""
    
    def test_parse_command(self):
        """Test parse_command function."""
        cmd, text = parse_command("/remember My note")
        
        assert cmd is not None
        assert cmd.command == "remember"
        assert text == "My note"
    
    def test_parse_command_no_command(self):
        """Test parse_command with no command."""
        cmd, text = parse_command("Hello world")
        
        assert cmd is None
        assert text == "Hello world"


class TestCommandTrainingCollector:
    """Test CommandTrainingCollector functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create a CommandTrainingCollector with temp database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        collector = CommandTrainingCollector(db_path=db_path)
        yield collector
        
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass
    
    @pytest.fixture
    def parser(self):
        return CommandParser()
    
    def test_init(self, collector):
        """Test collector initialization."""
        assert collector is not None
    
    def test_log_command(self, collector, parser):
        """Test logging a command."""
        commands, _ = parser.parse("/remember My sister Sarah")
        
        log_id = collector.log_command(
            user_id="user123",
            message="/remember My sister Sarah",
            command=commands[0],
            system_tier=3,
            system_ego_score=0.6
        )
        
        assert log_id is not None
    
    def test_log_tier_correction(self, collector, parser):
        """Test logging tier correction."""
        commands, _ = parser.parse("/tier2 Important info")
        
        collector.log_command(
            user_id="user123",
            message="/tier2 Important info",
            command=commands[0],
            system_tier=3  # System would have said Tier 3
        )
        
        corrections = collector.get_tier_corrections()
        assert len(corrections) == 1
        assert corrections[0]["system_tier"] == 3
        assert corrections[0]["user_tier"] == 2
    
    def test_get_training_dataset(self, collector, parser):
        """Test getting training dataset."""
        # Log some commands
        cmds1, _ = parser.parse("/remember Test 1")
        cmds2, _ = parser.parse("/tier2 Test 2")
        
        collector.log_command("user1", "/remember Test 1", cmds1[0])
        collector.log_command("user1", "/tier2 Test 2", cmds2[0])
        
        dataset = collector.get_training_dataset()
        
        assert len(dataset) == 2
    
    def test_get_stats(self, collector, parser):
        """Test getting statistics."""
        cmds, _ = parser.parse("/remember Test")
        collector.log_command("user1", "/remember Test", cmds[0])
        
        stats = collector.get_stats()
        
        assert stats["total_commands"] == 1
        assert "remember" in stats["by_command"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

