"""
DAPPY Command Parser

Parses explicit user commands from messages.
These commands provide ground truth signals for ML training.

Key Principle:
- Users can signal importance, but CANNOT directly control Tier 1
- Tier 1 is detected by the system and confirmed via Shadow Tier
- User commands are HINTS, not FORCES (except for /tier2, /tier3, /tier4)

Command Categories:
1. memory_control: /remember, /forget, /remember-all
2. tier_control: /tier2, /tier3, /tier4 (NO /tier1 - system only)
3. privacy: /private, /sensitive
4. corrections: /correct, /evolve, /update
5. graph: /link, /unlink
6. retrieval: /search, /context, /recall

Phase 1C Implementation
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of user commands."""
    MEMORY_CONTROL = "memory_control"
    TIER_CONTROL = "tier_control"
    PRIVACY = "privacy"
    CORRECTIONS = "corrections"
    GRAPH = "graph"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


@dataclass
class UserCommand:
    """
    Represents a parsed user command.
    
    Attributes:
        command: The command name (e.g., "remember", "forget")
        category: Command category
        args: Additional arguments (e.g., entity name for /link)
        target_text: The message content after the command
        config: Command configuration from COMMAND_CONFIG
        confidence: Always 1.0 for explicit commands
    """
    command: str
    category: CommandCategory
    args: List[str] = field(default_factory=list)
    target_text: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Explicit commands = 100% confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "category": self.category.value,
            "args": self.args,
            "target_text": self.target_text,
            "config": self.config,
            "confidence": self.confidence
        }


# Command configuration
# Each command has:
# - action: What the system should do
# - category: Which category it belongs to
# - weight: Importance weight for ExplicitImportanceScorer
# - tier: Target tier (if applicable)
# - requires_confirmation: Whether to confirm before action

COMMAND_CONFIG = {
    # ==================== MEMORY CONTROL ====================
    "/remember": {
        "action": "boost_importance",
        "category": CommandCategory.MEMORY_CONTROL,
        "weight": 0.8,
        "max_tier": 2,  # Can boost to Tier 2, NOT Tier 1
        "description": "Mark as important, boost to Tier 2"
    },
    "/forget": {
        "action": "delete",
        "category": CommandCategory.MEMORY_CONTROL,
        "weight": 0.0,
        "requires_confirmation": True,
        "description": "Delete from all memory stores"
    },
    "/remember-all": {
        "action": "bulk_store",
        "category": CommandCategory.MEMORY_CONTROL,
        "weight": 0.7,
        "max_tier": 2,
        "requires_confirmation": True,
        "description": "Store entire conversation"
    },
    
    # ==================== TIER CONTROL ====================
    # NOTE: /tier1 is INTENTIONALLY EXCLUDED
    # Tier 1 is detected by the system and confirmed via Shadow Tier
    # Users lack the epistemological self-awareness to identify core identity
    
    "/tier2": {
        "action": "force_tier",
        "category": CommandCategory.TIER_CONTROL,
        "tier": 2,
        "weight": 0.8,
        "description": "Force to Tier 2 (long-term important)"
    },
    "/tier3": {
        "action": "force_tier",
        "category": CommandCategory.TIER_CONTROL,
        "tier": 3,
        "weight": 0.5,
        "description": "Force to Tier 3 (short-term useful)"
    },
    "/tier4": {
        "action": "force_tier",
        "category": CommandCategory.TIER_CONTROL,
        "tier": 4,
        "weight": 0.3,
        "description": "Force to Tier 4 (temporary/hot)"
    },
    
    # ==================== PRIVACY ====================
    "/private": {
        "action": "mark_sensitive",
        "category": CommandCategory.PRIVACY,
        "weight": 0.85,
        "encrypt": True,
        "pii_flag": True,
        "description": "Mark as private, encrypt storage"
    },
    "/sensitive": {
        "action": "mark_sensitive",
        "category": CommandCategory.PRIVACY,
        "weight": 0.85,
        "encrypt": True,
        "pii_flag": True,
        "description": "Mark as sensitive (alias for /private)"
    },
    
    # ==================== CORRECTIONS ====================
    "/correct": {
        "action": "trigger_correction",
        "category": CommandCategory.CORRECTIONS,
        "weight": 0.75,
        "creates_edge": True,
        "edge_type": "contradicts",
        "description": "Correct a previous memory"
    },
    "/evolve": {
        "action": "create_supersedes_edge",
        "category": CommandCategory.CORRECTIONS,
        "weight": 0.8,
        "creates_edge": True,
        "edge_type": "supersedes",
        "description": "Mark as evolution of previous belief"
    },
    "/update": {
        "action": "update_memory",
        "category": CommandCategory.CORRECTIONS,
        "weight": 0.7,
        "description": "Update existing memory (alias for /evolve)"
    },
    
    # ==================== GRAPH CONTROL ====================
    "/link": {
        "action": "create_edge",
        "category": CommandCategory.GRAPH,
        "weight": 0.6,
        "requires_args": True,
        "description": "Create edge to entity (usage: /link <entity>)"
    },
    "/unlink": {
        "action": "delete_edge",
        "category": CommandCategory.GRAPH,
        "weight": 0.0,
        "requires_args": True,
        "requires_confirmation": True,
        "description": "Remove edge to entity (usage: /unlink <entity>)"
    },
    
    # ==================== RETRIEVAL ====================
    "/search": {
        "action": "search",
        "category": CommandCategory.RETRIEVAL,
        "weight": 0.0,  # Retrieval doesn't affect importance
        "is_query": True,
        "description": "Search memories (usage: /search <query>)"
    },
    "/context": {
        "action": "show_context",
        "category": CommandCategory.RETRIEVAL,
        "weight": 0.0,
        "is_query": True,
        "description": "Show relevant context for current conversation"
    },
    "/recall": {
        "action": "recall",
        "category": CommandCategory.RETRIEVAL,
        "weight": 0.0,
        "is_query": True,
        "description": "Recall specific memory (alias for /search)"
    },
}

# Shorthand aliases for power users
COMMAND_ALIASES = {
    "/r": "/remember",
    "/f": "/forget",
    "/ra": "/remember-all",
    "/t2": "/tier2",
    "/t3": "/tier3",
    "/t4": "/tier4",
    "/p": "/private",
    "/s": "/search",
    "/c": "/context",
    "/l": "/link",
    "/u": "/unlink",
}

# Commands that are SYSTEM ONLY (user cannot use directly)
# If user tries these, gracefully handle by treating as /remember
SYSTEM_ONLY_COMMANDS = [
    "/tier1",
    "/core",
    "/identity",
    "/t1",
]


class CommandParser:
    """
    Parses explicit user commands from messages.
    
    Key principles:
    1. Commands are case-insensitive
    2. Commands must be at the start of the message
    3. Multiple commands can be combined (e.g., /remember /private)
    4. Unknown commands are ignored (message passed through)
    5. /tier1 is SYSTEM ONLY - users cannot directly assign Tier 1
    
    Usage:
        parser = CommandParser()
        commands, remaining_text = parser.parse("/remember My sister Sarah lives in NYC")
        # commands = [UserCommand(command="remember", ...)]
        # remaining_text = "My sister Sarah lives in NYC"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.commands = COMMAND_CONFIG.copy()
        self.aliases = COMMAND_ALIASES.copy()
        self.system_only = SYSTEM_ONLY_COMMANDS.copy()
        
        logger.info(f"✅ CommandParser initialized")
        logger.info(f"   Available commands: {len(self.commands)}")
        logger.info(f"   Aliases: {len(self.aliases)}")
    
    def parse(self, message: str) -> Tuple[List[UserCommand], str]:
        """
        Parse commands from message.
        
        Args:
            message: User message that may contain commands
        
        Returns:
            Tuple of (list of commands, remaining text after commands)
        
        Examples:
            "/remember My name is Sarah"
            → ([UserCommand(command="remember")], "My name is Sarah")
            
            "/remember /private My SSN is 123"
            → ([UserCommand("remember"), UserCommand("private")], "My SSN is 123")
            
            "Hello world"
            → ([], "Hello world")
        """
        if not message or not message.strip():
            return [], message
        
        message = message.strip()
        commands = []
        remaining = message
        
        # Parse commands from the start of the message
        while remaining.startswith('/'):
            # Find the command (up to next space or next /)
            match = re.match(r'^(/\S+)', remaining)
            if not match:
                break
            
            cmd_text = match.group(1).lower()
            
            # Check for system-only commands
            if cmd_text in self.system_only:
                logger.info(f"User tried system-only command {cmd_text}, treating as /remember")
                cmd_text = "/remember"
            
            # Resolve alias
            if cmd_text in self.aliases:
                cmd_text = self.aliases[cmd_text]
            
            # Check if valid command
            if cmd_text not in self.commands:
                # Unknown command - stop parsing, treat rest as text
                break
            
            # Get command config
            cmd_config = self.commands[cmd_text]
            
            # Move past the command
            remaining = remaining[len(match.group(1)):].strip()
            
            # Check if command requires arguments
            args = []
            if cmd_config.get("requires_args"):
                # Extract argument (next word)
                arg_match = re.match(r'^(\S+)', remaining)
                if arg_match:
                    args.append(arg_match.group(1))
                    remaining = remaining[len(arg_match.group(1)):].strip()
            
            # Create command object
            command = UserCommand(
                command=cmd_text.lstrip('/'),
                category=cmd_config["category"],
                args=args,
                target_text="",  # Will be set after all commands parsed
                config=cmd_config,
                confidence=1.0
            )
            commands.append(command)
        
        # Set target_text for all commands
        for cmd in commands:
            cmd.target_text = remaining
        
        return commands, remaining
    
    def parse_single(self, message: str) -> Tuple[Optional[UserCommand], str]:
        """
        Parse a single command from message.
        Convenience method that returns only the first command.
        
        Returns:
            Tuple of (command or None, remaining text)
        """
        commands, remaining = self.parse(message)
        if commands:
            return commands[0], remaining
        return None, remaining
    
    def has_command(self, message: str) -> bool:
        """Check if message starts with a command."""
        if not message or not message.strip():
            return False
        
        message = message.strip().lower()
        
        # Check direct commands
        for cmd in self.commands:
            if message.startswith(cmd):
                return True
        
        # Check aliases
        for alias in self.aliases:
            if message.startswith(alias):
                return True
        
        # Check system-only (these will be converted to /remember)
        for cmd in self.system_only:
            if message.startswith(cmd):
                return True
        
        return False
    
    def get_command_help(self, command: str = None) -> str:
        """
        Get help text for commands.
        
        Args:
            command: Specific command to get help for, or None for all
        
        Returns:
            Help text
        """
        if command:
            cmd = command if command.startswith('/') else f"/{command}"
            if cmd in self.commands:
                config = self.commands[cmd]
                return f"{cmd}: {config.get('description', 'No description')}"
            return f"Unknown command: {command}"
        
        # All commands
        lines = ["Available commands:", ""]
        
        # Group by category
        by_category = {}
        for cmd, config in self.commands.items():
            cat = config["category"].value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append((cmd, config))
        
        for category, cmds in by_category.items():
            lines.append(f"  {category.upper()}:")
            for cmd, config in cmds:
                desc = config.get('description', '')
                lines.append(f"    {cmd}: {desc}")
            lines.append("")
        
        # Aliases
        lines.append("  SHORTCUTS:")
        for alias, target in self.aliases.items():
            lines.append(f"    {alias} → {target}")
        
        return "\n".join(lines)
    
    def get_importance_weight(self, commands: List[UserCommand]) -> float:
        """
        Get combined importance weight from commands.
        
        Takes the maximum weight from all commands.
        """
        if not commands:
            return 0.5  # Default
        
        weights = [cmd.config.get("weight", 0.5) for cmd in commands]
        return max(weights)
    
    def get_target_tier(self, commands: List[UserCommand]) -> Optional[int]:
        """
        Get target tier from commands.
        
        Returns tier if explicitly set, None otherwise.
        """
        for cmd in commands:
            if cmd.config.get("action") == "force_tier":
                return cmd.config.get("tier")
        return None
    
    def requires_confirmation(self, commands: List[UserCommand]) -> bool:
        """Check if any command requires confirmation."""
        return any(cmd.config.get("requires_confirmation", False) for cmd in commands)
    
    def is_retrieval_only(self, commands: List[UserCommand]) -> bool:
        """Check if commands are retrieval-only (no storage)."""
        return all(cmd.config.get("is_query", False) for cmd in commands)
    
    def should_encrypt(self, commands: List[UserCommand]) -> bool:
        """Check if content should be encrypted."""
        return any(cmd.config.get("encrypt", False) for cmd in commands)
    
    def creates_edge(self, commands: List[UserCommand]) -> Optional[str]:
        """Check if commands create an edge, return edge type."""
        for cmd in commands:
            if cmd.config.get("creates_edge"):
                return cmd.config.get("edge_type")
        return None


# Convenience function for quick parsing
def parse_command(message: str) -> Tuple[Optional[UserCommand], str]:
    """
    Quick parse a single command from message.
    
    Usage:
        cmd, text = parse_command("/remember My name is Sarah")
    """
    parser = CommandParser()
    return parser.parse_single(message)


# Export command categories for reference
COMMAND_CATEGORIES = {
    "memory_control": ["/remember", "/forget", "/remember-all"],
    "tier_control": ["/tier2", "/tier3", "/tier4"],  # NO /tier1!
    "privacy": ["/private", "/sensitive"],
    "corrections": ["/correct", "/evolve", "/update"],
    "graph": ["/link", "/unlink"],
    "retrieval": ["/search", "/context", "/recall"],
}

