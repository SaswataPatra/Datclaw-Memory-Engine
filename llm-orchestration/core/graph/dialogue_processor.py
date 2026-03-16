"""
DAPPY Dialogue Processor

Handles conversational/dialogue formats for relation extraction.

Supports:
- LoCoMo dataset format (DATE markers + "X said, 'Y'" format)
- Raw dialogue format (Speaker: message)
- Narrative format (already supported)

Phase 1F Implementation
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single turn in a conversation."""
    speaker: str
    text: str
    date: Optional[datetime] = None
    turn_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDialogue:
    """Processed dialogue with speaker attribution and date context."""
    turns: List[DialogueTurn]
    dates: List[datetime]
    format_type: str  # "locomo", "raw_dialogue", "narrative"
    original_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def speakers(self) -> List[str]:
        """Get unique list of speakers from turns."""
        return list(set(turn.speaker for turn in self.turns))


class DialogueProcessor:
    """
    Pre-processes dialogue formats for relation extraction.
    
    Handles:
    1. LoCoMo format: DATE markers + "X said, 'Y'"
    2. Raw dialogue: "Speaker: message"
    3. Narrative: "X said Y" (already works)
    
    Key features:
    - Date context tracking
    - Speaker attribution
    - Quoted speech extraction
    - Temporal marker resolution ("yesterday", "last week")
    """
    
    # Regex patterns
    DATE_MARKER_PATTERN = r'DATE:\s*(.+?)(?:\n|$)'
    LOCOMO_SAID_PATTERN = r'(\w+)\s+said,?\s*["\'](.+?)["\']'
    RAW_DIALOGUE_PATTERN = r'^(\w+):\s*(.+)$'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dialogue processor."""
        self.config = config or {}
        logger.info("✅ DialogueProcessor initialized")
    
    def is_dialogue_format(self, text: str) -> Tuple[bool, str]:
        """
        Detect if text is in dialogue format.
        
        Returns:
            (is_dialogue, format_type)
            format_type: "locomo", "raw_dialogue", "narrative", or "unknown"
        """
        # Check for LoCoMo format (DATE markers)
        if re.search(self.DATE_MARKER_PATTERN, text, re.IGNORECASE):
            logger.debug("📋 Detected LoCoMo dialogue format (DATE markers)")
            return (True, "locomo")
        
        # Check for "X said, 'Y'" pattern (LoCoMo without DATE markers)
        if re.search(self.LOCOMO_SAID_PATTERN, text):
            logger.debug("📋 Detected LoCoMo dialogue format ('X said' pattern)")
            return (True, "locomo")
        
        # Check for raw dialogue format (Speaker: message)
        # Handle both multi-line and single-line with embedded dialogue
        lines = text.strip().split('\n')
        
        # Also check for inline dialogue (e.g., "Melanie: ... Caroline: ...")
        # Split by common speaker patterns
        inline_speakers = re.findall(r'\b([A-Z][a-z]+):\s+', text)
        
        dialogue_lines = sum(1 for line in lines if re.match(self.RAW_DIALOGUE_PATTERN, line.strip()))
        
        if dialogue_lines >= 2:  # At least 2 dialogue lines (multi-line format)
            logger.debug("📋 Detected raw dialogue format (Speaker: message) - multi-line")
            return (True, "raw_dialogue")
        elif len(inline_speakers) >= 2:  # At least 2 speakers in inline format
            logger.debug(f"📋 Detected raw dialogue format (Speaker: message) - inline with {len(inline_speakers)} speakers: {inline_speakers}")
            return (True, "raw_dialogue")
        
        # Not dialogue format
        return (False, "narrative")
    
    def process(self, text: str) -> ProcessedDialogue:
        """
        Process dialogue text into structured format.
        
        Args:
            text: Raw dialogue text
        
        Returns:
            ProcessedDialogue with speaker attribution and date context
        """
        is_dialogue, format_type = self.is_dialogue_format(text)
        
        if not is_dialogue:
            # Return as single narrative turn
            logger.debug("📋 Processing as narrative (no dialogue detected)")
            return ProcessedDialogue(
                turns=[DialogueTurn(speaker="narrator", text=text, turn_index=0)],
                dates=[],
                format_type="narrative",
                original_text=text
            )
        
        logger.info(f"📋 Processing dialogue (format={format_type})")
        
        if format_type == "locomo":
            return self._process_locomo(text)
        elif format_type == "raw_dialogue":
            return self._process_raw_dialogue(text)
        else:
            # Fallback to narrative
            return ProcessedDialogue(
                turns=[DialogueTurn(speaker="narrator", text=text, turn_index=0)],
                dates=[],
                format_type="narrative",
                original_text=text
            )
    
    def _process_locomo(self, text: str) -> ProcessedDialogue:
        """
        Process LoCoMo format dialogue.
        
        Format:
            DATE: 1:56 pm on 8 May, 2023
            
            CONVERSATION:
            
            Caroline said, "I went to a support group yesterday."
            Melanie said, "That's amazing!"
        """
        turns = []
        dates = []
        current_date = None
        turn_index = 0
        
        # Extract DATE markers
        for match in re.finditer(self.DATE_MARKER_PATTERN, text, re.IGNORECASE):
            date_str = match.group(1).strip()
            try:
                # Parse date (simplified - you may need more robust parsing)
                current_date = self._parse_date(date_str)
                dates.append(current_date)
                logger.debug(f"📅 Extracted DATE: {current_date}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to parse date '{date_str}': {e}")
        
        # Extract dialogue turns ("X said, 'Y'")
        for match in re.finditer(self.LOCOMO_SAID_PATTERN, text):
            speaker = match.group(1).strip()
            speech = match.group(2).strip()
            
            turn = DialogueTurn(
                speaker=speaker,
                text=speech,
                date=current_date,
                turn_index=turn_index
            )
            turns.append(turn)
            turn_index += 1
            
            logger.debug(f"💬 Turn {turn_index}: {speaker} → '{speech[:50]}...'")
        
        logger.info(f"✅ Processed LoCoMo dialogue: {len(turns)} turns, {len(dates)} dates")
        
        return ProcessedDialogue(
            turns=turns,
            dates=dates,
            format_type="locomo",
            original_text=text,
            metadata={"date_markers": len(dates)}
        )
    
    def _process_raw_dialogue(self, text: str) -> ProcessedDialogue:
        """
        Process raw dialogue format.
        
        Formats supported:
        1. Multi-line:
            Melanie: Music's amazing, isn't it?
            Caroline: I 100% agree, Mel.
        
        2. Inline (single line):
            Melanie: Music's amazing, isn't it? Caroline: I 100% agree, Mel.
        """
        turns = []
        turn_index = 0
        
        # First, try to extract preamble (text before first speaker)
        # e.g., "here is a conversation i want you to remember :- Melanie: ..."
        first_speaker_match = re.search(r'\b([A-Z][a-z]+):\s+', text)
        preamble = ""
        dialogue_text = text
        
        if first_speaker_match:
            preamble_end = first_speaker_match.start()
            if preamble_end > 0:
                preamble = text[:preamble_end].strip()
                dialogue_text = text[preamble_end:]
                logger.debug(f"📝 Extracted preamble: '{preamble[:50]}...'")
        
        # Try multi-line format first
        lines = dialogue_text.strip().split('\n')
        multi_line_turns = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(self.RAW_DIALOGUE_PATTERN, line)
            if match:
                speaker = match.group(1).strip()
                message = match.group(2).strip()
                multi_line_turns.append((speaker, message))
        
        # If multi-line worked, use it
        if len(multi_line_turns) >= 2:
            for speaker, message in multi_line_turns:
                turn = DialogueTurn(
                    speaker=speaker,
                    text=message,
                    turn_index=turn_index
                )
                turns.append(turn)
                turn_index += 1
                logger.debug(f"💬 Turn {turn_index}: {speaker} → '{message[:50]}...'")
        else:
            # Try inline format: split by speaker patterns
            # Pattern: "Speaker: text" followed by another "Speaker: text"
            speaker_splits = re.split(r'\b([A-Z][a-z]+):\s+', dialogue_text)
            
            # speaker_splits will be: ['', 'Melanie', 'text...', 'Caroline', 'text...']
            # Pair up speakers with their text
            for i in range(1, len(speaker_splits), 2):
                if i + 1 < len(speaker_splits):
                    speaker = speaker_splits[i].strip()
                    message = speaker_splits[i + 1].strip()
                    
                    # Remove trailing speaker name if present (for next turn)
                    # e.g., "Music's amazing, isn't it? Caroline:" → "Music's amazing, isn't it?"
                    message = re.sub(r'\s+[A-Z][a-z]+:\s*$', '', message)
                    
                    if message:  # Only add if there's actual content
                        turn = DialogueTurn(
                            speaker=speaker,
                            text=message,
                            turn_index=turn_index
                        )
                        turns.append(turn)
                        turn_index += 1
                        logger.debug(f"💬 Turn {turn_index} (inline): {speaker} → '{message[:50]}...'")
        
        logger.info(f"✅ Processed raw dialogue: {len(turns)} turns")
        if preamble:
            logger.info(f"   → Preamble detected and removed: '{preamble[:50]}...'")
        
        return ProcessedDialogue(
            turns=turns,
            dates=[],
            format_type="raw_dialogue",
            original_text=text,
            metadata={"preamble": preamble} if preamble else {}
        )
    
    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse date string from LoCoMo format.
        
        Examples:
            "1:56 pm on 8 May, 2023"
            "8 May, 2023"
            "May 8, 2023"
        """
        # Try various date formats
        formats = [
            "%I:%M %p on %d %B, %Y",  # "1:56 pm on 8 May, 2023"
            "%d %B, %Y",              # "8 May, 2023"
            "%B %d, %Y",              # "May 8, 2023"
            "%Y-%m-%d",               # "2023-05-08"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Fallback: return current date
        logger.warning(f"⚠️  Could not parse date '{date_str}', using current date")
        return datetime.utcnow()
    
    def resolve_temporal_references(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> str:
        """
        Resolve temporal references like "yesterday", "last week".
        
        Args:
            text: Text with temporal references
            reference_date: Reference date for resolution (defaults to now)
        
        Returns:
            Text with resolved dates
        """
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        # Map temporal references to timedelta
        temporal_map = {
            "yesterday": timedelta(days=-1),
            "today": timedelta(days=0),
            "tomorrow": timedelta(days=1),
            "last week": timedelta(weeks=-1),
            "next week": timedelta(weeks=1),
            "last month": timedelta(days=-30),
            "next month": timedelta(days=30),
        }
        
        resolved_text = text
        for ref, delta in temporal_map.items():
            if ref in text.lower():
                resolved_date = reference_date + delta
                date_str = resolved_date.strftime("%B %d, %Y")
                # Replace with explicit date
                resolved_text = re.sub(
                    ref,
                    f"{ref} ({date_str})",
                    resolved_text,
                    flags=re.IGNORECASE
                )
                logger.debug(f"📅 Resolved '{ref}' → {date_str}")
        
        return resolved_text
    
    def to_narrative(self, dialogue: ProcessedDialogue) -> str:
        """
        Convert dialogue to narrative format for easier processing.
        
        Input:
            Caroline said, "I went to a support group."
        
        Output:
            Caroline went to a support group.
        """
        narrative_parts = []
        
        for turn in dialogue.turns:
            # Convert quoted speech to narrative
            # "I went" → "Caroline went"
            narrative = self._convert_to_third_person(turn.text, turn.speaker)
            
            # Add date context if available
            if turn.date:
                date_str = turn.date.strftime("%B %d, %Y")
                narrative = f"On {date_str}, {narrative}"
            
            narrative_parts.append(narrative)
        
        return " ".join(narrative_parts)
    
    def to_speaker_attributed_text(self, turn: DialogueTurn) -> str:
        """
        Return the original dialogue text WITHOUT pronoun conversion.
        
        The speaker context is passed via metadata, and pronouns (I/me/my)
        will be resolved to the speaker during entity resolution.
        
        Input:
            DialogueTurn(speaker="Caroline", text="I 100% agree, Mel.")
        
        Output:
            "I 100% agree, Mel."  (original text, no conversion)
        
        Why no conversion?
        - Preserves original text for provenance
        - Avoids complex verb conjugation
        - Dependency parser handles pronouns naturally
        - Entity resolver maps "I" → speaker using metadata
        """
        # Return original text - no pronoun conversion needed!
        return turn.text
    
    def extract_aliases(self, dialogue: ProcessedDialogue) -> Dict[str, List[str]]:
        """
        Extract nickname/alias mappings from dialogue.
        
        Examples:
            "I 100% agree, Mel" → {"Melanie": ["Mel"]}
            "Hey Sarah, ..." → {"Sarah": ["Sarah"]}
        
        Returns:
            Dict mapping canonical names to list of aliases
        """
        aliases = {}
        
        # Common nickname patterns
        # Look for direct address: "Hey X", "X,", "Thanks X"
        for turn in dialogue.turns:
            # Find all capitalized words that might be names
            potential_names = re.findall(r'\b([A-Z][a-z]+)\b', turn.text)
            
            for name in potential_names:
                # Skip the speaker themselves
                if name.lower() == turn.speaker.lower():
                    continue
                
                # Check if this might be a nickname (shorter than 5 chars, or ends in common suffixes)
                is_nickname = (
                    len(name) <= 4 or  # Short names like "Mel", "Sam", "Joe"
                    name.endswith(('ie', 'y'))  # "Katie", "Jimmy"
                )
                
                if is_nickname:
                    # Try to find the full name in other speakers
                    for other_speaker in dialogue.speakers:
                        if other_speaker.lower().startswith(name.lower()):
                            # Found a match! "Mel" → "Melanie"
                            if other_speaker not in aliases:
                                aliases[other_speaker] = []
                            if name not in aliases[other_speaker]:
                                aliases[other_speaker].append(name)
                                logger.debug(f"🏷️  Detected alias: {name} → {other_speaker}")
        
        return aliases
    
    def _convert_to_third_person(self, text: str, speaker: str) -> str:
        """
        Convert first-person speech to third-person narrative.
        
        Examples:
            "I went" → "Caroline went"
            "I'm happy" → "Caroline is happy"
            "I 100% agree" → "Caroline 100% agrees"
        """
        
        
        # More comprehensive pronoun replacement
        # Order matters! Do contractions first, then pronouns
        replacements = [
            # Contractions (must come first)
            (r"\bI'm\b", f"{speaker} is"),
            (r"\bI've\b", f"{speaker} has"),
            (r"\bI'll\b", f"{speaker} will"),
            (r"\bI'd\b", f"{speaker} would"),
            
            # Pronouns as subjects
            (r'\bI\b', speaker),
            
            # Pronouns as objects
            (r'\bme\b', speaker),
            
            # Possessive
            (r'\bmy\b', f"{speaker}'s"),
            (r'\bmine\b', f"{speaker}'s"),
            
            # Reflexive
            (r'\bmyself\b', f"{speaker}"),
        ]
        
        narrative = text
        for pattern, replacement in replacements:
            narrative = re.sub(pattern, replacement, narrative)
        
        # Fix verb conjugation for third person
        # Common patterns: "Caroline agree" → "Caroline agrees"
        verb_fixes = [
            (rf'\b{re.escape(speaker)}\s+(agree)\b', f'{speaker} agrees'),
            (rf'\b{re.escape(speaker)}\s+(love)\b', f'{speaker} loves'),
            (rf'\b{re.escape(speaker)}\s+(like)\b', f'{speaker} likes'),
            (rf'\b{re.escape(speaker)}\s+(want)\b', f'{speaker} wants'),
            (rf'\b{re.escape(speaker)}\s+(need)\b', f'{speaker} needs'),
            (rf'\b{re.escape(speaker)}\s+(think)\b', f'{speaker} thinks'),
            (rf'\b{re.escape(speaker)}\s+(believe)\b', f'{speaker} believes'),
            (rf'\b{re.escape(speaker)}\s+(feel)\b', f'{speaker} feels'),
            (rf'\b{re.escape(speaker)}\s+(know)\b', f'{speaker} knows'),
            (rf'\b{re.escape(speaker)}\s+(understand)\b', f'{speaker} understands'),
            (rf'\b{re.escape(speaker)}\s+(work)\b', f'{speaker} works'),
            (rf'\b{re.escape(speaker)}\s+(live)\b', f'{speaker} lives'),
            (rf'\b{re.escape(speaker)}\s+(study)\b', f'{speaker} studies'),
            (rf'\b{re.escape(speaker)}\s+(play)\b', f'{speaker} plays'),
            (rf'\b{re.escape(speaker)}\s+(enjoy)\b', f'{speaker} enjoys'),
        ]
        
        for pattern, replacement in verb_fixes:
            narrative = re.sub(pattern, replacement, narrative, flags=re.IGNORECASE)
        
        return narrative

