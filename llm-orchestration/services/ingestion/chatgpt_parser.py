"""
ChatGPT Share Link Parser

Fetches and parses shared ChatGPT conversations from chatgpt.com/share/* URLs.
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx

from .models import BaseParser, ConversationChunk

logger = logging.getLogger(__name__)

# Browser-like defaults; ChatGPT often returns 403 without cookies / real browser TLS.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class ChatGPTShareParser(BaseParser):
    """
    Parser for ChatGPT shared conversation links.
    
    Fetches the conversation JSON from chatgpt.com/backend-api/share/{id}
    and converts it into ConversationChunks.
    """
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    @property
    def source_type(self) -> str:
        return "chatgpt"
    
    async def parse(self, source: str, **kwargs) -> List[ConversationChunk]:
        """
        Parse a ChatGPT share URL.
        
        Args:
            source: ChatGPT share URL (e.g., https://chatgpt.com/share/abc123)
            
        Returns:
            List of ConversationChunk objects (user messages paired with assistant responses)
        """
        share_id = self._extract_share_id(source)
        if not share_id:
            raise ValueError(f"Invalid ChatGPT share URL: {source}")
        
        logger.info(f"Fetching ChatGPT conversation: {share_id}")
        
        conversation_data = await self._fetch_conversation(share_id)
        chunks = self._parse_conversation(conversation_data)
        
        logger.info(f"Parsed {len(chunks)} conversation chunks from ChatGPT share")
        return chunks
    
    def _extract_share_id(self, url: str) -> Optional[str]:
        """Extract the share ID from a ChatGPT share URL."""
        patterns = [
            r'chatgpt\.com/share/([a-zA-Z0-9\-]+)',
            r'chatgpt\.com/c/([a-zA-Z0-9\-]+)',
            r'^([a-zA-Z0-9\-]+)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _browser_headers(self, share_id: str, *, for_html: bool) -> Dict[str, str]:
        share_url = f"https://chatgpt.com/share/{share_id}"
        base: Dict[str, str] = {
            "User-Agent": _UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://chatgpt.com",
            "Referer": share_url,
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
        }
        if for_html:
            base.update(
                {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                }
            )
        else:
            base.update(
                {
                    "Accept": "application/json, text/plain, */*",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin",
                }
            )
        return base

    async def _fetch_conversation(self, share_id: str) -> Dict[str, Any]:
        """Fetch conversation JSON from ChatGPT backend API."""
        api_url = f"https://chatgpt.com/backend-api/share/{share_id}"
        share_url = f"https://chatgpt.com/share/{share_id}"

        # Optional: paste browser Cookie header while logged in at chatgpt.com (dev only).
        extra_cookie = (os.getenv("CHATGPT_SHARE_COOKIE") or "").strip()

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            try:
                # 1) Load public share page — often sets Cloudflare / session cookies for same client.
                await client.get(share_url, headers=self._browser_headers(share_id, for_html=True))
            except httpx.RequestError as e:
                logger.warning("ChatGPT share page prefetch failed (continuing): %s", e)

            api_headers = self._browser_headers(share_id, for_html=False)
            if extra_cookie:
                api_headers["Cookie"] = extra_cookie

            try:
                response = await client.get(api_url, headers=api_headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise ValueError(f"ChatGPT conversation not found: {share_id}") from e
                if e.response.status_code == 403:
                    raise ValueError(self._forbidden_help_message()) from e
                raise ValueError(f"Failed to fetch ChatGPT conversation: {e}") from e
            except httpx.RequestError as e:
                raise ValueError(f"Network error fetching ChatGPT conversation: {e}") from e

    @staticmethod
    def _forbidden_help_message() -> str:
        return (
            "ChatGPT returned 403 Forbidden (share API is often blocked for automated requests). "
            "Try: (1) Set CHATGPT_SHARE_COOKIE in .env to a Cookie header copied from your browser "
            "while logged in at chatgpt.com (same machine / dev only); "
            "(2) Export the chat and use Session JSON or paste text import; "
            "(3) Run import from a network that is not flagged. "
            "See docs/TROUBLESHOOTING.md (ChatGPT share)."
        )
    
    def _parse_conversation(self, data: Dict[str, Any]) -> List[ConversationChunk]:
        """
        Parse the ChatGPT conversation JSON into ConversationChunks.
        
        The JSON structure is a tree of messages in the 'mapping' field.
        Each message has an id, parent, children, and message content.
        """
        title = data.get('title', 'Untitled Conversation')
        create_time = data.get('create_time')
        mapping = data.get('mapping', {})
        
        if not mapping:
            logger.warning("Empty conversation mapping")
            return []
        
        # Build the conversation thread by following parent-child relationships
        messages = self._build_message_thread(mapping)
        
        # Pair user messages with assistant responses
        chunks = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Skip system messages
            if msg['role'] == 'system':
                i += 1
                continue
            
            # User message
            if msg['role'] == 'user':
                user_content = msg['content']
                user_timestamp = msg.get('timestamp')
                
                # Look ahead for assistant response
                assistant_content = None
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    assistant_content = messages[i + 1]['content']
                    i += 2
                else:
                    i += 1
                
                if user_content:
                    chunk = ConversationChunk(
                        content=user_content,
                        context=assistant_content,
                        timestamp=user_timestamp,
                        source_type=self.source_type,
                        metadata={
                            'conversation_title': title,
                            'conversation_create_time': create_time,
                        }
                    )
                    chunks.append(chunk)
            else:
                i += 1
        
        return chunks
    
    def _build_message_thread(self, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a linear thread of messages from the tree structure.
        
        ChatGPT stores conversations as a tree (for branching). We follow
        the main thread by starting from the root and following the first child.
        """
        # Find root node (has no parent or parent is root)
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get('parent')
            if parent is None or parent == node_id:
                root_id = node_id
                break
        
        if not root_id:
            logger.warning("Could not find root node in conversation")
            return []
        
        # Walk the tree depth-first, following the first child
        messages = []
        current_id = root_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id)
            
            if not node:
                break
            
            # Extract message content
            message_data = node.get('message')
            if message_data:
                role = message_data.get('author', {}).get('role')
                content_data = message_data.get('content', {})
                
                # Extract text from parts
                parts = content_data.get('parts', [])
                content = ' '.join(str(part) for part in parts if part).strip()
                
                # Extract timestamp
                create_time = message_data.get('create_time')
                timestamp = None
                if create_time:
                    try:
                        timestamp = datetime.fromtimestamp(create_time).isoformat()
                    except (ValueError, OSError):
                        pass
                
                if content and role:
                    messages.append({
                        'role': role,
                        'content': content,
                        'timestamp': timestamp,
                    })
            
            # Move to first child
            children = node.get('children', [])
            current_id = children[0] if children else None
        
        return messages
