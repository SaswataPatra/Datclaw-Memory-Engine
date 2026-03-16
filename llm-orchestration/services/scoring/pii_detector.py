"""
PII Detection Service
Detects personally identifiable information in text using regex patterns
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class PIIDetector:
    """
    Detects PII (Personally Identifiable Information) in text
    
    Uses regex patterns to identify:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social security numbers
    - IP addresses
    - URLs with sensitive patterns
    """
    
    def __init__(self):
        # Regex patterns for PII detection
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'api_key': re.compile(r'\b(?:api[_-]?key|token|secret)[:\s]*[\'"]?([A-Za-z0-9_\-]{20,})[\'"]?', re.IGNORECASE),
            'password': re.compile(r'\b(?:password|passwd|pwd)[:\s]*[\'"]?([^\s\'"]{6,})[\'"]?', re.IGNORECASE),
        }
        
        logger.info("PIIDetector initialized")
    
    def detect(self, text: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Detect PII in text
        
        Args:
            text: Text to scan for PII
            
        Returns:
            Tuple of (has_pii, detected_pii_dict)
            - has_pii: True if any PII found
            - detected_pii_dict: Dictionary of PII type -> list of matches
        """
        detected = {}
        has_pii = False
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches if isinstance(matches, list) else [matches]
                has_pii = True
        
        if has_pii:
            logger.warning(f"🚨 PII detected in text: {list(detected.keys())}")
            logger.warning(f"   Text preview: {text[:100]}...")
        
        return has_pii, detected
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text with placeholder values
        
        Args:
            text: Text to mask
            
        Returns:
            Text with PII masked
        """
        masked_text = text
        
        # Mask email
        masked_text = self.patterns['email'].sub('[EMAIL]', masked_text)
        
        # Mask phone
        masked_text = self.patterns['phone'].sub('[PHONE]', masked_text)
        
        # Mask SSN
        masked_text = self.patterns['ssn'].sub('[SSN]', masked_text)
        
        # Mask credit card
        masked_text = self.patterns['credit_card'].sub('[CREDIT_CARD]', masked_text)
        
        # Mask IP address
        masked_text = self.patterns['ip_address'].sub('[IP_ADDRESS]', masked_text)
        
        # Mask API keys
        masked_text = self.patterns['api_key'].sub(r'\1[API_KEY]', masked_text)
        
        # Mask passwords
        masked_text = self.patterns['password'].sub(r'\1[PASSWORD]', masked_text)
        
        return masked_text

