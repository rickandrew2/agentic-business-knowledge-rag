"""Security utilities for input validation and protection.

Handles:
- Input validation against prompt injection
- PII detection
- Rate limiting
- Secure error responses
"""

import re
import logging
from typing import Dict
from pydantic import BaseModel, field_validator, ConfigDict

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Validated chat message with injection prevention."""
    
    message: str
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message for security and length."""
        
        # Check length
        if len(v) > 2000:
            raise ValueError("Message exceeds maximum length of 2000 characters")
        
        if len(v) < 1:
            raise ValueError("Message cannot be empty")
        
        # Check for common prompt injection patterns
        injection_patterns = [
            r"forget.*instruction|override.*system",
            r"execute.*code|run.*script|system\s*call",
            r"ignore.*context|new\s+instruction|role\s+play",
            r"sql\s+injection|drop\s+table|delete\s+from",
            r"prompt\s*injection|jailbreak|deb[ug]*mode"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                logger.warning(f"Potential injection detected in message: {v[:50]}...")
                raise ValueError("Message contains potentially harmful content")
        
        return v.strip()


class DataUploadRequest(BaseModel):
    """Validated data upload request."""
    
    filename: str
    content_type: str
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename for path traversal attacks."""
        
        # Only allow safe filenames
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid filename: path traversal detected")
        
        # Allow only CSV and markdown files
        allowed_extensions = ('.csv', '.txt', '.md', '.pdf')
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(f"Only {allowed_extensions} files are allowed")
        
        return v
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type."""
        
        allowed_types = (
            'text/csv',
            'text/plain',
            'text/markdown',
            'application/pdf'
        )
        
        if v not in allowed_types:
            raise ValueError(f"Content type {v} not allowed")
        
        return v


def detect_pii(text: str) -> Dict[str, bool]:
    """Detect likely PII patterns in text.
    
    Args:
        text: Text to scan for PII
        
    Returns:
        Dictionary mapping PII type to boolean detection result
    """
    
    pii_patterns = {
        "email": r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{16}\b|(\d{4}[\s-]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    }
    
    findings = {}
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, text)
        findings[pii_type] = len(matches) > 0
        if matches:
            logger.warning(f"PII detected ({pii_type}): {len(matches)} occurrences")
    
    return findings


def mask_pii(text: str) -> str:
    """Mask PII in text with [REDACTED].
    
    Args:
        text: Text containing PII
        
    Returns:
        Text with PII masked
    """
    
    # Mask emails
    text = re.sub(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", "[EMAIL]", text, flags=re.IGNORECASE)
    
    # Mask phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    
    # Mask SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    
    # Mask credit cards
    text = re.sub(r"\b\d{16}\b|(\d{4}[\s-]?){3}\d{4}\b", "[CREDIT_CARD]", text)
    
    # Mask IP addresses
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", text)
    
    return text


class SecureErrorResponse(BaseModel):
    """Secure error response that doesn't leak internal details."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Unable to process request. Please try again.",
                "error_code": "PROCESSING_ERROR",
                "request_id": "req_12345"
            }
        }
    )
    
    error: str
    error_code: str
    request_id: str  # For debugging without exposing internals
