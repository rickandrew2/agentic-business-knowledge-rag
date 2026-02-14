"""Logging configuration for the RAG application.

Provides structured logging with JSON output and security event tracking.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            JSON string representation of log record
        """
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename',
                               'funcName', 'levelname', 'levelno', 'lineno',
                               'module', 'msecs', 'message', 'pathname', 'process',
                               'processName', 'relativeCreated', 'thread', 'threadName']:
                    log_obj[key] = value
        
        return json.dumps(log_obj)


def setup_logging(logger_name: str = "rag_assistant", level: int = logging.INFO) -> logging.Logger:
    """Set up application logging.
    
    Args:
        logger_name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(console_handler)
    
    return logger


def log_rag_event(logger: logging.Logger, event_type: str, 
                  user_id: str, query: str, result: bool, 
                  extra_data: Dict[str, Any] = None):
    """Log RAG-specific events for audit trail.
    
    Args:
        logger: Logger instance
        event_type: Type of event (QUERY, RETRIEVAL, RESPONSE, ERROR)
        user_id: User identifier
        query: The user query
        result: Success/failure boolean
        extra_data: Additional context to log
    """
    
    log_data = {
        "event_type": event_type,
        "user_id": user_id,
        "query_length": len(query),
        "success": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if extra_data:
        log_data.update(extra_data)
    
    logger.info(f"RAG_EVENT", extra=log_data)
