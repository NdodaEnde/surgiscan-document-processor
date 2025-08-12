"""
Centralized logging configuration for the document processing microservice.
Provides structured logging with different formatters for development and production.
"""

import logging
import logging.config
import sys
import json
from datetime import datetime
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

from .config import settings


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'document-processor'
        log_record['version'] = settings.VERSION
        
        # Add level name
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


def setup_logging() -> None:
    """Setup logging configuration based on environment"""
    
    # Determine log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Different configs for development vs production
    if settings.DEBUG:
        # Development: Human-readable format
        logging.basicConfig(
            level=log_level,
            format=settings.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        # Production: Structured JSON logging
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'structured': {
                    '()': StructuredFormatter,
                    'format': '%(timestamp)s %(level)s %(name)s %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'stream': sys.stdout,
                    'formatter': 'structured',
                    'level': log_level
                }
            },
            'root': {
                'level': log_level,
                'handlers': ['console']
            },
            'loggers': {
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': ['console'],
                    'propagate': False
                },
                'uvicorn.error': {
                    'level': 'INFO',
                    'handlers': ['console'],
                    'propagate': False
                },
                'uvicorn.access': {
                    'level': 'INFO',
                    'handlers': ['console'],
                    'propagate': False
                }
            }
        }
        logging.config.dictConfig(config)
    
    # Setup Sentry if configured
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
            from sentry_sdk.integrations.asyncio import AsyncioIntegration
            
            sentry_logging = LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors as events
            )
            
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                integrations=[sentry_logging, AsyncioIntegration()],
                environment="production" if not settings.DEBUG else "development",
                traces_sample_rate=0.1,
                release=settings.VERSION
            )
            
            logging.info("Sentry error tracking initialized")
            
        except ImportError:
            logging.warning("Sentry SDK not installed, error tracking disabled")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


# Context manager for request logging
class RequestLogger:
    """Context manager for logging request processing"""
    
    def __init__(self, logger: logging.Logger, request_id: str, operation: str):
        self.logger = logger
        self.request_id = request_id
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"Starting {self.operation}",
            extra={
                'request_id': self.request_id,
                'operation': self.operation,
                'status': 'started'
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                extra={
                    'request_id': self.request_id,
                    'operation': self.operation,
                    'status': 'completed',
                    'duration_seconds': duration
                }
            )
        else:
            self.logger.error(
                f"Failed {self.operation}: {exc_val}",
                extra={
                    'request_id': self.request_id,
                    'operation': self.operation,
                    'status': 'failed',
                    'duration_seconds': duration,
                    'error': str(exc_val),
                    'error_type': exc_type.__name__ if exc_type else None
                }
            )


# Utility functions for common logging patterns
def log_processing_start(logger: logging.Logger, document_id: str, filename: str, mode: str) -> None:
    """Log the start of document processing"""
    logger.info(
        "Starting document processing",
        extra={
            'document_id': document_id,
            'document_filename': filename,
            'processing_mode': mode,
            'event': 'processing_started'
        }
    )


def log_processing_complete(logger: logging.Logger, document_id: str, result: Dict[str, Any]) -> None:
    """Log successful document processing completion"""
    logger.info(
        "Document processing completed",
        extra={
            'document_id': document_id,
            'document_types_found': len(result.get('extracted_data', {})),
            'total_fields': result.get('processing_summary', {}).get('total_fields_extracted', 0),
            'processing_time': result.get('processing_summary', {}).get('processing_time', 0),
            'event': 'processing_completed'
        }
    )


def log_processing_error(logger: logging.Logger, document_id: str, error: Exception) -> None:
    """Log document processing errors"""
    logger.error(
        "Document processing failed",
        extra={
            'document_id': document_id,
            'error': str(error),
            'error_type': type(error).__name__,
            'event': 'processing_failed'
        },
        exc_info=True
    )


def log_api_request(logger: logging.Logger, method: str, path: str, request_id: str, **kwargs) -> None:
    """Log API request details"""
    logger.info(
        f"{method} {path}",
        extra={
            'method': method,
            'path': path,
            'request_id': request_id,
            'event': 'api_request',
            **kwargs
        }
    )


def log_database_operation(logger: logging.Logger, operation: str, collection: str, **kwargs) -> None:
    """Log database operations"""
    logger.info(
        f"Database {operation}",
        extra={
            'database_operation': operation,
            'collection': collection,
            'event': 'database_operation',
            **kwargs
        }
    )