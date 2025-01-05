import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional


# Configure logging
class JSONFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        # Base log record attributes
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if they exist
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra dict if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logger(name: str = "wellplanned") -> logging.Logger:
    """Set up and configure the application logger"""
    logger = logging.getLogger(name)

    # Set log level based on environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level))

    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    return logger


# Create the logger instance
logger = setup_logger()


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter to add context to log records"""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        # Ensure extra dict exists
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Add request_id if present in context
        if hasattr(self, "request_id"):
            kwargs["extra"]["request_id"] = self.request_id

        # Add user_id if present in context
        if hasattr(self, "user_id"):
            kwargs["extra"]["user_id"] = self.user_id

        return msg, kwargs


def get_logger(
    request_id: Optional[str] = None, user_id: Optional[str] = None
) -> LoggerAdapter:
    """Get a logger instance with optional request and user context"""
    adapter = LoggerAdapter(logger, {})
    if request_id:
        adapter.request_id = request_id
    if user_id:
        adapter.user_id = user_id
    return adapter


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        log = get_logger(request_id=request_id)

        try:
            log.info(
                f"Starting {func.__name__}",
                extra={"extra_data": {"args": str(args), "kwargs": str(kwargs)}},
            )
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            log.info(
                f"Completed {func.__name__}",
                extra={"extra_data": {"duration_ms": duration_ms}},
            )
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log.error(
                f"Error in {func.__name__}",
                exc_info=True,
                extra={"extra_data": {"duration_ms": duration_ms, "error": str(e)}},
            )
            raise

    return wrapper


def log_execution_time_async(func: Callable) -> Callable:
    """Decorator to log async function execution time"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        log = get_logger(request_id=request_id)

        try:
            log.info(
                f"Starting {func.__name__}",
                extra={"extra_data": {"args": str(args), "kwargs": str(kwargs)}},
            )
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            log.info(
                f"Completed {func.__name__}",
                extra={"extra_data": {"duration_ms": duration_ms}},
            )
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log.error(
                f"Error in {func.__name__}",
                exc_info=True,
                extra={"extra_data": {"duration_ms": duration_ms, "error": str(e)}},
            )
            raise

    return wrapper
