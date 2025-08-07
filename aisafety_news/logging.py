"""Structured JSON logging configuration for AI Safety Newsletter Agent."""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog import processors, stdlib

from .config import get_settings


def setup_logging(
    log_level: str | None = None,
    json_logging: bool | None = None,
    log_file: Path | None = None
) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_logging: Enable JSON formatting
        log_file: Optional log file path
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    log_level = log_level or settings.log_level
    json_logging = json_logging if json_logging is not None else settings.json_logging

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Processor chain for structured logging
    processors_list = [
        # Add log level and timestamp
        stdlib.filter_by_level,
        stdlib.add_logger_name,
        stdlib.add_log_level,
        stdlib.PositionalArgumentsFormatter(),
        processors.TimeStamper(fmt="iso"),
        processors.StackInfoRenderer(),
        processors.format_exc_info,
    ]

    if json_logging:
        # JSON output for production
        processors_list.append(
            processors.JSONRenderer(serializer=json.dumps, indent=None)
        )
    else:
        # Human-readable output for development
        processors_list.extend([
            processors.add_log_level,
            processors.CallsiteParameterAdder(
                parameters=[processors.CallsiteParameter.FILENAME,
                           processors.CallsiteParameter.LINENO]
            ),
            stdlib.ProcessorFormatter.wrap_for_formatter,
        ])

    # Configure structlog
    structlog.configure(
        processors=processors_list,
        wrapper_class=stdlib.BoundLogger,
        logger_factory=stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        if json_logging:
            file_formatter = logging.Formatter('%(message)s')
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> dict[str, Any]:
    """Create a standardized log entry for function calls.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function arguments to log
        
    Returns:
        Structured log data
    """
    return {
        "event": "function_call",
        "function": func_name,
        "args": kwargs
    }


def log_api_request(
    method: str,
    url: str,
    status_code: int | None = None,
    response_time: float | None = None,
    **kwargs: Any
) -> dict[str, Any]:
    """Create a standardized log entry for API requests.
    
    Args:
        method: HTTP method
        url: Request URL
        status_code: Response status code
        response_time: Response time in seconds
        **kwargs: Additional request data
        
    Returns:
        Structured log data
    """
    log_data = {
        "event": "api_request",
        "method": method,
        "url": url,
        **kwargs
    }

    if status_code is not None:
        log_data["status_code"] = status_code

    if response_time is not None:
        log_data["response_time"] = response_time

    return log_data


def log_processing_stage(
    stage: str,
    input_count: int,
    output_count: int,
    duration: float | None = None,
    **kwargs: Any
) -> dict[str, Any]:
    """Create a standardized log entry for processing stages.
    
    Args:
        stage: Processing stage name
        input_count: Number of input items
        output_count: Number of output items
        duration: Processing duration in seconds
        **kwargs: Additional processing data
        
    Returns:
        Structured log data
    """
    log_data = {
        "event": "processing_stage",
        "stage": stage,
        "input_count": input_count,
        "output_count": output_count,
        **kwargs
    }

    if duration is not None:
        log_data["duration"] = duration

    return log_data


def log_error(
    error: Exception,
    context: str | None = None,
    **kwargs: Any
) -> dict[str, Any]:
    """Create a standardized log entry for errors.
    
    Args:
        error: Exception that occurred
        context: Additional context about the error
        **kwargs: Additional error data
        
    Returns:
        Structured log data
    """
    log_data = {
        "event": "error",
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        **kwargs
    }

    if context:
        log_data["context"] = context

    return log_data


# Performance monitoring helpers
class PerformanceLogger:
    """Context manager for logging performance metrics."""

    def __init__(self, operation: str, logger: structlog.stdlib.BoundLogger):
        self.operation = operation
        self.logger = logger
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceLogger":
        import time
        self.start_time = time.time()
        self.logger.info("operation_started", operation=self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import time
        if self.start_time is not None:
            duration = time.time() - self.start_time
            if exc_type is None:
                self.logger.info(
                    "operation_completed",
                    operation=self.operation,
                    duration=duration
                )
            else:
                self.logger.error(
                    "operation_failed",
                    operation=self.operation,
                    duration=duration,
                    error_type=exc_type.__name__ if exc_type else None,
                    error_message=str(exc_val) if exc_val else None
                )


# Initialize logging on module import
setup_logging()
