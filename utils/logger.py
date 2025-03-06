"""
Advanced Logging System for Legion AGI

This module provides a centralized, configurable logging system for the Legion AGI framework.
It extends loguru with additional functionality specific to AGI development, including:
- Multi-destination logging (file, console, network)
- Structured logging for machine analysis
- Context-aware logging for tracing agent interactions
- Performance monitoring
"""

import os
import sys
import time
import json
import socket
import threading
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime

from loguru import logger

from legion_agi.config import (
    LOG_LEVEL, 
    LOG_FORMAT, 
    LOG_FILE, 
    LOG_ROTATION,
    LOG_DIR
)


class LogContext:
    """Context manager for tracking nested execution contexts in logs."""
    
    _thread_local = threading.local()
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize log context.
        
        Args:
            name: Context name
            **kwargs: Additional context attributes
        """
        self.name = name
        self.attrs = kwargs
        self.start_time = None
        
    def __enter__(self):
        # Get or initialize context stack for this thread
        if not hasattr(self._thread_local, 'context_stack'):
            self._thread_local.context_stack = []
            
        # Push new context to stack
        self._thread_local.context_stack.append(self)
        self.start_time = time.time()
        
        # Log context entry
        logger.bind(
            context=self.name,
            context_level=len(self._thread_local.context_stack),
            **self.attrs
        ).debug(f"Entering context: {self.name}")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Log context exit
        elapsed = time.time() - self.start_time
        
        log_func = logger.exception if exc_type else logger.debug
        log_func = logger.bind(
            context=self.name,
            context_level=len(self._thread_local.context_stack),
            elapsed_ms=int(elapsed * 1000),
            **self.attrs
        )
        
        if exc_type:
            log_func(f"Exiting context with error: {self.name} ({exc_val})")
        else:
            log_func(f"Exiting context: {self.name}")
            
        # Pop context from stack
        if hasattr(self._thread_local, 'context_stack') and self._thread_local.context_stack:
            self._thread_local.context_stack.pop()
            
    @classmethod
    def current(cls) -> Optional[Dict[str, Any]]:
        """
        Get the current context information.
        
        Returns:
            Current context info or None if no active context
        """
        if not hasattr(cls._thread_local, 'context_stack') or not cls._thread_local.context_stack:
            return None
            
        contexts = []
        for ctx in cls._thread_local.context_stack:
            contexts.append({
                "name": ctx.name,
                "attrs": ctx.attrs,
                "elapsed_ms": int((time.time() - ctx.start_time) * 1000)
            })
            
        return {
            "current": contexts[-1]["name"],
            "depth": len(contexts),
            "path": "/".join(ctx["name"] for ctx in contexts),
            "contexts": contexts
        }


class PerformanceTimer:
    """
    Timer class for performance monitoring and logging.
    Usage: 
        with PerformanceTimer("operation_name", threshold_ms=100):
            # code to time
    """
    
    def __init__(self, operation: str, threshold_ms: int = 0, **kwargs):
        """
        Initialize performance timer.
        
        Args:
            operation: Operation name to log
            threshold_ms: Only log if execution exceeds this threshold (0 to always log)
            **kwargs: Additional context attributes to log
        """
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.attrs = kwargs
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_ms = int(elapsed_time * 1000)
        
        # Only log if exceeded threshold (or if exception occurred)
        if elapsed_ms >= self.threshold_ms or exc_type:
            log_func = logger.warning if elapsed_ms > 5 * self.threshold_ms else logger.info
            
            # If threshold is 0 (always log) and fast operation, use debug level
            if self.threshold_ms == 0 and elapsed_ms < 100:
                log_func = logger.debug
                
            # Get current context if available
            context_info = LogContext.current()
            context_attrs = {}
            
            if context_info:
                context_attrs = {
                    "context": context_info["current"],
                    "context_path": context_info["path"]
                }
                
            # Log performance data
            log_func = log_func.bind(
                operation=self.operation,
                elapsed_ms=elapsed_ms,
                **context_attrs,
                **self.attrs
            )
            
            if exc_type:
                log_func(f"Operation failed: {self.operation} - {elapsed_ms}ms - {exc_val}")
            else:
                log_func(f"Operation timing: {self.operation} - {elapsed_ms}ms")


class AgentLogger:
    """
    Specialized logger for agent activities with context tracking.
    Provides agent-specific logging with contextual information.
    """
    
    def __init__(self, agent_id: str, agent_name: str, agent_role: str):
        """
        Initialize agent logger.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            agent_role: Agent role
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_role = agent_role
        self._logger = logger.bind(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role
        )
        
    def debug(self, message: str, **kwargs):
        """Log debug message with agent context."""
        self._logger.bind(**kwargs).debug(message)
        
    def info(self, message: str, **kwargs):
        """Log info message with agent context."""
        self._logger.bind(**kwargs).info(message)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with agent context."""
        self._logger.bind(**kwargs).warning(message)
        
    def error(self, message: str, **kwargs):
        """Log error message with agent context."""
        self._logger.bind(**kwargs).error(message)
        
    def critical(self, message: str, **kwargs):
        """Log critical message with agent context."""
        self._logger.bind(**kwargs).critical(message)
        
    def activity(self, activity_type: str, data: Dict[str, Any]) -> None:
        """
        Log agent activity with structured data.
        
        Args:
            activity_type: Type of activity (e.g., "reasoning", "collaboration")
            data: Activity data
        """
        self._logger.bind(
            activity_type=activity_type,
            timestamp=datetime.now().isoformat(),
            **data
        ).info(f"Agent activity: {activity_type}")


class NetworkLogSink:
    """
    Custom loguru sink that forwards logs to a network service.
    Useful for distributed logging or monitoring.
    """
    
    def __init__(self, host: str, port: int):
        """
        Initialize network log sink.
        
        Args:
            host: Target host
            port: Target port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.reconnect_interval = 5  # seconds
        self.last_reconnect_attempt = 0
        
    def __call__(self, message):
        """Process and forward log message."""
        # Extract record dict from message
        record = message.record
        
        # Convert to JSON-serializable format
        log_entry = {
            "time": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "function": record["function"],
            "file": record["file"].name,
            "line": record["line"]
        }
        
        # Add extra attributes
        for key, value in record["extra"].items():
            log_entry[key] = value
            
        # Send over network
        self._send_log(log_entry)
        
    def _send_log(self, log_entry: Dict[str, Any]) -> None:
        """Send log entry to network target."""
        # Check connection state
        if not self.connected:
            # Avoid frequent reconnection attempts
            current_time = time.time()
            if current_time - self.last_reconnect_attempt < self.reconnect_interval:
                return
                
            self.last_reconnect_attempt = current_time
            self._connect()
            
        # Send if connected
        if self.connected:
            try:
                serialized = json.dumps(log_entry) + "\n"
                self.socket.sendall(serialized.encode('utf-8'))
            except Exception:
                self.connected = False
                self.socket = None
                
    def _connect(self) -> None:
        """Establish connection to log server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
        except Exception:
            self.connected = False
            self.socket = None


def setup_logging(
    console_level: str = LOG_LEVEL,
    file_level: str = LOG_LEVEL,
    file_path: str = LOG_FILE,
    format_str: str = LOG_FORMAT,
    rotation: str = LOG_ROTATION,
    network_logging: bool = False,
    network_host: str = "localhost",
    network_port: int = 9020
) -> None:
    """
    Setup logging for the Legion AGI system.
    
    Args:
        console_level: Log level for console output
        file_level: Log level for file output
        file_path: Path to log file
        format_str: Log format string
        rotation: Log rotation pattern
        network_logging: Enable network logging
        network_host: Network logging host
        network_port: Network logging port
    """
    # Remove default handlers
    logger.remove()
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=console_level,
        format=format_str,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        file_path,
        level=file_level,
        format=format_str,
        rotation=rotation,
        compression="zip"
    )
    
    # Add structured JSON log file
    json_format = "{message}"
    logger.add(
        os.path.join(os.path.dirname(file_path), "legion_structured.jsonl"),
        level=file_level,
        format=json_format,
        rotation=rotation,
        compression="zip",
        serialize=True  # Enables structured logging
    )
    
    # Add network logging if enabled
    if network_logging:
        network_sink = NetworkLogSink(network_host, network_port)
        logger.add(network_sink, level=file_level)
        
    logger.info("Logging system initialized")


# Initialize default logging configuration
setup_logging()

# Export main classes and functions
__all__ = [
    'logger', 
    'LogContext', 
    'PerformanceTimer', 
    'AgentLogger',
    'NetworkLogSink', 
    'setup_logging'
]
