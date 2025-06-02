#!/usr/bin/env python3
"""
Unified Error Logging System

Provides centralized error handling and logging for the entire trading application.
Features:
- Structured error logging with context
- Error categorization
- Error statistics and reporting
- Configurable severity levels
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sqlite3

# Configure logs directory
os.makedirs("logs", exist_ok=True)

class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "NETWORK"
    API = "API"
    RATE_LIMIT = "RATE_LIMIT"
    AUTH = "AUTHENTICATION"
    BALANCE = "BALANCE"
    ORDER = "ORDER"
    DATA = "DATA"
    CONFIG = "CONFIG"
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"
    MARKET_DATA = "MARKET_DATA"
    ORDER_EXECUTION = "ORDER_EXECUTION"
    SIGNAL = "SIGNAL"
    SYSTEM = "SYSTEM"
    DATABASE = "DATABASE"

class UnifiedErrorLogger:
    """Centralized error logging system for trading applications"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logger instance exists"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UnifiedErrorLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, log_file: str = "logs/error.log", max_size: int = 10 * 1024 * 1024, 
                backup_count: int = 5, console_output: bool = True):
        """Initialize the error logger"""
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return
            
        # Base configuration
        self.log_file = log_file
        self.max_size = max_size
        self.backup_count = backup_count
        self.console_output = console_output
        
        # Setup logger
        self.logger = logging.getLogger('unified_error_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't propagate to parent loggers
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            # Colored formatting for console
            console_formatter = self._get_colored_formatter()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = defaultdict(lambda: deque(maxlen=100))
        self.recent_errors = deque(maxlen=1000)  # Store recent errors
        
        # Alert thresholds
        self.alert_thresholds = {
            ErrorCategory.RATE_LIMIT.value: 10,   # 10 rate limit errors
            ErrorCategory.NETWORK.value: 5,       # 5 network errors
            ErrorCategory.AUTH.value: 3,          # 3 auth errors
            ErrorCategory.BALANCE.value: 3,       # 3 balance errors
            ErrorCategory.API.value: 5            # 5 API errors
        }
        
        # Alert cooldowns (to prevent spamming)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300.0  # 5 minutes between alerts for same category
        
        # Completed initialization
        self._initialized = True
    
    def _get_colored_formatter(self) -> logging.Formatter:
        """Create a formatter with color coding for console output"""
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        green = "\x1b[32;20m"
        blue = "\x1b[34;20m"
        reset = "\x1b[0m"
        
        formats = {
            logging.DEBUG: grey + "%(asctime)s - [%(levelname)s] - %(message)s" + reset,
            logging.INFO: green + "%(asctime)s - [%(levelname)s] - %(message)s" + reset,
            logging.WARNING: yellow + "%(asctime)s - [%(levelname)s] - %(message)s" + reset,
            logging.ERROR: red + "%(asctime)s - [%(levelname)s] - %(message)s" + reset,
            logging.CRITICAL: bold_red + "%(asctime)s - [%(levelname)s] - %(message)s" + reset
        }
        
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                log_fmt = formats.get(record.levelno)
                formatter = logging.Formatter(log_fmt)
                return formatter.format(record)
        
        return ColoredFormatter()
    
    def log_error(self, 
                 message: str,
                 category: Union[ErrorCategory, str] = ErrorCategory.UNKNOWN,
                 severity: Union[ErrorSeverity, int, str] = ErrorSeverity.ERROR,
                 error: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None,
                 stack_trace: bool = True) -> Dict:
        """
        Log an error with full context
        
        Args:
            message: Main error message
            category: Error category for classification
            severity: Error severity level
            error: Exception object if available
            context: Additional context dictionary
            stack_trace: Whether to include stack trace for exceptions
            
        Returns:
            Dict containing the logged error details
        """
        # Process category
        if isinstance(category, str):
            try:
                category = ErrorCategory(category)
            except ValueError:
                category = ErrorCategory.UNKNOWN
        
        # Process severity
        if isinstance(severity, str):
            try:
                severity = ErrorSeverity[severity]
            except KeyError:
                severity = ErrorSeverity.ERROR
        elif isinstance(severity, int):
            severity = ErrorSeverity(severity) if severity in [e.value for e in ErrorSeverity] else ErrorSeverity.ERROR
        
        # Format context
        context_dict = context or {}
        
        # Extract exception details if provided
        error_details = {}
        if error:
            error_details = {
                "type": type(error).__name__,
                "message": str(error)
            }
            if stack_trace:
                error_details["traceback"] = traceback.format_exc()
                
            # Add exception details to context
            context_dict.update({"exception": error_details})
        
        # Create structured error record
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "category": category.value,
            "severity": severity.value,
            "context": context_dict
        }
        
        # Convert to log message
        log_message = self._format_log_message(error_record)
        
        # Log at appropriate level
        if severity == ErrorSeverity.DEBUG:
            self.logger.debug(log_message)
        elif severity == ErrorSeverity.INFO:
            self.logger.info(log_message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        
        # Track error statistics
        self._track_error(error_record)
        
        # Check for alert thresholds
        self._check_alert_threshold(category.value)
        
        return error_record
    
    def _format_log_message(self, error_record: Dict) -> str:
        """Format error record into a log message string"""
        # Basic message with category
        message = f"[{error_record['category']}] {error_record['message']}"
        
        # Add exception info if present
        context = error_record.get('context', {})
        exception = context.get('exception', {})
        if exception:
            exc_type = exception.get('type')
            exc_msg = exception.get('message')
            if exc_type and exc_msg:
                message += f" | Exception: {exc_type}: {exc_msg}"
        
        # Add important context keys (filter out large or less relevant ones)
        context_str = []
        for key, value in context.items():
            if key == 'exception':
                continue
            if isinstance(value, (str, int, float, bool)):
                context_str.append(f"{key}={value}")
        
        if context_str:
            message += f" | Context: {', '.join(context_str)}"
            
        return message
    
    def _track_error(self, error_record: Dict) -> None:
        """Track error for statistics and history"""
        category = error_record['category']
        self.error_counts[category] += 1
        self.error_history[category].append(error_record)
        self.recent_errors.append(error_record)
    
    def _check_alert_threshold(self, category: str) -> None:
        """Check if error count has reached alert threshold"""
        # Skip if no threshold for this category
        if category not in self.alert_thresholds:
            return
            
        # Check cooldown period
        current_time = time.time()
        last_alert = self.last_alert_time.get(category, 0)
        if current_time - last_alert < self.alert_cooldown:
            return
            
        # Count recent errors (last 15 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=15)
        recent_count = 0
        
        for error in self.error_history[category]:
            error_time = datetime.fromisoformat(error['timestamp'])
            if error_time >= cutoff_time:
                recent_count += 1
        
        # Check if threshold exceeded
        threshold = self.alert_thresholds[category]
        if recent_count >= threshold:
            alert_message = (f"⚠️ ALERT: {category} errors threshold exceeded! "
                           f"{recent_count} errors in the last 15 minutes (threshold: {threshold})")
            
            # Log alert
            self.logger.critical(alert_message)
            
            # Update last alert time
            self.last_alert_time[category] = current_time
    
    def get_error_stats(self, 
                       time_window: Optional[timedelta] = None,
                       categories: Optional[List[Union[ErrorCategory, str]]] = None) -> Dict:
        """
        Get error statistics for specified time window and categories
        
        Args:
            time_window: Time window for stats (default: all time)
            categories: List of categories to include (default: all)
            
        Returns:
            Dict with error statistics
        """
        # Process categories
        if categories:
            cat_values = []
            for cat in categories:
                if isinstance(cat, ErrorCategory):
                    cat_values.append(cat.value)
                elif isinstance(cat, str):
                    cat_values.append(cat)
        else:
            cat_values = [cat.value for cat in ErrorCategory]
        
        # Define cutoff time if window specified
        cutoff_time = None
        if time_window:
            cutoff_time = datetime.now() - time_window
        
        # Collect stats
        stats = {
            "total_errors": 0,
            "by_category": defaultdict(int),
            "by_severity": defaultdict(int),
            "recent_errors": []
        }
        
        # Process recent errors
        for error in self.recent_errors:
            # Skip if not in requested categories
            if error['category'] not in cat_values:
                continue
                
            # Skip if outside time window
            if cutoff_time:
                error_time = datetime.fromisoformat(error['timestamp'])
                if error_time < cutoff_time:
                    continue
            
            # Count error
            stats["total_errors"] += 1
            stats["by_category"][error['category']] += 1
            stats["by_severity"][error['severity']] += 1
            
            # Add to recent errors list (limited to last 10)
            if len(stats["recent_errors"]) < 10:
                # Simplify for output
                simplified = {
                    "timestamp": error['timestamp'],
                    "category": error['category'],
                    "message": error['message']
                }
                stats["recent_errors"].append(simplified)
        
        return stats
    
    def export_errors_to_json(self, filepath: str, 
                            time_window: Optional[timedelta] = None) -> bool:
        """
        Export errors to a JSON file
        
        Args:
            filepath: Path to export JSON file
            time_window: Optional time window to filter errors
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Define cutoff time if window specified
            cutoff_time = None
            if time_window:
                cutoff_time = datetime.now() - time_window
            
            # Filter errors
            filtered_errors = []
            for error in self.recent_errors:
                if cutoff_time:
                    error_time = datetime.fromisoformat(error['timestamp'])
                    if error_time < cutoff_time:
                        continue
                filtered_errors.append(error)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "exported_at": datetime.now().isoformat(),
                    "error_count": len(filtered_errors),
                    "errors": filtered_errors
                }, f, indent=2)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to export errors to JSON: {e}")
            return False
    
    def clear_error_history(self) -> None:
        """Clear error history and reset counters"""
        self.error_counts.clear()
        self.error_history.clear()
        self.recent_errors.clear()
        self.last_alert_time.clear()

# Global instance
_error_logger = None

def get_error_logger() -> UnifiedErrorLogger:
    """Get the global error logger instance"""
    global _error_logger
    if _error_logger is None:
        _error_logger = UnifiedErrorLogger()
    return _error_logger

# Convenience function for logging errors
def log_error(message: str, 
             category: Union[ErrorCategory, str] = ErrorCategory.UNKNOWN,
             severity: Union[ErrorSeverity, int, str] = ErrorSeverity.ERROR,
             error: Optional[Exception] = None,
             context: Optional[Dict[str, Any]] = None,
             stack_trace: bool = True) -> Dict:
    """Convenience function to log an error using the global error logger"""
    logger = get_error_logger()
    return logger.log_error(
        message=message,
        category=category,
        severity=severity,
        error=error,
        context=context,
        stack_trace=stack_trace
    )

# Initialize database for error storage
def init_error_db():
    """Initialize SQLite database for persistent error storage"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/errors.db")
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        category TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        error_details TEXT,
        context TEXT,
        stack_trace TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Ensure DB is initialized
init_error_db()

def log_error_to_db(message, category="UNKNOWN", severity="ERROR", error=None, context=None, stack_trace=True):
    """
    Log an error with detailed information
    
    Args:
        message (str): Main error message
        category (str): Error category (use ErrorCategory enum)
        severity (str): Error severity (use ErrorSeverity enum)
        error (Exception, optional): Exception object if available
        context (dict, optional): Additional context information
        stack_trace (bool): Whether to include stack trace
    
    Returns:
        dict: Error information that was logged
    """
    # Convert enum to string if needed
    if isinstance(category, ErrorCategory):
        category = category.value
    if isinstance(severity, ErrorSeverity):
        severity = severity.value
    
    # Get timestamp
    timestamp = datetime.now().isoformat()
    
    # Process error details
    error_details = str(error) if error else None
    
    # Get stack trace if requested and error exists
    trace = None
    if stack_trace and error:
        trace = traceback.format_exc()
    
    # Context as JSON
    context_json = json.dumps(context) if context else None
    
    # Prepare error info
    error_info = {
        "timestamp": timestamp,
        "category": category,
        "severity": severity,
        "message": message,
        "error_details": error_details,
        "context": context_json,
        "stack_trace": trace
    }
    
    # Log to file
    logger = get_error_logger()
    log_message = f"{category} - {message}"
    
    if error:
        log_message += f": {error}"
    
    # Choose log level based on severity
    if severity == "CRITICAL":
        logger.critical(log_message)
    elif severity == "ERROR":
        logger.error(log_message)
    elif severity == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)
    
    # Save to database
    try:
        conn = sqlite3.connect("data/errors.db")
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO errors (timestamp, category, severity, message, error_details, context, stack_trace)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            category,
            severity,
            message,
            error_details,
            context_json,
            trace
        ))
        
        conn.commit()
        conn.close()
    except Exception as e:
        # If DB storage fails, at least log it
        logger.error(f"Failed to store error in database: {e}")
    
    return error_info

# Example usage
if __name__ == "__main__":
    # Test the error logger
    logger = get_error_logger()
    
    # Log different types of errors
    try:
        result = 1 / 0
    except Exception as e:
        logger.log_error(
            message="Division by zero error", 
            category=ErrorCategory.INTERNAL, 
            error=e,
            context={"operation": "division", "numerator": 1, "denominator": 0}
        )
    
    # Log API error
    logger.log_error(
        message="API request failed",
        category=ErrorCategory.API,
        severity=ErrorSeverity.WARNING,
        context={"endpoint": "/api/data", "status_code": 429, "retry_after": 30}
    )
    
    # Log rate limit error
    logger.log_error(
        message="Rate limit exceeded",
        category=ErrorCategory.RATE_LIMIT,
        context={"endpoint": "fetch_ticker", "limit": 20, "period": "1s"}
    )
    
    # Print error stats
    stats = logger.get_error_stats(time_window=timedelta(hours=1))
    print(f"Error Stats (last hour): {json.dumps(stats, indent=2)}")
    
    print("Error logger test complete. Check logs/error.log for output.") 