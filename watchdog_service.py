#!/usr/bin/env python3
"""
Watchdog Service for Trading Bot Code Changes

Monitors Python files for changes and automatically restarts the main trading process
without interrupting existing API connections or losing state.
"""

import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
import signal
import hashlib
import json
from datetime import datetime

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Configure logger
logger = logging.getLogger("watchdog_service")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class CodeChangeHandler(FileSystemEventHandler):
    """Handles file system events for Python code files"""
    
    def __init__(self, callback: Callable, file_patterns: List[str] = None):
        """
        Initialize the handler
        
        Args:
            callback: Function to call when a code change is detected
            file_patterns: List of file patterns to monitor (e.g., ["*.py"])
        """
        self.callback = callback
        self.file_patterns = file_patterns or ["*.py"]
        self.last_reload_time = time.time()
        self.min_reload_interval = 2.0  # Minimum seconds between reloads
        self.modified_files = set()
        self.processing_lock = threading.Lock()
        
    def on_modified(self, event: FileSystemEvent) -> None:
        """Called when a file is modified"""
        if not event.is_directory and self._should_handle_file(event.src_path):
            with self.processing_lock:
                self.modified_files.add(event.src_path)
                self._schedule_reload()
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Called when a file is created"""
        if not event.is_directory and self._should_handle_file(event.src_path):
            with self.processing_lock:
                self.modified_files.add(event.src_path)
                self._schedule_reload()
    
    def _should_handle_file(self, path: str) -> bool:
        """Check if the file should trigger a reload based on patterns"""
        file_path = Path(path)
        
        # Skip temporary files, __pycache__, and non-Python files by default
        if (file_path.name.startswith(".") or 
            "__pycache__" in file_path.parts or
            file_path.name.endswith(".pyc")):
            return False
            
        # Check against file patterns
        for pattern in self.file_patterns:
            if pattern.startswith("*."):
                ext = pattern[1:]  # Get extension including the dot
                if file_path.name.endswith(ext):
                    return True
            elif pattern in file_path.name:
                return True
                
        return False
    
    def _schedule_reload(self) -> None:
        """Schedule a reload with debouncing to prevent rapid consecutive reloads"""
        current_time = time.time()
        time_since_last_reload = current_time - self.last_reload_time
        
        if time_since_last_reload < self.min_reload_interval:
            # If we recently reloaded, schedule a delayed reload
            if not hasattr(self, '_reload_timer') or not self._reload_timer.is_alive():
                self._reload_timer = threading.Timer(
                    self.min_reload_interval - time_since_last_reload + 0.1, 
                    self._execute_reload
                )
                self._reload_timer.daemon = True
                self._reload_timer.start()
        else:
            # Otherwise reload immediately
            self._execute_reload()
    
    def _execute_reload(self) -> None:
        """Execute the actual reload with the collected modified files"""
        with self.processing_lock:
            if not self.modified_files:
                return
                
            modified_files_list = list(self.modified_files)
            self.modified_files.clear()
            self.last_reload_time = time.time()
            
            logger.info(f"Code changes detected in {len(modified_files_list)} file(s):")
            for file_path in modified_files_list:
                logger.info(f"  - {os.path.basename(file_path)}")
                
            try:
                self.callback(modified_files_list)
            except Exception as e:
                logger.error(f"Error during reload callback: {e}")

class WatchdogService:
    """Service for watching code changes and reloading modules"""
    
    def __init__(self, 
                watch_paths: List[str] = None, 
                file_patterns: List[str] = None,
                reload_callback: Optional[Callable] = None):
        """
        Initialize the watchdog service
        
        Args:
            watch_paths: List of directory paths to monitor
            file_patterns: List of file patterns to monitor
            reload_callback: Optional custom callback for reload events
        """
        self.watch_paths = watch_paths or ["."]
        self.file_patterns = file_patterns or ["*.py"]
        self.custom_reload_callback = reload_callback
        self.observer = Observer()
        self.reloaded_modules = {}  # Track successfully reloaded modules
        self.reload_failures = {}   # Track modules that failed to reload
        
        # Convert relative paths to absolute
        self.watch_paths = [os.path.abspath(path) for path in self.watch_paths]
        
        # Setup event handler
        self.event_handler = CodeChangeHandler(
            callback=self._handle_code_change,
            file_patterns=self.file_patterns
        )
    
    def start(self) -> None:
        """Start the file monitoring service"""
        for path in self.watch_paths:
            if os.path.exists(path):
                self.observer.schedule(self.event_handler, path, recursive=True)
                logger.info(f"Monitoring for code changes in: {path}")
            else:
                logger.warning(f"Watch path does not exist: {path}")
        
        self.observer.start()
        logger.info(f"Watchdog service started. Monitoring {len(self.watch_paths)} paths for changes.")
    
    def stop(self) -> None:
        """Stop the file monitoring service"""
        self.observer.stop()
        self.observer.join()
        logger.info("Watchdog service stopped.")
    
    def _handle_code_change(self, modified_files: List[str]) -> None:
        """Handle code change event by reloading affected modules"""
        try:
            # Convert to module names and filter Python files
            modules_to_reload = self._get_modules_from_files(modified_files)
            
            if not modules_to_reload:
                logger.info("No Python modules to reload.")
                return
                
            # Reload the modules
            for module_name in modules_to_reload:
                try:
                    if module_name in sys.modules:
                        logger.info(f"Reloading module: {module_name}")
                        importlib.reload(sys.modules[module_name])
                        self.reloaded_modules[module_name] = time.time()
                    else:
                        logger.info(f"Module not loaded yet: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to reload module {module_name}: {e}")
                    self.reload_failures[module_name] = (time.time(), str(e))
            
            # Call custom reload callback if provided
            if self.custom_reload_callback:
                self.custom_reload_callback(modules_to_reload)
                
            logger.info(f"Code reload completed for {len(modules_to_reload)} module(s)")
            
        except Exception as e:
            logger.error(f"Error handling code change: {e}")
    
    def _get_modules_from_files(self, file_paths: List[str]) -> List[str]:
        """Convert file paths to Python module names"""
        modules = []
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
                
            # Get the absolute path
            abs_path = os.path.abspath(file_path)
            
            # Find module name based on path relative to any Python path
            module_name = None
            for path in sys.path:
                path = os.path.abspath(path)
                if abs_path.startswith(path):
                    rel_path = os.path.relpath(abs_path, path)
                    # Convert path to module notation
                    module_name = rel_path.replace(os.path.sep, '.')
                    if module_name.endswith('.py'):
                        module_name = module_name[:-3]
                    break
            
            # If no match in sys.path, use the filename without extension
            if not module_name:
                module_name = os.path.basename(file_path)[:-3]
                
            if module_name:
                modules.append(module_name)
                
        return modules

# Singleton instance for global access
_watchdog_instance = None

def get_watchdog_service() -> WatchdogService:
    """Get or create the global watchdog service instance"""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = WatchdogService(
            watch_paths=["."],
            file_patterns=["*.py"]
        )
    return _watchdog_instance

def start_watchdog(watch_paths=None, reload_callback=None):
    """Start the watchdog service with the specified configuration"""
    watchdog = get_watchdog_service()
    
    if watch_paths:
        watchdog.watch_paths = watch_paths
    
    if reload_callback:
        watchdog.custom_reload_callback = reload_callback
        
    watchdog.start()
    return watchdog

class FileWatcher:
    """
    Watches files for changes and triggers callback when detected
    """
    def __init__(self, paths, callback, interval=1.0):
        """
        Initialize the file watcher
        
        Args:
            paths (list): List of file paths or directories to watch
            callback (callable): Function to call when changes are detected
            interval (float): How often to check for changes in seconds
        """
        self.paths = [Path(p) for p in paths]
        self.callback = callback
        self.interval = interval
        self.snapshots = {}
        self.running = False
        self.thread = None
        self.exclude_patterns = [
            '.git', 
            '__pycache__', 
            '.pyc', 
            '.log', 
            '.db', 
            '.sqlite'
        ]
    
    def is_excluded(self, path):
        """Check if a path should be excluded from watching"""
        path_str = str(path)
        return any(pattern in path_str for pattern in self.exclude_patterns)
    
    def get_files_recursive(self):
        """Get all files in watched paths recursively, excluding certain patterns"""
        all_files = []
        for path in self.paths:
            if path.is_file() and not self.is_excluded(path):
                all_files.append(path)
            elif path.is_dir():
                for root, dirs, files in os.walk(path):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not self.is_excluded(d)]
                    for file in files:
                        file_path = Path(root) / file
                        if not self.is_excluded(file_path):
                            all_files.append(file_path)
        return all_files
    
    def take_snapshot(self):
        """Take a snapshot of current file states"""
        snapshot = {}
        for file_path in self.get_files_recursive():
            try:
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    # Use modification time as an efficient first check
                    if file_path in self.snapshots and self.snapshots[file_path] == mtime:
                        # No change in mtime, skip computing hash
                        snapshot[file_path] = mtime
                    else:
                        # Compute file hash for definitive change detection
                        snapshot[file_path] = mtime
            except (FileNotFoundError, PermissionError):
                # Skip files that can't be accessed
                pass
        return snapshot
    
    def detect_changes(self):
        """Detect if any files have changed since last snapshot"""
        current = self.take_snapshot()
        
        if not self.snapshots:
            # First run, just save the snapshot
            self.snapshots = current
            return False
        
        # Check for any modifications
        changed_files = []
        
        # Find modified or added files
        for file_path, mtime in current.items():
            if file_path not in self.snapshots or self.snapshots[file_path] != mtime:
                changed_files.append(file_path)
        
        # Find deleted files
        for file_path in self.snapshots:
            if file_path not in current:
                changed_files.append(file_path)
        
        if changed_files:
            logger.info(f"Detected changes in {len(changed_files)} files")
            for file_path in changed_files[:5]:  # Show first 5 files only
                logger.info(f"  - {file_path}")
            if len(changed_files) > 5:
                logger.info(f"  - ... and {len(changed_files) - 5} more files")
            
            # Update snapshot
            self.snapshots = current
            return True
        
        # Update snapshot
        self.snapshots = current
        return False
    
    def watch(self):
        """Start watching for changes in a loop"""
        while self.running:
            if self.detect_changes():
                self.callback()
            time.sleep(self.interval)
    
    def start(self):
        """Start the file watcher in a thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Watcher is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()
        logger.info(f"File watcher started, monitoring {len(self.paths)} paths")
    
    def stop(self):
        """Stop the file watcher"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        logger.info("File watcher stopped")

class ServiceManager:
    """
    Manages services, handling starting, stopping, and restarting
    """
    def __init__(self, service_config):
        """
        Initialize the service manager
        
        Args:
            service_config (dict): Configuration for services
        """
        self.services = {}
        self.service_config = service_config
        self.load_services(service_config)
    
    def load_services(self, config):
        """Load service configurations"""
        for service_name, service_info in config.items():
            self.services[service_name] = {
                'command': service_info.get('command'),
                'watch_paths': service_info.get('watch_paths', []),
                'process': None,
                'auto_restart': service_info.get('auto_restart', True),
                'restart_delay': service_info.get('restart_delay', 2.0),
                'graceful_shutdown': service_info.get('graceful_shutdown', True),
                'env': service_info.get('env', {}),
                'working_dir': service_info.get('working_dir', None),
            }
    
    def start_service(self, service_name):
        """Start a service by name"""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        
        if service['process'] is not None and service['process'].poll() is None:
            logger.warning(f"Service {service_name} is already running")
            return True
        
        try:
            # Prepare environment variables
            env = os.environ.copy()
            env.update(service['env'])
            
            # Start the process
            process = subprocess.Popen(
                service['command'],
                shell=True,
                env=env,
                cwd=service['working_dir'],
                # Redirect stdout/stderr to prevent deadlocks
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Make process a session leader so child processes are also terminated
                start_new_session=True
            )
            
            service['process'] = process
            logger.info(f"Started service: {service_name}")
            
            # Start a thread to handle process output
            threading.Thread(
                target=self._handle_process_output,
                args=(service_name, process),
                daemon=True
            ).start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            return False
    
    def _handle_process_output(self, service_name, process):
        """Handle and log process output"""
        try:
            for line in iter(process.stdout.readline, b''):
                line_str = line.decode('utf-8', errors='replace').strip()
                if line_str:
                    logger.info(f"[{service_name}] {line_str}")
            
            for line in iter(process.stderr.readline, b''):
                line_str = line.decode('utf-8', errors='replace').strip()
                if line_str:
                    logger.error(f"[{service_name}] {line_str}")
        except Exception as e:
            logger.error(f"Error handling process output for {service_name}: {e}")
    
    def stop_service(self, service_name):
        """Stop a service by name"""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        process = service['process']
        
        if process is None or process.poll() is not None:
            logger.warning(f"Service {service_name} is not running")
            service['process'] = None
            return True
        
        try:
            if service['graceful_shutdown']:
                # Try to terminate gracefully first
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait for a short time for graceful shutdown
                for _ in range(10):
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
            
            # If still running, force kill
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            
            service['process'] = None
            logger.info(f"Stopped service: {service_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False
    
    def restart_service(self, service_name):
        """Restart a service by name"""
        logger.info(f"Restarting service: {service_name}")
        self.stop_service(service_name)
        
        # Wait before restarting
        service = self.services[service_name]
        time.sleep(service['restart_delay'])
        
        return self.start_service(service_name)
    
    def start_all_services(self):
        """Start all services"""
        for service_name in self.services:
            self.start_service(service_name)
    
    def stop_all_services(self):
        """Stop all services"""
        for service_name in self.services:
            self.stop_service(service_name)
    
    def restart_all_services(self):
        """Restart all services"""
        for service_name in self.services:
            self.restart_service(service_name)

class Watchdog:
    """
    Main watchdog service that coordinates file watching and service management
    """
    def __init__(self, config_file=None):
        """
        Initialize the watchdog with optional config file
        
        Args:
            config_file (str): Path to the JSON configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.service_manager = ServiceManager(self.config.get('services', {}))
        self.file_watchers = []
        self.setup_watchers()
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        default_config = {
            'services': {
                'main': {
                    'command': 'python supertrend_pullback_live.py',
                    'watch_paths': ['supertrend_pullback_live.py', 'bitget_utils.py', 'error_logger.py'],
                    'auto_restart': True,
                    'restart_delay': 2.0,
                    'graceful_shutdown': True,
                    'working_dir': None
                }
            },
            'watcher': {
                'interval': 1.0,
                'exclude_patterns': ['.git', '__pycache__', '.pyc', '.log', '.db', '.sqlite']
            }
        }
        
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_file}: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def setup_watchers(self):
        """Set up file watchers for each service"""
        for service_name, service in self.service_manager.services.items():
            if service['auto_restart'] and service['watch_paths']:
                watch_paths = service['watch_paths']
                
                # Create a callback function for this specific service
                def create_restart_callback(svc_name):
                    def restart_callback():
                        logger.info(f"File changes detected, restarting {svc_name}")
                        self.service_manager.restart_service(svc_name)
                    return restart_callback
                
                watcher = FileWatcher(
                    watch_paths,
                    create_restart_callback(service_name),
                    interval=self.config.get('watcher', {}).get('interval', 1.0)
                )
                
                self.file_watchers.append(watcher)
    
    def start(self):
        """Start the watchdog service"""
        logger.info("Starting watchdog service")
        
        # Start all services
        self.service_manager.start_all_services()
        
        # Start all file watchers
        for watcher in self.file_watchers:
            watcher.start()
        
        logger.info("Watchdog service started")
    
    def stop(self):
        """Stop the watchdog service"""
        logger.info("Stopping watchdog service")
        
        # Stop all file watchers
        for watcher in self.file_watchers:
            watcher.stop()
        
        # Stop all services
        self.service_manager.stop_all_services()
        
        logger.info("Watchdog service stopped")

def start_watchdog(config_file=None):
    """Start the watchdog service with an optional config file"""
    watchdog = Watchdog(config_file)
    watchdog.start()
    return watchdog

# For running as a standalone script
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/watchdog.log")
        ]
    )
    
    # Get config file from command line argument if provided
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Start watchdog
    watchdog = start_watchdog(config_file)
    
    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("Keyboard interrupt received, stopping watchdog")
        watchdog.stop()
    except Exception as e:
        logger.error(f"Error in watchdog main loop: {e}")
        watchdog.stop() 