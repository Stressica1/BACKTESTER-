#!/usr/bin/env python3
"""
Trading Bot Launcher with Error Handling and Watchdog

This script manages the launch, monitoring, and recovery of the trading bot.
Features:
- Automatic dependency installation
- Error monitoring and recovery
- Code change detection and hot reloading
- Health checks and status reporting
"""

import argparse
import asyncio
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure paths exist
os.makedirs("logs", exist_ok=True)
os.makedirs("config", exist_ok=True)

# Configure logging
logger = logging.getLogger("launcher")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/launcher.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter("\033[92m%(asctime)s\033[0m - \033[94m%(levelname)s\033[0m - %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Flag to track if bot is running
bot_process = None
bot_thread = None
stop_event = threading.Event()
watchdog_started = False

def check_dependencies() -> bool:
    """Check and install required dependencies"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "ccxt", "numpy", "pandas", "aiohttp", "websockets", 
        "watchdog", "python-dotenv", "tqdm", "aiolimiter"
    ]
    
    try:
        # Check requirements.txt first
        req_file = Path("requirements.txt")
        if req_file.exists():
            logger.info("Found requirements.txt, installing dependencies...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies from requirements.txt: {result.stderr}")
                # Fall back to manual installation
            else:
                logger.info("Successfully installed dependencies from requirements.txt")
                return True
                
        # Manual installation of critical packages
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.split(">=")[0].strip())
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing_packages,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
                
        logger.info("All dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return False

def setup_watchdog():
    """Setup file system watchdog for hot reloading"""
    global watchdog_started
    
    try:
        # Import here to ensure it's installed
        from watchdog_service import start_watchdog
        
        def reload_callback(modules):
            logger.info(f"Code changes detected in {len(modules)} modules. Hot reloading...")
            # The actual reloading is handled by the module logic
            
        # Start watchdog for current directory
        start_watchdog(
            watch_paths=["."],
            reload_callback=reload_callback
        )
        
        watchdog_started = True
        logger.info("File watchdog started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start watchdog: {e}")
        watchdog_started = False

def restart_bot():
    """Restart the trading bot"""
    stop_bot()
    time.sleep(2)  # Give time for cleanup
    start_bot()
    
def stop_bot():
    """Stop the trading bot"""
    global bot_process, bot_thread, stop_event
    
    if bot_process:
        logger.info("Stopping trading bot...")
        
        try:
            # Set stop event
            stop_event.set()
            
            # Wait for thread to terminate
            if bot_thread and bot_thread.is_alive():
                bot_thread.join(timeout=5)
            
            # If process still running, terminate it
            if bot_process:
                # Send SIGTERM first for graceful shutdown
                bot_process.terminate()
                
                try:
                    # Wait up to 5 seconds for graceful termination
                    bot_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if still running
                    logger.warning("Bot did not terminate gracefully, forcing shutdown")
                    bot_process.kill()
                    
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
        
        bot_process = None
        logger.info("Trading bot stopped")

def monitor_process():
    """Monitor the bot process and restart if needed"""
    global bot_process, stop_event
    
    while not stop_event.is_set():
        if bot_process:
            return_code = bot_process.poll()
            
            # Check if process has exited
            if return_code is not None:
                if return_code != 0:
                    logger.warning(f"Bot process exited with code {return_code}, restarting...")
                    time.sleep(5)  # Wait before restarting
                    start_bot_process()
                else:
                    logger.info("Bot process exited normally")
                    return
                    
        time.sleep(1)

def log_health_check():
    """Log health information for monitoring"""
    try:
        # Check error log count
        error_count = 0
        try:
            from error_logger import get_error_logger
            error_logger = get_error_logger()
            stats = error_logger.get_error_stats(time_window=timedelta(hours=1))
            error_count = stats.get("total_errors", 0)
        except:
            pass
            
        # Log health status
        logger.info(f"Health Check - Bot Running: {bot_process is not None}, "
                   f"Errors (last hour): {error_count}, "
                   f"Watchdog Active: {watchdog_started}")
                   
    except Exception as e:
        logger.error(f"Health check error: {e}")

def start_bot_process():
    """Start the trading bot as a subprocess"""
    global bot_process
    
    try:
        # Start bot as subprocess
        logger.info("Starting trading bot process...")
        bot_process = subprocess.Popen(
            [sys.executable, "supertrend_pullback_live.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        logger.info(f"Trading bot started with PID {bot_process.pid}")
        
        # Start output reader threads
        threading.Thread(
            target=read_process_output,
            args=(bot_process.stdout, False),
            daemon=True
        ).start()
        
        threading.Thread(
            target=read_process_output,
            args=(bot_process.stderr, True),
            daemon=True
        ).start()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        bot_process = None
        return False

def read_process_output(stream, is_error):
    """Read and log output from the bot process"""
    prefix = "ERROR" if is_error else "INFO"
    for line in stream:
        line = line.strip()
        if line:
            logger.log(
                logging.ERROR if is_error else logging.INFO,
                f"Bot: {line}"
            )

def start_bot():
    """Start the trading bot and monitoring thread"""
    global bot_thread, stop_event
    
    # Reset stop event
    stop_event.clear()
    
    # Start the process
    success = start_bot_process()
    
    if success:
        # Start monitoring thread
        bot_thread = threading.Thread(target=monitor_process)
        bot_thread.daemon = True
        bot_thread.start()
        
        # Setup periodic health check
        threading.Thread(target=periodic_health_check, daemon=True).start()

def periodic_health_check():
    """Run health checks periodically"""
    while not stop_event.is_set():
        log_health_check()
        time.sleep(300)  # Check every 5 minutes

def handle_signal(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    stop_bot()
    sys.exit(0)

def check_for_updates():
    """Check if the code has local modifications"""
    try:
        # Check if git is available
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            modified_files = [
                line.strip().split(" ", 1)[1] 
                for line in result.stdout.strip().split("\n") 
                if line.strip()
            ]
            
            if modified_files:
                logger.info(f"Local modifications detected in {len(modified_files)} files:")
                for file in modified_files[:5]:  # Show first 5
                    logger.info(f"  - {file}")
                if len(modified_files) > 5:
                    logger.info(f"  - ...and {len(modified_files) - 5} more")
    except:
        # Git not available or not a git repo, ignore
        pass

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Trading Bot Launcher")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies without starting the bot")
    parser.add_argument("--restart", action="store_true", help="Restart the bot if already running")
    parser.add_argument("--version", action="store_true", help="Display version information")
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    if args.version:
        from datetime import date
        print(f"SuperTrend Pullback Trading Bot v1.0.0")
        print(f"Build Date: {date.today().isoformat()}")
        return
    
    logger.info("=== Trading Bot Launcher Started ===")
    
    # Check for updates
    check_for_updates()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Failed to install required dependencies. Exiting.")
        return 1
    
    # Setup watchdog for code changes
    setup_watchdog()
    
    if args.check_only:
        logger.info("Dependency check completed. Not starting bot (--check-only specified).")
        return 0
    
    if args.restart:
        stop_bot()
        time.sleep(2)
    
    # Start the bot
    start_bot()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        stop_bot()
    
    logger.info("=== Trading Bot Launcher Exited ===")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 