import os
import sys
import psutil
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Terminal-Reset")

def set_working_directory():
    """Set working directory to BACKTESTER"""
    try:
        # First try to change directly to C:\BACKTESTER
        try:
            os.chdir(r"C:\BACKTESTER")
            logger.info(f"Working directory set to: {os.getcwd()}")
            return True
        except:
            pass

        # If that fails, try to find BACKTESTER in the current path
        current_dir = os.getcwd()
        if 'BACKTESTER' in current_dir:
            backtester_path = current_dir.split('BACKTESTER')[0] + 'BACKTESTER'
            os.chdir(backtester_path)
            logger.info(f"Working directory set to: {os.getcwd()}")
            return True

        # If we're in C:\Users\Letme, try to go up one level and find BACKTESTER
        if current_dir.endswith('Letme'):
            os.chdir('..')
            if os.path.exists('BACKTESTER'):
                os.chdir('BACKTESTER')
                logger.info(f"Working directory set to: {os.getcwd()}")
                return True

        logger.error("Could not find BACKTESTER directory")
        return False

    except Exception as e:
        logger.error(f"Failed to set working directory: {e}")
        return False

def reset_terminals():
    """Reset all terminals and kill related processes"""
    logger.info("Starting terminal reset...")
    
    # First set the working directory
    if not set_working_directory():
        logger.error("Failed to set working directory to BACKTESTER. Aborting reset.")
        return
    
    # Kill all Python processes related to our trading system
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip current process
            if proc.pid == current_pid:
                continue
                
            # Check if it's a Python process running our scripts
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any(x in ' '.join(cmdline) for x in ['supertrend', 'super_z']):
                    logger.info(f"Terminating process: {proc.pid} - {' '.join(cmdline)}")
                    proc.terminate()
                    proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
    
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Verify we're in the correct directory
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # List files in directory to verify
    logger.info("\nFiles in current directory:")
    for file in os.listdir():
        logger.info(f"- {file}")
    
    logger.info("\nTerminal reset complete!")

if __name__ == "__main__":
    reset_terminals() 