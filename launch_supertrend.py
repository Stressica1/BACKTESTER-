import os
import sys
import subprocess
import threading
import time
import webbrowser
import signal
import http.server
import socketserver
import logging
from pathlib import Path
import atexit
import psutil  # Add this import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SuperTrend-Launcher")

# Global variables to track processes
http_server = None
strategy_process = None
pullback_process = None
optimized_process = None

def kill_python_processes():
    """Kill all Python processes related to our trading system"""
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

def cleanup():
    """Clean up all processes"""
    logger.info("Cleaning up processes...")
    
    # Kill all related Python processes
    kill_python_processes()
    
    # Kill strategy process
    if strategy_process:
        try:
            strategy_process.terminate()
            strategy_process.wait(timeout=5)
            logger.info("Strategy process terminated")
        except:
            strategy_process.kill()
            logger.info("Strategy process killed")
    
    # Kill pullback analyzer
    if pullback_process:
        try:
            pullback_process.terminate()
            pullback_process.wait(timeout=5)
            logger.info("Pullback analyzer terminated")
        except:
            pullback_process.kill()
            logger.info("Pullback analyzer killed")
    
    # Kill optimized analyzer
    if optimized_process:
        try:
            optimized_process.terminate()
            optimized_process.wait(timeout=5)
            logger.info("Optimized analyzer terminated")
        except:
            optimized_process.kill()
            logger.info("Optimized analyzer killed")
    
    # Shutdown HTTP server
    if http_server:
        try:
            http_server.shutdown()
            logger.info("HTTP server terminated")
        except:
            logger.info("HTTP server already terminated")
    
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    logger.info("All processes terminated. Terminal reset complete.")

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    logger.info("\nReceived shutdown signal. Cleaning up...")
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup function
atexit.register(cleanup)

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

def start_http_server(port=8000):
    """Start HTTP server for serving static files and dashboard"""
    global http_server
    try:
        # Set working directory
        if not set_working_directory():
            raise Exception("Failed to set working directory to BACKTESTER")
        
        # Create handler
        handler = http.server.SimpleHTTPRequestHandler
        
        # Try different ports if the default is in use
        for port in range(8000, 8010):
            try:
                # Create server
                http_server = socketserver.TCPServer(("", port), handler)
                logger.info(f"Starting HTTP server at http://localhost:{port}")
                http_server.serve_forever()
                break
            except OSError as e:
                if port == 8009:  # Last attempt
                    raise
                logger.warning(f"Port {port} in use, trying next port...")
                continue
    except Exception as e:
        logger.error(f"Error starting HTTP server: {e}")
        raise

def start_strategy():
    """Start the SuperTrend strategy"""
    global strategy_process
    try:
        # Set working directory
        if not set_working_directory():
            raise Exception("Failed to set working directory to BACKTESTER")
        
        logger.info("Starting SuperTrend strategy on Bitget TESTNET...")
        
        # Verify test_batch.json exists and has correct testnet settings
        config_path = os.path.join(os.getcwd(), "test_batch.json")
        if not os.path.exists(config_path):
            logger.error("test_batch.json not found! Cannot start trading.")
            return None
            
        # Start the strategy in a new process with explicit testnet mode
        strategy_process = subprocess.Popen(
            [sys.executable, "supertrend_live.py", "--testnet", "--exchange", "bitget"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=os.getcwd()  # Use current directory
        )
        
        # Log output with emphasis on trading activity
        while True:
            output = strategy_process.stdout.readline()
            if output == '' and strategy_process.poll() is not None:
                break
            if output:
                # Highlight important trading events
                if "BUY Signal:" in output:
                    logger.info(f"🔵 {output.strip()}")
                elif "SELL Signal:" in output:
                    logger.info(f"🔴 {output.strip()}")
                elif "Order placed:" in output:
                    logger.info(f"💫 {output.strip()}")
                elif "Order executed:" in output:
                    logger.info(f"✅ {output.strip()}")
                elif "Error" in output or "Failed" in output:
                    logger.error(f"❌ {output.strip()}")
                else:
                    logger.info(output.strip())
        
        # Log errors
        for error in strategy_process.stderr:
            logger.error(f"❌ {error.strip()}")
        
        return_code = strategy_process.poll()
        if return_code != 0:
            logger.error(f"Strategy process exited with error code {return_code}")
        else:
            logger.info("Strategy process completed successfully")
        
        return strategy_process
    except Exception as e:
        logger.error(f"Error starting strategy: {e}")
        raise

def open_dashboard(port=8000):
    """Open the dashboard in a web browser"""
    try:
        # Wait a moment for servers to start
        time.sleep(2)
        
        dashboard_url = f"http://localhost:{port}/templates/dashboard_supertrend.html"
        logger.info(f"Opening dashboard at {dashboard_url}")
        
        # Open in browser
        webbrowser.open(dashboard_url)
    except Exception as e:
        logger.error(f"Error opening dashboard: {e}")

def _log_process_output(process, name):
    """Log subprocess output for a named process"""
    for line in process.stdout:
        logger.info(f"[{name}] {line.strip()}")
    for line in process.stderr:
        logger.error(f"[{name} ERROR] {line.strip()}")


def start_pullback_analyzer():
    """Start the SuperZ Pullback Analyzer"""
    try:
        logger.info("Starting SuperZ Pullback Analyzer...")
        process = subprocess.Popen(
            [sys.executable, "super_z_pullback_analyzer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        threading.Thread(
            target=_log_process_output,
            args=(process, "PullbackAnalyzer"),
            daemon=True
        ).start()
        return process
    except Exception as e:
        logger.error(f"Error starting pullback analyzer: {e}")
        return None


def start_optimized_analyzer():
    """Start the SuperZ Optimized Analyzer"""
    try:
        logger.info("Starting SuperZ Optimized Analyzer...")
        process = subprocess.Popen(
            [sys.executable, "super_z_optimized.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        threading.Thread(
            target=_log_process_output,
            args=(process, "OptimizedAnalyzer"),
            daemon=True
        ).start()
        return process
    except Exception as e:
        logger.error(f"Error starting optimized analyzer: {e}")
        return None

def main():
    """Main function to start everything"""
    try:
        # Check if supertrend_live.py exists
        if not Path("supertrend_live.py").exists():
            logger.error("supertrend_live.py not found. Make sure you're in the correct directory.")
            return
        
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        
        # Start strategy with explicit testnet mode
        strategy_process = start_strategy()
        if not strategy_process:
            logger.error("Failed to start strategy. Check logs for details.")
            return
            
        # Start additional analyzers
        pullback_process = start_pullback_analyzer()
        optimized_process = start_optimized_analyzer()
        
        # Open dashboard
        open_dashboard()
        
        # Wait for Ctrl+C
        logger.info("Trading system running on Bitget TESTNET. Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down trading system...")
            cleanup()
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        cleanup()

if __name__ == "__main__":
    main()
