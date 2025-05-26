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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SuperTrend-Launcher")

def start_http_server(port=8000):
    """Start HTTP server for serving static files and dashboard"""
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_dir)
        
        # Create handler
        handler = http.server.SimpleHTTPRequestHandler
        
        # Create server
        httpd = socketserver.TCPServer(("", port), handler)
        
        logger.info(f"Starting HTTP server at http://localhost:{port}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error starting HTTP server: {e}")
        raise

def start_strategy():
    """Start the SuperTrend strategy"""
    try:
        logger.info("Starting SuperTrend strategy...")
        
        # Start the strategy in a new process
        process = subprocess.Popen([sys.executable, "supertrend_live.py"], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Log output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Log errors
        for error in process.stderr:
            logger.error(error.strip())
        
        # Get return code
        return_code = process.poll()
        logger.info(f"Strategy process exited with code {return_code}")
        
        return process
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
            text=True
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
            text=True
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
        
        # Check if dashboard template exists
        if not Path("templates/dashboard_supertrend.html").exists():
            logger.error("Dashboard template not found. Make sure templates/dashboard_supertrend.html exists.")
            return
        
        # Start HTTP server in a separate thread
        http_thread = threading.Thread(target=start_http_server, daemon=True)
        http_thread.start()
        
        # Start strategy
        strategy_process = start_strategy()
        
        # Start additional analyzers
        pullback_process = start_pullback_analyzer()
        optimized_process = start_optimized_analyzer()
        
        # Open dashboard
        open_dashboard()
        
        # Wait for Ctrl+C
        logger.info("Press Ctrl+C to stop all services...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            
            # Terminate strategy process
            if strategy_process:
                strategy_process.terminate()
                logger.info("Strategy process terminated")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
