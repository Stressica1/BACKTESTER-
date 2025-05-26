import os
import json
import asyncio
import logging
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
from threading import Thread
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the ultra-fast volatility scanner
from volatility_scanner import VolatilityScanner, run_volatility_scan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store the latest volatility data
volatility_cache = {}
last_scan_time = {}

# Initialize the ultra-fast scanner
scanner = VolatilityScanner(
    api_key=os.getenv('BITGET_API_KEY'),
    api_secret=os.getenv('BITGET_API_SECRET'),
    passphrase=os.getenv('BITGET_PASSPHRASE'),
    testnet=False
)

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/volatility')
def get_volatility():
    """API endpoint to get ultra-fast volatility data"""
    timeframe = request.args.get('timeframe', '4h')
    top_n = int(request.args.get('top_n', 20))
    min_volume = float(request.args.get('min_volume', 50000))
    
    # Check if we have cached data for this timeframe
    cache_key = f"{timeframe}_{top_n}_{min_volume}"
    if cache_key in volatility_cache and cache_key in last_scan_time:
        # Check if the cache is less than 2 minutes old (faster refresh for ultra-fast scanner)
        if datetime.now() - last_scan_time[cache_key] < timedelta(minutes=2):
            return jsonify(volatility_cache[cache_key])
    
    # Run the ultra-fast scan to get fresh data
    try:
        async def run_scan():
            return await scanner.scan_all_markets_ultra_fast(
                timeframe=timeframe,
                top_n=top_n,
                min_volume=min_volume
            )
        
        # Run the async scan in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_scan())
        finally:
            loop.close()
        
        # Cache the results
        volatility_cache[cache_key] = results
        last_scan_time[cache_key] = datetime.now()
        
        logger.info(f"Ultra-fast scan completed for {timeframe}, found {len(results)} markets")
        
        # Return the data as JSON
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error running ultra-fast volatility scan: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/volatility/ultra-fast')
def get_volatility_ultra_fast():
    """Ultra-fast API endpoint for real-time volatility scanning"""
    timeframe = request.args.get('timeframe', '4h')
    top_n = int(request.args.get('top_n', 10))
    min_volume = float(request.args.get('min_volume', 25000))
    
    start_time = time.time()
    
    try:
        async def run_ultra_fast_scan():
            return await scanner.scan_all_markets_ultra_fast(
                timeframe=timeframe,
                top_n=top_n,
                min_volume=min_volume
            )
        
        # Run the async scan
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_ultra_fast_scan())
        finally:
            loop.close()
        
        scan_time = time.time() - start_time
        
        response_data = {
            "results": results,
            "meta": {
                "scan_time_seconds": round(scan_time, 3),
                "markets_found": len(results),
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "performance": "ultra-fast" if scan_time < 5 else "fast" if scan_time < 15 else "normal"
            }
        }
        
        logger.info(f"Ultra-fast API scan completed in {scan_time:.3f}s, found {len(results)} markets")
        
        return jsonify(response_data)
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error in ultra-fast scan after {error_time:.3f}s: {str(e)}")
        return jsonify({
            "error": str(e),
            "meta": {
                "scan_time_seconds": round(error_time, 3),
                "timestamp": datetime.now().isoformat()
            }
        }), 500

@app.route('/api/health')
def health_check():
    """API endpoint to check the health of the application"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "balance": "10000.00"  # Mock balance for demo
    })

@socketio.on('connect')
def handle_connect():
    """Handle client WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_success', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription to real-time data"""
    channels = data.get('channels', [])
    logger.info(f"Client {request.sid} subscribed to channels: {channels}")
    
    # Acknowledge subscription
    emit('subscription_success', {
        'channels': channels,
        'message': 'Successfully subscribed to channels'
    })

def background_scanner():
    """Background thread to periodically run the volatility scanner"""
    while True:
        try:
            # Get API credentials from environment
            api_key = os.getenv('BITGET_API_KEY')
            api_secret = os.getenv('BITGET_API_SECRET')
            passphrase = os.getenv('BITGET_PASSPHRASE')
            
            if not all([api_key, api_secret, passphrase]):
                logger.warning("API credentials not configured, skipping volatility scan")
                time.sleep(300)
                continue
              # Run the scanner for all timeframes
            results = run_volatility_scan(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                top_n=20,
                min_volume=100000
            )
            
            # Update the cache
            for timeframe, data in results.items():
                volatility_cache[timeframe] = data
                last_scan_time[timeframe] = datetime.now()
                
                # Emit the new data to connected clients
                socketio.emit('volatility_update', {
                    'type': 'volatility_update',
                    'timeframe': timeframe,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                })
                
            logger.info(f"Volatility scan completed for all timeframes")
        except Exception as e:
            logger.error(f"Error in background scanner: {str(e)}")
        
        # Sleep for 5 minutes before running again
        time.sleep(300)

if __name__ == '__main__':
    # Start the background scanner in a separate thread
    scanner_thread = Thread(target=background_scanner, daemon=True)
    scanner_thread.start()
    
    # Run the Flask application with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
