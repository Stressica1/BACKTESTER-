import os
import socket
import uvicorn
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_free_port():
    """Find a free port to run the server on"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_ngrok_tunnel(port):
    """Start an ngrok tunnel to the specified port"""
    try:
        # Try to import ngrok
        from pyngrok import ngrok
        
        # Set ngrok auth token if provided
        ngrok_token = os.getenv('NGROK_AUTH_TOKEN')
        if not ngrok_token:
            logger.warning("No NGROK_AUTH_TOKEN found in environment variables")
            return None
            
        try:
            ngrok.set_auth_token(ngrok_token)
        except Exception as e:
            logger.error(f"Error setting ngrok auth token: {str(e)}")
            return None
        
        # Start the tunnel
        try:
            tunnel = ngrok.connect(port)
            webhook_url = tunnel.public_url
            logger.info(f"Ngrok tunnel established at: {webhook_url}")
            return webhook_url
        except Exception as e:
            logger.error(f"Error starting ngrok tunnel: {str(e)}")
            return None
            
    except ImportError:
        logger.warning("pyngrok not installed. Running without ngrok tunnel.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with ngrok: {str(e)}")
        return None

def main():
    try:
        # Find a free port
        port = find_free_port()
        logger.info(f"Starting server on port: {port}")
        
        # Start ngrok tunnel
        webhook_url = start_ngrok_tunnel(port)
        if not webhook_url:
            logger.warning("Failed to start ngrok tunnel, using localhost only")
            webhook_url = f"http://localhost:{port}"
        
        # Set environment variables
        os.environ['PORT'] = str(port)
        os.environ['NGROK_URL'] = webhook_url
        
        logger.info(f"Server will be accessible at: {webhook_url}")
        
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            reload_dirs=["."],
            reload_delay=0.5,
            log_level="info",
            access_log=True,
            workers=1
        )
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 