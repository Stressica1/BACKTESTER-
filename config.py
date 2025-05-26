import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # Bitget API Keys
    BITGET_API_KEY = os.getenv('BITGET_API_KEY')
    BITGET_API_SECRET = os.getenv('BITGET_API_SECRET')
    BITGET_PASSPHRASE = os.getenv('BITGET_PASSPHRASE')
    BITGET_TESTNET = os.getenv('BITGET_TESTNET', 'true').lower() == 'true'
    
    # Legacy API Keys (for backwards compatibility)
    TESTNET_API_KEY = os.getenv('TESTNET_API_KEY')
    TESTNET_API_SECRET = os.getenv('TESTNET_API_SECRET')
    MAINNET_API_KEY = os.getenv('MAINNET_API_KEY')
    MAINNET_API_SECRET = os.getenv('MAINNET_API_SECRET')
    
    # Ngrok Auth Token
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'testnet')
    
    @classmethod
    def get_api_key(cls):
        # Prioritize Bitget credentials
        if cls.BITGET_API_KEY:
            return cls.BITGET_API_KEY
        # Fallback to legacy credentials
        return cls.TESTNET_API_KEY if cls.ENVIRONMENT == 'testnet' else cls.MAINNET_API_KEY
    
    @classmethod
    def get_api_secret(cls):
        # Prioritize Bitget credentials
        if cls.BITGET_API_SECRET:
            return cls.BITGET_API_SECRET
        # Fallback to legacy credentials
        return cls.TESTNET_API_SECRET if cls.ENVIRONMENT == 'testnet' else cls.MAINNET_API_SECRET
    
    @classmethod
    def get_passphrase(cls):
        return cls.BITGET_PASSPHRASE
    
    @classmethod
    def is_testnet(cls):
        # Use Bitget testnet setting if available, otherwise fallback to environment
        if cls.BITGET_API_KEY:
            return cls.BITGET_TESTNET
        return cls.ENVIRONMENT == 'testnet'
    
    @staticmethod
    def get_webhook_secret():
        return os.getenv('WEBHOOK_SECRET', 'your-webhook-secret')
    
    @staticmethod
    def get_allowed_symbols():
        return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    @staticmethod
    def get_max_position_size():
        return float(os.getenv('MAX_POSITION_SIZE', '1.0'))
    
    @staticmethod
    def get_risk_per_trade():
        return float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% risk per trade 