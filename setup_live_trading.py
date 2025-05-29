#!/usr/bin/env python3
"""
Secure Setup Script for Live Trading Configuration
"""
import json
import os
import getpass
from pathlib import Path

def setup_live_trading():
    """Setup Bitget API configuration for live trading"""
    
    print("🔧 BITGET LIVE TRADING SETUP")
    print("=" * 50)
    print("This will help you securely configure your API credentials")
    print("=" * 50)
    
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "bitget_config.json"
    
    # Check if config already exists
    if config_file.exists():
        print(f"\n⚠️  Configuration file already exists at: {config_file}")
        overwrite = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if overwrite != 'yes':
            print("❌ Setup cancelled.")
            return
    
    print("\n📋 Enter your Bitget API credentials:")
    print("(These will be saved locally in config/bitget_config.json)")
    print()
    
    # Get API credentials
    api_key = input("API Key: ").strip()
    secret = getpass.getpass("Secret Key (hidden): ").strip()
    passphrase = getpass.getpass("Passphrase (hidden): ").strip()
    
    # Ask about sandbox mode
    print("\n🏖️  Sandbox Mode:")
    print("1. Production (Real Money)")
    print("2. Sandbox (Test Environment)")
    mode = input("Select mode (1 or 2): ").strip()
    
    sandbox = mode == '2'
    
    # Create configuration
    config = {
        "api_key": api_key,
        "secret": secret,
        "passphrase": passphrase,
        "sandbox": sandbox
    }
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n✅ Configuration saved to: {config_file}")
    
    # Set file permissions (Windows doesn't support chmod, but we can try)
    try:
        import stat
        os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)  # Read/write for owner only
        print("✅ File permissions set (owner read/write only)")
    except:
        print("⚠️  Could not set file permissions (normal on Windows)")
    
    # Test configuration
    print("\n🧪 Testing configuration...")
    test_config()

def test_config():
    """Test the configuration by checking exchange connection"""
    try:
        import ccxt
        
        # Load config
        with open('config/bitget_config.json', 'r') as f:
            config = json.load(f)
        
        # Create exchange instance
        exchange_class = ccxt.bitget
        exchange_params = {
            'apiKey': config['api_key'],
            'secret': config['secret'],
            'password': config['passphrase'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        }
        
        if config.get('sandbox'):
            exchange_params['urls'] = {
                'api': {
                    'public': 'https://api-sandbox.bitget.com',
                    'private': 'https://api-sandbox.bitget.com',
                }
            }
            print("🏖️  Using SANDBOX mode")
        else:
            print("💰 Using PRODUCTION mode")
        
        exchange = exchange_class(exchange_params)
        
        # Test connection
        print("Testing API connection...")
        balance = exchange.fetch_balance()
        
        # Show USDT balance
        usdt_balance = balance.get('USDT', {})
        free_usdt = usdt_balance.get('free', 0)
        
        print(f"\n✅ CONNECTION SUCCESSFUL!")
        print(f"💵 USDT Balance: ${free_usdt:.2f}")
        
        if free_usdt < 100:
            print(f"⚠️  WARNING: Low balance. Recommended minimum: $100 USDT")
        
        print("\n🚀 Your bot is ready for live trading!")
        print("Run: python supertrend_pullback_live.py")
        print("Select option 2 for Live Trading")
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {str(e)}")
        print("\nPlease check:")
        print("1. API credentials are correct")
        print("2. API has trading permissions enabled")
        print("3. Your IP is whitelisted on Bitget")
        print("4. You have internet connection")

if __name__ == "__main__":
    setup_live_trading() 