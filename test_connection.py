import ccxt
import json

# Load configuration
with open('config/bitget_config.json', 'r') as f:
    config = json.load(f)

# Create exchange instance
try:
    exchange = ccxt.bitget({
        'apiKey': config['api_key'],
        'secret': config['secret'],
        'password': config['passphrase'],
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    
    # Test connection
    print("Testing connection to Bitget...")
    balance = exchange.fetch_balance()
    
    # Get USDT balance
    usdt = balance.get('USDT', {})
    free_balance = float(usdt.get('free', 0))
    
    print(f"\n✅ CONNECTION SUCCESSFUL!")
    print(f"💰 USDT Balance: ${free_balance:.2f}")
    
    if free_balance < 100:
        print(f"⚠️  Warning: Low balance. Recommended minimum: $100")
    else:
        print(f"✨ You're ready for live trading!")
        
except Exception as e:
    print(f"\n❌ Connection failed: {str(e)}")
    print("\nPlease check:")
    print("1. Your API credentials are correct")
    print("2. API has Spot Trading permission enabled")
    print("3. Your IP address is whitelisted on Bitget") 