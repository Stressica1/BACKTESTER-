"""
BITGET API CREDENTIALS SETUP HELPER
This script helps you set up proper API credentials for live trading
"""

import os
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)

def setup_bitget_credentials():
    print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}🚨 BITGET API CREDENTIALS SETUP 🚨{Style.RESET_ALL}")
    print(f"{Back.YELLOW}{Fore.BLACK}⚠️  IMPORTANT: You need MAINNET API credentials for live trading! ⚠️{Style.RESET_ALL}")
    print()
    
    print("To get your Bitget API credentials:")
    print("1. Go to https://www.bitget.com/api")
    print("2. Login to your Bitget account")
    print("3. Go to API Management")
    print("4. Create a new API key with:")
    print("   - Trading permissions enabled")
    print("   - Futures trading enabled")
    print("   - IP whitelist (recommended)")
    print("5. Copy the API Key, Secret Key, and Passphrase")
    print()
    
    print(f"{Back.CYAN}{Fore.BLACK}Current .env file configuration:{Style.RESET_ALL}")
    
    # Read current .env
    try:
        with open('.env', 'r') as f:
            content = f.read()
            
        if 'BITGET_API_KEY=' in content:
            print("✅ BITGET_API_KEY is set")
        else:
            print("❌ BITGET_API_KEY is missing")
            
        if 'BITGET_API_SECRET=' in content:
            print("✅ BITGET_API_SECRET is set")
        else:
            print("❌ BITGET_API_SECRET is missing")
            
        if 'BITGET_PASSPHRASE=' in content:
            print("✅ BITGET_PASSPHRASE is set")
        else:
            print("❌ BITGET_PASSPHRASE is missing")
            
        if 'BITGET_TESTNET=false' in content:
            print("✅ MAINNET mode enabled")
        elif 'BITGET_TESTNET=true' in content:
            print(f"{Back.YELLOW}{Fore.BLACK}⚠️  TESTNET mode enabled - switch to false for live trading{Style.RESET_ALL}")
        else:
            print("❌ BITGET_TESTNET setting missing")
            
    except FileNotFoundError:
        print(f"{Back.RED}{Fore.WHITE}❌ .env file not found!{Style.RESET_ALL}")
        return False
    
    print()
    print(f"{Back.GREEN}{Fore.BLACK}To fix your credentials:{Style.RESET_ALL}")
    print("1. Edit the .env file")
    print("2. Replace the API credentials with your MAINNET credentials")
    print("3. Set BITGET_TESTNET=false for live trading")
    print("4. Save the file and restart the bot")
    print()
    
    # Test with dummy credentials to show the format
    print(f"{Back.MAGENTA}{Fore.WHITE}Example .env format:{Style.RESET_ALL}")
    print("BITGET_API_KEY=bg_your_actual_api_key_here")
    print("BITGET_API_SECRET=your_actual_secret_key_here") 
    print("BITGET_PASSPHRASE=your_actual_passphrase_here")
    print("BITGET_TESTNET=false")
    print()
    
    return True

if __name__ == "__main__":
    setup_bitget_credentials()
    
    print(f"{Back.CYAN}{Fore.BLACK}After setting up credentials, run:{Style.RESET_ALL}")
    print("python test_live_trading_diagnostic.py")
    print("to verify everything works!") 