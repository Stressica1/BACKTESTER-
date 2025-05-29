import ccxt
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)

# Your credentials
api_key = 'bg_33b25387b50e7f874c18ddf34f5cbb14'
api_secret = '4b3cab211d44a155c5cc63dd025fad43025d09155ee6eef3769ef2f6f85c9715'
api_password = '22672267'

print(f"{Back.CYAN}{Fore.BLACK}Testing TESTNET...{Style.RESET_ALL}")
try:
    exchange_testnet = ccxt.bitget({
        'apiKey': api_key,
        'secret': api_secret,
        'password': api_password,
        'sandbox': True,
        'enableRateLimit': True,
    })
    balance = exchange_testnet.fetch_balance()
    print(f"{Back.GREEN}{Fore.BLACK}✅ TESTNET WORKS! USDT: {balance['total'].get('USDT', 0)}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Back.RED}{Fore.WHITE}❌ TESTNET FAILED: {e}{Style.RESET_ALL}")

print(f"{Back.CYAN}{Fore.BLACK}Testing MAINNET...{Style.RESET_ALL}")
try:
    exchange_mainnet = ccxt.bitget({
        'apiKey': api_key,
        'secret': api_secret,
        'password': api_password,
        'sandbox': False,
        'enableRateLimit': True,
    })
    balance = exchange_mainnet.fetch_balance()
    print(f"{Back.GREEN}{Fore.BLACK}✅ MAINNET WORKS! USDT: {balance['total'].get('USDT', 0)}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Back.RED}{Fore.WHITE}❌ MAINNET FAILED: {e}{Style.RESET_ALL}") 