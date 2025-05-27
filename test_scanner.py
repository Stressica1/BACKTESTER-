import asyncio
import logging
from volatility_scanner import VolatilityScanner, EnhancedMarketRanker
import os
from dotenv import load_dotenv
import ccxt
import time
import json
import csv

# Load environment variables
load_dotenv()

# Configure logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scanner.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("ScannerTest")

# Strict buy threshold for signals
STRICT_BUY_THRESHOLD = 70

def print_summary_table(results, ranker, tf=None):
    # Print a simple summary table for the top 20 results
    print("\n=== TOP 20 SIGNALS ===")
    print(f"{'#':<3} {'Symbol':<15} {'TF':<5} {'Signal':<6} {'Price':^12} {'Score':^6} {'Tier':<22} {'Volume':^12} {'Lev':^4} {'Filters':<20}")
    print("-"*120)
    for i, res in enumerate(results, 1):
        symbol = res.get('symbol', '-')
        timeframe = tf if tf else res.get('timeframe', '-')
        trend = res.get('mtf_trend', '-')
        signal = 'BUY' if trend == 'bullish' else 'SELL' if trend == 'bearish' else 'HOLD'
        price = res.get('last_price', res.get('close', '-'))
        score = res.get('score', '-')
        tier = res.get('tier', '-')
        volume = f"{res.get('volume', 0):,.0f}"
        lev = res.get('leverage', '-')
        filters = []
        if res.get('volume', 0) > 1_000_000:
            filters.append('HighVol')
        if res.get('daily_volatility', 0) > 5:
            filters.append('HighVolatility')
        if res.get('rsi', 50) > 70:
            filters.append('RSI>70')
        if res.get('rsi', 50) < 30:
            filters.append('RSI<30')
        filters_str = ', '.join(filters) if filters else '-'
        if score != '-' and tier != '-':
            print(f"{i:<3} {symbol:<15} {timeframe:<5} {signal:<6} {price:^12.4f} {score:^6} {tier:<22} {volume:^12} {lev:^4} {filters_str:<20}")
    print("-"*120)

async def test_scanner(suppress_ohlcv=False):
    # Check Bitget credentials
    api_key = os.getenv('BITGET_API_KEY')
    api_secret = os.getenv('BITGET_API_SECRET')
    passphrase = os.getenv('BITGET_PASSPHRASE')
    if not api_key or not api_secret or not passphrase:
        logger.warning("Bitget API credentials are missing or incomplete! Please check your .env file.")
    
    # Initialize scanner with API credentials
    scanner = VolatilityScanner(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=passphrase,
        testnet=False  # Use live API
    )
    ranker = EnhancedMarketRanker()

    timeframes = ['5m', '15m', '30m', '1h']
    all_results = {}
    for tf in timeframes:
        start_time = time.time()
        results = await scanner.scan_all_markets(timeframe=tf, top_n=20, min_volume=100000)
        elapsed = time.time() - start_time
        logger.info(f'Scanner for {tf} completed successfully in {elapsed:.2f} seconds!')
        logger.info(f'Found {len(results)} results for {tf}:')
        all_results[tf] = results
        # Export results to JSON and CSV per timeframe
        with open(f'scanner_results_{tf}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Exported results to scanner_results_{tf}.json')
        if results:
            keys = results[0].keys()
            with open(f'scanner_results_{tf}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            logger.info(f'Exported results to scanner_results_{tf}.csv')
        # Print summary table
        print(f"\n{'='*20} {tf.upper()} TIMEFRAME {'='*20}")
        print_summary_table(results, ranker, tf)

    # Log failed pairs if any (from scanner's cache)
    failed_pairs = []
    if hasattr(scanner, 'data_cache'):
        for k, v in scanner.data_cache.items():
            if v is None or (isinstance(v, tuple) and v[1] is None):
                failed_pairs.append(k)
    if failed_pairs:
        logger.warning(f"Failed to fetch data for {len(failed_pairs)} pairs: {failed_pairs}")
    else:
        logger.info("No failed pairs detected.")

    # Try the minimal fetch (optional print)
    exchange = ccxt.bitget({
        'apiKey': api_key,
        'secret': api_secret,
        'password': passphrase,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=10)
    if not suppress_ohlcv:
        print(ohlcv)
    logger.info('Fetched sample OHLCV for BTC/USDT:USDT (5m)')

if __name__ == "__main__":
    asyncio.run(test_scanner(suppress_ohlcv=False))

# BUSSIED!!!!!
