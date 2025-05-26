import asyncio
import logging
from volatility_scanner import VolatilityScanner
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_scanner():
    # Initialize scanner with API credentials
    scanner = VolatilityScanner(
        api_key=os.getenv('BITGET_API_KEY'),
        api_secret=os.getenv('BITGET_API_SECRET'),
        passphrase=os.getenv('BITGET_PASSPHRASE'),
        testnet=False  # Use live API
    )
    
    # Test with a small number of pairs
    results = await scanner.scan_all_markets(timeframe='4h', top_n=5, min_volume=100000)
    
    print(f'\nScanner completed successfully!')
    print(f'Found {len(results)} results:')
    for result in results:
        print(f"  {result['symbol']}: Appeal Score = {result.get('appeal_score', 'N/A'):.3f}, Vol = {result.get('daily_volatility', 'N/A'):.2f}%")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_scanner())
