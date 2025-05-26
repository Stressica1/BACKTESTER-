import asyncio
import requests
import json
import time

# All 20 symbols for comprehensive testing
symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT',
    'SOL/USDT', 'DOT/USDT', 'LINK/USDT', 'AVAX/USDT', 'MATIC/USDT',
    'ATOM/USDT', 'NEAR/USDT', 'ALGO/USDT', 'FTM/USDT', 'ONE/USDT',
    'LTC/USDT', 'BCH/USDT', 'ETC/USDT', 'DOGE/USDT', 'SHIB/USDT'
]

timeframes = ['1m', '5m', '15m']

payload = {
    'symbols': symbols,
    'timeframes': timeframes,
    'days': 30,
    'max_concurrent': 20
}

print('🚀 Starting comprehensive Super Z analysis...')
print(f'📊 Testing {len(symbols)} symbols across {len(timeframes)} timeframes')
print(f'🔄 Total combinations: {len(symbols) * len(timeframes)} analyses')

start_time = time.time()

try:
    response = requests.post(
        'http://localhost:8000/api/super-z-analysis/optimized',
        json=payload,
        timeout=300  # 5 minute timeout
    )
    
    if response.status_code == 200:
        result = response.json()
        execution_time = time.time() - start_time
        
        print(f'✅ Analysis completed in {execution_time:.2f} seconds')
        
        if result.get('status') == 'success':
            stats = result.get('aggregate_statistics', {})
            print(f'📈 FINAL RESULTS:')
            print(f'   Symbols tested: {stats.get("symbols_analyzed", 0)}')
            print(f'   Total signals: {stats.get("total_signals_across_markets", 0)}')
            print(f'   Total pullbacks: {stats.get("total_pullback_events", 0)}')
            print(f'   Pullback rate: {stats.get("overall_pullback_rate", 0):.1f}%')
            print(f'   Hypothesis confirmed: {stats.get("hypothesis_confirmed", False)}')
            print(f'   Avg time per analysis: {stats.get("average_execution_time_per_analysis", 0):.3f}s')
        else:
            print(f'❌ Analysis failed: {result.get("error", "Unknown error")})')
    else:
        print(f'❌ HTTP Error {response.status_code}: {response.text}')
        
except Exception as e:
    print(f'❌ Error: {e}') 