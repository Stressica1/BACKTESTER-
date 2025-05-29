import asyncio
import sys
sys.path.append('.')
from supertrend_pullback_live import SuperTrendBot

async def test_signals():
    print('🔍 TESTING SIGNAL GENERATION...')
    bot = SuperTrendBot()
    
    # Test a few symbols
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT']
    
    print('\n📊 TESTING SIGNAL GENERATION FOR MULTIPLE SYMBOLS:')
    print('='*60)
    
    signals_found = 0
    for symbol in test_symbols:
        try:
            signal = await bot.generate_signal_fast(symbol)
            if signal:
                print(f'✅ SIGNAL: {symbol} | {signal["side"].upper()} | {signal["confidence"]:.1f}% | {signal["reason"]}')
                signals_found += 1
            else:
                print(f'❌ NO SIGNAL: {symbol}')
        except Exception as e:
            print(f'⚠️ ERROR: {symbol} - {e}')
    
    print(f'\n📈 RESULTS: {signals_found}/{len(test_symbols)} symbols generated signals')
    
    if signals_found > 0:
        print('✅ SIGNAL GENERATION IS WORKING!')
    else:
        print('❌ NO SIGNALS GENERATED - Need to adjust thresholds')

if __name__ == "__main__":
    asyncio.run(test_signals()) 