#!/usr/bin/env python3
"""
QUICK TEST: ITERATION 2 Signal Generation (Further Reduced Strictness)
"""

import sys
import asyncio
from supertrend_pullback_live import AggressivePullbackTrader
import time

async def test_iteration2():
    """Test iteration 2 signal generation directly"""
    print("🧪 TESTING ITERATION 2 - FURTHER REDUCED STRICTNESS")
    print("=" * 60)
    
    # Create trader in simulation mode
    trader = AggressivePullbackTrader(simulation_mode=True)
    
    print(f"🎯 Testing {len(trader.active_symbols[:10])} pairs...")
    print("📊 NEW THRESHOLDS:")
    print("   📈 BUY RSI: < 50 (was 45)")
    print("   📉 SELL RSI: > 50 (was 55)")
    print("   🚀 Momentum: 0.0015 (was 0.002)")
    print("   📊 Volume: > 1.0 (was 1.1)")
    print("   🎯 Confidence: ≥ 65% (was 70%)")
    print("-" * 60)
    
    signals_found = 0
    symbols_tested = 0
    
    for symbol in trader.active_symbols[:15]:  # Test first 15 symbols
        try:
            signals_found += 1
            symbols_tested += 1
            
            signal = await trader.generate_signal(symbol)
            
            if signal:
                print(f"✅ SIGNAL: {symbol} {signal['side'].upper()} | "
                     f"Confidence: {signal['confidence']:.1f}% | "
                     f"RSI: {signal['rsi']:.1f} | "
                     f"Momentum: {signal['momentum']:.4f}")
                signals_found += 1
            else:
                print(f"⚪ No signal: {symbol}")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("-" * 60)
    print(f"📊 ITERATION 2 RESULTS:")
    print(f"   🎯 Symbols Tested: {symbols_tested}")
    print(f"   ✅ Signals Found: {signals_found}")
    print(f"   📈 Signal Rate: {(signals_found/symbols_tested*100) if symbols_tested > 0 else 0:.1f}%")
    
    if signals_found == 0:
        print("\n❌ STILL TOO STRICT - NEED ITERATION 3")
    elif signals_found < 3:
        print("\n⚠️ SOME SIGNALS - MAY NEED FURTHER ADJUSTMENT")
    else:
        print("\n✅ GOOD SIGNAL RATE - TESTING COMPLETE")

if __name__ == "__main__":
    asyncio.run(test_iteration2()) 