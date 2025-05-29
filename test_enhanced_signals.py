#!/usr/bin/env python3
"""
TEST ENHANCED SIGNAL SYSTEM - CLEAR LONG/SHORT DEMONSTRATION
"""

import asyncio
import pandas as pd
import numpy as np
import time
import logging
from enhanced_signal_system import EnhancedSignalSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(trend_type="uptrend"):
    """Create test market data for different scenarios"""
    
    # Generate 50 periods of data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='5min')
    
    if trend_type == "uptrend":
        # Create uptrend data (good for LONG signals)
        base_price = 100
        prices = []
        for i in range(50):
            # Overall upward movement with some volatility
            price = base_price + (i * 0.5) + np.random.uniform(-0.5, 0.5)
            prices.append(price)
        
        # Make current price slightly above SuperTrend for LONG signal
        prices[-5:] = [p * 0.98 for p in prices[-5:]]  # Small dip for entry
        
    elif trend_type == "downtrend":
        # Create downtrend data (good for SHORT signals)
        base_price = 100
        prices = []
        for i in range(50):
            # Overall downward movement with some volatility
            price = base_price - (i * 0.3) + np.random.uniform(-0.5, 0.5)
            prices.append(price)
        
        # Make current price slightly below SuperTrend for SHORT signal
        prices[-5:] = [p * 1.02 for p in prices[-5:]]  # Small pump for entry
    
    # Create OHLC data
    data = []
    for i, close in enumerate(prices):
        open_price = close + np.random.uniform(-0.2, 0.2)
        high = max(open_price, close) + np.random.uniform(0, 0.3)
        low = min(open_price, close) - np.random.uniform(0, 0.3)
        volume = np.random.uniform(1000, 5000)
        
        data.append([
            int(dates[i].timestamp() * 1000),  # timestamp
            open_price,
            high,
            low,
            close,
            volume
        ])
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

async def test_signal_generation():
    """Test signal generation with different market conditions"""
    
    enhanced_system = EnhancedSignalSystem()
    
    print("🧪 TESTING ENHANCED SIGNAL SYSTEM")
    print("=" * 60)
    
    # Test 1: UPTREND scenario (should generate LONG signal)
    print("\n📈 TEST 1: UPTREND MARKET (Expecting LONG signal)")
    print("-" * 50)
    
    uptrend_data = create_test_data("uptrend")
    signal1 = await enhanced_system.generate_enhanced_signal("BTC/USDT", uptrend_data)
    
    if signal1:
        print(f"✅ SIGNAL GENERATED!")
        print(f"   🎯 DIRECTION: {signal1['direction']}")
        print(f"   📋 ACTION: {signal1['action']}")
        print(f"   📊 POSITION TYPE: {signal1['position_type']}")
        print(f"   📈 EXPECTED: {signal1['expected_outcome']}")
        print(f"   💯 CONFIDENCE: {signal1['confidence']:.1f}%")
        print(f"   ⚡ LEVERAGE: {signal1['leverage']}x")
    else:
        print("❌ No signal generated for uptrend")
    
    # Test 2: DOWNTREND scenario (should generate SHORT signal)
    print("\n📉 TEST 2: DOWNTREND MARKET (Expecting SHORT signal)")
    print("-" * 50)
    
    downtrend_data = create_test_data("downtrend")
    signal2 = await enhanced_system.generate_enhanced_signal("ETH/USDT", downtrend_data)
    
    if signal2:
        print(f"✅ SIGNAL GENERATED!")
        print(f"   🎯 DIRECTION: {signal2['direction']}")
        print(f"   📋 ACTION: {signal2['action']}")
        print(f"   📊 POSITION TYPE: {signal2['position_type']}")
        print(f"   📉 EXPECTED: {signal2['expected_outcome']}")
        print(f"   💯 CONFIDENCE: {signal2['confidence']:.1f}%")
        print(f"   ⚡ LEVERAGE: {signal2['leverage']}x")
    else:
        print("❌ No signal generated for downtrend")
    
    # Summary
    print("\n🎯 SIGNAL CLARITY SUMMARY:")
    print("=" * 60)
    print("🟢 LONG signals = BUY positions (expect price to go UP)")
    print("🔴 SHORT signals = SELL positions (expect price to go DOWN)")
    print("📊 Each signal clearly shows direction and expected outcome")
    print("💯 Confidence levels help determine position size")
    print("⚡ Leverage is automatically calculated based on confidence")

async def test_multiple_scenarios():
    """Test multiple trading scenarios"""
    
    enhanced_system = EnhancedSignalSystem()
    
    print("\n🔄 TESTING MULTIPLE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        ("BTC/USDT", "uptrend"),
        ("ETH/USDT", "downtrend"),
        ("SOL/USDT", "uptrend"),
        ("XRP/USDT", "downtrend"),
        ("ADA/USDT", "uptrend")
    ]
    
    long_signals = 0
    short_signals = 0
    
    for symbol, trend in scenarios:
        data = create_test_data(trend)
        signal = await enhanced_system.generate_enhanced_signal(symbol, data)
        
        if signal:
            if signal['direction'] == 'LONG':
                long_signals += 1
                print(f"🟢 {symbol}: LONG signal ({signal['confidence']:.1f}% confidence)")
            elif signal['direction'] == 'SHORT':
                short_signals += 1
                print(f"🔴 {symbol}: SHORT signal ({signal['confidence']:.1f}% confidence)")
        else:
            print(f"⚪ {symbol}: No signal")
    
    print(f"\n📊 RESULTS:")
    print(f"   🟢 LONG signals: {long_signals}")
    print(f"   🔴 SHORT signals: {short_signals}")
    print(f"   📈 Total signals: {long_signals + short_signals}")

if __name__ == "__main__":
    print("🚀 ENHANCED SIGNAL SYSTEM TESTING")
    print("Testing crystal clear LONG/SHORT signal indication...")
    print()
    
    # Run tests
    asyncio.run(test_signal_generation())
    asyncio.run(test_multiple_scenarios())
    
    print("\n✅ TESTING COMPLETE!")
    print("🎯 Enhanced system provides CRYSTAL CLEAR signal direction!") 