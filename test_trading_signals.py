#!/usr/bin/env python3
"""
Test script for Super Z Trading Signal System
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from super_z_trading_signals import SuperZTradingSignals

async def test_trading_signals():
    print('🎯 Super Z Trading Signal System Test')
    print('=' * 50)
    
    try:
        # Initialize signal system
        signal_system = SuperZTradingSignals()
        print('✅ Trading signal system initialized')
        
        # Test signal detection
        print('\n📡 Testing signal detection on BTC/USDT...')
        signals = await signal_system.detect_signals(['BTC/USDT'], ['5m'])
        
        print(f'✅ Found {len(signals)} trading signals')
        
        if signals:
            for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                print(f'\n📊 Signal #{i+1}:')
                print(f'   Symbol: {signal["symbol"]}')
                print(f'   Timeframe: {signal["timeframe"]}')
                print(f'   Action: {signal["action"]}')
                print(f'   Price: ${signal["price"]:,.2f}')
                print(f'   Pullback Probability: {signal["pullback_probability"]:.1f}%')
                print(f'   Stop Loss: ${signal["stop_loss"]:,.2f}')
                print(f'   Take Profit: ${signal["take_profit"]:,.2f}')
                print(f'   Risk/Reward: {signal["risk_reward_ratio"]:.2f}')
        else:
            print('   No signals found at this time')
        
        # Test market scan
        print('\n🔍 Testing market scan...')
        pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        scan_results = await signal_system.scan_market(pairs, ['5m'])
        
        print(f'✅ Market scan completed: {len(scan_results)} opportunities found')
        
        if scan_results:
            print('\n🎯 Top opportunities:')
            for result in scan_results[:5]:  # Top 5 opportunities
                print(f'   {result["symbol"]}: {result["action"]} - {result["pullback_probability"]:.1f}% pullback prob')
        
        print('\n🎉 Trading signal system test completed successfully!')
        return True
        
    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_trading_signals())
    if success:
        print('\n✅ All trading signal tests passed!')
    else:
        print('\n❌ Trading signal tests failed!')
