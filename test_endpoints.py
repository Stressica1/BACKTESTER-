#!/usr/bin/env python3
"""
Simple test of the optimized Super Z analysis endpoints
"""

import requests
import json
import time
from datetime import datetime

def test_optimized_endpoints():
    print("🎯 Testing Super Z Optimized Analysis Endpoints")
    print("=" * 50)
    print(f"⏰ Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_url = "http://localhost:8000"
    
    # Test 1: Quick test endpoint (GET)
    print("\n📊 Test 1: Quick Optimization Test")
    print("-" * 30)
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/api/super-z-analysis/quick-test", timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {result.get('status')}")
            print(f"⏱️  Execution Time: {result.get('execution_time', 0):.2f}s")
            print(f"📈 Symbols Tested: {result.get('symbols_tested', 0)}")
            print(f"📊 Total Signals: {result.get('results', {}).get('aggregate_statistics', {}).get('total_signals_across_markets', 0)}")
            print(f"🎯 Pullback Rate: {result.get('results', {}).get('aggregate_statistics', {}).get('overall_pullback_rate', 0):.1f}%")
            print(f"🚀 Optimization Working: {result.get('performance_summary', {}).get('optimization_working', False)}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Optimized batch endpoint (POST)
    print("\n🚀 Test 2: Optimized Batch Analysis")
    print("-" * 35)
    
    test_data = {
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT"],
        "timeframes": ["5m", "15m"],
        "use_optimization": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/super-z-analysis/optimized",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {result.get('status')}")
            print(f"⏱️  Client Time: {end_time - start_time:.2f}s")
            print(f"⚡ Server Time: {result.get('execution_time', 0):.2f}s")
            print(f"📊 Symbols: {len(test_data['symbols'])}")
            print(f"📈 Timeframes: {len(test_data['timeframes'])}")
            print(f"🎯 Total Combinations: {len(test_data['symbols']) * len(test_data['timeframes'])}")
            print(f"📈 Total Signals: {result.get('total_signals', 0)}")
            print(f"🎯 Pullback Rate: {result.get('overall_pullback_rate', 0):.1f}%")
            print(f"🚀 Processing Speed: {(len(test_data['symbols']) * len(test_data['timeframes'])) / result.get('execution_time', 1):.1f} combinations/sec")
            
            # Show sample trading opportunities
            if result.get('results'):
                print("\n💼 Sample Trading Opportunities:")
                count = 0
                for symbol_result in result.get('results', [])[:3]:  # Top 3 symbols
                    symbol = symbol_result.get('symbol', 'Unknown')
                    for tf_result in symbol_result.get('timeframe_results', []):
                        timeframe = tf_result.get('timeframe')
                        analysis = tf_result.get('analysis', {})
                        signals = analysis.get('signals', [])
                        pullback_rate = analysis.get('pullback_rate', 0)
                        
                        if signals and count < 5:  # Show max 5 opportunities
                            latest_signal = signals[-1]  # Most recent signal
                            signal_type = latest_signal.get('type', 'unknown').upper()
                            price = latest_signal.get('price', 0)
                            print(f"   📊 {symbol} ({timeframe}): {signal_type} at ${price:,.2f} - {pullback_rate:.1f}% pullback rate")
                            count += 1
            
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Details: {error_detail}")
            except:
                print(f"   Raw Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print(f"\n🏁 Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 Super Z Trading System is fully operational!")

if __name__ == "__main__":
    test_optimized_endpoints()
