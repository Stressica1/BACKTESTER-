"""
Test script for optimized Super Z analysis
Tests 20 pairs across 1m, 5m, 15m timeframes with 200% speed improvement
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedAnalysisTest:
    """Test harness for optimized Super Z analysis"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_quick_optimization(self):
        """Test quick optimization with 5 symbols"""
        logger.info("🚀 Testing quick optimization (5 symbols x 3 timeframes)")
        
        try:
            start_time = time.time()
            
            async with self.session.get(f"{self.base_url}/api/super-z-analysis/quick-test") as response:
                if response.status == 200:
                    result = await response.json()
                    execution_time = time.time() - start_time
                    
                    print(f"\n✅ Quick Test Results:")
                    print(f"   Execution Time: {execution_time:.2f} seconds")
                    print(f"   Server Reported Time: {result.get('execution_time', 'N/A'):.2f} seconds")
                    print(f"   Symbols Tested: {result.get('symbols_tested', 0)}")
                    print(f"   Timeframes Tested: {result.get('timeframes_tested', 0)}")
                    print(f"   Total Combinations: {result.get('symbols_tested', 0) * result.get('timeframes_tested', 0)}")
                    
                    if result.get('status') == 'success':
                        perf = result.get('performance_summary', {})
                        print(f"   Avg Time per Combination: {perf.get('avg_time_per_combination', 0):.3f} seconds")
                        print(f"   Optimization Working: {perf.get('optimization_working', False)}")
                        print(f"   Estimated Full Test Time: {perf.get('estimated_full_test_time', 0):.1f} seconds")
                        
                        # Analyze results
                        analysis_results = result.get('results', {})
                        if 'aggregate_statistics' in analysis_results:
                            stats = analysis_results['aggregate_statistics']
                            print(f"\n📊 Analysis Results:")
                            print(f"   Total Signals: {stats.get('total_signals_across_markets', 0)}")
                            print(f"   Total Pullbacks: {stats.get('total_pullback_events', 0)}")
                            print(f"   Overall Pullback Rate: {stats.get('overall_pullback_rate', 0):.1f}%")
                            print(f"   Hypothesis Confirmed: {stats.get('hypothesis_confirmed', False)}")
                        
                        return True
                    else:
                        print(f"❌ Quick test failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Quick test error: {e}")
            return False
    
    async def test_full_optimization(self):
        """Test full optimization with 20 symbols across 3 timeframes"""
        logger.info("🚀 Testing full optimization (20 symbols x 3 timeframes)")
        
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
            "SOL/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT",
            "ATOM/USDT", "NEAR/USDT", "ALGO/USDT", "FTM/USDT", "ONE/USDT",
            "LTC/USDT", "BCH/USDT", "ETC/USDT", "DOGE/USDT", "SHIB/USDT"
        ]
        
        timeframes = ["1m", "5m", "15m"]
        
        payload = {
            "symbols": symbols,
            "timeframes": timeframes,
            "days": 30,
            "max_concurrent": 20
        }
        
        try:
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/super-z-analysis/optimized",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    execution_time = time.time() - start_time
                    
                    print(f"\n✅ Full Optimization Test Results:")
                    print(f"   Total Execution Time: {execution_time:.2f} seconds")
                    print(f"   Symbols Processed: {len(symbols)}")
                    print(f"   Timeframes Processed: {len(timeframes)}")
                    print(f"   Total Combinations: {len(symbols) * len(timeframes)}")
                    print(f"   Avg Time per Combination: {execution_time / (len(symbols) * len(timeframes)):.3f} seconds")
                    
                    if result.get('status') == 'success':
                        analysis_results = result.get('analysis_results', {})
                        
                        # Performance metrics
                        perf_metrics = result.get('performance_metrics', {})
                        print(f"\n🚀 Performance Metrics:")
                        print(f"   Speed Improvement: {perf_metrics.get('speed_improvement_estimate', 'N/A')}")
                        print(f"   Concurrent Processing: {perf_metrics.get('concurrent_processing', False)}")
                        print(f"   Optimization Active: {perf_metrics.get('optimization_active', False)}")
                        
                        # Aggregate statistics
                        if 'aggregate_statistics' in analysis_results:
                            stats = analysis_results['aggregate_statistics']
                            print(f"\n📊 Aggregate Analysis Results:")
                            print(f"   Successful Analyses: {stats.get('successful_analyses', 0)}")
                            print(f"   Total Signals: {stats.get('total_signals_across_markets', 0)}")
                            print(f"   Total Pullbacks: {stats.get('total_pullback_events', 0)}")
                            print(f"   Overall Pullback Rate: {stats.get('overall_pullback_rate', 0):.1f}%")
                            print(f"   Hypothesis Confirmed: {stats.get('hypothesis_confirmed', False)}")
                            print(f"   Average Execution Time per Analysis: {stats.get('average_execution_time_per_analysis', 0):.3f} seconds")
                        
                        # Sample some individual results
                        individual_results = analysis_results.get('results', {})
                        print(f"\n📈 Sample Individual Results:")
                        
                        count = 0
                        for symbol, timeframe_results in individual_results.items():
                            if count >= 5:  # Show first 5 symbols
                                break
                            
                            print(f"\n   {symbol}:")
                            for timeframe, result_data in timeframe_results.items():
                                if result_data.get('status') == 'success':
                                    signals = result_data.get('total_signals', 0)
                                    pullbacks = result_data.get('pullback_events', 0)
                                    rate = result_data.get('pullback_rate', 0)
                                    exec_time = result_data.get('execution_time', 0)
                                    print(f"     {timeframe}: {signals} signals, {pullbacks} pullbacks ({rate:.1f}%), {exec_time:.3f}s")
                                else:
                                    print(f"     {timeframe}: Error - {result_data.get('error', 'Unknown')}")
                            count += 1
                        
                        return True
                    else:
                        print(f"❌ Full test failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ HTTP Error {response.status}: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Full test error: {e}")
            return False
    
    async def compare_with_original(self):
        """Compare optimized version with original implementation"""
        logger.info("🔍 Comparing optimized vs original implementation")
        
        test_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        # Test original implementation
        print("\n⏱️  Testing Original Implementation...")
        original_times = []
        
        for symbol in test_symbols:
            try:
                start_time = time.time()
                
                async with self.session.get(
                    f"{self.base_url}/api/super-z-analysis",
                    params={
                        "symbol": symbol,
                        "timeframe": "5m",
                        "days": 30,
                        "st_length": 50,
                        "st_multiplier": 1.0
                    }
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        execution_time = time.time() - start_time
                        original_times.append(execution_time)
                        
                        if result.get('status') == 'success':
                            stats = result.get('results', {}).get('statistics', {})
                            signals = result.get('results', {}).get('total_signals', 0)
                            print(f"   {symbol}: {execution_time:.3f}s, {signals} signals")
                        else:
                            print(f"   {symbol}: Error - {result.get('error', 'Unknown')}")
                    else:
                        print(f"   {symbol}: HTTP Error {response.status}")
                        
            except Exception as e:
                print(f"   {symbol}: Exception - {e}")
        
        # Test optimized implementation  
        print("\n🚀 Testing Optimized Implementation...")
        
        payload = {
            "symbols": test_symbols,
            "timeframes": ["5m"],
            "days": 30,
            "max_concurrent": 10
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/super-z-analysis/optimized",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    optimized_time = time.time() - start_time
                    
                    if result.get('status') == 'success':
                        analysis_results = result.get('analysis_results', {})
                        individual_results = analysis_results.get('results', {})
                        
                        optimized_times = []
                        for symbol in test_symbols:
                            if symbol in individual_results and "5m" in individual_results[symbol]:
                                result_data = individual_results[symbol]["5m"]
                                if result_data.get('status') == 'success':
                                    exec_time = result_data.get('execution_time', 0)
                                    optimized_times.append(exec_time)
                                    signals = result_data.get('total_signals', 0)
                                    print(f"   {symbol}: {exec_time:.3f}s, {signals} signals")
                        
                        print(f"   Total concurrent time: {optimized_time:.3f}s")
                        
                        # Calculate improvement
                        if original_times and optimized_times:
                            avg_original = sum(original_times) / len(original_times)
                            avg_optimized = sum(optimized_times) / len(optimized_times)
                            total_original = sum(original_times)
                            
                            print(f"\n📊 Performance Comparison:")
                            print(f"   Original Average: {avg_original:.3f}s per symbol")
                            print(f"   Optimized Average: {avg_optimized:.3f}s per symbol")
                            print(f"   Original Total: {total_original:.3f}s (sequential)")
                            print(f"   Optimized Total: {optimized_time:.3f}s (concurrent)")
                            print(f"   Speed Improvement: {(total_original / optimized_time):.1f}x faster")
                            print(f"   Time Saved: {total_original - optimized_time:.3f}s")
                            
                            if total_original / optimized_time >= 2.0:
                                print("   ✅ 200% speed improvement achieved!")
                            else:
                                print("   ⚠️  Speed improvement below 200%")
                        
                        return True
                    else:
                        print(f"❌ Optimized test failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Optimized test HTTP Error: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Optimized test error: {e}")
            return False

async def main():
    """Main test function"""
    print("🧪 Super Z Optimized Analysis Test Suite")
    print("=" * 50)
    
    async with OptimizedAnalysisTest() as test_harness:
        # Run tests in sequence
        tests = [
            ("Quick Optimization Test", test_harness.test_quick_optimization),
            ("Performance Comparison", test_harness.compare_with_original),
            ("Full Optimization Test", test_harness.test_full_optimization),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            
            try:
                result = await test_func()
                results[test_name] = result
                
                if result:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n{'='*20} TEST SUMMARY {'='*20}")
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Optimization is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
