"""
LIVE TRADING DIAGNOSTIC SCRIPT
Tests each component of the live trading system step by step
"""

import asyncio
import logging
import pandas as pd
from super_z_pullback_analyzer import SuperZPullbackAnalyzer
from datetime import datetime
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('diagnostic.log'),
        logging.StreamHandler()
    ]
)

async def test_live_trading_components():
    """Test each component of live trading system"""
    
    print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}🚨 LIVE TRADING DIAGNOSTIC TEST 🚨{Style.RESET_ALL}")
    
    # Step 1: Initialize analyzer
    print(f"{Back.CYAN}{Fore.BLACK}Step 1: Initializing SuperZPullbackAnalyzer...{Style.RESET_ALL}")
    try:
        analyzer = SuperZPullbackAnalyzer()
        print(f"{Back.GREEN}{Fore.BLACK}✅ Analyzer initialized successfully{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Failed to initialize: {e}{Style.RESET_ALL}")
        return False
    
    # Step 2: Test Bitget connection
    print(f"{Back.CYAN}{Fore.BLACK}Step 2: Testing Bitget connection...{Style.RESET_ALL}")
    try:
        balance = analyzer.exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)
        print(f"{Back.GREEN}{Fore.BLACK}✅ Connected to Bitget! USDT Balance: {usdt_balance:.2f}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Bitget connection failed: {e}{Style.RESET_ALL}")
        return False
    
    # Step 3: Test data fetching
    print(f"{Back.CYAN}{Fore.BLACK}Step 3: Testing data fetching for BTC/USDT:USDT...{Style.RESET_ALL}")
    try:
        df = await analyzer.fetch_data('BTC/USDT:USDT', '5m', days=1)
        if df.empty:
            print(f"{Back.RED}{Fore.WHITE}❌ No data received{Style.RESET_ALL}")
            return False
        print(f"{Back.GREEN}{Fore.BLACK}✅ Data fetched! Candles: {len(df)}, Latest: {df.index[-1]}{Style.RESET_ALL}")
        print(f"Latest BTC price: {df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Data fetching failed: {e}{Style.RESET_ALL}")
        return False
    
    # Step 4: Test signal detection
    print(f"{Back.CYAN}{Fore.BLACK}Step 4: Testing signal detection...{Style.RESET_ALL}")
    try:
        # Test with minimal data for speed
        test_df = df.tail(100)  # Only last 100 candles
        signals, df_with_indicators = analyzer.detect_signals(test_df, timeframe='5m')
        print(f"{Back.GREEN}{Fore.BLACK}✅ Signal detection working! Found {len(signals)} signals{Style.RESET_ALL}")
        
        if signals:
            latest_signal = signals[-1]
            print(f"Latest signal: {latest_signal['type']} @ {latest_signal['price']:.2f} at {latest_signal['time']}")
            print(f"Signal confidence: {latest_signal['confidence']:.1f}")
        
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Signal detection failed: {e}{Style.RESET_ALL}")
        return False
    
    # Step 5: Test scoring system
    print(f"{Back.CYAN}{Fore.BLACK}Step 5: Testing scoring system...{Style.RESET_ALL}")
    try:
        if signals:
            score, breakdown, datapoints, normalized = analyzer.enhanced_score_all_data('BTC/USDT:USDT', df_with_indicators, latest_signal)
            print(f"{Back.GREEN}{Fore.BLACK}✅ Scoring system working! Score: {score:.1f}/100{Style.RESET_ALL}")
            print(f"Score breakdown: {breakdown}")
        else:
            print(f"{Back.YELLOW}{Fore.BLACK}⚠️ No signals to score{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Scoring failed: {e}{Style.RESET_ALL}")
        return False
    
    # Step 6: Test order sizing
    print(f"{Back.CYAN}{Fore.BLACK}Step 6: Testing order sizing...{Style.RESET_ALL}")
    try:
        current_price = df['close'].iloc[-1]
        can_trade, amount = analyzer.can_place_trade('BTC/USDT:USDT', current_price)
        print(f"{Back.GREEN}{Fore.BLACK}✅ Order sizing working! Can trade: {can_trade}, Amount: {amount:.6f}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Order sizing failed: {e}{Style.RESET_ALL}")
        return False
    
    # Step 7: Test live signal generation on current data
    print(f"{Back.CYAN}{Fore.BLACK}Step 7: Testing LIVE signal generation...{Style.RESET_ALL}")
    try:
        # Fetch the most recent data
        fresh_df = await analyzer.fetch_data('BTC/USDT:USDT', '5m', days=1)
        fresh_signals, fresh_indicators = analyzer.detect_signals(fresh_df.tail(50), timeframe='5m')
        
        print(f"{Back.GREEN}{Fore.BLACK}✅ Live signal generation working!{Style.RESET_ALL}")
        
        if fresh_signals:
            latest_fresh = fresh_signals[-1]
            is_current = latest_fresh['time'] == fresh_indicators.index[-1]
            print(f"Latest signal time: {latest_fresh['time']}")
            print(f"Latest candle time: {fresh_indicators.index[-1]}")
            print(f"Signal is on current candle: {is_current}")
            
            if is_current:
                print(f"{Back.GREEN}{Fore.BLACK}🚀 CURRENT CANDLE SIGNAL DETECTED! This would trigger a live trade!{Style.RESET_ALL}")
            else:
                print(f"{Back.YELLOW}{Fore.BLACK}⚠️ Signal is historical, not on current candle{Style.RESET_ALL}")
        else:
            print(f"{Back.YELLOW}{Fore.BLACK}⚠️ No current signals found{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}❌ Live signal generation failed: {e}{Style.RESET_ALL}")
        return False
    
    print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}✅ ALL DIAGNOSTIC TESTS PASSED! LIVE TRADING SYSTEM IS READY!{Style.RESET_ALL}")
    return True

async def run_single_scan_test():
    """Run a single scan cycle to prove live trading works"""
    
    print(f"{Back.MAGENTA}{Fore.WHITE}{Style.BRIGHT}🔥 SINGLE SCAN TEST - PROVING LIVE TRADING WORKS 🔥{Style.RESET_ALL}")
    
    analyzer = SuperZPullbackAnalyzer()
    
    # Test symbols - major pairs only
    symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    for symbol in symbols:
        print(f"{Back.CYAN}{Fore.BLACK}Testing {symbol}...{Style.RESET_ALL}")
        
        try:
            # Get fresh data
            df = await analyzer.fetch_data(symbol, '5m', days=1)
            if df.empty:
                continue
                
            # Get signals
            signals, df_with_indicators = analyzer.detect_signals(df.tail(50), timeframe='5m')
            
            if signals:
                latest_signal = signals[-1]
                score, breakdown, datapoints, normalized = analyzer.enhanced_score_all_data(symbol, df_with_indicators, latest_signal)
                
                is_current = latest_signal['time'] == df_with_indicators.index[-1]
                
                print(f"  Signal: {latest_signal['type']} @ {latest_signal['price']:.2f}")
                print(f"  Score: {score:.1f}/100")
                print(f"  Current candle: {is_current}")
                
                if is_current and score >= 85:
                    print(f"{Back.GREEN}{Fore.BLACK}🚀 WOULD PLACE LIVE ORDER: {symbol} {latest_signal['type']} @ {latest_signal['price']:.2f}{Style.RESET_ALL}")
                    
                    # Test order sizing
                    can_trade, amount = analyzer.can_place_trade(symbol, latest_signal['price'])
                    print(f"  Order details: Can trade: {can_trade}, Amount: {amount:.6f}")
                    
                    if can_trade:
                        print(f"{Back.RED}{Fore.WHITE}💰 LIVE ORDER READY! (Not executing in test mode){Style.RESET_ALL}")
                else:
                    print(f"  Not meeting criteria for live trade")
            else:
                print(f"  No signals found")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}✅ SINGLE SCAN TEST COMPLETE!{Style.RESET_ALL}")

if __name__ == "__main__":
    print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}🚨 STARTING LIVE TRADING DIAGNOSTIC 🚨{Style.RESET_ALL}")
    
    # Run diagnostic tests
    success = asyncio.run(test_live_trading_components())
    
    if success:
        # Run single scan test
        asyncio.run(run_single_scan_test())
    else:
        print(f"{Back.RED}{Fore.WHITE}❌ DIAGNOSTIC FAILED - LIVE TRADING NOT READY{Style.RESET_ALL}") 