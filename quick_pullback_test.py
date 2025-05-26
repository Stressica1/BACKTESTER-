"""
Quick Super Z Pullback Test
A simple script to quickly test if your Super Z signals are followed by pullbacks
"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickPullbackTest:
    """Quick test for Super Z pullback pattern"""
    
    def __init__(self):
        # Initialize exchange (you may need to add your API credentials)
        self.exchange = ccxt.bitget({
            'enableRateLimit': True,
            'sandbox': False,  # Set to True for testing
        })
        
    def calculate_vhma(self, df: pd.DataFrame, period: int = 21) -> pd.Series:
        """Simplified VHMA calculation"""
        # Volume-weighted close
        vwc = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        
        # Simplified Hull MA (close approximation)
        n2 = period // 2
        wma1 = vwc.rolling(n2).mean()
        wma2 = vwc.rolling(period).mean()
        raw_hma = 2 * wma1 - wma2
        
        # Final smoothing
        sqrt_period = int(np.sqrt(period))
        vhma = raw_hma.rolling(sqrt_period).mean()
        
        return vhma
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 50, multiplier: float = 1.0):
        """Calculate SuperTrend"""
        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # SuperTrend bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # SuperTrend line
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = lower_band.iloc[0]
        trend.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                trend.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                trend.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                trend.iloc[i] = trend.iloc[i-1]
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend, trend
    
    async def fetch_data(self, symbol: str, timeframe: str = '4h', days: int = 30):
        """Fetch OHLCV data"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df.sort_index()
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def find_signals_and_pullbacks(self, df: pd.DataFrame):
        """Find Super Z signals and analyze subsequent pullbacks"""
        if len(df) < 50:
            return []
        
        # Calculate indicators
        df['vhma'] = self.calculate_vhma(df)
        df['supertrend'], df['trend'] = self.calculate_supertrend(df)
        
        # VHMA color (green when rising, red when falling)
        df['vhma_rising'] = df['vhma'] > df['vhma'].shift(1)
        
        signals = []
        
        for i in range(2, len(df) - 10):  # Leave room to analyze pullbacks
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Long signal conditions
            long_signal = (
                current['close'] > current['supertrend'] and  # Price above SuperTrend
                prev['close'] <= prev['supertrend'] and      # Previous was below/at SuperTrend
                current['vhma_rising'] and                   # VHMA turning green
                not prev['vhma_rising']                      # Previous VHMA was red
            )
            
            # Short signal conditions  
            short_signal = (
                current['close'] < current['supertrend'] and # Price below SuperTrend
                prev['close'] >= prev['supertrend'] and     # Previous was above/at SuperTrend
                not current['vhma_rising'] and              # VHMA turning red
                prev['vhma_rising']                         # Previous VHMA was green
            )
            
            if long_signal or short_signal:
                signal_type = 'LONG' if long_signal else 'SHORT'
                
                # Analyze next 10 candles for pullback
                pullback_analysis = self.analyze_pullback(df, i, signal_type)
                
                signals.append({
                    'time': current.name,
                    'type': signal_type,
                    'price': current['close'],
                    'vhma': current['vhma'],
                    'supertrend': current['supertrend'],
                    'pullback': pullback_analysis
                })
        
        return signals
    
    def analyze_pullback(self, df: pd.DataFrame, signal_idx: int, signal_type: str):
        """Analyze pullback pattern after signal"""
        signal_price = df.iloc[signal_idx]['close']
        
        # Look at next 10 candles
        future_candles = df.iloc[signal_idx+1:signal_idx+11]
        
        if len(future_candles) == 0:
            return {'no_data': True}
        
        red_vhma_candles = 0
        max_pullback_pct = 0
        pullback_occurred = False
        recovery_occurred = False
        
        for i, (timestamp, candle) in enumerate(future_candles.iterrows()):
            # Count red VHMA areas
            if not candle['vhma_rising']:  # Red VHMA
                red_vhma_candles += 1
            
            # Calculate pullback percentage
            if signal_type == 'LONG':
                # For long signals, pullback is when price goes below signal price
                if candle['low'] < signal_price:
                    pullback_pct = ((signal_price - candle['low']) / signal_price) * 100
                    max_pullback_pct = max(max_pullback_pct, pullback_pct)
                    pullback_occurred = True
                
                # Recovery is when price goes back above signal price
                if candle['close'] > signal_price and pullback_occurred:
                    recovery_occurred = True
                    
            else:  # SHORT signal
                # For short signals, pullback is when price goes above signal price
                if candle['high'] > signal_price:
                    pullback_pct = ((candle['high'] - signal_price) / signal_price) * 100
                    max_pullback_pct = max(max_pullback_pct, pullback_pct)
                    pullback_occurred = True
                
                # Recovery is when price goes back below signal price
                if candle['close'] < signal_price and pullback_occurred:
                    recovery_occurred = True
        
        return {
            'pullback_occurred': pullback_occurred,
            'max_pullback_pct': max_pullback_pct,
            'red_vhma_candles': red_vhma_candles,
            'recovery_occurred': recovery_occurred
        }
    
    async def quick_test(self, symbols: list, timeframe: str = '4h'):
        """Run quick test on multiple symbols"""
        results = {}
        
        for symbol in symbols:
            print(f"\n🔍 Testing {symbol}...")
            
            df = await self.fetch_data(symbol, timeframe)
            if df.empty:
                print(f"❌ No data for {symbol}")
                continue
            
            signals = self.find_signals_and_pullbacks(df)
            
            if not signals:
                print(f"📊 No signals found for {symbol}")
                continue
            
            # Analyze results
            total_signals = len(signals)
            signals_with_pullbacks = len([s for s in signals if s['pullback']['pullback_occurred']])
            signals_with_red_vhma = len([s for s in signals if s['pullback']['red_vhma_candles'] > 0])
            
            avg_pullback = np.mean([s['pullback']['max_pullback_pct'] for s in signals 
                                  if s['pullback']['pullback_occurred']])
            
            recovery_rate = len([s for s in signals if s['pullback']['recovery_occurred']]) / total_signals
            
            results[symbol] = {
                'total_signals': total_signals,
                'pullback_rate': signals_with_pullbacks / total_signals,
                'red_vhma_rate': signals_with_red_vhma / total_signals,
                'avg_pullback_pct': avg_pullback if signals_with_pullbacks > 0 else 0,
                'recovery_rate': recovery_rate,
                'signals': signals
            }
            
            print(f"📈 Results for {symbol}:")
            print(f"   Total signals: {total_signals}")
            print(f"   Pullback rate: {signals_with_pullbacks/total_signals:.1%}")
            print(f"   Red VHMA rate: {signals_with_red_vhma/total_signals:.1%}")
            print(f"   Avg pullback: {avg_pullback:.2f}%" if signals_with_pullbacks > 0 else "   Avg pullback: No pullbacks")
            print(f"   Recovery rate: {recovery_rate:.1%}")
        
        return results
    
    def print_summary(self, results: dict):
        """Print overall summary"""
        if not results:
            print("\n❌ No results to summarize")
            return
        
        print("\n" + "="*60)
        print("🎯 SUPER Z PULLBACK PATTERN ANALYSIS SUMMARY")
        print("="*60)
        
        total_signals = sum(r['total_signals'] for r in results.values())
        all_pullback_rates = [r['pullback_rate'] for r in results.values()]
        all_red_vhma_rates = [r['red_vhma_rate'] for r in results.values()]
        all_recovery_rates = [r['recovery_rate'] for r in results.values()]
        
        # Get all individual pullback percentages
        all_pullbacks = []
        for r in results.values():
            for signal in r['signals']:
                if signal['pullback']['pullback_occurred']:
                    all_pullbacks.append(signal['pullback']['max_pullback_pct'])
        
        print(f"📊 Total signals analyzed: {total_signals}")
        print(f"📉 Average pullback rate: {np.mean(all_pullback_rates):.1%}")
        print(f"🔴 Average red VHMA rate: {np.mean(all_red_vhma_rates):.1%}")
        print(f"📈 Average recovery rate: {np.mean(all_recovery_rates):.1%}")
        
        if all_pullbacks:
            print(f"💥 Average pullback size: {np.mean(all_pullbacks):.2f}%")
            print(f"💥 Max pullback observed: {max(all_pullbacks):.2f}%")
        
        print("\n🧠 INSIGHTS:")
        
        avg_pullback_rate = np.mean(all_pullback_rates)
        avg_red_vhma_rate = np.mean(all_red_vhma_rates)
        
        if avg_pullback_rate > 0.7:  # 70%+ pullback rate
            print("✅ PATTERN CONFIRMED: Signals are frequently followed by pullbacks!")
            
            if avg_red_vhma_rate > 0.6:  # 60%+ involve red VHMA
                print("✅ RED VHMA CONNECTION: Pullbacks often coincide with red VHMA areas!")
            
            if all_pullbacks and np.mean(all_pullbacks) > 2:
                print(f"⚠️  SIGNIFICANT PULLBACKS: Average {np.mean(all_pullbacks):.1f}% - consider wider stops!")
            
            if np.mean(all_recovery_rates) > 0.6:
                print("💪 GOOD RECOVERY: Most pullbacks recover - pattern may be buyable!")
        
        elif avg_pullback_rate > 0.4:  # 40-70% pullback rate
            print("🤔 MODERATE PATTERN: Some signals show pullbacks, but not consistent")
        
        else:
            print("❌ PATTERN NOT CONFIRMED: Pullbacks are rare after signals")
        
        print("\n" + "="*60)

async def main():
    """Run the quick test"""
    tester = QuickPullbackTest()
    
    # Test symbols (USDT futures format for Bitget)
    symbols = [
        'BTCUSDT',  # Bitcoin
        'ETHUSDT',  # Ethereum  
        'ADAUSDT',  # Cardano
        'SOLUSDT',  # Solana
        'DOTUSDT',  # Polkadot
    ]
    
    print("🚀 Starting Super Z Pullback Pattern Test...")
    print("🎯 Testing hypothesis: Signals are followed by pullbacks to red VHMA areas")
    
    results = await tester.quick_test(symbols, timeframe='4h')
    tester.print_summary(results)
    
    return results

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
