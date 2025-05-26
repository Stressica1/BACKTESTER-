"""
Super Z Strategy Pullback Analysis
Analyzes the pattern where signals are followed by pullbacks to red VHMA areas
"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import colorama
from colorama import Fore, Style, Back
from prettytable import PrettyTable
import glob
import traceback
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tail_log_files(n=30):
    log_files = glob.glob("*.log")
    for log_file in log_files:
        print(f"\n--- Last {n} lines of {log_file} ---")
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-n:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"Could not read {log_file}: {e}")

tail_log_files()

@dataclass
class PullbackEvent:
    """Represents a pullback event after a signal"""
    signal_time: datetime
    signal_type: str  # 'long' or 'short'
    signal_price: float
    pullback_start_time: datetime
    pullback_end_time: datetime
    pullback_low_price: float  # For long signals
    pullback_high_price: float  # For short signals
    pullback_percentage: float
    vhma_red_duration: int  # Number of candles in red VHMA area
    recovered: bool  # Whether price recovered above signal level
    recovery_time: Optional[datetime]

class SuperZPullbackAnalyzer:
    """
    Analyzes Super Z strategy signals and subsequent pullbacks to red VHMA areas
    """
    
    def __init__(self, exchange_id: str = 'bitget'):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': '',
            'secret': '',
            'password': '',
            'sandbox': True,  # Enable testnet mode
            'enableRateLimit': True,
        })
        print(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}WARNING: Running in TESTNET/SANDBOX mode! All data is from the exchange testnet.\n{Style.RESET_ALL}")
        self.pullback_events: List[PullbackEvent] = []
        
    def calculate_vhma(self, df: pd.DataFrame, length: int = 21) -> pd.Series:
        """
        Calculate Volume-Weighted Hull Moving Average (VHMA)
        Based on the Pine Script implementation
        """
        # Volume-weighted price
        vwp = (df['close'] * df['volume']).rolling(window=length).sum() / df['volume'].rolling(window=length).sum()
        
        # Hull MA calculation
        def hull_ma(src, period):
            half_length = period // 2
            sqrt_length = int(np.sqrt(period))
            
            wma_half = src.rolling(window=half_length).apply(self._wma, raw=False)
            wma_full = src.rolling(window=period).apply(self._wma, raw=False)
            hull = (2 * wma_half - wma_full).rolling(window=sqrt_length).apply(self._wma, raw=False)
            
            return hull
        
        vhma = hull_ma(vwp, length)
        return vhma
    
    def _wma(self, x):
        """Weighted Moving Average calculation"""
        weights = np.arange(1, len(x) + 1)
        return np.sum(weights * x) / np.sum(weights)
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 50, multiplier: float = 1.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator
        Returns: (supertrend_line, trend_direction)
        """
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Calculate SuperTrend
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            # Calculate trend
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                trend.iloc[i] = 1
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
            
            # Calculate SuperTrend line
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        return supertrend, trend
    
    def detect_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Super Z strategy signals based on VHMA and SuperTrend
        """
        # Calculate indicators
        vhma = self.calculate_vhma(df)
        supertrend, trend = self.calculate_supertrend(df)
        
        # Add to dataframe
        df = df.copy()
        df['vhma'] = vhma
        df['supertrend'] = supertrend
        df['trend'] = trend
        df['vhma_color'] = np.where(df['vhma'] > df['vhma'].shift(1), 'green', 'red')
        
        signals = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Long signal: Price crosses above SuperTrend AND VHMA turns green
            if (current['close'] > current['supertrend'] and 
                previous['close'] <= previous['supertrend'] and
                current['vhma_color'] == 'green' and previous['vhma_color'] == 'red'):
                
                signals.append({
                    'time': current.name,
                    'type': 'long',
                    'price': current['close'],
                    'vhma': current['vhma'],
                    'supertrend': current['supertrend']
                })
            
            # Short signal: Price crosses below SuperTrend AND VHMA turns red
            elif (current['close'] < current['supertrend'] and 
                  previous['close'] >= previous['supertrend'] and
                  current['vhma_color'] == 'red' and previous['vhma_color'] == 'green'):
                
                signals.append({
                    'time': current.name,
                    'type': 'short',
                    'price': current['close'],
                    'vhma': current['vhma'],
                    'supertrend': current['supertrend']
                })
        
        return signals, df
    
    def analyze_pullbacks_after_signals(self, df: pd.DataFrame, signals: List[Dict], 
                                       lookback_candles: int = 20) -> List[PullbackEvent]:
        """
        Analyze pullbacks that occur after signals, focusing on red VHMA areas
        """
        pullback_events = []
        
        for signal in signals:
            signal_idx = df.index.get_loc(signal['time'])
            
            # Look ahead for pullbacks
            end_idx = min(signal_idx + lookback_candles, len(df) - 1)
            future_data = df.iloc[signal_idx:end_idx + 1].copy()
            
            if len(future_data) < 5:  # Need at least 5 candles to analyze
                continue
            
            # Analyze pullback pattern
            pullback_event = self._analyze_single_pullback(
                signal, future_data, signal_idx
            )
            
            if pullback_event:
                pullback_events.append(pullback_event)
        
        return pullback_events
    
    def _analyze_single_pullback(self, signal: Dict, future_data: pd.DataFrame, 
                                signal_idx: int) -> Optional[PullbackEvent]:
        """
        Analyze a single pullback event after a signal
        """
        signal_price = signal['price']
        signal_time = signal['time']
        signal_type = signal['type']
        
        # Count red VHMA candles and find pullback extremes
        red_vhma_count = 0
        pullback_start_time = None
        pullback_end_time = None
        extreme_price = signal_price
        extreme_time = signal_time
        
        recovered = False
        recovery_time = None
        
        for i, (timestamp, row) in enumerate(future_data.iterrows()):
            if i == 0:  # Skip signal candle
                continue
                
            # Count red VHMA areas
            if row['vhma_color'] == 'red':
                red_vhma_count += 1
                if pullback_start_time is None:
                    pullback_start_time = timestamp
                pullback_end_time = timestamp
            
            # Track extreme prices during pullback
            if signal_type == 'long':
                if row['low'] < extreme_price:
                    extreme_price = row['low']
                    extreme_time = timestamp
                
                # Check for recovery (price back above signal level)
                if row['close'] > signal_price and not recovered:
                    recovered = True
                    recovery_time = timestamp
                    
            else:  # short signal
                if row['high'] > extreme_price:
                    extreme_price = row['high']
                    extreme_time = timestamp
                
                # Check for recovery (price back below signal level)
                if row['close'] < signal_price and not recovered:
                    recovered = True
                    recovery_time = timestamp
        
        # Only consider it a pullback if we had some red VHMA areas
        if red_vhma_count == 0:
            return None
        
        # Calculate pullback percentage
        pullback_percentage = abs((extreme_price - signal_price) / signal_price) * 100
        
        return PullbackEvent(
            signal_time=signal_time,
            signal_type=signal_type,
            signal_price=signal_price,
            pullback_start_time=pullback_start_time or signal_time,
            pullback_end_time=pullback_end_time or signal_time,
            pullback_low_price=extreme_price if signal_type == 'long' else signal_price,
            pullback_high_price=extreme_price if signal_type == 'short' else signal_price,
            pullback_percentage=pullback_percentage,
            vhma_red_duration=red_vhma_count,
            recovered=recovered,
            recovery_time=recovery_time
        )
    
    async def fetch_data(self, symbol: str, timeframe: str = '4h', days: int = 90) -> pd.DataFrame:
        """Fetch OHLCV data for analysis"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe,
                since=since,
                limit=1000
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_pullback_statistics(self, pullback_events: List[PullbackEvent]) -> Dict:
        """
        Calculate comprehensive statistics about pullback patterns
        """
        if not pullback_events:
            return {}
        
        stats = {
            'total_signals': len(pullback_events),
            'pullback_frequency': len([e for e in pullback_events if e.vhma_red_duration > 0]) / len(pullback_events),
            'average_pullback_percentage': np.mean([e.pullback_percentage for e in pullback_events]),
            'median_pullback_percentage': np.median([e.pullback_percentage for e in pullback_events]),
            'max_pullback_percentage': max([e.pullback_percentage for e in pullback_events]),
            'average_red_vhma_duration': np.mean([e.vhma_red_duration for e in pullback_events]),
            'recovery_rate': len([e for e in pullback_events if e.recovered]) / len(pullback_events),
            'long_signals': len([e for e in pullback_events if e.signal_type == 'long']),
            'short_signals': len([e for e in pullback_events if e.signal_type == 'short']),
        }
        
        # Separate analysis for long and short signals
        long_events = [e for e in pullback_events if e.signal_type == 'long']
        short_events = [e for e in pullback_events if e.signal_type == 'short']
        
        if long_events:
            stats['long_avg_pullback'] = np.mean([e.pullback_percentage for e in long_events])
            stats['long_recovery_rate'] = len([e for e in long_events if e.recovered]) / len(long_events)
        
        if short_events:
            stats['short_avg_pullback'] = np.mean([e.pullback_percentage for e in short_events])
            stats['short_recovery_rate'] = len([e for e in short_events if e.recovered]) / len(short_events)
        
        return stats
    
    async def run_comprehensive_analysis(self, symbols: List[str], 
                                       timeframe: str = '4h', 
                                       days: int = 90) -> Dict:
        """
        Run comprehensive pullback analysis across multiple symbols
        """
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")
            
            # Fetch data
            df = await self.fetch_data(symbol, timeframe, days)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            
            # Detect signals
            signals, df_with_indicators = self.detect_signals(df)
            logger.info(f"Found {len(signals)} signals for {symbol}")
            
            if not signals:
                continue
            
            # Analyze pullbacks
            pullback_events = self.analyze_pullbacks_after_signals(
                df_with_indicators, signals, lookback_candles=20
            )
            
            # Calculate statistics
            stats = self.analyze_pullback_statistics(pullback_events)
            
            all_results[symbol] = {
                'signals': signals,
                'pullback_events': pullback_events,
                'statistics': stats,
                'data': df_with_indicators
            }
        
        return all_results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a comprehensive text report of the analysis
        """
        report = []
        report.append("=" * 80)
        report.append("SUPER Z STRATEGY PULLBACK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        total_signals = sum(len(data['signals']) for data in results.values())
        total_pullbacks = sum(len(data['pullback_events']) for data in results.values())
        
        report.append(f"OVERALL SUMMARY:")
        report.append(f"- Analyzed symbols: {len(results)}")
        report.append(f"- Total signals found: {total_signals}")
        report.append(f"- Total pullback events: {total_pullbacks}")
        report.append("")
        
        # Per-symbol analysis
        for symbol, data in results.items():
            stats = data['statistics']
            if not stats:
                continue
                
            report.append(f"SYMBOL: {symbol}")
            report.append("-" * 40)
            report.append(f"Total signals: {stats['total_signals']}")
            report.append(f"Pullback frequency: {stats['pullback_frequency']:.1%}")
            report.append(f"Average pullback: {stats['average_pullback_percentage']:.2f}%")
            report.append(f"Median pullback: {stats['median_pullback_percentage']:.2f}%")
            report.append(f"Max pullback: {stats['max_pullback_percentage']:.2f}%")
            report.append(f"Average red VHMA duration: {stats['average_red_vhma_duration']:.1f} candles")
            report.append(f"Recovery rate: {stats['recovery_rate']:.1%}")
            
            if 'long_avg_pullback' in stats:
                report.append(f"Long signals avg pullback: {stats['long_avg_pullback']:.2f}%")
                report.append(f"Long signals recovery rate: {stats['long_recovery_rate']:.1%}")
            
            if 'short_avg_pullback' in stats:
                report.append(f"Short signals avg pullback: {stats['short_avg_pullback']:.2f}%")
                report.append(f"Short signals recovery rate: {stats['short_recovery_rate']:.1%}")
            
            report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS:")
        report.append("-" * 40)
        
        # Calculate cross-symbol averages
        all_pullback_percentages = []
        all_recovery_rates = []
        all_red_durations = []
        
        for data in results.values():
            events = data['pullback_events']
            if events:
                all_pullback_percentages.extend([e.pullback_percentage for e in events])
                all_red_durations.extend([e.vhma_red_duration for e in events])
                all_recovery_rates.append(len([e for e in events if e.recovered]) / len(events))
        
        if all_pullback_percentages:
            avg_pullback = np.mean(all_pullback_percentages)
            avg_recovery = np.mean(all_recovery_rates)
            avg_red_duration = np.mean(all_red_durations)
            
            report.append(f"✓ PULLBACK PATTERN CONFIRMED: {len(all_pullback_percentages)} events analyzed")
            report.append(f"✓ Average pullback after signal: {avg_pullback:.2f}%")
            report.append(f"✓ Average recovery rate: {avg_recovery:.1%}")
            report.append(f"✓ Average time spent in red VHMA: {avg_red_duration:.1f} candles")
            
            # Determine if pattern is significant
            if avg_pullback > 2.0:  # More than 2% average pullback
                report.append(f"⚠️  SIGNIFICANT PULLBACK PATTERN DETECTED")
                report.append(f"   - Consider waiting for pullback completion before entry")
                report.append(f"   - Set wider stop losses to account for {avg_pullback:.1f}% average pullback")
            
            if avg_recovery > 0.7:  # 70% recovery rate
                report.append(f"✅ HIGH RECOVERY RATE ({avg_recovery:.1%})")
                report.append(f"   - Signals tend to recover, pullbacks may be buying opportunities")
            
            # Pattern strength assessment
            correlation_strength = "moderate" if len(set(all_red_durations)) > 1 else "weak"
            report.append(f"📊 Pattern consistency: {correlation_strength}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main function to run the strategy"""
    analyzer = SuperZPullbackAnalyzer()

    # Dynamically fetch all USDT-margined swap symbols from Bitget
    print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}Fetching all Bitget USDT-margined swap symbols...{Style.RESET_ALL}")
    markets = await asyncio.to_thread(analyzer.exchange.load_markets)
    symbols = [s for s, m in markets.items() if m.get('swap') and m.get('quote') == 'USDT']
    print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}Scanning {len(symbols)} symbols: {symbols[:10]}... (+{len(symbols)-10} more){Style.RESET_ALL}")

    # Run analysis
    results = await analyzer.run_comprehensive_analysis(symbols=symbols, timeframe='4h', days=60)

    # Generate and print report
    report = analyzer.generate_report(results)
    print(report)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'super_z_pullback_analysis_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Save detailed data as JSON
    serializable_results = {
        symbol: {
            'statistics': data['statistics'],
            'signal_count': len(data['signals']),
            'pullback_count': len(data['pullback_events'])
        }
        for symbol, data in results.items()
    }
    with open(f'super_z_analysis_data_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nAnalysis complete! Report saved to super_z_pullback_analysis_{timestamp}.txt")
    return results

if __name__ == "__main__":
    results = asyncio.run(main())

# After analysis, build a cyberpunk PrettyTable for signals

def print_cyberpunk_table(signals):
    table = PrettyTable()
    table.field_names = [
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Symbol",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}TF",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Tier",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Signal",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Price",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Time",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Volume",
        f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Filters"
    ]
    for s in signals:
        # Assign tier and color
        tier, emoji, color = get_signal_tier(s)
        table.add_row([
            f"{Back.BLACK}{Fore.CYAN}{s['symbol']}",
            f"{Back.BLACK}{Fore.MAGENTA}{s['timeframe']}",
            f"{color}{tier} {emoji}",
            f"{color}{s['signal']}",
            f"{Fore.YELLOW}{s['price']:.2f}",
            f"{Fore.GREEN}{s['time']}",
            f"{Fore.CYAN}{s['volume']:.2f}",
            f"{Fore.LIGHTMAGENTA_EX}{s['filters']}"
        ])
    print(f"\n{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}=== CYBERPUNK SUPER Z SIGNALS ==={Style.RESET_ALL}")
    print(table)
    print(f"{Back.MAGENTA}{Fore.CYAN}Legend: 🟢 Strong Buy | 🟡 Buy | ⚪️ Hold | 🔴 Sell | 💀 Strong Sell | 🔄 Trend Change | 🌀 Pullback{Style.RESET_ALL}\n")

# Helper to assign tier, emoji, and color

def get_signal_tier(signal):
    # Example logic, adjust as needed
    if signal['signal'] == 'BUY' and signal['filters'].count('✔️') >= 2:
        return 'Strong Buy', '🟢', Back.GREEN + Fore.BLACK
    elif signal['signal'] == 'BUY':
        return 'Buy', '🟡', Back.YELLOW + Fore.BLACK
    elif signal['signal'] == 'SELL' and signal['filters'].count('✔️') >= 2:
        return 'Strong Sell', '💀', Back.RED + Fore.WHITE
    elif signal['signal'] == 'SELL':
        return 'Sell', '🔴', Back.LIGHTRED_EX + Fore.WHITE
    elif 'Trend' in signal.get('tier', ''):
        return 'Trend Change', '🔄', Back.CYAN + Fore.BLACK
    elif 'Pullback' in signal.get('tier', ''):
        return 'Pullback', '🌀', Back.MAGENTA + Fore.WHITE
    else:
        return 'Hold', '⚪️', Back.BLACK + Fore.WHITE
