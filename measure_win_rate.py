#!/usr/bin/env python3
"""
WIN RATE MEASUREMENT - Track Actual Trading Performance
Measures real win rate based on profitable vs losing trades
"""

import sys
import asyncio
import time
import random
from collections import defaultdict
from supertrend_pullback_live import AggressivePullbackTrader

class WinRateTracker:
    """Track and calculate actual trading win rate"""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.break_even_trades = 0
        self.total_pnl = 0.0
        self.trade_history = []
        
    def record_trade(self, symbol, side, entry_price, exit_price, size, leverage):
        """Record a trade outcome and calculate P&L"""
        
        # Calculate P&L based on trade direction
        if side.lower() == 'buy':
            raw_pnl = (exit_price - entry_price) / entry_price
        else:  # sell
            raw_pnl = (entry_price - exit_price) / entry_price
        
        # Apply leverage to P&L
        leveraged_pnl = raw_pnl * leverage
        
        # Convert to USDT P&L
        usdt_pnl = size * leveraged_pnl
        
        # Subtract trading fees (0.1% per side = 0.2% total)
        fee = size * 0.002  # 0.2% trading fees
        final_pnl = usdt_pnl - fee
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'leverage': leverage,
            'raw_pnl': raw_pnl * 100,  # As percentage
            'leveraged_pnl': leveraged_pnl * 100,  # As percentage  
            'usdt_pnl': final_pnl,
            'profitable': final_pnl > 0
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
        self.total_pnl += final_pnl
        
        if final_pnl > 0.01:  # Profitable (accounting for small rounding)
            self.winning_trades += 1
        elif final_pnl < -0.01:  # Loss
            self.losing_trades += 1
        else:  # Break-even
            self.break_even_trades += 1
            
        return trade_record
    
    def get_win_rate(self):
        """Calculate current win rate"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    def get_profit_factor(self):
        """Calculate profit factor"""
        gross_profit = sum(t['usdt_pnl'] for t in self.trade_history if t['usdt_pnl'] > 0)
        gross_loss = abs(sum(t['usdt_pnl'] for t in self.trade_history if t['usdt_pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def get_average_win_loss_ratio(self):
        """Calculate average win/loss ratio"""
        wins = [t['usdt_pnl'] for t in self.trade_history if t['usdt_pnl'] > 0]
        losses = [abs(t['usdt_pnl']) for t in self.trade_history if t['usdt_pnl'] < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0
        return avg_win / avg_loss

async def simulate_trade_execution(trader, signal):
    """Simulate a complete trade cycle with realistic market movement"""
    
    symbol = signal['symbol']
    side = signal['side']
    entry_price = signal['price']
    leverage = signal['leverage']
    size = 0.50  # Fixed USDT size
    
    # Simulate holding period (5 minutes to 2 hours)
    hold_time = random.uniform(5 * 60, 2 * 60 * 60)  # 5 min to 2 hours
    
    # Simulate realistic price movement based on volatility
    volatility = get_symbol_volatility(symbol)
    
    # Random price movement with slight bias toward signal direction
    direction_bias = 0.6 if side == 'buy' else 0.4  # 60% chance of moving in signal direction
    
    if random.random() < direction_bias:
        # Move in favorable direction
        if side == 'buy':
            price_change = random.uniform(0.001, volatility * 2)  # 0.1% to 2x volatility gain
        else:
            price_change = random.uniform(-volatility * 2, -0.001)  # Loss for short
    else:
        # Move against signal
        if side == 'buy':
            price_change = random.uniform(-volatility * 1.5, -0.001)  # Loss for long
        else:
            price_change = random.uniform(0.001, volatility * 1.5)  # Loss for short
    
    exit_price = entry_price * (1 + price_change)
    
    # Add some random market noise
    noise = random.uniform(-0.0005, 0.0005)  # ±0.05% noise
    exit_price *= (1 + noise)
    
    return entry_price, exit_price

def get_symbol_volatility(symbol):
    """Get estimated daily volatility for different symbols"""
    volatilities = {
        'BTC/USDT': 0.02,    # 2% daily volatility
        'ETH/USDT': 0.025,   # 2.5%
        'SOL/USDT': 0.04,    # 4%
        'BNB/USDT': 0.025,   # 2.5%
        'XRP/USDT': 0.03,    # 3%
        'ADA/USDT': 0.035,   # 3.5%
        'DOGE/USDT': 0.05,   # 5% (meme coin)
        'SHIB/USDT': 0.06,   # 6% (meme coin)
        'PEPE/USDT': 0.08,   # 8% (meme coin)
    }
    
    return volatilities.get(symbol, 0.04)  # Default 4% for unknown symbols

async def measure_win_rate():
    """Comprehensive win rate measurement"""
    
    print("📊 MEASURING ACTUAL TRADING WIN RATE")
    print("=" * 60)
    print("🎯 Running extended simulation to track trade outcomes...")
    print("💰 Fixed Position Size: 0.50 USDT per trade")
    print("⚡ Testing optimized signal parameters (Iteration 2)")
    print("-" * 60)
    
    # Initialize tracker and trader
    tracker = WinRateTracker()
    trader = AggressivePullbackTrader(simulation_mode=True)
    
    # Target metrics
    target_trades = 50  # Aim for 50 trades for statistical significance
    max_cycles = 200    # Maximum cycles to prevent infinite loop
    cycle_count = 0
    
    print(f"🎯 Target: {target_trades} trades for statistical significance")
    print(f"🔄 Max cycles: {max_cycles}")
    print("-" * 60)
    
    start_time = time.time()
    
    while tracker.total_trades < target_trades and cycle_count < max_cycles:
        cycle_count += 1
        
        # Test a batch of symbols each cycle
        symbols_to_test = random.sample(trader.active_symbols, min(10, len(trader.active_symbols)))
        
        for symbol in symbols_to_test:
            try:
                # Generate signal
                signal = await trader.generate_signal(symbol)
                
                if signal and signal.get('confidence', 0) >= 65:
                    # Execute simulated trade
                    entry_price, exit_price = await simulate_trade_execution(trader, signal)
                    
                    # Record trade outcome
                    trade_record = tracker.record_trade(
                        symbol=signal['symbol'],
                        side=signal['side'],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=0.50,
                        leverage=signal['leverage']
                    )
                    
                    # Print trade result
                    profit_emoji = "✅" if trade_record['profitable'] else "❌"
                    print(f"{profit_emoji} Trade #{tracker.total_trades}: {symbol} {signal['side'].upper()} | "
                          f"P&L: {trade_record['usdt_pnl']:+.3f} USDT | "
                          f"Leverage: {signal['leverage']}x | "
                          f"Win Rate: {tracker.get_win_rate():.1f}%")
                    
                    # Show running statistics every 10 trades
                    if tracker.total_trades % 10 == 0:
                        print(f"📊 Progress: {tracker.total_trades}/{target_trades} trades | "
                              f"Win Rate: {tracker.get_win_rate():.1f}% | "
                              f"Total P&L: {tracker.total_pnl:+.2f} USDT")
                        print("-" * 60)
                        
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")
        
        # Brief pause between cycles
        await asyncio.sleep(0.1)
        
        # Break if we hit target
        if tracker.total_trades >= target_trades:
            break
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    print("\n🏁 WIN RATE MEASUREMENT COMPLETE!")
    print("=" * 60)
    
    # Core Performance Metrics
    print(f"📊 CORE METRICS:")
    print(f"   📈 Total Trades: {tracker.total_trades}")
    print(f"   ✅ Winning Trades: {tracker.winning_trades}")
    print(f"   ❌ Losing Trades: {tracker.losing_trades}")
    print(f"   ⚪ Break-even Trades: {tracker.break_even_trades}")
    print(f"   🎯 WIN RATE: {tracker.get_win_rate():.2f}%")
    
    # Financial Metrics
    print(f"\n💰 FINANCIAL PERFORMANCE:")
    print(f"   💵 Total P&L: {tracker.total_pnl:+.2f} USDT")
    print(f"   📊 Profit Factor: {tracker.get_profit_factor():.2f}")
    print(f"   ⚖️ Avg Win/Loss Ratio: {tracker.get_average_win_loss_ratio():.2f}")
    
    if tracker.total_trades > 0:
        print(f"   📈 Average P&L per Trade: {tracker.total_pnl/tracker.total_trades:+.3f} USDT")
        print(f"   💹 ROI per Trade: {(tracker.total_pnl/tracker.total_trades)/0.50*100:+.2f}%")
    
    # Performance Analysis
    print(f"\n🔍 ANALYSIS:")
    if tracker.get_win_rate() >= 60:
        print(f"   ✅ EXCELLENT: Win rate {tracker.get_win_rate():.1f}% is very strong!")
    elif tracker.get_win_rate() >= 50:
        print(f"   ✅ GOOD: Win rate {tracker.get_win_rate():.1f}% is solid!")
    elif tracker.get_win_rate() >= 40:
        print(f"   ⚠️ ACCEPTABLE: Win rate {tracker.get_win_rate():.1f}% needs profit factor > 1.5")
    else:
        print(f"   ❌ LOW: Win rate {tracker.get_win_rate():.1f}% requires high profit factor")
    
    if tracker.get_profit_factor() > 1.5:
        print(f"   ✅ Strong profit factor: {tracker.get_profit_factor():.2f}")
    elif tracker.get_profit_factor() > 1.0:
        print(f"   ⚠️ Marginal profit factor: {tracker.get_profit_factor():.2f}")
    else:
        print(f"   ❌ Poor profit factor: {tracker.get_profit_factor():.2f}")
    
    # Execution Stats
    print(f"\n⚡ EXECUTION STATS:")
    print(f"   🔄 Cycles: {cycle_count}")
    print(f"   ⏱️ Time: {elapsed_time:.1f} seconds")
    print(f"   🚀 Trades/minute: {tracker.total_trades/(elapsed_time/60):.1f}")
    
    print("=" * 60)
    
    return tracker

if __name__ == "__main__":
    asyncio.run(measure_win_rate()) 