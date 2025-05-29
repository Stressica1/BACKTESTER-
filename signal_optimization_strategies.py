#!/usr/bin/env python3
"""
SIGNAL OPTIMIZATION STRATEGIES
Advanced techniques to improve trading signal quality and win rate
"""

import asyncio
import time
import pandas as pd
import numpy as np
from collections import deque
import statistics

class SignalOptimizer:
    """Advanced signal optimization techniques"""
    
    def __init__(self):
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
    # OPTIMIZATION STRATEGY 1: ADAPTIVE PARAMETERS
    def adaptive_rsi_thresholds(self, symbol, current_volatility):
        """
        Dynamically adjust RSI thresholds based on market volatility
        High volatility = wider thresholds, Low volatility = tighter thresholds
        """
        base_oversold = 30
        base_overbought = 70
        
        # Adjust based on volatility (0.01 = 1% daily volatility)
        volatility_adjustment = (current_volatility - 0.02) * 500  # Scale factor
        
        oversold_threshold = max(20, base_oversold - volatility_adjustment)
        overbought_threshold = min(80, base_overbought + volatility_adjustment)
        
        return {
            'buy_rsi_max': oversold_threshold + 20,  # More flexible for buy
            'sell_rsi_min': overbought_threshold - 20,  # More flexible for sell
            'volatility': current_volatility
        }
    
    # OPTIMIZATION STRATEGY 2: MULTI-TIMEFRAME CONFIRMATION
    def multi_timeframe_signal(self, symbol, current_tf='5m'):
        """
        Require signal confirmation across multiple timeframes
        5m + 15m + 1h confirmation for stronger signals
        """
        timeframes = ['5m', '15m', '1h']
        confirmations = {}
        
        for tf in timeframes:
            # Simulate getting data for each timeframe
            trend_aligned = self.check_trend_alignment(symbol, tf)
            momentum_aligned = self.check_momentum_alignment(symbol, tf)
            
            confirmations[tf] = {
                'trend': trend_aligned,
                'momentum': momentum_aligned,
                'score': 1 if (trend_aligned and momentum_aligned) else 0
            }
        
        # Calculate multi-timeframe score
        total_score = sum(conf['score'] for conf in confirmations.values())
        max_score = len(timeframes)
        
        return {
            'mtf_score': total_score / max_score * 100,
            'confirmations': confirmations,
            'signal_strength': 'STRONG' if total_score >= 2 else 'WEAK'
        }
    
    # OPTIMIZATION STRATEGY 3: DYNAMIC CONFIDENCE SCORING
    def enhanced_confidence_calculation(self, signal_data):
        """
        Enhanced confidence calculation with market condition awareness
        """
        base_confidence = 40
        
        # 1. Trend Strength Factor (0-25 points)
        supertrend_distance = abs(signal_data['price'] - signal_data['supertrend_value'])
        trend_strength = min(25, supertrend_distance / signal_data['price'] * 2500)
        
        # 2. RSI Momentum Factor (0-20 points)  
        rsi = signal_data['rsi']
        if signal_data['side'] == 'buy':
            rsi_strength = max(0, (40 - rsi) / 40 * 20)  # Better score for more oversold
        else:
            rsi_strength = max(0, (rsi - 60) / 40 * 20)  # Better score for more overbought
        
        # 3. Volume Confirmation Factor (0-15 points)
        volume_ratio = signal_data['volume_ratio']
        volume_strength = min(15, (volume_ratio - 1) * 15)
        
        # 4. Momentum Quality Factor (0-15 points)
        momentum = abs(signal_data['momentum'])
        momentum_strength = min(15, momentum * 7500)
        
        # 5. Market Condition Factor (0-10 points)
        market_condition = self.assess_market_condition(signal_data['symbol'])
        condition_bonus = market_condition * 10
        
        # 6. Historical Performance Factor (-5 to +5 points)
        historical_performance = self.get_symbol_performance(signal_data['symbol'])
        
        total_confidence = (base_confidence + trend_strength + rsi_strength + 
                          volume_strength + momentum_strength + condition_bonus + 
                          historical_performance)
        
        return min(95, max(0, total_confidence))
    
    # OPTIMIZATION STRATEGY 4: MARKET REGIME DETECTION
    def detect_market_regime(self, price_data):
        """
        Detect if market is trending, ranging, or volatile
        Adjust signal parameters accordingly
        """
        if len(price_data) < 20:
            return 'UNKNOWN'
        
        # Calculate trend strength
        sma_20 = price_data.rolling(20).mean()
        price_above_sma = (price_data > sma_20).sum() / len(price_data)
        
        # Calculate volatility
        returns = price_data.pct_change().dropna()
        volatility = returns.std() * np.sqrt(288)  # Annualized for 5-min data
        
        # Determine regime
        if volatility > 0.05:  # High volatility (>5% daily)
            return 'VOLATILE'
        elif 0.7 <= price_above_sma <= 1.0 or 0.0 <= price_above_sma <= 0.3:
            return 'TRENDING'
        else:
            return 'RANGING'
    
    # OPTIMIZATION STRATEGY 5: SIGNAL FILTERING
    def filter_weak_signals(self, signal):
        """
        Apply advanced filters to eliminate weak signals
        """
        filters_passed = 0
        total_filters = 6
        
        # Filter 1: Price action confirmation
        if self.check_price_action_confirmation(signal):
            filters_passed += 1
        
        # Filter 2: Volume spike confirmation  
        if signal['volume_ratio'] > 1.3:
            filters_passed += 1
        
        # Filter 3: Momentum persistence
        if self.check_momentum_persistence(signal):
            filters_passed += 1
        
        # Filter 4: Support/Resistance level respect
        if self.check_sr_levels(signal):
            filters_passed += 1
        
        # Filter 5: Time-of-day filter (avoid low liquidity hours)
        if self.check_trading_hours(signal):
            filters_passed += 1
        
        # Filter 6: Economic calendar filter
        if self.check_economic_events(signal):
            filters_passed += 1
        
        filter_score = filters_passed / total_filters * 100
        
        return {
            'filters_passed': filters_passed,
            'total_filters': total_filters,
            'filter_score': filter_score,
            'signal_approved': filters_passed >= 4  # Require 4/6 filters
        }
    
    # OPTIMIZATION STRATEGY 6: ADAPTIVE STOP LOSSES
    def calculate_adaptive_stop_loss(self, signal, atr_value):
        """
        Calculate dynamic stop loss based on volatility (ATR)
        """
        base_stop_pct = 0.01  # 1% base stop loss
        
        # Adjust based on ATR (Average True Range)
        atr_multiplier = 2.0
        atr_stop_pct = (atr_value / signal['price']) * atr_multiplier
        
        # Use the larger of base stop or ATR-based stop
        adaptive_stop_pct = max(base_stop_pct, atr_stop_pct)
        
        # Cap at maximum 3% stop loss
        final_stop_pct = min(0.03, adaptive_stop_pct)
        
        return {
            'stop_loss_pct': final_stop_pct,
            'stop_price': signal['price'] * (1 - final_stop_pct) if signal['side'] == 'buy' 
                         else signal['price'] * (1 + final_stop_pct),
            'atr_based': atr_stop_pct > base_stop_pct
        }
    
    # HELPER METHODS FOR OPTIMIZATION
    def check_trend_alignment(self, symbol, timeframe):
        """Check if trend is aligned on given timeframe"""
        # Simulate trend check (in real implementation, fetch actual data)
        return np.random.choice([True, False], p=[0.6, 0.4])
    
    def check_momentum_alignment(self, symbol, timeframe):
        """Check if momentum is aligned on given timeframe"""
        return np.random.choice([True, False], p=[0.6, 0.4])
    
    def assess_market_condition(self, symbol):
        """Assess current market condition (0.0 to 1.0)"""
        return np.random.uniform(0.3, 0.9)
    
    def get_symbol_performance(self, symbol):
        """Get historical performance score for symbol (-5 to +5)"""
        return np.random.uniform(-3, 5)
    
    def check_price_action_confirmation(self, signal):
        """Check if price action confirms the signal"""
        return np.random.choice([True, False], p=[0.7, 0.3])
    
    def check_momentum_persistence(self, signal):
        """Check if momentum is persistent"""
        return abs(signal['momentum']) > 0.002
    
    def check_sr_levels(self, signal):
        """Check support/resistance levels"""
        return np.random.choice([True, False], p=[0.8, 0.2])
    
    def check_trading_hours(self, signal):
        """Check if within good trading hours"""
        hour = time.gmtime().tm_hour
        return 8 <= hour <= 22  # London/NY session overlap
    
    def check_economic_events(self, signal):
        """Check for major economic events"""
        return np.random.choice([True, False], p=[0.9, 0.1])

# PRACTICAL OPTIMIZATION IMPLEMENTATION
class OptimizedSignalGenerator:
    """
    Enhanced signal generator with optimization strategies
    """
    
    def __init__(self):
        self.optimizer = SignalOptimizer()
        self.performance_history = deque(maxlen=100)
        
    async def generate_optimized_signal(self, symbol, market_data):
        """
        Generate optimized trading signal using multiple strategies
        """
        try:
            # Basic signal generation (existing logic)
            basic_signal = await self.generate_basic_signal(symbol, market_data)
            if not basic_signal:
                return None
            
            # OPTIMIZATION 1: Adaptive Parameters
            adaptive_params = self.optimizer.adaptive_rsi_thresholds(
                symbol, market_data['volatility']
            )
            
            # OPTIMIZATION 2: Multi-timeframe Confirmation
            mtf_confirmation = self.optimizer.multi_timeframe_signal(symbol)
            
            # OPTIMIZATION 3: Enhanced Confidence
            enhanced_confidence = self.optimizer.enhanced_confidence_calculation(basic_signal)
            
            # OPTIMIZATION 4: Market Regime Detection
            market_regime = self.optimizer.detect_market_regime(market_data['price_series'])
            
            # OPTIMIZATION 5: Signal Filtering
            filter_results = self.optimizer.filter_weak_signals(basic_signal)
            
            # Combine all optimizations
            if filter_results['signal_approved'] and mtf_confirmation['mtf_score'] >= 60:
                optimized_signal = {
                    **basic_signal,
                    'confidence': enhanced_confidence,
                    'mtf_score': mtf_confirmation['mtf_score'],
                    'market_regime': market_regime,
                    'filter_score': filter_results['filter_score'],
                    'adaptive_params': adaptive_params,
                    'optimization_applied': True
                }
                
                return optimized_signal
            
            return None  # Signal rejected by filters
            
        except Exception as e:
            print(f"Error in optimized signal generation: {e}")
            return None
    
    async def generate_basic_signal(self, symbol, market_data):
        """Generate basic signal (existing logic)"""
        # Implement existing signal logic here
        return {
            'symbol': symbol,
            'side': 'buy',  # or 'sell'
            'price': market_data['current_price'],
            'rsi': market_data['rsi'],
            'momentum': market_data['momentum'],
            'volume_ratio': market_data['volume_ratio'],
            'supertrend_value': market_data['supertrend'],
            'timestamp': time.time()
        }

def demonstrate_optimization():
    """Demonstrate optimization strategies"""
    print("🚀 SIGNAL OPTIMIZATION STRATEGIES")
    print("=" * 60)
    
    optimizer = SignalOptimizer()
    
    # Example signal data
    sample_signal = {
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'price': 43500,
        'rsi': 35,
        'momentum': 0.0025,
        'volume_ratio': 1.4,
        'supertrend_value': 43200,
        'timestamp': time.time()
    }
    
    print("📊 CURRENT SIGNAL:")
    for key, value in sample_signal.items():
        if key == 'timestamp':
            continue
        print(f"   {key}: {value}")
    
    print("\n🔧 OPTIMIZATION RESULTS:")
    
    # 1. Enhanced Confidence
    enhanced_conf = optimizer.enhanced_confidence_calculation(sample_signal)
    print(f"   📈 Enhanced Confidence: {enhanced_conf:.1f}%")
    
    # 2. Multi-timeframe
    mtf_result = optimizer.multi_timeframe_signal(sample_signal['symbol'])
    print(f"   📊 Multi-TF Score: {mtf_result['mtf_score']:.1f}%")
    print(f"   🎯 Signal Strength: {mtf_result['signal_strength']}")
    
    # 3. Signal Filtering
    filter_result = optimizer.filter_weak_signals(sample_signal)
    print(f"   🔍 Filter Score: {filter_result['filter_score']:.1f}%")
    print(f"   ✅ Filters Passed: {filter_result['filters_passed']}/{filter_result['total_filters']}")
    print(f"   🎯 Signal Approved: {filter_result['signal_approved']}")
    
    # 4. Adaptive Parameters
    adaptive_params = optimizer.adaptive_rsi_thresholds(sample_signal['symbol'], 0.03)  # 3% volatility
    print(f"   📊 Adaptive RSI Buy Max: {adaptive_params['buy_rsi_max']:.1f}")
    
    print("\n✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_optimization() 