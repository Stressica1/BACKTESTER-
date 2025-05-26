"""
Super Z Trading Signal System
Converts the validated pullback hypothesis into actionable trading signals with precise entry/exit rules
"""

import pandas as pd
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json
from super_z_optimized import SuperZOptimizedAnalyzer, PullbackEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Represents a complete trading signal with entry/exit rules"""
    symbol: str
    timeframe: str
    signal_type: str  # 'long' or 'short'
    timestamp: datetime
    
    # Initial signal data
    initial_signal_price: float
    vhma_value: float
    supertrend_value: float
    
    # Expected pullback data
    expected_pullback_percentage: float
    expected_pullback_duration: int
    
    # Entry strategy
    entry_strategy: str  # 'immediate', 'wait_for_pullback', 'pullback_recovery'
    entry_price: Optional[float] = None
    entry_triggered: bool = False
    
    # Exit strategy
    stop_loss_price: float = None
    take_profit_price: float = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # Risk management
    position_size_percentage: float = 1.0  # % of portfolio
    risk_reward_ratio: float = 3.0
    max_risk_percentage: float = 2.0  # % of portfolio at risk
    
    # Tracking data
    pullback_detected: bool = False
    pullback_low_price: Optional[float] = None
    pullback_high_price: Optional[float] = None
    recovery_detected: bool = False
    
    # Performance
    pnl_percentage: Optional[float] = None
    status: str = 'pending'  # 'pending', 'active', 'closed', 'cancelled'
    
    confidence_score: float = 85.0  # Based on historical success rate

@dataclass
class TradingStrategy:
    """Configuration for different trading strategies"""
    name: str
    description: str
    entry_method: str
    risk_percentage: float
    reward_ratio: float
    timeframe_priority: List[str]
    
class SuperZTradingSignalGenerator:
    """
    Generates actionable trading signals based on Super Z pullback analysis
    """
    
    def __init__(self):
        self.analyzer = SuperZOptimizedAnalyzer(pool_size=5, use_cache=True)
        self.active_signals: List[TradingSignal] = []
        self.closed_signals: List[TradingSignal] = []
        
        # Trading strategies based on our findings
        self.strategies = {
            'aggressive': TradingStrategy(
                name="Aggressive Pullback Entry",
                description="Enter immediately on signal, expecting pullback for better average price",
                entry_method="immediate",
                risk_percentage=2.0,
                reward_ratio=3.0,
                timeframe_priority=['5m', '15m', '1h']
            ),
            'conservative': TradingStrategy(
                name="Conservative Pullback Wait",
                description="Wait for pullback to red VHMA, then enter on recovery",
                entry_method="wait_for_pullback",
                risk_percentage=1.5,
                reward_ratio=4.0,
                timeframe_priority=['15m', '1h', '4h']
            ),
            'scalp': TradingStrategy(
                name="Scalp Pullback Trade",
                description="Quick trades on pullback patterns",
                entry_method="pullback_recovery",
                risk_percentage=3.0,
                reward_ratio=2.0,
                timeframe_priority=['1m', '5m']
            )
        }
    
    async def generate_trading_signals(self, symbols: List[str], timeframes: List[str] = ['5m', '15m', '1h'],
                                     strategy_name: str = 'conservative', days: int = 7) -> List[TradingSignal]:
        """
        Generate trading signals for multiple symbols using the Super Z pullback pattern
        """
        logger.info(f"Generating {strategy_name} trading signals for {len(symbols)} symbols")
        
        strategy = self.strategies.get(strategy_name, self.strategies['conservative'])
        signals = []
        
        # Analyze each symbol for recent signals
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Get recent data and detect signals
                    df = await self.analyzer.fetch_data_optimized(symbol, timeframe, days)
                    
                    if df.empty:
                        continue
                    
                    # Detect signals
                    detected_signals, df_with_indicators = self.analyzer.detect_signals_vectorized(df)
                    
                    # Only consider the most recent signals (last 24 hours)
                    recent_signals = [s for s in detected_signals 
                                    if s['time'] > datetime.now() - timedelta(hours=24)]
                    
                    if not recent_signals:
                        continue
                    
                    # Analyze historical pullback behavior for this symbol/timeframe
                    historical_pullbacks = await asyncio.to_thread(
                        self.analyzer._analyze_pullbacks_vectorized,
                        df_with_indicators, detected_signals, 20
                    )
                    
                    # Calculate expected pullback statistics
                    expected_stats = self._calculate_expected_pullback_stats(historical_pullbacks)
                    
                    # Generate signals for recent detections
                    for signal_data in recent_signals:
                        trading_signal = self._create_trading_signal(
                            symbol, timeframe, signal_data, strategy, expected_stats, df_with_indicators
                        )
                        signals.append(trading_signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
                    continue
        
        # Sort by confidence score and timestamp
        signals.sort(key=lambda x: (x.confidence_score, x.timestamp), reverse=True)
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    def _calculate_expected_pullback_stats(self, pullback_events: List[PullbackEvent]) -> Dict:
        """Calculate expected pullback statistics from historical data"""
        if not pullback_events:
            return {
                'avg_pullback_percentage': 3.0,
                'avg_duration': 10,
                'success_rate': 0.7
            }
        
        pullback_percentages = [e.pullback_percentage for e in pullback_events]
        durations = [e.vhma_red_duration for e in pullback_events]
        
        return {
            'avg_pullback_percentage': np.mean(pullback_percentages),
            'median_pullback_percentage': np.median(pullback_percentages),
            'max_pullback_percentage': np.max(pullback_percentages),
            'avg_duration': np.mean(durations),
            'success_rate': len(pullback_events) / max(len(pullback_events), 1),
            'recovery_rate': sum(1 for e in pullback_events if e.recovered) / len(pullback_events)
        }
    
    def _create_trading_signal(self, symbol: str, timeframe: str, signal_data: Dict,
                             strategy: TradingStrategy, expected_stats: Dict,
                             df: pd.DataFrame) -> TradingSignal:
        """Create a comprehensive trading signal"""
        
        signal_price = signal_data['price']
        signal_type = signal_data['type']
        
        # Calculate position sizing and risk management
        expected_pullback = expected_stats.get('avg_pullback_percentage', 3.0)
        
        # Entry strategy based on strategy type
        if strategy.entry_method == 'immediate':
            entry_price = signal_price
            entry_triggered = True
        else:
            entry_price = None
            entry_triggered = False
        
        # Calculate stop loss and take profit based on expected pullback
        if signal_type == 'long':
            # For long signals, expect pullback down, so set stop below expected pullback
            stop_loss_price = signal_price * (1 - (expected_pullback + 1.0) / 100)
            take_profit_price = signal_price * (1 + (expected_pullback * strategy.reward_ratio) / 100)
        else:
            # For short signals, expect pullback up
            stop_loss_price = signal_price * (1 + (expected_pullback + 1.0) / 100)
            take_profit_price = signal_price * (1 - (expected_pullback * strategy.reward_ratio) / 100)
        
        # Calculate confidence score based on historical success
        base_confidence = expected_stats.get('success_rate', 0.7) * 100
        recovery_bonus = expected_stats.get('recovery_rate', 0.8) * 20
        timeframe_bonus = 10 if timeframe in strategy.timeframe_priority[:2] else 0
        
        confidence_score = min(95, base_confidence + recovery_bonus + timeframe_bonus)
        
        return TradingSignal(
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            timestamp=signal_data['time'],
            initial_signal_price=signal_price,
            vhma_value=signal_data['vhma'],
            supertrend_value=signal_data['supertrend'],
            expected_pullback_percentage=expected_pullback,
            expected_pullback_duration=int(expected_stats.get('avg_duration', 10)),
            entry_strategy=strategy.entry_method,
            entry_price=entry_price,
            entry_triggered=entry_triggered,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_size_percentage=strategy.risk_percentage,
            risk_reward_ratio=strategy.reward_ratio,
            max_risk_percentage=strategy.risk_percentage,
            confidence_score=confidence_score
        )
    
    async def monitor_active_signals(self, signals: List[TradingSignal]) -> List[Dict]:
        """
        Monitor active trading signals and update their status based on real-time data
        """
        logger.info(f"Monitoring {len(signals)} active signals")
        
        updates = []
        
        for signal in signals:
            try:
                # Fetch current price data
                df = await self.analyzer.fetch_data_optimized(
                    signal.symbol, signal.timeframe, 1
                )
                
                if df.empty:
                    continue
                
                current_price = df['close'].iloc[-1]
                current_time = df.index[-1]
                
                # Check for entry triggers
                if not signal.entry_triggered:
                    entry_update = self._check_entry_conditions(signal, df)
                    if entry_update:
                        updates.append(entry_update)
                
                # Check for exit conditions
                if signal.entry_triggered:
                    exit_update = self._check_exit_conditions(signal, current_price, current_time)
                    if exit_update:
                        updates.append(exit_update)
                
                # Check for pullback detection
                pullback_update = self._check_pullback_conditions(signal, df)
                if pullback_update:
                    updates.append(pullback_update)
                    
            except Exception as e:
                logger.error(f"Error monitoring signal {signal.symbol}: {e}")
                continue
        
        return updates
    
    def _check_entry_conditions(self, signal: TradingSignal, df: pd.DataFrame) -> Optional[Dict]:
        """Check if entry conditions are met for pending signals"""
        if signal.entry_triggered:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Calculate VHMA for current data
        df['vhma'] = self.analyzer.calculate_vhma_vectorized(df)
        df['vhma_color'] = np.where(df['vhma'] > df['vhma'].shift(1), 'green', 'red')
        
        current_vhma_color = df['vhma_color'].iloc[-1]
        
        if signal.entry_strategy == 'wait_for_pullback':
            # Wait for red VHMA (pullback) then enter on green recovery
            if (current_vhma_color == 'green' and 
                df['vhma_color'].iloc[-2] == 'red' and 
                signal.pullback_detected):
                
                signal.entry_price = current_price
                signal.entry_triggered = True
                signal.recovery_detected = True
                
                return {
                    'type': 'entry_triggered',
                    'signal': signal,
                    'message': f"Entry triggered for {signal.symbol} on pullback recovery at {current_price}"
                }
        
        elif signal.entry_strategy == 'pullback_recovery':
            # Enter on first sign of recovery after pullback
            if signal.pullback_detected and current_vhma_color == 'green':
                signal.entry_price = current_price
                signal.entry_triggered = True
                signal.recovery_detected = True
                
                return {
                    'type': 'entry_triggered',
                    'signal': signal,
                    'message': f"Scalp entry triggered for {signal.symbol} at {current_price}"
                }
        
        return None
    
    def _check_exit_conditions(self, signal: TradingSignal, current_price: float, 
                             current_time: datetime) -> Optional[Dict]:
        """Check if exit conditions are met"""
        if not signal.entry_triggered or signal.status == 'closed':
            return None
        
        # Check stop loss
        if signal.signal_type == 'long' and current_price <= signal.stop_loss_price:
            return self._close_signal(signal, current_price, 'stop_loss')
        
        if signal.signal_type == 'short' and current_price >= signal.stop_loss_price:
            return self._close_signal(signal, current_price, 'stop_loss')
        
        # Check take profit
        if signal.signal_type == 'long' and current_price >= signal.take_profit_price:
            return self._close_signal(signal, current_price, 'take_profit')
        
        if signal.signal_type == 'short' and current_price <= signal.take_profit_price:
            return self._close_signal(signal, current_price, 'take_profit')
        
        # Check time-based exit (if signal is too old)
        if current_time > signal.timestamp + timedelta(hours=48):
            return self._close_signal(signal, current_price, 'timeout')
        
        return None
    
    def _check_pullback_conditions(self, signal: TradingSignal, df: pd.DataFrame) -> Optional[Dict]:
        """Check for pullback detection"""
        if signal.pullback_detected:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Calculate VHMA color
        df['vhma'] = self.analyzer.calculate_vhma_vectorized(df)
        df['vhma_color'] = np.where(df['vhma'] > df['vhma'].shift(1), 'green', 'red')
        
        # Check for red VHMA areas (pullback indication)
        recent_colors = df['vhma_color'].tail(5).tolist()
        red_count = recent_colors.count('red')
        
        if red_count >= 2:  # At least 2 red candles in last 5
            signal.pullback_detected = True
            
            if signal.signal_type == 'long':
                signal.pullback_low_price = min(signal.pullback_low_price or current_price, current_price)
            else:
                signal.pullback_high_price = max(signal.pullback_high_price or current_price, current_price)
            
            return {
                'type': 'pullback_detected',
                'signal': signal,
                'message': f"Pullback detected for {signal.symbol} at {current_price}"
            }
        
        return None
    
    def _close_signal(self, signal: TradingSignal, exit_price: float, reason: str) -> Dict:
        """Close a trading signal and calculate P&L"""
        signal.exit_price = exit_price
        signal.exit_reason = reason
        signal.status = 'closed'
        
        # Calculate P&L
        if signal.entry_price:
            if signal.signal_type == 'long':
                signal.pnl_percentage = ((exit_price - signal.entry_price) / signal.entry_price) * 100
            else:
                signal.pnl_percentage = ((signal.entry_price - exit_price) / signal.entry_price) * 100
        
        return {
            'type': 'signal_closed',
            'signal': signal,
            'message': f"Signal closed for {signal.symbol}: {reason}, P&L: {signal.pnl_percentage:.2f}%"
        }
    
    def get_signal_dashboard_data(self, signals: List[TradingSignal]) -> Dict:
        """Generate dashboard data for trading signals"""
        active_signals = [s for s in signals if s.status != 'closed']
        closed_signals = [s for s in signals if s.status == 'closed']
        
        # Performance metrics
        if closed_signals:
            winning_trades = [s for s in closed_signals if s.pnl_percentage and s.pnl_percentage > 0]
            win_rate = len(winning_trades) / len(closed_signals) * 100
            avg_win = np.mean([s.pnl_percentage for s in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([s.pnl_percentage for s in closed_signals if s.pnl_percentage and s.pnl_percentage < 0]) or 0
            total_pnl = sum(s.pnl_percentage for s in closed_signals if s.pnl_percentage)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            total_pnl = 0
        
        return {
            'summary': {
                'total_signals': len(signals),
                'active_signals': len(active_signals),
                'closed_signals': len(closed_signals),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'active_signals': [asdict(s) for s in active_signals],
            'recent_closed': [asdict(s) for s in closed_signals[-10:]],
            'performance_by_timeframe': self._group_performance_by_timeframe(closed_signals),
            'performance_by_symbol': self._group_performance_by_symbol(closed_signals)
        }
    
    def _group_performance_by_timeframe(self, signals: List[TradingSignal]) -> Dict:
        """Group performance metrics by timeframe"""
        grouped = {}
        for signal in signals:
            if signal.timeframe not in grouped:
                grouped[signal.timeframe] = []
            grouped[signal.timeframe].append(signal)
        
        result = {}
        for timeframe, tf_signals in grouped.items():
            pnls = [s.pnl_percentage for s in tf_signals if s.pnl_percentage is not None]
            if pnls:
                result[timeframe] = {
                    'count': len(tf_signals),
                    'avg_pnl': np.mean(pnls),
                    'total_pnl': sum(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100
                }
        
        return result
    
    def _group_performance_by_symbol(self, signals: List[TradingSignal]) -> Dict:
        """Group performance metrics by symbol"""
        grouped = {}
        for signal in signals:
            if signal.symbol not in grouped:
                grouped[signal.symbol] = []
            grouped[signal.symbol].append(signal)
        
        result = {}
        for symbol, symbol_signals in grouped.items():
            pnls = [s.pnl_percentage for s in symbol_signals if s.pnl_percentage is not None]
            if pnls:
                result[symbol] = {
                    'count': len(symbol_signals),
                    'avg_pnl': np.mean(pnls),
                    'total_pnl': sum(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100
                }
        
        return result

# Global instance
_signal_generator = None

def get_signal_generator() -> SuperZTradingSignalGenerator:
    """Get singleton signal generator instance"""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = SuperZTradingSignalGenerator()
    return _signal_generator
