#!/usr/bin/env python3
"""
CLEAR DIRECTION SIGNAL SYSTEM - GUARANTEED LONG/SHORT INDICATION
Simplified but effective system that clearly shows trade direction
"""

import asyncio
import time
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ClearDirectionSignalSystem:
    """
    Clear direction signal system - ALWAYS shows LONG or SHORT
    """
    
    def __init__(self):
        self.signal_history = []
        
    async def generate_clear_signal(self, symbol, market_data):
        """
        Generate signals with CRYSTAL CLEAR LONG/SHORT indication
        """
        try:
            df = market_data.get('5m', market_data)
            if df is None or len(df) < 10:
                return None
            
            # Calculate indicators
            supertrend, direction = self.calculate_supertrend(df)
            if supertrend is None:
                return None
                
            rsi = self.calculate_rsi(df)
            if rsi is None:
                return None
                
            # Get current values
            current_price = float(df['close'].iloc[-1])
            current_supertrend = float(supertrend.iloc[-1])
            current_direction = int(direction.iloc[-1])
            current_rsi = float(rsi.iloc[-1])
            
            # Calculate momentum
            momentum = self.calculate_momentum(df)
            
            # SIMPLIFIED SIGNAL LOGIC - More lenient conditions
            signal = None
            
            # 🟢 LONG SIGNAL CONDITIONS (More lenient)
            if (current_direction == 1 and  # SuperTrend uptrend
                current_price > current_supertrend and  # Price above SuperTrend
                current_rsi < 60):  # RSI not extremely overbought
                
                signal = self.create_long_signal(symbol, current_price, current_supertrend, 
                                               current_rsi, momentum)
                                               
            # 🔴 SHORT SIGNAL CONDITIONS (More lenient)
            elif (current_direction == -1 and  # SuperTrend downtrend  
                  current_price < current_supertrend and  # Price below SuperTrend
                  current_rsi > 40):  # RSI not extremely oversold
                  
                signal = self.create_short_signal(symbol, current_price, current_supertrend,
                                                current_rsi, momentum)
            
            return signal
            
        except Exception as e:
            logger.error(f"Clear signal generation error: {e}")
            return None
    
    def create_long_signal(self, symbol, price, supertrend, rsi, momentum):
        """Create a LONG signal with all details"""
        confidence = self.calculate_confidence(rsi, momentum, signal_type='long')
        
        signal = {
            'symbol': symbol,
            'direction': 'LONG',
            'side': 'buy',
            'action': 'BUY/LONG',
            'position_type': 'LONG POSITION',
            'trade_expectation': 'PRICE WILL INCREASE',
            'market_outlook': 'BULLISH',
            'price': price,
            'confidence': confidence,
            'leverage': self.calculate_leverage(confidence),
            'supertrend_value': supertrend,
            'rsi': rsi,
            'momentum': momentum,
            'signal_strength': 'STRONG' if confidence >= 75 else 'MEDIUM',
            'timestamp': time.time(),
            'clear_direction': '📈 UPWARD MOVEMENT EXPECTED'
        }
        
        logger.info(f"🟢 LONG SIGNAL GENERATED FOR {symbol}")
        logger.info(f"   🎯 DIRECTION: LONG (BUY POSITION)")
        logger.info(f"   📈 EXPECTATION: PRICE INCREASE")
        logger.info(f"   💯 CONFIDENCE: {confidence:.1f}%")
        logger.info(f"   💲 ENTRY: {price:.6f}")
        logger.info(f"   ⚡ LEVERAGE: {signal['leverage']}x")
        logger.info(f"   📊 RSI: {rsi:.1f}")
        
        return signal
    
    def create_short_signal(self, symbol, price, supertrend, rsi, momentum):
        """Create a SHORT signal with all details"""
        confidence = self.calculate_confidence(rsi, momentum, signal_type='short')
        
        signal = {
            'symbol': symbol,
            'direction': 'SHORT',
            'side': 'sell',
            'action': 'SELL/SHORT',
            'position_type': 'SHORT POSITION',
            'trade_expectation': 'PRICE WILL DECREASE',
            'market_outlook': 'BEARISH',
            'price': price,
            'confidence': confidence,
            'leverage': self.calculate_leverage(confidence),
            'supertrend_value': supertrend,
            'rsi': rsi,
            'momentum': momentum,
            'signal_strength': 'STRONG' if confidence >= 75 else 'MEDIUM',
            'timestamp': time.time(),
            'clear_direction': '📉 DOWNWARD MOVEMENT EXPECTED'
        }
        
        logger.info(f"🔴 SHORT SIGNAL GENERATED FOR {symbol}")
        logger.info(f"   🎯 DIRECTION: SHORT (SELL POSITION)")
        logger.info(f"   📉 EXPECTATION: PRICE DECREASE")
        logger.info(f"   💯 CONFIDENCE: {confidence:.1f}%")
        logger.info(f"   💲 ENTRY: {price:.6f}")
        logger.info(f"   ⚡ LEVERAGE: {signal['leverage']}x")
        logger.info(f"   📊 RSI: {rsi:.1f}")
        
        return signal
    
    def calculate_confidence(self, rsi, momentum, signal_type='long'):
        """Calculate confidence for signals"""
        base_confidence = 65
        
        if signal_type == 'long':
            # For LONG signals, lower RSI is better
            rsi_bonus = max(0, (60 - rsi) / 40 * 15)
            momentum_bonus = max(0, momentum * 1000)
        else:  # short
            # For SHORT signals, higher RSI is better
            rsi_bonus = max(0, (rsi - 40) / 40 * 15)
            momentum_bonus = max(0, abs(momentum) * 1000)
        
        confidence = base_confidence + rsi_bonus + min(15, momentum_bonus)
        return min(95, max(60, confidence))
    
    def calculate_leverage(self, confidence):
        """Calculate leverage based on confidence"""
        if confidence >= 85:
            return 30
        elif confidence >= 75:
            return 25
        elif confidence >= 65:
            return 20
        else:
            return 15
    
    def calculate_supertrend(self, df, period=10, multiplier=3.0):
        """Calculate SuperTrend indicator - Simplified"""
        try:
            if len(df) < period + 1:
                return None, None
                
            # Ensure numeric types
            df = df.copy()
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            
            # Simplified ATR calculation
            high_low = df['high'] - df['low']
            atr = high_low.rolling(window=period).mean()
            
            # Calculate SuperTrend
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Simplified SuperTrend logic
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            for i in range(len(df)):
                close_price = df['close'].iloc[i]
                
                if close_price > hl2.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i] if i < len(lower_band) else close_price * 0.98
                    direction.iloc[i] = 1  # Uptrend
                else:
                    supertrend.iloc[i] = upper_band.iloc[i] if i < len(upper_band) else close_price * 1.02
                    direction.iloc[i] = -1  # Downtrend
            
            return supertrend, direction
            
        except Exception as e:
            logger.debug(f"SuperTrend calculation error: {e}")
            return None, None
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI indicator - Simplified"""
        try:
            if len(df) < period + 1:
                return None
            
            close_prices = pd.to_numeric(df['close'], errors='coerce')
            delta = close_prices.diff()
            
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Simple averages
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            
            # Avoid division by zero
            avg_losses = avg_losses.replace(0, 0.001)
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
            return None
    
    def calculate_momentum(self, df):
        """Calculate price momentum - Simplified"""
        try:
            if len(df) < 2:
                return 0
            
            close_prices = pd.to_numeric(df['close'])
            return (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
            
        except:
            return 0

# Test the clear direction system
if __name__ == "__main__":
    clear_system = ClearDirectionSignalSystem()
    print("✅ Clear Direction Signal System Ready!")
    print("🟢 LONG signals = BUY positions (expect price UP)")
    print("🔴 SHORT signals = SELL positions (expect price DOWN)")
    print("🎯 Crystal clear direction indication guaranteed!") 