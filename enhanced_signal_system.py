#!/usr/bin/env python3
"""
ENHANCED SIGNAL SYSTEM - CLEAR LONG/SHORT INDICATION
Provides crystal clear signal direction for trading
"""

import asyncio
import time
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class EnhancedSignalSystem:
    """
    Enhanced signal system with CRYSTAL CLEAR LONG/SHORT indication
    """
    
    def __init__(self):
        self.signal_history = []
        
    async def generate_enhanced_signal(self, symbol, market_data):
        """
        Generate signals with CLEAR LONG/SHORT indication
        """
        try:
            df = market_data.get('5m', market_data)
            if df is None or len(df) < 20:
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
            
            # Calculate momentum and volume
            momentum = self.calculate_momentum(df)
            volume_score = self.calculate_volume_score(df)
            
            # SIGNAL DECISION LOGIC
            signal = None
            
            # 🟢 LONG SIGNAL CONDITIONS
            if self.is_long_signal(current_direction, current_price, current_supertrend, 
                                 current_rsi, momentum, volume_score):
                signal = self.create_long_signal(symbol, current_price, current_supertrend, 
                                               current_rsi, momentum, volume_score)
                                               
            # 🔴 SHORT SIGNAL CONDITIONS
            elif self.is_short_signal(current_direction, current_price, current_supertrend,
                                    current_rsi, momentum, volume_score):
                signal = self.create_short_signal(symbol, current_price, current_supertrend,
                                                current_rsi, momentum, volume_score)
            
            return signal
            
        except Exception as e:
            logger.error(f"Enhanced signal generation error: {e}")
            return None
    
    def is_long_signal(self, direction, price, supertrend, rsi, momentum, volume):
        """Check if conditions meet LONG signal criteria"""
        return (
            direction == 1 and              # SuperTrend uptrend
            price > supertrend and          # Price above SuperTrend
            20 < rsi < 50 and              # RSI in oversold to neutral range
            momentum > 0.001 and           # Positive momentum
            volume > 60                    # Adequate volume
        )
    
    def is_short_signal(self, direction, price, supertrend, rsi, momentum, volume):
        """Check if conditions meet SHORT signal criteria"""
        return (
            direction == -1 and            # SuperTrend downtrend
            price < supertrend and         # Price below SuperTrend
            50 < rsi < 80 and             # RSI in overbought to neutral range
            momentum < -0.001 and         # Negative momentum
            volume > 60                   # Adequate volume
        )
    
    def create_long_signal(self, symbol, price, supertrend, rsi, momentum, volume):
        """Create a LONG signal with all details"""
        confidence = self.calculate_long_confidence(rsi, momentum, volume)
        
        signal = {
            'symbol': symbol,
            'direction': 'LONG',
            'side': 'buy',
            'action': 'BUY/LONG',
            'position_type': 'LONG POSITION',
            'price': price,
            'confidence': confidence,
            'leverage': self.calculate_leverage(confidence),
            'supertrend_value': supertrend,
            'rsi': rsi,
            'momentum': momentum,
            'volume_score': volume,
            'expected_outcome': 'PRICE INCREASE',
            'signal_strength': 'STRONG' if confidence >= 75 else 'MEDIUM',
            'timestamp': time.time()
        }
        
        logger.info(f"🟢 LONG SIGNAL GENERATED FOR {symbol}")
        logger.info(f"   🎯 DIRECTION: LONG (BUY)")
        logger.info(f"   📈 EXPECTATION: PRICE INCREASE")
        logger.info(f"   💯 CONFIDENCE: {confidence:.1f}%")
        logger.info(f"   💲 ENTRY PRICE: {price:.6f}")
        logger.info(f"   ⚡ LEVERAGE: {signal['leverage']}x")
        
        return signal
    
    def create_short_signal(self, symbol, price, supertrend, rsi, momentum, volume):
        """Create a SHORT signal with all details"""
        confidence = self.calculate_short_confidence(rsi, momentum, volume)
        
        signal = {
            'symbol': symbol,
            'direction': 'SHORT',
            'side': 'sell',
            'action': 'SELL/SHORT',
            'position_type': 'SHORT POSITION',
            'price': price,
            'confidence': confidence,
            'leverage': self.calculate_leverage(confidence),
            'supertrend_value': supertrend,
            'rsi': rsi,
            'momentum': momentum,
            'volume_score': volume,
            'expected_outcome': 'PRICE DECREASE',
            'signal_strength': 'STRONG' if confidence >= 75 else 'MEDIUM',
            'timestamp': time.time()
        }
        
        logger.info(f"🔴 SHORT SIGNAL GENERATED FOR {symbol}")
        logger.info(f"   🎯 DIRECTION: SHORT (SELL)")
        logger.info(f"   📉 EXPECTATION: PRICE DECREASE")
        logger.info(f"   💯 CONFIDENCE: {confidence:.1f}%")
        logger.info(f"   💲 ENTRY PRICE: {price:.6f}")
        logger.info(f"   ⚡ LEVERAGE: {signal['leverage']}x")
        
        return signal
    
    def calculate_long_confidence(self, rsi, momentum, volume):
        """Calculate confidence for LONG signals"""
        base_confidence = 60
        
        # RSI bonus (lower RSI = higher confidence for longs)
        rsi_bonus = max(0, (50 - rsi) / 30 * 15)  # 0-15 points
        
        # Momentum bonus
        momentum_bonus = min(10, momentum * 2500)  # 0-10 points
        
        # Volume bonus
        volume_bonus = min(10, (volume - 60) / 40 * 10)  # 0-10 points
        
        confidence = base_confidence + rsi_bonus + momentum_bonus + volume_bonus
        return min(95, max(60, confidence))
    
    def calculate_short_confidence(self, rsi, momentum, volume):
        """Calculate confidence for SHORT signals"""
        base_confidence = 60
        
        # RSI bonus (higher RSI = higher confidence for shorts)
        rsi_bonus = max(0, (rsi - 50) / 30 * 15)  # 0-15 points
        
        # Momentum bonus (negative momentum)
        momentum_bonus = min(10, abs(momentum) * 2500)  # 0-10 points
        
        # Volume bonus
        volume_bonus = min(10, (volume - 60) / 40 * 10)  # 0-10 points
        
        confidence = base_confidence + rsi_bonus + momentum_bonus + volume_bonus
        return min(95, max(60, confidence))
    
    def calculate_leverage(self, confidence):
        """Calculate leverage based on confidence"""
        if confidence >= 85:
            return 25
        elif confidence >= 75:
            return 20
        elif confidence >= 65:
            return 15
        else:
            return 10
    
    def calculate_supertrend(self, df, period=10, multiplier=3.0):
        """Calculate SuperTrend indicator"""
        try:
            if len(df) < period + 1:
                return None, None
                
            # Ensure numeric types
            df = df.copy()
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            
            # Calculate ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            # Calculate SuperTrend
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Initialize SuperTrend series
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            for i in range(period, len(df)):
                close_price = df['close'].iloc[i]
                
                if i == period:
                    # First calculation
                    if close_price > hl2.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                else:
                    # Subsequent calculations
                    if direction.iloc[i-1] == 1:  # Was uptrend
                        if close_price > lower_band.iloc[i]:
                            supertrend.iloc[i] = lower_band.iloc[i]
                            direction.iloc[i] = 1
                        else:
                            supertrend.iloc[i] = upper_band.iloc[i]
                            direction.iloc[i] = -1
                    else:  # Was downtrend
                        if close_price < upper_band.iloc[i]:
                            supertrend.iloc[i] = upper_band.iloc[i]
                            direction.iloc[i] = -1
                        else:
                            supertrend.iloc[i] = lower_band.iloc[i]
                            direction.iloc[i] = 1
            
            return supertrend, direction
            
        except Exception as e:
            logger.debug(f"SuperTrend calculation error: {e}")
            return None, None
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        try:
            if len(df) < period + 1:
                return None
            
            close_prices = pd.to_numeric(df['close'], errors='coerce')
            delta = close_prices.diff()
            
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Avoid division by zero
            avg_losses = avg_losses.replace(0, 0.001)
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
            return None
    
    def calculate_momentum(self, df):
        """Calculate price momentum"""
        try:
            if len(df) < 2:
                return 0
            
            close_prices = pd.to_numeric(df['close'])
            return (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
            
        except:
            return 0
    
    def calculate_volume_score(self, df):
        """Calculate volume score"""
        try:
            if len(df) < 5:
                return 50
            
            volumes = pd.to_numeric(df['volume'])
            avg_volume = volumes.rolling(5).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_score = min(100, max(0, 50 + (volume_ratio - 1) * 25))
            
            return volume_score
            
        except:
            return 50

# Test the enhanced system
if __name__ == "__main__":
    enhanced_system = EnhancedSignalSystem()
    print("✅ Enhanced Signal System Ready!")
    print("🟢 LONG signals = BUY positions (expect price increase)")
    print("🔴 SHORT signals = SELL positions (expect price decrease)") 