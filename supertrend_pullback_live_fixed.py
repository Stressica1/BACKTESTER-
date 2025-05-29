#!/usr/bin/env python3
"""
FIXED SuperTrend Pullback Trading Bot with CRYSTAL CLEAR LONG/SHORT INDICATION
- Sets leverage FIRST
- Uses 0.50 USDT as margin (not position value)
- Clear LONG/SHORT signal direction
- No minimum balance requirements
"""

import ccxt
import asyncio
import pandas as pd
import numpy as np
import time
import logging
import json
import os
import sqlite3
from collections import deque
import random

# Setup logging with color coding
class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    blue = "\x1b[34;20m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """Setup enhanced logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(CustomFormatter())
    
    # File handler
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    file_handler = logging.FileHandler("logs/trading_bot.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

class FixedSuperTrendBot:
    """Fixed SuperTrend bot with clear LONG/SHORT indication"""
    
    def __init__(self, config_file="config/bitget_config.json", simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.config = self.load_config(config_file)
        self.exchange = self.setup_exchange()
        
        # Trading parameters
        self.st_period = 10
        self.st_multiplier = 3.0
        self.fixed_margin = 0.50  # Fixed 0.50 USDT margin
        
        # Rate limiting
        self.rate_limits = {
            'fetch_ticker': 20,
            'fetch_ohlcv': 20,
            'create_order': 10,
            'set_leverage': 10,
            'fetch_balance': 10
        }
        self.rate_limit_counters = {key: deque() for key in self.rate_limits}
        
        # Statistics
        self.signal_stats = {'total_signals': 0, 'trades_executed': 0}
        
        logger.info("🤖 Fixed SuperTrend Bot initialized")
        logger.info(f"   💰 Fixed margin: {self.fixed_margin} USDT")
        logger.info(f"   🎯 Clear LONG/SHORT signal indication enabled")
        logger.info(f"   ⚡ Leverage-first execution enabled")
    
    def load_config(self, config_file):
        """Load configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Configuration loaded")
                return config
            else:
                logger.warning("⚠️ Config file not found, using defaults")
                return {
                    "api_key": "", "secret": "", "passphrase": "", 
                    "sandbox": True
                }
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            return {"api_key": "", "secret": "", "passphrase": "", "sandbox": True}
    
    def setup_exchange(self):
        """Setup exchange"""
        if self.simulation_mode:
            return self.create_mock_exchange()
        
        try:
            exchange = ccxt.bitget({
                "apiKey": self.config["api_key"],
                "secret": self.config["secret"],
                "password": self.config["passphrase"],
                "sandbox": self.config.get("sandbox", True),
                "timeout": 10000,
                "rateLimit": 100,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"}
            })
            
            exchange.load_markets()
            logger.info("✅ Exchange connected")
            return exchange
        except Exception as e:
            logger.error(f"Exchange setup failed: {e}")
            logger.warning("Falling back to simulation mode")
            self.simulation_mode = True
            return self.create_mock_exchange()
    
    def create_mock_exchange(self):
        """Create mock exchange for simulation"""
        class MockExchange:
            def __init__(self):
                self.balance = {'USDT': {'free': 10000, 'used': 0, 'total': 10000}}
                
            def fetch_ticker(self, symbol):
                time.sleep(0.001)
                base_price = self._get_base_price(symbol)
                return {
                    'last': base_price + random.uniform(-base_price*0.01, base_price*0.01),
                    'timestamp': time.time() * 1000
                }
                
            def fetch_ohlcv(self, symbol, timeframe='5m', since=None, limit=100):
                time.sleep(0.001)
                return self._generate_realistic_ohlcv(symbol, limit)
                
            def fetch_balance(self):
                time.sleep(0.001)
                return self.balance
                
            def create_market_order(self, symbol, side, amount, price=None, params=None):
                time.sleep(0.002)
                current_price = self._get_base_price(symbol)
                
                return {
                    'id': f'mock_{int(time.time() * 1000)}',
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'filled': amount,
                    'status': 'closed',
                    'price': current_price,
                    'timestamp': time.time(),
                    'fees': {'cost': amount * 0.001, 'currency': 'USDT'}
                }
                
            def set_leverage(self, leverage, symbol, params=None):
                time.sleep(0.001)
                return {'leverage': leverage, 'symbol': symbol}
                
            def load_markets(self):
                time.sleep(0.001)
                return {}
                
            def _get_base_price(self, symbol):
                prices = {
                    'BTC/USDT': 95000, 'ETH/USDT': 3400, 'SOL/USDT': 210,
                    'BNB/USDT': 650, 'XRP/USDT': 2.3, 'ADA/USDT': 0.95
                }
                return prices.get(symbol, random.uniform(1, 100))
                
            def _generate_realistic_ohlcv(self, symbol, limit):
                base_price = self._get_base_price(symbol)
                now = int(time.time() * 1000)
                interval_ms = 5 * 60 * 1000
                
                ohlcv = []
                current_price = base_price
                
                for i in range(limit):
                    timestamp = now - (limit - i) * interval_ms
                    change = random.uniform(-0.02, 0.02)
                    new_price = current_price * (1 + change)
                    
                    open_price = current_price
                    close_price = new_price
                    high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                    low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                    volume = random.uniform(1000, 10000)
                    
                    ohlcv.append([timestamp, open_price, high_price, low_price, close_price, volume])
                    current_price = new_price
                
                return ohlcv
                
        return MockExchange()
    
    async def rate_limit(self, endpoint='default'):
        """Rate limiting"""
        current_time = time.time()
        limit = self.rate_limits.get(endpoint, 10)
        window = self.rate_limit_counters[endpoint]
        
        while window and current_time - window[0] > 1.0:
            window.popleft()
        
        if len(window) >= limit:
            sleep_time = 1.0 - (current_time - window[0]) + 0.01
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        window.append(current_time)
    
    async def get_market_data(self, symbol):
        """Get market data"""
        try:
            await self.rate_limit('fetch_ohlcv')
            ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Market data error for {symbol}: {e}")
            return None
    
    def calculate_supertrend(self, df, period=10, multiplier=3.0):
        """Calculate SuperTrend indicator"""
        try:
            if len(df) < period + 1:
                return None, None
                
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
        """Calculate RSI indicator"""
        try:
            if len(df) < period + 1:
                return None
            
            close_prices = pd.to_numeric(df['close'], errors='coerce')
            delta = close_prices.diff()
            
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            
            avg_losses = avg_losses.replace(0, 0.001)
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.debug(f"RSI calculation error: {e}")
            return None
    
    def calculate_momentum_simple(self, df):
        """Calculate price momentum"""
        try:
            if len(df) < 2:
                return 0
            
            close_prices = pd.to_numeric(df['close'])
            return (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
            
        except:
            return 0
    
    async def generate_signal(self, symbol):
        """🎯 CLEAR DIRECTION SIGNAL GENERATION - CRYSTAL CLEAR LONG/SHORT INDICATION"""
        try:
            # Get market data
            df = await self.get_market_data(symbol)
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
            momentum = self.calculate_momentum_simple(df)
            
            # SIMPLIFIED SIGNAL LOGIC - Clear LONG/SHORT conditions
            signal = None
            
            # 🟢 LONG SIGNAL CONDITIONS (Clear and simple)
            if (current_direction == 1 and  # SuperTrend uptrend
                current_price > current_supertrend and  # Price above SuperTrend
                current_rsi < 60):  # RSI not extremely overbought
                
                signal = self.create_long_signal(symbol, current_price, current_supertrend, 
                                               current_rsi, momentum)
                                               
            # 🔴 SHORT SIGNAL CONDITIONS (Clear and simple)
            elif (current_direction == -1 and  # SuperTrend downtrend  
                  current_price < current_supertrend and  # Price below SuperTrend
                  current_rsi > 40):  # RSI not extremely oversold
                  
                signal = self.create_short_signal(symbol, current_price, current_supertrend,
                                                current_rsi, momentum)
            
            if signal:
                self.signal_stats['total_signals'] += 1
                
            return signal
                
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def create_long_signal(self, symbol, price, supertrend, rsi, momentum):
        """Create a LONG signal with all details"""
        confidence = self.calculate_confidence_simple(rsi, momentum, signal_type='long')
        
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
            'leverage': self.calculate_leverage_simple(confidence),
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
        logger.info(f"   🟢 EXECUTION: BUY ORDER - GOING LONG")
        logger.info(f"   📈 EXPECTATION: PRICE WILL GO UP")
        
        return signal
    
    def create_short_signal(self, symbol, price, supertrend, rsi, momentum):
        """Create a SHORT signal with all details"""
        confidence = self.calculate_confidence_simple(rsi, momentum, signal_type='short')
        
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
            'leverage': self.calculate_leverage_simple(confidence),
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
        logger.info(f"   🔴 EXECUTION: SELL ORDER - GOING SHORT")
        logger.info(f"   📉 EXPECTATION: PRICE WILL GO DOWN")
        
        return signal
    
    def calculate_confidence_simple(self, rsi, momentum, signal_type='long'):
        """Calculate confidence for signals"""
        base_confidence = 70
        
        if signal_type == 'long':
            # For LONG signals, lower RSI is better
            rsi_bonus = max(0, (60 - rsi) / 40 * 15)
            momentum_bonus = max(0, momentum * 1000)
        else:  # short
            # For SHORT signals, higher RSI is better
            rsi_bonus = max(0, (rsi - 40) / 40 * 15)
            momentum_bonus = max(0, abs(momentum) * 1000)
        
        confidence = base_confidence + rsi_bonus + min(15, momentum_bonus)
        return min(95, max(65, confidence))
    
    def calculate_leverage_simple(self, confidence):
        """Calculate leverage based on confidence"""
        if confidence >= 85:
            return 30
        elif confidence >= 75:
            return 25
        elif confidence >= 65:
            return 20
        else:
            return 15
    
    async def execute_trade(self, signal):
        """FIXED: Execute trade with LEVERAGE FIRST, then 0.50 USDT margin"""
        execution_start = time.time()
        
        try:
            symbol = signal['symbol']
            side = signal['side']
            leverage = signal.get('leverage', 20)
            
            # STEP 1: SET LEVERAGE FIRST (CRITICAL)
            logger.info(f"🔧 SETTING LEVERAGE FIRST: {symbol} -> {leverage}x")
            
            if not self.simulation_mode:
                try:
                    await self.rate_limit('set_leverage')
                    await self.set_leverage(symbol, leverage)
                    logger.info(f"✅ Leverage set: {leverage}x")
                except Exception as e:
                    logger.error(f"❌ Leverage setting failed: {e}")
                    return None
            else:
                logger.info(f"📊 SIMULATION: Leverage set to {leverage}x")
            
            # STEP 2: Get current price
            await self.rate_limit('fetch_ticker')
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = float(ticker['last'])
            
            # STEP 3: Calculate position details
            margin_usdt = self.fixed_margin  # 0.50 USDT margin
            effective_position_value = margin_usdt * leverage  # Effective position value
            quantity = effective_position_value / current_price  # Quantity to trade
            
            # LOG CRYSTAL CLEAR TRADE EXECUTION
            logger.info(f"⚡ EXECUTING {signal.get('direction', signal['side'].upper())} TRADE: {symbol}")
            logger.info(f"   🎯 DIRECTION: {signal.get('direction', 'LONG' if signal['side'] == 'buy' else 'SHORT')}")
            logger.info(f"   📊 POSITION: {signal.get('position_type', 'LONG POSITION' if signal['side'] == 'buy' else 'SHORT POSITION')}")
            logger.info(f"   📈 EXPECTATION: {signal.get('trade_expectation', 'PRICE INCREASE' if signal['side'] == 'buy' else 'PRICE DECREASE')}")
            logger.info(f"   💰 Margin Used: {margin_usdt} USDT")
            logger.info(f"   📈 Leverage: {leverage}x") 
            logger.info(f"   💵 Effective Position: {effective_position_value} USDT")
            logger.info(f"   📊 Quantity: {quantity:.6f} coins")
            logger.info(f"   💲 Entry Price: {current_price:.6f}")
            logger.info(f"   📋 Side: {side.upper()} ({'🟢 LONG' if side == 'buy' else '🔴 SHORT'})")
            
            # STEP 4: Execute the trade
            if not self.simulation_mode:
                await self.rate_limit('create_order')
                order = self.exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=quantity,
                    params={'marginMode': 'isolated'}
                )
                
                logger.info(f"✅ TRADE EXECUTED - ORDER ID: {order.get('id', 'N/A')}")
                self.signal_stats['trades_executed'] += 1
                
                return order
            else:
                logger.info(f"📊 SIMULATION: {side.upper()} order executed for {quantity:.6f} coins")
                self.signal_stats['trades_executed'] += 1
                
                return {
                    'id': f'sim_{int(time.time() * 1000)}',
                    'symbol': symbol,
                    'side': side,
                    'amount': quantity,
                    'price': current_price,
                    'status': 'closed',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
            return None
    
    async def set_leverage(self, symbol, leverage):
        """Set leverage for symbol"""
        try:
            result = self.exchange.set_leverage(leverage, symbol)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Failed to set leverage: {e}")
            raise
    
    async def run_test(self):
        """Run test with clear LONG/SHORT signals"""
        logger.info("🚀 STARTING FIXED BOT TEST - CLEAR LONG/SHORT SIGNALS")
        logger.info("=" * 80)
        
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
        
        for symbol in test_symbols:
            logger.info(f"\n🔍 Testing {symbol}...")
            
            signal = await self.generate_signal(symbol)
            
            if signal:
                logger.info(f"✅ SIGNAL GENERATED FOR {symbol}")
                logger.info(f"   🎯 DIRECTION: {signal['direction']}")
                logger.info(f"   📊 POSITION: {signal['position_type']}")
                logger.info(f"   📈 EXPECTATION: {signal['trade_expectation']}")
                logger.info(f"   💯 CONFIDENCE: {signal['confidence']:.1f}%")
                logger.info(f"   ⚡ LEVERAGE: {signal['leverage']}x")
                
                # Execute trade
                trade_result = await self.execute_trade(signal)
                if trade_result:
                    logger.info(f"✅ Trade executed successfully")
                else:
                    logger.error(f"❌ Trade execution failed")
            else:
                logger.info(f"⚪ No signal for {symbol}")
            
            await asyncio.sleep(1)  # Small delay between tests
        
        # Summary
        logger.info("\n📊 TEST SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"🔢 Total signals: {self.signal_stats['total_signals']}")
        logger.info(f"⚡ Trades executed: {self.signal_stats['trades_executed']}")
        logger.info("✅ FIXED BOT TEST COMPLETE!")
        logger.info("🎯 All signals now show CRYSTAL CLEAR LONG/SHORT direction!")

if __name__ == "__main__":
    async def main():
        bot = FixedSuperTrendBot(simulation_mode=True)
        await bot.run_test()
    
    asyncio.run(main()) 