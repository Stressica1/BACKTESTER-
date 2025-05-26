import json
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import websockets
import colorama
from colorama import Fore, Style, Back
colorama.init(autoreset=True)
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 
import concurrent.futures
from tqdm import tqdm
import prettytable
import multiprocessing
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SuperTrend")

def fetch_ohlcv_data(args):
    symbol, tf, exchange_config = args
    try:
        # Create a new exchange instance for each process
        exchange_id = exchange_config['exchange']
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': exchange_config['api_key'],
            'secret': exchange_config['api_secret'],
            'enableRateLimit': True,
        })
        if exchange_config.get('use_sandbox', False):
            exchange.set_sandbox_mode(True)
            
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
        if not ohlcv or len(ohlcv) == 0:
            return (symbol, tf), None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return (symbol, tf), df
    except Exception:
        logger.exception(f"Error fetching {symbol} {tf}")
        return (symbol, tf), None

class EnhancedSuperTrend:
    def __init__(self, config_file='test_batch.json'):
        """Initialize the Enhanced SuperTrend strategy"""
        self.config = self.load_config(config_file)
        print(f"\nLoaded config: {json.dumps(self.config, indent=2)}\n")
        self.exchange = self.setup_exchange()
        # Determine symbol list
        if self.config.get('testnet', {}).get('enabled') and self.config['testnet'].get('use_sandbox'):
            print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}Fetching all Bitget USDT-margined swap pairs from testnet...{Style.RESET_ALL}")
            try:
                markets = self.exchange.load_markets()
                symbols = [s for s, m in markets.items() if m.get('swap') and m.get('quote') == 'USDT']
            except Exception as e:
                logger.exception(f"Could not load markets from testnet: {e}. Using config symbols.")
                symbols = self.config.get('symbols', [])
            self.config['symbols'] = symbols
            print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}Scanning {len(symbols)} testnet pairs: {symbols[:10]}... (+{len(symbols)-10} more){Style.RESET_ALL}")
            print(f"[WARNING] Exchange is in testnet/sandbox mode. Some exchanges do not support OHLCV data in sandbox mode!\n")
        else:
            # Use symbols defined in configuration file
            symbols = self.config.get('symbols', [])
            print(f"{Fore.GREEN}Using symbols from configuration file: {symbols[:10]}... (+{len(symbols)-10} more if applicable){Style.RESET_ALL}")
            self.config['symbols'] = symbols
        self.active_trades = {}
        self.position_status = {}
        self.dashboard_data = {
            'strategy_status': 'idle',
            'active_symbols': [],
            'signals': [],
            'positions': [],
            'performance': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0
            }
        }
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
            logger.info(f"Configuration loaded successfully from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
        
    def setup_exchange(self):
        """Initialize exchange connection with testnet"""
        try:
            exchange_id = self.config['testnet']['exchange']
            exchange_class = getattr(ccxt, exchange_id)
            
            exchange = exchange_class({
                'apiKey': self.config['testnet']['api_key'],
                'secret': self.config['testnet']['api_secret'],
                'enableRateLimit': True,
            })
            
            # Use testnet if enabled
            if self.config['testnet']['use_sandbox']:
                exchange.set_sandbox_mode(True)
                logger.info(f"Connected to {exchange_id} testnet")
            else:
                logger.info(f"Connected to {exchange_id} live")
                
            return exchange
        
        except Exception as e:
            logger.error(f"Error setting up exchange: {e}")
            raise
        
    def calculate_supertrend(self, df, length=None, multiplier=None):
        """Calculate SuperTrend indicator"""
        length = length or self.config['st_length']
        multiplier = multiplier or self.config['st_multiplier']
        
        # Calculate ATR
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(length).mean()
        
        # Calculate SuperTrend
        df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
        df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
        df['in_uptrend'] = True
        
        for i in range(1, len(df)):
            if df['close'].loc[df.index[i]] > df['upperband'].loc[df.index[i-1]]:
                df['in_uptrend'].loc[df.index[i]] = True
            elif df['close'].loc[df.index[i]] < df['lowerband'].loc[df.index[i-1]]:
                df['in_uptrend'].loc[df.index[i]] = False
            else:
                df['in_uptrend'].loc[df.index[i]] = df['in_uptrend'].loc[df.index[i-1]]
                
                if df['in_uptrend'].loc[df.index[i]] and df['lowerband'].loc[df.index[i]] < df['lowerband'].loc[df.index[i-1]]:
                    df['lowerband'].loc[df.index[i]] = df['lowerband'].loc[df.index[i-1]]
                if not df['in_uptrend'].loc[df.index[i]] and df['upperband'].loc[df.index[i]] > df['upperband'].loc[df.index[i-1]]:
                    df['upperband'].loc[df.index[i]] = df['upperband'].loc[df.index[i-1]]
        
        # Add SuperTrend values
        df['supertrend'] = np.nan
        for i in range(len(df)):
            if df['in_uptrend'].loc[df.index[i]]:
                df['supertrend'].loc[df.index[i]] = df['lowerband'].loc[df.index[i]]
            else:
                df['supertrend'].loc[df.index[i]] = df['upperband'].loc[df.index[i]]
                
        return df
    
    def calculate_rsi(self, df, period=None):
        """Calculate RSI indicator"""
        period = period or self.config['enhanced_st']['rsi_period']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def apply_volume_filter(self, df):
        """Apply volume filter to signals"""
        if self.config['enhanced_st']['volume_filter']:
            # Calculate average volume
            df['avg_volume'] = df['volume'].rolling(20).mean()
            # Only consider signals when volume is above threshold
            df['volume_signal'] = df['volume'] > (df['avg_volume'] * self.config['enhanced_st']['volume_threshold'])
            return df['volume_signal']
        return pd.Series([True] * len(df), index=df.index)
    
    def apply_rsi_filter(self, df):
        """Apply RSI filter to signals"""
        if self.config['enhanced_st']['rsi_filter']:
            overbought = self.config['enhanced_st']['rsi_overbought']
            oversold = self.config['enhanced_st']['rsi_oversold']
            
            # Create RSI signals
            df['rsi_buy_signal'] = df['rsi'] < oversold
            df['rsi_sell_signal'] = df['rsi'] > overbought
            
            return df['rsi_buy_signal'], df['rsi_sell_signal']
        return pd.Series([True] * len(df), index=df.index), pd.Series([True] * len(df), index=df.index)
    
    def get_ohlcv_data(self, symbol, timeframe):
        """Fetch OHLCV data from exchange"""
        try:
            print(f"Requesting OHLCV for {symbol} {timeframe}...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            if not ohlcv or len(ohlcv) == 0:
                print(f"[ERROR] No OHLCV data returned for {symbol} {timeframe}!")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"First rows for {symbol} {timeframe}:\n{df.head()}\n")
            return df
        except Exception as e:
            print(f"[ERROR] Exception fetching data for {symbol} {timeframe}: {e}")
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol, entry_price):
        """Calculate position size based on risk parameters"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['total']['USDT']
            risk_amount = usdt_balance * (self.config['risk_management']['position_size_percent'] / 100)
            position_size = risk_amount / entry_price
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def place_order(self, symbol, side, amount, price=None):
        """Place order on exchange"""
        try:
            # Skip actual order placement if paper trading
            if not self.config['execution']['live_trading']:
                logger.info(f"Paper trading: {side} {amount} {symbol} @ {price}")
                return {
                    "id": f"paper_{int(time.time())}",
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price or self.exchange.fetch_ticker(symbol)['last'],
                    "timestamp": datetime.now().timestamp() * 1000,
                    "status": "filled",
                    "paper": True
                }
            
            # Place real order
            order_type = 'market'
            params = {}
            
            # Add stop loss and take profit for market orders
            if side == 'buy':
                stop_price = price * (1 - self.config['risk_management']['stop_loss_percent']/100)
                take_profit = price * (1 + self.config['risk_management']['take_profit_percent']/100)
            else:
                stop_price = price * (1 + self.config['risk_management']['stop_loss_percent']/100)
                take_profit = price * (1 - self.config['risk_management']['take_profit_percent']/100)
                
            # Place the main order
            order = self.exchange.create_order(
                symbol, order_type, side, amount, price, params
            )
            
            logger.info(f"Order placed: {side} {amount} {symbol} @ {price}")
            
            # Update dashboard data
            self.update_dashboard({
                'type': 'order',
                'data': {
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'filled',
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            
            # Update dashboard with error
            self.update_dashboard({
                'type': 'error',
                'data': {
                    'message': f"Error placing order: {e}",
                    'symbol': symbol,
                    'side': side,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            return None
    
    def analyze_symbol(self, symbol, timeframe=None):
        """Analyze a symbol for SuperTrend signals"""
        timeframe = timeframe or self.config['timeframe']
        
        # Fetch data
        df = self.get_ohlcv_data(symbol, timeframe)
        if df is None or len(df) < 50:
            logger.warning(f"Not enough data for {symbol}")
            return None
        
        # Calculate indicators
        df = self.calculate_supertrend(df)
        df = self.calculate_rsi(df)
        
        # Apply filters
        volume_signal = self.apply_volume_filter(df)
        rsi_buy_signal, rsi_sell_signal = self.apply_rsi_filter(df)
        
        # Get latest data point
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Detect SuperTrend changes (signals)
        signal = None
        if not previous['in_uptrend'] and current['in_uptrend'] and volume_signal.loc[df.index[-1]] and rsi_buy_signal.loc[df.index[-1]]:
            signal = {
                'type': 'buy',
                'symbol': symbol,
                'price': current['close'],
                'supertrend': current['supertrend'],
                'rsi': current['rsi'],
                'timestamp': current.name.isoformat(),
                'volume_ratio': current['volume'] / current['avg_volume'] if 'avg_volume' in current else 1.0
            }
            logger.info(f"BUY Signal: {symbol} @ {current['close']} (SuperTrend: {current['supertrend']:.2f}, RSI: {current['rsi']:.2f})")
        
        elif previous['in_uptrend'] and not current['in_uptrend'] and rsi_sell_signal.loc[df.index[-1]]:
            signal = {
                'type': 'sell',
                'symbol': symbol,
                'price': current['close'],
                'supertrend': current['supertrend'],
                'rsi': current['rsi'],
                'timestamp': current.name.isoformat(),
                'volume_ratio': current['volume'] / current['avg_volume'] if 'avg_volume' in current else 1.0
            }
            logger.info(f"SELL Signal: {symbol} @ {current['close']} (SuperTrend: {current['supertrend']:.2f}, RSI: {current['rsi']:.2f})")
        
        # Update dashboard with signal
        if signal:
            self.update_dashboard({
                'type': 'signal',
                'data': signal
            })
            
        return {
            'symbol': symbol,
            'last_close': current['close'],
            'supertrend': current['supertrend'],
            'in_uptrend': current['in_uptrend'],
            'rsi': current['rsi'],
            'signal': signal,
            'timestamp': current.name.isoformat()
        }
    
    def update_dashboard(self, update):
        """Update dashboard data for UI"""
        if update['type'] == 'signal':
            self.dashboard_data['signals'].append(update['data'])
            # Keep only the last 20 signals
            self.dashboard_data['signals'] = self.dashboard_data['signals'][-20:]
            
        elif update['type'] == 'order':
            # Update positions list
            symbol = update['data']['symbol']
            side = update['data']['side']
            price = update['data']['price']
            amount = update['data']['amount']
            
            # Find existing position or create new one
            position_exists = False
            for i, pos in enumerate(self.dashboard_data['positions']):
                if pos['symbol'] == symbol:
                    position_exists = True
                    if side == 'sell':
                        # Close position
                        entry_price = pos['entry_price']
                        pnl = (price - entry_price) * pos['amount'] if pos['side'] == 'buy' else (entry_price - price) * pos['amount']
                        self.dashboard_data['positions'].pop(i)
                        
                        # Update performance metrics
                        self.dashboard_data['performance']['total_trades'] += 1
                        if pnl > 0:
                            self.dashboard_data['performance']['winning_trades'] += 1
                        else:
                            self.dashboard_data['performance']['losing_trades'] += 1
                        self.dashboard_data['performance']['total_profit'] += pnl
                        
                        # Calculate win rate and profit factor
                        if self.dashboard_data['performance']['total_trades'] > 0:
                            self.dashboard_data['performance']['win_rate'] = (
                                self.dashboard_data['performance']['winning_trades'] / 
                                self.dashboard_data['performance']['total_trades']
                            ) * 100
                    break
            
            if not position_exists and side == 'buy':
                # Add new position
                self.dashboard_data['positions'].append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry_price': price,
                    'current_price': price,
                    'pnl': 0.0,
                    'pnl_percent': 0.0,
                    'timestamp': update['data']['timestamp']
                })
            
            # Update active symbols list
            active_symbols = [pos['symbol'] for pos in self.dashboard_data['positions']]
            self.dashboard_data['active_symbols'] = active_symbols
            
        elif update['type'] == 'price':
            # Update price for a symbol
            symbol = update['data']['symbol']
            price = update['data']['price']
            
            # Update position current price and pnl
            for pos in self.dashboard_data['positions']:
                if pos['symbol'] == symbol:
                    pos['current_price'] = price
                    if pos['side'] == 'buy':
                        pos['pnl'] = (price - pos['entry_price']) * pos['amount']
                        pos['pnl_percent'] = ((price / pos['entry_price']) - 1) * 100
                    else:
                        pos['pnl'] = (pos['entry_price'] - price) * pos['amount']
                        pos['pnl_percent'] = ((pos['entry_price'] / price) - 1) * 100
                    break
        
        elif update['type'] == 'status':
            # Update strategy status
            self.dashboard_data['strategy_status'] = update['data']['status']
        
        elif update['type'] == 'error':
            # Add error to signals list with special type
            error_signal = {
                'type': 'error',
                'message': update['data']['message'],
                'symbol': update['data'].get('symbol', 'system'),
                'timestamp': update['data']['timestamp']
            }
            self.dashboard_data['signals'].append(error_signal)
            # Keep only the last 20 signals
            self.dashboard_data['signals'] = self.dashboard_data['signals'][-20:]
    
    def get_dashboard_data(self):
        """Get current dashboard data for UI"""
        # Add timestamp to the data
        dashboard_copy = self.dashboard_data.copy()
        dashboard_copy['timestamp'] = datetime.now().isoformat()
        return dashboard_copy
    
    async def start_dashboard_server(self, host='0.0.0.0', port=8765):
        """Start a WebSocket server for dashboard updates"""
        async def handler(websocket, path):
            """Handle WebSocket connections"""
            logger.info(f"New dashboard connection established")
            
            # Send initial data
            await websocket.send(json.dumps({
                'type': 'dashboard_update',
                'data': self.get_dashboard_data()
            }))
            
            try:
                # Keep connection alive and send updates
                while True:
                    try:
                        # Handle incoming messages (e.g., for commands)
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(message)
                        
                        # Handle commands
                        if data.get('type') == 'command':
                            cmd = data.get('command')
                            if cmd == 'start':
                                self.update_dashboard({
                                    'type': 'status',
                                    'data': {'status': 'running'}
                                })
                            elif cmd == 'stop':
                                self.update_dashboard({
                                    'type': 'status',
                                    'data': {'status': 'stopped'}
                                })
                            elif cmd == 'refresh':
                                # Force refresh all data
                                pass
                    
                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass
                    
                    # Send regular updates
                    await websocket.send(json.dumps({
                        'type': 'dashboard_update',
                        'data': self.get_dashboard_data()
                    }))
                    
                    await asyncio.sleep(1)
            
            except websockets.exceptions.ConnectionClosed:
                logger.info("Dashboard connection closed")
            except Exception as e:
                logger.error(f"Error in dashboard connection: {e}")
        
        # Start WebSocket server
        logger.info(f"Starting dashboard server on ws://{host}:{port}")
        server = await websockets.serve(handler, host, port)
        
        # Keep server running
        await server.wait_closed()
    
    def print_signal_table(self, results):
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}{'SuperTrend Signal Summary':^80}")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}{'Symbol':<12}{'TF':<6}{'Signal':<10}{'Price':<12}{'Time':<20}{'Volume':<8}")
        print(f"{Fore.CYAN}{'-'*80}")
        for r in results:
            color = Fore.GREEN if r['signal']=='BUY' else Fore.RED if r['signal']=='SELL' else Fore.WHITE
            print(f"{color}{r['symbol']:<12}{r['timeframe']:<6}{r['signal']:<10}{r['price']:<12.2f}{r['time']:<20}{r['volume']:<8.2f}")
        print(f"{Fore.CYAN}{'='*80}\n")

    def print_performance_summary(self):
        perf = self.dashboard_data['performance']
        print(f"{Fore.MAGENTA}{'🏆 Performance Summary 🏆':^80}")
        print(f"{Fore.CYAN}{'━'*80}")
        print(f"{Fore.YELLOW}Total Trades: {perf['total_trades']}  |  Win Rate: {perf['win_rate']:.2f}%  |  Profit Factor: {perf['profit_factor']:.2f}")
        print(f"{Fore.GREEN}Total Profit: {perf['total_profit']:.2f}  |  Wins: {perf['winning_trades']}  |  Losses: {perf['losing_trades']}")
        print(f"{Fore.CYAN}{'━'*80}\n")
        print(f"{Fore.LIGHTMAGENTA_EX}{'✨🚀💰🔥🤑💎👑🌈🎉🎊🎯🥇🥈🥉🏅🏆':^80}{Style.RESET_ALL}\n")

    def detect_signal(self, last, vol_signal, rsi_buy, rsi_sell, method='strict'):
        """Detect signal with different strictness methods"""
        # Strict: All filters must be true
        if method == 'strict':
            buy = last['in_uptrend'] and vol_signal and rsi_buy
            sell = not last['in_uptrend'] and vol_signal and rsi_sell
        # Moderate: At least one filter (SuperTrend + (vol or rsi))
        elif method == 'moderate':
            buy = last['in_uptrend'] and (vol_signal or rsi_buy)
            sell = not last['in_uptrend'] and (vol_signal or rsi_sell)
        # Loose: Only SuperTrend
        elif method == 'loose':
            buy = last['in_uptrend']
            sell = not last['in_uptrend']
        return buy, sell

    def batch_fetch_ohlcv(self, tasks):
        """Fetch OHLCV data for all symbol-timeframe pairs in parallel using multiprocessing.Pool"""
        pool_args = [(symbol, tf, self.config['testnet']) for symbol, tf in tasks]
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count()*2, 32)) as pool:
            results = pool.map(fetch_ohlcv_data, pool_args)
        return dict(results)

    def analyze_symbol_timeframe_fast(self, args_df):
        (symbol, tf), df = args_df
        if df is None or df.empty:
            return None
        df = self.calculate_supertrend(df)
        df = self.calculate_rsi(df)
        vol_signal = self.apply_volume_filter(df)
        rsi_buy, rsi_sell = self.apply_rsi_filter(df)
        last = df.iloc[-1]
        results = {}
        for method in ['strict', 'moderate', 'loose']:
            buy, sell = self.detect_signal(last, vol_signal.loc[df.index[-1]], rsi_buy.loc[df.index[-1]], rsi_sell.loc[df.index[-1]], method)
            if buy:
                signal = 'BUY'
            elif sell:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            filters = {
                'vol': bool(vol_signal.loc[df.index[-1]]),
                'rsi': bool(rsi_buy.loc[df.index[-1]]) if buy else bool(rsi_sell.loc[df.index[-1]]) if sell else False,
                'st': bool(last['in_uptrend']) if buy else not last['in_uptrend'] if sell else False
            }
            results[method] = {
                                'symbol': symbol,
                'timeframe': tf,
                'signal': signal,
                'price': last['close'],
                'time': last.name.strftime('%Y-%m-%d %H:%M'),
                'volume': last['volume'],
                'filters': filters
            }
        return results

    def print_cyberpunk_table(self, results, method):
        table = prettytable.PrettyTable()
        table.field_names = [
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Symbol 🪙",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}TF ⏰",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Signal 🚦",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Price 💲",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Time 🕒",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Volume 🔊",
            f"{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}Filters 🧪"
        ]
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for r in results:
            if not r: continue
            r = r[method]
            if r['signal'] == 'BUY':
                color = Back.GREEN + Fore.BLACK
                emoji = '🟢🚀💰🤑✨🔥'
                signal_counts['BUY'] += 1
            elif r['signal'] == 'SELL':
                color = Back.RED + Fore.WHITE
                emoji = '🔴⬇️⚡💀😱💣'
                signal_counts['SELL'] += 1
            else:
                color = Back.BLACK + Fore.WHITE
                emoji = '⚪️⏸️🤔💤🌫️'
                signal_counts['HOLD'] += 1
            filters = r.get('filters', {})
            fstr = ''
            fstr += f"{'✔️' if filters.get('vol') else '❌'}Vol "
            fstr += f"{'✔️' if filters.get('rsi') else '❌'}RSI "
            fstr += f"{'✔️' if filters.get('st') else '❌'}ST "
            table.add_row([
                f"{color}{r['symbol']} {emoji}{Style.RESET_ALL}",
                f"{color}{r['timeframe']} {emoji}{Style.RESET_ALL}",
                f"{color}{emoji} {r['signal']}{Style.RESET_ALL}",
                f"{color}{r['price']:.2f} {emoji}{Style.RESET_ALL}",
                f"{color}{r['time']} {emoji}{Style.RESET_ALL}",
                f"{color}{r['volume']:.2f} {emoji}{Style.RESET_ALL}",
                f"{Back.MAGENTA}{Fore.WHITE}{fstr}{Style.RESET_ALL}"
            ])
        print(f"\n{Back.MAGENTA}{Fore.CYAN}{Style.BRIGHT}=== 🚀 CYBERPUNK SUPER Z SIGNALS ({method.title()}) 🚀 ==={Style.RESET_ALL}")
        print("""
███████╗██╗   ██╗███████╗██████╗ ███████╗██████╗ ██╗   ██╗███╗   ██╗██╗  ██╗
██╔════╝██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗██║   ██║████╗  ██║██║ ██╔╝
███████╗██║   ██║█████╗  ██████╔╝███████║██████╔╝██║   ██║██╔██╗ ██║█████╔╝ 
╚════██║██║   ██║██╔══╝  ██╔══██╗╚════██║██╔══██╗██║   ██║██║╚██╗██║██╔═██╗ 
███████║╚██████╔╝███████╗██║  ██║███████║██║  ██║╚██████╔╝██║ ╚████║██║  ██╗
╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝
""")
        print(table)
        print(f"{Back.MAGENTA}{Fore.CYAN}Legend: 🟢 Strong Buy | 🟡 Buy | ⚪️ Hold | 🔴 Sell | 💀 Strong Sell | 🔄 Trend Change | 🌀 Pullback | 🚀 Moon | 💣 Dump | 🤑 Profit | 😱 Panic | ✨ Magic | 🔥 Hot | 🤔 Wait | 💤 Sleep | 🌫️ Fog{Style.RESET_ALL}\n")
        print(f"{Style.BRIGHT}{Fore.MAGENTA}Summary: {Fore.GREEN}BUY: {signal_counts['BUY']}  {Fore.RED}SELL: {signal_counts['SELL']}  {Fore.WHITE}HOLD: {signal_counts['HOLD']}\n{Style.RESET_ALL}")

    async def run_strategy(self):
        import itertools
        from tqdm import tqdm
        print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}\n\n███████╗██╗   ██╗███████╗██████╗ ███████╗██████╗ ██╗   ██╗██████╗ ███╗   ██╗██╗  ██╗\n██╔════╝██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗██║   ██║████╗  ██║██║ ██╔╝\n███████╗██║   ██║█████╗  ██████╔╝███████║██████╔╝██║   ██║██╔██╗ ██║█████╔╝ \n╚════██║██║   ██║██╔══╝  ██╔══██╗╚════██║██╔══██╗██║   ██║██║╚██╗██║██╔═██╗ \n███████║╚██████╔╝███████╗██║  ██║███████║██║  ██║╚██████╔╝██║ ╚████║██║  ██╗\n╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝\n{Style.RESET_ALL}")
        methods = ['strict', 'moderate', 'loose']
        timeframes = self.config.get('timeframes', [self.config.get('timeframe', '1m')])
        symbols = self.config['symbols']
        print(f"{Back.CYAN}{Fore.BLACK}{Style.BRIGHT}Scanning {len(symbols)} symbols x {len(timeframes)} timeframes = {len(symbols)*len(timeframes)} tasks...{Style.RESET_ALL}")
        tasks = list(itertools.product(symbols, timeframes))
        # Batch fetch all OHLCV data in parallel
        ohlcv_data = self.batch_fetch_ohlcv(tasks)
        # Analyze in parallel
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count()*2, 32)) as pool:
            all_results = list(tqdm(pool.imap(self.analyze_symbol_timeframe_fast, ohlcv_data.items()), total=len(tasks), desc='Analyzing', ncols=120, colour='magenta'))
        for method in methods:
            self.print_cyberpunk_table(all_results, method)
        self.print_performance_summary()
        print(f"{Fore.CYAN}Dry-run mode: No real trades were placed.\n{Style.RESET_ALL}")

async def main():
    """Main function to run the strategy"""
    try:
        strategy = EnhancedSuperTrend()
        await strategy.run_strategy()
    except Exception:
        logger.exception("Main error")

if __name__ == "__main__":
    asyncio.run(main())
 