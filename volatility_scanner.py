import ccxt
import pandas as pd
import numpy as np
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import ta
from statsmodels.tsa.stattools import adfuller
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import json
import requests
from requests.adapters import HTTPAdapter

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Fast calculation functions using pure numpy
def fast_atr(high, low, close, period=14):
    """Fast ATR calculation using pure numpy"""
    n = len(high)
    if n < period:
        return np.nan
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr[-1]

def fast_volatility(returns, period=20):
    """Fast volatility calculation using numpy"""
    if len(returns) < period:
        return np.nan
    return np.std(returns[-period:]) * 100

def fast_rsi(closes, period=14):
    """Fast RSI calculation using numpy"""
    if len(closes) < period + 1:
        return 50.0
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fast_bb_squeeze(closes, period=20, std_dev=2):
    """Fast Bollinger Band squeeze calculation"""
    if len(closes) < period:
        return 0.5
    
    recent_closes = closes[-period:]
    sma = np.mean(recent_closes)
    std = np.std(recent_closes)
    
    bb_width = (2 * std_dev * std) / sma * 100
    
    # Compare to historical average (simplified)
    if len(closes) >= period * 2:
        hist_closes = closes[-period*2:-period]
        hist_sma = np.mean(hist_closes)
        hist_std = np.std(hist_closes)
        hist_bb_width = (2 * std_dev * hist_std) / hist_sma * 100
        
        if hist_bb_width > 0:
            return 1 - min(bb_width / hist_bb_width, 1)
    
    return 0.5

class VolatilityScanner:
    """Ultra-fast advanced scanner that identifies the most volatile and appealing coins on Bitget USDT-M futures"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, testnet: bool = True):
        """Initialize the scanner with Bitget API credentials"""
        # Temporary session before defining max_concurrent_requests
        temp_session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        temp_session.mount('https://', adapter)
        temp_session.mount('http://', adapter)
        # Initialize CCXT exchange with custom HTTP session to increase pool size
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # Bitget requires passphrase
            'enableRateLimit': False,  # Disable to go faster
            'session': temp_session,
            'options': {
                'defaultType': 'swap',  # For USDT-M futures
                'adjustForTimeDifference': True
            }
        })
        
        # Set testnet if applicable
        if testnet and 'test' in self.exchange.urls:
            self.exchange.urls['api'] = self.exchange.urls['test']
            self.exchange.set_sandbox_mode(True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Timeframes to analyze
        self.timeframes = ['5m', '15m', '30m']  # Analyze 5, 15, and 30 minute intervals
        
        # Volatility calculation parameters
        self.atr_period = 14
        self.vol_period = 20
        self.price_change_periods = [1, 3, 7, 14]  # days
        
        # Additional technical parameters
        self.rsi_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.volume_sma_period = 20
        self.mean_reversion_period = 30
        
        # Ultra-fast rate limiting - minimal delays
        self.last_api_call = 0
        self.min_api_interval = 0.2  # 200ms between API calls (safer)
        self.max_concurrent_requests = 8  # Lower concurrency to avoid rate limits
        
        # Enhanced caching with persistence
        self.data_cache = {}
        self.cache_duration = 180  # 3 minutes cache
        
        # Batch processing settings
        self.batch_size = 10
        
        # Connection pool for aiohttp
        self.session = None
        
        # Override CCXT HTTP session with increased pool size to avoid 'pool is full' warnings
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=self.max_concurrent_requests, pool_maxsize=self.max_concurrent_requests)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        # Assign to CCXT exchange
        self.exchange.session = session
        
        # Pre-computed constants for faster calculations
        self.volatility_regimes = {
            'low': (0, 3),        # 0-3% daily volatility
            'medium': (3, 7),     # 3-7% daily volatility
            'high': (7, 15),      # 7-15% daily volatility
            'extreme': (15, 100)  # 15%+ daily volatility
        }
          # Market cycle detection
        self.market_cycle_periods = 90  # days to analyze for market cycle
    
    async def rate_limit(self):
        """Apply minimal rate limiting to API calls"""
        now = time.time()
        elapsed = now - self.last_api_call
        if elapsed < self.min_api_interval:
            await asyncio.sleep(self.min_api_interval - elapsed)
        self.last_api_call = time.time()

    async def fetch_all_market_data_batch(self, symbols: List[str], timeframe: str = '4h', lookback_days: int = 7) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel with aggressive batching"""
        tasks = []
        results = {}
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def fetch_single(symbol):
            async with semaphore:
                try:
                    data = await self.fetch_market_data_fast(symbol, timeframe, lookback_days)
                    if data is not None:
                        return symbol, data
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    return symbol, None
        
        # Create all tasks
        tasks = [fetch_single(symbol) for symbol in symbols]
        
        # Execute all tasks in parallel
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in completed_tasks:
            if isinstance(result, tuple) and result[1] is not None:
                symbol, data = result
                results[symbol] = data
        
        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    async def fetch_market_data_fast(self, symbol: str, timeframe: str = '4h', lookback_days: int = 7) -> pd.DataFrame:
        """Ultra-fast market data fetching with minimal processing"""
        try:
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                if (datetime.now() - cache_timestamp).total_seconds() < self.cache_duration:
                    return df
            await self.rate_limit()
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    ohlcv = await asyncio.to_thread(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        timeframe,
                        since=since,
                        limit=500
                    )
                    break
                except ccxt.RateLimitExceeded:
                    logger.warning(f"[429] Rate limit exceeded for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if '429' in str(e):
                        logger.warning(f"[429] Too Many Requests for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
            else:
                logger.error(f"Failed to fetch {symbol} {timeframe} after {max_retries} retries due to rate limits.")
                return None
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df['returns'] = df['close'].pct_change()
            df.attrs['timeframe'] = timeframe
            self.data_cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            logger.error(f"Error fetching fast market data for {symbol}: {str(e)}")
            return None

    async def get_all_usdt_m_pairs_with_leverage(self) -> List[Tuple[str, float]]:
        """Fetch all available USDT-M pairs and their max leverage from Bitget"""
        try:
            await self.rate_limit()
            logger.info("Fetching markets from Bitget...")
            markets = await asyncio.to_thread(self.exchange.fetch_markets)
            pairs_with_leverage = []
            for market in markets:
                symbol = market['symbol']
                if not market.get('active', False):
                    continue
                if market.get('type') not in ['swap', 'future']:
                    continue
                is_usdt_pair = (
                    (':USDT' in symbol) or 
                    (symbol.endswith('USDT')) or 
                    (symbol.endswith('/USDT')) or
                    (market.get('quote') == 'USDT' and market.get('settle') == 'USDT')
                )
                if is_usdt_pair:
                    max_leverage = market.get('limits', {}).get('leverage', {}).get('max', 1)
                    pairs_with_leverage.append((symbol, max_leverage))
            logger.info(f"Found {len(pairs_with_leverage)} active USDT-M pairs with leverage on Bitget")
            return pairs_with_leverage
        except Exception as e:
            logger.error(f"Error fetching USDT-M pairs with leverage: {str(e)}")
            return []

    async def fetch_market_data(self, symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch market data for a specific symbol and timeframe"""
        try:
            cache_key = f"{symbol}_{timeframe}_{lookback_days}"
            if cache_key in self.data_cache:
                cache_timestamp, df = self.data_cache[cache_key]
                if (datetime.now() - cache_timestamp).total_seconds() < 300:
                    return df
            await self.rate_limit()
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    ohlcv = await asyncio.to_thread(
                        self.exchange.fetch_ohlcv,
                        symbol,
                        timeframe,
                        since=since,
                        limit=1000
                    )
                    break
                except ccxt.RateLimitExceeded:
                    logger.warning(f"[429] Rate limit exceeded for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if '429' in str(e):
                        logger.warning(f"[429] Too Many Requests for {symbol} {timeframe}, retrying (attempt {attempt+1})...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
                    return None
            else:
                logger.error(f"Failed to fetch {symbol} {timeframe} after {max_retries} retries due to rate limits.")
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            if len(df) < 30:
                logger.warning(f"Not enough data for {symbol} on {timeframe}, only {len(df)} candles")
                return None
            self.data_cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
            return None

    def calculate_volatility_metrics_fast(self, df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Ultra-fast volatility metrics calculation using optimized functions"""
        if df is None or len(df) < self.vol_period:
            return None
        
        try:
            # Convert to numpy arrays for speed
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values
            
            current_price = closes[-1]
            
            # Fast ATR calculation using numba
            atr = fast_atr(highs, lows, closes, self.atr_period)
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            # Fast volatility using numba
            returns = np.diff(closes) / closes[:-1]  # Calculate returns directly
            daily_volatility = fast_volatility(returns, self.vol_period)
            
            # Fast RSI using numba
            rsi = fast_rsi(closes, self.rsi_period)
            
            # Fast BB squeeze using numba
            bb_squeeze = fast_bb_squeeze(closes, self.bollinger_period, self.bollinger_std)
            
            # Quick volume metrics
            volume = volumes[-1]
            avg_volume = np.mean(volumes[-self.volume_sma_period:]) if len(volumes) >= self.volume_sma_period else volume
            vol_change_pct = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Quick price changes (simplified for speed)
            price_changes = {}
            tf_multiplier = {'5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1}
            tf = df.attrs.get('timeframe', '4h')
            
            for days in [1, 3, 7]:  # Reduced for speed
                periods = days * tf_multiplier.get(tf, 6)
                if len(closes) > periods:
                    past_price = closes[-periods-1]
                    change_pct = ((current_price - past_price) / past_price) * 100
                    price_changes[f"{days}d_change"] = change_pct
            
            # Quick Bollinger Band metrics
            if len(closes) >= self.bollinger_period:
                bb_middle = np.mean(closes[-self.bollinger_period:])
                bb_std = np.std(closes[-self.bollinger_period:])
                bb_upper = bb_middle + self.bollinger_std * bb_std
                bb_lower = bb_middle - self.bollinger_std * bb_std
                bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            else:
                bb_width = 0
                bb_position = 0.5
            
            # Quick volume volatility
            if len(volumes) >= self.vol_period:
                volume_returns = np.diff(volumes) / volumes[:-1]
                volume_volatility = np.std(volume_returns[-self.vol_period:]) * 100
            else:
                volume_volatility = 0
            
            # Quick mean reversion score
            if len(closes) >= self.mean_reversion_period:
                price_sma = np.mean(closes[-self.mean_reversion_period:])
                price_std = np.std(closes[-self.mean_reversion_period:])
                z_score = (current_price - price_sma) / price_std if price_std > 0 else 0
                mean_reversion_score = min(abs(z_score) / 3, 1) if z_score != 0 else 0
                mean_reversion_direction = -1 if z_score > 0 else 1 if z_score < 0 else 0
            else:
                z_score = 0
                mean_reversion_score = 0
                mean_reversion_direction = 0
            
            # Latest candle range
            latest_candle_range_pct = ((highs[-1] - lows[-1]) / lows[-1]) * 100 if lows[-1] > 0 else 0
            
            # Volatility regime (simplified)
            volatility_regime = "medium"
            if daily_volatility < 3:
                volatility_regime = "low"
            elif daily_volatility > 15:
                volatility_regime = "extreme"
            elif daily_volatility > 7:
                volatility_regime = "high"
            
            # Volume trend (simplified)
            if len(volumes) >= 10:
                recent_vol_avg = np.mean(volumes[-5:])
                prev_vol_avg = np.mean(volumes[-10:-5])
                volume_trend = ((recent_vol_avg - prev_vol_avg) / prev_vol_avg) * 100 if prev_vol_avg > 0 else 0
            else:
                volume_trend = 0
            
            # OBV calculation
            if len(closes) > 1:
                obv = 0
                obv_arr = np.zeros_like(closes)
                for i in range(1, len(closes)):
                    if closes[i] > closes[i-1]:
                        obv += volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv -= volumes[i]
                    obv_arr[i] = obv
                obv_val = obv_arr[-1]
            else:
                obv_val = 0

            # Momentum (10-period ROC)
            momentum_period = 10
            if len(closes) > momentum_period:
                momentum = (closes[-1] - closes[-momentum_period-1]) / closes[-momentum_period-1]
            else:
                momentum = 0
            
            # Simplified metrics for speed
            metrics = {
                'symbol': symbol,
                'last_price': current_price,
                
                # Core volatility metrics
                'atr': atr,
                'atr_pct': atr_pct,
                'daily_volatility': daily_volatility,
                'volatility_regime': volatility_regime,
                
                # Price action metrics
                'latest_range_pct': latest_candle_range_pct,
                'bb_width': bb_width,
                'bb_position': bb_position,
                'bb_squeeze': bb_squeeze,
                
                # Technical indicators
                'rsi': rsi,
                'obv': obv_val,
                'momentum': momentum,
                
                # Volume metrics
                'volume': volume,
                'avg_volume': avg_volume,
                'volume_change_pct': vol_change_pct,
                'relative_volume': relative_volume,
                'volume_volatility': volume_volatility,
                'volume_trend': volume_trend,
                
                # Mean reversion metrics
                'z_score': z_score,
                'mean_reversion_score': mean_reversion_score,
                'mean_reversion_direction': mean_reversion_direction,
                  # Add price changes
                **price_changes,
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {str(e)}")
            return None

    def calculate_appeal_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an enhanced overall appeal score based on multiple metrics"""
        if not metrics:
            return 0
        
        # Expanded weights for different metrics
        weights = {
            # Volatility metrics (50% of score)
            'atr_pct': 0.10,                 # Higher ATR % means more volatility
            'daily_volatility': 0.10,         # Standard deviation of daily returns
            'bb_squeeze': 0.10,               # Bollinger Band squeeze (higher = more compressed and ready to expand)
            'latest_range_pct': 0.05,         # Latest candle range as % of price
            'volatility_premium': 0.05,       # Short-term vs long-term volatility premium
            'bb_width': 0.05,                 # Bollinger Band width as % of price
            'volume_volatility': 0.05,        # Volatility of volume
            
            # Momentum metrics (25% of score)
            'mean_reversion_score': 0.10,     # Mean reversion potential (higher = more likely to revert)
            'volume_change_pct': 0.05,        # Recent volume change
            'relative_volume': 0.05,          # Current volume relative to average
            'volume_trend': 0.05,             # Recent volume trend
            
            # Trend metrics (15% of score)
            'trend_strength': 0.05,           # Strength of current trend
            '1d_change': 0.025,               # Short-term price change
            '3d_change': 0.025,               # Medium-term price change
            
            # Technical metrics (10% of score)
            'rsi': 0.10,                      # RSI (adjusted for mean reversion or trend following)
        }
        
        # Normalize price changes
        for days in metrics.get('price_change_periods', [1, 3, 7, 14]):
            key = f"{days}d_change"
            if key in metrics and key in weights:
                # Convert to a 0-1 score (more positive is better, consider +/- 20% as the range)
                metrics[f"{key}_score"] = (metrics[key] + 20) / 40
                metrics[f"{key}_score"] = max(0, min(1, metrics[f"{key}_score"]))
        
        # Calculate RSI score, optimizing for either trend following or mean reversion
        # For trend following: higher RSI = better for uptrends, lower RSI = better for downtrends
        # For mean reversion: extreme RSI (high or low) = better opportunity
        rsi = metrics.get('rsi', 50)
        
        # Determine if we should score for mean reversion or trend following based on ADX
        trend_strength = metrics.get('trend_strength', 0)
        
        if trend_strength > 0.3:  # Strong trend, use trend following scoring
            # If trend is up (positive price change), higher RSI is better
            # If trend is down (negative price change), lower RSI is better
            recent_change = metrics.get('1d_change', 0)
            if recent_change > 0:  # Uptrend
                metrics['rsi_score'] = rsi / 100  # Higher RSI is better
            else:  # Downtrend
                metrics['rsi_score'] = (100 - rsi) / 100  # Lower RSI is better
        else:  # Weak trend, use mean reversion scoring
            # Extreme RSI values are better for mean reversion
            metrics['rsi_score'] = abs(rsi - 50) / 50  # 0 = neutral, 1 = extreme
        
        # Calculate appeal score
        score = 0
        used_weights_sum = 0
        
        for metric, weight in weights.items():
            if metric == 'rsi' and 'rsi_score' in metrics:
                score += metrics['rsi_score'] * weight
                used_weights_sum += weight
            elif metric.endswith('_change') and f"{metric}_score" in metrics:
                score += metrics[f"{metric}_score"] * weight
                used_weights_sum += weight
            elif metric in metrics and not pd.isna(metrics[metric]):
                # Normalize metrics to 0-1 scale based on typical ranges
                norm_ranges = {
                    'atr_pct': (0, 10),               # 0-10% ATR
                    'daily_volatility': (0, 15),       # 0-15% daily volatility
                    'bb_width': (0, 30),               # 0-30% BB width
                    'bb_squeeze': (0, 1),              # Already 0-1
                    'latest_range_pct': (0, 15),       # 0-15% candle range
                    'volatility_premium': (-5, 5),     # -5% to +5% premium
                    'mean_reversion_score': (0, 1),    # Already 0-1
                    'volume_change_pct': (-50, 200),   # -50% to +200% volume change
                    'relative_volume': (0, 3),         # 0-3x relative volume
                    'volume_trend': (-50, 100),        # -50% to +100% trend
                    'trend_strength': (0, 1),          # Already 0-1
                    'volume_volatility': (0, 100),     # 0-100% volume volatility
                }
                
                if metric in norm_ranges:
                    min_val, max_val = norm_ranges[metric]
                    value = metrics[metric]
                    
                    # Standard normalization
                    norm_value = (value - min_val) / (max_val - min_val)
                    
                    # Clip to 0-1 range
                    norm_value = max(0, min(1, norm_value))
                    score += norm_value * weight
                    used_weights_sum += weight
        
        # Normalize score based on weights actually used
        if used_weights_sum > 0:
            score = score / used_weights_sum
          # Add bonus for extreme opportunities (very high volatility + significant mean reversion potential)
        if metrics.get('daily_volatility', 0) > 10 and metrics.get('mean_reversion_score', 0) > 0.8:
            score *= 1.2  # 20% bonus
        
        # Cap at 1.0
        return min(score, 1.0)

    async def scan_all_markets(self, timeframe: str = '4h', top_n: int = 20, min_volume: float = 100000) -> List[Dict[str, Any]]:
        """Scan all markets using ultra-fast method by default"""
        return await self.scan_all_markets_ultra_fast(timeframe, top_n, min_volume)

    async def scan_all_markets_ultra_fast(self, timeframe: str = '4h', top_n: int = 20, min_volume: float = 100000) -> List[Dict[str, Any]]:
        """Ultra-fast market scanning with parallel processing and optimized calculations"""
        # Get all USDT-M pairs with leverage
        pairs_with_leverage = await self.get_all_usdt_m_pairs_with_leverage()
        # Filter for leverage >= 25x
        pairs = [s for s, lev in pairs_with_leverage if lev >= 25]
        leverage_map = {s: lev for s, lev in pairs_with_leverage}
        if not pairs:
            logger.error("No pairs found to scan after leverage filter! Returning empty result.")
            return []
        logger.info(f"Starting ultra-fast volatility scan for {len(pairs)} pairs on {timeframe} timeframe")
        # Batch fetch all market data in parallel
        start_time = time.time()
        all_data = await self.fetch_all_market_data_batch(pairs, timeframe, lookback_days=7)
        fetch_time = time.time() - start_time
        logger.info(f"Fetched data for {len(all_data)} pairs in {fetch_time:.2f} seconds")
        results = []
        def calculate_metrics_wrapper(args):
            symbol, df = args
            if df is not None:
                metrics = self.calculate_volatility_metrics_fast(df, symbol)
                if metrics:
                    metrics['leverage'] = leverage_map.get(symbol, 1)
                return metrics
            return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            start_calc_time = time.time()
            metric_results = list(executor.map(calculate_metrics_wrapper, all_data.items()))
            calc_time = time.time() - start_calc_time
            logger.info(f"Calculated metrics for {len(metric_results)} pairs in {calc_time:.2f} seconds")
        for metrics in metric_results:
            if metrics and metrics.get('avg_volume', 0) >= min_volume and metrics.get('leverage', 1) >= 25:
                # Add full enhanced score and tier
                ranker = EnhancedMarketRanker()
                perf = {
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'profit_factor': metrics.get('profit_factor', 1),
                    'win_rate': metrics.get('win_rate', 0.5),
                    'max_drawdown': metrics.get('max_drawdown', 0.2),
                    'consistency': metrics.get('consistency', 0.5),
                }
                mtf_signals = metrics.get('mtf_signals', None)
                scored = ranker.score_market(metrics, mtf_signals, perf)
                metrics['score'] = scored['score']
                metrics['tier'] = scored['tier']
                metrics['mtf_trend'] = scored['mtf_trend']
                results.append(metrics)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        top_results = results[:top_n]
        total_time = time.time() - start_time
        logger.info(f"Completed ultra-fast scan in {total_time:.2f}s, found {len(results)} valid pairs, returning top {len(top_results)}")
        return top_results

    def calculate_appeal_score_fast(self, metrics: Dict[str, float]) -> float:
        """Ultra-fast appeal score calculation with simplified weights"""
        if not metrics:
            return 0
        
        # Simplified scoring for speed
        score = 0
        
        # Volatility components (60%)
        daily_vol = min(metrics.get('daily_volatility', 0) / 10, 1) * 0.20
        atr_pct = min(metrics.get('atr_pct', 0) / 5, 1) * 0.15
        bb_squeeze = metrics.get('bb_squeeze', 0) * 0.15
        latest_range = min(metrics.get('latest_range_pct', 0) / 8, 1) * 0.10
        
        # Volume components (25%)
        vol_change = min(abs(metrics.get('volume_change_pct', 0)) / 100, 1) * 0.10
        rel_volume = min(metrics.get('relative_volume', 1) / 3, 1) * 0.15
        
        # Mean reversion (15%)
        mean_rev = metrics.get('mean_reversion_score', 0) * 0.15
        
        score = daily_vol + atr_pct + bb_squeeze + latest_range + vol_change + rel_volume + mean_rev
        
        return min(score, 1.0)

    async def get_volatility_rankings(self, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
        """Get volatility rankings for multiple timeframes"""
        results = {}
        
        for timeframe in self.timeframes:
            results[timeframe] = await self.scan_all_markets(timeframe, top_n, min_volume)
            
        return results
        
    def predict_future_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Use historical volatility patterns to predict future volatility
        Note: This is a simplified prediction and not a full GARCH model
        """
        try:
            # Extract returns
            returns = df['returns'].dropna()
            
            if len(returns) < 30:
                return {}
                
            # Calculate recent volatility (last 10 periods)
            recent_vol = returns[-10:].std() * 100
            
            # Calculate volatility of volatility (how stable is volatility?)
            # Higher values mean volatility itself is volatile
            rolling_vol = returns.rolling(window=5).std() * 100
            vol_of_vol = rolling_vol.std()
            
            # Estimate volatility persistence (autocorrelation of squared returns)
            # Higher values mean volatility tends to cluster
            squared_returns = returns**2
            vol_persistence = squared_returns.autocorr(lag=1)
            
            # Simple volatility forecast based on recent trend
            vol_5d = returns[-5:].std() * 100
            vol_10d = returns[-10:].std() * 100
            vol_20d = returns[-20:].std() * 100
            
            # If recent volatility is increasing, predict it continues
            vol_trend = (vol_5d - vol_10d) / vol_10d if vol_10d > 0 else 0
            
            # Predicted change in volatility
            pred_vol_change = vol_trend * (1 + abs(vol_persistence))
            
            # Predicted volatility
            predicted_vol = recent_vol * (1 + pred_vol_change)
            
            # Confidence score (0-1, higher = more confident)
            # Based on stability of volatility and sample size
            prediction_confidence = 1 - min(vol_of_vol / recent_vol, 1) if recent_vol > 0 else 0
            
            return {
                'recent_volatility': recent_vol,
                'volatility_of_volatility': vol_of_vol,
                'volatility_persistence': vol_persistence,
                'volatility_trend': vol_trend,
                'predicted_volatility': predicted_vol,
                'prediction_confidence': prediction_confidence,
            }
        except Exception as e:
            logger.error(f"Error predicting volatility: {str(e)}")
            return {}

class EnhancedMarketRanker:
    """
    Enhanced market ranking system with ultra-modern visual experience
    """
    # ANSI color codes
    COLOR_RESET = '\033[0m'
    COLOR_BOLD = '\033[1m'
    COLOR_UNDERLINE = '\033[4m'
    COLOR_BLINK = '\033[5m'

    # Enhanced visual elements
    TIERS = [
        (90, "GODLIKE UNICORN BUSSY TIER 🌈✨💫"),
        (80, "ULTRA PREMIUM BUSSY TIER 💎🔥⚡"),
        (70, "PREMIUM TIER ✨🌟💫"),
        (60, "HIGH POTENTIAL TIER 🔥🚀💪"),
        (50, "STRONG PERFORMER TIER ⚡💯🎯"),
        (40, "SOLID MARKET TIER 🥇💫✨"),
        (30, "AVERAGE MARKET TIER 🥈📊📈"),
        (20, "BASIC MARKET TIER 🥉📉📊"),
        (10, "LOW QUALITY TIER 🪫⚠️📉"),
        (0,  "PURE SHIT TIER 💩🚫❌")
    ]

    # Enhanced metric emojis with dynamic states
    METRIC_EMOJIS = {
        'volume': ['💰', '💎', '💵', '💸', '🤑'],
        'volatility': ['🌋', '⚡', '🔥', '💥', '🌪️'],
        'leverage': ['🔥', '💪', '⚡', '🚀', '💫'],
        'tick_size': ['🎯', '🎨', '🎪', '🎭', '🎪'],
        'liquidity': ['🌊', '💧', '🌊', '💦', '🌊'],
        'obv': ['📊', '📈', '📉', '📊', '📈'],
        'momentum': ['🚀', '💫', '⚡', '🔥', '💪'],
        'rsi': ['📈', '📊', '📉', '📈', '📊'],
        'supertrend': ['🔮', '✨', '🌟', '💫', '⭐'],
        'sharpe': ['📏', '📐', '📊', '📈', '📉'],
        'profit_factor': ['💹', '📈', '💯', '🎯', '✨'],
        'win_loss': ['🏆', '🎯', '💪', '🔥', '⚡'],
        'drawdown': ['📉', '⚠️', '🚫', '❌', '💩'],
        'consistency': ['🔄', '⚡', '💫', '✨']
    }

    # Enhanced ANSI colors with gradients
    METRIC_COLORS = {
        'volume': '\033[38;2;255;215;0m',      # Gold gradient
        'volatility': '\033[38;2;255;0;255m',  # Magenta gradient
        'leverage': '\033[38;2;255;0;0m',      # Red gradient
        'tick_size': '\033[38;2;0;191;255m',   # Deep sky blue
        'liquidity': '\033[38;2;0;255;255m',   # Cyan gradient
        'obv': '\033[38;2;0;255;0m',           # Green gradient
        'momentum': '\033[38;2;255;69;0m',     # Orange-red gradient
        'rsi': '\033[38;2;30;144;255m',        # Dodger blue
        'supertrend': '\033[38;2;147;112;219m', # Medium purple
        'sharpe': '\033[38;2;50;205;50m',      # Lime green
        'profit_factor': '\033[38;2;255;215;0m', # Gold
        'win_loss': '\033[38;2;0;255;127m',    # Spring green
        'drawdown': '\033[38;2;255;0;0m',      # Red
        'consistency': '\033[38;2;0;191;255m'  # Deep sky blue
    }

    # Animation frames for loading states
    LOADING_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, mtf_timeframes=None):
        self.mtf_timeframes = mtf_timeframes or ["15m", "1h", "4h", "1d"]
        self.st_period = 50
        self.st_multiplier = 1.0
        self.animation_index = 0
        self.last_update = time.time()

    def get_animated_emoji(self, metric: str, value: float) -> str:
        emojis = self.METRIC_EMOJIS.get(metric, ['❓'])
        # Normalize value to 0-1 for emoji index
        norm = min(max(float(value) / 20, 0), 1)
        index = min(int(norm * (len(emojis)-1)), len(emojis) - 1)
        return emojis[index]

    def get_gradient_color(self, metric: str, value: float) -> str:
        base_color = self.METRIC_COLORS.get(metric, '\033[38;2;255;255;255m')
        # Optionally, could modulate color intensity by value
        return base_color

    def create_progress_bar(self, value: float, width: int = 20) -> str:
        value = min(max(value, 0), 1)
        filled = int(width * value)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {value*100:.1f}%"

    def create_metric_line(self, metric: str, value: float, score: float) -> str:
        emoji = self.get_animated_emoji(metric, score)
        color = self.get_gradient_color(metric, score)
        progress = self.create_progress_bar(score/20)  # Normalize to 0-1 range
        return f"{color}{emoji} {metric.title():<12} {progress}{self.COLOR_RESET}"

    def create_tier_banner(self, tier: str, score: float) -> str:
        banner = f"\n{self.COLOR_BOLD}{'='*60}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{' ' * 15}{tier}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{' ' * 20}Score: {score:.2f}{self.COLOR_RESET}\n"
        banner += f"{self.COLOR_BOLD}{'='*60}{self.COLOR_RESET}\n"
        return banner

    def create_visual_breakdown(self, metrics: Dict, breakdown: Dict) -> str:
        visual_lines = []
        for metric, score in breakdown.items():
            value = metrics.get(metric, score)
            try:
                visual_lines.append(self.create_metric_line(metric, value, score))
            except Exception:
                visual_lines.append(f"{metric}: {score:.1f}")
        return "\n".join(visual_lines)

    def score_market(self, metrics, mtf_signals=None, perf=None):
        # Define max points for each metric
        metric_points = {
            'volume': 20,
            'volatility': 15,
            'leverage': 10,
            'tick_size': 10,
            'liquidity': 10,
            'obv': 7,
            'momentum': 10,
            'rsi': 10,
            'supertrend': 15,
            'sharpe': 10,
            'profit_factor': 10,
            'win_loss': 10,
            'drawdown': 10,
            'consistency': 10,
        }
        breakdown = {}
        visual_breakdown = {}
        # Volume (log scale, max 20)
        volume = metrics.get('volume', 0)
        volume_score = min(metric_points['volume'], 5 * np.log10(max(volume, 1)))
        breakdown['volume'] = volume_score
        visual_breakdown['volume'] = self.create_metric_line('volume', volume, volume_score)
        # Volatility (max 15)
        vol_score = min(metric_points['volatility'], metrics.get('daily_volatility', 0) / 2)
        breakdown['volatility'] = vol_score
        visual_breakdown['volatility'] = self.create_metric_line('volatility', metrics.get('daily_volatility', 0), vol_score)
        # Leverage (max 10, assume 50x+ is 10)
        leverage = metrics.get('leverage', 50)
        lev_score = metric_points['leverage'] if leverage >= 100 else 8 if leverage >= 50 else 5
        breakdown['leverage'] = lev_score
        visual_breakdown['leverage'] = self.create_metric_line('leverage', leverage, lev_score)
        # Tick Size (max 10, smaller is better)
        tick = metrics.get('tick_size', 0.01)
        tick_score = metric_points['tick_size'] if tick < 0.001 else 8 if tick < 0.01 else 5
        breakdown['tick_size'] = tick_score
        visual_breakdown['tick_size'] = self.create_metric_line('tick_size', tick, tick_score)
        # Liquidity (max 10, based on volume)
        liq_score = min(metric_points['liquidity'], volume_score)
        breakdown['liquidity'] = liq_score
        visual_breakdown['liquidity'] = self.create_metric_line('liquidity', volume, liq_score)
        # OBV (max 7, normalized)
        obv = metrics.get('obv', 0)
        obv_score = min(metric_points['obv'], abs(obv) / 1e7)
        breakdown['obv'] = obv_score
        visual_breakdown['obv'] = self.create_metric_line('obv', obv, obv_score)
        # Momentum (max 10, normalized ROC)
        momentum = metrics.get('momentum', 0)
        mom_score = min(metric_points['momentum'], abs(momentum) * 10)
        breakdown['momentum'] = mom_score
        visual_breakdown['momentum'] = self.create_metric_line('momentum', momentum, mom_score)
        # RSI (max 10, extreme values get higher score)
        rsi = metrics.get('rsi', 50)
        rsi_score = metric_points['rsi'] if rsi < 40 or rsi > 60 else 5
        breakdown['rsi'] = rsi_score
        visual_breakdown['rsi'] = self.create_metric_line('rsi', rsi, rsi_score)
        # SuperTrend (max 15, strong trend = high score)
        st_metrics = metrics.get('supertrend_metrics', {})
        st_score = 0
        trend_score = st_metrics.get('supertrend', 0)
        st_score += 5 if trend_score != 0 else 0
        trend_strength = st_metrics.get('trend_strength', 0)
        st_score += min(5, trend_strength * 10)
        trend_consistency = st_metrics.get('trend_consistency', 0)
        st_score += min(5, trend_consistency * 10)
        st_score = min(metric_points['supertrend'], st_score)
        breakdown['supertrend'] = st_score
        visual_breakdown['supertrend'] = self.create_metric_line('supertrend', st_score, st_score)
        # Sharpe Ratio (max 10)
        sharpe = perf.get('sharpe_ratio', 0) if perf else 0
        sharpe_score = min(metric_points['sharpe'], max(0, sharpe * 2))
        breakdown['sharpe'] = sharpe_score
        visual_breakdown['sharpe'] = self.create_metric_line('sharpe', sharpe, sharpe_score)
        # Profit Factor (max 10)
        pf = perf.get('profit_factor', 1) if perf else 1
        pf_score = min(metric_points['profit_factor'], max(0, (pf-1)*5))
        breakdown['profit_factor'] = pf_score
        visual_breakdown['profit_factor'] = self.create_metric_line('profit_factor', pf, pf_score)
        # Win/Loss (max 10)
        win_rate = perf.get('win_rate', 0.5) if perf else 0.5
        wl_score = min(metric_points['win_loss'], win_rate * 10)
        breakdown['win_loss'] = wl_score
        visual_breakdown['win_loss'] = self.create_metric_line('win_loss', win_rate, wl_score)
        # Drawdown (max 10, lower is better)
        dd = perf.get('max_drawdown', 0.2) if perf else 0.2
        dd_score = metric_points['drawdown'] if dd < 0.1 else 7 if dd < 0.2 else 4
        breakdown['drawdown'] = dd_score
        visual_breakdown['drawdown'] = self.create_metric_line('drawdown', dd, dd_score)
        # Consistency (max 10, higher = more stable)
        consistency = perf.get('consistency', 0.5) if perf else 0.5
        cons_score = min(metric_points['consistency'], consistency * 10)
        breakdown['consistency'] = cons_score
        visual_breakdown['consistency'] = self.create_metric_line('consistency', consistency, cons_score)
        # Sum all metric points for raw score (max 100)
        raw_score = sum(breakdown.get(k, 0) for k in metric_points)
        # 2. MTF Trend Bonus/Penalty
        mtf_trend = 'neutral'
        mtf_bonus = 0
        mtf_icon = ''
        if mtf_signals:
            bullish = sum(1 for s in mtf_signals if s == 'bullish')
            bearish = sum(1 for s in mtf_signals if s == 'bearish')
            if bullish > len(mtf_signals)//2:
                mtf_trend = 'bullish'
                mtf_bonus = raw_score * 0.5
                mtf_icon = '📈'
            elif bearish > len(mtf_signals)//2:
                mtf_trend = 'bearish'
                mtf_bonus = -raw_score * 0.5
                mtf_icon = '📉'
        final_score = raw_score + mtf_bonus
        final_score = max(0, min(100, final_score))
        # 3. Assign tier
        tier = next(t for s, t in self.TIERS if final_score >= s)
        # 4. Visual summary string
        summary = f"{self.create_tier_banner(tier, final_score)}\n{self.create_visual_breakdown(metrics, breakdown)}\n{mtf_icon}"
        # 5. Return full breakdown
        return {
            'score': round(final_score, 2),
            'tier': tier,
            'mtf_trend': mtf_trend,
            'metrics_breakdown': breakdown,
            'visual_breakdown': visual_breakdown,
            'summary': summary,
            'animated_breakdown': self.create_animated_breakdown(breakdown)
        }

    def create_animated_breakdown(self, breakdown: Dict[str, float]) -> str:
        self.animation_index = (self.animation_index + 1) % len(self.LOADING_FRAMES)
        loading = self.LOADING_FRAMES[self.animation_index]
        animated = []
        for metric, value in breakdown.items():
            emoji = self.get_animated_emoji(metric, value)
            color = self.get_gradient_color(metric, value)
            animated.append(f"{loading} {color}{emoji} {metric}: {value:.1f}{self.COLOR_RESET}")
        return "\n".join(animated)

    def print_market_analysis(self, result: Dict):
        print("\033[2J\033[H")  # Clear screen
        print(result['summary'])
        print("\nReal-time Analysis:")
        print(result['animated_breakdown'])
        if 'perf' in result:
            print(f"\n{self.COLOR_BOLD}Performance Metrics:{self.COLOR_RESET}")
            for metric, value in result['perf'].items():
                print(f"  {metric}: {value:.2f}")

# Function to run the scanner
async def scan_for_volatile_coins(api_key: str = None, api_secret: str = None, 
                                testnet: bool = True, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
    """Run the enhanced volatility scanner and return results"""
    scanner = VolatilityScanner(api_key, api_secret, testnet)
    return await scanner.get_volatility_rankings(top_n, min_volume)

# Synchronous wrapper for the scanner
def run_volatility_scan(api_key: str = None, api_secret: str = None, 
                      testnet: bool = True, top_n: int = 20, min_volume: float = 1000000) -> Dict[str, List[Dict[str, Any]]]:
    """Synchronous wrapper to run the enhanced volatility scanner"""
    return asyncio.run(scan_for_volatile_coins(api_key, api_secret, testnet, top_n, min_volume))

if __name__ == "__main__":
    # Run independently for testing
    results = run_volatility_scan(top_n=10, testnet=False)
    ranker = EnhancedMarketRanker()
    # Print results with full visual scoring
    for timeframe, coins in results.items():
        print(f"\nTop volatile coins for {timeframe} timeframe:")
        for i, coin in enumerate(coins, 1):
            # Use the ranker to get the full visual summary
            perf = {
                'sharpe_ratio': coin.get('sharpe_ratio', 0),
                'profit_factor': coin.get('profit_factor', 1),
                'win_rate': coin.get('win_rate', 0.5),
                'max_drawdown': coin.get('max_drawdown', 0.2),
                'consistency': coin.get('consistency', 0.5),
            }
            mtf_signals = coin.get('mtf_signals', None)
            result = ranker.score_market(coin, mtf_signals, perf)
            ranker.print_market_analysis(result)