import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

class DataFetcher:
    def __init__(self, exchange_id: str = 'bitget', testnet: bool = True):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'testnet': testnet
            }
        })
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv(self, symbol: str, timeframe: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            # Convert timeframe to exchange format
            timeframe_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            exchange_tf = timeframe_map.get(timeframe, timeframe)
            
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                exchange_tf,
                since=int(start_time.timestamp() * 1000) if start_time else None,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end time if provided
            if end_time:
                df = df[df.index <= end_time]
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        return df

    def prepare_data(self, symbol: str, timeframe: str,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """Fetch and prepare data for backtesting"""
        # If no start time provided, default to 30 days ago
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
            
        # If no end time provided, default to now
        if not end_time:
            end_time = datetime.now()
            
        # Fetch OHLCV data
        df = self.fetch_ohlcv(symbol, timeframe, start_time, end_time, limit)
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Clean data (remove NaN values)
        df = df.dropna()
        
        return df

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get market information for position sizing"""
        try:
            market = self.exchange.market(symbol)
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'min_amount': market['limits']['amount']['min'],
                'min_cost': market['limits']['cost']['min'],
                'price_precision': market['precision']['price'],
                'amount_precision': market['precision']['amount'],
                'current_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': (ticker['ask'] - ticker['bid']) / ticker['bid']
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market info: {str(e)}")
            raise 