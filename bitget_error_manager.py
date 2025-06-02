#!/usr/bin/env python3
"""
Bitget Error Manager - Comprehensive error handling for Bitget API

Features:
- Complete Bitget error code database with solutions
- Structured error categorization
- Smart retry mechanisms with adaptive backoff
- Detailed error logging with context
- Automatic recovery actions for common errors
"""

import asyncio
import json
import logging
import random
import re
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ccxt

# Set up logging
logger = logging.getLogger("bitget_error_manager")
if not logger.handlers:
    handler = logging.FileHandler("logs/bitget_errors.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Import error_logger if available
try:
    from error_logger import log_error, ErrorCategory, ErrorSeverity
    HAS_ERROR_LOGGER = True
except ImportError:
    HAS_ERROR_LOGGER = False

class BitgetErrorType(Enum):
    """Categorization of Bitget error types"""
    RATE_LIMIT = "RATE_LIMIT"
    AUTHENTICATION = "AUTHENTICATION"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INVALID_ORDER = "INVALID_ORDER"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MARGIN_MODE_ERROR = "MARGIN_MODE_ERROR"
    LEVERAGE_ERROR = "LEVERAGE_ERROR"
    PRICE_DEVIATION = "PRICE_DEVIATION"
    SYMBOL_ERROR = "SYMBOL_ERROR"
    POSITION_ERROR = "POSITION_ERROR"
    MARKET_ERROR = "MARKET_ERROR"

class BitgetErrorCode:
    """Database of all known Bitget error codes with solutions"""
    
    # Authentication Errors (40xxx)
    AUTH_ERRORS = {
        '40001': {
            'message': 'Invalid API Key',
            'solution': 'Verify API key is correct and has proper permissions',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'check_api_key'
        },
        '40002': {
            'message': 'Invalid signature',
            'solution': 'Check signature generation algorithm and secret key',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'check_signature'
        },
        '40003': {
            'message': 'Invalid timestamp',
            'solution': 'Ensure timestamp is within 30 seconds of server time',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'sync_time'
        },
        '40004': {
            'message': 'Invalid passphrase',
            'solution': 'Verify passphrase matches the one set during API key creation',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'check_passphrase'
        },
        '40005': {
            'message': 'IP not whitelisted',
            'solution': 'Add your IP address to the API key whitelist',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'whitelist_ip'
        },
        '40006': {
            'message': 'API key permissions insufficient',
            'solution': 'Enable required permissions (read/trade/withdraw) for API key',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'check_permissions'
        },
        '40007': {
            'message': 'API key expired',
            'solution': 'Renew or create a new API key',
            'type': BitgetErrorType.AUTHENTICATION,
            'action': 'renew_api_key'
        }
    }
    
    # Trading Errors (50xxx)
    TRADING_ERRORS = {
        '50001': {
            'message': 'Insufficient balance',
            'solution': 'Check account balance and ensure sufficient funds',
            'type': BitgetErrorType.INSUFFICIENT_BALANCE,
            'action': 'check_balance'
        },
        '50002': {
            'message': 'Order size too small',
            'solution': 'Increase order size to meet minimum requirements',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'increase_size'
        },
        '50003': {
            'message': 'Order size too large',
            'solution': 'Reduce order size to within maximum limits',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'decrease_size'
        },
        '50004': {
            'message': 'Order price deviation too large',
            'solution': 'Adjust order price closer to current market price',
            'type': BitgetErrorType.PRICE_DEVIATION,
            'action': 'adjust_price'
        },
        '50005': {
            'message': 'Order not found',
            'solution': 'Verify order ID exists and belongs to your account',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'check_order_exists'
        },
        '50006': {
            'message': 'Order already cancelled',
            'solution': 'Check order status before attempting to cancel',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'check_order_status'
        },
        '50007': {
            'message': 'Order already filled',
            'solution': 'Cannot modify/cancel filled orders',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'check_order_status'
        },
        '50008': {
            'message': 'Trading pair not available',
            'solution': 'Check if trading pair is active and properly formatted',
            'type': BitgetErrorType.SYMBOL_ERROR,
            'action': 'check_symbol'
        },
        '50009': {
            'message': 'Market closed',
            'solution': 'Wait for market to reopen or check trading hours',
            'type': BitgetErrorType.MARKET_ERROR,
            'action': 'wait_market_open'
        },
        '50010': {
            'message': 'Position size limit exceeded',
            'solution': 'Reduce position size to within allowed limits',
            'type': BitgetErrorType.POSITION_ERROR,
            'action': 'reduce_position_size'
        },
        '50067': {
            'message': 'Order price deviates greatly from index price',
            'solution': 'Adjust order price closer to index price or use market order',
            'type': BitgetErrorType.PRICE_DEVIATION,
            'action': 'adjust_price_to_index'
        },
        '43012': {
            'message': 'Insufficient margin',
            'solution': 'Add more margin or reduce position size',
            'type': BitgetErrorType.INSUFFICIENT_BALANCE,
            'action': 'add_margin'
        }
    }
    
    # Rate Limiting Errors
    RATE_LIMIT_ERRORS = {
        '429': {
            'message': 'Too many requests',
            'solution': 'Implement exponential backoff and reduce request frequency',
            'type': BitgetErrorType.RATE_LIMIT,
            'action': 'exponential_backoff'
        },
        '30001': {
            'message': 'Request too frequent',
            'solution': 'Add delays between requests and implement rate limiting',
            'type': BitgetErrorType.RATE_LIMIT,
            'action': 'exponential_backoff'
        },
        '30002': {
            'message': 'IP rate limit exceeded',
            'solution': 'Reduce request frequency or distribute across multiple IPs',
            'type': BitgetErrorType.RATE_LIMIT,
            'action': 'ip_rate_limit'
        },
        '30003': {
            'message': 'UID rate limit exceeded',
            'solution': 'Implement per-user rate limiting',
            'type': BitgetErrorType.RATE_LIMIT,
            'action': 'uid_rate_limit'
        },
        '30007': {
            'message': 'Request over limit, connection close',
            'solution': 'Reduce WebSocket subscriptions and implement rate limiting',
            'type': BitgetErrorType.RATE_LIMIT,
            'action': 'reduce_subscriptions'
        }
    }
    
    # System Errors
    SYSTEM_ERRORS = {
        '10001': {
            'message': 'System error',
            'solution': 'Retry request after delay, contact support if persistent',
            'type': BitgetErrorType.SERVER_ERROR,
            'action': 'retry_with_delay'
        },
        '10002': {
            'message': 'System maintenance',
            'solution': 'Wait for maintenance to complete, check announcements',
            'type': BitgetErrorType.SERVER_ERROR,
            'action': 'wait_maintenance'
        },
        '10003': {
            'message': 'Request timeout',
            'solution': 'Retry request with exponential backoff',
            'type': BitgetErrorType.NETWORK_ERROR,
            'action': 'retry_with_backoff'
        }
    }
    
    # Margin and Leverage Errors
    MARGIN_ERRORS = {
        '40309': {
            'message': 'Symbol has been removed',
            'solution': 'Remove this symbol from your trading list',
            'type': BitgetErrorType.SYMBOL_ERROR,
            'action': 'remove_symbol'
        },
        '40797': {
            'message': 'Leverage exceeds maximum',
            'solution': 'Reduce leverage to within allowed limits',
            'type': BitgetErrorType.LEVERAGE_ERROR,
            'action': 'reduce_leverage'
        },
        '400172': {
            'message': 'Margin coin cannot be empty',
            'solution': 'Add correct marginCoin parameter',
            'type': BitgetErrorType.MARGIN_MODE_ERROR,
            'action': 'add_margin_coin'
        },
        '45110': {
            'message': 'Minimum trading amount',
            'solution': 'Increase trading amount to meet minimum requirements',
            'type': BitgetErrorType.INVALID_ORDER,
            'action': 'increase_amount'
        },
        '40019': {
            'message': 'Parameter holdSide error',
            'solution': 'Correct holdSide parameter or omit it',
            'type': BitgetErrorType.VALIDATION_ERROR,
            'action': 'fix_hold_side'
        }
    }
    
    # Combined error database
    ALL_ERRORS = {
        **AUTH_ERRORS,
        **TRADING_ERRORS,
        **RATE_LIMIT_ERRORS,
        **SYSTEM_ERRORS,
        **MARGIN_ERRORS
    }
    
    @classmethod
    def get_error_info(cls, error_code: str) -> dict:
        """Get error information for a specific error code"""
        # Default error info for unknown codes
        default_error = {
            'message': 'Unknown error',
            'solution': 'Check the error code in Bitget documentation',
            'type': BitgetErrorType.VALIDATION_ERROR,
            'action': 'manual_review'
        }
        
        # Check in all error categories
        return cls.ALL_ERRORS.get(error_code, default_error)
    
    @classmethod
    def extract_error_code(cls, error_str: str) -> Optional[str]:
        """Extract error code from error message string"""
        # Common error code patterns in Bitget API responses
        patterns = [
            r'code[\'"]*:[\'"]*(\d+)',  # code:"50067" or code:50067
            r'code[\'"]?\s*:\s*[\'"]?(\d+)',  # code: "50067" with spaces
            r'HTTP\s+(\d+)',  # HTTP 429
            r'"code"\s*:\s*"?(\d+)"?',  # "code": "50067" or "code": 50067
            r'"code":"(\d+)"',  # "code":"50067" 
            r'bitget.*?(\d{5,})',  # bitget 50067
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Special case for HTTP 429
        if "429" in error_str or "too many requests" in error_str.lower():
            return "429"
            
        return None

class BitgetRateLimiter:
    """Advanced rate limiter for Bitget API with per-endpoint tracking"""
    
    def __init__(self):
        # Define Bitget API rate limits per endpoint
        self.rate_limits = {
            # Public endpoints
            'fetch_ticker': 20,
            'fetch_tickers': 20,
            'fetch_ohlcv': 20,
            'fetch_order_book': 20,
            'fetch_trades': 20,
            'fetch_markets': 20,
            
            # Private account endpoints
            'fetch_balance': 10,
            'fetch_positions': 10,
            
            # Private trading endpoints
            'create_order': 10,
            'cancel_order': 10,
            'cancel_all_orders': 10,
            'fetch_orders': 20,
            'fetch_open_orders': 20,
            'fetch_closed_orders': 20,
            'fetch_my_trades': 20,
            
            # Settings endpoints
            'set_leverage': 10,
            'set_margin_mode': 10,
            
            # Default for unspecified endpoints
            'default': 10
        }
        
        # Track request timestamps per endpoint
        self.request_timestamps = defaultdict(lambda: deque(maxlen=100))
        
        # Backoff tracking
        self.backoff_until = {}
        self.consecutive_429s = defaultdict(int)
        
    async def limit(self, endpoint: str = 'default') -> None:
        """Apply rate limiting for the specified endpoint"""
        current_time = time.time()
        
        # Check if we're in a backoff period for this endpoint
        backoff_time = self.backoff_until.get(endpoint, 0)
        if current_time < backoff_time:
            sleep_time = backoff_time - current_time
            logger.debug(f"Rate limit backoff for {endpoint}: waiting {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            current_time = time.time()  # Update current time after sleep
        
        # Get rate limit for endpoint (requests per second)
        limit = self.rate_limits.get(endpoint, self.rate_limits['default'])
        window = self.request_timestamps[endpoint]
        
        # Remove old requests (older than 1 second)
        while window and current_time - window[0] > 1.0:
            window.popleft()
        
        # Check if we need to wait
        if len(window) >= limit:
            sleep_time = 1.0 - (current_time - window[0]) + 0.01  # Extra 10ms buffer
            if sleep_time > 0:
                logger.debug(f"Rate limit sleep for {endpoint}: {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Add current request timestamp
        self.request_timestamps[endpoint].append(time.time())
    
    def register_429(self, endpoint: str) -> float:
        """Register a 429 (Too Many Requests) response and calculate backoff time"""
        self.consecutive_429s[endpoint] += 1
        count = self.consecutive_429s[endpoint]
        
        # Exponential backoff with jitter
        base_delay = min(60, 2 ** count)  # Cap at 60 seconds
        jitter = random.uniform(0, 0.25 * base_delay)  # 0-25% jitter
        backoff = base_delay + jitter
        
        # Set backoff until timestamp
        self.backoff_until[endpoint] = time.time() + backoff
        
        logger.warning(f"Rate limit hit for {endpoint} (#{count}). Backing off for {backoff:.2f}s")
        return backoff
    
    def reset_429(self, endpoint: str) -> None:
        """Reset consecutive 429 counter after successful request"""
        if endpoint in self.consecutive_429s and self.consecutive_429s[endpoint] > 0:
            self.consecutive_429s[endpoint] = 0

class BitgetErrorManager:
    """Comprehensive error manager for Bitget API with recovery actions"""
    
    def __init__(self):
        self.rate_limiter = BitgetRateLimiter()
        self.server_time_offset = 0
        self.recovery_actions = {
            'sync_time': self._sync_time,
            'exponential_backoff': self._exponential_backoff,
            'check_balance': self._check_balance,
            'adjust_price': self._adjust_price,
            'adjust_price_to_index': self._adjust_price_to_index,
            'reduce_leverage': self._reduce_leverage,
            'add_margin_coin': self._add_margin_coin,
            'fix_hold_side': self._fix_hold_side,
            'remove_symbol': self._remove_symbol
        }
        
        # Error statistics for monitoring
        self.error_counts = defaultdict(int)
        self.error_timestamps = defaultdict(list)
        self.last_cleanup = time.time()
        
        # Store recently removed symbols
        self.removed_symbols = set()
        
    async def handle_error(self, error: Exception, exchange: ccxt.Exchange, 
                          endpoint: str = 'default', retry_count: int = 0, 
                          max_retries: int = 3, **context) -> Tuple[bool, Any]:
        """
        Handle Bitget API errors with recovery actions
        
        Args:
            error: The exception that occurred
            exchange: The ccxt exchange instance
            endpoint: API endpoint that was called
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries
            context: Additional context (symbol, params, etc.)
            
        Returns:
            Tuple[bool, Any]: (should_retry, result_or_error)
        """
        error_str = str(error)
        
        # Extract error code from error message
        error_code = BitgetErrorCode.extract_error_code(error_str)
        
        # Clean up old error statistics periodically
        current_time = time.time()
        if current_time - self.last_cleanup > 3600:  # Every hour
            self._cleanup_error_stats()
            self.last_cleanup = current_time
        
        # Get error information
        error_info = BitgetErrorCode.get_error_info(error_code) if error_code else {
            'message': 'Unknown error',
            'solution': 'Check Bitget documentation',
            'type': BitgetErrorType.VALIDATION_ERROR,
            'action': 'manual_review'
        }
        
        error_type = error_info['type']
        error_action = error_info['action']
        
        # Update error statistics
        self.error_counts[error_code] += 1
        self.error_timestamps[error_code].append(time.time())
        
        # Log the error
        self._log_error(
            error_code=error_code,
            error_message=error_str,
            error_type=error_type,
            endpoint=endpoint,
            retry_count=retry_count,
            context=context
        )
        
        # Special handling for removed symbols (40309)
        if error_code == '40309' and 'symbol' in context:
            symbol = context.get('symbol')
            if symbol:
                logger.error(f"Symbol {symbol} has been removed from the exchange")
                self.removed_symbols.add(symbol)
                return False, error
        
        # Check if we've exceeded max retries
        if retry_count >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded for {endpoint}")
            return False, error
        
        # Handle rate limiting specifically
        if error_type == BitgetErrorType.RATE_LIMIT:
            backoff_time = self.rate_limiter.register_429(endpoint)
            await asyncio.sleep(backoff_time)
            return True, None
        
        # Execute recovery action if available
        if error_action in self.recovery_actions:
            try:
                recovery_result = await self.recovery_actions[error_action](
                    exchange=exchange,
                    error_code=error_code,
                    endpoint=endpoint,
                    retry_count=retry_count,
                    **context
                )
                
                if recovery_result is not None:
                    logger.info(f"Recovery action '{error_action}' succeeded")
                    return True, recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery action '{error_action}' failed: {recovery_error}")
        
        # If no specific recovery or recovery failed, use generic retry logic
        if retry_count < max_retries:
            # Calculate delay with exponential backoff
            delay = 1.0 * (2 ** retry_count)
            logger.info(f"Retrying {endpoint} in {delay:.2f}s (attempt {retry_count+1}/{max_retries})")
            await asyncio.sleep(delay)
            return True, None
        
        return False, error
    
    def _log_error(self, error_code: str, error_message: str, error_type: BitgetErrorType,
                  endpoint: str, retry_count: int, context: dict) -> None:
        """Log error with context"""
        # Get error info from database
        error_info = BitgetErrorCode.get_error_info(error_code) if error_code else {
            'message': 'Unknown error',
            'solution': 'Check Bitget documentation'
        }
        
        # Prepare log message
        log_message = (
            f"Bitget API error [{error_code or 'UNKNOWN'}]: {error_message}"
            f" | Endpoint: {endpoint} | Retry: {retry_count}"
            f" | Solution: {error_info['solution']}"
        )
        
        # Determine severity based on error type and retry count
        if error_type in [BitgetErrorType.AUTHENTICATION, BitgetErrorType.SERVER_ERROR]:
            logger.error(log_message)
            severity = "ERROR"
        elif retry_count >= 2:
            logger.error(log_message)
            severity = "ERROR"
        else:
            logger.warning(log_message)
            severity = "WARNING"
        
        # Log to unified error logger if available
        if HAS_ERROR_LOGGER:
            log_error(
                message=log_message,
                category=f"BITGET_{error_type.value}" if error_type else "BITGET_API",
                severity=severity,
                context={
                    "error_code": error_code,
                    "endpoint": endpoint,
                    "retry_count": retry_count,
                    "solution": error_info['solution'],
                    **{k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, type(None)))}
                }
            )
    
    async def _sync_time(self, exchange: ccxt.Exchange, **kwargs) -> Optional[int]:
        """Synchronize local time with server time"""
        try:
            server_time = await exchange.fetch_time()
            local_time = int(time.time() * 1000)
            self.server_time_offset = server_time - local_time
            logger.info(f"Synced server time, offset: {self.server_time_offset}ms")
            return server_time
        except Exception as e:
            logger.error(f"Failed to sync server time: {e}")
            return None
    
    async def _exponential_backoff(self, retry_count: int = 0, **kwargs) -> None:
        """Implement exponential backoff"""
        delay = 1.0 * (2 ** retry_count) + random.uniform(0, 0.5)
        logger.info(f"Rate limit backoff: waiting {delay:.2f}s")
        await asyncio.sleep(delay)
        return True
    
    async def _check_balance(self, exchange: ccxt.Exchange, **kwargs) -> Optional[Dict]:
        """Check account balance and verify sufficiency"""
        try:
            balance = await exchange.fetch_balance()
            if 'USDT' in balance:
                usdt_balance = balance['USDT']
                logger.info(f"USDT Balance: Free={usdt_balance['free']}, Used={usdt_balance['used']}, Total={usdt_balance['total']}")
                return balance
            else:
                logger.warning("USDT balance not found")
                return None
        except Exception as e:
            logger.error(f"Failed to check balance: {e}")
            return None
    
    async def _adjust_price(self, retry_count: int = 0, symbol: str = None, 
                          exchange: ccxt.Exchange = None, side: str = None, **kwargs) -> Optional[float]:
        """Adjust order price based on current market conditions"""
        try:
            if not exchange or not symbol:
                return None
                
            # Fetch ticker for current price
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate adjusted price with increased slippage for each retry
            slippage_percent = min(0.05 * (retry_count + 1), 0.15)  # 5% base, max 15%
            
            if side == 'buy':
                # For buy orders, increase price to ensure execution
                adjusted_price = current_price * (1 + slippage_percent)
            else:
                # For sell orders, decrease price to ensure execution
                adjusted_price = current_price * (1 - slippage_percent)
                
            logger.info(f"Adjusted price for {symbol} {side}: {current_price} -> {adjusted_price} (slippage: {slippage_percent*100:.1f}%)")
            return adjusted_price
            
        except Exception as e:
            logger.error(f"Failed to adjust price: {e}")
            return None
    
    async def _adjust_price_to_index(self, retry_count: int = 0, symbol: str = None,
                                   exchange: ccxt.Exchange = None, side: str = None, **kwargs) -> Optional[float]:
        """Adjust order price relative to index price for price deviation errors"""
        try:
            if not exchange or not symbol:
                return None
                
            # Fetch index price if available (exchange specific)
            index_price = None
            try:
                # Try to get index price from different APIs
                index_data = await exchange.fetch_index_ohlcv(symbol, '1m', limit=1)
                if index_data and len(index_data) > 0:
                    index_price = index_data[0][4]  # Close price
            except:
                pass
                
            # Fall back to ticker if index price not available
            if not index_price:
                ticker = await exchange.fetch_ticker(symbol)
                index_price = ticker['last']
            
            # Calculate adjusted price with increased slippage for each retry
            # 50067 errors need more aggressive slippage
            slippage_percent = min(0.03 * (retry_count + 2), 0.15)  # Start at 6%, max 15%
            
            if side == 'buy':
                # For buy orders, increase price to ensure execution
                adjusted_price = index_price * (1 + slippage_percent)
            else:
                # For sell orders, decrease price to ensure execution
                adjusted_price = index_price * (1 - slippage_percent)
                
            logger.info(f"Adjusted price for {symbol} {side} to index: {index_price} -> {adjusted_price} (slippage: {slippage_percent*100:.1f}%)")
            return adjusted_price
            
        except Exception as e:
            logger.error(f"Failed to adjust price to index: {e}")
            return None
    
    async def _reduce_leverage(self, exchange: ccxt.Exchange = None, 
                             symbol: str = None, leverage: int = None, **kwargs) -> Optional[int]:
        """Reduce leverage when it exceeds maximum allowed"""
        try:
            if not exchange or not symbol or not leverage:
                return None
                
            # Get the maximum allowed leverage from error message if possible
            error_message = kwargs.get('error_message', '')
            max_leverage = None
            
            try:
                # Try to extract max leverage from error message
                match = re.search(r'maximum.*?(\d+)', error_message.lower())
                if match:
                    max_leverage = int(match.group(1))
            except:
                pass
                
            # Use extracted max leverage or fallback to a safe value
            if max_leverage:
                new_leverage = max_leverage
            else:
                # Reduce leverage by 50% if we can't determine the maximum
                new_leverage = max(1, int(leverage * 0.5))
                
            logger.info(f"Reducing leverage for {symbol}: {leverage}x -> {new_leverage}x")
            
            # Set the new leverage
            params = {
                "marginCoin": "USDT",
                "symbol": symbol.replace("/", "") if "/" in symbol else symbol
            }
            
            result = await exchange.set_leverage(new_leverage, symbol=symbol, params=params)
            logger.info(f"Successfully set reduced leverage to {new_leverage}x")
            return new_leverage
            
        except Exception as e:
            logger.error(f"Failed to reduce leverage: {e}")
            return None
    
    async def _add_margin_coin(self, exchange: ccxt.Exchange = None, 
                              params: dict = None, **kwargs) -> Optional[dict]:
        """Add missing marginCoin parameter"""
        try:
            if not params:
                params = {}
                
            # Add the marginCoin parameter if missing
            if 'marginCoin' not in params:
                params['marginCoin'] = 'USDT'
                
            logger.info(f"Added marginCoin parameter: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to add marginCoin parameter: {e}")
            return None
    
    async def _fix_hold_side(self, exchange: ccxt.Exchange = None, 
                            params: dict = None, **kwargs) -> Optional[dict]:
        """Fix holdSide parameter issues"""
        try:
            if not params:
                params = {}
                
            # Remove problematic holdSide parameter if present
            if 'holdSide' in params:
                del params['holdSide']
                logger.info("Removed problematic holdSide parameter")
                
            return params
            
        except Exception as e:
            logger.error(f"Failed to fix holdSide parameter: {e}")
            return None
    
    async def _remove_symbol(self, symbol: str = None, **kwargs) -> bool:
        """Mark symbol as removed from exchange"""
        if symbol:
            logger.info(f"Marking symbol {symbol} as removed from exchange")
            self.removed_symbols.add(symbol)
            return True
        return False
    
    def _cleanup_error_stats(self) -> None:
        """Clean up old error statistics"""
        current_time = time.time()
        one_day_ago = current_time - (24 * 3600)
        
        for error_code in list(self.error_timestamps.keys()):
            # Keep only timestamps from the last 24 hours
            self.error_timestamps[error_code] = [
                ts for ts in self.error_timestamps[error_code] if ts > one_day_ago
            ]
            
            # Update counts based on remaining timestamps
            self.error_counts[error_code] = len(self.error_timestamps[error_code])
            
            # Remove empty entries
            if self.error_counts[error_code] == 0:
                del self.error_counts[error_code]
                del self.error_timestamps[error_code]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        stats = {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': dict(self.error_counts),
            'error_rates': {},
            'removed_symbols': list(self.removed_symbols)
        }
        
        # Calculate error rates (errors per hour)
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        for error_code, timestamps in self.error_timestamps.items():
            recent_errors = [ts for ts in timestamps if ts > one_hour_ago]
            stats['error_rates'][error_code] = len(recent_errors)
            
        return stats

# Create singleton instance
_error_manager = None

def get_error_manager() -> BitgetErrorManager:
    """Get the singleton BitgetErrorManager instance"""
    global _error_manager
    if _error_manager is None:
        _error_manager = BitgetErrorManager()
    return _error_manager

# Utility function for easy access
async def handle_bitget_error(error: Exception, exchange: ccxt.Exchange, 
                            endpoint: str = 'default', retry_count: int = 0, 
                            max_retries: int = 3, **context) -> Tuple[bool, Any]:
    """Handle Bitget API error with the singleton error manager"""
    manager = get_error_manager()
    return await manager.handle_error(
        error=error,
        exchange=exchange,
        endpoint=endpoint,
        retry_count=retry_count,
        max_retries=max_retries,
        **context
    )

# Utility function to create a decorated execute_with_retry method
async def execute_with_retry(method: Callable, *args, 
                           endpoint: str = 'default',
                           max_retries: int = 3,
                           exchange: ccxt.Exchange = None,
                           **kwargs) -> Any:
    """
    Execute a method with automatic retry handling for Bitget API errors
    
    Args:
        method: The method to execute
        endpoint: API endpoint category for rate limiting
        max_retries: Maximum number of retry attempts
        exchange: The ccxt exchange instance
        *args, **kwargs: Arguments to pass to the method
        
    Returns:
        The result of the method call
    """
    manager = get_error_manager()
    rate_limiter = manager.rate_limiter
    
    # Apply rate limiting before request
    await rate_limiter.limit(endpoint)
    
    retry_count = 0
    context = {'endpoint': endpoint, **kwargs}
    
    while True:
        try:
            # Execute the method
            result = await method(*args, **kwargs)
            
            # Reset rate limit counter on success
            rate_limiter.reset_429(endpoint)
            
            return result
            
        except Exception as error:
            # Handle the error
            should_retry, recovery_result = await handle_bitget_error(
                error=error,
                exchange=exchange,
                endpoint=endpoint,
                retry_count=retry_count,
                max_retries=max_retries,
                **context
            )
            
            if should_retry:
                retry_count += 1
                # If recovery produced a result, use it
                if recovery_result is not None:
                    return recovery_result
                # Otherwise continue the loop to retry
                continue
            else:
                # Re-raise the error if we shouldn't retry
                raise 