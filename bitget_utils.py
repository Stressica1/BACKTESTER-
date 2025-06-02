#!/usr/bin/env python3
"""
Bitget API Utilities - Comprehensive error handling, rate limiting, and recovery

This module provides robust utilities for working with the Bitget API:
- Rate limiting with adaptive backoff
- Error handling with specific error code mapping
- Recovery mechanisms for common errors
- Logging standardization
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ccxt

# Configure module logger
logger = logging.getLogger("bitget_utils")

class BitgetErrorType(Enum):
    RATE_LIMIT = "RATE_LIMIT"
    AUTHENTICATION = "AUTHENTICATION"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INVALID_ORDER = "INVALID_ORDER"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CROSS_MARGIN_ERROR = "CROSS_MARGIN_ERROR"
    LEVERAGE_ERROR = "LEVERAGE_ERROR"
    POSITION_ERROR = "POSITION_ERROR"

class BitgetAPIError(Exception):
    """Enhanced exception for Bitget API errors with detailed context"""
    def __init__(self, error_code: str, message: str, error_type: BitgetErrorType, 
                 response_data: Optional[Dict] = None, recovery_action: Optional[str] = None):
        self.error_code = error_code
        self.message = message
        self.error_type = error_type
        self.response_data = response_data or {}
        self.recovery_action = recovery_action
        self.timestamp = datetime.now().isoformat()
        super().__init__(f"[{error_code}] {message}")

class BitgetRateLimiter:
    """Advanced rate limiter with per-endpoint tracking and adaptive backoff"""
    
    def __init__(self):
        # Default rate limits based on Bitget docs
        self.rate_limits = {
            'fetch_ticker': 20,
            'fetch_ohlcv': 20, 
            'fetch_balance': 10,
            'create_order': 10,
            'cancel_order': 10,
            'fetch_markets': 20,
            'set_leverage': 10,
            'set_margin_mode': 10,
            'fetch_positions': 10,
            'default': 10  # Default for endpoints not explicitly listed
        }
        
        # Timestamp tracking for each endpoint
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

class BitgetErrorHandler:
    """Comprehensive error handler for Bitget API"""
    
    def __init__(self):
        self.rate_limiter = BitgetRateLimiter()
        self.error_map = self._build_error_map()
        self.recovery_actions = {
            'sync_time': self._sync_time,
            'exponential_backoff': self._exponential_backoff,
            'check_balance': self._check_balance,
            'adjust_price': self._adjust_price,
            'switch_to_isolated': self._switch_to_isolated,
            'reduce_leverage': self._reduce_leverage
        }
        self.server_time_offset = 0
    
    def _build_error_map(self) -> Dict[str, Tuple[BitgetErrorType, str]]:
        """Build comprehensive error code mapping"""
        return {
            # Authentication errors
            '40001': (BitgetErrorType.AUTHENTICATION, 'check_api_key'),
            '40002': (BitgetErrorType.AUTHENTICATION, 'check_signature'),
            '40003': (BitgetErrorType.AUTHENTICATION, 'sync_time'),
            '40004': (BitgetErrorType.AUTHENTICATION, 'check_passphrase'),
            '40005': (BitgetErrorType.AUTHENTICATION, 'whitelist_ip'),
            '40006': (BitgetErrorType.AUTHENTICATION, 'check_permissions'),
            
            # Trading errors
            '50001': (BitgetErrorType.INSUFFICIENT_BALANCE, 'check_balance'),
            '50002': (BitgetErrorType.INVALID_ORDER, 'increase_size'),
            '50003': (BitgetErrorType.INVALID_ORDER, 'decrease_size'),
            '50004': (BitgetErrorType.CROSS_MARGIN_ERROR, 'switch_to_isolated'),
            '50005': (BitgetErrorType.INVALID_ORDER, 'check_order_exists'),
            '50006': (BitgetErrorType.INVALID_ORDER, 'check_order_status'),
            '50007': (BitgetErrorType.INVALID_ORDER, 'check_order_status'),
            '50067': (BitgetErrorType.INVALID_ORDER, 'adjust_price'),
            '43012': (BitgetErrorType.INSUFFICIENT_BALANCE, 'check_balance'),
            
            # Rate limiting
            '429': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            '30001': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            '30002': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            '30003': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            
            # System errors
            '10001': (BitgetErrorType.SERVER_ERROR, 'retry_with_delay'),
            '10002': (BitgetErrorType.SERVER_ERROR, 'wait_maintenance'),
            '10003': (BitgetErrorType.NETWORK_ERROR, 'retry_with_backoff'),
            
            # Position errors
            '45110': (BitgetErrorType.POSITION_ERROR, 'check_minimum_amount')
        }
    
    async def handle_error(self, error: Exception, exchange: ccxt.Exchange, 
                           endpoint: str = 'default', retry_count: int = 0, 
                           max_retries: int = 3, **context) -> Tuple[bool, Any]:
        """
        Handle Bitget API errors with appropriate recovery actions
        
        Args:
            error: The exception that occurred
            exchange: The ccxt exchange instance
            endpoint: API endpoint that was called
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries
            context: Additional context for error handling (symbol, params, etc.)
            
        Returns:
            Tuple[bool, Any]: (should_retry, result_or_error)
                - should_retry: Whether to retry the operation
                - result_or_error: Result if recovery successful, or original error
        """
        error_str = str(error)
        error_code = "unknown"
        
        # Extract error code from various error formats
        if hasattr(error, 'args') and len(error.args) > 0:
            for arg in error.args:
                arg_str = str(arg)
                # Try to find common error code patterns
                for code_pattern in ['code":"', 'code":', 'code=', '[code:', 'code ']:
                    if code_pattern in arg_str:
                        parts = arg_str.split(code_pattern, 1)
                        if len(parts) > 1:
                            potential_code = ''.join(c for c in parts[1].split('"')[0].split('}')[0].split(',')[0] if c.isdigit())
                            if potential_code:
                                error_code = potential_code
                                break
        
        # Special case for HTTP 429
        if "429" in error_str or "too many requests" in error_str.lower():
            error_code = "429"
        
        # Get error type and recovery action
        error_type, recovery_action = self.error_map.get(
            error_code, (BitgetErrorType.VALIDATION_ERROR, 'manual_review'))
        
        # Log the error with context
        log_message = (f"Bitget API error [{error_code}]: {error_str} "
                      f"(endpoint={endpoint}, retry={retry_count}/{max_retries})")
        
        if error_type in [BitgetErrorType.AUTHENTICATION, BitgetErrorType.INSUFFICIENT_BALANCE, 
                         BitgetErrorType.SERVER_ERROR]:
            logger.error(log_message)
        else:
            logger.warning(log_message)
        
        # Check if we've exceeded max retries
        if retry_count >= max_retries:
            logger.error(f"Max retries ({max_retries}) exceeded for {endpoint}")
            return False, error
        
        # Handle rate limiting specifically
        if error_type == BitgetErrorType.RATE_LIMIT:
            backoff_time = self.rate_limiter.register_429(endpoint)
            await asyncio.sleep(backoff_time)
            return True, None
        
        # Execute appropriate recovery action if available
        if recovery_action in self.recovery_actions:
            recovery_result = await self.recovery_actions[recovery_action](
                exchange=exchange, error_code=error_code, endpoint=endpoint, 
                retry_count=retry_count, **context)
            
            if recovery_result is not None:
                return True, recovery_result
        
        # Default retry with exponential backoff
        wait_time = min(60, 2 ** retry_count)  # Cap at 60 seconds
        jitter = random.uniform(0, 0.1 * wait_time)  # Add 0-10% jitter
        total_wait = wait_time + jitter
        
        logger.info(f"Retrying in {total_wait:.2f}s (attempt {retry_count + 1}/{max_retries})")
        await asyncio.sleep(total_wait)
        
        return True, None
    
    # Recovery action implementations
    async def _sync_time(self, exchange: ccxt.Exchange, **kwargs) -> Optional[int]:
        """Synchronize local time with server time"""
        try:
            server_time = await exchange.fetch_time()
            local_time = int(time.time() * 1000)
            self.server_time_offset = server_time - local_time
            logger.info(f"Time synchronized with Bitget server. Offset: {self.server_time_offset}ms")
            return server_time
        except Exception as e:
            logger.error(f"Failed to sync server time: {e}")
            return None
    
    async def _exponential_backoff(self, retry_count: int = 0, **kwargs) -> None:
        """Apply exponential backoff for rate limits"""
        base_delay = 2 ** retry_count
        jitter = random.uniform(0, 0.2 * base_delay)
        wait_time = base_delay + jitter
        
        logger.info(f"Rate limit backoff: waiting {wait_time:.2f}s")
        await asyncio.sleep(wait_time)
        return None
    
    async def _check_balance(self, exchange: ccxt.Exchange, **kwargs) -> Optional[Dict]:
        """Check account balance"""
        try:
            # Get balance for USDT-M futures
            balance = await exchange.fetch_balance({
                'type': 'swap',
                'product_type': 'umcbl'  # USDT-margined contracts
            })
            
            usdt_info = balance.get('USDT', {})
            free = usdt_info.get('free', 0)
            free_balance = float(free) if free is not None and free != '' else 0.0
            
            logger.info(f"Available USDT balance: {free_balance}")
            return {"available_balance": free_balance}
        except Exception as e:
            logger.error(f"Failed to check balance: {e}")
            return None
    
    async def _adjust_price(self, retry_count: int = 0, symbol: str = None, 
                           exchange: ccxt.Exchange = None, side: str = None, **kwargs) -> Optional[float]:
        """Adjust order price based on current market conditions"""
        if not symbol or not exchange:
            return None
            
        try:
            # Get current market price
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker.get('last', ticker.get('close', 0))
            
            if not current_price or current_price <= 0:
                return None
                
            # Apply slippage in the appropriate direction based on side and retry count
            slippage_factor = 0.002 * (retry_count + 1)  # Increase with each retry
            
            if side == 'buy':
                # For buy orders, increase price to beat price deviation checks
                adjusted_price = current_price * (1 + slippage_factor)
            else:
                # For sell orders, decrease price to beat price deviation checks
                adjusted_price = current_price * (1 - slippage_factor)
                
            logger.info(f"Adjusted price for {symbol} {side}: {current_price} → {adjusted_price}")
            return adjusted_price
        except Exception as e:
            logger.error(f"Failed to adjust price: {e}")
            return None
    
    async def _switch_to_isolated(self, exchange: ccxt.Exchange = None, 
                                symbol: str = None, **kwargs) -> Optional[Dict]:
        """Switch to isolated margin mode when cross margin is not supported"""
        if not exchange or not symbol:
            return None
            
        try:
            # Set isolated margin mode
            margin_params = {
                "symbol": symbol.replace("/", ""),
                "marginMode": "isolated"
            }
            
            await exchange.set_margin_mode('isolated', symbol, params=margin_params)
            logger.info(f"Switched to isolated margin mode for {symbol}")
            
            return {"margin_mode": "isolated"}
        except Exception as e:
            logger.error(f"Failed to switch to isolated margin: {e}")
            return None
    
    async def _reduce_leverage(self, exchange: ccxt.Exchange = None, 
                              symbol: str = None, leverage: int = None, **kwargs) -> Optional[int]:
        """Reduce leverage when current setting exceeds maximum allowed"""
        if not exchange or not symbol:
            return None
            
        try:
            # Try progressively lower leverage values
            leverage_options = [20, 15, 10, 5, 3]
            
            # Start with the highest leverage below our current value
            start_index = 0
            if leverage:
                for i, option in enumerate(leverage_options):
                    if option < leverage:
                        start_index = i
                        break
            
            # Try each leverage option
            for new_leverage in leverage_options[start_index:]:
                try:
                    params = {
                        "marginCoin": "USDT",
                        "holdSide": kwargs.get("holdSide", "long")
                    }
                    
                    await exchange.set_leverage(new_leverage, symbol, params=params)
                    logger.info(f"Reduced leverage for {symbol} to {new_leverage}x")
                    return new_leverage
                except Exception:
                    # Try next lower leverage
                    continue
                    
            # If all options failed, return the lowest value as a suggestion
            logger.warning(f"Could not set any leverage for {symbol}, suggesting minimum: {leverage_options[-1]}x")
            return leverage_options[-1]
        except Exception as e:
            logger.error(f"Failed to reduce leverage: {e}")
            return None

class BitgetUtils:
    """Unified interface for Bitget API utilities"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.rate_limiter = BitgetRateLimiter()
        self.error_handler = BitgetErrorHandler()
        self.logger = logger
        
        # Performance tracking
        self.api_calls = defaultdict(int)
        self.api_times = defaultdict(list)
        self.errors = defaultdict(int)
        
    async def execute_with_retry(self, method_name: str, *args, 
                               max_retries: int = 3, **kwargs) -> Any:
        """
        Execute any exchange method with automatic rate limiting and error handling
        
        Args:
            method_name: The ccxt exchange method to call (e.g., 'fetch_ticker')
            *args: Positional arguments to pass to the method
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Result from the exchange method
        
        Raises:
            BitgetAPIError: If all retries fail
        """
        start_time = time.time()
        method = getattr(self.exchange, method_name)
        retry_count = 0
        
        # Extract symbol for context if present
        symbol = kwargs.get('symbol', args[0] if args else None)
        context = {
            'symbol': symbol,
            **kwargs
        }
        
        while retry_count <= max_retries:
            try:
                # Apply rate limiting
                await self.rate_limiter.limit(method_name)
                
                # Execute API call
                result = await method(*args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*args, **kwargs)
                
                # Record successful call
                self.api_calls[method_name] += 1
                self.api_times[method_name].append(time.time() - start_time)
                
                # Reset rate limit counter on success
                self.rate_limiter.reset_429(method_name)
                
                return result
                
            except Exception as e:
                self.errors[method_name] += 1
                
                # Handle error and determine if retry is needed
                should_retry, recovery_result = await self.error_handler.handle_error(
                    error=e,
                    exchange=self.exchange,
                    endpoint=method_name,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    **context
                )
                
                if not should_retry:
                    # Convert to BitgetAPIError with full context
                    error_code = "unknown"
                    error_type = BitgetErrorType.VALIDATION_ERROR
                    
                    # Try to extract error code
                    error_str = str(e)
                    if "50067" in error_str:
                        error_code = "50067"
                        error_type = BitgetErrorType.INVALID_ORDER
                    elif "43012" in error_str or "insufficient balance" in error_str.lower():
                        error_code = "43012"
                        error_type = BitgetErrorType.INSUFFICIENT_BALANCE
                    elif "50004" in error_str:
                        error_code = "50004"
                        error_type = BitgetErrorType.CROSS_MARGIN_ERROR
                    elif "429" in error_str:
                        error_code = "429"
                        error_type = BitgetErrorType.RATE_LIMIT
                    
                    raise BitgetAPIError(
                        error_code=error_code,
                        message=str(e),
                        error_type=error_type,
                        response_data={"original_error": str(e), "method": method_name, "args": args, "kwargs": kwargs}
                    )
                
                # If recovery produced a result, return it
                if recovery_result is not None and recovery_result != e:
                    return recovery_result
                    
                retry_count += 1
        
        # If we've exhausted retries, raise the original error
        raise BitgetAPIError(
            error_code="max_retries",
            message=f"Maximum retries ({max_retries}) exceeded for {method_name}",
            error_type=BitgetErrorType.NETWORK_ERROR,
            response_data={"method": method_name, "retries": max_retries}
        )
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        stats = {
            "calls": dict(self.api_calls),
            "errors": dict(self.errors),
            "avg_times": {}
        }
        
        # Calculate average times
        for method, times in self.api_times.items():
            if times:
                stats["avg_times"][method] = sum(times) / len(times)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset API statistics"""
        self.api_calls.clear()
        self.api_times.clear()
        self.errors.clear()
    
    # Convenience wrappers for common API calls
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker with error handling and rate limiting"""
        return await self.execute_with_retry('fetch_ticker', symbol)
        
    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, 
                        limit: Optional[int] = None, params: Dict = None) -> List:
        """Fetch OHLCV data with error handling and rate limiting"""
        return await self.execute_with_retry('fetch_ohlcv', symbol, timeframe, since, limit, 
                                          **(params or {}))
    
    async def create_order(self, symbol: str, type: str, side: str, 
                         amount: float, price: Optional[float] = None, 
                         params: Dict = None) -> Dict:
        """Create order with error handling and rate limiting"""
        return await self.execute_with_retry('create_order', symbol, type, side, 
                                          amount, price, **(params or {}))
    
    async def set_leverage(self, leverage: int, symbol: str, params: Dict = None) -> Dict:
        """Set leverage with error handling and rate limiting"""
        return await self.execute_with_retry('set_leverage', leverage, symbol, 
                                          **(params or {}))
    
    async def set_margin_mode(self, margin_mode: str, symbol: str, params: Dict = None) -> Dict:
        """Set margin mode with error handling and rate limiting"""
        return await self.execute_with_retry('set_margin_mode', margin_mode, symbol, 
                                          **(params or {}))
    
    async def fetch_balance(self, params: Dict = None) -> Dict:
        """Fetch balance with error handling and rate limiting"""
        return await self.execute_with_retry('fetch_balance', **(params or {})) 