# Bitget API Rate Limit Management

This document provides guidance on managing Bitget API rate limits to prevent `429 Too Many Requests` errors that are appearing in the logs.

## Current Issues

The logs show multiple rate limit errors:
```
ERROR:__main__:Error fetching data for NOT/USDT:USDT: bitget {"code":"429","msg":"Too Many Requests","requestTime":1748721720474,"data":null}
ERROR:__main__:Error fetching data for FLOKI/USDT:USDT: bitget {"code":"429","msg":"Too Many Requests","requestTime":1748721720606,"data":null}
```

These errors occur when our trading bot exceeds Bitget's API rate limits (typically 20 requests/second for public endpoints and 10 requests/second for private endpoints).

## Implementation Steps

### 1. Add Rate Limiting to All API Calls

Update the `rate_limit` method in `supertrend_pullback_live.py` and similar files:

```python
async def rate_limit(self, endpoint='default'):
    """Enhanced rate limiter with proper throttling and tracking"""
    # Track endpoint specific limits - most public endpoints allow 20 req/s, private 10 req/s
    if not hasattr(self, '_rate_limit_tracker'):
        self._rate_limit_tracker = {
            'default': {'limit': 10, 'tokens': 10, 'last_refill': time.time()},
            'market': {'limit': 20, 'tokens': 20, 'last_refill': time.time()},
            'account': {'limit': 10, 'tokens': 10, 'last_refill': time.time()},
            'trade': {'limit': 10, 'tokens': 10, 'last_refill': time.time()}
        }
    
    # Refill tokens based on time elapsed (token bucket algorithm)
    bucket = self._rate_limit_tracker.get(endpoint, self._rate_limit_tracker['default'])
    now = time.time()
    time_passed = now - bucket['last_refill']
    
    # Calculate refill (don't exceed max)
    refill = min(bucket['limit'], bucket['tokens'] + time_passed * bucket['limit'])
    bucket['tokens'] = refill
    bucket['last_refill'] = now
    
    # If tokens available, proceed immediately
    if bucket['tokens'] >= 1:
        bucket['tokens'] -= 1
        return
    
    # Otherwise, wait until we have a token
    wait_time = (1 - bucket['tokens']) / bucket['limit']
    logger.debug(f"Rate limit reached for {endpoint}, waiting {wait_time:.2f}s")
    await asyncio.sleep(wait_time)
    bucket['tokens'] = 0  # Reset to exactly 0 after waiting
    bucket['last_refill'] = time.time()
```

### 2. Create a Connection Pool Manager

Implement a connection pool manager to avoid `Connection pool is full` warnings:

```python
def setup_connection_pool(self):
    """Set up and configure connection pooling for HTTP requests"""
    import urllib3
    
    # Create a pooling manager with proper settings
    self.http_pool_manager = urllib3.PoolManager(
        num_pools=10,        # Number of connection pools
        maxsize=20,          # Connections per pool
        timeout=5.0,         # Default timeout
        retries=1,           # Default retry count
        block=False          # Don't block when pool is full
    )
    
    # Apply to CCXT if using it
    if hasattr(self.exchange, 'session'):
        # Configure the session used by CCXT
        self.exchange.session.mount('https://', 
            requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=1
            )
        )
```

### 3. Implement Batch Requests Where Possible

Convert multiple single requests into batch requests:

```python
async def fetch_multiple_symbols_data(self, symbols, timeframe):
    """Fetch data for multiple symbols in a batch-oriented way"""
    # Group symbols into batches of 5 to reduce request count
    batch_size = 5
    symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
    
    results = {}
    for batch in symbol_batches:
        # Process each batch with a delay between batches
        batch_tasks = [self.fetch_data(symbol, timeframe) for symbol in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Store results
        for symbol, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
                results[symbol] = None
            else:
                results[symbol] = result
                
        # Add delay between batches to stay within rate limits
        await asyncio.sleep(0.5)
        
    return results
```

### 4. Implement Exponential Backoff for Rate Limit Errors

Add proper exponential backoff when rate limits are hit:

```python
async def handle_rate_limit_error(self, error, endpoint, retry_count=0):
    """Handle rate limit errors with exponential backoff"""
    # Check if it's a rate limit error
    is_rate_limit = False
    error_str = str(error)
    
    if '429' in error_str or 'Too Many Requests' in error_str:
        is_rate_limit = True
    elif '30001' in error_str or '30002' in error_str or '30003' in error_str:
        # Bitget specific rate limit error codes
        is_rate_limit = True
        
    if is_rate_limit:
        # Implement exponential backoff
        if retry_count >= 5:
            logger.error(f"Maximum retries exceeded for rate limit on {endpoint}")
            raise Exception(f"Rate limit retry exhausted: {error}")
            
        # Calculate backoff time: 2^retry * (0.5-1.5 random jitter)
        backoff = (2 ** retry_count) * (0.5 + random.random())
        logger.warning(f"Rate limit hit for {endpoint}, backing off for {backoff:.2f}s (retry {retry_count+1}/5)")
        
        # Wait for backoff period
        await asyncio.sleep(backoff)
        
        return True  # Signal to retry
    
    return False  # Not a rate limit error, don't retry
```

### 5. Implement Request Throttling

Add a request throttling mechanism to the main trading loop:

```python
def adjust_throttling(self):
    """Dynamically adjust request throttling based on recent rate limit errors"""
    if not hasattr(self, '_rate_limit_errors'):
        self._rate_limit_errors = []
    
    # Keep only errors from the last 60 seconds
    now = time.time()
    self._rate_limit_errors = [t for t in self._rate_limit_errors if now - t < 60]
    
    # Calculate error rate and adjust throttling
    error_count = len(self._rate_limit_errors)
    if error_count > 10:
        # Severe rate limiting - process one symbol at a time with 1s delay
        self.symbols_per_iteration = 1
        self.iteration_delay = 1.0
        logger.warning("⚠️ Severe rate limiting! Processing 1 symbol at a time with 1s delay.")
    elif error_count > 5:
        # Moderate rate limiting - reduce batch size and increase delay
        self.symbols_per_iteration = 2
        self.iteration_delay = 0.5
        logger.warning("⚠️ Moderate rate limiting! Reduced processing to 2 symbols with 0.5s delay.")
    else:
        # Normal operation
        self.symbols_per_iteration = 5
        self.iteration_delay = 0.2
```

### 6. Monitor Rate Limit Usage

Create a simple rate limit monitoring system:

```python
def log_rate_limit_usage(self):
    """Log current rate limit usage statistics"""
    if hasattr(self, '_rate_limit_tracker'):
        logger.info("📊 Current API Rate Limit Usage:")
        for endpoint, data in self._rate_limit_tracker.items():
            usage_pct = ((data['limit'] - data['tokens']) / data['limit']) * 100
            status = "🟢" if usage_pct < 70 else "🟠" if usage_pct < 90 else "🔴"
            logger.info(f"  {status} {endpoint}: {usage_pct:.1f}% used ({data['limit'] - data['tokens']:.1f}/{data['limit']} req/s)")
```

### 7. Integrate with Bitget Error Management

Ensure the rate limiting integrates with the broader error management system:

```python
async def handle_bitget_error(self, error, symbol=None, retry_count=0):
    """Enhanced error handler with rate limit management"""
    error_str = str(error)
    error_code = None
    
    # Extract error code if available
    if '"code":"' in error_str:
        try:
            error_code = error_str.split('"code":"')[1].split('"')[0]
        except:
            pass
    
    # Handle rate limit errors
    if error_code == '429' or error_code == '30001' or error_code == '30002' or error_code == '30003':
        # Record the rate limit error
        if not hasattr(self, '_rate_limit_errors'):
            self._rate_limit_errors = []
        self._rate_limit_errors.append(time.time())
        
        # Apply backoff
        is_rate_limit = await self.handle_rate_limit_error(error, 'bitget_api', retry_count)
        if is_rate_limit:
            return True  # Retry the operation
    
    # Handle other errors...
    return False
```

## Monitoring and Troubleshooting

1. **Check the rate limit logs:**
   ```
   Get-Content -Tail 30 logs/bitget_rate_limits.log
   ```

2. **Monitor connection pool warnings:**
   ```
   Get-Content -Tail 100 logs/trading_unified.log | Select-String "Connection pool"
   ```

3. **Analyze request frequency:**
   ```
   Get-Content logs/trading_unified.log | Select-String "API request" | Measure-Object -Line
   ```

4. **Visualize rate limit trends:**
   Create a simple script to visualize rate limit usage over time using the logged data.

## Best Practices

1. **Batch similar requests** where possible
2. **Prioritize critical requests** over non-essential ones
3. **Cache responses** that don't change frequently
4. **Implement circuit breakers** to prevent cascading failures
5. **Use asynchronous programming** efficiently
6. **Monitor and log rate limit usage** regularly
7. **Adjust request frequency dynamically** based on error rates

By implementing these measures, we can dramatically reduce the 429 errors while maintaining efficient trading operations.

BUSSIED!!!!! 