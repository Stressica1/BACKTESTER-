---
description: 
globs: 
alwaysApply: true
---
target_file: .roo/rules/bitget_error_management.md
instructions: Create a comprehensive error management and version control rule for Bitget API integration with all error codes and their solutions.
code_edit: ---
description: Comprehensive Bitget API error management, version control, and troubleshooting guide
globs: scripts/bitget_*, src/exchange/bitget/**, any/bitget/**, *.py, *.js, *.ts, src/ccxt/**, exchange/**, trading/**, logs/**, error_handling/**
alwaysApply: true
---

- **Bitget API Error Management & Version Control**
    - **All API integrations must implement comprehensive error handling with proper logging and recovery mechanisms.**
    - **Version control all configuration changes and API implementations.**
    - **Maintain detailed error logs with contextual information for debugging.**

- **HTTP Status Codes & Handling**
    - **Standard HTTP Error Responses:**
        ```typescript
        const httpErrors = {
            400: 'Bad Request - Invalid request format',
            401: 'Unauthorized - Invalid API Key or signature',
            403: 'Forbidden - No access to requested resource',
            404: 'Not Found - Endpoint or resource not found',
            429: 'Too Many Requests - Rate limit exceeded',
            500: 'Internal Server Error - Server-side error',
            502: 'Bad Gateway - Network error',
            503: 'Service Unavailable - Maintenance mode'
        };
        ```

    - **Error Handling Implementation:**
        ```python
        # ✅ DO: Comprehensive error handling
        import logging
        from enum import Enum
        from typing import Optional, Dict, Any
        
        class BitgetErrorType(Enum):
            RATE_LIMIT = "RATE_LIMIT"
            AUTHENTICATION = "AUTHENTICATION"
            INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
            INVALID_ORDER = "INVALID_ORDER"
            NETWORK_ERROR = "NETWORK_ERROR"
            SERVER_ERROR = "SERVER_ERROR"
            VALIDATION_ERROR = "VALIDATION_ERROR"
        
        class BitgetAPIError(Exception):
            def __init__(self, error_code: str, message: str, error_type: BitgetErrorType, 
                         response_data: Optional[Dict] = None):
                self.error_code = error_code
                self.message = message
                self.error_type = error_type
                self.response_data = response_data or {}
                super().__init__(f"[{error_code}] {message}")
        
        def handle_api_response(response):
            if response.status_code == 200:
                data = response.json()
                if data.get('code') != '00000':
                    raise BitgetAPIError(
                        error_code=data.get('code', 'UNKNOWN'),
                        message=data.get('msg', 'Unknown error'),
                        error_type=get_error_type(data.get('code')),
                        response_data=data
                    )
                return data['data']
            else:
                raise BitgetAPIError(
                    error_code=str(response.status_code),
                    message=f"HTTP {response.status_code}: {response.text}",
                    error_type=BitgetErrorType.NETWORK_ERROR,
                    response_data={'status_code': response.status_code, 'text': response.text}
                )
        ```

- **Bitget API Error Codes & Solutions**
    - **Authentication Errors (40xxx):**
        ```typescript
        const authErrors = {
            '40001': {
                message: 'Invalid API Key',
                solution: 'Verify API key is correct and has proper permissions',
                code: 'INVALID_API_KEY'
            },
            '40002': {
                message: 'Invalid signature',
                solution: 'Check signature generation algorithm and secret key',
                code: 'INVALID_SIGNATURE'
            },
            '40003': {
                message: 'Invalid timestamp',
                solution: 'Ensure timestamp is within 30 seconds of server time',
                code: 'INVALID_TIMESTAMP'
            },
            '40004': {
                message: 'Invalid passphrase',
                solution: 'Verify passphrase matches the one set during API key creation',
                code: 'INVALID_PASSPHRASE'
            },
            '40005': {
                message: 'IP not whitelisted',
                solution: 'Add your IP address to the API key whitelist',
                code: 'IP_NOT_WHITELISTED'
            },
            '40006': {
                message: 'API key permissions insufficient',
                solution: 'Enable required permissions (read/trade/withdraw) for API key',
                code: 'INSUFFICIENT_PERMISSIONS'
            }
        };
        ```

    - **Trading Errors (50xxx):**
        ```typescript
        const tradingErrors = {
            '50001': {
                message: 'Insufficient balance',
                solution: 'Check account balance and ensure sufficient funds',
                code: 'INSUFFICIENT_BALANCE'
            },
            '50002': {
                message: 'Order size too small',
                solution: 'Increase order size to meet minimum requirements',
                code: 'ORDER_SIZE_TOO_SMALL'
            },
            '50003': {
                message: 'Order size too large',
                solution: 'Reduce order size to within maximum limits',
                code: 'ORDER_SIZE_TOO_LARGE'
            },
            '50004': {
                message: 'Order price deviation too large',
                solution: 'Adjust order price closer to current market price',
                code: 'PRICE_DEVIATION_TOO_LARGE'
            },
            '50005': {
                message: 'Order not found',
                solution: 'Verify order ID exists and belongs to your account',
                code: 'ORDER_NOT_FOUND'
            },
            '50006': {
                message: 'Order already cancelled',
                solution: 'Check order status before attempting to cancel',
                code: 'ORDER_ALREADY_CANCELLED'
            },
            '50007': {
                message: 'Order already filled',
                solution: 'Cannot modify/cancel filled orders',
                code: 'ORDER_ALREADY_FILLED'
            },
            '50008': {
                message: 'Trading pair not available',
                solution: 'Check if trading pair is active and properly formatted',
                code: 'PAIR_NOT_AVAILABLE'
            },
            '50009': {
                message: 'Market closed',
                solution: 'Wait for market to reopen or check trading hours',
                code: 'MARKET_CLOSED'
            },
            '50010': {
                message: 'Position size limit exceeded',
                solution: 'Reduce position size to within allowed limits',
                code: 'POSITION_LIMIT_EXCEEDED'
            },
            '50067': {
                message: 'Order price deviates greatly from index price, opening position will be risky',
                solution: 'Adjust order price closer to index price or use market order',
                code: 'PRICE_DEVIATION_RISK'
            }
        };
        ```

    - **Rate Limiting Errors (429 & 30xxx):**
        ```typescript
        const rateLimitErrors = {
            '429': {
                message: 'Too many requests',
                solution: 'Implement exponential backoff and reduce request frequency',
                code: 'RATE_LIMIT_EXCEEDED'
            },
            '30001': {
                message: 'Request too frequent',
                solution: 'Add delays between requests and implement rate limiting',
                code: 'REQUEST_TOO_FREQUENT'
            },
            '30002': {
                message: 'IP rate limit exceeded',
                solution: 'Reduce request frequency or distribute across multiple IPs',
                code: 'IP_RATE_LIMIT'
            },
            '30003': {
                message: 'UID rate limit exceeded',
                solution: 'Implement per-user rate limiting',
                code: 'UID_RATE_LIMIT'
            }
        };
        ```

    - **WebSocket Errors (30xxx):**
        ```typescript
        const websocketErrors = {
            '30001': {
                message: 'WebSocket connection limit exceeded',
                solution: 'Close unused connections, max 20 per IP',
                code: 'WS_CONNECTION_LIMIT'
            },
            '30002': {
                message: 'Subscription limit exceeded',
                solution: 'Reduce subscriptions, max 240 per connection',
                code: 'WS_SUBSCRIPTION_LIMIT'
            },
            '30003': {
                message: 'Invalid subscription parameters',
                solution: 'Check channel name and parameters format',
                code: 'WS_INVALID_PARAMS'
            },
            '30004': {
                message: 'Authentication failed for private channel',
                solution: 'Verify login credentials and signature',
                code: 'WS_AUTH_FAILED'
            }
        };
        ```

    - **System Errors (10xxx):**
        ```typescript
        const systemErrors = {
            '10001': {
                message: 'System error',
                solution: 'Retry request after delay, contact support if persistent',
                code: 'SYSTEM_ERROR'
            },
            '10002': {
                message: 'System maintenance',
                solution: 'Wait for maintenance to complete, check announcements',
                code: 'SYSTEM_MAINTENANCE'
            },
            '10003': {
                message: 'Request timeout',
                solution: 'Retry request with exponential backoff',
                code: 'REQUEST_TIMEOUT'
            }
        };
        ```

- **Error Recovery Strategies**
    - **Rate Limit Recovery:**
        ```python
        # ✅ DO: Exponential backoff for rate limits
        import asyncio
        import random
        from datetime import datetime, timedelta
        
        class RateLimitHandler:
            def __init__(self):
                self.reset_time = None
                self.retry_count = 0
                self.max_retries = 5
            
            async def handle_rate_limit(self, error_response):
                self.retry_count += 1
                
                if self.retry_count > self.max_retries:
                    raise Exception("Max retries exceeded for rate limit")
                
                # Calculate backoff time
                base_delay = 2 ** self.retry_count
                jitter = random.uniform(0.1, 0.5)
                delay = base_delay + jitter
                
                logger.warning(f"Rate limit hit, waiting {delay:.2f}s (attempt {self.retry_count})")
                await asyncio.sleep(delay)
                
                return True  # Indicate retry should be attempted
        ```

    - **Authentication Error Recovery:**
        ```python
        # ✅ DO: Handle authentication errors
        class AuthHandler:
            def __init__(self):
                self.token_refresh_time = None
                self.server_time_offset = 0
            
            async def handle_auth_error(self, error_code):
                if error_code == '40003':  # Invalid timestamp
                    await self.sync_server_time()
                elif error_code == '40002':  # Invalid signature
                    await self.refresh_signature()
                elif error_code in ['40001', '40004', '40006']:  # Key/permission issues
                    raise Exception(f"Configuration error: {error_code}")
                
            async def sync_server_time(self):
                server_time = await self.get_server_time()
                local_time = int(time.time() * 1000)
                self.server_time_offset = server_time - local_time
                logger.info(f"Synced server time, offset: {self.server_time_offset}ms")
        ```

- **Comprehensive Error Mapping Function**
    ```python
    # ✅ DO: Centralized error mapping
    def get_error_type_and_action(error_code: str) -> tuple:
        error_map = {
            # Authentication errors - usually configuration issues
            '40001': (BitgetErrorType.AUTHENTICATION, 'check_api_key'),
            '40002': (BitgetErrorType.AUTHENTICATION, 'check_signature'),
            '40003': (BitgetErrorType.AUTHENTICATION, 'sync_time'),
            '40004': (BitgetErrorType.AUTHENTICATION, 'check_passphrase'),
            '40005': (BitgetErrorType.AUTHENTICATION, 'whitelist_ip'),
            '40006': (BitgetErrorType.AUTHENTICATION, 'check_permissions'),
            
            # Trading errors - business logic issues
            '50001': (BitgetErrorType.INSUFFICIENT_BALANCE, 'check_balance'),
            '50002': (BitgetErrorType.INVALID_ORDER, 'increase_size'),
            '50003': (BitgetErrorType.INVALID_ORDER, 'decrease_size'),
            '50004': (BitgetErrorType.INVALID_ORDER, 'adjust_price'),
            '50067': (BitgetErrorType.INVALID_ORDER, 'adjust_price_to_index'),
            
            # Rate limiting - throttling issues
            '429': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            '30001': (BitgetErrorType.RATE_LIMIT, 'exponential_backoff'),
            '30002': (BitgetErrorType.RATE_LIMIT, 'reduce_frequency'),
            
            # System errors - server issues
            '10001': (BitgetErrorType.SERVER_ERROR, 'retry_with_delay'),
            '10002': (BitgetErrorType.SERVER_ERROR, 'wait_maintenance'),
            '10003': (BitgetErrorType.NETWORK_ERROR, 'retry_with_backoff'),
        }
        
        return error_map.get(error_code, (BitgetErrorType.VALIDATION_ERROR, 'manual_review'))
    ```

- **Version Control Requirements**
    - **API Configuration Management:**
        ```yaml
        # bitget_config.yml - Version controlled configuration
        api_version: "v1.2.3"
        endpoints:
          base_url: "https://api.bitget.com"
          websocket_url: "wss://ws.bitget.com/spot/v1/stream"
        
        rate_limits:
          public_endpoints: 20  # requests per second
          private_endpoints: 10
          trading_endpoints: 10
          wallet_endpoints: 5
        
        retry_config:
          max_retries: 3
          base_delay: 1.0
          max_delay: 30.0
          exponential_base: 2
        
        error_handling:
          log_all_errors: true
          alert_on_auth_errors: true
          auto_retry_rate_limits: true
        ```

    - **Change Log Management:**
        ```markdown
        # CHANGELOG.md - Track all API changes
        
        ## [1.2.3] - 2024-01-15
        ### Added
        - New error handling for price deviation (50067)
        - Enhanced rate limiting with per-endpoint tracking
        ### Changed  
        - Updated signature generation for RSA support
        - Improved WebSocket reconnection logic
        ### Fixed
        - Fixed timestamp synchronization issues
        - Resolved batch order rate limit calculation
        ```

- **Logging & Monitoring Standards**
    - **Structured Logging Format:**
        ```python
        # ✅ DO: Comprehensive API logging
        import structlog
        
        logger = structlog.get_logger()
        
        def log_api_call(endpoint, method, request_data, response_data, duration, error=None):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'endpoint': endpoint,
                'method': method,
                'request_size': len(str(request_data)) if request_data else 0,
                'response_size': len(str(response_data)) if response_data else 0,
                'duration_ms': duration * 1000,
                'success': error is None
            }
            
            if error:
                log_entry.update({
                    'error_code': getattr(error, 'error_code', 'UNKNOWN'),
                    'error_message': str(error),
                    'error_type': getattr(error, 'error_type', 'UNKNOWN')
                })
                logger.error("API call failed", **log_entry)
            else:
                logger.info("API call successful", **log_entry)
        ```

    - **Error Metrics & Alerting:**
        ```python
        # ✅ DO: Track error metrics
        from collections import defaultdict, Counter
        from datetime import datetime, timedelta
        
        class ErrorMetrics:
            def __init__(self):
                self.error_counts = Counter()
                self.error_history = defaultdict(list)
                self.alert_thresholds = {
                    'auth_errors': 5,      # 5 auth errors in 5 minutes
                    'rate_limits': 10,     # 10 rate limits in 1 minute  
                    'server_errors': 20    # 20 server errors in 5 minutes
                }
            
            def record_error(self, error_type, error_code):
                now = datetime.utcnow()
                self.error_counts[error_type] += 1
                self.error_history[error_type].append(now)
                
                # Clean old entries
                cutoff = now - timedelta(minutes=5)
                self.error_history[error_type] = [
                    t for t in self.error_history[error_type] if t > cutoff
                ]
                
                # Check alert thresholds
                recent_count = len(self.error_history[error_type])
                threshold = self.alert_thresholds.get(error_type.value, 50)
                
                if recent_count >= threshold:
                    self.send_alert(error_type, recent_count, threshold)
            
            def send_alert(self, error_type, count, threshold):
                logger.critical(
                    f"Error threshold exceeded: {error_type.value}",
                    count=count,
                    threshold=threshold,
                    action_required=True
                )
        ```

- **Recovery Procedures**
    - **Automated Recovery Actions:**
        ```python
        # ✅ DO: Implement automated recovery
        class RecoveryManager:
            def __init__(self):
                self.recovery_actions = {
                    'sync_time': self.sync_server_time,
                    'check_balance': self.validate_balance,
                    'adjust_price': self.get_market_price_suggestion,
                    'exponential_backoff': self.apply_backoff,
                    'reduce_frequency': self.reduce_request_frequency,
                    'retry_with_delay': self.schedule_retry
                }
            
            async def execute_recovery(self, action_name, context):
                if action_name in self.recovery_actions:
                    return await self.recovery_actions[action_name](mdc:context)
                else:
                    logger.warning(f"Unknown recovery action: {action_name}")
                    return False
        ```

- **Testing Error Scenarios**
    - **Error Simulation for Testing:**
        ```python
        # ✅ DO: Test error handling paths
        class ErrorSimulator:
            def __init__(self):
                self.error_scenarios = {
                    'rate_limit': {'code': '429', 'trigger_after': 10},
                    'auth_failure': {'code': '40001', 'trigger_probability': 0.1},
                    'insufficient_balance': {'code': '50001', 'balance_threshold': 100},
                    'network_timeout': {'code': '10003', 'delay_threshold': 5000}
                }
            
            def should_simulate_error(self, scenario_name, context):
                scenario = self.error_scenarios.get(scenario_name)
                if not scenario:
                    return False
                
                # Implement scenario-specific logic
                if scenario_name == 'rate_limit':
                    return context.get('request_count', 0) > scenario['trigger_after']
                elif scenario_name == 'auth_failure':
                    return random.random() < scenario['trigger_probability']
                
                return False
        ```

- **❌ DON'T: Error Handling Anti-Patterns**
    - Don't ignore error codes - always handle them appropriately
    - Don't retry authentication errors without fixing the underlying issue
    - Don't implement infinite retry loops
    - Don't log sensitive information (API keys, signatures)
    - Don't catch all exceptions with generic handlers
    - Don't continue trading operations after persistent errors
    - Don't hardcode error codes - use constants or enums

    - Don't forget to implement circuit breakers for cascading failures