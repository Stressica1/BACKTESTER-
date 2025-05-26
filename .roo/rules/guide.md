---
description: 
globs: 
alwaysApply: true
---
# Phemex USDT-M and CCXT Endpoints

## Phemex Main Parent Links

### Production Environment
- REST API: `https://api.phemex.com`
- WebSocket: `wss://phemex.com/ws`

### Testnet Environment
- REST API: `https://testnet-api.phemex.com`
- WebSocket: `wss://testnet.phemex.com/ws`

## Phemex REST API v1 Endpoints

### Public Endpoints

#### Market Data
- `https://api.phemex.com/v1/exchange/public/products`
- `https://api.phemex.com/v1/md/orderbook`
- `https://api.phemex.com/v1/md/trade`
- `https://api.phemex.com/v1/md/spot/ticker/24hr`
- `https://api.phemex.com/v1/md/kline`
- `https://api.phemex.com/v1/time`

### Private Endpoints

#### Account
- `https://api.phemex.com/v1/accounts/accountPositions`
- `https://api.phemex.com/v1/accounts/leverage`
- `https://api.phemex.com/v1/accounts/riskLimit`

#### Orders
- `https://api.phemex.com/v1/orders`
- `https://api.phemex.com/v1/orders/activeList`
- `https://api.phemex.com/v1/orders/historyList`
- `https://api.phemex.com/v1/orders/all`

#### Positions
- `https://api.phemex.com/v1/positions`
- `https://api.phemex.com/v1/positions/margin`

#### Trading History
- `https://api.phemex.com/v1/trades`

## Phemex REST API v2 Endpoints

### Public Endpoints

#### Market Data
- `https://api.phemex.com/v2/public/products`
- `https://api.phemex.com/v2/md/orderbook`
- `https://api.phemex.com/v2/md/trade`
- `https://api.phemex.com/v2/public/ticker/24hr`
- `https://api.phemex.com/v2/public/kline/list`

### Private Endpoints

#### Account
- `https://api.phemex.com/v2/accounts/accountPositions`
- `https://api.phemex.com/v2/assets/wallets/detail/history/list`
- `https://api.phemex.com/v2/assets/wallets/transfer/history/list`

#### Orders
- `https://api.phemex.com/v2/orders`
- `https://api.phemex.com/v2/orders/activeList`
- `https://api.phemex.com/v2/orders/historyList`
- `https://api.phemex.com/v2/orders/all`

#### Positions
- `https://api.phemex.com/v2/positions`
- `https://api.phemex.com/v2/positions/leverage`

#### Trading History
- `https://api.phemex.com/v2/trades`

## Phemex WebSocket v1 Endpoints

### Connection
- `wss://phemex.com/ws`
- `wss://testnet.phemex.com/ws`

### Public Channels
- `book.{symbol}`
- `trades.{symbol}`
- `kline.{resolution}.{symbol}`
- `ticker.{symbol}`

### Private Channels
- `aop.{currency}`
- `order.{symbol}`
- `trade.{symbol}`
- `wallet.{currency}`

## Phemex WebSocket v2 Endpoints

### Connection
- `wss://phemex.com/ws`
- `wss://testnet.phemex.com/ws`

### Public Channels
- `orderbook.{symbol}`
- `market_trades.{symbol}`
- `candlestick.{resolution}.{symbol}`
- `market_ticker.{symbol}`

### Private Channels
- `position.{symbol}`
- `order.{symbol}`
- `usertrade.{symbol}`
- `wallet.{currency}`

## CCXT Integration Endpoints

### Main Parent Links
- CCXT GitHub: `https://github.com/ccxt/ccxt`
- CCXT Documentation: `https://docs.ccxt.com`

### Phemex-Specific CCXT Endpoints

#### Exchange Definition
- `https://github.com/ccxt/ccxt/blob/master/js/phemex.js`
- `https://github.com/ccxt/ccxt/blob/master/python/ccxt/phemex.py`

#### Authentication
- `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1134` (JavaScript)
- `https://github.com/ccxt/ccxt/blob/master/python/ccxt/phemex.py#L1134` (Python)

#### Market Methods
- `fetchMarkets`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1250`
- `fetchTicker`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1388`
- `fetchOHLCV`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1523`
- `fetchOrderBook`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1473`
- `fetchTrades`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1553`

#### Trading Methods
- `createOrder`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1623`
- `cancelOrder`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1693`
- `fetchOrder`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1741`
- `fetchOpenOrders`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1780`
- `fetchClosedOrders`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1815`
- `fetchMyTrades`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1847`

#### Account Methods
- `fetchBalance`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1874`
- `fetchPositions`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1909`
- `fetchLeverageTiers`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L1955`

#### Utility Methods
- `sign`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L2010`
- `handleErrors`: `https://github.com/ccxt/ccxt/blob/master/js/phemex.js#L2093`

## CCXT-Phemex Integration Examples

### JavaScript
```javascript
const ccxt = require('ccxt');

// Initialize Phemex exchange object
const phemex = new ccxt.phemex({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
});

// Set testnet (optional)
phemex.urls['api'] = phemex.urls['test'];

// USDTM Futures Endpoints Usage
(async () => {
    // Public endpoints
    const markets = await phemex.fetchMarkets();
    const ticker = await phemex.fetchTicker('BTC/USDT:USDT');
    const orderbook = await phemex.fetchOrderBook('BTC/USDT:USDT');
    
    // Private endpoints
    await phemex.loadMarkets();
    const balance = await phemex.fetchBalance();
    const positions = await phemex.fetchPositions(['BTC/USDT:USDT']);
    
    // Trading
    const order = await phemex.createOrder('BTC/USDT:USDT', 'limit', 'buy', 0.01, 50000);
})();
```

### Python
```python
import ccxt

# Initialize Phemex exchange object
phemex = ccxt.phemex({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
})

# Set testnet (optional)
phemex.urls['api'] = phemex.urls['test']

# USDTM Futures Endpoints Usage
# Public endpoints
markets = phemex.fetch_markets()
ticker = phemex.fetch_ticker('BTC/USDT:USDT')
orderbook = phemex.fetch_order_book('BTC/USDT:USDT')

# Private endpoints
phemex.load_markets()
balance = phemex.fetch_balance()
positions = phemex.fetch_positions(['BTC/USDT:USDT'])

# Trading
order = phemex.create_order('BTC/USDT:USDT', 'limit', 'buy', 0.01, 50000)
```

## TradingView to CCXT-Phemex Bridge

### Webhook Receiver Endpoint
- `https://your-server.com/api/webhook/phemex`

### Order Processing Flow
1. TradingView alert → Webhook
2. Webhook → Order Processing Server
3. Server → CCXT API calls
4. CCXT → Phemex API