---
description: Bitget USDT-M Futures API endpoints and CCXT integration guide
globs: 
alwaysApply: true
---
# Bitget USDT-M Futures and CCXT Endpoints

## Bitget Main Parent Links

### Production Environment
- REST API: `https://api.bitget.com`
- WebSocket: `wss://stream.bitget.com`

### Testnet Environment
- REST API: `https://api.bitgetapi.com` (testnet)
- WebSocket: `wss://testnet-stream.bitget.com`

## Bitget REST API v1 Endpoints

### Public Endpoints

#### Market Data
- `https://api.bitget.com/api/mix/v1/market/contracts`
- `https://api.bitget.com/api/mix/v1/market/depth`
- `https://api.bitget.com/api/mix/v1/market/ticker`
- `https://api.bitget.com/api/mix/v1/market/tickers`
- `https://api.bitget.com/api/mix/v1/market/candles`
- `https://api.bitget.com/api/mix/v1/market/fills`

### Private Endpoints

#### Account
- `https://api.bitget.com/api/mix/v1/account/account`
- `https://api.bitget.com/api/mix/v1/account/accounts`
- `https://api.bitget.com/api/mix/v1/position/allPosition`
- `https://api.bitget.com/api/mix/v1/position/singlePosition`

#### Orders
- `https://api.bitget.com/api/mix/v1/order/placeOrder`
- `https://api.bitget.com/api/mix/v1/order/current`
- `https://api.bitget.com/api/mix/v1/order/history`
- `https://api.bitget.com/api/mix/v1/order/fills`
- `https://api.bitget.com/api/mix/v1/order/cancel-order`

#### Risk Management
- `https://api.bitget.com/api/mix/v1/account/setLeverage`
- `https://api.bitget.com/api/mix/v1/account/setMargin`
- `https://api.bitget.com/api/mix/v1/account/setMarginMode`

## Bitget WebSocket Endpoints

### Connection
- `wss://stream.bitget.com/mix/v1/stream`
- `wss://testnet-stream.bitget.com/mix/v1/stream` (testnet)

### Public Channels
- `ticker` - Real-time ticker data
- `depth` - Order book updates
- `candle15m` - 15-minute candlestick data
- `candle1H` - 1-hour candlestick data
- `trade` - Recent trades

### Private Channels
- `account` - Account balance updates
- `positions` - Position updates
- `orders` - Order status updates

## CCXT Integration Endpoints

### Main Parent Links
- CCXT GitHub: `https://github.com/ccxt/ccxt`
- CCXT Documentation: `https://docs.ccxt.com`

### Bitget-Specific CCXT Endpoints

#### Exchange Definition
- `https://github.com/ccxt/ccxt/blob/master/js/bitget.js`
- `https://github.com/ccxt/ccxt/blob/master/python/ccxt/bitget.py`

#### Market Methods
- `fetchMarkets` - Get all available trading pairs
- `fetchTicker` - Get 24hr ticker statistics
- `fetchOHLCV` - Get candlestick/kline data
- `fetchOrderBook` - Get order book depth
- `fetchTrades` - Get recent trades

#### Trading Methods
- `createOrder` - Place new order
- `cancelOrder` - Cancel existing order
- `fetchOrder` - Get order status
- `fetchOpenOrders` - Get all open orders
- `fetchClosedOrders` - Get order history
- `fetchMyTrades` - Get trade history

#### Account Methods
- `fetchBalance` - Get account balances
- `fetchPositions` - Get open positions
- `setLeverage` - Set position leverage

## CCXT-Bitget Integration Examples

### JavaScript
```javascript
const ccxt = require('ccxt');

// Initialize Bitget exchange object
const bitget = new ccxt.bitget({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'password': 'YOUR_PASSPHRASE', // Bitget requires passphrase
    'sandbox': true, // Use testnet
});

// USDT-M Futures Endpoints Usage
(async () => {
    // Public endpoints
    const markets = await bitget.fetchMarkets();
    const ticker = await bitget.fetchTicker('BTC/USDT:USDT');
    const orderbook = await bitget.fetchOrderBook('BTC/USDT:USDT');
    const ohlcv = await bitget.fetchOHLCV('BTC/USDT:USDT', '15m');
    
    // Private endpoints
    await bitget.loadMarkets();
    const balance = await bitget.fetchBalance();
    const positions = await bitget.fetchPositions(['BTC/USDT:USDT']);
    
    // Trading
    const order = await bitget.createOrder('BTC/USDT:USDT', 'market', 'buy', 0.01);
})();
```

### Python
```python
import ccxt

# Initialize Bitget exchange object
bitget = ccxt.bitget({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'password': 'YOUR_PASSPHRASE',
    'sandbox': True,  # Use testnet
})

# USDT-M Futures Trading
async def trade_bitget():
    # Public endpoints
    markets = await bitget.fetch_markets()
    ticker = await bitget.fetch_ticker('BTC/USDT:USDT')
    orderbook = await bitget.fetch_order_book('BTC/USDT:USDT')
    
    # Private endpoints
    await bitget.load_markets()
    balance = await bitget.fetch_balance()
    positions = await bitget.fetch_positions(['BTC/USDT:USDT'])
    
    # Trading
    order = await bitget.create_order('BTC/USDT:USDT', 'market', 'buy', 0.01)
```

## TradingView to CCXT-Bitget Bridge

### Webhook Endpoint
- `https://your-server.com/api/webhook/bitget`

### Signal Flow
1. TradingView Alert
2. Webhook POST
3. Signal Processing
4. CCXT → Bitget API

## Bitget USDT-M Symbol Format

### Standard Format
- Spot: `BTC/USDT`
- USDT-M Futures: `BTC/USDT:USDT`
- Coin-M Futures: `BTC/USD:BTC`

### Symbol Examples
- `BTCUSDT_UMCBL` (Bitget native)
- `BTC/USDT:USDT` (CCXT format)

## Rate Limits
- REST API: 20 requests per second
- WebSocket: 240 connections per IP
- Order limits: 100 orders per second

## Error Codes
- `30001`: Order does not exist
- `30002`: Insufficient balance
- `30003`: Order size too small
- `40004`: Invalid symbol
- `43025`: Order would trigger immediately