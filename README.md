# Trading Bot Dashboard with Backtesting

A comprehensive trading bot dashboard with backtesting capabilities, built with FastAPI and modern web technologies.

## Features

- Real-time trading dashboard
- Backtesting engine with optimization
- Technical analysis indicators
- Risk management system
- Webhook integration with TradingView
- Dark/light theme support
- Responsive design

## New Enhanced Dashboard (10000x)

We now have a completely optimized trading dashboard with:

- **Robust WebSocket Connection**: Automatically recovers from disconnections
- **Advanced UI**: Modern, cyberpunk-inspired interface with animations
- **Performance Optimizations**: Hardware acceleration and efficient rendering
- **Enhanced Error Handling**: Comprehensive error recovery and user feedback

[See the full Dashboard documentation](./README_DASHBOARD.md)

## Prerequisites

- Python 3.8+
- ngrok account (for webhook functionality)
- TradingView account (for alerts)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot-dashboard.git
cd trading-bot-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
NGROK_AUTH_TOKEN=your_ngrok_auth_token
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_API_SECRET=your_exchange_api_secret
TESTNET=true
```

## Usage

1. Start the server:
```bash
python start_server.py
```

2. Access the dashboard:
- Local: http://localhost:8000
- Public: Use the ngrok URL displayed in the terminal

3. Configure TradingView alerts:
- Use the webhook URL displayed in the dashboard
- Set up alerts in TradingView with the following format:
```json
{
    "symbol": "BTC/USDT",
    "action": "buy",
    "price": 50000,
    "time": "2024-03-20T12:00:00Z"
}
```

## Backtesting

The dashboard includes a powerful backtesting engine with the following features:

- Multiple timeframe support
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Risk management (position sizing, stop loss, take profit)
- Parameter optimization using Optuna
- Performance metrics (Sharpe ratio, max drawdown, win rate)

To run a backtest:

1. Navigate to the Backtesting section
2. Configure your strategy parameters
3. Select the time range and timeframe
4. Click "Run Backtest" or "Optimize Parameters"

## Risk Management

The system includes comprehensive risk management features:

- Position sizing based on account balance
- Multiple take profit levels
- Stop loss protection
- Maximum drawdown limits
- Maximum open trades limit

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. # FInal_v1
# BACKTESTER-
