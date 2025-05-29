# SuperTrend Pullback Trading Bot

High-performance algorithmic trading bot using SuperTrend and pullback strategies for Bitget exchange.

## 🚀 Features

- **Advanced SuperTrend Strategy**: Optimized parameters for maximum profitability
- **Aggressive Pullback Detection**: Captures market reversals and momentum
- **Dynamic Leverage**: Automatically adjusts leverage based on signal strength
- **Risk Management**: Stop losses, take profits, and trailing stops
- **Bitget API Integration**: Full compliance with rate limits and error handling
- **Simulation Mode**: Test strategies without risking real money
- **Multi-Symbol Trading**: Automatically selects most volatile trading pairs

## 📋 Requirements

- Python 3.8+
- Bitget account with API access
- Any amount of USDT for live trading (no minimum balance required)

## 🛠 Installation

1. **Clone or download the repository**
```bash
git clone <repository_url>
cd BACKTESTER
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the bot**
```bash
python supertrend_pullback_live.py
```

## ⚙️ Configuration

### First Time Setup

1. Run the bot and select option **3** (Configure Bitget API)
2. Enter your Bitget API credentials:
   - API Key
   - Secret Key  
   - Passphrase
   - Sandbox mode (recommended for testing)

### Getting Bitget API Credentials

1. Login to [Bitget](https://www.bitget.com)
2. Go to **API Management**
3. Create new API key with **Trading** permissions
4. **Important**: Whitelist your IP address
5. Save your credentials securely

## 🎮 Usage

### Simulation Mode (Recommended First)

```bash
python supertrend_pullback_live.py
# Select option 1: Run Simulation
# Choose number of days to simulate
```

### Live Trading Mode

```bash
python supertrend_pullback_live.py
# Select option 2: Live Trading
# Type 'CONFIRM' to proceed with real money
```

## ⚠️ Risk Warning

**This bot trades with real money in live mode. Cryptocurrency trading involves significant risk of loss. Only trade with money you can afford to lose.**

## 📊 Strategy Parameters

- **Position Size**: 0.50 USDT margin per trade (with leverage applied)
- **Leverage System**: Dynamic leverage 20x-50x applied FIRST
- **Effective Position**: 0.50 × leverage (e.g., 0.50 × 50x = 25 USDT effective)
- **Max Positions**: 50 concurrent trades
- **SuperTrend Period**: 10 (balanced reaction)
- **SuperTrend Multiplier**: 3.0 (balanced bands)
- **Stop Loss**: 1% (tight control)
- **Take Profits**: 0.8%, 1.5%, 2.5%

## 📈 Performance Features

- **Leverage-First Execution**: Sets leverage before position calculation
- **Ultra-High Win Rate**: 85%+ win rate targeting system
- **Smart Position Sizing**: 0.50 USDT margin × dynamic leverage
- **Multi-threading**: Parallel processing for faster execution
- **Rate Limiting**: Full Bitget API compliance
- **Error Recovery**: Automatic retry with exponential backoff
- **Price Deviation Protection**: Handles Bitget error 50067
- **Real-time Logging**: Comprehensive trade and error logs

## 📝 Logs

All trading activity is logged to:
- `logs/supertrend_pullback.log` - Main trading log
- `logs/simulation_trades.csv` - Detailed simulation results

## 🔧 Customization

Edit parameters in the `AggressivePullbackTrader` class:

```python
# Trading parameters
self.FIXED_POSITION_SIZE_USDT = 0.50  # 0.50 USDT margin per trade
self.max_positions = 50
self.pullback_threshold = 0.002  # 0.2% pullback
self.stop_loss_pct = 0.01  # 1% stop loss

# SuperTrend parameters  
self.st_period = 10
self.st_multiplier = 3.0

# Leverage system (applied FIRST)
self.min_leverage = 20
self.max_leverage = 50
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Errors**: Check API credentials and IP whitelist
2. **Rate Limit Errors**: Bot handles these automatically with backoff
3. **Price Deviation Error 50067**: Bot gets current market price automatically
4. **Position Size Issues**: Leverage is set FIRST, then 0.50 USDT margin applied

### Error Codes

- `50067`: Price deviation (handled automatically)
- `40001-40006`: Authentication issues
- `50001`: Insufficient balance
- `429/30001/30002`: Rate limiting

## 📞 Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Verify API configuration
3. Check Bitget API status

## ⚖️ Disclaimer

This software is for educational purposes. The authors are not responsible for any financial losses. Always test in simulation mode first and never risk more than you can afford to lose.

## 🔄 Updates

- **LEVERAGE-FIRST EXECUTION**: Sets leverage before calculating position size
- Enhanced error handling for all Bitget API responses
- **Ultra-High Win Rate System**: 85%+ win rate targeting
- **Smart Position Sizing**: 0.50 USDT margin × dynamic leverage
- Improved signal generation with multi-timeframe analysis
- Real-time performance monitoring
- **200+ Trading Pairs**: Dynamic pair discovery system

---

**BUSSIED!!!!** - Your trading bot is now ready for action! 🚀
