#!/usr/bin/env python3
"""
SIMPLE SUPERTREND BOT - NO BALANCE CHECK BULLSHIT
Just fucking runs and trades!
"""

from supertrend_pullback_live import AggressivePullbackTrader
import asyncio
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTradingBot(AggressivePullbackTrader):
    """Simplified bot that bypasses balance checks and enforces min 25x leverage"""
    
    async def check_account_balance(self):
        """BYPASS: Always return True - fuck the balance check"""
        logger.info("💰 Balance check BYPASSED - assuming sufficient balance")
        return True, 100.0  # Always return sufficient balance
    
    async def execute_trade(self, signal):
        """FORCED: Execute trade with MINIMUM 25x LEVERAGE, always live trading, and log every real order"""
        execution_start = time.time()
        try:
            symbol = signal['symbol']
            side = signal['side']
            signal['leverage'] = max(25, signal.get('leverage', 50))  # FORCE MINIMUM 25x IN SIGNAL
            leverage = signal['leverage']
            logger.info(f"🔧 SETTING LEVERAGE FIRST: {symbol} -> {leverage}x (MIN 25x ENFORCED)")
            try:
                await self.rate_limit('set_leverage')
                await self.set_leverage(symbol, leverage)
                logger.info(f"✅ Leverage set: {leverage}x for {symbol}")
            except Exception as e:
                logger.warning(f"⚠️ Leverage setting failed for {symbol}: {e}")
                leverage = 25
                signal['leverage'] = 25
            margin_usdt = self.FIXED_POSITION_SIZE_USDT
            effective_position_value = margin_usdt * leverage
            current_price = signal.get('price', 0)
            if current_price <= 0:
                current_price = await self.get_current_market_price(symbol)
                if not current_price:
                    logger.error(f"❌ Cannot get price for {symbol}")
                    return None
            quantity = float(effective_position_value) / float(current_price)
            quantity = self.adjust_quantity_for_precision(symbol, quantity)
            logger.info(f"⚡ EXECUTING TRADE: {symbol} {side.upper()} (REAL ORDER)")
            logger.info(f"   💰 Margin Used: {margin_usdt} USDT")
            logger.info(f"   📈 Leverage: {leverage}x (MIN 25x ENFORCED)")
            logger.info(f"   💵 Effective Position: {effective_position_value} USDT")
            logger.info(f"   📊 Quantity: {quantity} coins")
            logger.info(f"   💲 Price: {current_price}")
            self.validate_position_size(margin_usdt)
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    await self.rate_limit('create_order')
                    order_params = {
                        'timeInForce': 'IOC',
                        'createMarketBuyOrderRequiresPrice': False
                    }
                    if side == 'buy':
                        cost_to_spend = margin_usdt
                        order = self.exchange.create_market_order(
                            symbol=symbol,
                            side=side,
                            amount=cost_to_spend,
                            params=order_params
                        )
                    else:
                        order = self.exchange.create_market_order(
                            symbol=symbol,
                            side=side,
                            amount=quantity,
                            params=order_params
                        )
                    execution_time = (time.time() - execution_start) * 1000
                    if order and order.get('status') in ['closed', 'filled']:
                        filled_price = order.get('price', current_price)
                        filled_quantity = order.get('filled', quantity)
                        actual_cost = order.get('cost', margin_usdt)
                        trade_data = {
                            'timestamp': time.time(),
                            'symbol': symbol,
                            'side': side,
                            'price': filled_price,
                            'margin_usdt': actual_cost,
                            'effective_value_usdt': actual_cost * leverage,
                            'leverage': leverage,
                            'quantity': filled_quantity,
                            'confidence': signal.get('confidence', 0),
                            'execution_time': execution_time,
                            'success': True
                        }
                        self.database.save_trade(trade_data)
                        self.total_trades += 1
                        logger.info(f"✅ REAL TRADE EXECUTED: {symbol} {side.upper()} @ {filled_price} x {filled_quantity} (LEVERAGE: {leverage}x)")
                        return order
                    else:
                        logger.warning(f"⚠️ Order not filled properly: {order}")
                        return None
                except Exception as e:
                    retry_count += 1
                    if await self.handle_bitget_error(e, symbol, retry_count):
                        continue
                    else:
                        break
            logger.error(f"❌ Trade execution failed after {max_retries} attempts")
            return None
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            logger.error(f"❌ Trade execution error in {execution_time:.1f}ms: {e}")
            return None
    
    async def main_trading_loop(self):
        """SIMPLIFIED: Main trading loop without balance validation"""
        logger.info("🚀 STARTING SIMPLIFIED TRADING LOOP")
        logger.info(f"💰 Position Size: {self.FIXED_POSITION_SIZE_USDT} USDT per trade")
        
        # Display configuration
        self.display_trading_pairs_config()
        
        signal_count = 0
        
        try:
            while True:
                await self.main_trading_loop_iteration()
                
                signal_count += 1
                if signal_count % 20 == 0:
                    logger.info(f"🔄 Processed {signal_count} signal cycles")
                
                await asyncio.sleep(3)  # 3-second interval between signals
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
            raise

async def main():
    """Run the simplified bot"""
    logger.info("🚀 SIMPLE SUPERTREND BOT - STARTING NOW!")
    
    # Create bot instance
    bot = SimpleTradingBot(simulation_mode=False)  # LIVE TRADING
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 