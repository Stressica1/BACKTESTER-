import asyncio
from supertrend_pullback_live import AggressivePullbackTrader

async def main():
    print("🚀 STARTING LIVE TRADING BOT")
    print("=" * 50)
    print("⚠️  LIVE TRADING MODE - REAL MONEY")
    print("=" * 50)
    
    # Create bot instance in live mode
    bot = AggressivePullbackTrader(simulation_mode=False)
    
    # Run the bot
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}") 