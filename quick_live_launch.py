#!/usr/bin/env python3
"""
🚀 QUICK LIVE LAUNCH - GET TRADING NOW!
Simple launcher for live trading bot
"""

import os
import sys
import subprocess

def main():
    print("🚀 SUPERTREND BOT LIVE LAUNCHER")
    print("=" * 50)
    
    print("\n🎯 CHOOSE YOUR OPTION:")
    print("1. 🔥 LIVE TRADING (Real Money)")
    print("2. 📊 SIMULATION MODE (Safe Testing)")
    print("3. 🛠️ FIXED BOT (Clear LONG/SHORT)")
    print("4. ❌ Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🔥 STARTING LIVE TRADING BOT...")
        print("⚠️  WARNING: This uses REAL MONEY!")
        confirm = input("Type 'LIVE' to confirm: ").strip()
        
        if confirm == "LIVE":
            print("🚀 Launching live trading bot...")
            subprocess.run([sys.executable, "supertrend_pullback_live.py"])
        else:
            print("❌ Live trading cancelled.")
            
    elif choice == "2":
        print("\n📊 STARTING SIMULATION MODE...")
        print("✅ Safe testing with fake money")
        subprocess.run([sys.executable, "supertrend_pullback_live.py", "--simulation"])
        
    elif choice == "3":
        print("\n🛠️ STARTING FIXED BOT...")
        print("✅ Crystal clear LONG/SHORT signals")
        subprocess.run([sys.executable, "supertrend_pullback_live_fixed.py"])
        
    elif choice == "4":
        print("\n👋 Goodbye! Trade safely!")
        
    else:
        print("\n❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    print("🎯 Welcome to SuperTrend Live Trading!")
    print("📖 Read LIVE_TRADING_SETUP_GUIDE.md first!")
    print()
    
    main() 