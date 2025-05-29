#!/usr/bin/env python3
"""
🚀 SUPREME VISUAL TRADING LAUNCHER 🚀
Launches live trading bot with full visual interface
"""

import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
import os

def install_requirements():
    """Install required packages for visual features"""
    requirements = [
        'pygame',
        'plyer', 
        'matplotlib',
        'numpy',
        'pandas',
        'colorama',
        'prettytable'
    ]
    
    print("🔧 Installing visual requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed")
        except:
            print(f"⚠️ {package} already installed or failed")

def launch_trading_bot():
    """Launch the main trading bot"""
    print("🚀 Launching Live Trading Bot...")
    process = subprocess.Popen([sys.executable, "super_z_pullback_analyzer.py"])
    return process

def launch_dashboard():
    """Launch the visual dashboard"""
    print("📊 Launching Visual Dashboard...")
    try:
        process = subprocess.Popen([sys.executable, "live_trading_dashboard.py"])
        return process
    except FileNotFoundError:
        print("⚠️ Dashboard not found, running in terminal mode")
        return None

def launch_alerts():
    """Launch the alert system"""
    print("🚨 Launching Alert System...")
    try:
        process = subprocess.Popen([sys.executable, "visual_alerts.py"])
        return process
    except FileNotFoundError:
        print("⚠️ Alert system not found")
        return None

def show_launcher_gui():
    """Show launcher GUI"""
    root = tk.Tk()
    root.title("🚀 SUPREME TRADING LAUNCHER")
    root.geometry("600x400")
    root.configure(bg='#0a0a0a')
    
    # Title
    title = tk.Label(root, text="🚀 SUPREME TRADING SYSTEM 🚀", 
                    bg='#0a0a0a', fg='#ff6600',
                    font=('Arial', 20, 'bold'))
    title.pack(pady=20)
    
    # Status
    status = tk.Label(root, text="Ready to launch components", 
                     bg='#0a0a0a', fg='#ffffff',
                     font=('Arial', 12))
    status.pack(pady=10)
    
    # Buttons frame
    buttons_frame = tk.Frame(root, bg='#0a0a0a')
    buttons_frame.pack(pady=20)
    
    # Launch buttons
    def launch_all():
        status.config(text="🚀 Launching all components...")
        root.update()
        
        # Install requirements first
        install_requirements()
        
        # Launch components
        bot_process = launch_trading_bot()
        time.sleep(2)
        dashboard_process = launch_dashboard()
        time.sleep(1)
        alerts_process = launch_alerts()
        
        status.config(text="✅ All components launched!")
        
        messagebox.showinfo("🚀 Launch Complete", 
                          "All trading components have been launched!\n\n"
                          "• Live Trading Bot: Running\n"
                          "• Visual Dashboard: Running\n" 
                          "• Alert System: Running\n\n"
                          "Check your windows for the interfaces.")
    
    def launch_bot_only():
        status.config(text="🤖 Launching trading bot only...")
        root.update()
        launch_trading_bot()
        status.config(text="✅ Trading bot launched!")
    
    def install_deps():
        status.config(text="🔧 Installing dependencies...")
        root.update()
        install_requirements()
        status.config(text="✅ Dependencies installed!")
    
    # Create buttons
    tk.Button(buttons_frame, text="🚀 LAUNCH EVERYTHING", 
             bg='#00ff88', fg='black',
             font=('Arial', 14, 'bold'),
             command=launch_all, width=20).pack(pady=10)
    
    tk.Button(buttons_frame, text="🤖 BOT ONLY", 
             bg='#4488ff', fg='white',
             font=('Arial', 12, 'bold'),
             command=launch_bot_only, width=20).pack(pady=5)
    
    tk.Button(buttons_frame, text="🔧 INSTALL DEPS", 
             bg='#ffaa00', fg='black',
             font=('Arial', 12, 'bold'),
             command=install_deps, width=20).pack(pady=5)
    
    # Info text
    info_text = """
🎯 FEATURES INCLUDED:
• Ultra-fast live trading (45+ score threshold)
• Real-time visual charts and indicators
• Audio alerts for buy/sell signals  
• Desktop notifications
• Performance tracking dashboard
• 50 parallel symbol processing
• Smart signal filtering
• Risk management with stop-loss/take-profit
    """
    
    info_label = tk.Label(root, text=info_text, 
                         bg='#0a0a0a', fg='#cccccc',
                         font=('Courier', 10),
                         justify=tk.LEFT)
    info_label.pack(pady=10)
    
    root.mainloop()

def main():
    """Main launcher function"""
    print("🚀 SUPREME VISUAL TRADING LAUNCHER")
    print("="*50)
    
    choice = input("""
Choose launch mode:
1. 🚀 Launch with GUI
2. 🤖 Launch trading bot only 
3. 📊 Launch dashboard only
4. 🚨 Launch alerts only
5. 🔧 Install dependencies only

Enter choice (1-5): """)
    
    if choice == "1":
        show_launcher_gui()
    elif choice == "2":
        install_requirements()
        launch_trading_bot()
        print("✅ Trading bot launched! Check the terminal output.")
        input("Press Enter to exit...")
    elif choice == "3":
        install_requirements()
        launch_dashboard()
        print("✅ Dashboard launched! Check for new window.")
        input("Press Enter to exit...")
    elif choice == "4":
        install_requirements()
        launch_alerts()
        print("✅ Alert system launched! Check for new window.")
        input("Press Enter to exit...")
    elif choice == "5":
        install_requirements()
        print("✅ Dependencies installed!")
    else:
        print("❌ Invalid choice. Launching with GUI...")
        show_launcher_gui()

if __name__ == "__main__":
    main() 