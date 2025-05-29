#!/usr/bin/env python3
"""
🚀 LIVE TRADING DASHBOARD 🚀
Real-time visual trading interface with advanced UI/UX
Inspired by modern trading platforms
"""

import asyncio
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import json
import requests
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import our analyzer
from super_z_pullback_analyzer import SuperZPullbackAnalyzer

class LiveTradingDashboard:
    def __init__(self):
        self.analyzer = SuperZPullbackAnalyzer()
        self.root = tk.Tk()
        self.root.title("🚀 SUPREME LIVE TRADING DASHBOARD 🚀")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0a0a')  # Dark theme
        
        # Data queues for real-time updates
        self.price_data = deque(maxlen=200)
        self.signals_queue = queue.Queue()
        self.trades_queue = queue.Queue()
        self.performance_data = {'pnl': 0, 'total_trades': 0, 'win_rate': 0}
        
        # Color scheme
        self.colors = {
            'bg': '#0a0a0a',
            'panel': '#1a1a1a', 
            'green': '#00ff88',
            'red': '#ff4466',
            'blue': '#4488ff',
            'yellow': '#ffaa00',
            'text': '#ffffff',
            'accent': '#ff6600'
        }
        
        self.setup_ui()
        self.start_live_data_feed()
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top status bar
        self.create_status_bar(main_frame)
        
        # Create three main panels
        panels_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        panels_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left panel - Chart and signals
        left_panel = tk.Frame(panels_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,2))
        
        # Right panel - Controls and data
        right_panel = tk.Frame(panels_frame, bg=self.colors['panel'], relief=tk.RAISED, bd=2, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(2,0))
        right_panel.pack_propagate(False)
        
        # Setup individual panels
        self.setup_chart_panel(left_panel)
        self.setup_control_panel(right_panel)
        
    def create_status_bar(self, parent):
        """Create top status bar with key metrics"""
        status_frame = tk.Frame(parent, bg=self.colors['panel'], height=60)
        status_frame.pack(fill=tk.X, pady=(0,5))
        status_frame.pack_propagate(False)
        
        # Connection status
        self.status_label = tk.Label(status_frame, text="🔴 DISCONNECTED", 
                                   bg=self.colors['panel'], fg=self.colors['red'],
                                   font=('Arial', 12, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=15)
        
        # Live PnL
        self.pnl_label = tk.Label(status_frame, text="P&L: $0.00", 
                                bg=self.colors['panel'], fg=self.colors['green'],
                                font=('Arial', 14, 'bold'))
        self.pnl_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Trade count
        self.trades_label = tk.Label(status_frame, text="Trades: 0", 
                                   bg=self.colors['panel'], fg=self.colors['text'],
                                   font=('Arial', 12, 'bold'))
        self.trades_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Win rate
        self.winrate_label = tk.Label(status_frame, text="Win Rate: 0%", 
                                    bg=self.colors['panel'], fg=self.colors['yellow'],
                                    font=('Arial', 12, 'bold'))
        self.winrate_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Current time
        self.time_label = tk.Label(status_frame, text="", 
                                 bg=self.colors['panel'], fg=self.colors['text'],
                                 font=('Arial', 10))
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=15)
        
    def setup_chart_panel(self, parent):
        """Setup main chart with price action and signals"""
        chart_frame = tk.Frame(parent, bg=self.colors['panel'])
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chart title
        title_label = tk.Label(chart_frame, text="🚀 LIVE PRICE CHART & SIGNALS 🚀", 
                             bg=self.colors['panel'], fg=self.colors['accent'],
                             font=('Arial', 16, 'bold'))
        title_label.pack(pady=5)
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                                      facecolor=self.colors['bg'],
                                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Style the charts
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#111111')
            ax.tick_params(colors=self.colors['text'])
            ax.spines['bottom'].set_color(self.colors['text'])
            ax.spines['top'].set_color(self.colors['text'])
            ax.spines['left'].set_color(self.colors['text'])
            ax.spines['right'].set_color(self.colors['text'])
        
        # Canvas for chart
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty chart
        self.update_chart([])
        
    def setup_control_panel(self, parent):
        """Setup control panel with settings and live data"""
        # Control buttons frame
        controls_frame = tk.Frame(parent, bg=self.colors['panel'])
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(controls_frame, text="🎛️ CONTROLS", 
                bg=self.colors['panel'], fg=self.colors['accent'],
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Start/Stop buttons
        button_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = tk.Button(button_frame, text="🚀 START TRADING", 
                                  bg=self.colors['green'], fg='black',
                                  font=('Arial', 10, 'bold'),
                                  command=self.start_trading)
        self.start_btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ STOP", 
                                 bg=self.colors['red'], fg='white',
                                 font=('Arial', 10, 'bold'),
                                 command=self.stop_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=2, fill=tk.X, expand=True)
        
        # Symbol selection
        symbol_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        symbol_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(symbol_frame, text="Symbol:", bg=self.colors['panel'], 
                fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.symbol_var = tk.StringVar(value="BTC/USDT:USDT")
        symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.symbol_var,
                                   values=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])
        symbol_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5,0))
        
        # Score threshold
        threshold_frame = tk.Frame(controls_frame, bg=self.colors['panel'])
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame, text="Score Threshold:", bg=self.colors['panel'], 
                fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.threshold_var = tk.StringVar(value="45")
        threshold_entry = tk.Entry(threshold_frame, textvariable=self.threshold_var, width=8)
        threshold_entry.pack(side=tk.RIGHT)
        
        # Live signals display
        signals_frame = tk.Frame(parent, bg=self.colors['panel'])
        signals_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(signals_frame, text="🎯 LIVE SIGNALS", 
                bg=self.colors['panel'], fg=self.colors['accent'],
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        # Signals listbox with scrollbar
        signals_list_frame = tk.Frame(signals_frame, bg=self.colors['panel'])
        signals_list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(signals_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.signals_listbox = tk.Listbox(signals_list_frame, bg='#1a1a1a', 
                                         fg=self.colors['text'],
                                         yscrollcommand=scrollbar.set,
                                         font=('Courier', 9))
        self.signals_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.signals_listbox.yview)
        
        # Performance metrics
        perf_frame = tk.Frame(parent, bg=self.colors['panel'])
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(perf_frame, text="📊 PERFORMANCE", 
                bg=self.colors['panel'], fg=self.colors['accent'],
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        self.perf_text = tk.Text(perf_frame, height=8, bg='#1a1a1a', 
                               fg=self.colors['text'], font=('Courier', 9))
        self.perf_text.pack(fill=tk.X, pady=5)
        
    def start_live_data_feed(self):
        """Start background thread for live data"""
        self.running = True
        self.data_thread = threading.Thread(target=self.data_feed_worker, daemon=True)
        self.data_thread.start()
        
        # Start UI update timer
        self.update_ui()
        
    def data_feed_worker(self):
        """Background worker for fetching live data"""
        while self.running:
            try:
                # Simulate live price data (replace with real API calls)
                current_time = datetime.now()
                symbol = self.symbol_var.get()
                
                # Mock price data
                if len(self.price_data) == 0:
                    price = 50000  # Starting price
                else:
                    last_price = self.price_data[-1]['price']
                    price = last_price + np.random.normal(0, 100)  # Random walk
                
                self.price_data.append({
                    'time': current_time,
                    'price': price,
                    'symbol': symbol
                })
                
                # Simulate signals (replace with real signal detection)
                if np.random.random() < 0.05:  # 5% chance of signal
                    signal_type = 'BUY' if np.random.random() > 0.5 else 'SELL'
                    score = np.random.randint(30, 100)
                    
                    self.signals_queue.put({
                        'time': current_time,
                        'symbol': symbol,
                        'type': signal_type,
                        'price': price,
                        'score': score
                    })
                
                asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"Data feed error: {e}")
                asyncio.sleep(5)
                
    def update_chart(self, price_data):
        """Update the live chart"""
        self.ax1.clear()
        self.ax2.clear()
        
        if len(price_data) > 0:
            times = [d['time'] for d in price_data]
            prices = [d['price'] for d in price_data]
            
            # Price chart
            self.ax1.plot(times, prices, color=self.colors['blue'], linewidth=2)
            self.ax1.set_title(f"{self.symbol_var.get()} - Live Price", 
                             color=self.colors['text'], fontsize=14)
            self.ax1.set_ylabel("Price", color=self.colors['text'])
            self.ax1.grid(True, alpha=0.3)
            
            # Volume chart (simulated)
            volumes = [abs(np.random.normal(1000, 200)) for _ in times]
            self.ax2.bar(times, volumes, color=self.colors['yellow'], alpha=0.7, width=0.0001)
            self.ax2.set_title("Volume", color=self.colors['text'])
            self.ax2.set_ylabel("Volume", color=self.colors['text'])
            
        self.canvas.draw()
        
    def update_ui(self):
        """Update UI elements with latest data"""
        try:
            # Update time
            self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Update chart if we have data
            if len(self.price_data) > 0:
                self.update_chart(list(self.price_data))
            
            # Process new signals
            while not self.signals_queue.empty():
                try:
                    signal = self.signals_queue.get_nowait()
                    self.add_signal_to_display(signal)
                except queue.Empty:
                    break
            
            # Update performance metrics
            self.update_performance_display()
            
        except Exception as e:
            print(f"UI update error: {e}")
        
        # Schedule next update
        self.root.after(1000, self.update_ui)  # Update every second
        
    def add_signal_to_display(self, signal):
        """Add signal to the signals display"""
        signal_text = f"{signal['time'].strftime('%H:%M:%S')} | {signal['symbol']} | {signal['type']} | ${signal['price']:.2f} | Score: {signal['score']}"
        
        self.signals_listbox.insert(0, signal_text)
        
        # Keep only last 50 signals
        if self.signals_listbox.size() > 50:
            self.signals_listbox.delete(50, tk.END)
        
        # Color code signals
        if signal['type'] == 'BUY':
            self.signals_listbox.itemconfig(0, fg=self.colors['green'])
        else:
            self.signals_listbox.itemconfig(0, fg=self.colors['red'])
            
    def update_performance_display(self):
        """Update performance metrics display"""
        perf_text = f"""
🏆 TRADING PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━
💰 Total P&L: ${self.performance_data['pnl']:.2f}
📈 Total Trades: {self.performance_data['total_trades']}
🎯 Win Rate: {self.performance_data['win_rate']:.1f}%
⚡ Status: {'LIVE TRADING' if hasattr(self, 'trading_active') and self.trading_active else 'STOPPED'}
🔄 Signals Today: {self.signals_listbox.size()}
⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, perf_text)
        
    def start_trading(self):
        """Start live trading"""
        self.trading_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="🟢 LIVE TRADING", fg=self.colors['green'])
        
        # Show confirmation dialog
        messagebox.showinfo("🚀 Trading Started", 
                          f"Live trading activated!\nSymbol: {self.symbol_var.get()}\nThreshold: {self.threshold_var.get()}")
        
    def stop_trading(self):
        """Stop live trading"""
        self.trading_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="🔴 STOPPED", fg=self.colors['red'])
        
        messagebox.showinfo("⏹️ Trading Stopped", "Live trading has been stopped.")
        
    def run(self):
        """Start the dashboard"""
        print("🚀 Starting Live Trading Dashboard...")
        self.root.mainloop()

def main():
    """Launch the dashboard"""
    dashboard = LiveTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 