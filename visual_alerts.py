#!/usr/bin/env python3
"""
🚨 VISUAL ALERTS & NOTIFICATIONS SYSTEM 🚨
Real-time visual and audio alerts for trading signals
Desktop notifications, popup alerts, and sound effects
"""

import tkinter as tk
from tkinter import messagebox
import pygame
import plyer
import time
import threading
from datetime import datetime
import os
import sys

class VisualAlertSystem:
    def __init__(self):
        self.alerts_enabled = True
        self.sound_enabled = True
        self.desktop_notifications = True
        
        # Initialize pygame for sound
        try:
            pygame.mixer.init()
            self.sound_available = True
        except:
            self.sound_available = False
            print("Sound system not available")
        
        # Create alert window
        self.setup_alert_display()
        
    def setup_alert_display(self):
        """Setup floating alert display window"""
        self.alert_window = tk.Toplevel()
        self.alert_window.title("🚨 TRADING ALERTS")
        self.alert_window.geometry("400x600+50+50")  # Position top-left
        self.alert_window.configure(bg='#000000')
        self.alert_window.attributes('-topmost', True)  # Always on top
        self.alert_window.resizable(False, False)
        
        # Header
        header = tk.Label(self.alert_window, text="🚨 LIVE ALERTS 🚨", 
                         bg='#000000', fg='#ff6600',
                         font=('Arial', 16, 'bold'))
        header.pack(pady=10)
        
        # Status indicator
        self.status_label = tk.Label(self.alert_window, text="🟢 MONITORING", 
                                   bg='#000000', fg='#00ff88',
                                   font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=5)
        
        # Alerts frame with scrollbar
        alerts_frame = tk.Frame(self.alert_window, bg='#000000')
        alerts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(alerts_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.alerts_listbox = tk.Listbox(alerts_frame, bg='#111111', fg='#ffffff',
                                        font=('Courier', 10),
                                        yscrollcommand=scrollbar.set)
        self.alerts_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.alerts_listbox.yview)
        
        # Control buttons
        controls_frame = tk.Frame(self.alert_window, bg='#000000')
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mute_btn = tk.Button(controls_frame, text="🔊 MUTE", 
                                 bg='#333333', fg='white',
                                 command=self.toggle_sound)
        self.mute_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = tk.Button(controls_frame, text="🗑️ CLEAR", 
                             bg='#666666', fg='white',
                             command=self.clear_alerts)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Close button (minimize to system tray)
        self.alert_window.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)
        
    def show_signal_alert(self, signal_data):
        """Show visual alert for trading signal"""
        if not self.alerts_enabled:
            return
            
        signal_type = signal_data.get('type', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'N/A')
        price = signal_data.get('price', 0)
        score = signal_data.get('score', 0)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Color coding
        if signal_type.upper() == 'BUY':
            color = '#00ff88'
            emoji = '🚀'
        else:
            color = '#ff4466'
            emoji = '📉'
            
        # Add to alerts list
        alert_text = f"{timestamp} {emoji} {signal_type} {symbol} ${price:.2f} Score:{score}"
        self.alerts_listbox.insert(0, alert_text)
        
        # Color the item
        self.alerts_listbox.itemconfig(0, fg=color)
        
        # Keep only last 100 alerts
        if self.alerts_listbox.size() > 100:
            self.alerts_listbox.delete(100, tk.END)
        
        # Flash the window
        self.flash_alert_window()
        
        # Play sound
        if signal_type.upper() == 'BUY':
            self.play_buy_sound()
        else:
            self.play_sell_sound()
        
        # Desktop notification
        if self.desktop_notifications:
            self.show_desktop_notification(signal_data)
        
        # High score popup
        if score >= 80:
            self.show_high_score_popup(signal_data)
            
    def flash_alert_window(self):
        """Flash the alert window to get attention"""
        original_bg = self.alert_window.cget('bg')
        
        def flash():
            for _ in range(3):
                self.alert_window.configure(bg='#ff3333')
                self.alert_window.update()
                time.sleep(0.1)
                self.alert_window.configure(bg=original_bg)
                self.alert_window.update()
                time.sleep(0.1)
        
        threading.Thread(target=flash, daemon=True).start()
        
    def play_buy_sound(self):
        """Play sound for buy signal"""
        if not self.sound_enabled or not self.sound_available:
            return
            
        try:
            # Create a simple buy sound programmatically
            self.create_buy_sound()
            pygame.mixer.music.load('buy_alert.wav')
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Sound error: {e}")
            
    def play_sell_sound(self):
        """Play sound for sell signal"""
        if not self.sound_enabled or not self.sound_available:
            return
            
        try:
            # Create a simple sell sound programmatically
            self.create_sell_sound()
            pygame.mixer.music.load('sell_alert.wav')
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Sound error: {e}")
            
    def create_buy_sound(self):
        """Create programmatic buy sound (rising tone)"""
        try:
            import numpy as np
            import wave
            
            sample_rate = 44100
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Rising frequency from 440Hz to 880Hz
            frequency = 440 + (440 * t / duration)
            wave_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to 16-bit integers
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open('buy_alert.wav', 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
                
        except Exception as e:
            print(f"Error creating buy sound: {e}")
            
    def create_sell_sound(self):
        """Create programmatic sell sound (falling tone)"""
        try:
            import numpy as np
            import wave
            
            sample_rate = 44100
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Falling frequency from 880Hz to 440Hz
            frequency = 880 - (440 * t / duration)
            wave_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to 16-bit integers
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open('sell_alert.wav', 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
                
        except Exception as e:
            print(f"Error creating sell sound: {e}")
            
    def show_desktop_notification(self, signal_data):
        """Show desktop notification"""
        try:
            signal_type = signal_data.get('type', 'SIGNAL')
            symbol = signal_data.get('symbol', 'N/A')
            price = signal_data.get('price', 0)
            score = signal_data.get('score', 0)
            
            title = f"🚨 {signal_type.upper()} SIGNAL!"
            message = f"{symbol}: ${price:.2f}\nScore: {score}/100"
            
            plyer.notification.notify(
                title=title,
                message=message,
                timeout=5,
                toast=True
            )
        except Exception as e:
            print(f"Desktop notification error: {e}")
            
    def show_high_score_popup(self, signal_data):
        """Show popup for high-score signals"""
        signal_type = signal_data.get('type', 'SIGNAL')
        symbol = signal_data.get('symbol', 'N/A')
        price = signal_data.get('price', 0)
        score = signal_data.get('score', 0)
        
        popup = tk.Toplevel()
        popup.title("🔥 HIGH SCORE SIGNAL! 🔥")
        popup.geometry("400x300")
        popup.configure(bg='#ff3333')
        popup.attributes('-topmost', True)
        
        # Center the popup
        popup.transient(self.alert_window)
        popup.grab_set()
        
        # Content
        tk.Label(popup, text="🔥 HIGH SCORE SIGNAL! 🔥", 
                bg='#ff3333', fg='white',
                font=('Arial', 18, 'bold')).pack(pady=20)
        
        tk.Label(popup, text=f"Type: {signal_type.upper()}", 
                bg='#ff3333', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        tk.Label(popup, text=f"Symbol: {symbol}", 
                bg='#ff3333', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        tk.Label(popup, text=f"Price: ${price:.2f}", 
                bg='#ff3333', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=5)
        
        tk.Label(popup, text=f"Score: {score}/100", 
                bg='#ff3333', fg='white',
                font=('Arial', 16, 'bold')).pack(pady=10)
        
        tk.Button(popup, text="✅ ACKNOWLEDGED", 
                 bg='white', fg='black',
                 font=('Arial', 12, 'bold'),
                 command=popup.destroy).pack(pady=20)
        
        # Auto-close after 10 seconds
        popup.after(10000, popup.destroy)
        
    def toggle_sound(self):
        """Toggle sound alerts"""
        self.sound_enabled = not self.sound_enabled
        
        if self.sound_enabled:
            self.mute_btn.config(text="🔊 MUTE")
            self.status_label.config(text="🟢 MONITORING (SOUND ON)")
        else:
            self.mute_btn.config(text="🔇 UNMUTE")
            self.status_label.config(text="🟡 MONITORING (MUTED)")
            
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts_listbox.delete(0, tk.END)
        
    def minimize_to_tray(self):
        """Minimize to system tray (hide window)"""
        self.alert_window.withdraw()
        
    def show_window(self):
        """Show the alert window"""
        self.alert_window.deiconify()
        self.alert_window.lift()
        
    def test_alerts(self):
        """Test all alert types"""
        # Test buy signal
        buy_signal = {
            'type': 'BUY',
            'symbol': 'BTC/USDT',
            'price': 45000.00,
            'score': 85
        }
        self.show_signal_alert(buy_signal)
        
        time.sleep(2)
        
        # Test sell signal
        sell_signal = {
            'type': 'SELL',
            'symbol': 'ETH/USDT',
            'price': 3200.00,
            'score': 92
        }
        self.show_signal_alert(sell_signal)

# Example usage and testing
if __name__ == "__main__":
    # Create alert system
    alert_system = VisualAlertSystem()
    
    print("🚨 Visual Alert System Started!")
    print("Testing alerts in 3 seconds...")
    
    # Test after 3 seconds
    def run_test():
        time.sleep(3)
        alert_system.test_alerts()
    
    threading.Thread(target=run_test, daemon=True).start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down alert system...") 