import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import optuna
from typing import List, Dict, Any, Optional
import logging

class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.current_balance = config.get('initial_balance', 10000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        self.max_open_trades = config.get('max_open_trades', 3)
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.position_sizing = config.get('position_sizing', 'fixed')
        self.slippage = config.get('slippage', 0.001)
        self.commission = config.get('commission', 0.001)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = self.current_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        
        if self.position_sizing == 'fixed':
            return position_size
        elif self.position_sizing == 'percentage':
            return min(position_size, self.current_balance * 0.1)  # Max 10% of balance
        else:
            return position_size

    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met"""
        conditions = self.config.get('entry_conditions', {})
        
        # Example conditions (customize based on your strategy)
        if 'rsi' in conditions:
            rsi = data['rsi'].iloc[index]
            if not (conditions['rsi']['min'] <= rsi <= conditions['rsi']['max']):
                return False
                
        if 'macd' in conditions:
            macd = data['macd'].iloc[index]
            signal = data['macd_signal'].iloc[index]
            if not (macd > signal if conditions['macd']['direction'] == 'long' else macd < signal):
                return False
                
        return True

    def check_exit_conditions(self, data: pd.DataFrame, index: int, trade: Dict[str, Any]) -> bool:
        """Check if exit conditions are met"""
        conditions = self.config.get('exit_conditions', {})
        current_price = data['close'].iloc[index]
        
        # Check take profit
        if 'take_profit' in conditions:
            tp_levels = conditions['take_profit']
            for level in tp_levels:
                if trade['direction'] == 'long' and current_price >= level:
                    return True
                elif trade['direction'] == 'short' and current_price <= level:
                    return True
                    
        # Check stop loss
        if 'stop_loss' in conditions:
            sl_price = trade['stop_loss']
            if trade['direction'] == 'long' and current_price <= sl_price:
                return True
            elif trade['direction'] == 'short' and current_price >= sl_price:
                return True
                
        return False

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the backtest on historical data"""
        self.trades = []
        self.equity_curve = [self.current_balance]
        open_trades = []
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Check for exits first
            for trade in open_trades[:]:
                if self.check_exit_conditions(data, i, trade):
                    exit_price = current_price * (1 - self.slippage if trade['direction'] == 'long' else 1 + self.slippage)
                    pnl = (exit_price - trade['entry_price']) * trade['size'] * (1 if trade['direction'] == 'long' else -1)
                    pnl -= (trade['entry_price'] * trade['size'] * self.commission)  # Entry commission
                    pnl -= (exit_price * trade['size'] * self.commission)  # Exit commission
                    
                    trade.update({
                        'exit_price': exit_price,
                        'exit_time': data.index[i],
                        'pnl': pnl
                    })
                    
                    self.current_balance += pnl
                    self.trades.append(trade)
                    open_trades.remove(trade)
            
            # Check for new entries
            if len(open_trades) < self.max_open_trades and self.check_entry_conditions(data, i):
                entry_price = current_price * (1 + self.slippage if self.config['direction'] == 'long' else 1 - self.slippage)
                stop_loss = entry_price * (1 - self.config['stop_loss'] if self.config['direction'] == 'long' else 1 + self.config['stop_loss'])
                
                position_size = self.calculate_position_size(entry_price, stop_loss)
                
                new_trade = {
                    'entry_price': entry_price,
                    'entry_time': data.index[i],
                    'direction': self.config['direction'],
                    'size': position_size,
                    'stop_loss': stop_loss
                }
                
                open_trades.append(new_trade)
            
            # Update equity curve
            unrealized_pnl = sum(
                (current_price - trade['entry_price']) * trade['size'] * (1 if trade['direction'] == 'long' else -1)
                for trade in open_trades
            )
            self.equity_curve.append(self.current_balance + unrealized_pnl)
        
        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
            
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in self.trades)
        
        # Calculate returns for Sharpe ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Calculate max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return {
            'total_trades': len(self.trades),
            'win_rate': winning_trades / len(self.trades) if self.trades else 0,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'equity_curve': self.equity_curve
        }

    def optimize_parameters(self, data: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize strategy parameters using Optuna"""
        def objective(trial):
            # Define parameter ranges
            params = {
                'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                'take_profit': [
                    trial.suggest_float('tp1', 0.02, 0.1),
                    trial.suggest_float('tp2', 0.03, 0.15),
                    trial.suggest_float('tp3', 0.04, 0.2)
                ]
            }
            
            # Update config with trial parameters
            trial_config = self.config.copy()
            trial_config.update(params)
            
            # Run backtest
            engine = BacktestEngine(trial_config)
            results = engine.run_backtest(data)
            
            # Return negative Sharpe ratio (we want to maximize it)
            return -results['sharpe_ratio']
        
        # Create and run the study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_config = self.config.copy()
        best_config.update(best_params)
        
        # Run final backtest with best parameters
        final_engine = BacktestEngine(best_config)
        final_results = final_engine.run_backtest(data)
        
        return {
            'best_parameters': best_params,
            'results': final_results
        } 