from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    # Time Range
    start_date: datetime
    end_date: datetime
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # Strategy Parameters
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    
    # Risk Parameters
    risk_per_trade: float = 0.025  # 2.5%
    max_open_trades: int = 3
    max_drawdown: float = 0.20  # 20%
    
    # Position Sizing
    position_sizing: str = "fixed"  # fixed, dynamic
    initial_balance: float = 10000.0
    
    # Optimization Parameters
    optimize_parameters: bool = False
    parameter_ranges: Dict[str, List[float]] = None
    
    # Performance Metrics
    target_sharpe_ratio: float = 1.5
    min_win_rate: float = 0.55
    max_daily_loss: float = 0.05  # 5%
    
    # Advanced Settings
    slippage: float = 0.001  # 0.1%
    commission: float = 0.0004  # 0.04%
    spread: float = 0.0002  # 0.02%
    
    @classmethod
    def from_dashboard(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """Create config from dashboard input"""
        return cls(
            start_date=datetime.strptime(data['start_date'], '%Y-%m-%d'),
            end_date=datetime.strptime(data['end_date'], '%Y-%m-%d'),
            timeframe=data['timeframe'],
            entry_conditions=data['entry_conditions'],
            exit_conditions=data['exit_conditions'],
            risk_per_trade=float(data['risk_per_trade']),
            max_open_trades=int(data['max_open_trades']),
            max_drawdown=float(data['max_drawdown']),
            position_sizing=data['position_sizing'],
            initial_balance=float(data['initial_balance']),
            optimize_parameters=data.get('optimize_parameters', False),
            parameter_ranges=data.get('parameter_ranges', None),
            target_sharpe_ratio=float(data['target_sharpe_ratio']),
            min_win_rate=float(data['min_win_rate']),
            max_daily_loss=float(data['max_daily_loss']),
            slippage=float(data['slippage']),
            commission=float(data['commission']),
            spread=float(data['spread'])
        )

class BacktestResults:
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.performance_metrics: Dict[str, float] = {}
        
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trades),
            'average_pnl': sum(t['pnl'] for t in self.trades) / total_trades if total_trades > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'profit_factor': self._calculate_profit_factor(),
            'average_win': sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0,
            'average_loss': sum(t['pnl'] for t in self.trades if t['pnl'] < 0) / losing_trades if losing_trades > 0 else 0
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0
            
        peak = self.equity_curve[0]
        max_drawdown = 0.0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0.0
            
        returns = [self.equity_curve[i] / self.equity_curve[i-1] - 1 
                  for i in range(1, len(self.equity_curve))]
        
        if not returns:
            return 0.0
            
        avg_return = sum(returns) / len(returns)
        std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        return avg_return / std_dev if std_dev != 0 else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf') 