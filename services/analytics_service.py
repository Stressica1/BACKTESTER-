"""
Analytics and Reporting Microservice

This service provides comprehensive analytics and reporting including:
- Portfolio performance analytics
- Trading strategy analysis
- Risk reporting
- P&L analytics
- Custom dashboards
- Automated report generation
- Business intelligence
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import io
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
import jinja2
import pdfkit
from datetime import timezone

from core.database import get_async_session, User, Portfolio, Trade, Position, Strategy
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient
from services.auth_service import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Analytics Models
class ReportType(str, Enum):
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    RISK_ANALYSIS = "risk_analysis"
    TRADING_ACTIVITY = "trading_activity"
    STRATEGY_PERFORMANCE = "strategy_performance"
    PNL_SUMMARY = "pnl_summary"
    COMPLIANCE = "compliance"
    EXECUTIVE_SUMMARY = "executive_summary"

class TimeFrame(str, Enum):
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"
    QUARTERLY = "3M"
    YEARLY = "1Y"
    ALL_TIME = "ALL"

class MetricType(str, Enum):
    RETURN = "return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

class AnalyticsRequest(BaseModel):
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    start_date: datetime
    end_date: datetime
    time_frame: TimeFrame = TimeFrame.DAILY
    metrics: List[MetricType] = Field(default_factory=lambda: [MetricType.RETURN])
    benchmark: Optional[str] = None

class ReportRequest(BaseModel):
    report_type: ReportType
    portfolio_ids: List[str]
    start_date: datetime
    end_date: datetime
    include_charts: bool = True
    format: str = Field(default="html", regex="^(html|pdf|json)$")
    custom_metrics: List[str] = Field(default_factory=list)

class DashboardConfig(BaseModel):
    user_id: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    refresh_interval: int = 300  # seconds
    name: str

# Analytics and Reporting Service
class AnalyticsService:
    def __init__(self):
        self.redis_client = RedisClient()
        self.mq_client = MessageQueueClient()
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('templates'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    async def initialize(self):
        """Initialize the analytics service"""
        await self.redis_client.connect()
        await self.mq_client.connect()
        
        logger.info("Analytics Service initialized")
    
    async def calculate_portfolio_performance(
        self, 
        request: AnalyticsRequest,
        db: AsyncSession
    ) -> PerformanceMetrics:
        """Calculate comprehensive portfolio performance metrics"""
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(request, db)
            
            if portfolio_data.empty:
                return self._empty_performance_metrics()
            
            # Calculate returns
            returns = portfolio_data['value'].pct_change().dropna()
            
            # Calculate metrics
            total_return = ((portfolio_data['value'].iloc[-1] / portfolio_data['value'].iloc[0]) - 1) * 100
            
            # Annualized return
            days = (request.end_date - request.start_date).days
            annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) * 100
            sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
            
            # Maximum drawdown
            rolling_max = portfolio_data['value'].expanding().max()
            drawdown = (portfolio_data['value'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Get trade data for additional metrics
            trade_metrics = await self._calculate_trade_metrics(request, db)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                avg_win=trade_metrics['avg_win'],
                avg_loss=trade_metrics['avg_loss'],
                total_trades=trade_metrics['total_trades'],
                winning_trades=trade_metrics['winning_trades'],
                losing_trades=trade_metrics['losing_trades']
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {str(e)}")
            return self._empty_performance_metrics()
    
    async def generate_performance_chart(
        self, 
        request: AnalyticsRequest,
        db: AsyncSession
    ) -> str:
        """Generate interactive performance chart"""
        try:
            # Get portfolio data
            portfolio_data = await self._get_portfolio_data(request, db)
            
            if portfolio_data.empty:
                return ""
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown'),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Portfolio value chart
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data.index,
                    y=portfolio_data['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Benchmark if provided
            if request.benchmark:
                benchmark_data = await self._get_benchmark_data(request, db)
                if not benchmark_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_data.index,
                            y=benchmark_data['value'],
                            mode='lines',
                            name=f'Benchmark ({request.benchmark})',
                            line=dict(color='gray', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
            
            # Daily returns
            returns = portfolio_data['value'].pct_change() * 100
            colors = ['green' if r > 0 else 'red' for r in returns]
            
            fig.add_trace(
                go.Bar(
                    x=portfolio_data.index[1:],
                    y=returns[1:],
                    name='Daily Returns (%)',
                    marker_color=colors[1:]
                ),
                row=2, col=1
            )
            
            # Drawdown
            rolling_max = portfolio_data['value'].expanding().max()
            drawdown = (portfolio_data['value'] - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_data.index,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown (%)',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance Analysis',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return ""
    
    async def generate_risk_analysis_chart(
        self, 
        request: AnalyticsRequest,
        db: AsyncSession
    ) -> str:
        """Generate risk analysis charts"""
        try:
            portfolio_data = await self._get_portfolio_data(request, db)
            
            if portfolio_data.empty:
                return ""
            
            returns = portfolio_data['value'].pct_change().dropna() * 100
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Returns Distribution', 'Rolling Volatility', 
                              'Risk-Return Scatter', 'VaR Analysis'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns Distribution',
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # Rolling volatility
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name='30-Day Rolling Volatility',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=2
            )
            
            # Risk-return scatter (rolling periods)
            periods = 30
            rolling_returns = returns.rolling(window=periods).mean() * 252
            rolling_vols = returns.rolling(window=periods).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_vols,
                    y=rolling_returns,
                    mode='markers',
                    name='Risk-Return',
                    marker=dict(
                        size=8,
                        color=rolling_returns,
                        colorscale='RdYlGn',
                        showscale=True
                    )
                ),
                row=2, col=1
            )
            
            # VaR analysis
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='VaR Analysis',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
            
            # Add VaR lines
            fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                         annotation_text=f"VaR 95%: {var_95:.2f}%", row=2, col=2)
            fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", 
                         annotation_text=f"VaR 99%: {var_99:.2f}%", row=2, col=2)
            
            fig.update_layout(
                title='Risk Analysis Dashboard',
                height=700,
                showlegend=False,
                template='plotly_white'
            )
            
            return fig.to_html(include_plotlyjs=True)
            
        except Exception as e:
            logger.error(f"Error generating risk analysis chart: {str(e)}")
            return ""
    
    async def generate_report(
        self, 
        request: ReportRequest,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate comprehensive report"""
        try:
            report_data = {
                "report_type": request.report_type,
                "generated_at": datetime.utcnow(),
                "portfolios": [],
                "summary": {},
                "charts": {} if request.include_charts else None
            }
            
            # Generate data for each portfolio
            for portfolio_id in request.portfolio_ids:
                analytics_request = AnalyticsRequest(
                    portfolio_id=portfolio_id,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                # Get performance metrics
                performance = await self.calculate_portfolio_performance(analytics_request, db)
                
                portfolio_data = {
                    "portfolio_id": portfolio_id,
                    "performance": performance.__dict__,
                    "positions": await self._get_portfolio_positions_summary(portfolio_id, db),
                    "trades": await self._get_portfolio_trades_summary(analytics_request, db)
                }
                
                # Add charts if requested
                if request.include_charts:
                    portfolio_data["charts"] = {
                        "performance": await self.generate_performance_chart(analytics_request, db),
                        "risk_analysis": await self.generate_risk_analysis_chart(analytics_request, db)
                    }
                
                report_data["portfolios"].append(portfolio_data)
            
            # Generate summary across all portfolios
            report_data["summary"] = await self._generate_report_summary(report_data["portfolios"])
            
            # Format report based on requested format
            if request.format == "html":
                return {"content": await self._format_html_report(report_data), "type": "html"}
            elif request.format == "pdf":
                html_content = await self._format_html_report(report_data)
                pdf_content = self._convert_html_to_pdf(html_content)
                return {"content": base64.b64encode(pdf_content).decode(), "type": "pdf"}
            else:  # json
                return {"content": report_data, "type": "json"}
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise HTTPException(status_code=500, detail="Report generation failed")
    
    async def create_dashboard(
        self, 
        config: DashboardConfig,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Create custom dashboard"""
        try:
            dashboard_data = {
                "user_id": config.user_id,
                "name": config.name,
                "widgets": [],
                "created_at": datetime.utcnow()
            }
            
            # Process each widget
            for widget_config in config.widgets:
                widget_data = await self._process_widget(widget_config, db)
                dashboard_data["widgets"].append(widget_data)
            
            # Store dashboard configuration
            await self.redis_client.set_json(
                f"dashboard:{config.user_id}:{config.name}",
                dashboard_data,
                expire=3600 * 24  # 24 hours
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise HTTPException(status_code=500, detail="Dashboard creation failed")
    
    async def get_market_overview(self, db: AsyncSession) -> Dict[str, Any]:
        """Get market overview data"""
        try:
            overview = {
                "timestamp": datetime.utcnow(),
                "market_data": await self._get_market_data(),
                "top_movers": await self._get_top_movers(),
                "sector_performance": await self._get_sector_performance(),
                "economic_indicators": await self._get_economic_indicators()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {str(e)}")
            return {}
    
    async def _get_portfolio_data(self, request: AnalyticsRequest, db: AsyncSession) -> pd.DataFrame:
        """Get portfolio historical data"""
        try:
            # This would query historical portfolio values
            # For now, generate sample data
            dates = pd.date_range(request.start_date, request.end_date, freq='D')
            np.random.seed(42)  # For reproducible results
            
            initial_value = 100000
            returns = np.random.normal(0.0005, 0.02, len(dates))
            values = [initial_value]
            
            for ret in returns[1:]:
                values.append(values[-1] * (1 + ret))
            
            return pd.DataFrame({
                'value': values
            }, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {str(e)}")
            return pd.DataFrame()
    
    async def _get_benchmark_data(self, request: AnalyticsRequest, db: AsyncSession) -> pd.DataFrame:
        """Get benchmark data"""
        try:
            # This would fetch actual benchmark data
            # For now, generate sample benchmark data
            dates = pd.date_range(request.start_date, request.end_date, freq='D')
            np.random.seed(123)  # Different seed for benchmark
            
            initial_value = 100000
            returns = np.random.normal(0.0003, 0.015, len(dates))
            values = [initial_value]
            
            for ret in returns[1:]:
                values.append(values[-1] * (1 + ret))
            
            return pd.DataFrame({
                'value': values
            }, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {str(e)}")
            return pd.DataFrame()
    
    async def _calculate_trade_metrics(self, request: AnalyticsRequest, db: AsyncSession) -> Dict:
        """Calculate trade-based metrics"""
        try:
            # This would query actual trade data
            # For now, return sample metrics
            return {
                'win_rate': 65.5,
                'profit_factor': 1.85,
                'avg_win': 2.3,
                'avg_loss': -1.8,
                'total_trades': 125,
                'winning_trades': 82,
                'losing_trades': 43
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {str(e)}")
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0
        )
    
    async def _get_portfolio_positions_summary(self, portfolio_id: str, db: AsyncSession) -> List[Dict]:
        """Get portfolio positions summary"""
        # This would query actual position data
        return []
    
    async def _get_portfolio_trades_summary(self, request: AnalyticsRequest, db: AsyncSession) -> List[Dict]:
        """Get portfolio trades summary"""
        # This would query actual trade data
        return []
    
    async def _generate_report_summary(self, portfolios: List[Dict]) -> Dict:
        """Generate summary across portfolios"""
        if not portfolios:
            return {}
        
        total_value = sum(p.get("performance", {}).get("total_return", 0) for p in portfolios)
        avg_return = total_value / len(portfolios) if portfolios else 0
        
        return {
            "total_portfolios": len(portfolios),
            "average_return": avg_return,
            "best_performer": max(portfolios, key=lambda p: p.get("performance", {}).get("total_return", 0), default={}),
            "worst_performer": min(portfolios, key=lambda p: p.get("performance", {}).get("total_return", 0), default={})
        }
    
    async def _format_html_report(self, report_data: Dict) -> str:
        """Format report as HTML"""
        try:
            template = self.jinja_env.get_template('report_template.html')
            return template.render(report=report_data)
        except Exception as e:
            logger.error(f"Error formatting HTML report: {str(e)}")
            return f"<html><body><h1>Report Error</h1><p>{str(e)}</p></body></html>"
    
    def _convert_html_to_pdf(self, html_content: str) -> bytes:
        """Convert HTML to PDF"""
        try:
            return pdfkit.from_string(html_content, False)
        except Exception as e:
            logger.error(f"Error converting to PDF: {str(e)}")
            return b""
    
    async def _process_widget(self, widget_config: Dict, db: AsyncSession) -> Dict:
        """Process dashboard widget"""
        widget_type = widget_config.get("type", "")
        
        if widget_type == "performance_chart":
            return await self._create_performance_widget(widget_config, db)
        elif widget_type == "metrics_table":
            return await self._create_metrics_widget(widget_config, db)
        elif widget_type == "risk_gauge":
            return await self._create_risk_widget(widget_config, db)
        else:
            return {"type": widget_type, "error": "Unknown widget type"}
    
    async def _create_performance_widget(self, config: Dict, db: AsyncSession) -> Dict:
        """Create performance widget"""
        # Implementation for performance widget
        return {"type": "performance_chart", "data": {}}
    
    async def _create_metrics_widget(self, config: Dict, db: AsyncSession) -> Dict:
        """Create metrics widget"""
        # Implementation for metrics widget
        return {"type": "metrics_table", "data": {}}
    
    async def _create_risk_widget(self, config: Dict, db: AsyncSession) -> Dict:
        """Create risk widget"""
        # Implementation for risk widget
        return {"type": "risk_gauge", "data": {}}
    
    async def _get_market_data(self) -> Dict:
        """Get current market data"""
        return {
            "indices": {
                "S&P 500": {"value": 4500, "change": 1.2},
                "NASDAQ": {"value": 15000, "change": 0.8},
                "DOW": {"value": 35000, "change": 0.5}
            }
        }
    
    async def _get_top_movers(self) -> Dict:
        """Get top moving stocks"""
        return {
            "gainers": [
                {"symbol": "AAPL", "change": 5.2},
                {"symbol": "MSFT", "change": 3.1}
            ],
            "losers": [
                {"symbol": "TSLA", "change": -2.8},
                {"symbol": "NVDA", "change": -1.5}
            ]
        }
    
    async def _get_sector_performance(self) -> Dict:
        """Get sector performance"""
        return {
            "Technology": 2.1,
            "Healthcare": 1.5,
            "Financials": -0.8,
            "Energy": 1.9
        }
    
    async def _get_economic_indicators(self) -> Dict:
        """Get economic indicators"""
        return {
            "VIX": 18.5,
            "10Y_Treasury": 4.2,
            "DXY": 103.5,
            "Gold": 1950.0
        }

# FastAPI App
app = FastAPI(title="Analytics & Reporting Service", version="1.0.0")
analytics_service = AnalyticsService()

@app.on_event("startup")
async def startup_event():
    await analytics_service.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/analytics/performance", response_model=PerformanceMetrics)
async def get_performance_analytics(
    request: AnalyticsRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get performance analytics"""
    return await analytics_service.calculate_portfolio_performance(request, db)

@app.post("/analytics/charts/performance")
async def get_performance_chart(
    request: AnalyticsRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get performance chart"""
    chart_html = await analytics_service.generate_performance_chart(request, db)
    return {"chart": chart_html}

@app.post("/analytics/charts/risk")
async def get_risk_analysis_chart(
    request: AnalyticsRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get risk analysis chart"""
    chart_html = await analytics_service.generate_risk_analysis_chart(request, db)
    return {"chart": chart_html}

@app.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Generate comprehensive report"""
    return await analytics_service.generate_report(request, db)

@app.post("/dashboards/create")
async def create_dashboard(
    config: DashboardConfig,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Create custom dashboard"""
    return await analytics_service.create_dashboard(config, db)

@app.get("/market/overview")
async def get_market_overview(
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get market overview"""
    return await analytics_service.get_market_overview(db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
