"""
Risk Management Microservice

This service handles all risk management operations including:
- Position risk monitoring
- Portfolio risk analysis
- Risk limit enforcement
- VaR calculations
- Stress testing
- Real-time risk alerts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from core.database import get_async_session, User, Portfolio, Trade, Position, RiskProfile
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient
from services.auth_service import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Risk Service Models
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(str, Enum):
    POSITION_SIZE = "position_size"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"

@dataclass
class RiskAlert:
    risk_type: RiskType
    level: RiskLevel
    message: str
    portfolio_id: str
    user_id: str
    timestamp: datetime
    current_value: float
    threshold: float
    suggested_action: str

class RiskMetrics(BaseModel):
    portfolio_id: str
    total_exposure: Decimal
    max_position_size: Decimal
    concentration_risk: float
    var_1d: float
    var_5d: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_score: float
    last_updated: datetime

class RiskLimits(BaseModel):
    max_position_size_pct: float = Field(default=10.0, description="Max position size as % of portfolio")
    max_sector_concentration: float = Field(default=25.0, description="Max sector concentration %")
    max_daily_var: float = Field(default=5.0, description="Max daily VaR %")
    max_drawdown: float = Field(default=15.0, description="Max drawdown %")
    max_leverage: float = Field(default=3.0, description="Max leverage ratio")
    min_liquidity_ratio: float = Field(default=10.0, description="Min cash/liquidity ratio %")

class StressTestScenario(BaseModel):
    name: str
    market_shock: float  # % market drop
    volatility_spike: float  # volatility multiplier
    correlation_increase: float  # correlation increase
    liquidity_reduction: float  # liquidity reduction %

# Risk Management Service
class RiskManagementService:
    def __init__(self):
        self.redis_client = RedisClient()
        self.mq_client = MessageQueueClient()
        self.risk_cache = {}
        self.monitoring_active = True
        
    async def initialize(self):
        """Initialize the risk management service"""
        await self.redis_client.connect()
        await self.mq_client.connect()
        
        # Subscribe to trade and position updates
        await self.mq_client.subscribe_to_queue(
            "position_updates", 
            self._handle_position_update
        )
        await self.mq_client.subscribe_to_queue(
            "trade_executions", 
            self._handle_trade_execution
        )
        
        logger.info("Risk Management Service initialized")
    
    async def calculate_portfolio_risk(
        self, 
        portfolio_id: str, 
        db: AsyncSession
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio"""
        try:
            # Get portfolio and positions
            portfolio = await self._get_portfolio(portfolio_id, db)
            positions = await self._get_portfolio_positions(portfolio_id, db)
            
            if not positions:
                return self._empty_risk_metrics(portfolio_id)
            
            # Calculate risk metrics
            total_exposure = sum(pos.market_value for pos in positions)
            max_position = max(pos.market_value for pos in positions)
            max_position_pct = (max_position / total_exposure) * 100 if total_exposure > 0 else 0
            
            # Get price history for VaR calculation
            price_data = await self._get_price_history(positions)
            
            # Calculate VaR
            var_1d = await self._calculate_var(price_data, confidence=0.95, days=1)
            var_5d = await self._calculate_var(price_data, confidence=0.95, days=5)
            
            # Calculate other metrics
            sharpe_ratio = await self._calculate_sharpe_ratio(price_data)
            max_drawdown = await self._calculate_max_drawdown(price_data)
            beta = await self._calculate_beta(price_data)
            correlation_matrix = await self._calculate_correlation_matrix(price_data)
            
            # Calculate concentration risk
            concentration_risk = await self._calculate_concentration_risk(positions)
            
            # Calculate overall risk score
            risk_score = await self._calculate_risk_score(
                max_position_pct, concentration_risk, var_1d, max_drawdown, sharpe_ratio
            )
            
            metrics = RiskMetrics(
                portfolio_id=portfolio_id,
                total_exposure=Decimal(str(total_exposure)),
                max_position_size=Decimal(str(max_position)),
                concentration_risk=concentration_risk,
                var_1d=var_1d,
                var_5d=var_5d,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                beta=beta,
                correlation_matrix=correlation_matrix,
                risk_score=risk_score,
                last_updated=datetime.utcnow()
            )
            
            # Cache metrics
            await self.redis_client.set_json(
                f"risk_metrics:{portfolio_id}", 
                metrics.dict(), 
                expire=300  # 5 minutes
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            raise HTTPException(status_code=500, detail="Risk calculation failed")
    
    async def check_risk_limits(
        self, 
        portfolio_id: str, 
        limits: RiskLimits,
        db: AsyncSession
    ) -> List[RiskAlert]:
        """Check portfolio against risk limits and generate alerts"""
        alerts = []
        
        try:
            metrics = await self.calculate_portfolio_risk(portfolio_id, db)
            portfolio = await self._get_portfolio(portfolio_id, db)
            
            # Position size check
            max_position_pct = (float(metrics.max_position_size) / float(metrics.total_exposure)) * 100
            if max_position_pct > limits.max_position_size_pct:
                alerts.append(RiskAlert(
                    risk_type=RiskType.POSITION_SIZE,
                    level=RiskLevel.HIGH if max_position_pct > limits.max_position_size_pct * 1.5 else RiskLevel.MEDIUM,
                    message=f"Position size exceeds limit: {max_position_pct:.2f}% > {limits.max_position_size_pct}%",
                    portfolio_id=portfolio_id,
                    user_id=portfolio.user_id,
                    timestamp=datetime.utcnow(),
                    current_value=max_position_pct,
                    threshold=limits.max_position_size_pct,
                    suggested_action="Reduce largest position size"
                ))
            
            # Concentration risk check
            if metrics.concentration_risk > limits.max_sector_concentration:
                alerts.append(RiskAlert(
                    risk_type=RiskType.PORTFOLIO_CONCENTRATION,
                    level=RiskLevel.HIGH,
                    message=f"Portfolio concentration too high: {metrics.concentration_risk:.2f}% > {limits.max_sector_concentration}%",
                    portfolio_id=portfolio_id,
                    user_id=portfolio.user_id,
                    timestamp=datetime.utcnow(),
                    current_value=metrics.concentration_risk,
                    threshold=limits.max_sector_concentration,
                    suggested_action="Diversify holdings across sectors"
                ))
            
            # VaR check
            if metrics.var_1d > limits.max_daily_var:
                alerts.append(RiskAlert(
                    risk_type=RiskType.VAR_BREACH,
                    level=RiskLevel.CRITICAL if metrics.var_1d > limits.max_daily_var * 1.5 else RiskLevel.HIGH,
                    message=f"Daily VaR exceeds limit: {metrics.var_1d:.2f}% > {limits.max_daily_var}%",
                    portfolio_id=portfolio_id,
                    user_id=portfolio.user_id,
                    timestamp=datetime.utcnow(),
                    current_value=metrics.var_1d,
                    threshold=limits.max_daily_var,
                    suggested_action="Reduce portfolio volatility"
                ))
            
            # Drawdown check
            if metrics.max_drawdown > limits.max_drawdown:
                alerts.append(RiskAlert(
                    risk_type=RiskType.DRAWDOWN,
                    level=RiskLevel.HIGH,
                    message=f"Max drawdown exceeds limit: {metrics.max_drawdown:.2f}% > {limits.max_drawdown}%",
                    portfolio_id=portfolio_id,
                    user_id=portfolio.user_id,
                    timestamp=datetime.utcnow(),
                    current_value=metrics.max_drawdown,
                    threshold=limits.max_drawdown,
                    suggested_action="Review risk management strategy"
                ))
            
            # Store alerts in Redis and send notifications
            if alerts:
                await self._store_and_notify_alerts(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return []
    
    async def run_stress_test(
        self, 
        portfolio_id: str, 
        scenario: StressTestScenario,
        db: AsyncSession
    ) -> Dict:
        """Run stress test scenario on portfolio"""
        try:
            positions = await self._get_portfolio_positions(portfolio_id, db)
            
            # Get current portfolio value
            current_value = sum(pos.market_value for pos in positions)
            
            # Apply stress scenario
            stressed_value = 0
            position_impacts = []
            
            for position in positions:
                # Apply market shock
                price_impact = position.current_price * (1 + scenario.market_shock / 100)
                
                # Apply volatility adjustment
                vol_adjustment = 1 + (scenario.volatility_spike - 1) * 0.1  # Simplified
                
                # Calculate stressed position value
                stressed_position_value = position.quantity * price_impact * vol_adjustment
                stressed_value += stressed_position_value
                
                position_impacts.append({
                    "symbol": position.symbol,
                    "current_value": position.market_value,
                    "stressed_value": stressed_position_value,
                    "impact": stressed_position_value - position.market_value,
                    "impact_pct": ((stressed_position_value - position.market_value) / position.market_value) * 100
                })
            
            total_impact = stressed_value - current_value
            total_impact_pct = (total_impact / current_value) * 100 if current_value > 0 else 0
            
            result = {
                "scenario": scenario.dict(),
                "portfolio_id": portfolio_id,
                "current_value": current_value,
                "stressed_value": stressed_value,
                "total_impact": total_impact,
                "total_impact_pct": total_impact_pct,
                "position_impacts": position_impacts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache result
            await self.redis_client.set_json(
                f"stress_test:{portfolio_id}:{scenario.name}", 
                result, 
                expire=3600  # 1 hour
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running stress test: {str(e)}")
            raise HTTPException(status_code=500, detail="Stress test failed")
    
    async def _handle_position_update(self, message: dict):
        """Handle position update messages"""
        try:
            portfolio_id = message.get("portfolio_id")
            if portfolio_id:
                # Invalidate cached risk metrics
                await self.redis_client.delete(f"risk_metrics:{portfolio_id}")
                
                # Trigger risk recalculation
                asyncio.create_task(self._recalculate_risk_async(portfolio_id))
                
        except Exception as e:
            logger.error(f"Error handling position update: {str(e)}")
    
    async def _handle_trade_execution(self, message: dict):
        """Handle trade execution messages"""
        try:
            portfolio_id = message.get("portfolio_id")
            if portfolio_id:
                # Check risk limits after trade
                asyncio.create_task(self._check_post_trade_risk(portfolio_id, message))
                
        except Exception as e:
            logger.error(f"Error handling trade execution: {str(e)}")
    
    async def _calculate_var(self, price_data: pd.DataFrame, confidence: float, days: int) -> float:
        """Calculate Value at Risk"""
        try:
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 30:  # Need sufficient data
                return 0.0
            
            # Calculate portfolio returns (simplified)
            portfolio_returns = returns.mean(axis=1)
            
            # Scale to specified days
            scaled_returns = portfolio_returns * np.sqrt(days)
            
            # Calculate VaR
            var = np.percentile(scaled_returns, (1 - confidence) * 100)
            
            return abs(var) * 100  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self, price_data: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 30:
                return 0.0
            
            portfolio_returns = returns.mean(axis=1)
            
            excess_returns = portfolio_returns.mean() * 252 - risk_free_rate  # Annualized
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            return excess_returns / volatility if volatility > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0
    
    async def _calculate_max_drawdown(self, price_data: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        try:
            portfolio_value = price_data.sum(axis=1)
            rolling_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - rolling_max) / rolling_max
            
            return abs(drawdown.min()) * 100  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    async def _calculate_beta(self, price_data: pd.DataFrame) -> float:
        """Calculate portfolio beta against market"""
        try:
            # Simplified beta calculation
            # In practice, you'd use a market index
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 30:
                return 1.0
            
            portfolio_returns = returns.mean(axis=1)
            market_returns = returns.iloc[:, 0]  # Use first asset as proxy
            
            covariance = np.cov(portfolio_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    async def _calculate_correlation_matrix(self, price_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between assets"""
        try:
            returns = price_data.pct_change().dropna()
            corr_matrix = returns.corr()
            
            # Convert to dict format
            result = {}
            for col in corr_matrix.columns:
                result[col] = {}
                for row in corr_matrix.index:
                    result[col][row] = float(corr_matrix.loc[row, col])
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return {}
    
    async def _calculate_concentration_risk(self, positions: List) -> float:
        """Calculate portfolio concentration risk"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.market_value for pos in positions)
            
            if total_value <= 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            weights = [pos.market_value / total_value for pos in positions]
            hhi = sum(w ** 2 for w in weights)
            
            # Convert to concentration percentage
            return (hhi - 1/len(positions)) / (1 - 1/len(positions)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {str(e)}")
            return 0.0
    
    async def _calculate_risk_score(
        self, 
        position_size_pct: float, 
        concentration: float, 
        var: float, 
        drawdown: float, 
        sharpe: float
    ) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Weighted risk components
            size_score = min(position_size_pct / 20 * 25, 25)  # 25% weight
            concentration_score = min(concentration / 30 * 25, 25)  # 25% weight
            var_score = min(var / 10 * 25, 25)  # 25% weight
            drawdown_score = min(drawdown / 20 * 20, 20)  # 20% weight
            sharpe_score = max(0, 5 - sharpe) if sharpe > 0 else 5  # 5% weight
            
            total_score = size_score + concentration_score + var_score + drawdown_score + sharpe_score
            
            return min(total_score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 50.0  # Default medium risk
    
    async def _get_portfolio(self, portfolio_id: str, db: AsyncSession):
        """Get portfolio from database"""
        result = await db.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_portfolio_positions(self, portfolio_id: str, db: AsyncSession):
        """Get portfolio positions from database"""
        # This would be implemented based on your Position model
        # For now, return empty list
        return []
    
    async def _get_price_history(self, positions: List) -> pd.DataFrame:
        """Get price history for positions"""
        # This would fetch historical price data
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def _empty_risk_metrics(self, portfolio_id: str) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            portfolio_id=portfolio_id,
            total_exposure=Decimal("0"),
            max_position_size=Decimal("0"),
            concentration_risk=0.0,
            var_1d=0.0,
            var_5d=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            beta=1.0,
            correlation_matrix={},
            risk_score=0.0,
            last_updated=datetime.utcnow()
        )
    
    async def _store_and_notify_alerts(self, alerts: List[RiskAlert]):
        """Store alerts and send notifications"""
        for alert in alerts:
            # Store in Redis
            alert_key = f"risk_alert:{alert.portfolio_id}:{alert.timestamp.timestamp()}"
            await self.redis_client.set_json(alert_key, alert.__dict__, expire=86400)  # 24 hours
            
            # Send to notification queue
            await self.mq_client.publish_message(
                "risk_alerts",
                {
                    "type": "risk_alert",
                    "alert": alert.__dict__,
                    "timestamp": alert.timestamp.isoformat()
                }
            )
    
    async def _recalculate_risk_async(self, portfolio_id: str):
        """Asynchronously recalculate risk metrics"""
        try:
            # This would recalculate risk metrics in background
            pass
        except Exception as e:
            logger.error(f"Error in async risk recalculation: {str(e)}")
    
    async def _check_post_trade_risk(self, portfolio_id: str, trade_data: dict):
        """Check risk limits after trade execution"""
        try:
            # This would check risk limits after a trade
            pass
        except Exception as e:
            logger.error(f"Error in post-trade risk check: {str(e)}")

# FastAPI App
app = FastAPI(title="Risk Management Service", version="1.0.0")
risk_service = RiskManagementService()

@app.on_event("startup")
async def startup_event():
    await risk_service.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/risk/portfolio/{portfolio_id}/metrics", response_model=RiskMetrics)
async def get_portfolio_risk_metrics(
    portfolio_id: str,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get risk metrics for a portfolio"""
    return await risk_service.calculate_portfolio_risk(portfolio_id, db)

@app.post("/risk/portfolio/{portfolio_id}/check-limits")
async def check_portfolio_risk_limits(
    portfolio_id: str,
    limits: RiskLimits,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Check portfolio against risk limits"""
    alerts = await risk_service.check_risk_limits(portfolio_id, limits, db)
    return {"alerts": [alert.__dict__ for alert in alerts]}

@app.post("/risk/portfolio/{portfolio_id}/stress-test")
async def run_portfolio_stress_test(
    portfolio_id: str,
    scenario: StressTestScenario,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Run stress test on portfolio"""
    return await risk_service.run_stress_test(portfolio_id, scenario, db)

@app.get("/risk/portfolio/{portfolio_id}/alerts")
async def get_portfolio_alerts(
    portfolio_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get recent risk alerts for portfolio"""
    alerts = await risk_service.redis_client.get_keys_by_pattern(
        f"risk_alert:{portfolio_id}:*"
    )
    
    alert_data = []
    for alert_key in alerts:
        alert = await risk_service.redis_client.get_json(alert_key)
        if alert:
            alert_data.append(alert)
    
    return {"alerts": alert_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
