"""
AI/ML Trading Strategy Engine

This service provides advanced AI/ML capabilities for trading including:
- Deep learning models for price prediction
- Reinforcement learning trading agents
- Feature engineering and selection
- Model training and backtesting
- Real-time inference and signal generation
- Strategy optimization
- Multi-timeframe analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pickle
import json
import io
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym import spaces
import ta
import yfinance as yf

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_async_session, User, Strategy
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient
from services.auth_service import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML Models and Strategy Types
class ModelType(str, Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRU = "gru"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE = "ensemble"

class StrategyType(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    PAIRS_TRADING = "pairs_trading"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    timeframe: str
    strategy_name: str
    model_name: str
    timestamp: datetime
    features: Dict[str, float]
    prediction: float

class ModelConfig(BaseModel):
    name: str
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    features: List[str]
    target: str
    lookback_window: int = 60
    prediction_horizon: int = 1
    training_data_size: int = 5000

class StrategyConfig(BaseModel):
    name: str
    strategy_type: StrategyType
    models: List[str]
    symbols: List[str]
    timeframes: List[str] = ["1h", "4h", "1d"]
    risk_per_trade: float = 0.02
    max_positions: int = 10
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TrainingRequest(BaseModel):
    model_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    retrain: bool = False

# Neural Network Models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last output
        
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size, model_dim, num_heads, num_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(model_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        
        return self.fc(x)

# Trading Environment for RL
class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + portfolio state
        obs_shape = data.shape[1] + 3  # features + position + balance + returns
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0=no position, 1=long, -1=short
        self.entry_price = 0
        self.total_reward = 0
        
        return self._get_observation()
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.balance *= (1 - self.transaction_cost)
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            pnl = (current_price - self.entry_price) / self.entry_price
            self.balance *= (1 + pnl - self.transaction_cost)
            reward = pnl
        
        # Calculate reward
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            reward = unrealized_pnl * 0.1  # Small reward for unrealized gains
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        self.total_reward += reward
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Market features
        market_data = self.data.iloc[self.current_step].values
        
        # Portfolio state
        portfolio_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self.total_reward
        ])
        
        return np.concatenate([market_data, portfolio_state]).astype(np.float32)

# AI/ML Strategy Engine
class AIMLStrategyEngine:
    def __init__(self):
        self.redis_client = RedisClient()
        self.mq_client = MessageQueueClient()
        self.models = {}
        self.strategies = {}
        self.scalers = {}
        
        # Model directories
        self.model_dir = "models"
        self.scaler_dir = "scalers"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the AI/ML strategy engine"""
        await self.redis_client.connect()
        await self.mq_client.connect()
        
        # Load existing models
        await self._load_models()
        
        logger.info("AI/ML Strategy Engine initialized")
    
    async def create_model(self, config: ModelConfig) -> str:
        """Create and configure a new ML model"""
        try:
            model_id = f"model_{config.name}_{int(datetime.utcnow().timestamp())}"
            
            if config.model_type == ModelType.LSTM:
                model = self._create_lstm_model(config)
            elif config.model_type == ModelType.TRANSFORMER:
                model = self._create_transformer_model(config)
            elif config.model_type == ModelType.RANDOM_FOREST:
                model = RandomForestRegressor(**config.hyperparameters)
            elif config.model_type == ModelType.XGBOOST:
                model = xgb.XGBRegressor(**config.hyperparameters)
            elif config.model_type == ModelType.REINFORCEMENT_LEARNING:
                model = self._create_rl_model(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Store model configuration
            model_info = {
                "id": model_id,
                "config": config.dict(),
                "created_at": datetime.utcnow().isoformat(),
                "trained": False,
                "performance_metrics": {}
            }
            
            self.models[model_id] = {
                "model": model,
                "config": config,
                "info": model_info
            }
            
            # Cache model info
            await self.redis_client.set_json(
                f"model_info:{model_id}", model_info, expire=None
            )
            
            logger.info(f"Created model {model_id} of type {config.model_type}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise HTTPException(status_code=500, detail="Model creation failed")
    
    async def train_model(
        self, 
        model_id: str, 
        training_request: TrainingRequest
    ) -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail="Model not found")
            
            model_data = self.models[model_id]
            model = model_data["model"]
            config = model_data["config"]
            
            # Get training data
            training_data = await self._prepare_training_data(
                training_request.symbol,
                training_request.start_date,
                training_request.end_date,
                config
            )
            
            if training_data.empty:
                raise HTTPException(status_code=400, detail="No training data available")
            
            # Train model based on type
            if config.model_type in [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.GRU]:
                metrics = await self._train_neural_network(
                    model, training_data, config, model_id
                )
            elif config.model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST]:
                metrics = await self._train_sklearn_model(
                    model, training_data, config, model_id
                )
            elif config.model_type == ModelType.REINFORCEMENT_LEARNING:
                metrics = await self._train_rl_model(
                    model, training_data, config, model_id
                )
            else:
                raise ValueError(f"Training not implemented for {config.model_type}")
            
            # Update model info
            self.models[model_id]["info"]["trained"] = True
            self.models[model_id]["info"]["performance_metrics"] = metrics
            self.models[model_id]["info"]["last_trained"] = datetime.utcnow().isoformat()
            
            # Save model and update cache
            await self._save_model(model_id)
            await self.redis_client.set_json(
                f"model_info:{model_id}", 
                self.models[model_id]["info"], 
                expire=None
            )
            
            logger.info(f"Model {model_id} trained successfully")
            return {
                "model_id": model_id,
                "metrics": metrics,
                "status": "trained"
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise HTTPException(status_code=500, detail="Model training failed")
    
    async def generate_signals(
        self, 
        strategy_name: str, 
        symbols: List[str],
        timeframe: str = "1h"
    ) -> List[TradingSignal]:
        """Generate trading signals using AI/ML models"""
        try:
            if strategy_name not in self.strategies:
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            strategy = self.strategies[strategy_name]
            signals = []
            
            for symbol in symbols:
                # Get recent market data
                market_data = await self._get_market_data(symbol, timeframe, lookback=100)
                
                if market_data.empty:
                    continue
                
                # Generate features
                features = await self._generate_features(market_data, strategy["config"])
                
                # Get predictions from each model in strategy
                predictions = []
                confidences = []
                
                for model_id in strategy["config"].models:
                    if model_id in self.models:
                        prediction, confidence = await self._get_model_prediction(
                            model_id, features, symbol
                        )
                        predictions.append(prediction)
                        confidences.append(confidence)
                
                if not predictions:
                    continue
                
                # Ensemble predictions
                ensemble_prediction = np.mean(predictions)
                ensemble_confidence = np.mean(confidences)
                
                # Generate signal based on strategy logic
                signal = await self._generate_signal_from_prediction(
                    symbol, ensemble_prediction, ensemble_confidence,
                    features, strategy, timeframe
                )
                
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
    
    async def create_strategy(self, config: StrategyConfig) -> str:
        """Create a new trading strategy"""
        try:
            strategy_id = f"strategy_{config.name}_{int(datetime.utcnow().timestamp())}"
            
            strategy_info = {
                "id": strategy_id,
                "config": config.dict(),
                "created_at": datetime.utcnow().isoformat(),
                "active": False,
                "performance": {}
            }
            
            self.strategies[strategy_id] = {
                "config": config,
                "info": strategy_info
            }
            
            # Cache strategy info
            await self.redis_client.set_json(
                f"strategy_info:{strategy_id}", strategy_info, expire=None
            )
            
            logger.info(f"Created strategy {strategy_id}")
            return strategy_id
            
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            raise HTTPException(status_code=500, detail="Strategy creation failed")
    
    async def backtest_strategy(
        self, 
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """Backtest a trading strategy"""
        try:
            if strategy_id not in self.strategies:
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            strategy = self.strategies[strategy_id]
            config = strategy["config"]
            
            # Get historical data for all symbols
            backtest_data = {}
            for symbol in config.symbols:
                data = await self._get_historical_data(symbol, start_date, end_date)
                if not data.empty:
                    backtest_data[symbol] = data
            
            if not backtest_data:
                raise HTTPException(status_code=400, detail="No backtest data available")
            
            # Run backtest simulation
            results = await self._run_backtest_simulation(
                strategy, backtest_data, initial_capital
            )
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_backtest_metrics(results)
            
            # Store backtest results
            backtest_result = {
                "strategy_id": strategy_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "final_value": performance_metrics.get("final_value", initial_capital),
                "total_return": performance_metrics.get("total_return", 0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0),
                "win_rate": performance_metrics.get("win_rate", 0),
                "total_trades": performance_metrics.get("total_trades", 0),
                "performance_metrics": performance_metrics,
                "trades": results.get("trades", []),
                "equity_curve": results.get("equity_curve", []),
                "created_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.set_json(
                f"backtest:{strategy_id}:{int(datetime.utcnow().timestamp())}",
                backtest_result,
                expire=86400 * 30  # 30 days
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise HTTPException(status_code=500, detail="Backtest failed")
    
    async def optimize_strategy(
        self, 
        strategy_id: str,
        optimization_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using genetic algorithm or grid search"""
        try:
            if strategy_id not in self.strategies:
                raise HTTPException(status_code=404, detail="Strategy not found")
            
            # Implementation for strategy optimization
            # This would use techniques like:
            # - Grid search
            # - Random search
            # - Genetic algorithms
            # - Bayesian optimization
            
            optimization_result = {
                "strategy_id": strategy_id,
                "optimization_method": optimization_parameters.get("method", "grid_search"),
                "best_parameters": {},
                "best_performance": {},
                "optimization_history": [],
                "completed_at": datetime.utcnow().isoformat()
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            raise HTTPException(status_code=500, detail="Strategy optimization failed")
    
    async def _create_lstm_model(self, config: ModelConfig) -> LSTMModel:
        """Create LSTM model"""
        hyperparams = config.hyperparameters
        return LSTMModel(
            input_size=len(config.features),
            hidden_size=hyperparams.get("hidden_size", 128),
            num_layers=hyperparams.get("num_layers", 2),
            output_size=1,
            dropout=hyperparams.get("dropout", 0.2)
        )
    
    async def _create_transformer_model(self, config: ModelConfig) -> TransformerModel:
        """Create Transformer model"""
        hyperparams = config.hyperparameters
        return TransformerModel(
            input_size=len(config.features),
            model_dim=hyperparams.get("model_dim", 256),
            num_heads=hyperparams.get("num_heads", 8),
            num_layers=hyperparams.get("num_layers", 4),
            output_size=1,
            dropout=hyperparams.get("dropout", 0.1)
        )
    
    async def _create_rl_model(self, config: ModelConfig):
        """Create reinforcement learning model"""
        # This would create a proper RL environment and model
        # For now, return a placeholder
        return None
    
    async def _prepare_training_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        config: ModelConfig
    ) -> pd.DataFrame:
        """Prepare training data with features and targets"""
        try:
            # Get historical data
            data = await self._get_historical_data(symbol, start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Generate technical indicators and features
            data = await self._add_technical_indicators(data)
            
            # Create target variable
            if config.target == "returns":
                data["target"] = data["close"].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)
            elif config.target == "price":
                data["target"] = data["close"].shift(-config.prediction_horizon)
            elif config.target == "direction":
                data["target"] = (data["close"].shift(-config.prediction_horizon) > data["close"]).astype(int)
            
            # Select features
            feature_columns = [col for col in config.features if col in data.columns]
            data = data[feature_columns + ["target"]].dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return pd.DataFrame()
    
    async def _train_neural_network(
        self, 
        model: nn.Module, 
        data: pd.DataFrame, 
        config: ModelConfig,
        model_id: str
    ) -> Dict[str, float]:
        """Train neural network model"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            # Prepare data
            features = data[config.features].values
            targets = data["target"].values
            
            # Scale data
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Store scaler
            self.scalers[model_id] = scaler
            
            # Create sequences for time series
            X, y = self._create_sequences(
                features_scaled, targets, config.lookback_window
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(device)
            X_test = torch.FloatTensor(X_test).to(device)
            y_train = torch.FloatTensor(y_train).to(device)
            y_test = torch.FloatTensor(y_test).to(device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training loop
            epochs = config.hyperparameters.get("epochs", 100)
            batch_size = config.hyperparameters.get("batch_size", 32)
            
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs.squeeze(), y_test)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f"{self.model_dir}/{model_id}.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= 10:  # Early stopping patience
                        break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {total_loss/len(X_train):.6f}, Val Loss: {val_loss:.6f}")
            
            # Calculate metrics
            model.eval()
            with torch.no_grad():
                predictions = model(X_test).squeeze().cpu().numpy()
                actual = y_test.cpu().numpy()
                
                mse = mean_squared_error(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                
                # Directional accuracy
                pred_direction = np.sign(predictions)
                actual_direction = np.sign(actual)
                directional_accuracy = np.mean(pred_direction == actual_direction)
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse)),
                "directional_accuracy": float(directional_accuracy),
                "final_train_loss": float(total_loss/len(X_train)),
                "final_val_loss": float(best_loss)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training neural network: {str(e)}")
            return {}
    
    async def _train_sklearn_model(
        self, 
        model, 
        data: pd.DataFrame, 
        config: ModelConfig,
        model_id: str
    ) -> Dict[str, float]:
        """Train sklearn-based model"""
        try:
            # Prepare data
            features = data[config.features].values
            targets = data["target"].values
            
            # Remove any remaining NaN values
            mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
            features = features[mask]
            targets = targets[mask]
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Store scaler
            self.scalers[model_id] = scaler
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, targets, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Directional accuracy
            train_dir_acc = np.mean(np.sign(y_pred_train) == np.sign(y_train))
            test_dir_acc = np.mean(np.sign(y_pred_test) == np.sign(y_test))
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    feature_importance[config.features[i]] = float(importance)
            
            metrics = {
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_mae": float(train_mae),
                "test_mae": float(test_mae),
                "train_rmse": float(np.sqrt(train_mse)),
                "test_rmse": float(np.sqrt(test_mse)),
                "train_directional_accuracy": float(train_dir_acc),
                "test_directional_accuracy": float(test_dir_acc),
                "feature_importance": feature_importance
            }
            
            # Save model
            with open(f"{self.model_dir}/{model_id}.pkl", "wb") as f:
                pickle.dump(model, f)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training sklearn model: {str(e)}")
            return {}
    
    async def _train_rl_model(
        self, 
        model, 
        data: pd.DataFrame, 
        config: ModelConfig,
        model_id: str
    ) -> Dict[str, float]:
        """Train reinforcement learning model"""
        try:
            # Create trading environment
            env = TradingEnvironment(data)
            
            # Create RL model (PPO in this case)
            rl_model = PPO("MlpPolicy", env, verbose=1)
            
            # Train model
            total_timesteps = config.hyperparameters.get("total_timesteps", 10000)
            rl_model.learn(total_timesteps=total_timesteps)
            
            # Evaluate model
            obs = env.reset()
            total_reward = 0
            
            for _ in range(len(data) - 1):
                action, _ = rl_model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            # Save model
            rl_model.save(f"{self.model_dir}/{model_id}")
            
            metrics = {
                "total_reward": float(total_reward),
                "final_balance": float(env.balance),
                "total_return": float((env.balance - env.initial_balance) / env.initial_balance)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training RL model: {str(e)}")
            return {}
    
    def _create_sequences(self, data, targets, lookback_window):
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(lookback_window, len(data)):
            X.append(data[i-lookback_window:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    async def _get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical market data"""
        try:
            # Use yfinance for demo purposes
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    async def _get_market_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback: int = 100
    ) -> pd.DataFrame:
        """Get recent market data"""
        try:
            # Calculate start date based on lookback
            if timeframe == "1h":
                start_date = datetime.utcnow() - timedelta(hours=lookback)
            elif timeframe == "4h":
                start_date = datetime.utcnow() - timedelta(hours=lookback * 4)
            elif timeframe == "1d":
                start_date = datetime.utcnow() - timedelta(days=lookback)
            else:
                start_date = datetime.utcnow() - timedelta(days=lookback)
            
            return await self._get_historical_data(symbol, start_date, datetime.utcnow())
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
    
    async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        try:
            if data.empty or len(data) < 20:
                return data
            
            # Price-based indicators
            data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
            data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
            data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            data['bb_upper'] = bb.bollinger_hband()
            data['bb_lower'] = bb.bollinger_lband()
            data['bb_middle'] = bb.bollinger_mavg()
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            # RSI
            data['rsi'] = ta.momentum.rsi(data['close'])
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            
            # ATR
            data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
            
            # Volume indicators
            data['volume_sma'] = ta.volume.volume_sma(data['close'], data['volume'])
            data['vwap'] = ta.volume.volume_weighted_average_price(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            # Price patterns
            data['returns'] = data['close'].pct_change()
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['price_position'] = (data['close'] - data['close'].rolling(window=20).min()) / \
                                   (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
            
            # Momentum indicators
            data['momentum'] = data['close'] / data['close'].shift(10) - 1
            data['roc'] = ta.momentum.roc(data['close'])
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    async def _generate_features(
        self, 
        data: pd.DataFrame, 
        strategy_config: StrategyConfig
    ) -> Dict[str, float]:
        """Generate features for prediction"""
        try:
            if data.empty:
                return {}
            
            # Get latest values
            latest = data.iloc[-1]
            
            features = {}
            
            # Technical indicators
            if 'rsi' in data.columns:
                features['rsi'] = float(latest['rsi'])
            if 'macd' in data.columns:
                features['macd'] = float(latest['macd'])
            if 'bb_width' in data.columns:
                features['bb_width'] = float(latest['bb_width'])
            if 'atr' in data.columns:
                features['atr'] = float(latest['atr'])
            if 'volume_ratio' in data.columns:
                features['volume_ratio'] = float(latest['volume'] / data['volume'].rolling(20).mean().iloc[-1])
            
            # Price patterns
            if 'returns' in data.columns:
                features['returns'] = float(latest['returns'])
            if 'volatility' in data.columns:
                features['volatility'] = float(latest['volatility'])
            if 'momentum' in data.columns:
                features['momentum'] = float(latest['momentum'])
            
            # Remove any NaN values
            features = {k: v for k, v in features.items() if not np.isnan(v)}
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            return {}
    
    async def _get_model_prediction(
        self, 
        model_id: str, 
        features: Dict[str, float],
        symbol: str
    ) -> Tuple[float, float]:
        """Get prediction from model"""
        try:
            if model_id not in self.models:
                return 0.0, 0.0
            
            model_data = self.models[model_id]
            model = model_data["model"]
            config = model_data["config"]
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in config.features:
                feature_vector.append(features.get(feature_name, 0.0))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features if scaler exists
            if model_id in self.scalers:
                feature_array = self.scalers[model_id].transform(feature_array)
            
            # Make prediction based on model type
            if config.model_type in [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.GRU]:
                model.eval()
                with torch.no_grad():
                    # For neural networks, we need sequences
                    # This is simplified - in practice, you'd maintain state
                    feature_tensor = torch.FloatTensor(feature_array)
                    prediction = model(feature_tensor.unsqueeze(0)).item()
                    confidence = min(abs(prediction) * 10, 1.0)  # Simplified confidence
            elif config.model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST]:
                prediction = model.predict(feature_array)[0]
                # For tree-based models, use prediction variance as confidence measure
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_array)[0]
                    confidence = max(proba) if len(proba) > 1 else 0.5
                else:
                    confidence = 0.7  # Default confidence
            else:
                prediction = 0.0
                confidence = 0.0
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error getting model prediction: {str(e)}")
            return 0.0, 0.0
    
    async def _generate_signal_from_prediction(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        features: Dict[str, float],
        strategy: Dict,
        timeframe: str
    ) -> Optional[TradingSignal]:
        """Generate trading signal from model prediction"""
        try:
            config = strategy["config"]
            
            # Define signal thresholds based on strategy type
            if config.strategy_type == StrategyType.TREND_FOLLOWING:
                buy_threshold = 0.02
                sell_threshold = -0.02
            elif config.strategy_type == StrategyType.MEAN_REVERSION:
                buy_threshold = -0.03  # Buy when oversold
                sell_threshold = 0.03   # Sell when overbought
            else:
                buy_threshold = 0.01
                sell_threshold = -0.01
            
            # Minimum confidence required
            min_confidence = 0.6
            
            if confidence < min_confidence:
                return None
            
            # Determine signal type
            if prediction > buy_threshold:
                signal_type = SignalType.STRONG_BUY if prediction > buy_threshold * 2 else SignalType.BUY
            elif prediction < sell_threshold:
                signal_type = SignalType.STRONG_SELL if prediction < sell_threshold * 2 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Only generate actionable signals
            if signal_type == SignalType.HOLD:
                return None
            
            # Calculate price targets and stop losses (simplified)
            current_price = features.get('close', 0)
            if current_price == 0:
                return None
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                price_target = current_price * (1 + abs(prediction))
                stop_loss = current_price * (1 - config.risk_per_trade)
            else:
                price_target = current_price * (1 - abs(prediction))
                stop_loss = current_price * (1 + config.risk_per_trade)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                timeframe=timeframe,
                strategy_name=config.name,
                model_name=strategy["info"]["id"],
                timestamp=datetime.utcnow(),
                features=features,
                prediction=prediction
            )
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    async def _save_model(self, model_id: str):
        """Save model to disk"""
        try:
            model_data = self.models[model_id]
            config = model_data["config"]
            
            # Save scaler if exists
            if model_id in self.scalers:
                with open(f"{self.scaler_dir}/{model_id}_scaler.pkl", "wb") as f:
                    pickle.dump(self.scalers[model_id], f)
            
            # PyTorch models are already saved during training
            # SKlearn models are saved during training
            
            logger.info(f"Model {model_id} saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    async def _load_models(self):
        """Load existing models from disk"""
        try:
            # This would load models and scalers from disk
            # Implementation depends on how models are stored
            pass
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    async def _run_backtest_simulation(
        self,
        strategy: Dict,
        backtest_data: Dict[str, pd.DataFrame],
        initial_capital: float
    ) -> Dict[str, Any]:
        """Run backtest simulation"""
        try:
            # Simplified backtest implementation
            # In practice, this would be much more sophisticated
            
            equity_curve = [initial_capital]
            trades = []
            current_capital = initial_capital
            
            # This is a placeholder implementation
            # Real backtesting would simulate day-by-day trading
            
            return {
                "equity_curve": equity_curve,
                "trades": trades,
                "final_value": current_capital
            }
            
        except Exception as e:
            logger.error(f"Error running backtest simulation: {str(e)}")
            return {"equity_curve": [], "trades": [], "final_value": initial_capital}
    
    async def _calculate_backtest_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate backtest performance metrics"""
        try:
            equity_curve = results.get("equity_curve", [])
            trades = results.get("trades", [])
            
            if not equity_curve or len(equity_curve) < 2:
                return {}
            
            initial_value = equity_curve[0]
            final_value = equity_curve[-1]
            
            # Basic metrics
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate other metrics (simplified)
            metrics = {
                "final_value": final_value,
                "total_return": total_return,
                "sharpe_ratio": 0.0,  # Would calculate properly
                "max_drawdown": 0.0,  # Would calculate properly
                "win_rate": 0.0,      # Would calculate from trades
                "total_trades": len(trades)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {str(e)}")
            return {}

# FastAPI App
app = FastAPI(title="AI/ML Strategy Engine", version="1.0.0")
engine = AIMLStrategyEngine()

@app.on_event("startup")
async def startup_event():
    await engine.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/models/create")
async def create_model(
    config: ModelConfig,
    current_user: User = Depends(get_current_user)
):
    """Create new ML model"""
    model_id = await engine.create_model(config)
    return {"model_id": model_id}

@app.post("/models/{model_id}/train")
async def train_model(
    model_id: str,
    request: TrainingRequest,
    current_user: User = Depends(get_current_user)
):
    """Train ML model"""
    result = await engine.train_model(model_id, request)
    return result

@app.post("/strategies/create")
async def create_strategy(
    config: StrategyConfig,
    current_user: User = Depends(get_current_user)
):
    """Create new trading strategy"""
    strategy_id = await engine.create_strategy(config)
    return {"strategy_id": strategy_id}

@app.post("/strategies/{strategy_name}/signals")
async def generate_signals(
    strategy_name: str,
    symbols: List[str],
    timeframe: str = "1h",
    current_user: User = Depends(get_current_user)
):
    """Generate trading signals"""
    signals = await engine.generate_signals(strategy_name, symbols, timeframe)
    return {"signals": [signal.__dict__ for signal in signals]}

@app.post("/strategies/{strategy_id}/backtest")
async def backtest_strategy(
    strategy_id: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    current_user: User = Depends(get_current_user)
):
    """Backtest trading strategy"""
    result = await engine.backtest_strategy(strategy_id, start_date, end_date, initial_capital)
    return result

@app.post("/strategies/{strategy_id}/optimize")
async def optimize_strategy(
    strategy_id: str,
    optimization_parameters: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Optimize strategy parameters"""
    result = await engine.optimize_strategy(strategy_id, optimization_parameters)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
