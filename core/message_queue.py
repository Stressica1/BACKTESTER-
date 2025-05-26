"""
Message Queue System using RabbitMQ
Handles asynchronous processing, event-driven architecture, and inter-service communication
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime
import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType
from aio_pika.abc import AbstractChannel, AbstractConnection, AbstractExchange, AbstractQueue
import pydantic
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class QueueMessage(BaseModel):
    """Standard message format for queue operations"""
    id: str
    type: str
    service: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 5
    retry_count: int = 0
    max_retries: int = 3

class MessageQueueClient:
    """Advanced RabbitMQ client for enterprise trading platform"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection: Optional[AbstractConnection] = None
        self.channel: Optional[AbstractChannel] = None
        self.exchanges: Dict[str, AbstractExchange] = {}
        self.queues: Dict[str, AbstractQueue] = {}
        self.consumers: Dict[str, Callable] = {}
        self.is_connected = False
        
    async def connect(self) -> None:
        """Establish connection to RabbitMQ with connection pooling"""
        try:
            connection_url = (
                f"amqp://{self.config['username']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['vhost']}"
            )
            
            self.connection = await aio_pika.connect_robust(
                connection_url,
                client_properties={"connection_name": "trading_platform"},
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # Setup exchanges
            await self._setup_exchanges()
            
            self.is_connected = True
            logger.info("Connected to RabbitMQ successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def _setup_exchanges(self) -> None:
        """Setup standard exchanges for different message types"""
        exchange_configs = [
            ("trading.topic", ExchangeType.TOPIC),
            ("market.data", ExchangeType.FANOUT),
            ("notifications", ExchangeType.DIRECT),
            ("deadletter", ExchangeType.DIRECT),
            ("analytics", ExchangeType.TOPIC),
            ("risk.management", ExchangeType.DIRECT),
        ]
        
        for exchange_name, exchange_type in exchange_configs:
            self.exchanges[exchange_name] = await self.channel.declare_exchange(
                exchange_name,
                exchange_type,
                durable=True
            )

    async def declare_queue(
        self,
        queue_name: str,
        routing_key: str = "",
        exchange_name: str = "trading.topic",
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
        arguments: Optional[Dict] = None
    ) -> AbstractQueue:
        """Declare a queue and bind it to an exchange"""
        
        if not self.is_connected:
            await self.connect()
            
        # Default arguments for dead letter queue
        if arguments is None:
            arguments = {
                "x-dead-letter-exchange": "deadletter",
                "x-dead-letter-routing-key": f"dlq.{queue_name}",
                "x-message-ttl": 3600000,  # 1 hour TTL
            }
        
        queue = await self.channel.declare_queue(
            queue_name,
            durable=durable,
            exclusive=exclusive,
            auto_delete=auto_delete,
            arguments=arguments
        )
        
        # Bind queue to exchange
        if exchange_name in self.exchanges:
            await queue.bind(self.exchanges[exchange_name], routing_key)
        
        self.queues[queue_name] = queue
        logger.info(f"Queue '{queue_name}' declared and bound to '{exchange_name}'")
        
        return queue

    async def publish_message(
        self,
        message: QueueMessage,
        exchange_name: str = "trading.topic",
        routing_key: str = "",
        priority: int = 5
    ) -> None:
        """Publish a message to an exchange"""
        
        if not self.is_connected:
            await self.connect()
            
        try:
            message_body = message.json().encode()
            
            rabbit_message = Message(
                message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority,
                timestamp=datetime.utcnow(),
                message_id=message.id,
                type=message.type,
                headers={
                    "service": message.service,
                    "retry_count": message.retry_count,
                }
            )
            
            await self.exchanges[exchange_name].publish(
                rabbit_message,
                routing_key=routing_key
            )
            
            logger.debug(f"Published message {message.id} to {exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def consume_messages(
        self,
        queue_name: str,
        callback: Callable,
        auto_ack: bool = False
    ) -> None:
        """Start consuming messages from a queue"""
        
        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not declared")
            
        queue = self.queues[queue_name]
        
        async def message_handler(message: aio_pika.IncomingMessage):
            try:
                # Parse message
                message_data = json.loads(message.body.decode())
                queue_message = QueueMessage(**message_data)
                
                # Process message
                success = await callback(queue_message)
                
                if success:
                    await message.ack()
                    logger.debug(f"Message {queue_message.id} processed successfully")
                else:
                    # Retry logic
                    if queue_message.retry_count < queue_message.max_retries:
                        queue_message.retry_count += 1
                        await self.publish_message(
                            queue_message,
                            routing_key=f"retry.{queue_name}"
                        )
                        await message.ack()
                    else:
                        await message.reject(requeue=False)  # Send to DLQ
                        logger.error(f"Message {queue_message.id} exceeded max retries")
                        
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await message.reject(requeue=False)
        
        await queue.consume(message_handler, no_ack=auto_ack)
        self.consumers[queue_name] = callback
        logger.info(f"Started consuming from queue '{queue_name}'")

    async def setup_trading_queues(self) -> None:
        """Setup standard trading platform queues"""
        
        # Trading queues
        await self.declare_queue("trade.execution", "trade.execute")
        await self.declare_queue("trade.orders", "trade.order.*")
        await self.declare_queue("trade.fills", "trade.fill")
        
        # Market data queues
        await self.declare_queue("market.prices", exchange_name="market.data")
        await self.declare_queue("market.orderbook", exchange_name="market.data")
        await self.declare_queue("market.trades", exchange_name="market.data")
        
        # Risk management queues
        await self.declare_queue("risk.alerts", "risk.alert", "risk.management")
        await self.declare_queue("risk.validation", "risk.validate", "risk.management")
        
        # Analytics queues
        await self.declare_queue("analytics.performance", "analytics.performance")
        await self.declare_queue("analytics.signals", "analytics.signal.*")
        
        # Notification queues
        await self.declare_queue("notifications.email", "notify.email", "notifications")
        await self.declare_queue("notifications.sms", "notify.sms", "notifications")
        await self.declare_queue("notifications.push", "notify.push", "notifications")
        
        logger.info("All trading queues setup completed")

    async def publish_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """Publish trade execution message"""
        message = QueueMessage(
            id=f"trade_{trade_data.get('id', 'unknown')}_{datetime.utcnow().timestamp()}",
            type="trade_execution",
            service="trading_engine",
            timestamp=datetime.utcnow(),
            data=trade_data,
            priority=8  # High priority for trades
        )
        await self.publish_message(message, routing_key="trade.execute")

    async def publish_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Publish market data update"""
        message = QueueMessage(
            id=f"market_{symbol}_{datetime.utcnow().timestamp()}",
            type="market_data",
            service="data_service",
            timestamp=datetime.utcnow(),
            data={"symbol": symbol, **data},
            priority=6
        )
        await self.publish_message(message, "market.data")

    async def publish_risk_alert(self, alert_data: Dict[str, Any]) -> None:
        """Publish risk management alert"""
        message = QueueMessage(
            id=f"risk_alert_{datetime.utcnow().timestamp()}",
            type="risk_alert",
            service="risk_service",
            timestamp=datetime.utcnow(),
            data=alert_data,
            priority=9  # Critical priority for risk alerts
        )
        await self.publish_message(message, "risk.management", "risk.alert")

    async def publish_notification(
        self,
        notification_type: str,
        recipient: str,
        content: Dict[str, Any]
    ) -> None:
        """Publish notification message"""
        message = QueueMessage(
            id=f"notification_{notification_type}_{datetime.utcnow().timestamp()}",
            type="notification",
            service="notification_service",
            timestamp=datetime.utcnow(),
            data={
                "type": notification_type,
                "recipient": recipient,
                "content": content
            },
            priority=4
        )
        await self.publish_message(
            message,
            "notifications",
            f"notify.{notification_type}"
        )

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues"""
        stats = {}
        
        for queue_name, queue in self.queues.items():
            try:
                # Get queue info
                info = await queue.channel.queue_declare(queue_name, passive=True)
                stats[queue_name] = {
                    "message_count": info.method.message_count,
                    "consumer_count": info.method.consumer_count,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for queue {queue_name}: {e}")
                stats[queue_name] = {"error": str(e)}
        
        return stats

    async def close(self) -> None:
        """Close connection and cleanup resources"""
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
            
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            
            self.is_connected = False
            logger.info("RabbitMQ connection closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")

# Singleton instance for global access
_message_queue_client: Optional[MessageQueueClient] = None

def get_message_queue_client() -> MessageQueueClient:
    """Get the global message queue client instance"""
    global _message_queue_client
    if _message_queue_client is None:
        raise RuntimeError("Message queue client not initialized")
    return _message_queue_client

async def initialize_message_queue(config: Dict[str, Any]) -> MessageQueueClient:
    """Initialize the global message queue client"""
    global _message_queue_client
    _message_queue_client = MessageQueueClient(config)
    await _message_queue_client.connect()
    await _message_queue_client.setup_trading_queues()
    return _message_queue_client

# Event handlers for common message types
class MessageHandlers:
    """Collection of standard message handlers for different services"""
    
    @staticmethod
    async def handle_trade_execution(message: QueueMessage) -> bool:
        """Handle trade execution messages"""
        try:
            trade_data = message.data
            logger.info(f"Processing trade execution: {trade_data}")
            
            # Add trade execution logic here
            # This would integrate with the trading engine
            
            return True
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
            return False

    @staticmethod
    async def handle_market_data(message: QueueMessage) -> bool:
        """Handle market data messages"""
        try:
            market_data = message.data
            logger.info(f"Processing market data for {market_data.get('symbol')}")
            
            # Add market data processing logic here
            # This would update caches, trigger strategies, etc.
            
            return True
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
            return False

    @staticmethod
    async def handle_risk_alert(message: QueueMessage) -> bool:
        """Handle risk management alerts"""
        try:
            alert_data = message.data
            logger.warning(f"Risk alert received: {alert_data}")
            
            # Add risk management logic here
            # This could trigger position closures, notifications, etc.
            
            return True
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
            return False

    @staticmethod
    async def handle_notification(message: QueueMessage) -> bool:
        """Handle notification messages"""
        try:
            notification_data = message.data
            logger.info(f"Sending {notification_data['type']} notification")
            
            # Add notification logic here
            # This would send emails, SMS, push notifications, etc.
            
            return True
        except Exception as e:
            logger.error(f"Error handling notification: {e}")
            return False
