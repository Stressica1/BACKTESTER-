"""
Notification Service

This service handles all notification operations including:
- Real-time alerts and notifications
- Email notifications
- SMS notifications
- Push notifications
- Slack/Discord integrations
- Notification preferences
- Alert templates
- Escalation policies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json

import aiohttp
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import jinja2
from twilio.rest import Client as TwilioClient
import slack_sdk
from discord_webhook import DiscordWebhook

from core.database import get_async_session, User
from core.redis_client import RedisClient
from core.message_queue import MessageQueueClient
from services.auth_service import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Notification Models
class NotificationType(str, Enum):
    TRADE_EXECUTION = "trade_execution"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_UPDATE = "portfolio_update"
    SYSTEM_ALERT = "system_alert"
    MARKET_NEWS = "market_news"
    PRICE_ALERT = "price_alert"
    STRATEGY_SIGNAL = "strategy_signal"
    ACCOUNT_ALERT = "account_alert"
    MAINTENANCE = "maintenance"

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBSOCKET = "websocket"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class NotificationMessage:
    id: str
    user_id: str
    type: NotificationType
    channel: NotificationChannel
    priority: Priority
    subject: str
    content: str
    data: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    status: NotificationStatus = NotificationStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

class NotificationPreferences(BaseModel):
    user_id: str
    email_enabled: bool = True
    sms_enabled: bool = False
    push_enabled: bool = True
    trade_notifications: bool = True
    risk_notifications: bool = True
    market_notifications: bool = False
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    channels_by_type: Dict[NotificationType, List[NotificationChannel]] = Field(default_factory=dict)

class NotificationTemplate(BaseModel):
    name: str
    type: NotificationType
    subject_template: str
    content_template: str
    variables: List[str]

class WebhookConfig(BaseModel):
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    method: str = "POST"
    retry_count: int = 3

class SlackConfig(BaseModel):
    webhook_url: str
    channel: str
    username: str = "TradingBot"

class DiscordConfig(BaseModel):
    webhook_url: str
    username: str = "TradingBot"

# Notification Service
class NotificationService:
    def __init__(self):
        self.redis_client = RedisClient()
        self.mq_client = MessageQueueClient()
        self.jinja_env = jinja2.Environment()
        self.active_websockets: Dict[str, WebSocket] = {}
        
        # Email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_username = ""  # Configure from environment
        self.smtp_password = ""  # Configure from environment
        
        # SMS configuration (Twilio)
        self.twilio_client = None
        self.twilio_from_number = ""  # Configure from environment
        
        # Push notification configuration
        self.fcm_server_key = ""  # Configure from environment
        
    async def initialize(self):
        """Initialize the notification service"""
        await self.redis_client.connect()
        await self.mq_client.connect()
        
        # Subscribe to notification queues
        await self.mq_client.subscribe_to_queue(
            "notifications", 
            self._handle_notification_message
        )
        
        await self.mq_client.subscribe_to_queue(
            "risk_alerts", 
            self._handle_risk_alert
        )
        
        await self.mq_client.subscribe_to_queue(
            "trade_executions", 
            self._handle_trade_notification
        )
        
        # Initialize Twilio client
        try:
            twilio_account_sid = ""  # Configure from environment
            twilio_auth_token = ""   # Configure from environment
            if twilio_account_sid and twilio_auth_token:
                self.twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token)
        except Exception as e:
            logger.warning(f"Twilio client initialization failed: {str(e)}")
        
        logger.info("Notification Service initialized")
    
    async def send_notification(
        self, 
        user_id: str,
        notification_type: NotificationType,
        subject: str,
        content: str,
        data: Dict[str, Any] = None,
        priority: Priority = Priority.MEDIUM,
        channels: List[NotificationChannel] = None,
        scheduled_at: datetime = None
    ) -> str:
        """Send notification to user"""
        try:
            # Get user preferences
            preferences = await self._get_user_preferences(user_id)
            
            # Determine channels to use
            if channels is None:
                channels = await self._get_channels_for_type(notification_type, preferences)
            
            # Check quiet hours
            if await self._is_quiet_hours(preferences):
                if priority not in [Priority.HIGH, Priority.CRITICAL]:
                    scheduled_at = await self._calculate_next_active_time(preferences)
            
            # Create notification message
            notification_id = f"notif_{user_id}_{int(datetime.utcnow().timestamp())}"
            
            notification = NotificationMessage(
                id=notification_id,
                user_id=user_id,
                type=notification_type,
                channel=channels[0] if channels else NotificationChannel.EMAIL,
                priority=priority,
                subject=subject,
                content=content,
                data=data or {},
                created_at=datetime.utcnow(),
                scheduled_at=scheduled_at
            )
            
            # Send to each channel
            for channel in channels:
                notification.channel = channel
                if scheduled_at and scheduled_at > datetime.utcnow():
                    # Schedule for later
                    await self._schedule_notification(notification)
                else:
                    # Send immediately
                    await self._send_notification_to_channel(notification)
            
            return notification_id
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            raise HTTPException(status_code=500, detail="Notification sending failed")
    
    async def send_bulk_notification(
        self,
        user_ids: List[str],
        notification_type: NotificationType,
        subject: str,
        content: str,
        data: Dict[str, Any] = None,
        priority: Priority = Priority.MEDIUM
    ) -> List[str]:
        """Send bulk notifications"""
        notification_ids = []
        
        for user_id in user_ids:
            try:
                notification_id = await self.send_notification(
                    user_id, notification_type, subject, content, data, priority
                )
                notification_ids.append(notification_id)
            except Exception as e:
                logger.error(f"Failed to send notification to user {user_id}: {str(e)}")
        
        return notification_ids
    
    async def register_websocket(self, user_id: str, websocket: WebSocket):
        """Register WebSocket connection for real-time notifications"""
        self.active_websockets[user_id] = websocket
        logger.info(f"Registered WebSocket for user {user_id}")
    
    async def unregister_websocket(self, user_id: str):
        """Unregister WebSocket connection"""
        if user_id in self.active_websockets:
            del self.active_websockets[user_id]
            logger.info(f"Unregistered WebSocket for user {user_id}")
    
    async def update_user_preferences(
        self, 
        user_id: str, 
        preferences: NotificationPreferences
    ):
        """Update user notification preferences"""
        try:
            await self.redis_client.set_json(
                f"notification_preferences:{user_id}",
                preferences.dict(),
                expire=None  # No expiration
            )
            
            logger.info(f"Updated notification preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update preferences")
    
    async def get_notification_history(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get notification history for user"""
        try:
            # Get notification keys
            keys = await self.redis_client.get_keys_by_pattern(
                f"notification_history:{user_id}:*"
            )
            
            # Get notifications
            notifications = []
            for key in keys[offset:offset + limit]:
                notification = await self.redis_client.get_json(key)
                if notification:
                    notifications.append(notification)
            
            # Sort by created_at
            notifications.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting notification history: {str(e)}")
            return []
    
    async def create_notification_template(
        self, 
        template: NotificationTemplate
    ):
        """Create notification template"""
        try:
            await self.redis_client.set_json(
                f"notification_template:{template.name}",
                template.dict(),
                expire=None
            )
            
            logger.info(f"Created notification template: {template.name}")
            
        except Exception as e:
            logger.error(f"Error creating template: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create template")
    
    async def send_template_notification(
        self,
        user_id: str,
        template_name: str,
        variables: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> str:
        """Send notification using template"""
        try:
            # Get template
            template_data = await self.redis_client.get_json(
                f"notification_template:{template_name}"
            )
            
            if not template_data:
                raise HTTPException(status_code=404, detail="Template not found")
            
            template = NotificationTemplate(**template_data)
            
            # Render templates
            subject_template = self.jinja_env.from_string(template.subject_template)
            content_template = self.jinja_env.from_string(template.content_template)
            
            subject = subject_template.render(**variables)
            content = content_template.render(**variables)
            
            # Send notification
            return await self.send_notification(
                user_id, template.type, subject, content, variables, priority
            )
            
        except Exception as e:
            logger.error(f"Error sending template notification: {str(e)}")
            raise HTTPException(status_code=500, detail="Template notification failed")
    
    async def _handle_notification_message(self, message: dict):
        """Handle notification messages from queue"""
        try:
            user_id = message.get("user_id")
            notification_type = message.get("type")
            subject = message.get("subject", "Notification")
            content = message.get("content", "")
            data = message.get("data", {})
            priority = Priority(message.get("priority", "medium"))
            
            await self.send_notification(
                user_id, NotificationType(notification_type), 
                subject, content, data, priority
            )
            
        except Exception as e:
            logger.error(f"Error handling notification message: {str(e)}")
    
    async def _handle_risk_alert(self, message: dict):
        """Handle risk alert messages"""
        try:
            alert = message.get("alert", {})
            user_id = alert.get("user_id")
            
            if user_id:
                subject = f"Risk Alert: {alert.get('risk_type', '').replace('_', ' ').title()}"
                content = f"""
                Risk Alert Generated:
                
                Type: {alert.get('risk_type', 'Unknown')}
                Level: {alert.get('level', 'Unknown')}
                Message: {alert.get('message', 'No details available')}
                Portfolio: {alert.get('portfolio_id', 'Unknown')}
                Current Value: {alert.get('current_value', 'N/A')}
                Threshold: {alert.get('threshold', 'N/A')}
                Suggested Action: {alert.get('suggested_action', 'Review portfolio')}
                
                Time: {alert.get('timestamp', datetime.utcnow().isoformat())}
                """
                
                priority = Priority.HIGH if alert.get('level') == 'critical' else Priority.MEDIUM
                
                await self.send_notification(
                    user_id, NotificationType.RISK_ALERT, 
                    subject, content, alert, priority
                )
        
        except Exception as e:
            logger.error(f"Error handling risk alert: {str(e)}")
    
    async def _handle_trade_notification(self, message: dict):
        """Handle trade execution notifications"""
        try:
            trade = message.get("trade", {})
            user_id = trade.get("user_id")
            
            if user_id:
                subject = f"Trade Executed: {trade.get('symbol', 'Unknown')}"
                content = f"""
                Trade Execution Notification:
                
                Symbol: {trade.get('symbol', 'Unknown')}
                Side: {trade.get('side', 'Unknown')}
                Quantity: {trade.get('quantity', 'Unknown')}
                Price: ${trade.get('price', 'Unknown')}
                Total Value: ${trade.get('total_value', 'Unknown')}
                Exchange: {trade.get('exchange', 'Unknown')}
                
                Time: {trade.get('timestamp', datetime.utcnow().isoformat())}
                """
                
                await self.send_notification(
                    user_id, NotificationType.TRADE_EXECUTION, 
                    subject, content, trade, Priority.MEDIUM
                )
        
        except Exception as e:
            logger.error(f"Error handling trade notification: {str(e)}")
    
    async def _send_notification_to_channel(self, notification: NotificationMessage):
        """Send notification to specific channel"""
        try:
            if notification.channel == NotificationChannel.EMAIL:
                await self._send_email(notification)
            elif notification.channel == NotificationChannel.SMS:
                await self._send_sms(notification)
            elif notification.channel == NotificationChannel.PUSH:
                await self._send_push_notification(notification)
            elif notification.channel == NotificationChannel.WEBSOCKET:
                await self._send_websocket_notification(notification)
            elif notification.channel == NotificationChannel.SLACK:
                await self._send_slack_notification(notification)
            elif notification.channel == NotificationChannel.DISCORD:
                await self._send_discord_notification(notification)
            elif notification.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(notification)
            
            # Store in history
            await self._store_notification_history(notification)
            
        except Exception as e:
            logger.error(f"Error sending to channel {notification.channel}: {str(e)}")
            await self._handle_notification_failure(notification, str(e))
    
    async def _send_email(self, notification: NotificationMessage):
        """Send email notification"""
        try:
            # Get user email
            user_email = await self._get_user_email(notification.user_id)
            
            if not user_email:
                raise Exception("User email not found")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = user_email
            msg['Subject'] = notification.subject
            
            msg.attach(MIMEText(notification.content, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                text = msg.as_string()
                server.sendmail(self.smtp_username, user_email, text)
            
            notification.status = NotificationStatus.SENT
            logger.info(f"Email sent to {user_email}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            raise
    
    async def _send_sms(self, notification: NotificationMessage):
        """Send SMS notification"""
        try:
            if not self.twilio_client:
                raise Exception("Twilio client not configured")
            
            # Get user phone
            user_phone = await self._get_user_phone(notification.user_id)
            
            if not user_phone:
                raise Exception("User phone not found")
            
            # Send SMS
            message = self.twilio_client.messages.create(
                body=f"{notification.subject}\n\n{notification.content}",
                from_=self.twilio_from_number,
                to=user_phone
            )
            
            notification.status = NotificationStatus.SENT
            logger.info(f"SMS sent to {user_phone}")
            
        except Exception as e:
            logger.error(f"SMS sending failed: {str(e)}")
            raise
    
    async def _send_push_notification(self, notification: NotificationMessage):
        """Send push notification"""
        try:
            # Get user FCM token
            fcm_token = await self._get_user_fcm_token(notification.user_id)
            
            if not fcm_token:
                raise Exception("FCM token not found")
            
            # Send push notification via FCM
            headers = {
                'Authorization': f'key={self.fcm_server_key}',
                'Content-Type': 'application/json',
            }
            
            payload = {
                'to': fcm_token,
                'notification': {
                    'title': notification.subject,
                    'body': notification.content,
                    'priority': 'high' if notification.priority in [Priority.HIGH, Priority.CRITICAL] else 'normal'
                },
                'data': notification.data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://fcm.googleapis.com/fcm/send',
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        notification.status = NotificationStatus.SENT
                        logger.info(f"Push notification sent to {notification.user_id}")
                    else:
                        raise Exception(f"FCM error: {response.status}")
            
        except Exception as e:
            logger.error(f"Push notification failed: {str(e)}")
            raise
    
    async def _send_websocket_notification(self, notification: NotificationMessage):
        """Send WebSocket notification"""
        try:
            if notification.user_id in self.active_websockets:
                websocket = self.active_websockets[notification.user_id]
                
                message = {
                    "type": "notification",
                    "id": notification.id,
                    "notification_type": notification.type,
                    "priority": notification.priority,
                    "subject": notification.subject,
                    "content": notification.content,
                    "data": notification.data,
                    "timestamp": notification.created_at.isoformat()
                }
                
                await websocket.send_json(message)
                notification.status = NotificationStatus.DELIVERED
                logger.info(f"WebSocket notification sent to {notification.user_id}")
            else:
                raise Exception("WebSocket not connected")
            
        except Exception as e:
            logger.error(f"WebSocket notification failed: {str(e)}")
            raise
    
    async def _send_slack_notification(self, notification: NotificationMessage):
        """Send Slack notification"""
        try:
            # Get Slack config for user
            slack_config = await self._get_user_slack_config(notification.user_id)
            
            if not slack_config:
                raise Exception("Slack configuration not found")
            
            webhook = DiscordWebhook(url=slack_config['webhook_url'])
            webhook.content = f"**{notification.subject}**\n{notification.content}"
            webhook.username = slack_config.get('username', 'TradingBot')
            
            response = webhook.execute()
            
            if response.status_code == 200:
                notification.status = NotificationStatus.SENT
                logger.info(f"Slack notification sent to {notification.user_id}")
            else:
                raise Exception(f"Slack error: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            raise
    
    async def _send_discord_notification(self, notification: NotificationMessage):
        """Send Discord notification"""
        try:
            # Get Discord config for user
            discord_config = await self._get_user_discord_config(notification.user_id)
            
            if not discord_config:
                raise Exception("Discord configuration not found")
            
            webhook = DiscordWebhook(url=discord_config['webhook_url'])
            webhook.content = f"**{notification.subject}**\n{notification.content}"
            webhook.username = discord_config.get('username', 'TradingBot')
            
            response = webhook.execute()
            
            if response.status_code == 200:
                notification.status = NotificationStatus.SENT
                logger.info(f"Discord notification sent to {notification.user_id}")
            else:
                raise Exception(f"Discord error: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Discord notification failed: {str(e)}")
            raise
    
    async def _send_webhook_notification(self, notification: NotificationMessage):
        """Send webhook notification"""
        try:
            # Get webhook config for user
            webhook_config = await self._get_user_webhook_config(notification.user_id)
            
            if not webhook_config:
                raise Exception("Webhook configuration not found")
            
            payload = {
                "notification": {
                    "id": notification.id,
                    "type": notification.type,
                    "priority": notification.priority,
                    "subject": notification.subject,
                    "content": notification.content,
                    "data": notification.data,
                    "timestamp": notification.created_at.isoformat()
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    webhook_config['method'],
                    webhook_config['url'],
                    headers=webhook_config.get('headers', {}),
                    json=payload
                ) as response:
                    if response.status in [200, 201, 202]:
                        notification.status = NotificationStatus.SENT
                        logger.info(f"Webhook notification sent to {notification.user_id}")
                    else:
                        raise Exception(f"Webhook error: {response.status}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {str(e)}")
            raise
    
    async def _get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences"""
        try:
            preferences_data = await self.redis_client.get_json(
                f"notification_preferences:{user_id}"
            )
            
            if preferences_data:
                return NotificationPreferences(**preferences_data)
            else:
                # Return default preferences
                return NotificationPreferences(user_id=user_id)
                
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return NotificationPreferences(user_id=user_id)
    
    async def _get_channels_for_type(
        self, 
        notification_type: NotificationType,
        preferences: NotificationPreferences
    ) -> List[NotificationChannel]:
        """Get notification channels for type based on preferences"""
        if notification_type in preferences.channels_by_type:
            return preferences.channels_by_type[notification_type]
        
        # Default channels based on preferences
        channels = []
        if preferences.email_enabled:
            channels.append(NotificationChannel.EMAIL)
        if preferences.push_enabled:
            channels.append(NotificationChannel.PUSH)
        
        return channels or [NotificationChannel.EMAIL]  # Always have at least email
    
    async def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is in quiet hours"""
        if not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False
        
        # Implementation for quiet hours check
        # This is simplified - would need proper timezone handling
        current_time = datetime.utcnow().time()
        return False  # Simplified for now
    
    async def _calculate_next_active_time(self, preferences: NotificationPreferences) -> datetime:
        """Calculate next active time after quiet hours"""
        # Simplified implementation
        return datetime.utcnow() + timedelta(hours=8)
    
    async def _schedule_notification(self, notification: NotificationMessage):
        """Schedule notification for later delivery"""
        try:
            # Store scheduled notification
            await self.redis_client.set_json(
                f"scheduled_notification:{notification.id}",
                notification.__dict__,
                expire=int((notification.scheduled_at - datetime.utcnow()).total_seconds())
            )
            
            logger.info(f"Scheduled notification {notification.id} for {notification.scheduled_at}")
            
        except Exception as e:
            logger.error(f"Error scheduling notification: {str(e)}")
    
    async def _store_notification_history(self, notification: NotificationMessage):
        """Store notification in history"""
        try:
            await self.redis_client.set_json(
                f"notification_history:{notification.user_id}:{notification.id}",
                {
                    "id": notification.id,
                    "type": notification.type,
                    "channel": notification.channel,
                    "priority": notification.priority,
                    "subject": notification.subject,
                    "content": notification.content,
                    "status": notification.status,
                    "created_at": notification.created_at.isoformat(),
                    "sent_at": datetime.utcnow().isoformat()
                },
                expire=86400 * 30  # 30 days
            )
            
        except Exception as e:
            logger.error(f"Error storing notification history: {str(e)}")
    
    async def _handle_notification_failure(self, notification: NotificationMessage, error: str):
        """Handle notification failure and retry logic"""
        try:
            notification.retry_count += 1
            notification.status = NotificationStatus.FAILED
            
            if notification.retry_count < notification.max_retries:
                # Schedule retry
                retry_delay = min(60 * (2 ** notification.retry_count), 3600)  # Exponential backoff
                notification.scheduled_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                notification.status = NotificationStatus.RETRY
                
                await self._schedule_notification(notification)
                logger.info(f"Scheduled retry {notification.retry_count} for notification {notification.id}")
            else:
                logger.error(f"Notification {notification.id} failed permanently: {error}")
            
        except Exception as e:
            logger.error(f"Error handling notification failure: {str(e)}")
    
    async def _get_user_email(self, user_id: str) -> Optional[str]:
        """Get user email from database"""
        # This would query the user database
        return "user@example.com"  # Placeholder
    
    async def _get_user_phone(self, user_id: str) -> Optional[str]:
        """Get user phone from database"""
        # This would query the user database
        return None  # Placeholder
    
    async def _get_user_fcm_token(self, user_id: str) -> Optional[str]:
        """Get user FCM token"""
        # This would query the user database or Redis
        return None  # Placeholder
    
    async def _get_user_slack_config(self, user_id: str) -> Optional[Dict]:
        """Get user Slack configuration"""
        return await self.redis_client.get_json(f"slack_config:{user_id}")
    
    async def _get_user_discord_config(self, user_id: str) -> Optional[Dict]:
        """Get user Discord configuration"""
        return await self.redis_client.get_json(f"discord_config:{user_id}")
    
    async def _get_user_webhook_config(self, user_id: str) -> Optional[Dict]:
        """Get user webhook configuration"""
        return await self.redis_client.get_json(f"webhook_config:{user_id}")

# FastAPI App
app = FastAPI(title="Notification Service", version="1.0.0")
notification_service = NotificationService()

@app.on_event("startup")
async def startup_event():
    await notification_service.initialize()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/notifications/send")
async def send_notification(
    user_id: str,
    notification_type: NotificationType,
    subject: str,
    content: str,
    priority: Priority = Priority.MEDIUM,
    data: Dict[str, Any] = None,
    current_user: User = Depends(get_current_user)
):
    """Send notification to user"""
    notification_id = await notification_service.send_notification(
        user_id, notification_type, subject, content, data, priority
    )
    return {"notification_id": notification_id}

@app.post("/notifications/bulk-send")
async def send_bulk_notification(
    user_ids: List[str],
    notification_type: NotificationType,
    subject: str,
    content: str,
    priority: Priority = Priority.MEDIUM,
    data: Dict[str, Any] = None,
    current_user: User = Depends(get_current_user)
):
    """Send bulk notifications"""
    notification_ids = await notification_service.send_bulk_notification(
        user_ids, notification_type, subject, content, data, priority
    )
    return {"notification_ids": notification_ids}

@app.websocket("/ws/notifications/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time notifications"""
    await websocket.accept()
    await notification_service.register_websocket(user_id, websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.ping()
    except Exception as e:
        logger.info(f"WebSocket disconnected for user {user_id}: {str(e)}")
    finally:
        await notification_service.unregister_websocket(user_id)

@app.post("/notifications/preferences")
async def update_preferences(
    preferences: NotificationPreferences,
    current_user: User = Depends(get_current_user)
):
    """Update notification preferences"""
    await notification_service.update_user_preferences(preferences.user_id, preferences)
    return {"status": "updated"}

@app.get("/notifications/history/{user_id}")
async def get_notification_history(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get notification history"""
    history = await notification_service.get_notification_history(user_id, limit, offset)
    return {"notifications": history}

@app.post("/notifications/templates")
async def create_template(
    template: NotificationTemplate,
    current_user: User = Depends(get_current_user)
):
    """Create notification template"""
    await notification_service.create_notification_template(template)
    return {"status": "created"}

@app.post("/notifications/send-template")
async def send_template_notification(
    user_id: str,
    template_name: str,
    variables: Dict[str, Any],
    priority: Priority = Priority.MEDIUM,
    current_user: User = Depends(get_current_user)
):
    """Send notification using template"""
    notification_id = await notification_service.send_template_notification(
        user_id, template_name, variables, priority
    )
    return {"notification_id": notification_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
