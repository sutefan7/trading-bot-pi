"""
Notification Manager
Handles operational notifications for ML overlay events
"""
import json
import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

try:
    from telegram_notifier import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("Telegram notifier not available")


class NotificationManager:
    """Manages operational notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.telegram_config = config.get('telegram', {})
        self.webhook_config = config.get('webhook', {})
        
        # Initialize notifiers
        self.telegram_notifier = None
        if self.enabled and TELEGRAM_AVAILABLE:
            try:
                # Get credentials from environment variables first, then config
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN') or self.telegram_config.get('bot_token')
                chat_id = os.getenv('TELEGRAM_CHAT_ID') or self.telegram_config.get('chat_id')
                
                if bot_token and chat_id:
                    self.telegram_notifier = TelegramNotifier(
                        bot_token=bot_token,
                        chat_id=chat_id
                    )
                    logger.info("Telegram notifier initialized")
                else:
                    logger.warning("Telegram credentials not found in environment or config")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram notifier: {e}")
        
        # Event filters
        self.event_filters = {
            'model_loaded': config.get('events', {}).get('model_loaded', True),
            'model_failed': config.get('events', {}).get('model_failed', True),
            'ml_overlay_disabled': config.get('events', {}).get('ml_overlay_disabled', True),
            'daily_loss_cap': config.get('events', {}).get('daily_loss_cap', True),
            'rollback': config.get('events', {}).get('rollback', True),
            'performance_alert': config.get('events', {}).get('performance_alert', True)
        }
        
        # Rate limiting
        self.last_notifications: Dict[str, datetime] = {}
        self.rate_limit_minutes = config.get('rate_limit_minutes', 5)
        
        logger.info(f"Notification manager initialized: enabled={self.enabled}")
    
    def send_notification(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Send notification for specific event type"""
        if not self.enabled:
            return
        
        if not self.event_filters.get(event_type, False):
            logger.debug(f"Notification filtered for event: {event_type}")
            return
        
        # Rate limiting
        if self._is_rate_limited(event_type):
            logger.debug(f"Notification rate limited for event: {event_type}")
            return
        
        # Format message
        formatted_message = self._format_message(event_type, message, data)
        
        # Send via Telegram
        if self.telegram_notifier:
            try:
                self.telegram_notifier.send_message(formatted_message)
                self.last_notifications[event_type] = datetime.now()
                logger.info(f"Notification sent: {event_type}")
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
        
        # Send via webhook
        if self.webhook_config.get('url'):
            try:
                self._send_webhook(event_type, formatted_message, data)
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")
    
    def _format_message(self, event_type: str, message: str, data: Optional[Dict[str, Any]]) -> str:
        """Format notification message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Event type emojis
        emojis = {
            'model_loaded': '🎯',
            'model_failed': '❌',
            'ml_overlay_disabled': '🚨',
            'daily_loss_cap': '💰',
            'rollback': '⏪',
            'performance_alert': '⚡'
        }
        
        emoji = emojis.get(event_type, '📢')
        
        # Base message
        formatted = f"{emoji} **Trading Bot Alert**\n"
        formatted += f"**Time:** {timestamp}\n"
        formatted += f"**Event:** {event_type.replace('_', ' ').title()}\n"
        formatted += f"**Message:** {message}\n"
        
        # Add data if available
        if data:
            formatted += "\n**Details:**\n"
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                formatted += f"• {key}: {value}\n"
        
        return formatted
    
    def _send_webhook(self, event_type: str, message: str, data: Optional[Dict[str, Any]]):
        """Send webhook notification"""
        webhook_data = {
            'event_type': event_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        response = requests.post(
            self.webhook_config['url'],
            json=webhook_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Webhook notification sent: {event_type}")
        else:
            logger.error(f"Webhook failed: {response.status_code} - {response.text}")
    
    def _is_rate_limited(self, event_type: str) -> bool:
        """Check if event type is rate limited"""
        if event_type not in self.last_notifications:
            return False
        
        last_time = self.last_notifications[event_type]
        time_diff = (datetime.now() - last_time).total_seconds() / 60
        
        return time_diff < self.rate_limit_minutes
    
    # Convenience methods for specific events
    def notify_model_loaded(self, model_version: str, symbols: List[str]):
        """Notify model loaded"""
        self.send_notification(
            'model_loaded',
            f"New ML model loaded: {model_version}",
            {
                'model_version': model_version,
                'symbols': symbols,
                'status': 'active'
            }
        )
    
    def notify_model_failed(self, failure_count: int, error_message: str):
        """Notify model failure"""
        self.send_notification(
            'model_failed',
            f"ML model failure #{failure_count}",
            {
                'failure_count': failure_count,
                'error': error_message,
                'status': 'degraded'
            }
        )
    
    def notify_ml_overlay_disabled(self, reason: str, daily_pnl: Optional[float] = None):
        """Notify ML overlay disabled"""
        data = {'reason': reason, 'status': 'disabled'}
        if daily_pnl is not None:
            data['daily_pnl'] = f"{daily_pnl:.2%}"
        
        self.send_notification(
            'ml_overlay_disabled',
            f"ML overlay disabled: {reason}",
            data
        )
    
    def notify_daily_loss_cap(self, daily_pnl: float, cap_threshold: float):
        """Notify daily loss cap triggered"""
        self.send_notification(
            'daily_loss_cap',
            f"Daily loss cap triggered: {daily_pnl:.2%} > {cap_threshold:.2%}",
            {
                'daily_pnl': f"{daily_pnl:.2%}",
                'cap_threshold': f"{cap_threshold:.2%}",
                'status': 'critical'
            }
        )
    
    def notify_rollback(self, from_version: str, to_version: str):
        """Notify rollback"""
        self.send_notification(
            'rollback',
            f"Model rollback: {from_version} → {to_version}",
            {
                'from_version': from_version,
                'to_version': to_version,
                'status': 'rolled_back'
            }
        )
    
    def notify_performance_alert(self, metric: str, value: float, threshold: float):
        """Notify performance alert"""
        self.send_notification(
            'performance_alert',
            f"Performance alert: {metric} = {value} (threshold: {threshold})",
            {
                'metric': metric,
                'value': value,
                'threshold': threshold,
                'status': 'warning'
            }
        )


# Convenience function
def create_notification_manager(config: Dict[str, Any]) -> Optional[NotificationManager]:
    """Create notification manager with configuration"""
    if not config.get('enabled', False):
        return None
    
    try:
        return NotificationManager(config)
    except Exception as e:
        logger.error(f"Failed to create notification manager: {e}")
        return None
