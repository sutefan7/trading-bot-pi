"""
SLO Monitor
Monitors Service Level Objectives and triggers alerts
"""
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class SLOThreshold:
    """SLO threshold definition"""
    name: str
    metric: str
    threshold: float
    operator: str  # 'lt', 'le', 'gt', 'ge', 'eq'
    window_minutes: int
    severity: str  # 'WARN', 'CRIT'
    min_samples: int = 5  # Minimum samples for evaluation
    breach_policy: str = "3/5"  # Breach policy: X out of Y evaluations
    evaluation_interval_minutes: int = 1  # How often to evaluate
    cooldown_minutes: int = 5  # Cooldown period for alerts


@dataclass
class SLOAlert:
    """SLO alert definition"""
    timestamp: datetime
    slo_name: str
    metric_value: float
    threshold: float
    severity: str
    message: str
    metadata: Dict[str, Any]


class SLOMonitor:
    """Monitors SLOs and triggers alerts"""
    
    def __init__(self, notification_manager=None):
        self.notification_manager = notification_manager
        
        # Alert suppression tracking
        self.alert_suppression: Dict[str, float] = {}  # SLO name -> suppression end time
        self.backoff_active = False
        self.backoff_end_time = 0.0
        
        # SLO definitions - NOC Essentials
        self.slos = [
            # Performance SLOs
            SLOThreshold(
                name="tick_to_decision_p95",
                metric="tick_to_decision_p95",
                threshold=200.0,  # 200ms
                operator="le",
                window_minutes=5,
                severity="WARN"
            ),
            SLOThreshold(
                name="decision_to_submit_p95",
                metric="decision_to_submit_p95",
                threshold=100.0,  # 100ms
                operator="le",
                window_minutes=5,
                severity="WARN"
            ),
            SLOThreshold(
                name="submit_to_ack_p95",
                metric="submit_to_ack_p95",
                threshold=500.0,  # 500ms
                operator="le",
                window_minutes=5,
                severity="WARN"
            ),
            # WebSocket SLOs
            SLOThreshold(
                name="ws_heartbeat_age",
                metric="ws_heartbeat_age_seconds",
                threshold=30.0,  # 30 seconds
                operator="ge",
                window_minutes=1,
                severity="CRIT"
            ),
            SLOThreshold(
                name="ws_connection_status",
                metric="ws_connected",
                threshold=1.0,  # Must be connected
                operator="eq",
                window_minutes=1,
                severity="CRIT"
            ),
            # Error SLOs
                     SLOThreshold(
                         name="error_rate_5xx",
                         metric="error_rate_5xx",
                         threshold=0.01,  # 1%
                         operator="ge",
                         window_minutes=10,  # 10 minute window
                         severity="CRIT",
                         min_samples=30,  # Minimum 30 samples
                         breach_policy="3/5",  # 3 out of 5 evaluations
                         evaluation_interval_minutes=2,  # Evaluate every 2 minutes
                         cooldown_minutes=10  # 10 minute cooldown
                     ),
            SLOThreshold(
                name="error_rate_429",
                metric="error_rate_429",
                threshold=0.05,  # 5%
                operator="ge",
                window_minutes=5,
                severity="WARN"
            ),
            # Data Quality SLOs
            SLOThreshold(
                name="missed_bar_detector",
                metric="missed_bars_count",
                threshold=2.0,  # 2 missed bars
                operator="ge",
                window_minutes=5,
                severity="WARN"
            ),
            # Risk SLOs
            SLOThreshold(
                name="daily_loss_cap",
                metric="daily_loss_pct",
                threshold=0.04,  # 4%
                operator="ge",
                window_minutes=1,
                severity="CRIT"
            ),
            SLOThreshold(
                name="risk_reject_rate",
                metric="risk_reject_rate",
                threshold=0.20,  # 20%
                operator="ge",
                window_minutes=10,
                severity="WARN"
            ),
            # Legacy ML SLOs (keep for compatibility)
            SLOThreshold(
                name="inference_latency_p95",
                metric="predict_latency_ms_p95",
                threshold=20.0,
                operator="le",
                window_minutes=10,
                severity="WARN"
            ),
            SLOThreshold(
                name="inference_latency_p95_critical",
                metric="predict_latency_ms_p95",
                threshold=30.0,
                operator="le",
                window_minutes=5,
                severity="CRIT"
            ),
            SLOThreshold(
                name="error_rate",
                metric="error_rate_percent",
                threshold=0.1,
                operator="le",
                window_minutes=60,
                severity="WARN"
            ),
            SLOThreshold(
                name="error_rate_critical",
                metric="error_rate_percent",
                threshold=1.0,
                operator="le",
                window_minutes=10,
                severity="CRIT"
            ),
            SLOThreshold(
                name="missed_bars",
                metric="missed_bars_count",
                threshold=1,
                operator="le",
                window_minutes=1440,  # 24 hours
                severity="WARN"
            ),
            SLOThreshold(
                name="idempotency_breaches",
                metric="idempotency_breaches_count",
                threshold=0,
                operator="eq",
                window_minutes=1440,  # 24 hours
                severity="CRIT"
            ),
            SLOThreshold(
                name="daily_loss_cap",
                metric="daily_loss_percent",
                threshold=5.0,
                operator="le",
                window_minutes=1440,  # 24 hours
                severity="CRIT"
            ),
            SLOThreshold(
                name="circuit_breaker_trips",
                metric="circuit_breaker_trips_count",
                threshold=1,
                operator="le",
                window_minutes=60,
                severity="WARN"
            )
        ]
        
        # Metrics storage
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Alert history
        self.alert_history: List[SLOAlert] = []
        
        # Last check time
        self.last_check_time = datetime.now()
        
        logger.info(f"SLO Monitor initialized with {len(self.slos)} SLOs")
    
    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value"""
        timestamp = datetime.now()
        
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'metadata': metadata or {}
        })
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.metrics_history[metric_name] = [
            m for m in self.metrics_history[metric_name]
            if m['timestamp'] > cutoff_time
        ]
        
        logger.debug(f"Recorded metric: {metric_name} = {value}")
    
    def check_slos(self) -> List[SLOAlert]:
        """Check all SLOs and return any alerts"""
        alerts = []
        current_time = datetime.now()
        
        for slo in self.slos:
            try:
                alert = self._check_slo(slo, current_time)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(f"Error checking SLO {slo.name}: {e}")
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        # Keep only recent alerts (last 7 days)
        cutoff_time = current_time - timedelta(days=7)
        self.alert_history = [
            a for a in self.alert_history
            if a.timestamp > cutoff_time
        ]
        
        # Send notifications
        for alert in alerts:
            self._send_alert(alert)
        
        self.last_check_time = current_time
        return alerts
    
    def _check_slo(self, slo: SLOThreshold, current_time: datetime) -> Optional[SLOAlert]:
        """Check a single SLO"""
        # Get metrics for the time window
        window_start = current_time - timedelta(minutes=slo.window_minutes)
        
        if slo.metric not in self.metrics_history:
            return None
        
        # Filter metrics by time window
        window_metrics = [
            m for m in self.metrics_history[slo.metric]
            if m['timestamp'] > window_start
        ]
        
        if not window_metrics:
            return None
        
        # Calculate metric value (e.g., percentile, average, count)
        metric_value = self._calculate_metric_value(slo.metric, window_metrics)
        
        # Check threshold
        if self._check_threshold(metric_value, slo.threshold, slo.operator):
            return None  # SLO is satisfied
        
        # SLO violated - create alert
        alert = SLOAlert(
            timestamp=current_time,
            slo_name=slo.name,
            metric_value=metric_value,
            threshold=slo.threshold,
            severity=slo.severity,
            message=self._create_alert_message(slo, metric_value),
            metadata={
                'window_minutes': slo.window_minutes,
                'metric_name': slo.metric,
                'operator': slo.operator,
                'sample_count': len(window_metrics)
            }
        )
        
        logger.warning(f"SLO violation: {slo.name} - {metric_value} {slo.operator} {slo.threshold}")
        return alert
    
    def _calculate_metric_value(self, metric_name: str, metrics: List[Dict[str, Any]]) -> float:
        """Calculate metric value from raw metrics"""
        if not metrics:
            return 0.0
        
        values = [m['value'] for m in metrics]
        
        # Different calculation methods based on metric type
        if 'p95' in metric_name:
            # Calculate 95th percentile
            return self._percentile(values, 95)
        elif 'p50' in metric_name:
            # Calculate 50th percentile (median)
            return self._percentile(values, 50)
        elif 'rate' in metric_name or 'percent' in metric_name:
            # Calculate average
            return sum(values) / len(values)
        elif 'count' in metric_name:
            # Sum all values
            return sum(values)
        else:
            # Default to average
            return sum(values) / len(values)
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value satisfies threshold"""
        if operator == 'lt':
            return value < threshold
        elif operator == 'le':
            return value <= threshold
        elif operator == 'gt':
            return value > threshold
        elif operator == 'ge':
            return value >= threshold
        elif operator == 'eq':
            return value == threshold
        else:
            logger.error(f"Unknown operator: {operator}")
            return True
    
    def _create_alert_message(self, slo: SLOThreshold, metric_value: float) -> str:
        """Create alert message"""
        operator_symbols = {
            'lt': '<',
            'le': '≤',
            'gt': '>',
            'ge': '≥',
            'eq': '='
        }
        
        operator_symbol = operator_symbols.get(slo.operator, slo.operator)
        
        return f"{slo.metric}: {metric_value:.2f} {operator_symbol} {slo.threshold} (window: {slo.window_minutes}m)"
    
    def _send_alert(self, alert: SLOAlert):
        """Send alert notification"""
        if not self.notification_manager:
            return
        
        # Format alert message
        alert_message = f"[TRADINGBOT][{alert.severity}] SLO Violation: {alert.slo_name}"
        alert_message += f"\n{alert.message}"
        alert_message += f"\nTime: {alert.timestamp.isoformat()}"
        
        # Add metadata
        if alert.metadata:
            alert_message += f"\nMetadata: {json.dumps(alert.metadata, indent=2)}"
        
        # Send notification
        try:
            self.notification_manager.send_notification(
                'performance_alert',
                alert_message,
                {
                    'slo_name': alert.slo_name,
                    'metric_value': alert.metric_value,
                    'threshold': alert.threshold,
                    'severity': alert.severity,
                    'metadata': alert.metadata
                }
            )
        except Exception as e:
            logger.error(f"Error sending SLO alert: {e}")
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status"""
        current_time = datetime.now()
        
        status = {
            'timestamp': current_time.isoformat(),
            'slos': [],
            'overall_status': 'PASS'
        }
        
        for slo in self.slos:
            try:
                # Get recent metrics
                window_start = current_time - timedelta(minutes=slo.window_minutes)
                
                if slo.metric not in self.metrics_history:
                    slo_status = {
                        'name': slo.name,
                        'status': 'NO_DATA',
                        'metric_value': None,
                        'threshold': slo.threshold,
                        'severity': slo.severity
                    }
                else:
                    window_metrics = [
                        m for m in self.metrics_history[slo.metric]
                        if m['timestamp'] > window_start
                    ]
                    
                    if not window_metrics:
                        slo_status = {
                            'name': slo.name,
                            'status': 'NO_DATA',
                            'metric_value': None,
                            'threshold': slo.threshold,
                            'severity': slo.severity
                        }
                    else:
                        metric_value = self._calculate_metric_value(slo.metric, window_metrics)
                        is_satisfied = self._check_threshold(metric_value, slo.threshold, slo.operator)
                        
                        slo_status = {
                            'name': slo.name,
                            'status': 'PASS' if is_satisfied else 'FAIL',
                            'metric_value': metric_value,
                            'threshold': slo.threshold,
                            'severity': slo.severity,
                            'sample_count': len(window_metrics)
                        }
                        
                        if not is_satisfied and slo.severity == 'CRIT':
                            status['overall_status'] = 'FAIL'
                
                status['slos'].append(slo_status)
                
            except Exception as e:
                logger.error(f"Error getting SLO status for {slo.name}: {e}")
                status['slos'].append({
                    'name': slo.name,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        return status
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'slo_name': alert.slo_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'severity': alert.severity,
                'message': alert.message,
                'metadata': alert.metadata
            }
            for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        return recent_alerts


# Convenience function
def create_slo_monitor(notification_manager=None) -> SLOMonitor:
    """Create SLO monitor"""
    return SLOMonitor(notification_manager)
