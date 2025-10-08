"""
ML Overlay Manager
Handles shadow/live/off modes with comprehensive logging and guardrails
"""
import hashlib
import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class MLOffset(Enum):
    """ML Overlay modes"""
    SHADOW = "shadow"  # Calculate but don't execute
    LIVE = "live"      # Execute ML decisions
    OFF = "off"        # Skip ML entirely


@dataclass
class MLDecision:
    """ML decision with metadata"""
    symbol: str
    ml_proba: float
    ml_buy: bool
    ml_sell: bool
    ml_confidence: float
    model_version: str
    features_hash: str
    timestamp: datetime
    mode: MLOffset
    executed: bool = False
    non_ml_signal: Optional[Dict[str, Any]] = None


class MLOverlayManager:
    """Manages ML overlay with shadow/live/off modes and guardrails"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = MLOffset(config.get('mode', 'off'))
        self.min_confidence_global = config.get('min_confidence_global', 0.55)
        self.per_pair_thresholds = config.get('per_pair_thresholds', True)
        self.daily_loss_cap = config.get('daily_loss_cap', 0.05)
        
        # Guardrails
        self.max_failures_per_5min = config.get('max_failures_per_5min', 3)
        self.failure_timestamps: List[datetime] = []
        self.daily_loss_triggered = False
        self.last_daily_loss_check = None
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.max_inference_history = 100
        
        # Notifications
        self.notifications_enabled = config.get('notifications', {}).get('enabled', False)
        self.notification_callbacks: List[callable] = []
        
        logger.info(f"ML Overlay Manager initialized: mode={self.mode.value}")
    
    def set_mode(self, mode: MLOffset):
        """Set ML overlay mode"""
        old_mode = self.mode
        self.mode = mode
        logger.info(f"ML Overlay mode changed: {old_mode.value} → {mode.value}")
        
        # Send notification
        if self.notifications_enabled:
            self._send_notification(
                f"ML Overlay mode changed to {mode.value}",
                {"old_mode": old_mode.value, "new_mode": mode.value}
            )
    
    def should_use_ml(self) -> bool:
        """Check if ML should be used based on mode and guardrails"""
        if self.mode == MLOffset.OFF:
            return False
        
        # Check daily loss cap
        if self._check_daily_loss_cap():
            if self.mode != MLOffset.OFF:
                logger.warning("🚨 Daily loss cap triggered - disabling ML overlay")
                self.set_mode(MLOffset.OFF)
                self._send_notification(
                    "ML Overlay disabled due to daily loss cap",
                    {"daily_loss_cap": self.daily_loss_cap}
                )
            return False
        
        # Check failure rate
        if self._check_failure_rate():
            if self.mode != MLOffset.OFF:
                logger.warning("🚨 High failure rate - disabling ML overlay")
                self.set_mode(MLOffset.OFF)
                self._send_notification(
                    "ML Overlay disabled due to high failure rate",
                    {"failure_count": len(self.failure_timestamps)}
                )
            return False
        
        return True
    
    def process_ml_decision(
        self,
        symbol: str,
        ml_result: Optional[Dict[str, Any]],
        non_ml_signals: List[Dict[str, Any]],
        model_version: str,
        features: np.ndarray
    ) -> MLDecision:
        """Process ML decision based on current mode"""
        
        start_time = time.time()
        
        # Calculate features hash for parity tracking
        features_hash = self._calculate_features_hash(features)
        
        # Initialize decision
        decision = MLDecision(
            symbol=symbol,
            ml_proba=0.0,
            ml_buy=False,
            ml_sell=False,
            ml_confidence=0.0,
            model_version=model_version,
            features_hash=features_hash,
            timestamp=datetime.now(),
            mode=self.mode,
            non_ml_signal=non_ml_signals[0] if non_ml_signals else None
        )
        
        # Process based on mode
        if self.mode == MLOffset.OFF or not self.should_use_ml():
            decision.executed = False
            self._log_decision(decision, "ML_OFF")
            return decision
        
        if ml_result is None:
            decision.executed = False
            self._log_decision(decision, "ML_UNAVAILABLE")
            return decision
        
        # Extract ML results
        decision.ml_proba = ml_result.get('proba', 0.0)
        decision.ml_buy = ml_result.get('buy', False)
        decision.ml_sell = ml_result.get('sell', False)
        decision.ml_confidence = ml_result.get('confidence', 0.0)
        
        # Apply confidence threshold
        if decision.ml_confidence < self.min_confidence_global:
            decision.executed = False
            self._log_decision(decision, "LOW_CONFIDENCE")
            return decision
        
        # Mode-specific processing
        if self.mode == MLOffset.SHADOW:
            decision.executed = False
            self._log_decision(decision, "SHADOW_MODE")
            
        elif self.mode == MLOffset.LIVE:
            decision.executed = True
            self._log_decision(decision, "LIVE_EXECUTED")
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self._track_inference_time(inference_time)
        
        return decision
    
    def _check_daily_loss_cap(self) -> bool:
        """Check if daily loss cap has been triggered"""
        # This would integrate with your risk manager
        # For now, return False - implement based on your risk manager
        return False
    
    def _check_failure_rate(self) -> bool:
        """Check if failure rate is too high"""
        now = datetime.now()
        
        # Remove old failures (older than 5 minutes)
        self.failure_timestamps = [
            ts for ts in self.failure_timestamps 
            if (now - ts).total_seconds() < 300
        ]
        
        return len(self.failure_timestamps) >= self.max_failures_per_5min
    
    def record_failure(self):
        """Record a failure for guardrail tracking"""
        self.failure_timestamps.append(datetime.now())
        logger.warning(f"ML inference failure recorded. Total failures in 5min: {len(self.failure_timestamps)}")
    
    def _calculate_features_hash(self, features: np.ndarray) -> str:
        """Calculate hash of features for parity tracking"""
        if not NUMPY_AVAILABLE:
            return "numpy_unavailable"
        
        try:
            # Convert to bytes and hash
            features_bytes = features.tobytes()
            return hashlib.md5(features_bytes).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Error calculating features hash: {e}")
            return "hash_error"
    
    def _track_inference_time(self, inference_time_ms: float):
        """Track inference time for performance monitoring"""
        self.inference_times.append(inference_time_ms)
        
        # Keep only recent times
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times = self.inference_times[-self.max_inference_history:]
    
    def _log_decision(self, decision: MLDecision, reason: str):
        """Log ML decision with structured logging"""
        log_data = {
            "symbol": decision.symbol,
            "mode": decision.mode.value,
            "ml_proba": decision.ml_proba,
            "ml_buy": decision.ml_buy,
            "ml_sell": decision.ml_sell,
            "ml_confidence": decision.ml_confidence,
            "model_version": decision.model_version,
            "features_hash": decision.features_hash,
            "executed": decision.executed,
            "reason": reason,
            "timestamp": decision.timestamp.isoformat()
        }
        
        if decision.non_ml_signal:
            log_data["non_ml_signal"] = decision.non_ml_signal
        
        logger.info(f"ML Decision: {reason}", extra=log_data)
    
    def _send_notification(self, message: str, data: Dict[str, Any]):
        """Send notification via registered callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(message, data)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    def add_notification_callback(self, callback: callable):
        """Add notification callback"""
        self.notification_callbacks.append(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.inference_times:
            return {"inference_times": "no_data"}
        
        times = self.inference_times
        return {
            "p50_latency_ms": np.percentile(times, 50),
            "p95_latency_ms": np.percentile(times, 95),
            "p99_latency_ms": np.percentile(times, 99),
            "avg_latency_ms": np.mean(times),
            "max_latency_ms": np.max(times),
            "sample_count": len(times),
            "failure_count_5min": len(self.failure_timestamps),
            "daily_loss_triggered": self.daily_loss_triggered
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "mode": self.mode.value,
            "min_confidence_global": self.min_confidence_global,
            "per_pair_thresholds": self.per_pair_thresholds,
            "daily_loss_cap": self.daily_loss_cap,
            "should_use_ml": self.should_use_ml(),
            "performance": self.get_performance_metrics()
        }


# Convenience function
def create_ml_overlay_manager(config: Dict[str, Any]) -> MLOverlayManager:
    """Create ML overlay manager with configuration"""
    return MLOverlayManager(config)
