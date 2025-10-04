"""
Inference Client with Failover
Manages ML model loading, hot-reload, and failover for trading bot
"""
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from loguru import logger

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.predict import ModelPredictor, ModelBundle
from features.pipeline import FeaturePipeline


class ModelManager:
    """Manages ML model lifecycle with failover capabilities"""
    
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        latest_file: str = "latest.txt",
        health_check_interval: int = 300,  # 5 minutes
        reload_interval: int = 60,  # 1 minute
        max_failures: int = 3
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.latest_file = Path(latest_file)
        self.health_check_interval = health_check_interval
        self.reload_interval = reload_interval
        self.max_failures = max_failures
        
        # State
        self.current_bundle: Optional[ModelBundle] = None
        self.model_available = False
        self.failure_count = 0
        self.last_health_check = None
        self.last_reload_check = None
        self.current_version = None
        
        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Feature pipeline
        self.feature_pipeline = FeaturePipeline()
        
        # Callbacks
        self.on_model_loaded: Optional[Callable] = None
        self.on_model_failed: Optional[Callable] = None
        self.on_failover: Optional[Callable] = None
        
        logger.info("ModelManager initialized")
    
    def start(self):
        """Start model monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor thread already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # Initial load
        self._try_load_latest_model()
        
        logger.info("ModelManager started")
    
    def stop(self):
        """Stop model monitoring thread"""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        
        logger.info("ModelManager stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for new model versions
                if (self.last_reload_check is None or 
                    (current_time - self.last_reload_check).total_seconds() > self.reload_interval):
                    self._check_for_new_model()
                    self.last_reload_check = current_time
                
                # Perform health check
                if (self.last_health_check is None or 
                    (current_time - self.last_health_check).total_seconds() > self.health_check_interval):
                    self._perform_health_check()
                    self.last_health_check = current_time
                
                # Sleep for a short interval
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _try_load_latest_model(self):
        """Try to load the latest model"""
        try:
            if not self.latest_file.exists():
                logger.warning(f"Latest file not found: {self.latest_file}")
                return
            
            with open(self.latest_file, 'r') as f:
                latest_path = f.read().strip()
            
            if not latest_path:
                logger.warning("Latest file is empty")
                return
            
            # Construct full path to model directory
            model_dir = self.artifacts_dir / latest_path
            self._load_model(str(model_dir))
            
        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            self._handle_model_failure()
    
    def _check_for_new_model(self):
        """Check for new model versions"""
        try:
            if not self.latest_file.exists():
                return
            
            with open(self.latest_file, 'r') as f:
                latest_path = f.read().strip()
            
            if not latest_path:
                return
            
            # Check if this is a new version
            # Compare directory names, not full paths
            current_dir = Path(self.current_version).name if self.current_version else None
            if latest_path != current_dir:
                logger.info(f"New model version detected: {latest_path}")
                # Construct full path to model directory
                model_dir = self.artifacts_dir / latest_path
                self._load_model(str(model_dir))
            
        except Exception as e:
            logger.error(f"Error checking for new model: {e}")
    
    def _load_model(self, model_path: str):
        """Load model from specified path"""
        try:
            with self._lock:
                logger.info(f"Loading model from: {model_path}")
                
                # Create predictor and load model
                predictor = ModelPredictor()
                bundle = predictor.load_artifact(model_path)
                
                # Validate model
                health_result = predictor.health_check(bundle)
                if health_result["status"] != "healthy":
                    raise ValueError(f"Model health check failed: {health_result}")
                
                # Update state
                self.current_bundle = bundle
                self.predictor = predictor
                self.model_available = True
                self.failure_count = 0
                self.current_version = model_path
                
                logger.info(f"✅ Model loaded successfully: {bundle.version}")
                
                # Call callback
                if self.on_model_loaded:
                    try:
                        self.on_model_loaded(bundle)
                    except Exception as e:
                        logger.error(f"Error in on_model_loaded callback: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self._handle_model_failure()
    
    def _perform_health_check(self):
        """Perform health check on current model"""
        try:
            if not self.model_available or not self.current_bundle:
                return
            
            # Use existing predictor if available, otherwise create new one
            if hasattr(self, 'predictor') and self.predictor:
                predictor = self.predictor
            else:
                predictor = ModelPredictor()
                self.predictor = predictor
            
            health_result = predictor.health_check(self.current_bundle)
            
            if health_result["status"] != "healthy":
                logger.warning(f"Model health check failed: {health_result}")
                self._handle_model_failure()
            else:
                logger.debug(f"Model health check passed: {self.current_bundle.version}")
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            self._handle_model_failure()
    
    def _handle_model_failure(self):
        """Handle model failure with failover logic"""
        try:
            with self._lock:
                self.failure_count += 1
                logger.warning(f"Model failure #{self.failure_count}")
                
                # Call failure callback
                if self.on_model_failed:
                    try:
                        self.on_model_failed(self.failure_count)
                    except Exception as e:
                        logger.error(f"Error in on_model_failed callback: {e}")
                
                # Check if we should disable ML overlay
                if self.failure_count >= self.max_failures:
                    logger.error(f"Max failures ({self.max_failures}) reached - disabling ML overlay")
                    self.model_available = False
                    self.current_bundle = None
                    
                    # Call failover callback
                    if self.on_failover:
                        try:
                            self.on_failover()
                        except Exception as e:
                            logger.error(f"Error in on_failover callback: {e}")
                
        except Exception as e:
            logger.error(f"Error handling model failure: {e}")
    
    def get_prediction(self, symbol: str, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Get ML prediction for symbol and features
        
        Args:
            symbol: Trading symbol
            features: Feature dictionary
            
        Returns:
            Prediction result or None if model unavailable
        """
        try:
            if not self.model_available or not self.current_bundle:
                logger.debug(f"Model not available for {symbol}")
                return None
            
            # Convert features to numpy array
            feature_vector = self._features_to_vector(features)
            if feature_vector is None:
                logger.warning(f"Could not convert features to vector for {symbol}")
                return None
            
            # Make prediction
            predictor = ModelPredictor()
            result = predictor.predict_one(self.current_bundle, feature_vector)
            
            # Add symbol and timestamp
            result['symbol'] = symbol
            result['timestamp'] = datetime.now().isoformat()
            
            logger.debug(f"ML prediction for {symbol}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            self._handle_model_failure()
            return None
    
    def _features_to_vector(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Convert feature dictionary to numpy vector"""
        try:
            import numpy as np
            
            if not self.current_bundle:
                return None
            
            # Create vector in correct order
            feature_vector = []
            for feature_name in self.current_bundle.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    logger.warning(f"Missing feature: {feature_name}")
                    feature_vector.append(0.0)  # Fill with default value
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error converting features to vector: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        with self._lock:
            return {
                'model_available': self.model_available,
                'current_version': self.current_version,
                'failure_count': self.failure_count,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'last_reload_check': self.last_reload_check.isoformat() if self.last_reload_check else None,
                'model_info': self.get_model_info() if self.current_bundle else None
            }
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed model information"""
        if not self.current_bundle:
            return None
        
        predictor = ModelPredictor()
        return predictor.get_model_info(self.current_bundle)
    
    def force_reload(self):
        """Force reload of latest model"""
        logger.info("Forcing model reload")
        self._try_load_latest_model()
    
    def reset_failure_count(self):
        """Reset failure count (useful for manual intervention)"""
        with self._lock:
            self.failure_count = 0
            logger.info("Failure count reset")
    
    def set_callbacks(
        self,
        on_model_loaded: Optional[Callable] = None,
        on_model_failed: Optional[Callable] = None,
        on_failover: Optional[Callable] = None
    ):
        """Set callback functions"""
        self.on_model_loaded = on_model_loaded
        self.on_model_failed = on_model_failed
        self.on_failover = on_failover
        logger.info("Callbacks set")


# Convenience function for integration
def create_model_manager(
    artifacts_dir: str = "artifacts",
    latest_file: str = "latest.txt",
    use_ml_overlay: bool = True
) -> Optional[ModelManager]:
    """
    Create and configure model manager
    
    Args:
        artifacts_dir: Directory containing model artifacts
        latest_file: File containing path to latest model
        use_ml_overlay: Whether to enable ML overlay
        
    Returns:
        ModelManager instance or None if ML overlay disabled
    """
    if not use_ml_overlay:
        logger.info("ML overlay disabled")
        return None
    
    try:
        manager = ModelManager(artifacts_dir, latest_file)
        
        # Set up default callbacks
        def on_model_loaded(bundle: ModelBundle):
            logger.info(f"🎯 ML model loaded: {bundle.version}")
        
        def on_model_failed(failure_count: int):
            logger.warning(f"⚠️ ML model failure #{failure_count}")
        
        def on_failover():
            logger.error("🚨 ML overlay disabled - falling back to non-ML strategies")
        
        manager.set_callbacks(on_model_loaded, on_model_failed, on_failover)
        
        return manager
        
    except Exception as e:
        logger.error(f"Error creating model manager: {e}")
        return None
