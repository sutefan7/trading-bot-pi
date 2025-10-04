"""
Circuit Breaker
Implements circuit breaker pattern for broker API calls
"""
import time
import random
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

try:
    import functools
    FUNCTOOLS_AVAILABLE = True
except ImportError:
    FUNCTOOLS_AVAILABLE = False


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        success_threshold: int = 3,
        ml_overlay_manager=None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.ml_overlay_manager = ml_overlay_manager
        
        # State
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_calls += 1
        
        # Check if circuit should be opened
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker: attempting reset (half-open)")
            else:
                logger.warning("Circuit breaker: OPEN - failing fast")
                self._notify_circuit_open()
                raise Exception("Circuit breaker is OPEN")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
        except Exception as e:
            # Unexpected exception - don't count towards circuit breaker
            logger.error(f"Unexpected exception in circuit breaker: {e}")
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker: CLOSED - service recovered")
                self._notify_circuit_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker: OPEN - service still failing")
            self._notify_circuit_open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker: OPEN - {self.failure_count} failures")
                self._notify_circuit_open()
    
    def _notify_circuit_open(self):
        """Notify that circuit is open"""
        if self.ml_overlay_manager:
            try:
                # Disable ML overlay when circuit is open
                self.ml_overlay_manager.set_mode(self.ml_overlay_manager.MLOffset.OFF)
                logger.warning("ML overlay disabled due to circuit breaker")
            except Exception as e:
                logger.error(f"Error disabling ML overlay: {e}")
    
    def _notify_circuit_closed(self):
        """Notify that circuit is closed"""
        if self.ml_overlay_manager:
            try:
                # Re-enable ML overlay when circuit is closed
                # Note: This should be done carefully, maybe with manual intervention
                logger.info("Circuit closed - ML overlay can be re-enabled manually")
            except Exception as e:
                logger.error(f"Error handling circuit close: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'failure_rate': self.total_failures / max(self.total_calls, 1)
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
    success_threshold: int = 3,
    ml_overlay_manager=None
):
    """Decorator for circuit breaker"""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold,
            ml_overlay_manager=ml_overlay_manager
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        # Add circuit breaker methods to wrapper
        wrapper.circuit_breaker = circuit_breaker
        wrapper.get_circuit_state = circuit_breaker.get_state
        wrapper.reset_circuit = circuit_breaker.reset
        
        return wrapper
    
    return decorator


class ExponentialBackoff:
    """Exponential backoff implementation"""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0
    
    def wait(self):
        """Wait for next retry"""
        if self.attempt == 0:
            self.attempt += 1
            return
        
        delay = min(
            self.base_delay * (self.multiplier ** (self.attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter_amount = delay * 0.1
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        logger.info(f"Exponential backoff: waiting {delay:.2f}s (attempt {self.attempt})")
        time.sleep(delay)
        self.attempt += 1
    
    def reset(self):
        """Reset backoff counter"""
        self.attempt = 0


def with_exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    max_retries: int = 3
):
    """Decorator for exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backoff = ExponentialBackoff(
                base_delay=base_delay,
                max_delay=max_delay,
                multiplier=multiplier,
                jitter=jitter
            )
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        backoff.wait()
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise last_exception
            
            return None
        
        return wrapper
    
    return decorator


# Combined decorator for circuit breaker + exponential backoff
def with_resilience(
    circuit_breaker_config: Optional[Dict[str, Any]] = None,
    backoff_config: Optional[Dict[str, Any]] = None,
    ml_overlay_manager=None
):
    """Combined decorator for circuit breaker and exponential backoff"""
    
    # Default configurations
    cb_config = circuit_breaker_config or {
        'failure_threshold': 5,
        'recovery_timeout': 60.0,
        'expected_exception': Exception,
        'success_threshold': 3
    }
    cb_config['ml_overlay_manager'] = ml_overlay_manager
    
    bo_config = backoff_config or {
        'base_delay': 1.0,
        'max_delay': 60.0,
        'multiplier': 2.0,
        'jitter': True,
        'max_retries': 3
    }
    
    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker
        cb_func = with_circuit_breaker(**cb_config)(func)
        
        # Apply exponential backoff
        resilient_func = with_exponential_backoff(**bo_config)(cb_func)
        
        return resilient_func
    
    return decorator
