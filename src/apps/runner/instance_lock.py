"""
Single Instance Lock
Prevents multiple instances of the trading bot from running simultaneously
"""
import os
import fcntl
import atexit
import signal
import sys
from pathlib import Path
from typing import Optional
from loguru import logger


class InstanceLock:
    """Single instance lock using PID file"""
    
    def __init__(self, lock_file: str = "/tmp/tradingbot.lock"):
        self.lock_file = Path(lock_file)
        self.lock_fd: Optional[int] = None
        
    def acquire(self) -> bool:
        """Acquire the instance lock"""
        try:
            # Create lock file if it doesn't exist
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open lock file
            self.lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            
            # Try to acquire exclusive lock (non-blocking)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write current PID
            os.write(self.lock_fd, str(os.getpid()).encode())
            os.fsync(self.lock_fd)
            
            # Register cleanup
            atexit.register(self.release)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            logger.info(f"Instance lock acquired: {self.lock_file}")
            return True
            
        except (OSError, IOError) as e:
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
            
            # Check if another instance is running
            if self._is_other_instance_running():
                logger.error("Another instance of trading bot is already running")
                return False
            else:
                logger.error(f"Failed to acquire instance lock: {e}")
                return False
    
    def release(self):
        """Release the instance lock"""
        try:
            if self.lock_fd is not None:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = None
            
            if self.lock_file.exists():
                self.lock_file.unlink()
            
            logger.info("Instance lock released")
            
        except Exception as e:
            logger.error(f"Error releasing instance lock: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, releasing lock...")
        self.release()
        sys.exit(0)
    
    def _is_other_instance_running(self) -> bool:
        """Check if another instance is running"""
        try:
            if not self.lock_file.exists():
                return False
            
            # Read PID from lock file
            with open(self.lock_file, 'r') as f:
                pid_str = f.read().strip()
            
            if not pid_str.isdigit():
                return False
            
            pid = int(pid_str)
            
            # Check if process is running
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                return True
            except OSError:
                # Process doesn't exist, remove stale lock file
                self.lock_file.unlink()
                return False
                
        except Exception as e:
            logger.error(f"Error checking other instance: {e}")
            return False


def require_single_instance(lock_file: str = "/tmp/tradingbot.lock") -> InstanceLock:
    """Decorator/context manager to require single instance"""
    lock = InstanceLock(lock_file)
    
    if not lock.acquire():
        logger.error("Cannot start: another instance is already running")
        sys.exit(1)
    
    return lock


# Convenience function
def check_single_instance() -> bool:
    """Check if only one instance is running"""
    try:
        import subprocess
        result = subprocess.run(
            ['pgrep', '-f', 'tradingbot'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return len(pids) == 1
        else:
            return True  # No instances running
            
    except Exception as e:
        logger.error(f"Error checking single instance: {e}")
        return False
