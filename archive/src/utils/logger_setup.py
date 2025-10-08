"""
Logger Setup voor Trading Bot v4
Geavanceerde logging configuratie met rotatie en verschillende niveaus
"""
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Setup geavanceerde logging configuratie
    
    Args:
        log_level: Logging niveau (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory voor log bestanden
    """
    # Maak logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Verwijder bestaande loguru handlers
    logger.remove()
    
    # Console handler met kleuren
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler voor alle logs
    logger.add(
        log_path / "trading_bot.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # File handler voor errors alleen
    logger.add(
        log_path / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # File handler voor trading signalen
    logger.add(
        log_path / "signals.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: "signal" in record["message"].lower() or "trade" in record["message"].lower(),
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # File handler voor performance metrics
    logger.add(
        log_path / "performance.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: any(keyword in record["message"].lower() for keyword in ["portfolio", "pnl", "drawdown", "sharpe"]),
        rotation="1 day",
        retention="90 days",
        compression="zip"
    )
    
    logger.info(f"📝 Logging geconfigureerd - niveau: {log_level}, directory: {log_path}")


def get_logger(name: str = None):
    """
    Krijg een logger instance
    
    Args:
        name: Logger naam
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class TradingLogger:
    """Speciale logger voor trading events"""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logger.bind(component=component)
    
    def signal_generated(self, symbol: str, action: str, confidence: float, price: float):
        """Log trading signal"""
        self.logger.info(f"📊 SIGNAL: {symbol} {action.upper()} @ {price:.2f} (confidence: {confidence:.3f})")
    
    def trade_executed(self, symbol: str, action: str, price: float, size: float):
        """Log trade uitvoering"""
        self.logger.info(f"📈 TRADE: {symbol} {action.upper()} @ {price:.2f}, size: {size:.4f}")
    
    def position_opened(self, symbol: str, side: str, entry_price: float, stop_loss: float, take_profit: float):
        """Log positie opening"""
        self.logger.info(f"🔓 POSITION OPENED: {symbol} {side.upper()} @ {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
    
    def position_closed(self, symbol: str, exit_price: float, pnl: float, reason: str):
        """Log positie sluiting"""
        pnl_color = "🟢" if pnl > 0 else "🔴"
        self.logger.info(f"🔒 POSITION CLOSED: {symbol} @ {exit_price:.2f}, P&L: {pnl_color} {pnl:.2f}, reason: {reason}")
    
    def portfolio_update(self, total_value: float, cash: float, pnl: float, drawdown: float):
        """Log portfolio update"""
        self.logger.info(f"💰 PORTFOLIO: €{total_value:.2f}, Cash: €{cash:.2f}, P&L: €{pnl:.2f}, DD: {drawdown:.2%}")
    
    def risk_alert(self, message: str, level: str = "WARNING"):
        """Log risico waarschuwing"""
        if level == "CRITICAL":
            self.logger.critical(f"🚨 RISK ALERT: {message}")
        elif level == "ERROR":
            self.logger.error(f"⚠️ RISK WARNING: {message}")
        else:
            self.logger.warning(f"⚠️ RISK NOTICE: {message}")
    
    def model_performance(self, model_name: str, metrics: dict):
        """Log model performance"""
        self.logger.info(f"🤖 MODEL {model_name}: AUC={metrics.get('auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")
    
    def data_update(self, symbol: str, rows: int, columns: int):
        """Log data update"""
        self.logger.info(f"📊 DATA UPDATE: {symbol} - {rows} rijen, {columns} kolommen")
    
    def error(self, message: str, exception: Exception = None):
        """Log error"""
        if exception:
            self.logger.error(f"❌ ERROR: {message} - {str(exception)}")
        else:
            self.logger.error(f"❌ ERROR: {message}")


def log_function_call(func):
    """Decorator om functie calls te loggen"""
    def wrapper(*args, **kwargs):
        logger.debug(f"🔧 Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} failed with error: {str(e)}")
            raise
    return wrapper


def log_execution_time(func):
    """Decorator om uitvoeringstijd te loggen"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"⏱️ Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⏱️ {func.__name__} completed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"⏱️ {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper
