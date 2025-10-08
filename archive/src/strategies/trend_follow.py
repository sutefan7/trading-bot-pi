"""
Trend Following Strategy
Follows strong trends using EMA crossovers, RSI, and MACD confirmation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class TradingSignal:
    """Trading signal with entry/exit levels"""
    symbol: str
    side: str  # 'buy' or 'sell'
    entry: float
    stop: float
    take_profit: float
    confidence: float
    strategy_name: str
    timestamp: datetime
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry': self.entry,
            'stop': self.stop,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'strategy_name': self.strategy_name,
            'timestamp': self.timestamp,
            'meta': self.meta
        }


class TrendFollowStrategy:
    """Trend following strategy using EMA crossovers and momentum"""
    
    def __init__(self, config: Dict[str, Any], indicators):
        """
        Initialize trend follow strategy
        
        Args:
            config: Strategy configuration
            indicators: TechnicalIndicators instance
        """
        self.config = config
        self.indicators = indicators
        
        # Parameters
        self.ema_short = config.get('ema_short', 20)
        self.ema_long = config.get('ema_long', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_entry_min = config.get('rsi_entry_min', 40)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.atr_mult_stop = config.get('atr_mult_stop', 2.0)
        self.atr_mult_trail = config.get('atr_mult_trail', 1.5)
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on trend following logic
        
        Args:
            df: DataFrame with OHLCV and indicators
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        try:
            if len(df) < max(self.ema_long, 50):
                return None
            
            # Get latest values
            close = df['close'].iloc[-1]
            ema_short = df.get('ema_12', df['close'].ewm(span=self.ema_short).mean()).iloc[-1]
            ema_long = df.get('ema_26', df['close'].ewm(span=self.ema_long).mean()).iloc[-1]
            rsi = df.get('rsi', pd.Series(50, index=df.index)).iloc[-1]
            macd = df.get('macd', pd.Series(0, index=df.index)).iloc[-1]
            macd_signal = df.get('macd_signal', pd.Series(0, index=df.index)).iloc[-1]
            atr = df.get('atr', df['close'].pct_change().rolling(14).std() * df['close']).iloc[-1]
            
            # Check for bullish signal
            if ema_short > ema_long and macd > macd_signal and rsi > self.rsi_entry_min and rsi < 70:
                # Calculate confidence based on indicator strength
                ema_diff_pct = (ema_short - ema_long) / ema_long
                macd_strength = abs(macd - macd_signal) / close if close > 0 else 0
                rsi_strength = (rsi - 50) / 50  # -1 to 1
                
                confidence = min(0.95, max(0.5,
                    0.3 * (1 if ema_diff_pct > 0.01 else 0.5) +
                    0.3 * min(1.0, macd_strength * 100) +
                    0.4 * min(1.0, abs(rsi_strength))
                ))
                
                # Calculate entry and stop levels
                entry = close
                stop = entry - (atr * self.atr_mult_stop)
                take_profit = entry + (atr * self.atr_mult_stop * 2.0)  # 2:1 R:R
                
                return TradingSignal(
                    symbol=symbol,
                    side='buy',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='trend_follow',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'trend_follow',
                        'ema_short': ema_short,
                        'ema_long': ema_long,
                        'rsi': rsi,
                        'macd': macd,
                        'atr': atr
                    }
                )
            
            # Check for bearish signal (short)
            elif ema_short < ema_long and macd < macd_signal and rsi < 60 and rsi > 30:
                # Calculate confidence
                ema_diff_pct = (ema_long - ema_short) / ema_long
                macd_strength = abs(macd - macd_signal) / close if close > 0 else 0
                rsi_strength = (50 - rsi) / 50
                
                confidence = min(0.95, max(0.5,
                    0.3 * (1 if ema_diff_pct > 0.01 else 0.5) +
                    0.3 * min(1.0, macd_strength * 100) +
                    0.4 * min(1.0, abs(rsi_strength))
                ))
                
                entry = close
                stop = entry + (atr * self.atr_mult_stop)
                take_profit = entry - (atr * self.atr_mult_stop * 2.0)
                
                return TradingSignal(
                    symbol=symbol,
                    side='sell',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='trend_follow',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'trend_follow',
                        'ema_short': ema_short,
                        'ema_long': ema_long,
                        'rsi': rsi,
                        'macd': macd,
                        'atr': atr
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating trend follow signal for {symbol}: {e}")
            return None




