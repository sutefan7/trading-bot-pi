"""
Breakout Strategy
Trades breakouts from consolidation using Bollinger Band squeeze and volume confirmation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
from .trend_follow import TradingSignal


class BreakoutStrategy:
    """Breakout strategy using BB squeeze and volume"""
    
    def __init__(self, config: Dict[str, Any], indicators):
        """
        Initialize breakout strategy
        
        Args:
            config: Strategy configuration
            indicators: TechnicalIndicators instance
        """
        self.config = config
        self.indicators = indicators
        
        # Parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_width_q = config.get('bb_width_q', 0.25)  # Squeeze quantile
        self.lookback_bars = config.get('lookback_bars', 20)
        self.volume_uptick_mult = config.get('volume_uptick_mult', 1.3)
        self.atr_mult_trail = config.get('atr_mult_trail', 2.0)
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on breakout logic
        
        Args:
            df: DataFrame with OHLCV and indicators
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        try:
            if len(df) < self.lookback_bars + 20:
                return None
            
            # Get latest values
            close = df['close'].iloc[-1]
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            volume = df.get('volume', pd.Series(0, index=df.index)).iloc[-1]
            
            bb_width = df.get('bb_width', pd.Series(0, index=df.index)).iloc[-20:]
            atr = df.get('atr', df['close'].pct_change().rolling(14).std() * df['close']).iloc[-1]
            
            # Volume analysis
            volume_ma = df.get('volume_sma', df.get('volume', pd.Series(1, index=df.index)).rolling(20).mean()).iloc[-1]
            volume_ratio = volume / volume_ma if volume_ma > 0 else 1.0
            
            # Check if we're in a squeeze (low BB width)
            if len(bb_width) > 0:
                bb_width_current = bb_width.iloc[-1]
                bb_width_quantile = bb_width.quantile(self.bb_width_q)
                in_squeeze = bb_width_current <= bb_width_quantile
            else:
                in_squeeze = False
            
            # Lookback high/low
            lookback_high = df['high'].iloc[-self.lookback_bars:].max()
            lookback_low = df['low'].iloc[-self.lookback_bars:].min()
            
            # Bullish breakout: price breaks above recent high with volume
            if close > lookback_high and volume_ratio >= self.volume_uptick_mult and in_squeeze:
                # Calculate confidence
                breakout_strength = (close - lookback_high) / lookback_high
                volume_strength = min(1.0, (volume_ratio - 1.0) / 2.0)  # 0 to 1
                squeeze_strength = 1.0 if in_squeeze else 0.5
                
                confidence = min(0.95, max(0.5,
                    0.3 * min(1.0, breakout_strength * 50) +
                    0.3 * volume_strength +
                    0.4 * squeeze_strength
                ))
                
                entry = close
                stop = lookback_high - atr  # Stop below breakout level
                risk = entry - stop
                take_profit = entry + (risk * 2.0)  # 2:1 R:R
                
                return TradingSignal(
                    symbol=symbol,
                    side='buy',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='breakout',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'breakout',
                        'lookback_high': lookback_high,
                        'lookback_low': lookback_low,
                        'volume_ratio': volume_ratio,
                        'in_squeeze': in_squeeze,
                        'atr': atr
                    }
                )
            
            # Bearish breakout: price breaks below recent low with volume
            elif close < lookback_low and volume_ratio >= self.volume_uptick_mult and in_squeeze:
                # Calculate confidence
                breakout_strength = (lookback_low - close) / lookback_low
                volume_strength = min(1.0, (volume_ratio - 1.0) / 2.0)
                squeeze_strength = 1.0 if in_squeeze else 0.5
                
                confidence = min(0.95, max(0.5,
                    0.3 * min(1.0, breakout_strength * 50) +
                    0.3 * volume_strength +
                    0.4 * squeeze_strength
                ))
                
                entry = close
                stop = lookback_low + atr  # Stop above breakout level
                risk = stop - entry
                take_profit = entry - (risk * 2.0)
                
                return TradingSignal(
                    symbol=symbol,
                    side='sell',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='breakout',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'breakout',
                        'lookback_high': lookback_high,
                        'lookback_low': lookback_low,
                        'volume_ratio': volume_ratio,
                        'in_squeeze': in_squeeze,
                        'atr': atr
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating breakout signal for {symbol}: {e}")
            return None



