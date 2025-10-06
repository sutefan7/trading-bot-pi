"""
Mean Reversion Strategy
Trades mean reversion using Bollinger Bands and RSI oversold/overbought conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
from .trend_follow import TradingSignal


class MeanRevertStrategy:
    """Mean reversion strategy using Bollinger Bands and RSI"""
    
    def __init__(self, config: Dict[str, Any], indicators):
        """
        Initialize mean reversion strategy
        
        Args:
            config: Strategy configuration
            indicators: TechnicalIndicators instance
        """
        self.config = config
        self.indicators = indicators
        
        # Parameters
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.tp_rr = config.get('tp_rr', 1.5)  # Take profit risk:reward
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on mean reversion logic
        
        Args:
            df: DataFrame with OHLCV and indicators
            symbol: Trading symbol
            
        Returns:
            TradingSignal or None
        """
        try:
            if len(df) < self.bb_period + 10:
                return None
            
            # Get latest values
            close = df['close'].iloc[-1]
            bb_upper = df.get('bb_upper', close * 1.02).iloc[-1]
            bb_lower = df.get('bb_lower', close * 0.98).iloc[-1]
            bb_middle = df.get('bb_middle', close).iloc[-1]
            rsi = df.get('rsi', pd.Series(50, index=df.index)).iloc[-1]
            atr = df.get('atr', df['close'].pct_change().rolling(14).std() * df['close']).iloc[-1]
            
            # Bullish mean reversion: price near lower band + RSI oversold
            if close <= bb_lower * 1.01 and rsi < self.rsi_oversold:
                # Calculate confidence
                distance_from_bb = (bb_lower - close) / bb_lower
                rsi_oversold_strength = (self.rsi_oversold - rsi) / self.rsi_oversold
                
                confidence = min(0.95, max(0.5,
                    0.5 * min(1.0, distance_from_bb * 50) +
                    0.5 * min(1.0, rsi_oversold_strength)
                ))
                
                entry = close
                stop = bb_lower - atr  # Stop below lower band
                risk = entry - stop
                take_profit = entry + (risk * self.tp_rr)  # Mean reversion to middle or above
                
                return TradingSignal(
                    symbol=symbol,
                    side='buy',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='mean_revert',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'mean_revert',
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle,
                        'rsi': rsi,
                        'atr': atr
                    }
                )
            
            # Bearish mean reversion: price near upper band + RSI overbought
            elif close >= bb_upper * 0.99 and rsi > (100 - self.rsi_oversold):
                # Calculate confidence
                distance_from_bb = (close - bb_upper) / bb_upper
                rsi_overbought_strength = (rsi - (100 - self.rsi_oversold)) / (100 - (100 - self.rsi_oversold))
                
                confidence = min(0.95, max(0.5,
                    0.5 * min(1.0, distance_from_bb * 50) +
                    0.5 * min(1.0, rsi_overbought_strength)
                ))
                
                entry = close
                stop = bb_upper + atr  # Stop above upper band
                risk = stop - entry
                take_profit = entry - (risk * self.tp_rr)
                
                return TradingSignal(
                    symbol=symbol,
                    side='sell',
                    entry=entry,
                    stop=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    strategy_name='mean_revert',
                    timestamp=datetime.now(),
                    meta={
                        'strategy': 'mean_revert',
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle,
                        'rsi': rsi,
                        'atr': atr
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None



