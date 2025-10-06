"""
Technical Indicators Calculator
Provides technical analysis indicators for trading strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not available - install with: pip install ta")


class TechnicalIndicators:
    """Calculator for technical indicators"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize indicators calculator
        
        Args:
            config: Configuration dictionary with indicator parameters
        """
        self.config = config or {}
        
        if not TA_AVAILABLE:
            raise ImportError("ta library is required. Install with: pip install ta")
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Add each indicator group
            result_df = self.add_trend_indicators(result_df)
            result_df = self.add_momentum_indicators(result_df)
            result_df = self.add_volatility_indicators(result_df)
            result_df = self.add_volume_indicators(result_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators (SMA, EMA, MACD, ADX)"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Simple Moving Averages
            sma_periods = self.config.get('moving_averages', {})
            df['sma_10'] = close.rolling(window=sma_periods.get('sma_short', 10), min_periods=1).mean()
            df['sma_50'] = close.rolling(window=sma_periods.get('sma_medium', 50), min_periods=1).mean()
            df['sma_200'] = close.rolling(window=sma_periods.get('sma_long', 200), min_periods=1).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = close.ewm(span=sma_periods.get('ema_short', 12), min_periods=1).mean()
            df['ema_26'] = close.ewm(span=sma_periods.get('ema_long', 26), min_periods=1).mean()
            
            # MACD
            macd_config = self.config.get('macd', {})
            macd = MACD(
                close,
                window_fast=macd_config.get('fast_period', 12),
                window_slow=macd_config.get('slow_period', 26),
                window_sign=macd_config.get('signal_period', 9)
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # ADX (trend strength)
            adx_period = self.config.get('adx', {}).get('period', 14)
            adx = ADXIndicator(high, low, close, window=adx_period)
            df['adx'] = adx.adx()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding trend indicators: {e}")
            return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (RSI, Stochastic)"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # RSI
            rsi_period = self.config.get('rsi', {}).get('period', 14)
            rsi = RSIIndicator(close, window=rsi_period)
            df['rsi'] = rsi.rsi()
            
            # Stochastic Oscillator
            stoch_k = 14
            stoch_d = 3
            stoch = StochasticOscillator(high, low, close, window=stoch_k, smooth_window=stoch_d)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {e}")
            return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators (Bollinger Bands, ATR)"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Bollinger Bands
            bb_config = self.config.get('bollinger_bands', {})
            bb_period = bb_config.get('period', 20)
            bb_std = bb_config.get('std_dev', 2.0)
            
            bb = BollingerBands(close, window=bb_period, window_dev=bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            atr_period = self.config.get('atr', {}).get('period', 14)
            atr = AverageTrueRange(high, low, close, window=atr_period)
            df['atr'] = atr.average_true_range()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volatility indicators: {e}")
            return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        try:
            if 'volume' not in df.columns:
                return df
            
            volume = df['volume']
            
            # Volume moving average
            df['volume_sma'] = volume.rolling(window=20, min_periods=1).mean()
            
            # Volume ratio
            df['volume_ratio'] = volume / df['volume_sma']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {e}")
            return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            atr = AverageTrueRange(high, low, close, window=period)
            return atr.average_true_range()
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(0, index=df.index)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI
        
        Args:
            df: DataFrame with close prices
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        try:
            close = df['close']
            rsi = RSIIndicator(close, window=period)
            return rsi.rsi()
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=df.index)
    
    def is_bullish_trend(self, df: pd.DataFrame) -> bool:
        """Check if current trend is bullish"""
        try:
            if len(df) < 2:
                return False
            
            # Check if short MA > long MA
            if 'sma_10' in df.columns and 'sma_50' in df.columns:
                return df['sma_10'].iloc[-1] > df['sma_50'].iloc[-1]
            
            # Fallback: check if price is rising
            return df['close'].iloc[-1] > df['close'].iloc[-10]
            
        except Exception as e:
            logger.error(f"Error checking trend: {e}")
            return False
    
    def is_bearish_trend(self, df: pd.DataFrame) -> bool:
        """Check if current trend is bearish"""
        try:
            if len(df) < 2:
                return False
            
            # Check if short MA < long MA
            if 'sma_10' in df.columns and 'sma_50' in df.columns:
                return df['sma_10'].iloc[-1] < df['sma_50'].iloc[-1]
            
            # Fallback: check if price is falling
            return df['close'].iloc[-1] < df['close'].iloc[-10]
            
        except Exception as e:
            logger.error(f"Error checking trend: {e}")
            return False




