"""
Shared Feature Pipeline
Common feature engineering for both training and serving
Ensures train/serve parity by using identical feature generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from loguru import logger

from .schema import FEATURE_SCHEMA, ensure_order_and_dtype, get_feature_names, get_feature_dtypes


class FeaturePipeline:
    """Shared feature engineering pipeline for train and serve"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_names = get_feature_names()
        self.feature_dtypes = get_feature_dtypes()
        
    def build_features(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Build features from raw OHLCV data
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            config: Optional configuration overrides
            
        Returns:
            DataFrame with engineered features in correct order and dtype
        """
        try:
            # Use provided config or default
            feature_config = config or self.config
            
            # ⚠️ MEMORY: Limit history on Raspberry Pi
            MAX_HISTORY_BARS = 250  # Enough for all indicators
            if len(df) > MAX_HISTORY_BARS:
                logger.debug(f"📊 Limiting history from {len(df)} to {MAX_HISTORY_BARS} bars for memory optimization")
                df = df.tail(MAX_HISTORY_BARS)
            
            # Validate input data
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create working copy
            result_df = df.copy()
            
            # Add all technical indicators
            result_df = self._add_technical_indicators(result_df, feature_config)
            
            # Add custom features
            result_df = self._add_custom_features(result_df, feature_config)
            
            # Ensure correct order and dtypes
            result_df = ensure_order_and_dtype(result_df)
            
            logger.debug(f"Built {len(self.feature_names)} features from {len(df)} rows")
            
            # ⚠️ MEMORY: Explicit cleanup of intermediate DataFrames
            del df
            import gc
            gc.collect()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error building features: {e}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add technical indicators using ta library"""
        try:
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.trend import MACD, SMAIndicator, EMAIndicator
            from ta.volatility import BollingerBands, AverageTrueRange
            
            close = df['close']
            high = df['high']
            low = df['low']
            open_price = df['open']
            
            # RSI
            rsi_period = config.get('rsi_period', 14)
            rsi_indicator = RSIIndicator(close, window=rsi_period)
            df['rsi'] = rsi_indicator.rsi()
            
            # MACD
            macd_fast = config.get('macd_fast', 12)
            macd_slow = config.get('macd_slow', 26)
            macd_signal = config.get('macd_signal', 9)
            macd_indicator = MACD(close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_histogram'] = macd_indicator.macd_diff()
            
            # Simple Moving Averages
            df['sma_10'] = close.rolling(window=10, min_periods=1).mean()
            df['sma_50'] = close.rolling(window=50, min_periods=1).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = close.ewm(span=12, min_periods=1).mean()
            df['ema_26'] = close.ewm(span=26, min_periods=1).mean()
            
            # Bollinger Bands
            bb_window = config.get('bb_window', 20)
            bb_std = config.get('bb_std', 2)
            bb_indicator = BollingerBands(close, window=bb_window, window_dev=bb_std)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            stoch_k = config.get('stoch_k', 14)
            stoch_d = config.get('stoch_d', 3)
            stoch_indicator = StochasticOscillator(high, low, close, window=stoch_k, smooth_window=stoch_d)
            df['stoch_k'] = stoch_indicator.stoch()
            df['stoch_d'] = stoch_indicator.stoch_signal()
            
            # ATR
            atr_window = config.get('atr_window', 14)
            atr_indicator = AverageTrueRange(high, low, close, window=atr_window)
            df['atr'] = atr_indicator.average_true_range()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise
    
    def _add_custom_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Add custom engineered features"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            open_price = df['open']
            
            # Price changes
            df['price_change_1d'] = close.pct_change(1)
            df['price_change_3d'] = close.pct_change(3)
            df['price_change_7d'] = close.pct_change(7)
            
            # Volatility (rolling standard deviation)
            df['volatility_5d'] = close.pct_change().rolling(window=5).std()
            df['volatility_20d'] = close.pct_change().rolling(window=20).std()
            
            # Momentum
            df['momentum_5d'] = close / close.shift(5) - 1
            df['momentum_20d'] = close / close.shift(20) - 1
            
            # Moving average features
            df['ma_trend'] = np.where(df['sma_10'] > df['sma_50'], 1, -1)
            df['ma_distance'] = (close - df['sma_50']) / df['sma_50']
            
            # RSI flags
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            
            # MACD features
            df['macd_cross'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1, 0
            )
            df['macd_strength'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands features
            df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.5, 1, 0)
            df['bb_expansion'] = np.where(df['bb_width'] > df['bb_width'].rolling(20).mean() * 1.5, 1, 0)
            
            # Candle features
            df['candle_body_size'] = abs(close - open_price) / close
            df['upper_shadow'] = (high - np.maximum(close, open_price)) / close
            df['lower_shadow'] = (np.minimum(close, open_price) - low) / close
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding custom features: {e}")
            raise
    
    def get_feature_vector(self, df: pd.DataFrame, row_idx: int = -1) -> np.ndarray:
        """
        Extract feature vector for a specific row
        
        Args:
            df: DataFrame with features
            row_idx: Row index to extract (default: last row)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Ensure correct schema
            df_schema = ensure_order_and_dtype(df)
            
            # Extract feature vector
            feature_vector = df_schema.iloc[row_idx].values
            
            # Handle NaN values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting feature vector: {e}")
            raise
    
    def save_featureset(self, obj: Dict[str, Any], path: str) -> None:
        """
        Save featureset configuration to file
        
        Args:
            obj: Featureset configuration dictionary
            path: Path to save the featureset
        """
        try:
            featureset_data = {
                'feature_names': self.feature_names,
                'feature_dtypes': {name: str(dtype) for name, dtype in self.feature_dtypes.items()},
                'config': obj,
                'schema_version': '1.0'
            }
            
            with open(path, 'w') as f:
                json.dump(featureset_data, f, indent=2)
                
            logger.info(f"Featureset saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving featureset: {e}")
            raise
    
    def load_featureset(self, path: str) -> Dict[str, Any]:
        """
        Load featureset configuration from file
        
        Args:
            path: Path to load the featureset from
            
        Returns:
            Featureset configuration dictionary
        """
        try:
            with open(path, 'r') as f:
                featureset_data = json.load(f)
            
            # Validate schema version
            schema_version = featureset_data.get('schema_version', '0.0')
            if schema_version != '1.0':
                logger.warning(f"Featureset schema version {schema_version} may not be compatible")
            
            logger.info(f"Featureset loaded from {path}")
            return featureset_data
            
        except Exception as e:
            logger.error(f"Error loading featureset: {e}")
            raise


# Convenience functions for backward compatibility
def build_features(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Build features using default pipeline"""
    pipeline = FeaturePipeline()
    return pipeline.build_features(df, config)


def load_featureset(path: str) -> Dict[str, Any]:
    """Load featureset using default pipeline"""
    pipeline = FeaturePipeline()
    return pipeline.load_featureset(path)


def save_featureset(obj: Dict[str, Any], path: str) -> None:
    """Save featureset using default pipeline"""
    pipeline = FeaturePipeline()
    return pipeline.save_featureset(obj, path)
