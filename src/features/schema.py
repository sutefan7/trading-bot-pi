"""
Feature Schema Definition
Defines explicit order and data types for features to ensure train/serve parity
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class FeatureSchema:
    """Schema definition for features"""
    name: str
    dtype: np.dtype
    description: str = ""


# Define the exact feature schema used by both train and serve
FEATURE_SCHEMA: List[FeatureSchema] = [
    # Basic price features
    FeatureSchema("rsi", np.float32, "Relative Strength Index (14 periods)"),
    FeatureSchema("macd", np.float32, "MACD line"),
    FeatureSchema("macd_signal", np.float32, "MACD signal line"),
    FeatureSchema("macd_histogram", np.float32, "MACD histogram"),
    
    # Moving averages
    FeatureSchema("sma_10", np.float32, "Simple Moving Average (10 periods)"),
    FeatureSchema("sma_50", np.float32, "Simple Moving Average (50 periods)"),
    FeatureSchema("ema_12", np.float32, "Exponential Moving Average (12 periods)"),
    FeatureSchema("ema_26", np.float32, "Exponential Moving Average (26 periods)"),
    
    # Bollinger Bands
    FeatureSchema("bb_upper", np.float32, "Bollinger Bands upper"),
    FeatureSchema("bb_lower", np.float32, "Bollinger Bands lower"),
    FeatureSchema("bb_middle", np.float32, "Bollinger Bands middle"),
    FeatureSchema("bb_width", np.float32, "Bollinger Bands width"),
    FeatureSchema("bb_percent", np.float32, "Bollinger Bands percentage"),
    
    # Stochastic
    FeatureSchema("stoch_k", np.float32, "Stochastic %K"),
    FeatureSchema("stoch_d", np.float32, "Stochastic %D"),
    
    # Volatility
    FeatureSchema("atr", np.float32, "Average True Range"),
    
    # Custom features
    FeatureSchema("price_change_1d", np.float32, "Price change 1 day"),
    FeatureSchema("price_change_3d", np.float32, "Price change 3 days"),
    FeatureSchema("price_change_7d", np.float32, "Price change 7 days"),
    FeatureSchema("volatility_5d", np.float32, "Volatility 5 days"),
    FeatureSchema("volatility_20d", np.float32, "Volatility 20 days"),
    FeatureSchema("momentum_5d", np.float32, "Momentum 5 days"),
    FeatureSchema("momentum_20d", np.float32, "Momentum 20 days"),
    FeatureSchema("ma_trend", np.float32, "Moving average trend"),
    FeatureSchema("ma_distance", np.float32, "Distance from moving average"),
    FeatureSchema("rsi_oversold", np.float32, "RSI oversold flag"),
    FeatureSchema("rsi_overbought", np.float32, "RSI overbought flag"),
    FeatureSchema("macd_cross", np.float32, "MACD crossover flag"),
    FeatureSchema("macd_strength", np.float32, "MACD strength"),
    FeatureSchema("bb_squeeze", np.float32, "Bollinger Bands squeeze"),
    FeatureSchema("bb_expansion", np.float32, "Bollinger Bands expansion"),
    FeatureSchema("candle_body_size", np.float32, "Candle body size"),
    FeatureSchema("upper_shadow", np.float32, "Upper shadow length"),
    FeatureSchema("lower_shadow", np.float32, "Lower shadow length"),
]


def get_feature_names() -> List[str]:
    """Get list of feature names in exact order"""
    return [feature.name for feature in FEATURE_SCHEMA]


def get_feature_dtypes() -> Dict[str, np.dtype]:
    """Get feature name to dtype mapping"""
    return {feature.name: feature.dtype for feature in FEATURE_SCHEMA}


def ensure_order_and_dtype(df: pd.DataFrame, schema: List[FeatureSchema] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has correct column order and data types
    
    Args:
        df: Input DataFrame with features
        schema: Feature schema to use (defaults to FEATURE_SCHEMA)
        
    Returns:
        DataFrame with correct column order and dtypes
        
    Raises:
        ValueError: If required features are missing
    """
    if schema is None:
        schema = FEATURE_SCHEMA
    
    # Get expected feature names
    expected_features = [f.name for f in schema]
    available_features = df.columns.tolist()
    
    # Check for missing features
    missing_features = set(expected_features) - set(available_features)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Reorder columns and ensure correct dtypes
    result_df = pd.DataFrame(index=df.index)
    
    for feature in schema:
        if feature.name in df.columns:
            # Convert to correct dtype
            result_df[feature.name] = df[feature.name].astype(feature.dtype)
        else:
            # Fill missing features with NaN
            result_df[feature.name] = np.nan
    
    return result_df


def validate_feature_schema(df: pd.DataFrame, schema: List[FeatureSchema] = None) -> bool:
    """
    Validate that DataFrame matches expected schema
    
    Args:
        df: DataFrame to validate
        schema: Feature schema to validate against
        
    Returns:
        True if schema matches, False otherwise
    """
    if schema is None:
        schema = FEATURE_SCHEMA
    
    expected_features = [f.name for f in schema]
    actual_features = df.columns.tolist()
    
    # Check column order and names
    if actual_features != expected_features:
        return False
    
    # Check data types
    for feature in schema:
        if df[feature.name].dtype != feature.dtype:
            return False
    
    return True


def create_empty_feature_df(index: pd.Index = None, schema: List[FeatureSchema] = None) -> pd.DataFrame:
    """
    Create empty DataFrame with correct feature schema
    
    Args:
        index: Index for the DataFrame
        schema: Feature schema to use
        
    Returns:
        Empty DataFrame with correct schema
    """
    if schema is None:
        schema = FEATURE_SCHEMA
    
    if index is None:
        index = pd.Index([])
    
    data = {feature.name: pd.Series(dtype=feature.dtype, index=index) for feature in schema}
    return pd.DataFrame(data, index=index)



