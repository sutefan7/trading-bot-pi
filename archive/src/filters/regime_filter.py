"""
Regime Filter voor Trading Bot v4
Bepaalt of de markt in een gunstige fase is voor trading
"""
from dataclasses import dataclass
import pandas as pd
from loguru import logger


@dataclass
class RegimeConfig:
    """Configuratie voor regime filter"""
    anchor_symbol: str
    timeframe: str
    sma_long: int
    adx_period: int
    adx_min: float
    vol_norm_window: int
    vol_min: float
    vol_max: float


class RegimeFilter:
    """Regime filter voor marktfase detectie"""
    
    def __init__(self, data_manager, indicators, cfg: RegimeConfig):
        self.dm = data_manager
        self.ind = indicators
        self.cfg = cfg
        self._last_check = None
        self._last_result = None
        
    def is_tradable(self) -> bool:
        """
        Returns True als markt in gunstige fase is:
          - Close > SMA(sma_long) (uptrend)
          - ADX(adx_period) >= adx_min (voldoende trendsterkte)
          - vol in [vol_min, vol_max], waarbij vol = ATR/Close (rolling)
        """
        try:
            # Haal data op voor anchor symbol
            df = self.dm.get_latest_data(self.cfg.anchor_symbol, days=365)
            if df is None or len(df) < max(self.cfg.sma_long, self.cfg.adx_period) + self.cfg.vol_norm_window + 5:
                logger.warning(f"Onvoldoende data voor regime check: {self.cfg.anchor_symbol}")
                return False
            
            # Resample naar gewenste timeframe als nodig
            if self.cfg.timeframe != "1h":
                df = self._resample_data(df, self.cfg.timeframe)
            
            # Voeg technische indicatoren toe
            df = self.ind.add_sma(df, self.cfg.sma_long)
            df = self.ind.add_atr(df, self.cfg.adx_period)  # ATR voor volatiliteit
            df = self.ind.add_adx(df, self.cfg.adx_period)
            
            # Check of we voldoende data hebben na indicator berekening
            if len(df) < 5:
                logger.warning("Onvoldoende data na indicator berekening")
                return False
            
            # Haal laatste waarden op
            last = df.iloc[-1]
            
            # Check 1: Close > SMA (uptrend)
            sma_col = f"SMA_{self.cfg.sma_long}"
            if sma_col not in df.columns:
                logger.error(f"SMA kolom niet gevonden: {sma_col}")
                return False
            
            uptrend = last["close"] > last[sma_col]
            
            # Check 2: ADX >= minimum (voldoende trendsterkte)
            adx_col = f"ADX_{self.cfg.adx_period}"
            if adx_col not in df.columns:
                logger.error(f"ADX kolom niet gevonden: {adx_col}")
                return False
            
            trend_strength = last[adx_col] >= self.cfg.adx_min
            
            # Check 3: Volatiliteit binnen range (ATR/Close)
            if "ATR" not in df.columns:
                logger.error("ATR kolom niet gevonden")
                return False
            
            # Bereken genormaliseerde volatiliteit (rolling ATR/Close)
            vol_norm = (df["ATR"].rolling(self.cfg.vol_norm_window).mean() / df["close"]).iloc[-1]
            vol_ok = self.cfg.vol_min <= vol_norm <= self.cfg.vol_max
            
            # Log resultaten
            logger.info(f"Regime check {self.cfg.anchor_symbol}: "
                       f"Uptrend={uptrend} (Close:{last['close']:.2f} > SMA:{last[sma_col]:.2f}), "
                       f"ADX={last[adx_col]:.1f} >= {self.cfg.adx_min}, "
                       f"Vol={vol_norm:.4f} in [{self.cfg.vol_min:.4f}, {self.cfg.vol_max:.4f}]")
            
            result = uptrend and trend_strength and vol_ok
            self._last_result = bool(result)  # Converteer naar Python boolean
            self._last_check = pd.Timestamp.now()
            
            return bool(result)  # Converteer naar Python boolean
            
        except Exception as e:
            logger.error(f"Fout bij regime check: {e}")
            return False
    
    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data naar gewenste timeframe"""
        try:
            if timeframe == "1d":
                # Resample naar dagelijkse data
                df_resampled = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            elif timeframe == "4h":
                # Resample naar 4-uur data
                df_resampled = df.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            else:
                # Onbekende timeframe, return originele data
                logger.warning(f"Onbekende timeframe: {timeframe}, gebruik originele data")
                return df
            
            return df_resampled
            
        except Exception as e:
            logger.error(f"Fout bij resampling naar {timeframe}: {e}")
            return df
    
    def get_regime_status(self) -> dict:
        """Krijg gedetailleerde regime status"""
        if self._last_result is None:
            self.is_tradable()  # Voer check uit als nog niet gedaan
        
        return {
            'is_tradable': self._last_result,
            'last_check': self._last_check.isoformat() if self._last_check else None,
            'anchor_symbol': self.cfg.anchor_symbol,
            'timeframe': self.cfg.timeframe
        }
