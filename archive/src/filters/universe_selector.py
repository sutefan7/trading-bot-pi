"""
Simplified Universe Selector voor Trading Bot v4 (Pi Version)
Selecteert coins op basis van Mac-getrainde modellen en recente data
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class UniverseConfig:
    """Configuratie voor universe selector"""
    rebalance_frequency: str = "weekly"
    max_assets: int = 3
    min_volume_24h: float = 1000000  # 1M USD
    momentum_days: int = 7


class UniverseSelector:
    """Simplified universe selector voor trading-time selectie"""
    
    def __init__(self, data_manager, cfg: UniverseConfig):
        self.dm = data_manager
        self.cfg = cfg
        self._cached_selection = None
        self._last_rebalance_ts = None
        
        # Mac-getrainde coins (Pi volgt Mac)
        self.mac_trained_coins = [
            'ADA-USD', 'AVAX-USD', 'BNB-USD', 'BTC-USD', 
            'DOGE-USD', 'ETH-USD', 'HYPE-USD', 'SOL-USD',
            'STETH-USD', 'TRX-USD', 'WBETH-USD', 'XRP-USD'
        ]
    
    def _get_recent_momentum(self, symbol: str, days: int = 7) -> float:
        """Bereken recente momentum (7 dagen)"""
        try:
            df = self.dm.get_latest_data(symbol, days=days)
            if df is None or len(df) < days:
                return 0.0
            
            # Bereken momentum als percentage verandering
            start_price = df['close'].iloc[0]
            end_price = df['close'].iloc[-1]
            momentum = (end_price - start_price) / start_price * 100
            
            return momentum
        except Exception as e:
            logger.warning(f"Fout bij momentum berekening voor {symbol}: {e}")
            return 0.0
    
    def _get_volume_24h(self, symbol: str) -> float:
        """Bereken 24h volume in USD"""
        try:
            # Haal 1h data op en sommeer laatste 24 bars
            df = self.dm.get_latest_data(symbol, days=2, interval="1h")
            if df is None or len(df) < 2:
                return 0.0
            
            recent = df.tail(24)
            if len(recent) == 0:
                return 0.0
            volume_24h = float((recent['volume'] * recent['close']).sum())
            return volume_24h
        except Exception as e:
            logger.warning(f"Fout bij volume berekening voor {symbol}: {e}")
            return 0.0
    
    def _get_ml_confidence(self, symbol: str) -> float:
        """Placeholder: neutrale confidence zodat Universe niet random is"""
        return 0.5
    
    def select(self, symbols: List[str]) -> List[str]:
        """
        Selecteer beste coins voor trading (trading-time selectie):
        1) Neem Mac-getrainde coins
        2) Score op recente momentum, volume en ML confidence
        3) Return top performers
        """
        try:
            logger.info(f"Universe selectie gestart voor {len(symbols)} symbols")
            
            # Filter op Mac-getrainde coins
            available_coins = [coin for coin in symbols if coin in self.mac_trained_coins]
            logger.info(f"Mac-getrainde coins beschikbaar: {len(available_coins)}")
            
            if not available_coins:
                logger.warning("Geen Mac-getrainde coins beschikbaar")
                return []
            
            # Score elke coin
            coin_scores = []
            for coin in available_coins:
                try:
                    # Bereken scores
                    momentum = self._get_recent_momentum(coin, self.cfg.momentum_days)
                    volume_24h = self._get_volume_24h(coin)
                    ml_confidence = self._get_ml_confidence(coin)
                    
                    # Filter op minimum volume
                    if volume_24h < self.cfg.min_volume_24h:
                        logger.debug(f"{coin} gefilterd: volume te laag ({volume_24h:,.0f})")
                        continue
                    
                    # Bereken totale score (zonder random ML factor)
                    # Momentum: 60%, Volume: 40%
                    score = (momentum * 0.6 + 
                            (volume_24h / 1000000) * 0.4)
                    
                    coin_scores.append((coin, score, momentum, volume_24h, ml_confidence))
                    
                except Exception as e:
                    logger.warning(f"Fout bij scoring {coin}: {e}")
                    continue
            
            if not coin_scores:
                logger.warning("Geen coins voldoen aan criteria")
                return []
            
            # Sorteer op score (hoogste eerst)
            coin_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Selecteer top performers
            selected_coins = coin_scores[:self.cfg.max_assets]
            selected_symbols = [coin for coin, score, momentum, volume, confidence in selected_coins]
            
            # Log resultaten
            logger.info(f"Universe selectie voltooid: {len(selected_symbols)} symbols geselecteerd")
            for i, (coin, score, momentum, volume, confidence) in enumerate(selected_coins):
                logger.info(f"   {i+1}. {coin}: momentum={momentum:.1f}%, volume=${volume:,.0f}, confidence={confidence:.1%}")
            
            # Cache resultaat
            self._cached_selection = selected_symbols
            self._last_rebalance_ts = datetime.now()
            
            return selected_symbols
            
        except Exception as e:
            logger.error(f"Fout bij universe selectie: {e}")
            return []
    
    def get_universe(self, symbols: List[str]) -> List[str]:
        """Get current universe (with caching)"""
        try:
            # Check cache
            if (self._cached_selection and 
                self._last_rebalance_ts and 
                datetime.now() - self._last_rebalance_ts < timedelta(days=7)):
                logger.debug("Using cached universe selection")
                return self._cached_selection
            
            # Rebalance universe
            logger.info("Universe herbalans...")
            return self.select(symbols)
            
        except Exception as e:
            logger.error(f"Fout bij universe ophalen: {e}")
            return []
    
    def should_rebalance(self) -> bool:
        """Check if universe should be rebalanced"""
        if not self._last_rebalance_ts:
            return True
        
        # Rebalance weekly
        if self.cfg.rebalance_frequency == "weekly":
            return datetime.now() - self._last_rebalance_ts > timedelta(days=7)
        
        # Default: rebalance daily
        return datetime.now() - self._last_rebalance_ts > timedelta(days=1)
    
    def get_universe_status(self) -> Dict[str, Any]:
        """Get universe status for logging"""
        return {
            'selected_coins': self._cached_selection or [],
            'last_rebalance': self._last_rebalance_ts,
            'should_rebalance': self.should_rebalance(),
            'mac_trained_coins': len(self.mac_trained_coins),
            'config': {
                'rebalance_frequency': self.cfg.rebalance_frequency,
                'max_assets': self.cfg.max_assets,
                'min_volume_24h': self.cfg.min_volume_24h,
                'momentum_days': self.cfg.momentum_days
            }
        }
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get universe selection statistics"""
        return {
            'mac_trained_coins': len(self.mac_trained_coins),
            'last_rebalance': self._last_rebalance_ts,
            'cached_selection': self._cached_selection,
            'config': {
                'rebalance_frequency': self.cfg.rebalance_frequency,
                'max_assets': self.cfg.max_assets,
                'min_volume_24h': self.cfg.min_volume_24h,
                'momentum_days': self.cfg.momentum_days
            }
        }