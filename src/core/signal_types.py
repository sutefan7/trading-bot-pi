"""
Uniforme signal types voor Trading Bot v4
Definieert standaard signal formaten voor alle strategieën
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SignalSide(Enum):
    """Trading signal side"""
    BUY = "buy"
    SELL = "sell"


class SignalType(Enum):
    """Signal type"""
    TREND_FOLLOW = "trend_follow"
    MEAN_REVERT = "mean_revert"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"


@dataclass
class TradingSignal:
    """Uniform trading signal format"""
    
    # Basis signal informatie
    symbol: str
    side: SignalSide
    signal_type: SignalType
    
    # Prijzen
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    
    # Risk management
    confidence: float = 0.5  # 0.0 - 1.0
    risk_reward_ratio: Optional[float] = None
    
    # Metadata
    strategy_name: str = "unknown"
    timestamp: datetime = None
    meta: Dict[str, Any] = None
    
    # Position sizing hints
    max_position_size: Optional[float] = None
    min_confidence_threshold: float = 0.6
    
    def __post_init__(self):
        """Post-initialization validatie"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        if self.meta is None:
            self.meta = {}
        
        # Bereken risk/reward ratio als niet opgegeven
        if self.risk_reward_ratio is None and self.take_profit is not None:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                self.risk_reward_ratio = reward / risk
        
        # Valideer confidence
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Converteer naar dictionary voor backward compatibility"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'entry': self.entry_price,
            'stop': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'risk_reward_ratio': self.risk_reward_ratio,
            'strategy': self.strategy_name,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp.isoformat(),
            'meta': self.meta or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Maak TradingSignal van dictionary"""
        return cls(
            symbol=data['symbol'],
            side=SignalSide(data['side']),
            signal_type=SignalType(data.get('signal_type', 'trend_follow')),
            entry_price=data['entry'],
            stop_loss=data['stop'],
            take_profit=data.get('take_profit'),
            confidence=data.get('confidence', 0.5),
            risk_reward_ratio=data.get('risk_reward_ratio'),
            strategy_name=data.get('strategy', 'unknown'),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            meta=data.get('meta', {})
        )
    
    def is_valid(self) -> bool:
        """Check of signal geldig is"""
        try:
            # Basis validatie
            if not self.symbol or not self.entry_price or not self.stop_loss:
                return False
            
            # Prijs validatie
            if self.entry_price <= 0 or self.stop_loss <= 0:
                return False
            
            # Stop loss moet logisch zijn
            if self.side == SignalSide.BUY and self.stop_loss >= self.entry_price:
                return False
            if self.side == SignalSide.SELL and self.stop_loss <= self.entry_price:
                return False
            
            # Take profit validatie
            if self.take_profit is not None:
                if self.take_profit <= 0:
                    return False
                if self.side == SignalSide.BUY and self.take_profit <= self.entry_price:
                    return False
                if self.side == SignalSide.SELL and self.take_profit >= self.entry_price:
                    return False
            
            # Confidence validatie
            if self.confidence < 0.0 or self.confidence > 1.0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_risk_amount(self) -> float:
        """Bereken risk amount per unit"""
        return abs(self.entry_price - self.stop_loss)
    
    def get_reward_amount(self) -> float:
        """Bereken reward amount per unit"""
        if self.take_profit is None:
            return 0.0
        return abs(self.take_profit - self.entry_price)
    
    def get_expected_value(self) -> float:
        """Bereken expected value van signal"""
        if self.risk_reward_ratio is None:
            return 0.0
        
        # Expected value = (win_prob * reward) - (loss_prob * risk)
        win_prob = self.confidence
        loss_prob = 1.0 - self.confidence
        
        expected_value = (win_prob * self.risk_reward_ratio) - (loss_prob * 1.0)
        return expected_value
    
    def __str__(self) -> str:
        """String representatie"""
        return (f"TradingSignal({self.symbol} {self.side.value} @ {self.entry_price:.4f}, "
                f"stop={self.stop_loss:.4f}, conf={self.confidence:.2f})")


# Convenience functies voor backward compatibility
def create_signal(symbol: str, side: str, entry: float, stop: float, 
                 take_profit: Optional[float] = None, confidence: float = 0.5,
                 strategy: str = "unknown", **meta) -> TradingSignal:
    """Maak TradingSignal met backward compatibility"""
    return TradingSignal(
        symbol=symbol,
        side=SignalSide(side),
        signal_type=SignalType.TREND_FOLLOW,  # Default
        entry_price=entry,
        stop_loss=stop,
        take_profit=take_profit,
        confidence=confidence,
        strategy_name=strategy,
        meta=meta
    )


def signal_to_dict(signal: TradingSignal) -> Dict[str, Any]:
    """Converteer signal naar dictionary"""
    return signal.to_dict()


def dict_to_signal(data: Dict[str, Any]) -> TradingSignal:
    """Maak signal van dictionary"""
    return TradingSignal.from_dict(data)
