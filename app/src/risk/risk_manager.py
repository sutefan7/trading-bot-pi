"""
Risicomanagement voor Trading Bot v4
Uitgebreide risicomanagement met position sizing, stop-loss en portfolio monitoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class Position:
    """Trading positie"""
    symbol: str
    side: str  # 'long' of 'short'
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_loss: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class Portfolio:
    """Portfolio status"""
    total_value: float
    cash: float
    positions: Dict[str, Position]
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    daily_pnl: float = 0.0


class RiskManager:
    """Centrale risicomanagement class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[Portfolio] = []
        
        # Nieuwe risicomanagement parameters
        self.per_trade_risk_pct = config.get('per_trade_risk_pct', 0.0075)
        self.daily_loss_cap_pct = config.get('daily_loss_cap_pct', 0.04)
        self.max_positions = config.get('max_open_positions', 3)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.03)
        
        # Backward compatibility
        self.max_position_size = config.get('max_position_size', 0.3)
        self.min_position_size = config.get('min_position_size', 0.05)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', 0.15)
        self.max_drawdown = config.get('max_drawdown', 0.20)
        self.trailing_stop = config.get('trailing_stop', True)
        
        # Portfolio tracking
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.cash = self.initial_balance
        self.realized_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.trade_history: List[Dict[str, Any]] = []
        self.win_trades = 0
        self.loss_trades = 0
        self.total_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
    def reset_daily_metrics(self):
        """Reset dagelijkse metrics"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info(f"Daily metrics gereset voor {current_date}")
    
    def update_daily_pnl(self, pnl: float):
        """Update dagelijkse P&L"""
        self.reset_daily_metrics()
        self.daily_pnl += pnl
        logger.debug(f"Daily P&L updated: {self.daily_pnl:.2f}")
        
    def calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Bereken position size gebaseerd op per-trade risk
        
        Args:
            symbol: Trading symbol
            signal: Signaal dictionary met entry, stop, confidence
            
        Returns:
            Position size in units
        """
        try:
            entry_price = signal['entry']
            stop_price = signal['stop']
            confidence = signal.get('confidence', 0.5)
            
            # Bereken risk per unit
            risk_per_unit = abs(entry_price - stop_price)
            if risk_per_unit <= 0:
                logger.warning(f"Invalid risk per unit for {symbol}: {risk_per_unit}")
                return 0.0
            
            # Gebruik totale portfolio waarde voor risk berekening
            portfolio_value = self.get_portfolio_value()
            max_risk_amount = portfolio_value * self.per_trade_risk_pct
            
            # Bereken position size gebaseerd op risk
            position_size = max_risk_amount / risk_per_unit
            
            # ⚠️ SAFETY FIX: More conservative confidence adjustment
            # Use non-linear scaling to be less aggressive
            confidence = max(0.3, confidence)  # Min 30% instead of 50%
            confidence_factor = confidence ** 1.5  # Non-linear makes it more conservative
            position_size *= confidence_factor
            
            # ⚠️ SAFETY FIX: Hard cap at 20% of portfolio
            MAX_POSITION_PCT = 0.20  # NEVER more than 20% in one position
            max_position_value_hard_cap = portfolio_value * MAX_POSITION_PCT
            max_position_size_hard_cap = max_position_value_hard_cap / entry_price
            
            if position_size * entry_price > max_position_value_hard_cap:
                logger.warning(
                    f"Position size capped at {MAX_POSITION_PCT:.0%} of portfolio for {symbol}: "
                    f"{position_size * entry_price:.2f} → {max_position_value_hard_cap:.2f}"
                )
                position_size = max_position_size_hard_cap
            
            # Limiteer tot beschikbare cash (max 90% van cash)
            max_position_value = self.cash * 0.90
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            # Minimum position size check
            min_position_value = portfolio_value * self.min_position_size
            min_position_size = min_position_value / entry_price
            
            if position_size < min_position_size:
                logger.info(f"Position size te klein voor {symbol}: {position_size:.4f} < {min_position_size:.4f}")
                return 0.0
            
            logger.debug(f"Position size berekening {symbol}: risk={max_risk_amount:.2f}, size={position_size:.4f}")
            return max(0.0, position_size)
            
        except Exception as e:
            logger.error(f"Fout bij position size berekening voor {symbol}: {e}")
            return 0.0
    
    def get_portfolio_value(self) -> float:
        """Bereken totale portfolio waarde"""
        try:
            # Basis cash
            total_value = self.cash
            
            # Voeg waarde van open posities toe (gebruik entry prijzen voor conservatieve schatting)
            for position in self.positions.values():
                position_value = position.size * position.entry_price
                total_value += position_value
            
            return total_value
        except Exception as e:
            logger.error(f"Fout bij portfolio waarde berekening: {e}")
            return self.cash
    
    def can_open_new_position(self, symbol: str, confidence: float = 0.5) -> bool:
        """Check of nieuwe positie geopend kan worden"""
        try:
            # Check max open positions
            if len(self.positions) >= self.max_positions:
                logger.debug(f"Max positions bereikt: {len(self.positions)}/{self.max_positions}")
                return False
            
            # Check of symbol al open positie heeft
            if symbol in self.positions:
                logger.debug(f"Positie voor {symbol} bestaat al")
                return False
            
            # Check confidence threshold
            confidence_threshold = self.config.get('confidence_threshold', 0.65)
            if confidence < confidence_threshold:
                logger.info(f"Confidence te laag voor {symbol}: {confidence:.3f} < {confidence_threshold}")
                return False
            
            # Check daily loss cap
            portfolio_value = self.get_portfolio_value()
            daily_loss_limit = portfolio_value * self.daily_loss_cap_pct
            if self.daily_pnl < -daily_loss_limit:
                logger.warning(f"Daily loss cap bereikt: {self.daily_pnl:.2f} < -{daily_loss_limit:.2f}")
                return False
            
            # Check beschikbare cash
            if self.cash <= 0:
                logger.warning("Geen beschikbare cash")
                return False
            
            # Check drawdown limiet
            current_drawdown = self.calculate_drawdown()
            if current_drawdown > self.max_drawdown:
                logger.warning(f"Drawdown limiet bereikt: {current_drawdown:.2%} > {self.max_drawdown:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fout bij can_open_new_position check: {e}")
            return False
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    stop_price: float, take_profit_price: Optional[float] = None, 
                    strategy: str = "unknown"):
        """Voeg nieuwe positie toe"""
        try:
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                entry_time=datetime.now(),
                size=size,
                stop_loss=stop_price,
                take_profit=take_profit_price or 0.0,
                max_loss=abs(entry_price - stop_price) * size
            )
            
            self.positions[symbol] = position
            
            # Update cash
            position_value = size * entry_price
            self.cash -= position_value
            
            logger.info(f"Positie toegevoegd: {symbol} {side} {size:.4f} @ {entry_price:.4f}")
            
        except Exception as e:
            logger.error(f"Fout bij toevoegen positie: {e}")
    
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        side: str, 
        atr: Optional[float] = None
    ) -> float:
        """
        Bereken stop-loss niveau
        
        Args:
            entry_price: Entry prijs
            side: 'long' of 'short'
            atr: Average True Range (optioneel)
        """
        if atr is not None:
            # Gebruik ATR-gebaseerde stop-loss
            atr_multiplier = 2.0
            stop_distance = atr * atr_multiplier
        else:
            # Gebruik percentage-gebaseerde stop-loss
            stop_distance = entry_price * self.stop_loss_pct
        
        if side == 'long':
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        side: str, 
        risk_reward_ratio: float = 3.0
    ) -> float:
        """
        Bereken take-profit niveau
        
        Args:
            entry_price: Entry prijs
            side: 'long' of 'short'
            risk_reward_ratio: Risk/reward ratio
        """
        # Bereken stop-loss afstand
        stop_distance = entry_price * self.stop_loss_pct
        
        # Take-profit afstand = stop-loss afstand * risk/reward ratio
        tp_distance = stop_distance * risk_reward_ratio
        
        if side == 'long':
            take_profit = entry_price + tp_distance
        else:  # short
            take_profit = entry_price - tp_distance
        
        return take_profit
    
    
    def open_position(
        self, 
        symbol: str, 
        side: str, 
        entry_price: float, 
        confidence: float,
        atr: Optional[float] = None
    ) -> Optional[Position]:
        """
        Open nieuwe trading positie
        
        Args:
            symbol: Trading symbol
            side: 'long' of 'short'
            entry_price: Entry prijs
            confidence: Model confidence
            atr: Average True Range voor stop-loss berekening
        """
        if not self.can_open_new_position(symbol, confidence):
            return None
        
        # Bereken position size (gebruik nieuwe signature)
        signal = {
            'entry': entry_price,
            'stop': self.calculate_stop_loss(entry_price, side, atr),
            'confidence': confidence
        }
        position_size = self.calculate_position_size(symbol, signal)
        
        # Bereken stop-loss en take-profit
        stop_loss = self.calculate_stop_loss(entry_price, side, atr)
        take_profit = self.calculate_take_profit(entry_price, side)
        
        # Maak positie
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(),
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=entry_price if self.trailing_stop else None
        )
        
        # Update portfolio
        self.positions[symbol] = position
        position_value = position_size * entry_price
        self.cash -= position_value
        
        logger.info(f"Positie geopend: {symbol} {side} @ {entry_price:.2f}, size: {position_size:.4f}")
        return position
    
    def update_position(
        self, 
        symbol: str, 
        current_price: float, 
        atr: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update positie met huidige prijs en check exit condities
        
        Args:
            symbol: Trading symbol
            current_price: Huidige prijs
            atr: Average True Range voor trailing stop
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Bereken P&L
        if position.side == 'long':
            position.pnl = (current_price - position.entry_price) * position.size
            position.pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            position.pnl = (position.entry_price - current_price) * position.size
            position.pnl_pct = (position.entry_price - current_price) / position.entry_price
        
        # Check stop-loss
        if self._check_stop_loss(position, current_price):
            return self._close_position(symbol, current_price, 'stop_loss')
        
        # Check take-profit
        if self._check_take_profit(position, current_price):
            return self._close_position(symbol, current_price, 'take_profit')
        
        # Update trailing stop
        if self.trailing_stop and position.trailing_stop is not None:
            self._update_trailing_stop(position, current_price, atr)
        
        return None
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check of stop-loss is getriggerd"""
        if position.side == 'long':
            return current_price <= position.stop_loss
        else:  # short
            return current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check of take-profit is getriggerd"""
        if position.side == 'long':
            return current_price >= position.take_profit
        else:  # short
            return current_price <= position.take_profit
    
    def _update_trailing_stop(self, position: Position, current_price: float, atr: Optional[float] = None):
        """Update trailing stop"""
        if position.side == 'long':
            # Voor long posities: verhoog trailing stop als prijs stijgt
            if current_price > position.entry_price:
                if atr is not None:
                    new_trailing_stop = current_price - (atr * 2)
                else:
                    new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                
                if new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    position.stop_loss = new_trailing_stop
        else:  # short
            # Voor short posities: verlaag trailing stop als prijs daalt
            if current_price < position.entry_price:
                if atr is not None:
                    new_trailing_stop = current_price + (atr * 2)
                else:
                    new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
                
                if new_trailing_stop < position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    position.stop_loss = new_trailing_stop
    
    def _close_position(self, symbol: str, current_price: float, reason: str) -> Dict[str, Any]:
        """Sluit positie"""
        position = self.positions[symbol]
        close_time = datetime.now()
        
        # Bereken exit waarde
        exit_value = position.size * current_price
        
        # Bereken P&L
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - current_price) * position.size
        
        # Update portfolio
        self.cash += exit_value
        self.realized_pnl += pnl
        self.current_balance = self.cash + self._calculate_positions_value(current_price)
        # Update daily P&L
        try:
            self.update_daily_pnl(pnl)
        except Exception as e:
            logger.warning(f"Kon daily P&L niet updaten: {e}")
        
        # Verwijder positie
        del self.positions[symbol]
        
        logger.info(f"Positie gesloten: {symbol} @ {current_price:.2f}, P&L: {pnl:.2f}, reden: {reason}")
        
        # Update trade stats
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'size': position.size,
            'pnl': pnl,
            'pnl_pct': pnl / (position.entry_price * position.size),
            'reason': reason,
            'entry_time': position.entry_time,
            'exit_time': close_time,
            'duration': close_time - position.entry_time
        }
        self.trade_history.append(trade_record)
        self.total_trades += 1
        if pnl >= 0:
            self.win_trades += 1
            self.gross_profit += pnl
        else:
            self.loss_trades += 1
            self.gross_loss += abs(pnl)

        return {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'size': position.size,
            'entry_time': position.entry_time,
            'exit_time': close_time,
            'pnl': pnl,
            'pnl_pct': pnl / (position.entry_price * position.size),
            'reason': reason,
            'duration': close_time - position.entry_time
        }
    
    def _calculate_positions_value(self, current_price: float) -> float:
        """Bereken totale waarde van alle posities"""
        total_value = 0.0
        for position in self.positions.values():
            if position.side == 'long':
                total_value += position.size * current_price
            else:  # short
                total_value += position.size * (2 * position.entry_price - current_price)
        return total_value
    
    def calculate_drawdown(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Bereken huidige drawdown"""
        if current_prices is None:
            # Gebruik entry prijzen als fallback
            current_value = self.cash + self._calculate_positions_value(0)
        else:
            # Gebruik huidige marktprijzen
            current_value = self.cash
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    if position.side == 'long':
                        current_value += position.size * current_price
                    else:  # short
                        current_value += position.size * (2 * position.entry_price - current_price)
                else:
                    # Fallback naar entry prijs
                    current_value += position.size * position.entry_price
        
        self.peak_balance = max(self.peak_balance, current_value)
        drawdown = (self.peak_balance - current_value) / self.peak_balance
        return drawdown
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Bereken Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Dagelijkse risk-free rate
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
        return sharpe
    
    def get_portfolio_status(self, current_prices: Optional[Dict[str, float]] = None) -> Portfolio:
        """Krijg huidige portfolio status"""
        # Bereken unrealized P&L
        unrealized_pnl = 0.0
        for position in self.positions.values():
            unrealized_pnl += position.pnl
        
        # Totale waarde
        if current_prices is None:
            total_value = self.cash + self._calculate_positions_value(0)  # Conservatieve schatting
        else:
            total_value = self.cash
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    if position.side == 'long':
                        total_value += position.size * current_price
                    else:  # short
                        total_value += position.size * (2 * position.entry_price - current_price)
                else:
                    # Fallback naar entry prijs
                    total_value += position.size * position.entry_price
        
        # P&L percentages
        total_pnl = self.realized_pnl + unrealized_pnl
        pnl_pct = total_pnl / self.initial_balance
        
        # Drawdown
        max_dd = self.calculate_drawdown()
        
        # Sharpe ratio (vereenvoudigd)
        sharpe = 0.0  # Zou berekend moeten worden met historische returns
        
        portfolio = Portfolio(
            total_value=total_value,
            cash=self.cash,
            positions=self.positions.copy(),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_pnl=total_pnl,
            pnl_pct=pnl_pct,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            daily_pnl=self.daily_pnl
        )
        
        self.portfolio_history.append(portfolio)
        return portfolio
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Krijg risico metrics"""
        portfolio = self.get_portfolio_status()
        
        # Position concentration
        total_exposure = sum(pos.size * pos.entry_price for pos in self.positions.values())
        concentration = total_exposure / portfolio.total_value if portfolio.total_value > 0 else 0
        
        # Correlation risk (vereenvoudigd)
        correlation_risk = len(self.positions) / self.max_positions

        # Win rate and trade stats
        win_rate = (self.win_trades / self.total_trades) if self.total_trades else 0.0
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else None
        
        return {
            'current_drawdown': portfolio.max_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'position_concentration': concentration,
            'max_concentration': self.max_position_size,
            'open_positions': len(self.positions),
            'max_positions': self.max_positions,
            'correlation_risk': correlation_risk,
            'cash_ratio': portfolio.cash / portfolio.total_value if portfolio.total_value > 0 else 1.0,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'profit_factor': profit_factor,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'wins': self.win_trades,
            'losses': self.loss_trades,
            'daily_pnl': self.daily_pnl,
            'total_pnl': portfolio.total_pnl,
            'cash': portfolio.cash,
            'sharpe_ratio': portfolio.sharpe_ratio
        }
    
    def emergency_close_all(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Noodgeval: sluit alle posities"""
        closed_positions = []
        
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                result = self._close_position(symbol, current_prices[symbol], 'emergency')
                closed_positions.append(result)
        
        logger.warning(f"Alle posities gesloten in noodgeval: {len(closed_positions)} posities")
        return closed_positions
