"""
Trade Executor voor Trading Bot v4
Voert trades uit met risicomanagement en cooldown controle
"""
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from loguru import logger


class TradeExecutor:
    """Trade executor met risicomanagement en cooldown controle"""
    
    def __init__(self, broker, risk_manager, cooldown_bars: int):
        self.broker = broker
        self.rm = risk_manager
        self.cooldown_bars = cooldown_bars
        self._last_trade_bar = {}  # Track laatste trade per symbol
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date = datetime.now().date()
    
    def maybe_execute(self, symbol: str, signal: Dict[str, Any], bar_ts: datetime) -> Optional[Dict[str, Any]]:
        """
        Voer trade uit indien alle condities voldaan zijn
        
        Args:
            symbol: Trading symbol
            signal: Signaal dictionary van strategie
            bar_ts: Timestamp van huidige bar
            
        Returns:
            Trade result dictionary of None
        """
        try:
            # Reset daily counters als nieuwe dag
            self._reset_daily_counters()
            
            # Check cooldown
            if not self._check_cooldown(symbol, bar_ts):
                logger.debug(f"Cooldown actief voor {symbol}")
                return None
            
            # Bereken position size eerst
            size = self.rm.calculate_position_size(symbol, signal)
            if size <= 0:
                logger.debug(f"Position size te klein voor {symbol}: {size}")
                return None
            
            # Check risk limits met berekende position size
            if not self._check_risk_limits(symbol, signal, size):
                logger.debug(f"Risk limits overschreden voor {symbol}")
                return None
            
            # Plaats order via broker
            trade_result = self._place_order(symbol, signal, size)
            
            if trade_result:
                # Update tracking
                self._last_trade_bar[symbol] = bar_ts
                self._daily_trades += 1
                
                logger.info(f"✅ Trade uitgevoerd: {symbol} {signal['side']} "
                           f"size={size:.4f} @ {signal['entry']:.4f}")
                # Update RiskManager daily P&L baseline via zero change (ensures daily reset)
                self.rm.update_daily_pnl(0.0)
                
                return trade_result
            else:
                logger.warning(f"❌ Trade gefaald voor {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Fout bij trade uitvoering voor {symbol}: {e}")
            return None
    
    def _check_cooldown(self, symbol: str, bar_ts: datetime) -> bool:
        """Check of cooldown periode is verstreken"""
        if symbol not in self._last_trade_bar:
            return True
        
        last_trade = self._last_trade_bar[symbol]
        time_diff = bar_ts - last_trade
        
        # Cooldown in uren
        cooldown_hours = self.cooldown_bars
        if time_diff.total_seconds() < (cooldown_hours * 3600):
            return False
        
        return True
    
    def _check_risk_limits(self, symbol: str, signal: Dict[str, Any], position_size: float) -> bool:
        """Check risk management limits"""
        try:
            # Check max open positions with signal confidence
            confidence = signal.get('confidence', 0.65)  # Match RiskManager default threshold
            if not self.rm.can_open_new_position(symbol, confidence):
                logger.debug(f"Max open positions bereikt of confidence te laag: {confidence:.3f}")
                return False
            
            # Check daily loss cap - gebruik get_portfolio_value() methode
            portfolio_value = self.rm.get_portfolio_value()
            daily_loss_cap = self.rm.config.get('daily_loss_cap_pct', 0.04)
            daily_loss_limit = portfolio_value * daily_loss_cap
            
            if self._daily_pnl < -daily_loss_limit:
                logger.warning(f"Daily loss cap bereikt: {self._daily_pnl:.2f} < -{daily_loss_limit:.2f}")
                return False
            
            # Check per-trade risk met berekende position size
            risk_amount = abs(signal['entry'] - signal['stop']) * position_size
            per_trade_risk_pct = self.rm.config.get('per_trade_risk_pct', 0.0075)
            max_risk = portfolio_value * per_trade_risk_pct
            
            if risk_amount > max_risk:
                logger.debug(f"Per-trade risk te hoog: {risk_amount:.2f} > {max_risk:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Fout bij risk check: {e}")
            return False
    
    def _place_order(self, symbol: str, signal: Dict[str, Any], size: float) -> Optional[Dict[str, Any]]:
        """Plaats order via broker"""
        try:
            side = signal['side']
            entry_price = signal['entry']
            stop_price = signal['stop']
            take_profit = signal.get('take_profit')
            
            # Plaats bracket order (entry + stop + take profit)
            order_result = self.broker.submit_bracket_order(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit
            )
            
            if order_result and order_result.get('success'):
                # Update portfolio
                self.rm.add_position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit,
                    strategy=signal.get('meta', {}).get('strategy', 'unknown')
                )
                
                return {
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'take_profit_price': take_profit,
                    'strategy': signal.get('meta', {}).get('strategy', 'unknown'),
                    'confidence': signal.get('confidence', 0.0),
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('order_id'),
                    'success': True
                }
            else:
                logger.error(f"Order gefaald: {order_result}")
                return None
                
        except Exception as e:
            logger.error(f"Fout bij order plaatsing: {e}")
            return None
    
    def _reset_daily_counters(self):
        """Reset daily counters bij nieuwe dag"""
        current_date = datetime.now().date()
        if current_date != self._last_reset_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = current_date
            logger.info("Daily counters gereset")
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L"""
        self._daily_pnl += pnl_change
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Krijg executor status"""
        return {
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'last_reset_date': self._last_reset_date.isoformat(),
            'cooldown_bars': self.cooldown_bars,
            'active_cooldowns': {
                symbol: (datetime.now() - last_trade).total_seconds() / 3600
                for symbol, last_trade in self._last_trade_bar.items()
            }
        }
