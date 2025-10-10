"""
Broker Interface voor Trading Bot v4
Abstracte broker interface voor paper en live trading
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from datetime import datetime
from loguru import logger
import uuid
import time


class BrokerInterface(ABC):
    """Abstracte broker interface"""
    
    @abstractmethod
    def submit_bracket_order(self, symbol: str, side: str, size: float, 
                           entry_price: float, stop_price: float, 
                           take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """Plaats bracket order (entry + stop + take profit)"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """Krijg account balance"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Krijg open posities"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Krijg order status"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Annuleer order"""
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Krijg alle open orders"""
        pass


class PaperBroker(BrokerInterface):
    """Paper trading broker voor testing"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions = {}
        self.order_id_counter = 1
        self.trade_history = []
        self.open_orders = {}  # Track open orders
        self.order_statuses = {}  # Track order statuses
        
        # Initialize trade logger
        try:
            from src.audit.trade_logger import trade_logger
            self.trade_logger = trade_logger
            self.audit_enabled = True
        except ImportError:
            self.trade_logger = None
            self.audit_enabled = False
            logger.warning("Trade logger not available - audit disabled")
        
    def submit_bracket_order(self, symbol: str, side: str, size: float, 
                           entry_price: float, stop_price: float, 
                           take_profit_price: Optional[float] = None,
                           model_id: Optional[str] = None,
                           model_ver: Optional[str] = None,
                           features_hash: Optional[str] = None,
                           config_hash: Optional[str] = None) -> Dict[str, Any]:
        """Simuleer bracket order"""
        start_time = time.time()
        req_id = str(uuid.uuid4())
        
        try:
            # Bereken order waarde
            order_value = size * entry_price
            
            # Check of we voldoende balance hebben
            if order_value > self.balance:
                logger.warning(f"Onvoldoende balance: {order_value:.2f} > {self.balance:.2f}")
                
                # Log rejected trade
                if self.audit_enabled:
                    self.trade_logger.log_trade_submission(
                        req_id=req_id,
                        side=side,
                        symbol=symbol,
                        qty_req=size,
                        order_type="market",
                        limit_px=entry_price,
                        sl_px=stop_price,
                        tp_px=take_profit_price,
                        model_id=model_id,
                        model_ver=model_ver,
                        features_hash=features_hash,
                        config_hash=config_hash
                    )
                    self.trade_logger.log_trade_update(
                        req_id=req_id,
                        status="rejected",
                        reason="Insufficient balance",
                        latency_ms=(time.time() - start_time) * 1000
                    )
                
                return {'success': False, 'error': 'Insufficient balance'}
            
            # Genereer order ID
            order_id = f"PAPER_{self.order_id_counter}"
            self.order_id_counter += 1
            
            # Simuleer order uitvoering
            self.balance -= order_value
            
            # Voeg positie toe
            self.positions[symbol] = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'order_id': order_id,
                'timestamp': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            # Track open order
            self.open_orders[order_id] = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'status': 'open',
                'timestamp': datetime.now()
            }
            
            # Update order status
            self.update_order_status(order_id, 'filled', 
                                   symbol=symbol, side=side, size=size, 
                                   entry_price=entry_price)
            
            # Log trade
            trade_record = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'timestamp': datetime.now(),
                'type': 'entry'
            }
            self.trade_history.append(trade_record)
            
            # Log to audit trail
            if self.audit_enabled:
                self.trade_logger.log_trade_submission(
                    req_id=req_id,
                    side=side,
                    symbol=symbol,
                    qty_req=size,
                    order_type="market",
                    limit_px=entry_price,
                    sl_px=stop_price,
                    tp_px=take_profit_price,
                    model_id=model_id,
                    model_ver=model_ver,
                    features_hash=features_hash,
                    config_hash=config_hash
                )
                self.trade_logger.log_trade_update(
                    req_id=req_id,
                    order_id=order_id,
                    status="filled",
                    qty_filled=size,
                    latency_ms=(time.time() - start_time) * 1000,
                    balance_after=self.balance
                )
            
            logger.info(f"Paper order geplaatst: {symbol} {side} {size:.4f} @ {entry_price:.4f}")
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'req_id': req_id
            }
            
        except Exception as e:
            logger.error(f"Fout bij paper order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_account_balance(self) -> Dict[str, float]:
        """Krijg paper account balance"""
        return {
            'cash': self.balance,
            'total_value': self.balance + self._calculate_positions_value(),
            'positions_value': self._calculate_positions_value()
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Krijg paper posities"""
        return self.positions.copy()
    
    def _calculate_positions_value(self) -> float:
        """Bereken totale waarde van posities"""
        total_value = 0.0
        for symbol, position in self.positions.items():
            # Voor paper trading gebruiken we entry price als huidige waarde
            # In echte implementatie zou je live prijzen ophalen
            position_value = position['size'] * position['entry_price']
            if position['side'] == 'sell':
                position_value = -position_value  # Short posities zijn negatief
            total_value += position_value
        return total_value
    
    def close_position(self, symbol: str, exit_price: float) -> Dict[str, Any]:
        """Sluit positie"""
        if symbol not in self.positions:
            return {'success': False, 'error': 'Position not found'}
        
        position = self.positions[symbol]
        
        # Bereken P&L
        if position['side'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Update balance (alleen de exit waarde, P&L is al inbegrepen)
        self.balance += position['size'] * exit_price
        
        # Log exit trade
        trade_record = {
            'order_id': f"EXIT_{self.order_id_counter}",
            'symbol': symbol,
            'side': 'close',
            'size': position['size'],
            'exit_price': exit_price,
            'pnl': pnl,
            'timestamp': datetime.now(),
            'type': 'exit'
        }
        self.trade_history.append(trade_record)
        self.order_id_counter += 1
        
        # Verwijder positie
        del self.positions[symbol]
        
        logger.info(f"Paper positie gesloten: {symbol} @ {exit_price:.4f}, P&L: {pnl:.2f}")
        
        return {
            'success': True,
            'symbol': symbol,
            'exit_price': exit_price,
            'pnl': pnl
        }
    
    def get_trade_history(self) -> list:
        """Krijg trade geschiedenis"""
        return self.trade_history.copy()
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Krijg order status"""
        if order_id in self.order_statuses:
            return self.order_statuses[order_id]
        else:
            return {'success': False, 'error': 'Order not found'}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Annuleer order"""
        if order_id in self.open_orders:
            # Mark order as cancelled
            self.order_statuses[order_id] = {
                'order_id': order_id,
                'status': 'cancelled',
                'timestamp': datetime.now()
            }
            
            # Remove from open orders
            del self.open_orders[order_id]
            
            logger.info(f"Paper order geannuleerd: {order_id}")
            return {'success': True, 'order_id': order_id, 'status': 'cancelled'}
        else:
            return {'success': False, 'error': 'Order not found or already closed'}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Krijg alle open orders"""
        return list(self.open_orders.values())
    
    def update_order_status(self, order_id: str, status: str, **kwargs):
        """Update order status"""
        self.order_statuses[order_id] = {
            'order_id': order_id,
            'status': status,
            'timestamp': datetime.now(),
            **kwargs
        }


class LiveBroker(BrokerInterface):
    """Live trading broker met Kraken API"""
    
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.positions = {}
        self.order_id_counter = 1
        self.open_orders = {}  # Track open orders
        self.order_statuses = {}  # Track order statuses
        
        # Import ccxt voor exchange connectie
        try:
            import ccxt
            if sandbox:
                self.exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'sandbox': True,
                    'enableRateLimit': True,
                })
            else:
                self.exchange = ccxt.kraken({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
            logger.info(f"Live broker geïnitialiseerd (sandbox: {sandbox})")
        except ImportError:
            logger.error("CCXT niet geïnstalleerd - installeer met: pip install ccxt")
            self.exchange = None
        except Exception as e:
            logger.error(f"Fout bij initialiseren exchange: {e}")
            self.exchange = None
    
    def submit_bracket_order(self, symbol: str, side: str, size: float, 
                           entry_price: float, stop_price: float, 
                           take_profit_price: Optional[float] = None) -> Dict[str, Any]:
        """Plaats live bracket order"""
        try:
            if not self.exchange:
                return {'success': False, 'error': 'Exchange not initialized'}
            
            # Converteer symbol naar Kraken formaat
            kraken_symbol = self._convert_symbol(symbol)
            if not kraken_symbol:
                return {'success': False, 'error': f'Unsupported symbol: {symbol}'}
            
            # Plaats market order (voor nu - in productie zou je limit orders gebruiken)
            order_type = 'market'
            order_side = 'buy' if side == 'buy' else 'sell'
            
            # Plaats entry order
            entry_order = self.exchange.create_order(
                symbol=kraken_symbol,
                type=order_type,
                side=order_side,
                amount=size,
                price=None  # Market order
            )
            
            if entry_order and entry_order.get('id'):
                order_id = f"LIVE_{entry_order['id']}"
                
                # Store position info
                self.positions[symbol] = {
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'take_profit_price': take_profit_price,
                    'order_id': order_id,
                    'timestamp': datetime.now(),
                    'kraken_order_id': entry_order['id']
                }
                
                logger.info(f"Live order geplaatst: {symbol} {side} {size:.4f}")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'kraken_order_id': entry_order['id']
                }
            else:
                return {'success': False, 'error': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Fout bij live order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_account_balance(self) -> Dict[str, float]:
        """Krijg live account balance"""
        try:
            if not self.exchange:
                return {'cash': 0.0, 'total_value': 0.0, 'positions_value': 0.0}
            
            balance = self.exchange.fetch_balance()
            
            # Kraken geeft balances per currency
            total_value = 0.0
            cash = 0.0
            
            for currency, amount in balance['total'].items():
                if amount > 0:
                    # Voor nu gebruiken we USD als basis
                    if currency == 'USD':
                        cash += amount
                        total_value += amount
                    # Voor crypto zou je de USD waarde moeten berekenen
            
            return {
                'cash': cash,
                'total_value': total_value,
                'positions_value': total_value - cash
            }
            
        except Exception as e:
            logger.error(f"Fout bij ophalen balance: {e}")
            return {'cash': 0.0, 'total_value': 0.0, 'positions_value': 0.0}
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Krijg live posities"""
        try:
            if not self.exchange:
                return {}
            
            # Kraken heeft geen directe positions API, dus we gebruiken onze tracking
            return self.positions.copy()
            
        except Exception as e:
            logger.error(f"Fout bij ophalen posities: {e}")
            return {}
    
    def _convert_symbol(self, symbol: str) -> Optional[str]:
        """Converteer symbol naar Kraken formaat"""
        # Kraken gebruikt andere symbol formaten
        symbol_map = {
            # Basis-set (bestaat zeker)
            'BTC-USD': 'BTC/USD',
            'ETH-USD': 'ETH/USD',
            'SOL-USD': 'SOL/USD',
            'ADA-USD': 'ADA/USD',
            'XRP-USD': 'XRP/USD',
            'LINK-USD': 'LINK/USD',
            # Bundel-set (gevalideerd via Kraken AssetPairs)
            'APT-USD': 'APT/USD',
            'BNB-USD': 'BNB/USD',
            'CAKE-USD': 'CAKE/USD',
            'MNT-USD': 'MNT/USD',
            'WLD-USD': 'WLD/USD',
            'WLFI-USD': 'WLFI/USD',
            'XMR-USD': 'XMR/USD',
            'XPL-USD': 'XPL/USD',
        }
        return symbol_map.get(symbol)
    
    def close_position(self, symbol: str, exit_price: float) -> Dict[str, Any]:
        """Sluit live positie"""
        try:
            if symbol not in self.positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = self.positions[symbol]
            
            if not self.exchange:
                return {'success': False, 'error': 'Exchange not initialized'}
            
            # Converteer symbol
            kraken_symbol = self._convert_symbol(symbol)
            if not kraken_symbol:
                return {'success': False, 'error': f'Unsupported symbol: {symbol}'}
            
            # Plaats exit order (tegenovergestelde kant)
            exit_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            exit_order = self.exchange.create_order(
                symbol=kraken_symbol,
                type='market',
                side=exit_side,
                amount=position['size'],
                price=None
            )
            
            if exit_order and exit_order.get('id'):
                # Bereken P&L
                if position['side'] == 'buy':
                    pnl = (exit_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['size']
                
                # Verwijder positie
                del self.positions[symbol]
                
                logger.info(f"Live positie gesloten: {symbol} @ {exit_price:.4f}, P&L: {pnl:.2f}")
                
                return {
                    'success': True,
                    'symbol': symbol,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'kraken_order_id': exit_order['id']
                }
            else:
                return {'success': False, 'error': 'Exit order failed'}
                
        except Exception as e:
            logger.error(f"Fout bij sluiten positie: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Krijg order status van Kraken"""
        try:
            if not self.exchange:
                return {'success': False, 'error': 'Exchange not initialized'}
            
            # Check local cache first
            if order_id in self.order_statuses:
                return self.order_statuses[order_id]
            
            # Fetch from exchange
            order = self.exchange.fetch_order(order_id)
            
            if order:
                status = {
                    'order_id': order_id,
                    'status': order.get('status', 'unknown'),
                    'symbol': order.get('symbol', ''),
                    'side': order.get('side', ''),
                    'amount': order.get('amount', 0),
                    'price': order.get('price', 0),
                    'filled': order.get('filled', 0),
                    'remaining': order.get('remaining', 0),
                    'timestamp': datetime.now()
                }
                
                # Cache the status
                self.order_statuses[order_id] = status
                return status
            else:
                return {'success': False, 'error': 'Order not found'}
                
        except Exception as e:
            logger.error(f"Fout bij ophalen order status: {e}")
            return {'success': False, 'error': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Annuleer order op Kraken"""
        try:
            if not self.exchange:
                return {'success': False, 'error': 'Exchange not initialized'}
            
            # Cancel order on exchange
            result = self.exchange.cancel_order(order_id)
            
            if result:
                # Update local cache
                self.order_statuses[order_id] = {
                    'order_id': order_id,
                    'status': 'cancelled',
                    'timestamp': datetime.now()
                }
                
                # Remove from open orders
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
                
                logger.info(f"Live order geannuleerd: {order_id}")
                return {'success': True, 'order_id': order_id, 'status': 'cancelled'}
            else:
                return {'success': False, 'error': 'Cancel failed'}
                
        except Exception as e:
            logger.error(f"Fout bij annuleren order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Krijg alle open orders van Kraken"""
        try:
            if not self.exchange:
                return []
            
            # Fetch open orders from exchange
            orders = self.exchange.fetch_open_orders()
            
            # Convert to our format
            open_orders = []
            for order in orders:
                order_data = {
                    'order_id': order.get('id', ''),
                    'symbol': order.get('symbol', ''),
                    'side': order.get('side', ''),
                    'amount': order.get('amount', 0),
                    'price': order.get('price', 0),
                    'status': order.get('status', 'open'),
                    'timestamp': datetime.now()
                }
                open_orders.append(order_data)
                
                # Cache the order
                self.open_orders[order.get('id', '')] = order_data
            
            return open_orders
            
        except Exception as e:
            logger.error(f"Fout bij ophalen open orders: {e}")
            return []
    
    def update_order_status(self, order_id: str, status: str, **kwargs):
        """Update order status in local cache"""
        self.order_statuses[order_id] = {
            'order_id': order_id,
            'status': status,
            'timestamp': datetime.now(),
            **kwargs
        }
