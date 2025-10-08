"""
Idempotent Order Execution
Ensures orders are placed only once with deterministic client_order_id
"""
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from loguru import logger

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class IdempotentExecutor:
    """Ensures idempotent order execution"""
    
    def __init__(self, broker, order_log_path: str = "logs/orders.json"):
        self.broker = broker
        self.order_log_path = Path(order_log_path)
        self.order_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing order log
        self.order_log = self._load_order_log()
        
    def place_order_idempotent(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        features_hash: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place order with idempotent client_order_id
        
        Args:
            side: 'buy' or 'sell'
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price (optional for market orders)
            order_type: 'market' or 'limit'
            features_hash: Hash of features used for decision
            strategy_name: Name of strategy that generated the order
            
        Returns:
            Order result dictionary
        """
        try:
            # Generate deterministic client_order_id
            client_order_id = self._generate_client_order_id(
                side, symbol, quantity, price, order_type, features_hash
            )
            
            # Check if order already exists
            if client_order_id in self.order_log:
                existing_order = self.order_log[client_order_id]
                
                # Check if order is still active
                if existing_order.get('status') in ['pending', 'open', 'partially_filled']:
                    logger.info(f"Order already exists and is active: {client_order_id}")
                    return {
                        'success': True,
                        'client_order_id': client_order_id,
                        'broker_order_id': existing_order.get('broker_order_id'),
                        'status': existing_order.get('status'),
                        'message': 'Order already exists'
                    }
                else:
                    logger.info(f"Order exists but is completed: {client_order_id}")
            
            # Log order intent before placing
            order_intent = {
                'client_order_id': client_order_id,
                'side': side,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'order_type': order_type,
                'features_hash': features_hash,
                'strategy_name': strategy_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'pending'
            }
            
            self._log_order_intent(order_intent)
            
            # Place order with broker
            broker_result = self.broker.place_order(
                side=side,
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                client_order_id=client_order_id
            )
            
            # Update order log with broker result
            order_intent.update({
                'broker_order_id': broker_result.get('broker_order_id'),
                'status': broker_result.get('status', 'open'),
                'broker_result': broker_result,
                'placed_at': datetime.now().isoformat()
            })
            
            self.order_log[client_order_id] = order_intent
            self._save_order_log()
            
            logger.info(f"Order placed successfully: {client_order_id} -> {broker_result.get('broker_order_id')}")
            
            return {
                'success': True,
                'client_order_id': client_order_id,
                'broker_order_id': broker_result.get('broker_order_id'),
                'status': broker_result.get('status'),
                'broker_result': broker_result
            }
            
        except Exception as e:
            logger.error(f"Error placing idempotent order: {e}")
            return {
                'success': False,
                'error': str(e),
                'client_order_id': client_order_id if 'client_order_id' in locals() else None
            }
    
    def _generate_client_order_id(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: Optional[float],
        order_type: str,
        features_hash: Optional[str]
    ) -> str:
        """Generate deterministic client_order_id"""
        
        # Create deterministic components
        timestamp = int(time.time() // 60) * 60  # Round to minute for idempotency
        price_str = f"{price:.8f}" if price else "market"
        quantity_str = f"{quantity:.8f}"
        
        # Create hash input
        hash_input = f"{timestamp}_{symbol}_{side}_{quantity_str}_{price_str}_{order_type}"
        
        if features_hash:
            hash_input += f"_{features_hash}"
        
        # Generate hash
        hash_obj = hashlib.md5(hash_input.encode())
        hash_hex = hash_obj.hexdigest()[:12]  # Use first 12 characters
        
        # Create client_order_id
        client_order_id = f"{symbol}_{side}_{timestamp}_{hash_hex}"
        
        return client_order_id
    
    def _log_order_intent(self, order_intent: Dict[str, Any]):
        """Log order intent before placing"""
        try:
            log_entry = {
                'action': 'order_intent',
                'timestamp': datetime.now().isoformat(),
                'order': order_intent
            }
            
            # Append to log file
            with open(self.order_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.debug(f"Order intent logged: {order_intent['client_order_id']}")
            
        except Exception as e:
            logger.error(f"Error logging order intent: {e}")
    
    def _load_order_log(self) -> Dict[str, Dict[str, Any]]:
        """Load existing order log"""
        try:
            if not self.order_log_path.exists():
                return {}
            
            order_log = {}
            
            with open(self.order_log_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('action') == 'order_intent':
                            order = entry.get('order', {})
                            client_order_id = order.get('client_order_id')
                            if client_order_id:
                                order_log[client_order_id] = order
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(order_log)} orders from log")
            return order_log
            
        except Exception as e:
            logger.error(f"Error loading order log: {e}")
            return {}
    
    def _save_order_log(self):
        """Save order log to file"""
        try:
            # Save as JSON for easy reading
            with open(self.order_log_path.with_suffix('.json'), 'w') as f:
                json.dump(self.order_log, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving order log: {e}")
    
    def get_order_status(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status by client_order_id"""
        return self.order_log.get(client_order_id)
    
    def update_order_status(self, client_order_id: str, status: str, filled_qty: float = 0, avg_price: float = 0):
        """Update order status"""
        if client_order_id in self.order_log:
            self.order_log[client_order_id].update({
                'status': status,
                'filled_qty': filled_qty,
                'avg_price': avg_price,
                'updated_at': datetime.now().isoformat()
            })
            self._save_order_log()
    
    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all active orders"""
        active_statuses = ['pending', 'open', 'partially_filled']
        return {
            client_order_id: order 
            for client_order_id, order in self.order_log.items()
            if order.get('status') in active_statuses
        }
    
    def cleanup_old_orders(self, days_to_keep: int = 7):
        """Clean up old completed orders"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            orders_to_remove = []
            for client_order_id, order in self.order_log.items():
                order_time = order.get('timestamp', '')
                try:
                    order_timestamp = datetime.fromisoformat(order_time.replace('Z', '+00:00')).timestamp()
                    if order_timestamp < cutoff_date and order.get('status') in ['filled', 'cancelled', 'rejected']:
                        orders_to_remove.append(client_order_id)
                except:
                    continue
            
            for client_order_id in orders_to_remove:
                del self.order_log[client_order_id]
            
            if orders_to_remove:
                self._save_order_log()
                logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
            
        except Exception as e:
            logger.error(f"Error cleaning up old orders: {e}")


# Convenience function
def create_idempotent_executor(broker, order_log_path: str = "logs/orders.json") -> IdempotentExecutor:
    """Create idempotent executor"""
    return IdempotentExecutor(broker, order_log_path)
