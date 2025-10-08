"""
Scheduler voor Trading Bot v4
Beheert periodieke taken en live data updates
"""
import schedule
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional, List
from loguru import logger
import pandas as pd


class TradingScheduler:
    """Scheduler voor trading bot taken"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.tasks = {}
        self.last_execution = {}
        
    def add_task(self, name: str, func: Callable, interval: str, **kwargs) -> None:
        """
        Voeg periodieke taak toe
        
        Args:
            name: Taak naam
            func: Functie om uit te voeren
            interval: Interval ('1h', '4h', '1d', 'weekly', etc.)
            **kwargs: Extra argumenten voor de functie
        """
        try:
            # Converteer interval naar schedule format
            schedule_func = self._get_schedule_func(interval)
            if schedule_func:
                schedule_func.do(self._wrap_task, name, func, **kwargs)
                self.tasks[name] = {
                    'func': func,
                    'interval': interval,
                    'kwargs': kwargs,
                    'last_run': None,
                    'success_count': 0,
                    'error_count': 0
                }
                logger.info(f"Taak toegevoegd: {name} (elke {interval})")
            else:
                logger.error(f"Ongeldig interval: {interval}")
                
        except Exception as e:
            logger.error(f"Fout bij toevoegen taak {name}: {e}")
    
    def _get_schedule_func(self, interval: str):
        """Converteer interval string naar schedule functie"""
        interval_map = {
            '1m': schedule.every(1).minute,
            '5m': schedule.every(5).minutes,
            '15m': schedule.every(15).minutes,
            '30m': schedule.every(30).minutes,
            '1h': schedule.every(1).hour,
            '2h': schedule.every(2).hours,
            '4h': schedule.every(4).hours,
            '6h': schedule.every(6).hours,
            '12h': schedule.every(12).hours,
            '1d': schedule.every(1).day,
            'weekly': schedule.every(1).week,
            'monthly': schedule.every(30).days  # 30 dagen als benadering voor maand
        }
        return interval_map.get(interval)
    
    def _wrap_task(self, name: str, func: Callable, **kwargs) -> None:
        """Wrapper voor taak uitvoering met error handling"""
        try:
            start_time = datetime.now()
            logger.info(f"🔄 Uitvoeren taak: {name}")
            
            # Voer functie uit
            result = func(**kwargs)
            
            # Update statistieken
            if name in self.tasks:
                self.tasks[name]['last_run'] = start_time
                self.tasks[name]['success_count'] += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Taak voltooid: {name} ({duration:.1f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Fout bij taak {name}: {e}")
            
            # Update error statistieken
            if name in self.tasks:
                self.tasks[name]['error_count'] += 1
            
            # Re-raise voor schedule library
            raise
    
    def start(self) -> None:
        """Start scheduler in aparte thread"""
        if self.running:
            logger.warning("Scheduler draait al")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info("🚀 Scheduler gestart")
    
    def stop(self) -> None:
        """Stop scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 Scheduler gestopt")
    
    def _run_scheduler(self) -> None:
        """Main scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)  # Check elke seconde
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)  # Wacht 5 seconden bij error
    
    def get_status(self) -> Dict[str, Any]:
        """Krijg scheduler status"""
        return {
            'running': self.running,
            'tasks': {
                name: {
                    'interval': task['interval'],
                    'last_run': task['last_run'].isoformat() if task['last_run'] else None,
                    'success_count': task['success_count'],
                    'error_count': task['error_count']
                }
                for name, task in self.tasks.items()
            }
        }
    
    def run_task_now(self, name: str) -> bool:
        """Voer taak direct uit"""
        if name not in self.tasks:
            logger.error(f"Taak niet gevonden: {name}")
            return False
        
        try:
            task = self.tasks[name]
            logger.info(f"🔄 Handmatig uitvoeren taak: {name}")
            
            result = task['func'](**task['kwargs'])
            
            # Update statistieken
            task['last_run'] = datetime.now()
            task['success_count'] += 1
            
            logger.info(f"✅ Handmatige taak voltooid: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fout bij handmatige taak {name}: {e}")
            if name in self.tasks:
                self.tasks[name]['error_count'] += 1
            return False


class DataFeedScheduler:
    """Speciale scheduler voor data feeds"""
    
    def __init__(self, data_manager, symbols: list, timeframes: list):
        self.data_manager = data_manager
        self.symbols = symbols
        self.timeframes = timeframes
        self.scheduler = TradingScheduler()
        self.last_update = {}
        
    def setup_data_feeds(self) -> None:
        """Setup periodieke data updates"""
        # 1-minuut updates voor 1h timeframe
        self.scheduler.add_task(
            'update_1h_data',
            self._update_1h_data,
            '1m'
        )
        
        # 5-minuut updates voor 4h timeframe
        self.scheduler.add_task(
            'update_4h_data',
            self._update_4h_data,
            '5m'
        )
        
        # 1-uur updates voor 1d timeframe
        self.scheduler.add_task(
            'update_1d_data',
            self._update_1d_data,
            '1h'
        )
        
        # Universe selectie (wekelijks)
        self.scheduler.add_task(
            'universe_selection',
            self._run_universe_selection,
            'weekly'
        )
        
        logger.info("Data feeds geconfigureerd")
    
    def _update_1h_data(self) -> None:
        """Update 1h data voor alle symbols"""
        try:
            for symbol in self.symbols:
                self.data_manager.get_historical_data(symbol, interval='1h', period='1d')
            self.last_update['1h'] = datetime.now()
            logger.debug("1h data bijgewerkt")
        except Exception as e:
            logger.error(f"Fout bij 1h data update: {e}")
    
    def _update_4h_data(self) -> None:
        """Update 4h data voor alle symbols"""
        try:
            for symbol in self.symbols:
                self.data_manager.get_historical_data(symbol, interval='4h', period='7d')
            self.last_update['4h'] = datetime.now()
            logger.debug("4h data bijgewerkt")
        except Exception as e:
            logger.error(f"Fout bij 4h data update: {e}")
    
    def _update_1d_data(self) -> None:
        """Update 1d data voor alle symbols"""
        try:
            for symbol in self.symbols:
                self.data_manager.get_historical_data(symbol, interval='1d', period='30d')
            self.last_update['1d'] = datetime.now()
            logger.debug("1d data bijgewerkt")
        except Exception as e:
            logger.error(f"Fout bij 1d data update: {e}")
    
    def _run_universe_selection(self) -> None:
        """Voer universe selectie uit"""
        try:
            # Dit zou de universe selector moeten aanroepen
            logger.info("Universe selectie uitgevoerd")
        except Exception as e:
            logger.error(f"Fout bij universe selectie: {e}")
    
    def start(self) -> None:
        """Start data feed scheduler"""
        self.scheduler.start()
        logger.info("Data feed scheduler gestart")
    
    def stop(self) -> None:
        """Stop data feed scheduler"""
        self.scheduler.stop()
        logger.info("Data feed scheduler gestopt")
    
    def get_status(self) -> Dict[str, Any]:
        """Krijg data feed status"""
        status = self.scheduler.get_status()
        status['last_updates'] = {
            timeframe: ts.isoformat() if ts else None
            for timeframe, ts in self.last_update.items()
        }
        return status


class WebSocketDataFeedScheduler:
    """Scheduler voor WebSocket data feeds"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.running = False
        self.loop = None
        self.thread = None
        self.data_feed = None
        
        # Import here to avoid circular imports
        from src.data.streams.kraken_ws import WebSocketDataFeed
        
        self.data_feed = WebSocketDataFeed(symbols)
        
    def start(self):
        """Start WebSocket data feed in background thread"""
        if self.running:
            logger.warning("WebSocket data feed already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        logger.info(f"WebSocket data feed started for symbols: {self.symbols}")
    
    def stop(self):
        """Stop WebSocket data feed"""
        if not self.running:
            return
            
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.data_feed.stop(), self.loop)
        
        if self.thread:
            self.thread.join(timeout=5)
            
        logger.info("WebSocket data feed stopped")
    
    def _run_async_loop(self):
        """Run async event loop in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._start_feed())
        except Exception as e:
            logger.error(f"Error in WebSocket data feed: {e}")
        finally:
            self.loop.close()
    
    async def _start_feed(self):
        """Start the data feed"""
        try:
            await self.data_feed.start()
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in WebSocket feed: {e}")
        finally:
            await self.data_feed.stop()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        if self.data_feed:
            return self.data_feed.get_latest_price(symbol)
        return None
    
    def get_latest_trades(self, symbol: str, limit: int = 10) -> List:
        """Get latest trades for symbol"""
        if self.data_feed:
            return self.data_feed.get_latest_trades(symbol, limit)
        return []
    
    def get_orderbook(self, symbol: str):
        """Get latest order book for symbol"""
        if self.data_feed:
            return self.data_feed.get_orderbook(symbol)
        return None
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.data_feed and self.data_feed.client.connected


# Global scheduler instance
trading_scheduler = TradingScheduler()
