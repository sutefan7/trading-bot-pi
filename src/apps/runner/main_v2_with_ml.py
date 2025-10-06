"""
Trading Bot v4 Runner with ML Overlay
Pi-based trading loop with optional ML inference and failover
"""
import sys
import os
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import ConfigManager
from data.data_manager import DataManager
from strategies.indicators import TechnicalIndicators
from risk.risk_manager import RiskManager
from utils.logger_setup import setup_logging
from loguru import logger

# Existing modules
from filters.regime_filter import RegimeFilter, RegimeConfig
from filters.universe_selector import UniverseSelector, UniverseConfig
from strategies.trend_follow import TrendFollowStrategy
from strategies.mean_revert import MeanRevertStrategy
from strategies.breakout import BreakoutStrategy
from execution.executor import TradeExecutor
from execution.broker import BrokerInterface, PaperBroker, LiveBroker
from core.secrets import secrets_manager, get_kraken_credentials
from core.scheduler import TradingScheduler, DataFeedScheduler

# New ML modules
from apps.runner.inference_client import ModelManager, create_model_manager
from apps.runner.ml_overlay import MLOverlayManager, MLOffset, create_ml_overlay_manager
from apps.runner.shadow_ml_observer import ShadowMLObserver
from apps.runner.notification_manager import create_notification_manager
from apps.runner.instance_lock import require_single_instance
# Removed dead imports (not needed for Pi version):
# from apps.runner.startup_reconcile import perform_startup_reconcile
# from execution.idempotent_executor import create_idempotent_executor
# from execution.circuit_breaker import with_resilience
# from monitoring.slo_monitor import create_slo_monitor
from features.pipeline import FeaturePipeline


class TradingBotV4WithML:
    """Trading Bot v4 with optional ML overlay and failover"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Setup logging
        setup_logging()
        logger.info("🚀 Trading Bot v4 (ML-enabled) wordt gestart...")
        
        # Single instance lock
        self.instance_lock = require_single_instance()
        logger.info("✅ Single instance lock acquired")
        
        # Laad configuratie
        self.config_manager = ConfigManager(config_path)
        if not self.config_manager.validate_config():
            raise ValueError("Configuratie validatie gefaald")
        
        # Initialiseer componenten
        self._init_components()
        
        # Trading state
        self.is_running = False
        self.last_update = None
        self.current_universe = []
        
        # ML state
        self.ml_overlay_enabled = False
        self.daily_loss_cap_triggered = False
        
        # ⚠️ PERFORMANCE: Data caching for Pi
        self._data_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl_seconds = 300  # 5 minutes cache
        
        logger.info("✅ Data caching enabled for Pi optimization")
        
        # Start ML model manager if available
        if self.model_manager:
            # Only load initial model, don't start monitoring
            self.model_manager._try_load_latest_model()
            logger.info("🤖 ML ModelManager gestart (monitoring disabled)")
        
        logger.info("✅ Trading Bot v4 (ML-enabled) succesvol geïnitialiseerd")
    
    def _init_components(self):
        """Initialiseer alle bot componenten"""
        # Data manager
        self.data_manager = DataManager("data")
        
        # Technische indicatoren
        indicators_config = self.config_manager.get_indicators_config()
        self.indicators = TechnicalIndicators(indicators_config)
        
        # Feature pipeline for ML
        self.feature_pipeline = FeaturePipeline()
        
        # Risicomanagement
        risk_config = self.config_manager.get_risk_config()
        self.risk_manager = RiskManager(risk_config.__dict__)
        
        # Trading configuratie
        self.trading_config = self.config_manager.get_trading_config()
        self.live_config = self.config_manager.get_live_config()
        
        # ML configuratie
        ml_config = self.config_manager.config_data.get('ml_overlay', {})
        self.use_ml_overlay = ml_config.get('enabled', False)
        self.artifacts_dir = ml_config.get('artifacts_dir', 'artifacts')
        self.latest_file = ml_config.get('latest_file', 'latest.txt')
        self.daily_loss_cap = ml_config.get('daily_loss_cap', 0.05)  # 5% daily loss cap
        
        # ML Overlay Manager
        self.ml_overlay_manager = create_ml_overlay_manager(ml_config)
        
        # Notification Manager
        notification_config = self.config_manager.config_data.get('notifications', {})
        self.notification_manager = create_notification_manager(notification_config)
        
        # Regime filter
        regime_config_dict = self.config_manager.config_data.get('regime', {})
        regime_config = RegimeConfig(**regime_config_dict)
        self.regime_filter = RegimeFilter(self.data_manager, self.indicators, regime_config)
        
        # Universe selector
        if hasattr(self.trading_config, 'universe'):
            universe_config = self._map_universe_config(self.trading_config.universe)
        else:
            universe_config = UniverseConfig()
        self.universe_selector = UniverseSelector(self.data_manager, universe_config)
        
        # Strategieën
        strategies_config = self.config_manager.config_data.get('strategies', {})
        self.strategies = {}
        
        if 'trend_follow' in strategies_config.get('enabled', []):
            self.strategies['trend_follow'] = TrendFollowStrategy(
                strategies_config.get('trend_follow', {}), self.indicators
            )
        
        if 'mean_revert' in strategies_config.get('enabled', []):
            self.strategies['mean_revert'] = MeanRevertStrategy(
                strategies_config.get('mean_revert', {}), self.indicators
            )
        
        if 'breakout' in strategies_config.get('enabled', []):
            self.strategies['breakout'] = BreakoutStrategy(
                strategies_config.get('breakout', {}), self.indicators
            )
        
        # ML Model Manager
        self.model_manager = create_model_manager(
            artifacts_dir=self.artifacts_dir,
            latest_file=self.latest_file,
            use_ml_overlay=self.use_ml_overlay
        )
        
        # Shadow ML Observer
        self.shadow_ml_observer = ShadowMLObserver()
        logger.info("📊 Shadow ML Observer geïnitialiseerd")
        
        # Broker
        if self.live_config.paper_trading:
            self.broker = PaperBroker(initial_balance=10000.0)
            logger.info("📄 Paper trading broker geïnitialiseerd")
        else:
            # Live broker met echte API credentials
            kraken_creds = get_kraken_credentials()
            if not kraken_creds['api_key']:
                logger.warning("Geen Kraken API credentials gevonden - fallback naar paper trading")
                self.broker = PaperBroker(initial_balance=10000.0)
            else:
                self.broker = LiveBroker(
                    api_key=kraken_creds['api_key'],
                    api_secret=kraken_creds['api_secret'],
                    sandbox=kraken_creds['sandbox']
                )
                logger.info(f"🔴 Live trading broker geïnitialiseerd (sandbox: {kraken_creds['sandbox']})")
        
        # Trade executor
        self.executor = TradeExecutor(
            self.broker, 
            self.risk_manager, 
            self.live_config.cooldown_bars
        )
        
        # Scheduler voor live trading
        self.scheduler = TradingScheduler()
        self.data_scheduler = DataFeedScheduler(
            self.data_manager, 
            self.trading_config.symbols, 
            self.trading_config.timeframes
        )
        
        logger.info(f"✅ Alle componenten geïnitialiseerd: {len(self.strategies)} strategieën")
        if self.model_manager:
            logger.info("🤖 ML overlay ingeschakeld")
        else:
            logger.info("📊 Alleen non-ML strategieën actief")
    
    def _get_cached_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        ⚠️ PERFORMANCE: Get data with caching for Pi optimization
        
        Args:
            symbol: Trading symbol
            days: Number of days of data
            
        Returns:
            DataFrame or None
        """
        try:
            cache_key = f"{symbol}_{days}d"
            current_time = datetime.now()
            
            # Check if cache exists and is fresh
            if cache_key in self._data_cache and cache_key in self._cache_timestamps:
                cache_age = (current_time - self._cache_timestamps[cache_key]).total_seconds()
                if cache_age < self._cache_ttl_seconds:
                    logger.debug(f"📦 Cache hit for {symbol} (age: {cache_age:.0f}s)")
                    return self._data_cache[cache_key]
            
            # Cache miss or stale - fetch new data
            logger.debug(f"📥 Cache miss for {symbol} - fetching data")
            df = self.data_manager.get_latest_data(symbol, days=days)
            
            if df is not None and len(df) > 0:
                # ⚠️ MEMORY: Limit to essential data only (last 200 bars max)
                MAX_CACHED_BARS = 200
                if len(df) > MAX_CACHED_BARS:
                    df = df.tail(MAX_CACHED_BARS)
                
                # Store in cache
                self._data_cache[cache_key] = df.copy()  # Copy to avoid mutations
                self._cache_timestamps[cache_key] = current_time
                
                # ⚠️ MEMORY: Cleanup old cache entries
                self._cleanup_stale_cache()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting cached data for {symbol}: {e}")
            # Fallback to direct fetch
            return self.data_manager.get_latest_data(symbol, days=days)
    
    def _cleanup_stale_cache(self):
        """
        ⚠️ MEMORY: Remove stale entries from cache
        """
        try:
            current_time = datetime.now()
            stale_keys = []
            
            for key, timestamp in self._cache_timestamps.items():
                cache_age = (current_time - timestamp).total_seconds()
                if cache_age > self._cache_ttl_seconds * 2:  # Double TTL for cleanup
                    stale_keys.append(key)
            
            for key in stale_keys:
                del self._data_cache[key]
                del self._cache_timestamps[key]
            
            if stale_keys:
                logger.debug(f"🗑️  Cleaned up {len(stale_keys)} stale cache entries")
                
                # ⚠️ MEMORY: Explicit garbage collection after cleanup
                import gc
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def run_trading_cycle(self):
        """Voer één volledige trading cycle uit met ML overlay"""
        try:
            logger.info("🔄 Trading cycle gestart...")
            
            # Check daily loss cap
            if self._check_daily_loss_cap():
                logger.warning("🚨 Daily loss cap triggered - disabling ML overlay")
                self._disable_ml_overlay()
                return
            
            # Stap 1: Check regime filter
            if not self.regime_filter.is_tradable():
                logger.info("❌ Regime filter: markt niet tradebaar")
                # Update bestaande posities maar open geen nieuwe
                self._update_existing_positions()
                return
            
            logger.info("✅ Regime filter: markt tradebaar")
            
            # Stap 2: Update universe indien nodig
            self._update_universe()
            
            # Stap 3: Genereer signalen voor universe (met ML overlay)
            signals = self._generate_signals_with_ml()
            
            # Stap 4: Voer beste signalen uit
            self._execute_signals(signals)
            
            # Stap 5: Update bestaande posities
            self._update_existing_positions()
            
            # Stap 6: Log status
            self._log_status()
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Fout in trading cycle: {e}")
    
    def _check_daily_loss_cap(self) -> bool:
        """Check if daily loss cap has been triggered"""
        try:
            portfolio = self.risk_manager.get_portfolio_status()
            daily_pnl_pct = portfolio.pnl_pct
            
            if daily_pnl_pct < -self.daily_loss_cap:
                if not self.daily_loss_cap_triggered:
                    logger.error(f"🚨 Daily loss cap triggered: {daily_pnl_pct:.2%} < -{self.daily_loss_cap:.2%}")
                    self.daily_loss_cap_triggered = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking daily loss cap: {e}")
            return False
    
    def _disable_ml_overlay(self):
        """Disable ML overlay due to loss cap or other reasons"""
        if self.model_manager:
            self.model_manager.stop()
            self.model_manager = None
            self.ml_overlay_enabled = False
            logger.warning("🤖 ML overlay disabled")
    
    def _generate_signals_with_ml(self) -> List[Dict]:
        """Genereer signalen met ML overlay"""
        signals = []
        
        for symbol in self.current_universe:
            try:
                # ⚠️ PERFORMANCE: Use cached data
                df = self._get_cached_data(symbol, days=30)
                if df is None or len(df) < 50:
                    logger.warning(f"Onvoldoende data voor {symbol}")
                    continue
                
                # Genereer non-ML signalen
                non_ml_signals = []
                for strategy_name, strategy in self.strategies.items():
                    try:
                        signal = strategy.generate_signal(df, symbol)
                        if signal:
                            sig_dict = signal.to_dict()
                            # Backward-compatible veldnaam en meta
                            sig_dict['strategy_name'] = signal.strategy_name
                            sig_dict.setdefault('meta', {})
                            sig_dict['meta']['strategy'] = signal.strategy_name
                            sig_dict['ml_enhanced'] = False
                            non_ml_signals.append(sig_dict)
                    except Exception as e:
                        logger.error(f"Fout bij {strategy_name} voor {symbol}: {e}")
                
                # Genereer ML signaal indien beschikbaar
                ml_result = None
                if self.model_manager and self.model_manager.model_available:
                    try:
                        ml_result = self._generate_ml_result(symbol, df)
                    except Exception as e:
                        logger.error(f"Fout bij ML signaal generatie voor {symbol}: {e}")
                        if self.ml_overlay_manager:
                            self.ml_overlay_manager.record_failure()
                
                # Process ML decision through overlay manager
                if self.ml_overlay_manager and ml_result:
                    try:
                        # Get model version
                        model_version = self.model_manager.get_model_version(self.model_manager.current_bundle)
                        
                        # Get feature vector for hash calculation
                        features_df = self.feature_pipeline.build_features(df)
                        feature_vector = self.feature_pipeline.get_feature_vector(features_df, -1)
                        
                        # Process decision
                        ml_decision = self.ml_overlay_manager.process_ml_decision(
                            symbol=symbol,
                            ml_result=ml_result,
                            non_ml_signals=non_ml_signals,
                            model_version=model_version,
                            features=feature_vector
                        )
                        
                        # Create signal based on decision
                        if ml_decision.executed and (ml_decision.ml_buy or ml_decision.ml_sell):
                            ml_signal = {
                                'symbol': symbol,
                                'strategy_name': 'ml_overlay',
                                'side': 'buy' if ml_decision.ml_buy else 'sell',
                                'confidence': ml_decision.ml_confidence,
                                'ml_enhanced': True,
                                'ml_proba': ml_decision.ml_proba,
                                'model_version': ml_decision.model_version,
                                'features_hash': ml_decision.features_hash
                            }
                            non_ml_signals.append(ml_signal)
                        
                    except Exception as e:
                        logger.error(f"Fout bij ML overlay processing voor {symbol}: {e}")
                
                # Kies beste signaal (alleen signalen met entry/stop zijn valide)
                valid_signals = [s for s in non_ml_signals if 'entry' in s and 'stop' in s]
                if valid_signals:
                    best_signal = max(valid_signals, key=lambda x: x.get('confidence', 0))
                    signal_type = "ML" if best_signal.get('ml_enhanced', False) else "Non-ML"
                    logger.info(f"🎯 {signal_type} signaal geselecteerd voor {symbol}: {best_signal['strategy_name']} "
                               f"(confidence: {best_signal['confidence']:.3f})")
                    signals.append(best_signal)
                
            except Exception as e:
                logger.error(f"Fout bij signaal generatie voor {symbol}: {e}")
        
        return signals
    
    def _generate_ml_result(self, symbol: str, df) -> Optional[Dict]:
        """Genereer ML signaal voor symbol"""
        try:
            # Bouw features
            features_df = self.feature_pipeline.build_features(df)
            
            # Haal laatste feature vector op
            feature_vector = self.feature_pipeline.get_feature_vector(features_df, -1)
            
            # Converteer naar dictionary voor model manager
            feature_dict = dict(zip(self.feature_pipeline.feature_names, feature_vector))
            
            # Krijg ML prediction
            ml_result = self.model_manager.get_prediction(symbol, feature_dict)
            
            if not ml_result:
                return None
            
            # ⚠️ CRITICAL FIX: ml_result is now a Dict (was float before)
            # Converteer ML result naar signaal format
            side = 'buy' if ml_result.get('buy', False) else 'sell' if ml_result.get('sell', False) else 'hold'
            
            signal = {
                'side': side,
                'confidence': ml_result.get('confidence', 0.5),
                'ml_proba': ml_result.get('proba', 0.5),
                'ml_buy_prob': ml_result.get('buy_prob', 0.5),
                'ml_sell_prob': ml_result.get('sell_prob', 0.5),
                'model_version': ml_result.get('model_version', 'unknown')
            }
            
            # Alleen returnen als er een duidelijke signaal is
            if signal['side'] != 'hold':
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating ML signal for {symbol}: {e}")
            return None
    
    def _update_universe(self):
        """Update universe indien nodig"""
        try:
            # Check of universe herbalans nodig is
            if self.universe_selector.should_rebalance() or not self.current_universe:
                logger.info("🔄 Universe herbalans...")
                
                # Download data voor alle symbols indien nodig
                for symbol in self.trading_config.symbols:
                    if not self._has_recent_data(symbol):
                        self._download_data(symbol)
                
                # Selecteer nieuwe universe
                self.current_universe = self.universe_selector.get_universe(
                    self.trading_config.symbols
                )
                
                logger.info(f"✅ Universe bijgewerkt: {self.current_universe}")
            else:
                logger.debug(f"Universe ongewijzigd: {self.current_universe}")
                
        except Exception as e:
            logger.error(f"Fout bij universe update: {e}")
    
    def _validate_signal(self, signal: Dict) -> bool:
        """
        ⚠️ SAFETY: Validate signal structure
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Required fields
            required_fields = ['symbol', 'side', 'entry', 'stop', 'confidence']
            for field in required_fields:
                if field not in signal:
                    logger.error(f"Signal missing required field: {field}")
                    return False
            
            # Validate side
            if signal['side'] not in ['buy', 'sell']:
                logger.error(f"Invalid side: {signal['side']}")
                return False
            
            # Validate numeric fields
            try:
                entry = float(signal['entry'])
                stop = float(signal['stop'])
                confidence = float(signal['confidence'])
                
                if entry <= 0 or stop <= 0:
                    logger.error(f"Invalid prices: entry={entry}, stop={stop}")
                    return False
                
                if confidence < 0 or confidence > 1:
                    logger.error(f"Invalid confidence: {confidence}")
                    return False
                
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid numeric values in signal: {e}")
                return False
            
            # Validate stop loss makes sense
            if signal['side'] == 'buy' and stop >= entry:
                logger.error(f"Invalid buy stop: stop ({stop}) >= entry ({entry})")
                return False
            
            if signal['side'] == 'sell' and stop <= entry:
                logger.error(f"Invalid sell stop: stop ({stop}) <= entry ({entry})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def _check_slippage(self, signal: Dict) -> bool:
        """
        ⚠️ SAFETY: Check slippage against current market price
        
        Args:
            signal: Signal dictionary with entry price
            
        Returns:
            True if slippage acceptable, False otherwise
        """
        try:
            symbol = signal['symbol']
            signal_entry = signal['entry']
            
            # Get current market price (use cache for performance)
            df = self._get_cached_data(symbol, days=1)
            if df is None or len(df) == 0:
                logger.warning(f"No data available for {symbol} - skipping slippage check")
                return True  # Allow if we can't check (data issue)
            
            current_price = df['close'].iloc[-1]
            
            # Calculate slippage
            slippage_pct = abs(current_price - signal_entry) / signal_entry
            
            MAX_SLIPPAGE = 0.02  # 2% maximum slippage
            
            if slippage_pct > MAX_SLIPPAGE:
                logger.warning(
                    f"⚠️ Slippage too high for {symbol}: {slippage_pct:.2%} > {MAX_SLIPPAGE:.2%} "
                    f"(signal_entry={signal_entry:.4f}, current={current_price:.4f})"
                )
                return False
            
            # Update entry price to current price if slippage acceptable
            if slippage_pct > 0.001:  # Only log if > 0.1%
                logger.info(f"Adjusting entry price for {symbol}: {signal_entry:.4f} → {current_price:.4f} (slippage: {slippage_pct:.2%})")
                signal['entry'] = current_price
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking slippage for {symbol}: {e}")
            return False  # Fail-safe: reject if we can't verify
    
    def _execute_signals(self, signals: List[Dict]):
        """Voer signalen uit via executor met validation en slippage protection"""
        for signal in signals:
            try:
                # ⚠️ SAFETY: Validate signal structure
                if not self._validate_signal(signal):
                    logger.warning(f"Invalid signal rejected: {signal.get('symbol', 'unknown')}")
                    continue
                
                # ⚠️ SAFETY: Check slippage
                if not self._check_slippage(signal):
                    logger.warning(f"Signal rejected due to high slippage: {signal['symbol']}")
                    continue
                
                symbol = signal['symbol']
                bar_ts = datetime.now()  # In echte implementatie: timestamp van bar
                
                result = self.executor.maybe_execute(symbol, signal, bar_ts)
                if result:
                    signal_type = "ML" if signal.get('ml_enhanced', False) else "Non-ML"
                    logger.info(f"✅ {signal_type} trade uitgevoerd: {symbol} {signal['side']} @ {signal['entry']:.4f}")
                else:
                    logger.debug(f"Trade niet uitgevoerd: {symbol}")
                    
            except Exception as e:
                logger.error(f"Fout bij trade uitvoering: {e}")
    
    def _update_existing_positions(self):
        """Update bestaande posities en check exit condities"""
        try:
            positions = self.risk_manager.positions.copy()
            
            for symbol, position in positions.items():
                try:
                    # ⚠️ PERFORMANCE: Use cached data
                    df = self._get_cached_data(symbol, days=1)
                    if df is None or len(df) == 0:
                        continue
                    
                    current_price = df['close'].iloc[-1]
                    
                    # Update positie
                    result = self.risk_manager.update_position(symbol, current_price)
                    if result:
                        logger.info(f"Positie gesloten: {symbol}")
                        
                except Exception as e:
                    logger.error(f"Fout bij positie update voor {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Fout bij posities update: {e}")
    
    def _download_data(self, symbol: str) -> bool:
        """Download data voor symbol"""
        try:
            logger.info(f"📥 Data downloaden voor {symbol}...")
            
            # Download data
            df = self.data_manager.download_yahoo_data(symbol, "2y", "1h")
            
            # Valideer data
            if not self.data_manager.validate_data(df):
                logger.error(f"Data validatie gefaald voor {symbol}")
                return False
            
            # Voeg technische indicatoren toe
            df = self.indicators.add_all_indicators(df)
            
            # Sla op
            filename = f"{symbol}_1h_2y"
            self.data_manager.save_data(df, filename)
            
            logger.info(f"✅ Data voorbereid voor {symbol}: {len(df)} rijen")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fout bij data download voor {symbol}: {e}")
            return False
    
    def _has_recent_data(self, symbol: str) -> bool:
        """Check of we recente data hebben voor symbol"""
        try:
            df = self.data_manager.get_latest_data(symbol, days=1)
            if df is None or len(df) == 0:
                return False
            
            # Check of data recenter is dan 24 uur (timezone-safe)
            last_update = df.index[-1]
            if isinstance(last_update, pd.Timestamp):
                tz = getattr(last_update, 'tz', None)
                now_ts = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
                time_diff_sec = (now_ts - last_update).total_seconds()
                return time_diff_sec < 24 * 3600
            
            return True
            
        except Exception as e:
            logger.error(f"Fout bij data check voor {symbol}: {e}")
            return False
    
    def _log_status(self):
        """Log huidige status inclusief ML info"""
        try:
            # Portfolio status
            portfolio = self.risk_manager.get_portfolio_status()
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Regime status
            regime_status = self.regime_filter.get_regime_status()
            
            # Universe status
            universe_status = self.universe_selector.get_universe_status()
            
            # Executor status
            executor_status = self.executor.get_executor_status()
            
            # ML status
            ml_status = self.model_manager.get_model_status() if self.model_manager else None
            
            logger.info("📊 Status Update:")
            logger.info(f"  💰 Portfolio: €{portfolio.total_value:.2f} (P&L: {portfolio.total_pnl:.2f})")
            logger.info(f"  📈 Risk: DD={risk_metrics['current_drawdown']:.2%}, Posities={risk_metrics['open_positions']}")
            logger.info(f"  🎯 Regime: {'Tradebaar' if regime_status['is_tradable'] else 'Niet tradebaar'}")
            logger.info(f"  🌟 Universe: {len(self.current_universe)} symbols")
            logger.info(f"  📊 Daily Trades: {executor_status['daily_trades']}, P&L: {executor_status['daily_pnl']:.2f}")
            
            if ml_status:
                ml_available = ml_status['model_available']
                ml_version = ml_status['current_version']
                logger.info(f"  🤖 ML: {'Actief' if ml_available else 'Inactief'} (v{ml_version})")
            else:
                logger.info(f"  🤖 ML: Uitgeschakeld")
            
        except Exception as e:
            logger.error(f"Fout bij status logging: {e}")
    
    def _map_universe_config(self, cfg_obj):
        """Map core.config.UniverseConfig naar filters.universe_selector.UniverseConfig"""
        try:
            # Afleiden van 24h volume drempel uit 30d volume
            min_30d_vol = getattr(cfg_obj, 'min_30d_volume_usd', 5000000)
            momentum_lookbacks = getattr(cfg_obj, 'momentum_lookbacks', [7])
            return UniverseConfig(
                rebalance_frequency=getattr(cfg_obj, 'rebalance_frequency', 'weekly'),
                max_assets=getattr(cfg_obj, 'max_assets', 3),
                min_volume_24h=float(min_30d_vol) / 30.0,
                momentum_days=int(momentum_lookbacks[0]) if momentum_lookbacks else 7,
            )
        except Exception as e:
            logger.warning(f"Kon universe config niet mappen, val terug op defaults: {e}")
            return UniverseConfig()
    def start_live_trading(self):
        """Start live trading met scheduler en ML monitoring"""
        if not self.live_config.enabled:
            logger.warning("Live trading niet ingeschakeld in configuratie")
            return
        
        logger.info("🚀 Live trading gestart!")
        self.is_running = True
        
        # Start ML model manager
        if self.model_manager:
            self.model_manager.start()
            logger.info("🤖 ML ModelManager gestart")
        
        # Setup data feed scheduler
        self.data_scheduler.setup_data_feeds()
        self.data_scheduler.start()
        
        # Setup trading scheduler
        self.scheduler.add_task(
            'trading_cycle',
            self.run_trading_cycle,
            f"{self.live_config.cooldown_bars}h"
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Eerste cycle direct uitvoeren
        self.run_trading_cycle()
        
        # Loop voor geplande cycles
        while self.is_running:
            try:
                time.sleep(60)  # Check elke minuut
            except KeyboardInterrupt:
                logger.info("⏹️ Live trading gestopt door gebruiker")
                self.stop_live_trading()
                break
            except Exception as e:
                logger.error(f"❌ Fout in live trading loop: {e}")
                time.sleep(300)  # Wacht 5 minuten bij fout
    
    def stop_live_trading(self):
        """Stop live trading en ML monitoring"""
        logger.info("⏹️ Live trading gestopt")
        self.is_running = False
        
        # Stop ML model manager
        if self.model_manager:
            self.model_manager.stop()
        
        # Stop schedulers
        self.scheduler.stop()
        self.data_scheduler.stop()
        
        # Clear old schedule (backward compatibility)
        schedule.clear()
    
    def get_status(self) -> Dict:
        """Krijg huidige bot status inclusief ML info"""
        try:
            portfolio = self.risk_manager.get_portfolio_status()
            risk_metrics = self.risk_manager.get_risk_metrics()
            regime_status = self.regime_filter.get_regime_status()
            universe_status = self.universe_selector.get_universe_status()
            executor_status = self.executor.get_executor_status()
            
            # ML status
            ml_status = self.model_manager.get_model_status() if self.model_manager else None
            
            status = {
                'is_running': self.is_running,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'portfolio': {
                    'total_value': portfolio.total_value,
                    'cash': portfolio.cash,
                    'total_pnl': portfolio.total_pnl,
                    'pnl_pct': portfolio.pnl_pct,
                    'open_positions': len(portfolio.positions)
                },
                'risk_metrics': risk_metrics,
                'regime': regime_status,
                'universe': {
                    'current_symbols': self.current_universe,
                    'status': universe_status
                },
                'executor': executor_status,
                'strategies': list(self.strategies.keys()),
                'scheduler': self.scheduler.get_status(),
                'data_scheduler': self.data_scheduler.get_status(),
                'ml_overlay': {
                    'enabled': self.use_ml_overlay,
                    'available': ml_status['model_available'] if ml_status else False,
                    'version': ml_status['current_version'] if ml_status else None,
                    'failure_count': ml_status['failure_count'] if ml_status else 0,
                    'daily_loss_cap_triggered': self.daily_loss_cap_triggered
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Fout bij status ophalen: {e}")
            return {'error': str(e)}


def main():
    """Hoofdfunctie"""
    try:
        # Maak trading bot instance
        bot = TradingBotV4WithML()
        
        # Non-interactive mode (voor systemd services)
        if len(sys.argv) > 1 and '--non-interactive' in sys.argv:
            logger.info("🚀 Starting Trading Bot (ML) in non-interactive mode (systemd service)")
            bot.start_live_trading()
            return
        
        # Menu interface
        print("🤖 Trading Bot v4 (ML-enabled) Menu:")
        print("1. Data downloaden")
        print("2. Universe selectie testen")
        print("3. Regime filter testen")
        print("4. Strategie signalen testen")
        print("5. ML model status")
        print("6. Live trading starten")
        print("7. Status bekijken")
        print("8. Exit")
        
        while True:
            choice = input("\nKies een optie (1-8): ").strip()
            
            if choice == '1':
                # Download data voor alle symbols
                for symbol in bot.trading_config.symbols:
                    bot._download_data(symbol)
                    
            elif choice == '2':
                # Test universe selectie
                print("\n🌟 Universe Selectie Test")
                print("=" * 30)
                universe = bot.universe_selector.select(bot.trading_config.symbols)
                print(f"Geselecteerde universe: {universe}")
                
            elif choice == '3':
                # Test regime filter
                print("\n🎯 Regime Filter Test")
                print("=" * 25)
                is_tradable = bot.regime_filter.is_tradable()
                status = bot.regime_filter.get_regime_status()
                print(f"Markt tradebaar: {is_tradable}")
                print(f"Status: {status}")
                
            elif choice == '4':
                # Test strategie signalen
                print("\n📊 Strategie Signal Test")
                print("=" * 28)
                signals = bot._generate_signals_with_ml()
                if signals:
                    for signal in signals:
                        signal_type = "ML" if signal.get('ml_enhanced', False) else "Non-ML"
                        print(f"{signal['symbol']}: {signal_type} {signal['strategy_name']} "
                              f"{signal['side']} (confidence: {signal['confidence']:.3f})")
                else:
                    print("Geen signalen gegenereerd")
                    
            elif choice == '5':
                # ML model status
                print("\n🤖 ML Model Status")
                print("=" * 20)
                if bot.model_manager:
                    status = bot.model_manager.get_model_status()
                    print(f"Model beschikbaar: {status['model_available']}")
                    print(f"Versie: {status['current_version']}")
                    print(f"Failure count: {status['failure_count']}")
                    if status['model_info']:
                        print(f"Feature count: {status['model_info']['feature_count']}")
                else:
                    print("ML overlay uitgeschakeld")
                    
            elif choice == '6':
                # Start live trading
                bot.start_live_trading()
                
            elif choice == '7':
                # Toon status
                status = bot.get_status()
                print(f"\nStatus: {status}")
                
            elif choice == '8':
                print("👋 Tot ziens!")
                break
                
            else:
                print("❌ Ongeldige keuze")
                
    except KeyboardInterrupt:
        print("\n👋 Bot gestopt door gebruiker")
    except Exception as e:
        logger.error(f"❌ Fout in main: {e}")


if __name__ == "__main__":
    main()
