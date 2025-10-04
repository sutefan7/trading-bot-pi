"""
Configuratie management voor Trading Bot v4
"""
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class UniverseConfig:
    """Universe selectie configuratie"""
    rebalance_frequency: str = "weekly"
    max_assets: int = 5
    min_30d_volume_usd: float = 5000000
    max_pair_correlation: float = 0.7
    momentum_lookbacks: list = field(default_factory=lambda: [30, 90])


@dataclass
class TradingConfig:
    """Trading configuratie parameters"""
    symbols: list = field(default_factory=lambda: ["BTC-USD"])
    timeframes: list = field(default_factory=lambda: ["1h"])
    primary_timeframe: str = "1h"
    universe: UniverseConfig = field(default_factory=UniverseConfig)


@dataclass
class APIConfig:
    """API configuratie"""
    kraken_sandbox: bool = True
    kraken_rate_limit: float = 1.0


@dataclass
class MLConfig:
    """Machine Learning configuratie"""
    models: list = field(default_factory=lambda: ["xgboost"])
    confidence_threshold: float = 0.65
    retrain_frequency_days: int = 30
    walk_forward_windows: int = 4
    feature_selection: bool = True


@dataclass
class RiskConfig:
    """Risicomanagement configuratie"""
    per_trade_risk_pct: float = 0.0075
    daily_loss_cap_pct: float = 0.04
    max_open_positions: int = 3
    trailing_stop_pct: float = 0.03
    # Backward compatibility
    max_position_size: float = 0.3
    min_position_size: float = 0.05
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    max_drawdown: float = 0.20
    trailing_stop: bool = True


@dataclass
class PortfolioConfig:
    """Portfolio configuratie"""
    initial_balance: float = 10000.0
    base_currency: str = "USD"
    rebalance_frequency: str = "weekly"
    max_correlation: float = 0.7


@dataclass
class BacktestConfig:
    """Backtesting configuratie"""
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    transaction_fee: float = 0.001
    slippage: float = 0.0005


@dataclass
class LiveConfig:
    """Live trading configuratie"""
    enabled: bool = False
    paper_trading: bool = True
    max_open_positions: int = 3
    cooldown_bars: int = 12


class ConfigManager:
    """Centrale configuratie manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config_data: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Laad configuratie uit YAML bestand"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self.config_data = yaml.safe_load(file)
                logger.info(f"Configuratie geladen van {self.config_path}")
            else:
                logger.warning(f"Configuratie bestand {self.config_path} niet gevonden, gebruik standaardwaarden")
                self.config_data = {}
        except Exception as e:
            logger.error(f"Fout bij laden configuratie: {e}")
            self.config_data = {}
    
    def get_trading_config(self) -> TradingConfig:
        """Haal trading configuratie op"""
        trading_data = self.config_data.get('trading', {})
        universe_data = trading_data.get('universe', {})
        
        universe_config = UniverseConfig(
            rebalance_frequency=universe_data.get('rebalance_frequency', 'weekly'),
            max_assets=universe_data.get('max_assets', 5),
            min_30d_volume_usd=universe_data.get('min_30d_volume_usd', 5000000),
            max_pair_correlation=universe_data.get('max_pair_correlation', 0.7),
            momentum_lookbacks=universe_data.get('momentum_lookbacks', [30, 90])
        )
        
        return TradingConfig(
            symbols=trading_data.get('symbols', ["BTC-USD"]),
            timeframes=trading_data.get('timeframes', ["1h"]),
            primary_timeframe=trading_data.get('primary_timeframe', "1h"),
            universe=universe_config
        )
    
    def get_api_config(self) -> APIConfig:
        """Haal API configuratie op"""
        api_data = self.config_data.get('api', {}).get('kraken', {})
        return APIConfig(
            kraken_sandbox=api_data.get('dry_run', True),  # Use dry_run instead of sandbox
            kraken_rate_limit=api_data.get('rate_limit', 1.0)
        )
    
    def get_ml_config(self) -> MLConfig:
        """Haal ML configuratie op"""
        ml_data = self.config_data.get('ml', {})
        return MLConfig(
            models=ml_data.get('model_types', ["xgboost"]),  # Use model_types instead of models
            confidence_threshold=ml_data.get('confidence_threshold', 0.65),
            retrain_frequency_days=ml_data.get('retrain_frequency_days', 30),
            walk_forward_windows=ml_data.get('walk_forward_windows', 4),
            feature_selection=ml_data.get('feature_selection', True)
        )
    
    def get_risk_config(self) -> RiskConfig:
        """Haal risicomanagement configuratie op"""
        risk_data = self.config_data.get('risk', {})
        return RiskConfig(
            per_trade_risk_pct=risk_data.get('per_trade_risk_pct', 0.0075),
            daily_loss_cap_pct=risk_data.get('daily_loss_cap_pct', 0.04),
            max_open_positions=risk_data.get('max_open_positions', 3),
            trailing_stop_pct=risk_data.get('trailing_stop_pct', 0.03),
            # Backward compatibility
            max_position_size=risk_data.get('max_position_size', 0.3),
            min_position_size=risk_data.get('min_position_size', 0.05),
            stop_loss_pct=risk_data.get('stop_loss_pct', 0.05),
            take_profit_pct=risk_data.get('take_profit_pct', 0.15),
            max_drawdown=risk_data.get('max_drawdown', 0.20),
            trailing_stop=risk_data.get('trailing_stop', True)
        )
    
    def get_portfolio_config(self) -> PortfolioConfig:
        """Haal portfolio configuratie op"""
        portfolio_data = self.config_data.get('portfolio', {})
        return PortfolioConfig(
            initial_balance=portfolio_data.get('initial_balance', 10000.0),
            base_currency=portfolio_data.get('base_currency', "USD"),
            rebalance_frequency=portfolio_data.get('rebalance_frequency', "weekly"),
            max_correlation=portfolio_data.get('max_correlation', 0.7)
        )
    
    def get_backtest_config(self) -> BacktestConfig:
        """Haal backtesting configuratie op"""
        backtest_data = self.config_data.get('backtest', {})
        return BacktestConfig(
            start_date=backtest_data.get('start_date', "2023-01-01"),
            end_date=backtest_data.get('end_date', "2024-12-31"),
            transaction_fee=backtest_data.get('transaction_fee', 0.001),
            slippage=backtest_data.get('slippage', 0.0005)
        )
    
    def get_live_config(self) -> LiveConfig:
        """Haal live trading configuratie op"""
        live_data = self.config_data.get('live', {})
        return LiveConfig(
            enabled=live_data.get('enabled', False),
            paper_trading=live_data.get('paper_trading', True),
            max_open_positions=live_data.get('max_open_positions', 3),
            cooldown_bars=live_data.get('cooldown_bars', 12)
        )
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """Haal technische indicatoren configuratie op"""
        return self.config_data.get('indicators', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Haal logging configuratie op"""
        return self.config_data.get('logging', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Haal performance tracking configuratie op"""
        return self.config_data.get('performance', {})
    
    def get_regime_config(self) -> Dict[str, Any]:
        """Haal regime filter configuratie op"""
        return self.config_data.get('regime', {})
    
    def get_strategies_config(self) -> Dict[str, Any]:
        """Haal strategies configuratie op"""
        return self.config_data.get('strategies', {})
    
    def get_universe_config(self) -> Dict[str, Any]:
        """Haal universe configuratie op"""
        return self.config_data.get('trading', {}).get('universe', {})
    
    def validate_config(self) -> bool:
        """Valideer configuratie - HARD FAIL bij conflicts"""
        try:
            # CRITICAL: Check for duplicate portfolio sections
            portfolio_sections = []
            for key, value in self.config_data.items():
                if key == 'portfolio':
                    portfolio_sections.append(value)
            
            if len(portfolio_sections) > 1:
                logger.error("🚨 CRITICAL: Multiple portfolio sections found in config!")
                logger.error("This causes unpredictable P&L and risk limit behavior.")
                logger.error("Please consolidate to ONE portfolio section with consistent base_currency.")
                return False
            
            # CRITICAL: Check for currency conflicts
            portfolio_data = self.config_data.get('portfolio', {})
            base_currency = portfolio_data.get('base_currency', 'USD')
            
            # Check if base_currency is consistent across config
            if 'fx_source' in portfolio_data:
                fx_source = portfolio_data.get('fx_source', 'default')
                if fx_source == 'kraken' and base_currency not in ['EUR', 'USD']:
                    logger.warning(f"FX source 'kraken' with base_currency '{base_currency}' may not be optimal")
            
            # Basis validaties
            trading_config = self.get_trading_config()
            if not trading_config.symbols:
                logger.error("Geen trading symbols geconfigureerd")
                return False
            
            risk_config = self.get_risk_config()
            if risk_config.max_position_size > 1.0:
                logger.error("Max position size kan niet groter zijn dan 100%")
                return False
            
            if risk_config.stop_loss_pct <= 0:
                logger.error("Stop loss moet groter zijn dan 0")
                return False
            
            # CRITICAL: Check for sandbox/dry_run conflicts
            api_data = self.config_data.get('api', {}).get('kraken', {})
            if 'sandbox' in api_data and 'dry_run' in api_data:
                logger.error("🚨 CRITICAL: Both 'sandbox' and 'dry_run' found in API config!")
                logger.error("Kraken doesn't support sandbox. Use 'dry_run' only.")
                return False
            
            # CRITICAL: Check for legacy risk fields
            risk_sections = [key for key in self.config_data.keys() if 'risk' in key.lower()]
            if len(risk_sections) > 1:
                logger.error(f"🚨 CRITICAL: Multiple risk sections found: {risk_sections}")
                logger.error("Use only one risk configuration to avoid conflicts.")
                return False
            
            # CRITICAL: Check for conflicting risk settings
            if 'risk' in self.config_data and 'risk_management' in self.config_data:
                logger.error("🚨 CRITICAL: Both 'risk' and 'risk_management' sections found!")
                logger.error("Use only one risk configuration.")
                return False
            
            # CRITICAL: Check for deprecated legacy risk fields
            deprecated_risk_fields = [
                'max_position_size', 'min_position_size', 'stop_loss_pct', 
                'take_profit_pct', 'max_drawdown', 'trailing_stop'
            ]
            
            risk_section = self.config_data.get('risk', {})
            found_deprecated = []
            for field in deprecated_risk_fields:
                if field in risk_section:
                    found_deprecated.append(field)
            
            if found_deprecated:
                logger.error("🚨 CRITICAL: Deprecated legacy risk fields found!")
                logger.error(f"Deprecated fields: {found_deprecated}")
                logger.error("These fields are deprecated and will cause validation errors.")
                logger.error("Use the new risk_guardrails section instead.")
                return False
            
            # CRITICAL: Check for dry_run and live flags
            if api_data.get('dry_run') and api_data.get('live'):
                logger.error("🚨 CRITICAL: Both 'dry_run' and 'live' flags are active!")
                logger.error("Use only one mode: dry_run for testing, live for production.")
                return False
            
            # CRITICAL: Check ML overlay consistency
            ml_enabled = self.config_data.get('ml', {}).get('enabled', False)
            ml_overlay_enabled = self.config_data.get('ml_overlay', {}).get('enabled', False)
            ml_overlay_mode = self.config_data.get('ml_overlay', {}).get('mode', 'off')
            
            if ml_enabled and ml_overlay_mode == 'live':
                logger.warning("⚠️ ML overlay is in 'live' mode - ensure this is intended for production")
            
            if ml_enabled and not ml_overlay_enabled:
                logger.warning("⚠️ ML is enabled but ml_overlay is disabled - ML features may not work")
            
            if ml_enabled and ml_overlay_enabled and ml_overlay_mode == 'live':
                logger.warning("⚠️  ML overlay is in 'live' mode - ensure this is intended for production!")
            
            logger.info("✅ Configuratie validatie succesvol")
            logger.info(f"   Base currency: {base_currency}")
            logger.info(f"   ML enabled: {ml_enabled}")
            logger.info(f"   ML overlay mode: {ml_overlay_mode}")
            return True
            
        except Exception as e:
            logger.error(f"🚨 Configuratie validatie gefaald: {e}")
            return False
