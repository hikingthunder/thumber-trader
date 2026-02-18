
from decimal import Decimal
from typing import Optional, Set
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings and configuration.
    Loads from environment variables and .env file.
    """
    # Product and Grid settings
    coinbase_api_key: SecretStr = None
    coinbase_api_secret: SecretStr = None
    product_id: str = "BTC-USD"
    product_ids: str = "BTC-USD"
    grid_lines: int = 8
    grid_band_pct: Decimal = Decimal("0.15")
    min_notional_usd: Decimal = Decimal("6")
    min_grid_profit_pct: Decimal = Decimal("0.015")
    maker_fee_pct: Decimal = Decimal("0.004")
    target_net_profit_pct: Decimal = Decimal("0.002")
    poll_seconds: int = 60
    log_level: str = "INFO"
    auto_start: bool = True

    # High Availability
    ha_failover_enabled: bool = False
    ha_instance_id: str = "thumber-trader-1"
    ha_takeover_poll_cycles: int = 3
    ha_lock_lease_seconds: int = 30
    ha_standby_sleep_seconds: int = 5

    # Multi-venue market data
    consensus_pricing_enabled: bool = True
    consensus_exchanges: str = "coinbase,binance,kraken,bybit"
    consensus_max_deviation_pct: Decimal = Decimal("0.02")

    # Alpha Fusion
    alpha_fusion_enabled: bool = True
    alpha_rsi_period: int = 14
    alpha_macd_fast: int = 12
    alpha_macd_slow: int = 26
    alpha_macd_signal: int = 9
    alpha_weight_rsi: Decimal = Decimal("0.3")
    alpha_weight_macd: Decimal = Decimal("0.3")
    alpha_weight_imbalance: Decimal = Decimal("0.4")

    # VPIN (Order Flow Toxicity)
    vpin_enabled: bool = True
    vpin_bucket_volume_base: Decimal = Decimal("0.25")
    vpin_rolling_buckets: int = 50
    vpin_history_size: int = 300
    vpin_threshold_percentile: Decimal = Decimal("0.95")
    vpin_response_mode: str = "widen"
    vpin_widen_band_multiplier: Decimal = Decimal("1.5")

    # Live Fee Adaptation
    dynamic_fee_tracking_enabled: bool = True
    fee_refresh_seconds: int = 3600

    # Exchange Bracket Orders
    use_exchange_bracket_orders: bool = False
    bracket_take_profit_pct: Decimal = Decimal("0.01")
    bracket_stop_loss_pct: Decimal = Decimal("0.01")

    # Grid Shape
    grid_spacing_mode: str = "arithmetic"

    # ATR (Volatility Adaptive Grid)
    atr_enabled: bool = False
    atr_period: int = 14
    atr_band_multiplier: Decimal = Decimal("4")
    atr_min_band_pct: Decimal = Decimal("0.03")
    atr_max_band_pct: Decimal = Decimal("0.35")

    # Capital and Risk Controls
    base_order_notional_usd: Decimal = Decimal("10")
    kelly_allocation_enabled: bool = True
    kelly_refresh_seconds: int = 900
    kelly_lookback_fills: int = 300
    kelly_min_closed_trades: int = 20
    kelly_min_allocation_frac: Decimal = Decimal("0.25")
    kelly_max_allocation_frac: Decimal = Decimal("2.50")
    black_litterman_tau: Decimal = Decimal("0.05")
    black_litterman_risk_aversion: Decimal = Decimal("2.5")
    black_litterman_confidence_floor: Decimal = Decimal("0.05")
    black_litterman_view_return_abs: Decimal = Decimal("0.01")
    quote_reserve_pct: Decimal = Decimal("0.25")
    max_btc_inventory_pct: Decimal = Decimal("0.65")
    hard_stop_loss_pct: Decimal = Decimal("0.08")
    liquidity_depth_check_enabled: bool = True
    liquidity_depth_levels: int = 50
    liquidity_max_book_share_pct: Decimal = Decimal("0.20")

    # Trend Signal Controls
    trend_candle_granularity: str = "ONE_HOUR"
    trend_candle_limit: int = 72
    trend_ema_fast: int = 9
    trend_ema_slow: int = 21
    trend_strength_threshold: Decimal = Decimal("0.003")
    adx_period: int = 14
    adx_ranging_threshold: Decimal = Decimal("20")
    adx_trending_threshold: Decimal = Decimal("25")
    adx_range_band_multiplier: Decimal = Decimal("0.8")
    adx_trend_band_multiplier: Decimal = Decimal("1.25")
    adx_trend_order_size_multiplier: Decimal = Decimal("0.7")

    # HMM Regime Detection
    hmm_regime_detection_enabled: bool = False
    hmm_states: int = 3
    hmm_lookback: int = 120
    hmm_iterations: int = 12
    hmm_min_variance: Decimal = Decimal("0.00000001")
    model_registry_enabled: bool = True
    model_drift_monitor_enabled: bool = True
    model_drift_retrain_enabled: bool = True
    model_registry_training_window: int = 2000
    model_registry_eval_window: int = 400
    model_drift_ks_threshold: Decimal = Decimal("0.20")
    model_drift_poll_seconds: int = 300

    # Dynamic Inventory Cap
    dynamic_inventory_cap_enabled: bool = False
    inventory_cap_min_pct: Decimal = Decimal("0.30")
    inventory_cap_max_pct: Decimal = Decimal("0.80")

    # Execution Mode
    paper_trading_mode: bool = False
    paper_start_usd: Decimal = Decimal("1000")
    paper_start_btc: Decimal = Decimal("0")
    paper_start_base: Decimal = Decimal("0")
    paper_fill_exceed_pct: Decimal = Decimal("0.0001")
    paper_fill_delay_seconds: int = 5
    paper_slippage_pct: Decimal = Decimal("0.0001")
    execution_rl_enabled: bool = False
    execution_rl_algo: str = "dqn"
    execution_rl_learning_rate: float = 0.03
    execution_rl_discount: float = 0.95
    execution_rl_epsilon: float = 0.15
    execution_rl_min_epsilon: float = 0.02
    execution_rl_epsilon_decay: float = 0.999
    execution_rl_chase_step_bps: Decimal = Decimal("1.5")
    execution_rl_max_chase_bps: Decimal = Decimal("12")
    execution_rl_update_interval_seconds: int = 2

    # Dashboard
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    dashboard_auth_token: str = ""
    dashboard_max_request_bytes: int = 1048576
    prometheus_enabled: bool = True
    prometheus_path: str = "/metrics"

    # Trailing Grid
    trailing_grid_enabled: bool = True
    trailing_trigger_levels: int = 2

    # Persistence
    state_db_path: str = "grid_state.db"
    legacy_orders_json_enabled: bool = False
    tax_lot_method: str = "FIFO"

    # Safe Start
    safe_start_enabled: bool = True
    base_buy_mode: str = "off"
    base_buy_execution_algo: str = "market"
    base_buy_execution_slices: int = 6
    base_buy_execution_window_seconds: int = 120
    base_buy_vwap_lookback_candles: int = 24
    base_buy_vwap_granularity: str = "FIVE_MINUTE"
    shared_usd_reserve_enabled: bool = True

    # Cross-Asset Risk Controls
    cross_asset_correlation_enabled: bool = True
    cross_asset_correlation_threshold: Decimal = Decimal("0.85")
    cross_asset_leader_inventory_trigger_pct: Decimal = Decimal("0.80")
    cross_asset_inventory_tightening_factor: Decimal = Decimal("0.65")
    cross_asset_inventory_min_pct: Decimal = Decimal("0.20")
    cross_asset_candle_lookback: int = 48
    cross_asset_refresh_seconds: int = 300

    # Strategy Stack
    strategy_stack_enabled: bool = False
    strategy_stack_layers: str = "core,alpha,hedge"
    core_layer_grid_band_multiplier: Decimal = Decimal("1.35")
    core_layer_notional_multiplier: Decimal = Decimal("0.85")
    alpha_layer_grid_band_multiplier: Decimal = Decimal("0.60")
    alpha_layer_notional_multiplier: Decimal = Decimal("0.60")
    alpha_layer_poll_seconds_multiplier: Decimal = Decimal("0.50")
    hedging_layer_grid_band_multiplier: Decimal = Decimal("0.75")
    hedging_layer_notional_multiplier: Decimal = Decimal("0.30")
    hedging_layer_inventory_frac: Decimal = Decimal("0.15")
    hedging_layer_requires_downtrend: bool = True
    strategy_layer_name: str = "single"
    strategy_layer_mode: str = "standard"

    # Cointegration Pair Trading
    cointegration_pair_trading_enabled: bool = False
    cointegration_pairs: str = ""
    cointegration_lookback: int = 96
    cointegration_min_correlation: Decimal = Decimal("0.75")
    cointegration_entry_z: Decimal = Decimal("2.0")
    cointegration_exit_z: Decimal = Decimal("0.75")
    cointegration_max_half_life_bars: Decimal = Decimal("72")

    # API Circuit Breaker
    api_circuit_breaker_enabled: bool = True
    api_latency_p95_threshold_ms: float = 2000.0
    api_failure_rate_threshold_pct: Decimal = Decimal("0.05")
    api_health_window_seconds: int = 300
    api_recovery_consecutive_minutes: int = 5

    # Sentiment Override
    sentiment_override_enabled: bool = False
    sentiment_source_url: str = ""
    sentiment_api_bearer_token: str = ""
    sentiment_json_path: str = "score"
    sentiment_asset_query_param: str = "symbol"
    sentiment_refresh_seconds: int = 300
    sentiment_lookback_seconds: int = 3600
    sentiment_negative_threshold: Decimal = Decimal("-0.6")
    sentiment_safe_inventory_cap_pct: Decimal = Decimal("0.20")

    # Notifications
    notifications_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_whitelist_chat_id: str = ""
    discord_webhook_url: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()
