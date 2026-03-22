"""
Crypto Futures AI Agent - Configuration
========================================
SINGLE SOURCE OF TRUTH for all settings.
Every module imports from here. Change settings HERE only.

FIXED: 2026-03-16
- Raised min_confidence from 0.15 → 0.55
- Raised min_agreement from 0.45 → 0.60  
- Added correlation group limits
- Reduced ML weight from 0.45 → 0.35
- Increased market_structure weight from 0.15 → 0.20
- Added position_correlation_limit
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DATA_CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR = DATA_CACHE_DIR
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "agent.db"

# Auto-create directories
for _dir in [SAVED_MODELS_DIR, DATA_CACHE_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

AGENT_MODE = "paper"

TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
]

PRIMARY_PAIR = "BTCUSDT"

REFERENCE_ASSETS = ["BTCUSDT", "ETHUSDT"]

TIMEFRAMES = {
    "entry": "1h",
    "swing": "4h",
    "macro": "1d",
}

LOOKBACK_CANDLES = {
    "1m": 1000,
    "5m": 1000,
    "15m": 1000,
    "1h": 720,
    "4h": 500,
    "1d": 365,
    "1w": 104,
}

PREDICTION_HORIZONS = {
    "intraday": 6,
    "short_swing": 24,
    "swing": 168,
}

ACTIVE_HORIZON = "short_swing"


FEATURE_CONFIG = {
    "rsi_periods": [7, 14, 21],
    "ema_periods": [9, 21, 50, 100, 200],
    "sma_periods": [20, 50, 100, 200],
    "bb_period": 20,
    "bb_std": 2.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "adx_period": 14,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "obv": True,
    "vwap": True,
    "ichimoku": True,
    "williams_r": True,
    "cci_period": 20,
    "mfi_period": 14,

    "returns_periods": [1, 3, 5, 10, 20],
    "volatility_periods": [10, 20, 30],
    "volume_ma_periods": [10, 20, 50],
    "lag_periods": [1, 2, 3, 5],
    "z_score_period": 20,

    "max_features": 50,
    "selection_method": "importance",
    "min_feature_importance": 0.001,
}


MODEL_CONFIG = {
    "test_size": 0.2,
    "validation_method": "walk_forward",
    "cv_splits": 5,

    # ── FIXED: was 0.55, ML needs higher bar ──
    "confidence_threshold": 0.58,
    "min_model_agreement": 0.6,

    "models": {
        "random_forest": {
            "enabled": True,
            "weight": 0.25,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "xgboost": {
            "enabled": True,
            "weight": 0.30,
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
        },
        "gradient_boosting": {
            "enabled": True,
            "weight": 0.25,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_leaf": 20,
                "random_state": 42,
            },
        },
        "extra_trees": {
            "enabled": True,
            "weight": 0.20,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 15,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    },

    "scaler": "robust",
}


AI_CONFIG = {
    "enabled": bool(HF_TOKEN),

    "models": {
        "analyst": "Qwen/Qwen3-8B",
        "strategist": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "risk_manager": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    },

    "fallback_models": [
        "Qwen/Qwen3-14B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ],

    "max_new_tokens": 500,
    "temperature": 0.6,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "timeout_seconds": 45,
}


# ══════════════════════════════════════════════════════════════
# RISK CONFIG — MAJOR FIXES APPLIED 2026-03-16
# ══════════════════════════════════════════════════════════════
RISK_CONFIG = {
    # Position sizing
    "max_leverage": 3,                    # ← REDUCED from 5 → 3
    "risk_per_trade_pct": 1.5,            # ← REDUCED from 2.0 → 1.5
    "max_position_size_pct": 15.0,        # ← REDUCED from 20 → 15
    "max_open_positions": 2,              # ← REDUCED from 3 → 2

    # Stop-loss / Take-profit
    "default_stop_loss_pct": 2.5,         # ← WIDENED from 2.0 → 2.5
    "default_take_profit_pct": 5.0,       # ← INCREASED from 3.0 → 5.0 (R:R = 2:1)
    "use_atr_stops": True,
    "atr_stop_multiplier": 3.0,           # ← WIDENED from 2.5 → 3.0
    "trailing_stop_pct": 3.5,

    "trailing_stop_tiers": [
        {"min_profit_pct": 3.5, "trail_pct": 1.0},
        {"min_profit_pct": 2.5, "trail_pct": 1.5},
        {"min_profit_pct": 1.5, "trail_pct": 2.0},
    ],

    # Circuit breakers
    "max_daily_loss_pct": 3.0,            # ← TIGHTENED from 5.0 → 3.0
    "max_weekly_loss_pct": 7.0,           # ← TIGHTENED from 10.0 → 7.0
    "max_total_drawdown_pct": 12.0,       # ← TIGHTENED from 15.0 → 12.0
    "cooldown_after_loss_minutes": 60,    # ← INCREASED from 30 → 60
    "max_consecutive_losses": 3,          # ← TIGHTENED from 5 → 3

    # ══ CONFIDENCE FILTERS — THE MAIN FIX ══
    "min_confidence_to_trade": 0.55,      # ← CRITICAL FIX: was 0.15 (!!!)
    "min_agreement_to_trade": 0.60,       # ← RESTORED from 0.45 → 0.60

    # ── NEW: Correlation limits ──
    # BTC/ETH/SOL move together - don't open same direction on all 3
    "correlation_groups": {
        "crypto_major": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "crypto_alt": ["BNBUSDT"],
    },
    "max_same_direction_per_group": 1,    # Only 1 SHORT or 1 LONG per group
}


BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "commission_pct": 0.04,
    "slippage_pct": 0.02,
    "start_date": "2024-01-01",
    "end_date": None,
    "use_leverage": True,
    "default_leverage": 2,                # ← REDUCED from 3 → 2
}


SENTIMENT_CONFIG = {
    "max_headlines": 30,

    "sources": {
        "google_news_rss": True,
        "coindesk_rss": True,
        "cointelegraph_rss": True,
        "reddit_cryptocurrency": True,
        "reddit_bitcoin": True,
        "fear_greed_index": True,
    },

    "finbert_model": "ProsusAI/finbert",
    "use_finbert": True,
    "fallback_to_keywords": True,

    "sentiment_weight": 0.15,
}


# ══════════════════════════════════════════════════════════════
# SIGNAL WEIGHTS — REBALANCED
# ML was too dominant at 0.45, causing short-only bias
# ══════════════════════════════════════════════════════════════
SIGNAL_WEIGHTS = {
    "ml_ensemble": 0.35,          # ← REDUCED from 0.45 → 0.35
    "sentiment": 0.15,            # unchanged
    "ai_reasoning": 0.15,         # unchanged
    "funding_rate": 0.10,         # unchanged
    "market_structure": 0.25,     # ← INCREASED from 0.15 → 0.25
}
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 0.01, "Signal weights must sum to 1.0!"


TELEGRAM_CONFIG = {
    "enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    "send_signals": True,
    "send_daily_report": True,
    "send_errors": True,
    "send_model_retrain": False,
    "quiet_hours": None,
}


SCHEDULE_CONFIG = {
    "analysis_interval_minutes": 60,
    "news_check_interval_minutes": 30,
    "retrain_interval_hours": 24,
    "heartbeat_interval_minutes": 5,
}


LOG_CONFIG = {
    "level": "INFO",
    "to_file": True,
    "to_console": True,
    "log_file": str(LOGS_DIR / "agent.log"),
    "max_file_size_mb": 10,
    "backup_count": 5,
    "format": "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}


BINANCE_CONFIG = {
    "base_url": "https://fapi.binance.com",
    "klines_endpoint": "/fapi/v1/klines",
    "ticker_endpoint": "/fapi/v1/ticker/24hr",
    "funding_rate_endpoint": "/fapi/v1/fundingRate",
    "open_interest_endpoint": "/fapi/v1/openInterest",
    "depth_endpoint": "/fapi/v1/depth",

    "requests_per_minute": 1200,
    "our_limit_per_minute": 60,
    "retry_on_429": True,
    "retry_delay_seconds": 3,

    "testnet_url": "https://testnet.binancefuture.com",
    "use_testnet": True,
}


def validate_config():
    """Run on import to catch config errors early."""
    errors = []

    if not TRADING_PAIRS:
        errors.append("TRADING_PAIRS is empty")

    if PRIMARY_PAIR not in TRADING_PAIRS:
        errors.append(f"PRIMARY_PAIR '{PRIMARY_PAIR}' not in TRADING_PAIRS")

    if AGENT_MODE not in ("backtest", "paper", "live"):
        errors.append(f"Invalid AGENT_MODE: '{AGENT_MODE}'")

    if AGENT_MODE == "live" and not BINANCE_API_KEY:
        errors.append("AGENT_MODE is 'live' but BINANCE_API_KEY is empty")

    if RISK_CONFIG["max_leverage"] > 20:
        errors.append("max_leverage > 20 is extremely dangerous")

    if RISK_CONFIG["risk_per_trade_pct"] > 10:
        errors.append("risk_per_trade_pct > 10% is extremely dangerous")

    if RISK_CONFIG["min_confidence_to_trade"] < 0.40:
        errors.append(f"min_confidence_to_trade={RISK_CONFIG['min_confidence_to_trade']} is too low! Minimum 0.40")

    total_weight = sum(
        m["weight"] for m in MODEL_CONFIG["models"].values() if m["enabled"]
    )
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Model weights sum to {total_weight}, should be 1.0")

    if errors:
        print("\n⚠️  CONFIG ERRORS:")
        for e in errors:
            print(f"   ❌ {e}")
        print()

    return len(errors) == 0

_config_valid = validate_config()