"""
Crypto Futures AI Agent - Configuration
========================================
SINGLE SOURCE OF TRUTH for all settings.
Every module imports from here. Change settings HERE only.

To switch modes:  Change AGENT_MODE
To add pairs:     Add to TRADING_PAIRS
To adjust risk:   Edit RISK_CONFIG
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ╔═══════════════════════════════════════════════════════════╗
# ║  PROJECT PATHS                                            ║
# ╚═══════════════════════════════════════════════════════════╝

BASE_DIR = Path(__file__).parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
DATA_CACHE_DIR = BASE_DIR / "cache"
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

AGENT_MODE = "backtest"

TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

PRIMARY_PAIR = "BTCUSDT"

REFERENCE_ASSETS = ["BTCUSDT", "ETHUSDT"]


# ╔═══════════════════════════════════════════════════════════╗
# ║  TIMEFRAMES                                                ║
# ║  Multi-timeframe analysis for better accuracy              ║
# ║                                                            ║
# ║  Strategy:                                                 ║
# ║    macro (1d)  → Identify the TREND                       ║
# ║    swing (4h)  → Identify the SETUP                       ║
# ║    entry (1h)  → Time the ENTRY                           ║
# ╚═══════════════════════════════════════════════════════════╝

TIMEFRAMES = {
    "entry": "1h",           # Entry timing
    "swing": "4h",           # Setup identification
    "macro": "1d",           # Trend direction
}

# How many candles to fetch per timeframe
LOOKBACK_CANDLES = {
    "1m": 1000,
    "5m": 1000,
    "15m": 1000,
    "1h": 720,     # 30 days
    "4h": 500,     # ~83 days
    "1d": 365,     # 1 year
    "1w": 104,     # 2 years
}

# Prediction horizons (in candles of entry timeframe)
# e.g., if entry=1h, horizon=24 means predict 24h ahead
PREDICTION_HORIZONS = {
    "intraday": 6,       # 6 hours ahead
    "short_swing": 24,   # 1 day ahead
    "swing": 168,        # 1 week ahead
}

# Active horizon for predictions
ACTIVE_HORIZON = "short_swing"


# ╔═══════════════════════════════════════════════════════════╗
# ║  FEATURE ENGINEERING                                       ║
# ╚═══════════════════════════════════════════════════════════╝

FEATURE_CONFIG = {
    # ── Technical Indicators ──
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

    # ── Derived / Statistical ──
    "returns_periods": [1, 3, 5, 10, 20],
    "volatility_periods": [10, 20, 30],
    "volume_ma_periods": [10, 20, 50],
    "lag_periods": [1, 2, 3, 5],
    "z_score_period": 20,

    # ── Feature Selection ──
    "max_features": 50,              # Top N features to keep
    "selection_method": "importance", # "importance" or "mutual_info"
    "min_feature_importance": 0.001,
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  ML MODELS                                                ║
# ╚═══════════════════════════════════════════════════════════╝

MODEL_CONFIG = {
    # Data split
    "test_size": 0.2,
    "validation_method": "walk_forward",  # "walk_forward" or "time_series_split"
    "cv_splits": 5,

    # Signal generation
    "confidence_threshold": 0.55,    # Min confidence to act
    "min_model_agreement": 0.6,      # Min % of models must agree

    # Individual models + weights
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

    # Scaler
    "scaler": "robust",  # "robust" or "standard" or "minmax"
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  AI BRAIN (HuggingFace Free Inference)                     ║
# ╚═══════════════════════════════════════════════════════════╝

AI_CONFIG = {
    "enabled": bool(HF_TOKEN),

    # Three AI "experts" for reasoning
    "models": {
        "analyst": "mistralai/Mistral-7B-Instruct-v0.3",
        "strategist": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "risk_manager": "microsoft/Phi-3-mini-4k-instruct",
    },

    # Fallback models if primary ones are busy/rate-limited
    "fallback_models": [
        "google/gemma-2-2b-it",
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/falcon-7b-instruct",
    ],

    "max_new_tokens": 500,
    "temperature": 0.3,
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
    "timeout_seconds": 30,
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  RISK MANAGEMENT                                          ║
# ║  THIS IS THE MOST IMPORTANT CONFIG                        ║
# ║  These settings PROTECT your capital                       ║
# ╚═══════════════════════════════════════════════════════════╝

RISK_CONFIG = {
    # Position sizing
    "max_leverage": 5,
    "risk_per_trade_pct": 2.0,       # Risk max 2% of capital per trade
    "max_position_size_pct": 20.0,   # Max 20% of capital in one position
    "max_open_positions": 3,

    # Stop-loss / Take-profit (as % from entry)
    "default_stop_loss_pct": 2.0,
    "default_take_profit_pct": 4.0,  # 2:1 reward-to-risk
    "use_atr_stops": True,           # Dynamic SL based on ATR
    "atr_stop_multiplier": 2.0,     # SL = entry ± (ATR × 2)
    "trailing_stop_pct": 1.5,

    # Circuit breakers
    "max_daily_loss_pct": 5.0,       # Pause trading if daily loss > 5%
    "max_weekly_loss_pct": 10.0,     # Pause trading if weekly loss > 10%
    "max_total_drawdown_pct": 15.0,  # STOP EVERYTHING if drawdown > 15%
    "cooldown_after_loss_minutes": 30,
    "max_consecutive_losses": 5,     # Pause after 5 losses in a row

    # Confidence filters
    "min_confidence_to_trade": 0.58, # Only trade if confidence > 58%
    "min_agreement_to_trade": 0.60,  # Only trade if 60%+ models agree
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  BACKTESTING                                               ║
# ╚═══════════════════════════════════════════════════════════╝

BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "commission_pct": 0.04,          # Binance futures taker fee
    "slippage_pct": 0.02,           # Estimated slippage
    "start_date": "2024-01-01",
    "end_date": None,               # None = up to today
    "use_leverage": True,
    "default_leverage": 3,
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  NEWS & SENTIMENT                                         ║
# ╚═══════════════════════════════════════════════════════════╝

SENTIMENT_CONFIG = {
    "max_headlines": 30,

    # Data sources (all free)
    "sources": {
        "google_news_rss": True,
        "coindesk_rss": True,
        "cointelegraph_rss": True,
        "reddit_cryptocurrency": True,
        "reddit_bitcoin": True,
        "fear_greed_index": True,
    },

    # FinBERT for financial sentiment
    "finbert_model": "ProsusAI/finbert",
    "use_finbert": True,
    "fallback_to_keywords": True,     # If FinBERT fails, use keyword method

    # Sentiment weight in final signal
    "sentiment_weight": 0.15,         # 15% of final signal
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  SIGNAL ENGINE WEIGHTS                                     ║
# ║  How much each component contributes to final signal       ║
# ╚═══════════════════════════════════════════════════════════╝

SIGNAL_WEIGHTS = {
    "ml_ensemble": 0.45,      # ML models = 45%
    "sentiment": 0.15,        # News sentiment = 15%
    "ai_reasoning": 0.15,     # HuggingFace AI = 15%
    "funding_rate": 0.10,     # Funding rate bias = 10%
    "market_structure": 0.15, # Trend alignment across timeframes = 15%
}
# Must sum to 1.0
assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 0.01, "Signal weights must sum to 1.0!"


# ╔═══════════════════════════════════════════════════════════╗
# ║  TELEGRAM NOTIFICATIONS                                    ║
# ╚═══════════════════════════════════════════════════════════╝

TELEGRAM_CONFIG = {
    "enabled": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    "send_signals": True,            # Send trade signals
    "send_daily_report": True,       # Daily P&L summary
    "send_errors": True,             # Send critical errors
    "send_model_retrain": False,     # Notify on model retraining
    "quiet_hours": None,             # e.g., (23, 7) to mute 11pm-7am
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  SCHEDULING                                                ║
# ╚═══════════════════════════════════════════════════════════╝

SCHEDULE_CONFIG = {
    "analysis_interval_minutes": 60,    # Full analysis every hour
    "news_check_interval_minutes": 30,  # Sentiment check every 30 min
    "retrain_interval_hours": 24,       # Retrain ML models daily
    "heartbeat_interval_minutes": 5,    # "I'm alive" check
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  LOGGING                                                   ║
# ╚═══════════════════════════════════════════════════════════╝

LOG_CONFIG = {
    "level": "INFO",                 # DEBUG | INFO | WARNING | ERROR
    "to_file": True,
    "to_console": True,
    "log_file": str(LOGS_DIR / "agent.log"),
    "max_file_size_mb": 10,
    "backup_count": 5,
    "format": "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  BINANCE API ENDPOINTS (public, no keys needed)            ║
# ╚═══════════════════════════════════════════════════════════╝

BINANCE_CONFIG = {
    # Futures endpoints (public data - no API key required)
    "base_url": "https://fapi.binance.com",
    "klines_endpoint": "/fapi/v1/klines",
    "ticker_endpoint": "/fapi/v1/ticker/24hr",
    "funding_rate_endpoint": "/fapi/v1/fundingRate",
    "open_interest_endpoint": "/fapi/v1/openInterest",
    "depth_endpoint": "/fapi/v1/depth",

    # Rate limiting (be respectful)
    "requests_per_minute": 1200,     # Binance limit
    "our_limit_per_minute": 60,      # Stay well under
    "retry_on_429": True,
    "retry_delay_seconds": 3,

    # Testnet (for paper/live trading later)
    "testnet_url": "https://testnet.binancefuture.com",
    "use_testnet": True,             # Always testnet until confident
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  VALIDATION - Catches config errors on import             ║
# ╚═══════════════════════════════════════════════════════════╝

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