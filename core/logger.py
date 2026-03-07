"""
core/logger.py
==============
Centralized logging for the entire agent.

Usage in any module:
    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is a message")
    logger.error("Something broke", exc_info=True)

Features:
    - Console output (colored by level)
    - Rotating file output (10MB max, 5 backups)
    - One call: get_logger("module_name")
    - All config from config.py
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Import config - handle case where logger is tested standalone
try:
    from config import LOG_CONFIG, LOGS_DIR
except ImportError:
    # Fallback defaults if config not available
    LOGS_DIR = Path(__file__).parent.parent / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
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


# ── ANSI Colors for console output ──────────────────────────

class ColorFormatter(logging.Formatter):
    """Adds color to console log output based on level."""

    COLORS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[91m",    # Red
        "CRITICAL": "\033[41m",    # Red background
    }
    RESET = "\033[0m"

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ── Track which loggers are already set up ───────────────────

_initialized_loggers = set()


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Module name (use __name__ from calling module)

    Returns:
        logging.Logger configured with console + file handlers

    Example:
        logger = get_logger(__name__)
        logger.info("Starting data collection")
        logger.warning("Rate limit approaching")
        logger.error("API call failed", exc_info=True)
    """

    # Shorten name for display: "data.binance_data" instead of full path
    short_name = name.replace("crypto_agent.", "").replace("__main__", "main")

    # Don't add handlers twice
    if short_name in _initialized_loggers:
        return logging.getLogger(short_name)

    logger = logging.getLogger(short_name)
    logger.setLevel(getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO))

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    log_format = LOG_CONFIG.get(
        "format",
        "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"
    )
    date_format = LOG_CONFIG.get("date_format", "%Y-%m-%d %H:%M:%S")

    # ── Console Handler ──
    if LOG_CONFIG.get("to_console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(
            getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO)
        )
        console_handler.setFormatter(ColorFormatter(log_format, date_format))
        logger.addHandler(console_handler)

    # ── File Handler (rotating) ──
    if LOG_CONFIG.get("to_file", True):
        log_file = Path(LOG_CONFIG.get("log_file", "logs/agent.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LOG_CONFIG.get("max_file_size_mb", 10) * 1024 * 1024,
            backupCount=LOG_CONFIG.get("backup_count", 5),
            encoding="utf-8",
        )
        file_handler.setLevel(
            getattr(logging, LOG_CONFIG["level"].upper(), logging.INFO)
        )
        # No colors in file
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    _initialized_loggers.add(short_name)
    return logger


def set_level(level: str):
    """
    Change log level globally at runtime.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR"

    Example:
        set_level("DEBUG")  # See everything
        set_level("WARNING")  # Quiet mode
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    for name in _initialized_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


# ╔═══════════════════════════════════════════════════════════╗
# ║  STANDALONE TEST                                           ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  LOGGER TEST")
    print("=" * 60)

    # Create loggers for different modules
    log1 = get_logger("data.binance_data")
    log2 = get_logger("models.ensemble")
    log3 = get_logger("trading.risk_manager")
    log4 = get_logger("main")

    # Test all levels
    log1.debug("Fetching BTCUSDT 1h candles (debug)")
    log1.info("Fetched 720 candles for BTCUSDT")
    log2.info("Training ensemble model on 500 samples")
    log2.warning("Feature 'funding_rate' has 15% missing values")
    log3.error("Daily loss limit reached: -5.2%")
    log4.info("Agent started in backtest mode")

    # Test duplicate prevention (should NOT add handlers again)
    log1_again = get_logger("data.binance_data")
    log1_again.info("This should appear only ONCE (no duplicate)")

    # Test level change
    print("\n--- Switching to WARNING level ---\n")
    set_level("WARNING")
    log1.info("This should NOT appear (INFO < WARNING)")
    log1.warning("This SHOULD appear (WARNING)")

    # Reset
    set_level("INFO")

    # Test error with traceback
    try:
        result = 1 / 0
    except Exception:
        log3.error("Division by zero caught", exc_info=True)

    print("\n" + "=" * 60)
    print(f"  ✅ Logger test complete")
    print(f"  📁 Check log file: {LOG_CONFIG.get('log_file', 'logs/agent.log')}")
    print("=" * 60)