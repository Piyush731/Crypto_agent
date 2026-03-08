"""
Crypto Futures AI Agent - Paper Trader
=======================================
Real-time paper trading with simulated execution.

Uses the full signal pipeline:
  SignalEngine (ML ensemble + sentiment + AI reasoning)
  → RiskManager (position sizing, SL/TP, circuit breakers)
  → Simulated execution (logged to SQLite)

Features:
  - Real Binance prices for entry/exit
  - Full risk management (position sizing, SL/TP, trailing, circuit breakers)
  - All trades logged to SQLite (survives restarts)
  - Automatic model retraining when stale
  - Capital tracking across restarts (DB-backed)
  - Single position per symbol (no stacking)
  - Continuous loop or single-cycle mode
  - Exit checks on every cycle for open positions
"""

import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    BACKTEST_CONFIG,
    TRADING_PAIRS,
    SCHEDULE_CONFIG,
    AI_CONFIG,
)
from core.logger import get_logger
from core.db import get_db
from trading.risk_manager import RiskManager

logger = get_logger(__name__)


class PaperTrader:
    """
    Real-time paper trader with simulated execution.

    Usage:
        pt = PaperTrader(capital=10000)

        # Single cycle (for testing / cron)
        result = pt.run_cycle()

        # Continuous loop
        pt.start(max_cycles=24)   # Run 24 hours @ 1h interval
        # or
        pt.start()                # Run forever until pt.stop()
    """

    # How often to check model staleness (in cycles)
    # At 1h analysis interval: 12 = every 12 hours
    MODEL_CHECK_INTERVAL = 12

    def __init__(self, capital: float = None, include_ai: bool = None):
        """
        Args:
            capital:     Starting capital (USDT). Default from BACKTEST_CONFIG.
            include_ai:  Include AI reasoning in signals. Default from AI_CONFIG.
        """
        self.initial_capital = capital or BACKTEST_CONFIG["initial_capital"]
        self.mode = "paper"
        self.include_ai = (
            include_ai if include_ai is not None
            else AI_CONFIG.get("enabled", False)
        )

        # ── Core components ──
        self.risk_mgr = RiskManager(capital=self.initial_capital, mode=self.mode)
        self.db = get_db()

        # ── Lazy-loaded components (avoid import overhead / circular) ──
        self._signal_engine = None
        self._binance = None
        self._trainer = None

        # ── Position state not stored in DB ──
        # {trade_id: {highest_price, lowest_price, current_stop}}
        self._position_state: Dict[int, Dict] = {}

        # ── Runtime state ──
        self._running = False
        self._cycles = 0
        self._last_cycle_time: Optional[datetime] = None

        # Commission / slippage from config
        self._commission_pct = BACKTEST_CONFIG.get("commission_pct", 0.04)
        self._slippage_pct = BACKTEST_CONFIG.get("slippage_pct", 0.02)

        # ── Load state from DB ──
        self._load_capital()
        self._init_position_states()

        logger.info(
            f"PaperTrader initialized | capital=${self.risk_mgr.get_capital():,.2f} | "
            f"mode={self.mode} | AI={'ON' if self.include_ai else 'OFF'} | "
            f"open_positions={len(self._position_state)}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  LAZY PROPERTIES
    # ═══════════════════════════════════════════════════════════════════

    @property
    def signal_engine(self):
        """Lazy-load SignalEngine to avoid circular imports."""
        if self._signal_engine is None:
            from analysis.signal_engine import SignalEngine
            self._signal_engine = SignalEngine()
        return self._signal_engine

    @property
    def binance(self):
        """Lazy-load BinanceData."""
        if self._binance is None:
            from data.binance_data import BinanceData
            self._binance = BinanceData()
        return self._binance

    @property
    def trainer(self):
        """Lazy-load ModelTrainer."""
        if self._trainer is None:
            from models.trainer import ModelTrainer
            self._trainer = ModelTrainer()
        return self._trainer

    # ═══════════════════════════════════════════════════════════════════
    #  STARTUP: load state from DB
    # ═══════════════════════════════════════════════════════════════════

    def _load_capital(self):
        """
        Reconstruct current capital from DB.

        capital = initial_capital + sum(net PnL of all closed paper trades)
        """
        try:
            total_pnl = self.db.get_total_pnl(self.mode)
            current = self.initial_capital + total_pnl
            self.risk_mgr.update_capital(current)
            logger.info(
                f"Capital loaded from DB: ${current:,.2f} "
                f"(initial=${self.initial_capital:,.2f}, "
                f"pnl=${total_pnl:+,.2f})"
            )
        except Exception as e:
            logger.warning(f"Could not load capital from DB: {e}")

    def _init_position_states(self):
        """
        Initialize position tracking for existing open trades.

        On restart, highest/lowest reset to current/entry price
        and trailing stop resets to initial SL. Conservative but safe.
        """
        try:
            open_trades = self.db.get_open_trades(self.mode)
            if not open_trades:
                return

            for trade in open_trades:
                trade_id = trade.get("id")
                entry_price = float(trade.get("entry_price", 0))
                stop_loss = float(trade.get("stop_loss", entry_price))
                symbol = trade.get("symbol", "")

                # Try to get current price for better initialization
                highest = entry_price
                lowest = entry_price
                try:
                    ticker = self.binance.get_ticker(symbol)
                    if ticker and ticker.get("last_price"):
                        current = float(ticker["last_price"])
                        highest = max(entry_price, current)
                        lowest = min(entry_price, current)
                except Exception:
                    pass

                self._position_state[trade_id] = {
                    "highest_price": highest,
                    "lowest_price": lowest,
                    "current_stop": stop_loss,
                }

            logger.info(
                f"Initialized {len(self._position_state)} open position states"
            )
        except Exception as e:
            logger.warning(f"Could not initialize position states: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════════════════════════

    def start(self, max_cycles: int = None):
        """
        Main continuous trading loop.

        Runs cycles at the configured interval until stopped.

        Args:
            max_cycles: Maximum cycles to run (None = forever).
        """
        self._running = True
        interval_s = SCHEDULE_CONFIG["analysis_interval_minutes"] * 60

        logger.info(
            f"{'═'*50}\n"
            f"  PAPER TRADER STARTING\n"
            f"  Capital: ${self.risk_mgr.get_capital():,.2f}\n"
            f"  Pairs: {TRADING_PAIRS}\n"
            f"  Interval: {interval_s // 60} min\n"
            f"  AI: {'ON' if self.include_ai else 'OFF'}\n"
            f"  Max cycles: {max_cycles or '∞'}\n"
            f"{'═'*50}"
        )

        cycle = 0
        while self._running:
            if max_cycles is not None and cycle >= max_cycles:
                logger.info(f"Reached max_cycles={max_cycles} — stopping")
                break

            try:
                self.run_cycle()
                cycle += 1

                if self._running and (max_cycles is None or cycle < max_cycles):
                    logger.info(
                        f"Next cycle in {interval_s // 60} min "
                        f"(cycle {cycle}/{max_cycles or '∞'})…"
                    )
                    # Sleep in 1-second chunks for responsive shutdown
                    for _ in range(interval_s):
                        if not self._running:
                            break
                        time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Paper trader interrupted (Ctrl+C)")
                self._running = False

            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                self.db.save_error("paper_trader", "cycle_error", str(e))
                # Wait 60s before retry on unhandled error
                for _ in range(60):
                    if not self._running:
                        break
                    time.sleep(1)

        self._running = False
        logger.info(
            f"Paper trader stopped after {cycle} cycles | "
            f"capital=${self.risk_mgr.get_capital():,.2f}"
        )

    def stop(self):
        """Request graceful shutdown (takes effect after current operation)."""
        self._running = False
        logger.info("Paper trader stop requested")

    # ═══════════════════════════════════════════════════════════════════
    #  SINGLE CYCLE
    # ═══════════════════════════════════════════════════════════════════

    def run_cycle(self) -> Dict:
        """
        Execute one full analysis cycle.

        Steps:
          1. Check open positions for exits
          2. Ensure models are trained (periodic)
          3. For each pair without a position: generate signal → open if approved
          4. Log summary

        Returns:
            Dict: cycle summary with counts and timing
        """
        t0 = time.time()
        self._cycles += 1

        result = {
            "cycle": self._cycles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions_checked": 0,
            "positions_closed": 0,
            "signals_generated": 0,
            "positions_opened": 0,
            "errors": [],
            "capital": 0.0,
        }

        logger.info(f"{'─'*50}")
        logger.info(
            f"Paper Cycle #{self._cycles} | "
            f"capital=${self.risk_mgr.get_capital():,.2f}"
        )

        # ── 1. Check open positions ──
        try:
            closed = self._check_open_positions()
            open_count = len(self.db.get_open_trades(self.mode))
            result["positions_checked"] = open_count + len(closed)
            result["positions_closed"] = len(closed)
        except Exception as e:
            logger.error(f"Position check error: {e}", exc_info=True)
            result["errors"].append(f"position_check: {e}")

        # ── 2. Ensure models are trained (periodic) ──
        if self._cycles == 1 or self._cycles % self.MODEL_CHECK_INTERVAL == 0:
            try:
                self._ensure_models_ready()
            except Exception as e:
                logger.error(f"Model check error: {e}", exc_info=True)
                result["errors"].append(f"model_check: {e}")

        # ── 3. Generate signals for pairs without open positions ──
        try:
            open_trades = self.db.get_open_trades(self.mode)
            open_symbols = {t.get("symbol") for t in open_trades}
        except Exception:
            open_symbols = set()

        for symbol in TRADING_PAIRS:
            if symbol in open_symbols:
                logger.debug(f"[{symbol}] Has open position — skip")
                continue

            try:
                sig_result = self._process_symbol(symbol)
                result["signals_generated"] += 1
                if sig_result.get("opened"):
                    result["positions_opened"] += 1
            except Exception as e:
                err_msg = f"[{symbol}] Error: {e}"
                logger.error(err_msg, exc_info=True)
                result["errors"].append(err_msg)

        # ── 4. Summary ──
        result["capital"] = round(self.risk_mgr.get_capital(), 2)
        result["cycle_time_s"] = round(time.time() - t0, 2)
        self._last_cycle_time = datetime.now(timezone.utc)

        logger.info(
            f"Cycle #{self._cycles} done | "
            f"opened={result['positions_opened']} closed={result['positions_closed']} | "
            f"capital=${result['capital']:,.2f} | "
            f"{result['cycle_time_s']}s"
        )

        return result

    # ═══════════════════════════════════════════════════════════════════
    #  CHECK OPEN POSITIONS
    # ═══════════════════════════════════════════════════════════════════

    def _check_open_positions(self) -> List[Dict]:
        """
        Check all open paper positions for exit conditions.

        Uses tick-level check (check_exit_conditions) since we see
        one price snapshot per cycle, not full bars.

        Returns:
            List of closed trade results
        """
        closed_trades: List[Dict] = []
        open_trades = self.db.get_open_trades(self.mode)

        if not open_trades:
            return closed_trades

        logger.info(f"Checking {len(open_trades)} open position(s)…")

        for trade in open_trades:
            try:
                trade_id = trade.get("id")
                symbol = trade.get("symbol", "")
                direction = int(trade.get("direction", 0))
                entry_price = float(trade.get("entry_price", 0))
                stop_loss = float(trade.get("stop_loss", 0))
                take_profit = float(trade.get("take_profit", 0))

                # ── Get current price ──
                ticker = self.binance.get_ticker(symbol)
                if not ticker or not ticker.get("last_price"):
                    logger.warning(
                        f"[{symbol}] Cannot get price — skip exit check"
                    )
                    continue

                current_price = float(ticker["last_price"])

                # ── Get / update position state ──
                state = self._position_state.get(trade_id, {
                    "highest_price": max(entry_price, current_price),
                    "lowest_price": min(entry_price, current_price),
                    "current_stop": stop_loss,
                })
                state["highest_price"] = max(
                    state["highest_price"], current_price
                )
                state["lowest_price"] = min(
                    state["lowest_price"], current_price
                )
                self._position_state[trade_id] = state

                # ── Check exit conditions (tick-level) ──
                exit_check = self.risk_mgr.check_exit_conditions(
                    entry_price=entry_price,
                    current_price=current_price,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    highest_price=state["highest_price"],
                    lowest_price=state["lowest_price"],
                    current_stop=state["current_stop"],
                )

                # Update trailing stop (even if no exit)
                state["current_stop"] = exit_check.get(
                    "new_trailing_stop", state["current_stop"]
                )

                if exit_check["should_exit"]:
                    exit_price = exit_check.get("exit_price", current_price)
                    exit_reason = exit_check.get("exit_reason", "unknown")

                    close_result = self._close_position(
                        trade, current_price, exit_reason, exit_price
                    )
                    closed_trades.append(close_result)
                else:
                    pnl_pct = exit_check.get("pnl_pct", 0)
                    logger.debug(
                        f"[{symbol}] Open | "
                        f"entry={entry_price:.2f} now={current_price:.2f} "
                        f"PnL={pnl_pct:+.2f}% "
                        f"trail_stop={state['current_stop']:.2f}"
                    )

            except Exception as e:
                logger.error(
                    f"Position check error (trade {trade.get('id')}): {e}",
                    exc_info=True,
                )

        return closed_trades

    # ═══════════════════════════════════════════════════════════════════
    #  PROCESS ONE SYMBOL (signal → evaluate → open)
    # ═══════════════════════════════════════════════════════════════════

    def _process_symbol(self, symbol: str) -> Dict:
        """
        Generate signal for one symbol and open position if approved.

        Returns:
            Dict: {symbol, signal, confidence, opened, trade_id?, rejected_reason?}
        """
        result = {
            "symbol": symbol,
            "signal": "HOLD",
            "opened": False,
        }

        try:
            # ── Generate signal via full pipeline ──
            signal = self.signal_engine.generate_signal(
                symbol, include_ai=self.include_ai
            )

            if not signal:
                logger.info(f"[{symbol}] No signal generated")
                return result

            sig_type = signal.get("signal", "HOLD")
            confidence = signal.get("confidence", 0)
            result["signal"] = sig_type
            result["confidence"] = confidence

            if sig_type == "HOLD":
                logger.info(
                    f"[{symbol}] HOLD | conf={confidence:.3f}"
                )
                return result

            logger.info(
                f"[{symbol}] Signal: {sig_type} | "
                f"conf={confidence:.3f} | "
                f"score={signal.get('combined_score', 0):+.3f}"
            )

            # ── Evaluate through risk manager ──
            eval_result = self.risk_mgr.evaluate_signal(signal)

            if not eval_result["approved"]:
                result["rejected_reason"] = eval_result["reason"]
                logger.info(
                    f"[{symbol}] Rejected: {eval_result['reason']}"
                )
                return result

            # ── Open position ──
            trade = self._open_position(symbol, signal, eval_result)
            if trade.get("trade_id"):
                result["opened"] = True
                result["trade_id"] = trade["trade_id"]

            return result

        except Exception as e:
            logger.error(f"[{symbol}] Process error: {e}", exc_info=True)
            result["error"] = str(e)
            return result

    # ═══════════════════════════════════════════════════════════════════
    #  OPEN / CLOSE POSITION
    # ═══════════════════════════════════════════════════════════════════

    def _open_position(
        self, symbol: str, signal: Dict, risk_eval: Dict
    ) -> Dict:
        """
        Open a simulated paper position.

        Gets real Binance price, applies slippage, saves to DB.
        """
        try:
            # ── Current price ──
            ticker = self.binance.get_ticker(symbol)
            if ticker and ticker.get("last_price"):
                market_price = float(ticker["last_price"])
            else:
                market_price = float(signal.get("entry_price", 0))

            if market_price <= 0:
                logger.error(f"[{symbol}] Invalid entry price: {market_price}")
                return {"error": "invalid price"}

            # ── Apply slippage ──
            direction = int(signal["direction"])
            slip = market_price * (self._slippage_pct / 100.0)
            entry_price = (
                market_price + slip if direction == 1
                else market_price - slip
            )

            now = datetime.now(timezone.utc)

            # ── Save to DB ──
            trade_data = {
                "symbol": symbol,
                "mode": self.mode,
                "signal_type": signal["signal"],
                "direction": direction,
                "entry_price": round(entry_price, 8),
                "position_size_usd": risk_eval["position_size_usd"],
                "leverage": risk_eval["leverage"],
                "stop_loss": risk_eval["stop_loss"],
                "take_profit": risk_eval["take_profit"],
                "confidence": signal.get("confidence", 0),
                "margin_used": risk_eval.get("margin_required", 0),
                "risk_reward_ratio": risk_eval.get("risk_reward_ratio", 0),
                "entry_time": now.isoformat(),
                "status": "open",
                "signal_id": signal.get("signal_id"),
            }

            trade_id = self.db.save_trade(trade_data)
            trade_data["trade_id"] = trade_id

            # ── Initialize position state ──
            self._position_state[trade_id] = {
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "current_stop": risk_eval["stop_loss"],
            }

            logger.info(
                f"📈 OPENED [{symbol}] {signal['signal']} | "
                f"entry=${entry_price:,.2f} | "
                f"size=${risk_eval['position_size_usd']:,.2f} | "
                f"lev={risk_eval['leverage']}x | "
                f"SL={risk_eval['stop_loss']:.2f} | "
                f"TP={risk_eval['take_profit']:.2f} | "
                f"id={trade_id}"
            )

            return trade_data

        except Exception as e:
            logger.error(f"Open position error [{symbol}]: {e}", exc_info=True)
            return {"error": str(e)}

    def _close_position(
        self,
        trade: Dict,
        current_price: float,
        reason: str,
        exit_price: float = None,
    ) -> Dict:
        """
        Close a simulated paper position.

        Calculates PnL, updates DB, updates risk manager state.
        """
        try:
            trade_id = trade["id"]
            symbol = trade.get("symbol", "?")
            direction = int(trade.get("direction", 0))
            entry_price = float(trade.get("entry_price", 0))
            position_size = float(trade.get("position_size_usd", 0))
            leverage = int(trade.get("leverage", 1))

            if exit_price is None:
                exit_price = current_price

            # ── PnL calculation ──
            if entry_price <= 0:
                logger.error(f"Invalid entry_price for trade {trade_id}")
                return {"error": "invalid entry_price"}

            price_frac = (exit_price - entry_price) / entry_price * direction
            gross_pnl = position_size * price_frac

            # Round-trip commission (entry + exit)
            commission = position_size * (self._commission_pct / 100.0) * 2
            net_pnl = gross_pnl - commission

            capital = self.risk_mgr.get_capital()
            pnl_pct = (
                (net_pnl / capital * 100.0) if capital > 0 else 0.0
            )

            # ── Close in DB ──
            self.db.close_trade(
                trade_id=trade_id,
                exit_price=round(exit_price, 8),
                exit_reason=reason,
                pnl_usd=round(net_pnl, 4),
                pnl_percent=round(pnl_pct, 4),
                commission=round(commission, 4),
            )

            # ── Update risk manager ──
            is_win = net_pnl > 0
            self.risk_mgr.record_trade_result(net_pnl, is_win)

            # ── Clean up position state ──
            self._position_state.pop(trade_id, None)

            emoji = "✅" if is_win else "❌"
            logger.info(
                f"{emoji} CLOSED [{symbol}] {reason} | "
                f"entry=${entry_price:,.2f} → exit=${exit_price:,.2f} | "
                f"PnL=${net_pnl:+,.2f} ({pnl_pct:+.2f}%) | "
                f"capital=${self.risk_mgr.get_capital():,.2f}"
            )

            return {
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": reason,
                "gross_pnl": round(gross_pnl, 4),
                "commission": round(commission, 4),
                "net_pnl": round(net_pnl, 4),
                "pnl_pct": round(pnl_pct, 4),
                "is_win": is_win,
            }

        except Exception as e:
            logger.error(
                f"Close position error (trade {trade.get('id')}): {e}",
                exc_info=True,
            )
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    #  MODEL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _ensure_models_ready(self):
        """
        Check all trading pairs for model staleness and retrain if needed.

        Uses ModelTrainer.retrain_if_needed() which checks the timestamp
        of saved model files against SCHEDULE_CONFIG.retrain_interval_hours.
        Only expensive when actual retraining is needed.
        """
        logger.info("Checking model freshness…")

        for symbol in TRADING_PAIRS:
            try:
                result = self.trainer.retrain_if_needed(symbol)

                if result.get("retrained"):
                    logger.info(
                        f"[{symbol}] Model retrained: {result.get('reason')}"
                    )
                elif result.get("error"):
                    logger.warning(
                        f"[{symbol}] Retrain failed: {result.get('error')}"
                    )
                else:
                    logger.debug(
                        f"[{symbol}] Model OK: {result.get('reason', 'fresh')}"
                    )

            except Exception as e:
                logger.warning(f"[{symbol}] Model check failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    #  EMERGENCY: FORCE CLOSE ALL
    # ═══════════════════════════════════════════════════════════════════

    def force_close_all(self) -> List[Dict]:
        """
        Force-close all open paper positions at current market price.

        Use for: emergency shutdown, end-of-day, testing cleanup.

        Returns:
            List of close results
        """
        open_trades = self.db.get_open_trades(self.mode)

        if not open_trades:
            logger.info("No open positions to close")
            return []

        logger.warning(
            f"FORCE CLOSING {len(open_trades)} open position(s)…"
        )

        results = []
        for trade in open_trades:
            try:
                symbol = trade.get("symbol", "")
                ticker = self.binance.get_ticker(symbol)

                if ticker and ticker.get("last_price"):
                    current_price = float(ticker["last_price"])
                else:
                    current_price = float(trade.get("entry_price", 0))

                close_result = self._close_position(
                    trade, current_price, "force_close"
                )
                results.append(close_result)

            except Exception as e:
                logger.error(
                    f"Force close error (trade {trade.get('id')}): {e}"
                )
                results.append({"error": str(e), "trade_id": trade.get("id")})

        total_pnl = sum(
            r.get("net_pnl", 0) for r in results if "net_pnl" in r
        )
        logger.warning(
            f"Force close complete: {len(results)} positions | "
            f"total PnL=${total_pnl:+,.2f}"
        )

        return results

    # ═══════════════════════════════════════════════════════════════════
    #  STATUS / QUERIES
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Get full paper trader status."""
        try:
            open_trades = self.db.get_open_trades(self.mode)
            capital = self.risk_mgr.get_capital()
            total_pnl = capital - self.initial_capital
            total_ret = (
                total_pnl / self.initial_capital * 100.0
                if self.initial_capital > 0 else 0.0
            )

            return {
                "mode": self.mode,
                "running": self._running,
                "cycles_completed": self._cycles,
                "last_cycle": (
                    self._last_cycle_time.isoformat()
                    if self._last_cycle_time else None
                ),
                "include_ai": self.include_ai,
                "capital": round(capital, 2),
                "initial_capital": round(self.initial_capital, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_ret, 2),
                "open_positions": len(open_trades),
                "open_symbols": [t.get("symbol") for t in open_trades],
                "trading_pairs": TRADING_PAIRS,
                "analysis_interval_min": SCHEDULE_CONFIG[
                    "analysis_interval_minutes"
                ],
            }

        except Exception as e:
            logger.error(f"Status error: {e}")
            return {"error": str(e)}

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions with live unrealized P&L.

        Fetches current prices from Binance for each position.
        """
        open_trades = self.db.get_open_trades(self.mode)
        positions = []

        for trade in open_trades:
            pos = dict(trade)
            symbol = trade.get("symbol", "")
            entry_price = float(trade.get("entry_price", 0))
            direction = int(trade.get("direction", 0))
            position_size = float(trade.get("position_size_usd", 0))

            try:
                ticker = self.binance.get_ticker(symbol)
                if ticker and ticker.get("last_price"):
                    current_price = float(ticker["last_price"])
                    price_frac = (
                        (current_price - entry_price)
                        / entry_price * direction
                    ) if entry_price > 0 else 0.0

                    unrealized = position_size * price_frac

                    pos["current_price"] = round(current_price, 2)
                    pos["unrealized_pnl_usd"] = round(unrealized, 2)
                    pos["unrealized_pnl_pct"] = round(price_frac * 100, 4)

                    # Position state info
                    state = self._position_state.get(trade.get("id"), {})
                    pos["current_stop"] = round(
                        state.get("current_stop", 0), 2
                    )
                    pos["highest_price"] = round(
                        state.get("highest_price", 0), 2
                    )
                    pos["lowest_price"] = round(
                        state.get("lowest_price", 0), 2
                    )

            except Exception as e:
                pos["price_error"] = str(e)

            positions.append(pos)

        return positions

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get closed paper trades (most recent first)."""
        return self.db.get_trades(
            mode=self.mode, status="closed", limit=limit
        )

    def get_performance(self) -> Dict:
        """
        Get paper trading performance summary.

        Combines DB stats with current capital tracking.
        """
        try:
            capital = self.risk_mgr.get_capital()
            total_pnl = capital - self.initial_capital

            stats = self.db.get_stats(mode=self.mode)
            risk = self.risk_mgr.get_risk_summary()

            return {
                "capital": round(capital, 2),
                "initial_capital": round(self.initial_capital, 2),
                "total_pnl_usd": round(total_pnl, 2),
                "total_return_pct": round(
                    total_pnl / self.initial_capital * 100.0
                    if self.initial_capital > 0 else 0.0, 2
                ),
                "db_stats": stats,
                "risk_summary": risk,
                "cycles_completed": self._cycles,
            }

        except Exception as e:
            logger.error(f"Performance query error: {e}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█" * 60)
    print("  PAPER TRADER — TEST")
    print("█" * 60)

    t_start = time.time()

    # ── 1. Create paper trader (AI off for faster test) ──
    pt = PaperTrader(capital=10000.0, include_ai=False)
    print(f"\n  ✓ PaperTrader created | capital=${pt.risk_mgr.get_capital():,.2f}")

    # ── 2. Show initial status ──
    status = pt.get_status()
    print(f"\n  STATUS:")
    print(f"    Capital:          ${status.get('capital', 0):,.2f}")
    print(f"    Total PnL:        ${status.get('total_pnl', 0):+,.2f}")
    print(f"    Return:           {status.get('total_return_pct', 0):+.2f}%")
    print(f"    Open positions:   {status.get('open_positions', 0)}")
    print(f"    Trading pairs:    {status.get('trading_pairs', [])}")
    print(f"    AI:               {'ON' if status.get('include_ai') else 'OFF'}")

    # ── 3. Show existing open positions ──
    open_pos = pt.get_open_positions()
    if open_pos:
        print(f"\n  OPEN POSITIONS ({len(open_pos)}):")
        for p in open_pos:
            sym = p.get("symbol", "?")
            sig = p.get("signal_type", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", 0)
            unreal = p.get("unrealized_pnl_usd", 0)
            pct = p.get("unrealized_pnl_pct", 0)
            print(
                f"    {sym} {sig} | "
                f"entry=${entry:,.2f} now=${current:,.2f} | "
                f"PnL=${unreal:+,.2f} ({pct:+.2f}%)"
            )
    else:
        print("\n  No open positions.")

    # ── 4. Run one cycle ──
    print(
        "\n  Running one paper trading cycle…\n"
        "  (First run may train models — can take a few minutes)\n"
    )

    try:
        cycle_result = pt.run_cycle()

        print(f"\n  CYCLE RESULT:")
        print(f"    Signals generated: {cycle_result.get('signals_generated', 0)}")
        print(f"    Positions opened:  {cycle_result.get('positions_opened', 0)}")
        print(f"    Positions closed:  {cycle_result.get('positions_closed', 0)}")
        print(f"    Capital:           ${cycle_result.get('capital', 0):,.2f}")
        print(f"    Cycle time:        {cycle_result.get('cycle_time_s', 0):.1f}s")
        if cycle_result.get("errors"):
            print(f"    Errors:            {len(cycle_result['errors'])}")
            for err in cycle_result["errors"][:3]:
                print(f"      - {err[:80]}")

    except Exception as e:
        print(f"\n  ✗ Cycle failed: {e}")

    # ── 5. Show positions after cycle ──
    open_pos = pt.get_open_positions()
    if open_pos:
        print(f"\n  OPEN POSITIONS AFTER CYCLE ({len(open_pos)}):")
        for p in open_pos:
            sym = p.get("symbol", "?")
            sig = p.get("signal_type", "?")
            entry = p.get("entry_price", 0)
            current = p.get("current_price", 0)
            unreal = p.get("unrealized_pnl_usd", 0)
            sl = p.get("stop_loss", 0)
            tp = p.get("take_profit", 0)
            print(
                f"    {sym} {sig} | "
                f"entry=${entry:,.2f} now=${current:,.2f} | "
                f"PnL=${unreal:+,.2f} | "
                f"SL={sl:.2f} TP={tp:.2f}"
            )

    # ── 6. Show trade history ──
    history = pt.get_trade_history(limit=10)
    if history:
        print(f"\n  RECENT CLOSED TRADES ({len(history)}):")
        for t in history[:5]:
            sym = t.get("symbol", "?")
            sig = t.get("signal_type", "?")
            pnl = t.get("pnl_usd", 0)
            reason = t.get("exit_reason", "?")
            emoji = "✅" if pnl > 0 else "❌"
            print(
                f"    {emoji} {sym} {sig} | "
                f"PnL=${pnl:+,.2f} | {reason}"
            )

    # ── 7. Performance summary ──
    perf = pt.get_performance()
    if "error" not in perf:
        print(f"\n  PERFORMANCE:")
        print(f"    Capital:     ${perf.get('capital', 0):,.2f}")
        print(f"    Total PnL:   ${perf.get('total_pnl_usd', 0):+,.2f}")
        print(f"    Return:      {perf.get('total_return_pct', 0):+.2f}%")
        db_stats = perf.get("db_stats", {})
        if db_stats:
            print(f"    Win rate:    {db_stats.get('win_rate', 0):.1f}%")
            print(f"    Total PnL:   ${db_stats.get('pnl', 0):+,.2f}")

    # ── 8. Final status ──
    final = pt.get_status()
    print(f"\n  FINAL STATUS:")
    print(f"    Capital:        ${final.get('capital', 0):,.2f}")
    print(f"    Open positions: {final.get('open_positions', 0)}")
    print(f"    Cycles:         {final.get('cycles_completed', 0)}")

    elapsed = time.time() - t_start
    print(f"\n  Total test time: {elapsed:.1f}s")

    print("\n" + "█" * 60)
    print("  PAPER TRADER TEST COMPLETE ✅")
    print("█" * 60 + "\n")