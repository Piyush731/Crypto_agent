"""
Crypto Futures AI Agent - Paper Trader
Real-time paper trading with simulated execution.

Uses the full signal pipeline:
  SignalEngine (ML ensemble + sentiment + AI reasoning)
  → RiskManager (position sizing, SL/TP, circuit breakers)
  → Simulated execution (logged to SQLite)

FIX LOG (v3.0.1):
  - _open_position: now stores position_size_usd AND quantity in DB
  - _close_position: handles legacy trades with position_size_usd=0
  - Direction display: stores signal_type ("LONG"/"SHORT") for display
"""

import time
import json
import signal as _signal
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    BACKTEST_CONFIG,
    RISK_CONFIG,
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
        result = pt.run_cycle()       # single cycle
        pt.start(max_cycles=24)       # continuous
    """

    MODEL_CHECK_INTERVAL = 12

    def __init__(self, capital: float = None, include_ai: bool = None):
        self.initial_capital = capital or BACKTEST_CONFIG["initial_capital"]
        self.mode = "paper"
        self.include_ai = (
            include_ai if include_ai is not None
            else AI_CONFIG.get("enabled", False)
        )

        self.risk_mgr = RiskManager(capital=self.initial_capital, mode=self.mode)
        self.db = get_db()

        self._signal_engine = None
        self._binance = None
        self._trainer = None

        self._position_state: Dict[int, Dict] = {}
        self._running = False
        self._cycles = 0
        self._last_cycle_time: Optional[datetime] = None

        self._commission_pct = BACKTEST_CONFIG.get("commission_pct", 0.04)
        self._slippage_pct = BACKTEST_CONFIG.get("slippage_pct", 0.02)

        self._load_capital()
        self._init_position_states()

        logger.info(
            f"PaperTrader initialized | capital=${self.risk_mgr.get_capital():,.2f} | "
            f"mode={self.mode} | AI={'ON' if self.include_ai else 'OFF'} | "
            f"open_positions={len(self._position_state)}"
        )

    # ═══════════════════════════════════════════════════════════
    #  LAZY PROPERTIES
    # ═══════════════════════════════════════════════════════════

    @property
    def signal_engine(self):
        if self._signal_engine is None:
            from analysis.signal_engine import SignalEngine
            self._signal_engine = SignalEngine()
        return self._signal_engine

    @property
    def binance(self):
        if self._binance is None:
            from data.binance_data import BinanceData
            self._binance = BinanceData()
        return self._binance

    @property
    def trainer(self):
        if self._trainer is None:
            from models.trainer import ModelTrainer
            self._trainer = ModelTrainer()
        return self._trainer

    # ═══════════════════════════════════════════════════════════
    #  STARTUP: load state from DB
    # ═══════════════════════════════════════════════════════════

    def _load_capital(self):
        """
        Reconstruct current capital from DB.
        FIX: get_total_pnl now returns USD (was returning pnl_percent).
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
        """Initialize position tracking for existing open trades."""
        try:
            open_trades = self.db.get_open_trades(self.mode)
            if not open_trades:
                return

            for trade in open_trades:
                trade_id = trade.get("id")
                entry_price = float(trade.get("entry_price", 0))
                stop_loss = float(trade.get("stop_loss", entry_price))
                symbol = trade.get("symbol", "")

                highest = entry_price
                lowest = entry_price
                current_stop = stop_loss

                # Load saved state from DB notes (survives restarts)
                notes_str = trade.get("notes", "")
                if notes_str:
                    try:
                        saved = json.loads(notes_str)
                        if isinstance(saved, dict):
                            highest = float(saved.get("highest_price", highest))
                            lowest = float(saved.get("lowest_price", lowest))
                            current_stop = float(saved.get("current_stop", stop_loss))
                            logger.info(
                                f"  [{symbol}] Restored state: "
                                f"stop={current_stop:.2f} high={highest:.2f} low={lowest:.2f}"
                            )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                # Also update with current live price
                try:
                    ticker = self.binance.get_ticker(symbol)
                    if ticker and ticker.get("last_price"):
                        current = float(ticker["last_price"])
                        highest = max(highest, current)
                        lowest = min(lowest, current)
                except Exception:
                    pass

                self._position_state[trade_id] = {
                    "highest_price": highest,
                    "lowest_price": lowest,
                    "current_stop": current_stop,
                }

            logger.info(
                f"Initialized {len(self._position_state)} open position states"
            )
        except Exception as e:
            logger.warning(f"Could not initialize position states: {e}")

    # ═══════════════════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════════════════

    # def start(self, max_cycles: int = None):
    #     """Main continuous trading loop."""
    #     self._running = True
    #     interval_s = SCHEDULE_CONFIG["analysis_interval_minutes"] * 60

    #     logger.info(
    #         f"{'═'*50}\n"
    #         f"  PAPER TRADER STARTING\n"
    #         f"  Capital: ${self.risk_mgr.get_capital():,.2f}\n"
    #         f"  Pairs: {TRADING_PAIRS}\n"
    #         f"  Interval: {interval_s // 60} min\n"
    #         f"  AI: {'ON' if self.include_ai else 'OFF'}\n"
    #         f"  Max cycles: {max_cycles or '∞'}\n"
    #         f"{'═'*50}"
    #     )

    #     cycle = 0
    #     while self._running:
    #         if max_cycles is not None and cycle >= max_cycles:
    #             logger.info(f"Reached max_cycles={max_cycles} — stopping")
    #             break

    #         try:
    #             self.run_cycle()
    #             cycle += 1

    #             if self._running and (max_cycles is None or cycle < max_cycles):
    #                 logger.info(
    #                     f"Next cycle in {interval_s // 60} min "
    #                     f"(cycle {cycle}/{max_cycles or '∞'})…"
    #                 )
    #                 for _ in range(interval_s):
    #                     if not self._running:
    #                         break
    #                     time.sleep(1)

    #         except KeyboardInterrupt:
    #             logger.info("Paper trader interrupted (Ctrl+C)")
    #             self._running = False

    #         except Exception as e:
    #             logger.error(f"Cycle error: {e}", exc_info=True)
    #             self.db.save_error("paper_trader", "cycle_error", str(e))
    #             for _ in range(60):
    #                 if not self._running:
    #                     break
    #                 time.sleep(1)

    #     self._running = False
    #     logger.info(
    #         f"Paper trader stopped after {cycle} cycles | "
    #         f"capital=${self.risk_mgr.get_capital():,.2f}"
    #     )
    def start(self, max_cycles: int = None):
        """Main continuous trading loop with crash recovery."""
        self._running = True
        interval_s = SCHEDULE_CONFIG["analysis_interval_minutes"] * 60

        # Register signal handlers for graceful shutdown
        def _handle_signal(signum, frame):
            logger.info(f"Signal {signum} received — shutting down gracefully")
            self._running = False

        try:
            _signal.signal(_signal.SIGTERM, _handle_signal)
            _signal.signal(_signal.SIGINT, _handle_signal)
        except Exception:
            pass  # May fail in non-main thread

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
        try:
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

                    # Notify via Telegram
                    try:
                        from notifications.telegram import get_notifier
                        get_notifier().send_error("paper_trader", "cycle_error", str(e))
                    except Exception:
                        pass

                    # Wait 60s before retrying (not full interval)
                    for _ in range(60):
                        if not self._running:
                            break
                        time.sleep(1)

        except Exception as fatal:
            logger.critical(f"FATAL paper trader crash: {fatal}", exc_info=True)
            self.db.save_error("paper_trader", "fatal_crash", str(fatal))
            try:
                from notifications.telegram import get_notifier
                get_notifier().send_error("paper_trader", "FATAL_CRASH", str(fatal))
            except Exception:
                pass

        finally:
            self._running = False
            logger.info(
                f"Paper trader stopped after {cycle} cycles | "
                f"capital=${self.risk_mgr.get_capital():,.2f}"
            )
   
    def stop(self):
        """Request graceful shutdown."""
        self._running = False
        logger.info("Paper trader stop requested")

    # ═══════════════════════════════════════════════════════════
    #  SINGLE CYCLE
    # ═══════════════════════════════════════════════════════════

    def run_cycle(self) -> Dict:
        """Execute one full analysis cycle."""
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

        # 1. Check open positions
        try:
            closed = self._check_open_positions()
            open_count = len(self.db.get_open_trades(self.mode))
            result["positions_checked"] = open_count + len(closed)
            result["positions_closed"] = len(closed)
        except Exception as e:
            logger.error(f"Position check error: {e}", exc_info=True)
            result["errors"].append(f"position_check: {e}")

        # 2. Ensure models trained (periodic)
        if self._cycles == 1 or self._cycles % self.MODEL_CHECK_INTERVAL == 0:
            try:
                self._ensure_models_ready()
            except Exception as e:
                logger.error(f"Model check error: {e}", exc_info=True)
                result["errors"].append(f"model_check: {e}")

        # 3. Generate signals for pairs without open positions
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

        # 4. Summary
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

    # ═══════════════════════════════════════════════════════════
    #  CHECK OPEN POSITIONS
    # ═══════════════════════════════════════════════════════════

    def _check_open_positions(self) -> List[Dict]:
        """Check all open positions for exit conditions."""
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

                ticker = self.binance.get_ticker(symbol)
                if not ticker or not ticker.get("last_price"):
                    logger.warning(f"[{symbol}] Cannot get price — skip exit check")
                    continue

                current_price = float(ticker["last_price"])

                state = self._position_state.get(trade_id, {
                    "highest_price": max(entry_price, current_price),
                    "lowest_price": min(entry_price, current_price),
                    "current_stop": stop_loss,
                })
                state["highest_price"] = max(state["highest_price"], current_price)
                state["lowest_price"] = min(state["lowest_price"], current_price)
                self._position_state[trade_id] = state

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

                state["current_stop"] = exit_check.get(
                    "new_trailing_stop", state["current_stop"]
                )

                try:
                    self.db.update_trade(trade_id, {
                        "notes": json.dumps({
                            "current_stop": round(state["current_stop"], 8),
                            "highest_price": round(state["highest_price"], 8),
                            "lowest_price": round(state["lowest_price"], 8),
                        })
                    })
                except Exception:
                    pass

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

    # ═══════════════════════════════════════════════════════════
    #  PROCESS ONE SYMBOL
    # ═══════════════════════════════════════════════════════════

    def _process_symbol(self, symbol: str) -> Dict:
        """Generate signal and open position if approved."""
        result = {"symbol": symbol, "signal": "HOLD", "opened": False}

        try:
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
                logger.info(f"[{symbol}] HOLD | conf={confidence:.3f}")
                return result

            logger.info(
                f"[{symbol}] Signal: {sig_type} | "
                f"conf={confidence:.3f} | "
                f"score={signal.get('combined_score', 0):+.3f}"
            )

            eval_result = self.risk_mgr.evaluate_signal(signal)

            if not eval_result["approved"]:
                result["rejected_reason"] = eval_result["reason"]
                logger.info(f"[{symbol}] Rejected: {eval_result['reason']}")
                return result

            trade = self._open_position(symbol, signal, eval_result)
            if trade.get("trade_id"):
                result["opened"] = True
                result["trade_id"] = trade["trade_id"]

            return result

        except Exception as e:
            logger.error(f"[{symbol}] Process error: {e}", exc_info=True)
            result["error"] = str(e)
            return result

    # ═══════════════════════════════════════════════════════════
    #  OPEN POSITION — FIX: stores position_size_usd + quantity
    # ═══════════════════════════════════════════════════════════

    def _open_position(
        self, symbol: str, signal: Dict, risk_eval: Dict
    ) -> Dict:
        """Open a simulated paper position."""
        try:
            # Current price
            ticker = self.binance.get_ticker(symbol)
            if ticker and ticker.get("last_price"):
                market_price = float(ticker["last_price"])
            else:
                market_price = float(signal.get("entry_price", 0))

            if market_price <= 0:
                logger.error(f"[{symbol}] Invalid entry price: {market_price}")
                return {"error": "invalid price"}

            # Apply slippage
            direction = int(signal["direction"])
            slip = market_price * (self._slippage_pct / 100.0)
            entry_price = (
                market_price + slip if direction == 1
                else market_price - slip
            )

            # ── FIX: get position size and compute quantity ──
            position_size_usd = float(risk_eval.get("position_size_usd", 0))
            leverage = int(risk_eval.get("leverage", 1))

            # Quantity = how many units of the asset
            quantity = (
                position_size_usd / entry_price
                if entry_price > 0 else 0.0
            )

            if position_size_usd <= 0:
                logger.error(
                    f"[{symbol}] Risk manager returned position_size_usd=0! "
                    f"Check risk_manager.evaluate_signal()"
                )
                return {"error": "zero position size"}

            now = datetime.now(timezone.utc)
            sig_label = "LONG" if direction == 1 else "SHORT"

            # ── Save to DB — all fields now stored correctly ──
            trade_data = {
                "symbol": symbol,
                "mode": self.mode,
                "signal_type": sig_label,
                "direction": direction,
                "entry_price": round(entry_price, 8),
                "quantity": round(quantity, 8),
                "position_size_usd": round(position_size_usd, 2),
                "leverage": leverage,
                "stop_loss": risk_eval["stop_loss"],
                "take_profit": risk_eval["take_profit"],
                "confidence": round(signal.get("confidence", 0), 4),
                "margin_used": round(risk_eval.get("margin_required", 0), 2),
                "entry_time": now.isoformat(),
                "status": "open",
                "signal_id": signal.get("signal_id"),
            }

            trade_id = self.db.save_trade(trade_data)
            trade_data["trade_id"] = trade_id

            # Initialize position state
            self._position_state[trade_id] = {
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "current_stop": risk_eval["stop_loss"],
            }

            # ── Telegram notification ──
            try:
                from notifications.telegram import get_notifier
                tg = get_notifier()
                tg.send_trade_opened(trade_data)
            except Exception:
                pass

            logger.info(
                f"📈 OPENED [{symbol}] {sig_label} | "
                f"entry=${entry_price:,.2f} | "
                f"size=${position_size_usd:,.2f} | "
                f"qty={quantity:.6f} | "
                f"lev={leverage}x | "
                f"SL={risk_eval['stop_loss']:.2f} | "
                f"TP={risk_eval['take_profit']:.2f} | "
                f"id={trade_id}"
            )

            return trade_data

        except Exception as e:
            logger.error(f"Open position error [{symbol}]: {e}", exc_info=True)
            return {"error": str(e)}

    # ═══════════════════════════════════════════════════════════
    #  CLOSE POSITION — FIX: handles legacy trades with size=0
    # ═══════════════════════════════════════════════════════════

    def _close_position(
        self,
        trade: Dict,
        current_price: float,
        reason: str,
        exit_price: float = None,
    ) -> Dict:
        """Close a simulated paper position with correct PnL."""
        try:
            trade_id = trade["id"]
            symbol = trade.get("symbol", "?")
            direction = int(trade.get("direction", 0))
            entry_price = float(trade.get("entry_price", 0))
            leverage = int(trade.get("leverage", 1))

            if exit_price is None:
                exit_price = current_price

            # ── FIX: read position_size_usd, handle legacy trades ──
            position_size = float(trade.get("position_size_usd", 0) or 0)

            if position_size <= 0:
                # Legacy trade that was created before the fix.
                # Estimate position size from SL distance + risk config.
                stop_loss = float(trade.get("stop_loss", 0) or 0)
                if entry_price > 0 and stop_loss > 0:
                    sl_distance_pct = abs(entry_price - stop_loss) / entry_price
                    if sl_distance_pct > 0:
                        capital = self.risk_mgr.get_capital()
                        risk_pct = RISK_CONFIG.get("risk_per_trade_pct", 2.0)
                        risk_amount = capital * risk_pct / 100.0
                        position_size = risk_amount / sl_distance_pct
                        max_size = capital * RISK_CONFIG.get("max_position_size_pct", 20.0) / 100.0
                        position_size = min(position_size, max_size)
                else:
                    # Last resort: 5% of capital
                    position_size = self.risk_mgr.get_capital() * 0.05

                logger.warning(
                    f"Trade #{trade_id}: legacy trade had position_size_usd=0, "
                    f"estimated ${position_size:,.2f}"
                )

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
            pnl_pct = (net_pnl / capital * 100.0) if capital > 0 else 0.0

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

            # ── Telegram notification ──
            sig_label = "LONG" if direction == 1 else "SHORT"
            try:
                from notifications.telegram import get_notifier
                tg = get_notifier()
                tg.send_trade_closed({
                    "symbol": symbol,
                    "direction": sig_label,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "pnl_usd": round(net_pnl, 4),
                    "pnl_percent": round(pnl_pct, 4),
                    "commission": round(commission, 4),
                    "position_size_usd": round(position_size, 2),
                })
            except Exception:
                pass

            emoji = "✅" if is_win else "❌"
            logger.info(
                f"{emoji} CLOSED [{symbol}] {sig_label} {reason} | "
                f"entry=${entry_price:,.2f} → exit=${exit_price:,.2f} | "
                f"size=${position_size:,.2f} | "
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
                "position_size_usd": round(position_size, 2),
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

    # ═══════════════════════════════════════════════════════════
    #  MODEL MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def _ensure_models_ready(self):
        """Check all pairs for model staleness and retrain if needed."""
        logger.info("Checking model freshness…")
        for symbol in TRADING_PAIRS:
            try:
                result = self.trainer.retrain_if_needed(symbol)
                if result.get("retrained"):
                    logger.info(f"[{symbol}] Model retrained: {result.get('reason')}")
                elif result.get("error"):
                    logger.warning(f"[{symbol}] Retrain failed: {result.get('error')}")
                else:
                    logger.debug(f"[{symbol}] Model OK: {result.get('reason', 'fresh')}")
            except Exception as e:
                logger.warning(f"[{symbol}] Model check failed: {e}")

    # ═══════════════════════════════════════════════════════════
    #  FORCE CLOSE ALL
    # ═══════════════════════════════════════════════════════════

    def force_close_all(self) -> List[Dict]:
        """Force-close all open paper positions at current market price."""
        open_trades = self.db.get_open_trades(self.mode)
        if not open_trades:
            logger.info("No open positions to close")
            return []

        logger.warning(f"FORCE CLOSING {len(open_trades)} open position(s)…")
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
                logger.error(f"Force close error (trade {trade.get('id')}): {e}")
                results.append({"error": str(e), "trade_id": trade.get("id")})

        total_pnl = sum(r.get("net_pnl", 0) for r in results if "net_pnl" in r)
        logger.warning(
            f"Force close complete: {len(results)} positions | "
            f"total PnL=${total_pnl:+,.2f}"
        )
        return results

    # ═══════════════════════════════════════════════════════════
    #  STATUS / QUERIES
    # ═══════════════════════════════════════════════════════════

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
                "cycles": self._cycles,
                "cycles_completed": self._cycles,
                "last_cycle": (
                    self._last_cycle_time.isoformat()
                    if self._last_cycle_time else None
                ),
                "include_ai": self.include_ai,
                "capital": round(capital, 2),
                "initial_capital": round(self.initial_capital, 2),
                "pnl": round(total_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_ret, 2),
                "open_positions": len(open_trades),
                "open_symbols": [t.get("symbol") for t in open_trades],
                "trading_pairs": TRADING_PAIRS,
                "analysis_interval_min": SCHEDULE_CONFIG["analysis_interval_minutes"],
            }
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {"error": str(e)}

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions with live unrealized P&L."""
        open_trades = self.db.get_open_trades(self.mode)
        positions = []

        for trade in open_trades:
            pos = dict(trade)
            symbol = trade.get("symbol", "")
            entry_price = float(trade.get("entry_price", 0))
            direction = int(trade.get("direction", 0))
            position_size = float(trade.get("position_size_usd", 0) or 0)

            # Direction label for display
            pos["direction_label"] = "LONG" if direction == 1 else "SHORT"

            try:
                ticker = self.binance.get_ticker(symbol)
                if ticker and ticker.get("last_price"):
                    current_price = float(ticker["last_price"])
                    price_frac = (
                        (current_price - entry_price) / entry_price * direction
                    ) if entry_price > 0 else 0.0

                    unrealized = position_size * price_frac

                    pos["current_price"] = round(current_price, 2)
                    pos["unrealized_pnl"] = round(unrealized, 2)
                    pos["unrealized_pnl_usd"] = round(unrealized, 2)
                    pos["unrealized_pnl_pct"] = round(price_frac * 100, 4)

                    state = self._position_state.get(trade.get("id"), {})
                    pos["current_stop"] = round(state.get("current_stop", 0), 2)
                    pos["highest_price"] = round(state.get("highest_price", 0), 2)
                    pos["lowest_price"] = round(state.get("lowest_price", 0), 2)
            except Exception as e:
                pos["price_error"] = str(e)

            positions.append(pos)

        return positions

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get closed paper trades (most recent first)."""
        return self.db.get_trades(mode=self.mode, status="closed", limit=limit)

    def get_performance(self) -> Dict:
        """Paper trading performance summary."""
        try:
            capital = self.risk_mgr.get_capital()
            total_pnl = capital - self.initial_capital
            stats = self.db.get_stats(mode=self.mode)
            risk = self.risk_mgr.get_risk_summary()
            return {
                "capital": round(capital, 2),
                "initial_capital": round(self.initial_capital, 2),
                "pnl": round(total_pnl, 2),
                "total_pnl_usd": round(total_pnl, 2),
                "return_pct": round(
                    total_pnl / self.initial_capital * 100.0
                    if self.initial_capital > 0 else 0.0, 2
                ),
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


# ═══════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█" * 60)
    print("  PAPER TRADER — TEST")
    print("█" * 60)

    t_start = time.time()

    pt = PaperTrader(capital=10000.0, include_ai=False)
    print(f"\n  ✓ PaperTrader created | capital=${pt.risk_mgr.get_capital():,.2f}")

    status = pt.get_status()
    print(f"\n  STATUS:")
    print(f"    Capital:        ${status.get('capital', 0):,.2f}")
    print(f"    Total PnL:      ${status.get('pnl', 0):+,.2f}")
    print(f"    Open positions: {status.get('open_positions', 0)}")
    print(f"    AI:             {'ON' if status.get('include_ai') else 'OFF'}")

    open_pos = pt.get_open_positions()
    if open_pos:
        print(f"\n  OPEN POSITIONS ({len(open_pos)}):")
        for p in open_pos:
            sym = p.get("symbol", "?")
            dl = p.get("direction_label", "?")
            entry = p.get("entry_price", 0)
            size = p.get("position_size_usd", 0)
            current = p.get("current_price", 0)
            unreal = p.get("unrealized_pnl_usd", 0)
            print(
                f"    {sym} {dl} | entry=${entry:,.2f} "
                f"size=${size:,.2f} now=${current:,.2f} | "
                f"PnL=${unreal:+,.2f}"
            )
    else:
        print("\n  No open positions.")

    print(f"\n  Running one paper trading cycle…")
    try:
        cycle_result = pt.run_cycle()
        print(f"\n  CYCLE RESULT:")
        print(f"    Signals generated: {cycle_result.get('signals_generated', 0)}")
        print(f"    Positions opened:  {cycle_result.get('positions_opened', 0)}")
        print(f"    Positions closed:  {cycle_result.get('positions_closed', 0)}")
        print(f"    Capital:           ${cycle_result.get('capital', 0):,.2f}")
        print(f"    Cycle time:        {cycle_result.get('cycle_time_s', 0):.1f}s")
    except Exception as e:
        print(f"\n  ✗ Cycle failed: {e}")

    elapsed = time.time() - t_start
    print(f"\n  Total test time: {elapsed:.1f}s")
    print("\n" + "█" * 60)
    print("  PAPER TRADER TEST COMPLETE ✅")
    print("█" * 60 + "\n")