"""
Crypto Futures AI Agent - Risk Manager
=======================================
Position sizing, SL/TP calculation, circuit breakers, signal approval/rejection.
Can APPROVE or REJECT signals based on configurable risk rules.

Used by: backtester.py, paper_trader.py, live_trader.py

Key responsibilities:
  - Evaluate signals → approve or reject with reason
  - Position sizing (fixed-fraction, max 2% risk per trade)
  - Stop-loss / take-profit (ATR-based or fixed %)
  - Trailing stop management
  - Circuit breakers (daily/weekly loss, drawdown, consecutive losses)
  - Dynamic leverage (confidence + volatility based)
  - Bar-level exit checking for backtester
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from config import (
    RISK_CONFIG,
    BACKTEST_CONFIG,
    AGENT_MODE,
)
from core.logger import get_logger
from core.db import get_db

logger = get_logger(__name__)


class RiskManager:
    """
    Risk management engine for crypto futures trading.

    Tracks capital, drawdown, consecutive losses internally.
    Uses SQLite DB for paper/live mode circuit breakers.
    Uses internal state for backtest mode circuit breakers.
    """

    def __init__(self, capital: float = None, mode: str = None):
        """
        Initialize RiskManager.

        Args:
            capital: Starting capital in USDT. Default: BACKTEST_CONFIG['initial_capital']
            mode:    Trading mode — 'backtest' | 'paper' | 'live'. Default: AGENT_MODE
        """
        self.initial_capital = capital or BACKTEST_CONFIG["initial_capital"]
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.mode = mode or AGENT_MODE

        # ── Risk settings from config ──
        self.max_leverage = RISK_CONFIG["max_leverage"]
        self.risk_per_trade_pct = RISK_CONFIG["risk_per_trade_pct"]
        self.max_position_size_pct = RISK_CONFIG["max_position_size_pct"]
        self.max_open_positions = RISK_CONFIG["max_open_positions"]

        self.default_sl_pct = RISK_CONFIG["default_stop_loss_pct"]
        self.default_tp_pct = RISK_CONFIG["default_take_profit_pct"]
        self.use_atr_stops = RISK_CONFIG["use_atr_stops"]
        self.atr_sl_multiplier = RISK_CONFIG["atr_stop_multiplier"]
        self.trailing_stop_pct = RISK_CONFIG["trailing_stop_pct"]

        #14th after gains updates below - gain locks

        self._trailing_tiers = sorted(
            RISK_CONFIG.get("trailing_stop_tiers", []),
            key=lambda t: t["min_profit_pct"],
            reverse=True,
        )

        self.max_daily_loss_pct = RISK_CONFIG["max_daily_loss_pct"]
        self.max_weekly_loss_pct = RISK_CONFIG["max_weekly_loss_pct"]
        self.max_total_dd_pct = RISK_CONFIG["max_total_drawdown_pct"]
        self.cooldown_minutes = RISK_CONFIG["cooldown_after_loss_minutes"]
        self.max_consecutive_losses = RISK_CONFIG["max_consecutive_losses"]

        self.min_confidence = RISK_CONFIG["min_confidence_to_trade"]
        self.min_agreement = RISK_CONFIG["min_agreement_to_trade"]

        # ── Commission from backtest config ──
        self.commission_pct = BACKTEST_CONFIG.get("commission_pct", 0.04)
        self.slippage_pct = BACKTEST_CONFIG.get("slippage_pct", 0.02)

        # ── Internal state tracking (used esp. in backtest mode) ──
        self._trade_results: List[Dict] = []   # {timestamp, pnl_usd, is_win}
        self._consecutive_losses = 0
        self._open_positions = 0
        self._sim_time: Optional[datetime] = None  # Set by backtester
        self.last_trade_time: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None

        # ── Counters ──
        self.signals_evaluated = 0
        self.signals_approved = 0
        self.signals_rejected = 0

        # ── Database ──
        self.db = get_db()

        logger.info(
            f"RiskManager initialized | capital=${self.initial_capital:,.2f} | "
            f"mode={self.mode} | max_leverage={self.max_leverage}x | "
            f"risk/trade={self.risk_per_trade_pct}% | "
            f"SL={self.default_sl_pct}% | TP={self.default_tp_pct}%"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN ENTRY: evaluate_signal
    # ═══════════════════════════════════════════════════════════════════

    def evaluate_signal(self, signal: Dict) -> Dict:
        """
        Evaluate a trading signal — approve or reject based on risk rules.

        Runs 8 sequential checks. First failure → rejection.

        Args:
            signal: Signal dict from SignalEngine.generate_signal()
                Expected keys: symbol, signal, direction, confidence,
                               entry_price, stop_loss, take_profit,
                               atr, atr_pct, agreement (optional)

        Returns:
            Dict:
                approved: bool
                reason: str
                position_size_usd: float      (notional value)
                position_size_units: float     (base asset qty)
                margin_required: float         (USDT margin)
                leverage: int
                stop_loss: float
                take_profit: float
                risk_amount_usd: float
                reward_amount_usd: float
                risk_reward_ratio: float
                risk_pct: float                (% of capital at risk)
                checks: Dict                   (individual check results)
        """
        self.signals_evaluated += 1

        result = {
            "approved": False,
            "reason": "",
            "position_size_usd": 0.0,
            "position_size_units": 0.0,
            "margin_required": 0.0,
            "leverage": 1,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_amount_usd": 0.0,
            "reward_amount_usd": 0.0,
            "risk_reward_ratio": 0.0,
            "risk_pct": 0.0,
            "checks": {},
        }

        try:
            # ── Check 0: Signal is actionable ──
            sig_type = signal.get("signal", "HOLD")
            if sig_type == "HOLD" or signal.get("direction", 0) == 0:
                result["reason"] = "Signal is HOLD — no trade"
                self.signals_rejected += 1
                return result

            symbol = signal.get("symbol", "UNKNOWN")
            direction = signal.get("direction", 0)
            confidence = signal.get("confidence", 0.0)
            # Use ML model agreement (3-4 models) not component agreement (4-5 components)
            # Components SHOULD disagree — that's why they have different weights
            ml_comp = signal.get("components", {}).get("ml_ensemble", {})
            if ml_comp.get("available") and ml_comp.get("agreement") is not None:
                agreement = float(ml_comp["agreement"])
            else:
                agreement = signal.get("agreement", 1.0)
            entry_price = signal.get("entry_price", 0.0)

            if entry_price <= 0:
                result["reason"] = f"Invalid entry price: {entry_price}"
                self.signals_rejected += 1
                return result

            # ── Check 1: Confidence & agreement ──
            chk = self._check_confidence(confidence, agreement)
            result["checks"]["confidence"] = chk
            if not chk["passed"]:
                result["reason"] = chk["reason"]
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: {chk['reason']}")
                return result

            # ── Check 2: Circuit breakers ──
            chk = self.check_circuit_breakers()
            result["checks"]["circuit_breakers"] = chk
            if not chk["can_trade"]:
                reasons = ", ".join(chk["active_breakers"])
                result["reason"] = f"Circuit breaker active: {reasons}"
                self.signals_rejected += 1
                logger.warning(f"REJECTED [{symbol}]: Circuit breaker — {reasons}")
                return result

            # ── Check 3: Max open positions ──
            chk = self._check_max_positions()
            result["checks"]["max_positions"] = chk
            if not chk["passed"]:
                result["reason"] = chk["reason"]
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: {chk['reason']}")
                return result

            # ── Check 4: Cooldown after loss ──
            chk = self._check_cooldown()
            result["checks"]["cooldown"] = chk
            if not chk["passed"]:
                result["reason"] = chk["reason"]
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: {chk['reason']}")
                return result

            # ── Calculate SL / TP ──
            atr = signal.get("atr")
            atr_pct = signal.get("atr_pct")

            sl_info = self.calculate_stop_loss(entry_price, direction, atr=atr)
            tp_info = self.calculate_take_profit(
                entry_price, direction, sl_info["stop_loss"], atr=atr
            )

            stop_loss = sl_info["stop_loss"]
            take_profit = tp_info["take_profit"]

            # Prefer signal's SL/TP if provided and valid
            if signal.get("stop_loss") and signal["stop_loss"] > 0:
                stop_loss = signal["stop_loss"]
            if signal.get("take_profit") and signal["take_profit"] > 0:
                take_profit = signal["take_profit"]

            # ── Check 5: Validate SL / TP ──
            chk = self._validate_sl_tp(entry_price, stop_loss, take_profit, direction)
            result["checks"]["sl_tp_valid"] = chk
            if not chk["passed"]:
                result["reason"] = chk["reason"]
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: {chk['reason']}")
                return result

            # ── Calculate leverage ──
            vol = atr_pct if atr_pct else self.default_sl_pct
            leverage = self.calculate_leverage(confidence, vol)

            # ── Calculate position size ──
            pos = self.calculate_position_size(
                self.current_capital, entry_price, stop_loss, leverage
            )

            # ── Check 6: Position size valid ──
            if pos["position_size_usd"] <= 0:
                result["reason"] = "Calculated position size is zero"
                self.signals_rejected += 1
                return result

            # ── Risk / reward amounts ──
            sl_dist_frac = abs(entry_price - stop_loss) / entry_price
            tp_dist_frac = abs(take_profit - entry_price) / entry_price

            risk_amount = pos["position_size_usd"] * sl_dist_frac
            reward_amount = pos["position_size_usd"] * tp_dist_frac
            rr_ratio = (reward_amount / risk_amount) if risk_amount > 0 else 0.0

            # ── Check 7: Minimum risk-reward ratio ──
            if rr_ratio < 1.0:
                result["checks"]["risk_reward"] = {
                    "passed": False,
                    "ratio": round(rr_ratio, 2),
                }
                result["reason"] = f"R:R ratio {rr_ratio:.2f} < 1.0 minimum"
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: R:R {rr_ratio:.2f} < 1.0")
                return result

            result["checks"]["risk_reward"] = {
                "passed": True,
                "ratio": round(rr_ratio, 2),
            }

            # ── Check 8: Margin available ──
            margin_req = pos["position_size_usd"] / leverage
            if margin_req > self.current_capital * 0.95:
                result["reason"] = (
                    f"Insufficient margin: need ${margin_req:,.2f}, "
                    f"have ${self.current_capital * 0.95:,.2f} available"
                )
                self.signals_rejected += 1
                logger.info(f"REJECTED [{symbol}]: {result['reason']}")
                return result

            # ═════════════════════════════════════════════
            #  ALL CHECKS PASSED → APPROVED
            # ═════════════════════════════════════════════
            result["approved"] = True
            result["reason"] = "All risk checks passed"
            result["position_size_usd"] = pos["position_size_usd"]
            result["position_size_units"] = pos["position_size_units"]
            result["margin_required"] = round(margin_req, 2)
            result["leverage"] = leverage
            result["stop_loss"] = stop_loss
            result["take_profit"] = take_profit
            result["risk_amount_usd"] = round(risk_amount, 2)
            result["reward_amount_usd"] = round(reward_amount, 2)
            result["risk_reward_ratio"] = round(rr_ratio, 2)
            result["risk_pct"] = pos["risk_pct"]

            self.signals_approved += 1
            now = self._sim_time or datetime.now(timezone.utc)
            self.last_trade_time = now

            logger.info(
                f"APPROVED [{symbol}] {sig_type} | "
                f"size=${pos['position_size_usd']:,.2f} | lev={leverage}x | "
                f"margin=${margin_req:,.2f} | "
                f"SL={stop_loss:.4f} | TP={take_profit:.4f} | "
                f"R:R={rr_ratio:.2f} | risk=${risk_amount:.2f} ({pos['risk_pct']:.1f}%)"
            )
            return result

        except Exception as e:
            result["reason"] = f"Risk evaluation error: {e}"
            logger.error(f"Risk evaluation error: {e}", exc_info=True)
            self.signals_rejected += 1
            return result

    # ═══════════════════════════════════════════════════════════════════
    #  POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        leverage: int = 1,
    ) -> Dict:
        """
        Calculate position size using fixed-fraction risk model.

        Formula:
            risk_amount  = capital × risk_per_trade_pct%
            notional     = risk_amount / SL_distance_fraction
            margin       = notional / leverage
            Cap notional by max_position_size_pct (as max margin %)

        Args:
            capital:     Available capital (USDT)
            entry_price: Entry price
            stop_loss:   Stop-loss price
            leverage:    Leverage multiplier

        Returns:
            Dict: position_size_usd, position_size_units, risk_amount,
                  risk_pct, leverage, capped
        """
        empty = {
            "position_size_usd": 0.0,
            "position_size_units": 0.0,
            "risk_amount": 0.0,
            "risk_pct": 0.0,
            "leverage": leverage,
            "capped": False,
        }

        try:
            if capital <= 0 or entry_price <= 0 or stop_loss <= 0:
                return empty

            # SL distance as fraction of entry
            sl_dist = abs(entry_price - stop_loss) / entry_price

            # Guard: too tight (< 0.05%) or too wide (> 50%)
            if sl_dist < 0.0005 or sl_dist > 0.50:
                sl_dist = self.default_sl_pct / 100.0
                logger.debug(
                    f"SL distance out of range, using default {self.default_sl_pct}%"
                )

            # Dollar risk
            risk_amount = capital * (self.risk_per_trade_pct / 100.0)

            # Notional position size
            notional = risk_amount / sl_dist

            # Cap 1: max margin as % of capital → max notional
            max_margin = capital * (self.max_position_size_pct / 100.0)
            max_notional = max_margin * leverage
            capped = False
            if notional > max_notional:
                notional = max_notional
                risk_amount = notional * sl_dist
                capped = True

            # Cap 2: available margin (keep 5% buffer)
            margin_needed = notional / leverage
            margin_avail = capital * 0.95
            if margin_needed > margin_avail:
                margin_needed = margin_avail
                notional = margin_needed * leverage
                risk_amount = notional * sl_dist
                capped = True

            units = notional / entry_price
            actual_risk_pct = (risk_amount / capital * 100.0) if capital > 0 else 0.0

            return {
                "position_size_usd": round(notional, 2),
                "position_size_units": round(units, 8),
                "risk_amount": round(risk_amount, 2),
                "risk_pct": round(actual_risk_pct, 2),
                "leverage": leverage,
                "capped": capped,
            }

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return empty

    # ═══════════════════════════════════════════════════════════════════
    #  STOP-LOSS / TAKE-PROFIT
    # ═══════════════════════════════════════════════════════════════════

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float = None,
    ) -> Dict:
        """
        Calculate stop-loss price.

        Uses ATR-based if available and enabled, otherwise fixed %.

        Args:
            entry_price: Entry price
            direction:   1 = LONG, -1 = SHORT
            atr:         ATR value (optional)

        Returns:
            Dict: stop_loss, method, distance_pct, distance_price
        """
        try:
            if self.use_atr_stops and atr and atr > 0:
                sl_distance = atr * self.atr_sl_multiplier
                method = f"ATR×{self.atr_sl_multiplier}"
            else:
                sl_distance = entry_price * (self.default_sl_pct / 100.0)
                method = f"fixed_{self.default_sl_pct}%"

            if direction == 1:  # LONG: SL below entry
                stop_loss = entry_price - sl_distance
                stop_loss = max(stop_loss, entry_price * 0.005)  # Floor > 0
            else:  # SHORT: SL above entry
                stop_loss = entry_price + sl_distance

            dist_pct = abs(entry_price - stop_loss) / entry_price * 100.0

            return {
                "stop_loss": round(stop_loss, 8),
                "method": method,
                "distance_pct": round(dist_pct, 4),
                "distance_price": round(sl_distance, 8),
            }

        except Exception as e:
            logger.error(f"Stop-loss calculation error: {e}")
            fallback = (
                entry_price * (1 - self.default_sl_pct / 100.0)
                if direction == 1
                else entry_price * (1 + self.default_sl_pct / 100.0)
            )
            return {
                "stop_loss": round(fallback, 8),
                "method": "fallback",
                "distance_pct": self.default_sl_pct,
                "distance_price": round(abs(entry_price - fallback), 8),
            }

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        stop_loss: float,
        atr: float = None,
        risk_reward_target: float = 2.0,
    ) -> Dict:
        """
        Calculate take-profit based on risk-reward ratio.

        TP distance = SL distance × risk_reward_target.

        Args:
            entry_price:        Entry price
            direction:          1 = LONG, -1 = SHORT
            stop_loss:          Stop-loss price (for SL distance calc)
            atr:                ATR (optional, unused — R:R based)
            risk_reward_target: Target R:R ratio (default 2.0)

        Returns:
            Dict: take_profit, method, distance_pct, distance_price, risk_reward_ratio
        """
        try:
            sl_distance = abs(entry_price - stop_loss)

            if sl_distance <= 0:
                tp_distance = entry_price * (self.default_tp_pct / 100.0)
                method = f"fixed_{self.default_tp_pct}%"
            else:
                tp_distance = sl_distance * risk_reward_target
                method = f"R:R_{risk_reward_target}"

            if direction == 1:  # LONG: TP above entry
                take_profit = entry_price + tp_distance
            else:  # SHORT: TP below entry
                take_profit = entry_price - tp_distance
                take_profit = max(take_profit, entry_price * 0.005)

            dist_pct = abs(take_profit - entry_price) / entry_price * 100.0
            rr = (tp_distance / sl_distance) if sl_distance > 0 else risk_reward_target

            return {
                "take_profit": round(take_profit, 8),
                "method": method,
                "distance_pct": round(dist_pct, 4),
                "distance_price": round(tp_distance, 8),
                "risk_reward_ratio": round(rr, 2),
            }

        except Exception as e:
            logger.error(f"Take-profit calculation error: {e}")
            fallback = (
                entry_price * (1 + self.default_tp_pct / 100.0)
                if direction == 1
                else entry_price * (1 - self.default_tp_pct / 100.0)
            )
            return {
                "take_profit": round(fallback, 8),
                "method": "fallback",
                "distance_pct": self.default_tp_pct,
                "distance_price": round(abs(fallback - entry_price), 8),
                "risk_reward_ratio": risk_reward_target,
            }

    # ═══════════════════════════════════════════════════════════════════
    #  TRAILING STOP
    # ═══════════════════════════════════════════════════════════════════

 ##   def calculate_trailing_stop(
 ##       self,
 ##       entry_price: float,
 ##       current_price: float,
 ##       direction: int,
 ##       current_stop: float,
 ##       highest_price: float = None,
 ##       lowest_price: float = None,
 ##   ) -> Dict:
 ##       """
 ##       Calculate trailing stop level.

 ##       Trailing stop only moves in the direction of profit, never backwards.

 ##       Args:
 ##           entry_price:   Original entry price
 ##           current_price: Current market price
 ##           direction:     1 = LONG, -1 = SHORT
 ##           current_stop:  Current stop-loss level
 ##           highest_price: Highest price since entry (LONG)
 ##           lowest_price:  Lowest price since entry (SHORT)

 ##       Returns:
 ##           Dict: new_stop, triggered, moved, pnl_pct
 ##       """
 ##       try:
 ##           trail_frac = self.trailing_stop_pct / 100.0

 ##           if direction == 1:  # LONG
 ##               ref = highest_price if highest_price else max(current_price, entry_price)
 ##               new_stop = ref * (1.0 - trail_frac)
 ##               new_stop = max(new_stop, current_stop)  # Never move down
 ##               triggered = current_price <= new_stop
 ##           else:  # SHORT
 ##               ref = lowest_price if lowest_price else min(current_price, entry_price)
 ##               new_stop = ref * (1.0 + trail_frac)
 ##               new_stop = min(new_stop, current_stop)  # Never move up
 ##               triggered = current_price >= new_stop

 ##           moved = abs(new_stop - current_stop) > 1e-10
 ##           pnl_pct = ((current_price - entry_price) / entry_price * 100.0) * direction

 ##           return {
 ##               "new_stop": round(new_stop, 8),
 ##               "triggered": triggered,
 ##               "moved": moved,
 ##               "ref_price": round(ref, 8),
 ##               "pnl_pct": round(pnl_pct, 4),
 ##           }

 ##       except Exception as e:
 ##           logger.error(f"Trailing stop error: {e}")
 ##           return {
 ##               "new_stop": current_stop,
 ##               "triggered": False,
 ##               "moved": False,
 ##               "ref_price": current_price,
 ##               "pnl_pct": 0.0,
 ##           } 


  #14th after gain - gains lock mechanism update
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: int,
        current_stop: float,
        highest_price: float = None,
        lowest_price: float = None,
    ) -> Dict:
        """
        Adaptive trailing stop — tightens as profit grows.

        Tiers (from config):
          profit < 1.5%  → 3.5% trail (wide, room to breathe)
          profit 1.5-2.5% → 2.0% trail (protect breakeven)
          profit 2.5-3.5% → 1.5% trail (lock real profit)
          profit > 3.5%   → 1.0% trail (tight, near TP)

        Stop only moves in profit direction, never backwards.
        """
        try:
            # Determine reference price (peak profit point)
            if direction == 1:  # LONG
                ref = highest_price if highest_price else max(current_price, entry_price)
                profit_pct = (ref - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
            else:  # SHORT
                ref = lowest_price if lowest_price else min(current_price, entry_price)
                profit_pct = (entry_price - ref) / entry_price * 100.0 if entry_price > 0 else 0.0

            # Adaptive trailing distance based on profit tier
            trail_frac = self.trailing_stop_pct / 100.0  # default fallback
            tier_used = "default"
            for tier in self._trailing_tiers:  # sorted highest-first
                if profit_pct >= tier["min_profit_pct"]:
                    trail_frac = tier["trail_pct"] / 100.0
                    tier_used = f"{tier['trail_pct']}%@{tier['min_profit_pct']}%"
                    break

            # Calculate new stop
            if direction == 1:  # LONG
                new_stop = ref * (1.0 - trail_frac)
                new_stop = max(new_stop, current_stop)  # Never move down
                triggered = current_price <= new_stop
            else:  # SHORT
                new_stop = ref * (1.0 + trail_frac)
                new_stop = min(new_stop, current_stop)  # Never move up
                triggered = current_price >= new_stop

            moved = abs(new_stop - current_stop) > 1e-10
            pnl_pct = ((current_price - entry_price) / entry_price * 100.0) * direction

            if moved:
                logger.debug(
                    f"Trailing stop moved: profit={profit_pct:.2f}% tier={tier_used} "
                    f"trail={trail_frac*100:.1f}% stop={current_stop:.2f}→{new_stop:.2f}"
                )

            return {
                "new_stop": round(new_stop, 8),
                "triggered": triggered,
                "moved": moved,
                "ref_price": round(ref, 8),
                "pnl_pct": round(pnl_pct, 4),
            }

        except Exception as e:
            logger.error(f"Trailing stop error: {e}")
            return {
                "new_stop": current_stop,
                "triggered": False,
                "moved": False,
                "ref_price": current_price,
                "pnl_pct": 0.0,
            }
    # ═══════════════════════════════════════════════════════════════════
    #  EXIT CONDITION CHECKS
    # ═══════════════════════════════════════════════════════════════════

    def check_exit_conditions(
        self,
        entry_price: float,
        current_price: float,
        direction: int,
        stop_loss: float,
        take_profit: float,
        highest_price: float = None,
        lowest_price: float = None,
        current_stop: float = None,
    ) -> Dict:
        """
        Check if any exit condition is met at a single price point.

        Priority: stop_loss → take_profit → trailing_stop.

        Used by paper_trader / live_trader for tick-level checks.
        For bar-level backtesting, use check_bar_exit() instead.

        Returns:
            Dict: should_exit, exit_reason, exit_price, pnl_pct, new_trailing_stop
        """
        try:
            eff_stop = current_stop if current_stop else stop_loss
            pnl_pct = ((current_price - entry_price) / entry_price * 100.0) * direction

            # SL check
            if direction == 1 and current_price <= eff_stop:
                return {
                    "should_exit": True,
                    "exit_reason": "stop_loss",
                    "exit_price": eff_stop,
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }
            if direction == -1 and current_price >= eff_stop:
                return {
                    "should_exit": True,
                    "exit_reason": "stop_loss",
                    "exit_price": eff_stop,
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }

            # TP check
            if direction == 1 and current_price >= take_profit:
                return {
                    "should_exit": True,
                    "exit_reason": "take_profit",
                    "exit_price": take_profit,
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }
            if direction == -1 and current_price <= take_profit:
                return {
                    "should_exit": True,
                    "exit_reason": "take_profit",
                    "exit_price": take_profit,
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }

            # Trailing stop check
            trail = self.calculate_trailing_stop(
                entry_price, current_price, direction,
                eff_stop, highest_price, lowest_price,
            )
            if trail["triggered"]:
                return {
                    "should_exit": True,
                    "exit_reason": "trailing_stop",
                    "exit_price": trail["new_stop"],
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": trail["new_stop"],
                }

            # No exit
            return {
                "should_exit": False,
                "exit_reason": None,
                "exit_price": None,
                "pnl_pct": round(pnl_pct, 4),
                "new_trailing_stop": trail["new_stop"],
            }

        except Exception as e:
            logger.error(f"Exit condition check error: {e}")
            return {
                "should_exit": False,
                "exit_reason": None,
                "exit_price": None,
                "pnl_pct": 0.0,
                "new_trailing_stop": stop_loss,
            }

    def check_bar_exit(
        self,
        entry_price: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        direction: int,
        stop_loss: float,
        take_profit: float,
        highest_since_entry: float = None,
        lowest_since_entry: float = None,
        current_stop: float = None,
    ) -> Dict:
        """
        Check exit conditions against a full OHLC bar.

        For backtesting: checks whether SL or TP was hit within the bar.
        SL is checked first (conservative assumption — worst case fills first).

        Args:
            entry_price:          Original entry price
            bar_high / bar_low:   Bar extremes
            bar_close:            Bar close price
            direction:            1 = LONG, -1 = SHORT
            stop_loss:            Stop-loss level
            take_profit:          Take-profit level
            highest_since_entry:  Highest high since entry (for trailing stop)
            lowest_since_entry:   Lowest low since entry (for trailing stop)
            current_stop:         Current trailing stop (may differ from initial SL)

        Returns:
            Dict: should_exit, exit_reason, exit_price, pnl_pct, new_trailing_stop
        """
        try:
            eff_stop = current_stop if current_stop else stop_loss

            # Update tracking extremes
            if highest_since_entry is None:
                highest_since_entry = bar_high
            else:
                highest_since_entry = max(highest_since_entry, bar_high)

            if lowest_since_entry is None:
                lowest_since_entry = bar_low
            else:
                lowest_since_entry = min(lowest_since_entry, bar_low)

            # ── SL check first (conservative) ──
            if direction == 1 and bar_low <= eff_stop:
                exit_price = eff_stop
                pnl_pct = ((exit_price - entry_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "stop_loss",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }
            if direction == -1 and bar_high >= eff_stop:
                exit_price = eff_stop
                pnl_pct = ((entry_price - exit_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "stop_loss",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }

            # ── TP check ──
            if direction == 1 and bar_high >= take_profit:
                exit_price = take_profit
                pnl_pct = ((exit_price - entry_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "take_profit",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }
            if direction == -1 and bar_low <= take_profit:
                exit_price = take_profit
                pnl_pct = ((entry_price - exit_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "take_profit",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": eff_stop,
                }

            # ── Trailing stop update ──
            trail = self.calculate_trailing_stop(
                entry_price,
                bar_close,
                direction,
                eff_stop,
                highest_since_entry,
                lowest_since_entry,
            )

            # Check if trailing stop was hit within bar
            if direction == 1 and trail["moved"] and bar_low <= trail["new_stop"]:
                exit_price = trail["new_stop"]
                pnl_pct = ((exit_price - entry_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "trailing_stop",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": trail["new_stop"],
                }
            if direction == -1 and trail["moved"] and bar_high >= trail["new_stop"]:
                exit_price = trail["new_stop"]
                pnl_pct = ((entry_price - exit_price) / entry_price * 100.0)
                return {
                    "should_exit": True,
                    "exit_reason": "trailing_stop",
                    "exit_price": round(exit_price, 8),
                    "pnl_pct": round(pnl_pct, 4),
                    "new_trailing_stop": trail["new_stop"],
                }

            # ── No exit ──
            pnl_pct = ((bar_close - entry_price) / entry_price * 100.0) * direction
            return {
                "should_exit": False,
                "exit_reason": None,
                "exit_price": None,
                "pnl_pct": round(pnl_pct, 4),
                "new_trailing_stop": trail["new_stop"],
            }

        except Exception as e:
            logger.error(f"Bar exit check error: {e}")
            return {
                "should_exit": False,
                "exit_reason": None,
                "exit_price": None,
                "pnl_pct": 0.0,
                "new_trailing_stop": stop_loss,
            }

    # ═══════════════════════════════════════════════════════════════════
    #  LEVERAGE CALCULATION
    # ═══════════════════════════════════════════════════════════════════

    def calculate_leverage(
        self, confidence: float, volatility_pct: float = None
    ) -> int:
        """
        Dynamic leverage based on confidence and volatility.

        Higher confidence + lower volatility → more leverage (up to max).
        Lower confidence or higher volatility → less leverage.

        Args:
            confidence:     Signal confidence 0-1
            volatility_pct: ATR as % of price (or similar vol measure)

        Returns:
            int: Leverage (1 to max_leverage)
        """
        try:
            # Base from confidence
            if confidence >= 0.85:
                base = 4
            elif confidence >= 0.75:
                base = 3
            elif confidence >= 0.65:
                base = 2
            else:
                base = 1

            # Volatility adjustment
            vol_adj = 0
            if volatility_pct and volatility_pct > 0:
                if volatility_pct > 5.0:
                    vol_adj = -2
                elif volatility_pct > 3.0:
                    vol_adj = -1
                elif volatility_pct < 1.0:
                    vol_adj = 1

            lev = max(1, min(base + vol_adj, self.max_leverage))
            return int(lev)

        except Exception as e:
            logger.error(f"Leverage calculation error: {e}")
            return 1

    # ═══════════════════════════════════════════════════════════════════
    #  COMMISSION ESTIMATION
    # ═══════════════════════════════════════════════════════════════════

    def estimate_commission(self, position_size_usd: float) -> Dict:
        """
        Estimate round-trip commission and slippage cost.

        Args:
            position_size_usd: Notional position size

        Returns:
            Dict: entry_commission, exit_commission, total_commission,
                  slippage_cost, total_cost, total_cost_pct
        """
        entry_comm = position_size_usd * (self.commission_pct / 100.0)
        exit_comm = position_size_usd * (self.commission_pct / 100.0)
        slip_cost = position_size_usd * (self.slippage_pct / 100.0) * 2  # Both sides
        total = entry_comm + exit_comm + slip_cost
        total_pct = (total / position_size_usd * 100.0) if position_size_usd > 0 else 0.0

        return {
            "entry_commission": round(entry_comm, 4),
            "exit_commission": round(exit_comm, 4),
            "total_commission": round(entry_comm + exit_comm, 4),
            "slippage_cost": round(slip_cost, 4),
            "total_cost": round(total, 4),
            "total_cost_pct": round(total_pct, 4),
        }

    # ═══════════════════════════════════════════════════════════════════
    #  CIRCUIT BREAKER CHECKS
    # ═══════════════════════════════════════════════════════════════════

    def check_circuit_breakers(self) -> Dict:
        """
        Run all circuit breaker checks.

        Returns:
            Dict: can_trade (bool), breakers (individual results),
                  active_breakers (list of triggered breaker descriptions)
        """
        breakers = {}
        active: List[str] = []

        daily = self._check_daily_loss()
        breakers["daily_loss"] = daily
        if not daily["passed"]:
            active.append(
                f"daily_loss ({daily['current_pct']:.1f}% >= {self.max_daily_loss_pct}%)"
            )

        weekly = self._check_weekly_loss()
        breakers["weekly_loss"] = weekly
        if not weekly["passed"]:
            active.append(
                f"weekly_loss ({weekly['current_pct']:.1f}% >= {self.max_weekly_loss_pct}%)"
            )

        dd = self._check_max_drawdown()
        breakers["max_drawdown"] = dd
        if not dd["passed"]:
            active.append(
                f"max_drawdown ({dd['drawdown_pct']:.1f}% >= {self.max_total_dd_pct}%)"
            )

        consec = self._check_consecutive_losses()
        breakers["consecutive_losses"] = consec
        if not consec["passed"]:
            active.append(
                f"consecutive_losses ({consec['count']} >= {self.max_consecutive_losses})"
            )

        can_trade = len(active) == 0

        if not can_trade:
            logger.warning(f"Circuit breakers ACTIVE: {active}")

        return {
            "can_trade": can_trade,
            "breakers": breakers,
            "active_breakers": active,
        }

    # ── Private check helpers ──

    def _check_confidence(self, confidence: float, agreement: float) -> Dict:
        """Check minimum confidence and agreement thresholds."""
        if confidence < self.min_confidence:
            return {
                "passed": False,
                "reason": (
                    f"Confidence {confidence:.3f} < min {self.min_confidence:.3f}"
                ),
                "confidence": confidence,
                "agreement": agreement,
            }
        if agreement < self.min_agreement:
            return {
                "passed": False,
                "reason": (
                    f"Agreement {agreement:.3f} < min {self.min_agreement:.3f}"
                ),
                "confidence": confidence,
                "agreement": agreement,
            }
        return {
            "passed": True,
            "reason": "OK",
            "confidence": confidence,
            "agreement": agreement,
        }

    def _check_max_positions(self) -> Dict:
        """Check open position count against limit."""
        try:
            if self.mode == "backtest":
                count = self._open_positions
            else:
                count = self.db.get_open_position_count(self.mode)

            passed = count < self.max_open_positions
            return {
                "passed": passed,
                "reason": (
                    f"Open: {count}/{self.max_open_positions}"
                    + ("" if passed else " — limit reached")
                ),
                "open_count": count,
                "max_allowed": self.max_open_positions,
            }
        except Exception as e:
            logger.error(f"Max positions check error: {e}")
            return {
                "passed": True,
                "reason": "Check failed — allowing",
                "open_count": 0,
                "max_allowed": self.max_open_positions,
            }

    def _check_cooldown(self) -> Dict:
        """Check if in cooldown period after a loss."""
        try:
            if not self.last_loss_time:
                return {"passed": True, "reason": "No recent loss", "remaining_min": 0}

            now = self._sim_time or datetime.now(timezone.utc)
            elapsed_min = (now - self.last_loss_time).total_seconds() / 60.0
            remaining = self.cooldown_minutes - elapsed_min

            if remaining > 0:
                return {
                    "passed": False,
                    "reason": f"Cooldown: {remaining:.0f} min remaining",
                    "remaining_min": round(remaining, 1),
                }
            return {"passed": True, "reason": "Cooldown expired", "remaining_min": 0}

        except Exception as e:
            logger.error(f"Cooldown check error: {e}")
            return {"passed": True, "reason": "Check failed — allowing", "remaining_min": 0}

    def _check_daily_loss(self) -> Dict:
        """Check daily loss limit."""
        try:
            if self.mode == "backtest":
                daily_pnl = self._get_internal_daily_pnl()
            else:
                daily_pnl = self.db.get_daily_pnl(self.mode)

            daily_pct = (
                (abs(daily_pnl) / self.current_capital * 100.0)
                if self.current_capital > 0 and daily_pnl < 0
                else 0.0
            )
            passed = daily_pct < self.max_daily_loss_pct

            return {
                "passed": passed,
                "current_pnl": round(daily_pnl, 2),
                "current_pct": round(daily_pct, 2),
                "limit_pct": self.max_daily_loss_pct,
            }
        except Exception as e:
            logger.error(f"Daily loss check error: {e}")
            return {
                "passed": True,
                "current_pnl": 0,
                "current_pct": 0,
                "limit_pct": self.max_daily_loss_pct,
            }

    def _check_weekly_loss(self) -> Dict:
        """Check weekly loss limit."""
        try:
            if self.mode == "backtest":
                weekly_pnl = self._get_internal_weekly_pnl()
            else:
                weekly_pnl = self.db.get_weekly_pnl(self.mode)

            weekly_pct = (
                (abs(weekly_pnl) / self.current_capital * 100.0)
                if self.current_capital > 0 and weekly_pnl < 0
                else 0.0
            )
            passed = weekly_pct < self.max_weekly_loss_pct

            return {
                "passed": passed,
                "current_pnl": round(weekly_pnl, 2),
                "current_pct": round(weekly_pct, 2),
                "limit_pct": self.max_weekly_loss_pct,
            }
        except Exception as e:
            logger.error(f"Weekly loss check error: {e}")
            return {
                "passed": True,
                "current_pnl": 0,
                "current_pct": 0,
                "limit_pct": self.max_weekly_loss_pct,
            }

    def _check_max_drawdown(self) -> Dict:
        """Check max drawdown from peak capital."""
        try:
            if self.peak_capital <= 0:
                return {
                    "passed": True,
                    "drawdown_pct": 0,
                    "limit_pct": self.max_total_dd_pct,
                }

            dd_pct = (
                (self.peak_capital - self.current_capital) / self.peak_capital * 100.0
            )
            passed = dd_pct < self.max_total_dd_pct

            return {
                "passed": passed,
                "drawdown_pct": round(dd_pct, 2),
                "current_capital": round(self.current_capital, 2),
                "peak_capital": round(self.peak_capital, 2),
                "limit_pct": self.max_total_dd_pct,
            }
        except Exception as e:
            logger.error(f"Max drawdown check error: {e}")
            return {
                "passed": True,
                "drawdown_pct": 0,
                "limit_pct": self.max_total_dd_pct,
            }

    def _check_consecutive_losses(self) -> Dict:
        """Check consecutive loss streak."""
        try:
            if self.mode == "backtest":
                count = self._consecutive_losses
            else:
                count = self.db.get_consecutive_losses(self.mode)

            passed = count < self.max_consecutive_losses
            return {
                "passed": passed,
                "count": count,
                "limit": self.max_consecutive_losses,
            }
        except Exception as e:
            logger.error(f"Consecutive losses check error: {e}")
            return {"passed": True, "count": 0, "limit": self.max_consecutive_losses}

    def _validate_sl_tp(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: int,
    ) -> Dict:
        """Validate SL and TP are on the correct sides and within sane range."""
        try:
            if direction == 1:
                sl_ok = stop_loss < entry_price
                tp_ok = take_profit > entry_price
            else:
                sl_ok = stop_loss > entry_price
                tp_ok = take_profit < entry_price

            sl_pct = abs(entry_price - stop_loss) / entry_price * 100.0
            range_ok = 0.05 <= sl_pct <= 25.0

            passed = sl_ok and tp_ok and range_ok

            reasons = []
            if not sl_ok:
                reasons.append(
                    f"SL wrong side (SL={stop_loss:.4f} entry={entry_price:.4f} dir={direction})"
                )
            if not tp_ok:
                reasons.append(
                    f"TP wrong side (TP={take_profit:.4f} entry={entry_price:.4f} dir={direction})"
                )
            if not range_ok:
                reasons.append(f"SL distance {sl_pct:.2f}% outside [0.05-25%]")

            return {
                "passed": passed,
                "reason": "; ".join(reasons) if reasons else "Valid",
                "sl_valid": sl_ok,
                "tp_valid": tp_ok,
                "sl_distance_pct": round(sl_pct, 4),
            }

        except Exception as e:
            logger.error(f"SL/TP validation error: {e}")
            return {
                "passed": False,
                "reason": f"Validation error: {e}",
                "sl_valid": False,
                "tp_valid": False,
                "sl_distance_pct": 0,
            }

    # ═══════════════════════════════════════════════════════════════════
    #  INTERNAL STATE FOR BACKTEST MODE
    # ═══════════════════════════════════════════════════════════════════

    def _get_internal_daily_pnl(self) -> float:
        """Sum PnL of trades from the current simulation day."""
        now = self._sim_time or datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return sum(
            t["pnl_usd"]
            for t in self._trade_results
            if t["timestamp"] >= day_start
        )

    def _get_internal_weekly_pnl(self) -> float:
        """Sum PnL of trades from the last 7 days."""
        now = self._sim_time or datetime.now(timezone.utc)
        week_start = now - timedelta(days=7)
        return sum(
            t["pnl_usd"]
            for t in self._trade_results
            if t["timestamp"] >= week_start
        )

    # ═══════════════════════════════════════════════════════════════════
    #  STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def record_trade_result(
        self,
        pnl_usd: float,
        is_win: bool,
        timestamp: datetime = None,
    ):
        """
        Record a completed trade result.

        Updates: capital, peak, consecutive losses, last_loss_time, internal log.

        Args:
            pnl_usd:   Profit / loss in USD
            is_win:    Whether the trade was profitable
            timestamp: Trade close time (default: sim_time or now)
        """
        ts = timestamp or self._sim_time or datetime.now(timezone.utc)

        self._trade_results.append({
            "timestamp": ts,
            "pnl_usd": pnl_usd,
            "is_win": is_win,
        })

        self.current_capital += pnl_usd

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self.last_loss_time = ts

        logger.debug(
            f"Trade result: PnL=${pnl_usd:+,.2f} | "
            f"{'WIN' if is_win else 'LOSS'} | "
            f"Capital=${self.current_capital:,.2f} | "
            f"Consec losses={self._consecutive_losses}"
        )

    def record_position_open(self):
        """Increment internal open position count (backtest mode)."""
        self._open_positions += 1

    def record_position_close(self):
        """Decrement internal open position count (backtest mode)."""
        self._open_positions = max(0, self._open_positions - 1)

    def update_capital(self, new_capital: float):
        """Set current capital and update peak if new high."""
        old = self.current_capital
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        logger.debug(
            f"Capital: ${old:,.2f} → ${new_capital:,.2f} "
            f"(peak: ${self.peak_capital:,.2f})"
        )

    def get_capital(self) -> float:
        """Return current capital."""
        return self.current_capital

    def set_simulation_time(self, dt: datetime):
        """
        Set the current simulation timestamp (for backtest mode).

        The backtester calls this before each bar so that cooldown
        and daily/weekly PnL checks use the correct reference time.
        """
        self._sim_time = dt

    def reset(self, capital: float = None):
        """
        Reset all internal state for a fresh run.

        Args:
            capital: New starting capital (default: initial_capital)
        """
        self.current_capital = capital or self.initial_capital
        self.peak_capital = self.current_capital
        self.last_trade_time = None
        self.last_loss_time = None
        self._trade_results.clear()
        self._consecutive_losses = 0
        self._open_positions = 0
        self._sim_time = None
        self.signals_evaluated = 0
        self.signals_approved = 0
        self.signals_rejected = 0
        logger.info(f"RiskManager RESET | capital=${self.current_capital:,.2f}")

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY / INFO
    # ═══════════════════════════════════════════════════════════════════

    def get_risk_summary(self) -> Dict:
        """Get full risk state summary."""
        try:
            breakers = self.check_circuit_breakers()
            dd_pct = (
                (self.peak_capital - self.current_capital) / self.peak_capital * 100.0
                if self.peak_capital > 0
                else 0.0
            )
            total_pnl = self.current_capital - self.initial_capital
            total_ret = (
                total_pnl / self.initial_capital * 100.0
                if self.initial_capital > 0
                else 0.0
            )

            wins = sum(1 for t in self._trade_results if t["is_win"])
            losses = sum(1 for t in self._trade_results if not t["is_win"])
            total_trades = len(self._trade_results)
            win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

            return {
                "mode": self.mode,
                "current_capital": round(self.current_capital, 2),
                "initial_capital": round(self.initial_capital, 2),
                "peak_capital": round(self.peak_capital, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_ret, 2),
                "max_drawdown_pct": round(dd_pct, 2),
                "consecutive_losses": self._consecutive_losses,
                "open_positions": self._open_positions,
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(win_rate, 1),
                "can_trade": breakers["can_trade"],
                "active_breakers": breakers["active_breakers"],
                "signals_evaluated": self.signals_evaluated,
                "signals_approved": self.signals_approved,
                "signals_rejected": self.signals_rejected,
                "approval_rate_pct": round(
                    self.signals_approved / max(1, self.signals_evaluated) * 100.0, 1
                ),
                "settings": {
                    "max_leverage": self.max_leverage,
                    "risk_per_trade_pct": self.risk_per_trade_pct,
                    "max_position_size_pct": self.max_position_size_pct,
                    "max_open_positions": self.max_open_positions,
                    "use_atr_stops": self.use_atr_stops,
                    "atr_sl_multiplier": self.atr_sl_multiplier,
                    "trailing_stop_pct": self.trailing_stop_pct,
                    "min_confidence": self.min_confidence,
                    "min_agreement": self.min_agreement,
                },
            }

        except Exception as e:
            logger.error(f"Risk summary error: {e}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    def _p(label, obj):
        """Pretty-print helper."""
        if isinstance(obj, dict):
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(f"  {label}: {obj}")

    print("\n" + "█" * 60)
    print("  RISK MANAGER — TESTS")
    print("█" * 60)

    rm = RiskManager(capital=10000.0, mode="backtest")

    # ── 1. Position sizing ──
    pos = rm.calculate_position_size(
        capital=10000, entry_price=50000, stop_loss=49000, leverage=3
    )
    _p("1. Position Sizing (BTC $50k, SL $49k, 3x)", pos)
    assert pos["position_size_usd"] > 0, "Position size must be > 0"
    assert pos["risk_pct"] <= rm.risk_per_trade_pct + 0.1, "Risk % must be ≤ configured"

    # ── 2. Stop-loss (ATR-based) ──
    sl = rm.calculate_stop_loss(entry_price=50000, direction=1, atr=1500)
    _p("2a. Stop-Loss LONG (ATR=1500)", sl)
    assert sl["stop_loss"] < 50000, "LONG SL must be below entry"

    sl_short = rm.calculate_stop_loss(entry_price=50000, direction=-1, atr=1500)
    _p("2b. Stop-Loss SHORT (ATR=1500)", sl_short)
    assert sl_short["stop_loss"] > 50000, "SHORT SL must be above entry"

    # ── 3. Stop-loss (fixed %) ──
    sl_fixed = rm.calculate_stop_loss(entry_price=50000, direction=1, atr=None)
    _p("3. Stop-Loss LONG (fixed %)", sl_fixed)
    assert sl_fixed["method"].startswith("fixed"), "Should use fixed method"

    # ── 4. Take-profit ──
    tp = rm.calculate_take_profit(
        entry_price=50000, direction=1, stop_loss=47000, atr=1500
    )
    _p("4. Take-Profit LONG (SL=47000)", tp)
    assert tp["take_profit"] > 50000, "LONG TP must be above entry"
    assert tp["risk_reward_ratio"] >= 1.9, "R:R should be ~2.0"

    # ── 5. Leverage calculation ──
    for conf, vol in [(0.60, 2.0), (0.70, 1.5), (0.80, 3.5), (0.90, 0.8)]:
        lev = rm.calculate_leverage(conf, vol)
        print(f"  5. Leverage: conf={conf}, vol={vol}% → {lev}x")
    assert rm.calculate_leverage(0.90, 0.8) >= rm.calculate_leverage(0.60, 5.0)

    # ── 6. Commission estimate ──
    comm = rm.estimate_commission(10000)
    _p("6. Commission ($10k notional)", comm)
    assert comm["total_cost"] > 0

    # ── 7. Trailing stop ──
    trail = rm.calculate_trailing_stop(
        entry_price=50000, current_price=53000, direction=1,
        current_stop=49000, highest_price=54000,
    )
    _p("7. Trailing Stop LONG (entry=50k, high=54k, now=53k)", trail)
    assert trail["new_stop"] > 49000, "Trailing stop should have moved up"

    # ── 8. Bar exit check ──
    # SL hit scenario
    exit_sl = rm.check_bar_exit(
        entry_price=50000, bar_high=50500, bar_low=46500, bar_close=47000,
        direction=1, stop_loss=47000, take_profit=56000,
    )
    _p("8a. Bar Exit — SL hit (low=46500, SL=47000)", exit_sl)
    assert exit_sl["should_exit"] and exit_sl["exit_reason"] == "stop_loss"

    # TP hit scenario
    exit_tp = rm.check_bar_exit(
        entry_price=50000, bar_high=57000, bar_low=50500, bar_close=56500,
        direction=1, stop_loss=47000, take_profit=56000,
    )
    _p("8b. Bar Exit — TP hit (high=57000, TP=56000)", exit_tp)
    assert exit_tp["should_exit"] and exit_tp["exit_reason"] == "take_profit"

    # Neither hit
    exit_none = rm.check_bar_exit(
        entry_price=50000, bar_high=50500, bar_low=50000, bar_close=50300,
        direction=1, stop_loss=47000, take_profit=56000,
    )
    _p("8c. Bar Exit — no exit", exit_none)
    assert not exit_none["should_exit"]

    # SHORT SL
    exit_short_sl = rm.check_bar_exit(
        entry_price=50000, bar_high=52000, bar_low=49800, bar_close=51800,
        direction=-1, stop_loss=51500, take_profit=47000,
    )
    _p("8d. Bar Exit SHORT — SL hit (high=52000, SL=51500)", exit_short_sl)
    assert exit_short_sl["should_exit"] and exit_short_sl["exit_reason"] == "stop_loss"

    # ── 9. Evaluate signal — APPROVED ──
    good_signal = {
        "symbol": "BTCUSDT",
        "signal": "LONG",
        "direction": 1,
        "confidence": 0.72,
        "agreement": 0.80,
        "entry_price": 50000.0,
        "stop_loss": 48500.0,
        "take_profit": 53000.0,
        "atr": 1500.0,
        "atr_pct": 3.0,
    }
    ev_good = rm.evaluate_signal(good_signal)
    _p("9. Evaluate GOOD signal", ev_good)
    assert ev_good["approved"], f"Good signal should be approved: {ev_good['reason']}"

    # ── 10. Evaluate signal — REJECTED (low confidence) ──
    weak_signal = {
        "symbol": "ETHUSDT",
        "signal": "SHORT",
        "direction": -1,
        "confidence": 0.40,
        "agreement": 0.50,
        "entry_price": 3000.0,
        "atr": 100.0,
        "atr_pct": 3.3,
    }
    ev_weak = rm.evaluate_signal(weak_signal)
    _p("10. Evaluate WEAK signal (low confidence)", ev_weak)
    assert not ev_weak["approved"]

    # ── 11. Evaluate HOLD signal ──
    hold_signal = {"symbol": "SOLUSDT", "signal": "HOLD", "direction": 0, "confidence": 0.9}
    ev_hold = rm.evaluate_signal(hold_signal)
    _p("11. Evaluate HOLD signal", ev_hold)
    assert not ev_hold["approved"]

    # ── 12. Circuit breakers — normal ──
    cb = rm.check_circuit_breakers()
    _p("12. Circuit Breakers (clean state)", cb)
    assert cb["can_trade"], "Should be able to trade with clean state"

    # ── 13. Record losses → trigger consecutive loss breaker ──
    rm_loss = RiskManager(capital=10000.0, mode="backtest")
    sim_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    rm_loss.set_simulation_time(sim_time)

    for i in range(5):
        rm_loss.record_trade_result(
            pnl_usd=-100.0,
            is_win=False,
            timestamp=sim_time + timedelta(hours=i),
        )
    rm_loss.set_simulation_time(sim_time + timedelta(hours=5))

    cb_loss = rm_loss.check_circuit_breakers()
    _p("13. Circuit Breakers after 5 losses", cb_loss)
    assert not cb_loss["can_trade"], "Should NOT trade after 5 consecutive losses"
    assert rm_loss._consecutive_losses == 5

    # ── 14. Record a win → reset consecutive losses ──
    rm_loss.record_trade_result(pnl_usd=50.0, is_win=True)
    assert rm_loss._consecutive_losses == 0, "Win should reset consecutive losses"
    print("\n  14. Win resets consecutive losses ✓")

    # ── 15. Max drawdown breaker ──
    rm_dd = RiskManager(capital=10000.0, mode="backtest")
    rm_dd.current_capital = 8400.0  # 16% drawdown from 10k
    cb_dd = rm_dd.check_circuit_breakers()
    _p("15. Circuit Breakers — 16% drawdown", cb_dd)
    assert not cb_dd["can_trade"], "16% DD should trigger breaker (limit=15%)"

    # ── 16. Cooldown check ──
    rm_cool = RiskManager(capital=10000.0, mode="backtest")
    now_utc = datetime.now(timezone.utc)
    rm_cool.last_loss_time = now_utc - timedelta(minutes=10)
    rm_cool._sim_time = now_utc
    cool = rm_cool._check_cooldown()
    _p("16. Cooldown (10 min after loss, 30 min cooldown)", cool)
    assert not cool["passed"], "Should still be in cooldown"

    # ── 17. Max positions ──
    rm_pos = RiskManager(capital=10000.0, mode="backtest")
    rm_pos._open_positions = 3
    pos_chk = rm_pos._check_max_positions()
    _p("17. Max Positions (3/3 open)", pos_chk)
    assert not pos_chk["passed"]

    rm_pos.record_position_close()
    pos_chk2 = rm_pos._check_max_positions()
    print(f"  17b. After close: {pos_chk2['open_count']}/{pos_chk2['max_allowed']} → passed={pos_chk2['passed']}")
    assert pos_chk2["passed"]

    # ── 18. Risk summary ──
    rm_sum = RiskManager(capital=10000.0, mode="backtest")
    rm_sum.record_trade_result(150.0, True)
    rm_sum.record_trade_result(-80.0, False)
    rm_sum.record_trade_result(200.0, True)
    summary = rm_sum.get_risk_summary()
    _p("18. Risk Summary (3 trades: +150, -80, +200)", summary)
    assert summary["total_trades"] == 3
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert abs(summary["total_pnl"] - 270.0) < 0.01

    # ── 19. Reset ──
    rm_sum.reset(capital=5000.0)
    assert rm_sum.current_capital == 5000.0
    assert rm_sum._consecutive_losses == 0
    assert len(rm_sum._trade_results) == 0
    print("\n  19. Reset to $5000 ✓")

    # ── 20. Exit conditions (tick-level) ──
    exit_tick = rm.check_exit_conditions(
        entry_price=50000, current_price=56500, direction=1,
        stop_loss=47000, take_profit=56000,
    )
    _p("20. Exit Conditions — TP hit (tick)", exit_tick)
    assert exit_tick["should_exit"] and exit_tick["exit_reason"] == "take_profit"

    # ── Done ──
    print("\n" + "█" * 60)
    print("  ALL RISK MANAGER TESTS PASSED ✅")
    print("█" * 60 + "\n")