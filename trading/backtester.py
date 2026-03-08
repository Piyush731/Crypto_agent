"""
Crypto Futures AI Agent - Backtester
=====================================
Walk-forward futures backtesting with full risk management.

Pipeline:
  1. Fetch data → build features + targets
  2. Walk-forward: expanding-window train, predict on unseen test window
  3. Simulate trades bar-by-bar (SL/TP/trailing/circuit breakers via RiskManager)
  4. Calculate metrics: Sharpe, max DD, win rate, profit factor, alpha vs buy&hold
  5. Save results to SQLite

Design:
  - One position at a time per symbol (no stacking)
  - SL checked BEFORE TP on each bar (conservative / worst-case first)
  - Entry at bar close + slippage; exit at SL/TP/trailing/max-holding/end-of-data
  - Fresh EnsemblePredictor per walk-forward window → no look-ahead bias
  - Fresh RiskManager per simulation → fully isolated capital tracking
  - Circuit breaker cooldown: after trigger, pause for N bars then auto-resume
    (prevents consecutive-loss breaker from permanently halting backtest)
  - Max drawdown: permanent halt if >= max_total_drawdown_pct
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import (
    BACKTEST_CONFIG,
    ACTIVE_HORIZON,
    PREDICTION_HORIZONS,
    TRADING_PAIRS,
)
from core.logger import get_logger
from core.db import get_db
from features.builder import FeatureBuilder
from models.ensemble import EnsemblePredictor
from trading.risk_manager import RiskManager

logger = get_logger(__name__)


class Backtester:
    """
    Walk-forward futures backtester with risk management.

    Usage:
        bt = Backtester(capital=10000)
        result = bt.run("BTCUSDT")
        print(result['metrics'])
    """

    # ── Walk-forward defaults ──
    MIN_TRAIN_SAMPLES = 200   # Minimum bars to train on
    TEST_WINDOW = 50          # Bars per test window
    MAX_HOLDING_BARS = 168    # Safety net: force-close after N bars (1 week @ 1h)

    # ── Circuit breaker auto-recovery (backtest only) ──
    # After a circuit breaker triggers, pause for N bars then auto-resume.
    # Prevents consecutive-loss breaker from permanently halting the backtest.
    # After cooldown: resets consecutive_losses=0, clears loss cooldown timer.
    # Max drawdown is handled separately as a permanent halt.
    BREAKER_COOLDOWN_BARS = 15  # 15 bars = 15 hours @ 1h TF

    def __init__(self, capital: float = None):
        """
        Args:
            capital: Starting capital (USDT). Default from BACKTEST_CONFIG.
        """
        self.initial_capital = capital or BACKTEST_CONFIG["initial_capital"]
        self.commission_pct = BACKTEST_CONFIG["commission_pct"]
        self.slippage_pct = BACKTEST_CONFIG["slippage_pct"]
        self.use_leverage = BACKTEST_CONFIG.get("use_leverage", True)
        self.default_leverage = BACKTEST_CONFIG.get("default_leverage", 3)

        self.feature_builder = FeatureBuilder()
        self.db = get_db()

        # Stored results per symbol
        self._results: Dict[str, Dict] = {}

        logger.info(
            f"Backtester initialized | capital=${self.initial_capital:,.2f} | "
            f"commission={self.commission_pct}% | slippage={self.slippage_pct}%"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLIC: run / run_all
    # ═══════════════════════════════════════════════════════════════════

    def run(self, symbol: str, dataset: Dict = None, horizon: str = None) -> Dict:
        """
        Full backtest for one symbol.

        Args:
            symbol:  Trading pair, e.g. "BTCUSDT"
            dataset: Pre-fetched dataset from DataManager (optional)
            horizon: Prediction horizon key (default: ACTIVE_HORIZON)

        Returns:
            Dict with keys:
                success, symbol, horizon, total_bars, date_range,
                trade_count, trades, equity_curve, metrics,
                walk_forward, signal_distribution, rejection_summary, total_time_s
        """
        t0 = time.time()
        horizon = horizon or ACTIVE_HORIZON

        logger.info(f"{'='*60}")
        logger.info(f"  BACKTEST START: {symbol} | horizon={horizon}")
        logger.info(f"{'='*60}")

        try:
            # ── 1. Get data ──
            if dataset is None:
                from data.manager import DataManager

                dm = DataManager()
                dataset = dm.get_full_dataset(symbol, include_news=False)

            if not dataset or "ohlcv" not in dataset:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "No data available",
                    "total_time_s": round(time.time() - t0, 2),
                }

            # ── 2. Build features + target ──
            logger.info(f"[{symbol}] Building features...")
            build = self.feature_builder.build_features(
                dataset, horizon=horizon, include_target=True
            )

            if "error" in build:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": build["error"],
                    "total_time_s": round(time.time() - t0, 2),
                }

            features = build.get("features")
            target = build.get("target")
            prices = build.get("prices")

            if features is None or features.empty:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Feature building produced empty result",
                    "total_time_s": round(time.time() - t0, 2),
                }

            n = len(features)
            min_needed = self.MIN_TRAIN_SAMPLES + self.TEST_WINDOW

            if n < min_needed:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": (
                        f"Insufficient data: {n} bars (need ≥{min_needed})"
                    ),
                    "total_time_s": round(time.time() - t0, 2),
                }

            logger.info(
                f"[{symbol}] Data ready: {n} bars | "
                f"{len(build.get('feature_names', []))} features | "
                f"target balance: {build.get('target_info', {}).get('balance', '?')}"
            )

            # ── 3. Walk-forward backtest ──
            result = self._walk_forward_backtest(features, target, prices, symbol)

            # Enrich result
            result["symbol"] = symbol
            result["horizon"] = horizon
            result["horizon_bars"] = PREDICTION_HORIZONS.get(horizon, 24)
            result["total_bars"] = n
            result["feature_count"] = len(build.get("feature_names", []))
            result["date_range"] = {
                "start": str(prices.index[0]),
                "end": str(prices.index[-1]),
            }
            result["total_time_s"] = round(time.time() - t0, 2)

            # ── 4. Store + log ──
            self._results[symbol] = result

            if result.get("success"):
                self._save_to_db(result)
                self._log_summary(result)

            return result

        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}", exc_info=True)
            return {
                "success": False,
                "symbol": symbol,
                "error": str(e),
                "total_time_s": round(time.time() - t0, 2),
            }

    def run_all(self, symbols: List[str] = None, horizon: str = None) -> Dict:
        """
        Run backtest for multiple symbols sequentially.

        Args:
            symbols: List of pairs (default: TRADING_PAIRS)
            horizon: Prediction horizon key

        Returns:
            Dict: per_symbol results + summary
        """
        symbols = symbols or TRADING_PAIRS
        t0 = time.time()

        logger.info(f"BACKTEST ALL: {len(symbols)} symbols — {symbols}")

        per_symbol = {}
        for sym in symbols:
            result = self.run(sym, horizon=horizon)
            per_symbol[sym] = result

        successful = [s for s, r in per_symbol.items() if r.get("success")]
        failed = [s for s, r in per_symbol.items() if not r.get("success")]

        avg_return = 0.0
        avg_sharpe = 0.0
        avg_winrate = 0.0
        total_trades = 0
        if successful:
            avg_return = float(np.mean([
                per_symbol[s].get("metrics", {}).get("total_return_pct", 0)
                for s in successful
            ]))
            avg_sharpe = float(np.mean([
                per_symbol[s].get("metrics", {}).get("sharpe_ratio", 0)
                for s in successful
            ]))
            avg_winrate = float(np.mean([
                per_symbol[s].get("metrics", {}).get("win_rate_pct", 0)
                for s in successful
            ]))
            total_trades = sum(
                per_symbol[s].get("metrics", {}).get("total_trades", 0)
                for s in successful
            )

        return {
            "per_symbol": per_symbol,
            "summary": {
                "total_symbols": len(symbols),
                "successful": len(successful),
                "failed": len(failed),
                "failed_symbols": failed,
                "avg_return_pct": round(avg_return, 2),
                "avg_sharpe": round(avg_sharpe, 3),
                "avg_win_rate_pct": round(avg_winrate, 1),
                "total_trades": total_trades,
                "total_time_s": round(time.time() - t0, 2),
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    #  WALK-FORWARD ENGINE
    # ═══════════════════════════════════════════════════════════════════

    def _walk_forward_backtest(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        prices: pd.DataFrame,
        symbol: str,
    ) -> Dict:
        """
        Walk-forward: generate predictions on expanding window, then simulate.

        Phase 1: Expanding-window train → predict on each unseen test chunk
        Phase 2: Bar-by-bar trade simulation across all predictions
        Phase 3: Performance metric calculation
        """
        n = len(features)

        # ════════════════════════════════════════════════
        #  Phase 1: Walk-forward prediction generation
        # ════════════════════════════════════════════════
        logger.info(f"[{symbol}] Phase 1: Walk-forward predictions")

        # Pre-allocate prediction storage
        pred_signals = pd.Series(index=features.index, dtype=object)
        pred_directions = pd.Series(index=features.index, dtype=float)
        pred_confidences = pd.Series(index=features.index, dtype=float)
        pred_prob_ups = pd.Series(index=features.index, dtype=float)

        window_results: List[Dict] = []
        model = EnsemblePredictor()
        step = 0
        train_end = self.MIN_TRAIN_SAMPLES

        while train_end < n:
            test_end = min(train_end + self.TEST_WINDOW, n)
            test_size = test_end - train_end

            if test_size < 5:
                break

            X_train = features.iloc[:train_end]
            y_train = target.iloc[:train_end]
            X_test = features.iloc[train_end:test_end]
            y_test = target.iloc[train_end:test_end]

            # Train fresh model on all available history
            train_info = model.train(X_train, y_train)

            if "error" in train_info:
                logger.warning(
                    f"[{symbol}] Window {step}: train failed — {train_info['error']}"
                )
                train_end = test_end
                step += 1
                continue

            # Evaluate accuracy on test set
            eval_info = model.evaluate(X_test, y_test)
            accuracy = eval_info.get("accuracy", 0)
            f1 = eval_info.get("f1_score", 0)

            # Batch predictions
            batch = model.predict_batch(X_test)

            if "error" not in batch and batch.get("count", 0) > 0:
                for j, idx in enumerate(X_test.index):
                    pred_signals.at[idx] = batch["signals"][j]
                    pred_directions.at[idx] = (
                        1 if batch["signals"][j] == "LONG"
                        else (-1 if batch["signals"][j] == "SHORT" else 0)
                    )
                    pred_confidences.at[idx] = float(batch["confidence"][j])
                    pred_prob_ups.at[idx] = float(batch["probability_up"][j])

            window_results.append({
                "window": step,
                "train_size": train_end,
                "test_size": test_size,
                "accuracy": round(accuracy, 4),
                "f1": round(f1, 4),
            })

            logger.debug(
                f"[{symbol}] Window {step}: train={train_end} "
                f"test={test_size} acc={accuracy:.3f} f1={f1:.3f}"
            )

            train_end = test_end
            step += 1

        # Combine into a predictions DataFrame (drop bars without predictions)
        predictions = pd.DataFrame({
            "signal": pred_signals,
            "direction": pred_directions,
            "confidence": pred_confidences,
            "probability_up": pred_prob_ups,
        })
        predictions = predictions.dropna(subset=["signal"])

        if predictions.empty:
            return {"success": False, "error": "No predictions generated (all windows failed)"}

        predictions["direction"] = predictions["direction"].astype(int)
        predictions["confidence"] = predictions["confidence"].astype(float)
        predictions["probability_up"] = predictions["probability_up"].astype(float)

        sig_counts = predictions["signal"].value_counts().to_dict()
        avg_accuracy = (
            float(np.mean([w["accuracy"] for w in window_results]))
            if window_results else 0.0
        )

        logger.info(
            f"[{symbol}] Phase 1 done: {len(predictions)} predictions | "
            f"{step} windows | avg_acc={avg_accuracy:.3f} | signals={sig_counts}"
        )

        # ════════════════════════════════════════════════
        #  Phase 2: Trade simulation
        # ════════════════════════════════════════════════
        logger.info(f"[{symbol}] Phase 2: Trade simulation")

        sim_result = self._simulate_trades(predictions, prices, features, symbol)

        trades = sim_result["trades"]
        equity_curve = sim_result["equity_curve"]
        rejection_summary = sim_result["rejection_summary"]

        logger.info(
            f"[{symbol}] Phase 2 done: {len(trades)} trades executed | "
            f"breaker cooldowns: {sim_result['breaker_triggers']} | "
            f"rejected: {sum(rejection_summary.values())}"
        )

        # ════════════════════════════════════════════════
        #  Phase 3: Metrics
        # ════════════════════════════════════════════════
        logger.info(f"[{symbol}] Phase 3: Metrics")

        pred_prices = prices.loc[prices.index.isin(predictions.index)]
        metrics = self._calculate_metrics(trades, equity_curve, pred_prices)

        return {
            "success": True,
            "trade_count": len(trades),
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "walk_forward": {
                "windows": step,
                "total_predictions": len(predictions),
                "avg_model_accuracy": round(avg_accuracy, 4),
                "window_results": window_results,
            },
            "signal_distribution": sig_counts,
            "rejection_summary": rejection_summary,
            "breaker_triggers": sim_result["breaker_triggers"],
            "max_dd_halted": sim_result["max_dd_halted"],
        }

    # ═══════════════════════════════════════════════════════════════════
    #  TRADE SIMULATION
    # ═══════════════════════════════════════════════════════════════════

    def _simulate_trades(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        symbol: str,
    ) -> Dict:
        """
        Bar-by-bar trade simulation with risk management.

        - One position at a time per symbol.
        - Exits: SL, TP, trailing stop, max holding, end-of-data.
        - Fresh RiskManager for isolated capital tracking.
        - Circuit breaker cooldown: pause N bars, then auto-resume
          (resets consecutive losses to prevent permanent lockout).
        - Max drawdown: permanent halt.

        Returns:
            Dict: trades, equity_curve, rejection_summary,
                  breaker_triggers, max_dd_halted
        """
        # Fresh, isolated risk manager
        risk_mgr = RiskManager(capital=self.initial_capital, mode="backtest")

        capital = self.initial_capital
        trades: List[Dict] = []
        equity_curve: List[Dict] = []
        position: Optional[Dict] = None

        # ── Circuit breaker cooldown state ──
        breaker_cooldown = 0        # Bars remaining in cooldown
        breaker_triggers = 0        # Total times breaker triggered
        max_dd_halted = False       # Permanent halt due to max drawdown

        # ── Rejection tracking ──
        rejection_counts = {
            "circuit_breaker": 0,
            "low_confidence": 0,
            "cooldown_skip": 0,
            "max_dd_halt": 0,
            "hold_signal": 0,
            "other": 0,
        }

        pred_indices = predictions.index.tolist()

        for i, idx in enumerate(pred_indices):
            # ── Get bar data ──
            if idx not in prices.index:
                continue

            bar = prices.loc[idx]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            pred = predictions.loc[idx]

            # ── Simulation timestamp ──
            sim_time = self._to_utc_datetime(idx)
            risk_mgr.set_simulation_time(sim_time)

            # ════════════════════════════════════════
            #  A. CHECK EXIT for existing position
            # ════════════════════════════════════════
            if position is not None:
                exit_result = risk_mgr.check_bar_exit(
                    entry_price=position["entry_price"],
                    bar_high=bar_high,
                    bar_low=bar_low,
                    bar_close=bar_close,
                    direction=position["direction"],
                    stop_loss=position["stop_loss"],
                    take_profit=position["take_profit"],
                    highest_since_entry=position["highest"],
                    lowest_since_entry=position["lowest"],
                    current_stop=position["current_stop"],
                )

                # Update price tracking
                position["highest"] = max(position["highest"], bar_high)
                position["lowest"] = min(position["lowest"], bar_low)
                position["holding_bars"] += 1

                should_close = exit_result["should_exit"]
                exit_reason = exit_result.get("exit_reason")
                exit_price = exit_result.get("exit_price", bar_close)

                # Max holding period safety net
                if not should_close and position["holding_bars"] >= self.MAX_HOLDING_BARS:
                    should_close = True
                    exit_reason = "max_holding"
                    exit_price = bar_close

                if should_close:
                    trade = self._close_position(
                        position, exit_price, exit_reason, capital, sim_time
                    )
                    trades.append(trade)

                    capital += trade["net_pnl_usd"]
                    is_win = trade["net_pnl_usd"] > 0

                    risk_mgr.record_trade_result(
                        trade["net_pnl_usd"], is_win, sim_time
                    )
                    risk_mgr.record_position_close()
                    risk_mgr.update_capital(capital)

                    position = None
                else:
                    # Update trailing stop
                    position["current_stop"] = exit_result.get(
                        "new_trailing_stop", position["current_stop"]
                    )

            # ════════════════════════════════════════
            #  B. CHECK ENTRY if no open position
            # ════════════════════════════════════════
            if position is None and capital > 0:

                # B1. Max drawdown hard stop (permanent, no recovery)
                if not max_dd_halted:
                    peak = risk_mgr.peak_capital
                    if peak > 0:
                        dd_pct = (peak - capital) / peak * 100.0
                        if dd_pct >= risk_mgr.max_total_dd_pct:
                            max_dd_halted = True
                            logger.warning(
                                f"[{symbol}] MAX DRAWDOWN {dd_pct:.1f}% "
                                f"(limit {risk_mgr.max_total_dd_pct}%) — "
                                f"permanently halting backtest"
                            )

                if max_dd_halted:
                    rejection_counts["max_dd_halt"] += 1
                    # Permanent halt — no more entries ever

                # B2. Circuit breaker cooldown (temporary pause)
                elif breaker_cooldown > 0:
                    breaker_cooldown -= 1
                    rejection_counts["cooldown_skip"] += 1

                    if breaker_cooldown == 0:
                        # Cooldown expired → reset state and resume
                        risk_mgr._consecutive_losses = 0
                        risk_mgr.last_loss_time = None
                        logger.info(
                            f"[{symbol}] Circuit breaker cooldown expired "
                            f"→ resuming trading (trigger #{breaker_triggers})"
                        )

                # B3. Normal entry evaluation
                else:
                    signal_str = str(pred["signal"])
                    direction = int(pred["direction"])
                    confidence = float(pred["confidence"])

                    if signal_str not in ("LONG", "SHORT") or direction == 0:
                        rejection_counts["hold_signal"] += 1
                    else:
                        atr, atr_pct = self._extract_atr(
                            features, idx, bar_close
                        )

                        signal_dict = {
                            "symbol": symbol,
                            "signal": signal_str,
                            "direction": direction,
                            "confidence": confidence,
                            "agreement": min(confidence * 1.2, 1.0),
                            "entry_price": bar_close,
                            "atr": atr,
                            "atr_pct": atr_pct,
                        }

                        eval_result = risk_mgr.evaluate_signal(signal_dict)

                        if eval_result["approved"]:
                            # ── OPEN POSITION ──
                            slip = bar_close * (self.slippage_pct / 100.0)
                            entry_price = (
                                bar_close + slip if direction == 1
                                else bar_close - slip
                            )

                            entry_comm = (
                                eval_result["position_size_usd"]
                                * (self.commission_pct / 100.0)
                            )

                            position = {
                                "symbol": symbol,
                                "entry_time": sim_time,
                                "entry_price": entry_price,
                                "entry_bar_close": bar_close,
                                "direction": direction,
                                "signal": signal_str,
                                "confidence": confidence,
                                "position_size_usd": eval_result[
                                    "position_size_usd"
                                ],
                                "leverage": eval_result["leverage"],
                                "stop_loss": eval_result["stop_loss"],
                                "take_profit": eval_result["take_profit"],
                                "current_stop": eval_result["stop_loss"],
                                "highest": bar_high,
                                "lowest": bar_low,
                                "holding_bars": 0,
                                "entry_commission": entry_comm,
                            }

                            risk_mgr.record_position_open()

                        else:
                            # ── REJECTED — categorize reason ──
                            reason = eval_result.get("reason", "")
                            reason_lower = reason.lower()

                            if "circuit breaker" in reason_lower:
                                # Trigger cooldown instead of spamming
                                breaker_cooldown = self.BREAKER_COOLDOWN_BARS
                                breaker_triggers += 1
                                rejection_counts["circuit_breaker"] += 1

                                logger.info(
                                    f"[{symbol}] Circuit breaker triggered "
                                    f"(#{breaker_triggers}) — "
                                    f"cooldown {self.BREAKER_COOLDOWN_BARS} bars | "
                                    f"reason: {reason}"
                                )
                            elif (
                                "confidence" in reason_lower
                                or "agreement" in reason_lower
                            ):
                                rejection_counts["low_confidence"] += 1
                            else:
                                rejection_counts["other"] += 1

            # ════════════════════════════════════════
            #  C. RECORD EQUITY (mark-to-market)
            # ════════════════════════════════════════
            if position is not None:
                unrealized = (
                    position["position_size_usd"]
                    * (bar_close - position["entry_price"])
                    / position["entry_price"]
                    * position["direction"]
                )
                equity = capital + unrealized - position["entry_commission"]
            else:
                equity = capital

            equity_curve.append({
                "timestamp": sim_time,
                "equity": round(equity, 2),
                "capital": round(capital, 2),
            })

        # ════════════════════════════════════════
        #  D. FORCE CLOSE remaining position
        # ════════════════════════════════════════
        if position is not None and pred_indices:
            last_idx = pred_indices[-1]
            if last_idx in prices.index:
                last_close = float(prices.loc[last_idx, "close"])
            else:
                last_close = position["entry_price"]

            last_time = (
                equity_curve[-1]["timestamp"]
                if equity_curve
                else datetime.now(timezone.utc)
            )

            trade = self._close_position(
                position, last_close, "end_of_data", capital, last_time
            )
            trades.append(trade)

            capital += trade["net_pnl_usd"]

            # Correct final equity point
            if equity_curve:
                equity_curve[-1]["equity"] = round(capital, 2)
                equity_curve[-1]["capital"] = round(capital, 2)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "rejection_summary": rejection_counts,
            "breaker_triggers": breaker_triggers,
            "max_dd_halted": max_dd_halted,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  POSITION CLOSE
    # ═══════════════════════════════════════════════════════════════════

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_reason: str,
        current_capital: float,
        exit_time: datetime,
    ) -> Dict:
        """Build a completed-trade record and compute PnL."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        size_usd = position["position_size_usd"]

        # Gross PnL (notional × price change fraction × direction)
        price_frac = (exit_price - entry_price) / entry_price * direction
        gross_pnl = size_usd * price_frac

        # Commissions (entry already stored, exit calculated now)
        entry_comm = position["entry_commission"]
        exit_comm = size_usd * (self.commission_pct / 100.0)
        total_comm = entry_comm + exit_comm

        net_pnl = gross_pnl - total_comm

        # Percentages
        pnl_pct = (net_pnl / current_capital * 100.0) if current_capital > 0 else 0.0
        price_pnl_pct = price_frac * 100.0
        leveraged_pnl_pct = price_pnl_pct * position["leverage"]

        return {
            "symbol": position.get("symbol", ""),
            "signal": position["signal"],
            "direction": direction,
            "confidence": position["confidence"],
            "leverage": position["leverage"],
            "entry_time": position["entry_time"],
            "entry_price": round(entry_price, 8),
            "exit_time": exit_time,
            "exit_price": round(exit_price, 8),
            "exit_reason": exit_reason,
            "position_size_usd": round(size_usd, 2),
            "stop_loss": round(position["stop_loss"], 8),
            "take_profit": round(position["take_profit"], 8),
            "holding_bars": position["holding_bars"],
            "gross_pnl_usd": round(gross_pnl, 4),
            "commission_usd": round(total_comm, 4),
            "net_pnl_usd": round(net_pnl, 4),
            "pnl_pct": round(pnl_pct, 4),
            "price_pnl_pct": round(price_pnl_pct, 4),
            "leveraged_pnl_pct": round(leveraged_pnl_pct, 4),
            "highest_price": round(position["highest"], 8),
            "lowest_price": round(position["lowest"], 8),
        }

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_utc_datetime(idx) -> datetime:
        """Convert pandas index element to timezone-aware datetime."""
        try:
            if hasattr(idx, "to_pydatetime"):
                dt = idx.to_pydatetime()
            elif isinstance(idx, datetime):
                dt = idx
            else:
                dt = pd.Timestamp(idx).to_pydatetime()

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _extract_atr(
        features: pd.DataFrame, idx, close_price: float
    ) -> Tuple[float, float]:
        """
        Extract ATR from features row.

        Falls back to 2% of price if not available.
        """
        try:
            if idx in features.index:
                row = features.loc[idx]
                for col in ("atr_14", "atr_20", "atr"):
                    if col in features.columns and pd.notna(row.get(col)):
                        atr = float(row[col])
                        atr_pct = (
                            atr / close_price * 100.0 if close_price > 0 else 2.0
                        )
                        return atr, atr_pct
                if "atr_pct" in features.columns and pd.notna(row.get("atr_pct")):
                    atr_pct = float(row["atr_pct"])
                    atr = close_price * atr_pct / 100.0
                    return atr, atr_pct
        except Exception:
            pass

        # Fallback
        atr = close_price * 0.02
        return atr, 2.0

    # ═══════════════════════════════════════════════════════════════════
    #  METRICS CALCULATION
    # ═══════════════════════════════════════════════════════════════════

    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        prices: pd.DataFrame,
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "total_return_pct": 0.0,
            "total_pnl_usd": 0.0,
            "avg_pnl_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "largest_win_usd": 0.0,
            "largest_loss_usd": 0.0,
            "profit_factor": 0.0,
            "expectancy_usd": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_usd": 0.0,
            "avg_holding_bars": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate_pct": 0.0,
            "short_win_rate_pct": 0.0,
            "long_pnl_usd": 0.0,
            "short_pnl_usd": 0.0,
            "buy_hold_return_pct": 0.0,
            "alpha_pct": 0.0,
            "total_commission_usd": 0.0,
            "initial_capital": self.initial_capital,
            "final_capital": self.initial_capital,
            "exit_reasons": {},
        }

        if not trades:
            if equity_curve:
                metrics["final_capital"] = equity_curve[-1].get(
                    "equity", self.initial_capital
                )
            return metrics

        # ── Basic trade stats ──
        n_trades = len(trades)
        wins = [t for t in trades if t["net_pnl_usd"] > 0]
        losses = [t for t in trades if t["net_pnl_usd"] <= 0]
        longs = [t for t in trades if t["direction"] == 1]
        shorts = [t for t in trades if t["direction"] == -1]
        long_wins = [t for t in longs if t["net_pnl_usd"] > 0]
        short_wins = [t for t in shorts if t["net_pnl_usd"] > 0]

        total_pnl = sum(t["net_pnl_usd"] for t in trades)
        total_comm = sum(t["commission_usd"] for t in trades)
        final = self.initial_capital + total_pnl

        metrics["total_trades"] = n_trades
        metrics["winning_trades"] = len(wins)
        metrics["losing_trades"] = len(losses)
        metrics["win_rate_pct"] = round(len(wins) / n_trades * 100.0, 2)
        metrics["total_pnl_usd"] = round(total_pnl, 2)
        metrics["avg_pnl_usd"] = round(total_pnl / n_trades, 2)
        metrics["total_return_pct"] = round(
            total_pnl / self.initial_capital * 100.0, 2
        )
        metrics["total_commission_usd"] = round(total_comm, 2)
        metrics["final_capital"] = round(final, 2)

        # ── Win / loss breakdowns ──
        if wins:
            win_pnls = [t["net_pnl_usd"] for t in wins]
            metrics["avg_win_usd"] = round(float(np.mean(win_pnls)), 2)
            metrics["largest_win_usd"] = round(max(win_pnls), 2)

        if losses:
            loss_pnls = [t["net_pnl_usd"] for t in losses]
            metrics["avg_loss_usd"] = round(float(np.mean(loss_pnls)), 2)
            metrics["largest_loss_usd"] = round(min(loss_pnls), 2)

        # ── Profit factor ──
        gross_profit = sum(t["net_pnl_usd"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["net_pnl_usd"] for t in losses)) if losses else 0
        metrics["profit_factor"] = round(
            (gross_profit / gross_loss) if gross_loss > 0 else
            (float("inf") if gross_profit > 0 else 0.0),
            3,
        )

        # ── Expectancy ──
        win_rate = len(wins) / n_trades
        avg_w = float(np.mean([t["net_pnl_usd"] for t in wins])) if wins else 0
        avg_l = abs(float(np.mean([t["net_pnl_usd"] for t in losses]))) if losses else 0
        metrics["expectancy_usd"] = round(
            win_rate * avg_w - (1 - win_rate) * avg_l, 2
        )

        # ── Holding period ──
        holding = [t["holding_bars"] for t in trades]
        metrics["avg_holding_bars"] = round(float(np.mean(holding)), 1)

        # ── Long / Short ──
        metrics["long_trades"] = len(longs)
        metrics["short_trades"] = len(shorts)
        metrics["long_win_rate_pct"] = round(
            len(long_wins) / len(longs) * 100.0 if longs else 0.0, 2
        )
        metrics["short_win_rate_pct"] = round(
            len(short_wins) / len(shorts) * 100.0 if shorts else 0.0, 2
        )
        metrics["long_pnl_usd"] = round(
            sum(t["net_pnl_usd"] for t in longs), 2
        )
        metrics["short_pnl_usd"] = round(
            sum(t["net_pnl_usd"] for t in shorts), 2
        )

        # ── Exit reasons ──
        exit_reasons: Dict[str, int] = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        metrics["exit_reasons"] = exit_reasons

        # ── Equity-curve based metrics ──
        if equity_curve and len(equity_curve) > 1:
            eq_vals = np.array(
                [e["equity"] for e in equity_curve], dtype=float
            )

            # Per-bar returns
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = np.diff(eq_vals) / eq_vals[:-1]
            returns = returns[np.isfinite(returns)]

            if len(returns) > 1:
                periods_per_year = 8760  # Hourly bars

                mean_r = float(np.mean(returns))
                std_r = float(np.std(returns, ddof=1))

                # Sharpe
                if std_r > 1e-12:
                    metrics["sharpe_ratio"] = round(
                        mean_r / std_r * np.sqrt(periods_per_year), 3
                    )

                # Sortino (downside deviation)
                down = returns[returns < 0]
                if len(down) > 0:
                    down_std = float(np.std(down, ddof=1))
                    if down_std > 1e-12:
                        metrics["sortino_ratio"] = round(
                            mean_r / down_std * np.sqrt(periods_per_year), 3
                        )

            # Max drawdown
            dd = self._calculate_max_drawdown(eq_vals)
            metrics["max_drawdown_pct"] = dd["max_dd_pct"]
            metrics["max_drawdown_usd"] = dd["max_dd_usd"]

            # Calmar
            if dd["max_dd_pct"] > 0 and len(eq_vals) > 1:
                total_bars = len(eq_vals)
                try:
                    ann_ret = (
                        (eq_vals[-1] / eq_vals[0])
                        ** (8760 / max(total_bars, 1))
                        - 1
                    ) * 100.0
                    metrics["calmar_ratio"] = round(
                        ann_ret / dd["max_dd_pct"], 3
                    )
                except Exception:
                    pass

        # ── Buy & Hold benchmark ──
        if len(prices) > 1:
            bh = self._calculate_buy_hold(prices)
            metrics["buy_hold_return_pct"] = bh["return_pct"]
            metrics["alpha_pct"] = round(
                metrics["total_return_pct"] - bh["return_pct"], 2
            )

        return metrics

    @staticmethod
    def _calculate_max_drawdown(equity_array: np.ndarray) -> Dict:
        """Calculate max drawdown from an equity array."""
        try:
            running_max = np.maximum.accumulate(equity_array)

            with np.errstate(divide="ignore", invalid="ignore"):
                dd_pct = (running_max - equity_array) / running_max * 100.0
            dd_pct = np.nan_to_num(dd_pct, nan=0.0)

            dd_usd = running_max - equity_array

            max_dd_pct = float(np.max(dd_pct))
            max_dd_usd = float(np.max(dd_usd))

            return {
                "max_dd_pct": round(max_dd_pct, 2),
                "max_dd_usd": round(max_dd_usd, 2),
            }
        except Exception:
            return {"max_dd_pct": 0.0, "max_dd_usd": 0.0}

    def _calculate_buy_hold(self, prices: pd.DataFrame) -> Dict:
        """Buy & hold return for comparison."""
        try:
            first = float(prices["close"].iloc[0])
            last = float(prices["close"].iloc[-1])
            ret = (last - first) / first * 100.0
            lev = self.default_leverage if self.use_leverage else 1

            return {
                "return_pct": round(ret, 2),
                "leveraged_return_pct": round(ret * lev, 2),
                "start_price": round(first, 4),
                "end_price": round(last, 4),
            }
        except Exception:
            return {
                "return_pct": 0.0,
                "leveraged_return_pct": 0.0,
                "start_price": 0,
                "end_price": 0,
            }

    # ═══════════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════

    def _save_to_db(self, result: Dict):
        """Save backtest summary to SQLite performance table."""
        try:
            m = result.get("metrics", {})
            wf = result.get("walk_forward", {})

            data = {
                "mode": "backtest",
                "symbol": result.get("symbol", ""),
                "total_return_pct": m.get("total_return_pct", 0),
                "win_rate_pct": m.get("win_rate_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "total_trades": m.get("total_trades", 0),
                "profit_factor": m.get("profit_factor", 0),
                "notes": json.dumps(
                    {
                        "horizon": result.get("horizon"),
                        "total_bars": result.get("total_bars"),
                        "wf_windows": wf.get("windows"),
                        "avg_accuracy": wf.get("avg_model_accuracy"),
                        "alpha_pct": m.get("alpha_pct"),
                        "signal_dist": result.get("signal_distribution"),
                        "exit_reasons": m.get("exit_reasons"),
                        "rejections": result.get("rejection_summary"),
                        "breaker_triggers": result.get("breaker_triggers"),
                    },
                    default=str,
                ),
            }
            self.db.save_performance(data)
            logger.debug(f"Backtest saved to DB: {result.get('symbol')}")

        except Exception as e:
            logger.error(f"DB save failed: {e}")

    def _log_summary(self, result: Dict):
        """Print a readable backtest summary to the logger."""
        sym = result.get("symbol", "?")
        m = result.get("metrics", {})
        wf = result.get("walk_forward", {})
        dr = result.get("date_range", {})
        rej = result.get("rejection_summary", {})

        lines = [
            f"{'─'*60}",
            f"  BACKTEST RESULTS: {sym}",
            f"{'─'*60}",
            f"  Period:        {dr.get('start','?')} → {dr.get('end','?')}",
            f"  Total bars:    {result.get('total_bars', 0)}",
            f"  WF windows:    {wf.get('windows', 0)} | "
            f"Avg model acc: {wf.get('avg_model_accuracy', 0):.1%}",
            f"  Trades:        {m.get('total_trades', 0)} "
            f"({m.get('long_trades', 0)}L / {m.get('short_trades', 0)}S)",
            f"  Win rate:      {m.get('win_rate_pct', 0):.1f}%",
            f"  Total PnL:     ${m.get('total_pnl_usd', 0):+,.2f} "
            f"({m.get('total_return_pct', 0):+.2f}%)",
            f"  Final capital: ${m.get('final_capital', 0):,.2f}",
            f"  Profit factor: {m.get('profit_factor', 0):.3f}",
            f"  Expectancy:    ${m.get('expectancy_usd', 0):+.2f}/trade",
            f"  Sharpe:        {m.get('sharpe_ratio', 0):.3f}",
            f"  Sortino:       {m.get('sortino_ratio', 0):.3f}",
            f"  Max drawdown:  {m.get('max_drawdown_pct', 0):.2f}% "
            f"(${m.get('max_drawdown_usd', 0):,.2f})",
            f"  Buy&Hold:      {m.get('buy_hold_return_pct', 0):+.2f}%",
            f"  Alpha:         {m.get('alpha_pct', 0):+.2f}%",
            f"  Avg holding:   {m.get('avg_holding_bars', 0):.1f} bars",
            f"  Commission:    ${m.get('total_commission_usd', 0):.2f}",
            f"  Exit reasons:  {m.get('exit_reasons', {})}",
            f"  Rejections:    {rej}",
            f"  Breaker cools: {result.get('breaker_triggers', 0)}",
            f"  Time:          {result.get('total_time_s', 0):.1f}s",
            f"{'─'*60}",
        ]
        for line in lines:
            logger.info(line)

    # ═══════════════════════════════════════════════════════════════════
    #  ACCESSORS
    # ═══════════════════════════════════════════════════════════════════

    def get_results(self, symbol: str = None) -> Optional[Dict]:
        """Get backtest results. None or all."""
        if symbol:
            return self._results.get(symbol)
        return dict(self._results)

    def get_summary(self) -> Dict:
        """Aggregate summary across all completed backtests."""
        if not self._results:
            return {"symbols": 0, "message": "No backtests run yet"}

        rows = []
        for sym, res in self._results.items():
            if not res.get("success"):
                continue
            m = res.get("metrics", {})
            rows.append({
                "symbol": sym,
                "return_pct": m.get("total_return_pct", 0),
                "win_rate_pct": m.get("win_rate_pct", 0),
                "sharpe": m.get("sharpe_ratio", 0),
                "max_dd_pct": m.get("max_drawdown_pct", 0),
                "trades": m.get("total_trades", 0),
                "profit_factor": m.get("profit_factor", 0),
                "alpha_pct": m.get("alpha_pct", 0),
            })

        if not rows:
            return {
                "symbols": len(self._results),
                "successful": 0,
                "message": "All backtests failed",
            }

        return {
            "symbols": len(rows),
            "per_symbol": rows,
            "avg_return_pct": round(float(np.mean([r["return_pct"] for r in rows])), 2),
            "avg_win_rate_pct": round(float(np.mean([r["win_rate_pct"] for r in rows])), 1),
            "avg_sharpe": round(float(np.mean([r["sharpe"] for r in rows])), 3),
            "avg_max_dd_pct": round(float(np.mean([r["max_dd_pct"] for r in rows])), 2),
            "avg_profit_factor": round(float(np.mean([r["profit_factor"] for r in rows])), 3),
            "total_trades": sum(r["trades"] for r in rows),
            "avg_alpha_pct": round(float(np.mean([r["alpha_pct"] for r in rows])), 2),
        }

    @staticmethod
    def print_trades(trades: List[Dict], max_rows: int = 50):
        """Pretty-print trade table to console."""
        if not trades:
            print("  No trades.")
            return

        print(
            f"\n  {'#':>3} {'Signal':<6} {'Entry':>12} {'Exit':>12} "
            f"{'Reason':<14} {'Bars':>4} {'Lev':>3} "
            f"{'Net PnL':>10} {'PnL%':>7} {'Capital%':>8}"
        )
        print("  " + "─" * 92)

        for i, t in enumerate(trades[:max_rows]):
            sig = t.get("signal", "?")
            ep = t.get("entry_price", 0)
            xp = t.get("exit_price", 0)
            xr = t.get("exit_reason", "?")[:13]
            hb = t.get("holding_bars", 0)
            lev = t.get("leverage", 1)
            npnl = t.get("net_pnl_usd", 0)
            ppnl = t.get("price_pnl_pct", 0)
            cpnl = t.get("pnl_pct", 0)
            w = "✓" if npnl > 0 else "✗"

            print(
                f"  {i+1:>3} {sig:<6} {ep:>12.2f} {xp:>12.2f} "
                f"{xr:<14} {hb:>4} {lev:>3}x "
                f"{npnl:>+10.2f} {ppnl:>+6.2f}% {cpnl:>+7.2f}% {w}"
            )

        if len(trades) > max_rows:
            print(f"  ... and {len(trades) - max_rows} more trades")


# ═══════════════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "█" * 60)
    print("  BACKTESTER — FULL TEST")
    print("█" * 60)

    t_start = time.time()

    # ── 1. Create backtester ──
    bt = Backtester(capital=10000.0)
    print(f"\n  ✓ Backtester created | capital=${bt.initial_capital:,.2f}")

    # ── 2. Fetch data ──
    print("\n  Fetching BTCUSDT data from Binance...")
    from data.manager import DataManager

    dm = DataManager()
    dataset = dm.get_full_dataset("BTCUSDT", include_news=False)

    if not dataset or "ohlcv" not in dataset:
        print("  ✗ Failed to fetch data — cannot run backtest")
    else:
        entry_tf = list(dataset["ohlcv"].keys())[0] if dataset["ohlcv"] else "?"
        n_bars = (
            len(list(dataset["ohlcv"].values())[0])
            if dataset["ohlcv"] else 0
        )
        print(f"  ✓ Data fetched: {entry_tf} → {n_bars} bars")

        # ── 3. Run backtest ──
        print("\n  Running walk-forward backtest (this may take a few minutes)...")
        result = bt.run("BTCUSDT", dataset=dataset)

        if result.get("success"):
            m = result.get("metrics", {})
            wf = result.get("walk_forward", {})
            rej = result.get("rejection_summary", {})

            print(f"\n  ✓ BACKTEST COMPLETE")
            print(f"  {'─'*50}")
            print(f"  Walk-forward windows:  {wf.get('windows', 0)}")
            print(f"  Avg model accuracy:    {wf.get('avg_model_accuracy', 0):.1%}")
            print(f"  Total predictions:     {wf.get('total_predictions', 0)}")
            print(f"  Signal distribution:   {result.get('signal_distribution', {})}")
            print(f"  {'─'*50}")
            print(f"  Total trades:          {m.get('total_trades', 0)}")
            print(f"  Win rate:              {m.get('win_rate_pct', 0):.1f}%")
            print(f"  Total PnL:             ${m.get('total_pnl_usd', 0):+,.2f}")
            print(f"  Total return:          {m.get('total_return_pct', 0):+.2f}%")
            print(f"  Final capital:         ${m.get('final_capital', 0):,.2f}")
            print(f"  Profit factor:         {m.get('profit_factor', 0):.3f}")
            print(f"  Expectancy:            ${m.get('expectancy_usd', 0):+.2f}/trade")
            print(f"  Sharpe ratio:          {m.get('sharpe_ratio', 0):.3f}")
            print(f"  Sortino ratio:         {m.get('sortino_ratio', 0):.3f}")
            print(f"  Max drawdown:          {m.get('max_drawdown_pct', 0):.2f}%")
            print(f"  Buy & hold return:     {m.get('buy_hold_return_pct', 0):+.2f}%")
            print(f"  Alpha vs B&H:          {m.get('alpha_pct', 0):+.2f}%")
            print(f"  Avg holding:           {m.get('avg_holding_bars', 0):.1f} bars")
            print(f"  Long:  {m.get('long_trades',0)} trades  "
                  f"WR={m.get('long_win_rate_pct',0):.0f}%  "
                  f"PnL=${m.get('long_pnl_usd',0):+,.2f}")
            print(f"  Short: {m.get('short_trades',0)} trades  "
                  f"WR={m.get('short_win_rate_pct',0):.0f}%  "
                  f"PnL=${m.get('short_pnl_usd',0):+,.2f}")
            print(f"  Commission total:      ${m.get('total_commission_usd', 0):.2f}")
            print(f"  Exit reasons:          {m.get('exit_reasons', {})}")
            print(f"  {'─'*50}")
            print(f"  REJECTION SUMMARY:")
            for reason, count in rej.items():
                if count > 0:
                    print(f"    {reason:<20}: {count}")
            print(f"  Breaker cooldowns:     {result.get('breaker_triggers', 0)}")
            print(f"  Max DD halted:         {result.get('max_dd_halted', False)}")

            # ── 4. Print trade table ──
            trades = result.get("trades", [])
            if trades:
                print(f"\n  TRADE LOG (showing up to 30):")
                Backtester.print_trades(trades, max_rows=30)

            # ── 5. Equity curve snippet ──
            ec = result.get("equity_curve", [])
            if ec:
                print(f"\n  EQUITY CURVE ({len(ec)} points):")
                print(f"    Start: ${ec[0]['equity']:,.2f}")
                mid = len(ec) // 2
                print(f"    Mid:   ${ec[mid]['equity']:,.2f}")
                print(f"    End:   ${ec[-1]['equity']:,.2f}")

        else:
            print(f"\n  ✗ Backtest failed: {result.get('error', 'unknown')}")

    # ── 6. Summary ──
    summary = bt.get_summary()
    print(f"\n  SUMMARY: {json.dumps(summary, indent=2, default=str)}")

    elapsed = time.time() - t_start
    print(f"\n  Total test time: {elapsed:.1f}s")

    print("\n" + "█" * 60)
    print("  BACKTESTER TEST COMPLETE ✅")
    print("█" * 60 + "\n")