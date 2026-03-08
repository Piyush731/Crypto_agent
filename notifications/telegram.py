"""
Crypto Futures AI Agent - Telegram Notifications
==================================================
Send trading signals, daily reports, error alerts via Telegram Bot API.

Features:
  - Rich formatted signal notifications (entry, SL, TP, confidence, components)
  - Daily/weekly performance reports
  - Paper trade open/close alerts
  - Backtest result summaries
  - Error / warning alerts
  - Quiet hours support
  - Rate limiting (max 20 msgs/min)
  - Message chunking for long content (Telegram 4096 char limit)
  - All free via Telegram Bot API
"""

import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_CONFIG,
)
from core.logger import get_logger

logger = get_logger(__name__)

# Telegram API limits
MAX_MESSAGE_LENGTH = 4096
MAX_MESSAGES_PER_MINUTE = 20


class TelegramNotifier:
    """
    Telegram Bot notification sender.

    Usage:
        tg = TelegramNotifier()
        tg.send_signal(signal_data)
        tg.send_daily_report(performance_data)
    """

    def __init__(self):
        """Initialize with config settings."""
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = TELEGRAM_CONFIG.get("enabled", False)

        self.send_signals_flag = TELEGRAM_CONFIG.get("send_signals", True)
        self.send_daily_flag = TELEGRAM_CONFIG.get("send_daily_report", True)
        self.send_errors_flag = TELEGRAM_CONFIG.get("send_errors", True)
        self.send_retrain_flag = TELEGRAM_CONFIG.get("send_model_retrain", False)
        self.quiet_hours = TELEGRAM_CONFIG.get("quiet_hours", None)

        self.base_url = f"https://api.telegram.org/bot{self.token}"

        # Rate limiting
        self._msg_timestamps: List[float] = []
        self._total_sent = 0
        self._total_errors = 0

        if self.enabled:
            logger.info("TelegramNotifier initialized ✅")
        else:
            logger.info(
                "TelegramNotifier initialized (DISABLED — "
                "set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)"
            )

    # ═══════════════════════════════════════════════════════════════════
    #  CORE: send_message
    # ═══════════════════════════════════════════════════════════════════

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_preview: bool = True,
        silent: bool = False,
    ) -> bool:
        """
        Send a message via Telegram Bot API.

        Handles: rate limiting, message chunking, quiet hours, retries.

        Args:
            text:            Message text (HTML or Markdown)
            parse_mode:      "HTML" or "MarkdownV2"
            disable_preview: Disable link previews
            silent:          Send without notification sound

        Returns:
            bool: True if sent successfully
        """
        if not self.enabled:
            logger.debug("Telegram disabled — message not sent")
            return False

        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id missing")
            return False

        # Quiet hours check
        if self._in_quiet_hours() and not silent:
            logger.debug("Quiet hours — message suppressed")
            return False

        # Rate limit check
        if not self._check_rate_limit():
            logger.warning("Telegram rate limit — message delayed")
            time.sleep(3)

        # Chunk long messages
        chunks = self._chunk_message(text)

        success = True
        for i, chunk in enumerate(chunks):
            try:
                payload = {
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": disable_preview,
                    "disable_notification": silent,
                }

                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=10,
                )

                if resp.status_code == 200:
                    self._msg_timestamps.append(time.time())
                    self._total_sent += 1
                    logger.debug(
                        f"Telegram sent ({i+1}/{len(chunks)}) "
                        f"[{len(chunk)} chars]"
                    )
                else:
                    error_info = resp.json() if resp.text else {}
                    err_desc = error_info.get("description", resp.text[:200])
                    logger.error(
                        f"Telegram API error {resp.status_code}: {err_desc}"
                    )

                    # If HTML parse fails, retry as plain text
                    if resp.status_code == 400 and "parse" in err_desc.lower():
                        logger.info("Retrying as plain text…")
                        payload["parse_mode"] = ""
                        # Strip HTML tags for plain text
                        import re
                        payload["text"] = re.sub(r"<[^>]+>", "", chunk)
                        retry = requests.post(
                            f"{self.base_url}/sendMessage",
                            json=payload,
                            timeout=10,
                        )
                        if retry.status_code == 200:
                            self._msg_timestamps.append(time.time())
                            self._total_sent += 1
                        else:
                            success = False
                            self._total_errors += 1
                    else:
                        success = False
                        self._total_errors += 1

                # Small delay between chunks
                if i < len(chunks) - 1:
                    time.sleep(0.5)

            except requests.exceptions.Timeout:
                logger.error("Telegram send timeout")
                success = False
                self._total_errors += 1

            except Exception as e:
                logger.error(f"Telegram send error: {e}")
                success = False
                self._total_errors += 1

        return success

    # ═══════════════════════════════════════════════════════════════════
    #  SIGNAL NOTIFICATION
    # ═══════════════════════════════════════════════════════════════════

    def send_signal(self, signal: Dict) -> bool:
        """
        Send a trading signal notification.

        Args:
            signal: Signal dict from SignalEngine.generate_signal()

        Returns:
            bool: True if sent
        """
        if not self.send_signals_flag:
            return False

        try:
            sym = signal.get("symbol", "???")
            sig_type = signal.get("signal", "HOLD")
            direction = signal.get("direction", 0)
            confidence = signal.get("confidence", 0)
            score = signal.get("combined_score", 0)
            entry = signal.get("entry_price", 0)
            sl = signal.get("stop_loss", 0)
            tp = signal.get("take_profit", 0)
            rr = signal.get("risk_reward_ratio", 0)
            risk_pct = signal.get("risk_pct", 0)
            reward_pct = signal.get("reward_pct", 0)
            atr_pct = signal.get("atr_pct", 0)

            # Signal emoji
            if sig_type == "LONG":
                emoji = "🟢"
                dir_emoji = "📈"
            elif sig_type == "SHORT":
                emoji = "🔴"
                dir_emoji = "📉"
            else:
                emoji = "⚪"
                dir_emoji = "➖"

            # Confidence bar
            conf_bar = self._progress_bar(confidence, 10)
            conf_pct = confidence * 100

            # Build message
            lines = [
                f"{emoji} <b>SIGNAL: {sig_type} {sym}</b> {dir_emoji}",
                "",
                f"💰 <b>Entry:</b>  <code>${entry:,.2f}</code>",
                f"🛑 <b>Stop Loss:</b>  <code>${sl:,.2f}</code>"
                + (f"  ({risk_pct:.1f}%)" if risk_pct else ""),
                f"🎯 <b>Take Profit:</b>  <code>${tp:,.2f}</code>"
                + (f"  ({reward_pct:.1f}%)" if reward_pct else ""),
                f"⚖️ <b>Risk:Reward:</b>  <code>1:{rr:.1f}</code>",
                "",
                f"📊 <b>Confidence:</b>  {conf_bar} {conf_pct:.0f}%",
                f"📏 <b>Score:</b>  <code>{score:+.3f}</code>",
            ]

            if atr_pct:
                lines.append(f"📐 <b>ATR:</b>  <code>{atr_pct:.2f}%</code>")

            # Components breakdown
            components = signal.get("components", {})
            if components:
                lines.append("")
                lines.append("🔬 <b>Components:</b>")

                comp_order = [
                    "ml_ensemble", "sentiment", "ai_reasoning",
                    "funding_rate", "market_structure",
                ]
                comp_names = {
                    "ml_ensemble": "ML Ensemble",
                    "sentiment": "Sentiment",
                    "ai_reasoning": "AI Reasoning",
                    "funding_rate": "Funding Rate",
                    "market_structure": "Mkt Structure",
                }

                for key in comp_order:
                    comp = components.get(key, {})
                    if not comp.get("available"):
                        continue

                    c_score = comp.get("score", 0)
                    c_signal = comp.get("signal", "?")
                    c_weight = comp.get("weight", 0)

                    if c_score > 0.1:
                        c_emoji = "🟢"
                    elif c_score < -0.1:
                        c_emoji = "🔴"
                    else:
                        c_emoji = "⚪"

                    name = comp_names.get(key, key)
                    lines.append(
                        f"  {c_emoji} {name}: "
                        f"<code>{c_score:+.3f}</code> "
                        f"({c_signal}) [{c_weight:.0%}]"
                    )

            # Data quality
            dq = signal.get("data_quality")
            if dq is not None:
                if isinstance(dq, dict):
                    dq_score = dq.get("score", 0)
                elif isinstance(dq, (int, float)):
                    dq_score = int(dq)
                else:
                    dq_score = 0
                lines.append(f"\n📡 <b>Data Quality:</b> {dq_score}/100")

            # Timestamp
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            lines.append(f"\n🕐 {now}")

            text = "\n".join(lines)
            return self.send_message(text)

        except Exception as e:
            logger.error(f"Signal notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  TRADE NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════

    def send_trade_opened(self, trade: Dict) -> bool:
        """
        Send notification when a paper/live trade is opened.

        Args:
            trade: Trade data dict from PaperTrader._open_position()
        """
        if not self.send_signals_flag:
            return False

        try:
            sym = trade.get("symbol", "???")
            sig = trade.get("signal_type", trade.get("signal", "?"))
            entry = float(trade.get("entry_price", 0))
            size = float(trade.get("position_size_usd", 0))
            lev = trade.get("leverage", 1)
            sl = float(trade.get("stop_loss", 0))
            tp = float(trade.get("take_profit", 0))
            conf = float(trade.get("confidence", 0))
            mode = trade.get("mode", "paper").upper()
            trade_id = trade.get("trade_id", trade.get("id", "?"))

            emoji = "🟢" if sig == "LONG" else "🔴"

            lines = [
                f"📈 <b>TRADE OPENED</b> [{mode}]",
                "",
                f"{emoji} <b>{sig} {sym}</b>",
                f"💰 Entry: <code>${entry:,.2f}</code>",
                f"📦 Size: <code>${size:,.2f}</code> ({lev}x)",
                f"🛑 SL: <code>${sl:,.2f}</code>",
                f"🎯 TP: <code>${tp:,.2f}</code>",
                f"📊 Confidence: {conf:.0%}",
                f"🆔 Trade #{trade_id}",
            ]

            now = datetime.now(timezone.utc).strftime("%H:%M UTC")
            lines.append(f"\n🕐 {now}")

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Trade opened notification error: {e}")
            return False

    def send_trade_closed(self, trade: Dict) -> bool:
        """
        Send notification when a trade is closed.

        Args:
            trade: Closed trade result dict
        """
        if not self.send_signals_flag:
            return False

        try:
            sym = trade.get("symbol", "???")
            direction = trade.get("direction", 0)
            sig = "LONG" if direction == 1 else "SHORT"
            entry = float(trade.get("entry_price", 0))
            exit_p = float(trade.get("exit_price", 0))
            reason = trade.get("exit_reason", "unknown")
            net_pnl = float(trade.get("net_pnl", trade.get("net_pnl_usd", 0)))
            pnl_pct = float(trade.get("pnl_pct", 0))
            commission = float(trade.get("commission", trade.get("commission_usd", 0)))
            is_win = trade.get("is_win", net_pnl > 0)

            if is_win:
                emoji = "✅"
                result = "WIN"
            else:
                emoji = "❌"
                result = "LOSS"

            reason_emojis = {
                "stop_loss": "🛑 Stop Loss",
                "take_profit": "🎯 Take Profit",
                "trailing_stop": "📐 Trailing Stop",
                "force_close": "⚡ Force Close",
                "max_holding": "⏰ Max Holding",
                "end_of_data": "📊 End of Data",
            }
            reason_text = reason_emojis.get(reason, f"📋 {reason}")

            lines = [
                f"{emoji} <b>TRADE CLOSED — {result}</b>",
                "",
                f"{'🟢' if direction == 1 else '🔴'} <b>{sig} {sym}</b>",
                f"💰 Entry: <code>${entry:,.2f}</code>",
                f"💰 Exit:  <code>${exit_p:,.2f}</code>",
                f"📋 Reason: {reason_text}",
                "",
                f"{'💵' if is_win else '💸'} <b>PnL: ${net_pnl:+,.2f} ({pnl_pct:+.2f}%)</b>",
            ]

            if commission > 0:
                lines.append(f"💳 Commission: ${commission:.2f}")

            now = datetime.now(timezone.utc).strftime("%H:%M UTC")
            lines.append(f"\n🕐 {now}")

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Trade closed notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  DAILY / PERFORMANCE REPORT
    # ═══════════════════════════════════════════════════════════════════

    def send_daily_report(self, performance: Dict) -> bool:
        """
        Send daily performance summary.

        Args:
            performance: Dict with capital, pnl, stats, risk info
        """
        if not self.send_daily_flag:
            return False

        try:
            capital = float(performance.get("capital", 0))
            initial = float(performance.get("initial_capital", 0))
            total_pnl = float(performance.get("total_pnl_usd", 0))
            total_ret = float(performance.get("total_return_pct", 0))

            stats = performance.get("db_stats", {})
            win_rate = stats.get("win_rate", 0)
            total_trades = stats.get("total_trades", 0)
            profit_factor = stats.get("profit_factor", 0)

            risk = performance.get("risk_summary", {})
            dd = risk.get("max_drawdown_pct", 0)
            consec_losses = risk.get("consecutive_losses", 0)
            open_pos = risk.get("open_positions", 0)
            can_trade = risk.get("can_trade", True)

            # Overall status emoji
            if total_ret > 2:
                status = "🚀"
            elif total_ret > 0:
                status = "📈"
            elif total_ret > -2:
                status = "📉"
            else:
                status = "🔥"

            pnl_emoji = "💵" if total_pnl >= 0 else "💸"

            lines = [
                f"📊 <b>DAILY REPORT</b> {status}",
                "",
                f"{pnl_emoji} <b>Capital:</b> <code>${capital:,.2f}</code>",
                f"📈 <b>Total PnL:</b> <code>${total_pnl:+,.2f} ({total_ret:+.2f}%)</code>",
                "",
                "📋 <b>Stats:</b>",
                f"  🎯 Win Rate: <code>{win_rate:.1f}%</code>",
                f"  📊 Trades: <code>{total_trades}</code>",
                f"  ⚖️ Profit Factor: <code>{profit_factor:.2f}</code>",
                "",
                "🛡️ <b>Risk:</b>",
                f"  📉 Max Drawdown: <code>{dd:.2f}%</code>",
                f"  🔄 Consec Losses: <code>{consec_losses}</code>",
                f"  📦 Open Positions: <code>{open_pos}</code>",
                f"  {'✅' if can_trade else '⛔'} Trading: "
                f"{'Active' if can_trade else 'PAUSED'}",
            ]

            # Active breakers warning
            breakers = risk.get("active_breakers", [])
            if breakers:
                lines.append("")
                lines.append("⚠️ <b>Circuit Breakers Active:</b>")
                for b in breakers:
                    lines.append(f"  🚫 {b}")

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"\n📅 {today}")

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Daily report notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  BACKTEST REPORT
    # ═══════════════════════════════════════════════════════════════════

    def send_backtest_report(self, result: Dict) -> bool:
        """
        Send backtest result summary.

        Args:
            result: Dict from Backtester.run()
        """
        if not self.send_signals_flag:
            return False

        try:
            sym = result.get("symbol", "???")
            m = result.get("metrics", {})
            wf = result.get("walk_forward", {})
            dr = result.get("date_range", {})

            total_ret = m.get("total_return_pct", 0)
            if total_ret > 0:
                emoji = "📈"
            else:
                emoji = "📉"

            lines = [
                f"🧪 <b>BACKTEST: {sym}</b> {emoji}",
                "",
                f"📅 {dr.get('start', '?')} → {dr.get('end', '?')}",
                f"📊 Bars: {result.get('total_bars', 0)} | "
                f"Windows: {wf.get('windows', 0)}",
                f"🎯 Model Accuracy: {wf.get('avg_model_accuracy', 0):.1%}",
                "",
                "💰 <b>Performance:</b>",
                f"  📈 Return: <code>{total_ret:+.2f}%</code>",
                f"  💵 PnL: <code>${m.get('total_pnl_usd', 0):+,.2f}</code>",
                f"  💼 Final: <code>${m.get('final_capital', 0):,.2f}</code>",
                "",
                "📋 <b>Stats:</b>",
                f"  🎯 Win Rate: <code>{m.get('win_rate_pct', 0):.1f}%</code>",
                f"  📊 Trades: <code>{m.get('total_trades', 0)}</code>"
                f" ({m.get('long_trades', 0)}L/{m.get('short_trades', 0)}S)",
                f"  ⚖️ Profit Factor: <code>{m.get('profit_factor', 0):.3f}</code>",
                f"  💡 Expectancy: <code>${m.get('expectancy_usd', 0):+.2f}</code>",
                "",
                "📐 <b>Risk Metrics:</b>",
                f"  📉 Sharpe: <code>{m.get('sharpe_ratio', 0):.3f}</code>",
                f"  📉 Sortino: <code>{m.get('sortino_ratio', 0):.3f}</code>",
                f"  📉 Max DD: <code>{m.get('max_drawdown_pct', 0):.2f}%</code>",
                "",
                f"  🏷️ B&H: <code>{m.get('buy_hold_return_pct', 0):+.2f}%</code>",
                f"  ⚡ Alpha: <code>{m.get('alpha_pct', 0):+.2f}%</code>",
            ]

            # Exit reasons
            exits = m.get("exit_reasons", {})
            if exits:
                lines.append("")
                lines.append("🚪 <b>Exit Reasons:</b>")
                for reason, count in sorted(
                    exits.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"  • {reason}: {count}")

            lines.append(f"\n⏱️ {result.get('total_time_s', 0):.1f}s")

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Backtest report notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  SCAN RESULTS
    # ═══════════════════════════════════════════════════════════════════

    def send_scan_results(self, scan: Dict) -> bool:
        """
        Send multi-pair scan results.

        Args:
            scan: Dict from SignalEngine.scan_all()
        """
        if not self.send_signals_flag:
            return False

        try:
            summary = scan.get("summary", {})
            all_results = scan.get("all", {})
            actionable = scan.get("actionable", {})

            lines = [
                "🔍 <b>MARKET SCAN</b>",
                "",
                f"📊 Scanned: {summary.get('total', 0)} pairs",
                f"🎯 Actionable: {summary.get('actionable', 0)}",
                f"🟢 Long: {summary.get('long', 0)} | "
                f"🔴 Short: {summary.get('short', 0)} | "
                f"⚪ Hold: {summary.get('hold', 0)}",
                "",
            ]

            # All pairs summary
            lines.append("📋 <b>All Pairs:</b>")
            for sym, res in sorted(all_results.items()):
                sig = res.get("signal", "?")
                conf = res.get("confidence", 0)
                score = res.get("combined_score", 0)

                if sig == "LONG":
                    e = "🟢"
                elif sig == "SHORT":
                    e = "🔴"
                else:
                    e = "⚪"

                lines.append(
                    f"  {e} <b>{sym}</b>: {sig} | "
                    f"conf={conf:.0%} | "
                    f"score={score:+.3f}"
                )

            # Actionable signals detail
            if actionable:
                lines.append("")
                lines.append("🎯 <b>Actionable Signals:</b>")
                for sym, res in actionable.items():
                    entry = res.get("entry_price", 0)
                    sl = res.get("stop_loss", 0)
                    tp = res.get("take_profit", 0)
                    sig = res.get("signal", "?")

                    lines.append(
                        f"\n  {'🟢' if sig == 'LONG' else '🔴'} "
                        f"<b>{sig} {sym}</b>"
                    )
                    lines.append(
                        f"    Entry: ${entry:,.2f} | "
                        f"SL: ${sl:,.2f} | TP: ${tp:,.2f}"
                    )

            lines.append(
                f"\n⏱️ {summary.get('scan_time_s', 0):.1f}s | "
                f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Scan results notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  ERROR / WARNING ALERTS
    # ═══════════════════════════════════════════════════════════════════

    def send_error(
        self, module: str, error_type: str, message: str
    ) -> bool:
        """
        Send error alert.

        Args:
            module:     Module name where error occurred
            error_type: Error classification
            message:    Error description
        """
        if not self.send_errors_flag:
            return False

        try:
            lines = [
                "🚨 <b>ERROR ALERT</b>",
                "",
                f"📦 <b>Module:</b> <code>{module}</code>",
                f"⚡ <b>Type:</b> <code>{error_type}</code>",
                f"📝 <b>Message:</b>",
                f"<code>{message[:500]}</code>",
                "",
                f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ]

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Error notification error: {e}")
            return False

    def send_warning(self, message: str) -> bool:
        """Send a warning notification."""
        if not self.send_errors_flag:
            return False

        try:
            text = (
                f"⚠️ <b>WARNING</b>\n\n"
                f"{message[:1000]}\n\n"
                f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            return self.send_message(text, silent=True)

        except Exception as e:
            logger.error(f"Warning notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  MODEL RETRAIN NOTIFICATION
    # ═══════════════════════════════════════════════════════════════════

    def send_model_retrained(self, result: Dict) -> bool:
        """
        Send model retraining notification.

        Args:
            result: Dict from ModelTrainer.train_symbol()
        """
        if not self.send_retrain_flag:
            return False

        try:
            sym = result.get("symbol", "???")
            success = result.get("success", False)
            test_eval = result.get("test_evaluation", {})
            acc = test_eval.get("accuracy", 0)
            f1 = test_eval.get("f1_score", 0)
            total_time = result.get("total_time_s", 0)

            emoji = "✅" if success else "❌"

            lines = [
                f"🔄 <b>MODEL RETRAINED</b> {emoji}",
                "",
                f"📦 <b>Symbol:</b> {sym}",
                f"{'✅' if success else '❌'} <b>Status:</b> "
                f"{'Success' if success else 'Failed'}",
            ]

            if success:
                lines.extend([
                    f"🎯 Accuracy: <code>{acc:.1%}</code>",
                    f"📊 F1 Score: <code>{f1:.3f}</code>",
                    f"📦 Features: {result.get('features', '?')}",
                    f"📊 Samples: {result.get('samples', '?')}",
                ])

                top = result.get("top_features", [])
                if top:
                    lines.append(
                        f"🏆 Top: {', '.join(top[:5])}"
                    )
            else:
                lines.append(
                    f"📝 Error: {result.get('error', 'unknown')}"
                )

            lines.append(f"\n⏱️ {total_time:.1f}s")

            return self.send_message("\n".join(lines), silent=True)

        except Exception as e:
            logger.error(f"Retrain notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  STARTUP / STATUS
    # ═══════════════════════════════════════════════════════════════════

    def send_startup(self, mode: str, capital: float, pairs: List[str]) -> bool:
        """Send agent startup notification."""
        try:
            lines = [
                "🤖 <b>AGENT STARTED</b>",
                "",
                f"⚙️ Mode: <code>{mode.upper()}</code>",
                f"💰 Capital: <code>${capital:,.2f}</code>",
                f"📦 Pairs: <code>{', '.join(pairs)}</code>",
                f"📊 Count: {len(pairs)}",
                "",
                f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ]

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Startup notification error: {e}")
            return False

    def send_shutdown(self, capital: float, total_pnl: float) -> bool:
        """Send agent shutdown notification."""
        try:
            emoji = "💵" if total_pnl >= 0 else "💸"
            lines = [
                "🛑 <b>AGENT STOPPED</b>",
                "",
                f"💰 Capital: <code>${capital:,.2f}</code>",
                f"{emoji} Session PnL: <code>${total_pnl:+,.2f}</code>",
                "",
                f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ]

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Shutdown notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  CIRCUIT BREAKER ALERT
    # ═══════════════════════════════════════════════════════════════════

    def send_circuit_breaker(self, breakers: List[str]) -> bool:
        """Send circuit breaker activation alert."""
        if not self.send_errors_flag:
            return False

        try:
            lines = [
                "⛔ <b>CIRCUIT BREAKER ACTIVATED</b>",
                "",
                "🚫 Trading paused due to:",
            ]

            for b in breakers:
                lines.append(f"  • {b}")

            lines.extend([
                "",
                "⏳ Trading will resume after cooldown.",
                f"\n🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
            ])

            return self.send_message("\n".join(lines))

        except Exception as e:
            logger.error(f"Circuit breaker notification error: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    #  CUSTOM / RAW MESSAGE
    # ═══════════════════════════════════════════════════════════════════

    def send_text(self, text: str, silent: bool = False) -> bool:
        """
        Send a plain text message (no formatting applied).

        Args:
            text:   Raw message text
            silent: Send without notification sound
        """
        return self.send_message(text, parse_mode="HTML", silent=silent)

    # ═══════════════════════════════════════════════════════════════════
    #  CONNECTION TEST
    # ═══════════════════════════════════════════════════════════════════

    def test_connection(self) -> Dict:
        """
        Test Telegram bot connection.

        Calls getMe API to verify token, then sends test message.

        Returns:
            Dict: {success, bot_name, bot_username, chat_id, message_sent}
        """
        result = {
            "success": False,
            "bot_name": None,
            "bot_username": None,
            "chat_id": self.chat_id,
            "message_sent": False,
        }

        if not self.token:
            result["error"] = "No TELEGRAM_BOT_TOKEN"
            return result

        try:
            # Test bot token
            resp = requests.get(
                f"{self.base_url}/getMe", timeout=10
            )

            if resp.status_code != 200:
                result["error"] = f"getMe failed: {resp.status_code}"
                return result

            data = resp.json()
            if not data.get("ok"):
                result["error"] = f"getMe not ok: {data}"
                return result

            bot = data.get("result", {})
            result["bot_name"] = bot.get("first_name", "?")
            result["bot_username"] = bot.get("username", "?")

            # Test send
            if self.chat_id:
                sent = self.send_message(
                    "🤖 <b>Connection Test</b>\n\n"
                    "✅ Crypto Agent is connected!\n"
                    f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                    silent=True,
                )
                result["message_sent"] = sent

            result["success"] = True
            return result

        except Exception as e:
            result["error"] = str(e)
            return result

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _progress_bar(value: float, width: int = 10) -> str:
        """Create a text progress bar: ████░░░░░░"""
        filled = int(value * width)
        filled = max(0, min(filled, width))
        empty = width - filled
        return "█" * filled + "░" * empty

    def _chunk_message(self, text: str) -> List[str]:
        """
        Split message into chunks respecting Telegram's 4096 char limit.

        Tries to split on newlines to keep formatting clean.
        """
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= MAX_MESSAGE_LENGTH:
                chunks.append(remaining)
                break

            # Find a good split point (newline near the limit)
            split_at = MAX_MESSAGE_LENGTH
            newline_pos = remaining.rfind("\n", 0, split_at)

            if newline_pos > split_at * 0.5:
                split_at = newline_pos + 1
            else:
                # No good newline — split at space
                space_pos = remaining.rfind(" ", 0, split_at)
                if space_pos > split_at * 0.5:
                    split_at = space_pos + 1

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        return chunks

    def _in_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours."""
        if not self.quiet_hours:
            return False

        try:
            start_h, end_h = self.quiet_hours
            current_h = datetime.now(timezone.utc).hour

            if start_h < end_h:
                return start_h <= current_h < end_h
            else:
                # Wraps midnight (e.g., 23 → 7)
                return current_h >= start_h or current_h < end_h

        except Exception:
            return False

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Clean old timestamps (older than 60s)
        self._msg_timestamps = [
            t for t in self._msg_timestamps if now - t < 60
        ]

        return len(self._msg_timestamps) < MAX_MESSAGES_PER_MINUTE

    # ═══════════════════════════════════════════════════════════════════
    #  STATUS
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Get notifier status."""
        return {
            "enabled": self.enabled,
            "has_token": bool(self.token),
            "has_chat_id": bool(self.chat_id),
            "send_signals": self.send_signals_flag,
            "send_daily_report": self.send_daily_flag,
            "send_errors": self.send_errors_flag,
            "send_model_retrain": self.send_retrain_flag,
            "quiet_hours": self.quiet_hours,
            "total_sent": self._total_sent,
            "total_errors": self._total_errors,
        }


# ═══════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════

_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get singleton TelegramNotifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


# ═══════════════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("\n" + "█" * 60)
    print("  TELEGRAM NOTIFIER — TEST")
    print("█" * 60)

    tg = TelegramNotifier()

    # ── 1. Status ──
    status = tg.get_status()
    print(f"\n  STATUS:")
    print(json.dumps(status, indent=4))

    if not tg.enabled:
        print("\n  ⚠️  Telegram is DISABLED")
        print("  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        print("  Showing message previews only:\n")

    # ── 2. Connection test ──
    if tg.enabled:
        print("\n  Testing connection…")
        conn = tg.test_connection()
        print(f"  Connection: {json.dumps(conn, indent=4)}")

    # ── 3. Test signal ──
    test_signal = {
        "symbol": "BTCUSDT",
        "signal": "SHORT",
        "direction": -1,
        "confidence": 0.72,
        "combined_score": -0.506,
        "entry_price": 68033.30,
        "stop_loss": 68814.09,
        "take_profit": 66471.73,
        "risk_reward_ratio": 2.0,
        "risk_pct": 1.15,
        "reward_pct": 2.29,
        "atr_pct": 2.3,
        "data_quality": {"score": 95},
        "components": {
            "ml_ensemble": {
                "available": True,
                "score": -0.647,
                "signal": "SHORT",
                "weight": 0.45,
            },
            "sentiment": {
                "available": True,
                "score": -0.275,
                "signal": "SHORT",
                "weight": 0.15,
            },
            "ai_reasoning": {
                "available": False,
                "score": 0.0,
                "signal": "HOLD",
                "weight": 0.15,
            },
            "funding_rate": {
                "available": True,
                "score": 0.022,
                "signal": "HOLD",
                "weight": 0.10,
            },
            "market_structure": {
                "available": True,
                "score": -0.667,
                "signal": "SHORT",
                "weight": 0.15,
            },
        },
    }

    print("\n  3. Sending test SIGNAL…")
    result = tg.send_signal(test_signal)
    print(f"     Sent: {result}")

    # ── 4. Test trade opened ──
    test_trade_open = {
        "symbol": "BTCUSDT",
        "signal_type": "SHORT",
        "entry_price": 68019.69,
        "position_size_usd": 6000.00,
        "leverage": 3,
        "stop_loss": 68814.09,
        "take_profit": 66471.73,
        "confidence": 0.72,
        "mode": "paper",
        "trade_id": 1,
    }

    print("\n  4. Sending TRADE OPENED…")
    result = tg.send_trade_opened(test_trade_open)
    print(f"     Sent: {result}")

    # ── 5. Test trade closed ──
    test_trade_close = {
        "symbol": "BTCUSDT",
        "direction": -1,
        "entry_price": 68019.69,
        "exit_price": 66471.73,
        "exit_reason": "take_profit",
        "net_pnl": 136.54,
        "pnl_pct": 1.37,
        "commission": 4.80,
        "is_win": True,
    }

    print("\n  5. Sending TRADE CLOSED…")
    result = tg.send_trade_closed(test_trade_close)
    print(f"     Sent: {result}")

    # ── 6. Test daily report ──
    test_perf = {
        "capital": 10136.54,
        "initial_capital": 10000.00,
        "total_pnl_usd": 136.54,
        "total_return_pct": 1.37,
        "db_stats": {
            "win_rate": 60.0,
            "total_trades": 5,
            "profit_factor": 1.85,
        },
        "risk_summary": {
            "max_drawdown_pct": 2.1,
            "consecutive_losses": 1,
            "open_positions": 2,
            "can_trade": True,
            "active_breakers": [],
        },
    }

    print("\n  6. Sending DAILY REPORT…")
    result = tg.send_daily_report(test_perf)
    print(f"     Sent: {result}")

    # ── 7. Test backtest report ──
    test_bt = {
        "symbol": "BTCUSDT",
        "total_bars": 494,
        "date_range": {
            "start": "2026-02-14",
            "end": "2026-03-07",
        },
        "metrics": {
            "total_return_pct": -1.95,
            "total_pnl_usd": -194.54,
            "final_capital": 9805.46,
            "win_rate_pct": 30.0,
            "total_trades": 40,
            "long_trades": 16,
            "short_trades": 24,
            "profit_factor": 0.838,
            "expectancy_usd": -4.86,
            "sharpe_ratio": -1.964,
            "sortino_ratio": -2.415,
            "max_drawdown_pct": 5.96,
            "buy_hold_return_pct": 4.93,
            "alpha_pct": -6.88,
            "exit_reasons": {
                "stop_loss": 19,
                "trailing_stop": 19,
                "take_profit": 1,
                "end_of_data": 1,
            },
        },
        "walk_forward": {
            "windows": 6,
            "avg_model_accuracy": 0.693,
        },
        "total_time_s": 31.0,
    }

    print("\n  7. Sending BACKTEST REPORT…")
    result = tg.send_backtest_report(test_bt)
    print(f"     Sent: {result}")

    # ── 8. Test error alert ──
    print("\n  8. Sending ERROR alert…")
    result = tg.send_error(
        "data.binance_data",
        "ConnectionError",
        "Failed to fetch OHLCV: Connection timeout after 30s",
    )
    print(f"     Sent: {result}")

    # ── 9. Test startup ──
    print("\n  9. Sending STARTUP…")
    result = tg.send_startup(
        "paper",
        10000.0,
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    )
    print(f"     Sent: {result}")

    # ── 10. Final status ──
    final = tg.get_status()
    print(f"\n  FINAL STATUS:")
    print(f"    Total sent:   {final['total_sent']}")
    print(f"    Total errors: {final['total_errors']}")

    print("\n" + "█" * 60)
    print("  TELEGRAM NOTIFIER TEST COMPLETE ✅")
    print("█" * 60 + "\n")