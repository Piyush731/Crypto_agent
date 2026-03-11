"""
Telegram Bot Command Handler — Remote Monitoring
==================================================
Listens for commands via long polling, responds with agent status.
Runs in a background daemon thread alongside paper trading.

Commands:
  /help        - Show available commands
  /status      - Capital, positions, P&L overview
  /balance     - Capital & P&L breakdown
  /positions   - Open positions with live prices
  /trades      - Recent closed trades
  /prices      - Live market prices + Fear & Greed
  /performance - Win rate, profit factor, stats
  /signals     - Recent signals generated

Security: Only responds to authorized TELEGRAM_CHAT_ID.
"""

import threading
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_CONFIG,
    TRADING_PAIRS,
    BACKTEST_CONFIG,
    RISK_CONFIG,
    SCHEDULE_CONFIG,
)
from core.logger import get_logger
from core.db import get_db

logger = get_logger("telegram_bot")


class TelegramCommandBot:
    """
    Telegram command handler with long polling.

    Runs in a daemon thread — auto-dies when main process exits.
    Read-only: only queries DB and Binance, never modifies state.

    Usage:
        bot = TelegramCommandBot()
        bot.start()   # non-blocking, starts background thread
        # ... paper trader runs ...
        bot.stop()    # graceful shutdown
    """

    def __init__(self):
        """Initialize bot with config settings."""
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = str(TELEGRAM_CHAT_ID).strip() if TELEGRAM_CHAT_ID else ""
        self.enabled = TELEGRAM_CONFIG.get("enabled", False)
        self.base_url = f"https://api.telegram.org/bot{self.token}"

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

        # Command registry — maps command string to handler method
        self._commands = {
            "/help": self._cmd_help,
            "/menu": self._cmd_help,
            "/start": self._cmd_help,
            "/status": self._cmd_status,
            "/pos": self._cmd_positions,
            "/positions": self._cmd_positions,
            "/trades": self._cmd_trades,
            "/prices": self._cmd_prices,
            "/perf": self._cmd_performance,
            "/performance": self._cmd_performance,
            "/signals": self._cmd_signals,
            "/bal": self._cmd_balance,
            "/balance": self._cmd_balance,
        }

        if self.enabled:
            logger.info("TelegramCommandBot initialized")
        else:
            logger.info("TelegramCommandBot disabled (no token/chat_id)")

    # ═══════════════════════════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════════════════════════

    def start(self):
        """Start listening for commands in background thread."""
        if not self.enabled:
            logger.info("Telegram command bot disabled — not starting")
            return

        if self._running:
            logger.warning("Command bot already running")
            return

        self._running = True
        self._clear_old_updates()

        self._thread = threading.Thread(
            target=self._poll_loop,
            name="TelegramCommandBot",
            daemon=True,  # dies when main process exits
        )
        self._thread.start()
        logger.info("Telegram command bot started ✅ (listening for /commands)")

    def stop(self):
        """Stop the polling loop gracefully."""
        if not self._running:
            return

        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("Telegram command bot stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ═══════════════════════════════════════════════════════════════════
    #  POLLING LOOP
    # ═══════════════════════════════════════════════════════════════════

    def _clear_old_updates(self):
        """Flush pending updates so we only process NEW commands after start."""
        try:
            resp = requests.get(
                f"{self.base_url}/getUpdates",
                params={"offset": -1, "limit": 1, "timeout": 1},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("result", [])
                if results:
                    self._offset = results[-1]["update_id"] + 1
                    logger.debug(f"Cleared old updates, offset={self._offset}")
        except Exception as e:
            logger.debug(f"Clear old updates: {e}")

    def _poll_loop(self):
        """Long-polling loop — runs in background thread."""
        logger.debug("Poll loop started")

        while self._running:
            try:
                resp = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self._offset,
                        "limit": 10,
                        "timeout": 30,  # long poll — blocks up to 30s
                    },
                    timeout=35,  # slightly more than poll timeout
                )

                if resp.status_code != 200:
                    logger.warning(f"getUpdates HTTP {resp.status_code}")
                    time.sleep(5)
                    continue

                data = resp.json()
                if not data.get("ok"):
                    logger.warning(f"getUpdates not ok: {data}")
                    time.sleep(5)
                    continue

                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)

            except requests.exceptions.Timeout:
                # Normal for long polling — no updates within 30s
                continue

            except requests.exceptions.ConnectionError:
                logger.warning("Connection error in poll loop, retrying in 15s")
                time.sleep(15)

            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                time.sleep(10)

        logger.debug("Poll loop ended")

    # ═══════════════════════════════════════════════════════════════════
    #  UPDATE HANDLER
    # ═══════════════════════════════════════════════════════════════════

    def _handle_update(self, update: Dict):
        """Process a single incoming update."""
        msg = update.get("message", {})
        if not msg:
            return

        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        user = msg.get("from", {}).get("first_name", "?")

        if not text or not chat_id:
            return

        # ── Security: only respond to authorized chat ──
        if chat_id != self.chat_id:
            logger.warning(
                f"Unauthorized command from chat_id={chat_id} user={user}: {text}"
            )
            return

        # Extract command — handle /command@botname format
        cmd = text.split()[0].lower().split("@")[0]
        logger.info(f"Command received: {cmd} (from {user})")

        handler = self._commands.get(cmd)
        if handler:
            try:
                response = handler()
                self._send_response(chat_id, response)
            except Exception as e:
                logger.error(f"Command {cmd} error: {e}", exc_info=True)
                self._send_response(chat_id, f"❌ Error processing {cmd}:\n{e}")
        elif text.startswith("/"):
            self._send_response(
                chat_id,
                f"❓ Unknown command: <code>{cmd}</code>\n\n"
                f"Type /help for available commands.",
            )

    def _send_response(self, chat_id: str, text: str):
        """Send response, handling chunking for long messages."""
        try:
            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )

                if resp.status_code != 200:
                    err = resp.json().get("description", resp.text[:200])
                    logger.error(f"Send response error: {err}")

                    # HTML parse error → retry plain text
                    if resp.status_code == 400 and "parse" in err.lower():
                        import re
                        plain = re.sub(r"<[^>]+>", "", chunk)
                        requests.post(
                            f"{self.base_url}/sendMessage",
                            json={
                                "chat_id": chat_id,
                                "text": plain,
                                "disable_web_page_preview": True,
                            },
                            timeout=10,
                        )

                if i < len(chunks) - 1:
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Send response error: {e}")

    @staticmethod
    def _chunk_text(text: str, max_len: int = 4096) -> List[str]:
        """Split text into Telegram-safe chunks."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break

            # Find good split point (newline)
            split_at = max_len
            nl = remaining.rfind("\n", 0, split_at)
            if nl > split_at * 0.5:
                split_at = nl + 1

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        return chunks

    # ═══════════════════════════════════════════════════════════════════
    #  COMMAND HANDLERS
    # ═══════════════════════════════════════════════════════════════════

    def _cmd_help(self) -> str:
        """Show available commands."""
        return (
            "🤖 <b>Crypto Agent — Remote Monitor</b>\n"
            "\n"
            "📊 <b>Monitoring</b>\n"
            "  /status — Agent overview\n"
            "  /balance — Capital &amp; P&amp;L\n"
            "  /positions — Open positions (live prices)\n"
            "  /trades — Recent closed trades\n"
            "  /signals — Recent generated signals\n"
            "  /performance — Win rate &amp; stats\n"
            "  /prices — Market prices &amp; Fear/Greed\n"
            "\n"
            "💡 <b>Shortcuts:</b> /pos /bal /perf\n"
            "\n"
            f"⏱️ Cycle interval: {SCHEDULE_CONFIG.get('analysis_interval_minutes', 60)} min\n"
            f"📦 Pairs: {', '.join(TRADING_PAIRS)}"
        )

    def _cmd_status(self) -> str:
        """Agent status overview: capital, positions, key stats."""
        try:
            db = get_db()

            # Capital
            initial = BACKTEST_CONFIG.get("initial_capital", 10000)
            total_pnl = db.get_total_pnl("paper") or 0
            capital = initial + total_pnl
            ret_pct = (total_pnl / initial * 100) if initial else 0

            # Positions
            open_trades = db.get_open_trades("paper")
            open_count = len(open_trades) if open_trades else 0
            max_pos = RISK_CONFIG.get("max_open_positions", 3)

            # Stats
            stats = db.get_stats(mode="paper") or {}
            total_trades = stats.get("total_trades", 0)
            win_rate = stats.get("win_rate", 0)

            # Daily PnL
            daily_pnl = db.get_daily_pnl("paper") or 0

            # Consecutive losses
            consec = db.get_consecutive_losses("paper") or 0

            pnl_emoji = "💵" if total_pnl >= 0 else "💸"
            daily_emoji = "📈" if daily_pnl >= 0 else "📉"

            lines = [
                "🤖 <b>AGENT STATUS</b>",
                "",
                f"💰 Capital: <code>${capital:,.2f}</code>",
                f"{pnl_emoji} Total PnL: <code>${total_pnl:+,.2f} ({ret_pct:+.1f}%)</code>",
                f"{daily_emoji} Today: <code>${daily_pnl:+,.2f}</code>",
                "",
                f"📦 Positions: {open_count}/{max_pos}",
                f"📊 Trades: {total_trades}",
                f"🎯 Win Rate: {win_rate:.1f}%",
                f"🔄 Consec Losses: {consec}",
            ]

            # List open position symbols
            if open_trades:
                pos_parts = []
                for t in open_trades:
                    sym = t.get("symbol", "?")
                    d = t.get("direction", "?")
                    d_str = str(d)
                    em = "🟢" if d_str in ("LONG", "1") else "🔴"
                    pos_parts.append(f"{em}{sym}")
                lines.append(f"\n📋 {' '.join(pos_parts)}")

            # Circuit breaker check
            max_consec = RISK_CONFIG.get("max_consecutive_losses", 5)
            if consec >= max_consec:
                lines.append(f"\n⛔ CIRCUIT BREAKER: {consec} consecutive losses!")

            now = datetime.now(timezone.utc).strftime("%H:%M UTC")
            lines.append(f"\n🕐 {now}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Status command error: {e}")
            return f"❌ Status error: {e}"

    def _cmd_balance(self) -> str:
        """Capital and P&L breakdown."""
        try:
            db = get_db()

            initial = BACKTEST_CONFIG.get("initial_capital", 10000)
            total_pnl = db.get_total_pnl("paper") or 0
            capital = initial + total_pnl
            ret_pct = (total_pnl / initial * 100) if initial else 0

            daily_pnl = db.get_daily_pnl("paper") or 0
            weekly_pnl = db.get_weekly_pnl("paper") or 0

            d_emoji = "📈" if daily_pnl >= 0 else "📉"
            w_emoji = "📈" if weekly_pnl >= 0 else "📉"

            return "\n".join([
                "💰 <b>BALANCE</b>",
                "",
                f"💵 Capital: <code>${capital:,.2f}</code>",
                f"📊 Initial: <code>${initial:,.2f}</code>",
                "",
                f"📈 Total: <code>${total_pnl:+,.2f} ({ret_pct:+.1f}%)</code>",
                f"{d_emoji} Today: <code>${daily_pnl:+,.2f}</code>",
                f"{w_emoji} Week: <code>${weekly_pnl:+,.2f}</code>",
            ])

        except Exception as e:
            return f"❌ Balance error: {e}"

    def _cmd_positions(self) -> str:
        """Open positions with live unrealized P&L."""
        try:
            db = get_db()
            open_trades = db.get_open_trades("paper")

            if not open_trades:
                return "📦 <b>No open positions</b>\n\nWaiting for signals with enough confidence."

            lines = [f"📦 <b>OPEN POSITIONS ({len(open_trades)})</b>", ""]

            # Fetch live prices
            prices = {}
            try:
                from data.binance_data import BinanceData
                bn = BinanceData()
                for t in open_trades:
                    sym = t.get("symbol", "")
                    if sym and sym not in prices:
                        tk = bn.get_ticker(sym, use_cache=False)
                        if tk:
                            prices[sym] = float(tk.get("last_price", 0))
                        time.sleep(0.1)
            except Exception as e:
                logger.debug(f"Price fetch for positions: {e}")

            for t in open_trades:
                sym = t.get("symbol", "?")
                direction = str(t.get("direction", "?"))
                entry = float(t.get("entry_price", 0))
                sl = float(t.get("stop_loss", 0))
                tp = float(t.get("take_profit", 0))
                size_usd = float(t.get("position_size_usd", 0))
                lev = t.get("leverage", 1)

                is_long = direction in ("LONG", "1")
                sig = "LONG" if is_long else "SHORT"
                emoji = "🟢" if is_long else "🔴"

                lines.append(f"{emoji} <b>{sig} {sym}</b>")

                # Format price
                if entry >= 1000:
                    lines.append(f"  💰 Entry: <code>${entry:,.2f}</code>")
                else:
                    lines.append(f"  💰 Entry: <code>${entry:.4f}</code>")

                # Live P&L
                current = prices.get(sym, 0)
                if current > 0:
                    if is_long:
                        pnl_pct = (current - entry) / entry * 100
                    else:
                        pnl_pct = (entry - current) / entry * 100

                    pnl_usd = size_usd * pnl_pct / 100
                    pc_emoji = "📈" if pnl_pct >= 0 else "📉"

                    if current >= 1000:
                        lines.append(f"  📍 Now: <code>${current:,.2f}</code>")
                    else:
                        lines.append(f"  📍 Now: <code>${current:.4f}</code>")

                    lines.append(
                        f"  {pc_emoji} PnL: <code>{pnl_pct:+.2f}%"
                        f" (${pnl_usd:+,.2f})</code>"
                    )

                # SL / TP distance
                if entry > 0:
                    if is_long:
                        sl_dist = (sl - entry) / entry * 100 if sl else 0
                        tp_dist = (tp - entry) / entry * 100 if tp else 0
                    else:
                        sl_dist = (entry - sl) / entry * 100 if sl else 0
                        tp_dist = (entry - tp) / entry * 100 if tp else 0

                if sl >= 1000:
                    lines.append(f"  🛑 SL: <code>${sl:,.2f}</code> ({sl_dist:+.1f}%)")
                elif sl > 0:
                    lines.append(f"  🛑 SL: <code>${sl:.4f}</code> ({sl_dist:+.1f}%)")

                if tp >= 1000:
                    lines.append(f"  🎯 TP: <code>${tp:,.2f}</code> ({tp_dist:+.1f}%)")
                elif tp > 0:
                    lines.append(f"  🎯 TP: <code>${tp:.4f}</code> ({tp_dist:+.1f}%)")

                lines.append(f"  📦 Size: <code>${size_usd:,.0f}</code> ({lev}x)")
                lines.append("")

            now = datetime.now(timezone.utc).strftime("%H:%M UTC")
            lines.append(f"🕐 {now}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Positions command error: {e}")
            return f"❌ Positions error: {e}"

    def _cmd_trades(self) -> str:
        """Recent closed trades."""
        try:
            db = get_db()
            trades = db.get_trades(mode="paper", status="closed", limit=10)

            if not trades:
                return "📜 <b>No closed trades yet</b>\n\nPositions will appear here after SL/TP hits."

            lines = [f"📜 <b>RECENT TRADES</b> (last {len(trades)})", ""]

            total_pnl = 0.0
            wins = 0
            count = 0

            for t in trades:
                sym = t.get("symbol", "?")
                direction = str(t.get("direction", "?"))
                pnl = float(t.get("pnl_usd", 0) or 0)
                pnl_pct = float(t.get("pnl_percent", 0) or 0)
                reason = t.get("exit_reason", "?")
                ts = t.get("closed_at", t.get("updated_at", ""))

                total_pnl += pnl
                count += 1
                if pnl > 0:
                    wins += 1

                is_long = direction in ("LONG", "1")
                sig = "L" if is_long else "S"
                emoji = "✅" if pnl >= 0 else "❌"

                # Short reason label
                reason_map = {
                    "stop_loss": "SL",
                    "take_profit": "TP",
                    "trailing_stop": "Trail",
                    "force_close": "Force",
                    "max_holding": "MaxH",
                }
                reason_short = reason_map.get(reason, reason[:6] if reason else "?")

                # Time
                time_str = ""
                if isinstance(ts, str) and len(ts) >= 16:
                    time_str = ts[5:16]  # MM-DD HH:MM

                lines.append(
                    f"{emoji} {sym} {sig} "
                    f"<code>${pnl:+.2f}</code> ({pnl_pct:+.1f}%) "
                    f"[{reason_short}] {time_str}"
                )

            # Summary line
            wr = (wins / count * 100) if count else 0
            pnl_emoji = "💵" if total_pnl >= 0 else "💸"

            lines.append("")
            lines.append(
                f"{pnl_emoji} Net: <code>${total_pnl:+,.2f}</code> | "
                f"WR: {wr:.0f}% ({wins}/{count})"
            )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Trades command error: {e}")
            return f"❌ Trades error: {e}"

    def _cmd_prices(self) -> str:
        """Live market prices and Fear & Greed index."""
        try:
            from data.binance_data import BinanceData
            bn = BinanceData()
        except Exception as e:
            return f"❌ Cannot connect to Binance: {e}"

        lines = ["📊 <b>MARKET PRICES</b>", ""]

        for sym in TRADING_PAIRS:
            try:
                tk = bn.get_ticker(sym, use_cache=False)
                if tk:
                    price = float(tk.get("last_price", 0))
                    chg = float(tk.get("price_change_pct", 0))
                    vol = float(tk.get("quote_volume_24h", 0))

                    emoji = "📈" if chg >= 0 else "📉"

                    if price >= 1000:
                        p_str = f"${price:,.2f}"
                    elif price >= 1:
                        p_str = f"${price:.4f}"
                    else:
                        p_str = f"${price:.6f}"

                    # Volume in M/B
                    if vol >= 1_000_000_000:
                        vol_str = f"${vol/1e9:.1f}B"
                    elif vol >= 1_000_000:
                        vol_str = f"${vol/1e6:.0f}M"
                    else:
                        vol_str = f"${vol:,.0f}"

                    lines.append(
                        f"{emoji} <b>{sym}</b>\n"
                        f"   <code>{p_str}</code> ({chg:+.2f}%) Vol:{vol_str}"
                    )
                else:
                    lines.append(f"❌ <b>{sym}</b>: fetch error")
            except Exception as e:
                lines.append(f"❌ <b>{sym}</b>: {e}")

            time.sleep(0.15)

        # Funding rates
        lines.append("")
        lines.append("💸 <b>Funding Rates:</b>")
        for sym in TRADING_PAIRS:
            try:
                fr = bn.get_funding_rate(sym, use_cache=False)
                if fr:
                    rate = float(fr.get("current_rate", 0))
                    ann = float(fr.get("annualized_rate", 0))
                    r_emoji = "🟢" if rate >= 0 else "🔴"
                    lines.append(
                        f"  {r_emoji} {sym}: <code>{rate:+.6f}</code>"
                        f" ({ann:+.1f}%/yr)"
                    )
                time.sleep(0.1)
            except Exception:
                pass

        # Fear & Greed
        try:
            from data.news_data import NewsData
            news = NewsData()
            fg = news.get_fear_greed_index()
            if fg:
                val = fg.get("value", "?")
                label = fg.get("label", "?")

                if isinstance(val, (int, float)):
                    if val <= 25:
                        fg_emoji = "😱"
                    elif val <= 45:
                        fg_emoji = "😰"
                    elif val <= 55:
                        fg_emoji = "😐"
                    elif val <= 75:
                        fg_emoji = "😊"
                    else:
                        fg_emoji = "🤑"
                else:
                    fg_emoji = "😱"

                lines.append(f"\n{fg_emoji} <b>Fear &amp; Greed:</b> {val} ({label})")
        except Exception:
            pass

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        lines.append(f"\n🕐 {now}")

        return "\n".join(lines)

    def _cmd_performance(self) -> str:
        """Performance statistics."""
        try:
            db = get_db()
            stats = db.get_stats(mode="paper") or {}

            total_trades = stats.get("total_trades", 0)
            if total_trades == 0:
                return (
                    "📊 <b>PERFORMANCE</b>\n\n"
                    "No closed trades yet.\n"
                    "Stats will appear after first trade closes."
                )

            initial = BACKTEST_CONFIG.get("initial_capital", 10000)
            total_pnl = db.get_total_pnl("paper") or 0
            capital = initial + total_pnl
            ret_pct = (total_pnl / initial * 100) if initial else 0

            win_rate = stats.get("win_rate", 0)
            profit_factor = stats.get("profit_factor", 0)
            pnl_db = stats.get("total_pnl", 0)

            # Open position count
            open_trades = db.get_open_trades("paper")
            open_count = len(open_trades) if open_trades else 0

            # Consecutive losses
            consec = db.get_consecutive_losses("paper") or 0

            # Daily / weekly
            daily_pnl = db.get_daily_pnl("paper") or 0
            weekly_pnl = db.get_weekly_pnl("paper") or 0

            # Grade
            if ret_pct > 5 and win_rate > 50 and profit_factor > 1.5:
                grade = "🏆 Excellent"
            elif ret_pct > 0 and profit_factor > 1.0:
                grade = "✅ Profitable"
            elif ret_pct > -3:
                grade = "⚠️ Marginal"
            else:
                grade = "❌ Needs Review"

            lines = [
                "📊 <b>PERFORMANCE</b>",
                "",
                f"💰 Capital: <code>${capital:,.2f}</code> ({ret_pct:+.1f}%)",
                f"💵 Total PnL: <code>${pnl_db:+,.2f}</code>",
                f"📅 Today: <code>${daily_pnl:+,.2f}</code>",
                f"📆 Week: <code>${weekly_pnl:+,.2f}</code>",
                "",
                f"📋 Trades: {total_trades}",
                f"🎯 Win Rate: {win_rate:.1f}%",
                f"⚖️ Profit Factor: {profit_factor:.2f}",
                f"📦 Open: {open_count}",
                f"🔄 Losing Streak: {consec}",
                "",
                f"📝 Grade: {grade}",
            ]

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Performance command error: {e}")
            return f"❌ Performance error: {e}"

    def _cmd_signals(self) -> str:
        """Recent generated signals."""
        try:
            db = get_db()
            signals = db.get_signals(limit=10)

            if not signals:
                return "📡 <b>No signals yet</b>\n\nSignals appear after each analysis cycle."

            lines = [f"📡 <b>RECENT SIGNALS</b> (last {len(signals)})", ""]

            for s in signals:
                sym = s.get("symbol", "?")
                sig_raw = s.get("direction", "?")
                sig = "LONG" if str(sig_raw) in ("1", "LONG") else (
                      "SHORT" if str(sig_raw) in ("-1", "SHORT") else "HOLD")
                conf = s.get("confidence", 0)
                score = s.get("combined_score", 0)
                ts = s.get("timestamp", "")
                status = s.get("status", "")

                # Time display
                time_str = ""
                if isinstance(ts, str):
                    if len(ts) >= 16:
                        time_str = ts[11:16]  # HH:MM
                    elif len(ts) >= 5:
                        time_str = ts[-5:]

                # Signal emoji
                if sig == "LONG":
                    emoji = "🟢"
                elif sig == "SHORT":
                    emoji = "🔴"
                else:
                    emoji = "⚪"

                # Confidence as percentage
                conf_pct = conf * 100 if isinstance(conf, (int, float)) and conf <= 1 else conf

                # Status indicator
                st_emoji = ""
                if status == "executed":
                    st_emoji = " ✅"
                elif status == "rejected":
                    st_emoji = " 🚫"

                lines.append(
                    f"{emoji} {sym} {sig} "
                    f"c={conf_pct:.0f}% "
                    f"s={score:+.3f} "
                    f"[{time_str}]{st_emoji}"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Signals command error: {e}")
            return f"❌ Signals error: {e}"

    # ═══════════════════════════════════════════════════════════════════
    #  STATUS
    # ═══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Get bot status."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "chat_id": self.chat_id,
            "commands": list(self._commands.keys()),
        }


# ═══════════════════════════════════════════════════════════════════════
#  TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  TELEGRAM COMMAND BOT — TEST")
    print("=" * 50)

    bot = TelegramCommandBot()
    status = bot.get_status()
    print(f"\n  Enabled: {status['enabled']}")
    print(f"  Commands: {len(status['commands'])}")
    print(f"  Chat ID: {status['chat_id'][:4]}..." if status['chat_id'] else "  Chat ID: not set")

    if not bot.enabled:
        print("\n  ⚠️  Bot disabled — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env")
        print("\n  Preview of command outputs:\n")

        # Show preview of each command
        for cmd_name in ["/help", "/status", "/balance", "/positions",
                         "/trades", "/performance", "/signals"]:
            handler = bot._commands.get(cmd_name)
            if handler:
                print(f"  {'─' * 40}")
                print(f"  Command: {cmd_name}")
                try:
                    import re
                    output = handler()
                    # Strip HTML for terminal display
                    plain = re.sub(r"<[^>]+>", "", output)
                    for line in plain.split("\n"):
                        print(f"    {line}")
                except Exception as e:
                    print(f"    Error: {e}")
                print()
    else:
        print(f"\n  Starting bot (Ctrl+C to stop)...")
        print(f"  Send /help in Telegram to test\n")

        bot.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopping...")
            bot.stop()

    print("\n" + "=" * 50)
    print("  TEST COMPLETE ✅")
    print("=" * 50 + "\n")