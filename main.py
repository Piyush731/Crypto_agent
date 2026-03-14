"""
Crypto Futures AI Agent v3.0 — Main Entry Point
================================================
Interactive CLI + command-line interface for all agent operations.

Phase 1: Backtest + Paper Trading + Telegram Signals
Phase 2: Paper Trading Optimization  
Phase 3: Live Trading (future)

Usage:
    python main.py                     # Interactive menu
    python main.py --status            # System health check
    python main.py --prices            # Quick price check (verify Binance data)
    python main.py --analyze BTCUSDT   # Analyze single pair
    python main.py --scan              # Scan all pairs
    python main.py --train BTCUSDT     # Train model for one pair
    python main.py --train-all         # Train all models
    python main.py --backtest BTCUSDT  # Backtest single pair
    python main.py --backtest-all      # Backtest all pairs
    python main.py --paper             # Start paper trading (Ctrl+C to stop)
"""

import sys
import time
import argparse
import traceback as tb_module
from datetime import datetime, timezone
from typing import Optional, Dict, List

# ─── Project Imports ──────────────────────────────────────────────────────
import config
from core.logger import get_logger
from core.db import get_db

logger = get_logger("main")
VERSION = "3.0"

PID_FILE = config.BASE_DIR / "paper_trader.pid"


def _write_pid():
    """Write current process PID for daemon management."""
    import os
    try:
        PID_FILE.write_text(str(os.getpid()))
        logger.info(f"PID {os.getpid()} written to {PID_FILE}")
    except Exception as e:
        logger.warning(f"Could not write PID file: {e}")


def _remove_pid():
    """Remove PID file on clean exit."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
            logger.debug("PID file removed")
    except Exception:
        pass




_cache = {}


def _get(name: str):
    """Lazy-load and cache a module component."""
    if name in _cache:
        return _cache[name]

    try:
        if name == "binance":
            from data.binance_data import BinanceData
            _cache[name] = BinanceData()

        elif name == "news":
            from data.news_data import NewsData
            _cache[name] = NewsData()

        elif name == "sentiment":
            from data.sentiment import SentimentAnalyzer
            _cache[name] = SentimentAnalyzer()

        elif name == "data_manager":
            from data.manager import DataManager
            _cache[name] = DataManager()

        elif name == "feature_builder":
            from features.builder import FeatureBuilder
            _cache[name] = FeatureBuilder()

        elif name == "trainer":
            from models.trainer import ModelTrainer
            _cache[name] = ModelTrainer()

        elif name == "signal_engine":
            from analysis.signal_engine import SignalEngine
            _cache[name] = SignalEngine()

        elif name == "ai_brain":
            from analysis.ai_brain import AIBrain
            _cache[name] = AIBrain()

        elif name == "backtester":
            from trading.backtester import Backtester
            _cache[name] = Backtester(
                capital=config.BACKTEST_CONFIG["initial_capital"]
            )

        elif name == "telegram":
            from notifications.telegram import get_notifier
            _cache[name] = get_notifier()

        else:
            raise ValueError(f"Unknown module: {name}")

    except Exception as e:
        logger.error(f"Failed to load module '{name}': {e}")
        raise

    return _cache[name]


def _new_paper_trader(include_ai: bool = None):
    """Create a fresh PaperTrader (not cached — include_ai may vary)."""
    from trading.paper_trader import PaperTrader
    if include_ai is None:
        include_ai = config.AI_CONFIG["enabled"]
    return PaperTrader(
        capital=config.BACKTEST_CONFIG["initial_capital"],
        include_ai=include_ai,
    )



class C:
    """ANSI colour shortcuts."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"


def _banner():
    pairs  = len(config.TRADING_PAIRS)
    mode   = config.AGENT_MODE.upper()
    cap    = config.BACKTEST_CONFIG["initial_capital"]
    ai_ok  = "✅" if config.AI_CONFIG["enabled"] else "❌"
    tg_ok  = "✅" if config.TELEGRAM_CONFIG["enabled"] else "❌"
    print(f"""
{C.CYAN}{'═'*60}
   🤖  Crypto Futures AI Agent v{VERSION}
{'═'*60}{C.RESET}
   Mode: {C.BOLD}{mode}{C.RESET}  │  Pairs: {C.BOLD}{pairs}{C.RESET}  │  Capital: {C.BOLD}${cap:,.0f}{C.RESET}
   AI: {ai_ok}   │  Telegram: {tg_ok}   │  Horizon: {config.ACTIVE_HORIZON}
{C.CYAN}{'═'*60}{C.RESET}""")


def _menu():
    print(f"""
  {C.BOLD}── ANALYSIS ─────────────────────────────{C.RESET}
   {C.GREEN} 1{C.RESET} │ Analyze Single Pair
   {C.GREEN} 2{C.RESET} │ Scan All Pairs
   {C.GREEN} 3{C.RESET} │ Quick Price Check  (verify Binance data)

  {C.BOLD}── ML MODELS ────────────────────────────{C.RESET}
   {C.GREEN} 4{C.RESET} │ Train Models
   {C.GREEN} 5{C.RESET} │ Model Status

  {C.BOLD}── TRADING ──────────────────────────────{C.RESET}
   {C.GREEN} 6{C.RESET} │ Backtest Single Pair
   {C.GREEN} 7{C.RESET} │ Backtest All Pairs
   {C.GREEN} 8{C.RESET} │ Paper Trading

  {C.BOLD}── SYSTEM ───────────────────────────────{C.RESET}
   {C.GREEN} 9{C.RESET} │ System Health Check
   {C.GREEN}10{C.RESET} │ Test Telegram
   {C.GREEN}11{C.RESET} │ View Recent Signals
   {C.GREEN}12{C.RESET} │ View Trade History
   {C.GREEN}13{C.RESET} │ Database Stats

   {C.RED} 0{C.RESET} │ Exit
""")


def _ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            raw = input(f"  {prompt} [{lo}-{hi}]: ").strip()
            if raw == "":
                continue
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  {C.RED}Enter a number between {lo} and {hi}.{C.RESET}")
        except ValueError:
            print(f"  {C.RED}Enter a number.{C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return lo  # default to lowest (usually 0 = cancel)


def _ask_symbol() -> Optional[str]:
    pairs = config.TRADING_PAIRS
    print("\n  Select trading pair:")
    for i, p in enumerate(pairs, 1):
        print(f"   {C.GREEN}{i}{C.RESET} │ {p}")
    print(f"   {C.RED}0{C.RESET} │ Cancel")
    c = _ask_int("Pair", 0, len(pairs))
    return None if c == 0 else pairs[c - 1]


def _ask_yn(prompt: str, default: bool = True) -> bool:
    tag = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"  {prompt} {tag}: ").strip().lower()
        if ans == "":
            return default
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _hdr(text: str):
    print(f"\n{C.CYAN}{'═'*60}\n  {text}\n{'═'*60}{C.RESET}")


def _price(v) -> str:
    if v is None or v == 0:
        return "N/A"
    if v >= 1000:
        return f"${v:,.2f}"
    if v >= 1:
        return f"${v:.4f}"
    return f"${v:.6f}"


def _pct(v) -> str:
    if v is None:
        return "N/A"
    c = C.GREEN if v >= 0 else C.RED
    return f"{c}{v:+.2f}%{C.RESET}"


def _pnl(v) -> str:
    if v is None:
        return "N/A"
    c = C.GREEN if v >= 0 else C.RED
    return f"{c}${v:+,.2f}{C.RESET}"


def _conf_bar(conf: float, width: int = 12) -> str:
    filled = int(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = conf * 100
    c = C.GREEN if pct >= 70 else (C.YELLOW if pct >= 50 else C.RED)
    return f"{c}{bar} {pct:.1f}%{C.RESET}"


def _sig_emoji(signal: str) -> str:
    if signal == "LONG":
        return f"{C.GREEN}🟢 LONG{C.RESET}"
    if signal == "SHORT":
        return f"{C.RED}🔴 SHORT{C.RESET}"
    return f"{C.YELLOW}⚪ HOLD{C.RESET}"



def _show_signal(r: Dict):
    """Pretty-print a signal from SignalEngine.generate_signal()."""
    if not r or "error" in r:
        err = (r or {}).get("error", "Unknown error")
        print(f"\n  {C.RED}❌ Analysis failed: {err}{C.RESET}")
        return

    sym   = r.get("symbol", "???")
    sig   = r.get("signal", "HOLD")
    conf  = r.get("confidence", 0)
    score = r.get("combined_score", 0)
    entry = r.get("entry_price")
    sl    = r.get("stop_loss")
    tp    = r.get("take_profit")
    rr    = r.get("risk_reward_ratio")
    rpct  = r.get("risk_pct")
    rwpct = r.get("reward_pct")

    _hdr(f"📊  ANALYSIS RESULT — {sym}")
    print(f"    Signal:      {_sig_emoji(sig)}")
    print(f"    Confidence:  {_conf_bar(conf)}")
    print(f"    Score:       {C.BOLD}{score:+.3f}{C.RESET}")
    print()

    if entry:
        print(f"    Entry:       {_price(entry)}")
    if sl:
        print(f"    Stop Loss:   {_price(sl)}  ({_pct(-abs(rpct) if rpct else None)})")
    if tp:
        print(f"    Take Profit: {_price(tp)}  ({_pct(abs(rwpct) if rwpct else None)})")
    if rr:
        print(f"    Risk:Reward: 1:{rr:.1f}")

    # ── Component breakdown ──
    comps = r.get("components", {})
    if comps:
        labels = {
            "ml_ensemble":      ("ML Ensemble   ", "45%"),
            "sentiment":        ("Sentiment     ", "15%"),
            "ai_reasoning":     ("AI Reasoning  ", "15%"),
            "funding_rate":     ("Funding Rate  ", "10%"),
            "market_structure": ("Mkt Structure ", "15%"),
        }
        print(f"\n  {C.BOLD}── Components ──{C.RESET}")
        for key, (lbl, wt) in labels.items():
            c = comps.get(key, {})
            if c.get("available"):
                sc = c.get("score", 0)
                cs = c.get("signal", "N/A")
                em = "🟢" if sc > 0.05 else ("🔴" if sc < -0.05 else "⚪")
                print(f"    {em} {lbl} {cs:<6s} score:{sc:+.3f}  (wt:{wt})")
            else:
                print(f"    ⬜ {lbl} {'N/A':<6s} (unavailable)     (wt:{wt})")

    # ── FIX: data_quality can be int, dict, or missing ──
    dq = r.get("data_quality")
    if dq is not None:
        if isinstance(dq, dict):
            dqs = dq.get("score", 0)
            issues = dq.get("issues", [])
        elif isinstance(dq, (int, float)):
            dqs = int(dq)
            issues = []
        else:
            dqs = 0
            issues = []
        dc = C.GREEN if dqs >= 70 else (C.YELLOW if dqs >= 50 else C.RED)
        print(f"\n    Data Quality: {dc}{dqs}/100{C.RESET}")
        for iss in issues[:3]:
            print(f"      ⚠️  {iss}")

    active = r.get("active_components", 0)
    total  = r.get("total_components", 5)
    at     = r.get("analysis_time_s", 0)
    print(f"\n    ⏱️  Analysis: {at:.1f}s  │  Components: {active}/{total} active")
    print(f"{'═'*60}")




def _show_backtest(r: Dict, brief: bool = False):
    if not r.get("success"):
        sym = r.get("symbol", "???")
        err = r.get("error", "Unknown")
        print(f"  {C.RED}❌ Backtest failed for {sym}: {err}{C.RESET}")
        return

    sym   = r.get("symbol", "???")
    m     = r.get("metrics", {})
    ret   = m.get("total_return_pct", 0)
    sha   = m.get("sharpe_ratio", 0)
    sor   = m.get("sortino_ratio", 0)
    cal   = m.get("calmar_ratio", 0)
    mdd   = m.get("max_drawdown_pct", 0)
    # FIX: backtester stores win_rate_pct (percentage like 31.8), not win_rate (decimal)
    wr    = m.get("win_rate_pct", m.get("win_rate", 0))
    pf    = m.get("profit_factor", 0)
    tc    = r.get("trade_count", m.get("total_trades", 0))
    # FIX: backtester stores alpha_pct, not alpha_vs_buyhold
    alpha = m.get("alpha_pct", m.get("alpha_vs_buyhold", 0))
    ts    = r.get("total_time_s", 0)

    if brief:
        rc = C.GREEN if ret > 0 else C.RED
        # FIX: wr is already percentage, use :.1f not :.1%
        print(f"    {sym:<10s} Return:{rc}{ret:>+8.2f}%{C.RESET}  "
              f"Sharpe:{sha:>6.2f}  WR:{wr:>6.1f}%  "
              f"MaxDD:{C.RED}{mdd:>7.2f}%{C.RESET}  "
              f"Trades:{tc:>4d}  ({ts:.0f}s)")
        return

    _hdr(f"📈  BACKTEST RESULT — {sym}")

    print(f"    {C.BOLD}── Performance ──{C.RESET}")
    print(f"    Total Return:   {_pct(ret)}")
    print(f"    Sharpe Ratio:   {sha:.2f}")
    print(f"    Sortino Ratio:  {sor:.2f}")
    print(f"    Calmar Ratio:   {cal:.2f}")
    print(f"    Max Drawdown:   {C.RED}{mdd:.2f}%{C.RESET}")
    print(f"    Alpha vs B&H:   {_pct(alpha)}")

    print(f"\n    {C.BOLD}── Trades ──{C.RESET}")
    print(f"    Total Trades:   {tc}")
    # FIX: wr is already percentage, use :.1f not :.1%
    print(f"    Win Rate:       {wr:.1f}%")
    print(f"    Profit Factor:  {pf:.2f}")
    # FIX: backtester stores expectancy_usd, not expectancy
    exp = m.get("expectancy_usd", m.get("expectancy", 0))
    print(f"    Expectancy:     {_pnl(exp)}")
    # FIX: backtester stores long_win_rate_pct / short_win_rate_pct (percentage)
    lwr = m.get("long_win_rate_pct", m.get("long_win_rate", 0))
    swr = m.get("short_win_rate_pct", m.get("short_win_rate", 0))
    print(f"    Long Trades:    {m.get('long_trades',0)} (WR: {lwr:.1f}%)")
    print(f"    Short Trades:   {m.get('short_trades',0)} (WR: {swr:.1f}%)")

    er = m.get("exit_reasons", {})
    if er:
        print(f"\n    {C.BOLD}── Exit Reasons ──{C.RESET}")
        for reason, cnt in sorted(er.items(), key=lambda x: -x[1]):
            print(f"    {reason:<22s} {cnt:>4d}")

    rj = r.get("rejection_summary", {})
    if rj and any(v > 0 for v in rj.values()):
        print(f"\n    {C.BOLD}── Rejections ──{C.RESET}")
        for reason, cnt in sorted(rj.items(), key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {reason:<25s} {cnt:>4d}")

    dr = r.get("date_range", {})
    if dr:
        print(f"\n    Period: {dr.get('start','?')} → {dr.get('end','?')}")
    print(f"    Total Time: {ts:.1f}s")
    print(f"{'═'*60}")


def _show_train(sym: str, r: Dict, brief: bool = False):
    if r.get("error"):
        if brief:
            print(f"    ❌ {sym:<10s} {r['error']}")
        else:
            print(f"\n  {C.RED}❌ Training failed for {sym}: {r['error']}{C.RESET}")
        return

    te  = r.get("test_evaluation", {})
    acc = te.get("accuracy", 0)
    f1  = te.get("f1_score", 0)
    roc = te.get("roc_auc", 0)
    cv  = r.get("cv_results", {})
    cva = cv.get("mean_accuracy", 0)
    cvs = cv.get("std_accuracy", 0)
    sa  = r.get("samples", 0)
    fe  = r.get("features", 0)
    ts  = r.get("total_time_s", 0)

    if brief:
        print(f"    ✅ {sym:<10s} acc:{acc:.1%}  f1:{f1:.3f}  "
              f"roc:{roc:.3f}  cv:{cva:.1%}  ({ts:.0f}s)")
        return

    _hdr(f"🧠  TRAINING RESULT — {sym}")
    print(f"    Samples:       {sa}")
    print(f"    Features:      {fe}")
    print(f"    Test Accuracy: {C.BOLD}{acc:.1%}{C.RESET}")
    print(f"    F1 Score:      {f1:.3f}")
    print(f"    ROC AUC:       {roc:.3f}")
    print(f"    CV Accuracy:   {cva:.1%} ± {cvs:.1%}")
    print(f"    Train Time:    {ts:.1f}s")

    top = r.get("top_features", {})
    if top:
        print(f"\n  {C.BOLD}── Top 10 Features ──{C.RESET}")
        for i, (feat, imp) in enumerate(list(top.items())[:10], 1):
            bar = "█" * int(imp * 100)
            print(f"    {i:2d}. {feat:<35s} {imp:.4f} {C.DIM}{bar}{C.RESET}")

    if r.get("model_saved"):
        print(f"\n    💾 Model saved → {config.SAVED_MODELS_DIR / sym}/")
    print(f"{'═'*60}")



def _startup_hints():
    """Show helpful hints for first-time users."""
    trained = sum(
        1 for p in config.TRADING_PAIRS
        if (config.SAVED_MODELS_DIR / p).exists()
        and any((config.SAVED_MODELS_DIR / p).glob("*.pkl"))
    )
    total = len(config.TRADING_PAIRS)

    if trained == 0:
        print(f"  {C.YELLOW}💡 No trained models found.{C.RESET}")
        print(f"  {C.YELLOW}   Recommended first steps:{C.RESET}")
        print(f"  {C.YELLOW}     1. System Health Check  (option 9){C.RESET}")
        print(f"  {C.YELLOW}     2. Quick Price Check    (option 3){C.RESET}")
        print(f"  {C.YELLOW}     3. Train Models         (option 4){C.RESET}")
        print(f"  {C.YELLOW}     4. Backtest             (option 6){C.RESET}")
        print()
    elif trained < total:
        print(f"  {C.YELLOW}💡 {trained}/{total} models trained. "
              f"Train remaining via option 4.{C.RESET}\n")




# ────────── 1. Analyze Single Pair ──────────

def action_analyze():
    symbol = _ask_symbol()
    if not symbol:
        return

    include_ai = config.AI_CONFIG["enabled"]
    if include_ai:
        include_ai = _ask_yn("Include AI analysis? (slower, uses HuggingFace)")

    print(f"\n  🔄 Running full analysis for {C.BOLD}{symbol}{C.RESET} ...")
    print(f"     Components: ML + sentiment{' + AI' if include_ai else ''}")
    print(f"     Estimated time: {'30-60' if include_ai else '15-30'}s\n")

    try:
        # Check / train model
        trainer = _get("trainer")
        st = trainer.get_training_status([symbol])
        if not st.get("per_symbol", {}).get(symbol, {}).get("exists"):
            print(f"  ⚠️  No trained model for {symbol}.")
            if _ask_yn("Train now? (~1-2 min)"):
                print(f"  🧠 Training ...")
                tr = trainer.train_symbol(symbol, save_model=True)
                if tr.get("success") or not tr.get("error"):
                    acc = tr.get("test_evaluation", {}).get("accuracy", 0)
                    print(f"  ✅ Trained!  Test accuracy: {acc:.1%}\n")
                else:
                    print(f"  ⚠️  Training issue: {tr.get('error','?')}. Continuing...\n")

        engine = _get("signal_engine")
        result = engine.generate_signal(symbol, include_ai=include_ai)
        _show_signal(result)

        # Offer Telegram
        if (config.TELEGRAM_CONFIG["enabled"]
                and result.get("signal") != "HOLD"):
            if _ask_yn("Send signal to Telegram?"):
                tg = _get("telegram")
                ok = tg.send_signal(result)
                print(f"  {'✅ Sent!' if ok else '❌ Send failed'}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Analysis failed: {e}{C.RESET}")


# ────────── 2. Scan All Pairs ──────────

def action_scan():
    include_ai = False
    if config.AI_CONFIG["enabled"]:
        include_ai = _ask_yn("Include AI? (much slower for all pairs)", default=False)

    pairs = config.TRADING_PAIRS
    est = len(pairs) * (60 if include_ai else 20)
    print(f"\n  🔄 Scanning {len(pairs)} pairs  (AI: {'on' if include_ai else 'off'})")
    print(f"     Estimated: ~{est}s\n")

    try:
        engine = _get("signal_engine")
        scan = engine.scan_all(include_ai=include_ai)

        summ = scan.get("summary", {})
        all_r = scan.get("all", {})
        act   = scan.get("actionable", {})

        _hdr("📊  SCAN RESULTS")
        print(f"    Pairs scanned: {summ.get('total', 0)}")
        print(f"    Actionable:    {C.GREEN}{summ.get('actionable',0)}{C.RESET}")
        print(f"    Long / Short / Hold: "
              f"{summ.get('long',0)} / {summ.get('short',0)} / {summ.get('hold',0)}")
        print(f"    Scan time:     {summ.get('scan_time_s',0):.1f}s")

        print(f"\n  {C.BOLD}── All Pairs ──{C.RESET}")
        for sym, res in all_r.items():
            if isinstance(res, dict) and "signal" in res:
                s = res["signal"]
                co = res.get("confidence", 0)
                sc = res.get("combined_score", 0)
                em = "🟢" if s == "LONG" else ("🔴" if s == "SHORT" else "⚪")
                print(f"    {em} {sym:<10s} {s:<6s} "
                      f"conf:{co:.1%}  score:{sc:+.3f}")
            else:
                e = res.get("error", "failed") if isinstance(res, dict) else "failed"
                print(f"    ❌ {sym:<10s} {e}")

        for sym, res in act.items():
            _show_signal(res)

        if config.TELEGRAM_CONFIG["enabled"] and act:
            if _ask_yn("Send scan results to Telegram?"):
                ok = _get("telegram").send_scan_results(scan)
                print(f"  {'✅ Sent!' if ok else '❌ Failed'}")

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Scan failed: {e}{C.RESET}")


# ────────── 3. Quick Price Check ──────────

def action_prices():
    print(f"\n  🔄 Fetching live Binance Futures prices ...\n")

    try:
        bn = _get("binance")
        if not bn.test_connection():
            print(f"  {C.RED}❌ Cannot reach Binance API{C.RESET}")
            return

        st = bn.get_server_time()
        if st:
            print(f"  Binance server time: {st.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

        # ── Prices ──
        print(f"  {'Symbol':<10s} {'Price':>12s} {'24h Chg':>10s} "
              f"{'24h High':>12s} {'24h Low':>12s} {'Vol (USDT)':>15s}")
        print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*12} {'─'*12} {'─'*15}")

        for sym in config.TRADING_PAIRS:
            tk = bn.get_ticker(sym, use_cache=False)
            if tk:
                p   = tk.get("last_price", 0)
                chg = tk.get("price_change_pct", 0)
                hi  = tk.get("high_24h", 0)
                lo  = tk.get("low_24h", 0)
                vol = tk.get("quote_volume_24h", 0)
                cc  = C.GREEN if chg >= 0 else C.RED
                print(f"  {sym:<10s} {_price(p):>12s} "
                      f"{cc}{chg:>+8.2f}%{C.RESET}  "
                      f"{_price(hi):>12s} {_price(lo):>12s} "
                      f"${vol:>12,.0f}")
            else:
                print(f"  {sym:<10s} {'ERROR':>12s}")
            time.sleep(0.1)

        # ── Funding Rates ──
        print(f"\n  {C.BOLD}── Funding Rates ──{C.RESET}")
        print(f"  {'Symbol':<10s} {'Rate':>14s} {'Annualized':>12s}")
        print(f"  {'─'*10} {'─'*14} {'─'*12}")

        for sym in config.TRADING_PAIRS:
            fr = bn.get_funding_rate(sym, use_cache=False)
            if fr:
                rate = fr.get("current_rate", 0)
                ann  = fr.get("annualized_rate", 0)
                rc   = C.GREEN if rate >= 0 else C.RED
                print(f"  {sym:<10s} {rc}{rate:>+12.6f}%{C.RESET}  "
                      f"{rc}{ann:>+10.2f}%{C.RESET}")
            else:
                print(f"  {sym:<10s} {'ERROR':>14s}")
            time.sleep(0.1)

        print(f"\n  {C.CYAN}💡 To verify manually:{C.RESET}")
        print(f"     https://www.binance.com/en/futures/BTCUSDT")
        print(f"     Compare 'Last Price', '24h Change', 'Volume'")
        print(f"     (These are USDT-M Futures prices, NOT Spot)")

    except Exception as e:
        logger.error(f"Price check failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Price check failed: {e}{C.RESET}")


# ────────── 4. Train Models ──────────

def action_train():
    print(f"\n  Train ML models:")
    print(f"   {C.GREEN}1{C.RESET} │ Single pair")
    print(f"   {C.GREEN}2{C.RESET} │ All pairs")
    print(f"   {C.RED}0{C.RESET} │ Cancel")
    c = _ask_int("Choice", 0, 2)
    if c == 0:
        return

    try:
        trainer = _get("trainer")

        if c == 1:
            sym = _ask_symbol()
            if not sym:
                return
            print(f"\n  🧠 Training {C.BOLD}{sym}{C.RESET}  "
                  f"(horizon: {config.ACTIVE_HORIZON}) ...\n")
            r = trainer.train_symbol(sym, save_model=True)
            _show_train(sym, r)

        else:
            n = len(config.TRADING_PAIRS)
            print(f"\n  🧠 Training {n} pairs  (~{n*2} min) ...\n")
            r = trainer.train_all()
            s = r.get("summary", {})
            _hdr("🧠  TRAINING SUMMARY")
            print(f"    Successful: {C.GREEN}{s.get('successful',0)}{C.RESET}"
                  f"/{s.get('total',0)}")
            print(f"    Failed:     {C.RED}{s.get('failed',0)}{C.RESET}")
            print(f"    Avg Accuracy: {s.get('avg_accuracy',0):.1%}")
            print(f"    Total Time:   {s.get('total_time_s',0):.1f}s")
            print(f"\n  {C.BOLD}── Per Symbol ──{C.RESET}")
            for sym, res in r.get("per_symbol", {}).items():
                _show_train(sym, res, brief=True)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Training failed: {e}{C.RESET}")


# ────────── 5. Model Status ──────────

def action_model_status():
    try:
        trainer = _get("trainer")
        status = trainer.get_training_status()
        s = status.get("summary", {})

        _hdr("🧠  MODEL STATUS")
        print(f"    Trained:   {C.GREEN}{s.get('trained',0)}{C.RESET}"
              f"/{s.get('total',0)}")
        print(f"    Untrained: {C.RED}{s.get('untrained',0)}{C.RESET}")
        print(f"    Stale:     {C.YELLOW}{s.get('stale',0)}{C.RESET}")

        print(f"\n  {'Symbol':<10s} {'Status':<12s} {'Age(hrs)':<10s} "
              f"{'Features':<10s} {'Accuracy':<10s}")
        print(f"  {'─'*10} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

        for sym, info in status.get("per_symbol", {}).items():
            if info.get("exists"):
                hrs = info.get("hours_since_train", 0)
                ft  = info.get("features", "?")
                ac  = info.get("test_accuracy", 0)
                ret_h = config.SCHEDULE_CONFIG.get("retrain_interval_hours", 24)
                ac2 = C.YELLOW if hrs > ret_h else C.GREEN
                print(f"  {sym:<10s} {C.GREEN}{'Trained':<12s}{C.RESET} "
                      f"{ac2}{hrs:<10.1f}{C.RESET} {str(ft):<10s} {ac:<10.1%}")
            else:
                print(f"  {sym:<10s} {C.RED}{'Not trained':<12s}{C.RESET} "
                      f"{'—':<10s} {'—':<10s} {'—':<10s}")

    except Exception as e:
        logger.error(f"Model status failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ {e}{C.RESET}")


# ────────── 6. Backtest Single ──────────

def action_backtest():
    sym = _ask_symbol()
    if not sym:
        return

    cap = config.BACKTEST_CONFIG["initial_capital"]
    print(f"\n  📈 Backtesting {C.BOLD}{sym}{C.RESET}")
    print(f"     Capital: ${cap:,.0f}  |  Commission: "
          f"{config.BACKTEST_CONFIG['commission_pct']}%")
    print(f"     Horizon: {config.ACTIVE_HORIZON}")
    print(f"     Estimated time: 2-5 min (walk-forward)\n")

    try:
        bt = _get("backtester")
        r = bt.run(sym)
        _show_backtest(r)

        if config.TELEGRAM_CONFIG["enabled"] and r.get("success"):
            if _ask_yn("Send backtest report to Telegram?"):
                ok = _get("telegram").send_backtest_report(r)
                print(f"  {'✅ Sent!' if ok else '❌ Failed'}")

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Backtest failed: {e}{C.RESET}")


# ────────── 7. Backtest All ──────────

def action_backtest_all():
    n = len(config.TRADING_PAIRS)
    print(f"\n  📈 Backtest all {n} pairs  (~{n*3} min)")
    if not _ask_yn("Proceed?"):
        return

    try:
        bt = _get("backtester")
        r = bt.run_all()
        s = r.get("summary", {})

        # FIX: backtester uses total_symbols, avg_return_pct, avg_win_rate_pct
        total_sym = s.get("total_symbols", s.get("total", 0))
        avg_ret = s.get("avg_return_pct", s.get("avg_return", 0))
        avg_wr = s.get("avg_win_rate_pct", s.get("avg_win_rate", 0))

        _hdr("📈  BACKTEST SUMMARY — ALL PAIRS")
        print(f"    Successful:   {C.GREEN}{s.get('successful',0)}{C.RESET}"
              f"/{total_sym}")
        print(f"    Avg Return:   {_pct(avg_ret)}")
        print(f"    Avg Sharpe:   {s.get('avg_sharpe', 0):.2f}")
        print(f"    Avg Win Rate: {avg_wr:.1f}%")
        print(f"    Total Trades: {s.get('total_trades', 0)}")
        print(f"    Total Time:   {s.get('total_time_s', 0):.1f}s")

        print(f"\n  {C.BOLD}── Per Symbol ──{C.RESET}")
        for sym, res in r.get("per_symbol", {}).items():
            _show_backtest(res, brief=True)

    except Exception as e:
        logger.error(f"Backtest-all failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Backtest failed: {e}{C.RESET}")


# ────────── 8. Paper Trading ──────────

def action_paper():
    print(f"\n  Paper Trading:")
    print(f"   {C.GREEN}1{C.RESET} │ Start continuous trading")
    print(f"   {C.GREEN}2{C.RESET} │ Run single cycle (test)")
    print(f"   {C.GREEN}3{C.RESET} │ View open positions")
    print(f"   {C.GREEN}4{C.RESET} │ View performance")
    print(f"   {C.GREEN}5{C.RESET} │ Force close all positions")
    print(f"   {C.RED}0{C.RESET} │ Cancel")
    c = _ask_int("Choice", 0, 5)
    if c == 0:
        return

    try:
        include_ai = config.AI_CONFIG["enabled"]
        paper = _new_paper_trader(include_ai)

        # ── Continuous ──
        if c == 1:
            intv = config.SCHEDULE_CONFIG["analysis_interval_minutes"]
            print(f"\n  🚀 Starting Paper Trading")
            print(f"     Capital:  ${config.BACKTEST_CONFIG['initial_capital']:,.0f}")
            print(f"     Pairs:    {', '.join(config.TRADING_PAIRS)}")
            print(f"     Interval: {intv} min")
            print(f"     AI:       {'on' if include_ai else 'off'}")
            print(f"\n     Press {C.BOLD}Ctrl+C{C.RESET} to stop\n")

            # if not _ask_yn("Start now?"):
            #     return

            # if config.TELEGRAM_CONFIG["enabled"]:
            #     _get("telegram").send_startup(
            #         "paper",
            #         config.BACKTEST_CONFIG["initial_capital"],
            #         config.TRADING_PAIRS,
            #     )

            # paper.start()  # blocks until Ctrl+C or max_cycles

            # st = paper.get_status()   old code before teh bot commadnwas introduced telgram bot cammads okay .

            if not _ask_yn("Start now?"):
                return

            if config.TELEGRAM_CONFIG["enabled"]:
                _get("telegram").send_startup(
                    "paper",
                    config.BACKTEST_CONFIG["initial_capital"],
                    config.TRADING_PAIRS,
                )

            # Start Telegram command bot for remote monitoring
            cmd_bot = None
            if config.TELEGRAM_CONFIG["enabled"]:
                try:
                    from notifications.telegram_bot import TelegramCommandBot
                    cmd_bot = TelegramCommandBot()
                    cmd_bot.start()
                    print(f"  📱 Telegram commands active: /help /status /positions /trades")
                except Exception as e:
                    logger.warning(f"Telegram command bot not available: {e}")

            paper.start()  # blocks until Ctrl+C or max_cycles

            if cmd_bot:
                cmd_bot.stop()

            st = paper.get_status()
            print(f"\n  Paper trading stopped.")
            print(f"  Cycles: {st.get('cycles', 0)}  |  "
                  f"Capital: ${st.get('capital', 0):,.2f}")

            if config.TELEGRAM_CONFIG["enabled"]:
                _get("telegram").send_shutdown(
                    st.get("capital", 0), st.get("pnl", 0)
                )

        # ── Single cycle ──
        elif c == 2:
            print(f"\n  🔄 Running one paper-trading cycle ...\n")
            cr = paper.run_cycle()
            print(f"  Cycle #{cr.get('cycle',0)} — {cr.get('cycle_time_s',0):.1f}s")
            print(f"  Positions checked:  {cr.get('positions_checked',0)}")
            print(f"  Positions closed:   {cr.get('positions_closed',0)}")
            print(f"  Signals generated:  {cr.get('signals_generated',0)}")
            print(f"  Positions opened:   {cr.get('positions_opened',0)}")
            print(f"  Errors:             {cr.get('errors',0)}")
            print(f"  Capital:            ${cr.get('capital',0):,.2f}")

        # ── Open positions ──
        elif c == 3:
            positions = paper.get_open_positions()
            if not positions:
                print(f"\n  No open positions.")
            else:
                print(f"\n  {C.BOLD}── Open Positions ({len(positions)}) ──{C.RESET}")
                for pos in positions:
                    s   = pos.get("symbol", "?")
                    d_raw = pos.get("direction", "?")
                    d   = pos.get("direction_label", 
                        "LONG" if str(d_raw) == "1" else 
                        "SHORT" if str(d_raw) == "-1" else str(d_raw))
                    ep  = pos.get("entry_price", 0)
                    cp  = pos.get("current_price", ep)
                    upnl = pos.get("unrealized_pnl", 0)
                    upct = pos.get("unrealized_pnl_pct", 0)
                    em  = "🟢" if d == "LONG" else "🔴"
                    pc  = C.GREEN if upnl >= 0 else C.RED
                    print(f"    {em} {s:<10s} {d:<6s}  "
                          f"entry:{_price(ep)}  now:{_price(cp)}  "
                          f"P&L:{pc}{upnl:+,.2f} ({upct:+.2f}%){C.RESET}")

        # ── Performance ──
        elif c == 4:
            perf = paper.get_performance()
            _hdr("💰  PAPER TRADING PERFORMANCE")
            print(f"    Capital:       ${perf.get('capital', 0):,.2f}")
            print(f"    Total P&L:     {_pnl(perf.get('pnl', 0))}")
            print(f"    Return:        {_pct(perf.get('return_pct', 0))}")

            ds = perf.get("db_stats", {})
            if ds:
                print(f"\n    Win Rate:      {ds.get('win_rate', 0):.1f}%")
                print(f"    Total Trades:  {ds.get('total_trades', 0)}")
                print(f"    Profit Factor: {ds.get('profit_factor', 0):.2f}")
                print(f"    Total P&L:     {_pnl(ds.get('total_pnl', 0))}")

            rs = perf.get("risk_summary", {})
            if rs:
                print(f"\n    {C.BOLD}── Risk Status ──{C.RESET}")
                print(f"    Open Positions:    {rs.get('open_positions', 0)}"
                      f"/{rs.get('max_positions', 3)}")
                print(f"    Consecutive Losses:{rs.get('consecutive_losses', 0)}")
                dd = rs.get("current_drawdown_pct", 0)
                dc = C.RED if dd > 5 else C.YELLOW if dd > 2 else C.GREEN
                print(f"    Current Drawdown:  {dc}{dd:.2f}%{C.RESET}")

            hist = paper.get_trade_history(limit=10)
            if hist:
                print(f"\n    {C.BOLD}── Last 10 Trades ──{C.RESET}")
                print(f"    {'Symbol':<10s} {'Dir':<6s} {'Entry':>10s} "
                      f"{'Exit':>10s} {'P&L':>10s} {'Reason':<15s}")
                print(f"    {'─'*10} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*15}")
                for t in hist:
                    ts = t.get("symbol", "?")
                    td = t.get("direction", "?")
                    te = t.get("entry_price", 0)
                    tx = t.get("exit_price", 0)
                    tp = t.get("pnl_usd", 0) or 0
                    tr = t.get("exit_reason", "?")
                    tc = C.GREEN if tp >= 0 else C.RED
                    print(f"    {ts:<10s} {td:<6s} {_price(te):>10s} "
                          f"{_price(tx) if tx else '—':>10s} {tc}{tp:>+9.2f}{C.RESET} "
                          f"{tr:<15s}")

        # ── Force close ──
        elif c == 5:
            positions = paper.get_open_positions()
            if not positions:
                print(f"\n  No open positions to close.")
            else:
                print(f"\n  ⚠️  Force-close {len(positions)} position(s)?")
                for pos in positions:
                    s = pos.get("symbol", "?")
                    d = pos.get("direction", "?")
                    print(f"    • {s} {d}")
                if _ask_yn("Confirm force-close ALL?", default=False):
                    closed = paper.force_close_all()
                    for cl in closed:
                        s  = cl.get("symbol", "?")
                        pn = cl.get("pnl_usd", 0)
                        print(f"    ✅ Closed {s}: {_pnl(pn)}")
                    print(f"  Done. {len(closed)} position(s) closed.")

    except KeyboardInterrupt:
        print(f"\n  Paper trading interrupted.")
    except Exception as e:
        logger.error(f"Paper trading error: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ Paper trading error: {e}{C.RESET}")


# ────────── 9. System Health Check ──────────

def action_health():
    _hdr("🩺  SYSTEM HEALTH CHECK")
    checks = []

    # 1. Config
    print(f"  Checking configuration ...", end=" ", flush=True)
    try:
        from config import _config_valid
        ok = _config_valid
    except ImportError:
        ok = True  # config loaded fine if we got this far
    checks.append(("Configuration", ok))
    print(f"{'✅' if ok else '❌'}")

    # 2. Directories
    print(f"  Checking directories ...", end=" ", flush=True)
    dirs_ok = all(d.exists() for d in [
        config.SAVED_MODELS_DIR, config.DATA_CACHE_DIR, config.LOGS_DIR
    ])
    checks.append(("Directories", dirs_ok))
    print(f"{'✅' if dirs_ok else '❌'}")

    # 3. Database
    print(f"  Checking database ...", end=" ", flush=True)
    try:
        db = get_db()
        info = db.get_table_info()
        db_ok = len(info) > 0
        checks.append(("Database", db_ok))
        print(f"✅  tables: {list(info.keys())}")
    except Exception as e:
        checks.append(("Database", False))
        print(f"❌  {e}")

    # 4. Binance API
    print(f"  Checking Binance API ...", end=" ", flush=True)
    try:
        bn = _get("binance")
        bn_ok = bn.test_connection()
        checks.append(("Binance API", bn_ok))
        if bn_ok:
            st = bn.get_server_time()
            print(f"✅  server: {st.strftime('%H:%M:%S UTC') if st else '?'}")
        else:
            print(f"❌  connection failed")
    except Exception as e:
        checks.append(("Binance API", False))
        print(f"❌  {e}")

    # 5. News sources
    print(f"  Checking news sources ...", end=" ", flush=True)
    try:
        news = _get("news")
        headlines = news.get_google_news(max_items=3)
        news_ok = len(headlines) > 0
        checks.append(("News (Google RSS)", news_ok))
        print(f"{'✅' if news_ok else '⚠️'}  headlines: {len(headlines)}")
    except Exception as e:
        checks.append(("News (Google RSS)", False))
        print(f"⚠️  {e}")

    # 6. Fear & Greed
    print(f"  Checking Fear & Greed ...", end=" ", flush=True)
    try:
        news = _get("news")
        fg = news.get_fear_greed_index()
        fg_ok = fg is not None
        checks.append(("Fear & Greed Index", fg_ok))
        if fg_ok:
            print(f"✅  value: {fg.get('value','?')} ({fg.get('label','?')})")
        else:
            print(f"⚠️  unavailable")
    except Exception as e:
        checks.append(("Fear & Greed Index", False))
        print(f"⚠️  {e}")

    # 7. Sentiment (FinBERT)
    print(f"  Checking FinBERT sentiment ...", end=" ", flush=True)
    try:
        sa = _get("sentiment")
        test = sa.analyze_headline("Bitcoin surges to new all-time high")
        sent_ok = test.get("score") is not None
        checks.append(("FinBERT Sentiment", sent_ok))
        method = test.get("method", "?")
        print(f"{'✅' if sent_ok else '⚠️'}  method: {method}  "
              f"score: {test.get('score', 0):+.3f}")
    except Exception as e:
        checks.append(("FinBERT Sentiment", False))
        print(f"⚠️  {e}")

    # 8. HuggingFace AI
    print(f"  Checking HuggingFace AI ...", end=" ", flush=True)
    if config.AI_CONFIG["enabled"]:
        try:
            ai = _get("ai_brain")
            ai_st = ai.get_status()
            ai_ok = ai_st.get("enabled", False) and ai_st.get("has_token", False)
            checks.append(("HuggingFace AI", ai_ok))
            models = ai_st.get("models", {})
            print(f"{'✅' if ai_ok else '⚠️'}  models: {len(models)}")
            for role, model in models.items():
                print(f"       {role}: {model}")
        except Exception as e:
            checks.append(("HuggingFace AI", False))
            print(f"⚠️  {e}")
    else:
        checks.append(("HuggingFace AI", None))
        print(f"⬜  disabled (no HF_TOKEN)")

    # 9. Telegram
    print(f"  Checking Telegram ...", end=" ", flush=True)
    if config.TELEGRAM_CONFIG["enabled"]:
        try:
            tg = _get("telegram")
            tg_r = tg.test_connection()
            tg_ok = tg_r.get("success", False)
            checks.append(("Telegram", tg_ok))
            bot = tg_r.get("bot_name", "?")
            print(f"{'✅' if tg_ok else '❌'}  bot: {bot}")
        except Exception as e:
            checks.append(("Telegram", False))
            print(f"❌  {e}")
    else:
        checks.append(("Telegram", None))
        print(f"⬜  disabled (no token/chat_id)")

    # 10. Trained models
    print(f"  Checking trained models ...", end=" ", flush=True)
    trained = []
    untrained = []
    for p in config.TRADING_PAIRS:
        model_dir = config.SAVED_MODELS_DIR / p
        if model_dir.exists() and any(model_dir.glob("*.pkl")):
            trained.append(p)
        else:
            untrained.append(p)
    models_ok = len(trained) > 0
    checks.append(("Trained Models", models_ok))
    print(f"{'✅' if models_ok else '⚠️'}  "
          f"{len(trained)}/{len(config.TRADING_PAIRS)}  "
          f"trained: {trained if trained else 'none'}")
    if untrained:
        print(f"       untrained: {untrained}")

    # ── Summary ──
    total = len(checks)
    passed = sum(1 for _, ok in checks if ok is True)
    failed = sum(1 for _, ok in checks if ok is False)
    skipped = sum(1 for _, ok in checks if ok is None)

    print(f"\n  {C.BOLD}── Summary ──{C.RESET}")
    print(f"    ✅ Passed:  {C.GREEN}{passed}{C.RESET}/{total}")
    if failed:
        print(f"    ❌ Failed:  {C.RED}{failed}{C.RESET}/{total}")
    if skipped:
        print(f"    ⬜ Skipped: {skipped}/{total}")

    if failed == 0:
        print(f"\n  {C.GREEN}🎉 All systems operational!{C.RESET}")
    elif failed <= 2:
        print(f"\n  {C.YELLOW}⚠️  Minor issues — agent can still run.{C.RESET}")
    else:
        print(f"\n  {C.RED}🚨 Multiple failures — check configuration.{C.RESET}")

    print(f"{'═'*60}")


# ────────── 10. Test Telegram ──────────

def action_test_telegram():
    if not config.TELEGRAM_CONFIG["enabled"]:
        print(f"\n  {C.YELLOW}Telegram not configured.{C.RESET}")
        print(f"  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    try:
        tg = _get("telegram")
        print(f"\n  Testing Telegram connection ...")
        result = tg.test_connection()

        if result.get("success"):
            print(f"  ✅ Connected to bot: {C.BOLD}{result.get('bot_name','?')}{C.RESET}")
            print(f"  ✅ Test message sent: {result.get('message_sent', False)}")
        else:
            print(f"  ❌ Connection failed: {result.get('error', '?')}")
            return

        print(f"\n  Send a test signal?")
        print(f"   {C.GREEN}1{C.RESET} │ Test signal message")
        print(f"   {C.GREEN}2{C.RESET} │ Test error message")
        print(f"   {C.GREEN}3{C.RESET} │ Test plain text")
        print(f"   {C.RED}0{C.RESET} │ Done")
        c = _ask_int("Choice", 0, 3)

        if c == 1:
            test_signal = {
                "symbol": "BTCUSDT",
                "signal": "LONG",
                "confidence": 0.72,
                "combined_score": 0.35,
                "entry_price": 67500.0,
                "stop_loss": 66200.0,
                "take_profit": 70100.0,
                "risk_reward_ratio": 2.0,
                "risk_pct": 1.93,
                "reward_pct": 3.85,
                "components": {
                    "ml_ensemble": {"available": True, "score": 0.42, "signal": "LONG", "confidence": 0.71},
                    "sentiment": {"available": True, "score": 0.18, "signal": "LONG", "confidence": 0.59},
                    "ai_reasoning": {"available": True, "score": 0.30, "signal": "LONG", "confidence": 0.65},
                    "funding_rate": {"available": True, "score": -0.05, "signal": "SHORT", "confidence": 0.52},
                    "market_structure": {"available": True, "score": 0.25, "signal": "LONG", "confidence": 0.63},
                },
                "active_components": 5,
                "total_components": 5,
                "data_quality": {"score": 85, "issues": []},
                "analysis_time_s": 42.3,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            ok = tg.send_signal(test_signal)
            print(f"  {'✅ Test signal sent!' if ok else '❌ Failed'}")

        elif c == 2:
            ok = tg.send_error("main", "TestError", "This is a test error message from health check.")
            print(f"  {'✅ Test error sent!' if ok else '❌ Failed'}")

        elif c == 3:
            ok = tg.send_text("🤖 Crypto Agent v3.0 — manual test message. All systems go! ✅")
            print(f"  {'✅ Sent!' if ok else '❌ Failed'}")

    except Exception as e:
        logger.error(f"Telegram test failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ {e}{C.RESET}")


# ────────── 11. View Recent Signals ──────────

def action_signals():
    try:
        db = get_db()
        print(f"\n  View signals for:")
        print(f"   {C.GREEN}1{C.RESET} │ All pairs")
        print(f"   {C.GREEN}2{C.RESET} │ Single pair")
        print(f"   {C.RED}0{C.RESET} │ Cancel")
        c = _ask_int("Choice", 0, 2)
        if c == 0:
            return

        symbol = None
        if c == 2:
            symbol = _ask_symbol()
            if not symbol:
                return

        signals = db.get_signals(symbol=symbol, limit=20)

        if not signals:
            print(f"\n  No signals found.")
            return

        _hdr(f"📡  RECENT SIGNALS" + (f" — {symbol}" if symbol else ""))
        print(f"  {'Time':<20s} {'Symbol':<10s} {'Signal':<7s} "
              f"{'Conf':>6s} {'Score':>7s} {'Entry':>10s}")
        print(f"  {'─'*20} {'─'*10} {'─'*7} {'─'*6} {'─'*7} {'─'*10}")

        for sig in signals:
            ts = sig.get("timestamp", "?")
            if isinstance(ts, str) and len(ts) > 19:
                ts = ts[:19]
            sy = sig.get("symbol", "?")
            si_raw = sig.get("direction", "?")
            si = "LONG" if str(si_raw) in ("1", "LONG") else (
                 "SHORT" if str(si_raw) in ("-1", "SHORT") else "HOLD")
            co = sig.get("confidence", 0)
            # FIX: read combined_score from DB (new column), fallback 0
            sc = sig.get("combined_score") or 0
            ep = sig.get("entry_price", 0)
            em = "🟢" if si == "LONG" else ("🔴" if si == "SHORT" else "⚪")
            print(f"  {str(ts):<20s} {sy:<10s} {em}{si:<6s} "
                  f"{co:>5.1%} {sc:>+6.3f} {_price(ep):>10s}")

        print(f"\n  Total: {len(signals)} signal(s)")

    except Exception as e:
        logger.error(f"Signal view failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ {e}{C.RESET}")


# ────────── 12. View Trade History ──────────

def action_trades():
    try:
        db = get_db()
        print(f"\n  Trade history for mode:")
        print(f"   {C.GREEN}1{C.RESET} │ Paper trades")
        print(f"   {C.GREEN}2{C.RESET} │ Backtest trades")
        print(f"   {C.GREEN}3{C.RESET} │ All trades")
        print(f"   {C.RED}0{C.RESET} │ Cancel")
        c = _ask_int("Choice", 0, 3)
        if c == 0:
            return

        mode_map = {1: "paper", 2: "backtest", 3: None}
        mode = mode_map[c]
        mode_label = mode or "all"

        trades = db.get_trades(mode=mode, limit=30)

        if not trades:
            print(f"\n  No {mode_label} trades found.")
            return

        _hdr(f"📜  TRADE HISTORY — {mode_label.upper()}")
        print(f"  {'Symbol':<10s} {'Dir':<6s} {'Entry':>10s} "
              f"{'Exit':>10s} {'P&L $':>9s} {'P&L %':>8s} "
              f"{'Reason':<15s} {'Status':<8s}")
        print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*10} "
              f"{'─'*9} {'─'*8} {'─'*15} {'─'*8}")

        total_pnl = 0
        wins = 0
        losses = 0
        for t in trades:
            sy = t.get("symbol", "?")
            d_raw = t.get("direction", "?")
            d = "LONG" if str(d_raw) in ("1", "LONG") else (
                "SHORT" if str(d_raw) in ("-1", "SHORT") else str(d_raw)
            )
            ep = t.get("entry_price", 0)
            xp = t.get("exit_price")
            pn = t.get("pnl_usd", 0) or 0
            pp = t.get("pnl_percent", 0) or 0
            er = t.get("exit_reason", "—")
            st = t.get("status", "?")

            if st == "closed":
                total_pnl += pn
                if pn >= 0:
                    wins += 1
                else:
                    losses += 1

            pc = C.GREEN if pn >= 0 else C.RED
            print(f"  {sy:<10s} {d:<6s} {_price(ep):>10s} "
                  f"{_price(xp) if xp else '—':>10s} "
                  f"{pc}{pn:>+8.2f}{C.RESET} "
                  f"{pc}{pp:>+7.2f}%{C.RESET} "
                  f"{str(er):<15s} {st:<8s}")

        total_closed = wins + losses
        wr = wins / total_closed if total_closed > 0 else 0

        print(f"\n  Total: {len(trades)} trade(s)  │  "
              f"Closed: {total_closed}  │  "
              f"Win Rate: {wr:.1%}  │  "
              f"P&L: {_pnl(total_pnl)}")

    except Exception as e:
        logger.error(f"Trade view failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ {e}{C.RESET}")


# ────────── 13. Database Stats ──────────

def action_db_stats():
    try:
        db = get_db()
        info = db.get_table_info()

        _hdr("🗄️  DATABASE STATS")
        print(f"    Path: {config.DB_PATH}")
        size_mb = config.DB_PATH.stat().st_size / (1024 * 1024) if config.DB_PATH.exists() else 0
        print(f"    Size: {size_mb:.2f} MB\n")

        print(f"    {'Table':<20s} {'Rows':>8s}")
        print(f"    {'─'*20} {'─'*8}")
        for table, count in info.items():
            print(f"    {table:<20s} {count:>8d}")

        # Per-mode stats
        for mode in ["paper", "backtest"]:
            stats = db.get_stats(mode=mode)
            if stats and stats.get("total_trades", 0) > 0:
                print(f"\n    {C.BOLD}── {mode.upper()} Stats ──{C.RESET}")
                print(f"    Total Trades:  {stats.get('total_trades', 0)}")
                print(f"    Win Rate:      {stats.get('win_rate', 0):.1f}%")
                print(f"    Total P&L:     {_pnl(stats.get('total_pnl', 0))}")
                pf = stats.get("profit_factor", 0)
                print(f"    Profit Factor: {pf:.2f}")

        # Recent errors
        errors = db.get_errors(limit=5)
        if errors:
            print(f"\n    {C.BOLD}── Recent Errors ──{C.RESET}")
            for err in errors:
                ts  = err.get("timestamp", "?")
                mod = err.get("module", "?")
                et  = err.get("error_type", "?")
                msg = err.get("message", "?")
                if isinstance(ts, str) and len(ts) > 19:
                    ts = ts[:19]
                print(f"    {C.RED}{ts}  {mod}  {et}{C.RESET}")
                print(f"      {msg[:80]}")

        print(f"{'═'*60}")

    except Exception as e:
        logger.error(f"DB stats failed: {e}", exc_info=True)
        print(f"\n  {C.RED}❌ {e}{C.RESET}")



def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crypto Futures AI Agent v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     Interactive menu
  python main.py --status            System health check
  python main.py --prices            Quick Binance price check
  python main.py --analyze BTCUSDT   Analyze single pair
  python main.py --scan              Scan all trading pairs
  python main.py --train BTCUSDT     Train model for one pair
  python main.py --train-all         Train models for all pairs
  python main.py --backtest BTCUSDT  Backtest single pair
  python main.py --backtest-all      Backtest all pairs
  python main.py --paper             Start paper trading
  python main.py --paper-cycle       Run one paper trading cycle
        """,
    )

    p.add_argument("--status", action="store_true",
                    help="System health check")
    p.add_argument("--prices", action="store_true",
                    help="Quick Binance price check")
    p.add_argument("--analyze", type=str, metavar="SYMBOL",
                    help="Full analysis for a symbol (e.g., BTCUSDT)")
    p.add_argument("--scan", action="store_true",
                    help="Scan all trading pairs")
    p.add_argument("--train", type=str, metavar="SYMBOL",
                    help="Train model for a symbol")
    p.add_argument("--train-all", action="store_true",
                    help="Train models for all pairs")
    p.add_argument("--backtest", type=str, metavar="SYMBOL",
                    help="Backtest a symbol")
    p.add_argument("--backtest-all", action="store_true",
                    help="Backtest all trading pairs")
    p.add_argument("--paper", action="store_true",
                    help="Start continuous paper trading")
    p.add_argument("--paper-cycle", action="store_true",
                    help="Run one paper trading cycle")
    p.add_argument("--no-ai", action="store_true",
                    help="Disable AI reasoning (faster)")
    p.add_argument("--signals", action="store_true",
                    help="View recent signals from DB")
    p.add_argument("--trades", action="store_true",
                    help="View trade history from DB")

    return p


def _run_cli(args):
    """Handle command-line arguments (non-interactive mode)."""
    no_ai = args.no_ai

    if args.status:
        action_health()
        return True

    if args.prices:
        action_prices()
        return True

    if args.analyze:
        sym = args.analyze.upper()
        if sym not in config.TRADING_PAIRS:
            print(f"  {C.YELLOW}⚠️  {sym} not in TRADING_PAIRS, proceeding anyway...{C.RESET}")
        print(f"\n  🔄 Analyzing {sym} ...")
        try:
            trainer = _get("trainer")
            st = trainer.get_training_status([sym])
            if not st.get("per_symbol", {}).get(sym, {}).get("exists"):
                print(f"  🧠 No model found, training {sym} first ...")
                tr = trainer.train_symbol(sym, save_model=True)
                if tr.get("error"):
                    print(f"  ⚠️  Training issue: {tr['error']}. Continuing...")

            engine = _get("signal_engine")
            include_ai = config.AI_CONFIG["enabled"] and not no_ai
            result = engine.generate_signal(sym, include_ai=include_ai)
            _show_signal(result)

            if config.TELEGRAM_CONFIG["enabled"] and result.get("signal") != "HOLD":
                tg = _get("telegram")
                tg.send_signal(result)
                print(f"  📱 Signal sent to Telegram")
        except Exception as e:
            logger.error(f"CLI analyze failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.scan:
        print(f"\n  🔄 Scanning all pairs ...")
        try:
            engine = _get("signal_engine")
            include_ai = config.AI_CONFIG["enabled"] and not no_ai
            scan = engine.scan_all(include_ai=include_ai)
            all_r = scan.get("all", {})
            summ = scan.get("summary", {})

            _hdr("📊  SCAN RESULTS")
            for sym, res in all_r.items():
                if isinstance(res, dict) and "signal" in res:
                    s = res["signal"]
                    co = res.get("confidence", 0)
                    sc = res.get("combined_score", 0)
                    em = "🟢" if s == "LONG" else ("🔴" if s == "SHORT" else "⚪")
                    print(f"    {em} {sym:<10s} {s:<6s} conf:{co:.1%} score:{sc:+.3f}")
                else:
                    e = res.get("error", "failed") if isinstance(res, dict) else "failed"
                    print(f"    ❌ {sym:<10s} {e}")

            act = scan.get("actionable", {})
            for sym, res in act.items():
                _show_signal(res)

            print(f"\n  Scan: {summ.get('total',0)} pairs, "
                  f"{summ.get('actionable',0)} actionable, "
                  f"{summ.get('scan_time_s',0):.1f}s")

            if config.TELEGRAM_CONFIG["enabled"] and act:
                tg = _get("telegram")
                tg.send_scan_results(scan)
                print(f"  📱 Results sent to Telegram")
        except Exception as e:
            logger.error(f"CLI scan failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.train:
        sym = args.train.upper()
        print(f"\n  🧠 Training {sym} ...")
        try:
            trainer = _get("trainer")
            r = trainer.train_symbol(sym, save_model=True)
            _show_train(sym, r)
        except Exception as e:
            logger.error(f"CLI train failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.train_all:
        print(f"\n  🧠 Training all {len(config.TRADING_PAIRS)} pairs ...")
        try:
            trainer = _get("trainer")
            r = trainer.train_all()
            s = r.get("summary", {})
            print(f"\n  ✅ Done: {s.get('successful',0)}/{s.get('total',0)} "
                  f"successful, avg accuracy: {s.get('avg_accuracy',0):.1%}")
            for sym, res in r.get("per_symbol", {}).items():
                _show_train(sym, res, brief=True)
        except Exception as e:
            logger.error(f"CLI train-all failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.backtest:
        sym = args.backtest.upper()
        print(f"\n  📈 Backtesting {sym} ...")
        try:
            bt = _get("backtester")
            r = bt.run(sym)
            _show_backtest(r)
        except Exception as e:
            logger.error(f"CLI backtest failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.backtest_all:
        print(f"\n  📈 Backtesting all {len(config.TRADING_PAIRS)} pairs ...")
        try:
            bt = _get("backtester")
            r = bt.run_all()
            s = r.get("summary", {})
            # FIX: backtester uses total_symbols and avg_return_pct
            total_sym = s.get("total_symbols", s.get("total", 0))
            avg_ret = s.get("avg_return_pct", s.get("avg_return", 0))
            print(f"\n  Summary: {s.get('successful',0)}/{total_sym} "
                  f"successful, avg return: {avg_ret:+.2f}%, "
                  f"avg Sharpe: {s.get('avg_sharpe',0):.2f}")
            for sym, res in r.get("per_symbol", {}).items():
                _show_backtest(res, brief=True)
        except Exception as e:
            logger.error(f"CLI backtest-all failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.paper:
        include_ai = config.AI_CONFIG["enabled"] and not no_ai
        print(f"\n  🚀 Starting paper trading (Ctrl+C to stop) ...")
        try:
            _write_pid()  # FIX: write PID file for daemon management
            paper = _new_paper_trader(include_ai)
            if config.TELEGRAM_CONFIG["enabled"]:
                _get("telegram").send_startup(
                    "paper",
                    config.BACKTEST_CONFIG["initial_capital"],
                    config.TRADING_PAIRS,
                )

            # Start Telegram command bot for remote monitoring
            cmd_bot = None
            if config.TELEGRAM_CONFIG["enabled"]:
                try:
                    from notifications.telegram_bot import TelegramCommandBot
                    cmd_bot = TelegramCommandBot()
                    cmd_bot.start()
                except Exception as e:
                    logger.warning(f"Telegram command bot: {e}")

            paper.start()

            if cmd_bot:
                cmd_bot.stop()

            st = paper.get_status()

            print(f"\n  Stopped. Capital: ${st.get('capital',0):,.2f}, "
                  f"P&L: {_pnl(st.get('pnl',0))}")
            if config.TELEGRAM_CONFIG["enabled"]:
                _get("telegram").send_shutdown(
                    st.get("capital", 0), st.get("pnl", 0)
                )
        except KeyboardInterrupt:
            print(f"\n  Paper trading stopped by user.")
        except Exception as e:
            logger.error(f"CLI paper failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        finally:
            _remove_pid()  # FIX: clean up PID file on exit
        return True

    if args.paper_cycle:
        include_ai = config.AI_CONFIG["enabled"] and not no_ai
        print(f"\n  🔄 Running one paper trading cycle ...")
        try:
            paper = _new_paper_trader(include_ai)
            cr = paper.run_cycle()
            print(f"  Cycle #{cr.get('cycle',0)} — {cr.get('cycle_time_s',0):.1f}s")
            print(f"  Positions: checked={cr.get('positions_checked',0)}, "
                  f"closed={cr.get('positions_closed',0)}, "
                  f"opened={cr.get('positions_opened',0)}")
            print(f"  Capital: ${cr.get('capital',0):,.2f}")
        except Exception as e:
            logger.error(f"CLI paper-cycle failed: {e}", exc_info=True)
            print(f"  {C.RED}❌ {e}{C.RESET}")
        return True

    if args.signals:
        action_signals()
        return True

    if args.trades:
        action_trades()
        return True

    return False  # no CLI args → fall through to interactive


_ACTION_MAP = {
    1:  action_analyze,
    2:  action_scan,
    3:  action_prices,
    4:  action_train,
    5:  action_model_status,
    6:  action_backtest,
    7:  action_backtest_all,
    8:  action_paper,
    9:  action_health,
    10: action_test_telegram,
    11: action_signals,
    12: action_trades,
    13: action_db_stats,
}


def _interactive():
    """Main interactive menu loop."""
    _banner()
    _startup_hints()

    while True:
        _menu()
        choice = _ask_int("Select", 0, 13)

        if choice == 0:
            print(f"\n  {C.CYAN}👋  Goodbye! Happy trading.{C.RESET}\n")
            break

        action = _ACTION_MAP.get(choice)
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print(f"\n  {C.YELLOW}Interrupted.{C.RESET}")
            except Exception as e:
                logger.error(f"Action {choice} failed: {e}", exc_info=True)
                print(f"\n  {C.RED}❌ Unexpected error: {e}{C.RESET}")
                print(f"  {C.DIM}Check logs/agent.log for details.{C.RESET}")

        input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")




def main():
    """Main entry point — CLI args or interactive menu."""
    try:
        parser = _build_parser()
        args = parser.parse_args()

        # If any CLI arg was given, run non-interactively
        if _run_cli(args):
            return

        # Otherwise → interactive menu
        _interactive()

    except KeyboardInterrupt:
        print(f"\n\n  {C.CYAN}👋  Goodbye!{C.RESET}\n")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\n  {C.RED}💀 Fatal error: {e}{C.RESET}")
        print(f"  {C.DIM}Check logs/agent.log for full traceback.{C.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()