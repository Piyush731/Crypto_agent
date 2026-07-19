"""Restart-safe, forward-only aggressive TSM paper runtime.

Public OKX data only. This module contains no private exchange client and no
order-placement method. It starts from a fresh ledger and never imports fills.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from data.okx_collector import collect
from paper_v4.store import PaperStore
from paper_v4.telegram import TelegramNotifier
from portfolio_v4.aggressive_forward_tsm import AggressiveForwardTimeSeriesMomentum
from providers.okx import OKXMarketData

SYMBOLS = {
    symbol: f"{symbol}-USDT-SWAP"
    for symbol in AggressiveForwardTimeSeriesMomentum.candidate_symbols
}
COLLECTOR_SYMBOLS = tuple(f"{symbol}USDT" for symbol in SYMBOLS)
PREFIX = "🔥 T12 HIGH-RISK FORWARD PAPER\nNO REAL ORDERS — NOT VALIDATED\n"


def money(value: float) -> str:
    return f"{value:,.2f} USDT"


def side(direction: int) -> str:
    return "LONG" if direction == 1 else "SHORT"


class AggressiveForwardRunner:
    def __init__(self, market_db: Path, paper_db: Path, starting_capital=10_000.0):
        self.market = MarketStore(market_db)
        self.paper = PaperStore(paper_db)
        self.strategy = AggressiveForwardTimeSeriesMomentum()
        self.starting_capital = float(starting_capital)

    def frames(self):
        now = datetime.now(timezone.utc)
        hourly_start = int((now - timedelta(days=110)).timestamp() * 1000)
        five_start = int((now - timedelta(days=10)).timestamp() * 1000)
        hourly = {
            symbol: self.market.get_candles(
                "okx", instrument, "1h", start_ms=hourly_start, completed_only=True
            )
            for symbol, instrument in SYMBOLS.items()
        }
        five = {
            symbol: self.market.get_candles(
                "okx", instrument, "5m", start_ms=five_start, completed_only=True
            )
            for symbol, instrument in SYMBOLS.items()
        }
        missing = [
            symbol for symbol in SYMBOLS
            if len(hourly[symbol]) < 2161 or five[symbol].empty
        ]
        if missing:
            raise RuntimeError(
                f"Insufficient 90-day warm-up or 5m data for: {missing}"
            )
        return hourly, five

    @staticmethod
    def common_timeline(five):
        common = None
        for frame in five.values():
            common = frame.index if common is None else common.intersection(frame.index)
        return common.sort_values()

    def run(self):
        hourly, five = self.frames()
        timeline = self.common_timeline(five)
        if timeline.empty:
            raise RuntimeError("No common completed 5m timeline")
        latest = timeline[-1]
        age = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60
        if age > 15:
            raise RuntimeError(f"Stale common 5m data: {latest.isoformat()} age={age:.1f}m")

        last_raw = self.paper.get_state("last_processed_bar")
        if last_raw is None:
            with self.paper.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.paper.set_state(connection, "capital", self.starting_capital)
                self.paper.set_state(connection, "peak_equity", self.starting_capital)
                self.paper.set_state(connection, "hard_halted", False)
                self.paper.set_state(connection, "last_processed_bar", latest.isoformat())
                self.paper.set_state(connection, "forward_start", datetime.now(timezone.utc).isoformat())
                self.paper.set_state(connection, "strategy", self.strategy.metadata())
                self.paper.queue(
                    connection,
                    "t12-forward-initialized",
                    "\n".join(
                        [
                            "✅ Aggressive forward ledger initialized",
                            f"Starting paper balance: {money(self.starting_capital)}",
                            f"Baseline bar: {latest.isoformat()}",
                            "Historical candles are warm-up only; imported fills: 0.",
                            "Risk: 0.20% per leg; max six legs; gross cap 75%.",
                            "Hard paper halt: 25% drawdown; manual review required.",
                        ]
                    ),
                )
            return {"initialized": True, "baseline_bar": latest.isoformat(), "processed": 0}

        last = pd.Timestamp(last_raw)
        new_timeline = timeline[timeline > last]
        if new_timeline.empty:
            return {"initialized": False, "processed": 0, "latest": latest.isoformat()}
        if int(new_timeline[0].value) - int(last.value) != 300_000_000_000:
            raise RuntimeError(f"Unsafe 5m gap after {last.isoformat()}")
        if len(new_timeline) > 1:
            deltas = new_timeline.asi8[1:] - new_timeline.asi8[:-1]
            if not (deltas == 300_000_000_000).all():
                raise RuntimeError("Unsafe missing common 5m candle")

        schedule = self.strategy.build_schedule(hourly)
        for timestamp in new_timeline:
            self.process_bar(timestamp, five, schedule)
        return {"initialized": False, "processed": len(new_timeline), "latest": new_timeline[-1].isoformat()}

    def process_bar(self, timestamp, five, schedule):
        cfg = self.strategy.config
        fee_rate = cfg.fee_bps_per_side / 10_000
        friction = (cfg.slippage_bps_per_side + cfg.half_spread_bps_per_side) / 10_000
        bars = {symbol: frame.loc[timestamp] for symbol, frame in five.items()}
        signal_time = pd.Timestamp(
            int(timestamp.value) - 300_000_000_000, unit="ns", tz="UTC"
        )

        with self.paper.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            capital = float(self.paper.get_state("capital", self.starting_capital))
            peak = float(self.paper.get_state("peak_equity", self.starting_capital))
            halted = bool(self.paper.get_state("hard_halted", False))
            positions = self.paper.positions(connection)

            def close_position(symbol, base_price, reason):
                nonlocal capital
                position = positions.pop(symbol)
                direction = int(position["direction"])
                exit_price = base_price * (1 - direction * friction)
                exit_fee = float(position["quantity"]) * exit_price * fee_rate
                gross = (
                    (exit_price - float(position["entry_price"]))
                    * float(position["quantity"]) * direction
                )
                net = gross - float(position["entry_fee"]) - exit_fee - float(position["funding_cost"])
                capital += net
                cursor = connection.execute(
                    """INSERT INTO trades(
                       symbol,direction,signal_time,entry_time,entry_price,quantity,
                       notional,stop_loss,entry_fee,funding_cost,exit_time,exit_price,
                       exit_fee,gross_pnl,net_pnl,exit_reason
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        symbol, direction, position["signal_time"], position["entry_time"],
                        position["entry_price"], position["quantity"], position["notional"],
                        position["stop_loss"], position["entry_fee"], position["funding_cost"],
                        timestamp.isoformat(), exit_price, exit_fee, gross, net, reason,
                    ),
                )
                connection.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
                self.paper.queue(
                    connection,
                    f"t12-close-{cursor.lastrowid}",
                    "\n".join(
                        [
                            f"{'✅' if net >= 0 else '❌'} CLOSED — {reason.upper()}",
                            f"{symbol} {side(direction)}",
                            f"Net P&L: {money(net)}",
                            f"Capital: {money(capital)}",
                            f"Time: {timestamp.isoformat()}",
                        ]
                    ),
                )

            equity_open = capital + sum(
                (float(bars[symbol]["open"]) - float(position["entry_price"]))
                * float(position["quantity"]) * int(position["direction"])
                - float(position["entry_fee"]) - float(position["funding_cost"])
                for symbol, position in positions.items()
            )
            peak = max(peak, equity_open)
            drawdown_pct = (equity_open / peak - 1) * 100
            if not halted and drawdown_pct <= -self.strategy.hard_halt_drawdown_pct:
                halted = True
                for symbol in list(positions):
                    close_position(symbol, float(bars[symbol]["open"]), "hard_drawdown_halt")
                self.paper.queue(
                    connection,
                    "t12-hard-halt",
                    f"🛑 PERMANENT PAPER HALT\nDrawdown: {drawdown_pct:.2f}%\nNo new positions until manual reset.",
                )

            if signal_time in schedule.index and not halted:
                already = connection.execute(
                    "SELECT 1 FROM processed_signals WHERE signal_time=?",
                    (signal_time.isoformat(),),
                ).fetchone()
                if already is None:
                    row = schedule.loc[signal_time]
                    desired = {symbol: 1 for symbol in tuple(row["long_symbols"])}
                    desired.update({symbol: -1 for symbol in tuple(row["short_symbols"])})
                    atr_pct = dict(row["atr_pct"])
                    for symbol in list(positions):
                        if desired.get(symbol) != int(positions[symbol]["direction"]):
                            close_position(symbol, float(bars[symbol]["open"]), "weekly_rebalance")

                    equity_open = capital + sum(
                        (float(bars[symbol]["open"]) - float(position["entry_price"]))
                        * float(position["quantity"]) * int(position["direction"])
                        - float(position["entry_fee"]) - float(position["funding_cost"])
                        for symbol, position in positions.items()
                    )
                    gross_limit = equity_open * cfg.max_gross_notional_pct / 100
                    gross = sum(float(position["notional"]) for position in positions.values())
                    opened = []
                    for symbol, direction in desired.items():
                        if symbol in positions:
                            continue
                        base = float(bars[symbol]["open"])
                        fill = base * (1 + direction * friction)
                        stop_distance = fill * float(atr_pct[symbol]) / 100 * cfg.stop_atr_multiple
                        risk_amount = equity_open * cfg.risk_per_leg_pct / 100
                        by_risk = risk_amount / stop_distance if stop_distance > 0 else 0
                        leg_limit = equity_open * cfg.max_notional_per_leg_pct / 100
                        available = max(gross_limit - gross, 0.0)
                        quantity = min(by_risk, min(leg_limit, available) / fill)
                        if quantity <= 0:
                            continue
                        notional = quantity * fill
                        position = {
                            "symbol": symbol, "direction": direction,
                            "signal_time": signal_time.isoformat(), "entry_time": timestamp.isoformat(),
                            "entry_price": fill, "quantity": quantity, "notional": notional,
                            "stop_loss": fill - direction * stop_distance,
                            "entry_fee": notional * fee_rate, "funding_cost": 0.0,
                        }
                        connection.execute(
                            """INSERT INTO positions(
                               symbol,direction,signal_time,entry_time,entry_price,quantity,
                               notional,stop_loss,entry_fee,funding_cost
                               ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            tuple(position[key] for key in [
                                "symbol", "direction", "signal_time", "entry_time", "entry_price",
                                "quantity", "notional", "stop_loss", "entry_fee", "funding_cost",
                            ]),
                        )
                        positions[symbol] = position
                        gross += notional
                        opened.append(symbol)
                        self.paper.queue(
                            connection,
                            f"t12-open-{signal_time.isoformat()}-{symbol}",
                            "\n".join(
                                [
                                    f"📈 OPENED — {symbol} {side(direction)}",
                                    f"Paper fill: {fill:.6f}",
                                    f"Notional: {money(notional)}",
                                    f"Stop: {position['stop_loss']:.6f}",
                                    "Take profit: none; weekly signal or 3×ATR stop.",
                                ]
                            ),
                        )
                    connection.execute(
                        """INSERT INTO processed_signals(
                           signal_time,execution_time,long_symbol,short_symbol
                           ) VALUES (?,?,?,?)""",
                        (
                            signal_time.isoformat(), timestamp.isoformat(),
                            ",".join(tuple(row["long_symbols"])),
                            ",".join(tuple(row["short_symbols"])),
                        ),
                    )
                    self.paper.queue(
                        connection,
                        f"t12-weekly-{signal_time.isoformat()}",
                        f"🔄 WEEKLY SIGNAL\nOpened: {', '.join(opened) if opened else 'none'}\nActive: {len(positions)}",
                    )

            for symbol in list(positions):
                position = positions[symbol]
                direction = int(position["direction"])
                bar = bars[symbol]
                stop = float(position["stop_loss"])
                hit = float(bar["low"]) <= stop if direction == 1 else float(bar["high"]) >= stop
                if hit:
                    base_exit = min(float(bar["open"]), stop) if direction == 1 else max(float(bar["open"]), stop)
                    close_position(symbol, base_exit, "atr_stop")
                else:
                    increment = float(position["notional"]) * cfg.funding_bps_per_8h / 10_000 / 96
                    position["funding_cost"] = float(position["funding_cost"]) + increment
                    connection.execute(
                        "UPDATE positions SET funding_cost=? WHERE symbol=?",
                        (position["funding_cost"], symbol),
                    )

            unrealized = sum(
                (float(bars[symbol]["close"]) - float(position["entry_price"]))
                * float(position["quantity"]) * int(position["direction"])
                - float(position["entry_fee"]) - float(position["funding_cost"])
                for symbol, position in positions.items()
            )
            equity = capital + unrealized
            peak = max(peak, equity)
            connection.execute(
                """INSERT INTO equity_snapshots(bar_time,equity,capital,position_count)
                   VALUES (?,?,?,?)""",
                (timestamp.isoformat(), equity, capital, len(positions)),
            )
            self.paper.set_state(connection, "capital", capital)
            self.paper.set_state(connection, "peak_equity", peak)
            self.paper.set_state(connection, "hard_halted", halted)
            self.paper.set_state(connection, "last_processed_bar", timestamp.isoformat())
            report_date = timestamp.strftime("%Y-%m-%d")
            if timestamp.hour == 0 and self.paper.get_state("last_daily_report") != report_date:
                trade_row = connection.execute(
                    "SELECT COUNT(*) count, COALESCE(SUM(net_pnl),0) pnl FROM trades"
                ).fetchone()
                current_dd = (equity / peak - 1) * 100
                self.paper.queue(
                    connection,
                    f"t12-daily-{report_date}",
                    "\n".join(
                        [
                            f"📊 DAILY SUMMARY — {report_date} UTC",
                            f"Equity: {money(equity)}",
                            f"Return: {(equity / self.starting_capital - 1) * 100:.2f}%",
                            f"Drawdown: {current_dd:.2f}%",
                            f"Realized P&L: {money(float(trade_row['pnl']))}",
                            f"Closed trades: {int(trade_row['count'])}",
                            f"Open positions: {len(positions)}",
                            f"Hard halted: {halted}",
                        ]
                    ),
                )
                self.paper.set_state(connection, "last_daily_report", report_date)


def validate_environment():
    required = {
        "AGENT_MODE": "paper",
        "MARKET_PROVIDER": "okx",
        "ALLOW_REAL_ORDERS": "false",
        "FORWARD_ONLY": "true",
    }
    for key, expected in required.items():
        if os.getenv(key, "").strip().lower() != expected:
            raise RuntimeError(f"{key} must equal {expected}")


def main():
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=os.getenv("T12_ENV_FILE", ".env.t12-forward"))
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env)
    validate_environment()
    market_db = Path(os.environ["MARKET_DB_PATH"])
    paper_db = Path(os.environ["PAPER_DB_PATH"])
    paper = PaperStore(paper_db)
    notifier = TelegramNotifier(paper, prefix=PREFIX)
    try:
        if not args.skip_collect:
            summary = collect(
                MarketStore(market_db), OKXMarketData(), COLLECTOR_SYMBOLS,
                ("5m", "1h"), 300,
            )
            if summary["errors"]:
                raise RuntimeError("; ".join(summary["errors"]))
        result = AggressiveForwardRunner(market_db, paper_db).run()
        print(result)
    except Exception as error:
        with paper.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            minute = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
            paper.queue(connection, f"t12-error-{minute}", f"🚨 RUNTIME ERROR\n{type(error).__name__}: {error}")
        notifier.flush()
        raise
    notifier.flush()


if __name__ == "__main__":
    main()
