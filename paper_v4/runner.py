"""Forward-only, restart-safe V7 paper runner using completed OKX candles.

This module has no authenticated exchange client and no order-placement method.
It reproduces the corrected backtest policy: a signal available on an hourly
boundary executes on the first completed 5m bar whose open is strictly later.
Notifications can therefore arrive shortly after that execution bar completes.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from data.okx_collector import DEFAULT_SYMBOLS, collect
from paper_v4.store import PaperStore
from paper_v4.telegram import TelegramNotifier
from portfolio_v4.registry import get_portfolio_strategy
from providers.okx import OKXMarketData

SYMBOLS = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "BNB": "BNB-USDT-SWAP",
}
STRATEGY_KEY = "cross_sectional_momentum_4h_v1"


def money(value: float) -> str:
    return f"{value:,.2f} USDT"


def side(direction: int) -> str:
    return "LONG" if direction == 1 else "SHORT"


class ForwardPaperRunner:
    def __init__(self, market_db: Path, paper_db: Path, starting_capital: float = 10_000.0):
        self.market = MarketStore(market_db)
        self.paper = PaperStore(paper_db)
        self.strategy = get_portfolio_strategy(STRATEGY_KEY)
        self.starting_capital = starting_capital

    def _frames(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        now = datetime.now(timezone.utc)
        start_ms = int((now - timedelta(days=10)).timestamp() * 1000)
        hourly = {
            symbol: self.market.get_candles(
                "okx", instrument, "1h", start_ms=start_ms, completed_only=True
            )
            for symbol, instrument in SYMBOLS.items()
        }
        five = {
            symbol: self.market.get_candles(
                "okx", instrument, "5m", start_ms=start_ms, completed_only=True
            )
            for symbol, instrument in SYMBOLS.items()
        }
        for name, collection in (("1h", hourly), ("5m", five)):
            missing = [symbol for symbol, frame in collection.items() if frame.empty]
            if missing:
                raise RuntimeError(f"Missing completed {name} candles: {missing}")
        return hourly, five

    @staticmethod
    def _common_timeline(five: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        common = None
        for frame in five.values():
            common = frame.index if common is None else common.intersection(frame.index)
        return common.sort_values()

    def initialize_or_run(self) -> dict:
        hourly, five = self._frames()
        timeline = self._common_timeline(five)
        if timeline.empty:
            raise RuntimeError("No common completed 5m timeline")
        latest = timeline[-1]
        age_minutes = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60
        if age_minutes > 15:
            raise RuntimeError(
                f"Market data stale: latest common 5m open={latest.isoformat()}, "
                f"age={age_minutes:.1f}m"
            )

        last_raw = self.paper.get_state("last_processed_bar")
        if last_raw is None:
            with self.paper.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self.paper.set_state(connection, "capital", self.starting_capital)
                self.paper.set_state(connection, "strategy_key", STRATEGY_KEY)
                self.paper.set_state(connection, "forward_start", datetime.now(timezone.utc).isoformat())
                self.paper.set_state(connection, "last_processed_bar", latest.isoformat())
                self.paper.queue(
                    connection,
                    "runtime-initialized",
                    "\n".join(
                        [
                            "✅ Forward paper ledger initialized",
                            f"Strategy: {STRATEGY_KEY}",
                            f"Starting balance: {money(self.starting_capital)}",
                            f"Baseline bar: {latest.isoformat()}",
                            "Old candles were used only for indicator warm-up.",
                            "No historical fills were imported.",
                            "Exit policy: 2×ATR stop or 4h rebalance; no take-profit rule.",
                        ]
                    ),
                )
            return {"initialized": True, "baseline_bar": latest.isoformat(), "processed": 0}

        last = pd.Timestamp(last_raw)
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        new_timeline = timeline[timeline > last]
        if new_timeline.empty:
            return {"initialized": False, "processed": 0, "latest": latest.isoformat()}
        if new_timeline[0] - last != pd.Timedelta(minutes=5):
            raise RuntimeError(
                f"Unsafe 5m gap: last={last.isoformat()} next={new_timeline[0].isoformat()}"
            )
        deltas = new_timeline.to_series().diff().dropna()
        if not deltas.empty and not (deltas == pd.Timedelta(minutes=5)).all():
            raise RuntimeError("Unsafe missing common 5m candle in processing range")

        schedule = self.strategy.build_schedule(hourly)
        for timestamp in new_timeline:
            self._process_bar(timestamp, five, schedule)
        return {
            "initialized": False,
            "processed": len(new_timeline),
            "latest": new_timeline[-1].isoformat(),
        }

    def _process_bar(
        self,
        timestamp: pd.Timestamp,
        five: dict[str, pd.DataFrame],
        schedule: pd.DataFrame,
    ) -> None:
        cfg = self.strategy.config
        fee_rate = cfg.fee_bps_per_side / 10_000
        friction = (cfg.slippage_bps_per_side + cfg.half_spread_bps_per_side) / 10_000
        bars = {symbol: frame.loc[timestamp] for symbol, frame in five.items()}
        signal_time = timestamp - pd.Timedelta(minutes=5)

        with self.paper.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            capital = float(self.paper.get_state("capital", self.starting_capital))
            positions = self.paper.positions(connection)

            def close_position(symbol: str, base_price: float, reason: str) -> None:
                nonlocal capital
                position = positions.pop(symbol)
                direction = int(position["direction"])
                exit_price = base_price * (1 - direction * friction)
                exit_notional = float(position["quantity"]) * exit_price
                exit_fee = exit_notional * fee_rate
                gross = (
                    (exit_price - float(position["entry_price"]))
                    * float(position["quantity"])
                    * direction
                )
                net = (
                    gross
                    - float(position["entry_fee"])
                    - exit_fee
                    - float(position["funding_cost"])
                )
                capital += net
                cursor = connection.execute(
                    """
                    INSERT INTO trades(
                        symbol,direction,signal_time,entry_time,entry_price,quantity,
                        notional,stop_loss,entry_fee,funding_cost,exit_time,exit_price,
                        exit_fee,gross_pnl,net_pnl,exit_reason
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        symbol, direction, position["signal_time"], position["entry_time"],
                        position["entry_price"], position["quantity"], position["notional"],
                        position["stop_loss"], position["entry_fee"], position["funding_cost"],
                        timestamp.isoformat(), exit_price, exit_fee, gross, net, reason,
                    ),
                )
                connection.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
                emoji = "✅" if net >= 0 else "❌"
                self.paper.queue(
                    connection,
                    f"trade-close-{cursor.lastrowid}",
                    "\n".join(
                        [
                            f"{emoji} POSITION CLOSED — {reason.upper()}",
                            f"{symbol} {side(direction)}",
                            f"Entry: {float(position['entry_price']):.6f}",
                            f"Exit: {exit_price:.6f}",
                            f"Net P&L: {money(net)}",
                            f"Funding model cost: {money(float(position['funding_cost']))}",
                            f"Realized capital: {money(capital)}",
                            f"Time: {timestamp.isoformat()}",
                        ]
                    ),
                )

            if signal_time in schedule.index:
                row = schedule.loc[signal_time]
                already = connection.execute(
                    "SELECT 1 FROM processed_signals WHERE signal_time=?",
                    (signal_time.isoformat(),),
                ).fetchone()
                if already is None:
                    desired: dict[str, tuple[int, float]] = {}
                    if pd.notna(row.get("long_symbol")) and row.get("long_symbol"):
                        desired[str(row["long_symbol"])] = (1, float(row["long_atr_pct"]))
                    if pd.notna(row.get("short_symbol")) and row.get("short_symbol"):
                        desired[str(row["short_symbol"])] = (-1, float(row["short_atr_pct"]))

                    for symbol in list(positions):
                        desired_direction = desired.get(symbol, (0, 0.0))[0]
                        if desired_direction != int(positions[symbol]["direction"]):
                            close_position(symbol, float(bars[symbol]["open"]), "rebalance")

                    equity_at_open = capital + sum(
                        (float(bars[symbol]["open"]) - float(position["entry_price"]))
                        * float(position["quantity"])
                        * int(position["direction"])
                        for symbol, position in positions.items()
                    )
                    opened = []
                    for symbol, (direction, atr_pct) in desired.items():
                        if symbol in positions:
                            continue
                        base_price = float(bars[symbol]["open"])
                        entry_price = base_price * (1 + direction * friction)
                        stop_distance = entry_price * atr_pct / 100 * cfg.stop_atr_multiple
                        risk_amount = equity_at_open * cfg.risk_per_leg_pct / 100
                        quantity_by_risk = risk_amount / stop_distance if stop_distance > 0 else 0
                        max_notional = equity_at_open * cfg.max_notional_per_leg_pct / 100
                        quantity = min(quantity_by_risk, max_notional / entry_price)
                        if quantity <= 0:
                            continue
                        notional = quantity * entry_price
                        position = {
                            "symbol": symbol,
                            "direction": direction,
                            "signal_time": signal_time.isoformat(),
                            "entry_time": timestamp.isoformat(),
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "notional": notional,
                            "stop_loss": entry_price - direction * stop_distance,
                            "entry_fee": notional * fee_rate,
                            "funding_cost": 0.0,
                        }
                        connection.execute(
                            """INSERT INTO positions(
                               symbol,direction,signal_time,entry_time,entry_price,quantity,
                               notional,stop_loss,entry_fee,funding_cost
                               ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            tuple(position[key] for key in [
                                "symbol", "direction", "signal_time", "entry_time",
                                "entry_price", "quantity", "notional", "stop_loss",
                                "entry_fee", "funding_cost",
                            ]),
                        )
                        positions[symbol] = position
                        opened.append(symbol)
                        self.paper.queue(
                            connection,
                            f"position-open-{signal_time.isoformat()}-{symbol}",
                            "\n".join(
                                [
                                    f"📈 POSITION OPENED — {symbol} {side(direction)}",
                                    f"Signal: {signal_time.isoformat()}",
                                    f"Paper fill: {entry_price:.6f}",
                                    f"Quantity: {quantity:.8f}",
                                    f"Notional: {money(notional)}",
                                    f"Stop loss: {position['stop_loss']:.6f}",
                                    "Take profit: none (exit at stop or rebalance)",
                                    f"Model risk: {cfg.risk_per_leg_pct:.2f}% of equity",
                                ]
                            ),
                        )
                    connection.execute(
                        """INSERT INTO processed_signals(
                           signal_time,execution_time,long_symbol,short_symbol
                           ) VALUES (?,?,?,?)""",
                        (
                            signal_time.isoformat(), timestamp.isoformat(),
                            row.get("long_symbol") if pd.notna(row.get("long_symbol")) else None,
                            row.get("short_symbol") if pd.notna(row.get("short_symbol")) else None,
                        ),
                    )
                    self.paper.queue(
                        connection,
                        f"rebalance-{signal_time.isoformat()}",
                        "\n".join(
                            [
                                "🔄 4H REBALANCE PROCESSED",
                                f"Signal: {signal_time.isoformat()}",
                                f"Execution bar: {timestamp.isoformat()}",
                                f"Target long: {row.get('long_symbol') or 'CASH'}",
                                f"Target short: {row.get('short_symbol') or 'CASH'}",
                                f"New positions: {', '.join(opened) if opened else 'none'}",
                            ]
                        ),
                    )

            # Match the research engine: rebalance first, then test the full
            # execution bar for a stop and accrue one 5m funding-model slice.
            for symbol in list(positions):
                position = positions[symbol]
                direction = int(position["direction"])
                bar = bars[symbol]
                stop = float(position["stop_loss"])
                hit = float(bar["low"]) <= stop if direction == 1 else float(bar["high"]) >= stop
                if hit:
                    base_exit = (
                        min(float(bar["open"]), stop)
                        if direction == 1
                        else max(float(bar["open"]), stop)
                    )
                    close_position(symbol, base_exit, "stop_loss")
                else:
                    funding_increment = (
                        float(position["notional"])
                        * cfg.funding_bps_per_8h / 10_000 / 96
                    )
                    position["funding_cost"] = float(position["funding_cost"]) + funding_increment
                    connection.execute(
                        "UPDATE positions SET funding_cost=? WHERE symbol=?",
                        (position["funding_cost"], symbol),
                    )

            unrealized = sum(
                (float(bars[symbol]["close"]) - float(position["entry_price"]))
                * float(position["quantity"])
                * int(position["direction"])
                - float(position["entry_fee"])
                - float(position["funding_cost"])
                for symbol, position in positions.items()
            )
            equity = capital + unrealized
            connection.execute(
                """INSERT INTO equity_snapshots(bar_time,equity,capital,position_count)
                   VALUES (?,?,?,?)""",
                (timestamp.isoformat(), equity, capital, len(positions)),
            )
            self.paper.set_state(connection, "capital", capital)
            self.paper.set_state(connection, "last_processed_bar", timestamp.isoformat())

            report_date = timestamp.strftime("%Y-%m-%d")
            previous_report = self.paper.get_state("last_daily_report")
            if timestamp.hour == 0 and previous_report != report_date:
                trade_row = connection.execute(
                    """SELECT COUNT(*) count, COALESCE(SUM(net_pnl),0) pnl
                       FROM trades"""
                ).fetchone()
                self.paper.queue(
                    connection,
                    f"daily-{report_date}",
                    "\n".join(
                        [
                            f"📊 DAILY SUMMARY — {report_date} UTC",
                            f"Equity: {money(equity)}",
                            f"Total return: {(equity / self.starting_capital - 1) * 100:.2f}%",
                            f"Realized P&L: {money(float(trade_row['pnl']))}",
                            f"Closed trades: {int(trade_row['count'])}",
                            f"Open positions: {len(positions)}",
                        ]
                    ),
                )
                self.paper.set_state(connection, "last_daily_report", report_date)


def validate_safety_environment() -> None:
    if os.getenv("AGENT_MODE", "").strip().lower() != "paper":
        raise RuntimeError("AGENT_MODE must equal paper")
    if os.getenv("MARKET_PROVIDER", "").strip().lower() != "okx":
        raise RuntimeError("MARKET_PROVIDER must equal okx")
    if os.getenv("ALLOW_REAL_ORDERS", "false").strip().lower() != "false":
        raise RuntimeError("ALLOW_REAL_ORDERS must equal false")


def main() -> None:
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=os.getenv("V4_ENV_FILE", ".env.v4"))
    parser.add_argument("--market-db", default=None)
    parser.add_argument("--paper-db", default=None)
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env)
    validate_safety_environment()

    market_db = Path(args.market_db or os.environ["MARKET_DB_PATH"])
    paper_db = Path(args.paper_db or os.environ["PAPER_DB_PATH"])
    paper = PaperStore(paper_db)
    notifier = TelegramNotifier(paper)

    try:
        if not args.skip_collect:
            provider = OKXMarketData()
            summary = collect(
                MarketStore(market_db), provider, DEFAULT_SYMBOLS, ("5m", "1h"), 300
            )
            if summary["errors"]:
                raise RuntimeError("; ".join(summary["errors"]))
        result = ForwardPaperRunner(market_db, paper_db).initialize_or_run()
        print(result)
    except Exception as error:
        with paper.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            minute = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
            paper.queue(connection, f"runtime-error-{minute}", f"🚨 RUNTIME ERROR\n{type(error).__name__}: {error}")
        notifier.flush()
        raise
    notifier.flush()


if __name__ == "__main__":
    main()
