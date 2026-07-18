# V4 Research Trial Registry

## Global research rules
- The final 20% chronological holdout remains untouched unless a candidate first
  passes every development and cost-stress gate.
- Signals use completed candles only. Execution is the first common 5m bar open
  strictly later than signal availability; trades record signal and entry time.
- Fees, slippage, spread and funding assumptions may not be reduced to rescue a
  failed strategy.
- Risk may not be increased to manufacture a larger return.
- A trial specification is frozen before its development result is viewed.

## Rejected single-symbol trials
- V4-T1: close-direction ML — negative gross expectancy
- V4-T2: triple-barrier ML — BTC PF 0.234
- V4-T3: 5m trend alignment — BTC PF 0.483
- V4-T4: 15m Donchian — BTC PF 0.695
- V4-T5: 1h Donchian — net PF 1.002, Sharpe 0.038; not economically viable
- V4-T6: 15m range reversion — BTC PF 0.352

## V4-T7 — Rejected as validated candidate; frozen forward shadow
- Universe: BTC/ETH/SOL/BNB OKX USDT linear swaps
- Rebalance: every 4h
- Corrected causal development result:
  - Return +13.9717%; CAGR 8.6259%
  - Net P&L +1,397.17 USDT on 10,000 USDT
  - PF 1.1135; Sharpe 1.0997; max DD -7.6192%
  - 70% profitable months; both long and short positive
  - ETH negative; 3/4 symbols non-negative
  - 1.5x costs: +3.5006%, PF 1.0271
  - 2.0x costs: -6.9706%, PF 0.9486
- Failed fixed PF and cost-robustness gates. Holdout was not evaluated.
- Deployed on Oracle only as `EXPERIMENTAL SHADOW PAPER — NO REAL ORDERS` with
  a new 10,000 USDT forward ledger. Deployment does not change rejection status.

## V4-T8 — Rejected
- Same ranking, universe, risk and costs as V4-T7; rebalance every 8h.
- Corrected causal development result:
  - Return +5.5154%; net P&L +551.54 USDT
  - PF 1.0518; Sharpe 0.4924; max DD -9.5719%
- Failed normal-cost and stressed-cost gates. Holdout was not evaluated.

## V4-T9 — Pre-registered; no result viewed

### Hypothesis
A broader liquid universe plus slower dual-horizon trend ranking and a rank
retention buffer can reduce boundary churn and concentration while retaining
cross-sectional trend edge after realistic costs.

### Universe eligibility (fixed before return analysis)
Candidate symbols:
`BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LINK, LTC, AVAX` as OKX USDT linear swaps.
A candidate is excluded only if its instrument is not live, has less than 95%
completed-candle coverage over the common research interval, or lacks enough
history for the 30-day lookback. Exclusions and data-quality counts must be
recorded before any strategy result is run. No exclusion may use P&L or returns.
At least 8 eligible symbols are required; otherwise V4-T9 is not run.

### Frozen signal
- Input: completed 1h candles; evaluate continuously but rebalance every 8h.
- 72h return weight: 40%.
- 30-day (720h) return weight: 60%.
- Score: weighted raw return divided by 30-day hourly realized volatility,
  clipped only at `1e-6` to prevent division by zero.
- Long candidate: highest score only when its weighted raw return is positive.
- Short candidate: lowest score only when its weighted raw return is negative.
- Rank buffer: retain the current long while it remains in the top 3 and raw
  return stays positive; retain the current short while it remains in the bottom
  3 and raw return stays negative. Otherwise switch to the current extreme.
- Never long and short the same instrument. Cash on either side is permitted.

### Frozen risk and costs
- Starting shared paper capital: 10,000 USDT.
- Risk per active leg: 0.25% of equity.
- Maximum notional per leg: 25% of equity.
- Stop: 2.0 x completed-1h ATR(14).
- Fee: 5 bps per side.
- Slippage: 2 bps per side.
- Half-spread: 1 bp per side.
- Funding model: 1 bp per 8h, matching V4-T7/V4-T8 comparison assumptions.
- No leverage/risk increase, take-profit, symbol removal or parameter change is
  permitted after the first development result is viewed.

### Split and maximum search budget
- Build one common chronological timeline across all eligible symbols.
- Development: first 80%; final holdout: last 20% untouched.
- V4-T9 permits exactly one parameterization above. No 4h/12h, alternate
  weights, alternate lookbacks or buffer-size variants on this sample.

### Development acceptance gates
All must pass:
- Net PF >= 1.20 and net return > 0.
- Sharpe >= 1.00 and maximum drawdown <= 15%.
- At least 60% profitable calendar months.
- Long and short net P&L both positive.
- At least 70% of traded symbols non-negative after costs.
- No one symbol contributes more than 40% of positive net P&L.
- 1.5x cost PF >= 1.10.
- 2.0x cost net result >= 0.

Only if every gate passes may the frozen strategy be evaluated once on the final
holdout. A target such as +40% or +50% is aspirational and is not an acceptance
substitute; a lower-return robust strategy outranks a larger fragile backtest.
