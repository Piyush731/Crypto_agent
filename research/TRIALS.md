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

## V4-T9 — Rejected; holdout untouched

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

### Development result and decision
- Return +7.9723%; net P&L +797.23; CAGR 5.1811%.
- PF 1.1089; Sharpe 0.5518; max DD -10.9748%.
- 47.37% profitable months; short side -196.39.
- Only 3/10 symbols positive; DOGE supplied about 58% of positive-symbol P&L.
- 1.5x cost PF 1.0504; 2.0x costs -32.55, PF 0.9959.
- Failed PF, Sharpe, monthly consistency, side, breadth, concentration and cost
  gates. Rejected. No parameter rescue and no holdout evaluation.

## V4-T10 — Pre-registered; no result viewed

### Hypothesis
A broad-market regime gate can avoid structurally wrong-direction positions,
while two-leg directional diversification, slow ranking retention, causal ATR
trailing stops and portfolio drawdown throttling can improve net robustness. This
is a new strategy family; V4-T9 parameters are not modified or rescued.

### Frozen universe and timing
- Exactly: `BTC, ETH, SOL, BNB, XRP, DOGE, ADA, LINK, LTC, AVAX` OKX swaps.
- All ten passed the pre-registered data eligibility audit.
- Completed 1h inputs; schedule at 00:00 and 12:00 UTC (12h rebalance).
- Fill at first common 5m open strictly after signal availability.
- Same first-80% development and untouched final-20% holdout split.

### Frozen regime layer
- BTC 30-day EMA uses 720 completed hourly closes.
- Breadth is the fraction of all ten symbols with positive 720h return.
- Bull: BTC above its 30-day EMA and breadth >= 60%.
- Bear: BTC below its 30-day EMA and breadth <= 40%.
- Mixed: all other states; target is cash with no active directional positions.

### Frozen selection and retention layer
- Score = `(50% x 168h return + 50% x 720h return) / 720h hourly volatility`.
- Bull target: two highest-score symbols whose weighted raw return is positive.
- Bear target: two lowest-score symbols whose weighted raw return is negative.
- Retain incumbent bull legs while in top 4 with positive raw return.
- Retain incumbent bear legs while in bottom 4 with negative raw return.
- Fill vacant target slots from the current score extremes. No symbol exclusions.

### Frozen execution, stops and costs
- Starting shared capital: 10,000 USDT.
- Risk per leg: 0.50%; maximum two legs; planned total risk <= 1.00%.
- Maximum notional per leg: 30%; maximum gross notional: 60%.
- Initial stop: 2.0 x completed-1h ATR(14).
- Arm trailing after favorable movement of 2.0 x entry ATR.
- Once armed, trail 2.0 x entry ATR from the best favorable price.
- A trailing update becomes active on the next 5m bar, avoiding unknown intrabar
  high/low ordering.
- Fee 5 bps/side; slippage 2 bps/side; half-spread 1 bp/side.
- Funding model 1 bp per 8h, unchanged from prior comparisons.

### Frozen portfolio protection
- Drawdown is measured from peak marked equity at a rebalance.
- Above -7.5%: normal risk for new positions.
- At or below -7.5%: half risk for new positions.
- At or below -12.5%: no new positions; existing stops/regime exits remain live.
- Normal or half risk resumes automatically only when marked drawdown moves back
  above the corresponding fixed threshold.

### Search budget and acceptance
- Exactly one V4-T10 parameterization. No alternate EMA, breadth, rebalance,
  retention, stop, trailing, risk or regime variants on this sample.
- Required: net return > 0, PF >= 1.20, Sharpe >= 1.00, max DD <= 20%, at least
  60% profitable months, bull and bear trade P&L both non-negative, at least 60%
  traded symbols non-negative, no symbol above 35% of positive-symbol P&L,
  1.5x-cost PF >= 1.10, and 2.0x-cost net result >= 0.
- Only an all-gates pass permits one final holdout evaluation.
