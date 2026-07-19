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

## V4-T10 — Rejected; holdout untouched

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

### Development result and decision
- Gross P&L -898.65; net P&L -1,240.90; PF 0.6567.
- Both bull (-648.86 net) and bear (-592.04 net) regimes lost.
- Initial stops lost -3,430.85 net; trailing exits gained +2,076.93 net.
- The -12.5% protection threshold stopped new entries in December 2024; the
  analyzer's last-trade period was shorter than the full processed timeline.
- Gross expectancy was negative, so costs and risk scaling cannot rescue it.
- Rejected without parameter changes and without holdout evaluation.

## V4-T11 — Rejected; recent and final periods untouched

### Independent corpus and universe
- Older data interval: 2021-01-01 00:00 through 2024-07-17 23:55 UTC.
- Recent 2024-07-18 onward data must not be loaded by the T11 runner.
- Eligible after the frozen 95% timestamp-coverage audit:
  `BTC, ETH, SOL, XRP, DOGE, ADA, LINK, LTC, AVAX`.
- BNB is excluded only because older-period 5m/1h coverage was 44.26% after its
  December 2022 listing. No return or P&L information was used for exclusion.

### Frozen signal and selection
- Completed 1h candles; weekly schedule Monday 00:00 UTC.
- Independent horizons: 168h, 720h and 2160h returns.
- Long if at least two horizon returns are positive; short if at least two are
  negative; otherwise no position for that instrument.
- Strength is the absolute mean of each horizon return divided by 720h hourly
  volatility and the square root of its own horizon.
- If more than six instruments signal, select the six greatest absolute
  strengths, with symbol name as deterministic tie-breaker.

### Frozen sizing, execution and costs
- Starting shared capital 10,000 USDT.
- Maximum six positions; risk 0.10% equity per leg; planned total <= 0.60%.
- Maximum notional 10% per leg and maximum gross notional 60%.
- Initial stop 3.0 x completed-1h ATR(14); no trailing stop and no take profit.
- Same-direction positions persist until weekly signal removal/reversal or stop.
- Fill at first common 5m open strictly after signal availability.
- Fee 5 bps/side, slippage 2 bps/side, half-spread 1 bp/side and funding model
  1 bp per 8h.

### Frozen date bounds and walk-forward reporting
- Feature warm-up starts 2021-01-01; full evaluation starts 2021-04-05.
- Full independent evaluation ends 2024-07-17 23:55.
- Reset-capital reporting windows: 2021-04-05--2021-12-31, calendar 2022,
  calendar 2023, and 2024-01-01--2024-07-17.
- Recent period and final holdout flags must both remain false.

### Search budget and acceptance
- Exactly one parameterization; no alternate horizons, weekday, position count,
  risk, stop or selection variants on this corpus.
- Full independent result requires net return > 0, PF >= 1.20, Sharpe >= 1.00,
  max DD <= 20%, long and short net P&L both positive, at least 60% traded
  symbols non-negative, 1.5x-cost PF >= 1.10 and 2.0x-cost net result >= 0.
- At least three of four reset-capital windows must be net positive, including
  the 2024H1 window. Failure rejects T11 without examining recent/final periods.

### Independent result and decision
- Full net +1,198.35 (+11.9835%) over about 3.23 years; CAGR 3.5685%.
- PF 1.2069; Sharpe 0.4394; max DD -12.9827%.
- 1.5x costs +843.98, PF 1.1415; 2.0x costs +489.62, PF 1.0798.
- Three of four windows positive, including 2024H1 (+78.02); 2022 negative.
- Long +1,939.95; short -741.60; only five of nine symbols non-negative.
- Failed Sharpe, short-side and symbol-breadth gates. Rejected without parameter
  changes. Recent period and final holdout remain untouched.

## V4-T12 — Pre-registered cross-venue validation; no result viewed

### Purpose and non-rescue rule
- Validate the frozen V4-T11 implementation on Binance Vision USD-M public
  archives. This is venue portability evidence, not a parameter search and
  cannot retroactively convert rejected T11 into an approved strategy.
- Every strategy, risk, cost, stop, universe and date parameter remains exactly
  equal to V4-T11. `strategy_parameters_changed` must be false.

### Isolated data
- New local SQLite DB: `t12_binance_crossvenue.db`; provider key
  `binance_vision`; instrument IDs use `SYMBOLUSDT`.
- Public monthly USD-M 5m archives only. Derived 1h candles require exactly 12
  completed 5m bars. No Oracle storage and no authenticated Binance API.
- Frozen interval remains 2021-01-01 through 2024-07-17; recent/final flags false.
- All nine symbols require >=95% 5m and 1h timestamp coverage. If fewer than all
  nine pass, T12 is not run; symbols may not be removed based on return or P&L.

### Validation decision
- Report the same full metrics, four reset-capital windows, directions, symbols
  and 1.5x/2.0x cost stress as T11.
- Cross-venue support requires full net positive, PF >=1.20, 2.0x costs positive,
  at least three of four windows positive including 2024H1, and the signs of
  long/short P&L to be reported without post-result removal.
- Regardless of outcome, no recent period or final holdout is evaluated by T12.

### V4-T12 result
- Binance Vision frozen cross-venue validation: +27.9620% (+2,796.20), CAGR
  7.7936%, PF 1.4207, Sharpe 0.5935, max marked DD -20.5175%.
- 1.5x costs +2,394.14 (PF 1.3499); 2.0x costs +1,992.08 (PF 1.2830).
- Three of four windows positive including 2024H1; 2022 negative.
- Long +3,449.90; short -653.70; six of nine symbols non-negative.
- Cross-venue trend portability supported, but rejected T11 is not rescued:
  Sharpe, drawdown and short-side consistency remain insufficient.

## FWD-HR1 — Frozen aggressive forward-only paper observation
- Not a historical trial and not validation. It may not evaluate any past fills.
- Signals are frozen T11 weekly 7d/30d/90d voting on nine OKX swaps.
- Fresh isolated 10,000 USDT paper ledger and separate market database.
- Risk 0.20% per leg, maximum six legs, maximum 15% notional per leg and 75%
  gross notional. Initial stop 3 x ATR; no take profit.
- Permanent paper halt at 25% marked-equity drawdown; no automatic reset.
- Public OKX data only, no private client and `ALLOW_REAL_ORDERS=false` mandatory.
- Performance begins only after runtime initialization; historical data is
  indicator warm-up and no retrospective fill may be imported.
- Telegram messages must identify `HIGH-RISK FORWARD PAPER — NOT VALIDATED`.
