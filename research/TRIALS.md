# V4 Research Trial Registry

## V4-T1 — Rejected
- Close-to-close ML target; negative gross expectancy on all symbols; holdout untouched.

## V4-T2 — Rejected
- Triple-barrier ML; BTC PF 0.234, drawdown 57.7%; holdout untouched.

## V4-T3 — Rejected
- Deterministic 5m trend alignment; BTC PF 0.483, drawdown 91.4%; holdout untouched.

## V4-T4 — Rejected
- Deterministic completed-15m 20-bar Donchian; BTC PF 0.695, drawdown 37.5%; holdout untouched.

## V4-T5 — Registered, not yet evaluated
- Plugin: `donchian_1h_v1`
- Completed-1h 55-bar breakout, volume >= average, ADX >= 20, EMA21 slope aligned
- 2 ATR stop, 6 ATR target, 576 x 5m maximum (48h)
- Holdout untouched

## V4-T6 — Registered, not yet evaluated
- Plugin: `range_reversion_15m_v1`
- Completed-15m Bollinger/RSI extreme only when completed-1h and 15m ADX indicate range
- 2 ATR stop, 2 ATR target, 48 x 5m maximum (4h)
- Holdout untouched
