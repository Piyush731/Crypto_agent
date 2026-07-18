# V4 Research Trial Registry

## V4-T1 — Rejected

- Target: 4-hour close-to-close return, +/-0.30% neutral band
- Model trials: dummy, logistic, HistGradientBoosting
- Execution: next 5m open, 2 ATR stop, 3 ATR target, 48-bar maximum
- Confidence threshold: 0.45
- Result: rejected on development OOS data
- Reason: negative gross expectancy and PF < 1 on all four symbols; excessive turnover
- Final holdout: untouched

## V4-T2 — Registered, not yet evaluated

- Strategy plugin: `triple_barrier_5m` v1
- Labels: LONG/SHORT only when the corresponding 3 ATR target wins before 2 ATR stop within 48 bars; HOLD otherwise/ambiguous
- Model trials: dummy, logistic, HistGradientBoosting
- Confidence threshold: 0.50
- Execution assumptions: inherited from `ExecutionPolicy`
- Final holdout: must remain untouched until development gates pass
