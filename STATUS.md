# BTC/USD Trading System — Project Status

> **Last Updated**: 2026-02-14
> **Branch**: `claude/btc-trading-system-UJqNd`
> **Phase**: 7 — Optimization Complete, Validation Next

---

## Current State Summary

Python-based backtester with 11 strategy modules, optimized on 4H and 2H timeframes.
Daily timeframe baseline results exist but underperform. **4H is the best timeframe.**

### Top Strategies (4H, In-Sample, Best Params)

| Rank | Strategy | PF | Sharpe | MaxDD% | Trades | WinRate% | Notes |
|------|----------|----|--------|--------|--------|----------|-------|
| 1 | STR (SuperTrend) | 10.67 | 2.57 | 6.3 | 5 | 80 | ⚠️ Too few trades |
| 2 | DIP (Dip Buy) | 2.59 | 1.90 | 9.6 | 19 | 52.6 | ✅ Good balance |
| 3 | BKD (Breakdown) | 5.25 | 1.52 | 10.5 | 8 | 50 | ⚠️ Few trades |
| 4 | M1 (BB+RSI MR) | 2.23 | 1.45 | 4.2 | 23 | 52.2 | ✅ Best risk-adj |
| 5 | RSI | 1.76 | 1.31 | 9.0 | 35 | 40 | ✅ Most trades |
| 6 | MOM (Momentum) | 2.84 | 1.12 | 18.1 | 8 | 50 | ⚠️ Few trades |
| 7 | DON (Donchian) | 2.01 | 0.92 | 12.6 | 13 | 53.8 | OK |
| 8 | MP (Multi-Period) | 1.83 | 0.92 | 12.7 | 17 | 41.2 | OK |
| 9 | SMA | 2.23 | 0.86 | 10.0 | 13 | 61.5 | OK |
| 10 | T1 (EMA+ADX) | 1.96 | 0.78 | 5.1 | 7 | 42.9 | ⚠️ Few trades |
| 11 | MACD | 0.98 | -0.09 | 32.8 | 126 | 30.2 | ❌ Losing |

### Daily Timeframe Baseline (Default Params)

| Strategy | PF | Sharpe | MaxDD% | Trades | Score |
|----------|----|--------|--------|--------|-------|
| T1 | 1.33 | 0.88 | 15.5 | 28 | 0.076 |
| M1 | 0.68 | -1.37 | 11.1 | 50 | -0.085 |
| H1 | 0.78 | -0.94 | 7.7 | 13 | -0.095 |
| H2 (Blend) | 0.79 | -0.85 | 10.2 | 28 | -0.067 |
| Buy & Hold | — | — | 76.7 | 1 | — |

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Market hypotheses | ✅ Done |
| 2 | Data collection & backtest spec | ✅ Done |
| 3 | Regime detection design | ✅ Done |
| 4 | Strategy candidates (11 strategies) | ✅ Done |
| 5 | Python backtester engine | ✅ Done |
| 6 | Optimizer (grid search, IS/OOS) | ✅ Done |
| 7 | Multi-timeframe optimization (4H, 2H) | ✅ Done |
| 8 | **Strategy combination / ensemble** | 🔲 Next |
| 9 | Walk-forward validation | 🔲 Pending |
| 10 | Monte Carlo / param sensitivity | 🔲 Pending |
| 11 | Final validation & go-live decision | 🔲 Pending |

---

## Architecture

```
data/                  → BTC-USD price CSVs (daily, 4h, 2h)
backtester/
  engine.py            → Core backtest engine (handles entries, exits, sizing)
  strategies.py        → 11 strategy modules (T1, M1, H1, DIP, DON, etc.)
  indicators.py        → Technical indicator calculations
  data_loader.py       → CSV loading & preprocessing
optimizer.py           → Grid search optimizer (daily)
optimizer_4h.py        → 4H timeframe optimizer
opt_single.py          → Single-strategy optimizer
opt_combine.py         → Multi-strategy combination optimizer
opt_results_4h/        → Optimization results per strategy (4H)
opt_results_2h/        → Optimization results per strategy (2H)
results/               → Backtest output (equity curves, trades)
run_backtest.py        → Run strategies with specific params
dashboard.py           → Web UI for viewing results
analysis/              → Anti-overfit toolkit (Monte Carlo, WFO, sensitivity)
pinescript/            → TradingView Pine Script versions (reference only)
```

---

## Key Decisions Made

1. **Timeframe**: 4H > Daily for most strategies (more trades, better Sharpe)
2. **Best candidates for ensemble**: DIP, M1, RSI (good trade count + metrics)
3. **Dropped**: MACD (negative on all timeframes)
4. **Risky**: STR, BKD, MOM — great metrics but too few trades (<10)

---

## Anti-Overfit Checklist

| Test | Pass Criteria | Status |
|------|--------------|--------|
| Walk-Forward (12 folds) | OOS Score ≥ 60% of IS | 🔲 Pending |
| Parameter Sensitivity | Score ±30% at ±20% param change | 🔲 Pending |
| Monte Carlo (1000 sims) | 95th pctile DD ≤ 25%, 5th pctile ret > 0% | 🔲 Pending |
| OOS Backtest | Run best params on OOS period | 🔲 Pending |
| Cost Stress (2×) | PF > 1.0 at 0.40% RT cost | 🔲 Pending |

---

## Backtest Configuration

| Parameter | Value |
|-----------|-------|
| Instrument | BTC/USD |
| Timeframes | Daily, 4H, 2H |
| Initial Capital | $100,000 |
| Commission | 0.05% per side |
| Slippage | Built into execution |
| Position Sizing | Volatility-targeted |
| Data Source | CSV files in `data/` |

---

## Next Steps

1. **Combine top strategies** (DIP + M1 + RSI) into ensemble on 4H
2. **Run OOS validation** — test best params on held-out data
3. **Walk-forward analysis** — rolling IS/OOS windows
4. **Parameter sensitivity** — check fragility of optimal params
5. **Monte Carlo** — bootstrap confidence intervals
6. Go/No-Go decision based on validation results

---

## How To Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest with default params
python run_backtest.py

# Run optimizer for a single strategy on 4H
python opt_single.py --strategy DIP --timeframe 4h

# Run all optimizations
bash opt_run_all.sh

# View dashboard
python dashboard.py

# Check project status
cat STATUS.md
```

---

*This file is the single source of truth. Update it after every significant change.*
