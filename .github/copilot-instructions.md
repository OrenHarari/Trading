# BTC/USD Daily Trading System — AI Agent Guide

## Project Overview
**Dual-language trading system**: Pine Script strategies (TradingView) + Python anti-overfit analysis toolkit. Goal: Build a regime-adaptive BTC/USD 1D strategy that passes rigorous statistical validation (Monte Carlo, Walk-Forward, Parameter Sensitivity).

### Key Metrics (see [STATUS.md](../STATUS.md))
- **Primary Score**: `(Profit Factor × Sharpe) / MaxDD` — must be > 0.10
- **Anti-Overfit Target**: OOS Score ≥ 60% of IS, ±20% param changes yield ≤30% score variation
- **Data Split**: 400-bar warmup, 50% IS (bars 400–1399), 30% OOS (bars 1400–1999)

## Architecture & Workflow

### 1. Strategy Development (Pine Script → `/pinescript/`)
```
Pine Script on TradingView → Backtest → Export CSV → Python Analysis
```
**Core Strategy**: `main_strategy.pine` (H2 Adaptive Blend) — switches between:
- **T1 (Trend)**: Dual EMA + ADX trend-following
- **M1 (Mean-Reversion)**: Bollinger Band + RSI reversion
- **H1 (Squeeze)**: BB/KC compression breakout

**Standalone variants**: `strategy_t1_*.pine`, `strategy_m1_*.pine`, etc. — used for ablation testing.

### 2. Python Analysis Toolkit (→ `/analysis/`)
Must be run **in order** after exporting TradingView trade lists:
```bash
cd analysis
python trade_export_parser.py --input ../results/trades.csv  # Step 1: Parse TradingView CSV
python monte_carlo.py --input ../results/trades_parsed.csv   # Step 2: Bootstrap resampling
python wfo_analysis.py --input ../results/trades_parsed.csv  # Step 3: Walk-forward analysis
python param_sensitivity.py --input ../results/trades_parsed.csv  # Step 4: Param grid sweep
```

## Pine Script Conventions

### Strict 8-Section Structure in `main_strategy.pine`
```pinescript
// SECTION 1: CONFIGURATION  — All input.* calls grouped by function
// SECTION 2: INDICATOR CALCULATIONS  — EMA, ADX, RSI, ATR, BB, VATS
// SECTION 3: REGIME CLASSIFICATION  — Two classifiers: Trend/Range + Vol Regime
// SECTION 4: POSITION SIZING  — Volatility-targeted sizing (30% annual vol)
// SECTION 5: STRATEGY LOGIC  — Entry signal generation for T1/M1/H1
// SECTION 6: EXECUTION  — strategy.entry() calls (priority: H1 > T1 > M1)
// SECTION 7: RISK MANAGEMENT  — strategy.close() with module-specific exits
// SECTION 8: VISUALIZATION  — Plots, regime backgrounds, dashboard table
```
**Never mix sections** — separates backtest logic from display for easier ablation.

### Input Parameter Groups (SECTION 1)
Group all `input.*` calls by category using `group=group_*`:
```pinescript
group_regime = "Regime Detection"
i_adx_threshold = input.float(22.0, "ADX Trend Threshold", group=group_regime)
// OVERFIT RISK: ADX threshold is sensitive — test ±20%
```
**Comment fragile parameters** with `// OVERFIT RISK` or `// DANGEROUS PARAM` inline.

### Ablation Toggles (`i_enable_*`)
Each strategy module has a boolean toggle:
```pinescript
i_enable_trend = input.bool(true, "Enable Trend Module (T1)", group=group_ablation)
i_enable_mr = input.bool(true, "Enable Mean-Reversion Module (M1)", group=group_ablation)
```
**Purpose**: Measure marginal contribution of each component (disable one → re-test → compare).

### Regime Filtering Pattern
Use `eff_regime` (effective regime after persistence filter) not raw regime signal:
```pinescript
eff_regime = i_enable_regime ? regime_a : 2  // 2 = TRANSITION (always trade if filter off)
can_trade = bar_index >= i_warmup_bars and (eff_regime != 2 or not i_enable_regime)
```

### Strategy-Specific Position Sizing
```pinescript
// T1 Trend: 100% of volatility-targeted size
// M1 Mean-Reversion: 70% (smaller, less certain)
// H1 Squeeze: 100% (rare, high-conviction)
// All shorts: 50% of long size
```

## Python Toolkit Conventions

### Trade CSV Format (after `trade_export_parser.py`)
Required columns: `trade_id`, `datetime`, `profit_pct`, `type`, `signal`
- **profit_pct**: Round-trip return in percentage (e.g., `2.45` = 2.45%)
- **chronological order**: Scripts assume trades are sorted by `datetime`

### Common Metrics Calculation Pattern
All Python scripts use identical metric formulas (see `monte_carlo.py` lines 33–56):
```python
# Profit Factor = gross_profit / gross_loss
# Sharpe = mean(pnl) / std(pnl) * sqrt(73)  # 73 ≈ trades/year for BTC daily
# MaxDD = max(running_max - equity)
# Score = (PF × Sharpe) / MaxDD  # Primary optimization target
```

### Walk-Forward Window Sizes
Default: **IS=60 trades, OOS=20 trades, step=20** (see `wfo_analysis.py` lines 51–55)
- Generates ~12 folds for 2000-day backtest (~5 years BTC daily)
- **Pass criteria**: OOS Score ≥ 60% of IS Score across all folds

### Parameter Sensitivity Variations
Test **±20% and ±40%** for all critical params (see `param_sensitivity.py` lines 25–35):
```python
"adx_threshold": {"base": 22, "variations": [13, 18, 22, 26, 31]},  # -40%, -20%, 0%, +20%, +40%
```
**Fragility rating**: High if score changes >30% at ±20% param variation.

## Critical Files

### Configuration & Tracking
- **[STATUS.md](../STATUS.md)**: Single source of truth — metrics targets, phase checklist, file inventory, backtest config
- **Update this first** when changing strategy logic or analysis requirements

### Pine Script Entry Points
- **[main_strategy.pine](../pinescript/main_strategy.pine)**: H2 Adaptive Blend (primary)
- Standalone variants for ablation: `strategy_t1_ema_crossover.pine`, `strategy_m1_bb_rsi.pine`, etc.

### Python Analysis Chain
1. **[trade_export_parser.py](../analysis/trade_export_parser.py)**: Normalizes TradingView CSV → `trades_parsed.csv`
2. **[monte_carlo.py](../analysis/monte_carlo.py)**: Bootstrap resampling (1000 sims)
3. **[wfo_analysis.py](../analysis/wfo_analysis.py)**: Rolling IS/OOS windows
4. **[param_sensitivity.py](../analysis/param_sensitivity.py)**: Grid search analysis framework

## Common Pitfalls

### Pine Script
- **Don't add `process_orders_on_close=true`** — all strategies use next-bar-open execution
- **Don't pyramid** — `pyramiding=0` enforced (one position at a time)
- **Don't skip warmup** — minimum 400 bars to stabilize indicators (configurable via `i_warmup_bars`)
- **Module priority matters**: Squeeze (H1) > Trend (T1) > Mean-Reversion (M1) in SECTION 6

### Python Analysis
- **Always parse first**: Never run `monte_carlo.py` on raw TradingView export — use `trade_export_parser.py` first
- **Check trade count**: Scripts warn if <20 trades (unreliable statistics)
- **Directory structure**: Scripts expect `../results/` folder for CSVs (create if missing)

### Workflow
- **TradingView → CSV export → Python** is one-way — can't re-import analyzed results to TradingView
- **Parameter sensitivity requires manual re-runs** — change Pine Script param → re-backtest → re-export CSV → re-analyze (no automation yet)

## Development Patterns

### Adding a New Strategy Module
1. Define entry signals in SECTION 5 (e.g., `m2_enter_long`, `m2_enter_short`)
2. Add `i_enable_m2` toggle in SECTION 1 (group=group_ablation)
3. Insert `strategy.entry()` calls in SECTION 6 (respect priority order)
4. Add module-specific exits in SECTION 7 (track `active_strategy`)
5. Update position sizing logic if needed (SECTION 4)

### Parameter Tuning (Anti-Overfit Protocol)
1. Identify parameter in Pine Script (e.g., `i_adx_threshold`)
2. Test ±20% variations (e.g., 18, 22, 26)
3. Export each trade list separately
4. Run `param_sensitivity.py` to compute fragility rating
5. If "High" fragility → flag for ensemble or remove parameter

### Validating a Strategy
After backtest, **run all three Python tools** and check:
```bash
# 1. Monte Carlo: 95th pctile MaxDD ≤ 25%, 5th pctile return > 0%
python monte_carlo.py --input trades_parsed.csv

# 2. Walk-Forward: OOS Score ≥ 60% of IS
python wfo_analysis.py --input trades_parsed.csv

# 3. Parameter Sensitivity: No "High" fragility params
python param_sensitivity.py --input trades_parsed.csv
```
**All three must pass** to proceed to live testing.

---

**Reference**: See [STATUS.md](../STATUS.md) for current phase, target metrics, and anti-overfit checklist. Update STATUS.md when completing analysis runs.
