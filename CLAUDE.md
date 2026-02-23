# CLAUDE.md — BTC/USD Algorithmic Trading System

This file provides context for AI assistants working in this codebase. It covers architecture, conventions, workflows, and critical rules to follow.

---

## Project Overview

A **dual-language algorithmic trading research system** for BTC/USD:

- **Pine Script** (TradingView): Strategy development, visual backtesting, parameter tuning
- **Python**: Rigorous anti-overfit validation (Monte Carlo, Walk-Forward, Parameter Sensitivity)

**Goal**: Build a regime-adaptive BTC/USD strategy that achieves strong backtested performance while passing all three statistical validation gates before live deployment.

**Primary Performance Metric**: `Score = (Profit Factor × Sharpe) / MaxDD` — must be > 0.10

---

## Repository Structure

```
Trading/
├── CLAUDE.md                          # This file
├── STATUS.md                          # Project dashboard — single source of truth
├── requirements.txt                   # Python dependencies
├── dashboard.py                       # Flask web UI for interactive strategy exploration
├── monitor.py                         # System status monitor
├── test_quick.py                      # Quick smoke test
├── run_backtest.py                    # Top-level backtest runner
├── optimizer.py                       # 1D parameter grid search
├── optimizer_4h.py                    # 4H parameter grid search
├── optimizer_short.py                 # Short position optimizer
├── opt_combine.py                     # Ensemble combination logic
├── opt_single.py                      # Single-strategy runner
├── opt_run_all.sh                     # Runs all optimizers sequentially
│
├── pinescript/                        # TradingView Pine Script strategies
│   ├── main_strategy.pine             # PRIMARY: H2 Adaptive Blend (T1+M1+H1)
│   ├── strategy_t1_ema_crossover.pine # Trend-only variant (ablation)
│   ├── strategy_m1_bb_rsi.pine        # Mean-reversion-only variant (ablation)
│   ├── strategy_h1_squeeze.pine       # Squeeze-only variant (ablation)
│   ├── strategy_t2_donchian.pine      # Donchian breakout variant
│   └── strategy_m2_vwma.pine          # VWMA mean-reversion variant
│
├── backtester/                        # MODERN Python backtest engine (use this)
│   ├── engine.py                      # Core execution: position tracking, P&L, equity curve
│   ├── strategies.py                  # Strategy implementations: T1, M1, H1, RSI, MACD, etc.
│   ├── indicators.py                  # Technical indicators: EMA, ATR, RSI, ADX, BB, etc.
│   └── data_loader.py                 # Data ingestion: Yahoo Finance, Binance, CSV
│
├── backtest/                          # LEGACY engine (kept for reference)
│   ├── engine.py
│   ├── strategies.py
│   ├── strategies_1h.py               # 10 aggressive 1H strategies
│   ├── run_backtest.py
│   ├── run_1h_monthly.py              # 1H monthly reporter with signal charts
│   ├── report_generator.py            # HTML dashboard generator
│   ├── data_fetcher.py
│   └── generate_data.py              # Synthetic BTC data fallback
│
├── analysis/                          # Anti-overfit validation toolkit
│   ├── trade_export_parser.py         # Step 1: Normalize TradingView CSV export
│   ├── monte_carlo.py                 # Step 2: Bootstrap resampling (1000 sims)
│   ├── wfo_analysis.py                # Step 3: Walk-forward optimization
│   ├── param_sensitivity.py           # Step 4: Parameter robustness scoring
│   └── requirements.txt
│
├── data/                              # BTC/USD OHLCV CSV files
│   ├── BTC-USD_daily.csv
│   ├── BTC-USD_4h.csv
│   └── BTC-USD_2h.csv
│
├── results/                           # Backtest outputs (HTML reports, CSVs)
├── snapshots/                         # Cached computation snapshots
├── opt_results_2h/                    # 2H optimization result JSONs
└── opt_results_4h/                    # 4H optimization result JSONs
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Strategy Development | Pine Script v5 (TradingView) |
| Backtesting (Python) | pandas, numpy (vectorized, no lookahead bias) |
| Statistical Validation | scipy, custom Monte Carlo + WFO |
| Web Dashboard | Flask + Plotly |
| Data Sources | yfinance (Yahoo Finance), Binance API, local CSVs |
| Languages | Python 3, Pine Script v5, Bash |

**Python dependencies** (`requirements.txt`):
```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
flask>=2.3.0
yfinance>=0.2.31
```

---

## Common Commands

### Running Backtests
```bash
# Quick smoke test (validates engine + single strategy)
python test_quick.py

# Full historical backtest (multi-timeframe)
python run_backtest.py

# 1H monthly backtest with signal charts
python backtest/run_1h_monthly.py

# Single strategy via modern engine
python opt_single.py --strategy T1
```

### Running Optimization
```bash
# Run all optimizers sequentially
bash opt_run_all.sh

# Individual optimizers
python optimizer.py          # 1D grid search
python optimizer_4h.py       # 4H grid search
python optimizer_short.py    # Short position tuning
python opt_combine.py        # Ensemble combination
```

### Anti-Overfit Validation Pipeline (run in order)
```bash
cd analysis
python trade_export_parser.py --input ../results/trades.csv        # Step 1: Parse
python monte_carlo.py --input ../results/trades_parsed.csv          # Step 2: MC
python wfo_analysis.py --input ../results/trades_parsed.csv         # Step 3: WFO
python param_sensitivity.py --input ../results/trades_parsed.csv    # Step 4: Sensitivity
```

### Web Dashboard
```bash
python dashboard.py     # Flask server → http://localhost:5000
```

### Monitoring
```bash
python monitor.py        # System status
python check_status.py   # Detailed status check
```

---

## Architecture: Python Backtest Engine

### `backtester/engine.py` — Core execution model

- **Execution model**: Next-bar-open (no lookahead bias). Signals fire on bar close; fills on next bar open.
- **Commission**: 0.05% per side
- **Slippage**: 0.05% per side
- **Warmup**: 400 bars minimum (indicator stabilization)
- **Position model**: One position at a time (`pyramiding=0`)
- **Shorting**: Allowed at 50% of long size by default

Key config constants in `engine.py`:
```python
warmup_bars = 400
commission_pct = 0.0005   # 0.05% per side
slippage_pct = 0.0005
atr_stop_mult = 2.5        # ATR-based initial stop
atr_trail_mult = 4.0       # ATR-based trailing stop
short_size_mult = 0.5      # Shorts are half-sized
target_annual_vol = 0.30   # Volatility targeting
```

### `backtester/strategies.py` — Strategy implementations

Available strategies and their modules:

| ID | Name | Type | Notes |
|----|------|------|-------|
| `T1` | EMA+ADX | Trend-following | Dual EMA crossover with ADX filter |
| `M1` | BB+RSI | Mean-reversion | Bollinger Band + RSI extremes |
| `H1` | Squeeze | Breakout | BB/KC compression + momentum |
| `RSI` | RSI | Momentum | Overbought/oversold |
| `MACD` | MACD | Trend | MACD crossover |
| `SMA` | SMA | Trend | Simple moving average cross |
| `MOM` | Momentum | Trend | Rate of change |
| `DIP` | Dip Buy | Reversal | Buy-the-dip |
| `DON` | Donchian | Breakout | Channel breakout |

### `backtester/indicators.py` — Indicator library

All indicators are implemented as pure functions operating on pandas Series:
```python
ema(series, period)         # Exponential moving average
sma(series, period)         # Simple moving average
atr(high, low, close, n)    # Average true range
rsi(series, period)         # Relative strength index
adx(high, low, close, n)    # Average directional index
macd(series, fast, slow, sig)  # MACD + signal
bollinger(series, n, mult)  # BB upper/middle/lower
keltner(high, low, close, n, mult)  # Keltner channels
```

---

## Architecture: Pine Script Strategies

### Strict 8-Section Structure (`main_strategy.pine`)

All Pine Script files in `/pinescript/` follow this mandatory layout:

```pinescript
// ═══════════════════════════════════════════════
// SECTION 1: CONFIGURATION
// ═══════════════════════════════════════════════
// All input.* calls, grouped by function using group= parameter

// ═══════════════════════════════════════════════
// SECTION 2: INDICATOR CALCULATIONS
// ═══════════════════════════════════════════════
// EMA, ADX, RSI, ATR, BB, Keltner, VATS indicators

// ═══════════════════════════════════════════════
// SECTION 3: REGIME CLASSIFICATION
// ═══════════════════════════════════════════════
// Trend/Range detector + Volatility regime classifier

// ═══════════════════════════════════════════════
// SECTION 4: POSITION SIZING
// ═══════════════════════════════════════════════
// Volatility-targeted sizing (30% annual vol target)

// ═══════════════════════════════════════════════
// SECTION 5: STRATEGY LOGIC
// ═══════════════════════════════════════════════
// Entry signal generation for T1/M1/H1 modules

// ═══════════════════════════════════════════════
// SECTION 6: EXECUTION
// ═══════════════════════════════════════════════
// strategy.entry() calls — priority: H1 > T1 > M1

// ═══════════════════════════════════════════════
// SECTION 7: RISK MANAGEMENT
// ═══════════════════════════════════════════════
// strategy.close() with ATR stops and trailing stops

// ═══════════════════════════════════════════════
// SECTION 8: VISUALIZATION
// ═══════════════════════════════════════════════
// Plots, regime backgrounds, dashboard table
```

**Never mix sections.** The separation enables ablation testing by commenting out individual sections.

### Pine Script Conventions

**Input parameter grouping** (SECTION 1):
```pinescript
group_regime = "Regime Detection"
group_vol    = "Volatility Regime"
group_entry  = "Entry Parameters"
group_risk   = "Risk Management"
group_ablate = "Ablation Toggles"
group_display= "Display"

i_adx_threshold = input.float(22.0, "ADX Trend Threshold", group=group_regime)
// OVERFIT RISK: ADX threshold is sensitive — test ±20%
```

Mark fragile parameters with `// OVERFIT RISK` or `// DANGEROUS PARAM` comments.

**Ablation toggles** (`i_enable_*` pattern):
```pinescript
i_enable_trend = input.bool(true, "Enable Trend Module (T1)", group=group_ablate)
i_enable_mr    = input.bool(true, "Enable Mean-Reversion Module (M1)", group=group_ablate)
i_enable_squeeze = input.bool(true, "Enable Squeeze Module (H1)", group=group_ablate)
```

**Regime filtering** (use `eff_regime`, not raw signal):
```pinescript
eff_regime = i_enable_regime ? regime_a : 2  // 2 = TRANSITION
can_trade  = bar_index >= i_warmup_bars and (eff_regime != 2 or not i_enable_regime)
```

**Position sizing by module**:
- T1 (Trend): 100% of volatility-targeted size
- M1 (Mean-Reversion): 70% (lower conviction)
- H1 (Squeeze): 100% (high conviction, rare)
- All shorts: 50% of long size

---

## Architecture: Anti-Overfit Validation

The analysis pipeline must be run sequentially after every significant change.

### Step 1: `trade_export_parser.py`
Normalizes raw TradingView CSV exports into the standard format.

**Required output columns**:
| Column | Type | Description |
|--------|------|-------------|
| `trade_id` | int | Sequential trade index |
| `datetime` | datetime | Entry timestamp (chronological order) |
| `profit_pct` | float | Round-trip return in % (e.g., `2.45` = 2.45%) |
| `type` | str | `"long"` or `"short"` |
| `signal` | str | Module that generated the trade (`T1`, `M1`, `H1`) |

**Always parse first** — never run downstream tools on raw TradingView exports.

### Step 2: `monte_carlo.py` — Bootstrap resampling
- Runs 1,000+ simulations by resampling trade returns with replacement
- **Pass criteria**: 95th percentile MaxDD ≤ 25%, 5th percentile return > 0%
- **Sharpe annualization**: `sqrt(73)` (≈ 73 trades/year for BTC daily)

### Step 3: `wfo_analysis.py` — Walk-Forward Optimization
- Rolling IS/OOS window validation
- Default windows: **IS=60 trades, OOS=20 trades, step=20**
- Generates ~12 folds for a 2000-day backtest
- **Pass criteria**: OOS Score ≥ 60% of IS Score across all folds

### Step 4: `param_sensitivity.py` — Parameter robustness
- Tests **±20% and ±40%** variations on all critical parameters
- **Pass criteria**: Score change ≤ 30% at ±20% variation (else "High" fragility)
- Default test grid:
  ```python
  "adx_threshold": {"base": 22, "variations": [13, 18, 22, 26, 31]}
  ```

### All Three Must Pass
```
Monte Carlo: 95th MaxDD ≤ 25% AND 5th return > 0%     ✓/✗
Walk-Forward: OOS Score ≥ 60% of IS                    ✓/✗
Param Sensitivity: No "High" fragility parameters       ✓/✗
```
Only proceed to live testing if all three pass.

---

## Data Split Convention

| Segment | Bar Range | Purpose |
|---------|-----------|---------|
| Warmup | 0–399 | Indicator stabilization (excluded from stats) |
| In-Sample (IS) | 400–1399 | Strategy fitting and optimization |
| Out-of-Sample (OOS) | 1400–1999 | Blind validation |

---

## Performance Metrics Reference

All Python analysis scripts use the **same formulas**:

```python
# Profit Factor
pf = gross_profit / gross_loss

# Sharpe Ratio (annualized for ~73 trades/year on BTC daily)
sharpe = mean(pnl) / std(pnl) * sqrt(73)

# Maximum Drawdown
running_max = equity.cummax()
maxdd = (running_max - equity).max()

# Primary Score (optimization target)
score = (pf * sharpe) / maxdd   # Must be > 0.10
```

---

## Current Best Results

See `STATUS.md` for full details. Summary as of Feb 2026:

| Strategy | Timeframe | Return | Sharpe | MaxDD | Score |
|----------|-----------|--------|--------|-------|-------|
| EMA+ADX v5 | 1D | 313.2% | 3.72 | 11.8% | Best raw return |
| Donchian+MACD v1 | 1D | 132.5% | — | 5.6% | 2.743 (best score) |
| Trend Pullback v3 | 1H | 64.1% | — | — | Best single month |

---

## Development Workflows

### Adding a New Strategy Module (Pine Script)

1. Define entry signals in **SECTION 5** (e.g., `m2_enter_long`, `m2_enter_short`)
2. Add `i_enable_m2` boolean toggle in **SECTION 1** (`group=group_ablate`)
3. Insert `strategy.entry()` calls in **SECTION 6** (respect priority: H1 > T1 > M1 > new)
4. Add module-specific exits in **SECTION 7** (keyed on `active_strategy`)
5. Update position sizing in **SECTION 4** if the module warrants different sizing
6. Update `STATUS.md` with the new module details

### Adding a New Python Strategy

1. Add a new class to `backtester/strategies.py` inheriting from `BaseStrategy`
2. Implement `generate_signals(data)` → returns DataFrame with `signal` column (`1`, `-1`, `0`)
3. Register the strategy ID in `backtester/engine.py`
4. Test with `python test_quick.py`

### Parameter Tuning Workflow (Anti-Overfit Protocol)

1. Identify the parameter in Pine Script (e.g., `i_adx_threshold = 22`)
2. Run baseline backtest → export CSV
3. Test ±20% variations (18, 26) → export CSVs for each
4. Run `param_sensitivity.py` across all three CSVs
5. If fragility is "High" → flag parameter or remove; do **not** over-optimize
6. Document findings in `STATUS.md`

### Validating a Modified Strategy

```bash
# 1. Export trade list from TradingView as CSV to results/trades.csv
# 2. Run full validation pipeline:
cd analysis
python trade_export_parser.py --input ../results/trades.csv
python monte_carlo.py --input ../results/trades_parsed.csv
python wfo_analysis.py --input ../results/trades_parsed.csv
python param_sensitivity.py --input ../results/trades_parsed.csv
# 3. All three tools must pass before proceeding
```

---

## Critical Rules and Pitfalls

### Pine Script Rules
- **No `process_orders_on_close=true`** — All strategies use next-bar-open execution
- **No pyramiding** — `pyramiding=0` always (one position at a time)
- **Never skip warmup** — Minimum 400 bars required (`i_warmup_bars`)
- **Module priority is fixed**: H1 (Squeeze) > T1 (Trend) > M1 (Mean-Reversion)
- **Use `eff_regime`** not raw regime signal for trade gating

### Python Analysis Rules
- **Always run `trade_export_parser.py` first** before any analysis script
- **Check trade count** before analysis — scripts warn if < 20 trades (unreliable stats)
- **Scripts expect `../results/` folder** — create it if missing
- **Sharpe annualizer is `sqrt(73)`** — do not change without updating all scripts

### General Workflow Rules
- **Update `STATUS.md` first** when changing strategy logic or analysis requirements
- **TradingView → CSV is one-way** — analyzed Python results cannot be re-imported to TradingView
- **Parameter sensitivity requires manual re-runs** — change Pine param → re-backtest → re-export → re-analyze (no automation yet)
- **Use `backtester/` not `backtest/`** for new work — `backtest/` is the legacy engine kept for reference

---

## File Conventions

### Output Files
| Pattern | Location | Contents |
|---------|----------|----------|
| `results/trades_*.csv` | `results/` | Trade lists by strategy |
| `results/equity_*.csv` | `results/` | Equity curves |
| `results/backtest_report.html` | `results/` | Main HTML dashboard |
| `opt_results_2h/*.json` | `opt_results_2h/` | 2H optimization results |
| `opt_results_4h/*.json` | `opt_results_4h/` | 4H optimization results |
| `snapshots/*.pkl` | `snapshots/` | Cached computation snapshots |

### Git Ignore Patterns
The following are excluded from version control:
- `__pycache__/`, `*.pyc`, `*.pyo`
- `.env` (API credentials)
- `*.log` (log files)

---

## Project Status

The project tracks its current phase, metrics, and anti-overfit checklist in **`STATUS.md`**. Always check `STATUS.md` before starting new work to understand current priorities.

**Current phase**: Phase 7 — Reporting & Validation

**Anti-overfit checklist** (must complete before live deployment):
- [ ] Walk-Forward Optimization (all folds pass OOS ≥ 60% IS)
- [ ] Monte Carlo (95th pctile MaxDD ≤ 25%, 5th pctile return > 0%)
- [ ] Parameter Sensitivity (no "High" fragility parameters)
- [ ] OOS Backtest (blind holdout period)
- [ ] Cost Stress Test (2× and 3× commission/slippage)

---

## Reference

- **Project status & targets**: `STATUS.md`
- **AI agent guidelines (legacy)**: `.github/copilot-instructions.md`
- **Primary Pine Script**: `pinescript/main_strategy.pine`
- **Python backtest engine**: `backtester/engine.py`
- **Analysis pipeline entry point**: `analysis/trade_export_parser.py`
