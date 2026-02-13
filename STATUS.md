# BTC/USD 1D Trading System — Project Status Dashboard

> **Last Updated**: 2026-02-13
> **Branch**: `claude/btc-trading-system-UJqNd`
> **Phase**: Implementation (Phase 6)

---

## Quick Reference Numbers

| Metric | Target | Current Status |
|--------|--------|---------------|
| Primary Score `(PF × Sharpe) / MaxDD` | > 0.10 | Pending backtest |
| Max Drawdown | ≤ 25% | Pending backtest |
| Minimum Trades | ≥ 120 | Pending backtest |
| Profit Factor | > 1.3 | Pending backtest |
| Sharpe Ratio | > 0.8 | Pending backtest |
| CAGR | Positive | Pending backtest |
| Win Rate | > 45% (trend) / > 55% (MR) | Pending backtest |
| Avg Trade | > 0.3% | Pending backtest |
| Max Consecutive Losses | < 10 | Pending backtest |
| Exposure | 30%–70% | Pending backtest |

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Operating Rules | Done |
| Phase 1 | Market Structure & Hypotheses | Done |
| Phase 2 | Data & Backtest Spec | Done |
| Phase 3 | Regime Detection Design | Done |
| Phase 4 | Strategy Candidates | Done |
| Phase 5 | Anti-Overfit Toolkit Design | Done |
| Phase 6 | Implementation | **In Progress** |
| Phase 7 | Reporting & Validation | Pending backtest data |

---

## Backtest Configuration

| Parameter | Value |
|-----------|-------|
| Instrument | BTCUSD 1D (BITSTAMP or INDEX on TradingView) |
| Initial Capital | $100,000 |
| Direction | Long + Short (shorts at 0.5× size) |
| Position Sizing | Volatility-targeted (30% annual vol target) |
| Commission | 0.05% per side |
| Slippage | 5 ticks per side |
| Total Round-Trip Cost | ~0.20% |
| Execution | Next-bar open (`process_orders_on_close=false`) |
| Warm-up | 400 bars (no trades) |
| IS Window | 50% of data (bars 400–1399) |
| OOS Window | 30% of data (bars 1400–1999) |

---

## Strategy Rankings

| Rank | Strategy | File | Type | Expected Sharpe | Expected PF |
|------|----------|------|------|----------------|-------------|
| 1 | **H2 Adaptive Blend** | `main_strategy.pine` | Hybrid | 1.2–1.6 | 1.5–2.0 |
| 2 | T1 EMA + ADX | `strategy_t1_ema_crossover.pine` | Trend | 1.0–1.4 | 1.4–1.8 |
| 3 | H1 Squeeze Breakout | `strategy_h1_squeeze.pine` | Hybrid | 0.9–1.3 | 1.3–1.7 |
| 4 | T2 Donchian | `strategy_t2_donchian.pine` | Trend | 0.8–1.2 | 1.3–1.6 |
| 5 | M1 BB + RSI | `strategy_m1_bb_rsi.pine` | Mean-Rev | 0.7–1.0 | 1.2–1.5 |
| 6 | M2 VWMA | `strategy_m2_vwma.pine` | Mean-Rev | 0.6–0.9 | 1.1–1.4 |

---

## Anti-Overfit Checklist

| Test | Pass Criteria | Status |
|------|--------------|--------|
| Walk-Forward (12 folds) | OOS Score ≥ 60% of IS | Pending |
| Parameter Sensitivity | Score within ±30% at ±20% param change | Pending |
| Monte Carlo (1000 resamples) | 95th pctile DD ≤ 25%, 5th pctile return > 0% | Pending |
| Regime Stress | No regime PF < 0.9 | Pending |
| Cost Stress (2×) | PF > 1.0 at 0.40% RT cost | Pending |

---

## File Inventory

### Pine Script (`/pinescript/`)
| File | Description | Status |
|------|-------------|--------|
| `main_strategy.pine` | H2 Adaptive Blend — primary strategy | Building |
| `strategy_t1_ema_crossover.pine` | Dual EMA + ADX trend-follower | Building |
| `strategy_m1_bb_rsi.pine` | Bollinger Band + RSI mean-reversion | Building |
| `strategy_h1_squeeze.pine` | BB/KC Squeeze Breakout | Building |
| `strategy_t2_donchian.pine` | Donchian Channel Breakout | Building |
| `strategy_m2_vwma.pine` | VWMA Reversion + Stochastic | Building |

### Analysis (`/analysis/`)
| File | Description | Status |
|------|-------------|--------|
| `monte_carlo.py` | Bootstrap resampling of trade P&L | Building |
| `wfo_analysis.py` | Walk-forward fold analysis | Building |
| `param_sensitivity.py` | Parameter grid sweep | Building |
| `trade_export_parser.py` | Parse TradingView CSV exports | Building |
| `requirements.txt` | Python dependencies | Building |

---

## How to Use

### TradingView Setup
1. Open TradingView → Pine Editor
2. Copy contents of desired `.pine` file
3. Apply to **BTCUSD 1D** chart (BITSTAMP:BTCUSD recommended)
4. Check Strategy Tester tab for results
5. Export trade list to CSV for Python analysis

### Running Analysis (after CSV export)
```bash
cd /home/user/Trading/analysis
pip install -r requirements.txt
python trade_export_parser.py --input ../results/trades.csv
python monte_carlo.py --input ../results/trades_parsed.csv
python wfo_analysis.py --input ../results/trades_parsed.csv
python param_sensitivity.py --input ../results/trades_parsed.csv
```

---

## Hypotheses Under Test

| ID | Hypothesis | Status |
|----|-----------|--------|
| H1 | BTC trends persist (ADX>25 + EMA → continues 20+ bars) | Testing via T1/H2 |
| H2 | Mean reversion in low-ADX regimes (BB touch → SMA revert) | Testing via M1/H2 |
| H3 | Vol compression precedes expansion (squeeze → 5%+ move) | Testing via H1 |
| H4 | Halving cycle long bias (post-halving long-only outperforms) | Structural in sizing |
| H5 | Day-of-week seasonality | Not yet implemented |

---

## Next Steps (What To Do After Backtest)

1. Load `main_strategy.pine` on BTCUSD 1D → record all metrics in table above
2. Load each standalone strategy → compare against H2
3. Export trades to CSV → run Python analysis suite
4. Fill in anti-overfit checklist results
5. Make GO/NO-GO decision based on final scorecard
6. If GO: consider multi-timeframe confirmation, walk-forward re-optimization
7. If NO-GO: identify weakest component and redesign
