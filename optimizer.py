#!/usr/bin/env python3
"""
Automated Strategy Optimizer for BTC/USD Daily Trading.

Uses REAL Yahoo Finance OHLC data (no fake/synthesized data).
Sweeps strategy parameters + engine config to find profitable combinations.
Validates with IS/OOS split to prevent overfitting.

Usage:
    python optimizer.py
"""

import sys
import time
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.data_loader import load_data
from backtester.engine import BacktestConfig
from backtester.strategies import STRATEGY_REGISTRY


# ── Configuration ─────────────────────────────────────

IS_END = "2022-01-01"      # In-Sample: 2014-09-17 → 2022-01-01
OOS_START = "2022-01-01"   # Out-of-Sample: 2022-01-01 → 2026-02-13

MIN_TRADES = 8             # Minimum trades to consider result valid
IS_RETURN_THRESHOLD = 15.0 # Minimum IS return % to qualify for OOS test
TOP_N_FOR_OOS = 80         # Top N IS results to validate on OOS

# Strategies to optimize (skip BH=benchmark, H2=composite, H1=too slow due to linreg)
OPTIMIZE_STRATEGIES = ["T1", "M1", "RSI", "MACD", "SMA", "MOM", "DIP", "DON"]


# ── Parameter Grids ──────────────────────────────────

STRATEGY_GRIDS = {
    "T1": {
        "ema_fast":      [10, 15, 21, 30],
        "ema_slow":      [40, 55, 80, 120],
        "adx_threshold": [15, 20, 25, 30],
    },
    "M1": {
        "bb_len":  [15, 20, 30],
        "bb_mult": [1.5, 2.0, 2.5],
        "rsi_os":  [20, 30, 40],
        "rsi_ob":  [60, 70, 80],
    },
    "H1": {
        "bb_len":       [15, 20, 30],
        "bb_mult":      [1.5, 2.0, 2.5],
        "kc_mult":      [1.0, 1.5, 2.0],
        "squeeze_bars": [3, 6, 10],
    },
    "RSI": {
        "rsi_len": [7, 14, 21],
        "rsi_os":  [20, 25, 30, 35],
        "rsi_ob":  [65, 70, 75, 80],
    },
    "MACD": {
        "macd_fast":   [8, 12, 16],
        "macd_slow":   [20, 26, 35],
        "macd_signal": [5, 9, 12],
    },
    "SMA": {
        "sma_fast": [20, 50, 100],
        "sma_slow": [100, 150, 200, 300],
    },
    "MOM": {
        "roc_len":        [10, 20, 30, 50],
        "roc_threshold":  [5, 10, 15, 20],
        "roc_exit":       [-5, 0, 5],
        "use_sma_filter": [0, 1],
    },
    "DIP": {
        "lookback":     [30, 60, 90],
        "dip_pct":      [10, 15, 20, 25, 30],
        "recovery_pct": [5, 10, 15, 20],
        "max_hold":     [60, 90],
    },
    "DON": {
        "entry_len": [20, 30, 55, 80],
        "exit_len":  [10, 20, 30],
    },
}

# Engine config variations (applied to all strategies)
CONFIG_GRID = {
    "target_annual_vol": [30, 50, 80],
    "atr_stop_mult":     [1.5, 2.0, 3.0],
    "atr_trail_mult":    [2.0, 3.0, 4.0],
    "short_size_mult":   [0.0, 0.25, 0.5],  # 0.0 = long-only
    "warmup_bars":       [200, 400],
    "max_bars_trend":    [40, 60, 90],
}

# For speed, use a reduced config grid per strategy type
CONFIG_PRESETS = [
    # Long-only, aggressive sizing, wide stops
    {"target_annual_vol": 80, "atr_stop_mult": 3.0, "atr_trail_mult": 4.0,
     "short_size_mult": 0.0, "warmup_bars": 200, "max_bars_trend": 90},
    # Long-only, medium sizing
    {"target_annual_vol": 50, "atr_stop_mult": 2.0, "atr_trail_mult": 3.0,
     "short_size_mult": 0.0, "warmup_bars": 200, "max_bars_trend": 60},
    # Long-only, aggressive, tight trail
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 2.5,
     "short_size_mult": 0.0, "warmup_bars": 200, "max_bars_trend": 60},
    # With shorts, aggressive
    {"target_annual_vol": 80, "atr_stop_mult": 3.0, "atr_trail_mult": 4.0,
     "short_size_mult": 0.5, "warmup_bars": 200, "max_bars_trend": 90},
    # Long-only, very wide trail (let winners run)
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 5.0,
     "short_size_mult": 0.0, "warmup_bars": 200, "max_bars_trend": 120},
]


def make_config(preset: dict) -> BacktestConfig:
    """Create a BacktestConfig from a preset dict."""
    cfg = BacktestConfig()
    for k, v in preset.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # Also propagate relevant params to MR/Squeeze
    cfg.mr_stop_mult = min(cfg.atr_stop_mult, 1.5)
    cfg.mr_time_exit = max(10, cfg.max_bars_trend // 3)
    cfg.squeeze_stop_mult = cfg.atr_stop_mult
    cfg.squeeze_trail_mult = cfg.atr_trail_mult
    cfg.squeeze_max_bars = min(cfg.max_bars_trend, 30)
    return cfg


def run_single(strategy_key, strategy_params, config, df):
    """Run a single backtest. Returns metrics dict or None."""
    try:
        strat_cls = STRATEGY_REGISTRY[strategy_key]
        strat = strat_cls(config, **strategy_params)
        result = strat.run(df)
        if "error" in result:
            return None
        if result["total_trades"] < MIN_TRADES:
            return None
        return result
    except Exception:
        return None


def generate_param_combos(grid: dict) -> list:
    """Generate all combinations from a parameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_optimization():
    """Main optimization loop."""
    print("=" * 70)
    print("  BTC/USD Strategy Optimizer")
    print("  Real OHLC Data from Yahoo Finance (no fake data)")
    print("=" * 70)

    # ── Load Data ──
    print("\n[1/5] Loading real OHLC data from Yahoo Finance...")
    df_full = load_data(source="BTC-USD")
    print(f"  Total bars: {len(df_full)} ({df_full['date'].iloc[0].date()} → {df_full['date'].iloc[-1].date()})")

    # ── Split IS/OOS ──
    print(f"\n[2/5] Splitting data: IS (→ {IS_END}) / OOS ({OOS_START} →)")
    df_is = df_full[df_full["date"] < IS_END].copy().reset_index(drop=True)
    df_oos = df_full[df_full["date"] >= OOS_START].copy().reset_index(drop=True)
    print(f"  IS:  {len(df_is)} bars ({df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()})")
    print(f"  OOS: {len(df_oos)} bars ({df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()})")

    # ── Count Total Combinations ──
    total_combos = 0
    for skey in OPTIMIZE_STRATEGIES:
        if skey in STRATEGY_GRIDS:
            param_combos = generate_param_combos(STRATEGY_GRIDS[skey])
            total_combos += len(param_combos) * len(CONFIG_PRESETS)

    print(f"\n[3/5] Running {total_combos} strategy+config combinations on IS data...")
    print(f"  Strategies: {', '.join(OPTIMIZE_STRATEGIES)}")
    print(f"  Config presets: {len(CONFIG_PRESETS)}")

    # ── Phase 1: IS Sweep ──
    all_results = []
    counter = 0
    t0 = time.time()
    best_so_far = -999

    for skey in OPTIMIZE_STRATEGIES:
        if skey not in STRATEGY_GRIDS:
            continue

        param_combos = generate_param_combos(STRATEGY_GRIDS[skey])

        for params in param_combos:
            for ci, preset in enumerate(CONFIG_PRESETS):
                counter += 1
                config = make_config(preset)
                result = run_single(skey, params, config, df_is)

                if result is not None:
                    ret = result["total_return_pct"]
                    entry = {
                        "strategy": skey,
                        "params": params.copy(),
                        "config_idx": ci,
                        "config": preset.copy(),
                        "is_return": ret,
                        "is_pf": result["profit_factor"],
                        "is_sharpe": result["sharpe_ratio"],
                        "is_maxdd": result["max_drawdown_pct"],
                        "is_trades": result["total_trades"],
                        "is_winrate": result["win_rate"],
                        "is_score": result["primary_score"],
                        "is_cagr": result["cagr_pct"],
                    }
                    all_results.append(entry)

                    if ret > best_so_far:
                        best_so_far = ret

                # Progress every 200 combos
                if counter % 200 == 0:
                    elapsed = time.time() - t0
                    rate = counter / elapsed
                    eta = (total_combos - counter) / rate if rate > 0 else 0
                    print(f"  [{counter}/{total_combos}] best IS return so far: {best_so_far:+.1f}% "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\n  Completed {counter} runs in {elapsed:.1f}s ({counter/elapsed:.0f} runs/sec)")
    print(f"  Valid results: {len(all_results)} (trades >= {MIN_TRADES})")

    # ── Filter IS Results ──
    qualifying = [r for r in all_results if r["is_return"] >= IS_RETURN_THRESHOLD]
    qualifying.sort(key=lambda x: x["is_return"], reverse=True)
    top_is = qualifying[:TOP_N_FOR_OOS]

    print(f"\n  Results with IS return >= {IS_RETURN_THRESHOLD}%: {len(qualifying)}")
    if not qualifying:
        # Lower threshold and try again
        print("  Lowering threshold to 5%...")
        qualifying = [r for r in all_results if r["is_return"] >= 5.0]
        qualifying.sort(key=lambda x: x["is_return"], reverse=True)
        top_is = qualifying[:TOP_N_FOR_OOS]
        print(f"  Results with IS return >= 5%: {len(qualifying)}")

    if not top_is:
        # Show best 20 regardless
        all_results.sort(key=lambda x: x["is_return"], reverse=True)
        top_is = all_results[:20]
        print(f"  Showing top 20 regardless of threshold")

    # ── Print IS Top Results ──
    print(f"\n{'='*90}")
    print(f"  TOP {len(top_is)} IN-SAMPLE RESULTS (sorted by return)")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strat':<5} {'Return':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD':>7} "
          f"{'Trades':>7} {'WinR':>6} {'CAGR':>7} {'Config':>8}")
    print("-" * 90)
    for i, r in enumerate(top_is[:30]):
        cfg_tag = f"v{r['config']['target_annual_vol']:.0f}" + \
                  ("L" if r['config']['short_size_mult'] == 0 else "S")
        print(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {r['is_pf']:>7.3f} "
              f"{r['is_sharpe']:>8.3f} {r['is_maxdd']:>6.1f}% {r['is_trades']:>7d} "
              f"{r['is_winrate']:>5.1f}% {r['is_cagr']:>+6.1f}% {cfg_tag:>8}")

    # ── Phase 2: OOS Validation ──
    print(f"\n[4/5] Validating top {len(top_is)} results on OOS data ({OOS_START} → end)...")

    oos_results = []
    for r in top_is:
        config = make_config(r["config"])
        result = run_single(r["strategy"], r["params"], config, df_oos)

        oos_entry = r.copy()
        if result is not None:
            oos_entry["oos_return"] = result["total_return_pct"]
            oos_entry["oos_pf"] = result["profit_factor"]
            oos_entry["oos_sharpe"] = result["sharpe_ratio"]
            oos_entry["oos_maxdd"] = result["max_drawdown_pct"]
            oos_entry["oos_trades"] = result["total_trades"]
            oos_entry["oos_winrate"] = result["win_rate"]
            oos_entry["oos_cagr"] = result["cagr_pct"]
        else:
            oos_entry["oos_return"] = None
            oos_entry["oos_pf"] = None
            oos_entry["oos_sharpe"] = None
            oos_entry["oos_maxdd"] = None
            oos_entry["oos_trades"] = 0
            oos_entry["oos_winrate"] = 0
            oos_entry["oos_cagr"] = None

        oos_results.append(oos_entry)

    # ── Sort by combined IS+OOS metric ──
    for r in oos_results:
        oos_ret = r["oos_return"] if r["oos_return"] is not None else -100
        r["combined_score"] = r["is_return"] * 0.4 + oos_ret * 0.6  # Weight OOS more

    oos_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # ── Print Final Results ──
    print(f"\n{'='*110}")
    print(f"  FINAL RESULTS — IS + OOS VALIDATED (sorted by combined score)")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Strat':<5} {'IS Ret':>9} {'OOS Ret':>9} {'IS PF':>7} {'OOS PF':>7} "
          f"{'IS DD':>7} {'OOS DD':>7} {'IS Tr':>6} {'OOS Tr':>6} {'Config':>8} {'Params'}")
    print("-" * 110)

    for i, r in enumerate(oos_results[:30]):
        cfg_tag = f"v{r['config']['target_annual_vol']:.0f}" + \
                  ("L" if r['config']['short_size_mult'] == 0 else "S")
        oos_ret_str = f"{r['oos_return']:>+8.1f}%" if r['oos_return'] is not None else "   N/A  "
        oos_pf_str = f"{r['oos_pf']:>7.3f}" if r['oos_pf'] is not None else "    N/A"
        oos_dd_str = f"{r['oos_maxdd']:>6.1f}%" if r['oos_maxdd'] is not None else "   N/A "
        oos_tr_str = f"{r['oos_trades']:>6d}" if r['oos_trades'] else "   N/A"

        # Compact param string
        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items())

        print(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {oos_ret_str} "
              f"{r['is_pf']:>7.3f} {oos_pf_str} {r['is_maxdd']:>6.1f}% {oos_dd_str} "
              f"{r['is_trades']:>6d} {oos_tr_str} {cfg_tag:>8}  {p_str}")

    # ── Highlight winners ──
    winners = [r for r in oos_results
               if r["oos_return"] is not None and r["oos_return"] > 0
               and r["is_return"] > 10]

    print(f"\n{'='*110}")
    if winners:
        print(f"  ★ PROFITABLE STRATEGIES (positive IS AND OOS): {len(winners)}")
        print(f"{'='*110}")
        for i, r in enumerate(winners[:10]):
            cfg_tag = f"vol={r['config']['target_annual_vol']:.0f}%, " + \
                      f"stop={r['config']['atr_stop_mult']}, trail={r['config']['atr_trail_mult']}, " + \
                      ("long-only" if r['config']['short_size_mult'] == 0 else f"shorts={r['config']['short_size_mult']}")
            p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(f"\n  #{i+1}: {r['strategy']} — IS: {r['is_return']:+.1f}% | OOS: {r['oos_return']:+.1f}%")
            print(f"       IS:  PF={r['is_pf']:.3f} Sharpe={r['is_sharpe']:.3f} MaxDD={r['is_maxdd']:.1f}% "
                  f"Trades={r['is_trades']} WinRate={r['is_winrate']:.1f}%")
            if r["oos_pf"] is not None:
                print(f"       OOS: PF={r['oos_pf']:.3f} Sharpe={r['oos_sharpe']:.3f} MaxDD={r['oos_maxdd']:.1f}% "
                      f"Trades={r['oos_trades']} WinRate={r['oos_winrate']:.1f}%")
            print(f"       Config: {cfg_tag}")
            print(f"       Params: {p_str}")
    else:
        print("  No strategies profitable on both IS and OOS periods.")
        print("  Top 5 IS results shown above for manual review.")
    print(f"{'='*110}")

    # ── Buy & Hold benchmark ──
    print("\n[5/5] Buy & Hold benchmark:")
    for label, df_period in [("IS", df_is), ("OOS", df_oos), ("Full", df_full)]:
        bh = STRATEGY_REGISTRY["BH"](BacktestConfig(warmup_bars=1))
        bh_res = bh.run(df_period)
        if "error" not in bh_res:
            print(f"  {label}: Return={bh_res['total_return_pct']:+.1f}% "
                  f"MaxDD={bh_res['max_drawdown_pct']:.1f}%")

    print("\nDone.")
    return oos_results


if __name__ == "__main__":
    run_optimization()
