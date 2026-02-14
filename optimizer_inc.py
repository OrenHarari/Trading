#!/usr/bin/env python3
"""
Incremental BTC/USD Strategy Optimizer — LONG + SHORT
Processes one strategy at a time, saves results after each.
Resilient to crashes — can resume from saved progress.

Usage:
    python optimizer_inc.py 2h     # 2-hour timeframe
    python optimizer_inc.py 4h     # 4-hour timeframe
"""

import gc
import json
import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestConfig
from backtester.strategies import STRATEGY_REGISTRY

# ── Config ──────────────────────────────────────
IS_MONTHS = 8
OOS_MONTHS = 4
MIN_TRADES = 5
IS_RETURN_THRESHOLD = 3.0
TOP_N_FOR_OOS = 80

STRATEGIES = ["T1", "M1", "RSI", "MACD", "SMA", "MOM", "DIP", "DON", "BKD", "STR", "MP"]

# Reduced grids for speed
GRIDS = {
    "T1":   {"ema_fast": [8, 21, 36], "ema_slow": [50, 100, 200], "adx_threshold": [15, 22]},
    "M1":   {"bb_len": [15, 20, 30], "bb_mult": [1.5, 2.0, 2.5], "rsi_os": [25, 35], "rsi_ob": [65, 75]},
    "RSI":  {"rsi_len": [7, 14, 21], "rsi_os": [25, 35], "rsi_ob": [65, 75]},
    "MACD": {"macd_fast": [8, 12], "macd_slow": [26, 40], "macd_signal": [5, 9]},
    "SMA":  {"sma_fast": [20, 50, 100], "sma_slow": [100, 200]},
    "MOM":  {"roc_len": [10, 20, 40], "roc_threshold": [5, 10, 20], "roc_exit": [-5, 0, 5], "use_sma_filter": [0, 1]},
    "DIP":  {"lookback": [30, 60, 120], "dip_pct": [8, 15, 25], "recovery_pct": [5, 10, 15], "max_hold": [48, 120]},
    "DON":  {"entry_len": [20, 40, 80], "exit_len": [10, 20]},
    "BKD":  {"sma_len": [20, 50], "rsi_len": [7, 14], "rsi_short": [40, 50], "rsi_long": [55, 65], "use_vol": [0, 1]},
    "STR":  {"lookback": [30, 60], "rip_pct": [15, 30], "pullback_pct": [5, 15], "dip_pct": [10, 20], "max_hold": [48, 120]},
    "MP":   {"ema_fast": [12, 21], "ema_slow": [50, 100], "roc_len": [10, 20], "structure_len": [30, 60], "entry_score": [2, 3]},
}

CONFIG_PRESETS = [
    {"target_annual_vol": 80,  "atr_stop_mult": 2.5, "atr_trail_mult": 4.0, "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 180},
    {"target_annual_vol": 80,  "atr_stop_mult": 1.5, "atr_trail_mult": 3.0, "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 120},
    {"target_annual_vol": 80,  "atr_stop_mult": 2.0, "atr_trail_mult": 5.0, "short_size_mult": 0.8, "warmup_bars": 72, "max_bars_trend": 180},
    {"target_annual_vol": 100, "atr_stop_mult": 2.0, "atr_trail_mult": 3.5, "short_size_mult": 1.2, "warmup_bars": 72, "max_bars_trend": 120},
    {"target_annual_vol": 80,  "atr_stop_mult": 2.0, "atr_trail_mult": 5.0, "short_size_mult": 0.0, "warmup_bars": 72, "max_bars_trend": 180},
]


def fetch_data(tf):
    import yfinance as yf
    cache = Path(f"data/BTC-USD_{tf}.csv")
    cache.parent.mkdir(exist_ok=True)
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < 12:
        df = pd.read_csv(cache)
        df["date"] = pd.to_datetime(df["date"])
        return df
    print(f"  Fetching 1h data from Yahoo Finance...")
    hist = yf.Ticker("BTC-USD").history(period="1y", interval="1h")
    if hist.empty:
        raise ValueError("No data")
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")
    rule = "2h" if tf == "2h" else "4h"
    ohlc = hist.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df = pd.DataFrame({
        "date": ohlc.index.tz_localize(None) if ohlc.index.tz else ohlc.index,
        "open": ohlc["Open"].values, "high": ohlc["High"].values,
        "low": ohlc["Low"].values, "close": ohlc["Close"].values,
        "volume": ohlc["Volume"].values,
    }).reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def make_config(preset, bpd):
    cfg = BacktestConfig()
    cfg.bars_per_day = bpd
    for k, v in preset.items():
        if hasattr(cfg, k): setattr(cfg, k, v)
    cfg.mr_stop_mult = min(cfg.atr_stop_mult, 1.5)
    cfg.mr_time_exit = max(10, cfg.max_bars_trend // 3)
    cfg.squeeze_stop_mult = cfg.atr_stop_mult
    cfg.squeeze_trail_mult = cfg.atr_trail_mult
    cfg.squeeze_max_bars = min(cfg.max_bars_trend, 60)
    return cfg


def run_one(skey, params, config, df):
    try:
        strat = STRATEGY_REGISTRY[skey](config, **params)
        r = strat.run(df)
        if "error" in r or r["total_trades"] < MIN_TRADES:
            return None
        return r
    except Exception:
        return None


def combos(grid):
    keys = list(grid.keys())
    return [dict(zip(keys, c)) for c in itertools.product(*grid.values())]


def run_optimizer(tf):
    bpd = 12 if tf == "2h" else 6
    progress_file = Path(f"optimizer_{tf}_progress.json")
    results_file = Path(f"optimizer_{tf}_short_results.txt")

    # Load data
    print(f"\n{'='*70}")
    print(f"  BTC/USD {tf.upper()} Optimizer — LONG + SHORT")
    print(f"{'='*70}")

    df_full = fetch_data(tf)
    print(f"  Bars: {len(df_full)} | {df_full['date'].iloc[0]} → {df_full['date'].iloc[-1]}")
    print(f"  Price range: ${df_full['close'].min():,.0f} — ${df_full['close'].max():,.0f}")

    # Split
    total_days = (df_full["date"].iloc[-1] - df_full["date"].iloc[0]).days
    is_end = df_full["date"].iloc[0] + pd.Timedelta(days=int(total_days * IS_MONTHS / 12))
    df_is = df_full[df_full["date"] < is_end].copy().reset_index(drop=True)
    df_oos = df_full[df_full["date"] >= is_end].copy().reset_index(drop=True)
    print(f"  IS: {len(df_is)} bars ({df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()})")
    print(f"  OOS: {len(df_oos)} bars ({df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()})")

    # Load progress
    completed_strats = set()
    all_results = []
    if progress_file.exists():
        data = json.loads(progress_file.read_text())
        completed_strats = set(data.get("completed", []))
        all_results = data.get("results", [])
        print(f"  Resuming: {len(completed_strats)} strategies already done")

    # IS Sweep — strategy by strategy
    total_combos = sum(len(combos(GRIDS[s])) * len(CONFIG_PRESETS) for s in STRATEGIES if s in GRIDS)
    print(f"\n  Total combos: {total_combos}")

    t0 = time.time()
    for skey in STRATEGIES:
        if skey not in GRIDS or skey in completed_strats:
            continue

        param_list = combos(GRIDS[skey])
        n_combos = len(param_list) * len(CONFIG_PRESETS)
        print(f"\n  >> {skey}: {n_combos} combos...", end="", flush=True)
        st = time.time()
        count = 0
        best = -999

        for params in param_list:
            for ci, preset in enumerate(CONFIG_PRESETS):
                cfg = make_config(preset, bpd)
                r = run_one(skey, params, cfg, df_is)
                count += 1
                if r is not None:
                    ret = r["total_return_pct"]
                    if ret > best: best = ret
                    all_results.append({
                        "strategy": skey,
                        "params": params,
                        "config_idx": ci,
                        "is_return": ret,
                        "is_pf": r["profit_factor"],
                        "is_sharpe": r["sharpe_ratio"],
                        "is_maxdd": r["max_drawdown_pct"],
                        "is_trades": r["total_trades"],
                        "is_winrate": r["win_rate"],
                    })

        elapsed = time.time() - st
        completed_strats.add(skey)
        print(f" done in {elapsed:.0f}s | best IS: {best:+.1f}% | valid: {count}", flush=True)

        # Save progress
        progress_file.write_text(json.dumps({
            "completed": list(completed_strats),
            "results": all_results,
        }))
        gc.collect()

    total_elapsed = time.time() - t0
    print(f"\n  IS sweep done: {len(all_results)} valid results in {total_elapsed:.0f}s")

    # Filter & OOS
    qualifying = sorted([r for r in all_results if r["is_return"] >= IS_RETURN_THRESHOLD],
                        key=lambda x: x["is_return"], reverse=True)[:TOP_N_FOR_OOS]
    if len(qualifying) < 10:
        qualifying = sorted(all_results, key=lambda x: x["is_return"], reverse=True)[:TOP_N_FOR_OOS]
    print(f"  Top {len(qualifying)} for OOS validation...")

    oos_results = []
    for r in qualifying:
        preset = CONFIG_PRESETS[r["config_idx"]]
        cfg = make_config(preset, bpd)
        res = run_one(r["strategy"], r["params"], cfg, df_oos)

        entry = r.copy()
        entry["config"] = preset
        if res:
            entry["oos_return"] = res["total_return_pct"]
            entry["oos_pf"] = res["profit_factor"]
            entry["oos_sharpe"] = res["sharpe_ratio"]
            entry["oos_maxdd"] = res["max_drawdown_pct"]
            entry["oos_trades"] = res["total_trades"]
            entry["oos_winrate"] = res["win_rate"]
        else:
            entry["oos_return"] = None
        oos_results.append(entry)

    # Combined score
    for r in oos_results:
        oos = r["oos_return"] if r["oos_return"] is not None else -100
        r["combined"] = r["is_return"] * 0.4 + oos * 0.6
    oos_results.sort(key=lambda x: x["combined"], reverse=True)

    # ── Build report ──
    lines = []
    def log(s=""): lines.append(s)

    log(f"{'='*110}")
    log(f"  BTC/USD {tf.upper()} OPTIMIZER — LONG + SHORT RESULTS")
    log(f"  Data: Yahoo Finance | IS: {df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()} | OOS: {df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()}")
    log(f"  IS bars: {len(df_is)} | OOS bars: {len(df_oos)} | {len(all_results)} IS results → top {len(qualifying)} validated")
    log(f"{'='*110}")

    # IS+OOS table
    log(f"\n{'#':>3} {'Strat':<5} {'IS Ret':>9} {'OOS Ret':>9} {'IS PF':>7} {'OOS PF':>7} {'IS DD':>7} {'OOS DD':>7} {'IS Tr':>6} {'OOS Tr':>6} {'Short':>6} {'Params'}")
    log("-" * 110)
    for i, r in enumerate(oos_results[:40]):
        sm = CONFIG_PRESETS[r["config_idx"]]["short_size_mult"]
        oos_r = f"{r['oos_return']:>+8.1f}%" if r["oos_return"] is not None else "     N/A "
        oos_p = f"{r['oos_pf']:>7.3f}" if r.get("oos_pf") else "    N/A"
        oos_d = f"{r['oos_maxdd']:>6.1f}%" if r.get("oos_maxdd") else "   N/A "
        oos_t = f"{r['oos_trades']:>6d}" if r.get("oos_trades") else "   N/A"
        p = " ".join(f"{k}={v}" for k, v in r["params"].items())
        log(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {oos_r} {r['is_pf']:>7.3f} {oos_p} {r['is_maxdd']:>6.1f}% {oos_d} {r['is_trades']:>6d} {oos_t} {sm:>5.1f}  {p}")

    # Winners
    winners = [r for r in oos_results if r["oos_return"] is not None and r["oos_return"] > 0]
    log(f"\n{'='*110}")
    if winners:
        log(f"  ★ PROFITABLE IS + OOS ({tf.upper()}): {len(winners)} strategies")
        log(f"{'='*110}")
        for i, r in enumerate(winners[:20]):
            p = CONFIG_PRESETS[r["config_idx"]]
            ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            log(f"\n  #{i+1}: {r['strategy']} — IS: {r['is_return']:+.1f}% | OOS: {r['oos_return']:+.1f}%")
            log(f"       IS:  PF={r['is_pf']:.3f} Sharpe={r['is_sharpe']:.3f} MaxDD={r['is_maxdd']:.1f}% Trades={r['is_trades']} WinRate={r['is_winrate']:.1f}%")
            if r.get("oos_pf"):
                log(f"       OOS: PF={r['oos_pf']:.3f} Sharpe={r['oos_sharpe']:.3f} MaxDD={r['oos_maxdd']:.1f}% Trades={r['oos_trades']} WinRate={r['oos_winrate']:.1f}%")
            log(f"       Config: vol={p['target_annual_vol']}% stop={p['atr_stop_mult']} trail={p['atr_trail_mult']} shorts={p['short_size_mult']}")
            log(f"       Params: {ps}")
    else:
        log("  No strategies profitable on both IS and OOS.")
    log(f"{'='*110}")

    # Pattern analysis
    log(f"\n{'='*70}")
    log(f"  PATTERN ANALYSIS — {tf.upper()}")
    log(f"{'='*70}")

    strat_stats = {}
    for r in all_results:
        s = r["strategy"]
        if s not in strat_stats: strat_stats[s] = {"n": 0, "pos": 0, "sum": 0, "best": -999}
        strat_stats[s]["n"] += 1
        if r["is_return"] > 0: strat_stats[s]["pos"] += 1
        strat_stats[s]["sum"] += r["is_return"]
        strat_stats[s]["best"] = max(strat_stats[s]["best"], r["is_return"])

    log(f"\n  Strategy robustness (IS):")
    log(f"  {'Strat':<6} {'Combos':>7} {'Positive':>9} {'%Pos':>6} {'AvgRet':>9} {'BestRet':>9}")
    log(f"  {'-'*50}")
    for s in sorted(strat_stats, key=lambda x: strat_stats[x]["pos"] / max(1, strat_stats[x]["n"]), reverse=True):
        d = strat_stats[s]
        log(f"  {s:<6} {d['n']:>7} {d['pos']:>9} {d['pos']/max(1,d['n'])*100:>5.0f}% {d['sum']/max(1,d['n']):>+8.1f}% {d['best']:>+8.1f}%")

    # Config preset analysis
    preset_stats = {}
    for r in all_results:
        ci = r["config_idx"]
        if ci not in preset_stats: preset_stats[ci] = {"n": 0, "pos": 0, "sum": 0}
        preset_stats[ci]["n"] += 1
        if r["is_return"] > 0: preset_stats[ci]["pos"] += 1
        preset_stats[ci]["sum"] += r["is_return"]

    log(f"\n  Config preset effectiveness:")
    log(f"  {'#':>3} {'%Pos':>7} {'AvgRet':>9} {'ShortMult':>10}")
    log(f"  {'-'*35}")
    for ci in sorted(preset_stats):
        d = preset_stats[ci]
        log(f"  {ci+1:>3} {d['pos']/max(1,d['n'])*100:>6.0f}% {d['sum']/max(1,d['n']):>+8.1f}% {CONFIG_PRESETS[ci]['short_size_mult']:>10.1f}")

    # Winning patterns
    if winners:
        log(f"\n  WINNING PATTERNS (IS + OOS > 0):")
        win_by = {}
        for w in winners:
            s = w["strategy"]
            if s not in win_by: win_by[s] = []
            win_by[s].append(w)
        for s, ws in sorted(win_by.items(), key=lambda x: -len(x[1])):
            log(f"\n  {s} ({len(ws)} winners):")
            for pkey in ws[0]["params"]:
                vals = sorted(set(w["params"][pkey] for w in ws))
                if len(vals) <= 3:
                    log(f"    {pkey}: {vals} (stable)")
                else:
                    log(f"    {pkey}: {min(vals)} → {max(vals)}")
            shorts_used = sum(1 for w in ws if CONFIG_PRESETS[w["config_idx"]]["short_size_mult"] > 0)
            avg_oos = np.mean([w["oos_return"] for w in ws if w["oos_return"] is not None])
            log(f"    Shorts: {shorts_used}/{len(ws)} | Avg OOS: {avg_oos:+.1f}%")

    # Buy & Hold
    log(f"\n  BUY & HOLD BENCHMARK:")
    for label, dfp in [("IS", df_is), ("OOS", df_oos), ("Full", df_full)]:
        bh = STRATEGY_REGISTRY["BH"](BacktestConfig(warmup_bars=1, bars_per_day=bpd))
        res = bh.run(dfp)
        if "error" not in res:
            log(f"    {label}: {res['total_return_pct']:+.1f}% (MaxDD={res['max_drawdown_pct']:.1f}%)")

    log(f"\n  Done — {tf.upper()} optimization complete.")

    # Save
    results_file.write_text("\n".join(lines))
    print(f"\n  Results saved to {results_file}")

    # Cleanup
    progress_file.unlink(missing_ok=True)
    return oos_results


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "2h"
    print(f"Starting {tf.upper()} optimization...", flush=True)
    run_optimizer(tf)
    print("✅ Done!", flush=True)
