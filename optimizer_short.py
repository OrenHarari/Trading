#!/usr/bin/env python3
"""
2H & 4H BTC/USD Strategy Optimizer — With Aggressive Shorts

Fetches 1h data from Yahoo Finance, resamples to 2h or 4h bars.
Focuses on finding high-profit strategies using BOTH long and short.
Includes pattern-learning strategies (BKD, STR, MP).

Results saved to: optimizer_{timeframe}_results.txt

Usage:
    python optimizer_short.py 2h    # Run 2-hour optimization
    python optimizer_short.py 4h    # Run 4-hour optimization
    python optimizer_short.py both  # Run both (default)
"""

import gc
import sys
import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from backtester.engine import BacktestConfig
from backtester.strategies import STRATEGY_REGISTRY


# ── Configuration ─────────────────────────────────────

IS_MONTHS = 8
OOS_MONTHS = 4
MIN_TRADES = 5
IS_RETURN_THRESHOLD = 3.0
TOP_N_FOR_OOS = 120

# All strategies including new short-focused ones
OPTIMIZE_STRATEGIES = ["T1", "M1", "RSI", "MACD", "SMA", "MOM", "DIP", "DON", "BKD", "STR", "MP"]

# ── Parameter Grids ──────────────────────────────────

STRATEGY_GRIDS = {
    "T1": {
        "ema_fast":      [8, 15, 24, 36],
        "ema_slow":      [50, 80, 120, 200],
        "adx_threshold": [15, 20, 25],
    },
    "M1": {
        "bb_len":  [15, 20, 30],
        "bb_mult": [1.5, 2.0, 2.5],
        "rsi_os":  [25, 35],
        "rsi_ob":  [65, 75],
    },
    "RSI": {
        "rsi_len": [7, 14, 21],
        "rsi_os":  [20, 30, 40],
        "rsi_ob":  [60, 70, 80],
    },
    "MACD": {
        "macd_fast":   [8, 12, 18],
        "macd_slow":   [20, 26, 40],
        "macd_signal": [5, 9, 14],
    },
    "SMA": {
        "sma_fast": [20, 50, 100],
        "sma_slow": [100, 200, 400],
    },
    "MOM": {
        "roc_len":        [10, 20, 30, 50],
        "roc_threshold":  [5, 10, 15, 20],
        "roc_exit":       [-5, 0, 5],
        "use_sma_filter": [0, 1],
    },
    "DIP": {
        "lookback":     [30, 60, 120],
        "dip_pct":      [8, 12, 18, 25],
        "recovery_pct": [5, 10, 15],
        "max_hold":     [36, 72, 144],
    },
    "DON": {
        "entry_len": [20, 36, 55, 80],
        "exit_len":  [10, 20, 36],
    },
    # NEW: Short-focused strategies
    "BKD": {
        "sma_len":   [20, 50, 80],
        "rsi_len":   [7, 14],
        "rsi_short": [35, 45, 50],
        "rsi_long":  [55, 65],
        "use_vol":   [0, 1],
    },
    "STR": {
        "lookback":     [30, 60, 120],
        "rip_pct":      [15, 25, 40],
        "pullback_pct": [5, 10, 20],
        "dip_pct":      [10, 20],
        "max_hold":     [36, 72, 120],
    },
    "MP": {
        "ema_fast":      [12, 21, 36],
        "ema_slow":      [50, 100, 200],
        "roc_len":       [10, 20],
        "structure_len": [30, 60],
        "entry_score":   [2, 3, 4],
    },
}

# Config presets — heavy on shorts! 
CONFIG_PRESETS = [
    # Short-heavy: high short size, aggressive
    {"target_annual_vol": 80, "atr_stop_mult": 2.5, "atr_trail_mult": 4.0,
     "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 180},
    # Short-heavy with tight stops
    {"target_annual_vol": 80, "atr_stop_mult": 1.5, "atr_trail_mult": 3.0,
     "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 120},
    # Balanced long+short, wide trail
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 5.0,
     "short_size_mult": 0.8, "warmup_bars": 72, "max_bars_trend": 180},
    # Pure short focus (short_size > 1 = leverage on shorts)
    {"target_annual_vol": 100, "atr_stop_mult": 2.0, "atr_trail_mult": 3.5,
     "short_size_mult": 1.2, "warmup_bars": 72, "max_bars_trend": 120},
    # Long-only aggressive (for comparison)
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 5.0,
     "short_size_mult": 0.0, "warmup_bars": 72, "max_bars_trend": 180},
    # Medium vol, balanced 
    {"target_annual_vol": 60, "atr_stop_mult": 2.0, "atr_trail_mult": 3.5,
     "short_size_mult": 0.8, "warmup_bars": 72, "max_bars_trend": 120},
]


def fetch_btc_intraday(timeframe: str) -> pd.DataFrame:
    """Fetch 1h BTC data and resample to 2h or 4h."""
    import yfinance as yf

    if timeframe == "2h":
        resample_rule = "2h"
        bars_per_day = 12
    elif timeframe == "4h":
        resample_rule = "4h"
        bars_per_day = 6
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    cache_file = Path(f"data/BTC-USD_{timeframe}.csv")
    cache_file.parent.mkdir(exist_ok=True)

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return df, bars_per_day

    print(f"  Fetching 1h data from Yahoo Finance...")
    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(period="1y", interval="1h")

    if hist.empty:
        raise ValueError("No hourly data from Yahoo Finance")

    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")

    ohlc = hist.resample(resample_rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    df = pd.DataFrame({
        "date": ohlc.index.tz_localize(None) if ohlc.index.tz else ohlc.index,
        "open": ohlc["Open"].values,
        "high": ohlc["High"].values,
        "low": ohlc["Low"].values,
        "close": ohlc["Close"].values,
        "volume": ohlc["Volume"].values,
    }).reset_index(drop=True)

    df.to_csv(cache_file, index=False)
    return df, bars_per_day


def make_config(preset: dict, bars_per_day: int) -> BacktestConfig:
    """Create a BacktestConfig from preset + bars_per_day."""
    cfg = BacktestConfig()
    cfg.bars_per_day = bars_per_day
    for k, v in preset.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.mr_stop_mult = min(cfg.atr_stop_mult, 1.5)
    cfg.mr_time_exit = max(10, cfg.max_bars_trend // 3)
    cfg.squeeze_stop_mult = cfg.atr_stop_mult
    cfg.squeeze_trail_mult = cfg.atr_trail_mult
    cfg.squeeze_max_bars = min(cfg.max_bars_trend, 60)
    return cfg


def run_single(strategy_key, strategy_params, config, df):
    """Run a single backtest."""
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
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_optimization_for_timeframe(timeframe: str):
    """Run full optimization for a single timeframe."""
    output_file = Path(f"optimizer_{timeframe}_short_results.txt")
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    bars_per_day = 12 if timeframe == "2h" else 6

    log("=" * 75)
    log(f"  BTC/USD {timeframe.upper()} Strategy Optimizer — LONG + SHORT")
    log(f"  Real OHLC: Yahoo Finance (last 1 year)")
    log("=" * 75)

    # ── Load Data ──
    log(f"\n[1/5] Loading {timeframe} data...")
    df_full, bpd = fetch_btc_intraday(timeframe)
    log(f"  Total bars: {len(df_full)} ({timeframe})")
    log(f"  Range: {df_full['date'].iloc[0]} → {df_full['date'].iloc[-1]}")
    log(f"  Price: ${df_full['close'].min():,.0f} → ${df_full['close'].max():,.0f}")

    # ── IS/OOS split ──
    total_days = (df_full["date"].iloc[-1] - df_full["date"].iloc[0]).days
    is_end = df_full["date"].iloc[0] + pd.Timedelta(days=int(total_days * IS_MONTHS / 12))

    df_is = df_full[df_full["date"] < is_end].copy().reset_index(drop=True)
    df_oos = df_full[df_full["date"] >= is_end].copy().reset_index(drop=True)

    log(f"\n[2/5] Split: IS ({IS_MONTHS}mo): {len(df_is)} bars | OOS ({OOS_MONTHS}mo): {len(df_oos)} bars")
    log(f"  IS:  {df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()}")
    log(f"  OOS: {df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()}")

    # ── Count combos ──
    total_combos = 0
    combo_breakdown = {}
    for skey in OPTIMIZE_STRATEGIES:
        if skey in STRATEGY_GRIDS:
            n = len(generate_param_combos(STRATEGY_GRIDS[skey]))
            combo_breakdown[skey] = n * len(CONFIG_PRESETS)
            total_combos += n * len(CONFIG_PRESETS)

    log(f"\n[3/5] Running {total_combos} combinations on IS data...")
    log(f"  Strategies: {', '.join(OPTIMIZE_STRATEGIES)}")
    for skey, n in combo_breakdown.items():
        log(f"    {skey}: {n} combos")
    log(f"  Config presets: {len(CONFIG_PRESETS)} (short_size_mult up to 1.2)")

    # ── Phase 1: IS Sweep ──
    all_results = []
    counter = 0
    t0 = time.time()
    best_so_far = -999

    for skey in OPTIMIZE_STRATEGIES:
        if skey not in STRATEGY_GRIDS:
            continue

        param_combos = generate_param_combos(STRATEGY_GRIDS[skey])
        strat_t0 = time.time()

        for params in param_combos:
            for ci, preset in enumerate(CONFIG_PRESETS):
                counter += 1
                config = make_config(preset, bpd)
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

                if counter % 500 == 0:
                    elapsed = time.time() - t0
                    rate = counter / elapsed
                    eta = (total_combos - counter) / rate if rate > 0 else 0
                    log(f"  [{counter}/{total_combos}] best IS: {best_so_far:+.1f}% "
                        f"({elapsed:.0f}s, ~{eta:.0f}s left)")

        strat_elapsed = time.time() - strat_t0
        log(f"  >> {skey} done in {strat_elapsed:.0f}s")
        gc.collect()

    elapsed = time.time() - t0
    log(f"\n  IS Sweep: {counter} runs in {elapsed:.1f}s ({counter/elapsed:.0f}/sec)")
    log(f"  Valid results: {len(all_results)}")

    # ── Filter ──
    qualifying = [r for r in all_results if r["is_return"] >= IS_RETURN_THRESHOLD]
    qualifying.sort(key=lambda x: x["is_return"], reverse=True)
    top_is = qualifying[:TOP_N_FOR_OOS]

    log(f"  IS return >= {IS_RETURN_THRESHOLD}%: {len(qualifying)}")

    if len(qualifying) < 10:
        all_results.sort(key=lambda x: x["is_return"], reverse=True)
        top_is = all_results[:TOP_N_FOR_OOS]
        log(f"  Using top {len(top_is)} regardless of threshold")

    # ── IS Table ──
    log(f"\n{'='*100}")
    log(f"  TOP {min(40, len(top_is))} IN-SAMPLE — {timeframe.upper()} (sorted by return)")
    log(f"{'='*100}")
    log(f"{'#':>3} {'Strat':<5} {'Return':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD':>7} "
        f"{'Trades':>7} {'WinR':>6} {'Short':>6} {'Params'}")
    log("-" * 100)
    for i, r in enumerate(top_is[:40]):
        short_tag = f"{r['config']['short_size_mult']:.1f}"
        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
        log(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {r['is_pf']:>7.3f} "
            f"{r['is_sharpe']:>8.3f} {r['is_maxdd']:>6.1f}% {r['is_trades']:>7d} "
            f"{r['is_winrate']:>5.1f}% {short_tag:>6}  {p_str}")

    # ── Phase 2: OOS ──
    log(f"\n[4/5] Validating top {len(top_is)} on OOS...")

    oos_results = []
    for r in top_is:
        config = make_config(r["config"], bpd)
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

    # Combined score
    for r in oos_results:
        oos_ret = r["oos_return"] if r["oos_return"] is not None else -100
        r["combined_score"] = r["is_return"] * 0.4 + oos_ret * 0.6

    oos_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # ── Final Table ──
    log(f"\n{'='*120}")
    log(f"  FINAL: {timeframe.upper()} IS+OOS VALIDATED (sorted by combined IS×0.4 + OOS×0.6)")
    log(f"{'='*120}")
    log(f"{'#':>3} {'Strat':<5} {'IS Ret':>9} {'OOS Ret':>9} {'IS PF':>7} {'OOS PF':>7} "
        f"{'IS DD':>7} {'OOS DD':>7} {'IS Tr':>6} {'OOS Tr':>6} {'Short':>6} {'Params'}")
    log("-" * 120)

    for i, r in enumerate(oos_results[:40]):
        short_tag = f"{r['config']['short_size_mult']:.1f}"
        oos_ret_str = f"{r['oos_return']:>+8.1f}%" if r['oos_return'] is not None else "   N/A  "
        oos_pf_str = f"{r['oos_pf']:>7.3f}" if r['oos_pf'] is not None else "    N/A"
        oos_dd_str = f"{r['oos_maxdd']:>6.1f}%" if r['oos_maxdd'] is not None else "   N/A "
        oos_tr_str = f"{r['oos_trades']:>6d}" if r['oos_trades'] else "   N/A"

        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
        log(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {oos_ret_str} "
            f"{r['is_pf']:>7.3f} {oos_pf_str} {r['is_maxdd']:>6.1f}% {oos_dd_str} "
            f"{r['is_trades']:>6d} {oos_tr_str} {short_tag:>6}  {p_str}")

    # ── Winners ──
    winners = [r for r in oos_results
               if r["oos_return"] is not None and r["oos_return"] > 0]

    log(f"\n{'='*120}")
    if winners:
        log(f"  ★ PROFITABLE BOTH IS + OOS ({timeframe.upper()}): {len(winners)} strategies")
        log(f"{'='*120}")
        for i, r in enumerate(winners[:20]):
            cfg = r["config"]
            cfg_str = (f"vol={cfg['target_annual_vol']}%, stop={cfg['atr_stop_mult']}, "
                       f"trail={cfg['atr_trail_mult']}, shorts={cfg['short_size_mult']}, "
                       f"max_bars={cfg['max_bars_trend']}")
            p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            log(f"\n  #{i+1}: {r['strategy']} — IS: {r['is_return']:+.1f}% | OOS: {r['oos_return']:+.1f}%")
            log(f"       IS:  PF={r['is_pf']:.3f} Sharpe={r['is_sharpe']:.3f} MaxDD={r['is_maxdd']:.1f}% "
                f"Trades={r['is_trades']} WinRate={r['is_winrate']:.1f}%")
            if r["oos_pf"] is not None:
                log(f"       OOS: PF={r['oos_pf']:.3f} Sharpe={r['oos_sharpe']:.3f} MaxDD={r['oos_maxdd']:.1f}% "
                    f"Trades={r['oos_trades']} WinRate={r['oos_winrate']:.1f}%")
            log(f"       Config: {cfg_str}")
            log(f"       Params: {p_str}")
    else:
        log("  No strategies profitable on both IS and OOS.")
    log(f"{'='*120}")

    # ── Pattern Analysis ──
    log(f"\n{'='*75}")
    log(f"  PATTERN ANALYSIS — Recurring Patterns That Work on {timeframe.upper()}")
    log(f"{'='*75}")

    # Analyze which strategies work across different configs
    strat_stats = {}
    for r in all_results:
        skey = r["strategy"]
        if skey not in strat_stats:
            strat_stats[skey] = {"count": 0, "pos": 0, "avg_ret": 0, "best": -999}
        strat_stats[skey]["count"] += 1
        if r["is_return"] > 0:
            strat_stats[skey]["pos"] += 1
        strat_stats[skey]["avg_ret"] += r["is_return"]
        strat_stats[skey]["best"] = max(strat_stats[skey]["best"], r["is_return"])

    log(f"\n  Strategy robustness (IS period):")
    log(f"  {'Strat':<6} {'Combos':>7} {'Positive':>9} {'%Pos':>6} {'AvgRet':>9} {'BestRet':>9}")
    log(f"  " + "-" * 50)
    for skey in sorted(strat_stats, key=lambda x: strat_stats[x]["pos"] / max(1, strat_stats[x]["count"]), reverse=True):
        s = strat_stats[skey]
        avg = s["avg_ret"] / max(1, s["count"])
        pct = s["pos"] / max(1, s["count"]) * 100
        log(f"  {skey:<6} {s['count']:>7} {s['pos']:>9} {pct:>5.0f}% {avg:>+8.1f}% {s['best']:>+8.1f}%")

    # Analyze which config presets work best
    preset_stats = {}
    for r in all_results:
        ci = r["config_idx"]
        if ci not in preset_stats:
            preset_stats[ci] = {"count": 0, "pos": 0, "avg_ret": 0}
        preset_stats[ci]["count"] += 1
        if r["is_return"] > 0:
            preset_stats[ci]["pos"] += 1
        preset_stats[ci]["avg_ret"] += r["is_return"]

    log(f"\n  Config preset effectiveness:")
    log(f"  {'Preset':>7} {'%Positive':>10} {'AvgReturn':>10} {'ShortMult':>10}")
    log(f"  " + "-" * 40)
    for ci in sorted(preset_stats.keys()):
        s = preset_stats[ci]
        avg = s["avg_ret"] / max(1, s["count"])
        pct = s["pos"] / max(1, s["count"]) * 100
        sm = CONFIG_PRESETS[ci]["short_size_mult"]
        log(f"  #{ci+1:>5} {pct:>9.0f}% {avg:>+9.1f}% {sm:>10.1f}")

    # Winning pattern summary
    if winners:
        log(f"\n  Winning patterns (IS + OOS profitable):")
        win_strats = {}
        for w in winners:
            sk = w["strategy"]
            if sk not in win_strats:
                win_strats[sk] = []
            win_strats[sk].append(w)

        for sk, ws in sorted(win_strats.items(), key=lambda x: -len(x[1])):
            log(f"\n  {sk} ({len(ws)} winning combos):")
            # Find common param ranges
            for pkey in ws[0]["params"]:
                vals = [w["params"][pkey] for w in ws]
                unique = sorted(set(vals))
                if len(unique) <= 3:
                    log(f"    {pkey}: {unique} (narrow — stable pattern)")
                else:
                    log(f"    {pkey}: {min(vals)} → {max(vals)} (wide range)")
            # Common config traits
            shorts_used = [w["config"]["short_size_mult"] > 0 for w in ws]
            log(f"    Uses shorts: {sum(shorts_used)}/{len(ws)} combos")
            avg_oos = np.mean([w["oos_return"] for w in ws if w["oos_return"] is not None])
            log(f"    Average OOS return: {avg_oos:+.1f}%")

    # ── Buy & Hold benchmark ──
    log(f"\n[5/5] Buy & Hold benchmark ({timeframe}):")
    for label, dfp in [("IS", df_is), ("OOS", df_oos), ("Full", df_full)]:
        bh = STRATEGY_REGISTRY["BH"](BacktestConfig(warmup_bars=1, bars_per_day=bpd))
        res = bh.run(dfp)
        if "error" not in res:
            log(f"  {label}: Return={res['total_return_pct']:+.1f}% MaxDD={res['max_drawdown_pct']:.1f}%")

    log(f"\nDone — {timeframe.upper()} optimization complete.")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {output_file}")

    return oos_results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("2h", "both"):
        print("\n" + "▓" * 75)
        print("  STARTING 2H OPTIMIZATION")
        print("▓" * 75 + "\n")
        run_optimization_for_timeframe("2h")

    if mode in ("4h", "both"):
        print("\n" + "▓" * 75)
        print("  STARTING 4H OPTIMIZATION")
        print("▓" * 75 + "\n")
        run_optimization_for_timeframe("4h")

    print("\n✅ All optimizations complete!")


if __name__ == "__main__":
    main()
