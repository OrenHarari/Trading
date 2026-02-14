#!/usr/bin/env python3
"""
4-Hour BTC/USD Strategy Optimizer — Last 1 Year

Fetches 1h data from Yahoo Finance, resamples to 4h bars,
then runs the same parameter sweep as the daily optimizer.
Uses IS/OOS split within the 1-year window.

Results saved to: optimizer_4h_results.txt

Usage:
    python optimizer_4h.py
"""

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

BARS_PER_DAY = 6  # 24h / 4h = 6 bars per day

# IS/OOS split: first 8 months IS, last 4 months OOS
IS_MONTHS = 8
OOS_MONTHS = 4

MIN_TRADES = 6             # Fewer trades on 1-year window
IS_RETURN_THRESHOLD = 5.0  # Lower threshold for shorter period
TOP_N_FOR_OOS = 100        # Top N IS results to validate on OOS

# Strategies to optimize (skip BH=benchmark, H2=composite, H1=too slow)
OPTIMIZE_STRATEGIES = ["T1", "M1", "RSI", "MACD", "SMA", "MOM", "DIP", "DON"]

# ── Parameter Grids (adapted for 4h timeframe) ──────

STRATEGY_GRIDS = {
    "T1": {
        "ema_fast":      [8, 15, 24, 36, 50],
        "ema_slow":      [50, 80, 120, 200],
        "adx_threshold": [15, 20, 25, 30],
    },
    "M1": {
        "bb_len":  [15, 20, 30, 50],
        "bb_mult": [1.5, 2.0, 2.5],
        "rsi_os":  [20, 30, 40],
        "rsi_ob":  [60, 70, 80],
    },
    "RSI": {
        "rsi_len": [6, 10, 14, 21],
        "rsi_os":  [20, 25, 30, 35],
        "rsi_ob":  [65, 70, 75, 80],
    },
    "MACD": {
        "macd_fast":   [8, 12, 18, 24],
        "macd_slow":   [20, 26, 36, 52],
        "macd_signal": [5, 9, 14],
    },
    "SMA": {
        "sma_fast": [20, 50, 100, 150],
        "sma_slow": [100, 200, 300, 500],
    },
    "MOM": {
        "roc_len":        [10, 20, 30, 50, 72],
        "roc_threshold":  [3, 5, 10, 15, 20],
        "roc_exit":       [-5, 0, 5],
        "use_sma_filter": [0, 1],
    },
    "DIP": {
        "lookback":     [30, 60, 90, 120],
        "dip_pct":      [5, 10, 15, 20, 25],
        "recovery_pct": [3, 5, 10, 15],
        "max_hold":     [36, 72, 144],  # 6-24 days in 4h bars
    },
    "DON": {
        "entry_len": [20, 36, 55, 80, 120],
        "exit_len":  [10, 20, 30, 55],
    },
}

# Engine config presets for 4h
CONFIG_PRESETS = [
    # Long-only, aggressive, wide trail (best from daily)
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 5.0,
     "short_size_mult": 0.0, "warmup_bars": 100, "max_bars_trend": 120,
     "bars_per_day": BARS_PER_DAY},
    # Long-only, aggressive, medium trail
    {"target_annual_vol": 80, "atr_stop_mult": 3.0, "atr_trail_mult": 4.0,
     "short_size_mult": 0.0, "warmup_bars": 100, "max_bars_trend": 180,
     "bars_per_day": BARS_PER_DAY},
    # Long-only, medium vol, tight trail
    {"target_annual_vol": 50, "atr_stop_mult": 2.0, "atr_trail_mult": 3.0,
     "short_size_mult": 0.0, "warmup_bars": 100, "max_bars_trend": 120,
     "bars_per_day": BARS_PER_DAY},
    # Long-only, aggressive, tight stops
    {"target_annual_vol": 80, "atr_stop_mult": 1.5, "atr_trail_mult": 2.5,
     "short_size_mult": 0.0, "warmup_bars": 100, "max_bars_trend": 90,
     "bars_per_day": BARS_PER_DAY},
    # With shorts, aggressive
    {"target_annual_vol": 80, "atr_stop_mult": 2.0, "atr_trail_mult": 4.0,
     "short_size_mult": 0.5, "warmup_bars": 100, "max_bars_trend": 180,
     "bars_per_day": BARS_PER_DAY},
]


def fetch_btc_4h() -> pd.DataFrame:
    """Fetch 1h BTC data from Yahoo Finance and resample to 4h bars."""
    import yfinance as yf

    cache_file = Path("data/BTC-USD_4h.csv")
    cache_file.parent.mkdir(exist_ok=True)

    # Use cache if < 12h old
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            print(f"  Loaded from cache: {len(df)} 4h bars")
            return df

    print("  Fetching 1h data from Yahoo Finance...")
    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(period="1y", interval="1h")

    if hist.empty:
        raise ValueError("No hourly data from Yahoo Finance")

    print(f"  Raw 1h bars: {len(hist)}")

    # Ensure timezone-aware index for resampling
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize("UTC")

    # Resample to 4h OHLCV
    ohlc = hist.resample("4h").agg({
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

    # Cache
    df.to_csv(cache_file, index=False)
    print(f"  Resampled to 4h: {len(df)} bars")
    print(f"  Range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    return df


def make_config(preset: dict) -> BacktestConfig:
    """Create a BacktestConfig from a preset dict."""
    cfg = BacktestConfig()
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
    """Generate all combinations from a parameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_optimization():
    """Main 4h optimization loop."""
    output_file = Path("optimizer_4h_results.txt")
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("  BTC/USD 4-Hour Strategy Optimizer")
    log("  Real OHLC: Yahoo Finance 1h → 4h resample (last 1 year)")
    log("=" * 70)

    # ── Load Data ──
    log("\n[1/5] Loading 4h BTC/USD data...")
    df_full = fetch_btc_4h()
    total_bars = len(df_full)
    log(f"  Total 4h bars: {total_bars}")
    log(f"  Date range: {df_full['date'].iloc[0]} → {df_full['date'].iloc[-1]}")
    log(f"  Price range: ${df_full['close'].min():,.0f} → ${df_full['close'].max():,.0f}")

    # ── IS/OOS split ──
    total_days = (df_full["date"].iloc[-1] - df_full["date"].iloc[0]).days
    is_end_date = df_full["date"].iloc[0] + pd.Timedelta(days=int(total_days * IS_MONTHS / 12))

    df_is = df_full[df_full["date"] < is_end_date].copy().reset_index(drop=True)
    df_oos = df_full[df_full["date"] >= is_end_date].copy().reset_index(drop=True)

    log(f"\n[2/5] Data Split:")
    log(f"  IS  ({IS_MONTHS}mo): {len(df_is)} bars ({df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()})")
    log(f"  OOS ({OOS_MONTHS}mo): {len(df_oos)} bars ({df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()})")

    # ── Count combinations ──
    total_combos = 0
    for skey in OPTIMIZE_STRATEGIES:
        if skey in STRATEGY_GRIDS:
            n_params = len(generate_param_combos(STRATEGY_GRIDS[skey]))
            total_combos += n_params * len(CONFIG_PRESETS)

    log(f"\n[3/5] Running {total_combos} combinations on IS data...")
    log(f"  Strategies: {', '.join(OPTIMIZE_STRATEGIES)}")
    log(f"  Config presets: {len(CONFIG_PRESETS)}")

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

                if counter % 500 == 0:
                    elapsed = time.time() - t0
                    rate = counter / elapsed
                    eta = (total_combos - counter) / rate if rate > 0 else 0
                    log(f"  [{counter}/{total_combos}] best IS return: {best_so_far:+.1f}% "
                        f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    log(f"\n  Completed {counter} runs in {elapsed:.1f}s ({counter/elapsed:.0f} runs/sec)")
    log(f"  Valid results: {len(all_results)} (trades >= {MIN_TRADES})")

    # ── Filter IS Results ──
    qualifying = [r for r in all_results if r["is_return"] >= IS_RETURN_THRESHOLD]
    qualifying.sort(key=lambda x: x["is_return"], reverse=True)
    top_is = qualifying[:TOP_N_FOR_OOS]

    log(f"\n  Results with IS return >= {IS_RETURN_THRESHOLD}%: {len(qualifying)}")

    if not qualifying:
        qualifying = [r for r in all_results if r["is_return"] >= 0]
        qualifying.sort(key=lambda x: x["is_return"], reverse=True)
        top_is = qualifying[:TOP_N_FOR_OOS]
        log(f"  Lowered threshold to 0%: {len(qualifying)} results")

    if not top_is:
        all_results.sort(key=lambda x: x["is_return"], reverse=True)
        top_is = all_results[:30]
        log(f"  Showing top {len(top_is)} regardless")

    # ── Print IS Top ──
    log(f"\n{'='*95}")
    log(f"  TOP {min(30, len(top_is))} IN-SAMPLE RESULTS (sorted by return)")
    log(f"{'='*95}")
    log(f"{'#':>3} {'Strat':<5} {'Return':>9} {'PF':>7} {'Sharpe':>8} {'MaxDD':>7} "
        f"{'Trades':>7} {'WinR':>6} {'CAGR':>7} {'Config':>8}")
    log("-" * 95)
    for i, r in enumerate(top_is[:30]):
        cfg_tag = f"v{r['config']['target_annual_vol']:.0f}" + \
                  ("L" if r['config']['short_size_mult'] == 0 else "S")
        log(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {r['is_pf']:>7.3f} "
            f"{r['is_sharpe']:>8.3f} {r['is_maxdd']:>6.1f}% {r['is_trades']:>7d} "
            f"{r['is_winrate']:>5.1f}% {r['is_cagr']:>+6.1f}% {cfg_tag:>8}")

    # ── Phase 2: OOS Validation ──
    log(f"\n[4/5] Validating top {len(top_is)} on OOS ({OOS_MONTHS} months)...")

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

    # Combined score
    for r in oos_results:
        oos_ret = r["oos_return"] if r["oos_return"] is not None else -100
        r["combined_score"] = r["is_return"] * 0.4 + oos_ret * 0.6

    oos_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # ── Print Final Results ──
    log(f"\n{'='*115}")
    log(f"  FINAL RESULTS — 4H TIMEFRAME — IS + OOS VALIDATED (sorted by combined score)")
    log(f"{'='*115}")
    log(f"{'#':>3} {'Strat':<5} {'IS Ret':>9} {'OOS Ret':>9} {'IS PF':>7} {'OOS PF':>7} "
        f"{'IS DD':>7} {'OOS DD':>7} {'IS Tr':>6} {'OOS Tr':>6} {'Config':>8} {'Params'}")
    log("-" * 115)

    for i, r in enumerate(oos_results[:30]):
        cfg_tag = f"v{r['config']['target_annual_vol']:.0f}" + \
                  ("L" if r['config']['short_size_mult'] == 0 else "S")
        oos_ret_str = f"{r['oos_return']:>+8.1f}%" if r['oos_return'] is not None else "   N/A  "
        oos_pf_str = f"{r['oos_pf']:>7.3f}" if r['oos_pf'] is not None else "    N/A"
        oos_dd_str = f"{r['oos_maxdd']:>6.1f}%" if r['oos_maxdd'] is not None else "   N/A "
        oos_tr_str = f"{r['oos_trades']:>6d}" if r['oos_trades'] else "   N/A"

        p_str = " ".join(f"{k}={v}" for k, v in r["params"].items())
        log(f"{i+1:>3} {r['strategy']:<5} {r['is_return']:>+8.1f}% {oos_ret_str} "
            f"{r['is_pf']:>7.3f} {oos_pf_str} {r['is_maxdd']:>6.1f}% {oos_dd_str} "
            f"{r['is_trades']:>6d} {oos_tr_str} {cfg_tag:>8}  {p_str}")

    # ── Winners ──
    winners = [r for r in oos_results
               if r["oos_return"] is not None and r["oos_return"] > 0
               and r["is_return"] > 3]

    log(f"\n{'='*115}")
    if winners:
        log(f"  ★ PROFITABLE ON BOTH IS AND OOS (4H chart, last 1 year): {len(winners)}")
        log(f"{'='*115}")
        for i, r in enumerate(winners[:15]):
            cfg_tag = f"vol={r['config']['target_annual_vol']:.0f}%, " + \
                      f"stop={r['config']['atr_stop_mult']}, trail={r['config']['atr_trail_mult']}, " + \
                      f"max_bars={r['config']['max_bars_trend']}, " + \
                      ("long-only" if r['config']['short_size_mult'] == 0 else f"shorts={r['config']['short_size_mult']}")
            p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            log(f"\n  #{i+1}: {r['strategy']} — IS: {r['is_return']:+.1f}% | OOS: {r['oos_return']:+.1f}%")
            log(f"       IS:  PF={r['is_pf']:.3f} Sharpe={r['is_sharpe']:.3f} MaxDD={r['is_maxdd']:.1f}% "
                f"Trades={r['is_trades']} WinRate={r['is_winrate']:.1f}%")
            if r["oos_pf"] is not None:
                log(f"       OOS: PF={r['oos_pf']:.3f} Sharpe={r['oos_sharpe']:.3f} MaxDD={r['oos_maxdd']:.1f}% "
                    f"Trades={r['oos_trades']} WinRate={r['oos_winrate']:.1f}%")
            log(f"       Config: {cfg_tag}")
            log(f"       Params: {p_str}")
    else:
        log("  No strategies profitable on both IS and OOS periods.")
        log("  Top 30 IS results shown above for manual review.")
    log(f"{'='*115}")

    # ── Buy & Hold benchmark ──
    log("\n[5/5] Buy & Hold benchmark (4H bars):")
    for label, df_period in [("IS", df_is), ("OOS", df_oos), ("Full", df_full)]:
        bh = STRATEGY_REGISTRY["BH"](BacktestConfig(warmup_bars=1, bars_per_day=BARS_PER_DAY))
        bh_res = bh.run(df_period)
        if "error" not in bh_res:
            log(f"  {label}: Return={bh_res['total_return_pct']:+.1f}% "
                f"MaxDD={bh_res['max_drawdown_pct']:.1f}%")

    log("\nDone.")

    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {output_file}")

    return oos_results


if __name__ == "__main__":
    run_optimization()
