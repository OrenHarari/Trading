#!/usr/bin/env python3
"""
Run ONE strategy optimization and save results to JSON.
Usage: python opt_single.py <timeframe> <strategy_key>
"""
import json, sys, time, itertools, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from backtester.engine import BacktestConfig
from backtester.strategies import STRATEGY_REGISTRY

IS_MONTHS, OOS_MONTHS, MIN_TRADES = 8, 4, 5

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

PRESETS = [
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
        df = pd.read_csv(cache); df["date"] = pd.to_datetime(df["date"]); return df
    hist = yf.Ticker("BTC-USD").history(period="1y", interval="1h")
    if hist.empty: raise ValueError("No data")
    if hist.index.tz is None: hist.index = hist.index.tz_localize("UTC")
    rule = "2h" if tf == "2h" else "4h"
    ohlc = hist.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    df = pd.DataFrame({"date": ohlc.index.tz_localize(None) if ohlc.index.tz else ohlc.index,
        "open": ohlc["Open"].values, "high": ohlc["High"].values, "low": ohlc["Low"].values,
        "close": ohlc["Close"].values, "volume": ohlc["Volume"].values}).reset_index(drop=True)
    df.to_csv(cache, index=False); return df

def make_cfg(p, bpd):
    c = BacktestConfig(); c.bars_per_day = bpd
    for k, v in p.items():
        if hasattr(c, k): setattr(c, k, v)
    c.mr_stop_mult = min(c.atr_stop_mult, 1.5); c.mr_time_exit = max(10, c.max_bars_trend // 3)
    c.squeeze_stop_mult = c.atr_stop_mult; c.squeeze_trail_mult = c.atr_trail_mult
    c.squeeze_max_bars = min(c.max_bars_trend, 60); return c

def run_one(skey, params, cfg, df):
    try:
        r = STRATEGY_REGISTRY[skey](cfg, **params).run(df)
        if "error" in r or r["total_trades"] < MIN_TRADES: return None
        return r
    except: return None

def main():
    tf = sys.argv[1]  # 2h or 4h
    skey = sys.argv[2]  # strategy key
    bpd = 12 if tf == "2h" else 6
    outdir = Path(f"opt_results_{tf}")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"{skey}.json"
    
    if outfile.exists():
        print(f"  {skey}: already done, skipping"); return

    df = fetch_data(tf)
    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    is_end = df["date"].iloc[0] + pd.Timedelta(days=int(total_days * IS_MONTHS / 12))
    df_is = df[df["date"] < is_end].copy().reset_index(drop=True)

    grid = GRIDS.get(skey, {})
    keys = list(grid.keys()); combos = [dict(zip(keys, c)) for c in itertools.product(*grid.values())]
    n = len(combos) * len(PRESETS)
    print(f"  {skey}: {n} combos on {len(df_is)} IS bars...", end="", flush=True)
    t0 = time.time()
    
    results = []
    for params in combos:
        for ci, preset in enumerate(PRESETS):
            cfg = make_cfg(preset, bpd)
            r = run_one(skey, params, cfg, df_is)
            if r:
                results.append({
                    "strategy": skey, "params": params, "config_idx": ci,
                    "is_return": round(r["total_return_pct"], 2),
                    "is_pf": round(r["profit_factor"], 3),
                    "is_sharpe": round(r["sharpe_ratio"], 3),
                    "is_maxdd": round(r["max_drawdown_pct"], 1),
                    "is_trades": r["total_trades"],
                    "is_winrate": round(r["win_rate"], 1),
                })
    
    elapsed = time.time() - t0
    best = max((r["is_return"] for r in results), default=-999)
    print(f" {elapsed:.0f}s | {len(results)} valid | best: {best:+.1f}%")
    
    outfile.write_text(json.dumps(results))
    gc.collect()

if __name__ == "__main__":
    main()
