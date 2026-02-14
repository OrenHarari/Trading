#!/usr/bin/env python3
"""
Combine per-strategy IS results, validate top ones on OOS, generate report.
Usage: python opt_combine.py <timeframe>
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from backtester.engine import BacktestConfig
from backtester.strategies import STRATEGY_REGISTRY

IS_MONTHS, OOS_MONTHS, MIN_TRADES = 8, 4, 5
TOP_N = 80

PRESETS = [
    {"target_annual_vol": 80,  "atr_stop_mult": 2.5, "atr_trail_mult": 4.0, "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 180},
    {"target_annual_vol": 80,  "atr_stop_mult": 1.5, "atr_trail_mult": 3.0, "short_size_mult": 1.0, "warmup_bars": 72, "max_bars_trend": 120},
    {"target_annual_vol": 80,  "atr_stop_mult": 2.0, "atr_trail_mult": 5.0, "short_size_mult": 0.8, "warmup_bars": 72, "max_bars_trend": 180},
    {"target_annual_vol": 100, "atr_stop_mult": 2.0, "atr_trail_mult": 3.5, "short_size_mult": 1.2, "warmup_bars": 72, "max_bars_trend": 120},
    {"target_annual_vol": 80,  "atr_stop_mult": 2.0, "atr_trail_mult": 5.0, "short_size_mult": 0.0, "warmup_bars": 72, "max_bars_trend": 180},
]

def fetch_data(tf):
    cache = Path(f"data/BTC-USD_{tf}.csv")
    df = pd.read_csv(cache); df["date"] = pd.to_datetime(df["date"]); return df

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
    tf = sys.argv[1]
    bpd = 12 if tf == "2h" else 6
    resdir = Path(f"opt_results_{tf}")
    
    # Load all IS results
    all_results = []
    for f in resdir.glob("*.json"):
        data = json.loads(f.read_text())
        all_results.extend(data)
        print(f"  Loaded {f.name}: {len(data)} results")
    print(f"  Total IS results: {len(all_results)}")
    
    # Load data
    df_full = fetch_data(tf)
    total_days = (df_full["date"].iloc[-1] - df_full["date"].iloc[0]).days
    is_end = df_full["date"].iloc[0] + pd.Timedelta(days=int(total_days * IS_MONTHS / 12))
    df_is = df_full[df_full["date"] < is_end].copy().reset_index(drop=True)
    df_oos = df_full[df_full["date"] >= is_end].copy().reset_index(drop=True)
    
    # Select top for OOS
    qualifying = sorted([r for r in all_results if r["is_return"] >= 3.0],
                        key=lambda x: x["is_return"], reverse=True)[:TOP_N]
    if len(qualifying) < 10:
        qualifying = sorted(all_results, key=lambda x: x["is_return"], reverse=True)[:TOP_N]
    
    print(f"  Validating top {len(qualifying)} on OOS ({len(df_oos)} bars)...")
    
    # OOS validation
    oos_results = []
    for r in qualifying:
        preset = PRESETS[r["config_idx"]]
        cfg = make_cfg(preset, bpd)
        res = run_one(r["strategy"], r["params"], cfg, df_oos)
        entry = r.copy()
        entry["config"] = preset
        if res:
            entry["oos_return"] = round(res["total_return_pct"], 2)
            entry["oos_pf"] = round(res["profit_factor"], 3)
            entry["oos_sharpe"] = round(res["sharpe_ratio"], 3)
            entry["oos_maxdd"] = round(res["max_drawdown_pct"], 1)
            entry["oos_trades"] = res["total_trades"]
            entry["oos_winrate"] = round(res["win_rate"], 1)
        else:
            entry["oos_return"] = None
        oos_results.append(entry)
    
    # Score and sort
    for r in oos_results:
        oos = r["oos_return"] if r["oos_return"] is not None else -100
        r["combined"] = r["is_return"] * 0.4 + oos * 0.6
    oos_results.sort(key=lambda x: x["combined"], reverse=True)
    
    # Generate report
    lines = []
    def log(s=""): lines.append(s)
    
    log(f"{'='*110}")
    log(f"  BTC/USD {tf.upper()} OPTIMIZER RESULTS — LONG + SHORT")
    log(f"  Yahoo Finance 1Y | IS: {df_is['date'].iloc[0].date()} → {df_is['date'].iloc[-1].date()} ({len(df_is)} bars)")
    log(f"  OOS: {df_oos['date'].iloc[0].date()} → {df_oos['date'].iloc[-1].date()} ({len(df_oos)} bars)")
    log(f"  {len(all_results)} IS combos → top {len(qualifying)} validated on OOS")
    log(f"{'='*110}")
    
    # Top table
    log(f"\n{'#':>3} {'Strat':<5} {'IS Ret':>9} {'OOS Ret':>9} {'IS PF':>7} {'OOS PF':>7} {'IS DD':>7} {'OOS DD':>7} {'IS Tr':>6} {'OOS Tr':>6} {'Short':>6} {'Params'}")
    log("-" * 120)
    for i, r in enumerate(oos_results[:40]):
        sm = PRESETS[r["config_idx"]]["short_size_mult"]
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
        for i, r in enumerate(winners[:25]):
            p = PRESETS[r["config_idx"]]
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
    for s in sorted(strat_stats, key=lambda x: strat_stats[x]["pos"]/max(1, strat_stats[x]["n"]), reverse=True):
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
        log(f"  {ci+1:>3} {d['pos']/max(1,d['n'])*100:>6.0f}% {d['sum']/max(1,d['n']):>+8.1f}% {PRESETS[ci]['short_size_mult']:>10.1f}")
    
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
            shorts_used = sum(1 for w in ws if PRESETS[w["config_idx"]]["short_size_mult"] > 0)
            avg_oos = np.mean([w["oos_return"] for w in ws if w["oos_return"] is not None])
            log(f"    Shorts: {shorts_used}/{len(ws)} | Avg OOS: {avg_oos:+.1f}%")
    
    # Buy & Hold
    log(f"\n  BUY & HOLD BENCHMARK:")
    for label, dfp in [("IS", df_is), ("OOS", df_oos), ("Full", df_full)]:
        bh = STRATEGY_REGISTRY["BH"](BacktestConfig(warmup_bars=1, bars_per_day=bpd))
        res = bh.run(dfp)
        if "error" not in res:
            log(f"    {label}: {res['total_return_pct']:+.1f}% (MaxDD={res['max_drawdown_pct']:.1f}%)")
    
    log(f"\n  Optimization complete.")
    
    report = "\n".join(lines)
    output_file = Path(f"optimizer_{tf}_short_results.txt")
    output_file.write_text(report)
    print(report)
    print(f"\n  Saved to: {output_file}")

if __name__ == "__main__":
    main()
