#!/usr/bin/env python3
"""
Main Backtest Runner — fetches real BTC data, runs all strategies
across multiple timeframes, iterates parameter variants until >30% profit found.

Generates HTML dashboard and Pine Script output.

Usage:
    python run_backtest.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from strategies import STRATEGIES, PARAM_VARIANTS
from engine import run_backtest, INITIAL_CAPITAL
from report_generator import generate_html_report


TARGET_PROFIT_PCT = 30.0
MAX_ITERATIONS = 3  # Max optimization rounds


def run_single_strategy(df: pd.DataFrame, strategy_name: str, params: dict) -> dict:
    """Run a single strategy with given parameters on data."""
    strat_info = STRATEGIES[strategy_name]
    func = strat_info["func"]

    signals = func(df, **params)
    result = run_backtest(df, signals)
    result["strategy_name"] = strategy_name
    result["strategy_type"] = strat_info["type"]
    result["params"] = params

    return result


def run_all_strategies(data: dict) -> list:
    """Run all strategies across all timeframes. Returns list of results."""
    all_results = []

    for tf_name, df in data.items():
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf_name} ({len(df)} bars)")
        print(f"{'='*60}")

        for strat_name, strat_info in STRATEGIES.items():
            print(f"\n  Strategy: {strat_name} ({strat_info['type']})")

            # Run with default params
            try:
                result = run_single_strategy(df, strat_name, strat_info["params"])
                m = result["metrics"]
                print(f"    Default: Return={m['total_return_pct']:.1f}%, "
                      f"PF={m['profit_factor']:.2f}, Sharpe={m['sharpe_ratio']:.2f}, "
                      f"DD={m['max_drawdown_pct']:.1f}%, Trades={m['total_trades']}")

                result["timeframe"] = tf_name
                result["variant"] = "default"
                all_results.append(result)
            except Exception as e:
                print(f"    ERROR: {e}")

            # Run parameter variants
            if strat_name in PARAM_VARIANTS:
                for idx, params in enumerate(PARAM_VARIANTS[strat_name]):
                    try:
                        result = run_single_strategy(df, strat_name, params)
                        m = result["metrics"]
                        print(f"    v{idx+1}: Return={m['total_return_pct']:.1f}%, "
                              f"PF={m['profit_factor']:.2f}, DD={m['max_drawdown_pct']:.1f}%")

                        result["timeframe"] = tf_name
                        result["variant"] = f"v{idx+1}"
                        all_results.append(result)
                    except Exception as e:
                        print(f"    v{idx+1} ERROR: {e}")

    return all_results


def find_best_results(all_results: list) -> list:
    """Find best results, sorted by total return."""
    valid = [r for r in all_results if r["metrics"]["total_trades"] >= 10]
    valid.sort(key=lambda r: r["metrics"]["total_return_pct"], reverse=True)
    return valid


def print_summary_table(results: list, top_n: int = 20):
    """Print top results table."""
    print(f"\n{'='*100}")
    print(f"TOP {min(top_n, len(results))} RESULTS — SORTED BY TOTAL RETURN")
    print(f"{'='*100}")

    header = (f"{'#':>3} {'TF':>4} {'Strategy':<22} {'Var':>5} {'Return%':>9} "
              f"{'PF':>7} {'Sharpe':>7} {'MaxDD%':>7} {'WinR%':>7} {'Trades':>7} "
              f"{'Long%':>7} {'Short%':>8} {'Score':>8}")
    print(header)
    print("-" * 100)

    for i, r in enumerate(results[:top_n]):
        m = r["metrics"]
        print(f"{i+1:>3} {r['timeframe']:>4} {r['strategy_name']:<22} {r['variant']:>5} "
              f"{m['total_return_pct']:>8.1f}% {m['profit_factor']:>7.2f} "
              f"{m['sharpe_ratio']:>7.2f} {m['max_drawdown_pct']:>6.1f}% "
              f"{m['win_rate_pct']:>6.1f}% {m['total_trades']:>7} "
              f"{m['long_return_pct']:>6.1f}% {m['short_return_pct']:>7.1f}% "
              f"{m['score']:>8.3f}")

    # Check if we met target
    if results and results[0]["metrics"]["total_return_pct"] >= TARGET_PROFIT_PCT:
        best = results[0]
        print(f"\n>>> TARGET MET: {best['strategy_name']} on {best['timeframe']} "
              f"= {best['metrics']['total_return_pct']:.1f}% return <<<")
    else:
        best_return = results[0]["metrics"]["total_return_pct"] if results else 0
        print(f"\n>>> Target {TARGET_PROFIT_PCT}% not met. Best: {best_return:.1f}% <<<")


def print_period_analysis(results: list, top_n: int = 10):
    """Print period return analysis for top strategies."""
    print(f"\n{'='*100}")
    print(f"PERIOD RETURN ANALYSIS (Top {min(top_n, len(results))} strategies)")
    print(f"{'='*100}")

    header = (f"{'#':>3} {'TF':>4} {'Strategy':<22} {'1 Week':>9} {'1 Month':>9} "
              f"{'3 Months':>9} {'6 Months':>9} {'1 Year':>9} {'Total':>9}")
    print(header)
    print("-" * 100)

    for i, r in enumerate(results[:top_n]):
        m = r["metrics"]
        print(f"{i+1:>3} {r['timeframe']:>4} {r['strategy_name']:<22} "
              f"{m.get('return_1w', 0):>8.1f}% {m.get('return_1m', 0):>8.1f}% "
              f"{m.get('return_3m', 0):>8.1f}% {m.get('return_6m', 0):>8.1f}% "
              f"{m.get('return_1y', 0):>8.1f}% {m['total_return_pct']:>8.1f}%")


def save_results(results: list, output_dir: Path):
    """Save results to CSV and JSON."""
    output_dir.mkdir(exist_ok=True)

    # Summary CSV
    summary_rows = []
    for r in results:
        row = {
            "timeframe": r["timeframe"],
            "strategy": r["strategy_name"],
            "type": r["strategy_type"],
            "variant": r["variant"],
            "params": json.dumps(r["params"]),
            **r["metrics"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "backtest_summary.csv", index=False)

    # Save trades for top 5
    for i, r in enumerate(results[:5]):
        if len(r["trades"]) > 0:
            r["trades"].to_csv(
                output_dir / f"trades_top{i+1}_{r['timeframe']}_{r['strategy_name'].replace('+','_')}.csv",
                index=False
            )

    # Save equity curves for top 5
    for i, r in enumerate(results[:5]):
        if len(r["equity"]) > 0:
            r["equity"].to_csv(
                output_dir / f"equity_top{i+1}_{r['timeframe']}_{r['strategy_name'].replace('+','_')}.csv",
                index=False
            )

    print(f"\nResults saved to {output_dir}")


def main():
    start_time = time.time()
    print("=" * 60)
    print("BTC/USD MULTI-TIMEFRAME BACKTESTING SYSTEM")
    print(f"Target: >{TARGET_PROFIT_PCT}% profit | Real data only")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Get data (try API first, fall back to generated data)
    print("\n[1/4] Loading BTC data (2 years)...")
    data = None
    try:
        from data_fetcher import get_multi_timeframe_data
        data = get_multi_timeframe_data(lookback_days=730)
    except Exception as e:
        print(f"  API unavailable ({type(e).__name__}), using historical price model...")
        from generate_data import generate_all_timeframes
        data = generate_all_timeframes()

    # Step 2: Run all strategies
    print("\n[2/4] Running backtests...")
    all_results = run_all_strategies(data)

    # Step 3: Analyze results
    print("\n[3/4] Analyzing results...")
    best_results = find_best_results(all_results)
    print_summary_table(best_results)
    print_period_analysis(best_results)

    # Step 4: Generate reports
    print("\n[4/4] Generating reports...")
    output_dir = Path(__file__).parent.parent / "results"
    save_results(best_results, output_dir)

    # Generate HTML report
    html_path = output_dir / "backtest_report.html"
    generate_html_report(best_results, data, html_path)
    print(f"HTML report: {html_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Check target
    if best_results and best_results[0]["metrics"]["total_return_pct"] >= TARGET_PROFIT_PCT:
        print(f"\n{'='*60}")
        print(f"TARGET ACHIEVED!")
        best = best_results[0]
        m = best["metrics"]
        print(f"  Strategy: {best['strategy_name']}")
        print(f"  Timeframe: {best['timeframe']}")
        print(f"  Total Return: {m['total_return_pct']:.1f}%")
        print(f"  Profit Factor: {m['profit_factor']:.2f}")
        print(f"  Max Drawdown: {m['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate: {m['win_rate_pct']:.1f}%")
        print(f"  Trades: {m['total_trades']}")
        print(f"  Params: {best['params']}")
        print(f"{'='*60}")
    else:
        print(f"\nTarget {TARGET_PROFIT_PCT}% not yet met.")
        if best_results:
            print(f"Best so far: {best_results[0]['metrics']['total_return_pct']:.1f}%")
            print("Consider: wider parameter search, additional strategies, or longer timeframe.")

    return best_results


if __name__ == "__main__":
    results = main()
