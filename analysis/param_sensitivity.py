"""
Parameter Sensitivity Analysis.

Tests how strategy performance changes when key parameters are varied
by +/-20% and +/-40%. Identifies fragile ("knife-edge") parameters.

Usage:
    python param_sensitivity.py --input ../results/trades_parsed.csv

Note: This script analyzes a SINGLE trade set. For true parameter sensitivity,
you need to re-run the strategy in TradingView with different parameter values
and export each trade list separately.

This script provides the analysis framework — you feed it multiple CSV files
from different parameter runs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Default parameter grid (base values from the strategy)
DEFAULT_PARAMS = {
    "ema_fast": {"base": 21, "variations": [13, 17, 21, 25, 29]},
    "ema_slow": {"base": 55, "variations": [33, 44, 55, 66, 77]},
    "adx_threshold": {"base": 22, "variations": [13, 18, 22, 26, 31]},
    "atr_stop_mult": {"base": 2.0, "variations": [1.2, 1.6, 2.0, 2.4, 2.8]},
    "atr_trail_mult": {"base": 2.5, "variations": [1.5, 2.0, 2.5, 3.0, 3.5]},
    "bb_mult": {"base": 2.0, "variations": [1.2, 1.6, 2.0, 2.4, 2.8]},
    "rsi_oversold": {"base": 30, "variations": [18, 24, 30, 36, 42]},
    "rsi_overbought": {"base": 70, "variations": [58, 64, 70, 76, 82]},
    "vats_trend_th": {"base": 1.5, "variations": [0.9, 1.2, 1.5, 1.8, 2.1]},
    "regime_persist": {"base": 3, "variations": [1, 2, 3, 4, 5]},
}


def compute_score(pnl: np.ndarray) -> dict:
    """Compute the primary score and supporting metrics."""
    if len(pnl) == 0:
        return {"score": 0, "pf": 0, "sharpe": 0, "max_dd": 0, "trades": 0}

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001

    pf = gross_profit / gross_loss
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    max_dd = (running_max - equity).max()

    sharpe = (np.mean(pnl) / np.std(pnl) * np.sqrt(73)) if np.std(pnl) > 0 else 0

    score = (pf * sharpe) / max_dd if max_dd > 0 else 0

    return {
        "score": round(score, 4),
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "trades": len(pnl),
    }


def rate_sensitivity(base_score: float, varied_scores: list) -> str:
    """Rate parameter sensitivity based on score variation at +/-20%.

    Low: All within 15% of base
    Medium: 15-30% from base
    High: >30% from base (fragile)
    """
    if base_score == 0:
        return "N/A"

    # The middle 3 values in a 5-element list are -20%, base, +20%
    pct_20_scores = varied_scores[1:4]  # -20%, base, +20%

    max_deviation = max(abs(s - base_score) / base_score for s in pct_20_scores)

    if max_deviation <= 0.15:
        return "Low"
    elif max_deviation <= 0.30:
        return "Medium"
    else:
        return "High"


def analyze_from_multiple_files(results_dir: str):
    """Analyze parameter sensitivity from multiple trade export files.

    Expected file naming: trades_param_{paramname}_{value}.csv
    """
    results_path = Path(results_dir)
    csv_files = sorted(results_path.glob("trades_param_*.csv"))

    if not csv_files:
        print("No parameter variation files found.")
        print("Expected files like: trades_param_ema_fast_13.csv")
        print("\nTo generate these, re-run the strategy in TradingView")
        print("with different parameter values and export each trade list.")
        return

    results = {}
    for f in csv_files:
        # Parse filename: trades_param_{name}_{value}.csv
        parts = f.stem.replace("trades_param_", "").rsplit("_", 1)
        if len(parts) == 2:
            param_name, param_val = parts
            df = pd.read_csv(f)
            if "profit_pct" in df.columns:
                pnl = df["profit_pct"].dropna().values
                metrics = compute_score(pnl)
                if param_name not in results:
                    results[param_name] = []
                results[param_name].append(
                    {"value": param_val, "score": metrics["score"], **metrics}
                )

    print_sensitivity_results(results)


def print_sensitivity_results(results: dict):
    """Print sensitivity analysis table."""
    print("\n" + "=" * 100)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)

    header = f"{'Parameter':<20} {'−40%':>10} {'−20%':>10} {'Base':>10} {'+20%':>10} {'+40%':>10} {'Rating':>10}"
    print(header)
    print("-" * 100)

    for param_name, entries in results.items():
        entries = sorted(entries, key=lambda x: float(x["value"]))
        scores = [e["score"] for e in entries]

        if len(scores) >= 5:
            base_score = scores[2]
            rating = rate_sensitivity(base_score, scores)
            print(
                f"{param_name:<20} {scores[0]:>10.4f} {scores[1]:>10.4f} "
                f"{scores[2]:>10.4f} {scores[3]:>10.4f} {scores[4]:>10.4f} {rating:>10}"
            )

    print("=" * 100)
    print("\nSensitivity Rating:")
    print("  Low:    All ±20% variants within 15% of base score (robust)")
    print("  Medium: Some variants 15-30% from base (acceptable)")
    print("  High:   Any variant >30% from base (fragile — consider removing)")


def print_parameter_grid():
    """Print the expected parameter grid for manual testing."""
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY GRID — Test These Values")
    print("=" * 80)

    header = f"{'Parameter':<20} {'−40%':>8} {'−20%':>8} {'Base':>8} {'+20%':>8} {'+40%':>8}"
    print(header)
    print("-" * 80)

    for name, cfg in DEFAULT_PARAMS.items():
        vals = cfg["variations"]
        print(
            f"{name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8}"
        )

    print("-" * 80)
    print("\nInstructions:")
    print("1. For each parameter, change ONLY that parameter in TradingView")
    print("2. Export the trade list as CSV")
    print("3. Name it: trades_param_{paramname}_{value}.csv")
    print("4. Place in the results/ directory")
    print("5. Re-run this script with: --dir ../results/")


def main():
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Analysis")
    parser.add_argument("--input", default=None, help="Path to base trades CSV")
    parser.add_argument("--dir", default=None, help="Directory with parameter variation CSVs")
    parser.add_argument("--grid", action="store_true", help="Print parameter test grid")
    args = parser.parse_args()

    if args.grid:
        print_parameter_grid()
        return

    if args.dir:
        analyze_from_multiple_files(args.dir)
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found", file=sys.stderr)
            sys.exit(1)

        df = pd.read_csv(str(input_path))
        pnl = df["profit_pct"].dropna().values
        base_metrics = compute_score(pnl)

        print(f"\nBase strategy performance ({len(pnl)} trades):")
        print(f"  Score: {base_metrics['score']}")
        print(f"  PF:    {base_metrics['pf']}")
        print(f"  Sharpe:{base_metrics['sharpe']}")
        print(f"  MaxDD: {base_metrics['max_dd']}%")
        print()
        print_parameter_grid()
        return

    print("Usage: Provide --input, --dir, or --grid")
    print("  --grid:  Show parameter values to test")
    print("  --input: Analyze base strategy and show grid")
    print("  --dir:   Analyze multiple parameter variation CSVs")


if __name__ == "__main__":
    main()
