"""
Monte Carlo simulation via bootstrap resampling of trade P&L.

Shuffles realized trades to estimate the distribution of:
- Total return
- Max drawdown
- Sharpe ratio
- Profit factor

Usage:
    python monte_carlo.py --input ../results/trades_parsed.csv --simulations 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_trades(filepath: str) -> np.ndarray:
    """Load trade P&L from parsed CSV."""
    df = pd.read_csv(filepath)
    if "profit_pct" not in df.columns:
        print("Error: 'profit_pct' column not found in CSV", file=sys.stderr)
        sys.exit(1)
    pnl = df["profit_pct"].dropna().values
    if len(pnl) < 20:
        print(f"Warning: Only {len(pnl)} trades found. Results may be unreliable.", file=sys.stderr)
    return pnl


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute max drawdown from cumulative equity curve."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    return drawdowns.max() if len(drawdowns) > 0 else 0.0


def compute_sharpe(pnl: np.ndarray, trades_per_year: float = 73.0) -> float:
    """Annualized Sharpe ratio."""
    if np.std(pnl) == 0:
        return 0.0
    return np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year)


def compute_profit_factor(pnl: np.ndarray) -> float:
    """Profit factor = gross profit / gross loss."""
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl <= 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def run_monte_carlo(pnl: np.ndarray, n_simulations: int = 1000, seed: int = 42) -> dict:
    """Run bootstrap resampling Monte Carlo on trade P&L."""
    rng = np.random.default_rng(seed)
    n_trades = len(pnl)

    results = {
        "total_return": np.zeros(n_simulations),
        "max_drawdown": np.zeros(n_simulations),
        "sharpe_ratio": np.zeros(n_simulations),
        "profit_factor": np.zeros(n_simulations),
    }

    for i in range(n_simulations):
        # Resample with replacement
        resampled = rng.choice(pnl, size=n_trades, replace=True)
        equity = np.cumsum(resampled)

        results["total_return"][i] = equity[-1]
        results["max_drawdown"][i] = compute_max_drawdown(equity)
        results["sharpe_ratio"][i] = compute_sharpe(resampled)
        results["profit_factor"][i] = compute_profit_factor(resampled)

    return results


def print_results(results: dict, pass_thresholds: dict):
    """Print Monte Carlo results with pass/fail assessment."""
    percentiles = [5, 25, 50, 75, 95]

    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 80)

    header = f"{'Metric':<20} {'Mean':>10} {'5th':>10} {'25th':>10} {'Median':>10} {'75th':>10} {'95th':>10} {'Pass?':>8}"
    print(header)
    print("-" * 80)

    for metric, values in results.items():
        pcts = np.percentile(values, percentiles)
        mean_val = np.mean(values)

        # Determine pass/fail
        threshold = pass_thresholds.get(metric)
        if threshold is not None:
            pctile_key, op, thresh_val = threshold
            test_val = pcts[percentiles.index(pctile_key)]
            passed = test_val > thresh_val if op == ">" else test_val <= thresh_val
            status = "PASS" if passed else "FAIL"
        else:
            status = "—"

        name = metric.replace("_", " ").title()
        print(
            f"{name:<20} {mean_val:>10.2f} {pcts[0]:>10.2f} {pcts[1]:>10.2f} "
            f"{pcts[2]:>10.2f} {pcts[3]:>10.2f} {pcts[4]:>10.2f} {status:>8}"
        )

    # Probability of negative return
    neg_pct = (results["total_return"] < 0).mean() * 100
    print(f"\nProbability of negative total return: {neg_pct:.1f}%")
    print(f"  {'PASS' if neg_pct < 10 else 'FAIL'}: Threshold is < 10%")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo trade P&L simulation")
    parser.add_argument("--input", required=True, help="Path to parsed trades CSV")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    pnl = load_trades(str(input_path))
    print(f"Loaded {len(pnl)} trades from {input_path}")
    print(f"Running {args.simulations} Monte Carlo simulations...")

    results = run_monte_carlo(pnl, n_simulations=args.simulations, seed=args.seed)

    # Pass/fail thresholds from Phase 5
    pass_thresholds = {
        "total_return": (5, ">", 0),  # 5th percentile return > 0%
        "max_drawdown": (95, "<=", 25),  # 95th percentile DD <= 25%
        "sharpe_ratio": (5, ">", 0.5),  # 5th percentile Sharpe > 0.5
        "profit_factor": (5, ">", 1.0),  # 5th percentile PF > 1.0
    }

    print_results(results, pass_thresholds)

    # Save detailed results
    output_path = input_path.with_name("monte_carlo_results.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
