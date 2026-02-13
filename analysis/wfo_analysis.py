"""
Walk-Forward Optimization (WFO) Analysis.

Splits trade sequence into rolling IS/OOS windows and evaluates
consistency of strategy performance across folds.

Usage:
    python wfo_analysis.py --input ../results/trades_parsed.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compute_window_metrics(pnl: np.ndarray) -> dict:
    """Compute performance metrics for a window of trades."""
    if len(pnl) == 0:
        return {
            "trades": 0,
            "total_return": 0,
            "profit_factor": 0,
            "sharpe": 0,
            "max_dd": 0,
            "score": 0,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else (9999 if gross_profit > 0 else 0)

    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    trades_per_year = 73  # approximate for daily BTC
    sharpe = (np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year)) if np.std(pnl) > 0 else 0

    score = (pf * sharpe) / max_dd if max_dd > 0 else (pf * sharpe if pf * sharpe > 0 else 0)

    return {
        "trades": len(pnl),
        "total_return": round(equity[-1], 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "score": round(score, 4),
    }


def run_walk_forward(
    pnl: np.ndarray,
    is_size: int = 60,
    oos_size: int = 20,
    step_size: int = 20,
) -> pd.DataFrame:
    """Run walk-forward analysis on trade P&L sequence.

    Args:
        pnl: Array of trade P&L percentages in chronological order
        is_size: Number of trades per in-sample window
        oos_size: Number of trades per out-of-sample window
        step_size: Number of trades to step forward each fold
    """
    n = len(pnl)
    folds = []
    fold_num = 1

    start = 0
    while start + is_size + oos_size <= n:
        is_pnl = pnl[start : start + is_size]
        oos_pnl = pnl[start + is_size : start + is_size + oos_size]

        is_metrics = compute_window_metrics(is_pnl)
        oos_metrics = compute_window_metrics(oos_pnl)

        # Compute OOS/IS ratio for score
        ratio = oos_metrics["score"] / is_metrics["score"] if is_metrics["score"] > 0 else 0

        folds.append(
            {
                "fold": fold_num,
                "is_start_trade": start + 1,
                "is_end_trade": start + is_size,
                "oos_start_trade": start + is_size + 1,
                "oos_end_trade": start + is_size + oos_size,
                "is_trades": is_metrics["trades"],
                "is_return": is_metrics["total_return"],
                "is_pf": is_metrics["profit_factor"],
                "is_sharpe": is_metrics["sharpe"],
                "is_max_dd": is_metrics["max_dd"],
                "is_score": is_metrics["score"],
                "oos_trades": oos_metrics["trades"],
                "oos_return": oos_metrics["total_return"],
                "oos_pf": oos_metrics["profit_factor"],
                "oos_sharpe": oos_metrics["sharpe"],
                "oos_max_dd": oos_metrics["max_dd"],
                "oos_score": oos_metrics["score"],
                "oos_is_ratio": round(ratio, 3),
            }
        )

        start += step_size
        fold_num += 1

    return pd.DataFrame(folds)


def print_wfo_results(df: pd.DataFrame):
    """Print WFO results with pass/fail assessment."""
    print("\n" + "=" * 100)
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 100)

    # Summary header
    header = (
        f"{'Fold':>4} | {'IS Trades':>9} | {'IS Return':>10} | {'IS Score':>9} | "
        f"{'OOS Trades':>10} | {'OOS Return':>11} | {'OOS Score':>10} | "
        f"{'OOS/IS':>7} | {'Pass?':>6}"
    )
    print(header)
    print("-" * 100)

    fail_count = 0
    for _, row in df.iterrows():
        # Pass criteria: OOS Score >= 60% of IS Score AND not (negative return + >30% DD)
        ratio_pass = row["oos_is_ratio"] >= 0.6
        dd_pass = not (row["oos_return"] < 0 and row["oos_max_dd"] > 30)
        passed = ratio_pass and dd_pass

        if not passed:
            fail_count += 1

        status = "PASS" if passed else "FAIL"
        print(
            f"{int(row['fold']):>4} | {int(row['is_trades']):>9} | {row['is_return']:>10.2f} | "
            f"{row['is_score']:>9.4f} | {int(row['oos_trades']):>10} | {row['oos_return']:>11.2f} | "
            f"{row['oos_score']:>10.4f} | {row['oos_is_ratio']:>7.3f} | {status:>6}"
        )

    print("-" * 100)

    # Averages
    avg_is_score = df["is_score"].mean()
    avg_oos_score = df["oos_score"].mean()
    avg_ratio = df["oos_is_ratio"].mean()
    n_folds = len(df)

    print(f"\nSummary:")
    print(f"  Total folds:        {n_folds}")
    print(f"  Average IS Score:   {avg_is_score:.4f}")
    print(f"  Average OOS Score:  {avg_oos_score:.4f}")
    print(f"  Average OOS/IS:     {avg_ratio:.3f}")
    print(f"  Failed folds:       {fail_count} / {n_folds}")

    # Overall pass/fail
    overall_ratio_pass = avg_oos_score >= 0.6 * avg_is_score
    overall_fold_pass = fail_count <= 3

    print(f"\n  OOS avg >= 60% IS avg: {'PASS' if overall_ratio_pass else 'FAIL'}")
    print(f"  <= 3 failed folds:     {'PASS' if overall_fold_pass else 'FAIL'}")
    print(f"  OVERALL:               {'PASS' if overall_ratio_pass and overall_fold_pass else 'FAIL'}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization Analysis")
    parser.add_argument("--input", required=True, help="Path to parsed trades CSV")
    parser.add_argument("--is-size", type=int, default=60, help="IS window size (trades)")
    parser.add_argument("--oos-size", type=int, default=20, help="OOS window size (trades)")
    parser.add_argument("--step", type=int, default=20, help="Step size (trades)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(str(input_path))
    if "profit_pct" not in df.columns:
        print("Error: 'profit_pct' column not found", file=sys.stderr)
        sys.exit(1)

    pnl = df["profit_pct"].dropna().values
    print(f"Loaded {len(pnl)} trades from {input_path}")

    results = run_walk_forward(pnl, is_size=args.is_size, oos_size=args.oos_size, step_size=args.step)

    if len(results) == 0:
        print("Warning: Not enough trades for walk-forward analysis with current window sizes.")
        print(f"  Need at least {args.is_size + args.oos_size} trades, have {len(pnl)}")
        sys.exit(0)

    print_wfo_results(results)

    output_path = input_path.with_name("wfo_results.csv")
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
