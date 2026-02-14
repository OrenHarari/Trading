"""
Main runner script — executes all strategies on BTC data and generates results.

Usage:
    python run_backtest.py                          # Full run with all data
    python run_backtest.py --start 2020-01-01       # From specific date
    python run_backtest.py --start 2024-01-01 --end 2026-01-01  # Date range
    python run_backtest.py --strategies T1,M1       # Specific strategies only
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtester.data_loader import load_btc_data, get_data_summary
from backtester.strategies import STRATEGY_REGISTRY
from backtester.engine import BacktestConfig


def run_all(data_path: str, start_date: str = None, end_date: str = None,
            strategies: list = None, output_dir: str = "results"):
    """Run backtests for all strategies and save results."""

    # Load data
    print(f"\n{'='*70}")
    print("BTC/USD BACKTESTING SYSTEM")
    print(f"{'='*70}")

    df = load_btc_data(data_path, start_date, end_date)
    summary = get_data_summary(df)
    print(f"\nData: {summary['total_bars']} bars | {summary['start_date']} → {summary['end_date']}")
    print(f"Price: ${summary['start_price']:,.2f} → ${summary['end_price']:,.2f} ({summary['total_return_pct']:+.1f}%)")
    print(f"Range: ${summary['min_price']:,.2f} – ${summary['max_price']:,.2f}")

    # Setup output dir
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    # Config
    config = BacktestConfig()

    # Select strategies
    if strategies:
        strat_keys = [s.strip().upper() for s in strategies]
    else:
        strat_keys = list(STRATEGY_REGISTRY.keys())

    # Run each strategy
    all_results = {}
    all_equity = {}
    all_trades = {}

    for key in strat_keys:
        if key not in STRATEGY_REGISTRY:
            print(f"\n⚠ Unknown strategy: {key}")
            continue

        strat_class = STRATEGY_REGISTRY[key]
        strat = strat_class(config)

        print(f"\n{'─'*50}")
        print(f"Running: {strat.name}")
        print(f"{'─'*50}")

        results = strat.run(df)

        if "error" in results:
            print(f"  ⚠ {results['error']}")
            continue

        all_results[key] = results
        all_equity[key] = strat.get_equity_df()
        all_trades[key] = strat.get_trades_df()

        # Print summary
        print(f"  Trades:    {results['total_trades']} ({results['long_trades']}L / {results['short_trades']}S)")
        print(f"  Win Rate:  {results['win_rate']}%")
        print(f"  PF:        {results['profit_factor']}")
        print(f"  Sharpe:    {results['sharpe_ratio']}")
        print(f"  Max DD:    {results['max_drawdown_pct']}%")
        print(f"  Score:     {results['primary_score']}")
        print(f"  CAGR:      {results['cagr_pct']}%")
        print(f"  Return:    {results['total_return_pct']}%")
        print(f"  Final Eq:  ${results['final_equity']:,.2f}")

        if results.get("strategy_breakdown"):
            for sname, sstats in results["strategy_breakdown"].items():
                print(f"    [{sname}] {sstats['trades']} trades, "
                      f"WR {sstats['win_rate']}%, avg {sstats['avg_profit']}%")

        # Save trades CSV
        trades_df = strat.get_trades_df()
        if not trades_df.empty:
            trades_df.to_csv(out_path / f"trades_{key}.csv", index=False)

        # Save equity CSV
        eq_df = strat.get_equity_df()
        if not eq_df.empty:
            eq_df.to_csv(out_path / f"equity_{key}.csv", index=False)

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print(f"{'='*70}")
        header = f"{'Strategy':<25} {'Trades':>7} {'WR%':>6} {'PF':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Score':>8} {'Return%':>10}"
        print(header)
        print("─" * 85)

        # Sort by primary score descending
        sorted_results = sorted(all_results.items(), key=lambda x: x[1].get("primary_score", 0), reverse=True)
        for key, res in sorted_results:
            strat_name = STRATEGY_REGISTRY[key].name if key in STRATEGY_REGISTRY else key
            print(f"{strat_name:<25} {res['total_trades']:>7} {res['win_rate']:>6} "
                  f"{res['profit_factor']:>8.3f} {res['sharpe_ratio']:>8.3f} "
                  f"{res['max_drawdown_pct']:>8.2f} {res['primary_score']:>8.4f} "
                  f"{res['total_return_pct']:>10.2f}")

    # Save combined results JSON
    with open(out_path / "backtest_results.json", "w") as f:
        # Convert non-serializable items
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {kk: (vv if not isinstance(vv, float) or not (vv != vv) else None)
                               for kk, vv in v.items()}
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n✓ Results saved to {out_path}/")
    return all_results, all_equity, all_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC/USD Backtesting System")
    parser.add_argument("--data", default="btc-usd-max (1).csv", help="Path to BTC CSV data")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategies", default=None, help="Comma-separated strategy keys (T1,M1,H1,H2,BH)")
    parser.add_argument("--output", default="results", help="Output directory")

    args = parser.parse_args()

    strats = args.strategies.split(",") if args.strategies else None
    run_all(args.data, args.start, args.end, strats, args.output)
