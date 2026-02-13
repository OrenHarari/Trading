"""
Parse TradingView strategy tester CSV exports into a standardized format.

Usage:
    python trade_export_parser.py --input ../results/trades_raw.csv --output ../results/trades_parsed.csv

TradingView exports trades in a specific format. This script normalizes them
for downstream analysis (Monte Carlo, WFO, etc.).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_tradingview_csv(filepath: str) -> pd.DataFrame:
    """Parse a TradingView strategy tester trade list CSV export.

    TradingView exports columns like:
    Trade #, Type, Signal, Date/Time, Price, Contracts, Profit, Cum. Profit, Run-up, Drawdown

    Returns a DataFrame with one row per round-trip trade.
    """
    df = pd.read_csv(filepath)

    # Normalize column names (TradingView exports vary slightly)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if "trade" in cl and "#" in cl:
            col_map[col] = "trade_num"
        elif cl == "type":
            col_map[col] = "type"
        elif cl == "signal":
            col_map[col] = "signal"
        elif "date" in cl or "time" in cl:
            col_map[col] = "datetime"
        elif cl == "price":
            col_map[col] = "price"
        elif "contract" in cl or "qty" in cl:
            col_map[col] = "contracts"
        elif cl in ("profit", "profit %", "profit%"):
            col_map[col] = "profit_pct"
        elif "cum" in cl:
            col_map[col] = "cum_profit"
        elif "run" in cl and "up" in cl:
            col_map[col] = "runup"
        elif "draw" in cl:
            col_map[col] = "drawdown"

    df = df.rename(columns=col_map)

    # Clean numeric columns
    for col in ["profit_pct", "cum_profit", "runup", "drawdown", "price", "contracts"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "")
                .str.replace(",", "")
                .str.replace("$", "")
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


def build_roundtrip_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Build round-trip trade table from parsed export.

    If the export already has one row per trade with profit,
    we just clean and return. Otherwise, pair entries and exits.
    """
    if "profit_pct" in df.columns and df["profit_pct"].notna().sum() > 10:
        # Already has profit per trade row — filter to exit rows
        trades = df[df["profit_pct"].notna()].copy()
        trades = trades.reset_index(drop=True)
        trades["trade_id"] = range(1, len(trades) + 1)
        return trades

    # If we need to pair entry/exit, do so by trade_num
    if "trade_num" in df.columns and "type" in df.columns:
        entries = df[df["type"].str.lower().str.contains("entry", na=False)]
        exits = df[df["type"].str.lower().str.contains("exit", na=False)]

        trades = []
        for _, entry_row in entries.iterrows():
            tnum = entry_row.get("trade_num")
            exit_match = exits[exits["trade_num"] == tnum]
            if len(exit_match) > 0:
                exit_row = exit_match.iloc[0]
                trades.append(
                    {
                        "trade_id": tnum,
                        "entry_date": entry_row.get("datetime"),
                        "exit_date": exit_row.get("datetime"),
                        "entry_price": entry_row.get("price"),
                        "exit_price": exit_row.get("price"),
                        "direction": entry_row.get("signal", ""),
                        "profit_pct": exit_row.get("profit_pct", 0),
                        "contracts": entry_row.get("contracts", 0),
                    }
                )
        return pd.DataFrame(trades)

    print("Warning: Could not parse trade structure. Returning raw data.", file=sys.stderr)
    return df


def compute_metrics(trades: pd.DataFrame) -> dict:
    """Compute key performance metrics from a trade list."""
    pnl = trades["profit_pct"].dropna().values

    if len(pnl) == 0:
        return {"error": "No trades found"}

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0

    # Cumulative equity curve
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    # Consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnl:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Sharpe (annualized, assuming daily trades approximation)
    avg_trade_days = 5  # rough estimate for daily strategy
    trades_per_year = 365 / avg_trade_days
    sharpe = (np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year)) if np.std(pnl) > 0 else 0

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(pnl) * 100
    avg_trade = np.mean(pnl)
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    expectancy = avg_trade
    score = (pf * sharpe) / max_dd if max_dd > 0 else float("inf")

    return {
        "total_trades": len(pnl),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_trade_pct": round(avg_trade, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "expectancy_pct": round(expectancy, 4),
        "max_consecutive_losses": max_consec_loss,
        "total_return_pct": round(equity[-1], 2),
        "score_pf_x_sharpe_div_dd": round(score, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Parse TradingView trade exports")
    parser.add_argument("--input", required=True, help="Path to TradingView CSV export")
    parser.add_argument(
        "--output", default=None, help="Output path for parsed trades (default: auto)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(input_path.with_name("trades_parsed.csv"))

    print(f"Parsing: {input_path}")
    raw = parse_tradingview_csv(str(input_path))
    trades = build_roundtrip_trades(raw)
    trades.to_csv(output_path, index=False)
    print(f"Saved parsed trades to: {output_path}")
    print(f"Total trades: {len(trades)}")

    if "profit_pct" in trades.columns:
        metrics = compute_metrics(trades)
        print("\n--- Performance Metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
