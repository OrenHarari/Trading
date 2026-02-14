"""
Backtesting Engine — runs strategies on OHLCV data with realistic execution.

Features:
- Next-bar execution (no lookahead)
- Commission + slippage
- Long + Short support
- ATR-based stop losses
- Per-trade tracking
- Comprehensive metrics
"""

import numpy as np
import pandas as pd
from typing import Callable


# ============================================================
# COST MODEL
# ============================================================
COMMISSION_PCT = 0.05  # 0.05% per side
SLIPPAGE_PCT = 0.05    # 0.05% per side
TOTAL_COST_PER_SIDE = COMMISSION_PCT + SLIPPAGE_PCT  # 0.10%
INITIAL_CAPITAL = 100_000


def run_backtest(df: pd.DataFrame, signals: pd.DataFrame,
                 initial_capital: float = INITIAL_CAPITAL,
                 commission_pct: float = TOTAL_COST_PER_SIDE,
                 position_size_pct: float = 95.0,
                 use_stop_loss: bool = True) -> dict:
    """Run a realistic backtest on signal data.

    Args:
        df: OHLCV DataFrame
        signals: DataFrame with 'signal' column (1=long, -1=short, 0=flat)
        initial_capital: Starting equity
        commission_pct: Cost per side as percentage
        position_size_pct: Percentage of equity per trade
        use_stop_loss: Whether to apply ATR stop losses

    Returns:
        dict with equity curve, trades, and metrics
    """
    equity = initial_capital
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_bar = 0
    stop_loss = 0.0

    equity_curve = []
    trades = []
    positions = []

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    sigs = signals["signal"].values
    stops = signals["stop_loss"].values if "stop_loss" in signals.columns else np.full(len(df), np.nan)

    for i in range(1, len(df)):
        current_close = closes[i]
        prev_signal = sigs[i]  # Signal computed from past data

        # Check stop loss first (intrabar)
        stopped_out = False
        if use_stop_loss and position != 0 and not np.isnan(stop_loss):
            if position == 1 and lows[i] <= stop_loss:
                # Long stopped out
                exit_price = stop_loss
                cost = abs(exit_price) * commission_pct / 100.0
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                size = equity * position_size_pct / 100.0
                pnl_dollar = size * pnl_pct / 100.0 - cost * 2 * size / entry_price
                equity += pnl_dollar

                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "direction": "LONG",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct - 2 * commission_pct, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "bars_held": i - entry_bar,
                    "exit_reason": "stop_loss",
                })
                position = 0
                stopped_out = True

            elif position == -1 and highs[i] >= stop_loss:
                exit_price = stop_loss
                cost = abs(exit_price) * commission_pct / 100.0
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
                size = equity * position_size_pct / 100.0
                pnl_dollar = size * pnl_pct / 100.0 - cost * 2 * size / entry_price
                equity += pnl_dollar

                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "direction": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct - 2 * commission_pct, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "bars_held": i - entry_bar,
                    "exit_reason": "stop_loss",
                })
                position = 0
                stopped_out = True

        # Position changes (at close, executing next bar open simulated by close)
        if not stopped_out:
            new_signal = prev_signal

            # Close existing position if signal changes
            if position != 0 and new_signal != position:
                exit_price = current_close
                if position == 1:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

                size = equity * position_size_pct / 100.0
                pnl_dollar = size * pnl_pct / 100.0 - abs(size) * 2 * commission_pct / 100.0

                equity += pnl_dollar

                trades.append({
                    "entry_date": df.index[entry_bar],
                    "exit_date": df.index[i],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(pnl_pct - 2 * commission_pct, 4),
                    "pnl_dollar": round(pnl_dollar, 2),
                    "bars_held": i - entry_bar,
                    "exit_reason": "signal",
                })
                position = 0

            # Open new position
            if position == 0 and new_signal != 0:
                position = new_signal
                entry_price = current_close
                entry_bar = i
                if not np.isnan(stops[i]):
                    stop_loss = stops[i]
                else:
                    stop_loss = np.nan

        # Track equity
        if position == 1:
            unrealized = (current_close - entry_price) / entry_price * equity * position_size_pct / 100.0
        elif position == -1:
            unrealized = (entry_price - current_close) / entry_price * equity * position_size_pct / 100.0
        else:
            unrealized = 0

        equity_curve.append({
            "datetime": df.index[i],
            "equity": equity + unrealized,
            "position": position,
            "close": current_close,
        })
        positions.append(position)

    # Close any remaining position
    if position != 0:
        exit_price = closes[-1]
        if position == 1:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
        size = equity * position_size_pct / 100.0
        pnl_dollar = size * pnl_pct / 100.0 - abs(size) * 2 * commission_pct / 100.0
        equity += pnl_dollar

        trades.append({
            "entry_date": df.index[entry_bar],
            "exit_date": df.index[-1],
            "direction": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct - 2 * commission_pct, 4),
            "pnl_dollar": round(pnl_dollar, 2),
            "bars_held": len(df) - 1 - entry_bar,
            "exit_reason": "end_of_data",
        })

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(trades_df, equity_df, initial_capital)

    return {
        "equity": equity_df,
        "trades": trades_df,
        "metrics": metrics,
        "final_equity": equity,
    }


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                    initial_capital: float) -> dict:
    """Compute comprehensive performance metrics."""
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "total_return_pct": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0, "win_rate_pct": 0,
            "avg_trade_pct": 0, "max_consecutive_losses": 0, "score": 0,
            "long_trades": 0, "short_trades": 0, "long_return_pct": 0,
            "short_return_pct": 0,
        }

    pnl = trades_df["pnl_pct"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001

    pf = gross_profit / max(gross_loss, 0.001)

    # Equity curve metrics
    if len(equity_df) > 0:
        eq = equity_df["equity"].values
        running_max = np.maximum.accumulate(eq)
        drawdown_pct = (running_max - eq) / running_max * 100
        max_dd = drawdown_pct.max()

        # Returns for Sharpe
        returns = pd.Series(eq).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365)
        else:
            sharpe = 0
    else:
        max_dd = 0
        sharpe = 0

    total_return = (equity_df["equity"].iloc[-1] / initial_capital - 1) * 100 if len(equity_df) > 0 else 0

    # Consecutive losses
    max_consec = 0
    current_streak = 0
    for p in pnl:
        if p <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    # Long/Short breakdown
    long_trades = trades_df[trades_df["direction"] == "LONG"]
    short_trades = trades_df[trades_df["direction"] == "SHORT"]

    score = (pf * sharpe) / max(max_dd, 0.01) if max_dd > 0 else 0

    # Period returns
    period_returns = compute_period_returns(equity_df, initial_capital)

    return {
        "total_trades": len(pnl),
        "total_return_pct": round(total_return, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(len(wins) / len(pnl) * 100, 1),
        "avg_trade_pct": round(pnl.mean(), 4),
        "avg_win_pct": round(wins.mean(), 4) if len(wins) > 0 else 0,
        "avg_loss_pct": round(losses.mean(), 4) if len(losses) > 0 else 0,
        "max_consecutive_losses": max_consec,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_return_pct": round(long_trades["pnl_pct"].sum(), 2) if len(long_trades) > 0 else 0,
        "short_return_pct": round(short_trades["pnl_pct"].sum(), 2) if len(short_trades) > 0 else 0,
        "score": round(score, 4),
        **period_returns,
    }


def compute_period_returns(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """Compute returns over different time periods (from end of data backwards)."""
    if len(equity_df) == 0:
        return {"return_1w": 0, "return_1m": 0, "return_3m": 0, "return_6m": 0, "return_1y": 0}

    eq = equity_df.set_index("datetime")["equity"]
    final = eq.iloc[-1]
    end_date = eq.index[-1]

    periods = {
        "return_1w": 7,
        "return_1m": 30,
        "return_3m": 90,
        "return_6m": 180,
        "return_1y": 365,
    }

    results = {}
    for key, days in periods.items():
        target_date = end_date - pd.Timedelta(days=days)
        # Find nearest date
        mask = eq.index <= target_date
        if mask.any():
            past_equity = eq[mask].iloc[-1]
            results[key] = round((final / past_equity - 1) * 100, 2)
        else:
            # Use initial capital if period extends before data start
            results[key] = round((final / initial_capital - 1) * 100, 2)

    return results
