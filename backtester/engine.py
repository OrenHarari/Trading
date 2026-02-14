"""
Backtesting engine for BTC/USD strategies.

Simulates strategy execution on historical data with realistic
position sizing, commissions, slippage, and risk management.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    """Represents a completed round-trip trade."""
    trade_id: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: str  # "long" or "short"
    strategy: str   # "T1", "M1", "H1"
    entry_price: float
    exit_price: float
    size_pct: float  # position size as % of equity
    profit_pct: float
    profit_usd: float
    bars_held: int
    exit_reason: str


@dataclass
class Position:
    """Active position state."""
    direction: str
    strategy: str
    entry_price: float
    entry_bar: int
    entry_date: pd.Timestamp
    size_pct: float
    highest_since: float
    lowest_since: float


@dataclass
class BacktestConfig:
    """Backtest configuration matching Pine Script settings."""
    initial_capital: float = 100_000
    commission_pct: float = 0.05   # per side
    slippage_pct: float = 0.05     # estimated slippage
    warmup_bars: int = 400
    pyramiding: int = 0            # 0 = one position at a time

    # Position sizing
    target_annual_vol: float = 30.0
    short_size_mult: float = 0.5
    mr_size_mult: float = 0.7

    # Timeframe
    bars_per_day: int = 1          # 1=daily, 6=4h, 24=1h

    # Risk management
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 2.5
    max_bars_trend: int = 60
    mr_stop_mult: float = 1.5
    mr_time_exit: int = 10
    squeeze_stop_mult: float = 2.0
    squeeze_trail_mult: float = 3.0
    squeeze_max_bars: int = 30


class BacktestEngine:
    """Core backtesting engine."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.equity = self.config.initial_capital
        self.peak_equity = self.equity
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.trade_counter = 0

    def reset(self):
        """Reset engine state for a new backtest."""
        self.equity = self.config.initial_capital
        self.peak_equity = self.equity
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.trade_counter = 0

    def _apply_costs(self, price: float, direction: str, is_entry: bool) -> float:
        """Apply commission and slippage to get execution price."""
        cost = self.config.commission_pct / 100 + self.config.slippage_pct / 100
        if (direction == "long" and is_entry) or (direction == "short" and not is_entry):
            return price * (1 + cost)  # buying: pay more
        else:
            return price * (1 - cost)  # selling: receive less

    def enter_position(self, bar_idx: int, date: pd.Timestamp, price: float,
                       direction: str, strategy: str, size_pct: float):
        """Open a new position."""
        if self.position is not None:
            return  # already in position (no pyramiding)

        exec_price = self._apply_costs(price, direction, is_entry=True)

        self.position = Position(
            direction=direction,
            strategy=strategy,
            entry_price=exec_price,
            entry_bar=bar_idx,
            entry_date=date,
            size_pct=size_pct,
            highest_since=price,
            lowest_since=price,
        )

    def exit_position(self, bar_idx: int, date: pd.Timestamp, price: float, reason: str):
        """Close an active position and record the trade."""
        if self.position is None:
            return

        pos = self.position
        exec_price = self._apply_costs(price, pos.direction, is_entry=False)

        # Calculate P&L
        if pos.direction == "long":
            profit_pct = (exec_price / pos.entry_price - 1) * 100
        else:
            profit_pct = (pos.entry_price / exec_price - 1) * 100

        # Apply position sizing to equity impact
        equity_impact_pct = profit_pct * (pos.size_pct / 100)
        profit_usd = self.equity * equity_impact_pct / 100

        self.equity += profit_usd
        self.peak_equity = max(self.peak_equity, self.equity)

        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            entry_date=pos.entry_date,
            exit_date=date,
            direction=pos.direction,
            strategy=pos.strategy,
            entry_price=pos.entry_price,
            exit_price=exec_price,
            size_pct=pos.size_pct,
            profit_pct=round(profit_pct, 4),
            profit_usd=round(profit_usd, 2),
            bars_held=bar_idx - pos.entry_bar,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None

    def update_position_tracking(self, high: float, low: float):
        """Update highest/lowest since entry for trailing stops."""
        if self.position:
            self.position.highest_since = max(self.position.highest_since, high)
            self.position.lowest_since = min(self.position.lowest_since, low)

    def record_equity(self, date: pd.Timestamp, close: float):
        """Record daily equity point."""
        # Mark-to-market if in position
        mtm_equity = self.equity
        if self.position:
            pos = self.position
            if pos.direction == "long":
                unrealized_pct = (close / pos.entry_price - 1) * 100
            else:
                unrealized_pct = (pos.entry_price / close - 1) * 100
            equity_impact = unrealized_pct * (pos.size_pct / 100)
            mtm_equity = self.equity * (1 + equity_impact / 100)

        self.equity_curve.append({
            "date": date,
            "equity": round(mtm_equity, 2),
            "close": close,
            "in_position": self.position is not None,
            "drawdown_pct": round((1 - mtm_equity / self.peak_equity) * 100, 2) if self.peak_equity > 0 else 0,
        })

    def get_results(self) -> dict:
        """Compute comprehensive backtest results."""
        if not self.trades:
            return {"error": "No trades executed"}

        pnl = np.array([t.profit_pct for t in self.trades])
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic stats
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001

        # Profit Factor
        pf = gross_profit / gross_loss

        # Sharpe (annualized)
        bars_per_year = 365 * self.config.bars_per_day
        trades_per_year = max(1, len(pnl) / max(1, (len(self.equity_curve) / bars_per_year)))
        sharpe = (np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year)) if np.std(pnl) > 0 else 0

        # Max Drawdown from equity curve
        if len(equity_df) > 0:
            eq = equity_df["equity"].values
            running_max = np.maximum.accumulate(eq)
            dd_pct = (running_max - eq) / running_max * 100
            max_dd = dd_pct.max()
        else:
            max_dd = 0

        # Primary Score
        score = (pf * sharpe) / max_dd if max_dd > 0 else 0

        # CAGR
        if len(equity_df) > 1:
            days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
            years = days / 365.25
            final_eq = equity_df["equity"].iloc[-1]
            cagr = ((final_eq / self.config.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        else:
            cagr = 0

        # Consecutive losses
        max_consec_losses = 0
        current_consec = 0
        for p in pnl:
            if p <= 0:
                current_consec += 1
                max_consec_losses = max(max_consec_losses, current_consec)
            else:
                current_consec = 0

        # Exposure
        if len(equity_df) > 0:
            exposure = equity_df["in_position"].mean() * 100
        else:
            exposure = 0

        # Strategy breakdown
        strategy_stats = {}
        for strat_name in set(t.strategy for t in self.trades):
            strat_trades = [t for t in self.trades if t.strategy == strat_name]
            strat_pnl = np.array([t.profit_pct for t in strat_trades])
            strat_wins = strat_pnl[strat_pnl > 0]
            strat_losses = strat_pnl[strat_pnl <= 0]
            strategy_stats[strat_name] = {
                "trades": len(strat_trades),
                "win_rate": round(len(strat_wins) / len(strat_trades) * 100, 1) if len(strat_trades) > 0 else 0,
                "avg_profit": round(np.mean(strat_pnl), 2),
                "total_profit": round(strat_pnl.sum(), 2),
                "avg_bars_held": round(np.mean([t.bars_held for t in strat_trades]), 1),
            }

        return {
            "total_trades": len(self.trades),
            "win_rate": round(len(wins) / len(pnl) * 100, 1),
            "profit_factor": round(pf, 3),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "primary_score": round(score, 4),
            "cagr_pct": round(cagr, 2),
            "avg_trade_pct": round(np.mean(pnl), 3),
            "best_trade_pct": round(pnl.max(), 2),
            "worst_trade_pct": round(pnl.min(), 2),
            "max_consecutive_losses": max_consec_losses,
            "exposure_pct": round(exposure, 1),
            "final_equity": round(equity_df["equity"].iloc[-1], 2) if len(equity_df) > 0 else self.config.initial_capital,
            "total_return_pct": round((equity_df["equity"].iloc[-1] / self.config.initial_capital - 1) * 100, 2) if len(equity_df) > 0 else 0,
            "strategy_breakdown": strategy_stats,
            "long_trades": len([t for t in self.trades if t.direction == "long"]),
            "short_trades": len([t for t in self.trades if t.direction == "short"]),
        }

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "trade_id": t.trade_id,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "direction": t.direction,
            "strategy": t.strategy,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_pct": t.size_pct,
            "profit_pct": t.profit_pct,
            "profit_usd": t.profit_usd,
            "bars_held": t.bars_held,
            "exit_reason": t.exit_reason,
        } for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)
