"""
Strategy implementations matching Pine Script logic.

Each strategy class computes signals and manages entries/exits
using the BacktestEngine. Mirrors the logic in main_strategy.pine.
"""

import numpy as np
import pandas as pd
from . import indicators as ind
from .engine import BacktestEngine, BacktestConfig


class StrategyBase:
    """Base class for all strategies."""

    name = "Base"
    description = ""
    PARAMS = {}

    def __init__(self, config: BacktestConfig = None, **params):
        self.config = config or BacktestConfig()
        self.engine = BacktestEngine(self.config)
        self._params = params

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> dict:
        self.engine.reset()
        df = self.compute_indicators(df.copy())
        df = self.generate_signals(df)
        self._execute(df)
        return self.engine.get_results()

    def _execute(self, df: pd.DataFrame):
        raise NotImplementedError

    def get_trades_df(self) -> pd.DataFrame:
        return self.engine.get_trades_df()

    def get_equity_df(self) -> pd.DataFrame:
        return self.engine.get_equity_df()

    @classmethod
    def explainer(cls) -> str:
        """Return human-readable explanation of what this strategy does."""
        return cls.description


# ─────────────────────────────────────────────────────
# T1: EMA Crossover + ADX Trend Following
# ─────────────────────────────────────────────────────

class T1_TrendEMA(StrategyBase):
    """Dual EMA crossover with ADX filter — trend following strategy."""

    name = "T1 Trend (EMA+ADX)"
    description = "Dual EMA crossover + ADX trend-following"

    PARAMS = {
        "ema_fast":      {"default": 21,   "min": 5,    "max": 50,   "step": 1,   "label": "EMA Fast Period"},
        "ema_slow":      {"default": 55,   "min": 20,   "max": 120,  "step": 5,   "label": "EMA Slow Period"},
        "adx_threshold": {"default": 22.0, "min": 10.0, "max": 40.0, "step": 1.0, "label": "ADX Threshold"},
        "adx_len":       {"default": 14,   "min": 7,    "max": 30,   "step": 1,   "label": "ADX Period"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.ema_fast_len = params.get("ema_fast", 21)
        self.ema_slow_len = params.get("ema_slow", 55)
        self.adx_len = params.get("adx_len", 14)
        self.adx_threshold = params.get("adx_threshold", 22.0)
        self.atr_len = params.get("atr_len", 14)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = ind.ema(df["close"], self.ema_fast_len)
        df["ema_slow"] = ind.ema(df["close"], self.ema_slow_len)
        df["adx"] = ind.adx(df["high"], df["low"], df["close"], self.adx_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_f = df["ema_fast"]
        ema_s = df["ema_slow"]
        cross_above = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        cross_below = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
        adx_ok = df["adx"] > self.adx_threshold
        df["long_signal"] = cross_above & adx_ok
        df["short_signal"] = cross_below & adx_ok
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar
                long_stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                short_stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                long_trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                short_trail = pos.lowest_since + cfg.atr_trail_mult * atr_val

                if pos.direction == "long":
                    eff_stop = max(long_stop, long_trail)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "T1 Stop")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "T1 Time Exit")
                else:
                    eff_stop = min(short_stop, short_trail)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "T1 Stop")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "T1 Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "T1", size)
                elif row["short_signal"]:
                    self.engine.enter_position(i, date, close, "short", "T1", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("T1 Trend uses a dual EMA crossover (fast 21 / slow 55) confirmed by ADX > 22. "
                "It enters long when fast EMA crosses above slow EMA in a trending market, "
                "and short when fast crosses below. Exits via ATR-based trailing stops or time limit (60 bars).")


# ─────────────────────────────────────────────────────
# M1: Bollinger Band + RSI Mean Reversion
# ─────────────────────────────────────────────────────

class M1_MeanRevBBRSI(StrategyBase):
    """BB + RSI mean-reversion."""

    name = "M1 Mean-Rev (BB+RSI)"
    description = "Bollinger Band + RSI mean-reversion"

    PARAMS = {
        "bb_len":  {"default": 20,   "min": 10,   "max": 40,   "step": 1,   "label": "BB Period"},
        "bb_mult": {"default": 2.0,  "min": 1.0,  "max": 3.5,  "step": 0.1, "label": "BB Multiplier"},
        "rsi_len": {"default": 14,   "min": 5,    "max": 30,   "step": 1,   "label": "RSI Period"},
        "rsi_ob":  {"default": 70.0, "min": 60.0, "max": 85.0, "step": 1.0, "label": "RSI Overbought"},
        "rsi_os":  {"default": 30.0, "min": 15.0, "max": 40.0, "step": 1.0, "label": "RSI Oversold"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.bb_len = params.get("bb_len", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self.rsi_len = params.get("rsi_len", 14)
        self.rsi_ob = params.get("rsi_ob", 70.0)
        self.rsi_os = params.get("rsi_os", 30.0)
        self.atr_len = params.get("atr_len", 14)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"] = ind.bollinger_bands(
            df["close"], self.bb_len, self.bb_mult
        )
        df["rsi"] = ind.rsi(df["close"], self.rsi_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["long_signal"] = (df["close"] < df["bb_lower"]) & (df["rsi"] < self.rsi_os)
        df["short_signal"] = (df["close"] > df["bb_upper"]) & (df["rsi"] > self.rsi_ob)
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02
            bb_mid = row["bb_mid"] if not np.isnan(row["bb_mid"]) else close

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    mr_stop = pos.entry_price - cfg.mr_stop_mult * atr_val
                    if close >= bb_mid:
                        self.engine.exit_position(i, date, close, "M1 Target")
                    elif low <= mr_stop:
                        self.engine.exit_position(i, date, mr_stop, "M1 Stop")
                    elif bars_held >= cfg.mr_time_exit:
                        self.engine.exit_position(i, date, close, "M1 Time Exit")
                else:
                    mr_stop = pos.entry_price + cfg.mr_stop_mult * atr_val
                    if close <= bb_mid:
                        self.engine.exit_position(i, date, close, "M1 Target")
                    elif high >= mr_stop:
                        self.engine.exit_position(i, date, mr_stop, "M1 Stop")
                    elif bars_held >= cfg.mr_time_exit:
                        self.engine.exit_position(i, date, close, "M1 Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10
                size *= cfg.mr_size_mult

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "M1", size)
                elif row["short_signal"]:
                    self.engine.enter_position(i, date, close, "short", "M1", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("M1 Mean-Reversion buys when price drops below the lower Bollinger Band and RSI < 30 "
                "(oversold). It sells short when price breaks above the upper BB and RSI > 70 (overbought). "
                "Targets the BB midline for profit; exits via ATR stop or 10-bar time limit.")


# ─────────────────────────────────────────────────────
# H1: BB/KC Squeeze Breakout
# ─────────────────────────────────────────────────────

class H1_SqueezeBreakout(StrategyBase):
    """Bollinger Band / Keltner Channel squeeze breakout."""

    name = "H1 Squeeze Breakout"
    description = "BB/KC squeeze compression breakout"

    PARAMS = {
        "bb_len":       {"default": 20,  "min": 10,  "max": 40,  "step": 1,   "label": "BB Period"},
        "bb_mult":      {"default": 2.0, "min": 1.0, "max": 3.5, "step": 0.1, "label": "BB Multiplier"},
        "kc_mult":      {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "label": "KC Multiplier"},
        "squeeze_bars": {"default": 6,   "min": 3,   "max": 15,  "step": 1,   "label": "Min Squeeze Bars"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.bb_len = params.get("bb_len", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self.kc_len = params.get("kc_len", 20)
        self.kc_mult = params.get("kc_mult", 1.5)
        self.squeeze_bars = params.get("squeeze_bars", 6)
        self.linreg_len = params.get("linreg_len", 20)
        self.atr_len = params.get("atr_len", 14)
        self.ema_len = params.get("ema_fast", 21)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bb_mid"], df["bb_upper"], df["bb_lower"], _ = ind.bollinger_bands(
            df["close"], self.bb_len, self.bb_mult
        )
        df["kc_basis"], df["kc_upper"], df["kc_lower"] = ind.keltner_channels(
            df["close"], df["high"], df["low"], self.kc_len, self.kc_mult
        )
        df["linreg_slope"] = ind.linear_regression_slope(df["close"], self.linreg_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["ema_fast"] = ind.ema(df["close"], self.ema_len)

        df["is_squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
        squeeze_count = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df["is_squeeze"].iloc[i]:
                squeeze_count.iloc[i] = squeeze_count.iloc[i-1] + 1
            else:
                squeeze_count.iloc[i] = 0
        df["squeeze_count"] = squeeze_count
        df["squeeze_fired"] = (~df["is_squeeze"]) & (df["squeeze_count"].shift(1) >= self.squeeze_bars)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["long_signal"] = df["squeeze_fired"] & (df["linreg_slope"] > 0) & (df["close"] > df["ema_fast"])
        df["short_signal"] = df["squeeze_fired"] & (df["linreg_slope"] < 0) & (df["close"] < df["ema_fast"])
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    init_stop = pos.entry_price - cfg.squeeze_stop_mult * atr_val
                    trail_stop = pos.highest_since - cfg.squeeze_trail_mult * atr_val
                    eff_stop = max(init_stop, trail_stop)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "H1 Stop")
                    elif bars_held >= cfg.squeeze_max_bars:
                        self.engine.exit_position(i, date, close, "H1 Time Exit")
                else:
                    init_stop = pos.entry_price + cfg.squeeze_stop_mult * atr_val
                    trail_stop = pos.lowest_since + cfg.squeeze_trail_mult * atr_val
                    eff_stop = min(init_stop, trail_stop)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "H1 Stop")
                    elif bars_held >= cfg.squeeze_max_bars:
                        self.engine.exit_position(i, date, close, "H1 Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row.get("long_signal", False):
                    self.engine.enter_position(i, date, close, "long", "H1", size)
                elif row.get("short_signal", False):
                    self.engine.enter_position(i, date, close, "short", "H1", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("H1 Squeeze detects when Bollinger Bands compress inside Keltner Channels (volatility squeeze). "
                "When the squeeze 'fires' (BB expands outside KC after 6+ bars), it enters in the direction "
                "of the linear regression slope. Wide trailing stops give breakouts room to run.")


# ─────────────────────────────────────────────────────
# RSI Strategy
# ─────────────────────────────────────────────────────

class RSI_Strategy(StrategyBase):
    """RSI momentum/reversal strategy."""

    name = "RSI Momentum"
    description = "RSI overbought/oversold reversals"

    PARAMS = {
        "rsi_len": {"default": 14,   "min": 5,    "max": 30,   "step": 1,   "label": "RSI Period"},
        "rsi_ob":  {"default": 70.0, "min": 60.0, "max": 85.0, "step": 1.0, "label": "Overbought Level"},
        "rsi_os":  {"default": 30.0, "min": 15.0, "max": 40.0, "step": 1.0, "label": "Oversold Level"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.rsi_len = params.get("rsi_len", 14)
        self.rsi_ob = params.get("rsi_ob", 70.0)
        self.rsi_os = params.get("rsi_os", 30.0)
        self.atr_len = params.get("atr_len", 14)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ind.rsi(df["close"], self.rsi_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["sma50"] = ind.sma(df["close"], 50)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = df["rsi"]
        # Buy when RSI crosses up from oversold, sell when crosses down from overbought
        df["long_signal"] = (rsi > self.rsi_os) & (rsi.shift(1) <= self.rsi_os)
        df["short_signal"] = (rsi < self.rsi_ob) & (rsi.shift(1) >= self.rsi_ob)
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    stop = pos.entry_price - cfg.mr_stop_mult * atr_val
                    trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    eff_stop = max(stop, trail)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "RSI Stop")
                    elif df["rsi"].iloc[i] > self.rsi_ob:
                        self.engine.exit_position(i, date, close, "RSI OB Exit")
                    elif bars_held >= 30:
                        self.engine.exit_position(i, date, close, "RSI Time Exit")
                else:
                    stop = pos.entry_price + cfg.mr_stop_mult * atr_val
                    trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    eff_stop = min(stop, trail)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "RSI Stop")
                    elif df["rsi"].iloc[i] < self.rsi_os:
                        self.engine.exit_position(i, date, close, "RSI OS Exit")
                    elif bars_held >= 30:
                        self.engine.exit_position(i, date, close, "RSI Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10
                size *= cfg.mr_size_mult

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "RSI", size)
                elif row["short_signal"]:
                    self.engine.enter_position(i, date, close, "short", "RSI", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("RSI Momentum buys when RSI crosses back above 30 (leaving oversold territory) "
                "and sells short when RSI drops below 70 (leaving overbought). Exits on opposite "
                "RSI extreme, ATR trailing stop, or 30-bar time limit.")


# ─────────────────────────────────────────────────────
# MACD Strategy
# ─────────────────────────────────────────────────────

class MACD_Strategy(StrategyBase):
    """MACD signal-line crossover strategy."""

    name = "MACD Crossover"
    description = "MACD / Signal line crossover"

    PARAMS = {
        "macd_fast":   {"default": 12,  "min": 5,   "max": 20,  "step": 1, "label": "MACD Fast"},
        "macd_slow":   {"default": 26,  "min": 15,  "max": 50,  "step": 1, "label": "MACD Slow"},
        "macd_signal": {"default": 9,   "min": 3,   "max": 15,  "step": 1, "label": "Signal Period"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal = params.get("macd_signal", 9)
        self.atr_len = params.get("atr_len", 14)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["macd_line"], df["macd_sig"], df["macd_hist"] = ind.macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_signal
        )
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ml = df["macd_line"]
        ms = df["macd_sig"]
        df["long_signal"] = (ml > ms) & (ml.shift(1) <= ms.shift(1))
        df["short_signal"] = (ml < ms) & (ml.shift(1) >= ms.shift(1))
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                    trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    eff_stop = max(stop, trail)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "MACD Stop")
                    elif row["short_signal"]:
                        self.engine.exit_position(i, date, close, "MACD Reverse")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "MACD Time Exit")
                else:
                    stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                    trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    eff_stop = min(stop, trail)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "MACD Stop")
                    elif row["long_signal"]:
                        self.engine.exit_position(i, date, close, "MACD Reverse")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "MACD Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "MACD", size)
                elif row["short_signal"]:
                    self.engine.enter_position(i, date, close, "short", "MACD", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("MACD Crossover enters long when the MACD line (12-EMA minus 26-EMA) crosses above "
                "the 9-period signal line, and short on the opposite crossover. Exits via ATR trailing "
                "stop, signal reversal, or 60-bar time limit.")


# ─────────────────────────────────────────────────────
# SMA Crossover Strategy
# ─────────────────────────────────────────────────────

class SMA_Crossover(StrategyBase):
    """Simple Moving Average crossover — classic trend strategy."""

    name = "SMA Crossover"
    description = "Golden/Death cross (SMA 50/200)"

    PARAMS = {
        "sma_fast": {"default": 50,  "min": 10,  "max": 100,  "step": 5, "label": "SMA Fast Period"},
        "sma_slow": {"default": 200, "min": 100, "max": 350,  "step": 10, "label": "SMA Slow Period"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        self.sma_fast_len = params.get("sma_fast", 50)
        self.sma_slow_len = params.get("sma_slow", 200)
        self.atr_len = params.get("atr_len", 14)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sma_fast"] = ind.sma(df["close"], self.sma_fast_len)
        df["sma_slow"] = ind.sma(df["close"], self.sma_slow_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sf = df["sma_fast"]
        ss = df["sma_slow"]
        df["long_signal"] = (sf > ss) & (sf.shift(1) <= ss.shift(1))
        df["short_signal"] = (sf < ss) & (sf.shift(1) >= ss.shift(1))
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    if low <= trail:
                        self.engine.exit_position(i, date, trail, "SMA Stop")
                    elif row["short_signal"]:
                        self.engine.exit_position(i, date, close, "SMA Reverse")
                else:
                    trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    if high >= trail:
                        self.engine.exit_position(i, date, trail, "SMA Stop")
                    elif row["long_signal"]:
                        self.engine.exit_position(i, date, close, "SMA Reverse")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "SMA", size)
                elif row["short_signal"]:
                    self.engine.enter_position(i, date, close, "short", "SMA", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("SMA Crossover uses the classic Golden Cross / Death Cross pattern. "
                "Enters long when SMA 50 crosses above SMA 200, short on the reverse. "
                "Exits via ATR trailing stop or signal reversal. Very few trades but captures major trends.")


# ─────────────────────────────────────────────────────
# H2: Adaptive Blend (main strategy) — IMPROVED
# ─────────────────────────────────────────────────────

class H2_AdaptiveBlend(StrategyBase):
    """H2 Adaptive Blend — combines T1 + M1 + H1 with regime detection.

    IMPROVED: Relaxed regime thresholds for more T1/M1 trades.
    Priority: H1 > T1 > M1
    """

    name = "H2 Adaptive Blend"
    description = "Regime-adaptive blend: T1 + M1 + H1"

    PARAMS = {
        "adx_threshold": {"default": 20.0, "min": 12.0, "max": 35.0, "step": 1.0, "label": "ADX Trend Threshold"},
        "adx_range_th":  {"default": 15.0, "min": 8.0,  "max": 25.0, "step": 1.0, "label": "ADX Range Threshold"},
        "vats_trend_th": {"default": 1.2,  "min": 0.5,  "max": 2.5,  "step": 0.1, "label": "VATS Trend Threshold"},
        "regime_persist": {"default": 2,   "min": 1,    "max": 5,    "step": 1,   "label": "Regime Persistence"},
        "rsi_ob":         {"default": 70.0, "min": 60.0, "max": 85.0, "step": 1.0, "label": "RSI Overbought"},
        "rsi_os":         {"default": 30.0, "min": 15.0, "max": 40.0, "step": 1.0, "label": "RSI Oversold"},
    }

    def __init__(self, config: BacktestConfig = None, **params):
        super().__init__(config, **params)
        # Regime params — RELAXED defaults
        self.ema_fast_len = params.get("ema_fast", 21)
        self.ema_slow_len = params.get("ema_slow", 55)
        self.adx_len = params.get("adx_len", 14)
        self.adx_threshold = params.get("adx_threshold", 20.0)   # was 22
        self.adx_range_th = params.get("adx_range_th", 15.0)     # was 18
        self.vats_lookback = params.get("vats_lookback", 20)
        self.vats_trend_th = params.get("vats_trend_th", 1.2)    # was 1.5
        self.vats_range_th = params.get("vats_range_th", 1.0)
        self.regime_persist = params.get("regime_persist", 2)     # was 3

        # BB / Squeeze params
        self.bb_len = params.get("bb_len", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self.kc_len = params.get("kc_len", 20)
        self.kc_mult = params.get("kc_mult", 1.5)
        self.squeeze_bars = params.get("squeeze_bars", 6)
        self.linreg_len = params.get("linreg_len", 20)

        # Entry params
        self.rsi_len = params.get("rsi_len", 14)
        self.rsi_ob = params.get("rsi_ob", 70.0)
        self.rsi_os = params.get("rsi_os", 30.0)
        self.atr_len = params.get("atr_len", 14)

        # Vol regime
        self.vol_high_th = params.get("vol_high_th", 75.0)
        self.vol_low_th = params.get("vol_low_th", 25.0)
        self.pctile_lookback = params.get("pctile_lookback", 100)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = ind.ema(df["close"], self.ema_fast_len)
        df["ema_slow"] = ind.ema(df["close"], self.ema_slow_len)
        df["adx"] = ind.adx(df["high"], df["low"], df["close"], self.adx_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["rsi"] = ind.rsi(df["close"], self.rsi_len)

        df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"] = ind.bollinger_bands(
            df["close"], self.bb_len, self.bb_mult
        )
        df["kc_basis"], df["kc_upper"], df["kc_lower"] = ind.keltner_channels(
            df["close"], df["high"], df["low"], self.kc_len, self.kc_mult
        )
        df["linreg_slope"] = ind.linear_regression_slope(df["close"], self.linreg_len)
        df["atr_norm"] = df["atr"] / df["close"] * 100
        df["atr_pctile"] = ind.percentile_rank(df["atr_norm"], self.pctile_lookback)
        df["bbw_pctile"] = ind.percentile_rank(df["bb_width"], self.pctile_lookback)
        df["vol_score"] = (df["atr_pctile"].fillna(50) + df["bbw_pctile"].fillna(50)) / 2
        df["vats"] = ind.vats(df["close"], df["atr"], self.vats_lookback)

        df["is_squeeze"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
        squeeze_count = pd.Series(0, index=df.index, dtype=int)
        for i in range(1, len(df)):
            if df["is_squeeze"].iloc[i]:
                squeeze_count.iloc[i] = squeeze_count.iloc[i-1] + 1
        df["squeeze_count"] = squeeze_count
        df["squeeze_fired"] = (~df["is_squeeze"]) & (df["squeeze_count"].shift(1) >= self.squeeze_bars)

        return df

    def _classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """Regime classification with persistence filter.
        Returns: 0=RANGE, 1=BULL, -1=BEAR, 2=TRANSITION
        """
        n = len(df)
        regime = pd.Series(0, index=df.index, dtype=int)
        counter = 0
        current_regime = 0

        for i in range(n):
            vats_val = df["vats"].iloc[i] if not np.isnan(df["vats"].iloc[i]) else 0
            adx_val = df["adx"].iloc[i] if not np.isnan(df["adx"].iloc[i]) else 0
            ema_f = df["ema_fast"].iloc[i]
            ema_s = df["ema_slow"].iloc[i]

            # Raw regime — RELAXED: allow "mild trends"
            if vats_val > self.vats_trend_th and ema_f > ema_s and adx_val > self.adx_threshold:
                raw = 1   # Bull
            elif vats_val > self.vats_trend_th and ema_f < ema_s and adx_val > self.adx_threshold:
                raw = -1  # Bear
            elif adx_val > self.adx_threshold and ema_f > ema_s:
                raw = 1   # Mild bull (ADX confirms but VATS borderline)
            elif adx_val > self.adx_threshold and ema_f < ema_s:
                raw = -1  # Mild bear
            elif vats_val < self.vats_range_th or adx_val < self.adx_range_th:
                raw = 0   # Range
            else:
                raw = 2   # Transition

            if raw == current_regime:
                counter = 0
            else:
                counter += 1
                if counter >= self.regime_persist:
                    current_regime = raw
                    counter = 0

            regime.iloc[i] = current_regime

        return regime

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["regime"] = self._classify_regime(df)

        ema_f = df["ema_fast"]
        ema_s = df["ema_slow"]

        cross_above = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        cross_below = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
        adx_ok = df["adx"] > self.adx_threshold

        # T1: allow in BULL, BEAR, and TRANSITION regimes (not just exact match)
        df["t1_long"] = cross_above & adx_ok & (df["regime"].isin([1, 2]))
        df["t1_short"] = cross_below & adx_ok & (df["regime"].isin([-1, 2]))

        # M1: range regime, low vol
        df["m1_long"] = (df["close"] < df["bb_lower"]) & (df["rsi"] < self.rsi_os) & \
                         (df["regime"] == 0) & (df["vol_score"] < self.vol_high_th)
        df["m1_short"] = (df["close"] > df["bb_upper"]) & (df["rsi"] > self.rsi_ob) & \
                          (df["regime"] == 0) & (df["vol_score"] < self.vol_high_th)

        # H1: squeeze breakout (no regime filter)
        df["h1_long"] = df["squeeze_fired"] & (df["linreg_slope"] > 0) & (df["close"] > ema_f)
        df["h1_short"] = df["squeeze_fired"] & (df["linreg_slope"] < 0) & (df["close"] < ema_f)

        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02
            bb_mid = row["bb_mid"] if not np.isnan(row["bb_mid"]) else close

            self.engine.update_position_tracking(high, low)

            # ── EXIT LOGIC ──
            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.strategy == "T1":
                    long_stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                    short_stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                    long_trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    short_trail = pos.lowest_since + cfg.atr_trail_mult * atr_val

                    if pos.direction == "long":
                        eff_stop = max(long_stop, long_trail)
                        if low <= eff_stop:
                            self.engine.exit_position(i, date, eff_stop, "T1 Stop")
                        elif bars_held >= cfg.max_bars_trend:
                            self.engine.exit_position(i, date, close, "T1 Time Exit")
                        elif row["regime"] == 0 and bars_held > 5:
                            self.engine.exit_position(i, date, close, "T1 Regime Exit")
                    else:
                        eff_stop = min(short_stop, short_trail)
                        if high >= eff_stop:
                            self.engine.exit_position(i, date, eff_stop, "T1 Stop")
                        elif bars_held >= cfg.max_bars_trend:
                            self.engine.exit_position(i, date, close, "T1 Time Exit")
                        elif row["regime"] == 0 and bars_held > 5:
                            self.engine.exit_position(i, date, close, "T1 Regime Exit")

                elif pos.strategy == "M1":
                    if pos.direction == "long":
                        mr_stop = pos.entry_price - cfg.mr_stop_mult * atr_val
                        if close >= bb_mid:
                            self.engine.exit_position(i, date, close, "M1 Target")
                        elif low <= mr_stop:
                            self.engine.exit_position(i, date, mr_stop, "M1 Stop")
                        elif bars_held >= cfg.mr_time_exit:
                            self.engine.exit_position(i, date, close, "M1 Time Exit")
                        elif row["regime"] in [1, -1]:
                            self.engine.exit_position(i, date, close, "M1 Regime Exit")
                    else:
                        mr_stop = pos.entry_price + cfg.mr_stop_mult * atr_val
                        if close <= bb_mid:
                            self.engine.exit_position(i, date, close, "M1 Target")
                        elif high >= mr_stop:
                            self.engine.exit_position(i, date, mr_stop, "M1 Stop")
                        elif bars_held >= cfg.mr_time_exit:
                            self.engine.exit_position(i, date, close, "M1 Time Exit")
                        elif row["regime"] in [1, -1]:
                            self.engine.exit_position(i, date, close, "M1 Regime Exit")

                elif pos.strategy == "H1":
                    if pos.direction == "long":
                        init_stop = pos.entry_price - cfg.squeeze_stop_mult * atr_val
                        trail_stop = pos.highest_since - cfg.squeeze_trail_mult * atr_val
                        eff_stop = max(init_stop, trail_stop)
                        if low <= eff_stop:
                            self.engine.exit_position(i, date, eff_stop, "H1 Stop")
                        elif bars_held >= cfg.squeeze_max_bars:
                            self.engine.exit_position(i, date, close, "H1 Time Exit")
                    else:
                        init_stop = pos.entry_price + cfg.squeeze_stop_mult * atr_val
                        trail_stop = pos.lowest_since + cfg.squeeze_trail_mult * atr_val
                        eff_stop = min(init_stop, trail_stop)
                        if high >= eff_stop:
                            self.engine.exit_position(i, date, eff_stop, "H1 Stop")
                        elif bars_held >= cfg.squeeze_max_bars:
                            self.engine.exit_position(i, date, close, "H1 Time Exit")

            # ── ENTRY LOGIC (Priority: H1 > T1 > M1) ──
            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                base_size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row.get("h1_long", False):
                    self.engine.enter_position(i, date, close, "long", "H1", base_size)
                elif row.get("h1_short", False):
                    self.engine.enter_position(i, date, close, "short", "H1", base_size * cfg.short_size_mult)
                elif row.get("t1_long", False):
                    self.engine.enter_position(i, date, close, "long", "T1", base_size)
                elif row.get("t1_short", False):
                    self.engine.enter_position(i, date, close, "short", "T1", base_size * cfg.short_size_mult)
                elif row.get("m1_long", False):
                    self.engine.enter_position(i, date, close, "long", "M1", base_size * cfg.mr_size_mult)
                elif row.get("m1_short", False):
                    self.engine.enter_position(i, date, close, "short", "M1",
                                               base_size * cfg.mr_size_mult * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return ("H2 Adaptive Blend is the main strategy. It classifies the market into regimes "
                "(Bull/Bear/Range/Transition) using ADX + VATS + EMA alignment, then activates: "
                "T1 (trend-following) in trends, M1 (mean-reversion) in ranges, H1 (squeeze breakout) "
                "anywhere. Priority: H1 > T1 > M1. Improved v2: relaxed regime thresholds (ADX 20, "
                "VATS 1.2, persist 2) so T1 fires more often.")


# ─────────────────────────────────────────────────────
# Buy & Hold Benchmark
# ─────────────────────────────────────────────────────

class BuyAndHold(StrategyBase):
    """Simple buy & hold benchmark for comparison."""

    name = "Buy & Hold"
    description = "Buy on first bar after warmup, hold forever"
    PARAMS = {}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], 14)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["long_signal"] = False
        warmup = min(self.config.warmup_bars, len(df) - 1)
        df.iloc[warmup, df.columns.get_loc("long_signal")] = True
        return df

    def _execute(self, df: pd.DataFrame):
        cfg = self.config
        warmup = min(cfg.warmup_bars, len(df) - 1)
        for i in range(len(df)):
            row = df.iloc[i]
            date = row["date"]
            close = row["close"]
            high = row["high"]
            low = row["low"]

            self.engine.update_position_tracking(high, low)

            if i == warmup and self.engine.position is None:
                self.engine.enter_position(i, date, close, "long", "B&H", 100)

            if i == len(df) - 1 and self.engine.position is not None:
                self.engine.exit_position(i, date, close, "End of Data")

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls) -> str:
        return "Buy & Hold simply buys at the start and holds until the end. Used as a benchmark."


# ─────────────────────────────────────────────────────
# Momentum: Rate of Change
# ─────────────────────────────────────────────────────

class MomentumROC(StrategyBase):
    """Buy when N-day rate-of-change exceeds threshold, exit when momentum fades."""

    name = "Momentum ROC"
    description = "Buy on strong upward momentum (ROC), exit on reversal"

    PARAMS = {
        "roc_len":       {"default": 20,  "min": 5,   "max": 60,  "step": 5,   "label": "ROC Lookback"},
        "roc_threshold": {"default": 10.0,"min": 3.0, "max": 30.0,"step": 1.0, "label": "ROC Entry Threshold %"},
        "roc_exit":      {"default": 0.0, "min":-10.0,"max": 5.0, "step": 1.0, "label": "ROC Exit Threshold %"},
        "use_sma_filter":{"default": 1,   "min": 0,   "max": 1,   "step": 1,   "label": "Use SMA 200 Filter"},
        "sma_len":       {"default": 200, "min": 50,  "max": 300, "step": 50,  "label": "SMA Filter Length"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.roc_len = int(params.get("roc_len", 20))
        self.roc_threshold = params.get("roc_threshold", 10.0)
        self.roc_exit = params.get("roc_exit", 0.0)
        self.use_sma_filter = bool(params.get("use_sma_filter", 1))
        self.sma_len = int(params.get("sma_len", 200))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["roc"] = (df["close"] / df["close"].shift(self.roc_len) - 1) * 100
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        if self.use_sma_filter:
            df["sma_filter"] = ind.sma(df["close"], self.sma_len)
        return df

    def generate_signals(self, df):
        long_cond = df["roc"] > self.roc_threshold
        if self.use_sma_filter:
            long_cond = long_cond & (df["close"] > df["sma_filter"])
        # Only enter on the bar ROC first crosses above threshold
        df["long_signal"] = long_cond & (~long_cond.shift(1).fillna(False))
        df["short_signal"] = False
        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02
            roc_val = row["roc"] if not np.isnan(row["roc"]) else 0

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar
                trail_stop = pos.highest_since - cfg.atr_trail_mult * atr_val

                if pos.direction == "long":
                    if low <= trail_stop:
                        self.engine.exit_position(i, date, trail_stop, "MOM Trail Stop")
                    elif roc_val < self.roc_exit:
                        self.engine.exit_position(i, date, close, "MOM Fade")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "MOM Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "MOM", size)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Momentum ROC buys when N-day rate of change exceeds a threshold "
                "(e.g., +10%), optionally filtered by price > SMA 200. "
                "Exits when momentum fades below exit threshold or trailing stop hit.")


# ─────────────────────────────────────────────────────
# Buy-the-Dip: Buy after significant drawdown from high
# ─────────────────────────────────────────────────────

class BuyTheDip(StrategyBase):
    """Buy after BTC drops N% from recent high — exploits recovery pattern."""

    name = "Buy the Dip"
    description = "Buy after N% drawdown from recent high, sell on recovery"

    PARAMS = {
        "lookback":      {"default": 60,  "min": 20,  "max": 200, "step": 10,  "label": "High Lookback"},
        "dip_pct":       {"default": 20.0,"min": 5.0, "max": 40.0,"step": 2.5, "label": "Dip Threshold %"},
        "recovery_pct":  {"default": 10.0,"min": 3.0, "max": 30.0,"step": 1.0, "label": "Recovery Target %"},
        "max_hold":      {"default": 90,  "min": 20,  "max": 200, "step": 10,  "label": "Max Bars Held"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.lookback = int(params.get("lookback", 60))
        self.dip_pct = params.get("dip_pct", 20.0)
        self.recovery_pct = params.get("recovery_pct", 10.0)
        self.max_hold = int(params.get("max_hold", 90))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["rolling_high"] = df["high"].rolling(self.lookback, min_periods=1).max()
        df["drawdown_from_high"] = (1 - df["close"] / df["rolling_high"]) * 100
        return df

    def generate_signals(self, df):
        # Enter when drawdown exceeds threshold % and starts recovering (close > yesterday)
        dip_hit = df["drawdown_from_high"] >= self.dip_pct
        recovering = df["close"] > df["close"].shift(1)
        # Only trigger on first bar of recovery after dip
        df["long_signal"] = dip_hit & recovering & (~(dip_hit.shift(1).fillna(False) & recovering.shift(1).fillna(False)))
        df["short_signal"] = False
        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar
                gain_pct = (close / pos.entry_price - 1) * 100
                stop_price = pos.entry_price - cfg.atr_stop_mult * atr_val

                if low <= stop_price:
                    self.engine.exit_position(i, date, stop_price, "DIP Stop")
                elif gain_pct >= self.recovery_pct:
                    self.engine.exit_position(i, date, close, "DIP Target")
                elif bars_held >= self.max_hold:
                    self.engine.exit_position(i, date, close, "DIP Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "DIP", size)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Buy the Dip buys when BTC drops N% from its recent high and starts recovering. "
                "Exits at a recovery target (e.g., +10%) or via ATR stop loss. "
                "Exploits BTC's historical pattern of sharp dips followed by recovery.")


# ─────────────────────────────────────────────────────
# Donchian Breakout (Turtle-style)
# ─────────────────────────────────────────────────────

class DonchianBreakout(StrategyBase):
    """Donchian Channel breakout — enter on new N-bar high, exit on new M-bar low."""

    name = "Donchian Breakout"
    description = "Enter on N-bar high breakout, exit on M-bar low"

    PARAMS = {
        "entry_len":  {"default": 55,  "min": 10,  "max": 100, "step": 5,  "label": "Entry Channel Length"},
        "exit_len":   {"default": 20,  "min": 5,   "max": 50,  "step": 5,  "label": "Exit Channel Length"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.entry_len = int(params.get("entry_len", 55))
        self.exit_len = int(params.get("exit_len", 20))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["dc_high"] = df["high"].rolling(self.entry_len, min_periods=1).max()
        df["dc_low"] = df["low"].rolling(self.entry_len, min_periods=1).min()
        df["dc_exit_low"] = df["low"].rolling(self.exit_len, min_periods=1).min()
        df["dc_exit_high"] = df["high"].rolling(self.exit_len, min_periods=1).max()
        return df

    def generate_signals(self, df):
        # Long: close breaks above N-bar high channel
        df["long_signal"] = df["close"] > df["dc_high"].shift(1)
        # Short: close breaks below N-bar low channel
        df["short_signal"] = df["close"] < df["dc_low"].shift(1)
        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    exit_level = row["dc_exit_low"] if not np.isnan(row["dc_exit_low"]) else low
                    if low <= exit_level:
                        self.engine.exit_position(i, date, exit_level, "DON Exit Low")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "DON Time Exit")
                    elif row.get("short_signal", False) and cfg.short_size_mult > 0:
                        self.engine.exit_position(i, date, close, "DON Reverse")
                else:
                    exit_level = row["dc_exit_high"] if not np.isnan(row["dc_exit_high"]) else high
                    if high >= exit_level:
                        self.engine.exit_position(i, date, exit_level, "DON Exit High")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "DON Time Exit")
                    elif row.get("long_signal", False):
                        self.engine.exit_position(i, date, close, "DON Reverse")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "DON", size)
                elif row["short_signal"] and cfg.short_size_mult > 0:
                    self.engine.enter_position(i, date, close, "short", "DON", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Donchian Breakout (Turtle-style) enters long when price breaks above the N-bar high, "
                "and exits when price drops to the M-bar low. A classic trend-following system.")


# ─────────────────────────────────────────────────────
# BreakdownShort: Short when price breaks below structure
# Pattern: price falls below SMA + RSI confirms weakness + volume spike
# ─────────────────────────────────────────────────────

class BreakdownShort(StrategyBase):
    """Short on structural breakdown — price below SMA + RSI weak + vol spike.
    
    PATTERN: Identifies tops/exhaustion by looking for:
    - Price crossing below key SMA (trend reversal)
    - RSI confirming weakness (< threshold)
    - Optional: volume above average (distribution)
    Also goes long on the inverse pattern for bi-directional trading.
    """

    name = "Breakdown Short"
    description = "Short on breakdown below SMA with RSI weakness confirmation"

    PARAMS = {
        "sma_len":       {"default": 50,  "min": 20,  "max": 200, "step": 10,  "label": "SMA Length"},
        "rsi_len":       {"default": 14,  "min": 6,   "max": 21,  "step": 1,   "label": "RSI Length"},
        "rsi_short":     {"default": 45,  "min": 30,  "max": 55,  "step": 5,   "label": "RSI Short Threshold"},
        "rsi_long":      {"default": 55,  "min": 45,  "max": 70,  "step": 5,   "label": "RSI Long Threshold"},
        "vol_mult":      {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.5, "label": "Volume Spike Multiple"},
        "use_vol":       {"default": 0,   "min": 0,   "max": 1,   "step": 1,   "label": "Use Volume Filter"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.sma_len = int(params.get("sma_len", 50))
        self.rsi_len = int(params.get("rsi_len", 14))
        self.rsi_short = params.get("rsi_short", 45)
        self.rsi_long = params.get("rsi_long", 55)
        self.vol_mult = params.get("vol_mult", 1.5)
        self.use_vol = bool(params.get("use_vol", 0))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["sma_key"] = ind.sma(df["close"], self.sma_len)
        df["rsi"] = ind.rsi(df["close"], self.rsi_len)
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["vol_sma"] = df["volume"].rolling(20, min_periods=1).mean()
        return df

    def generate_signals(self, df):
        # Short: price crosses below SMA + RSI weak
        below_sma = (df["close"] < df["sma_key"]) & (df["close"].shift(1) >= df["sma_key"].shift(1))
        rsi_weak = df["rsi"] < self.rsi_short

        # Long: price crosses above SMA + RSI strong
        above_sma = (df["close"] > df["sma_key"]) & (df["close"].shift(1) <= df["sma_key"].shift(1))
        rsi_strong = df["rsi"] > self.rsi_long

        if self.use_vol:
            vol_spike = df["volume"] > df["vol_sma"] * self.vol_mult
            df["short_signal"] = below_sma & rsi_weak & vol_spike
            df["long_signal"] = above_sma & rsi_strong & vol_spike
        else:
            df["short_signal"] = below_sma & rsi_weak
            df["long_signal"] = above_sma & rsi_strong

        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "short":
                    short_stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                    short_trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    eff_stop = min(short_stop, short_trail)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "BKD Short Stop")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "BKD Time Exit")
                    elif row.get("long_signal", False):
                        self.engine.exit_position(i, date, close, "BKD Reverse Signal")
                else:  # long
                    long_stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                    long_trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    eff_stop = max(long_stop, long_trail)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "BKD Long Stop")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "BKD Time Exit")
                    elif row.get("short_signal", False):
                        self.engine.exit_position(i, date, close, "BKD Reverse Signal")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["short_signal"] and cfg.short_size_mult > 0:
                    self.engine.enter_position(i, date, close, "short", "BKD", size * cfg.short_size_mult)
                elif row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "BKD", size)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Breakdown Short identifies structural breakdowns: price crossing below SMA "
                "with RSI confirming weakness. Also goes long on the inverse pattern. "
                "Key pattern: distribution (high volume) + trend break = shorting opportunity.")


# ─────────────────────────────────────────────────────
# SellTheRip: Short after sharp rallies (inverse of BuyTheDip)
# Pattern: price spikes N% above rolling low → overbought → short
# ─────────────────────────────────────────────────────

class SellTheRip(StrategyBase):
    """Short after BTC spikes N% above recent low — exploits mean reversion.
    
    PATTERN: After a sharp rally, price often pulls back 30-50% of the move.
    This strategy shorts the overextension and rides the pullback.
    Also buys dips as the long-side counterpart.
    """

    name = "Sell the Rip"
    description = "Short after N% rally from recent low, cover on pullback"

    PARAMS = {
        "lookback":      {"default": 60,  "min": 20,  "max": 200, "step": 10,  "label": "Low Lookback"},
        "rip_pct":       {"default": 30.0,"min": 10.0,"max": 60.0,"step": 5.0, "label": "Rip Threshold %"},
        "pullback_pct":  {"default": 10.0,"min": 3.0, "max": 25.0,"step": 1.0, "label": "Pullback Target %"},
        "dip_pct":       {"default": 15.0,"min": 5.0, "max": 40.0,"step": 5.0, "label": "Dip Entry (Long) %"},
        "max_hold":      {"default": 72,  "min": 20,  "max": 200, "step": 10,  "label": "Max Bars Held"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.lookback = int(params.get("lookback", 60))
        self.rip_pct = params.get("rip_pct", 30.0)
        self.pullback_pct = params.get("pullback_pct", 10.0)
        self.dip_pct = params.get("dip_pct", 15.0)
        self.max_hold = int(params.get("max_hold", 72))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["rolling_low"] = df["low"].rolling(self.lookback, min_periods=1).min()
        df["rolling_high"] = df["high"].rolling(self.lookback, min_periods=1).max()
        df["rally_from_low"] = (df["close"] / df["rolling_low"] - 1) * 100
        df["drop_from_high"] = (1 - df["close"] / df["rolling_high"]) * 100
        df["rsi"] = ind.rsi(df["close"], 14)
        return df

    def generate_signals(self, df):
        # Short: rally too far from low + overbought + starts dropping
        rip_hit = df["rally_from_low"] >= self.rip_pct
        overbought = df["rsi"] > 65
        dropping = df["close"] < df["close"].shift(1)
        df["short_signal"] = rip_hit & overbought & dropping & \
                             (~(rip_hit.shift(1).fillna(False) & dropping.shift(1).fillna(False)))

        # Long: dip too far from high + oversold + recovering
        dip_hit = df["drop_from_high"] >= self.dip_pct
        oversold = df["rsi"] < 35
        recovering = df["close"] > df["close"].shift(1)
        df["long_signal"] = dip_hit & oversold & recovering & \
                            (~(dip_hit.shift(1).fillna(False) & recovering.shift(1).fillna(False)))

        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "short":
                    # Cover when price pulls back target % from entry
                    target_price = pos.entry_price * (1 - self.pullback_pct / 100)
                    short_stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                    short_trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    eff_stop = min(short_stop, short_trail)

                    if low <= target_price:
                        self.engine.exit_position(i, date, target_price, "STR Target Hit")
                    elif high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "STR Short Stop")
                    elif bars_held >= self.max_hold:
                        self.engine.exit_position(i, date, close, "STR Time Exit")
                else:  # long
                    target_price = pos.entry_price * (1 + self.pullback_pct / 100)
                    long_stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                    long_trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    eff_stop = max(long_stop, long_trail)

                    if high >= target_price:
                        self.engine.exit_position(i, date, target_price, "STR Long Target")
                    elif low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "STR Long Stop")
                    elif bars_held >= self.max_hold:
                        self.engine.exit_position(i, date, close, "STR Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                # Priority: short signals first (this is a short-focused strategy)
                if row["short_signal"] and cfg.short_size_mult > 0:
                    self.engine.enter_position(i, date, close, "short", "STR", size * cfg.short_size_mult)
                elif row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "STR", size)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Sell the Rip shorts after BTC rallies N% from recent low + RSI overbought. "
                "Covers on pullback target or trailing stop. Also buys dips on inverse pattern. "
                "Key insight: sharp rallies often retrace 30-50%.")


# ─────────────────────────────────────────────────────
# MultiPattern: Combines multiple repeating patterns
# Learns from: EMA cross + RSI divergence + volume + structure
# ─────────────────────────────────────────────────────

class MultiPattern(StrategyBase):
    """Multi-signal pattern scorer — combines several indicators into a
    composite score for higher-confidence entries.
    
    PATTERNS DETECTED:
    1. Trend alignment (EMA fast vs slow)
    2. Momentum (ROC direction)
    3. Mean reversion (RSI extreme zones)
    4. Structure (price vs recent high/low)
    5. Volatility expansion (ATR breakout)
    
    Score -5 to +5: >= threshold → long, <= -threshold → short
    """

    name = "Multi Pattern"
    description = "Composite pattern scorer combining trend, momentum, RSI, structure"

    PARAMS = {
        "ema_fast":      {"default": 21,  "min": 8,   "max": 50,  "step": 1,   "label": "EMA Fast"},
        "ema_slow":      {"default": 55,  "min": 30,  "max": 200, "step": 5,   "label": "EMA Slow"},
        "roc_len":       {"default": 20,  "min": 5,   "max": 50,  "step": 5,   "label": "ROC Lookback"},
        "structure_len": {"default": 50,  "min": 20,  "max": 100, "step": 10,  "label": "Structure Lookback"},
        "entry_score":   {"default": 3,   "min": 2,   "max": 5,   "step": 1,   "label": "Entry Score Threshold"},
    }

    def __init__(self, config=None, **params):
        super().__init__(config, **params)
        self.ema_fast_len = int(params.get("ema_fast", 21))
        self.ema_slow_len = int(params.get("ema_slow", 55))
        self.roc_len = int(params.get("roc_len", 20))
        self.structure_len = int(params.get("structure_len", 50))
        self.entry_score = int(params.get("entry_score", 3))
        self.atr_len = 14

    def compute_indicators(self, df):
        df["ema_fast"] = ind.ema(df["close"], self.ema_fast_len)
        df["ema_slow"] = ind.ema(df["close"], self.ema_slow_len)
        df["rsi"] = ind.rsi(df["close"], 14)
        df["roc"] = (df["close"] / df["close"].shift(self.roc_len) - 1) * 100
        df["atr"] = ind.atr(df["high"], df["low"], df["close"], self.atr_len)
        df["atr_sma"] = df["atr"].rolling(50, min_periods=1).mean()
        df["rolling_high"] = df["high"].rolling(self.structure_len, min_periods=1).max()
        df["rolling_low"] = df["low"].rolling(self.structure_len, min_periods=1).min()
        return df

    def generate_signals(self, df):
        score = pd.Series(0.0, index=df.index)

        # Pattern 1: Trend alignment (+1/-1)
        trend_bull = df["ema_fast"] > df["ema_slow"]
        score = score + trend_bull.astype(float) - (~trend_bull).astype(float)

        # Pattern 2: Momentum direction (+1/-1)
        mom_bull = df["roc"] > 0
        score = score + mom_bull.astype(float) - (~mom_bull).astype(float)

        # Pattern 3: RSI zones (+1 oversold, -1 overbought)
        score = score + (df["rsi"] < 30).astype(float)   # oversold = bullish
        score = score - (df["rsi"] > 70).astype(float)   # overbought = bearish

        # Pattern 4: Structure (+1 near low, -1 near high)
        price_range = (df["rolling_high"] - df["rolling_low"]).replace(0, np.nan)
        position_in_range = (df["close"] - df["rolling_low"]) / price_range
        score = score + (position_in_range < 0.2).astype(float)   # near low = bullish
        score = score - (position_in_range > 0.8).astype(float)   # near high = bearish

        # Pattern 5: Volatility expansion (+1/-1 follows trend direction)
        vol_expand = df["atr"] > df["atr_sma"] * 1.2
        score = score + (vol_expand & trend_bull).astype(float)
        score = score - (vol_expand & ~trend_bull).astype(float)

        df["pattern_score"] = score

        # Trigger on first bar score crosses threshold
        above_thresh = score >= self.entry_score
        below_thresh = score <= -self.entry_score
        df["long_signal"] = above_thresh & (~above_thresh.shift(1).fillna(False))
        df["short_signal"] = below_thresh & (~below_thresh.shift(1).fillna(False))

        return df

    def _execute(self, df):
        cfg = self.config
        for i in range(len(df)):
            row = df.iloc[i]
            date, close, high, low = row["date"], row["close"], row["high"], row["low"]
            atr_val = row["atr"] if not np.isnan(row["atr"]) else close * 0.02
            score = row["pattern_score"] if not np.isnan(row["pattern_score"]) else 0

            self.engine.update_position_tracking(high, low)

            if self.engine.position is not None:
                pos = self.engine.position
                bars_held = i - pos.entry_bar

                if pos.direction == "long":
                    long_stop = pos.entry_price - cfg.atr_stop_mult * atr_val
                    long_trail = pos.highest_since - cfg.atr_trail_mult * atr_val
                    eff_stop = max(long_stop, long_trail)
                    if low <= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "MP Long Stop")
                    elif score <= -self.entry_score:
                        self.engine.exit_position(i, date, close, "MP Score Flip")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "MP Time Exit")
                else:
                    short_stop = pos.entry_price + cfg.atr_stop_mult * atr_val
                    short_trail = pos.lowest_since + cfg.atr_trail_mult * atr_val
                    eff_stop = min(short_stop, short_trail)
                    if high >= eff_stop:
                        self.engine.exit_position(i, date, eff_stop, "MP Short Stop")
                    elif score >= self.entry_score:
                        self.engine.exit_position(i, date, close, "MP Score Flip")
                    elif bars_held >= cfg.max_bars_trend:
                        self.engine.exit_position(i, date, close, "MP Time Exit")

            if i >= cfg.warmup_bars and self.engine.position is None:
                atr_norm = atr_val / close * 100 if close > 0 else 1
                bar_vol_target = cfg.target_annual_vol / np.sqrt(365 * cfg.bars_per_day)
                size = min(100, max(5, bar_vol_target / atr_norm * 100)) if atr_norm > 0 else 10

                if row["long_signal"]:
                    self.engine.enter_position(i, date, close, "long", "MP", size)
                elif row["short_signal"] and cfg.short_size_mult > 0:
                    self.engine.enter_position(i, date, close, "short", "MP", size * cfg.short_size_mult)

            self.engine.record_equity(date, close)

    @classmethod
    def explainer(cls):
        return ("Multi Pattern scores 5 recurring patterns: EMA trend, ROC momentum, RSI zones, "
                "price structure (near high/low), and volatility expansion. "
                "Long when score >= threshold, short when <= -threshold. "
                "Designed to find repeating high-probability setups.")


# Registry of all strategies
STRATEGY_REGISTRY = {
    "T1": T1_TrendEMA,
    "M1": M1_MeanRevBBRSI,
    "H1": H1_SqueezeBreakout,
    "H2": H2_AdaptiveBlend,
    "RSI": RSI_Strategy,
    "MACD": MACD_Strategy,
    "SMA": SMA_Crossover,
    "BH": BuyAndHold,
    "MOM": MomentumROC,
    "DIP": BuyTheDip,
    "DON": DonchianBreakout,
    "BKD": BreakdownShort,
    "STR": SellTheRip,
    "MP":  MultiPattern,
}
