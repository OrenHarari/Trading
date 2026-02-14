"""
Aggressive 1H strategies optimized for 1-month BTC trading.
Target: >50% return in a single month using Long + Short.

These strategies are faster, more aggressive, and use tighter
timeframe indicators suited for hourly analysis.
"""

import numpy as np
import pandas as pd


# ============================================================
# INDICATOR HELPERS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, mult: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid + mult * std, mid, mid - mult * std

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d

def adx(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().where(lambda x: (x > -low.diff()) & (x > 0), 0.0)
    minus_dm = (-low.diff()).where(lambda x: (x > high.diff()) & (x > 0), 0.0)
    atr_val = atr(df, period)
    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period), plus_di, minus_di

def vwap_bands(df: pd.DataFrame, period: int = 20, mult: float = 2.0):
    """Volume-weighted price with bands."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (tp * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()
    std = tp.rolling(period).std()
    return vwap + mult * std, vwap, vwap - mult * std


# ============================================================
# STRATEGY 1: Scalp EMA Crossover (very fast, many trades)
# ============================================================
def strategy_scalp_ema(df: pd.DataFrame, fast: int = 5, mid: int = 13,
                       slow: int = 34, atr_mult: float = 1.2) -> pd.DataFrame:
    """Triple EMA crossover scalping. Aggressive entries on fast cross mid, slow as filter."""
    close = df["close"]
    ema_f = ema(close, fast)
    ema_m = ema(close, mid)
    ema_s = ema(close, slow)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: fast > mid > slow (aligned uptrend)
    long_cond = (ema_f.shift(1) > ema_m.shift(1)) & (ema_m.shift(1) > ema_s.shift(1))
    # Short: fast < mid < slow (aligned downtrend)
    short_cond = (ema_f.shift(1) < ema_m.shift(1)) & (ema_m.shift(1) < ema_s.shift(1))

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1
    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 2: RSI Momentum Breakout (catch strong moves)
# ============================================================
def strategy_rsi_momentum(df: pd.DataFrame, rsi_len: int = 7,
                          rsi_long_entry: float = 55, rsi_short_entry: float = 45,
                          rsi_long_exit: float = 75, rsi_short_exit: float = 25,
                          atr_mult: float = 1.5) -> pd.DataFrame:
    """RSI momentum: enter when RSI shows strength, exit at extremes."""
    close = df["close"]
    rsi_val = rsi(close, rsi_len)
    ema_20 = ema(close, 20)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: RSI > 55 and rising, price above EMA20
    long_cond = (rsi_val.shift(1) > rsi_long_entry) & (rsi_val.shift(1) > rsi_val.shift(2)) & (close.shift(1) > ema_20.shift(1))
    # Short: RSI < 45 and falling, price below EMA20
    short_cond = (rsi_val.shift(1) < rsi_short_entry) & (rsi_val.shift(1) < rsi_val.shift(2)) & (close.shift(1) < ema_20.shift(1))
    # Flat zones
    flat_long = rsi_val.shift(1) > rsi_long_exit
    flat_short = rsi_val.shift(1) < rsi_short_exit

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1
    signals.loc[flat_long & (signals["signal"] == 1), "signal"] = 0
    signals.loc[flat_short & (signals["signal"] == -1), "signal"] = 0

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 3: BB Squeeze Breakout (catch volatility expansion)
# ============================================================
def strategy_bb_squeeze(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0,
                        kc_len: int = 20, kc_mult: float = 1.5,
                        atr_mult: float = 1.8) -> pd.DataFrame:
    """Bollinger/Keltner squeeze breakout with MACD direction."""
    close = df["close"]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    kc_basis = ema(close, kc_len)
    kc_range = atr(df, kc_len) * kc_mult
    kc_upper = kc_basis + kc_range
    kc_lower = kc_basis - kc_range

    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    atr_val = atr(df, 10)

    # Squeeze: BB inside KC
    squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_count = squeeze.rolling(6).sum()
    was_squeezed = squeeze_count.shift(1) >= 4
    squeeze_released = ~squeeze & was_squeezed

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    long_cond = squeeze_released & (hist.shift(1) > 0) & (close.shift(1) > bb_mid.shift(1))
    short_cond = squeeze_released & (hist.shift(1) < 0) & (close.shift(1) < bb_mid.shift(1))

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 4: VWAP Bounce (institutional level trading)
# ============================================================
def strategy_vwap_bounce(df: pd.DataFrame, vwap_len: int = 20,
                         vwap_mult: float = 1.5, rsi_len: int = 7,
                         atr_mult: float = 1.3) -> pd.DataFrame:
    """Trade bounces off VWAP bands with RSI confirmation."""
    close = df["close"]
    vwap_upper, vwap_mid, vwap_lower = vwap_bands(df, vwap_len, vwap_mult)
    rsi_val = rsi(close, rsi_len)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: price bounces off lower VWAP band + RSI oversold
    long_cond = (close.shift(1) < vwap_lower.shift(1)) & (rsi_val.shift(1) < 30) & (close.shift(1) > close.shift(2))
    # Short: price rejected at upper VWAP band + RSI overbought
    short_cond = (close.shift(1) > vwap_upper.shift(1)) & (rsi_val.shift(1) > 70) & (close.shift(1) < close.shift(2))
    # Target midline
    at_mid_long = close.shift(1) > vwap_mid.shift(1)
    at_mid_short = close.shift(1) < vwap_mid.shift(1)

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 5: Multi-Indicator Confluence (high probability entries)
# ============================================================
def strategy_confluence(df: pd.DataFrame, ema_fast: int = 8, ema_slow: int = 21,
                        rsi_len: int = 14, stoch_len: int = 14,
                        atr_mult: float = 1.5) -> pd.DataFrame:
    """Enter only when EMA, RSI, Stochastic, and MACD all agree."""
    close = df["close"]
    ema_f = ema(close, ema_fast)
    ema_s = ema(close, ema_slow)
    rsi_val = rsi(close, rsi_len)
    k, d = stochastic(df, stoch_len)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: ALL must agree
    long_cond = (
        (ema_f.shift(1) > ema_s.shift(1)) &  # EMA bullish
        (rsi_val.shift(1) > 50) & (rsi_val.shift(1) < 75) &  # RSI bullish but not overbought
        (k.shift(1) > d.shift(1)) &  # Stochastic bullish
        (hist.shift(1) > 0)  # MACD bullish
    )
    # Short: ALL must agree
    short_cond = (
        (ema_f.shift(1) < ema_s.shift(1)) &
        (rsi_val.shift(1) < 50) & (rsi_val.shift(1) > 25) &
        (k.shift(1) < d.shift(1)) &
        (hist.shift(1) < 0)
    )

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 6: Aggressive Breakout (catch big moves fast)
# ============================================================
def strategy_aggressive_breakout(df: pd.DataFrame, lookback: int = 12,
                                  volume_mult: float = 1.5,
                                  atr_mult: float = 1.0) -> pd.DataFrame:
    """Trade breakouts of recent high/low with volume confirmation."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]
    atr_val = atr(df, 10)

    recent_high = high.rolling(lookback).max()
    recent_low = low.rolling(lookback).min()
    avg_vol = vol.rolling(lookback).mean()

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: close breaks above recent high with above-average volume
    long_cond = (close.shift(1) > recent_high.shift(2)) & (vol.shift(1) > avg_vol.shift(1) * volume_mult)
    # Short: close breaks below recent low with above-average volume
    short_cond = (close.shift(1) < recent_low.shift(2)) & (vol.shift(1) > avg_vol.shift(1) * volume_mult)

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# REGISTRY
# ============================================================

AGGRESSIVE_STRATEGIES = {
    "Scalp EMA": {
        "func": strategy_scalp_ema,
        "type": "Trend-Scalp",
        "params": {"fast": 5, "mid": 13, "slow": 34, "atr_mult": 1.2},
    },
    "RSI Momentum": {
        "func": strategy_rsi_momentum,
        "type": "Momentum",
        "params": {"rsi_len": 7, "rsi_long_entry": 55, "rsi_short_entry": 45, "rsi_long_exit": 75, "rsi_short_exit": 25, "atr_mult": 1.5},
    },
    "BB Squeeze": {
        "func": strategy_bb_squeeze,
        "type": "Breakout",
        "params": {"bb_len": 20, "bb_mult": 2.0, "kc_len": 20, "kc_mult": 1.5, "atr_mult": 1.8},
    },
    "VWAP Bounce": {
        "func": strategy_vwap_bounce,
        "type": "Mean-Reversion",
        "params": {"vwap_len": 20, "vwap_mult": 1.5, "rsi_len": 7, "atr_mult": 1.3},
    },
    "Confluence": {
        "func": strategy_confluence,
        "type": "Multi-Indicator",
        "params": {"ema_fast": 8, "ema_slow": 21, "rsi_len": 14, "stoch_len": 14, "atr_mult": 1.5},
    },
    "Aggressive Breakout": {
        "func": strategy_aggressive_breakout,
        "type": "Breakout",
        "params": {"lookback": 12, "volume_mult": 1.5, "atr_mult": 1.0},
    },
}


AGGRESSIVE_PARAM_VARIANTS = {
    "Scalp EMA": [
        {"fast": 3, "mid": 8, "slow": 21, "atr_mult": 1.0},
        {"fast": 5, "mid": 13, "slow": 34, "atr_mult": 1.2},
        {"fast": 5, "mid": 10, "slow": 25, "atr_mult": 0.8},
        {"fast": 3, "mid": 5, "slow": 13, "atr_mult": 0.7},
    ],
    "RSI Momentum": [
        {"rsi_len": 5, "rsi_long_entry": 55, "rsi_short_entry": 45, "rsi_long_exit": 80, "rsi_short_exit": 20, "atr_mult": 1.2},
        {"rsi_len": 7, "rsi_long_entry": 55, "rsi_short_entry": 45, "rsi_long_exit": 75, "rsi_short_exit": 25, "atr_mult": 1.5},
        {"rsi_len": 7, "rsi_long_entry": 52, "rsi_short_entry": 48, "rsi_long_exit": 70, "rsi_short_exit": 30, "atr_mult": 1.0},
        {"rsi_len": 10, "rsi_long_entry": 60, "rsi_short_entry": 40, "rsi_long_exit": 80, "rsi_short_exit": 20, "atr_mult": 1.5},
    ],
    "BB Squeeze": [
        {"bb_len": 15, "bb_mult": 1.5, "kc_len": 15, "kc_mult": 1.0, "atr_mult": 1.5},
        {"bb_len": 20, "bb_mult": 2.0, "kc_len": 20, "kc_mult": 1.5, "atr_mult": 1.8},
        {"bb_len": 10, "bb_mult": 2.0, "kc_len": 10, "kc_mult": 1.5, "atr_mult": 1.2},
    ],
    "VWAP Bounce": [
        {"vwap_len": 14, "vwap_mult": 1.0, "rsi_len": 5, "atr_mult": 1.0},
        {"vwap_len": 20, "vwap_mult": 1.5, "rsi_len": 7, "atr_mult": 1.3},
        {"vwap_len": 10, "vwap_mult": 2.0, "rsi_len": 7, "atr_mult": 1.5},
    ],
    "Confluence": [
        {"ema_fast": 5, "ema_slow": 15, "rsi_len": 7, "stoch_len": 7, "atr_mult": 1.2},
        {"ema_fast": 8, "ema_slow": 21, "rsi_len": 14, "stoch_len": 14, "atr_mult": 1.5},
        {"ema_fast": 10, "ema_slow": 30, "rsi_len": 14, "stoch_len": 21, "atr_mult": 2.0},
    ],
    "Aggressive Breakout": [
        {"lookback": 8, "volume_mult": 1.2, "atr_mult": 0.8},
        {"lookback": 12, "volume_mult": 1.5, "atr_mult": 1.0},
        {"lookback": 6, "volume_mult": 1.0, "atr_mult": 0.7},
        {"lookback": 16, "volume_mult": 2.0, "atr_mult": 1.2},
    ],
}
