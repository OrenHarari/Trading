"""
5 Trading Strategy Models for BTC backtesting.

Each strategy returns entry/exit signals as a DataFrame with columns:
  - signal: 1 (long), -1 (short), 0 (flat)
  - stop_loss: price level for stop
  - take_profit: price level for TP (optional)

All strategies use ONLY past data — no lookahead bias.
"""

import numpy as np
import pandas as pd


# ============================================================
# INDICATOR HELPERS (all use past data only)
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(df, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)

    return adx_val, plus_di, minus_di


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


def donchian(df: pd.DataFrame, period: int):
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Supertrend indicator."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)

    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = max(lower_band.iloc[i],
                                     supertrend.iloc[i - 1] if direction.iloc[i - 1] == 1 else lower_band.iloc[i])
        else:
            supertrend.iloc[i] = min(upper_band.iloc[i],
                                     supertrend.iloc[i - 1] if direction.iloc[i - 1] == -1 else upper_band.iloc[i])

    return supertrend, direction


# ============================================================
# STRATEGY 1: EMA Crossover + ADX Trend (improved from T1)
# ============================================================

def strategy_ema_adx(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 21,
                     adx_threshold: float = 20, atr_stop: float = 2.0) -> pd.DataFrame:
    """EMA crossover filtered by ADX strength. Long + Short."""
    close = df["close"]
    ema_f = ema(close, fast_period)
    ema_s = ema(close, slow_period)
    adx_val, plus_di, minus_di = adx(df, 14)
    atr_val = atr(df, 14)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Use shift(1) for all conditions — no lookahead
    long_cond = (ema_f.shift(1) > ema_s.shift(1)) & (adx_val.shift(1) > adx_threshold) & (plus_di.shift(1) > minus_di.shift(1))
    short_cond = (ema_f.shift(1) < ema_s.shift(1)) & (adx_val.shift(1) > adx_threshold) & (minus_di.shift(1) > plus_di.shift(1))

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1
    signals.loc[~long_cond & ~short_cond, "signal"] = 0

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_stop * atr_val,
        np.where(signals["signal"] == -1, close + atr_stop * atr_val, np.nan)
    )

    return signals


# ============================================================
# STRATEGY 2: MACD + RSI Momentum
# ============================================================

def strategy_macd_rsi(df: pd.DataFrame, rsi_period: int = 14,
                      rsi_ob: float = 70, rsi_os: float = 30,
                      atr_stop: float = 2.5) -> pd.DataFrame:
    """MACD crossover confirmed by RSI. Long + Short."""
    close = df["close"]
    macd_line, signal_line, hist = macd(close)
    rsi_val = rsi(close, rsi_period)
    atr_val = atr(df, 14)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: MACD crosses above signal, RSI not overbought
    long_cond = (hist.shift(1) > 0) & (hist.shift(2) <= 0) & (rsi_val.shift(1) < rsi_ob) & (rsi_val.shift(1) > 40)
    # Short: MACD crosses below signal, RSI not oversold
    short_cond = (hist.shift(1) < 0) & (hist.shift(2) >= 0) & (rsi_val.shift(1) > rsi_os) & (rsi_val.shift(1) < 60)

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_stop * atr_val,
        np.where(signals["signal"] == -1, close + atr_stop * atr_val, np.nan)
    )

    return signals


# ============================================================
# STRATEGY 3: Bollinger Band Mean-Reversion + Stochastic
# ============================================================

def strategy_bb_stoch(df: pd.DataFrame, bb_period: int = 20, bb_mult: float = 2.0,
                      stoch_k: int = 14, atr_stop: float = 1.5) -> pd.DataFrame:
    """BB touch with stochastic confirmation. Long + Short."""
    close = df["close"]
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_period, bb_mult)
    k, d = stochastic(df, stoch_k)
    atr_val = atr(df, 14)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: price below lower BB + stochastic oversold
    long_cond = (close.shift(1) < bb_lower.shift(1)) & (k.shift(1) < 20)
    # Short: price above upper BB + stochastic overbought
    short_cond = (close.shift(1) > bb_upper.shift(1)) & (k.shift(1) > 80)
    # Exit to flat when price returns to midline
    at_mid = (close.shift(1) > bb_mid.shift(1) * 0.99) & (close.shift(1) < bb_mid.shift(1) * 1.01)

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_stop * atr_val,
        np.where(signals["signal"] == -1, close + atr_stop * atr_val, np.nan)
    )

    return signals


# ============================================================
# STRATEGY 4: Supertrend Trend-Following
# ============================================================

def strategy_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
                        atr_stop: float = 2.0) -> pd.DataFrame:
    """Supertrend direction-based trend-following. Long + Short."""
    close = df["close"]
    st_line, st_dir = supertrend(df, period, multiplier)
    atr_val = atr(df, 14)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Use direction from previous bar (no lookahead)
    signals["signal"] = st_dir.shift(1).fillna(0).astype(int)

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_stop * atr_val,
        np.where(signals["signal"] == -1, close + atr_stop * atr_val, np.nan)
    )

    return signals


# ============================================================
# STRATEGY 5: Donchian Breakout + MACD Confirmation (Hybrid)
# ============================================================

def strategy_donchian_macd(df: pd.DataFrame, entry_period: int = 20,
                           exit_period: int = 10, atr_stop: float = 2.5) -> pd.DataFrame:
    """Donchian channel breakout confirmed by MACD momentum. Long + Short."""
    close = df["close"]
    dc_upper, dc_mid, dc_lower = donchian(df, entry_period)
    exit_upper, _, exit_lower = donchian(df, exit_period)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    atr_val = atr(df, 14)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: breakout above Donchian upper + MACD positive
    long_cond = (close.shift(1) > dc_upper.shift(2)) & (hist.shift(1) > 0)
    # Short: breakdown below Donchian lower + MACD negative
    short_cond = (close.shift(1) < dc_lower.shift(2)) & (hist.shift(1) < 0)

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_stop * atr_val,
        np.where(signals["signal"] == -1, close + atr_stop * atr_val, np.nan)
    )

    return signals


# ============================================================
# REGISTRY
# ============================================================

STRATEGIES = {
    "EMA+ADX Trend": {
        "func": strategy_ema_adx,
        "type": "Trend-Following",
        "params": {"fast_period": 9, "slow_period": 21, "adx_threshold": 20, "atr_stop": 2.0},
    },
    "MACD+RSI Momentum": {
        "func": strategy_macd_rsi,
        "type": "Momentum",
        "params": {"rsi_period": 14, "rsi_ob": 70, "rsi_os": 30, "atr_stop": 2.5},
    },
    "BB+Stoch Reversion": {
        "func": strategy_bb_stoch,
        "type": "Mean-Reversion",
        "params": {"bb_period": 20, "bb_mult": 2.0, "stoch_k": 14, "atr_stop": 1.5},
    },
    "Supertrend": {
        "func": strategy_supertrend,
        "type": "Trend-Following",
        "params": {"period": 10, "multiplier": 3.0, "atr_stop": 2.0},
    },
    "Donchian+MACD": {
        "func": strategy_donchian_macd,
        "type": "Hybrid",
        "params": {"entry_period": 20, "exit_period": 10, "atr_stop": 2.5},
    },
}


# Parameter variations for optimization
PARAM_VARIANTS = {
    "EMA+ADX Trend": [
        {"fast_period": 8, "slow_period": 21, "adx_threshold": 18, "atr_stop": 1.5},
        {"fast_period": 9, "slow_period": 21, "adx_threshold": 20, "atr_stop": 2.0},
        {"fast_period": 12, "slow_period": 26, "adx_threshold": 22, "atr_stop": 2.0},
        {"fast_period": 9, "slow_period": 30, "adx_threshold": 18, "atr_stop": 2.5},
        {"fast_period": 5, "slow_period": 15, "adx_threshold": 15, "atr_stop": 1.5},
    ],
    "MACD+RSI Momentum": [
        {"rsi_period": 10, "rsi_ob": 65, "rsi_os": 35, "atr_stop": 2.0},
        {"rsi_period": 14, "rsi_ob": 70, "rsi_os": 30, "atr_stop": 2.5},
        {"rsi_period": 14, "rsi_ob": 75, "rsi_os": 25, "atr_stop": 3.0},
        {"rsi_period": 7, "rsi_ob": 70, "rsi_os": 30, "atr_stop": 2.0},
    ],
    "BB+Stoch Reversion": [
        {"bb_period": 15, "bb_mult": 1.5, "stoch_k": 10, "atr_stop": 1.5},
        {"bb_period": 20, "bb_mult": 2.0, "stoch_k": 14, "atr_stop": 1.5},
        {"bb_period": 20, "bb_mult": 2.5, "stoch_k": 14, "atr_stop": 2.0},
        {"bb_period": 30, "bb_mult": 2.0, "stoch_k": 21, "atr_stop": 2.0},
    ],
    "Supertrend": [
        {"period": 7, "multiplier": 2.0, "atr_stop": 1.5},
        {"period": 10, "multiplier": 3.0, "atr_stop": 2.0},
        {"period": 10, "multiplier": 2.0, "atr_stop": 2.0},
        {"period": 14, "multiplier": 3.0, "atr_stop": 2.5},
        {"period": 7, "multiplier": 3.0, "atr_stop": 2.0},
    ],
    "Donchian+MACD": [
        {"entry_period": 15, "exit_period": 7, "atr_stop": 2.0},
        {"entry_period": 20, "exit_period": 10, "atr_stop": 2.5},
        {"entry_period": 30, "exit_period": 15, "atr_stop": 3.0},
        {"entry_period": 10, "exit_period": 5, "atr_stop": 1.5},
    ],
}
