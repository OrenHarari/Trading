"""
Technical indicators library.

Mirrors the Pine Script indicators used in main_strategy.pine.
All functions operate on pandas Series/DataFrames.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(period, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def bollinger_bands(close: pd.Series, period: int = 20, mult: float = 2.0):
    """Bollinger Bands. Returns (mid, upper, lower, width)."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=1).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid
    return mid, upper, lower, width


def keltner_channels(close: pd.Series, high: pd.Series, low: pd.Series,
                     period: int = 20, mult: float = 1.5):
    """Keltner Channels. Returns (basis, upper, lower)."""
    basis = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = basis + mult * atr_val
    lower = basis - mult * atr_val
    return basis, upper, lower


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = np.where((high - prev_high) > (prev_low - low),
                       np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0), 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr_val = atr(high, low, close, period)

    plus_di = 100 * ema(plus_dm, period) / atr_val.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = ema(dx, period)

    return adx_val


def linear_regression_slope(close: pd.Series, period: int = 20) -> pd.Series:
    """Linear regression slope over rolling window."""
    def _slope(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        coeffs = np.polyfit(x, window, 1)
        return coeffs[0]

    return close.rolling(period, min_periods=2).apply(_slope, raw=True)


def percentile_rank(series: pd.Series, period: int = 100) -> pd.Series:
    """Percentile rank of current value within lookback window."""
    def _pctile(window):
        if len(window) < 2:
            return 50.0
        current = window.iloc[-1]
        count = (window.iloc[:-1] < current).sum()
        return count / (len(window) - 1) * 100.0

    return series.rolling(period, min_periods=2).apply(_pctile)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD (Moving Average Convergence Divergence).

    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def vats(close: pd.Series, atr_val: pd.Series, lookback: int = 20) -> pd.Series:
    """Volatility-Adjusted Trend Strength.
    
    Measures price displacement relative to expected displacement.
    VATS > 1.5 = trending, VATS < 1.0 = ranging
    """
    displacement = (close - close.shift(lookback)).abs()
    expected = atr_val * np.sqrt(lookback)
    expected = expected.replace(0, np.nan)
    return displacement / expected
