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
# STRATEGY 7: Trend Pullback (buy dips in uptrends)
# ============================================================
def strategy_trend_pullback(df: pd.DataFrame, trend_ema: int = 50,
                            fast_ema: int = 9, rsi_len: int = 7,
                            rsi_oversold: float = 35, rsi_overbought: float = 65,
                            atr_mult: float = 1.5) -> pd.DataFrame:
    """Buy dips in uptrends, sell rallies in downtrends. High win-rate approach."""
    close = df["close"]
    ema_trend = ema(close, trend_ema)
    ema_fast = ema(close, fast_ema)
    rsi_val = rsi(close, rsi_len)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Long: price above trend EMA (uptrend) + RSI dipped below oversold + now recovering
    uptrend = close.shift(1) > ema_trend.shift(1)
    rsi_dip = (rsi_val.shift(2) < rsi_oversold) & (rsi_val.shift(1) > rsi_val.shift(2))
    price_above_fast = close.shift(1) > ema_fast.shift(1)

    # Short: price below trend EMA (downtrend) + RSI popped above overbought + now declining
    downtrend = close.shift(1) < ema_trend.shift(1)
    rsi_pop = (rsi_val.shift(2) > rsi_overbought) & (rsi_val.shift(1) < rsi_val.shift(2))
    price_below_fast = close.shift(1) < ema_fast.shift(1)

    long_cond = uptrend & rsi_dip
    short_cond = downtrend & rsi_pop

    # Also enter long if fast EMA crosses above trend EMA (trend start)
    ema_cross_up = (ema_fast.shift(1) > ema_trend.shift(1)) & (ema_fast.shift(2) <= ema_trend.shift(2))
    ema_cross_down = (ema_fast.shift(1) < ema_trend.shift(1)) & (ema_fast.shift(2) >= ema_trend.shift(2))

    signals.loc[long_cond | ema_cross_up, "signal"] = 1
    signals.loc[short_cond | ema_cross_down, "signal"] = -1

    # Exit when trend reverses
    signals.loc[downtrend & (signals["signal"] == 1), "signal"] = 0
    signals.loc[uptrend & (signals["signal"] == -1), "signal"] = 0

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 8: Momentum Acceleration (catch early big moves)
# ============================================================
def strategy_momentum_accel(df: pd.DataFrame, macd_fast: int = 8,
                            macd_slow: int = 21, macd_sig: int = 5,
                            adx_len: int = 10, adx_thresh: float = 20,
                            atr_mult: float = 1.3) -> pd.DataFrame:
    """Enter when MACD histogram accelerates with ADX confirming trend strength."""
    close = df["close"]
    macd_line, signal_line, hist = macd(close, macd_fast, macd_slow, macd_sig)
    adx_val, plus_di, minus_di = adx(df, adx_len)
    atr_val = atr(df, 10)
    ema_20 = ema(close, 20)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # MACD histogram acceleration (increasing positive = bullish momentum)
    hist_accel = hist.shift(1) > hist.shift(2)
    hist_positive = hist.shift(1) > 0
    hist_negative = hist.shift(1) < 0
    hist_decel = hist.shift(1) < hist.shift(2)

    # ADX showing trend strength
    adx_strong = adx_val.shift(1) > adx_thresh

    # Long: histogram positive + accelerating + ADX strong + price above EMA
    long_cond = hist_positive & hist_accel & adx_strong & (close.shift(1) > ema_20.shift(1))
    # Short: histogram negative + accelerating downward + ADX strong + price below EMA
    short_cond = hist_negative & hist_decel & adx_strong & (close.shift(1) < ema_20.shift(1))

    # Exit on momentum loss
    long_exit = hist_negative | (hist_positive & hist_decel & (hist.shift(1) < hist.shift(1).rolling(5).mean()))
    short_exit = hist_positive | (hist_negative & hist_accel & (hist.shift(1) > hist.shift(1).rolling(5).mean()))

    signals.loc[long_cond, "signal"] = 1
    signals.loc[short_cond, "signal"] = -1
    signals.loc[long_exit & (signals["signal"] == 1), "signal"] = 0
    signals.loc[short_exit & (signals["signal"] == -1), "signal"] = 0

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 9: Adaptive Regime (switch trend/MR based on ADX)
# ============================================================
def strategy_adaptive_regime(df: pd.DataFrame, adx_len: int = 14,
                             adx_trend_thresh: float = 25, adx_range_thresh: float = 18,
                             ema_fast: int = 9, ema_slow: int = 21,
                             bb_len: int = 20, rsi_len: int = 7,
                             atr_mult: float = 1.5) -> pd.DataFrame:
    """Switch between trend-following (high ADX) and mean-reversion (low ADX)."""
    close = df["close"]
    adx_val, plus_di, minus_di = adx(df, adx_len)
    ema_f = ema(close, ema_fast)
    ema_s = ema(close, ema_slow)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_len)
    rsi_val = rsi(close, rsi_len)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    is_trending = adx_val.shift(1) > adx_trend_thresh
    is_ranging = adx_val.shift(1) < adx_range_thresh

    # TREND MODE: EMA crossover
    trend_long = is_trending & (ema_f.shift(1) > ema_s.shift(1)) & (plus_di.shift(1) > minus_di.shift(1))
    trend_short = is_trending & (ema_f.shift(1) < ema_s.shift(1)) & (minus_di.shift(1) > plus_di.shift(1))

    # RANGE MODE: BB bounce with RSI
    range_long = is_ranging & (close.shift(1) < bb_lower.shift(1)) & (rsi_val.shift(1) < 30)
    range_short = is_ranging & (close.shift(1) > bb_upper.shift(1)) & (rsi_val.shift(1) > 70)

    signals.loc[trend_long | range_long, "signal"] = 1
    signals.loc[trend_short | range_short, "signal"] = -1

    # Exit conditions per regime
    # Trend exits: when EMA crosses back
    trend_exit_long = is_trending & (ema_f.shift(1) < ema_s.shift(1))
    trend_exit_short = is_trending & (ema_f.shift(1) > ema_s.shift(1))
    # Range exits: when price returns to BB midline
    range_exit_long = is_ranging & (close.shift(1) > bb_mid.shift(1)) & (signals["signal"] == 1)
    range_exit_short = is_ranging & (close.shift(1) < bb_mid.shift(1)) & (signals["signal"] == -1)

    signals.loc[(trend_exit_long | range_exit_long) & (signals["signal"] == 1), "signal"] = 0
    signals.loc[(trend_exit_short | range_exit_short) & (signals["signal"] == -1), "signal"] = 0

    signals["stop_loss"] = np.where(
        signals["signal"] == 1, close - atr_mult * atr_val,
        np.where(signals["signal"] == -1, close + atr_mult * atr_val, np.nan))
    return signals


# ============================================================
# STRATEGY 10: Ensemble Vote (combine multiple indicators)
# ============================================================
def strategy_ensemble_vote(df: pd.DataFrame, ema_fast: int = 9,
                           ema_slow: int = 21, rsi_len: int = 7,
                           macd_fast: int = 12, macd_slow: int = 26,
                           stoch_len: int = 14, bb_len: int = 20,
                           min_votes: int = 3, atr_mult: float = 1.3) -> pd.DataFrame:
    """Enter only when >=min_votes indicators agree on direction. High conviction."""
    close = df["close"]
    ema_f = ema(close, ema_fast)
    ema_s = ema(close, ema_slow)
    rsi_val = rsi(close, rsi_len)
    macd_line, signal_line, hist = macd(close, macd_fast, macd_slow, 9)
    k, d = stochastic(df, stoch_len)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_len)
    atr_val = atr(df, 10)

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals["stop_loss"] = np.nan

    # Vote counting (each is +1 for long, -1 for short)
    v_ema = pd.Series(0, index=df.index)
    v_ema[ema_f.shift(1) > ema_s.shift(1)] = 1
    v_ema[ema_f.shift(1) < ema_s.shift(1)] = -1

    v_rsi = pd.Series(0, index=df.index)
    v_rsi[(rsi_val.shift(1) > 50) & (rsi_val.shift(1) < 80)] = 1
    v_rsi[(rsi_val.shift(1) < 50) & (rsi_val.shift(1) > 20)] = -1

    v_macd = pd.Series(0, index=df.index)
    v_macd[hist.shift(1) > 0] = 1
    v_macd[hist.shift(1) < 0] = -1

    v_stoch = pd.Series(0, index=df.index)
    v_stoch[(k.shift(1) > d.shift(1)) & (k.shift(1) < 80)] = 1
    v_stoch[(k.shift(1) < d.shift(1)) & (k.shift(1) > 20)] = -1

    v_bb = pd.Series(0, index=df.index)
    v_bb[close.shift(1) > bb_mid.shift(1)] = 1
    v_bb[close.shift(1) < bb_mid.shift(1)] = -1

    total_vote = v_ema + v_rsi + v_macd + v_stoch + v_bb

    signals.loc[total_vote >= min_votes, "signal"] = 1
    signals.loc[total_vote <= -min_votes, "signal"] = -1

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
    "Trend Pullback": {
        "func": strategy_trend_pullback,
        "type": "Trend-Pullback",
        "params": {"trend_ema": 50, "fast_ema": 9, "rsi_len": 7, "rsi_oversold": 35, "rsi_overbought": 65, "atr_mult": 1.5},
    },
    "Momentum Accel": {
        "func": strategy_momentum_accel,
        "type": "Momentum",
        "params": {"macd_fast": 8, "macd_slow": 21, "macd_sig": 5, "adx_len": 10, "adx_thresh": 20, "atr_mult": 1.3},
    },
    "Adaptive Regime": {
        "func": strategy_adaptive_regime,
        "type": "Adaptive",
        "params": {"adx_len": 14, "adx_trend_thresh": 25, "adx_range_thresh": 18, "ema_fast": 9, "ema_slow": 21, "bb_len": 20, "rsi_len": 7, "atr_mult": 1.5},
    },
    "Ensemble Vote": {
        "func": strategy_ensemble_vote,
        "type": "Ensemble",
        "params": {"ema_fast": 9, "ema_slow": 21, "rsi_len": 7, "macd_fast": 12, "macd_slow": 26, "stoch_len": 14, "bb_len": 20, "min_votes": 3, "atr_mult": 1.3},
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
    "Trend Pullback": [
        {"trend_ema": 30, "fast_ema": 5, "rsi_len": 5, "rsi_oversold": 30, "rsi_overbought": 70, "atr_mult": 1.2},
        {"trend_ema": 50, "fast_ema": 9, "rsi_len": 7, "rsi_oversold": 35, "rsi_overbought": 65, "atr_mult": 1.5},
        {"trend_ema": 40, "fast_ema": 8, "rsi_len": 7, "rsi_oversold": 40, "rsi_overbought": 60, "atr_mult": 1.0},
        {"trend_ema": 20, "fast_ema": 5, "rsi_len": 5, "rsi_oversold": 30, "rsi_overbought": 70, "atr_mult": 0.8},
    ],
    "Momentum Accel": [
        {"macd_fast": 5, "macd_slow": 15, "macd_sig": 3, "adx_len": 7, "adx_thresh": 18, "atr_mult": 1.0},
        {"macd_fast": 8, "macd_slow": 21, "macd_sig": 5, "adx_len": 10, "adx_thresh": 20, "atr_mult": 1.3},
        {"macd_fast": 12, "macd_slow": 26, "macd_sig": 9, "adx_len": 14, "adx_thresh": 22, "atr_mult": 1.5},
        {"macd_fast": 6, "macd_slow": 18, "macd_sig": 4, "adx_len": 8, "adx_thresh": 15, "atr_mult": 0.8},
    ],
    "Adaptive Regime": [
        {"adx_len": 10, "adx_trend_thresh": 22, "adx_range_thresh": 15, "ema_fast": 5, "ema_slow": 15, "bb_len": 15, "rsi_len": 5, "atr_mult": 1.2},
        {"adx_len": 14, "adx_trend_thresh": 25, "adx_range_thresh": 18, "ema_fast": 9, "ema_slow": 21, "bb_len": 20, "rsi_len": 7, "atr_mult": 1.5},
        {"adx_len": 10, "adx_trend_thresh": 20, "adx_range_thresh": 15, "ema_fast": 8, "ema_slow": 21, "bb_len": 15, "rsi_len": 7, "atr_mult": 1.0},
    ],
    "Ensemble Vote": [
        {"ema_fast": 5, "ema_slow": 15, "rsi_len": 5, "macd_fast": 8, "macd_slow": 21, "stoch_len": 7, "bb_len": 15, "min_votes": 3, "atr_mult": 1.0},
        {"ema_fast": 9, "ema_slow": 21, "rsi_len": 7, "macd_fast": 12, "macd_slow": 26, "stoch_len": 14, "bb_len": 20, "min_votes": 3, "atr_mult": 1.3},
        {"ema_fast": 5, "ema_slow": 13, "rsi_len": 5, "macd_fast": 8, "macd_slow": 17, "stoch_len": 9, "bb_len": 15, "min_votes": 4, "atr_mult": 1.5},
        {"ema_fast": 9, "ema_slow": 21, "rsi_len": 7, "macd_fast": 12, "macd_slow": 26, "stoch_len": 14, "bb_len": 20, "min_votes": 2, "atr_mult": 1.0},
    ],
}
