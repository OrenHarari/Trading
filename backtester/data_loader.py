"""
Data loader for BTC/USD and ETH/USD daily price data.

Supports:
- Yahoo Finance (real OHLC via yfinance)
- CoinGecko CSV (close-only, synthesized OHLC)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Asset Registry ──────────────────────────────────────
ASSETS = {
    "BTC-USD": {"name": "Bitcoin", "symbol": "BTC-USD", "source": "yahoo"},
    "ETH-USD": {"name": "Ethereum", "symbol": "ETH-USD", "source": "yahoo"},
    "CSV":     {"name": "BTC (CSV File)", "symbol": "CSV", "source": "csv"},
}


def load_data(source: str = "BTC-USD", start_date: str = None,
              end_date: str = None, csv_path: str = None) -> pd.DataFrame:
    """Universal data loader — Yahoo Finance or CSV.

    Args:
        source: Asset key from ASSETS registry (e.g. "BTC-USD", "ETH-USD", "CSV")
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        csv_path: Path to CSV file (used when source="CSV")

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if source == "CSV":
        path = csv_path or "btc-usd-max (1).csv"
        return load_csv_data(path, start_date, end_date)
    else:
        return fetch_yahoo_ohlc(source, start_date, end_date)


def fetch_yahoo_ohlc(symbol: str = "BTC-USD", start_date: str = None,
                     end_date: str = None) -> pd.DataFrame:
    """Fetch real OHLCV data from Yahoo Finance with local caching.

    Args:
        symbol: Yahoo Finance ticker (e.g. "BTC-USD", "ETH-USD")
        start_date: Start date (YYYY-MM-DD), defaults to 2014-01-01
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    import yfinance as yf

    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{symbol}_daily.csv"

    # Try cache first (refresh if older than 24 hours)
    if cache_file.exists():
        import time
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return _filter_dates(df, start_date, end_date)

    # Download full history from Yahoo Finance (always fetch max range, filter later)
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start="2014-01-01", end=pd.Timestamp.now().strftime("%Y-%m-%d"), interval="1d")

    if hist.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {symbol}")

    df = pd.DataFrame({
        "date": hist.index.tz_localize(None) if hist.index.tz else hist.index,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    })

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)

    return _filter_dates(df, start_date, end_date)


def load_csv_data(filepath: str, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
    """Load price data from CoinGecko CSV export (close-only, synthesizes OHLC).

    Args:
        filepath: Path to the CSV file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse date
    if "snapped_at" in df.columns:
        df["date"] = pd.to_datetime(df["snapped_at"], utc=True).dt.tz_localize(None)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    else:
        raise ValueError("No date column found in CSV")

    # Get price
    if "price" in df.columns:
        df["close"] = df["price"].astype(float)
    elif "close" in df.columns:
        df["close"] = df["close"].astype(float)
    else:
        raise ValueError("No price/close column found in CSV")

    # Volume
    if "total_volume" in df.columns:
        df["volume"] = df["total_volume"].astype(float)
    elif "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0

    # Synthesize OHLC from close-only data
    df = df.sort_values("date").reset_index(drop=True)
    returns = df["close"].pct_change()
    rolling_vol = returns.rolling(20, min_periods=1).std().fillna(0.01)

    df["high"] = df["close"] * (1 + rolling_vol * 0.5)
    df["low"] = df["close"] * (1 - rolling_vol * 0.5)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    result = df[["date", "open", "high", "low", "close", "volume"]].copy()
    result = result.reset_index(drop=True)

    return _filter_dates(result, start_date, end_date)


# Legacy alias for backward compatibility
def load_btc_data(filepath: str, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
    """Legacy wrapper — loads CoinGecko CSV."""
    return load_csv_data(filepath, start_date, end_date)


def _filter_dates(df: pd.DataFrame, start_date: str = None,
                  end_date: str = None) -> pd.DataFrame:
    """Apply date filters and return clean DataFrame."""
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df.reset_index(drop=True)


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics for the loaded data."""
    return {
        "total_bars": len(df),
        "start_date": str(df["date"].iloc[0].date()) if len(df) > 0 else "",
        "end_date": str(df["date"].iloc[-1].date()) if len(df) > 0 else "",
        "start_price": round(df["close"].iloc[0], 2) if len(df) > 0 else 0,
        "end_price": round(df["close"].iloc[-1], 2) if len(df) > 0 else 0,
        "total_return_pct": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2) if len(df) > 1 else 0,
        "max_price": round(df["close"].max(), 2) if len(df) > 0 else 0,
        "min_price": round(df["close"].min(), 2) if len(df) > 0 else 0,
    }
