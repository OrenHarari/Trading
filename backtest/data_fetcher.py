"""
Real BTC/USD data fetcher using public APIs.
Fetches OHLCV data from CoinGecko and Binance public APIs.
No API key required.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1d",
                         start_date: str = "2024-01-01", end_date: str = None) -> pd.DataFrame:
    """Fetch OHLCV data from Binance public API (no key needed).

    Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    url = "https://api.binance.com/api/v3/klines"

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    if end_date:
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    else:
        end_ts = int(datetime.now().timestamp() * 1000)

    all_data = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Failed to fetch data after 5 attempts: {e}")

        if not data:
            break

        all_data.extend(data)
        current_start = data[-1][0] + 1  # Next ms after last candle

        if len(data) < 1000:
            break

        time.sleep(0.2)  # Rate limit

    if not all_data:
        raise RuntimeError(f"No data fetched for {symbol} {interval}")

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df = df.set_index("datetime")
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def get_btc_data(interval: str = "1d", lookback_days: int = 730) -> pd.DataFrame:
    """Get BTC/USDT data for a given interval.

    Args:
        interval: '1d', '4h', '2h', '1h'
        lookback_days: How many days back to fetch
    """
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    cache_file = DATA_DIR / f"btc_{interval}_{lookback_days}d.csv"

    # Use cache if less than 12 hours old
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=12):
            print(f"  Loading cached data: {cache_file}")
            df = pd.read_csv(cache_file, index_col="datetime", parse_dates=True)
            return df

    print(f"  Fetching BTC {interval} data from Binance ({lookback_days} days)...")
    df = fetch_binance_klines("BTCUSDT", interval, start_date)

    # Save cache
    df.to_csv(cache_file)
    print(f"  Saved {len(df)} candles to {cache_file}")

    return df


def get_multi_timeframe_data(lookback_days: int = 730) -> dict:
    """Fetch BTC data for all required timeframes."""
    timeframes = {
        "1D": "1d",
        "4H": "4h",
        "2H": "2h",
    }

    data = {}
    for label, interval in timeframes.items():
        print(f"Fetching {label} data...")
        data[label] = get_btc_data(interval, lookback_days)
        print(f"  Got {len(data[label])} candles ({data[label].index[0]} to {data[label].index[-1]})")

    return data


if __name__ == "__main__":
    data = get_multi_timeframe_data()
    for tf, df in data.items():
        print(f"\n{tf}: {len(df)} bars, {df.index[0]} → {df.index[-1]}")
        print(f"  Price range: ${df['close'].min():.0f} – ${df['close'].max():.0f}")
