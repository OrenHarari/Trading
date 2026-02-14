#!/usr/bin/env python3
"""Quick test: can we run a single strategy backtest?"""
import sys, traceback
sys.path.insert(0, '.')

try:
    from backtester.engine import BacktestConfig
    from backtester.strategies import STRATEGY_REGISTRY
    import pandas as pd
    
    df = pd.read_csv('data/BTC-USD_2h.csv')
    df['date'] = pd.to_datetime(df['date'])
    df_is = df.iloc[:2916].copy().reset_index(drop=True)
    
    cfg = BacktestConfig()
    cfg.bars_per_day = 12
    cfg.warmup_bars = 72
    cfg.target_annual_vol = 80
    cfg.atr_stop_mult = 2.5
    cfg.atr_trail_mult = 4.0
    cfg.short_size_mult = 1.0
    cfg.max_bars_trend = 180
    
    strat = STRATEGY_REGISTRY['T1'](cfg, ema_fast=8, ema_slow=50, adx_threshold=15)
    result = strat.run(df_is)
    
    with open('test_quick_output.txt', 'w') as f:
        if 'error' in result:
            f.write(f'ERROR: {result["error"]}\n')
        else:
            f.write(f'OK: return={result["total_return_pct"]:+.1f}% trades={result["total_trades"]}\n')
        f.write('TEST COMPLETE\n')
except Exception as e:
    with open('test_quick_output.txt', 'w') as f:
        f.write(f'EXCEPTION: {e}\n')
        traceback.print_exc(file=f)
