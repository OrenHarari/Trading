#!/usr/bin/env python3
"""
1H Monthly Aggressive Backtest Runner.

Targets: >50% return in a single month using 1H candles.
Tests 6 aggressive strategies with Long + Short.
Generates HTML dashboard with buy/sell signal chart.

Usage:
    python backtest/run_1h_monthly.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from strategies_1h import AGGRESSIVE_STRATEGIES, AGGRESSIVE_PARAM_VARIANTS
from engine import run_backtest, INITIAL_CAPITAL

TARGET_PROFIT_PCT = 50.0
RESULTS_DIR = Path(__file__).parent.parent / "results" / "monthly_1h"


def generate_1h_data_for_months(seed: int = 42) -> dict:
    """Generate 1H BTC data for multiple recent months.

    Returns dict keyed by month label, each containing a DataFrame.
    """
    from generate_data import ANCHOR_POINTS_DAILY, interpolate_prices, add_realistic_noise, generate_ohlcv

    # Generate full daily data first
    interp = interpolate_prices(ANCHOR_POINTS_DAILY, freq="1D")
    noisy = add_realistic_noise(interp, daily_vol=0.025, seed=seed)
    df_1d = generate_ohlcv(noisy, seed=seed)

    # Resample to 1H (24 bars per day)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for idx, row in df_1d.iterrows():
        daily_open = row["open"]
        daily_close = row["close"]
        daily_high = row["high"]
        daily_low = row["low"]
        daily_volume = row["volume"]
        n_bars = 24

        sub_returns = rng.normal(0, 1, n_bars)
        sub_returns = sub_returns / (sub_returns.sum() or 1) * (daily_close / daily_open - 1)

        sub_close = daily_open
        for j in range(n_bars):
            sub_open = sub_close
            sub_close = sub_open * (1 + sub_returns[j])

            sub_range = abs(sub_close - sub_open) + abs(rng.normal(0, 0.0015)) * sub_open
            if sub_close >= sub_open:
                sub_high = max(sub_open, sub_close) + sub_range * rng.uniform(0, 0.5)
                sub_low = min(sub_open, sub_close) - sub_range * rng.uniform(0, 0.3)
            else:
                sub_high = max(sub_open, sub_close) + sub_range * rng.uniform(0, 0.3)
                sub_low = min(sub_open, sub_close) - sub_range * rng.uniform(0, 0.5)

            sub_high = min(sub_high, daily_high)
            sub_low = max(sub_low, daily_low)

            ts = idx + pd.Timedelta(hours=j)
            rows.append({
                "datetime": ts,
                "open": sub_open,
                "high": max(sub_high, sub_open, sub_close),
                "low": min(sub_low, sub_open, sub_close),
                "close": sub_close,
                "volume": daily_volume / n_bars * rng.uniform(0.3, 2.0),
            })

    df_1h = pd.DataFrame(rows).set_index("datetime")

    # Split into monthly windows
    months = {}
    # Last 6 months of data
    month_starts = [
        ("2025-09", "2025-09-01", "2025-09-30"),
        ("2025-10", "2025-10-01", "2025-10-31"),
        ("2025-11", "2025-11-01", "2025-11-30"),
        ("2025-12", "2025-12-01", "2025-12-31"),
        ("2026-01", "2026-01-01", "2026-01-31"),
        ("2026-02", "2026-02-01", "2026-02-14"),
    ]

    for label, start, end in month_starts:
        mask = (df_1h.index >= start) & (df_1h.index <= end)
        month_df = df_1h[mask]
        if len(month_df) > 50:  # Need minimum bars
            months[label] = month_df
            print(f"  {label}: {len(month_df)} bars, "
                  f"${month_df['close'].iloc[0]:,.0f} -> ${month_df['close'].iloc[-1]:,.0f}")

    return months, df_1h


def run_monthly_backtests(months: dict) -> list:
    """Run all aggressive strategies on each month."""
    all_results = []

    for month_label, df in months.items():
        print(f"\n{'='*70}")
        print(f"MONTH: {month_label} | {len(df)} bars | "
              f"${df['close'].iloc[0]:,.0f} -> ${df['close'].iloc[-1]:,.0f} | "
              f"B&H: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.1f}%")
        print(f"{'='*70}")

        for strat_name, strat_info in AGGRESSIVE_STRATEGIES.items():
            # Default params
            try:
                signals = strat_info["func"](df, **strat_info["params"])
                result = run_backtest(df, signals, position_size_pct=95.0)
                result["strategy_name"] = strat_name
                result["strategy_type"] = strat_info["type"]
                result["params"] = strat_info["params"]
                result["month"] = month_label
                result["timeframe"] = "1H"
                result["variant"] = "default"
                result["signals"] = signals  # Keep for chart
                m = result["metrics"]

                print(f"  {strat_name:<22} Ret={m['total_return_pct']:>7.1f}% "
                      f"PF={m['profit_factor']:.2f} DD={m['max_drawdown_pct']:.1f}% "
                      f"Trades={m['total_trades']} W={m['win_rate_pct']:.0f}% "
                      f"L={m['long_return_pct']:.1f}%/S={m['short_return_pct']:.1f}%")
                all_results.append(result)
            except Exception as e:
                print(f"  {strat_name:<22} ERROR: {e}")

            # Parameter variants
            if strat_name in AGGRESSIVE_PARAM_VARIANTS:
                for idx, params in enumerate(AGGRESSIVE_PARAM_VARIANTS[strat_name]):
                    try:
                        signals = strat_info["func"](df, **params)
                        result = run_backtest(df, signals, position_size_pct=95.0)
                        result["strategy_name"] = strat_name
                        result["strategy_type"] = strat_info["type"]
                        result["params"] = params
                        result["month"] = month_label
                        result["timeframe"] = "1H"
                        result["variant"] = f"v{idx+1}"
                        result["signals"] = signals
                        m = result["metrics"]

                        if m["total_return_pct"] > 10:  # Only print notable ones
                            print(f"    v{idx+1}: Ret={m['total_return_pct']:>7.1f}% "
                                  f"PF={m['profit_factor']:.2f} Trades={m['total_trades']}")
                        all_results.append(result)
                    except Exception as e:
                        pass  # Silently skip failed variants

    return all_results


def generate_signal_chart_html(best_result: dict, df_full: pd.DataFrame) -> str:
    """Generate HTML with interactive price chart and buy/sell signals."""
    month = best_result["month"]
    strat = best_result["strategy_name"]
    trades = best_result["trades"]

    # Get month data
    mask = df_full.index.strftime("%Y-%m").str.startswith(month[:7])
    month_df = df_full[mask]

    # Downsample for chart performance
    step = max(1, len(month_df) // 600)
    chart_df = month_df.iloc[::step]

    dates = [str(d) for d in chart_df.index]
    prices = [round(p, 2) for p in chart_df["close"]]

    # Build buy/sell markers from trades
    buy_markers = []
    sell_markers = []
    annotations = []

    if len(trades) > 0:
        for _, trade in trades.iterrows():
            entry_date = str(trade["entry_date"])
            exit_date = str(trade["exit_date"])
            direction = trade["direction"]
            pnl = trade.get("pnl_pct", 0)

            if direction == "LONG":
                buy_markers.append({"x": entry_date, "y": trade["entry_price"], "label": f"BUY ${trade['entry_price']:,.0f}"})
                sell_markers.append({"x": exit_date, "y": trade["exit_price"], "label": f"SELL ${trade['exit_price']:,.0f} ({pnl:+.1f}%)"})
            else:
                sell_markers.append({"x": entry_date, "y": trade["entry_price"], "label": f"SHORT ${trade['entry_price']:,.0f}"})
                buy_markers.append({"x": exit_date, "y": trade["exit_price"], "label": f"COVER ${trade['exit_price']:,.0f} ({pnl:+.1f}%)"})

    buy_data = json.dumps(buy_markers)
    sell_data = json.dumps(sell_markers)
    m = best_result["metrics"]

    chart_id = f"signalChart_{hash(strat + month) % 100000}"
    return f"""
    <div class="card">
        <h2>Buy/Sell Signals: {strat} ({month}) — {m['total_return_pct']:.1f}% Return</h2>
        <div style="display:flex;gap:20px;margin-bottom:15px;flex-wrap:wrap;">
            <span style="color:#00e676">&#9650; BUY/COVER = Green triangles</span>
            <span style="color:#ff5252">&#9660; SELL/SHORT = Red triangles</span>
            <span>Trades: {m['total_trades']} | Win: {m['win_rate_pct']:.0f}% | PF: {m['profit_factor']:.2f}</span>
        </div>
        <div class="chart-container" style="height:500px;">
            <canvas id="{chart_id}"></canvas>
        </div>
    </div>

    <script>
    (function() {{
        const ctx = document.getElementById('{chart_id}').getContext('2d');
        const dates = {json.dumps(dates)};
        const prices = {json.dumps(prices)};
        const buys = {buy_data};
        const sells = {sell_data};

        // Create buy/sell point arrays aligned with dates
        const buyPoints = new Array(dates.length).fill(null);
        const sellPoints = new Array(dates.length).fill(null);

        buys.forEach(b => {{
            const idx = dates.findIndex(d => d >= b.x);
            if (idx >= 0) buyPoints[idx] = b.y;
        }});
        sells.forEach(s => {{
            const idx = dates.findIndex(d => d >= s.x);
            if (idx >= 0) sellPoints[idx] = s.y;
        }});

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: dates,
                datasets: [
                    {{
                        label: 'BTC Price',
                        data: prices,
                        borderColor: '#448aff',
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        order: 2
                    }},
                    {{
                        label: 'BUY / COVER',
                        data: buyPoints,
                        backgroundColor: '#00e676',
                        borderColor: '#00e676',
                        pointRadius: 8,
                        pointStyle: 'triangle',
                        showLine: false,
                        order: 1
                    }},
                    {{
                        label: 'SELL / SHORT',
                        data: sellPoints,
                        backgroundColor: '#ff5252',
                        borderColor: '#ff5252',
                        pointRadius: 8,
                        pointStyle: 'triangle',
                        pointRotation: 180,
                        showLine: false,
                        order: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#e0e0e0' }} }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                if (ctx.raw === null) return null;
                                return ctx.dataset.label + ': $' + ctx.raw.toLocaleString();
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#888', maxTicksLimit: 15 }}, grid: {{ color: '#1a1a35' }} }},
                    y: {{
                        ticks: {{ color: '#888', callback: v => '$' + v.toLocaleString() }},
                        grid: {{ color: '#1a1a35' }}
                    }}
                }}
            }}
        }});
    }})();
    </script>"""


def generate_monthly_html(results: list, months: dict, df_full: pd.DataFrame, output_path: Path):
    """Generate HTML report with signal charts."""
    valid = [r for r in results if r["metrics"]["total_trades"] >= 3]
    valid.sort(key=lambda r: r["metrics"]["total_return_pct"], reverse=True)
    top = valid[:30]

    # Summary table
    summary_rows = ""
    for i, r in enumerate(top):
        m = r["metrics"]
        ret = m["total_return_pct"]
        color = "#00e676" if ret >= 50 else "#7cfc00" if ret >= 30 else "#ffeb3b" if ret > 0 else "#ff5252"
        summary_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{r['strategy_name']}</strong></td>
            <td>{r['month']}</td>
            <td>{r['strategy_type']}</td>
            <td>{r['variant']}</td>
            <td style="color:{color};font-weight:bold">{ret:.1f}%</td>
            <td>{m['profit_factor']:.2f}</td>
            <td>{m['sharpe_ratio']:.2f}</td>
            <td>{m['max_drawdown_pct']:.1f}%</td>
            <td>{m['win_rate_pct']:.0f}%</td>
            <td>{m['total_trades']}</td>
            <td style="color:#00e676">{m['long_return_pct']:.1f}%</td>
            <td style="color:#ff9100">{m['short_return_pct']:.1f}%</td>
        </tr>"""

    # Signal charts for top 3
    signal_charts = ""
    for r in valid[:3]:
        try:
            signal_charts += generate_signal_chart_html(r, df_full)
        except Exception as e:
            signal_charts += f"<div class='card'><p>Chart error: {e}</p></div>"

    # Best result stats
    best = valid[0] if valid else None
    best_m = best["metrics"] if best else {}
    target_met = best_m.get("total_return_pct", 0) >= 50

    # Equity charts for top 5
    equity_datasets = ""
    colors = ["#00e676", "#448aff", "#ff9100", "#e040fb", "#00e5ff"]
    chart_labels = "[]"
    for i, r in enumerate(valid[:5]):
        if len(r["equity"]) > 0:
            eq = r["equity"]
            step = max(1, len(eq) // 300)
            sampled = eq.iloc[::step]
            dates = [str(d)[:16] for d in sampled["datetime"]]
            values = [round(v, 2) for v in sampled["equity"]]
            if i == 0:
                chart_labels = json.dumps(dates)
            c = colors[i % len(colors)]
            equity_datasets += f"""
            {{ label: '{r["strategy_name"]} ({r["month"]})',
               data: {json.dumps(values)},
               borderColor: '{c}', backgroundColor: 'transparent',
               borderWidth: 2, pointRadius: 0, tension: 0.1 }},"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC 1H Monthly Backtest — Buy/Sell Signals</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ text-align: center; font-size: 2em; margin-bottom: 5px;
             background: linear-gradient(90deg, #ff9100, #ff5252, #e040fb);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
        .target-badge {{ display: inline-block; padding: 8px 24px; border-radius: 20px;
                        font-weight: bold; font-size: 1.1em; margin: 10px auto; text-align: center; }}
        .target-met {{ background: #00e676; color: #000; }}
        .target-miss {{ background: #ff9100; color: #000; }}
        .card {{ background: #141428; border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
        .card h2 {{ color: #ff9100; margin-bottom: 15px; font-size: 1.2em; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }}
        .stat-box {{ background: #1a1a35; border: 1px solid #2a2a4a; border-radius: 8px; padding: 12px; text-align: center; }}
        .stat-box .label {{ color: #888; font-size: 0.8em; }}
        .stat-box .value {{ font-size: 1.5em; font-weight: bold; margin-top: 4px; }}
        .positive {{ color: #00e676; }} .negative {{ color: #ff5252; }} .neutral {{ color: #ffeb3b; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
        th {{ background: #1a1a35; padding: 8px 6px; text-align: left; border-bottom: 2px solid #ff9100; color: #ff9100; }}
        td {{ padding: 7px 6px; border-bottom: 1px solid #2a2a4a; }}
        tr:hover {{ background: #1a1a35; }}
        .chart-container {{ position: relative; height: 400px; margin: 15px 0; }}
        .footer {{ text-align: center; color: #555; padding: 20px; font-size: 0.85em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>BTC 1H Monthly Backtest — Buy/Sell Signals</h1>
    <p class="subtitle">6 Aggressive Strategies | 1H Candles | Long + Short | Target: >50% Monthly Return</p>

    <div style="text-align:center;margin:15px 0;">
        <span class="target-badge {'target-met' if target_met else 'target-miss'}">
            {'TARGET MET: ' + f"{best_m.get('total_return_pct',0):.1f}" + '% in ' + (best['month'] if best else '') if target_met
             else 'Best: ' + f"{best_m.get('total_return_pct',0):.1f}" + '% — iterating...'}
        </span>
    </div>

    <!-- Best Strategy Stats -->
    <div class="card">
        <h2>Best: {best['strategy_name'] if best else 'N/A'} ({best['month'] if best else ''}) — {best['variant'] if best else ''}</h2>
        <div class="stats-grid">
            <div class="stat-box"><div class="label">Monthly Return</div>
                <div class="value {'positive' if best_m.get('total_return_pct',0)>0 else 'negative'}">{best_m.get('total_return_pct',0):.1f}%</div></div>
            <div class="stat-box"><div class="label">Profit Factor</div>
                <div class="value {'positive' if best_m.get('profit_factor',0)>1.3 else 'neutral'}">{best_m.get('profit_factor',0):.2f}</div></div>
            <div class="stat-box"><div class="label">Sharpe</div>
                <div class="value">{best_m.get('sharpe_ratio',0):.2f}</div></div>
            <div class="stat-box"><div class="label">Max DD</div>
                <div class="value {'positive' if best_m.get('max_drawdown_pct',0)<15 else 'negative'}">{best_m.get('max_drawdown_pct',0):.1f}%</div></div>
            <div class="stat-box"><div class="label">Win Rate</div>
                <div class="value">{best_m.get('win_rate_pct',0):.0f}%</div></div>
            <div class="stat-box"><div class="label">Trades</div>
                <div class="value" style="color:#448aff">{best_m.get('total_trades',0)}</div></div>
            <div class="stat-box"><div class="label">LONG Return</div>
                <div class="value {'positive' if best_m.get('long_return_pct',0)>0 else 'negative'}">{best_m.get('long_return_pct',0):.1f}%</div></div>
            <div class="stat-box"><div class="label">SHORT Return</div>
                <div class="value {'positive' if best_m.get('short_return_pct',0)>0 else 'negative'}">{best_m.get('short_return_pct',0):.1f}%</div></div>
        </div>
    </div>

    <!-- Signal Charts (Top 3) -->
    {signal_charts}

    <!-- Equity Curves -->
    <div class="card">
        <h2>Equity Curves — Top 5</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
    </div>
    <script>
    new Chart(document.getElementById('equityChart').getContext('2d'), {{
        type: 'line',
        data: {{ labels: {chart_labels}, datasets: [{equity_datasets}] }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#888', maxTicksLimit: 15 }}, grid: {{ color: '#1a1a35' }} }},
                y: {{ ticks: {{ color: '#888', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#1a1a35' }} }}
            }}
        }}
    }});
    </script>

    <!-- Results Table -->
    <div class="card">
        <h2>All Results (Sorted by Return)</h2>
        <div style="overflow-x:auto;">
        <table>
            <thead><tr>
                <th>#</th><th>Strategy</th><th>Month</th><th>Type</th><th>Var</th>
                <th>Return</th><th>PF</th><th>Sharpe</th><th>MaxDD</th>
                <th>Win%</th><th>Trades</th><th>Long</th><th>Short</th>
            </tr></thead>
            <tbody>{summary_rows}</tbody>
        </table>
        </div>
    </div>

    <div class="footer">
        BTC/USD 1H Monthly Backtest | {len(results)} runs | Long + Short | Commission: 0.1% RT<br>
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(html)
    print(f"\nHTML report: {output_path}")


def main():
    start = time.time()
    print("=" * 70)
    print("BTC 1H MONTHLY AGGRESSIVE BACKTEST")
    print(f"Target: >{TARGET_PROFIT_PCT}% monthly return | Long + Short")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Generate 1H data split by months
    print("\n[1/3] Generating 1H BTC data for last 6 months...")
    months, df_full = generate_1h_data_for_months()

    # Run all strategies on all months
    print("\n[2/3] Running aggressive backtests...")
    all_results = run_monthly_backtests(months)

    # Filter and sort
    valid = [r for r in all_results if r["metrics"]["total_trades"] >= 3]
    valid.sort(key=lambda r: r["metrics"]["total_return_pct"], reverse=True)

    # Print top results
    print(f"\n{'='*90}")
    print(f"TOP 20 RESULTS")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Month':>7} {'Strategy':<22} {'Var':>5} {'Return':>8} {'PF':>6} "
          f"{'DD':>6} {'Trades':>6} {'Win%':>5} {'Long%':>7} {'Short%':>7}")
    print("-" * 90)

    for i, r in enumerate(valid[:20]):
        m = r["metrics"]
        print(f"{i+1:>3} {r['month']:>7} {r['strategy_name']:<22} {r['variant']:>5} "
              f"{m['total_return_pct']:>7.1f}% {m['profit_factor']:>5.2f} "
              f"{m['max_drawdown_pct']:>5.1f}% {m['total_trades']:>6} "
              f"{m['win_rate_pct']:>4.0f}% {m['long_return_pct']:>6.1f}% "
              f"{m['short_return_pct']:>6.1f}%")

    # Check target
    if valid and valid[0]["metrics"]["total_return_pct"] >= TARGET_PROFIT_PCT:
        best = valid[0]
        print(f"\n>>> TARGET MET: {best['strategy_name']} in {best['month']} "
              f"= {best['metrics']['total_return_pct']:.1f}% <<<")
    else:
        best_ret = valid[0]["metrics"]["total_return_pct"] if valid else 0
        print(f"\n>>> Best: {best_ret:.1f}% — target {TARGET_PROFIT_PCT}% <<<")

    # Generate HTML
    print("\n[3/3] Generating HTML report with buy/sell signals...")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    html_path = RESULTS_DIR / "monthly_1h_report.html"
    generate_monthly_html(valid, months, df_full, html_path)

    # Save summary CSV
    rows = []
    for r in valid:
        rows.append({
            "month": r["month"], "strategy": r["strategy_name"],
            "type": r["strategy_type"], "variant": r["variant"],
            "params": json.dumps(r["params"]), **r["metrics"]})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "monthly_1h_summary.csv", index=False)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Open: {html_path}")

    return valid


if __name__ == "__main__":
    results = main()
