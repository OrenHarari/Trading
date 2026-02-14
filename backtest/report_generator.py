"""
HTML Report Generator — creates a clean, visual dashboard showing
backtest results across strategies and timeframes.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def generate_html_report(results: list, data: dict, output_path: Path):
    """Generate comprehensive HTML backtest report."""

    # Prepare data for charts
    top_results = results[:15]

    # Build equity chart data for top 5
    equity_charts = []
    for i, r in enumerate(top_results[:5]):
        if len(r["equity"]) > 0:
            eq = r["equity"]
            # Downsample for performance
            step = max(1, len(eq) // 500)
            sampled = eq.iloc[::step]
            dates = [str(d)[:10] for d in sampled["datetime"]]
            values = [round(v, 2) for v in sampled["equity"]]
            equity_charts.append({
                "name": f"{r['strategy_name']} ({r['timeframe']})",
                "dates": dates,
                "values": values,
            })

    # Summary table rows
    summary_rows = ""
    for i, r in enumerate(top_results):
        m = r["metrics"]
        ret = m["total_return_pct"]
        color = "#00e676" if ret >= 30 else "#ffeb3b" if ret > 0 else "#ff5252"
        pf_color = "#00e676" if m["profit_factor"] > 1.5 else "#ffeb3b" if m["profit_factor"] > 1.0 else "#ff5252"

        summary_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{r['strategy_name']}</strong></td>
            <td>{r['timeframe']}</td>
            <td>{r['strategy_type']}</td>
            <td style="color:{color};font-weight:bold">{ret:.1f}%</td>
            <td style="color:{pf_color}">{m['profit_factor']:.2f}</td>
            <td>{m['sharpe_ratio']:.2f}</td>
            <td>{m['max_drawdown_pct']:.1f}%</td>
            <td>{m['win_rate_pct']:.1f}%</td>
            <td>{m['total_trades']}</td>
            <td>{m['long_return_pct']:.1f}%</td>
            <td>{m['short_return_pct']:.1f}%</td>
        </tr>"""

    # Period returns table
    period_rows = ""
    for i, r in enumerate(top_results[:10]):
        m = r["metrics"]
        def pc(v):
            color = "#00e676" if v >= 30 else "#7cfc00" if v > 10 else "#ffeb3b" if v > 0 else "#ff5252"
            return f'<td style="color:{color}">{v:.1f}%</td>'

        period_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td><strong>{r['strategy_name']}</strong></td>
            <td>{r['timeframe']}</td>
            {pc(m.get('return_1w', 0))}
            {pc(m.get('return_1m', 0))}
            {pc(m.get('return_3m', 0))}
            {pc(m.get('return_6m', 0))}
            {pc(m.get('return_1y', 0))}
            {pc(m['total_return_pct'])}
        </tr>"""

    # Long vs Short comparison
    ls_rows = ""
    for i, r in enumerate(top_results[:10]):
        m = r["metrics"]
        long_c = "#00e676" if m["long_return_pct"] > 0 else "#ff5252"
        short_c = "#00e676" if m["short_return_pct"] > 0 else "#ff5252"
        ls_rows += f"""
        <tr>
            <td>{r['strategy_name']} ({r['timeframe']})</td>
            <td>{m['long_trades']}</td>
            <td style="color:{long_c};font-weight:bold">{m['long_return_pct']:.1f}%</td>
            <td>{m['short_trades']}</td>
            <td style="color:{short_c};font-weight:bold">{m['short_return_pct']:.1f}%</td>
        </tr>"""

    # Best result summary
    best = top_results[0] if top_results else None
    best_m = best["metrics"] if best else {}
    target_met = best_m.get("total_return_pct", 0) >= 30

    # Equity chart JS data
    chart_datasets = ""
    colors = ["#00e676", "#448aff", "#ff9100", "#e040fb", "#00e5ff"]
    for i, ec in enumerate(equity_charts):
        c = colors[i % len(colors)]
        chart_datasets += f"""
        {{
            label: '{ec["name"]}',
            data: {json.dumps(ec["values"])},
            borderColor: '{c}',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
        }},"""

    chart_labels = json.dumps(equity_charts[0]["dates"]) if equity_charts else "[]"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Trading System — Backtest Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2em;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #00e676, #448aff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .target-badge {{
            display: inline-block;
            padding: 8px 24px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
            margin: 10px auto;
            text-align: center;
        }}
        .target-met {{ background: #00e676; color: #000; }}
        .target-miss {{ background: #ff5252; color: #fff; }}
        .card {{
            background: #141428;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card h2 {{
            color: #448aff;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #1a1a35;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-box .label {{ color: #888; font-size: 0.85em; }}
        .stat-box .value {{ font-size: 1.6em; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #00e676; }}
        .negative {{ color: #ff5252; }}
        .neutral {{ color: #ffeb3b; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th {{
            background: #1a1a35;
            padding: 10px 8px;
            text-align: left;
            border-bottom: 2px solid #448aff;
            color: #448aff;
            font-weight: 600;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #2a2a4a;
        }}
        tr:hover {{ background: #1a1a35; }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #555;
            padding: 20px;
            font-size: 0.85em;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC/USD Trading System — Backtest Report</h1>
        <p class="subtitle">Real Data | Multi-Timeframe | {len(results)} strategy variants tested | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div style="text-align:center;margin:15px 0;">
            <span class="target-badge {'target-met' if target_met else 'target-miss'}">
                {'TARGET MET: ' + str(best_m.get("total_return_pct", 0)) + '% return' if target_met else 'TARGET IN PROGRESS — Best: ' + str(best_m.get("total_return_pct", 0)) + '%'}
            </span>
        </div>

        <!-- Best Strategy Stats -->
        <div class="card">
            <h2>Best Strategy: {best['strategy_name'] if best else 'N/A'} ({best['timeframe'] if best else ''})</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="label">Total Return</div>
                    <div class="value {'positive' if best_m.get('total_return_pct', 0) > 0 else 'negative'}">{best_m.get('total_return_pct', 0):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="label">Profit Factor</div>
                    <div class="value {'positive' if best_m.get('profit_factor', 0) > 1.3 else 'neutral'}">{best_m.get('profit_factor', 0):.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value {'positive' if best_m.get('sharpe_ratio', 0) > 0.8 else 'neutral'}">{best_m.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Max Drawdown</div>
                    <div class="value {'positive' if best_m.get('max_drawdown_pct', 0) < 25 else 'negative'}">{best_m.get('max_drawdown_pct', 0):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="label">Win Rate</div>
                    <div class="value {'positive' if best_m.get('win_rate_pct', 0) > 50 else 'neutral'}">{best_m.get('win_rate_pct', 0):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="label">Total Trades</div>
                    <div class="value" style="color:#448aff">{best_m.get('total_trades', 0)}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Long Return</div>
                    <div class="value {'positive' if best_m.get('long_return_pct', 0) > 0 else 'negative'}">{best_m.get('long_return_pct', 0):.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="label">Short Return</div>
                    <div class="value {'positive' if best_m.get('short_return_pct', 0) > 0 else 'negative'}">{best_m.get('short_return_pct', 0):.1f}%</div>
                </div>
            </div>
        </div>

        <!-- Equity Curve Chart -->
        <div class="card">
            <h2>Equity Curves — Top 5 Strategies</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <!-- Summary Table -->
        <div class="card">
            <h2>All Strategy Results (Sorted by Return)</h2>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>#</th><th>Strategy</th><th>TF</th><th>Type</th>
                            <th>Return</th><th>PF</th><th>Sharpe</th><th>MaxDD</th>
                            <th>Win%</th><th>Trades</th><th>Long</th><th>Short</th>
                        </tr>
                    </thead>
                    <tbody>{summary_rows}</tbody>
                </table>
            </div>
        </div>

        <!-- Period Returns -->
        <div class="card">
            <h2>Returns by Period</h2>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>#</th><th>Strategy</th><th>TF</th>
                            <th>1 Week</th><th>1 Month</th><th>3 Months</th>
                            <th>6 Months</th><th>1 Year</th><th>Total</th>
                        </tr>
                    </thead>
                    <tbody>{period_rows}</tbody>
                </table>
            </div>
        </div>

        <!-- Long vs Short -->
        <div class="card">
            <h2>Long vs Short Analysis</h2>
            <div style="overflow-x:auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Strategy</th><th>Long Trades</th><th>Long Return</th>
                            <th>Short Trades</th><th>Short Return</th>
                        </tr>
                    </thead>
                    <tbody>{ls_rows}</tbody>
                </table>
            </div>
        </div>

        <div class="footer">
            BTC/USD Trading System | Real Data Backtest | No lookahead bias | Commission: 0.1% RT<br>
            Generated by automated backtesting pipeline
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_labels},
                datasets: [{chart_datasets}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e0e0e0', font: {{ size: 12 }} }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#888', maxTicksLimit: 20 }},
                        grid: {{ color: '#1a1a35' }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#888',
                            callback: function(v) {{ return '$' + v.toLocaleString(); }}
                        }},
                        grid: {{ color: '#1a1a35' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(html)
    print(f"  HTML report saved: {output_path}")
