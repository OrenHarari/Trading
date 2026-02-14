"""
Web UI Dashboard v2 for BTC/USD & ETH/USD Backtesting System.

Features:
- Real OHLC data from Yahoo Finance (BTC + ETH)
- 8 strategies with parameter tuning
- Persistent snapshot cache (instant reload)
- Custom capital presets ($1K - $100K)
- Investment P&L summary
- Strategy explainer panel
- Interactive Plotly charts

Usage:
    python dashboard.py
    # Opens at http://localhost:5000
"""

import json
import sys
import hashlib
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string, jsonify, request, make_response
import pandas as pd
import numpy as np
import gzip

from backtester.data_loader import load_data, load_btc_data, get_data_summary, ASSETS
from backtester.strategies import STRATEGY_REGISTRY
from backtester.engine import BacktestConfig

app = Flask(__name__)

MAX_CHART_POINTS = 500  # Downsample equity/price to this many points for charts


def _downsample(records, max_points=MAX_CHART_POINTS):
    """Downsample a list of dicts to max_points, keeping first and last."""
    n = len(records)
    if n <= max_points:
        return records
    # Always keep first and last; evenly space the rest
    indices = set([0, n - 1])
    step = (n - 1) / (max_points - 1)
    for i in range(max_points):
        indices.add(int(round(i * step)))
    indices = sorted(indices)
    return [records[i] for i in indices]


def _slim_equity(equity_records):
    """Remove unused fields and round floats in equity data."""
    slim = []
    for r in equity_records:
        slim.append({
            "date": r["date"],
            "equity": round(r["equity"], 2),
            "drawdown_pct": round(r.get("drawdown_pct", 0), 2),
        })
    return slim


def _slim_price(price_records):
    """Round price data floats."""
    return [{"date": r["date"], "close": round(r["close"], 2)} for r in price_records]


def _gzip_json_response(data):
    """Return a gzipped JSON response if client supports it, otherwise plain JSON."""
    content = json.dumps(data, default=str, separators=(',', ':'))
    accept = request.headers.get('Accept-Encoding', '')
    if 'gzip' in accept:
        compressed = gzip.compress(content.encode('utf-8'), compresslevel=6)
        resp = make_response(compressed)
        resp.headers['Content-Encoding'] = 'gzip'
        resp.headers['Content-Type'] = 'application/json'
        resp.headers['Content-Length'] = len(compressed)
        return resp
    else:
        resp = make_response(content)
        resp.headers['Content-Type'] = 'application/json'
        return resp

# ── Persistent Snapshot Cache ──────────────────────────
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)


def _snapshot_key(source, start_date, end_date, param_overrides):
    """Generate a unique cache key for a backtest configuration."""
    raw = json.dumps({"s": source, "start": start_date, "end": end_date,
                       "p": param_overrides or {}}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _snapshot_path(key: str) -> Path:
    return SNAPSHOT_DIR / f"{key}.json"


def _load_snapshot(key: str):
    """Load a cached snapshot from disk. Returns None if not found."""
    path = _snapshot_path(key)
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_snapshot(key: str, data: dict):
    """Save backtest results to disk as a JSON snapshot."""
    path = _snapshot_path(key)
    try:
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except IOError as e:
        print(f"Warning: could not save snapshot: {e}")


def run_strategies(source="BTC-USD", start_date=None, end_date=None,
                   strategy_keys=None, initial_capital=100000, param_overrides=None):
    """Run backtests with persistent snapshot cache.

    Results are saved to disk so repeat runs with the same source/dates/params
    are returned instantly. Capital scaling is done client-side — snapshots
    store results at the base capital level (100k) for reuse across capitals.
    """
    snap_key = _snapshot_key(source, start_date, end_date, param_overrides)

    # Check disk cache first (capital-independent — we store base results)
    cached = _load_snapshot(snap_key)
    if cached is not None:
        # Adjust capital in cached results
        cached["initial_capital"] = initial_capital
        cached["_cached"] = True
        return cached

    # Run backtests fresh
    print(f"[backtest] Computing {source} {start_date}–{end_date} ...")
    t0 = _time.time()

    df = load_data(source=source, start_date=start_date, end_date=end_date)
    config = BacktestConfig(initial_capital=100_000)  # always run at 100k base

    keys = strategy_keys or list(STRATEGY_REGISTRY.keys())
    results = {}
    param_overrides = param_overrides or {}

    for key in keys:
        if key not in STRATEGY_REGISTRY:
            continue
        params = param_overrides.get(key, {})
        strat = STRATEGY_REGISTRY[key](config, **params)
        res = strat.run(df)
        if "error" not in res:
            eq_raw = strat.get_equity_df().to_dict(orient="records")
            eq_slim = _slim_equity(eq_raw)
            eq_down = _downsample(eq_slim)
            results[key] = {
                "metrics": res,
                "equity": eq_down,
                "trades": strat.get_trades_df().to_dict(orient="records"),
                "name": strat.name,
                "explainer": strat.explainer(),
            }

    data_summary = get_data_summary(df)
    price_data = df[["date", "close"]].copy()
    price_data["date"] = price_data["date"].astype(str)
    price_slim = _slim_price(price_data.to_dict(orient="records"))
    price_down = _downsample(price_slim)

    output = {
        "results": results,
        "data_summary": data_summary,
        "price_data": price_down,
        "initial_capital": initial_capital,
        "_base_capital": 100_000,
    }

    # Serialize dates before caching
    output = json.loads(json.dumps(output, default=str))

    # Save snapshot to disk
    _save_snapshot(snap_key, output)
    elapsed = _time.time() - t0
    print(f"[backtest] Done in {elapsed:.1f}s — snapshot saved as {snap_key}")

    output["initial_capital"] = initial_capital
    return output


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Backtester Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0d1117; color: #c9d1d9; min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #161b22, #1c2333);
            padding: 16px 24px; border-bottom: 1px solid #30363d;
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 12px;
        }
        .header h1 { font-size: 22px; background: linear-gradient(90deg, #f7931a, #ffd700);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header .subtitle { font-size: 12px; color: #8b949e; margin-top: 2px; }
        .controls {
            display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap;
        }
        .ctrl-group { display: flex; flex-direction: column; gap: 2px; }
        .ctrl-group label { font-size: 11px; color: #8b949e; }
        .ctrl-group input, .ctrl-group select {
            background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
            padding: 5px 10px; border-radius: 6px; font-size: 13px;
        }
        .btn-run {
            background: #238636; border: 1px solid #2ea043; color: #fff;
            padding: 6px 18px; border-radius: 6px; font-size: 13px;
            font-weight: 600; cursor: pointer;
        }
        .btn-run:hover { background: #2ea043; }
        .btn-run:disabled { opacity: 0.5; cursor: wait; }
        .container { padding: 16px 24px; max-width: 1600px; margin: 0 auto; }

        /* Loading */
        .loading { text-align: center; padding: 60px; font-size: 16px; color: #8b949e; }
        .loading .spinner {
            display: inline-block; width: 36px; height: 36px;
            border: 3px solid #30363d; border-top-color: #f7931a;
            border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Investment Banner */
        .invest-banner {
            background: linear-gradient(135deg, #1c2333, #161b22);
            border: 1px solid #30363d; border-radius: 10px;
            padding: 18px 24px; margin-bottom: 16px;
            display: flex; align-items: center; justify-content: center;
            gap: 40px; flex-wrap: wrap;
        }
        .invest-banner .inv-item { text-align: center; }
        .invest-banner .inv-label { font-size: 11px; color: #8b949e; margin-bottom: 2px; }
        .invest-banner .inv-val { font-size: 26px; font-weight: 700; }
        .invest-banner .inv-val.profit { color: #3fb950; }
        .invest-banner .inv-val.loss { color: #f85149; }
        .invest-banner .inv-strat { font-size: 13px; color: #f7931a; font-weight: 600; }
        .invest-banner .inv-arrow { font-size: 32px; color: #30363d; }

        /* Data Bar */
        .data-bar {
            display: flex; gap: 16px; padding: 12px 18px;
            background: #161b22; border-radius: 8px;
            margin-bottom: 16px; border: 1px solid #30363d; flex-wrap: wrap;
        }
        .data-bar .stat { text-align: center; flex: 1; min-width: 80px; }
        .data-bar .stat .value { font-size: 18px; font-weight: 700; color: #e6edf3; }
        .data-bar .stat .label { font-size: 10px; color: #8b949e; margin-top: 2px; }
        .data-bar .stat .value.positive { color: #3fb950; }
        .data-bar .stat .value.negative { color: #f85149; }

        /* Comparison Table */
        .comparison-table {
            width: 100%; border-collapse: collapse;
            background: #161b22; border-radius: 8px; overflow: hidden;
            margin-bottom: 16px; border: 1px solid #30363d;
        }
        .comparison-table th {
            background: #1c2333; padding: 10px 12px; text-align: left;
            font-size: 11px; color: #8b949e; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px;
            border-bottom: 1px solid #30363d;
        }
        .comparison-table td {
            padding: 10px 12px; border-bottom: 1px solid #21262d;
            font-size: 13px; font-variant-numeric: tabular-nums;
        }
        .comparison-table tr:hover { background: #1c2333; cursor: pointer; }
        .comparison-table tr.best td { background: rgba(35, 134, 54, 0.1); }
        .comparison-table .strat-name { font-weight: 600; color: #e6edf3; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #d29922; }
        .badge {
            display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 10px; font-weight: 600;
        }
        .badge.pass { background: rgba(35,134,54,0.2); color: #3fb950; }
        .badge.fail { background: rgba(248,81,73,0.2); color: #f85149; }

        /* Charts */
        .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .chart-full { grid-column: 1 / -1; }
        .chart-box {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 14px; overflow: hidden;
        }
        .chart-box h3 {
            font-size: 13px; color: #e6edf3; margin-bottom: 10px;
            padding-bottom: 6px; border-bottom: 1px solid #21262d;
        }

        /* Strategy Tabs */
        .strat-tabs { display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }
        .strat-tab {
            padding: 5px 14px; border-radius: 20px;
            background: #21262d; border: 1px solid #30363d;
            cursor: pointer; font-size: 12px; transition: all 0.2s;
        }
        .strat-tab:hover { border-color: #8b949e; }
        .strat-tab.active {
            background: #f7931a; color: #0d1117;
            border-color: #f7931a; font-weight: 600;
        }

        /* Explainer */
        .explainer-box {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 16px; margin-bottom: 16px;
        }
        .explainer-box h3 { font-size: 14px; color: #f7931a; margin-bottom: 8px; }
        .explainer-box p { font-size: 13px; line-height: 1.6; color: #8b949e; }

        /* Param Panel */
        .param-panel {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 8px; padding: 16px; margin-bottom: 16px;
        }
        .param-panel h3 {
            font-size: 14px; color: #e6edf3; margin-bottom: 12px;
            padding-bottom: 6px; border-bottom: 1px solid #21262d;
        }
        .param-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 10px;
        }
        .param-item { display: flex; flex-direction: column; gap: 3px; }
        .param-item label { font-size: 11px; color: #8b949e; }
        .param-item .param-row { display: flex; align-items: center; gap: 8px; }
        .param-item input[type=range] {
            flex: 1; accent-color: #f7931a; height: 4px;
        }
        .param-item .param-val {
            font-size: 12px; color: #e6edf3; font-weight: 600;
            min-width: 40px; text-align: right;
        }

        /* Trade Log */
        .trades-table { width: 100%; border-collapse: collapse; font-size: 11px; }
        .trades-table th {
            background: #1c2333; padding: 6px 8px; text-align: left;
            color: #8b949e; position: sticky; top: 0;
        }
        .trades-table td { padding: 5px 8px; border-bottom: 1px solid #21262d; }
        .trades-table tr:hover { background: #1c2333; }
        .trade-log-container { max-height: 350px; overflow-y: auto; border-radius: 0 0 8px 8px; }

        @media (max-width: 900px) {
            .chart-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Trading Backtester</h1>
            <div class="subtitle">Regime-Adaptive Strategy Analysis</div>
        </div>
        <div class="controls">
            <div class="ctrl-group">
                <label>Asset</label>
                <select id="assetSelect">
                    <option value="BTC-USD" selected>BTC/USD (Yahoo)</option>
                    <option value="ETH-USD">ETH/USD (Yahoo)</option>
                    <option value="CSV">BTC (CSV File)</option>
                </select>
            </div>
            <div class="ctrl-group">
                <label>Start Date</label>
                <input type="date" id="startDate" value="2020-01-01">
            </div>
            <div class="ctrl-group">
                <label>End Date</label>
                <input type="date" id="endDate" value="2026-02-13">
            </div>
            <div class="ctrl-group">
                <label>Capital ($)</label>
                <select id="capitalSelect">
                    <option value="1000">$1,000</option>
                    <option value="5000">$5,000</option>
                    <option value="10000" selected>$10,000</option>
                    <option value="50000">$50,000</option>
                    <option value="100000">$100,000</option>
                </select>
            </div>
            <button class="btn-run" id="runBtn" onclick="runBacktest(false)">Run Backtest</button>
            <button class="btn-run" id="rerunBtn" onclick="runBacktest(true)" 
                style="background:#6e40c9;border-color:#8957e5;font-size:11px;padding:6px 10px">
                Force Rerun</button>
        </div>
    </div>

    <div class="container" id="mainContent">
        <div class="loading" id="initialLoad">
            <div class="spinner"></div>
            <div>Loading strategies & running backtests...</div>
            <div style="font-size:12px; color:#6e7681; margin-top:6px">Cached results load instantly</div>
        </div>
    </div>

    <script>
    let currentData = null;
    let selectedStrategy = null;
    let paramOverrides = {};

    const COLORS = {
        'T1': '#58a6ff', 'M1': '#d2a8ff', 'H1': '#ffa657',
        'H2': '#3fb950', 'RSI': '#f778ba', 'MACD': '#79c0ff',
        'SMA': '#d29922', 'BH': '#8b949e'
    };

    const plotLayout = {
        paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
        font: { color: '#c9d1d9', size: 11 },
        margin: { l: 55, r: 15, t: 10, b: 35 },
        xaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
        yaxis: { gridcolor: '#21262d', linecolor: '#30363d' },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
        hovermode: 'x unified',
    };

    function fmt$(v) {
        if (Math.abs(v) >= 1e6) return '$' + (v/1e6).toFixed(1) + 'M';
        if (Math.abs(v) >= 1e3) return '$' + (v/1e3).toFixed(1) + 'K';
        return '$' + v.toFixed(0);
    }

    async function runBacktest(forceRerun) {
        const btn = document.getElementById('runBtn');
        btn.disabled = true; btn.textContent = 'Running...';
        document.getElementById('mainContent').innerHTML =
            '<div class="loading"><div class="spinner"></div><div>Running backtests...</div></div>';

        const source = document.getElementById('assetSelect').value;
        const start = document.getElementById('startDate').value || '';
        const end = document.getElementById('endDate').value || '';
        const capital = parseInt(document.getElementById('capitalSelect').value);

        try {
            const resp = await fetch('/api/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    source, start, end, capital,
                    params: Object.keys(paramOverrides).length > 0 ? paramOverrides : undefined,
                    force: !!forceRerun
                })
            });
            currentData = await resp.json();
            renderDashboard();
        } catch (e) {
            document.getElementById('mainContent').innerHTML =
                '<div class="loading" style="color:#f85149">Error: ' + e.message + '</div>';
        }
        btn.disabled = false; btn.textContent = 'Run Backtest';
    }

    function renderDashboard() {
        const d = currentData;
        const keys = Object.keys(d.results);
        if (!selectedStrategy || !keys.includes(selectedStrategy))
            selectedStrategy = keys.includes('H2') ? 'H2' : keys[0];

        const capital = d.initial_capital;
        let html = '';

        // ── Investment Banner ──
        const sorted = keys
            .filter(k => d.results[k].metrics.primary_score !== undefined)
            .sort((a, b) => (d.results[b].metrics.primary_score || 0) - (d.results[a].metrics.primary_score || 0));
        const bestKey = sorted[0];
        if (bestKey) {
            const bm = d.results[bestKey].metrics;
            const finalVal = capital * (1 + bm.total_return_pct / 100);
            const pnl = finalVal - capital;
            const isProfit = pnl >= 0;
            html += '<div class="invest-banner">' +
                '<div class="inv-item"><div class="inv-label">You Invested</div>' +
                '<div class="inv-val">' + fmt$(capital) + '</div></div>' +
                '<div class="inv-arrow">&rarr;</div>' +
                '<div class="inv-item"><div class="inv-label">Best Strategy Result</div>' +
                '<div class="inv-val ' + (isProfit ? 'profit' : 'loss') + '">' + fmt$(finalVal) + '</div></div>' +
                '<div class="inv-item"><div class="inv-label">P&L</div>' +
                '<div class="inv-val ' + (isProfit ? 'profit' : 'loss') + '">' +
                (isProfit ? '+' : '') + fmt$(pnl) + ' (' + (isProfit ? '+' : '') + bm.total_return_pct.toFixed(1) + '%)</div></div>' +
                '<div class="inv-item"><div class="inv-label">Best Strategy</div>' +
                '<div class="inv-strat">' + d.results[bestKey].name + '</div></div>' +
                '</div>';
        }

        // ── Data Summary Bar ──
        const ds = d.data_summary;
        const retClass = ds.total_return_pct >= 0 ? 'positive' : 'negative';
        html += '<div class="data-bar">' +
            '<div class="stat"><div class="value">' + ds.total_bars.toLocaleString() + '</div><div class="label">Total Bars</div></div>' +
            '<div class="stat"><div class="value">' + ds.start_date + '</div><div class="label">Start</div></div>' +
            '<div class="stat"><div class="value">' + ds.end_date + '</div><div class="label">End</div></div>' +
            '<div class="stat"><div class="value">$' + ds.start_price.toLocaleString() + '</div><div class="label">Start Price</div></div>' +
            '<div class="stat"><div class="value">$' + ds.end_price.toLocaleString() + '</div><div class="label">End Price</div></div>' +
            '<div class="stat"><div class="value ' + retClass + '">' + (ds.total_return_pct > 0 ? '+' : '') + ds.total_return_pct + '%</div><div class="label">Asset Return</div></div>' +
            '</div>';

        // ── Comparison Table ──
        html += '<table class="comparison-table"><thead><tr>' +
            '<th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th>' +
            '<th>Sharpe</th><th>Max DD</th><th>Score</th><th>CAGR</th>' +
            '<th>P&L ($)</th><th>Final Value</th><th>Status</th>' +
            '</tr></thead><tbody>';

        for (const key of sorted) {
            const m = d.results[key].metrics;
            const isBest = key === bestKey;
            const finalVal = capital * (1 + m.total_return_pct / 100);
            const pnl = finalVal - capital;
            const allPass = m.primary_score > 0.10 && m.max_drawdown_pct <= 25 && m.profit_factor > 1.3;

            html += '<tr class="' + (isBest ? 'best' : '') + '" onclick="selectStrategy(\'' + key + '\')">' +
                '<td class="strat-name" style="color:' + (COLORS[key] || '#c9d1d9') + '">' +
                (isBest ? '&#9733; ' : '') + d.results[key].name + '</td>' +
                '<td>' + m.total_trades + ' <span style="color:#8b949e">(' + m.long_trades + 'L/' + m.short_trades + 'S)</span></td>' +
                '<td>' + m.win_rate + '%</td>' +
                '<td class="' + (m.profit_factor >= 1.3 ? 'positive' : m.profit_factor >= 1.0 ? 'neutral' : 'negative') + '">' + m.profit_factor.toFixed(3) + '</td>' +
                '<td class="' + (m.sharpe_ratio >= 0.8 ? 'positive' : m.sharpe_ratio >= 0 ? 'neutral' : 'negative') + '">' + m.sharpe_ratio.toFixed(3) + '</td>' +
                '<td class="' + (m.max_drawdown_pct <= 25 ? 'positive' : 'negative') + '">' + m.max_drawdown_pct.toFixed(1) + '%</td>' +
                '<td class="' + (m.primary_score >= 0.10 ? 'positive' : 'negative') + '" style="font-weight:700">' + m.primary_score.toFixed(4) + '</td>' +
                '<td class="' + (m.cagr_pct >= 0 ? 'positive' : 'negative') + '">' + m.cagr_pct.toFixed(1) + '%</td>' +
                '<td class="' + (pnl >= 0 ? 'positive' : 'negative') + '">' + (pnl >= 0 ? '+' : '') + fmt$(pnl) + '</td>' +
                '<td>' + fmt$(finalVal) + '</td>' +
                '<td><span class="badge ' + (allPass ? 'pass' : 'fail') + '">' + (allPass ? 'PASS' : 'REVIEW') + '</span></td>' +
                '</tr>';
        }
        html += '</tbody></table>';

        // ── Strategy Tabs ──
        html += '<div class="strat-tabs" id="stratTabs">';
        for (const key of sorted) {
            html += '<div class="strat-tab ' + (key === selectedStrategy ? 'active' : '') + '" ' +
                'onclick="selectStrategy(\'' + key + '\')">' + d.results[key].name + '</div>';
        }
        html += '</div>';

        // ── Explainer ──
        html += '<div class="explainer-box" id="explainerBox">' +
            '<h3 id="explainerTitle">' + (d.results[selectedStrategy]?.name || '') + ' — How it works</h3>' +
            '<p id="explainerText">' + (d.results[selectedStrategy]?.explainer || '') + '</p></div>';

        // ── Parameter Tuning Panel ──
        html += '<div class="param-panel" id="paramPanel">' +
            '<h3>Parameter Tuning — <span id="paramTitle">' + (d.results[selectedStrategy]?.name || '') + '</span></h3>' +
            '<div class="param-grid" id="paramGrid"></div></div>';

        // ── Charts ──
        html += '<div class="chart-grid">' +
            '<div class="chart-box chart-full"><h3>Equity Curves — All Strategies</h3><div id="equityChart"></div></div>' +
            '<div class="chart-box chart-full"><h3>Price + Trade Markers</h3><div id="priceChart"></div></div>' +
            '<div class="chart-box"><h3>Drawdown Comparison</h3><div id="ddChart"></div></div>' +
            '<div class="chart-box"><h3>Monthly Returns</h3><div id="monthlyChart"></div></div>' +
            '<div class="chart-box"><h3>Trade P&L Distribution</h3><div id="pnlChart"></div></div>' +
            '<div class="chart-box"><h3>Win Rate by Strategy</h3><div id="wrChart"></div></div>' +
            '</div>';

        // ── Trade Log ──
        html += '<div class="chart-box"><h3>Trade Log — <span id="tradeLogTitle">' +
            (d.results[selectedStrategy]?.name || '') + '</span></h3>' +
            '<div class="trade-log-container"><table class="trades-table"><thead><tr>' +
            '<th>#</th><th>Entry</th><th>Exit</th><th>Dir</th><th>Module</th>' +
            '<th>Entry $</th><th>Exit $</th><th>P&L %</th><th>P&L $</th><th>Bars</th><th>Exit Reason</th>' +
            '</tr></thead><tbody id="tradeLogBody"></tbody></table></div></div>';

        document.getElementById('mainContent').innerHTML = html;

        renderParamPanel(selectedStrategy);
        renderEquityChart(d);
        renderPriceChart(d);
        renderDrawdownChart(d);
        renderMonthlyChart(d);
        renderPnlChart(d);
        renderWinRateChart(d);
        renderTradeLog(d, selectedStrategy);
    }

    function selectStrategy(key) {
        selectedStrategy = key;
        document.querySelectorAll('.strat-tab').forEach(t => {
            t.classList.toggle('active', t.textContent.trim() === currentData.results[key]?.name);
        });
        const r = currentData.results[key];
        if (r) {
            document.getElementById('explainerTitle').textContent = r.name + ' — How it works';
            document.getElementById('explainerText').textContent = r.explainer;
            document.getElementById('tradeLogTitle').textContent = r.name;
            document.getElementById('paramTitle').textContent = r.name;
        }
        renderParamPanel(key);
        renderTradeLog(currentData, key);
        renderPriceChart(currentData);
        renderMonthlyChart(currentData);
        renderPnlChart(currentData);
    }

    async function renderParamPanel(key) {
        const grid = document.getElementById('paramGrid');
        if (!grid) return;

        try {
            const resp = await fetch('/api/params?strategy=' + key);
            const params = await resp.json();
            grid.innerHTML = '';

            if (!params || Object.keys(params).length === 0) {
                grid.innerHTML = '<div style="color:#8b949e;font-size:12px">No tunable parameters for this strategy.</div>';
                return;
            }

            for (const [pname, pinfo] of Object.entries(params)) {
                const currentVal = (paramOverrides[key] && paramOverrides[key][pname] !== undefined)
                    ? paramOverrides[key][pname] : pinfo.default;
                const step = pinfo.step || 1;

                const item = document.createElement('div');
                item.className = 'param-item';
                item.innerHTML =
                    '<label>' + pinfo.label + '</label>' +
                    '<div class="param-row">' +
                    '<input type="range" min="' + pinfo.min + '" max="' + pinfo.max + '" ' +
                    'step="' + step + '" value="' + currentVal + '" ' +
                    'oninput="updateParam(\'' + key + '\', \'' + pname + '\', this.value, ' + step + ')">' +
                    '<span class="param-val" id="pv_' + key + '_' + pname + '">' +
                    (step < 1 ? parseFloat(currentVal).toFixed(1) : currentVal) + '</span>' +
                    '</div>';
                grid.appendChild(item);
            }
        } catch (e) {
            grid.innerHTML = '<div style="color:#f85149;font-size:12px">Could not load params.</div>';
        }
    }

    function updateParam(stratKey, paramName, value, step) {
        if (!paramOverrides[stratKey]) paramOverrides[stratKey] = {};
        paramOverrides[stratKey][paramName] = step < 1 ? parseFloat(value) : parseInt(value);
        const el = document.getElementById('pv_' + stratKey + '_' + paramName);
        if (el) el.textContent = step < 1 ? parseFloat(value).toFixed(1) : value;
    }

    function renderEquityChart(d) {
        const traces = [];
        const capital = d.initial_capital;
        const scale = capital / (d._base_capital || 100000);
        for (const [key, val] of Object.entries(d.results)) {
            const eq = val.equity;
            if (!eq || eq.length === 0) continue;
            traces.push({
                x: eq.map(e => e.date),
                y: eq.map(e => e.equity * scale),
                name: val.name, type: 'scatter', mode: 'lines',
                line: { color: COLORS[key] || '#8b949e', width: key === 'BH' ? 1 : 2,
                        dash: key === 'BH' ? 'dot' : 'solid' },
            });
        }
        Plotly.newPlot('equityChart', traces, {
            ...plotLayout, height: 380,
            yaxis: { ...plotLayout.yaxis, title: 'Equity ($)', tickprefix: '$' },
        }, { responsive: true });
    }

    function renderPriceChart(d) {
        const price = d.price_data;
        const traces = [{
            x: price.map(p => p.date), y: price.map(p => p.close),
            name: 'Price', type: 'scatter', mode: 'lines',
            line: { color: '#f7931a', width: 1.5 },
        }];

        if (selectedStrategy && d.results[selectedStrategy]) {
            const trades = d.results[selectedStrategy].trades;
            const wins = trades.filter(t => t.profit_pct > 0);
            const losses = trades.filter(t => t.profit_pct <= 0);

            if (wins.length > 0)
                traces.push({ x: wins.map(t => t.entry_date), y: wins.map(t => t.entry_price),
                    name: 'Win', mode: 'markers',
                    marker: { symbol: 'triangle-up', size: 7, color: '#3fb950' }});
            if (losses.length > 0)
                traces.push({ x: losses.map(t => t.entry_date), y: losses.map(t => t.entry_price),
                    name: 'Loss', mode: 'markers',
                    marker: { symbol: 'x', size: 6, color: '#f85149' }});
        }

        Plotly.newPlot('priceChart', traces, {
            ...plotLayout, height: 380,
            yaxis: { ...plotLayout.yaxis, title: 'Price ($)', tickprefix: '$', type: 'log' },
        }, { responsive: true });
    }

    function renderDrawdownChart(d) {
        const traces = [];
        for (const [key, val] of Object.entries(d.results)) {
            const eq = val.equity;
            if (!eq || eq.length === 0) continue;
            traces.push({
                x: eq.map(e => e.date), y: eq.map(e => -e.drawdown_pct),
                name: val.name, type: 'scatter', mode: 'lines',
                fill: key === selectedStrategy ? 'tozeroy' : undefined,
                line: { color: COLORS[key] || '#8b949e', width: 1.5 },
            });
        }
        Plotly.newPlot('ddChart', traces, {
            ...plotLayout, height: 280,
            yaxis: { ...plotLayout.yaxis, title: 'Drawdown %', ticksuffix: '%' },
        }, { responsive: true });
    }

    function renderMonthlyChart(d) {
        if (!selectedStrategy || !d.results[selectedStrategy]) return;
        const trades = d.results[selectedStrategy].trades;
        if (!trades || trades.length === 0) return;

        const monthly = {};
        trades.forEach(t => {
            const dt = new Date(t.exit_date);
            const ym = dt.getFullYear() + '-' + String(dt.getMonth()+1).padStart(2,'0');
            monthly[ym] = (monthly[ym] || 0) + (t.profit_pct || 0);
        });

        const items = Object.entries(monthly).sort((a, b) => a[0].localeCompare(b[0]));
        Plotly.newPlot('monthlyChart', [{
            x: items.map(([k]) => k), y: items.map(([, v]) => v),
            type: 'bar', marker: { color: items.map(([, v]) => v >= 0 ? '#3fb950' : '#f85149') },
        }], {
            ...plotLayout, height: 280, showlegend: false,
            yaxis: { ...plotLayout.yaxis, title: 'Return %', ticksuffix: '%' },
        }, { responsive: true });
    }

    function renderPnlChart(d) {
        if (!selectedStrategy || !d.results[selectedStrategy]) return;
        const trades = d.results[selectedStrategy].trades;
        if (!trades || trades.length === 0) return;

        Plotly.newPlot('pnlChart', [{
            x: trades.map(t => t.profit_pct), type: 'histogram', nbinsx: 40,
            marker: { color: '#58a6ff', line: { color: '#1c2333', width: 1 } },
        }], {
            ...plotLayout, height: 280, showlegend: false,
            xaxis: { ...plotLayout.xaxis, title: 'P&L %' },
            yaxis: { ...plotLayout.yaxis, title: 'Count' },
        }, { responsive: true });
    }

    function renderWinRateChart(d) {
        const keys = Object.keys(d.results);
        Plotly.newPlot('wrChart', [{
            x: keys.map(k => d.results[k].name),
            y: keys.map(k => d.results[k].metrics.win_rate),
            type: 'bar', marker: { color: keys.map(k => COLORS[k] || '#8b949e') },
            text: keys.map(k => d.results[k].metrics.win_rate + '%'),
            textposition: 'outside', textfont: { color: '#c9d1d9', size: 11 },
        }], {
            ...plotLayout, height: 280, showlegend: false,
            yaxis: { ...plotLayout.yaxis, title: 'Win Rate %', range: [0, 100] },
        }, { responsive: true });
    }

    function renderTradeLog(d, key) {
        const trades = d.results[key]?.trades || [];
        const capital = d.initial_capital;
        const scale = capital / (d._base_capital || 100000);
        const body = document.getElementById('tradeLogBody');
        if (!body) return;

        body.innerHTML = trades.slice().reverse().map(t => {
            const pnlDollar = (t.profit_usd || 0) * scale;
            return '<tr>' +
                '<td>' + t.trade_id + '</td>' +
                '<td>' + (t.entry_date?.substring(0, 10) || '') + '</td>' +
                '<td>' + (t.exit_date?.substring(0, 10) || '') + '</td>' +
                '<td style="color:' + (t.direction === 'long' ? '#3fb950' : '#f85149') + '">' + t.direction.toUpperCase() + '</td>' +
                '<td style="color:' + (COLORS[t.strategy] || '#c9d1d9') + '">' + t.strategy + '</td>' +
                '<td>$' + (t.entry_price?.toLocaleString(undefined, {maximumFractionDigits:0}) || '') + '</td>' +
                '<td>$' + (t.exit_price?.toLocaleString(undefined, {maximumFractionDigits:0}) || '') + '</td>' +
                '<td class="' + (t.profit_pct >= 0 ? 'positive' : 'negative') + '">' +
                (t.profit_pct >= 0 ? '+' : '') + (t.profit_pct?.toFixed(2) || 0) + '%</td>' +
                '<td class="' + (pnlDollar >= 0 ? 'positive' : 'negative') + '">' +
                (pnlDollar >= 0 ? '+' : '') + '$' + Math.abs(pnlDollar).toFixed(0) + '</td>' +
                '<td>' + (t.bars_held || 0) + '</td>' +
                '<td style="color:#8b949e">' + (t.exit_reason || '') + '</td>' +
                '</tr>';
        }).join('');
    }

    // Auto-run on load
    window.addEventListener('load', runBacktest);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/run", methods=["GET", "POST"])
def api_run():
    if request.method == "POST":
        data = request.get_json() or {}
        source = data.get("source", "BTC-USD")
        start = data.get("start") or None
        end = data.get("end") or None
        capital = int(data.get("capital", 10000))
        param_overrides = data.get("params", {})
        force = bool(data.get("force", False))
    else:
        source = request.args.get("source", "BTC-USD")
        start = request.args.get("start") or None
        end = request.args.get("end") or None
        capital = int(request.args.get("capital", 10000))
        param_overrides = {}
        force = False

    if start in ("null", ""): start = None
    if end in ("null", ""): end = None

    # Convert param override values to proper types
    for skey, sparams in param_overrides.items():
        for pname, pval in sparams.items():
            if isinstance(pval, str):
                try:
                    sparams[pname] = float(pval) if '.' in pval else int(pval)
                except ValueError:
                    pass

    # If force rerun, delete existing snapshot first
    if force:
        snap_key = _snapshot_key(source, start, end, param_overrides)
        snap_path = _snapshot_path(snap_key)
        if snap_path.exists():
            snap_path.unlink()
            print(f"[cache] Deleted snapshot {snap_key} (force rerun)")

    output = run_strategies(
        source=source,
        start_date=start,
        end_date=end,
        initial_capital=capital,
        param_overrides=param_overrides,
    )

    # Return gzipped JSON for fast transfer over Codespace proxy
    return _gzip_json_response(output)


@app.route("/api/params")
def api_params():
    """Return tunable parameters for a given strategy."""
    key = request.args.get("strategy", "H2")
    if key not in STRATEGY_REGISTRY:
        return jsonify({})
    strat_cls = STRATEGY_REGISTRY[key]
    return jsonify(getattr(strat_cls, "PARAMS", {}))


if __name__ == "__main__":
    # Clear old snapshots (format may have changed)
    for old in SNAPSHOT_DIR.glob("*.json"):
        old.unlink()
    print("Cleared old snapshots.")

    # Pre-warm with the exact default form values so first load is instant
    default_start = "2020-01-01"
    default_end = "2026-02-13"
    print(f"Pre-generating default BTC-USD snapshot ({default_start} to {default_end})...")
    run_strategies("BTC-USD", default_start, default_end)
    print("Default snapshot ready — first load will be instant.")

    n_snaps = len(list(SNAPSHOT_DIR.glob("*.json")))
    print(f"\n{'=' * 55}")
    print(f"  Trading Backtester Dashboard v2")
    print(f"  http://localhost:5000")
    print(f"  Snapshots cached: {n_snaps}")
    print(f"  Features: BTC + ETH | 8 Strategies | Param Tuning")
    print(f"{'=' * 55}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
