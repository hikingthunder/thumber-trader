"""HTML renderers for dashboard pages."""

from __future__ import annotations

import html
import json
from typing import Any, Dict


def render_dashboard_home_html(snapshot: Dict[str, Any]) -> str:
    rows = []
    for oid, order in snapshot.get("orders", {}).items():
        rows.append(
            f"<tr><td>{html.escape(str(oid))}</td><td>{html.escape(str(order.get('side')))}</td><td>{html.escape(str(order.get('price')))}</td>"
            f"<td>{html.escape(str(order.get('base_size')))}</td><td>{html.escape(str(order.get('grid_index')))}</td></tr>"
        )
    events = "".join(f"<li>{html.escape(str(event))}</li>" for event in snapshot.get("recent_events", []))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Thumber Trader Dashboard</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin:0; background:#0b1220; color:#e8eefc; }}
    .container {{ max-width: 1100px; margin: 20px auto; padding: 0 16px 24px; }}
    .card {{ background:#121a2b; border:1px solid #2d3a58; border-radius:10px; padding:12px; margin-bottom:12px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
    .k {{ font-size:12px; color:#93a0bf; }} .v {{ font-size:18px; font-weight:600; }}
    table {{ width:100%; border-collapse:collapse; }}
    td, th {{ border-bottom:1px solid #2d3a58; padding:8px; text-align:left; }}
    button, .btn {{ border:0; border-radius:8px; padding:10px 12px; cursor:pointer; margin-right:8px; text-decoration:none; display:inline-block; }}
    .primary {{ background:#4f8cff; color:white; }} .danger {{ background:#d95f5f; color:white; }}
    .muted {{ color:#93a0bf; }} #action-status {{ margin-top:8px; }}
    #price-chart {{ height: 440px; width: 100%; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Thumber Trader Dashboard</h1>
    <p class="muted">Live runtime status and controls.</p>

    <div class="card grid">
      <div><div class="k">Product</div><div class="v" id="product-id">{snapshot.get('product_id')}</div></div>
      <div><div class="k">Last Price</div><div class="v" id="last-price">{snapshot.get('last_price')}</div></div>
      <div><div class="k">Trend</div><div class="v" id="trend-bias">{snapshot.get('trend_bias')}</div></div>
      <div><div class="k">Portfolio (USD)</div><div class="v" id="portfolio-value-usd">{snapshot.get('portfolio_value_usd')}</div></div>
      <div><div class="k">Active Orders</div><div class="v" id="active-orders">{snapshot.get('active_orders')}</div></div>
      <div><div class="k">Fills</div><div class="v" id="fills">{snapshot.get('fills')}</div></div>
      <div><div class="k">Survival Prob. (30d)</div><div class="v" id="survival-probability">{snapshot.get('risk_metrics', {}).get('survival_probability_30d', '1')}</div></div>
      <div><div class="k">Risk of Ruin (30d)</div><div class="v" id="risk-of-ruin">{snapshot.get('risk_metrics', {}).get('risk_of_ruin_30d', '0')}</div></div>
    </div>

    <div class="card">
      <h2>Grid Price Action</h2>
      <p class="muted">Candles stream live with order layers and fill markers.</p>
      <div id="price-chart"></div>
    </div>

    <div class="card">
      <button class="danger" onclick="sendAction('kill_switch')">Emergency Kill Switch</button>
      <button class="primary" onclick="sendAction('reanchor')">Manual Re-anchor</button>
      <button class="primary" onclick="openConfigWindow()">Open Configuration</button>
      <a class="btn primary" href="/config">Open Config Page</a>
      <div id="action-status" aria-live="polite"></div>
    </div>

    <div class="card"><h2>Open Orders</h2><table><thead><tr><th>Order ID</th><th>Side</th><th>Price</th><th>Base Size</th><th>Grid Index</th></tr></thead><tbody id="orders-body">{''.join(rows)}</tbody></table></div>
    <div class="card"><h2>Recent Events</h2><ul id="recent-events">{events}</ul><p><a style="color:#4f8cff" href="/api/status">JSON API</a> · <a style="color:#4f8cff" href="/api/tax_report.csv">Tax CSV</a></p></div>
  </div>

  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <script>
    const initialSnapshot = {json.dumps(snapshot)};
    let chart;
    let candleSeries;
    let buyOrderSeries = [];
    let sellOrderSeries = [];

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}
    function asNumber(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }}
    function buildOrderLayers(orders) {{
      const grouped = {{ BUY: [], SELL: [] }};
      Object.values(orders || {{}}).forEach((order) => {{
        const side = String(order?.side || '').toUpperCase();
        const price = asNumber(order?.price);
        if (!price || !(side in grouped)) return;
        grouped[side].push(price);
      }});
      grouped.BUY.sort((a, b) => a - b);
      grouped.SELL.sort((a, b) => a - b);
      return grouped;
    }}
    function renderOrderLayers(snapshot) {{
      for (const series of buyOrderSeries) chart.removeSeries(series);
      for (const series of sellOrderSeries) chart.removeSeries(series);
      buyOrderSeries = [];
      sellOrderSeries = [];

      const layers = buildOrderLayers(snapshot.orders || {{}});
      const candles = snapshot.chart?.candles || [];
      if (!candles.length) return;
      const firstTime = candles[0].time;
      const lastTime = candles[candles.length - 1].time;

      for (const price of layers.BUY) {{
        const line = chart.addLineSeries({{ color: 'rgba(65, 216, 135, 0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
        line.setData([{{ time: firstTime, value: price }}, {{ time: lastTime, value: price }}]);
        buyOrderSeries.push(line);
      }}
      for (const price of layers.SELL) {{
        const line = chart.addLineSeries({{ color: 'rgba(255, 103, 103, 0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
        line.setData([{{ time: firstTime, value: price }}, {{ time: lastTime, value: price }}]);
        sellOrderSeries.push(line);
      }}
    }}
    function renderFillMarkers(snapshot) {{
      const fills = (snapshot.chart?.recent_fills || []).map((fill) => {{
        const ts = asNumber(fill?.ts);
        const price = asNumber(fill?.price);
        const side = String(fill?.side || '').toUpperCase();
        if (!ts || !price || !side) return null;
        const isBuy = side === 'BUY';
        return {{
          time: Math.floor(ts),
          position: isBuy ? 'belowBar' : 'aboveBar',
          color: isBuy ? '#38d489' : '#ff6b6b',
          shape: isBuy ? 'arrowUp' : 'arrowDown',
          text: `${{side}} @ $${{price.toFixed(2)}}`,
        }};
      }}).filter(Boolean);
      candleSeries.setMarkers(fills);
    }}
    function renderChart(snapshot) {{
      if (!chart || !candleSeries) return;
      const candles = snapshot.chart?.candles || [];
      if (candles.length) {{
        candleSeries.setData(candles.map((c) => ({{
          time: Number(c.time),
          open: Number(c.open),
          high: Number(c.high),
          low: Number(c.low),
          close: Number(c.close),
        }})));
      }}
      renderOrderLayers(snapshot);
      renderFillMarkers(snapshot);
      chart.timeScale().fitContent();
    }}
    function initChart() {{
      const chartContainer = document.getElementById('price-chart');
      chart = LightweightCharts.createChart(chartContainer, {{
        layout: {{ background: {{ color: '#121a2b' }}, textColor: '#d6e3ff' }},
        grid: {{ vertLines: {{ color: '#24314e' }}, horzLines: {{ color: '#24314e' }} }},
        rightPriceScale: {{ borderColor: '#2d3a58' }},
        timeScale: {{ borderColor: '#2d3a58', timeVisible: true }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      }});
      candleSeries = chart.addCandlestickSeries({{
        upColor: '#38d489',
        downColor: '#ff6b6b',
        borderVisible: false,
        wickUpColor: '#38d489',
        wickDownColor: '#ff6b6b',
      }});

      const observer = new ResizeObserver(() => {{
        chart.applyOptions({{ width: chartContainer.clientWidth, height: chartContainer.clientHeight }});
      }});
      observer.observe(chartContainer);
    }}
    function applySnapshot(snapshot) {{
      const setText = (id, value) => {{
        const el = document.getElementById(id);
        if (el) el.textContent = value ?? '';
      }};
      setText('product-id', snapshot.product_id);
      setText('last-price', snapshot.last_price);
      setText('trend-bias', snapshot.trend_bias);
      setText('portfolio-value-usd', snapshot.portfolio_value_usd);
      setText('active-orders', snapshot.active_orders);
      setText('fills', snapshot.fills);
      setText('survival-probability', snapshot.risk_metrics?.survival_probability_30d);
      setText('risk-of-ruin', snapshot.risk_metrics?.risk_of_ruin_30d);

      const ordersBody = document.getElementById('orders-body');
      if (ordersBody) {{
        const rows = Object.entries(snapshot.orders || {{}}).map(([oid, order]) =>
          `<tr><td>${{escapeHtml(oid)}}</td><td>${{escapeHtml(order?.side)}}</td><td>${{escapeHtml(order?.price)}}</td><td>${{escapeHtml(order?.base_size)}}</td><td>${{escapeHtml(order?.grid_index)}}</td></tr>`
        );
        ordersBody.innerHTML = rows.join('');
      }}

      const recentEvents = document.getElementById('recent-events');
      if (recentEvents) {{
        const events = (snapshot.recent_events || []).map((event) => `<li>${{escapeHtml(event)}}</li>`);
        recentEvents.innerHTML = events.join('');
      }}

      renderChart(snapshot);
    }}
    function connectEventStream() {{
      const source = new EventSource('/api/stream');
      source.onmessage = (event) => {{
        try {{
          applySnapshot(JSON.parse(event.data));
        }} catch (_err) {{}}
      }};
      source.onerror = () => {{
        source.close();
        setTimeout(connectEventStream, 1000);
      }};
    }}

    function renderStatus(msg, ok=true) {{
      const el = document.getElementById('action-status');
      el.style.color = ok ? '#b9f0d6' : '#ffd1d1';
      el.textContent = msg;
    }}
    async function sendAction(action, payload={{}}) {{
      try {{
        const resp = await fetch('/api/action', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{action, ...payload}}) }});
        const data = await resp.json();
        renderStatus(data.ok ? `Action "${{action}}" completed.` : (data.error || 'Action failed'), Boolean(data.ok));
      }} catch (err) {{ renderStatus(`Request failed: ${{err}}`, false); }}
    }}
    function openConfigWindow() {{
      const popup = window.open('/config?popup=1', 'thumber-config', 'width=980,height=820');
      if (!popup) window.location.href = '/config';
    }}

    initChart();
    applySnapshot(initialSnapshot);
    connectEventStream();
  </script>
</body>
</html>"""



def render_config_html(snapshot: Dict[str, Any], popup_mode: bool = False) -> str:
    config = snapshot.get("config", {})
    config_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td><input data-key=\"{html.escape(str(k))}\" value=\"{html.escape(str(v))}\" /></td></tr>"
        for k, v in sorted(config.items())
    )

    dismiss_js = "window.close();" if popup_mode else "window.location.href='/'"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Thumber Trader Configuration</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin:0; background:#0b1220; color:#e8eefc; }}
    .container {{ max-width: 980px; margin: 20px auto; padding: 0 16px 24px; }}
    .card {{ background:#121a2b; border:1px solid #2d3a58; border-radius:10px; padding:12px; margin-bottom:12px; }}
    table {{ width:100%; border-collapse:collapse; }}
    td, th {{ border-bottom:1px solid #2d3a58; padding:8px; text-align:left; }}
    input {{ width:100%; background:#1a2439; color:#e8eefc; border:1px solid #2d3a58; border-radius:6px; padding:6px; }}
    button {{ border:0; border-radius:8px; padding:10px 12px; cursor:pointer; margin-right:8px; }}
    .primary {{ background:#4f8cff; color:white; }} .neutral {{ background:#4b5776; color:white; }}
    .muted {{ color:#93a0bf; }} #action-status {{ margin-top:10px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"card\">
      <h1>Configuration</h1>
      <p class=\"muted\">Edit values, then save to `.env` and apply without SSH edits.</p>
      <table>
        <thead><tr><th>Variable</th><th>Value</th></tr></thead>
        <tbody>{config_rows}</tbody>
      </table>
      <div style=\"margin-top:10px\">
        <button class=\"primary\" onclick=\"applyRuntime()\">Apply Runtime</button>
        <button class=\"primary\" onclick=\"saveConfig()\">Save to .env + Apply</button>
        <button class=\"neutral\" onclick=\"dismiss()\">Dismiss</button>
      </div>
      <div id=\"action-status\" aria-live=\"polite\"></div>
    </div>
  </div>

  <script>
    function collectUpdates() {{
      const updates = {{}};
      document.querySelectorAll('input[data-key]').forEach((el) => {{ updates[el.dataset.key] = el.value; }});
      return updates;
    }}
    function renderStatus(msg, ok=true) {{
      const el = document.getElementById('action-status');
      el.style.color = ok ? '#b9f0d6' : '#ffd1d1';
      el.textContent = msg;
    }}
    async function call(action, payload={{}}) {{
      const resp = await fetch('/api/action', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{action, ...payload}}) }});
      return resp.json();
    }}
    async function applyRuntime() {{
      try {{
        const data = await call('reload_config', {{updates: collectUpdates()}});
        renderStatus(data.ok ? 'Runtime config applied.' : (data.error || 'Apply failed'), Boolean(data.ok));
      }} catch (err) {{ renderStatus(`Request failed: ${{err}}`, false); }}
    }}
    async function saveConfig() {{
      try {{
        const envPath = prompt('Config file path (.env default):', '.env') || '.env';
        const data = await call('save_config', {{updates: collectUpdates(), env_path: envPath}});
        if (!data.ok) {{ renderStatus(data.error || 'Save failed', false); return; }}
        renderStatus('Configuration saved and applied. Closing...', true);
        setTimeout(() => dismiss(), 700);
      }} catch (err) {{ renderStatus(`Request failed: ${{err}}`, false); }}
    }}
    function dismiss() {{ {dismiss_js}; }}
  </script>
</body>
</html>"""
