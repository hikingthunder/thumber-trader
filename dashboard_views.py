"""HTML renderers for dashboard pages."""

from __future__ import annotations

import html
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
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"refresh\" content=\"10\" />
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
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Thumber Trader Dashboard</h1>
    <p class=\"muted\">Live runtime status and controls.</p>

    <div class=\"card grid\">
      <div><div class=\"k\">Product</div><div class=\"v\">{snapshot.get('product_id')}</div></div>
      <div><div class=\"k\">Last Price</div><div class=\"v\">{snapshot.get('last_price')}</div></div>
      <div><div class=\"k\">Trend</div><div class=\"v\">{snapshot.get('trend_bias')}</div></div>
      <div><div class=\"k\">Portfolio (USD)</div><div class=\"v\">{snapshot.get('portfolio_value_usd')}</div></div>
      <div><div class=\"k\">Active Orders</div><div class=\"v\">{snapshot.get('active_orders')}</div></div>
      <div><div class=\"k\">Fills</div><div class=\"v\">{snapshot.get('fills')}</div></div>
    </div>

    <div class=\"card\">
      <button class=\"danger\" onclick=\"sendAction('kill_switch')\">Emergency Kill Switch</button>
      <button class=\"primary\" onclick=\"sendAction('reanchor')\">Manual Re-anchor</button>
      <button class=\"primary\" onclick=\"openConfigWindow()\">Open Configuration</button>
      <a class=\"btn primary\" href=\"/config\">Open Config Page</a>
      <div id=\"action-status\" aria-live=\"polite\"></div>
    </div>

    <div class=\"card\"><h2>Open Orders</h2><table><thead><tr><th>Order ID</th><th>Side</th><th>Price</th><th>Base Size</th><th>Grid Index</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <div class=\"card\"><h2>Recent Events</h2><ul>{events}</ul><p><a style=\"color:#4f8cff\" href=\"/api/status\">JSON API</a> · <a style=\"color:#4f8cff\" href=\"/api/tax_report.csv\">Tax CSV</a></p></div>
  </div>

  <script>
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
