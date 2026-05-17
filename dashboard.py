"""
Interactive dashboard: order book viewer + spread dynamics.
Slider controls both charts simultaneously.
Output: outputs/dashboard.html
"""

import json
import pandas as pd
import numpy as np
from orderbook import OrderBook, load_spot, load_futures
from analysis import compute_spread

FREQ  = "1min"
START = "2025-06-10 10:00"
END   = "2025-06-10 18:45"
DEPTH = 12

print("Loading data...")
spot_df = load_spot("data/20250610_SBER.parquet")
fut_df  = load_futures("data/SBERF_2025_06_10.parquet")

spot_ob = OrderBook(spot_df)
fut_ob  = OrderBook(fut_df)

print("Building book snapshots (single pass)...")
spot_books = spot_ob.build_book_series(FREQ, START, END, DEPTH)
fut_books  = fut_ob.build_book_series(FREQ, START, END, DEPTH)

print("Building spread series...")
spot_mid = spot_ob.build_mid_series(FREQ, START, END)
fut_mid  = fut_ob.build_mid_series(FREQ, START, END)
spread_df = compute_spread(spot_mid, fut_mid)

# Serialize to JS-friendly format
timestamps = [s["ts"] for s in spot_books]

spread_times  = [t.strftime("%H:%M") for t in spread_df.index]
spread_tickvals = [t for t in spread_times if t.endswith(':00') or t.endswith(':30')]
spread_vals   = [round(v, 4) if not np.isnan(v) else None for v in spread_df["spread"]]
spot_mid_vals = [round(v, 4) if not np.isnan(v) else None for v in spread_df["spot_mid"]]
fut_mid_vals  = [round(v, 4) if not np.isnan(v) else None for v in spread_df["futures_mid"]]

# z-score with same window as the backtest (so dashboard echoes the strategy view)
Z_WINDOW = 60
roll_mean = spread_df["spread"].rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).mean()
roll_std  = spread_df["spread"].rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).std()
z_score   = (spread_df["spread"] - roll_mean) / roll_std.replace(0, np.nan)
z_vals    = [round(v, 2) if not (isinstance(v, float) and np.isnan(v)) else None for v in z_score]

def book_to_js(books):
    result = []
    for b in books:
        result.append({
            "ts":          b["ts"],
            "mid":         b["mid"],
            "bid_prices":  [p for p, _ in b["bids"]],
            "bid_vols":    [v for _, v in b["bids"]],
            "ask_prices":  [p for p, _ in b["asks"]],
            "ask_vols":    [v for _, v in b["asks"]],
        })
    return result

def np_default(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    raise TypeError

spot_js = json.dumps(book_to_js(spot_books), default=np_default)
fut_js  = json.dumps(book_to_js(fut_books),  default=np_default)
spread_js = json.dumps({
    "times":     spread_times,
    "spread":    spread_vals,
    "spot_mid":  spot_mid_vals,
    "fut_mid":   fut_mid_vals,
    "zscore":    z_vals,
})

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>SBER vs SBERF — Дашборд · 10.06.2025</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
}}
header {{
  background: #1a1a2e;
  color: #fff;
  padding: 16px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 3px solid #3498db;
}}
header h1 {{ font-size: 18px; font-weight: 600; letter-spacing: .4px; }}
header span {{ font-size: 12px; color: #7f8c8d; }}

.container {{ padding: 20px 28px; display: flex; flex-direction: column; gap: 18px; }}

.card {{
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,.07);
  padding: 18px 22px;
}}
.card h2 {{
  font-size: 14px;
  font-weight: 600;
  border-left: 4px solid #3498db;
  padding-left: 10px;
  margin-bottom: 12px;
  color: #2c3e50;
}}

/* Time slider */
.slider-row {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 14px 22px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,.07);
  flex-wrap: wrap;
}}
.slider-row label {{ font-size: 13px; font-weight: 600; white-space: nowrap; color: #2c3e50; }}
#timeSlider {{
  flex: 1;
  min-width: 200px;
  height: 6px;
  accent-color: #3498db;
  cursor: pointer;
}}
#timeLabel {{
  font-size: 16px;
  font-weight: 700;
  color: #3498db;
  min-width: 48px;
  text-align: center;
}}
.time-input-wrap {{
  display: flex;
  align-items: center;
  gap: 6px;
}}
#timeInput {{
  width: 70px;
  padding: 6px 10px;
  border: 2px solid #dfe6e9;
  border-radius: 6px;
  font-size: 15px;
  font-weight: 600;
  text-align: center;
  color: #2c3e50;
  outline: none;
  transition: border-color .2s;
}}
#timeInput:focus {{ border-color: #3498db; }}
#timeInput.error {{ border-color: #e74c3c; }}
#goBtn {{
  padding: 6px 14px;
  background: #3498db;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background .2s;
}}
#goBtn:hover {{ background: #2980b9; }}
#timeError {{
  font-size: 11px;
  color: #e74c3c;
  min-width: 120px;
}}

/* Stats row */
.stats {{
  display: flex;
  gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}}
.stat {{
  background: #f8f9fa;
  border-radius: 8px;
  padding: 8px 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 120px;
}}
.stat .label {{ font-size: 11px; color: #7f8c8d; margin-bottom: 2px; }}
.stat .value {{ font-size: 16px; font-weight: 700; }}
.stat .value.bid {{ color: #2980b9; }}
.stat .value.ask {{ color: #c0392b; }}
.stat .value.mid {{ color: #27ae60; }}
.stat .value.spread {{ color: #8e44ad; }}

/* Two-column books */
.books-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
.book-pair {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 4px;
}}
.book-pair > div {{ background: #fafbfc; border-radius: 6px; padding: 4px; }}

/* Live spread panel */
.spread-panel {{
  display: grid;
  grid-template-columns: 1fr 1fr 1.4fr 1fr;
  gap: 14px;
  padding: 16px 22px;
  background: linear-gradient(135deg, #1a1a2e 0%, #2c3e50 100%);
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,.12);
  color: #fff;
}}
.spread-panel .item {{
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4px 0;
}}
.spread-panel .item.divider {{
  border-left: 1px solid rgba(255,255,255,.18);
}}
.spread-panel .lab {{
  font-size: 11px;
  color: #95a5a6;
  text-transform: uppercase;
  letter-spacing: .8px;
  margin-bottom: 4px;
}}
.spread-panel .val {{
  font-size: 22px;
  font-weight: 700;
  font-family: 'Consolas', 'Monaco', monospace;
}}
.spread-panel .val.spot {{ color: #5dade2; }}
.spread-panel .val.fut  {{ color: #f5b041; }}
.spread-panel .val.spread {{ color: #58d68d; font-size: 26px; }}
.spread-panel .sub {{
  font-size: 11px;
  color: #bdc3c7;
  margin-top: 2px;
}}
</style>
</head>
<body>

<header>
  <h1>SBER / SBERF · 10 июня 2025</h1>
  <span>Full Orders Log · Тип А · МОСБИРЖА</span>
</header>

<div class="container">

  <!-- Slider + time input -->
  <div class="slider-row">
    <label>Время торгов:</label>
    <input type="range" id="timeSlider" min="0" max="{len(timestamps)-1}" value="0" step="1">
    <span id="timeLabel">{timestamps[0]}</span>
    <div class="time-input-wrap">
      <input type="text" id="timeInput" placeholder="HH:MM" maxlength="5" spellcheck="false">
      <button id="goBtn">→</button>
      <span id="timeError"></span>
    </div>
  </div>

  <!-- Live spread numeric panel -->
  <div class="spread-panel" id="spreadPanel">
    <div class="item">
      <span class="lab">SBER (спот) mid</span>
      <span class="val spot" id="livSpot">—</span>
      <span class="sub">руб.</span>
    </div>
    <div class="item divider">
      <span class="lab">SBERF (фьюч) mid</span>
      <span class="val fut" id="livFut">—</span>
      <span class="sub">руб.</span>
    </div>
    <div class="item divider">
      <span class="lab">Спред: SBERF − SBER</span>
      <span class="val spread" id="livSpread">—</span>
      <span class="sub" id="livSpreadPct">—</span>
    </div>
    <div class="item divider">
      <span class="lab">z-score (window=60)</span>
      <span class="val" id="livZ" style="color:#e74c3c">—</span>
      <span class="sub" id="livZdesc">—</span>
    </div>
  </div>

  <!-- Spread chart -->
  <div class="card">
    <h2>Динамика спреда и цен за день (SBERF − SBER)</h2>
    <div id="spreadChart" style="height:320px"></div>
  </div>

  <!-- Order books -->
  <div class="books-row">
    <div class="card">
      <h2>Стакан SBER (спот)</h2>
      <div class="stats" id="spotStats"></div>
      <div class="book-pair">
        <div id="spotBid" style="height:380px"></div>
        <div id="spotAsk" style="height:380px"></div>
      </div>
    </div>
    <div class="card">
      <h2>Стакан SBERF (фьючерс)</h2>
      <div class="stats" id="futStats"></div>
      <div class="book-pair">
        <div id="futBid" style="height:380px"></div>
        <div id="futAsk" style="height:380px"></div>
      </div>
    </div>
  </div>

</div>

<script>
const SPOT_BOOKS  = {spot_js};
const FUT_BOOKS   = {fut_js};
const SPREAD_DATA = {spread_js};

// ── Init spread chart ─────────────────────────────────────────────────────
const spreadLayout = {{
  height: 320,
  margin: {{t: 10, b: 40, l: 56, r: 56}},
  template: 'plotly_white',
  hovermode: 'x unified',
  legend: {{orientation: 'h', y: 1.08, x: 0}},
  yaxis:  {{title: 'Цена (руб.)', side: 'left'}},
  yaxis2: {{title: 'Спред (руб.)', side: 'right', overlaying: 'y', showgrid: false}},
  xaxis:  {{
    title: '',
    tickmode: 'array',
    tickvals: {json.dumps(spread_tickvals)},
    tickangle: -45,
  }},
  shapes: [{{
    type: 'line', xref: 'x', yref: 'paper',
    x0: SPREAD_DATA.times[0], x1: SPREAD_DATA.times[0],
    y0: 0, y1: 1,
    line: {{color: '#e74c3c', width: 2, dash: 'dot'}},
  }}],
}};

const spreadTraces = [
  {{
    x: SPREAD_DATA.times, y: SPREAD_DATA.spot_mid,
    name: 'SBER (спот)', type: 'scatter', mode: 'lines',
    line: {{color: '#2980b9', width: 1.5}},
  }},
  {{
    x: SPREAD_DATA.times, y: SPREAD_DATA.fut_mid,
    name: 'SBERF (фьючерс)', type: 'scatter', mode: 'lines',
    line: {{color: '#e67e22', width: 1.5}},
  }},
  {{
    x: SPREAD_DATA.times, y: SPREAD_DATA.spread,
    name: 'Спред', type: 'scatter', mode: 'lines',
    yaxis: 'y2',
    line: {{color: '#8e44ad', width: 2}},
    fill: 'tozeroy', fillcolor: 'rgba(142,68,173,0.08)',
  }},
];

Plotly.newPlot('spreadChart', spreadTraces, spreadLayout, {{responsive: true, displayModeBar: false}});

// ── Init book charts (classic exchange style) ────────────────────────────
// BID (зелёный): max price at top, decreasing going down
// ASK (красный): min price at top, increasing going down
// Both use autorange:'reversed' on category y-axis — since bid_prices come
// descending and ask_prices ascending, position 0 (the "best" level) goes to TOP.

const sideLayout = (title, color) => ({{
  height: 380,
  margin: {{t: 30, b: 38, l: 62, r: 14}},
  title: {{text: title, font: {{color: color, size: 13}}, x: 0.5, xanchor: 'center'}},
  template: 'plotly_white',
  showlegend: false,
  xaxis: {{title: '', zeroline: true, zerolinecolor: '#ccc'}},
  yaxis: {{title: 'Цена (руб.)', type: 'category', autorange: 'reversed'}},
  bargap: 0.25,
}});

function bidTrace(book) {{
  return [{{
    type: 'bar', orientation: 'h',
    x: book.bid_vols,
    y: book.bid_prices.map(p => p.toFixed(2)),
    marker: {{color: 'rgba(39,174,96,0.78)', line: {{color:'#27ae60', width:1}}}},
    text: book.bid_vols.map(v => v.toLocaleString('ru')),
    textposition: 'outside',
    hovertemplate: 'Цена: %{{y}}<br>Объём: %{{x:,}}<extra>BID</extra>',
  }}];
}}

function askTrace(book) {{
  return [{{
    type: 'bar', orientation: 'h',
    x: book.ask_vols,
    y: book.ask_prices.map(p => p.toFixed(2)),
    marker: {{color: 'rgba(231,76,60,0.78)', line: {{color:'#c0392b', width:1}}}},
    text: book.ask_vols.map(v => v.toLocaleString('ru')),
    textposition: 'outside',
    hovertemplate: 'Цена: %{{y}}<br>Объём: %{{x:,}}<extra>ASK</extra>',
  }}];
}}

const cfg = {{responsive: true, displayModeBar: false}};
Plotly.newPlot('spotBid', bidTrace(SPOT_BOOKS[0]), sideLayout('BID', '#27ae60'), cfg);
Plotly.newPlot('spotAsk', askTrace(SPOT_BOOKS[0]), sideLayout('ASK', '#c0392b'), cfg);
Plotly.newPlot('futBid',  bidTrace(FUT_BOOKS[0]),  sideLayout('BID', '#27ae60'), cfg);
Plotly.newPlot('futAsk',  askTrace(FUT_BOOKS[0]),  sideLayout('ASK', '#c0392b'), cfg);

// ── Stats helper ──────────────────────────────────────────────────────────
function renderStats(elId, book, otherMid) {{
  const bid    = book.bid_prices.length ? book.bid_prices[0] : null;
  const ask    = book.ask_prices.length ? book.ask_prices[0] : null;
  const mid    = book.mid;
  const baSpread = (bid && ask) ? (ask - bid).toFixed(4) : '—';
  const futSpread = (mid && otherMid) ? (mid - otherMid).toFixed(4) : '—';

  document.getElementById(elId).innerHTML = `
    <div class="stat"><span class="label">Best BID</span><span class="value bid">${{bid ? bid.toFixed(2) : '—'}}</span></div>
    <div class="stat"><span class="label">Best ASK</span><span class="value ask">${{ask ? ask.toFixed(2) : '—'}}</span></div>
    <div class="stat"><span class="label">Mid</span><span class="value mid">${{mid ? mid.toFixed(4) : '—'}}</span></div>
    <div class="stat"><span class="label">Bid-Ask спред</span><span class="value spread">${{baSpread}}</span></div>
  `;
}}

renderStats('spotStats', SPOT_BOOKS[0], FUT_BOOKS[0].mid);
renderStats('futStats',  FUT_BOOKS[0],  SPOT_BOOKS[0].mid);

// ── Live spread panel ─────────────────────────────────────────────────────
// Maps slider index → spread series index (book timestamps include seconds
// because we sample at HH:MM:00, but the spread series uses HH:MM strings)
const TS_TO_SPREAD_IDX = {{}};
SPREAD_DATA.times.forEach((t, i) => TS_TO_SPREAD_IDX[t] = i);

function updateSpreadPanel(ts) {{
  const i = TS_TO_SPREAD_IDX[ts];
  if (i === undefined) return;
  const spot = SPREAD_DATA.spot_mid[i];
  const fut  = SPREAD_DATA.fut_mid[i];
  const sp   = SPREAD_DATA.spread[i];
  const z    = SPREAD_DATA.zscore[i];

  document.getElementById('livSpot').textContent = spot != null ? spot.toFixed(4) : '—';
  document.getElementById('livFut').textContent  = fut  != null ? fut.toFixed(4)  : '—';

  if (sp != null) {{
    const sign = sp >= 0 ? '+' : '';
    document.getElementById('livSpread').textContent = sign + sp.toFixed(4) + ' руб';
    const pct = spot ? (sp / spot * 100) : 0;
    document.getElementById('livSpreadPct').textContent = sign + pct.toFixed(4) + ' %  от спота';
    document.getElementById('livSpread').style.color = sp >= 0 ? '#58d68d' : '#ec7063';
  }} else {{
    document.getElementById('livSpread').textContent = '—';
    document.getElementById('livSpreadPct').textContent = '—';
  }}

  const zEl = document.getElementById('livZ');
  const zDesc = document.getElementById('livZdesc');
  if (z != null) {{
    const zSign = z >= 0 ? '+' : '';
    zEl.textContent = zSign + z.toFixed(2) + ' σ';
    if (Math.abs(z) >= 2.0) {{
      zEl.style.color = '#e74c3c';
      zDesc.textContent = z > 0 ? 'СИГНАЛ: SHORT spread' : 'СИГНАЛ: LONG spread';
    }} else if (Math.abs(z) >= 1.0) {{
      zEl.style.color = '#f39c12';
      zDesc.textContent = 'умеренное отклонение';
    }} else {{
      zEl.style.color = '#95a5a6';
      zDesc.textContent = 'около среднего';
    }}
  }} else {{
    zEl.textContent = '—';
    zDesc.textContent = 'window прогревается';
    zEl.style.color = '#7f8c8d';
  }}
}}

updateSpreadPanel(SPOT_BOOKS[0].ts);

// ── Build time index for fast lookup ─────────────────────────────────────
const TIME_INDEX = {{}};
SPOT_BOOKS.forEach((b, i) => TIME_INDEX[b.ts] = i);

// ── Core update function ──────────────────────────────────────────────────
function goToIndex(idx) {{
  if (idx < 0 || idx >= SPOT_BOOKS.length) return;
  const ts = SPOT_BOOKS[idx].ts;
  slider.value = idx;
  label.textContent = ts;

  Plotly.react('spotBid', bidTrace(SPOT_BOOKS[idx]), sideLayout('BID', '#27ae60'), cfg);
  Plotly.react('spotAsk', askTrace(SPOT_BOOKS[idx]), sideLayout('ASK', '#c0392b'), cfg);
  Plotly.react('futBid',  bidTrace(FUT_BOOKS[idx]),  sideLayout('BID', '#27ae60'), cfg);
  Plotly.react('futAsk',  askTrace(FUT_BOOKS[idx]),  sideLayout('ASK', '#c0392b'), cfg);

  Plotly.relayout('spreadChart', {{'shapes[0].x0': ts, 'shapes[0].x1': ts}});
  renderStats('spotStats', SPOT_BOOKS[idx], FUT_BOOKS[idx].mid);
  renderStats('futStats',  FUT_BOOKS[idx],  SPOT_BOOKS[idx].mid);
  updateSpreadPanel(ts);
}}

// ── Slider ────────────────────────────────────────────────────────────────
const slider = document.getElementById('timeSlider');
const label  = document.getElementById('timeLabel');
slider.addEventListener('input', function() {{
  goToIndex(parseInt(this.value));
}});

// ── Time input ────────────────────────────────────────────────────────────
const timeInput = document.getElementById('timeInput');
const goBtn     = document.getElementById('goBtn');
const timeError = document.getElementById('timeError');

function applyTimeInput() {{
  let val = timeInput.value.trim();

  // Auto-format: "1230" → "12:30"
  if (/^\d{{4}}$/.test(val)) val = val.slice(0,2) + ':' + val.slice(2);

  timeError.textContent = '';
  timeInput.classList.remove('error');

  if (!/^\d{{2}}:\d{{2}}$/.test(val)) {{
    timeInput.classList.add('error');
    timeError.textContent = 'Формат: ЧЧ:ММ (напр. 12:30)';
    return;
  }}

  if (val in TIME_INDEX) {{
    timeInput.value = val;
    goToIndex(TIME_INDEX[val]);
  }} else {{
    // Find nearest available minute
    const [hh, mm] = val.split(':').map(Number);
    const target = hh * 60 + mm;
    let best = -1, bestDiff = Infinity;
    SPOT_BOOKS.forEach((b, i) => {{
      const [bh, bm] = b.ts.split(':').map(Number);
      const d = Math.abs(bh * 60 + bm - target);
      if (d < bestDiff) {{ bestDiff = d; best = i; }}
    }});
    if (best >= 0) {{
      timeError.textContent = `→ ближайшее: ${{SPOT_BOOKS[best].ts}}`;
      goToIndex(best);
    }} else {{
      timeInput.classList.add('error');
      timeError.textContent = 'Вне торговой сессии (10:00–18:45)';
    }}
  }}
}}

goBtn.addEventListener('click', applyTimeInput);
timeInput.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') applyTimeInput(); }});
</script>
</body>
</html>"""

out = "outputs/dashboard.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Done → {out}")
