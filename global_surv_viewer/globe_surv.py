"""
╔═══════════════════════════════════════════════════════════╗
║   GEO-SURVEILLANCE FEED  ·  VERSION 0.2 (TEST BUILD)     ║
║   Global Vision Platform — Public OSINT Edition           ║
╠═══════════════════════════════════════════════════════════╣
║  NEW in v0.2:                                             ║
║   🔥 NASA FIRMS wildfire hotspot layer                    ║
║   🎛️  Layer toggle panel (show/hide each feed)            ║
║   🖱️  Click-to-inspect any aircraft or earthquake         ║
║   📊 Log tabs: Aircraft | Earthquakes split view          ║
║   ✈️  Altitude filter slider for aircraft                  ║
║   🚢 Maritime layer stub (v0.3: AIS WebSocket)            ║
╠═══════════════════════════════════════════════════════════╣
║  INSTALL (same as v0.1, no new packages):                 ║
║    pip install dash dash-bootstrap-components pandas      ║
║               plotly requests opencv-python numpy         ║
║               --break-system-packages                     ║
║                                                           ║
║  OPTIONAL — NASA FIRMS fires layer:                       ║
║    1. Go to: https://firms.modaps.eosdis.nasa.gov/api/    ║
║    2. Enter your email → get MAP_KEY in seconds           ║
║    3. Paste it into FIRMS_MAP_KEY below                   ║
║                                                           ║
║  RUN:   python3 gsf_v02.py                                ║
║  URL:   http://127.0.0.1:8050                             ║
╚═══════════════════════════════════════════════════════════╝
"""

import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
import cv2
import base64
import threading
import time
import sys
import io
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit here
# ─────────────────────────────────────────────────────────────
VERSION = "0.2.0"

# ── Camera ────────────────────────────────────────────────────
CAMERA_SOURCES = [
    "http://pendelcam.dorfcam.de/mjpg/video.mjpg",
    "http://webcam1.lpl.arizona.edu/mjpeg.cgi",
    "http://195.196.36.242/mjpg/video.mjpg",
]

# ── Public APIs ───────────────────────────────────────────────
OPENSKY_URL = "https://opensky-network.org/api/states/all"
USGS_EQ_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"

# NASA FIRMS — paste your free MAP_KEY here (leave blank to skip)
FIRMS_MAP_KEY = ""   # ← e.g. "d1a1c34c52f82f09e2327e87df27f073"
FIRMS_URL_TPL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    "/{key}/VIIRS_SNPP_NRT/world/1/today"
)

# ── Refresh rates (ms) ────────────────────────────────────────
MAP_REFRESH_MS  = 14000   # ADS-B + globe re-render
CAM_REFRESH_MS  = 120     # camera frames
EQ_REFRESH_MS   = 45000   # earthquakes
FIRE_REFRESH_MS = 120000  # fires (2 min — NASA updates every 3 h)

# ── Caps ──────────────────────────────────────────────────────
MAX_FLIGHTS = 450
MAX_EQ      = 120
MAX_FIRES   = 500

# ─────────────────────────────────────────────────────────────
# CAMERA STREAM — async MJPEG with auto-fallback
# ─────────────────────────────────────────────────────────────
class TacticalStream:
    def __init__(self, sources=CAMERA_SOURCES):
        self.sources   = sources
        self.cap_index = 0
        self.stopped   = False
        self.lock      = threading.Lock()
        self.frame     = None
        self.grabbed   = False
        self.stream    = None
        self._connect()

    def _connect(self):
        src = self.sources[self.cap_index % len(self.sources)]
        print(f"[CAM] → {src}")
        if self.stream:
            self.stream.release()
        self.stream = cv2.VideoCapture(src)
        if self.stream.isOpened():
            self.grabbed, self.frame = self.stream.read()
            print("[CAM] ✓ online")
        else:
            print("[CAM] ✗ offline — will cycle to next source")
            self.grabbed = False
            self.frame   = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        fails = 0
        while not self.stopped:
            if self.stream and self.stream.isOpened():
                ok, frame = self.stream.read()
                if ok:
                    with self.lock:
                        self.grabbed, self.frame = ok, frame
                    fails = 0
                else:
                    fails += 1
                    if fails > 20:
                        self.cap_index += 1
                        self._connect()
                        fails = 0
            else:
                time.sleep(2)
                self._connect()
            time.sleep(0.033)

    def read_encoded(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                ts  = datetime.now().strftime("%H:%M:%S")
                cv2.putText(img, "NO SIGNAL — RETRYING", (110, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 55, 55), 2)
                cv2.putText(img, ts, (255, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 70), 1)
                _, buf = cv2.imencode('.jpg', img)
            else:
                fr = cv2.resize(self.frame, (640, 360))
                ts = datetime.now(timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S")
                cv2.putText(fr, "◉ LIVE", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
                cv2.putText(fr, ts, (8, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 100), 1)
                _, buf = cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode('utf-8')

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()


cam = TacticalStream().start()

# ─────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────

def fetch_flights():
    try:
        r = requests.get(OPENSKY_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        if 'states' not in data or not data['states']:
            return pd.DataFrame()
        df = pd.DataFrame(data['states']).iloc[:MAX_FLIGHTS, [0,1,2,5,6,7,9,10]]
        df.columns = ['icao','callsign','country','lon','lat','alt','vel','hdg']
        df.dropna(subset=['lon','lat'], inplace=True)
        df['alt']      = df['alt'].fillna(0).round(0)
        df['vel']      = df['vel'].fillna(0).round(1)
        df['hdg']      = df['hdg'].fillna(0).round(0)
        df['callsign'] = df['callsign'].str.strip().replace('', 'UNKNOWN')
        df['alt_km']   = (df['alt'] / 1000).round(2)
        return df
    except Exception as e:
        print(f"[ADS-B] {e}")
        return pd.DataFrame()


def fetch_earthquakes():
    try:
        r = requests.get(USGS_EQ_URL, timeout=10)
        r.raise_for_status()
        rows = []
        for f in r.json().get('features', [])[:MAX_EQ]:
            p, c = f['properties'], f['geometry']['coordinates']
            rows.append({
                'lon':   c[0], 'lat': c[1],
                'depth': round(c[2], 1),
                'mag':   p.get('mag', 0) or 0,
                'place': p.get('place', 'Unknown'),
                'time':  datetime.fromtimestamp(
                             p['time']/1000, tz=timezone.utc
                         ).strftime('%H:%M UTC'),
                'url':   p.get('url', ''),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[EQ] {e}")
        return pd.DataFrame()


def fetch_fires():
    """NASA FIRMS VIIRS — requires free MAP_KEY."""
    if not FIRMS_MAP_KEY:
        return pd.DataFrame(), "no_key"
    try:
        url = FIRMS_URL_TPL.format(key=FIRMS_MAP_KEY)
        r   = requests.get(url, timeout=15)
        r.raise_for_status()
        df  = pd.read_csv(io.StringIO(r.text))
        if df.empty or 'latitude' not in df.columns:
            return pd.DataFrame(), "empty"
        df = df[['latitude','longitude','bright_ti4','frp','daynight']].dropna()
        df = df.rename(columns={'latitude':'lat','longitude':'lon',
                                'bright_ti4':'brightness','frp':'power'})
        df['power'] = df['power'].fillna(0).round(1)
        df = df.head(MAX_FIRES)
        return df, "ok"
    except Exception as e:
        print(f"[FIRES] {e}")
        return pd.DataFrame(), "error"


# ─────────────────────────────────────────────────────────────
# GLOBE FIGURE BUILDER
# ─────────────────────────────────────────────────────────────
_GEO_BASE = dict(
    showframe      = False,
    bgcolor        = 'rgba(0,0,0,0)',
    showcoastlines = True,  coastlinecolor = '#1a6644',
    showland       = True,  landcolor      = 'rgb(14,19,16)',
    showocean      = True,  oceancolor     = 'rgb(5,10,20)',
    showlakes      = True,  lakecolor      = 'rgb(7,12,25)',
    showcountries  = True,  countrycolor   = 'rgb(30,50,38)',
    showrivers     = True,  rivercolor     = 'rgb(7,30,50)',
)
_LAYOUT_BASE = dict(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = dict(color='#a0ffcc', family='monospace', size=11),
    margin        = dict(l=0, r=0, t=28, b=0),
    showlegend    = True,
    legend        = dict(
        bgcolor='rgba(0,18,8,0.85)', bordercolor='#0d3320', borderwidth=1,
        font=dict(size=9, color='#a0ffcc'), x=0.01, y=0.99,
    ),
)


def build_globe(df_fl, df_eq, df_fire, layers, view_mode, alt_min):
    traces = []

    # ── Flights ──────────────────────────────────────────────
    if 'flights' in layers and not df_fl.empty:
        filt = df_fl[df_fl['alt'] >= alt_min * 1000]
        if not filt.empty:
            hover = (
                "<b>✈ " + filt['callsign'] + "</b><br>"
                + "Country: " + filt['country'] + "<br>"
                + "Alt: " + filt['alt'].astype(int).astype(str) + " m  ("
                           + filt['alt_km'].astype(str) + " km)<br>"
                + "Speed: " + filt['vel'].astype(str) + " m/s<br>"
                + "Heading: " + filt['hdg'].astype(int).astype(str) + "°<br>"
                + "ICAO: <i>" + filt['icao'] + "</i>"
            )
            traces.append(go.Scattergeo(
                lon=filt['lon'], lat=filt['lat'],
                text=hover, hoverinfo='text',
                name=f"✈ Aircraft ({len(filt)})",
                mode='markers',
                marker=dict(
                    size=5, opacity=0.85,
                    color=filt['alt'],
                    colorscale=[[0,'#002a18'],[0.3,'#00884a'],[0.7,'#00ddaa'],[1,'#00ffee']],
                    cmin=0, cmax=13000,
                    colorbar=dict(
                        title=dict(text="Alt (m)", font=dict(color='#a0ffcc',size=9)),
                        thickness=7, len=0.38, x=1.01,
                        tickfont=dict(color='#a0ffcc',size=8),
                        bgcolor='rgba(0,0,0,0)', bordercolor='#0d3320',
                    ),
                    line=dict(width=0),
                ),
                customdata=filt[['icao','callsign','country','alt','vel','hdg']].values,
            ))

    # ── Earthquakes ───────────────────────────────────────────
    if 'quakes' in layers and not df_eq.empty:
        hover = (
            "<b>⚡ M" + df_eq['mag'].round(1).astype(str) + "</b><br>"
            + df_eq['place'] + "<br>"
            + "Depth: " + df_eq['depth'].astype(str) + " km<br>"
            + df_eq['time']
        )
        traces.append(go.Scattergeo(
            lon=df_eq['lon'], lat=df_eq['lat'],
            text=hover, hoverinfo='text',
            name=f"⚡ Earthquakes ({len(df_eq)})",
            mode='markers',
            marker=dict(
                size=(df_eq['mag'].clip(2,8)*3).tolist(),
                opacity=0.78,
                color=df_eq['mag'].tolist(),
                colorscale=[[0,'#2a0800'],[0.5,'#cc4400'],[1,'#ff1100']],
                cmin=2.5, cmax=7.5,
                symbol='diamond',
                line=dict(width=1, color='rgba(255,80,0,0.4)'),
            ),
            customdata=df_eq[['mag','place','depth','time']].values,
        ))

    # ── Wildfires ─────────────────────────────────────────────
    if 'fires' in layers and not df_fire.empty:
        hover = (
            "<b>🔥 Wildfire Hotspot</b><br>"
            + "Brightness: " + df_fire['brightness'].round(1).astype(str) + " K<br>"
            + "FRP: " + df_fire['power'].astype(str) + " MW<br>"
            + "Day/Night: " + df_fire['daynight']
        )
        traces.append(go.Scattergeo(
            lon=df_fire['lon'], lat=df_fire['lat'],
            text=hover, hoverinfo='text',
            name=f"🔥 Fire Hotspots ({len(df_fire)})",
            mode='markers',
            marker=dict(
                size=4, opacity=0.7,
                color=df_fire['brightness'].tolist(),
                colorscale=[[0,'#330000'],[0.5,'#ff4400'],[1,'#ffcc00']],
                cmin=300, cmax=500,
                line=dict(width=0),
            ),
        ))

    geo = dict(**_GEO_BASE)
    if view_mode == 'globe':
        geo['projection_type'] = 'orthographic'
    else:
        geo['projection_type'] = 'natural earth'

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"LIVE TACTICAL GLOBE  ·  {datetime.utcnow().strftime('%H:%M:%S UTC')}",
            font=dict(size=10, color='#00bb55'), x=0.01,
        ),
        geo=geo,
        **_LAYOUT_BASE,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# APP & LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title=f"GSF v{VERSION}",
    update_title=None,
)

CSS = """
body { background:#040c06 !important; font-family:'Courier New',monospace; }
.card { background:#07100a !important; border:1px solid #0b2e18 !important; }
.card-header {
    background:#091508 !important; color:#00cc66 !important;
    font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase;
    border-bottom:1px solid #0b2e18 !important; padding:5px 12px !important;
}
.stat-box {
    background:#060e08; border:1px solid #0b2e18; border-radius:3px;
    padding:5px 8px; text-align:center; margin-bottom:6px;
}
.stat-value { color:#00ff88; font-size:1.3rem; font-weight:bold; line-height:1.2; }
.stat-label { color:#2a5535; font-size:0.6rem; letter-spacing:0.12em; }
.scanline {
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
        rgba(0,255,100,0.012) 2px, rgba(0,255,100,0.012) 4px);
    pointer-events:none; position:fixed; top:0; left:0; width:100%; height:100%; z-index:9999;
}
.log-row { border-bottom:1px solid #091408; padding:2px 0; line-height:1.4; }
.layer-check .form-check-label { color:#337744 !important; font-size:0.72rem; }
.layer-check .form-check-input:checked { background-color:#00cc55; border-color:#00cc55; }
.inspect-panel { border:1px solid #0b3318; border-radius:4px; background:#050e07;
                 padding:10px; font-size:0.75rem; color:#88ccaa; min-height:80px; }
.inspect-title { color:#00ff88; font-size:0.8rem; font-weight:bold; margin-bottom:6px; }
.inspect-kv { color:#336644; }
.inspect-kv span { color:#aaffcc; }
.blink { animation:blink 1.1s step-end infinite; }
@keyframes blink { 50%{opacity:0} }
"""

# ── Layer checkbox options
LAYER_OPTIONS = [
    {'label': ' ✈  Aircraft (ADS-B)',   'value': 'flights'},
    {'label': ' ⚡ Earthquakes',         'value': 'quakes'},
    {'label': ' 🔥 Wildfires (FIRMS)',   'value': 'fires'},
]

app.layout = html.Div([
    html.Div(className='scanline'),
    html.Style(CSS),

    dbc.Container(fluid=True, children=[

        # ── Header ────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("◈ ", style={'color':'#00ff88'}),
                    html.Span("GEO-SURVEILLANCE FEED", style={
                        'color':'#00cc66','fontSize':'1.0rem',
                        'letterSpacing':'0.15em','fontWeight':'bold',
                    }),
                    html.Span(f"  v{VERSION}  ·  PUBLIC OSINT EDITION", style={
                        'color':'#2a5535','fontSize':'0.68rem','marginLeft':'8px',
                    }),
                ], style={'padding':'9px 0 5px'}),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("ONLINE ", style={'color':'#00ff88','fontSize':'0.68rem'}),
                    html.Span("■ ", className='blink', style={'color':'#00ff44','fontSize':'0.68rem'}),
                    html.Br(),
                    html.Span(id='clock', style={'color':'#1e4428','fontSize':'0.62rem'}),
                ], style={'textAlign':'right','padding':'8px 0'}),
            ], width=4),
        ]),

        # ── Main grid ─────────────────────────────────────────
        dbc.Row([

            # LEFT — Globe
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                dbc.Tabs([
                                    dbc.Tab(label="⬡ FLAT MAP",   tab_id='tab-flat',
                                            label_style={'color':'#005522','fontSize':'0.7rem'},
                                            active_label_style={'color':'#00ff88'}),
                                    dbc.Tab(label="◎ ORTHOGRAPHIC", tab_id='tab-globe',
                                            label_style={'color':'#005522','fontSize':'0.7rem'},
                                            active_label_style={'color':'#00ff88'}),
                                ], id='view-tabs', active_tab='tab-flat',
                                   style={'background':'transparent','borderBottom':'none'}),
                            ], width=6),
                            dbc.Col([
                                # Layer toggles inline in header
                                dbc.Checklist(
                                    id='layer-toggles',
                                    options=LAYER_OPTIONS,
                                    value=['flights','quakes','fires'],
                                    inline=True,
                                    className='layer-check',
                                    style={'marginTop':'4px'},
                                ),
                            ], width=6),
                        ], align='center'),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='globe-fig',
                                  style={'height':'56vh'},
                                  config={
                                      'displaylogo': False,
                                      'modeBarButtonsToRemove': ['select2d','lasso2d'],
                                      'scrollZoom': True,
                                  }),
                        dcc.Interval(id='tick-map',  interval=MAP_REFRESH_MS,  n_intervals=0),
                        dcc.Interval(id='tick-eq',   interval=EQ_REFRESH_MS,   n_intervals=0),
                        dcc.Interval(id='tick-fire', interval=FIRE_REFRESH_MS, n_intervals=0),
                    ], style={'padding':'4px','backgroundColor':'#040c06'}),
                ]),

                # ── Altitude filter slider ─────────────────────
                dbc.Card([
                    dbc.CardHeader("✈  ALTITUDE FILTER — minimum altitude shown"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Slider(
                                    id='alt-slider',
                                    min=0, max=13, step=0.5, value=0,
                                    marks={i: {'label':f"{i}km",'style':{'color':'#336644','fontSize':'0.6rem'}}
                                           for i in range(0,14,2)},
                                    tooltip={'placement':'top','always_visible':True},
                                ),
                            ], width=10),
                            dbc.Col([
                                html.Div(id='alt-label', style={'color':'#00cc66','fontSize':'0.72rem',
                                                                 'marginTop':'6px','textAlign':'center'}),
                            ], width=2),
                        ]),
                    ], style={'padding':'10px 16px 6px','backgroundColor':'#040c06'}),
                ], className='mt-2'),

                # ── Click-to-inspect panel ─────────────────────
                dbc.Card([
                    dbc.CardHeader("🔎 CLICK-TO-INSPECT — select any marker on the globe"),
                    dbc.CardBody([
                        html.Div(id='inspect-panel', className='inspect-panel',
                                 children=[
                                     html.Span("Click any aircraft ✈ or earthquake ⚡ to inspect.",
                                               style={'color':'#224433','fontSize':'0.72rem'})
                                 ]),
                    ], style={'padding':'8px','backgroundColor':'#040c06'}),
                ], className='mt-2'),

            ], width=8),

            # RIGHT — Camera + Stats + Log
            dbc.Col([

                # Stats row (5 boxes now)
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-flights', className='stat-value'),
                        html.Div("AIRCRAFT", className='stat-label'),
                    ], className='stat-box'), width=4),
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-eq', className='stat-value'),
                        html.Div("QUAKES", className='stat-label'),
                    ], className='stat-box'), width=4),
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-fires', className='stat-value'),
                        html.Div("FIRE PTS", className='stat-label'),
                    ], className='stat-box'), width=4),
                ], className='mb-1'),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-alt', className='stat-value'),
                        html.Div("MAX ALT km", className='stat-label'),
                    ], className='stat-box'), width=6),
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-speed', className='stat-value'),
                        html.Div("MAX SPD m/s", className='stat-label'),
                    ], className='stat-box'), width=6),
                ], className='mb-2'),

                # Camera
                dbc.Card([
                    dbc.CardHeader("◈ PUBLIC OSINT CAMERA — LIVE MJPEG"),
                    dbc.CardBody([
                        html.Img(id='cam-img', style={
                            'width':'100%',
                            'border':'1px solid #0b2e18',
                            'borderRadius':'3px',
                        }),
                        dcc.Interval(id='tick-cam', interval=CAM_REFRESH_MS, n_intervals=0),
                    ], style={'padding':'5px','backgroundColor':'#030807'}),
                ], className='mb-2'),

                # Tabbed log
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs([
                            dbc.Tab(label="✈ ADS-B LOG", tab_id='log-flights',
                                    label_style={'color':'#005522','fontSize':'0.7rem'},
                                    active_label_style={'color':'#00ff88'}),
                            dbc.Tab(label="⚡ QUAKE LOG", tab_id='log-quakes',
                                    label_style={'color':'#005522','fontSize':'0.7rem'},
                                    active_label_style={'color':'#ff6633'}),
                        ], id='log-tabs', active_tab='log-flights',
                           style={'background':'transparent','borderBottom':'none'}),
                    ),
                    dbc.CardBody([
                        html.Div(id='log-panel', style={
                            'height':'23vh',
                            'overflowY':'auto',
                            'fontSize':'0.7rem',
                            'fontFamily':'monospace',
                        }),
                    ], style={'padding':'6px','backgroundColor':'#030807'}),
                ]),

            ], width=4),
        ]),

        # Footer
        dbc.Row([dbc.Col(html.Div(
            f"GSF v{VERSION}  ·  ADS-B: OpenSky  ·  Seismic: USGS  ·  "
            f"Fires: NASA FIRMS  ·  Camera: public MJPEG  ·  All feeds public/unauthenticated",
            style={'color':'#183322','fontSize':'0.58rem','padding':'6px 0','textAlign':'center'},
        ))]),
    ]),

    # Data stores
    dcc.Store(id='store-flights'),
    dcc.Store(id='store-eq'),
    dcc.Store(id='store-fires'),
])

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

# Clock
@app.callback(Output('clock','children'), Input('tick-cam','n_intervals'))
def update_clock(n):
    return datetime.utcnow().strftime("UTC  %Y-%m-%d  %H:%M:%S")

# Camera
@app.callback(Output('cam-img','src'), Input('tick-cam','n_intervals'))
def update_cam(n):
    enc = cam.read_encoded()
    return f"data:image/jpeg;base64,{enc}" if enc else dash.no_update

# Altitude slider label
@app.callback(Output('alt-label','children'), Input('alt-slider','value'))
def update_alt_label(v):
    return f"≥ {v} km"

# Flights fetch
@app.callback(
    Output('store-flights','data'),
    Output('stat-flights','children'),
    Output('stat-alt','children'),
    Output('stat-speed','children'),
    Input('tick-map','n_intervals'),
)
def refresh_flights(n):
    df = fetch_flights()
    if df.empty:
        return '{}', '—', '—', '—'
    return (
        df.to_json(),
        str(len(df)),
        f"{df['alt_km'].max():.1f}",
        f"{df['vel'].max():.0f}",
    )

# Earthquakes fetch
@app.callback(
    Output('store-eq','data'),
    Output('stat-eq','children'),
    Input('tick-eq','n_intervals'),
)
def refresh_quakes(n):
    df = fetch_earthquakes()
    if df.empty:
        return '{}', '—'
    return df.to_json(), str(len(df))

# Fires fetch
@app.callback(
    Output('store-fires','data'),
    Output('stat-fires','children'),
    Input('tick-fire','n_intervals'),
)
def refresh_fires(n):
    df, status = fetch_fires()
    if df.empty:
        label = 'NO KEY' if status == 'no_key' else '—'
        return '{}', label
    return df.to_json(), str(len(df))

# Globe render
@app.callback(
    Output('globe-fig','figure'),
    Input('store-flights','data'),
    Input('store-eq','data'),
    Input('store-fires','data'),
    Input('layer-toggles','value'),
    Input('view-tabs','active_tab'),
    Input('alt-slider','value'),
)
def render_globe(fl_json, eq_json, fire_json, layers, tab, alt_min):
    def safe_read(j):
        try:
            return pd.read_json(j) if j and j != '{}' else pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    df_fl   = safe_read(fl_json)
    df_eq   = safe_read(eq_json)
    df_fire = safe_read(fire_json)
    view    = 'globe' if tab == 'tab-globe' else 'flat'
    return build_globe(df_fl, df_eq, df_fire, layers or [], view, alt_min or 0)

# Log panel (tabbed)
@app.callback(
    Output('log-panel','children'),
    Input('store-flights','data'),
    Input('store-eq','data'),
    Input('log-tabs','active_tab'),
)
def update_log(fl_json, eq_json, active_tab):
    if active_tab == 'log-flights':
        try:
            df = pd.read_json(fl_json) if fl_json and fl_json != '{}' else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            return [html.P("Awaiting ADS-B telemetry…", style={'color':'#224433'})]
        top = df.nlargest(80, 'alt')
        return [html.Div([
            html.Span(f"[{r['icao'].upper()}]",  style={'color':'#996600','marginRight':'5px'}),
            html.Span(f"{str(r['callsign'])[:8].ljust(8)}", style={'color':'#00bb44','marginRight':'5px'}),
            html.Span(f"{str(r['country'])[:10].ljust(10)}", style={'color':'#2a5535','marginRight':'5px'}),
            html.Span(f"↑{int(r['alt']):>6,}m", style={'color':'#009977','marginRight':'5px'}),
            html.Span(f"{r['vel']:>5.1f}m/s", style={'color':'#336644'}),
        ], className='log-row') for _, r in top.iterrows()]

    else:  # quake log
        try:
            df = pd.read_json(eq_json) if eq_json and eq_json != '{}' else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            return [html.P("No recent earthquakes…", style={'color':'#224433'})]
        top = df.nlargest(60, 'mag')
        return [html.Div([
            html.Span(f"M{r['mag']:.1f}", style={'color':'#ff6633','marginRight':'6px','fontWeight':'bold'}),
            html.Span(f"{str(r['place'])[:28]}", style={'color':'#cc8844','marginRight':'6px'}),
            html.Span(f"↓{r['depth']}km", style={'color':'#664422','marginRight':'6px'}),
            html.Span(r['time'], style={'color':'#443322'}),
        ], className='log-row') for _, r in top.iterrows()]

# Click-to-inspect
@app.callback(
    Output('inspect-panel','children'),
    Input('globe-fig','clickData'),
)
def inspect_click(click):
    if not click:
        return html.Span("Click any aircraft ✈ or earthquake ⚡ to inspect.",
                         style={'color':'#224433','fontSize':'0.72rem'})
    pt = click['points'][0]
    txt = pt.get('text','')

    def kv(k, v):
        return html.Div([
            html.Span(f"{k}: ", className='inspect-kv'),
            html.Span(str(v), className='inspect-kv', style={'color':'#aaffcc'}),
        ])

    # Detect type by hover text prefix
    if '✈' in txt:
        cd = pt.get('customdata', [])
        children = [html.Div("✈  AIRCRAFT DETAIL", className='inspect-title')]
        if len(cd) >= 6:
            children += [
                kv("ICAO",     cd[0].upper()),
                kv("Callsign", cd[1]),
                kv("Country",  cd[2]),
                kv("Altitude", f"{int(cd[3]):,} m  ({round(cd[3]/1000,2)} km)"),
                kv("Speed",    f"{cd[4]} m/s  ({round(cd[4]*3.6,1)} km/h)"),
                kv("Heading",  f"{int(cd[5])}°"),
            ]
    elif '⚡' in txt:
        cd = pt.get('customdata', [])
        children = [html.Div("⚡  EARTHQUAKE DETAIL", className='inspect-title',
                             style={'color':'#ff6633'})]
        if len(cd) >= 4:
            children += [
                kv("Magnitude", f"M{cd[0]}"),
                kv("Location",  cd[1]),
                kv("Depth",     f"{cd[2]} km"),
                kv("Time",      cd[3]),
            ]
    elif '🔥' in txt:
        children = [
            html.Div("🔥  WILDFIRE HOTSPOT", className='inspect-title', style={'color':'#ff4400'}),
            html.Div(txt.replace('<b>','').replace('</b>','').replace('<br>','  |  '),
                     style={'color':'#cc7733','fontSize':'0.72rem'}),
        ]
    else:
        children = [html.Div(txt[:200], style={'color':'#aaffcc','fontSize':'0.7rem'})]

    children.append(html.Div([
        html.Span(f"  Lat: {pt.get('lat','?'):.4f}  Lon: {pt.get('lon','?'):.4f}",
                  style={'color':'#224433','fontSize':'0.66rem','marginTop':'6px'}),
    ]))
    return children


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fire_status = "ACTIVE" if FIRMS_MAP_KEY else "NO KEY — add FIRMS_MAP_KEY to config"
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  GEO-SURVEILLANCE FEED  v{VERSION}  —  BOOTING              ║
╠═══════════════════════════════════════════════════════════╣
║  ✈  ADS-B aircraft      →  OpenSky Network               ║
║  ⚡  Seismic M2.5+       →  USGS Earthquake Feed          ║
║  🔥  Wildfire hotspots   →  {fire_status:<32s}║
║  📹  OSINT camera        →  Public MJPEG streams          ║
╠═══════════════════════════════════════════════════════════╣
║  NEW in v0.2: layer toggles · click-inspect · alt-filter ║
║               tabbed log · fire layer · 5 stat panels     ║
╠═══════════════════════════════════════════════════════════╣
║  Dashboard:  http://127.0.0.1:8050                        ║
║  Ctrl+C to exit cleanly                                   ║
╚═══════════════════════════════════════════════════════════╝
""")
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n[*] Shutting down…")
        cam.stop()
        sys.exit(0)
