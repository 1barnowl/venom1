"""
╔═══════════════════════════════════════════════════════════╗
║     GEO-SURVEILLANCE FEED  ·  VERSION 0.1 (TEST BUILD)    ║
║     Global Vision Platform - Public OSINT Edition         ║
╚═══════════════════════════════════════════════════════════╝

INSTALL:
    pip install dash dash-bootstrap-components pandas plotly \
                requests opencv-python numpy --break-system-packages

RUN:
    python3 gsf_v01.py

BROWSER:
    http://127.0.0.1:8050
"""

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
import cv2
import base64
import threading
import time
import sys
import json
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
VERSION = "0.1.0"

# Public MJPEG streams — cycle through fallbacks if one is offline
CAMERA_SOURCES = [
    "http://pendelcam.dorfcam.de/mjpg/video.mjpg",
    "http://webcam1.lpl.arizona.edu/mjpeg.cgi",
    "http://195.196.36.242/mjpg/video.mjpg",
]

# Public REST APIs — all unauthenticated free tiers
OPENSKY_URL   = "https://opensky-network.org/api/states/all"
USGS_EQ_URL   = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
NASA_FIRE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/FIRMS_KEY_PLACEHOLDER/VIIRS_SNPP_NRT/world/1/today"
GDELT_URL     = "https://api.gdeltproject.org/api/v2/summary/summary?d=web&t=summary&k=disaster+emergency&ts=default&svt=zoom&stab=0&evds=e&last24=1&outputtype=json"

MAP_REFRESH_MS  = 12000   # ms between flight map refreshes
CAM_REFRESH_MS  = 120     # ms between camera frame grabs
EQ_REFRESH_MS   = 30000   # ms between earthquake refreshes
MAX_TARGETS     = 400     # ADS-B targets to render
MAX_EQ          = 100     # Earthquake markers to render

# ─────────────────────────────────────────────────────────────
# CAMERA STREAM HANDLER (Non-blocking)
# ─────────────────────────────────────────────────────────────
class TacticalStream:
    """Async MJPEG capture with automatic fallback."""
    def __init__(self, sources=CAMERA_SOURCES):
        self.sources = sources
        self.cap_index = 0
        self.stopped  = False
        self.lock     = threading.Lock()
        self.frame    = None
        self.grabbed  = False
        self.stream   = None
        self._connect()

    def _connect(self):
        src = self.sources[self.cap_index % len(self.sources)]
        print(f"[CAM] Connecting to: {src}")
        if self.stream:
            self.stream.release()
        self.stream = cv2.VideoCapture(src)
        if self.stream.isOpened():
            self.grabbed, self.frame = self.stream.read()
            print(f"[CAM] ✓ Stream online")
        else:
            print(f"[CAM] ✗ Offline — will retry next source")
            self.grabbed = False
            self.frame   = None

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        fails = 0
        while not self.stopped:
            if self.stream and self.stream.isOpened():
                grabbed, frame = self.stream.read()
                if grabbed:
                    with self.lock:
                        self.grabbed = grabbed
                        self.frame   = frame
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
            time.sleep(0.033)   # ~30 fps cap

    def read_encoded(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                black = np.zeros((360, 640, 3), dtype=np.uint8)
                ts    = datetime.now().strftime("%H:%M:%S")
                cv2.putText(black, "NO SIGNAL / STREAM OFFLINE", (90, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 60), 2)
                cv2.putText(black, f"RETRYING ...  {ts}", (130, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 80), 1)
                _, buf = cv2.imencode('.jpg', black)
            else:
                frame_r = cv2.resize(self.frame, (640, 360))
                # Timestamp overlay
                ts = datetime.now(timezone.utc).strftime("UTC %Y-%m-%d %H:%M:%S")
                cv2.putText(frame_r, ts, (8, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
                cv2.putText(frame_r, "LIVE", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                _, buf = cv2.imencode('.jpg', frame_r, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode('utf-8')

    def stop(self):
        self.stopped = True
        if self.stream:
            self.stream.release()

# Boot camera thread
cam = TacticalStream().start()

# ─────────────────────────────────────────────────────────────
# DATA INGESTION SERVICES
# ─────────────────────────────────────────────────────────────

def fetch_flights():
    """Pull live ADS-B transponder states from OpenSky."""
    try:
        r = requests.get(OPENSKY_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        if 'states' not in data or not data['states']:
            return pd.DataFrame()

        df = pd.DataFrame(data['states']).iloc[:MAX_TARGETS, [0,1,2,5,6,7,9,10]]
        df.columns = ['icao', 'callsign', 'country', 'lon', 'lat', 'alt', 'vel', 'hdg']
        df.dropna(subset=['lon', 'lat'], inplace=True)
        df['alt']      = df['alt'].fillna(0).round(0)
        df['vel']      = df['vel'].fillna(0).round(1)
        df['callsign'] = df['callsign'].str.strip().replace('', 'UNKNOWN')
        df['alt_km']   = (df['alt'] / 1000).round(2)
        return df
    except Exception as e:
        print(f"[ADS-B] Fetch failed: {e}")
        return pd.DataFrame()


def fetch_earthquakes():
    """Pull recent M2.5+ earthquakes from USGS."""
    try:
        r = requests.get(USGS_EQ_URL, timeout=10)
        r.raise_for_status()
        features = r.json().get('features', [])[:MAX_EQ]
        rows = []
        for f in features:
            props = f['properties']
            coords = f['geometry']['coordinates']
            rows.append({
                'lon':   coords[0],
                'lat':   coords[1],
                'depth': round(coords[2], 1),
                'mag':   props.get('mag', 0),
                'place': props.get('place', 'Unknown'),
                'time':  datetime.fromtimestamp(
                             props['time'] / 1000, tz=timezone.utc
                         ).strftime('%H:%M UTC'),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[EQ] Fetch failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────
_dark_geo = dict(
    showframe      = False,
    bgcolor        = 'rgba(0,0,0,0)',
    showcoastlines = True,  coastlinecolor = '#1a6644',
    showland       = True,  landcolor      = 'rgb(15,20,18)',
    showocean      = True,  oceancolor     = 'rgb(6,12,22)',
    showlakes      = True,  lakecolor      = 'rgb(8,14,28)',
    showcountries  = True,  countrycolor   = 'rgb(35,55,45)',
    showrivers     = True,  rivercolor     = 'rgb(8,35,55)',
    projection_type= 'natural earth',
)
_dark_layout = dict(
    paper_bgcolor  = 'rgba(0,0,0,0)',
    plot_bgcolor   = 'rgba(0,0,0,0)',
    font           = dict(color='#a0ffcc', family='monospace', size=11),
    margin         = dict(l=0, r=0, t=28, b=0),
    showlegend     = True,
    legend         = dict(
        bgcolor     = 'rgba(0,20,10,0.8)',
        bordercolor = '#1a6644',
        borderwidth = 1,
        font        = dict(size=10, color='#a0ffcc'),
        x=0.01, y=0.99,
    ),
)


def build_globe_flight(df_flights, df_eq, view_mode):
    """Compose the main tactical globe figure."""
    traces = []

    # ── Flights layer ─────────────────────────────────────────
    if not df_flights.empty:
        hover = (
            "<b>" + df_flights['callsign'] + "</b><br>"
            + "Country: " + df_flights['country'] + "<br>"
            + "Alt: " + df_flights['alt'].astype(str) + " m (" + df_flights['alt_km'].astype(str) + " km)<br>"
            + "Speed: " + df_flights['vel'].astype(str) + " m/s<br>"
            + "ICAO: " + df_flights['icao']
        )
        traces.append(go.Scattergeo(
            lon       = df_flights['lon'],
            lat       = df_flights['lat'],
            text      = hover,
            hoverinfo = 'text',
            name      = f"✈ Aircraft ({len(df_flights)})",
            mode      = 'markers',
            marker    = dict(
                size          = 5,
                opacity       = 0.85,
                color         = df_flights['alt'],
                colorscale    = [[0, '#003322'], [0.4, '#00aa55'], [1, '#00ffcc']],
                cmin          = 0,
                cmax          = 13000,
                colorbar      = dict(
                    title      = dict(text="Alt (m)", font=dict(color='#a0ffcc', size=10)),
                    thickness  = 8, len=0.4, x=1.01,
                    tickfont   = dict(color='#a0ffcc', size=8),
                    bgcolor    = 'rgba(0,0,0,0)',
                    bordercolor= '#1a6644',
                ),
                symbol        = 'circle',
                line          = dict(width=0),
            )
        ))

    # ── Earthquake layer ──────────────────────────────────────
    if not df_eq.empty:
        eq_hover = (
            "<b>M" + df_eq['mag'].astype(str) + " Earthquake</b><br>"
            + df_eq['place'] + "<br>"
            + "Depth: " + df_eq['depth'].astype(str) + " km<br>"
            + df_eq['time']
        )
        eq_size  = (df_eq['mag'].clip(2, 8) * 3).tolist()
        eq_color = df_eq['mag'].tolist()
        traces.append(go.Scattergeo(
            lon       = df_eq['lon'],
            lat       = df_eq['lat'],
            text      = eq_hover,
            hoverinfo = 'text',
            name      = f"⚡ Earthquakes M2.5+ ({len(df_eq)})",
            mode      = 'markers',
            marker    = dict(
                size      = eq_size,
                opacity   = 0.75,
                color     = eq_color,
                colorscale= [[0,'#331100'], [0.5,'#ff6600'], [1,'#ff0000']],
                cmin      = 2.5,
                cmax      = 8,
                symbol    = 'diamond',
                line      = dict(width=1, color='#ff4400'),
            )
        ))

    fig = go.Figure(data=traces)
    geo_cfg = dict(**_dark_geo)
    if view_mode == 'globe':
        geo_cfg['projection_type'] = 'orthographic'
    fig.update_layout(
        title = dict(
            text=f"LIVE TACTICAL GLOBE  ·  {datetime.utcnow().strftime('%H:%M:%S UTC')}",
            font=dict(size=11, color='#00cc66'),
            x=0.01,
        ),
        geo = geo_cfg,
        **_dark_layout,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# DASH APP LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title=f"GSF v{VERSION}",
    update_title=None,
)

# Custom dark-green military terminal CSS
CUSTOM_CSS = """
body {
    background: #050d08 !important;
    font-family: 'Courier New', monospace;
}
.card {
    background: #080f0a !important;
    border: 1px solid #0d3320 !important;
}
.card-header {
    background: #0a1a0e !important;
    color: #00cc66 !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #0d3320 !important;
    padding: 6px 12px !important;
}
.stat-box {
    background: #070e09;
    border: 1px solid #0d3320;
    border-radius: 4px;
    padding: 6px 10px;
    text-align: center;
    margin-bottom: 6px;
}
.stat-value { color: #00ff88; font-size: 1.4rem; font-weight: bold; }
.stat-label { color: #336644; font-size: 0.65rem; letter-spacing: 0.1em; }
.scanline {
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,100,0.015) 2px, rgba(0,255,100,0.015) 4px
    );
    pointer-events: none;
    position: fixed; top:0; left:0; width:100%; height:100%;
    z-index: 9999;
}
.log-entry { border-bottom: 1px solid #0a1a0e; padding: 3px 0; }
.blink { animation: blink 1.2s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
"""

app.layout = html.Div([
    # scanline overlay
    html.Div(className='scanline'),

    # Injected CSS
    html.Style(CUSTOM_CSS),

    # ── Top header bar ────────────────────────────────────────
    dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("◈ ", style={'color':'#00ff88'}),
                    html.Span(f"GEO-SURVEILLANCE FEED", style={
                        'color':'#00cc66','fontSize':'1.05rem',
                        'letterSpacing':'0.15em','fontWeight':'bold',
                    }),
                    html.Span(f"  v{VERSION}  ·  PUBLIC OSINT EDITION", style={
                        'color':'#336644','fontSize':'0.7rem','marginLeft':'8px',
                    }),
                ], style={'padding':'10px 0 6px'}),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("STATUS: ", style={'color':'#336644','fontSize':'0.7rem'}),
                    html.Span("ONLINE ", style={'color':'#00ff88','fontSize':'0.7rem'}),
                    html.Span("■", className='blink', style={'color':'#00ff44','fontSize':'0.7rem'}),
                    html.Br(),
                    html.Span(id='clock', style={'color':'#225533','fontSize':'0.65rem'}),
                ], style={'textAlign':'right','padding':'8px 0'}),
            ], width=4),
        ]),

        # ── Main content grid ─────────────────────────────────
        dbc.Row([

            # LEFT: Globe + tabs
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Tabs([
                            dbc.Tab(label="⬡ GLOBE / FLAT", tab_id='tab-flat',
                                    label_style={'color':'#006633','fontSize':'0.72rem'},
                                    active_label_style={'color':'#00ff88'}),
                            dbc.Tab(label="◎ ORTHOGRAPHIC", tab_id='tab-globe',
                                    label_style={'color':'#006633','fontSize':'0.72rem'},
                                    active_label_style={'color':'#00ff88'}),
                        ], id='view-tabs', active_tab='tab-flat',
                           style={'background':'transparent','borderBottom':'none'}),
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id='globe-fig',
                                  style={'height':'62vh'},
                                  config={'displaylogo':False,
                                          'modeBarButtonsToRemove':['select2d','lasso2d']}),
                        dcc.Interval(id='tick-map', interval=MAP_REFRESH_MS, n_intervals=0),
                    ], style={'padding':'4px','backgroundColor':'#050d08'}),
                ]),
            ], width=8),

            # RIGHT: Camera + stats + log
            dbc.Col([

                # Stat boxes row
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-flights', className='stat-value'),
                        html.Div("AIRCRAFT LIVE", className='stat-label'),
                    ], className='stat-box'), width=4),
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-eq', className='stat-value'),
                        html.Div("EQ EVENTS", className='stat-label'),
                    ], className='stat-box'), width=4),
                    dbc.Col(html.Div([
                        html.Div("—", id='stat-alt', className='stat-value'),
                        html.Div("MAX ALT (km)", className='stat-label'),
                    ], className='stat-box'), width=4),
                ], className='mb-2'),

                # Camera feed
                dbc.Card([
                    dbc.CardHeader("◈ PUBLIC OSINT CAMERA NODE — LIVE"),
                    dbc.CardBody([
                        html.Img(id='cam-img', style={
                            'width':'100%',
                            'border':'1px solid #0d3320',
                            'borderRadius':'3px',
                            'imageRendering':'crisp-edges',
                        }),
                        dcc.Interval(id='tick-cam', interval=CAM_REFRESH_MS, n_intervals=0),
                    ], style={'padding':'6px','backgroundColor':'#030806'}),
                ], className='mb-2'),

                # Signal intel log
                dbc.Card([
                    dbc.CardHeader("◈ SIGNAL INTELLIGENCE — ADS-B TARGETS"),
                    dbc.CardBody([
                        html.Div(id='sig-log', style={
                            'height':'22vh',
                            'overflowY':'auto',
                            'fontSize':'0.72rem',
                            'fontFamily':'monospace',
                        }),
                        dcc.Interval(id='tick-eq', interval=EQ_REFRESH_MS, n_intervals=0),
                    ], style={'padding':'6px','backgroundColor':'#030806'}),
                ]),

            ], width=4),
        ]),

        # Footer
        dbc.Row([
            dbc.Col(html.Div(
                f"GSF v{VERSION}  ·  Data: OpenSky (ADS-B) · USGS (Seismic) · Public Cameras  ·  "
                f"All sources public domain / unauthenticated",
                style={'color':'#1a4425','fontSize':'0.6rem','padding':'6px 0','textAlign':'center'}
            )),
        ]),
    ]),

    # Hidden stores
    dcc.Store(id='store-flights'),
    dcc.Store(id='store-eq'),
])

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(Output('clock', 'children'), Input('tick-cam', 'n_intervals'))
def update_clock(n):
    return datetime.utcnow().strftime("UTC  %Y-%m-%d  %H:%M:%S")


@app.callback(Output('cam-img', 'src'), Input('tick-cam', 'n_intervals'))
def update_camera(n):
    enc = cam.read_encoded()
    return f"data:image/jpeg;base64,{enc}" if enc else dash.no_update


@app.callback(
    Output('store-flights', 'data'),
    Output('stat-flights',  'children'),
    Output('stat-alt',      'children'),
    Output('sig-log',       'children'),
    Input('tick-map', 'n_intervals')
)
def refresh_flights(n):
    df = fetch_flights()
    if df.empty:
        return '{}', '—', '—', [html.P("Awaiting ADS-B telemetry…", style={'color':'#336644'})]

    max_alt = f"{df['alt_km'].max():.1f}"

    # Build signal log entries
    top = df.nlargest(60, 'alt')
    entries = []
    for _, row in top.iterrows():
        entries.append(html.Div([
            html.Span(f"[{row['icao'].upper()}]", style={'color':'#cc7700','marginRight':'6px'}),
            html.Span(f"{str(row['callsign']).ljust(8)}", style={'color':'#00cc55','marginRight':'6px'}),
            html.Span(f"{row['country'][:12].ljust(12)}", style={'color':'#336644','marginRight':'6px'}),
            html.Span(f"↑{int(row['alt']):,}m", style={'color':'#00aa88','marginRight':'6px'}),
            html.Span(f"{row['vel']}m/s", style={'color':'#447755'}),
        ], className='log-entry'))

    return df.to_json(), str(len(df)), max_alt, entries


@app.callback(
    Output('store-eq', 'data'),
    Output('stat-eq', 'children'),
    Input('tick-eq', 'n_intervals')
)
def refresh_earthquakes(n):
    df = fetch_earthquakes()
    if df.empty:
        return '{}', '—'
    return df.to_json(), str(len(df))


@app.callback(
    Output('globe-fig', 'figure'),
    Input('store-flights', 'data'),
    Input('store-eq',      'data'),
    Input('view-tabs',     'active_tab'),
)
def render_globe(flights_json, eq_json, tab):
    try:
        df_flights = pd.read_json(flights_json) if flights_json and flights_json != '{}' else pd.DataFrame()
    except Exception:
        df_flights = pd.DataFrame()
    try:
        df_eq = pd.read_json(eq_json) if eq_json and eq_json != '{}' else pd.DataFrame()
    except Exception:
        df_eq = pd.DataFrame()

    view_mode = 'globe' if tab == 'tab-globe' else 'flat'
    return build_globe_flight(df_flights, df_eq, view_mode)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════════════╗
║  GEO-SURVEILLANCE FEED  v0.1  —  BOOTING                 ║
╠═══════════════════════════════════════════════════════════╣
║  ✈  ADS-B live aircraft   →  OpenSky Network             ║
║  ⚡  Seismic events M2.5+  →  USGS Earthquake Feed        ║
║  📹  Public camera OSINT   →  Open MJPEG Streams          ║
╠═══════════════════════════════════════════════════════════╣
║  Dashboard:  http://127.0.0.1:8050                        ║
║  Press Ctrl+C to shut down cleanly                        ║
╚═══════════════════════════════════════════════════════════╝
""")
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n[*] Shutdown signal received. Releasing streams…")
        cam.stop()
        print("[*] Goodbye.")
        sys.exit(0)
