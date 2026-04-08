"""
GEO-SURVEILLANCE TACTICAL ENGINE - PUBLIC FEED EDITION
Dependencies: pip install dash dash-bootstrap-components pandas plotly requests opencv-python --break-system-packages
run, then http://127.0.0.1:8050 in browser
"""

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
import cv2
import base64
import threading
import time
import sys
import numpy as np

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
# Defaulting to a public unauthenticated IP camera stream. 
# Swap this string with any RTSP/HTTP stream URL you scrape from Shodan/Insecam.
CAMERA_SOURCE = "http://pendelcam.dorfcam.de/mjpg/video.mjpg" 
OPENSKY_API_URL = "https://opensky-network.org/api/states/all"
MAP_REFRESH_RATE = 10000  # ms
CAM_REFRESH_RATE = 100    # ms

# ==========================================
# THREADED HARDWARE/STREAM SENSORS
# ==========================================
class TacticalStream:
    """Handles asynchronous video capture to prevent UI blocking."""
    def __init__(self, src=CAMERA_SOURCE):
        print(f"[*] Connecting to public stream: {src}")
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.lock = threading.Lock()
        
        # Crash prevention: If the URL is dead, it won't kill the script
        if not self.stream.isOpened():
            print(f"[!] Warning: Cannot connect to {src}. Stream may be offline.")
            self.grabbed = False
            self.frame = None
        else:
            self.grabbed, self.frame = self.stream.read()

    def start(self):
        threading.Thread(target=self._update, args=(), daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            if self.stream.isOpened():
                grabbed, frame = self.stream.read()
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.03) # Cap at ~30fps to save CPU

    def read_encoded(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                # Return a black frame with a "NO SIGNAL" overlay instead of crashing
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "STREAM OFFLINE / NO SIGNAL", (80, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', black_frame)
            else:
                frame_resized = cv2.resize(self.frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame_resized)
            return base64.b64encode(buffer).decode('utf-8')

    def stop(self):
        self.stopped = True
        self.stream.release()

# Initialize stream thread
video_feed = TacticalStream().start()

# ==========================================
# OSINT DATA INGESTION
# ==========================================
def fetch_telemetry():
    """Fetches public ADSB transponder data from OpenSky."""
    try:
        resp = requests.get(OPENSKY_API_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        
        if 'states' not in data or not data['states']:
            return pd.DataFrame()

        raw_states = data['states']
        
        # Limit to 300 targets to maintain 3D rendering performance
        df = pd.DataFrame(raw_states).iloc[:300, [0, 1, 2, 5, 6, 7, 9, 10]]
        df.columns = ['ICAO24', 'Callsign', 'Country', 'Longitude', 'Latitude', 'Altitude', 'Velocity', 'Heading']
        
        # Clean data
        df.dropna(subset=['Longitude', 'Latitude', 'Altitude'], inplace=True)
        df['Callsign'] = df['Callsign'].str.strip()
        df['Callsign'] = df['Callsign'].replace('', 'UNKNOWN')
        
        return df
    except Exception as e:
        print(f"[-] Telemetry Ingestion Failed: {e}")
        return pd.DataFrame()

# ==========================================
# UI ARCHITECTURE
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "OSINT Geo-Surveillance"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Global OSINT Tracker & Public Surveillance", className="text-center text-success mt-3 mb-4"), width=12)
    ]),
    
    dbc.Row([
        # Left Column: 3D Geospatial View
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Tactical 3D Viewport - Live ADS-B Targets", className="font-weight-bold"),
                dbc.CardBody([
                    dcc.Graph(id='3d-globe', style={'height': '65vh'}),
                    dcc.Interval(id='map-tick', interval=MAP_REFRESH_RATE, n_intervals=0)
                ], className="p-1")
            ], color="secondary", outline=True)
        ], width=8),
        
        # Right Column: Camera & Raw Data
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Public Video Feed (OSINT Node)", className="font-weight-bold"),
                dbc.CardBody([
                    html.Img(id='cam-viewport', style={'width': '100%', 'border': '1px solid #28a745', 'borderRadius': '4px'}),
                    dcc.Interval(id='cam-tick', interval=CAM_REFRESH_RATE, n_intervals=0)
                ], className="p-2 bg-dark text-center")
            ], color="secondary", outline=True, className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader("Signal Intelligence (Active Targets)", className="font-weight-bold"),
                dbc.CardBody([
                    html.Div(id='telemetry-log', style={'height': '30vh', 'overflowY': 'auto', 'fontSize': '0.85rem'})
                ], className="p-2 bg-dark text-light")
            ], color="secondary", outline=True)
        ], width=4)
    ])
], fluid=True, style={'backgroundColor': '#121212', 'minHeight': '100vh'})

# ==========================================
# CALLBACKS & ENGINE LOGIC
# ==========================================

@app.callback(
    Output('cam-viewport', 'src'),
    Input('cam-tick', 'n_intervals')
)
def update_video_stream(n):
    encoded_image = video_feed.read_encoded()
    if encoded_image:
        return f"data:image/jpeg;base64,{encoded_image}"
    return dash.no_update

@app.callback(
    [Output('3d-globe', 'figure'),
     Output('telemetry-log', 'children')],
    Input('map-tick', 'n_intervals')
)
def update_geospatial_engine(n):
    df = fetch_telemetry()
    
    if not df.empty:
        hover_text = df['Callsign'] + "<br>Origin: " + df['Country'] + "<br>Alt: " + df['Altitude'].astype(str) + "m"
        
        fig = go.Figure(data=go.Scattergeo(
            lon = df['Longitude'],
            lat = df['Latitude'],
            text = hover_text,
            mode = 'markers',
            marker = dict(
                size = 5,
                opacity = 0.8,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(width=1, color='rgba(102, 102, 102)'),
                colorscale = 'Blues',
                cmin = 0,
                color = df['Altitude'],
                cmax = df['Altitude'].max(),
                colorbar_title = "Alt (m)"
            )
        ))
    else:
        fig = go.Figure(data=go.Scattergeo())

    fig.update_layout(
        title = 'Live Target Telemetry',
        geo = dict(
            projection_type = 'orthographic',
            showcoastlines = True,
            coastlinecolor = "RebeccaPurple",
            showland = True,
            landcolor = "rgb(25, 25, 25)",
            showocean = True,
            oceancolor = "rgb(10, 10, 20)",
            showlakes = True,
            lakecolor = "rgb(10, 10, 20)",
            showcountries = True,
            countrycolor = "rgb(50, 50, 50)",
            bgcolor = 'rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    if not df.empty:
        log_entries = []
        for index, row in df.iterrows():
            entry = html.Div([
                html.Span(f"[{row['ICAO24']}] ", className="text-warning"),
                html.Span(f"{row['Callsign']} | ", className="text-success font-weight-bold"),
                html.Span(f"Alt: {row['Altitude']}m | Spd: {row['Velocity']}m/s")
            ], className="mb-1 border-bottom border-secondary pb-1")
            log_entries.append(entry)
    else:
        log_entries = [html.P("Awaiting telemetry data or API rate limit reached...", className="text-danger")]

    return fig, log_entries

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == '__main__':
    try:
        print("[*] Booting Geo-Surveillance Interface...")
        print("[*] Connect via browser at http://127.0.0.1:8050")
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n[*] Shutting down tactical stream...")
        video_feed.stop()
        sys.exit(0)
