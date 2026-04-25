"""
╔═══════════════════════════════════════════════════════════╗
║   GEO-SURVEILLANCE FEED  ·  VERSION 1.1  — INTERACTIVE   ║
║   Global Vision Platform — Public OSINT Edition           ║
╠═══════════════════════════════════════════════════════════╣
║  ARCHITECTURE CHANGE:                                     ║
║   🗺  3D globe (CesiumJS) REMOVED — replaced by          ║
║       Scattermapbox tile map (CartoDB Dark, no token)     ║
║       Zoom from global → street level, fully interactive  ║
║                                                           ║
║  NEW in v1.0:                                             ║
║   🗺  Tile map primary view — zoomable, pannable, fast   ║
║       Flight trails drawn as polylines on real tiles      ║
║       Geofence zones rendered as polygon outlines         ║
║   ⚙️  gsf.ini config — all keys/settings external        ║
║       Run --init-config to generate the file             ║
║   🤖 Anomaly detection — rolling z-score baseline,       ║
║       fires alerts on statistical spikes per layer        ║
║   🔲 Geofencing engine — named bounding boxes,           ║
║       ingress/egress alerts, drawn on tile map            ║
║   🔗 Convergence detector — 3+ co-located hazard types  ║
║       in same cell triggers multi-layer alert             ║
║   🛸 ISS pass predictor — /api/v1/iss/passes?lat=&lon=  ║
║   🚀 Production WSGI — gunicorn-ready, CORS, /healthz   ║
║   📋 Versioned API — /api/v1/ with full JSON index       ║
╠═══════════════════════════════════════════════════════════╣
║  INSTALL (same as v0.9 — no new packages):               ║
║    pip install dash dash-bootstrap-components pandas      ║
║               plotly requests opencv-python numpy         ║
║               websocket-client sgp4                       ║
║               --break-system-packages                     ║
║                                                           ║
║  CONFIG:  python3 gsf_v10.py --init-config               ║
║  RUN:     python3 gsf_v10.py                              ║
║  PROD:    gunicorn -w 1 -b 0.0.0.0:8050 gsf_v10:application
║  URL:     http://127.0.0.1:8050                           ║
║  WebGL:   http://127.0.0.1:8050/deck                      ║
║  API:     http://127.0.0.1:8050/api/v1/                  ║
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
import os
import json as json_lib
import math
import sqlite3
import zlib
import configparser
import numpy as np
from collections import deque
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from flask import Response, jsonify, stream_with_context, request as flask_request
import queue as _queue

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False

try:
    from sgp4.api import Satrec, jday as sgp4_jday
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False

# ─────────────────────────────────────────────────────────────
# CONFIG FILE
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = """\
[gsf]
host  = 0.0.0.0
port  = 8050
debug = false

[keys]
# Free API keys — no source editing required
ais_api_key      =
firms_map_key    =
openaq_api_key   =

[cameras]
source_0 = http://pendelcam.dorfcam.de/mjpg/video.mjpg
source_1 = http://webcam1.lpl.arizona.edu/mjpeg.cgi
source_2 = http://195.196.36.242/mjpg/video.mjpg
source_3 =

[limits]
max_flights   = 450
max_eq        = 120
max_fires     = 500
max_ships     = 300
max_aq        = 300
max_events    = 200
max_starlink  = 150
ring_capacity = 240
trail_max_pos = 12

[intervals]
flight_s   = 15
quake_s    = 45
eq30d_s    = 600
fire_s     = 120
radar_s    = 300
iss_s      = 30
aq_s       = 300
gdelt_s    = 300
starlink_s = 60
threat_s   = 60
record_s   = 30

[alerts]
eq_mag_threshold      = 5.0
aq_pm25_threshold     = 150.4
gdelt_count_threshold = 25
anomaly_zscore        = 2.5
convergence_layers    = 3

[threat]
weight_seismic  = 0.35
weight_airqual  = 0.25
weight_conflict = 0.25
weight_fire     = 0.15
grid_deg        = 5

[events]
# Ticketmaster Discovery API — free, 5000 calls/day
# Get your key in 60 sec: https://developer.ticketmaster.com/
ticketmaster_api_key =
events_radius_km     = 25
events_max_results   = 30
"""

_CONFIG_FILE = 'gsf.ini'
_cfg = configparser.ConfigParser()

if '--init-config' in sys.argv:
    with open(_CONFIG_FILE, 'w') as f: f.write(DEFAULT_CONFIG)
    print(f"[CFG] Written {_CONFIG_FILE} — edit [keys] section then re-run.")
    sys.exit(0)

if os.path.exists(_CONFIG_FILE):
    _cfg.read(_CONFIG_FILE); print(f"[CFG] Loaded {_CONFIG_FILE}")
else:
    _cfg.read_string(DEFAULT_CONFIG); print(f"[CFG] Using defaults (run --init-config to create gsf.ini)")

def _cg(s, k, fb=''): 
    try: return _cfg.get(s, k).strip()
    except: return fb
def _ci(s, k, fb=0):
    try: return int(_cfg.get(s, k))
    except: return fb
def _cf(s, k, fb=0.0):
    try: return float(_cfg.get(s, k))
    except: return fb

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
VERSION = "1.2.0"

AIS_API_KEY    = os.environ.get('AIS_API_KEY',    _cg('keys','ais_api_key'))
FIRMS_MAP_KEY  = os.environ.get('FIRMS_MAP_KEY',  _cg('keys','firms_map_key'))
OPENAQ_API_KEY       = os.environ.get('OPENAQ_API_KEY',       _cg('keys','openaq_api_key'))
TICKETMASTER_API_KEY = os.environ.get('TICKETMASTER_API_KEY', _cg('events','ticketmaster_api_key'))
EVENTS_RADIUS_KM     = _ci('events','events_radius_km',   25)
EVENTS_MAX_RESULTS   = _ci('events','events_max_results',  30)
HOST = _cg('gsf','host','0.0.0.0')
PORT = _ci('gsf','port',8050)

CAMERA_SOURCES = [_cg('cameras',f'source_{i}','') for i in range(4)]

OPENSKY_URL      = "https://opensky-network.org/api/states/all"
USGS_EQ_URL      = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
USGS_EQ_30D_URL  = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.geojson"
FIRMS_URL_TPL    = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/world/1/today"
AIS_WS_URL       = "wss://stream.aisstream.io/v0/stream"
OPEN_METEO_URL   = "https://api.open-meteo.com/v1/forecast"
RAINVIEWER_META  = "https://api.rainviewer.com/public/weather-maps.json"
ISS_URL          = "https://api.wheretheiss.at/v1/satellites/25544"
ISS_TLE_URL      = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle"
OPENAQ_URL       = "https://api.openaq.org/v2/locations"
GDELT_GEO_URL    = ("https://api.gdeltproject.org/api/v2/geo/geo"
                    "?query=conflict+OR+disaster+OR+flood+OR+explosion+OR+emergency"
                    "&mode=pointdata&maxrecords=200&timespan=4h&format=json")
STARLINK_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"

FLIGHT_INTERVAL   = _ci('intervals','flight_s',  15)
QUAKE_INTERVAL    = _ci('intervals','quake_s',   45)
EQ30D_INTERVAL    = _ci('intervals','eq30d_s',   600)
FIRE_INTERVAL     = _ci('intervals','fire_s',    120)
RADAR_INTERVAL    = _ci('intervals','radar_s',   300)
ISS_INTERVAL      = _ci('intervals','iss_s',     30)
AQ_INTERVAL       = _ci('intervals','aq_s',      300)
GDELT_INTERVAL    = _ci('intervals','gdelt_s',   300)
STARLINK_INTERVAL = _ci('intervals','starlink_s',60)
THREAT_INTERVAL   = _ci('intervals','threat_s',  60)
RECORD_INTERVAL   = _ci('intervals','record_s',  30)

MAX_FLIGHTS   = _ci('limits','max_flights',  450)
MAX_EQ        = _ci('limits','max_eq',       120)
MAX_EQ30D     = 800
MAX_FIRES     = _ci('limits','max_fires',    500)
MAX_SHIPS     = _ci('limits','max_ships',    300)
MAX_AQ        = _ci('limits','max_aq',       300)
MAX_EVENTS    = _ci('limits','max_events',   200)
MAX_STARLINK  = _ci('limits','max_starlink', 150)
RING_CAPACITY = _ci('limits','ring_capacity',240)
TRAIL_MAX_POS = _ci('limits','trail_max_pos', 12)
ISS_TRAIL_MAX = 20
SHIP_TTL      = 300

ALERT_EQ_MAG      = _cf('alerts','eq_mag_threshold',      5.0)
ALERT_AQ_LEVEL    = _cf('alerts','aq_pm25_threshold',     150.4)
ALERT_GDELT_COUNT = _ci('alerts','gdelt_count_threshold', 25)
ANOMALY_ZSCORE    = _cf('alerts','anomaly_zscore',        2.5)
CONVERGENCE_LAYERS= _ci('alerts','convergence_layers',    3)

THREAT_W = {
    'seismic':  _cf('threat','weight_seismic',  0.35),
    'airqual':  _cf('threat','weight_airqual',  0.25),
    'conflict': _cf('threat','weight_conflict', 0.25),
    'fire':     _cf('threat','weight_fire',     0.15),
}
THREAT_GRID_DEG = _ci('threat','grid_deg', 5)

CAM_REFRESH_MS    = 120
UI_REFRESH_MS     = 15000
ALERT_REFRESH_MS  = 3000
PB_REFRESH_MS     = 1000

AQI_BREAKS = [
    (0,    12,   'Good',           '#00e400'),
    (12,   35.4, 'Moderate',       '#ffff00'),
    (35.4, 55.4, 'Unhealthy*',     '#ff7e00'),
    (55.4, 150.4,'Unhealthy',      '#ff0000'),
    (150.4,250.4,'Very Unhealthy', '#8f3f97'),
    (250.4,9999, 'Hazardous',      '#7e0023'),
]

def aqi_color(pm25):
    for lo, hi, label, color in AQI_BREAKS:
        if pm25 <= hi: return color, label
    return '#7e0023', 'Hazardous'

# ─────────────────────────────────────────────────────────────
# SGP4 HELPERS
# ─────────────────────────────────────────────────────────────
def teme_to_lla(r, jd_total):
    x, y, z = r
    T   = (jd_total - 2451545.0) / 36525.0
    gmst= (280.46061837 + 360.98564736629*(jd_total-2451545.0) + 0.000387933*T*T) % 360.0
    g   = math.radians(gmst)
    xe  =  x*math.cos(g) + y*math.sin(g)
    ye  = -x*math.sin(g) + y*math.cos(g)
    ze  = z
    a=6378.137; f=1/298.257223563; b=a*(1-f); e2=1-(b/a)**2
    lon=math.degrees(math.atan2(ye,xe)); p=math.sqrt(xe**2+ye**2)
    lat=math.degrees(math.atan2(ze,p*(1-e2)))
    for _ in range(4):
        sl=math.sin(math.radians(lat)); N=a/math.sqrt(1-e2*sl*sl)
        lat=math.degrees(math.atan2(ze+e2*N*sl,p))
    sl=math.sin(math.radians(lat)); N=a/math.sqrt(1-e2*sl*sl); cl=math.cos(math.radians(lat))
    alt=(p/cl-N) if abs(lat)<89 else (abs(ze)/sl-N*(1-e2))
    return round(lat,4), round(lon,4), round(alt,1)

def propagate_tle_now(line1, line2):
    if not HAS_SGP4: return None
    try:
        sat=Satrec.twoline2rv(line1,line2); now=datetime.now(timezone.utc)
        jd,fr=sgp4_jday(now.year,now.month,now.day,now.hour,now.minute,now.second+now.microsecond/1e6)
        e,rv,_=sat.sgp4(jd,fr)
        if e!=0 or rv is None: return None
        return teme_to_lla(rv,jd+fr)
    except Exception: return None

def parse_tle_block(text):
    sats=[]; lines=[l.strip() for l in text.splitlines() if l.strip()]; i=0
    while i+2<len(lines):
        n,l1,l2=lines[i],lines[i+1],lines[i+2]
        if l1.startswith('1 ') and l2.startswith('2 '): sats.append((n,l1,l2)); i+=3
        else: i+=1
    return sats

# ─────────────────────────────────────────────────────────────
# ISS PASS PREDICTOR
# ─────────────────────────────────────────────────────────────
_iss_tle_cache = {'line1':'','line2':'','fetched':0}
_iss_tle_lock  = threading.Lock()

def _fetch_iss_tle():
    with _iss_tle_lock:
        if time.time()-_iss_tle_cache['fetched']<3600 and _iss_tle_cache['line1']:
            return _iss_tle_cache['line1'], _iss_tle_cache['line2']
    try:
        r=requests.get(ISS_TLE_URL,timeout=8); r.raise_for_status()
        lines=[l.strip() for l in r.text.splitlines() if l.strip()]
        if len(lines)>=3 and lines[1].startswith('1 ') and lines[2].startswith('2 '):
            with _iss_tle_lock:
                _iss_tle_cache.update({'line1':lines[1],'line2':lines[2],'fetched':time.time()})
            return lines[1],lines[2]
    except Exception as e: print(f"[ISS TLE]{e}")
    with _iss_tle_lock: return _iss_tle_cache.get('line1',''),_iss_tle_cache.get('line2','')

def predict_iss_passes(obs_lat, obs_lon, obs_alt_km=0.0, n_passes=5, min_elev=10.0, days_ahead=2):
    if not HAS_SGP4: return []
    line1,line2=_fetch_iss_tle()
    if not line1: return []
    try:
        sat=Satrec.twoline2rv(line1,line2); start=datetime.now(timezone.utc)
        dt=60.0; passes=[]; in_pass=False; aos_ts=tca_ts=los_ts=None; max_el=-90.0
        a=6378.137; f=1/298.257223563; e2=2*f-f*f
        obs_lat_r=math.radians(obs_lat); obs_lon_r=math.radians(obs_lon)
        sl_o=math.sin(obs_lat_r); cl_o=math.cos(obs_lat_r)
        N_o=a/math.sqrt(1-e2*sl_o*sl_o)
        ox=(N_o+obs_alt_km)*cl_o*math.cos(obs_lon_r)
        oy=(N_o+obs_alt_km)*cl_o*math.sin(obs_lon_r)
        oz=(N_o*(1-e2)+obs_alt_km)*sl_o
        for step in range(int(days_ahead*86400/dt)):
            t=start+timedelta(seconds=step*dt)
            jd,fr=sgp4_jday(t.year,t.month,t.day,t.hour,t.minute,t.second+t.microsecond/1e6)
            e,rv,_=sat.sgp4(jd,fr)
            if e!=0 or rv is None: continue
            lat_s,lon_s,alt_s=teme_to_lla(rv,jd+fr)
            sl=math.sin(math.radians(lat_s)); cl=math.cos(math.radians(lat_s))
            N_s=a/math.sqrt(1-e2*sl*sl)
            sx=(N_s+alt_s)*cl*math.cos(math.radians(lon_s))
            sy=(N_s+alt_s)*cl*math.sin(math.radians(lon_s))
            sz=(N_s*(1-e2)+alt_s)*sl
            dx=sx-ox; dy=sy-oy; dz=sz-oz
            mag=math.sqrt(dx*dx+dy*dy+dz*dz)
            if mag<1e-6: continue
            dx/=mag; dy/=mag; dz/=mag
            ux=-sl_o*math.cos(obs_lon_r); uy=-sl_o*math.sin(obs_lon_r); uz=cl_o
            elev=math.degrees(math.asin(dx*ux+dy*uy+dz*uz))
            if not in_pass and elev>=min_elev:
                in_pass=True; aos_ts=t; max_el=elev; tca_ts=t
            elif in_pass:
                if elev>max_el: max_el=elev; tca_ts=t
                if elev<min_elev:
                    in_pass=False; los_ts=t
                    passes.append({'aos_utc':aos_ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                   'tca_utc':tca_ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                   'los_utc':los_ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                   'max_elev':round(max_el,1),
                                   'duration_s':int((los_ts-aos_ts).total_seconds())})
                    if len(passes)>=n_passes: break
        return passes
    except Exception as e: print(f"[ISS PASSES]{e}"); return []

# ─────────────────────────────────────────────────────────────
# THREAT SCORE ENGINE
# ─────────────────────────────────────────────────────────────
def compute_threat_scores(df_eq, df_aq, df_ev, df_fi):
    G=THREAT_GRID_DEG; grid={}
    def cell(lat,lon): return (int(lat//G)*G, int(lon//G)*G)
    def ensure(c):
        if c not in grid: grid[c]={'seismic':0.,'airqual':0.,'conflict':0.,'fire':0.}
    if not df_eq.empty and 'mag' in df_eq.columns:
        for _,r in df_eq.iterrows(): c=cell(r['lat'],r['lon']); ensure(c); grid[c]['seismic']+=10**float(r.get('mag',0))
    if not df_aq.empty and 'pm25' in df_aq.columns:
        for _,r in df_aq.iterrows(): c=cell(r['lat'],r['lon']); ensure(c); grid[c]['airqual']=max(grid[c]['airqual'],float(r.get('pm25',0)))
    if not df_ev.empty and 'count' in df_ev.columns:
        for _,r in df_ev.iterrows(): c=cell(r['lat'],r['lon']); ensure(c); grid[c]['conflict']+=float(r.get('count',1))*(1+max(0,-float(r.get('tone',0)))/20)
    if not df_fi.empty and 'power' in df_fi.columns:
        for _,r in df_fi.iterrows(): c=cell(r['lat'],r['lon']); ensure(c); grid[c]['fire']+=float(r.get('power',0))
    if not grid: return pd.DataFrame()
    rows=[{'lat_bin':lb+G/2,'lon_bin':lb2+G/2,'seismic':v['seismic'],'airqual':v['airqual'],'conflict':v['conflict'],'fire':v['fire']} for (lb,lb2),v in grid.items()]
    df=pd.DataFrame(rows)
    def nlog(s): s=s.copy(); s[s<=0]=1e-9; ls=np.log10(s); mn,mx=ls.min(),ls.max(); return (ls-mn)/(mx-mn+1e-9)
    def nlin(s): mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)
    df['n_seismic']=nlog(df['seismic']); df['n_airqual']=nlin(df['airqual'])
    df['n_conflict']=nlin(df['conflict']); df['n_fire']=nlog(df['fire'].clip(lower=0)+1)
    w=THREAT_W
    df['score']=(df['n_seismic']*w['seismic']+df['n_airqual']*w['airqual']+df['n_conflict']*w['conflict']+df['n_fire']*w['fire'])*100
    df['score']=df['score'].clip(0,100).round(1)
    return df.sort_values('score',ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# ANOMALY DETECTOR
# ─────────────────────────────────────────────────────────────
class AnomalyDetector:
    WINDOW  = 20
    METRICS = ['n_fl','n_eq','max_mag','max_aq']
    LABELS  = {
        'n_fl':   ('✈ FLIGHT COUNT SPIKE',   'warning'),
        'n_eq':   ('⚡ SEISMIC RATE SPIKE',  'critical'),
        'max_mag':('⚡ HIGH MAGNITUDE',       'critical'),
        'max_aq': ('💨 AQ SPIKE',            'warning'),
    }
    def __init__(self, threshold=ANOMALY_ZSCORE):
        self._lock=threading.Lock(); self._threshold=threshold
        self._windows={m:deque(maxlen=self.WINDOW) for m in self.METRICS}
        self._last_alert={m:0. for m in self.METRICS}; self._status={}
    def ingest(self, frame_entry):
        with self._lock:
            for m in self.METRICS: self._windows[m].append(float(frame_entry.get(m,0) or 0))
    def compute(self):
        anomalies=[]; now=time.time()
        with self._lock:
            for m in self.METRICS:
                win=list(self._windows[m])
                if len(win)<max(5,self.WINDOW//2): self._status[m]={'mean':0,'std':0,'z':0,'anomaly':False,'value':0}; continue
                arr=np.array(win,dtype=float); mean=float(np.mean(arr[:-1])); std=float(np.std(arr[:-1])); val=arr[-1]
                z=(val-mean)/(std+1e-9); is_anom=abs(z)>self._threshold and std>0.5
                self._status[m]={'mean':round(mean,2),'std':round(std,2),'z':round(z,2),'anomaly':is_anom,'value':round(val,2)}
                if is_anom and now-self._last_alert.get(m,0)>600:
                    self._last_alert[m]=now; lbl,sev=self.LABELS.get(m,(m,'info'))
                    anomalies.append({'metric':m,'severity':sev,'message':f"{lbl} — val={val:.1f} z={z:.1f}σ (base={mean:.1f}±{std:.1f})"})
        return anomalies
    def get_status(self):
        with self._lock: return dict(self._status)

_anomaly=AnomalyDetector(threshold=ANOMALY_ZSCORE)

# ─────────────────────────────────────────────────────────────
# GEOFENCE ENGINE
# ─────────────────────────────────────────────────────────────
class GeofenceEngine:
    def __init__(self):
        self._lock=threading.Lock(); self._zones={}; self._inside={}
    def add_zone(self,name,lat_min,lat_max,lon_min,lon_max):
        name=name.strip()[:32]
        with self._lock: self._zones[name]={'lat_min':lat_min,'lat_max':lat_max,'lon_min':lon_min,'lon_max':lon_max,'created':datetime.utcnow().isoformat()+'Z'}
        return name
    def remove_zone(self,name):
        with self._lock: self._zones.pop(name,None); [self._inside.pop(k,None) for k in [k for k in self._inside if k[2]==name]]
    def get_zones(self):
        with self._lock: return dict(self._zones)
    def _in(self,z,lat,lon): return z['lat_min']<=lat<=z['lat_max'] and z['lon_min']<=lon<=z['lon_max']
    def check(self,flights,ships):
        events=[]
        with self._lock: zones=dict(self._zones)
        for zn,z in zones.items():
            if not flights.empty:
                for _,row in flights.iterrows():
                    eid=str(row.get('icao','?')).upper(); key=('fl',eid,zn)
                    inside=self._in(z,float(row['lat']),float(row['lon']))
                    prev=self._inside.get(key,None)
                    with self._lock: self._inside[key]=inside
                    if prev is False and inside: events.append({'severity':'info','message':f"✈ {eid} ({str(row.get('callsign','?')).strip()}) entered [{zn}]",'key':f"gf-in:{eid}:{zn}"})
                    elif prev is True and not inside: events.append({'severity':'info','message':f"✈ {eid} exited [{zn}]",'key':f"gf-out:{eid}:{zn}"})
            for s in ships:
                eid=str(s.get('mmsi','?')); key=('sh',eid,zn)
                inside=self._in(z,float(s.get('lat',0)),float(s.get('lon',0)))
                prev=self._inside.get(key,None)
                with self._lock: self._inside[key]=inside
                if prev is False and inside: events.append({'severity':'info','message':f"🚢 {s.get('name','?')} entered [{zn}]",'key':f"gf-in:{eid}:{zn}"})
                elif prev is True and not inside: events.append({'severity':'info','message':f"🚢 {s.get('name','?')} exited [{zn}]",'key':f"gf-out:{eid}:{zn}"})
        return events

_geofence=GeofenceEngine()

# ─────────────────────────────────────────────────────────────
# CONVERGENCE DETECTOR
# ─────────────────────────────────────────────────────────────
class ConvergenceDetector:
    _NAMES={'quakes':'⚡ Seismic','fires':'🔥 Fire','aq':'💨 AQ','events':'📰 Conflict'}
    def __init__(self,min_layers=CONVERGENCE_LAYERS,grid_deg=THREAT_GRID_DEG):
        self._lock=threading.Lock(); self._min=min_layers; self._G=grid_deg; self._last_fire={}
    def detect(self,quakes,fires,aq,events):
        from collections import defaultdict
        cl=defaultdict(set)
        def add(df,name):
            if df is not None and not df.empty and 'lat' in df.columns:
                for _,r in df.iterrows():
                    c=(int(r['lat']//self._G)*self._G, int(r['lon']//self._G)*self._G); cl[c].add(name)
        add(quakes,'quakes'); add(fires,'fires'); add(aq,'aq'); add(events,'events')
        alerts=[]; now=time.time()
        for cell,layers in cl.items():
            if len(layers)<self._min: continue
            with self._lock:
                if now-self._last_fire.get(cell,0)<1800: continue
                self._last_fire[cell]=now
            layer_str=', '.join(self._NAMES.get(l,l) for l in sorted(layers))
            alerts.append({'severity':'critical' if len(layers)>=4 else 'warning',
                           'message':f"🔗 CONVERGENCE {cell[0]}°,{cell[1]}° — {len(layers)} hazards: {layer_str}",
                           'key':f"conv:{cell}:{now:.0f}"})
        return alerts

_convergence=ConvergenceDetector()

# ─────────────────────────────────────────────────────────────
# ALERT QUEUE
# ─────────────────────────────────────────────────────────────
class AlertQueue:
    def __init__(self,maxlen=100):
        self._lock=threading.Lock(); self._alerts=deque(maxlen=maxlen); self._seen=set()
    def push(self,severity,icon,message,key=None):
        with self._lock:
            k=key or message
            if k in self._seen: return
            self._seen.add(k)
            self._alerts.appendleft({'id':int(time.time()*1000),'severity':severity,'icon':icon,'message':message,'ts':datetime.utcnow().strftime('%H:%M:%S UTC')})
    def get_all(self):
        with self._lock: return list(self._alerts)
    def clear_seen(self):
        with self._lock: self._seen.clear()

_alerts=AlertQueue()

# ─────────────────────────────────────────────────────────────
# SSE BUS
# ─────────────────────────────────────────────────────────────
class SSEBus:
    def __init__(self):
        self._lock=threading.Lock(); self._listeners=[]; self._last=None
    def subscribe(self):
        q=_queue.Queue(maxsize=4)
        with self._lock:
            self._listeners.append(q)
            if self._last:
                try: q.put_nowait(self._last)
                except: pass
        return q
    def unsubscribe(self,q):
        with self._lock:
            try: self._listeners.remove(q)
            except ValueError: pass
    def publish(self,data):
        self._last=data; dead=[]
        with self._lock: ls=list(self._listeners)
        for q in ls:
            try: q.put_nowait(data)
            except: dead.append(q)
        if dead:
            with self._lock:
                for q in dead:
                    try: self._listeners.remove(q)
                    except ValueError: pass

_sse_bus=SSEBus()

# ─────────────────────────────────────────────────────────────
# RING BUFFER
# ─────────────────────────────────────────────────────────────
class RingBuffer:
    def __init__(self,capacity=RING_CAPACITY):
        self.capacity=capacity; self._lock=threading.Lock()
        self._con=sqlite3.connect(':memory:',check_same_thread=False); self._cur=self._con.cursor()
        self._cur.execute("CREATE TABLE frames(idx INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT NOT NULL,ts_epoch REAL NOT NULL,n_fl INTEGER DEFAULT 0,n_eq INTEGER DEFAULT 0,n_sh INTEGER DEFAULT 0,max_mag REAL DEFAULT 0,max_aq REAL DEFAULT 0,n_fires INTEGER DEFAULT 0,n_events INTEGER DEFAULT 0,blob BLOB NOT NULL)")
        self._con.commit(); self._total=0
    def push(self,snap):
        raw=json_lib.dumps(snap,default=str).encode(); blob=zlib.compress(raw,level=6)
        ts=snap.get('ts',datetime.utcnow().isoformat()+'Z'); counts=snap.get('counts',{})
        quakes=snap.get('quakes',[]); aq_vals=[r.get('pm25',0) for r in snap.get('aq',[])]
        max_mag=max((q.get('mag',0) for q in quakes),default=0.)
        max_aq=max(aq_vals) if aq_vals else 0.
        try: ts_epoch=datetime.fromisoformat(ts.rstrip('Z')).replace(tzinfo=timezone.utc).timestamp()
        except: ts_epoch=time.time()
        with self._lock:
            self._cur.execute("INSERT INTO frames(ts,ts_epoch,n_fl,n_eq,n_sh,max_mag,max_aq,n_fires,n_events,blob) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ts,ts_epoch,counts.get('flights',0),counts.get('quakes',0),counts.get('ships',0),max_mag,max_aq,counts.get('fires',0),counts.get('events',0),blob))
            self._con.commit(); self._total+=1
            self._cur.execute("SELECT COUNT() FROM frames"); cnt=self._cur.fetchone()[0]
            if cnt>self.capacity:
                self._cur.execute("DELETE FROM frames WHERE idx IN (SELECT idx FROM frames ORDER BY idx ASC LIMIT ?)",(cnt-self.capacity,)); self._con.commit()
    def get_frame(self,idx):
        with self._lock: self._cur.execute("SELECT blob FROM frames WHERE idx=?",(idx,)); row=self._cur.fetchone()
        if not row: return None
        try: return json_lib.loads(zlib.decompress(row[0]))
        except: return None
    def get_latest(self):
        with self._lock: self._cur.execute("SELECT idx FROM frames ORDER BY idx DESC LIMIT 1"); row=self._cur.fetchone()
        return self.get_frame(row[0]) if row else None
    def list_frames(self):
        with self._lock:
            self._cur.execute("SELECT idx,ts,ts_epoch,n_fl,n_eq,n_sh,max_mag,max_aq,n_fires,n_events FROM frames ORDER BY idx ASC")
            rows=self._cur.fetchall()
        return [{'idx':r[0],'ts':r[1],'ts_epoch':r[2],'n_fl':r[3],'n_eq':r[4],'n_sh':r[5],'max_mag':r[6],'max_aq':r[7],'n_fires':r[8],'n_events':r[9]} for r in rows]
    def stats(self):
        with self._lock:
            self._cur.execute("SELECT COUNT(),SUM(LENGTH(blob)) FROM frames"); cnt,sz=self._cur.fetchone(); sz=sz or 0
        return {'count':cnt or 0,'capacity':self.capacity,'fill_pct':round((cnt or 0)/self.capacity*100,1),'total_written':self._total,'size_kb':round(sz/1024,1)}

_ring=RingBuffer()

# ─────────────────────────────────────────────────────────────
# MULTI-CAMERA
# ─────────────────────────────────────────────────────────────
class MultiCam:
    N=4
    def __init__(self,sources):
        self._sources=list((sources+['']*self.N)[:self.N])
        self._locks=[threading.Lock() for _ in range(self.N)]
        self._frames=[None]*self.N; self._grabbed=[False]*self.N; self._caps=[None]*self.N; self._stopped=False
        for i in range(self.N):
            if self._sources[i]: self._connect(i)
    def _connect(self,idx):
        src=self._sources[idx]
        if not src: return
        if self._caps[idx]: self._caps[idx].release()
        cap=cv2.VideoCapture(src); self._caps[idx]=cap
        ok,frame=cap.read() if cap.isOpened() else (False,None)
        with self._locks[idx]: self._grabbed[idx]=ok; self._frames[idx]=frame
        print(f"[CAM{idx}] {'✓' if ok else '✗'} {src[:40]}")
    def set_source(self,idx,url):
        if 0<=idx<self.N: self._sources[idx]=url.strip(); self._connect(idx)
    def start(self):
        for i in range(self.N): threading.Thread(target=self._run,args=(i,),daemon=True,name=f'cam-{i}').start()
        return self
    def _run(self,idx):
        fails=0
        while not self._stopped:
            if self._caps[idx] and self._caps[idx].isOpened():
                ok,frame=self._caps[idx].read()
                if ok:
                    with self._locks[idx]: self._grabbed[idx]=ok; self._frames[idx]=frame
                    fails=0
                else:
                    fails+=1
                    if fails>20: self._connect(idx); fails=0
            else:
                time.sleep(2)
                if self._sources[idx]: self._connect(idx)
            time.sleep(0.033)
    def read_encoded(self,idx):
        if idx<0 or idx>=self.N: return None
        with self._locks[idx]: grabbed=self._grabbed[idx]; frame=self._frames[idx]
        if not grabbed or frame is None:
            img=np.zeros((240,320,3),dtype=np.uint8)
            cv2.putText(img,"NO SIGNAL",(70,110),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,55,55),1)
            _,buf=cv2.imencode('.jpg',img)
        else:
            fr=cv2.resize(frame,(320,240))
            cv2.putText(fr,f"CAM{idx} ◉",(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,80),1)
            cv2.putText(fr,datetime.now(timezone.utc).strftime("UTC %H:%M:%S"),(4,230),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,200,100),1)
            _,buf=cv2.imencode('.jpg',fr,[cv2.IMWRITE_JPEG_QUALITY,72])
        return base64.b64encode(buf).decode('utf-8')
    def stop(self):
        self._stopped=True
        for c in self._caps:
            if c: c.release()

cams=MultiCam(CAMERA_SOURCES).start()

# ─────────────────────────────────────────────────────────────
# FETCHERS
# ─────────────────────────────────────────────────────────────
def fetch_flights():
    try:
        r=requests.get(OPENSKY_URL,timeout=10); r.raise_for_status(); data=r.json()
        if 'states' not in data or not data['states']: return pd.DataFrame()
        df=pd.DataFrame(data['states']).iloc[:MAX_FLIGHTS,[0,1,2,5,6,7,9,10]]
        df.columns=['icao','callsign','country','lon','lat','alt','vel','hdg']
        df.dropna(subset=['lon','lat'],inplace=True)
        df['alt']=df['alt'].fillna(0).round(0); df['vel']=df['vel'].fillna(0).round(1); df['hdg']=df['hdg'].fillna(0).round(0)
        df['callsign']=df['callsign'].str.strip().replace({'':'UNKNOWN'}); df['alt_km']=(df['alt']/1000).round(2)
        return df
    except Exception as e: print(f"[ADS-B]{e}"); return pd.DataFrame()

def _parse_usgs(rj,max_rows):
    rows=[]
    for f in rj.get('features',[])[:max_rows]:
        p,c=f['properties'],f['geometry']['coordinates']
        rows.append({'lon':c[0],'lat':c[1],'depth':round(c[2],1),'mag':p.get('mag',0) or 0,
                     'place':p.get('place','Unknown'),'time_ms':p.get('time',0),
                     'time':datetime.fromtimestamp(p['time']/1000,tz=timezone.utc).strftime('%H:%M UTC'),
                     'url':p.get('url',''),'usgs_id':f.get('id','')})
    return pd.DataFrame(rows)

def fetch_earthquakes():
    try: r=requests.get(USGS_EQ_URL,timeout=10); r.raise_for_status(); return _parse_usgs(r.json(),MAX_EQ)
    except Exception as e: print(f"[EQ]{e}"); return pd.DataFrame()

def fetch_earthquakes_30d():
    try: r=requests.get(USGS_EQ_30D_URL,timeout=20); r.raise_for_status(); return _parse_usgs(r.json(),MAX_EQ30D)
    except Exception as e: print(f"[EQ-30d]{e}"); return pd.DataFrame()

def fetch_fires():
    if not FIRMS_MAP_KEY: return pd.DataFrame()
    try:
        r=requests.get(FIRMS_URL_TPL.format(key=FIRMS_MAP_KEY),timeout=15); r.raise_for_status()
        df=pd.read_csv(io.StringIO(r.text))
        if df.empty or 'latitude' not in df.columns: return pd.DataFrame()
        df=df[['latitude','longitude','bright_ti4','frp','daynight']].dropna()
        df=df.rename(columns={'latitude':'lat','longitude':'lon','bright_ti4':'brightness','frp':'power'})
        df['power']=df['power'].fillna(0).round(1); return df.head(MAX_FIRES)
    except Exception as e: print(f"[FIRES]{e}"); return pd.DataFrame()

def fetch_rainviewer_meta():
    try:
        r=requests.get(RAINVIEWER_META,timeout=6); r.raise_for_status()
        past=r.json().get('radar',{}).get('past',[]); return past[-1].get('path','') if past else ''
    except Exception as e: print(f"[RADAR]{e}"); return ''

def fetch_iss():
    try:
        r=requests.get(ISS_URL,timeout=6); r.raise_for_status(); d=r.json()
        return {'lat':round(float(d['latitude']),4),'lon':round(float(d['longitude']),4),
                'alt_km':round(float(d['altitude']),1),'vel_kph':round(float(d['velocity']),0),
                'vis':d.get('visibility','unknown'),'ts':time.time()}
    except Exception as e: print(f"[ISS]{e}"); return {}

def fetch_air_quality():
    try:
        headers={'X-API-Key':OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
        params={'limit':MAX_AQ,'parameter':'pm25','order_by':'lastUpdated','sort':'desc','has_geo':'true'}
        r=requests.get(OPENAQ_URL,params=params,headers=headers,timeout=12); r.raise_for_status()
        rows=[]
        for loc in r.json().get('results',[]):
            coords=loc.get('coordinates') or {}; lat=coords.get('latitude'); lon=coords.get('longitude')
            if lat is None or lon is None: continue
            pm25=None
            for param in loc.get('parameters',[]):
                if param.get('parameter')=='pm25' and param.get('lastValue') is not None: pm25=round(float(param['lastValue']),1); break
            if pm25 is None or pm25<0: continue
            color,label=aqi_color(pm25); country=loc.get('country',{})
            rows.append({'lat':round(float(lat),4),'lon':round(float(lon),4),'pm25':pm25,'label':label,'color':color,
                         'name':loc.get('name','?')[:30],'country':country.get('code','??') if isinstance(country,dict) else str(country)})
        return pd.DataFrame(rows)
    except Exception as e: print(f"[AQ]{e}"); return pd.DataFrame()

def fetch_gdelt_events():
    try:
        r=requests.get(GDELT_GEO_URL,timeout=12); r.raise_for_status(); rows=[]
        for feat in r.json().get('features',[])[:MAX_EVENTS]:
            geom=feat.get('geometry',{}); props=feat.get('properties',{})
            c=geom.get('coordinates',[])
            if len(c)<2: continue
            rows.append({'lon':round(float(c[0]),4),'lat':round(float(c[1]),4),
                         'title':str(props.get('name',props.get('title','Event')))[:80],
                         'url':str(props.get('url','')),'tone':float(props.get('tone',0) or 0),
                         'count':int(props.get('count',1) or 1)})
        return pd.DataFrame(rows)
    except Exception as e: print(f"[GDELT]{e}"); return pd.DataFrame()

def fetch_starlink():
    if not HAS_SGP4: return pd.DataFrame()
    try:
        r=requests.get(STARLINK_TLE_URL,timeout=20); r.raise_for_status()
        tles=parse_tle_block(r.text); rows=[]
        for name,l1,l2 in tles[:MAX_STARLINK*2]:
            res=propagate_tle_now(l1,l2)
            if res is None: continue
            lat,lon,alt=res
            if alt<200 or alt>2000: continue
            rows.append({'name':name.strip(),'lat':lat,'lon':lon,'alt_km':alt,'norad':l1[2:7].strip()})
            if len(rows)>=MAX_STARLINK: break
        print(f"[STARLINK]{len(rows)}"); return pd.DataFrame(rows)
    except Exception as e: print(f"[STARLINK]{e}"); return pd.DataFrame()

def fetch_weather_point(lat,lon):
    def _f():
        try:
            params={'latitude':lat,'longitude':lon,'forecast_days':2,
                    'current':'temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,cloud_cover,precipitation',
                    'hourly':'temperature_2m','wind_speed_unit':'kmh'}
            r=requests.get(OPEN_METEO_URL,params=params,timeout=5); r.raise_for_status(); return r.json()
        except: return {}
    try:
        with ThreadPoolExecutor(max_workers=1) as ex: return ex.submit(_f).result(timeout=5.5)
    except (FutTimeout,Exception): return {}

WMO_CODES={0:'Clear sky',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',45:'Fog',48:'Icy fog',
    51:'Light drizzle',53:'Drizzle',55:'Dense drizzle',61:'Light rain',63:'Rain',65:'Heavy rain',
    71:'Light snow',73:'Snow',75:'Heavy snow',80:'Light showers',81:'Showers',82:'Heavy showers',
    95:'Thunderstorm',96:'Thunderstorm+hail',99:'Heavy hail'}

# ── POI category → icon/colour map ──────────────────────────
POI_ICONS = {
    'restaurant':'🍽', 'cafe':'☕', 'fast_food':'🍟', 'bar':'🍺', 'pub':'🍻',
    'hospital':'🏥', 'clinic':'🏥', 'pharmacy':'💊', 'doctors':'⚕',
    'school':'🏫', 'university':'🎓', 'library':'📚',
    'supermarket':'🛒', 'convenience':'🏪', 'marketplace':'🏪',
    'fuel':'⛽', 'parking':'🅿', 'bank':'🏦', 'atm':'💳',
    'hotel':'🏨', 'hostel':'🏨', 'museum':'🏛', 'theatre':'🎭',
    'cinema':'🎬', 'park':'🌳', 'police':'🚓', 'fire_station':'🚒',
    'place_of_worship':'⛪', 'post_office':'📮', 'toilets':'🚻',
    'charging_station':'⚡', 'bicycle_rental':'🚲', 'bus_station':'🚌',
}
POI_COLORS = {
    'restaurant':'#ff8844', 'cafe':'#cc8822', 'fast_food':'#ff6600', 'bar':'#aa5500',
    'hospital':'#ff4466', 'clinic':'#ff4466', 'pharmacy':'#ff88aa',
    'school':'#44aaff', 'university':'#2288ff', 'library':'#4488cc',
    'fuel':'#ffcc00', 'parking':'#888888', 'bank':'#aacc44',
    'hotel':'#cc88ff', 'museum':'#ff88cc', 'park':'#44cc66',
    'police':'#0088ff', 'default':'#aaaaaa',
}

def _poi_color(cat):
    for k,v in POI_COLORS.items():
        if k in cat: return v
    return POI_COLORS['default']

def _poi_icon(cat):
    for k,v in POI_ICONS.items():
        if k in cat: return v
    return '📍'


def fetch_overpass_poi(lat, lon, radius_m=3000):
    """Query OpenStreetMap Overpass API for nearby amenities. Free, no key."""
    query = f"""
[out:json][timeout:8];
(
  node["amenity"](around:{radius_m},{lat},{lon});
  node["shop"](around:{radius_m},{lat},{lon});
  node["tourism"](around:{radius_m},{lat},{lon});
  node["leisure"]["leisure"!="grass"](around:{radius_m},{lat},{lon});
);
out body 300;
"""
    try:
        r = requests.post('https://overpass-api.de/api/interpreter',
                          data={'data': query}, timeout=10)
        r.raise_for_status()
        rows = []
        for el in r.json().get('elements', []):
            lat_e = el.get('lat'); lon_e = el.get('lon')
            if lat_e is None or lon_e is None: continue
            tags = el.get('tags', {})
            cat  = (tags.get('amenity') or tags.get('shop') or
                    tags.get('tourism') or tags.get('leisure') or 'place')
            name = tags.get('name', cat)[:40]
            rows.append({'lat':round(float(lat_e),6),'lon':round(float(lon_e),6),
                         'name':name,'category':cat,
                         'icon':_poi_icon(cat),'color':_poi_color(cat)})
        print(f"[POI] {len(rows)} results for ({lat:.3f},{lon:.3f}) r={radius_m}m")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[POI] {e}"); return pd.DataFrame()


def geocode_location(query: str):
    """Nominatim geocoder — returns (lat, lon, display_name) or None."""
    try:
        r = requests.get('https://nominatim.openstreetmap.org/search',
                         params={'q': query, 'format': 'json', 'limit': 1},
                         headers={'Accept-Language': 'en', 'User-Agent': f'GSF/{VERSION}'},
                         timeout=6)
        r.raise_for_status()
        data = r.json()
        if not data: return None
        best = data[0]
        return (round(float(best['lat']), 5),
                round(float(best['lon']), 5),
                best.get('display_name', query)[:60])
    except Exception as e:
        print(f"[GEOCODE] {e}"); return None


# ─────────────────────────────────────────────────────────────
# EVENT INTELLIGENCE ENGINE  ← v1.2
# ─────────────────────────────────────────────────────────────

TM_BASE = "https://app.ticketmaster.com/discovery/v2"

# Ticketmaster segment/genre labels → user-readable + emoji
TM_SEGMENTS = {
    'KZFzniwnSyZfZ7v7nJ': ('🎵','Music'),
    'KZFzniwnSyZfZ7v7nE': ('⚽','Sports'),
    'KZFzniwnSyZfZ7v7na': ('🎨','Arts & Theatre'),
    'KZFzniwnSyZfZ7v7nn': ('🎡','Miscellaneous'),
    'KZFzniwnSyZfZ7v7n1': ('🎪','Film & Media'),
    'KZFzniwnSyZfZ7v7nk': ('🏛','Family'),
}

def _tm_segment_label(seg_id: str):
    return TM_SEGMENTS.get(seg_id, ('🎟','Event'))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R  = 6371.0
    φ1 = math.radians(lat1); φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_ticketmaster_events(lat: float, lon: float,
                              radius_km: int = EVENTS_RADIUS_KM,
                              size: int = EVENTS_MAX_RESULTS) -> list[dict]:
    """
    Query Ticketmaster Discovery API for real upcoming events near a location.
    Free API key (5000 calls/day): https://developer.ticketmaster.com/
    Falls back to empty list if no key or API error.
    """
    if not TICKETMASTER_API_KEY:
        return []
    try:
        r = requests.get(f"{TM_BASE}/events.json", params={
            'apikey':    TICKETMASTER_API_KEY,
            'latlong':   f"{lat},{lon}",
            'radius':    str(radius_km),
            'unit':      'km',
            'size':      str(size),
            'sort':      'date,asc',
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get('_embedded', {}).get('events', [])
        results = []
        for ev in items:
            dates = ev.get('dates', {})
            start = dates.get('start', {})
            date_str  = start.get('localDate', '')
            time_str  = start.get('localTime', '')
            tba       = start.get('timeTBA', False)
            status    = dates.get('status', {}).get('code', 'onsale')
            venues    = ev.get('_embedded', {}).get('venues', [{}])
            venue     = venues[0] if venues else {}
            v_name    = venue.get('name', 'Unknown Venue')
            v_city    = venue.get('city', {}).get('name', '')
            v_country = venue.get('country', {}).get('name', '')
            v_loc     = venue.get('location', {})
            v_lat     = float(v_loc.get('latitude',  lat))
            v_lon     = float(v_loc.get('longitude', lon))
            seg_id    = ''
            class_list= ev.get('classifications', [{}])
            if class_list:
                seg = class_list[0].get('segment', {})
                seg_id = seg.get('id', '')
            icon, label = _tm_segment_label(seg_id)
            dist = haversine_km(lat, lon, v_lat, v_lon)
            price_ranges = ev.get('priceRanges', [])
            price_str = ''
            if price_ranges:
                pr = price_ranges[0]
                cur = pr.get('currency', 'USD')
                mn  = pr.get('min', 0); mx = pr.get('max', 0)
                price_str = f"{cur} {mn:.0f}–{mx:.0f}" if mn != mx else f"{cur} {mn:.0f}"
            time_display = (time_str[:5] if time_str and not tba else 'TBA')
            results.append({
                'source':   'ticketmaster',
                'id':       ev.get('id', ''),
                'name':     ev.get('name', 'Unknown Event')[:70],
                'icon':     icon,
                'category': label,
                'date':     date_str,
                'time':     time_display,
                'datetime_sort': date_str + 'T' + (time_str or '99:99'),
                'venue':    v_name[:40],
                'city':     v_city,
                'country':  v_country,
                'lat':      v_lat,
                'lon':      v_lon,
                'dist_km':  round(dist, 1),
                'status':   status,
                'price':    price_str,
                'url':      ev.get('url', ''),
            })
        return sorted(results, key=lambda x: x['datetime_sort'])
    except Exception as e:
        print(f"[TM] {e}"); return []


def fetch_overpass_events(lat: float, lon: float,
                          radius_m: int = 20000) -> list[dict]:
    """
    Query OpenStreetMap Overpass for event venues, stadiums, theatres,
    festival grounds, conference centres, etc. No API key needed.
    """
    query = f"""
[out:json][timeout:10];
(
  node["amenity"~"events_venue|cinema|theatre|concert_hall|music_venue|conference_centre"](around:{radius_m},{lat},{lon});
  node["leisure"~"stadium|sports_centre|arena|dance|amusement_arcade"](around:{radius_m},{lat},{lon});
  node["tourism"~"museum|attraction|theme_park|zoo"](around:{radius_m},{lat},{lon});
  node["building"~"stadium|arena|exhibition_centre|sports_hall"](around:{radius_m},{lat},{lon});
  way ["amenity"~"events_venue|theatre|concert_hall|cinema"](around:{radius_m},{lat},{lon});
  way ["leisure"~"stadium|sports_centre|arena"](around:{radius_m},{lat},{lon});
);
out center 80;
"""
    try:
        r = requests.post('https://overpass-api.de/api/interpreter',
                          data={'data': query}, timeout=12)
        r.raise_for_status()
        rows = []
        seen = set()
        for el in r.json().get('elements', []):
            # Ways return centre; nodes return lat/lon directly
            if el['type'] == 'way':
                centre = el.get('center', {})
                v_lat = centre.get('lat'); v_lon = centre.get('lon')
            else:
                v_lat = el.get('lat'); v_lon = el.get('lon')
            if v_lat is None or v_lon is None: continue
            tags  = el.get('tags', {})
            name  = tags.get('name', '')
            if not name or name in seen: continue
            seen.add(name)
            cat_raw = (tags.get('amenity') or tags.get('leisure') or
                       tags.get('tourism') or tags.get('building') or 'venue')
            cat_map = {
                'theatre':'🎭 Theatre','cinema':'🎬 Cinema','concert_hall':'🎵 Concert Hall',
                'music_venue':'🎵 Music Venue','events_venue':'🎪 Event Venue',
                'conference_centre':'🏛 Conference','stadium':'🏟 Stadium',
                'sports_centre':'⚽ Sports Centre','arena':'🏟 Arena',
                'museum':'🏛 Museum','attraction':'🌟 Attraction',
                'theme_park':'🎡 Theme Park','zoo':'🐾 Zoo',
                'dance':'💃 Dance','amusement_arcade':'🎮 Arcade',
            }
            icon_label = cat_map.get(cat_raw, '🏢 Venue')
            icon, label = icon_label.split(' ', 1)
            dist = haversine_km(lat, lon, float(v_lat), float(v_lon))
            rows.append({
                'source':   'osm_venue',
                'id':       f"osm-{el['id']}",
                'name':     name[:60],
                'icon':     icon,
                'category': label,
                'date':     '',
                'time':     '',
                'datetime_sort': 'VENUE',
                'venue':    name[:40],
                'city':     tags.get('addr:city', ''),
                'country':  tags.get('addr:country', ''),
                'lat':      float(v_lat),
                'lon':      float(v_lon),
                'dist_km':  round(dist, 1),
                'status':   'venue',
                'price':    '',
                'url':      tags.get('website', tags.get('contact:website', '')),
            })
        return sorted(rows, key=lambda x: x['dist_km'])
    except Exception as e:
        print(f"[OSM events] {e}"); return []


def nearby_gdelt_context(lat: float, lon: float,
                          radius_km: float,
                          df_ev: pd.DataFrame) -> list[dict]:
    """
    Filter the cached GDELT events DataFrame to those within radius_km
    of the given point. Returns list of dicts sorted by article count desc.
    """
    if df_ev.empty or 'lat' not in df_ev.columns:
        return []
    results = []
    for _, row in df_ev.iterrows():
        dist = haversine_km(lat, lon, float(row['lat']), float(row['lon']))
        if dist <= radius_km:
            tone = float(row.get('tone', 0))
            # Simple sentiment label
            if tone < -10:     sentiment = '🔴 Negative'
            elif tone < -3:    sentiment = '🟠 Concerning'
            elif tone > 3:     sentiment = '🟢 Positive'
            else:              sentiment = '🟡 Neutral'
            results.append({
                'title':     str(row.get('title', 'News'))[:80],
                'count':     int(row.get('count', 1)),
                'tone':      round(tone, 1),
                'sentiment': sentiment,
                'dist_km':   round(dist, 1),
                'lat':       float(row['lat']),
                'lon':       float(row['lon']),
                'url':       str(row.get('url', '')),
            })
    return sorted(results, key=lambda x: x['count'], reverse=True)[:20]


# ─────────────────────────────────────────────────────────────
# DATA CACHE
# ─────────────────────────────────────────────────────────────
class DataCache:
    def __init__(self):
        self._lock=threading.Lock()
        self._flights=pd.DataFrame(); self._quakes=pd.DataFrame(); self._quakes_30d=pd.DataFrame()
        self._fires=pd.DataFrame(); self._ships={}; self._trails={}; self._radar_path=''
        self._iss={}; self._iss_trail=[]; self._aq=pd.DataFrame()
        self._events=pd.DataFrame(); self._starlink=pd.DataFrame(); self._threat=pd.DataFrame()
        self._anomaly_status={}; self._version=0

    def _bump(self): self._version+=1

    def set_flights(self,df):
        with self._lock:
            self._flights=df.copy(); now_iso=datetime.utcnow().strftime('%H:%M:%S')
            for _,row in df.iterrows():
                icao=str(row['icao']); pos=(float(row['lon']),float(row['lat']),float(row['alt']),now_iso)
                if icao not in self._trails: self._trails[icao]=[]
                trail=self._trails[icao]
                if not trail or trail[-1][:2]!=pos[:2]: trail.append(pos)
                if len(trail)>TRAIL_MAX_POS: trail.pop(0)
            active=set(df['icao'].astype(str))
            for k in list(self._trails):
                if k not in active: del self._trails[k]
            self._bump()

    def set_quakes(self,df):   self._safe_set('_quakes',df)
    def set_quakes_30d(self,df):
        with self._lock: self._quakes_30d=df.copy()
    def set_fires(self,df):    self._safe_set('_fires',df)
    def set_aq(self,df):       self._safe_set('_aq',df)
    def set_events(self,df):   self._safe_set('_events',df)
    def set_starlink(self,df): self._safe_set('_starlink',df)
    def set_threat(self,df):   self._safe_set('_threat',df)
    def _safe_set(self,attr,df):
        with self._lock: setattr(self,attr,df.copy()); self._bump()

    def upsert_ship(self,mmsi,ship):
        with self._lock: self._ships[mmsi]=ship; self._bump()
    def prune_ships(self):
        cutoff=time.time()-SHIP_TTL
        with self._lock: self._ships={k:v for k,v in self._ships.items() if v.get('ts',0)>cutoff}
    def set_radar_path(self,path):
        with self._lock:
            if path!=self._radar_path: self._radar_path=path; self._bump()
    def set_iss(self,data):
        with self._lock:
            self._iss=data
            if data:
                pos=(data['lon'],data['lat'],data['alt_km']*1000)
                if not self._iss_trail or self._iss_trail[-1][:2]!=pos[:2]: self._iss_trail.append(pos)
                if len(self._iss_trail)>ISS_TRAIL_MAX: self._iss_trail.pop(0)
            self._bump()
    def set_anomaly_status(self,st):
        with self._lock: self._anomaly_status=st

    def get_version(self):
        with self._lock: return self._version
    def get_flights(self):
        with self._lock: return self._flights.copy()
    def get_quakes(self):
        with self._lock: return self._quakes.copy()
    def get_quakes_30d(self):
        with self._lock: return self._quakes_30d.copy()
    def get_fires(self):
        with self._lock: return self._fires.copy()
    def get_ships(self):
        cutoff=time.time()-SHIP_TTL
        with self._lock: return [v for v in self._ships.values() if v.get('ts',0)>cutoff][:MAX_SHIPS]
    def get_trails(self):
        with self._lock: return {k:list(v) for k,v in self._trails.items()}
    def get_trail(self,icao):
        with self._lock: return list(self._trails.get(icao,[]))
    def get_radar_path(self):
        with self._lock: return self._radar_path
    def get_iss(self):
        with self._lock: return dict(self._iss),list(self._iss_trail)
    def get_aq(self):
        with self._lock: return self._aq.copy()
    def get_events(self):
        with self._lock: return self._events.copy()
    def get_starlink(self):
        with self._lock: return self._starlink.copy()
    def get_threat(self):
        with self._lock: return self._threat.copy()
    def get_anomaly_status(self):
        with self._lock: return dict(self._anomaly_status)

    def to_snapshot(self):
        def srec(df,cols):
            if df.empty: return []
            return df[[c for c in cols if c in df.columns]].dropna().to_dict('records')
        with self._lock:
            return {
                'ts':        datetime.utcnow().isoformat()+'Z',
                'v':         self._version,
                'flights':   srec(self._flights,['icao','callsign','country','lon','lat','alt','vel','hdg']),
                'quakes':    srec(self._quakes, ['lon','lat','mag','depth','place','time']),
                'fires':     srec(self._fires,  ['lon','lat','brightness','power']),
                'ships':     list(self._ships.values())[:MAX_SHIPS],
                'trails':    {k:[p[:3] for p in v] for k,v in self._trails.items()},
                'radar_path':self._radar_path,
                'iss':       dict(self._iss),
                'iss_trail': list(self._iss_trail),
                'aq':        srec(self._aq,['lon','lat','pm25','label','color','name','country']),
                'events':    srec(self._events,['lon','lat','title','tone','count']),
                'starlink':  srec(self._starlink,['lon','lat','alt_km','name','norad']),
                'threat':    srec(self._threat,['lat_bin','lon_bin','score','n_seismic','n_airqual','n_conflict','n_fire']),
                'geofences': dict(_geofence.get_zones()),
                'anomaly':   dict(self._anomaly_status),
                'counts': {
                    'flights': len(self._flights),'quakes':len(self._quakes),
                    'fires':   len(self._fires),  'ships': len(self._ships),
                    'aq':      len(self._aq),     'events':len(self._events),
                    'starlink':len(self._starlink),'threat':len(self._threat),
                },
            }

    def to_geojson(self):
        feats=[]
        def add(lon,lat,props): feats.append({'type':'Feature','geometry':{'type':'Point','coordinates':[lon,lat]},'properties':props})
        with self._lock:
            for _,r in self._flights.iterrows(): add(r['lon'],r['lat'],{'layer':'flight','icao':str(r['icao']),'callsign':str(r['callsign']),'alt_m':float(r['alt']),'vel_ms':float(r['vel'])})
            for _,r in self._quakes.iterrows(): add(r['lon'],r['lat'],{'layer':'earthquake','mag':float(r['mag']),'depth_km':float(r['depth']),'place':str(r['place'])})
            for _,r in self._fires.iterrows(): add(r['lon'],r['lat'],{'layer':'fire','brightness_k':float(r['brightness']),'frp_mw':float(r['power'])})
            for v in list(self._ships.values())[:MAX_SHIPS]: add(v['lon'],v['lat'],{'layer':'ship','mmsi':v['mmsi'],'name':v['name'],'sog_kn':v['sog']})
            if self._iss: add(self._iss['lon'],self._iss['lat'],{'layer':'iss','alt_km':self._iss['alt_km']})
            for _,r in self._aq.iterrows(): add(r['lon'],r['lat'],{'layer':'air_quality','name':str(r['name']),'pm25':float(r['pm25'])})
            for _,r in self._starlink.iterrows(): add(r['lon'],r['lat'],{'layer':'starlink','name':str(r['name']),'alt_km':float(r['alt_km'])})
            for _,r in self._threat.iterrows(): add(r['lon_bin'],r['lat_bin'],{'layer':'threat','score':float(r['score'])})
        return {'type':'FeatureCollection','features':feats,'metadata':{'source':f'GSF v{VERSION}','generated':datetime.utcnow().isoformat()+'Z','count':len(feats)}}

_cache=DataCache()

# ─────────────────────────────────────────────────────────────
# BACKGROUND UPDATER
# ─────────────────────────────────────────────────────────────
_seen_eq=set(); _seen_aq=set(); _seen_ev=set()

def _check_alerts():
    df=_cache.get_quakes()
    if not df.empty and 'mag' in df.columns:
        for _,r in df[df['mag']>=ALERT_EQ_MAG].iterrows():
            key=str(r.get('usgs_id',r.get('place','')))
            if key and key not in _seen_eq: _seen_eq.add(key); _alerts.push('critical','⚡',f"M{round(float(r['mag']),1)} — {str(r.get('place','?'))[:40]}",key=key)
    df=_cache.get_aq()
    if not df.empty and 'pm25' in df.columns:
        for _,r in df[df['pm25']>=ALERT_AQ_LEVEL].iterrows():
            key=f"aq:{r.get('name','')}:{r.get('country','')}"
            if key not in _seen_aq: _seen_aq.add(key); _alerts.push('warning','💨',f"Hazardous AQ — {r.get('name','?')} PM2.5={r['pm25']}",key=key)
    df=_cache.get_events()
    if not df.empty and 'count' in df.columns:
        for _,r in df[df['count']>=ALERT_GDELT_COUNT].iterrows():
            key=f"ev:{str(r.get('title',''))[:40]}"
            if key not in _seen_ev: _seen_ev.add(key); _alerts.push('info','📰',f"{r['count']} articles — {str(r.get('title',''))[:50]}",key=key)

def _bg_updater():
    last={k:0 for k in ['flights','quakes','eq30d','fires','radar','iss','aq','gdelt','starlink','threat','record','prune','seen']}
    last_v=-1
    while True:
        now=time.time()
        if now-last['flights']  >= FLIGHT_INTERVAL:
            df=fetch_flights();          (not df.empty) and _cache.set_flights(df);   last['flights']=now
        if now-last['quakes']   >= QUAKE_INTERVAL:
            df=fetch_earthquakes();      (not df.empty) and _cache.set_quakes(df);    last['quakes']=now
        if now-last['eq30d']    >= EQ30D_INTERVAL:
            df=fetch_earthquakes_30d();  (not df.empty) and _cache.set_quakes_30d(df);last['eq30d']=now
        if now-last['fires']    >= FIRE_INTERVAL:
            df=fetch_fires();            (not df.empty) and _cache.set_fires(df);     last['fires']=now
        if now-last['radar']    >= RADAR_INTERVAL:
            p=fetch_rainviewer_meta();   p and _cache.set_radar_path(p);              last['radar']=now
        if now-last['iss']      >= ISS_INTERVAL:
            d=fetch_iss();               d and _cache.set_iss(d);                     last['iss']=now
        if now-last['aq']       >= AQ_INTERVAL:
            df=fetch_air_quality();      (not df.empty) and _cache.set_aq(df);        last['aq']=now
        if now-last['gdelt']    >= GDELT_INTERVAL:
            df=fetch_gdelt_events();     (not df.empty) and _cache.set_events(df);    last['gdelt']=now
        if now-last['starlink'] >= STARLINK_INTERVAL:
            df=fetch_starlink();         (not df.empty) and _cache.set_starlink(df);  last['starlink']=now
        if now-last['threat']   >= THREAT_INTERVAL:
            df=compute_threat_scores(_cache.get_quakes(),_cache.get_aq(),_cache.get_events(),_cache.get_fires())
            if not df.empty: _cache.set_threat(df)
            last['threat']=now
        # Anomaly detection (feed latest ring frame metrics)
        frames=_ring.list_frames()
        if frames:
            _anomaly.ingest(frames[-1])
            anom_events=_anomaly.compute()
            _cache.set_anomaly_status(_anomaly.get_status())
            for av in anom_events: _alerts.push(av['severity'],'🤖',av['message'],key=f"anom:{av['metric']}:{now:.0f}")
        # Geofence check
        if _geofence.get_zones():
            for gev in _geofence.check(_cache.get_flights(),_cache.get_ships()):
                _alerts.push(gev['severity'],'🔲',gev['message'],key=gev.get('key',gev['message']))
        # Convergence
        for cev in _convergence.detect(_cache.get_quakes(),_cache.get_fires(),_cache.get_aq(),_cache.get_events()):
            _alerts.push(cev['severity'],'🔗',cev['message'],key=cev.get('key',cev['message']))
        # Ring buffer record
        if now-last['record'] >= RECORD_INTERVAL:
            snap=_cache.to_snapshot(); snap['alerts']=_alerts.get_all(); _ring.push(snap); last['record']=now
        _check_alerts()
        v=_cache.get_version()
        if v!=last_v:
            snap=_cache.to_snapshot(); snap['alerts']=_alerts.get_all()
            try: _sse_bus.publish(json_lib.dumps(snap,default=str))
            except Exception as e: print(f"[SSE]{e}")
            last_v=v
        if now-last['prune'] >= 60: _cache.prune_ships(); last['prune']=now
        if now-last['seen']  >= 3600: _seen_eq.clear(); _seen_aq.clear(); _seen_ev.clear(); _alerts.clear_seen(); last['seen']=now
        time.sleep(5)

threading.Thread(target=_bg_updater,daemon=True,name='bg').start()

# ─────────────────────────────────────────────────────────────
# AIS WEBSOCKET
# ─────────────────────────────────────────────────────────────
class AISStream:
    def __init__(self,api_key,cache): self.api_key=api_key; self.cache=cache
    def start(self):
        if not self.api_key or not HAS_WS: return self
        threading.Thread(target=self._run,daemon=True,name='ais').start(); return self
    def _run(self):
        while True:
            try:
                ws=websocket.WebSocketApp(AIS_WS_URL,on_open=self._on_open,on_message=self._on_msg,on_error=lambda ws,e:None,on_close=lambda ws,c,m:None)
                ws.run_forever(ping_interval=30,ping_timeout=10)
            except: pass
            time.sleep(15)
    def _on_open(self,ws):
        ws.send(json_lib.dumps({"APIKey":self.api_key,"BoundingBoxes":[[[-90,-180],[90,180]]],"FilterMessageTypes":["PositionReport","StandardClassBPositionReport"]}))
    def _on_msg(self,ws,raw):
        try:
            data=json_lib.loads(raw); meta=data.get('MetaData',{}); msg=data.get('Message',{})
            pos=msg.get('PositionReport') or msg.get('StandardClassBPositionReport',{})
            if not pos: return
            mmsi=str(pos.get('UserID','')); lat=pos.get('Latitude'); lon=pos.get('Longitude')
            if not mmsi or lat is None or lon is None: return
            self.cache.upsert_ship(mmsi,{'mmsi':mmsi,'name':meta.get('ShipName','UNKNOWN').strip() or mmsi,
                'lat':round(float(lat),5),'lon':round(float(lon),5),
                'cog':round(float(pos.get('Cog',0) or 0),1),'sog':round(float(pos.get('Sog',0) or 0),1),'ts':time.time()})
        except: pass

AISStream(AIS_API_KEY,_cache).start()

# ─────────────────────────────────────────────────────────────
# DECK.GL WebGL BONUS VIEW  (/deck)
# ─────────────────────────────────────────────────────────────
DECK_HTML = r"""<!DOCTYPE html><html><head>
  <meta charset="utf-8"><title>GSF __VERSION__ · deck.gl WebGL</title>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <script src="https://unpkg.com/maplibre-gl@3/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@3/dist/maplibre-gl.css" rel="stylesheet"/>
  <style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:#040c06;font-family:'Courier New',monospace}
  #dc{position:absolute;top:0;left:0;width:100%;height:100%}
  #ui{position:absolute;top:0;left:0;right:0;height:34px;background:rgba(4,12,6,.92);border-bottom:1px solid #0b2e18;display:flex;align-items:center;padding:0 12px;gap:10px;z-index:10}
  .brand{color:#00ff88;font-size:.7rem;letter-spacing:.12em}.brand span{color:#1e5530}
  .sp{font-size:.6rem;color:#336644;white-space:nowrap}.sp .v{color:#00ff88;font-weight:bold}
  #clk{color:#1e5530;font-size:.58rem;margin-left:auto}
  #tt{position:absolute;pointer-events:none;z-index:20;background:rgba(4,10,6,.93);border:1px solid #0d3320;border-radius:4px;padding:7px 9px;font-size:.67rem;color:#aaffcc;max-width:210px;display:none}
  #tt .t{color:#00ff88;font-weight:bold;margin-bottom:3px}.kv{margin:1px 0}.kv .k{color:#2a5535}.kv .v{color:#aaffcc}
  #lb{position:absolute;bottom:0;left:0;right:0;height:34px;background:rgba(4,12,6,.9);border-top:1px solid #0b2e18;display:flex;align-items:center;padding:0 8px;gap:5px;z-index:10}
  .lb{font-family:'Courier New',monospace;background:rgba(0,20,10,.8);border:1px solid #0d3320;color:#336644;font-size:.58rem;padding:3px 6px;border-radius:3px;cursor:pointer}
  .lb.on{border-color:#00cc55;color:#00ff88;background:rgba(0,40,20,.9)}
  #st{margin-left:auto;font-size:.58rem;color:#1a3a22}</style>
</head><body>
<div id="dc"></div>
<div id="ui">
  <div class="brand">◈ GSF <span>· v__VERSION__ · WebGL GPU</span></div>
  <div class="sp">✈ <span class="v" id="n-fl">—</span></div>
  <div class="sp">⚡ <span class="v" id="n-eq">—</span></div>
  <div class="sp">🎯 <span class="v" id="n-th">—</span></div>
  <div class="sp">🚢 <span class="v" id="n-sh">—</span></div>
  <div class="sp">🛰 <span class="v" id="n-sl">—</span></div>
  <div id="clk">UTC --:--:--</div>
</div>
<div id="tt"></div>
<div id="lb">
  <button class="lb on" id="b-fl" onclick="tl('flights')">✈ AC</button>
  <button class="lb on" id="b-hx" onclick="tl('hex')">⬡ DENSITY</button>
  <button class="lb on" id="b-eq" onclick="tl('quakes')">⚡ EQ</button>
  <button class="lb on" id="b-sh" onclick="tl('ships')">🚢</button>
  <button class="lb on" id="b-sl" onclick="tl('starlink')">🛰</button>
  <button class="lb on" id="b-th" onclick="tl('threat')">🎯</button>
  <button class="lb on" id="b-fi" onclick="tl('fires')">🔥</button>
  <div id="st">SSE ◌</div>
</div>
<script>
'use strict';
const {DeckGL,ScatterplotLayer,HexagonLayer}=deck;
const lv={flights:true,hex:true,quakes:true,ships:true,starlink:true,threat:true,fires:true};
let snap=null;
const dgl=new DeckGL({container:'dc',mapStyle:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',initialViewState:{longitude:0,latitude:20,zoom:1.8,pitch:35,bearing:0},controller:true});
const tt=document.getElementById('tt');
function kv(k,v){return `<div class="kv"><span class="k">${k}:</span> <span class="v">${v}</span></div>`;}
function showTT(info){if(!info?.object){tt.style.display='none';return;}const{x,y,object:d,layer}=info;let h='';
  if(layer?.id==='flights')h=`<div class="t">✈ ${(d.icao||'').toUpperCase()}</div>${kv('Callsign',d.callsign||'—')}${kv('Alt',`${Math.round(d.alt||0).toLocaleString()} m`)}${kv('Speed',`${d.vel||0} m/s`)}`;
  else if(layer?.id==='quakes')h=`<div class="t" style="color:#ff6633">⚡ M${(d.mag||0).toFixed(1)}</div>${kv('Location',(d.place||'').substring(0,28))}${kv('Depth',`${d.depth||0} km`)}`;
  else if(layer?.id==='ships')h=`<div class="t" style="color:#00ccff">🚢 ${d.name||d.mmsi||'?'}</div>${kv('MMSI',d.mmsi||'—')}${kv('Speed',`${d.sog||0} kn`)}`;
  else if(layer?.id==='starlink')h=`<div class="t" style="color:#aaaaff">🛰 ${(d.name||'').replace('STARLINK-','SL-')}</div>${kv('NORAD',d.norad||'—')}${kv('Alt',`${d.alt_km} km`)}`;
  else if(layer?.id==='fires')h=`<div class="t" style="color:#ff5500">🔥 FIRE</div>${kv('Brightness',`${(d.brightness||0).toFixed(1)} K`)}${kv('FRP',`${d.power||0} MW`)}`;
  else if(layer?.id==='threat')h=`<div class="t" style="color:#ff8844">🎯 THREAT</div>${kv('Score',`${(d.score||0).toFixed(1)} / 100`)}`;
  else if(layer?.id==='hex')h=`<div class="t">⬡ HEX</div>${kv('Flights',d.points?.length||0)}`;
  else{tt.style.display='none';return;}
  tt.innerHTML=h;tt.style.display='block';tt.style.left=(x+12)+'px';tt.style.top=(y+12)+'px';}
function buildLayers(d){if(!d)return[];const L=[];const fl=d.flights||[],eq=d.quakes||[],sh=d.ships||[],sl=d.starlink||[],fi=d.fires||[],th=d.threat||[];
  if(lv.flights&&fl.length)L.push(new ScatterplotLayer({id:'flights',data:fl,getPosition:d=>[d.lon,d.lat,d.alt||0],getColor:d=>{const t=Math.min((d.alt||0)/13000,1);return[0,Math.round(55+t*200),Math.round(30+t*195),220];},getRadius:18000,radiusMinPixels:3,radiusMaxPixels:8,pickable:true,onHover:showTT}));
  if(lv.hex&&fl.length)L.push(new HexagonLayer({id:'hex',data:fl,getPosition:d=>[d.lon,d.lat],radius:200000,elevationScale:5000,extruded:true,pickable:true,onHover:showTT,colorRange:[[0,80,30,80],[0,150,80,110],[0,200,130,140],[0,255,180,170],[50,255,220,200],[100,255,255,230]],opacity:0.55,coverage:0.88}));
  if(lv.quakes&&eq.length)L.push(new ScatterplotLayer({id:'quakes',data:eq,getPosition:d=>[d.lon,d.lat,0],getColor:d=>{const t=Math.min((d.mag-2.5)/5.5,1);return[Math.round(80+t*175),20,0,200];},getRadius:d=>Math.pow(10,(d.mag||2.5)-1)*8000,radiusMinPixels:4,radiusMaxPixels:22,pickable:true,onHover:showTT}));
  if(lv.ships&&sh.length)L.push(new ScatterplotLayer({id:'ships',data:sh,getPosition:d=>[d.lon,d.lat,0],getColor:()=>[0,200,255,180],getRadius:25000,radiusMinPixels:3,radiusMaxPixels:7,pickable:true,onHover:showTT}));
  if(lv.starlink&&sl.length)L.push(new ScatterplotLayer({id:'starlink',data:sl,getPosition:d=>[d.lon,d.lat,(d.alt_km||550)*1000],getColor:()=>[170,170,255,160],getRadius:30000,radiusMinPixels:2,radiusMaxPixels:5,pickable:true,onHover:showTT}));
  if(lv.fires&&fi.length)L.push(new ScatterplotLayer({id:'fires',data:fi,getPosition:d=>[d.lon,d.lat,0],getColor:d=>{const t=Math.min((d.brightness-300)/200,1);return[Math.round(200+t*55),Math.round(50+t*75),0,185];},getRadius:20000,radiusMinPixels:2,radiusMaxPixels:6,pickable:true,onHover:showTT}));
  if(lv.threat&&th.length)L.push(new ScatterplotLayer({id:'threat',data:th,getPosition:d=>[d.lon_bin,d.lat_bin,0],getColor:d=>{const t=(d.score||0)/100;return[Math.round(40+t*215),Math.round(150-t*130),25,Math.round(80+t*120)];},getRadius:450000,radiusMinPixels:10,radiusMaxPixels:55,pickable:true,onHover:showTT}));
  return L;}
function tl(n){lv[n]=!lv[n];const m={flights:'b-fl',hex:'b-hx',quakes:'b-eq',ships:'b-sh',starlink:'b-sl',threat:'b-th',fires:'b-fi'};document.getElementById(m[n])?.classList.toggle('on',lv[n]);if(snap)dgl.setProps({layers:buildLayers(snap)});}
function us(s){document.getElementById('n-fl').textContent=s.counts?.flights??'—';document.getElementById('n-eq').textContent=s.counts?.quakes??'—';document.getElementById('n-th').textContent=s.counts?.threat??'—';document.getElementById('n-sh').textContent=s.counts?.ships??'—';document.getElementById('n-sl').textContent=s.counts?.starlink??'—';}
function uc(){const n=new Date(),p=x=>String(x).padStart(2,'0');document.getElementById('clk').textContent=`UTC ${n.getUTCFullYear()}-${p(n.getUTCMonth()+1)}-${p(n.getUTCDate())} ${p(n.getUTCHours())}:${p(n.getUTCMinutes())}:${p(n.getUTCSeconds())}`;}
setInterval(uc,1000);uc();
function connectSSE(){const st=document.getElementById('st');st.textContent='SSE ◌…';
const es=new EventSource('/api/stream');
es.onopen=()=>{st.textContent='SSE ● LIVE';};
es.onmessage=e=>{try{snap=JSON.parse(e.data);dgl.setProps({layers:buildLayers(snap)});us(snap);}catch(err){console.warn(err);}};
es.onerror=()=>{st.textContent='SSE ✗ retry…';es.close();setTimeout(connectSSE,5000);};}
connectSSE();
</script></body></html>
"""

# ─────────────────────────────────────────────────────────────
# DASH APP + PRODUCTION WSGI EXPORT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title=f"GSF v{VERSION}",
    update_title=None,
)
application = app.server  # Gunicorn entry point: gunicorn -w 1 gsf_v10:application

# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────
def _cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

@app.server.route('/healthz')
def route_health():
    s=_ring.stats()
    return _cors(jsonify({'status':'ok','version':VERSION,'buffer_frames':s['count'],'buffer_fill_pct':s['fill_pct'],'ts':datetime.utcnow().isoformat()+'Z'}))

@app.server.route('/metrics')
def route_metrics():
    s=_ring.stats(); iss,_=_cache.get_iss(); alerts=_alerts.get_all(); zones=_geofence.get_zones()
    lines=[
        f'gsf_flights_total {len(_cache.get_flights())}',
        f'gsf_earthquakes_total {len(_cache.get_quakes())}',
        f'gsf_ships_total {len(_cache.get_ships())}',
        f'gsf_alerts_total {len(alerts)}',
        f'gsf_buffer_frames {s["count"]}',
        f'gsf_buffer_fill_pct {s["fill_pct"]}',
        f'gsf_geofence_zones {len(zones)}',
        f'gsf_iss_alt_km {iss.get("alt_km",0) if iss else 0}',
    ]
    return Response('\n'.join(lines)+'\n',mimetype='text/plain')

@app.server.route('/api/v1/')
def route_api_index():
    base=flask_request.host_url.rstrip('/')
    return _cors(jsonify({'version':VERSION,'endpoints':{
        'snapshot':    f'{base}/api/v1/snapshot',
        'alerts':      f'{base}/api/v1/alerts',
        'flights':     f'{base}/api/v1/flights',
        'earthquakes': f'{base}/api/v1/earthquakes',
        'ships':       f'{base}/api/v1/ships',
        'threat':      f'{base}/api/v1/threat',
        'geofences':   f'{base}/api/v1/geofences',
        'anomaly':     f'{base}/api/v1/anomaly',
        'iss_position':f'{base}/api/v1/iss/position',
        'iss_passes':  f'{base}/api/v1/iss/passes?lat=48.85&lon=2.35&n=5',
        'events':      f'{base}/api/v1/events?lat=48.85&lon=2.35&radius_km=25',
        'frames_list': f'{base}/api/frames/list',
        'frame':       f'{base}/api/frame/{{idx}}',
        'export':      f'{base}/api/export.geojson',
        'stream':      f'{base}/api/stream',
        'health':      f'{base}/healthz',
        'metrics':     f'{base}/metrics',
    }}))

@app.server.route('/api/v1/events')
def route_v1_events():
    """
    Nearby event intelligence endpoint.
    Query params: lat, lon, radius_km (default 25), source (all|ticketmaster|osm|gdelt)
    """
    try:
        lat       = float(flask_request.args.get('lat', 0))
        lon       = float(flask_request.args.get('lon', 0))
        radius_km = int(flask_request.args.get('radius_km', EVENTS_RADIUS_KM))
        source    = flask_request.args.get('source', 'all')

        result = {'lat': lat, 'lon': lon, 'radius_km': radius_km,
                  'ts': datetime.utcnow().isoformat()+'Z'}

        if source in ('all', 'ticketmaster'):
            result['ticketmaster_events'] = fetch_ticketmaster_events(lat, lon, radius_km)

        if source in ('all', 'osm'):
            result['osm_venues'] = fetch_overpass_events(lat, lon, radius_km * 1000)

        if source in ('all', 'gdelt'):
            df_ev = _cache.get_events()
            result['gdelt_news_context'] = nearby_gdelt_context(lat, lon, radius_km, df_ev)

        return _cors(jsonify(result))
    except Exception as e:
        return _cors(jsonify({'error': str(e)})), 400

@app.server.route('/api/v1/snapshot')
def route_v1_snap():
    snap=_cache.to_snapshot(); snap['alerts']=_alerts.get_all(); return _cors(jsonify(snap))

@app.server.route('/api/v1/alerts')
def route_v1_alerts(): return _cors(jsonify(_alerts.get_all()))

@app.server.route('/api/v1/flights')
def route_v1_flights():
    df=_cache.get_flights(); return _cors(jsonify(df.to_dict('records') if not df.empty else []))

@app.server.route('/api/v1/earthquakes')
def route_v1_quakes():
    df=_cache.get_quakes(); return _cors(jsonify(df.to_dict('records') if not df.empty else []))

@app.server.route('/api/v1/ships')
def route_v1_ships(): return _cors(jsonify(_cache.get_ships()))

@app.server.route('/api/v1/threat')
def route_v1_threat():
    df=_cache.get_threat()
    if df.empty: return _cors(jsonify([]))
    cols=['lat_bin','lon_bin','score','n_seismic','n_airqual','n_conflict','n_fire']
    return _cors(jsonify(df[[c for c in cols if c in df.columns]].to_dict('records')))

@app.server.route('/api/v1/anomaly')
def route_v1_anomaly(): return _cors(jsonify(_cache.get_anomaly_status()))

@app.server.route('/api/v1/iss/position')
def route_v1_iss():
    iss,trail=_cache.get_iss(); return _cors(jsonify({'position':iss,'trail':trail,'ts':datetime.utcnow().isoformat()+'Z'}))

@app.server.route('/api/v1/iss/passes')
def route_v1_iss_passes():
    try:
        lat=float(flask_request.args.get('lat',0)); lon=float(flask_request.args.get('lon',0))
        n=int(flask_request.args.get('n',5)); elev=float(flask_request.args.get('min_elev',10.0))
        passes=predict_iss_passes(lat,lon,n_passes=min(n,10),min_elev=elev)
        return _cors(jsonify({'observer':{'lat':lat,'lon':lon},'passes':passes,'sgp4':HAS_SGP4,'computed':datetime.utcnow().isoformat()+'Z'}))
    except Exception as e: return _cors(jsonify({'error':str(e)})),400

@app.server.route('/api/v1/geofences', methods=['GET'])
def route_v1_gf_get(): return _cors(jsonify(_geofence.get_zones()))

@app.server.route('/api/v1/geofences', methods=['POST'])
def route_v1_gf_post():
    try:
        d=flask_request.get_json(force=True)
        name=_geofence.add_zone(d['name'],float(d['lat_min']),float(d['lat_max']),float(d['lon_min']),float(d['lon_max']))
        return _cors(jsonify({'created':name,'zones':_geofence.get_zones()})),201
    except Exception as e: return _cors(jsonify({'error':str(e)})),400

@app.server.route('/api/v1/geofences/<name>', methods=['DELETE'])
def route_v1_gf_delete(name): _geofence.remove_zone(name); return _cors(jsonify({'deleted':name,'zones':_geofence.get_zones()}))

@app.server.route('/api/stream')
def route_sse():
    def generate():
        q=_sse_bus.subscribe()
        try:
            while True:
                try: data=q.get(timeout=30); yield f"data: {data}\n\n"
                except _queue.Empty: yield ": heartbeat\n\n"
        except GeneratorExit: pass
        finally: _sse_bus.unsubscribe(q)
    return Response(stream_with_context(generate()),mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no','Connection':'keep-alive'})

@app.server.route('/api/snapshot')
def route_snap(): snap=_cache.to_snapshot(); snap['alerts']=_alerts.get_all(); return jsonify(snap)

@app.server.route('/api/alerts')
def route_alerts(): return jsonify(_alerts.get_all())

@app.server.route('/api/export.geojson')
def route_geojson():
    gj=_cache.to_geojson(); resp=Response(json_lib.dumps(gj,indent=2,default=str),mimetype='application/geo+json')
    resp.headers['Content-Disposition']=f'attachment; filename="gsf_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.geojson"'
    return _cors(resp)

@app.server.route('/api/weather')
def route_weather():
    try: return jsonify(fetch_weather_point(float(flask_request.args.get('lat',0)),float(flask_request.args.get('lon',0))))
    except Exception as e: return jsonify({'error':str(e)}),400

@app.server.route('/api/trail/<icao>')
def route_trail(icao): return jsonify({'icao':icao.upper(),'trail':_cache.get_trail(icao.upper())})

@app.server.route('/api/frames/list')
def route_frames_list(): return jsonify(_ring.list_frames())

@app.server.route('/api/frame/<int:idx>')
def route_frame(idx):
    snap=_ring.get_frame(idx)
    if snap is None: return jsonify({'error':f'frame {idx} not found'}),404
    return jsonify(snap)

@app.server.route('/api/buffer/stats')
def route_buf_stats(): return jsonify(_ring.stats())

@app.server.route('/deck')
def route_deck(): return Response(DECK_HTML.replace('__VERSION__',VERSION),mimetype='text/html')

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
CSS = """
body{background:#040c06!important;font-family:'Courier New',monospace}
.card{background:#07100a!important;border:1px solid #0b2e18!important}
.card-header{background:#091508!important;color:#00cc66!important;font-size:.7rem;
             letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid #0b2e18!important;padding:5px 12px!important}
.stat-box{background:#060e08;border:1px solid #0b2e18;border-radius:3px;padding:4px 5px;text-align:center;margin-bottom:5px}
.stat-value{color:#00ff88;font-size:.88rem;font-weight:bold;line-height:1.2}
.stat-value.sl-val{color:#aaaaff}.stat-value.iss-val{color:#ffee44}.stat-value.al-val{color:#ff6666}
.stat-value.gf-val{color:#88ff88}.stat-value.rec-val{color:#00ffaa}
.stat-label{color:#2a5535;font-size:.5rem;letter-spacing:.08em}
.scanline{background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,100,.012) 2px,rgba(0,255,100,.012) 4px);
  pointer-events:none;position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999}
.log-row{border-bottom:1px solid #091408;padding:2px 0;line-height:1.4}
.layer-check .form-check-label{color:#337744!important;font-size:.62rem}
.layer-check .form-check-input:checked{background-color:#00cc55;border-color:#00cc55}
.inspect-panel{border:1px solid #0b3318;border-radius:4px;background:#050e07;padding:8px;font-size:.69rem;color:#88ccaa;min-height:50px}
.inspect-title{color:#00ff88;font-size:.72rem;font-weight:bold;margin-bottom:4px}
.wx-box{background:#040e06;border:1px solid #0b2e18;border-radius:3px;padding:5px 8px;font-size:.62rem;color:#88ccaa;margin-top:3px}
.wx-title{color:#00cc66;font-size:.62rem;margin-bottom:2px}
.track-box{background:#040c06;border:1px solid #0b2e18;border-radius:3px;padding:4px 7px;margin-top:3px}
.playback-panel{background:#040e06;border:1px solid #1a3318;border-radius:4px;padding:7px 9px}
.pb-btn{font-family:'Courier New',monospace;background:rgba(0,20,10,.85);border:1px solid #0d3320;color:#00cc55;font-size:.68rem;padding:4px 8px;border-radius:3px;cursor:pointer;transition:all .12s}
.pb-btn:hover{border-color:#00cc55;color:#00ff88}
.pb-btn.live-active{border-color:#00ff44;color:#00ff44;background:rgba(0,30,10,.9)}
.alert-row{padding:3px 8px;border-bottom:1px solid #111100;font-size:.62rem;display:flex;align-items:flex-start;gap:5px}
.alert-row.critical{border-left:3px solid #ff3333;background:rgba(50,5,5,.5)}
.alert-row.warning {border-left:3px solid #ffaa00;background:rgba(40,20,0,.5)}
.alert-row.info    {border-left:3px solid #4488ff;background:rgba(5,10,30,.5)}
.alert-msg{color:#ccddaa;font-size:.61rem}.alert-ts{color:#334433;font-size:.55rem}
.cam-cell{border:1px solid #0b2e18;border-radius:3px;background:#020804}
.anom-block{padding:4px 6px;margin:2px 0;border-radius:3px;font-size:.63rem}
.anom-block.ok{background:rgba(0,30,10,.5);color:#336644}
.anom-block.anom{background:rgba(40,15,0,.8);color:#ff8844;border:1px solid #884400}
.gf-row{border-bottom:1px solid #0b2e18;padding:2px 0;font-size:.62rem;display:flex;align-items:center;gap:5px}
.blink{animation:blink 1.1s step-end infinite}@keyframes blink{50%{opacity:0}}
/* v1.1 additions */
.bm-btn{font-family:'Courier New',monospace;background:rgba(0,18,8,.88);border:1px solid #0d3320;
        color:#224433;font-size:.6rem;padding:2px 8px;border-radius:3px;cursor:pointer;transition:all .12s;
        margin-right:3px}
.bm-btn:hover{border-color:#005522;color:#009944}
.bm-btn.bm-active{border-color:#00cc66;color:#00ff88;background:rgba(0,30,15,.9)}
.draw-btn{font-family:'Courier New',monospace;background:rgba(0,18,8,.88);border:1px solid #0d3320;
          color:#336644;font-size:.62rem;padding:3px 8px;border-radius:3px;cursor:pointer;transition:all .12s;width:100%}
.draw-btn:hover{border-color:#005522;color:#009944}
.draw-btn.draw-active{border-color:#ffcc00;color:#ffee44;background:rgba(30,20,0,.9);animation:draw-pulse 1s step-end infinite}
@keyframes draw-pulse{50%{border-color:#886600;color:#ccaa00}}
.search-result-label{color:#00ff88;font-size:.62rem;padding:2px 4px;
                     background:rgba(0,30,10,.85);border-radius:2px;margin-top:2px;display:inline-block}
.poi-badge{display:inline-flex;align-items:center;gap:3px;background:rgba(0,20,40,.85);
           border:1px solid #0d3355;border-radius:3px;padding:2px 6px;
           font-size:.6rem;color:#88aaff;margin:1px}
/* v1.2 Event Intelligence */
.ev-card{border-bottom:1px solid #1a0030;padding:5px 0;font-size:.63rem}
.ev-card:last-child{border-bottom:none}
.ev-name{color:#cc88ff;font-weight:bold;font-size:.65rem;line-height:1.3}
.ev-meta{color:#664488;font-size:.59rem;margin-top:1px}
.ev-date{color:#88aaff;font-size:.61rem;font-weight:bold;min-width:70px;flex-shrink:0}
.ev-dist{color:#336644;font-size:.58rem;white-space:nowrap}
.ev-badge{display:inline-block;padding:1px 5px;border-radius:2px;font-size:.58rem;
          font-weight:bold;margin-left:4px}
.ev-badge.onsale{background:rgba(0,180,50,.2);color:#44dd66;border:1px solid #226633}
.ev-badge.cancelled{background:rgba(180,0,0,.2);color:#ff4444;border:1px solid #662222}
.ev-badge.postponed{background:rgba(180,100,0,.2);color:#ffaa44;border:1px solid #664422}
.ev-badge.sold-out{background:rgba(80,0,80,.2);color:#cc44ff;border:1px solid #442266}
.ev-badge.venue{background:rgba(0,40,80,.3);color:#4488ff;border:1px solid #224466}
.ev-price{color:#aaff88;font-size:.6rem;margin-left:4px}
.ev-link{color:#5d3d8a;font-size:.58rem;text-decoration:none}
.ev-link:hover{color:#cc88ff}
.ev-news-item{border-bottom:1px solid #1a0030;padding:4px 0;font-size:.62rem}
.ev-news-title{color:#ccaaff;line-height:1.3}
.ev-news-meta{color:#443366;font-size:.58rem;margin-top:1px}
.ev-section-hdr{color:#5d2d8a;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;
                padding:3px 0;border-bottom:1px solid #1a0030;margin-bottom:4px}
.ev-empty{color:#332244;font-size:.65rem;padding:12px 6px;text-align:center}
.ev-tm-note{color:#442244;font-size:.58rem;padding:3px 0}
"""

LAYER_OPTIONS=[
    {'label':' ✈','value':'flights'},  {'label':' ⚡','value':'quakes'},
    {'label':' 🔥','value':'fires'},   {'label':' 🚢','value':'ships'},
    {'label':' 🛰','value':'starlink'},{'label':' 🛸','value':'iss'},
    {'label':' 💨','value':'aq'},      {'label':' 📰','value':'events'},
    {'label':' 🎯','value':'threat'},  {'label':' 🔲','value':'geofence'},
    {'label':' 🛤','value':'trails'},  {'label':' 🏙','value':'poi'},
    {'label':' 🌧','value':'radar'},
]

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
app.layout = html.Div([
    html.Div(className='scanline'),
    html.Style(CSS),
    dbc.Container(fluid=True, children=[

        # ── Header ─────────────────────────────────────────
        dbc.Row([
            dbc.Col([html.Div([
                html.Span("◈ ",style={'color':'#00ff88'}),
                html.Span("GEO-SURVEILLANCE FEED",style={'color':'#00cc66','fontSize':'1.0rem','letterSpacing':'.15em','fontWeight':'bold'}),
                html.Span(f"  v{VERSION}  ·  ADVANCED  ·  2D NATIVE",style={'color':'#2a5535','fontSize':'.65rem','marginLeft':'8px'}),
            ],style={'padding':'8px 0 4px'})],width=5),
            dbc.Col([html.Div([
                html.Span("ONLINE ",style={'color':'#00ff88','fontSize':'.65rem'}),
                html.Span("■ ",className='blink',style={'color':'#00ff44','fontSize':'.65rem'}),
                html.Br(),
                html.Span(id='clock',style={'color':'#1e4428','fontSize':'.59rem'}),
            ],style={'textAlign':'right','padding':'7px 0'})],width=2),
            dbc.Col([
                dbc.Row([
                    dbc.Col([html.A("⬡ WebGL",    href='/deck',           target='_blank',style={'display':'block','marginTop':'9px','background':'rgba(0,15,30,.9)','border':'1px solid #0d3355','color':'#88aaff','fontSize':'.6rem','padding':'4px 6px','borderRadius':'3px','textDecoration':'none','textAlign':'center'})],width=3),
                    dbc.Col([html.A("📋 API v1",  href='/api/v1/',        target='_blank',style={'display':'block','marginTop':'9px','background':'rgba(10,10,0,.9)','border':'1px solid #3d3500','color':'#ccaa44','fontSize':'.6rem','padding':'4px 6px','borderRadius':'3px','textDecoration':'none','textAlign':'center'})],width=3),
                    dbc.Col([html.A("💾 GeoJSON", href='/api/export.geojson',target='_blank',style={'display':'block','marginTop':'9px','background':'rgba(0,10,30,.9)','border':'1px solid #0d3355','color':'#44aaff','fontSize':'.6rem','padding':'4px 6px','borderRadius':'3px','textDecoration':'none','textAlign':'center'})],width=3),
                    dbc.Col([html.A("❤ Health",  href='/healthz',         target='_blank',style={'display':'block','marginTop':'9px','background':'rgba(0,20,5,.9)','border':'1px solid #1a4422','color':'#44cc88','fontSize':'.6rem','padding':'4px 6px','borderRadius':'3px','textDecoration':'none','textAlign':'center'})],width=3),
                ],className='g-1'),
            ],width=5),
        ]),

        # ── Main 3-column ───────────────────────────────────
        dbc.Row([

            # LEFT — Maps (6 col)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                dbc.Tabs([
                                    dbc.Tab(label="🗺 TILE MAP",   tab_id='t-map',
                                            label_style={'color':'#005522','fontSize':'.66rem'},
                                            active_label_style={'color':'#00ffee'}),
                                    dbc.Tab(label="🌍 GLOBE",     tab_id='t-globe',
                                            label_style={'color':'#005522','fontSize':'.66rem'},
                                            active_label_style={'color':'#00ff88'}),
                                    dbc.Tab(label="🌡 DENSITY",   tab_id='t-density',
                                            label_style={'color':'#005522','fontSize':'.66rem'},
                                            active_label_style={'color':'#ff8844'}),
                                ],id='view-tabs',active_tab='t-map',
                                   style={'background':'transparent','borderBottom':'none'}),
                            ],width=5),
                            dbc.Col([
                                dbc.Checklist(id='layer-toggles',options=LAYER_OPTIONS,
                                    value=['flights','quakes','fires','ships','starlink','iss','aq','events','threat','geofence','trails','radar'],
                                    inline=True,className='layer-check',
                                    style={'marginTop':'4px','fontSize':'.58rem'}),
                            ],width=7),
                        ],align='center'),
                        # ── Search + Basemap row ────────────────────────
                        dbc.Row([
                            dbc.Col([
                                dbc.InputGroup([
                                    dbc.Input(id='search-input',
                                              placeholder="Search location… (city, address, landmark)",
                                              type='text', debounce=False,
                                              style={'background':'#040e06','border':'1px solid #0b2e18',
                                                     'color':'#a0ffcc','fontSize':'.65rem',
                                                     'fontFamily':'monospace','height':'28px'}),
                                    dbc.Button("⌕", id='search-btn', n_clicks=0,
                                               style={'background':'rgba(0,30,15,.9)','border':'1px solid #0d5530',
                                                      'color':'#00cc66','fontSize':'.7rem','padding':'0 10px',
                                                      'height':'28px'}),
                                ], size='sm'),
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    html.Span("BASE: ", style={'color':'#336644','fontSize':'.6rem','marginRight':'4px'}),
                                    html.Button("DARK",   id='bm-dark',   n_clicks=0, className='bm-btn bm-active'),
                                    html.Button("LIGHT",  id='bm-light',  n_clicks=0, className='bm-btn'),
                                    html.Button("STREET", id='bm-street', n_clicks=0, className='bm-btn'),
                                    html.Span("  🌧", id='radar-badge',
                                              style={'color':'#224433','fontSize':'.65rem','marginLeft':'6px',
                                                     'cursor':'default'}),
                                ], style={'display':'flex','alignItems':'center','height':'28px'}),
                            ], width=6),
                        ], className='mt-1 mb-0 g-1',
                           style={'padding':'0 4px 4px 4px'}),
                    ]),
                    dbc.CardBody([
                        # Primary: tile map (Scattermapbox)
                        html.Div(id='view-map',children=[
                            dcc.Graph(id='map-fig',style={'height':'46vh'},
                                      config={'displaylogo':False,'scrollZoom':True,
                                              'modeBarButtonsToRemove':['select2d','lasso2d']}),
                        ]),
                        # Secondary: globe (Scattergeo)
                        html.Div(id='view-globe',style={'display':'none'},children=[
                            dcc.Graph(id='globe-fig',style={'height':'46vh'},
                                      config={'displaylogo':False,'scrollZoom':True,
                                              'modeBarButtonsToRemove':['select2d','lasso2d']}),
                        ]),
                        # Density heatmap
                        html.Div(id='view-density',style={'display':'none'},children=[
                            dcc.Graph(id='density-fig',style={'height':'46vh'},
                                      config={'displaylogo':False,'scrollZoom':True,
                                              'modeBarButtonsToRemove':['select2d','lasso2d']}),
                        ]),
                        dcc.Interval(id='tick-ui',    interval=UI_REFRESH_MS,   n_intervals=0),
                        dcc.Interval(id='tick-alerts',interval=ALERT_REFRESH_MS,n_intervals=0),
                        dcc.Interval(id='tick-pb',    interval=PB_REFRESH_MS,   n_intervals=0),
                        dcc.Interval(id='tick-rec',   interval=5000,            n_intervals=0),
                    ],style={'padding':'4px','backgroundColor':'#040c06'}),
                ]),

                # EQ slider
                html.Div(id='ctrl-wrap',children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("⏱ EQ HISTORY (days)"),
                                dbc.CardBody([
                                    dcc.Slider(id='eq-days-slider',min=0,max=30,step=1,value=1,
                                               marks={0:{'label':'NOW','style':{'color':'#ff6633','fontSize':'.55rem'}},
                                                      30:{'label':'30d','style':{'color':'#336644','fontSize':'.55rem'}}},
                                               tooltip={'placement':'top','always_visible':True}),
                                    html.Div(id='eq-days-label',style={'color':'#ff6633','fontSize':'.6rem','marginTop':'2px','textAlign':'center'}),
                                ],style={'padding':'7px 13px 3px','backgroundColor':'#040c06'}),
                            ]),
                        ],width=6),
                        dbc.Col([
                            # Playback controls
                            dbc.Card([
                                dbc.CardHeader("⏯ PLAYBACK  ·  RING BUFFER"),
                                dbc.CardBody([
                                    html.Div(className='playback-panel',children=[
                                        dbc.Row([
                                            dbc.Col([html.Div(id='pb-status-text',style={'color':'#336644','fontSize':'.6rem'})],width=8),
                                            dbc.Col([html.Div(id='pb-mode-badge',style={'textAlign':'right','fontSize':'.6rem'})],width=4),
                                        ],className='mb-1'),
                                        dcc.Slider(id='pb-slider',min=0,max=239,step=1,value=239,marks={},
                                                   tooltip={'placement':'bottom','always_visible':False},updatemode='drag'),
                                        html.Div(id='pb-time-label',style={'color':'#88ccaa','fontSize':'.58rem','textAlign':'center','marginTop':'2px'}),
                                        dbc.Row([
                                            dbc.Col([html.Button("⬤ LIVE",id='pb-live-btn',n_clicks=0,className='pb-btn live-active',style={'width':'100%','fontSize':'.62rem'})],width=3),
                                            dbc.Col([html.Button("◀◀",    id='pb-back-btn',n_clicks=0,className='pb-btn',style={'width':'100%','fontSize':'.62rem'})],width=3),
                                            dbc.Col([html.Button("▶ PLAY",id='pb-play-btn',n_clicks=0,className='pb-btn',style={'width':'100%','fontSize':'.62rem'})],width=3),
                                            dbc.Col([html.Button("▶▶",    id='pb-fwd-btn', n_clicks=0,className='pb-btn',style={'width':'100%','fontSize':'.62rem'})],width=3),
                                        ],className='mt-1 g-1'),
                                    ]),
                                ],style={'padding':'6px','backgroundColor':'#040c06'}),
                            ]),
                        ],width=6),
                    ],className='mt-2 g-2'),
                ]),

                # Inspect
                html.Div(id='inspect-wrap',children=[
                    dbc.Card([
                        dbc.CardHeader("🔎 INSPECT  ·  🌡 WEATHER  ·  ✈ ALTITUDE TRAIL"),
                        dbc.CardBody([
                            html.Div(id='inspect-panel',className='inspect-panel',
                                     children=[html.Span("Click any marker on the map to inspect.",style={'color':'#224433','fontSize':'.68rem'})]),
                        ],style={'padding':'7px','backgroundColor':'#040c06'}),
                    ],className='mt-2'),
                ]),
            ],width=6),

            # MIDDLE — Stats + cameras + alerts + log (3 col)
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div([html.Div("—",id='s-fl',className='stat-value'),html.Div("AC",className='stat-label')],className='stat-box'),width=4),
                    dbc.Col(html.Div([html.Div("—",id='s-eq',className='stat-value'),html.Div("EQ",className='stat-label')],className='stat-box'),width=4),
                    dbc.Col(html.Div([html.Div("—",id='s-sh',className='stat-value'),html.Div("SHIPS",className='stat-label')],className='stat-box'),width=4),
                ],className='mb-1'),
                dbc.Row([
                    dbc.Col(html.Div([html.Div("—",id='s-sl',className='stat-value sl-val'),html.Div("STARLINK",className='stat-label')],className='stat-box'),width=3),
                    dbc.Col(html.Div([html.Div("0",id='s-al',className='stat-value al-val'),html.Div("ALERTS",className='stat-label')],className='stat-box'),width=3),
                    dbc.Col(html.Div([html.Div("—",id='s-gf',className='stat-value gf-val'),html.Div("GF ZONES",className='stat-label')],className='stat-box'),width=3),
                    dbc.Col(html.Div([html.Div("—",id='s-rec',className='stat-value rec-val'),html.Div("BUF",className='stat-label')],className='stat-box'),width=3),
                ],className='mb-2'),

                # Camera
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Tabs([
                            dbc.Tab(label="📹 SINGLE", tab_id='cam-single',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#00ff88'}),
                            dbc.Tab(label="⊞ GRID",    tab_id='cam-grid',  label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#00cc66'}),
                        ],id='cam-tabs',active_tab='cam-single',style={'background':'transparent','borderBottom':'none'}),
                    ]),
                    dbc.CardBody([
                        html.Div(id='cam-single-view',children=[html.Img(id='cam-img-0',style={'width':'100%','borderRadius':'2px'})]),
                        html.Div(id='cam-grid-view',style={'display':'none'},children=[
                            dbc.Row([dbc.Col([html.Img(id='cam-img-1',style={'width':'100%'})],width=6,className='cam-cell p-1'),dbc.Col([html.Img(id='cam-img-2',style={'width':'100%'})],width=6,className='cam-cell p-1')],className='g-0 mb-1'),
                            dbc.Row([dbc.Col([html.Img(id='cam-img-3',style={'width':'100%'})],width=6,className='cam-cell p-1'),dbc.Col([html.Img(id='cam-img-4',style={'width':'100%'})],width=6,className='cam-cell p-1')],className='g-0'),
                        ]),
                        dcc.Interval(id='tick-cam',interval=CAM_REFRESH_MS,n_intervals=0),
                        dbc.Row([
                            dbc.Col([dbc.Input(id='cam-url-input',placeholder="MJPEG URL…",type='text',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem','fontFamily':'monospace','marginTop':'4px'})],width=7),
                            dbc.Col([dbc.Select(id='cam-slot-select',options=[{'label':f'S{i}','value':str(i)} for i in range(4)],value='0',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem','fontFamily':'monospace','marginTop':'4px'})],width=2),
                            dbc.Col([dbc.Button("SET",id='cam-url-btn',n_clicks=0,style={'marginTop':'4px','fontSize':'.59rem','background':'rgba(0,30,15,.9)','border':'1px solid #0d5530','color':'#00cc66','padding':'4px 6px','width':'100%'})],width=3),
                        ],className='g-1'),
                        html.Div(id='cam-status',style={'color':'#336644','fontSize':'.56rem','marginTop':'2px'}),
                    ],style={'padding':'5px','backgroundColor':'#030807'}),
                ],className='mb-2'),

                dbc.Card([
                    dbc.CardHeader([html.Span("🔔 ALERTS "),html.Span(id='alert-badge',style={'color':'#ff6666','fontWeight':'bold'})]),
                    dbc.CardBody([html.Div(id='alert-panel',style={'height':'7vh','overflowY':'auto','fontSize':'.62rem'})],style={'padding':'0','backgroundColor':'#030807'}),
                ],className='mb-2'),

                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs([
                            dbc.Tab(label="✈", tab_id='l-fl',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#00ff88'}),
                            dbc.Tab(label="⚡",tab_id='l-eq',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#ff6633'}),
                            dbc.Tab(label="🛰",tab_id='l-sl',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#aaaaff'}),
                            dbc.Tab(label="🚢",tab_id='l-sh',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#00ccff'}),
                            dbc.Tab(label="📰",tab_id='l-ev',label_style={'color':'#005522','fontSize':'.62rem'},active_label_style={'color':'#ffaacc'}),
                        ],id='log-tabs',active_tab='l-fl',style={'background':'transparent','borderBottom':'none'}),
                    ),
                    dbc.CardBody([html.Div(id='log-panel',style={'height':'10vh','overflowY':'auto','fontSize':'.62rem','fontFamily':'monospace'})],style={'padding':'5px','backgroundColor':'#030807'}),
                ]),
            ],width=3),

            # RIGHT — Intelligence panels (3 col)
            dbc.Col([
                # Anomaly
                dbc.Card([
                    dbc.CardHeader("🤖 ANOMALY DETECTION  ·  Z-SCORE"),
                    dbc.CardBody([
                        html.Div(id='anomaly-panel',style={'fontSize':'.64rem'}),
                        dcc.Interval(id='tick-anom',interval=UI_REFRESH_MS,n_intervals=0),
                    ],style={'padding':'7px','backgroundColor':'#040c06'}),
                ],className='mb-2'),

                # Geofence
                dbc.Card([
                    dbc.CardHeader("🔲 GEOFENCE ZONES"),
                    dbc.CardBody([
                        html.Div(id='gf-zone-list',style={'maxHeight':'8vh','overflowY':'auto','fontSize':'.62rem','marginBottom':'5px'}),
                        # Draw mode row
                        dbc.Row([
                            dbc.Col([
                                html.Button("✏️ DRAW ON MAP", id='gf-draw-toggle', n_clicks=0,
                                            className='draw-btn',
                                            style={'width':'100%','fontSize':'.62rem','padding':'4px 6px'}),
                            ], width=6),
                            dbc.Col([
                                html.Div(id='gf-draw-status',
                                         style={'color':'#336644','fontSize':'.6rem',
                                                'display':'flex','alignItems':'center','height':'100%'}),
                            ], width=6),
                        ], className='g-1 mb-2'),
                        dbc.Input(id='gf-name',placeholder="Zone name (or type + click map to draw)…",type='text',
                            style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.6rem','fontFamily':'monospace','marginBottom':'4px'}),
                        dbc.Row([
                            dbc.Col([dbc.Input(id='gf-lat-min',placeholder="Lat min",type='number',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem'})],width=3),
                            dbc.Col([dbc.Input(id='gf-lat-max',placeholder="Lat max",type='number',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem'})],width=3),
                            dbc.Col([dbc.Input(id='gf-lon-min',placeholder="Lon min",type='number',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem'})],width=3),
                            dbc.Col([dbc.Input(id='gf-lon-max',placeholder="Lon max",type='number',style={'background':'#040e06','border':'1px solid #0b2e18','color':'#a0ffcc','fontSize':'.59rem'})],width=3),
                        ],className='g-1 mb-2'),
                        dbc.Button("➕ ADD ZONE",id='gf-add-btn',n_clicks=0,
                            style={'fontSize':'.62rem','background':'rgba(0,30,15,.9)','border':'1px solid #0d5530','color':'#00cc66','padding':'4px 8px','width':'100%'}),
                        html.Div(id='gf-status',style={'color':'#336644','fontSize':'.58rem','marginTop':'3px'}),
                        dcc.Interval(id='tick-gf',interval=UI_REFRESH_MS,n_intervals=0),
                    ],style={'padding':'8px','backgroundColor':'#040c06'}),
                ],className='mb-2'),

                # POI panel
                dbc.Card([
                    dbc.CardHeader("🏙 POINTS OF INTEREST  ·  OVERPASS / OSM"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id='poi-center-input',
                                          placeholder="City / address to load POIs…",
                                          type='text', debounce=False,
                                          style={'background':'#040e06','border':'1px solid #0b2e18',
                                                 'color':'#a0ffcc','fontSize':'.62rem','fontFamily':'monospace'}),
                            ], width=8),
                            dbc.Col([
                                dbc.Button("LOAD POI", id='poi-load-btn', n_clicks=0,
                                           style={'fontSize':'.62rem','background':'rgba(0,20,30,.9)',
                                                  'border':'1px solid #0d3355','color':'#44aaff',
                                                  'padding':'4px 8px','width':'100%'}),
                            ], width=4),
                        ], className='g-1 mb-1'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Select(id='poi-radius-select',
                                           options=[{'label':'500 m','value':'500'},
                                                    {'label':'1 km', 'value':'1000'},
                                                    {'label':'2 km', 'value':'2000'},
                                                    {'label':'5 km', 'value':'5000'}],
                                           value='2000',
                                           style={'background':'#040e06','border':'1px solid #0b2e18',
                                                  'color':'#a0ffcc','fontSize':'.61rem'}),
                            ], width=5),
                            dbc.Col([
                                html.Div(id='poi-status',
                                         style={'color':'#336644','fontSize':'.6rem',
                                                'display':'flex','alignItems':'center','height':'100%'}),
                            ], width=7),
                        ], className='g-1'),
                    ],style={'padding':'8px','backgroundColor':'#040c06'}),
                ],className='mb-2'),

                # ── EVENT INTELLIGENCE  ← v1.2 ───────────────────
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🎟 EVENT INTELLIGENCE  ·  NEARBY EVENTS & VENUES"),
                    ]),
                    dbc.CardBody([
                        # Search row
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(id='ev-search-input',
                                          placeholder="City / address…",
                                          type='text', debounce=False,
                                          style={'background':'#040e06','border':'1px solid #0b2e18',
                                                 'color':'#a0ffcc','fontSize':'.62rem','fontFamily':'monospace'}),
                            ], width=6),
                            dbc.Col([
                                dbc.Select(id='ev-radius-select',
                                           options=[{'label':'5 km','value':'5'},
                                                    {'label':'15 km','value':'15'},
                                                    {'label':'25 km','value':'25'},
                                                    {'label':'50 km','value':'50'},
                                                    {'label':'100 km','value':'100'}],
                                           value='25',
                                           style={'background':'#040e06','border':'1px solid #0b2e18',
                                                  'color':'#a0ffcc','fontSize':'.61rem'}),
                            ], width=3),
                            dbc.Col([
                                dbc.Button("SEARCH", id='ev-search-btn', n_clicks=0,
                                           style={'fontSize':'.62rem','background':'rgba(20,0,30,.9)',
                                                  'border':'1px solid #5d2d8a','color':'#cc88ff',
                                                  'padding':'4px 8px','width':'100%'}),
                            ], width=3),
                        ], className='g-1 mb-2'),

                        # Tab pills
                        dbc.Tabs([
                            dbc.Tab(label="🎫 LIVE EVENTS", tab_id='ev-tab-events',
                                    label_style={'color':'#5d2d8a','fontSize':'.63rem'},
                                    active_label_style={'color':'#cc88ff'}),
                            dbc.Tab(label="🏟 VENUES", tab_id='ev-tab-venues',
                                    label_style={'color':'#5d2d8a','fontSize':'.63rem'},
                                    active_label_style={'color':'#cc88ff'}),
                            dbc.Tab(label="📰 NEWS CONTEXT", tab_id='ev-tab-news',
                                    label_style={'color':'#5d2d8a','fontSize':'.63rem'},
                                    active_label_style={'color':'#ffaacc'}),
                        ], id='ev-tabs', active_tab='ev-tab-events',
                           style={'background':'transparent','borderBottom':'1px solid #1a0030','marginBottom':'6px'}),

                        # Status line
                        html.Div(id='ev-status',
                                 style={'color':'#5d2d8a','fontSize':'.6rem','marginBottom':'4px'}),

                        # Results panel
                        html.Div(id='ev-results-panel',
                                 style={'height':'22vh','overflowY':'auto',
                                        'fontSize':'.63rem','fontFamily':'monospace'}),

                        # Ticketmaster key notice
                        html.Div(id='ev-key-notice',
                                 style={'color':'#442244','fontSize':'.58rem','marginTop':'4px'}),
                    ], style={'padding':'8px','backgroundColor':'#040c06'}),
                ], className='mb-2'),

                # Telemetry charts
                dbc.Card([
                    dbc.CardHeader("📊 TELEMETRY  ·  2H ROLLING"),
                    dbc.CardBody([
                        dcc.Graph(id='telem-flights',style={'height':'7vh'},config={'displayModeBar':False}),
                        dcc.Graph(id='telem-quakes', style={'height':'7vh'},config={'displayModeBar':False}),
                        dcc.Graph(id='telem-mag',    style={'height':'7vh'},config={'displayModeBar':False}),
                        dcc.Interval(id='tick-telem',interval=UI_REFRESH_MS,n_intervals=0),
                    ],style={'padding':'4px','backgroundColor':'#040c06'}),
                ],className='mb-2'),

                # Threat leaderboard
                dbc.Card([
                    dbc.CardHeader("🎯 THREAT LEADERBOARD"),
                    dbc.CardBody([
                        html.Div(id='threat-panel',style={'height':'12vh','overflowY':'auto','fontSize':'.62rem','fontFamily':'monospace'}),
                        dcc.Interval(id='tick-threat',interval=UI_REFRESH_MS,n_intervals=0),
                    ],style={'padding':'6px','backgroundColor':'#040c06'}),
                ]),
            ],width=3),
        ]),

        dbc.Row([dbc.Col(html.Div(
            f"GSF v{VERSION}  ·  2D Tile Map  ·  Event Intelligence (Ticketmaster+OSM)  ·  "
            f"Anomaly Detection  ·  Geofencing  ·  Convergence  ·  ISS Passes  ·  "
            f"Config: gsf.ini  ·  API: /api/v1/  ·  /healthz  ·  /metrics",
            style={'color':'#183322','fontSize':'.5rem','padding':'4px 0','textAlign':'center'},
        ))]),
    ]),

    dcc.Store(id='store-snap'),
    dcc.Store(id='store-pb'),
    dcc.Store(id='store-basemap', data='carto-darkmatter'),
    dcc.Store(id='store-search',  data={'lat': 20.0, 'lon': 0.0, 'zoom': 1.5, 'label': ''}),
    dcc.Store(id='store-poi',     data=[]),
    dcc.Store(id='store-draw',    data={'active': False, 'clicks': []}),
    dcc.Store(id='store-events',  data={'events': [], 'venues': [], 'news': [], 'centre': {}}),
])

# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

@app.callback(Output('clock','children'), Input('tick-cam','n_intervals'))
def cb_clock(n): return datetime.utcnow().strftime("UTC  %Y-%m-%d  %H:%M:%S")

@app.callback(Output('cam-img-0','src'), Input('tick-cam','n_intervals'))
def cb_cam0(n):
    enc=cams.read_encoded(0); return f"data:image/jpeg;base64,{enc}" if enc else dash.no_update

@app.callback(Output('cam-img-1','src'),Output('cam-img-2','src'),Output('cam-img-3','src'),Output('cam-img-4','src'),
              Input('tick-cam','n_intervals'))
def cb_cam_grid(n):
    return [f"data:image/jpeg;base64,{cams.read_encoded(i)}" if cams.read_encoded(i) else dash.no_update for i in range(4)]

@app.callback(Output('cam-single-view','style'),Output('cam-grid-view','style'),Input('cam-tabs','active_tab'))
def cb_cam_tab(tab):
    show,hide={'display':'block'},{'display':'none'}
    return (hide,show) if tab=='cam-grid' else (show,hide)

@app.callback(Output('cam-status','children'),Input('cam-url-btn','n_clicks'),
              State('cam-url-input','value'),State('cam-slot-select','value'),prevent_initial_call=True)
def cb_cam_swap(n,url,slot):
    if not url or not url.strip(): return "⚠ Enter a URL first"
    try: cams.set_source(int(slot or 0),url.strip()); return f"✓ Slot {slot} → {url.strip()[:38]}…"
    except Exception as e: return f"✗ {e}"

@app.callback(
    Output('view-map','style'), Output('view-globe','style'), Output('view-density','style'),
    Output('ctrl-wrap','style'), Output('inspect-wrap','style'),
    Input('view-tabs','active_tab'),
)
def cb_view(tab):
    show,hide={'display':'block'},{'display':'none'}
    if tab=='t-globe':   return hide, show, hide, show, show
    if tab=='t-density': return hide, hide, show, hide, hide
    return show, hide, hide, show, show   # t-map default

@app.callback(Output('eq-days-label','children'), Input('eq-days-slider','value'))
def cb_eq_label(v):
    return "Live 24h feed" if v<=1 else f"Last {v} days (30d archive)"

# Main data refresh
@app.callback(
    Output('store-snap','data'),
    Output('s-fl','children'),Output('s-eq','children'),Output('s-sh','children'),
    Output('s-sl','children'),Output('s-al','children'),Output('s-rec','children'),
    Input('tick-ui','n_intervals'),
    State('store-pb','data'),
)
def cb_refresh(n, pb_state):
    pb = pb_state or {}
    is_live = pb.get('live', True)
    pos     = pb.get('pos', -1)
    if not is_live and _ring.stats()['count'] > 0:
        frames = _ring.list_frames()
        snap = _ring.get_frame(frames[pos]['idx']) if 0<=pos<len(frames) else (_ring.get_latest() or {})
    else:
        df_fl=_cache.get_flights(); df_eq=_cache.get_quakes(); df_fi=_cache.get_fires()
        ships=_cache.get_ships(); trails=_cache.get_trails(); iss,iss_trail=_cache.get_iss()
        df_aq=_cache.get_aq(); df_ev=_cache.get_events(); df_sl=_cache.get_starlink(); df_th=_cache.get_threat()
        snap={
            'flights':  df_fl.to_dict('records') if not df_fl.empty else [],
            'quakes':   df_eq.to_dict('records') if not df_eq.empty else [],
            'fires':    df_fi.to_dict('records') if not df_fi.empty else [],
            'ships':    ships,'trails':trails,'iss':iss,'iss_trail':iss_trail,
            'aq':       df_aq.to_dict('records') if not df_aq.empty else [],
            'events':   df_ev.to_dict('records') if not df_ev.empty else [],
            'starlink': df_sl.to_dict('records') if not df_sl.empty else [],
            'threat':   df_th.to_dict('records') if not df_th.empty else [],
            'geofences':_geofence.get_zones(),
        }
    alerts=_alerts.get_all(); buf_s=_ring.stats()
    n_fl=len(snap.get('flights',[])); n_eq=len(snap.get('quakes',[])); n_sh=len(snap.get('ships',[]))
    n_sl=len(snap.get('starlink',[]))
    return (snap, str(n_fl) if n_fl else '—', str(n_eq) if n_eq else '—', str(n_sh),
            str(n_sl) if n_sl else ('NO SGP4' if not HAS_SGP4 else '—'),
            str(len(alerts)), f"{buf_s['count']}/{buf_s['capacity']}")

# Alerts
@app.callback(Output('alert-badge','children'),Output('alert-panel','children'),Input('tick-alerts','n_intervals'))
def cb_alerts(n):
    alerts=_alerts.get_all(); count=len(alerts)
    if not alerts: children=[html.Div("No alerts.",style={'color':'#224433','padding':'6px','fontSize':'.62rem'})]
    else: children=[html.Div([html.Span(a['icon'],style={'fontSize':'.85rem','marginRight':'4px'}),
                              html.Div([html.Div(a['message'][:60],className='alert-msg'),html.Div(a['ts'],className='alert-ts')])],
                             className=f"alert-row {a['severity']}") for a in alerts[:12]]
    return (f"({count})" if count else ""), children

# ── BASEMAP SWITCHER ─────────────────────────────────────────
BASEMAP_STYLES = {
    'dark':   'carto-darkmatter',
    'light':  'carto-positron',
    'street': 'open-street-map',
}

@app.callback(
    Output('store-basemap','data'),
    Output('bm-dark','className'),
    Output('bm-light','className'),
    Output('bm-street','className'),
    Input('bm-dark','n_clicks'),
    Input('bm-light','n_clicks'),
    Input('bm-street','n_clicks'),
    State('store-basemap','data'),
    prevent_initial_call=False,
)
def cb_basemap(n_dark, n_light, n_street, current):
    triggered = ctx.triggered_id
    mapping = {
        'bm-dark':   'carto-darkmatter',
        'bm-light':  'carto-positron',
        'bm-street': 'open-street-map',
    }
    chosen = mapping.get(triggered, current or 'carto-darkmatter')
    def cls(key):
        return 'bm-btn bm-active' if mapping[key] == chosen else 'bm-btn'
    return chosen, cls('bm-dark'), cls('bm-light'), cls('bm-street')


# ── SEARCH / FLY-TO ──────────────────────────────────────────
@app.callback(
    Output('store-search','data'),
    Output('search-input','style'),
    Input('search-btn','n_clicks'),
    Input('search-input','n_submit'),
    State('search-input','value'),
    State('store-search','data'),
    prevent_initial_call=True,
)
def cb_search(n_click, n_submit, query, current):
    base_style = {'background':'#040e06','border':'1px solid #0b2e18',
                  'color':'#a0ffcc','fontSize':'.65rem','fontFamily':'monospace','height':'28px'}
    if not query or not query.strip():
        return current or {}, base_style
    result = geocode_location(query.strip())
    if result is None:
        err_style = dict(base_style, border='1px solid #882200', color='#ff8866')
        return current or {}, err_style
    lat, lon, label = result
    ok_style = dict(base_style, border='1px solid #00aa44')
    return {'lat': lat, 'lon': lon, 'zoom': 11, 'label': label}, ok_style


# ── GEOFENCE DRAW MODE ────────────────────────────────────────
@app.callback(
    Output('store-draw','data'),
    Output('gf-draw-toggle','className'),
    Output('gf-draw-status','children'),
    Output('gf-lat-min','value'),
    Output('gf-lat-max','value'),
    Output('gf-lon-min','value'),
    Output('gf-lon-max','value'),
    Input('gf-draw-toggle','n_clicks'),
    Input('map-fig','clickData'),
    State('store-draw','data'),
    State('gf-lat-min','value'),
    State('gf-lat-max','value'),
    State('gf-lon-min','value'),
    State('gf-lon-max','value'),
    prevent_initial_call=True,
)
def cb_gf_draw(n_toggle, map_click, draw_state, lat_min, lat_max, lon_min, lon_max):
    draw = draw_state or {'active': False, 'clicks': []}
    triggered = ctx.triggered_id

    if triggered == 'gf-draw-toggle':
        # Toggle draw mode on/off
        new_active = not draw.get('active', False)
        new_state = {'active': new_active, 'clicks': []}
        btn_cls = 'draw-btn draw-active' if new_active else 'draw-btn'
        status = "🟡 Click corner 1 on the map…" if new_active else ""
        return new_state, btn_cls, status, lat_min, lat_max, lon_min, lon_max

    if triggered == 'map-fig' and draw.get('active') and map_click:
        pt  = map_click['points'][0]
        lat = pt.get('lat'); lon = pt.get('lon')
        if lat is None or lon is None:
            return draw, 'draw-btn draw-active', "⚠ Click a map area (not a marker)", lat_min, lat_max, lon_min, lon_max

        clicks = list(draw.get('clicks', []))
        clicks.append({'lat': float(lat), 'lon': float(lon)})

        if len(clicks) == 1:
            new_state = {'active': True, 'clicks': clicks}
            status = f"🟠 Corner 1: ({lat:.3f}, {lon:.3f}) — click corner 2"
            return new_state, 'draw-btn draw-active', status, lat_min, lat_max, lon_min, lon_max
        else:
            # Two clicks — populate the form fields
            c1, c2 = clicks[0], clicks[1]
            new_lat_min = round(min(c1['lat'], c2['lat']), 4)
            new_lat_max = round(max(c1['lat'], c2['lat']), 4)
            new_lon_min = round(min(c1['lon'], c2['lon']), 4)
            new_lon_max = round(max(c1['lon'], c2['lon']), 4)
            new_state = {'active': False, 'clicks': []}
            status = f"✅ Box ready — name it and click ADD ZONE"
            return new_state, 'draw-btn', status, new_lat_min, new_lat_max, new_lon_min, new_lon_max

    return draw, ('draw-btn draw-active' if draw.get('active') else 'draw-btn'), "", lat_min, lat_max, lon_min, lon_max


# ── POI LOADER ────────────────────────────────────────────────
@app.callback(
    Output('store-poi','data'),
    Output('poi-status','children'),
    Input('poi-load-btn','n_clicks'),
    State('poi-center-input','value'),
    State('poi-radius-select','value'),
    State('store-search','data'),
    prevent_initial_call=True,
)
def cb_poi_load(n, query, radius, search_state):
    if not n:
        return [], ""
    # Determine centre: use search store if available and no new query
    if query and query.strip():
        result = geocode_location(query.strip())
        if result is None:
            return [], "✗ Location not found"
        lat, lon, label = result
    elif search_state and search_state.get('lat'):
        lat  = search_state['lat']
        lon  = search_state['lon']
        label = search_state.get('label', f"{lat:.3f},{lon:.3f}")
    else:
        return [], "⚠ Enter a location or search first"

    radius_m = int(radius or 2000)
    df = fetch_overpass_poi(lat, lon, radius_m)
    if df.empty:
        return [], f"No POI found within {radius_m}m"

    status = html.Span([
        html.Span(f"✓ {len(df)} POIs", style={'color':'#44aaff','fontWeight':'bold'}),
        html.Span(f" near {label[:30]}", style={'color':'#336644'}),
    ])
    return df.to_dict('records'), status


# ── RADAR BADGE UPDATE ────────────────────────────────────────
@app.callback(
    Output('radar-badge','style'),
    Input('layer-toggles','value'),
)
def cb_radar_badge(layers):
    active = 'radar' in (layers or [])
    return {'color': '#44aaff' if active else '#224433',
            'fontSize': '.65rem', 'marginLeft': '6px', 'cursor': 'default',
            'fontWeight': 'bold' if active else 'normal'}


# ── EVENT INTELLIGENCE CALLBACK  ← v1.2 ──────────────────────
@app.callback(
    Output('store-events','data'),
    Output('ev-status','children'),
    Output('ev-key-notice','children'),
    Input('ev-search-btn','n_clicks'),
    Input('ev-search-input','n_submit'),
    State('ev-search-input','value'),
    State('ev-radius-select','value'),
    State('store-search','data'),
    State('store-snap','data'),
    prevent_initial_call=True,
)
def cb_ev_search(n_click, n_submit, query, radius, search_state, snap):
    radius_km = int(radius or 25)

    # Resolve centre point
    if query and query.strip():
        result = geocode_location(query.strip())
        if result is None:
            return ({'events':[], 'venues':[], 'news':[], 'centre':{}},
                    html.Span("✗ Location not found", style={'color':'#ff4444'}), "")
        lat, lon, label = result
    elif search_state and search_state.get('lat'):
        lat   = search_state['lat']
        lon   = search_state['lon']
        label = search_state.get('label', f"{lat:.3f},{lon:.3f}")
    else:
        return ({'events':[], 'venues':[], 'news':[], 'centre':{}},
                html.Span("⚠ Enter a location or use the map search first",
                          style={'color':'#886600'}), "")

    # 1. Live events from Ticketmaster (if key set)
    tm_events = fetch_ticketmaster_events(lat, lon, radius_km=radius_km)

    # 2. Venue discovery from OSM (always available)
    osm_venues = fetch_overpass_events(lat, lon, radius_m=radius_km * 1000)

    # 3. GDELT news context from cached data
    try:
        df_ev = pd.DataFrame((snap or {}).get('events', []))
    except Exception:
        df_ev = pd.DataFrame()
    news_ctx = nearby_gdelt_context(lat, lon, radius_km, df_ev)

    store = {
        'events':  tm_events,
        'venues':  osm_venues,
        'news':    news_ctx,
        'centre':  {'lat': lat, 'lon': lon, 'label': label, 'radius_km': radius_km},
    }

    n_ev = len(tm_events); n_ve = len(osm_venues); n_nw = len(news_ctx)
    status = html.Div([
        html.Span(f"📍 {label[:35]}  ·  {radius_km} km radius  —  ",
                  style={'color':'#664488'}),
        html.Span(f"🎫 {n_ev} live events  ", style={'color':'#cc88ff','fontWeight':'bold'}),
        html.Span(f"🏟 {n_ve} venues  ", style={'color':'#4488ff'}),
        html.Span(f"📰 {n_nw} news items", style={'color':'#ffaacc'}),
    ])

    key_notice = ""
    if not TICKETMASTER_API_KEY:
        key_notice = html.Div([
            html.Span("🔑 No Ticketmaster key — showing OSM venues only. "
                      "Get a free key (5000 calls/day) at: "),
            html.A("developer.ticketmaster.com",
                   href="https://developer.ticketmaster.com/",
                   target="_blank",
                   className='ev-link'),
            html.Span(" → add to gsf.ini [events] section."),
        ], className='ev-tm-note')

    return store, status, key_notice


@app.callback(
    Output('ev-results-panel','children'),
    Input('store-events','data'),
    Input('ev-tabs','active_tab'),
)
def cb_ev_results(store, active_tab):
    store = store or {}
    centre = store.get('centre', {})

    if not centre:
        return html.Div([
            html.Div("🎟", style={'fontSize':'2rem','marginBottom':'8px','color':'#2a0040'}),
            html.Div("Search a location to discover nearby events, venues, and news context.",
                     style={'color':'#443355','fontSize':'.65rem','lineHeight':'1.5'}),
        ], className='ev-empty')

    # ── Tab: Live Events ──────────────────────────────────────
    if active_tab == 'ev-tab-events':
        events = store.get('events', [])
        if not events:
            if not TICKETMASTER_API_KEY:
                return html.Div([
                    html.Div("🔑 Add your free Ticketmaster API key to gsf.ini to see live events.",
                             style={'color':'#443355','fontSize':'.65rem'}),
                    html.Br(),
                    html.Div("Meanwhile, check the 🏟 VENUES tab for event spaces in this area.",
                             style={'color':'#5d2d8a','fontSize':'.63rem'}),
                ], className='ev-empty')
            return html.Div(
                f"No upcoming events found within {centre.get('radius_km', 25)} km. "
                "Try increasing the radius.",
                className='ev-empty')

        rows = []
        current_date = None
        for ev in events:
            date_str = ev.get('date', '')
            # Date separator
            if date_str and date_str != current_date:
                current_date = date_str
                try:
                    from datetime import datetime as _dt
                    d = _dt.strptime(date_str, '%Y-%m-%d')
                    date_display = d.strftime('%A, %d %b %Y')
                except Exception:
                    date_display = date_str
                rows.append(html.Div(date_display, className='ev-section-hdr'))

            status_code = ev.get('status', 'onsale')
            status_map = {
                'onsale': ('ON SALE','onsale'),
                'offsale': ('OFF SALE','cancelled'),
                'cancelled': ('CANCELLED','cancelled'),
                'postponed': ('POSTPONED','postponed'),
                'rescheduled': ('RESCHEDULED','postponed'),
                'soldOut': ('SOLD OUT','sold-out'),
            }
            s_label, s_cls = status_map.get(status_code, (status_code.upper(), 'onsale'))

            price_el = html.Span(ev['price'], className='ev-price') if ev.get('price') else html.Span()

            url = ev.get('url', '')
            name_el = (html.A(ev['name'], href=url, target='_blank', className='ev-name')
                       if url else html.Span(ev['name'], className='ev-name'))

            rows.append(html.Div([
                html.Div([
                    html.Span(ev['icon'] + ' ', style={'fontSize':'.85rem'}),
                    name_el,
                    html.Span(s_label, className=f'ev-badge {s_cls}'),
                    price_el,
                ], style={'display':'flex','alignItems':'baseline','flexWrap':'wrap','gap':'3px'}),
                html.Div([
                    html.Span(f"🕐 {ev['time']}  " if ev.get('time') and ev['time'] != 'TBA' else '🕐 TBA  ',
                              style={'color':'#88aaff','fontSize':'.6rem'}),
                    html.Span(f"📍 {ev['venue']}", style={'color':'#664488','fontSize':'.6rem'}),
                    html.Span(f"  ·  {ev['city']}" if ev.get('city') else '',
                              style={'color':'#443366','fontSize':'.6rem'}),
                    html.Span(f"  ·  {ev['dist_km']} km away",
                              style={'color':'#336644','fontSize':'.59rem'}),
                ], className='ev-meta'),
            ], className='ev-card'))

        return html.Div(rows)

    # ── Tab: Venues ───────────────────────────────────────────
    elif active_tab == 'ev-tab-venues':
        venues = store.get('venues', [])
        if not venues:
            return html.Div("No event venues found in this area via OpenStreetMap.",
                            className='ev-empty')
        rows = [html.Div("Event spaces, stadiums & cultural venues from OpenStreetMap",
                         className='ev-section-hdr')]
        for ve in venues:
            url = ve.get('url', '')
            name_el = (html.A(ve['name'], href=url, target='_blank', className='ev-name')
                       if url else html.Span(ve['name'], className='ev-name'))
            rows.append(html.Div([
                html.Div([
                    html.Span(ve['icon'] + ' ', style={'fontSize':'.85rem'}),
                    name_el,
                    html.Span('VENUE', className='ev-badge venue'),
                ], style={'display':'flex','alignItems':'baseline','gap':'3px'}),
                html.Div([
                    html.Span(ve['category'] + '  ', style={'color':'#664488','fontSize':'.6rem'}),
                    html.Span(f"📍 {ve['city']}" if ve.get('city') else '',
                              style={'color':'#443366','fontSize':'.6rem'}),
                    html.Span(f"  ·  {ve['dist_km']} km away",
                              style={'color':'#336644','fontSize':'.59rem'}),
                ], className='ev-meta'),
            ], className='ev-card'))
        return html.Div(rows)

    # ── Tab: News Context ─────────────────────────────────────
    else:
        news = store.get('news', [])
        if not news:
            return html.Div([
                html.Div("No recent GDELT news events found near this location.",
                         style={'color':'#443355','fontSize':'.65rem','marginBottom':'6px'}),
                html.Div("GDELT tracks global conflict, disaster, protest, and emergency news "
                         "in near-real-time. A quiet area means no major incidents in the last 4 hours.",
                         style={'color':'#332244','fontSize':'.62rem','lineHeight':'1.4'}),
            ], className='ev-empty')

        rows = [html.Div("Recent news events within this area (GDELT · last 4 hours)",
                         className='ev-section-hdr')]
        for item in news:
            sentiment = item.get('sentiment', '🟡 Neutral')
            url = item.get('url', '')
            title_el = (html.A(item['title'], href=url, target='_blank',
                               style={'color':'#ccaaff','textDecoration':'none'})
                        if url else html.Span(item['title'], className='ev-news-title'))
            rows.append(html.Div([
                html.Div(title_el, className='ev-news-title'),
                html.Div([
                    html.Span(sentiment + '  ', style={'fontSize':'.6rem'}),
                    html.Span(f"📰 {item['count']} articles  ",
                              style={'color':'#664488','fontSize':'.59rem'}),
                    html.Span(f"tone: {item['tone']:+.1f}  ",
                              style={'color':'#443366','fontSize':'.59rem'}),
                    html.Span(f"📍 {item['dist_km']} km away",
                              style={'color':'#336644','fontSize':'.59rem'}),
                ], className='ev-news-meta'),
            ], className='ev-news-item'))
        return html.Div(rows)


# ── EVENT MARKERS ON TILE MAP  (added to cb_map via store-events) ──
# cb_map already declared below; we patch it by adding store-events as Input.


# ── PRIMARY: TILE MAP (Scattermapbox) ─────────────────────────
# Fetches RainViewer radar path from cache and composes
# all layers including POI, radar tiles, basemap, search centre.

@app.callback(
    Output('map-fig','figure'),
    Input('store-snap','data'),
    Input('layer-toggles','value'),
    Input('store-pb','data'),
    Input('eq-days-slider','value'),
    Input('store-basemap','data'),
    Input('store-search','data'),
    Input('store-poi','data'),
    Input('store-events','data'),
)
def cb_map(snap, layers, pb_state, eq_days, basemap_style, search_state, poi_data, ev_store):
    if not snap: return go.Figure()
    layers   = layers or []; eq_days = eq_days or 1
    pb       = pb_state or {}; is_playback = not pb.get('live', True)
    mapstyle = basemap_style or 'carto-darkmatter'

    def sdf(key):
        try: return pd.DataFrame(snap.get(key, []))
        except: return pd.DataFrame()

    df_fl = sdf('flights'); df_fi = sdf('fires');  df_sh = sdf('ships')
    df_aq = sdf('aq');      df_ev = sdf('events'); df_sl = sdf('starlink')
    df_th = sdf('threat');  iss   = snap.get('iss', {}) or {}
    trails    = snap.get('trails', {}) or {}
    geofences = snap.get('geofences', {}) or _geofence.get_zones()

    if eq_days <= 1:
        df_eq = sdf('quakes')
    else:
        df_30 = _cache.get_quakes_30d()
        if df_30.empty: df_eq = sdf('quakes')
        else:
            cutoff_ms = (datetime.utcnow() - timedelta(days=eq_days)).timestamp() * 1000
            df_eq = df_30[df_30['time_ms'] >= cutoff_ms].head(MAX_EQ)

    traces = []

    # ── Flight trails ──────────────────────────────────────────
    if 'trails' in layers and trails:
        lons_t = []; lats_t = []
        for pts in trails.values():
            if len(pts) < 2: continue
            for p in pts: lons_t.append(p[0]); lats_t.append(p[1])
            lons_t.append(None); lats_t.append(None)
        if lons_t:
            traces.append(go.Scattermapbox(
                lon=lons_t, lat=lats_t, mode='lines',
                line=dict(color='rgba(0,200,130,0.35)', width=1.5),
                hoverinfo='skip', name='Trails', showlegend=False))

    # ── Flights ────────────────────────────────────────────────
    if 'flights' in layers and not df_fl.empty:
        traces.append(go.Scattermapbox(
            lon=df_fl['lon'], lat=df_fl['lat'], mode='markers',
            marker=dict(size=7, color=df_fl['alt'].tolist(),
                        colorscale=[[0,'rgba(0,55,30,0.8)'],[.4,'rgba(0,140,80,0.85)'],[1,'rgba(0,255,220,0.9)']],
                        cmin=0, cmax=13000, opacity=0.9,
                        colorbar=dict(title=dict(text='Alt(m)',font=dict(color='#a0ffcc',size=8)),
                                      thickness=7,len=0.25,x=1.01,
                                      tickfont=dict(color='#a0ffcc',size=7),
                                      bgcolor='rgba(0,0,0,0)',bordercolor='#0d3320')),
            text="<b>✈ "+df_fl['callsign']+"</b><br>"+df_fl['country']+"<br>↑"+df_fl['alt'].astype(int).astype(str)+"m  "+df_fl['vel'].astype(str)+"m/s",
            hoverinfo='text', name=f"✈ Aircraft ({len(df_fl)})",
            customdata=df_fl[['icao','callsign','country','alt','vel','hdg']].values,
        ))

    # ── Earthquakes ────────────────────────────────────────────
    if 'quakes' in layers and not df_eq.empty:
        mag_sz = (df_eq['mag'].clip(2, 8) * 3.5).tolist()
        traces.append(go.Scattermapbox(
            lon=df_eq['lon'], lat=df_eq['lat'], mode='markers',
            marker=dict(size=mag_sz, color=df_eq['mag'].tolist(),
                        colorscale=[[0,'rgba(80,10,0,0.75)'],[.5,'rgba(220,60,0,0.85)'],[1,'rgba(255,230,0,0.95)']],
                        cmin=2.5, cmax=7.5, opacity=0.9),
            text="<b>⚡ M"+df_eq['mag'].round(1).astype(str)+"</b><br>"+df_eq['place']+"<br>↓"+df_eq['depth'].astype(str)+"km  "+df_eq['time'],
            hoverinfo='text', name=f"⚡ Earthquakes ({len(df_eq)})",
            customdata=df_eq[['mag','place','depth','time']].values,
        ))

    # ── Fires ──────────────────────────────────────────────────
    if 'fires' in layers and not df_fi.empty:
        traces.append(go.Scattermapbox(
            lon=df_fi['lon'], lat=df_fi['lat'], mode='markers',
            marker=dict(size=5, color=df_fi['brightness'].tolist(),
                        colorscale=[[0,'rgba(80,20,0,0.7)'],[.5,'rgba(255,80,0,0.85)'],[1,'rgba(255,230,0,0.95)']],
                        cmin=300, cmax=500, opacity=0.85),
            text="<b>🔥 Fire</b><br>"+df_fi['brightness'].round(1).astype(str)+" K  FRP: "+df_fi['power'].astype(str)+" MW",
            hoverinfo='text', name=f"🔥 Fires ({len(df_fi)})",
        ))

    # ── Ships ──────────────────────────────────────────────────
    if 'ships' in layers and not df_sh.empty and 'lon' in df_sh.columns:
        traces.append(go.Scattermapbox(
            lon=df_sh['lon'], lat=df_sh['lat'], mode='markers',
            marker=dict(size=8, color='rgba(0,200,255,0.85)',
                        symbol='harbor'),
            text="<b>🚢 "+df_sh['name']+"</b><br>MMSI: "+df_sh['mmsi'].astype(str)+"<br>"+df_sh['sog'].astype(str)+" kn",
            hoverinfo='text', name=f"🚢 Ships ({len(df_sh)})",
            customdata=df_sh[['mmsi','name','sog','cog']].values
                       if all(c in df_sh.columns for c in ['mmsi','name','sog','cog']) else None,
        ))

    # ── Starlink ───────────────────────────────────────────────
    if 'starlink' in layers and not df_sl.empty and 'lon' in df_sl.columns:
        traces.append(go.Scattermapbox(
            lon=df_sl['lon'], lat=df_sl['lat'], mode='markers',
            marker=dict(size=5, color='rgba(170,170,255,0.7)'),
            text="<b>🛰 "+df_sl['name']+"</b><br>Alt: "+df_sl['alt_km'].astype(str)+" km",
            hoverinfo='text', name=f"🛰 Starlink ({len(df_sl)})",
            customdata=df_sl[['name','norad','alt_km']].values
                       if all(c in df_sl.columns for c in ['name','norad','alt_km']) else None,
        ))

    # ── ISS ────────────────────────────────────────────────────
    if 'iss' in layers and iss and iss.get('lat') is not None:
        traces.append(go.Scattermapbox(
            lon=[iss['lon']], lat=[iss['lat']], mode='markers+text',
            marker=dict(size=16, color='rgba(255,238,68,0.95)'),
            text=["🛸"], textposition='top center',
            textfont=dict(size=14),
            hovertext=f"<b>🛸 ISS</b><br>Alt: {iss.get('alt_km','—')} km<br>Vel: {int(iss.get('vel_kph',0)):,} km/h<br>Vis: {iss.get('vis','—')}",
            hoverinfo='text', name="🛸 ISS",
        ))

    # ── Air Quality ────────────────────────────────────────────
    if 'aq' in layers and not df_aq.empty and 'lon' in df_aq.columns:
        traces.append(go.Scattermapbox(
            lon=df_aq['lon'], lat=df_aq['lat'], mode='markers',
            marker=dict(size=9, color=df_aq['pm25'].tolist(),
                        colorscale=[[0,'rgba(0,228,0,0.8)'],[.25,'rgba(255,255,0,0.8)'],
                                    [.5,'rgba(255,126,0,0.85)'],[.75,'rgba(255,0,0,0.9)'],[1,'rgba(126,0,35,0.9)']],
                        cmin=0, cmax=150, opacity=0.88),
            text="<b>💨 "+df_aq['name']+"</b><br>PM2.5: "+df_aq['pm25'].astype(str)+" µg/m³<br>"+df_aq['label'],
            hoverinfo='text', name=f"💨 Air Quality ({len(df_aq)})",
            customdata=df_aq[['pm25','label','name','country']].values
                       if all(c in df_aq.columns for c in ['pm25','label','name','country']) else None,
        ))

    # ── GDELT Events ───────────────────────────────────────────
    if 'events' in layers and not df_ev.empty and 'lon' in df_ev.columns:
        traces.append(go.Scattermapbox(
            lon=df_ev['lon'], lat=df_ev['lat'], mode='markers',
            marker=dict(size=(df_ev['count'].clip(1, 10) / 1.5 + 5).tolist(),
                        color=df_ev['tone'].tolist(),
                        colorscale=[[0,'rgba(255,0,68,0.8)'],[.5,'rgba(136,68,34,0.75)'],[1,'rgba(68,100,68,0.7)']],
                        cmin=-20, cmax=5, opacity=0.8),
            text="<b>📰 "+df_ev['title'].str[:55]+"</b><br>"+df_ev['count'].astype(str)+" articles",
            hoverinfo='text', name=f"📰 Events ({len(df_ev)})",
            customdata=df_ev[['title','count','tone']].values
                       if all(c in df_ev.columns for c in ['title','count','tone']) else None,
        ))

    # ── Threat zones ───────────────────────────────────────────
    if 'threat' in layers and not df_th.empty and 'lon_bin' in df_th.columns:
        traces.append(go.Scattermapbox(
            lon=df_th['lon_bin'], lat=df_th['lat_bin'], mode='markers',
            marker=dict(size=(df_th['score'].clip(0, 100) / 7 + 8).tolist(),
                        color=df_th['score'].tolist(),
                        colorscale=[[0,'rgba(0,51,0,0.4)'],[.4,'rgba(136,102,0,0.55)'],
                                    [.7,'rgba(255,68,0,0.65)'],[1,'rgba(255,0,0,0.75)']],
                        cmin=0, cmax=100, opacity=0.7),
            text="<b>🎯 Threat Zone</b><br>Score: "+df_th['score'].round(1).astype(str)+" / 100",
            hoverinfo='text', name=f"🎯 Threat ({len(df_th)})",
        ))

    # ── Geofence polygon outlines ──────────────────────────────
    if 'geofence' in layers and geofences:
        for zone_name, z in geofences.items():
            lats_z = [z['lat_min'],z['lat_min'],z['lat_max'],z['lat_max'],z['lat_min']]
            lons_z = [z['lon_min'],z['lon_max'],z['lon_max'],z['lon_min'],z['lon_min']]
            traces.append(go.Scattermapbox(
                lon=lons_z, lat=lats_z, mode='lines',
                line=dict(color='rgba(0,255,68,0.75)', width=2),
                fill='toself', fillcolor='rgba(0,255,68,0.05)',
                hoverinfo='text',
                text=f"<b>🔲 {zone_name}</b><br>{z['lat_min']:.2f}–{z['lat_max']:.2f}°N  {z['lon_min']:.2f}–{z['lon_max']:.2f}°E",
                name=f"🔲 {zone_name}", showlegend=True,
            ))

    # ── POI layer ──────────────────────────────────────────────
    if 'poi' in layers and poi_data:
        try:
            df_poi = pd.DataFrame(poi_data)
            if not df_poi.empty and 'lon' in df_poi.columns:
                # Group by category color for cleaner rendering
                for cat_color in df_poi['color'].unique():
                    subset = df_poi[df_poi['color'] == cat_color]
                    traces.append(go.Scattermapbox(
                        lon=subset['lon'], lat=subset['lat'], mode='markers+text',
                        marker=dict(size=11, color=cat_color, opacity=0.85),
                        text=subset['icon'].tolist(),
                        textposition='middle center',
                        textfont=dict(size=9),
                        hovertext="<b>"+subset['icon']+" "+subset['name']+"</b><br>"+subset['category'],
                        hoverinfo='text',
                        name=f"🏙 POI ({len(df_poi)} total)" if cat_color == df_poi['color'].iloc[0] else None,
                        showlegend=(cat_color == df_poi['color'].iloc[0]),
                    ))
        except Exception as e:
            print(f"[POI render] {e}")

    # ── Event markers (Ticketmaster + OSM venues) ──────────────
    ev_store_data = ev_store or {}
    ev_centre     = ev_store_data.get('centre', {})
    if ev_centre:
        # Live Ticketmaster events
        tm_evs = ev_store_data.get('events', [])
        if tm_evs:
            ev_df = pd.DataFrame(tm_evs)
            ev_df = ev_df.dropna(subset=['lat','lon'])
            if not ev_df.empty:
                traces.append(go.Scattermapbox(
                    lon=ev_df['lon'], lat=ev_df['lat'],
                    mode='markers+text',
                    marker=dict(size=14, color='rgba(200,120,255,0.92)',
                                symbol='circle'),
                    text=ev_df['icon'].tolist(),
                    textposition='middle center',
                    textfont=dict(size=10),
                    hovertext=(
                        "<b>" + ev_df['icon'] + " " + ev_df['name'] + "</b><br>" +
                        ev_df['date'] + "  " + ev_df['time'] + "<br>" +
                        "📍 " + ev_df['venue'] + "<br>" +
                        ev_df['dist_km'].astype(str) + " km away" +
                        ev_df['price'].apply(lambda p: f"<br>💰 {p}" if p else "")
                    ).tolist(),
                    hoverinfo='text',
                    name=f"🎫 Events ({len(ev_df)})",
                    customdata=ev_df[['name','date','venue','dist_km']].values,
                ))

        # OSM Venues
        osm_vs = ev_store_data.get('venues', [])
        if osm_vs:
            vs_df = pd.DataFrame(osm_vs)
            vs_df = vs_df.dropna(subset=['lat','lon'])
            if not vs_df.empty:
                traces.append(go.Scattermapbox(
                    lon=vs_df['lon'], lat=vs_df['lat'],
                    mode='markers+text',
                    marker=dict(size=11, color='rgba(68,136,255,0.82)',
                                symbol='circle'),
                    text=vs_df['icon'].tolist(),
                    textposition='middle center',
                    textfont=dict(size=9),
                    hovertext=(
                        "<b>" + vs_df['icon'] + " " + vs_df['name'] + "</b><br>" +
                        vs_df['category'] + "<br>" +
                        vs_df['dist_km'].astype(str) + " km away"
                    ).tolist(),
                    hoverinfo='text',
                    name=f"🏟 Venues ({len(vs_df)})",
                ))

        # Draw search-radius circle hint as a faint polygon
        if ev_centre.get('lat') and ev_centre.get('radius_km'):
            c_lat = ev_centre['lat']; c_lon = ev_centre['lon']
            r_km  = ev_centre['radius_km']
            # Approximate circle with 36-point polygon
            import math as _m
            R_earth = 6371.0
            circle_lats = []; circle_lons = []
            for deg in range(0, 361, 10):
                θ = _m.radians(deg)
                dlat = (r_km / R_earth) * _m.cos(θ)
                dlon = (r_km / R_earth) * _m.sin(θ) / _m.cos(_m.radians(c_lat))
                circle_lats.append(c_lat + _m.degrees(dlat))
                circle_lons.append(c_lon + _m.degrees(dlon))
            traces.append(go.Scattermapbox(
                lon=circle_lons, lat=circle_lats,
                mode='lines',
                line=dict(color='rgba(180,100,255,0.30)', width=1.5),
                fill='toself', fillcolor='rgba(180,100,255,0.04)',
                hoverinfo='skip',
                name=f"Event search radius ({r_km} km)",
                showlegend=True,
            ))

    # ── Map layout ─────────────────────────────────────────────
    ts_label = (snap.get('ts', ''))[11:19] + ' UTC' if snap.get('ts') else datetime.utcnow().strftime('%H:%M:%S UTC')
    pb_label = f"  ⏪ PLAYBACK {ts_label}" if is_playback else f"  ●  {ts_label}"

    # Search-driven center (if available, override default)
    s = search_state or {}
    center_lat = s.get('lat', 20.0)
    center_lon = s.get('lon',  0.0)
    zoom_level = s.get('zoom', 1.5)

    # If an event search was done, zoom to that area
    if ev_centre.get('lat') and ev_centre.get('lon'):
        r_km = ev_centre.get('radius_km', 25)
        center_lat = ev_centre['lat']
        center_lon = ev_centre['lon']
        # Pick zoom level based on radius
        if r_km <= 5:    zoom_level = 13
        elif r_km <= 15: zoom_level = 11
        elif r_km <= 25: zoom_level = 10
        elif r_km <= 50: zoom_level = 9
        else:            zoom_level = 8

    # Radar tiles as mapbox layer
    mapbox_layers = []
    if 'radar' in layers:
        radar_path = _cache.get_radar_path()
        if radar_path:
            mapbox_layers.append({
                'sourcetype': 'raster',
                'source': [f"https://tilecache.rainviewer.com{radar_path}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"],
                'opacity': 0.45,
                'type': 'raster',
                'tileSize': 256,
            })

    fig = go.Figure(data=traces)
    fig.update_layout(
        mapbox=dict(
            style=mapstyle,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
            uirevision='gsf-map',
            layers=mapbox_layers,
        ),
        title=dict(
            text=f"◈ GSF LIVE MAP{pb_label}",
            font=dict(size=10, color='#ff8844' if is_playback else '#00bb55'),
            x=0.01, y=0.99,
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0ffcc', family='monospace', size=10),
        margin=dict(l=0, r=0, t=28, b=0),
        showlegend=True,
        legend=dict(bgcolor='rgba(0,18,8,0.85)', bordercolor='#0d3320', borderwidth=1,
                    font=dict(size=7, color='#a0ffcc'), x=0.01, y=0.01, orientation='v'),
        uirevision='gsf-map',
    )
    return fig


# ── SECONDARY: GLOBE (Scattergeo) ────────────────────────────
@app.callback(
    Output('globe-fig','figure'),
    Input('store-snap','data'),
    Input('layer-toggles','value'),
    Input('eq-days-slider','value'),
)
def cb_globe(snap, layers, eq_days):
    if not snap: return go.Figure()
    layers = layers or []; eq_days = eq_days or 1

    def sdf(key):
        try: return pd.DataFrame(snap.get(key,[]))
        except: return pd.DataFrame()

    df_fl=sdf('flights'); df_fi=sdf('fires'); df_sh=sdf('ships')
    df_aq=sdf('aq');      df_ev=sdf('events'); df_sl=sdf('starlink')
    df_th=sdf('threat');  iss=snap.get('iss',{}) or {}

    if eq_days <= 1:
        df_eq = sdf('quakes')
    else:
        df_30 = _cache.get_quakes_30d()
        if df_30.empty: df_eq = sdf('quakes')
        else:
            cutoff_ms = (datetime.utcnow()-timedelta(days=eq_days)).timestamp()*1000
            df_eq = df_30[df_30['time_ms'] >= cutoff_ms].head(MAX_EQ)

    traces = []
    if 'flights' in layers and not df_fl.empty:
        traces.append(go.Scattergeo(lon=df_fl['lon'],lat=df_fl['lat'],hoverinfo='text',
            text="<b>✈ "+df_fl['callsign']+"</b><br>"+df_fl['country']+"<br>↑"+df_fl['alt'].astype(int).astype(str)+"m",
            name=f"✈ {len(df_fl)}",mode='markers',
            marker=dict(size=4,opacity=.8,color=df_fl['alt'],colorscale=[[0,'#002a18'],[.4,'#00884a'],[1,'#00ffee']],cmin=0,cmax=13000,line=dict(width=0))))
    if 'quakes' in layers and not df_eq.empty:
        traces.append(go.Scattergeo(lon=df_eq['lon'],lat=df_eq['lat'],hoverinfo='text',
            text="<b>⚡ M"+df_eq['mag'].round(1).astype(str)+"</b><br>"+df_eq['place'],
            name=f"⚡ {len(df_eq)}",mode='markers',
            marker=dict(size=(df_eq['mag'].clip(2,8)*3).tolist(),opacity=.78,color=df_eq['mag'].tolist(),
                        colorscale=[[0,'#2a0800'],[.5,'#cc4400'],[1,'#ff1100']],cmin=2.5,cmax=7.5,symbol='diamond',line=dict(width=1,color='rgba(255,80,0,0.4)'))))
    if 'fires' in layers and not df_fi.empty:
        traces.append(go.Scattergeo(lon=df_fi['lon'],lat=df_fi['lat'],hoverinfo='text',
            text="<b>🔥</b> "+df_fi['brightness'].round(1).astype(str)+" K",name=f"🔥 {len(df_fi)}",mode='markers',
            marker=dict(size=3,opacity=.7,color=df_fi['brightness'].tolist(),colorscale=[[0,'#330000'],[.5,'#ff4400'],[1,'#ffcc00']],cmin=300,cmax=500,line=dict(width=0))))
    if 'ships' in layers and not df_sh.empty and 'lon' in df_sh.columns:
        traces.append(go.Scattergeo(lon=df_sh['lon'],lat=df_sh['lat'],hoverinfo='text',
            text="<b>🚢 "+df_sh['name']+"</b>",name=f"🚢 {len(df_sh)}",mode='markers',
            marker=dict(size=4,opacity=.8,color='#00ccff',symbol='triangle-up',line=dict(width=0))))
    if 'starlink' in layers and not df_sl.empty and 'lon' in df_sl.columns:
        traces.append(go.Scattergeo(lon=df_sl['lon'],lat=df_sl['lat'],hoverinfo='text',
            text="<b>🛰 "+df_sl['name']+"</b>",name=f"🛰 {len(df_sl)}",mode='markers',
            marker=dict(size=3,opacity=.7,color='#aaaaff',line=dict(width=0))))
    if 'iss' in layers and iss and iss.get('lat') is not None:
        traces.append(go.Scattergeo(lon=[iss['lon']],lat=[iss['lat']],hoverinfo='text',
            text=f"<b>🛸 ISS</b><br>Alt: {iss.get('alt_km','—')} km",name="🛸 ISS",mode='markers',
            marker=dict(size=14,opacity=.95,color='#ffee44',symbol='star',line=dict(width=2,color='#886600'))))
    if 'aq' in layers and not df_aq.empty and 'lon' in df_aq.columns:
        traces.append(go.Scattergeo(lon=df_aq['lon'],lat=df_aq['lat'],hoverinfo='text',
            text="<b>💨 "+df_aq['name']+"</b><br>PM2.5: "+df_aq['pm25'].astype(str)+" µg/m³",
            name=f"💨 {len(df_aq)}",mode='markers',
            marker=dict(size=6,opacity=.8,color=df_aq['pm25'].tolist(),colorscale=[[0,'#00e400'],[.5,'#ff7e00'],[1,'#7e0023']],cmin=0,cmax=150,line=dict(width=0))))
    if 'events' in layers and not df_ev.empty and 'lon' in df_ev.columns:
        traces.append(go.Scattergeo(lon=df_ev['lon'],lat=df_ev['lat'],hoverinfo='text',
            text="<b>📰 "+df_ev['title'].str[:50]+"</b>",name=f"📰 {len(df_ev)}",mode='markers',
            marker=dict(size=(df_ev['count'].clip(1,10)/2+3).tolist(),opacity=.75,
                        color=df_ev['tone'].tolist(),colorscale=[[0,'#ff0044'],[.5,'#884422'],[1,'#446644']],cmin=-20,cmax=5,symbol='diamond',line=dict(width=0))))
    if 'threat' in layers and not df_th.empty and 'lon_bin' in df_th.columns:
        traces.append(go.Scattergeo(lon=df_th['lon_bin'],lat=df_th['lat_bin'],hoverinfo='text',
            text="<b>🎯 Threat</b><br>"+df_th['score'].round(1).astype(str),name=f"🎯 {len(df_th)}",mode='markers',
            marker=dict(size=(df_th['score'].clip(0,100)/8+5).tolist(),opacity=.55,
                        color=df_th['score'].tolist(),colorscale=[[0,'#003300'],[.4,'#886600'],[.7,'#ff4400'],[1,'#ff0000']],cmin=0,cmax=100,symbol='square',line=dict(width=0))))

    fig=go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"GLOBAL OVERVIEW  ·  {datetime.utcnow().strftime('%H:%M:%S UTC')}",font=dict(size=10,color='#00bb55'),x=.01),
        geo=dict(showframe=False,bgcolor='rgba(0,0,0,0)',showcoastlines=True,coastlinecolor='#1a6644',
                 showland=True,landcolor='rgb(14,19,16)',showocean=True,oceancolor='rgb(5,10,20)',
                 showlakes=True,lakecolor='rgb(7,12,25)',showcountries=True,countrycolor='rgb(30,50,38)',
                 showrivers=True,rivercolor='rgb(7,30,50)',projection_type='natural earth'),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0ffcc',family='monospace',size=10),
        margin=dict(l=0,r=0,t=26,b=0),showlegend=True,
        legend=dict(bgcolor='rgba(0,18,8,0.85)',bordercolor='#0d3320',borderwidth=1,font=dict(size=7,color='#a0ffcc'),x=.01,y=.99),
        uirevision='gsf-globe',
    )
    return fig

# ── DENSITY HEATMAP ──────────────────────────────────────────
@app.callback(Output('density-fig','figure'),Input('store-snap','data'),Input('view-tabs','active_tab'))
def cb_density(snap, tab):
    if not snap or tab!='t-density': return go.Figure()
    def sdf(k):
        try: return pd.DataFrame(snap.get(k,[]))
        except: return pd.DataFrame()
    df_fl=sdf('flights'); df_eq=sdf('quakes'); df_fi=sdf('fires')
    traces=[]
    if not df_fl.empty and 'lon' in df_fl.columns:
        traces.append(go.Densitymapbox(lon=df_fl['lon'],lat=df_fl['lat'],z=[1]*len(df_fl),radius=20,opacity=0.65,
            colorscale=[[0,'rgba(0,30,15,0)'],[.2,'rgba(0,80,50,0.5)'],[.6,'rgba(0,200,120,0.7)'],[1,'rgba(0,255,180,0.9)']],
            showscale=False,name='Flight Density',hoverinfo='skip'))
    if not df_eq.empty and 'lon' in df_eq.columns:
        traces.append(go.Densitymapbox(lon=df_eq['lon'],lat=df_eq['lat'],z=(df_eq['mag'].clip(2,9)**2).tolist(),radius=40,opacity=0.6,
            colorscale=[[0,'rgba(40,0,0,0)'],[.3,'rgba(180,40,0,0.5)'],[.7,'rgba(255,100,0,0.7)'],[1,'rgba(255,220,0,0.9)']],
            showscale=False,name='Seismic Intensity',hoverinfo='skip'))
    if not df_fi.empty and 'lon' in df_fi.columns:
        traces.append(go.Densitymapbox(lon=df_fi['lon'],lat=df_fi['lat'],z=df_fi['power'].clip(0,500).fillna(0).tolist(),radius=18,opacity=0.55,
            colorscale=[[0,'rgba(50,0,0,0)'],[.5,'rgba(255,50,0,0.6)'],[1,'rgba(255,255,0,0.85)']],
            showscale=False,name='Fire Intensity',hoverinfo='skip'))
    fig=go.Figure(data=traces)
    fig.update_layout(mapbox=dict(style='carto-darkmatter',center=dict(lat=20,lon=0),zoom=1.2),
        title=dict(text="🌡 DENSITY  ·  Flights · Seismic · Fire",font=dict(size=10,color='#00cc66'),x=.01),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0ffcc',family='monospace',size=10),
        margin=dict(l=0,r=0,t=28,b=0),showlegend=True,
        legend=dict(bgcolor='rgba(0,18,8,0.88)',bordercolor='#0d3320',borderwidth=1,font=dict(size=8,color='#a0ffcc'),x=.01,y=.99),
        uirevision='gsf-density')
    return fig

# ── INSPECT (click on map-fig or globe-fig) ──────────────────
@app.callback(
    Output('inspect-panel','children'),
    Input('map-fig','clickData'),
    Input('globe-fig','clickData'),
)
def cb_inspect(map_click, globe_click):
    click = map_click or globe_click
    if not click:
        return html.Span("Click any marker on the tile map or globe to inspect.",
                         style={'color':'#224433','fontSize':'.68rem'})
    pt  = click['points'][0]
    txt = pt.get('text','') or ''
    lat = pt.get('lat'); lon = pt.get('lon')
    cd  = pt.get('customdata',[]) or []

    def kv(k,v): return html.Div([html.Span(f"{k}: ",style={'color':'#2a5535'}),html.Span(str(v),style={'color':'#aaffcc'})])

    children = []; is_flight = False; selected_icao = None

    if '✈' in txt and len(cd) >= 6:
        is_flight = True; selected_icao = str(cd[0]).upper()
        children = [html.Div("✈  AIRCRAFT",className='inspect-title'),
                    kv("ICAO",selected_icao),kv("Callsign",cd[1]),kv("Country",cd[2]),
                    kv("Altitude",f"{int(cd[3]):,} m  ({round(cd[3]/1000,2)} km)"),
                    kv("Speed",f"{cd[4]} m/s  ({round(float(cd[4])*3.6,1)} km/h)"),kv("Heading",f"{int(cd[5])}°")]
    elif '⚡' in txt and len(cd) >= 4:
        children = [html.Div("⚡  EARTHQUAKE",className='inspect-title',style={'color':'#ff6633'}),
                    kv("Magnitude",f"M{cd[0]}"),kv("Location",str(cd[1])[:32]),kv("Depth",f"{cd[2]} km"),kv("Time",cd[3])]
    elif '🚢' in txt and len(cd) >= 4:
        children = [html.Div("🚢  VESSEL",className='inspect-title',style={'color':'#00ccff'}),
                    kv("MMSI",cd[0]),kv("Name",cd[1]),kv("Speed",f"{cd[2]} kn"),kv("Heading",f"{float(cd[3]):.0f}°")]
    elif '🛰' in txt and len(cd) >= 3:
        children = [html.Div("🛰  STARLINK",className='inspect-title',style={'color':'#aaaaff'}),
                    kv("Name",cd[0]),kv("NORAD",cd[1]),kv("Altitude",f"{cd[2]} km")]
    elif '🛸' in txt:
        iss,_=_cache.get_iss()
        children = [html.Div("🛸  ISS",className='inspect-title',style={'color':'#ffee44'})]
        if iss: children += [kv("Altitude",f"{iss.get('alt_km','—')} km"),kv("Velocity",f"{int(iss.get('vel_kph',0)):,} km/h"),kv("Visibility",iss.get('vis','—'))]
    elif '💨' in txt and len(cd) >= 4:
        color,_=aqi_color(float(cd[0]) if cd[0] else 0)
        children = [html.Div("💨  AIR QUALITY",className='inspect-title',style={'color':'#44ccff'}),
                    kv("Station",cd[2]),kv("Country",cd[3]),kv("PM2.5",f"{cd[0]} µg/m³"),
                    html.Div(str(cd[1]),style={'color':color,'fontWeight':'bold','marginTop':'3px'})]
    elif '📰' in txt and len(cd) >= 3:
        children = [html.Div("📰  NEWS EVENT",className='inspect-title',style={'color':'#ffaacc'}),
                    html.Div(str(cd[0])[:75],style={'color':'#cc8899','fontSize':'.67rem','marginBottom':'3px'}),
                    kv("Articles",cd[1]),kv("Tone",f"{float(cd[2]):.1f}")]
    elif '🔥' in txt:
        children = [html.Div("🔥  FIRE HOTSPOT",className='inspect-title',style={'color':'#ff5500'}),
                    html.Div(txt[:80].replace('<b>','').replace('</b>','').replace('<br>','  '),style={'color':'#cc7733','fontSize':'.68rem'})]
    elif '🎯' in txt:
        children = [html.Div("🎯  THREAT ZONE",className='inspect-title',style={'color':'#ff8844'}),
                    html.Span("See threat leaderboard →",style={'color':'#664422','fontSize':'.65rem'})]
    elif '🔲' in txt:
        children = [html.Div("🔲  GEOFENCE ZONE",className='inspect-title',style={'color':'#88ff88'}),
                    html.Div(txt[:60].replace('<b>','').replace('</b>','').replace('<br>','  '),style={'color':'#44cc88','fontSize':'.67rem'})]
    else:
        children = [html.Div("📍 MAP LOCATION",className='inspect-title',style={'color':'#aaffcc'})]

    if lat is not None and lon is not None:
        children.append(html.Div(f"Lat {float(lat):.4f}  Lon {float(lon):.4f}",
                                 style={'color':'#1a3a22','fontSize':'.6rem','marginTop':'4px'}))
        wx_full = fetch_weather_point(float(lat),float(lon))
        wx = wx_full.get('current',{}) if isinstance(wx_full,dict) else {}
        if wx:
            code=int(wx.get('weather_code',-1))
            children.append(html.Div([
                html.Div("🌡 WEATHER",className='wx-title'),
                html.Div(WMO_CODES.get(code,f"Code {code}"),style={'color':'#aaffcc','marginBottom':'2px'}),
                html.Div([html.Span(f"🌡 {wx.get('temperature_2m','—')}°C  ",style={'color':'#ffaa44'}),
                          html.Span(f"💧 {wx.get('relative_humidity_2m','—')}%  ",style={'color':'#44aaff'}),
                          html.Span(f"💨 {wx.get('wind_speed_10m','—')}km/h",style={'color':'#88ccaa'})]),
            ],className='wx-box'))
            hourly=wx_full.get('hourly',{}) if isinstance(wx_full,dict) else {}
            temps=hourly.get('temperature_2m',[]); times=hourly.get('time',[])
            if temps and len(temps)>=24:
                spark=go.Figure(go.Scatter(x=[t[-5:] for t in times[:24]],y=temps[:24],mode='lines',
                    line=dict(color='#ff9944',width=1.5),fill='tozeroy',fillcolor='rgba(255,120,40,0.08)'))
                spark.update_layout(height=60,margin=dict(l=20,r=4,t=2,b=14),
                    paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False,tickfont=dict(size=6,color='#336644'),nticks=6),
                    yaxis=dict(showgrid=True,gridcolor='rgba(50,80,50,0.2)',tickfont=dict(size=6,color='#336644'),ticksuffix='°'),showlegend=False)
                children.append(html.Div([
                    html.Div("24h TEMP FORECAST",style={'color':'#336644','fontSize':'.55rem','marginTop':'2px'}),
                    dcc.Graph(figure=spark,config={'displayModeBar':False},style={'height':'60px','width':'100%'}),
                ],className='track-box'))

    if is_flight and selected_icao:
        trail=_cache.get_trail(selected_icao)
        if len(trail)>=2:
            times_t=[p[3] if len(p)>3 else f"t-{i}" for i,p in enumerate(trail)]
            alts=[p[2]/1000 for p in trail]
            tf=go.Figure(go.Scatter(x=times_t,y=alts,mode='lines+markers',line=dict(color='#00ffcc',width=1.5),marker=dict(size=4,color='#00ffcc')))
            tf.update_layout(height=68,margin=dict(l=24,r=4,t=2,b=16),
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False,tickfont=dict(size=6,color='#336644'),nticks=5),
                yaxis=dict(showgrid=True,gridcolor='rgba(50,80,50,0.2)',tickfont=dict(size=6,color='#336644'),ticksuffix='km'),showlegend=False)
            children.append(html.Div([
                html.Div(f"✈ ALTITUDE TRAIL — {selected_icao}",style={'color':'#336644','fontSize':'.55rem','marginTop':'2px'}),
                dcc.Graph(figure=tf,config={'displayModeBar':False},style={'height':'68px','width':'100%'}),
            ],className='track-box'))
    return children

# ── PLAYBACK ─────────────────────────────────────────────────
@app.callback(
    Output('store-pb','data'),
    Output('pb-slider','max'), Output('pb-slider','value'),
    Output('pb-status-text','children'), Output('pb-time-label','children'),
    Output('pb-mode-badge','children'), Output('pb-live-btn','className'),
    Input('pb-live-btn','n_clicks'), Input('pb-back-btn','n_clicks'),
    Input('pb-play-btn','n_clicks'), Input('pb-fwd-btn','n_clicks'),
    Input('pb-slider','value'), Input('tick-pb','n_intervals'),
    State('store-pb','data'),
    prevent_initial_call=False,
)
def cb_playback(n_live,n_back,n_play,n_fwd,slider_val,n_tick,pb_state):
    triggered=ctx.triggered_id
    pb=pb_state or {'live':True,'pos':-1,'playing':False}
    frames=_ring.list_frames(); total=len(frames); buf_s=_ring.stats()

    if triggered=='pb-live-btn': pb={'live':True,'pos':total-1,'playing':False}
    elif triggered=='pb-back-btn': cur=pb.get('pos',total-1); pb={'live':False,'pos':max(0,cur-1),'playing':False}
    elif triggered=='pb-fwd-btn': cur=pb.get('pos',total-1); pb={'live':False,'pos':min(total-1,cur+1),'playing':False}
    elif triggered=='pb-play-btn':
        if not pb.get('live',True): pb['playing']=not pb.get('playing',False)
    elif triggered=='pb-slider': pb={'live':False,'pos':int(slider_val or 0),'playing':False}
    elif triggered=='tick-pb':
        if pb.get('playing') and not pb.get('live') and total>0:
            nxt=pb.get('pos',0)+1
            pb['pos']=min(total-1,nxt)
            if pb['pos']>=total-1: pb['playing']=False

    cur_pos=max(0,min(total-1,int(pb.get('pos',total-1)))) if total>0 else 0
    pb['pos']=cur_pos
    sl_max=max(0,total-1); sl_val=cur_pos

    if 0<=cur_pos<len(frames):
        ts=frames[cur_pos].get('ts','—')
        time_label=f"Frame {cur_pos+1}/{total}  ·  {ts[11:19]+' UTC' if ts!='—' else '—'}"
    else: time_label="Buffer filling…"

    status=f"◉ REC  {buf_s['count']}/{buf_s['capacity']} frames  ·  {buf_s['fill_pct']:.0f}% full  ·  {buf_s['size_kb']:.0f} KB"

    if pb.get('live',True): badge=html.Span("◉ LIVE",style={'color':'#00ff44','fontWeight':'bold','fontSize':'.63rem'})
    elif pb.get('playing'): badge=html.Span("▶ PLAYING",style={'color':'#ff8844','fontWeight':'bold','fontSize':'.63rem'})
    else: badge=html.Span("⏸ PAUSED",style={'color':'#ffcc44','fontSize':'.63rem'})

    live_cls='pb-btn live-active' if pb.get('live',True) else 'pb-btn'
    return pb, sl_max, sl_val, status, time_label, badge, live_cls

# ── ANOMALY PANEL ────────────────────────────────────────────
@app.callback(Output('anomaly-panel','children'),Input('tick-anom','n_intervals'))
def cb_anomaly(n):
    status=_cache.get_anomaly_status()
    labels={'n_fl':'✈ AIRCRAFT','n_eq':'⚡ SEISMIC RATE','max_mag':'⚡ MAX MAGNITUDE','max_aq':'💨 MAX PM2.5'}
    if not status: return [html.P("Collecting baseline…",style={'color':'#224433','fontSize':'.62rem'})]
    rows=[]
    for m,lbl in labels.items():
        v=status.get(m,{})
        if not v: continue
        anom=v.get('anomaly',False); z=v.get('z',0); val=v.get('value',0); mean=v.get('mean',0); std=v.get('std',0)
        rows.append(html.Div([
            html.Span(('🔴 ' if anom else '🟢 ')+lbl,style={'fontWeight':'bold' if anom else 'normal','fontSize':'.63rem'}),
            html.Span(f"  {val:.1f}  z={z:+.1f}σ",style={'color':'#ffaa44' if anom else '#336644','marginLeft':'5px','fontSize':'.62rem'}),
            html.Br(),
            html.Span(f"baseline: {mean:.1f}±{std:.1f}",style={'color':'#2a4433','fontSize':'.57rem'}),
        ],className=f'anom-block {"anom" if anom else "ok"}'))
    return rows or [html.P("No data yet.",style={'color':'#224433'})]

# ── GEOFENCE PANEL ───────────────────────────────────────────
@app.callback(
    Output('gf-zone-list','children'),Output('gf-status','children'),Output('s-gf','children'),
    Input('gf-add-btn','n_clicks'),Input('tick-gf','n_intervals'),
    State('gf-name','value'),State('gf-lat-min','value'),State('gf-lat-max','value'),
    State('gf-lon-min','value'),State('gf-lon-max','value'),
)
def cb_geofence(n_add,n_tick,name,lat_min,lat_max,lon_min,lon_max):
    msg=""
    if ctx.triggered_id=='gf-add-btn' and n_add:
        if not name or any(v is None for v in [lat_min,lat_max,lon_min,lon_max]): msg="⚠ Fill all fields"
        else:
            try: msg=f"✓ Zone [{_geofence.add_zone(name,float(lat_min),float(lat_max),float(lon_min),float(lon_max))}] added"
            except Exception as e: msg=f"✗ {e}"
    zones=_geofence.get_zones()
    rows=[]
    for zn,z in zones.items():
        rows.append(html.Div([
            html.Span(f"🔲 {zn}",style={'color':'#88ff88','fontWeight':'bold','marginRight':'6px'}),
            html.Span(f"{z['lat_min']:.1f}–{z['lat_max']:.1f}°N  {z['lon_min']:.1f}–{z['lon_max']:.1f}°E",style={'color':'#336644','fontSize':'.6rem'}),
        ],className='gf-row'))
    if not rows: rows=[html.Div("No zones. Add below.",style={'color':'#224433','fontSize':'.62rem','padding':'3px 0'})]
    return rows, msg, str(len(zones))

# ── TELEMETRY CHARTS ─────────────────────────────────────────
@app.callback(
    Output('telem-flights','figure'),Output('telem-quakes','figure'),Output('telem-mag','figure'),
    Input('tick-telem','n_intervals'),
)
def cb_telemetry(n):
    frames=_ring.list_frames()
    def ef(title,color):
        fig=go.Figure(); fig.update_layout(height=75,margin=dict(l=28,r=4,t=14,b=16),
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=title,font=dict(size=8,color=color),x=0.02,y=0.98),
            xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            yaxis=dict(showgrid=True,gridcolor='rgba(50,80,50,0.18)',tickfont=dict(size=7,color='#336644'),zeroline=False),
            showlegend=False); return fig
    if not frames: return ef("✈ AIRCRAFT",'#00ff88'),ef("⚡ QUAKES",'#ff6633'),ef("⚡ MAX MAG",'#ff4400')
    times=[f['ts'][11:19] for f in frames]
    def mf(y,title,color,fill):
        fig=go.Figure(go.Scatter(x=times,y=y,mode='lines',line=dict(color=color,width=1.5),fill='tozeroy',fillcolor=fill))
        fig.update_layout(height=75,margin=dict(l=28,r=4,t=14,b=16),paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',title=dict(text=title,font=dict(size=8,color=color),x=0.02,y=0.98),
            xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            yaxis=dict(showgrid=True,gridcolor='rgba(50,80,50,0.18)',tickfont=dict(size=7,color='#336644'),zeroline=False),
            showlegend=False); return fig
    return (mf([f['n_fl'] for f in frames],"✈ AIRCRAFT",'#00ff88','rgba(0,255,136,0.07)'),
            mf([f['n_eq'] for f in frames],"⚡ QUAKES", '#ff6633','rgba(255,102,51,0.07)'),
            mf([f['max_mag'] for f in frames],"⚡ MAX MAG",'#ff4400','rgba(255,68,0,0.07)'))

# ── THREAT LEADERBOARD ───────────────────────────────────────
@app.callback(Output('threat-panel','children'),Input('tick-threat','n_intervals'))
def cb_threat(n):
    df=_cache.get_threat()
    if df.empty: return [html.P("Computing…",style={'color':'#224433','fontSize':'.62rem'})]
    rows=[]
    for i,(_,row) in enumerate(df.head(10).iterrows()):
        score=float(row['score']); color='#ff3333' if score>70 else '#ff8844' if score>40 else '#aacc44'
        rows.append(html.Div([
            html.Div([html.Span(f"#{i+1:02d}",style={'color':'#336644','marginRight':'4px','fontSize':'.6rem'}),
                      html.Span(f"{row['lat_bin']:.0f}°,{row['lon_bin']:.0f}°",style={'color':'#88ccaa','marginRight':'6px','fontSize':'.63rem'}),
                      html.Span(f"{score:.1f}",style={'color':color,'fontWeight':'bold','fontSize':'.68rem'})],
                     style={'display':'flex','alignItems':'center'}),
            html.Div(style={'height':'3px','width':f"{int(score)}%",
                            'background':f"linear-gradient(to right,{color}44,{color})",
                            'borderRadius':'2px','marginBottom':'2px'}),
        ],style={'borderBottom':'1px solid #0e1800','paddingBottom':'2px','marginBottom':'2px'}))
    return rows

# ── LOG PANEL ────────────────────────────────────────────────
@app.callback(Output('log-panel','children'),Input('store-snap','data'),Input('log-tabs','active_tab'),Input('eq-days-slider','value'))
def cb_log(snap,tab,eq_days):
    if not snap: return []
    eq_days=eq_days or 1
    if tab=='l-fl':
        try: df=pd.DataFrame(snap.get('flights',[]))
        except: df=pd.DataFrame()
        if df.empty: return [html.P("Awaiting ADS-B…",style={'color':'#224433'})]
        top=df.nlargest(80,'alt') if 'alt' in df.columns else df
        return [html.Div([html.Span(f"[{r.get('icao','?').upper()}]",style={'color':'#996600','marginRight':'4px'}),
                          html.Span(f"{str(r.get('callsign','?'))[:8].ljust(8)}",style={'color':'#00bb44','marginRight':'4px'}),
                          html.Span(f"↑{int(r.get('alt',0)):>6,}m",style={'color':'#009977','marginRight':'3px'}),
                          html.Span(f"{float(r.get('vel',0)):>5.1f}m/s",style={'color':'#336644'})],
                         className='log-row') for r in top.to_dict('records')]
    elif tab=='l-eq':
        if eq_days<=1:
            try: df=pd.DataFrame(snap.get('quakes',[]))
            except: df=pd.DataFrame()
        else:
            df30=_cache.get_quakes_30d()
            if df30.empty: df=pd.DataFrame(snap.get('quakes',[]))
            else:
                cutoff_ms=(datetime.utcnow()-timedelta(days=eq_days)).timestamp()*1000
                df=df30[df30['time_ms']>=cutoff_ms].head(MAX_EQ)
        if df.empty: return [html.P("No quakes.",style={'color':'#224433'})]
        top=df.nlargest(60,'mag') if 'mag' in df.columns else df
        return [html.Div([html.Span(f"M{float(r.get('mag',0)):.1f}",
                                   style={'color':'#ff3333' if float(r.get('mag',0))>=5 else '#ff6633','marginRight':'4px','fontWeight':'bold'}),
                          html.Span(f"{str(r.get('place','?'))[:26]}",style={'color':'#cc8844','marginRight':'4px'}),
                          html.Span(str(r.get('time','?')),style={'color':'#443322'})],
                         className='log-row') for r in top.to_dict('records')]
    elif tab=='l-sl':
        try: df=pd.DataFrame(snap.get('starlink',[]))
        except: df=pd.DataFrame()
        if df.empty: return [html.P("Install sgp4" if not HAS_SGP4 else "Loading TLEs…",style={'color':'#224433','fontSize':'.61rem'})]
        top=df.nlargest(80,'alt_km') if 'alt_km' in df.columns else df
        return [html.Div([html.Span(f"[{r.get('norad','?')}]",style={'color':'#776699','marginRight':'4px'}),
                          html.Span(f"{str(r.get('name','?')).replace('STARLINK-','SL-')[:12].ljust(12)}",style={'color':'#aaaaff','marginRight':'4px'}),
                          html.Span(f"↑{float(r.get('alt_km',0)):.0f}km",style={'color':'#667799'})],
                         className='log-row') for r in top.to_dict('records')]
    elif tab=='l-sh':
        ships=snap.get('ships',[])
        if not ships: return [html.P("No ships — add AIS_API_KEY",style={'color':'#224433','fontSize':'.61rem'})]
        return [html.Div([html.Span(f"[{r.get('mmsi','?')}]",style={'color':'#007799','marginRight':'4px'}),
                          html.Span(f"{str(r.get('name','?'))[:14].ljust(14)}",style={'color':'#00aacc','marginRight':'4px'}),
                          html.Span(f"{float(r.get('sog',0)):.1f}kn",style={'color':'#005566'})],
                         className='log-row') for r in ships[:60]]
    else:
        try: df=pd.DataFrame(snap.get('events',[]))
        except: df=pd.DataFrame()
        if df.empty: return [html.P("Loading GDELT…",style={'color':'#224433'})]
        top=df.nlargest(60,'count') if 'count' in df.columns else df
        return [html.Div([html.Span(f"[{float(r.get('tone',0)):+.1f}]",
                                   style={'color':'#ff4466' if float(r.get('tone',0))<-5 else '#668866','marginRight':'4px','fontSize':'.6rem'}),
                          html.Span(f"{str(r.get('title','?'))[:52]}",style={'color':'#cc8899','fontSize':'.6rem'})],
                         className='log-row') for r in top.to_dict('records')]

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║  GEO-SURVEILLANCE FEED  v{VERSION}  —  ADVANCED EDITION    ║
╠═══════════════════════════════════════════════════════════╣
║  🗺  PRIMARY:  Scattermapbox tile map (CartoDB Dark)      ║
║      NO CesiumJS · no token needed · zoom to street level ║
║  🌍  SECONDARY: Scattergeo global overview                ║
║  🌡  DENSITY:   Densitymapbox heatmap                     ║
╠═══════════════════════════════════════════════════════════╣
║  ✈  ADS-B aircraft    → OpenSky (no key)                 ║
║  ⚡  Seismic 30d       → USGS (no key)                    ║
║  🛰  Starlink ~150     → CelesTrak+SGP4 {'✓' if HAS_SGP4 else '✗ pip install sgp4'}         ║
║  🛸  ISS              → wheretheiss.at + pass predictor   ║
║  💨  Air quality       → OpenAQ v2 (no key baseline)      ║
║  📰  News events       → GDELT (no key)                   ║
║  🌧  Rain radar        → RainViewer (no key)              ║
║  🌡  Weather           → Open-Meteo (no key)              ║
║  🤖 Anomaly engine    → rolling z-score, 20-frame window  ║
║  🔲 Geofencing        → named bbox zones on tile map      ║
║  🔗 Convergence       → multi-hazard co-location alerts   ║
║  🕐 Ring buffer       → SQLite :memory:, 2h playback      ║
║  📊 Telemetry charts  → 2h rolling from ring buffer       ║
║  📡 SSE push          → /api/stream                       ║
║  ⬡  deck.gl WebGL     → /deck (GPU scatter + hexagon)     ║
║  🔥  Wildfires         → {'KEY SET' if FIRMS_MAP_KEY else 'add FIRMS_MAP_KEY':<36s}║
║  🚢  Ships AIS         → {'KEY SET' if AIS_API_KEY else 'add AIS_API_KEY':<36s}║
╠═══════════════════════════════════════════════════════════╣
║  Dashboard:  http://127.0.0.1:{PORT}                       ║
║  WebGL:      http://127.0.0.1:{PORT}/deck                  ║
║  API v1:     http://127.0.0.1:{PORT}/api/v1/               ║
║  Health:     http://127.0.0.1:{PORT}/healthz               ║
║  Metrics:    http://127.0.0.1:{PORT}/metrics               ║
║  Gunicorn:   gunicorn -w 1 gsf_v10:application            ║
╚═══════════════════════════════════════════════════════════╝
""")
    try:
        app.run(debug=False, host=HOST, port=PORT)
    except KeyboardInterrupt:
        print("\n[*] Shutting down…")
        cams.stop()
        sys.exit(0)
