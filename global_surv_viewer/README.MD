# Geo-Surveillance Feed (GSF)

A live, multi-layer geospatial intelligence dashboard built with Dash, Plotly, and Flask. It combines public data feeds into a single interactive map and supporting panels for tracking aircraft, earthquakes, fires, ships, the ISS, air quality, news events, geofences, anomalies, and threat hotspots.

## What it does

GSF continuously polls multiple sources and renders them in a browser-based dashboard:

- **Aircraft tracking** from OpenSky
- **Earthquakes** from USGS
- **Wildfires** from NASA FIRMS
- **Ships / AIS** from AISStream
- **ISS position and pass prediction**
- **Air quality** from OpenAQ
- **News / incident events** from GDELT
- **Nearby events and venues** from Ticketmaster and OpenStreetMap
- **Geofencing** with ingress / egress alerts
- **Anomaly detection** using rolling z-scores
- **Convergence detection** for overlapping hazards
- **Ring-buffer playback** for recent history
- **SSE live updates** and a small API surface for external use

The dashboard also includes a **2D tile map**, a **global overview**, and a **density view** for layering and comparison.

## Main features

### Live map views
- Tile map with zoom and pan
- Layer toggles for flights, quakes, fires, ships, Starlink, ISS, air quality, events, threat scores, geofences, trails, radar, and POIs
- Optional deck.gl WebGL view at `/deck`
- Basemap switching between dark, light, and street styles

### Intelligence modules
- **Threat scoring**: aggregates seismic, air quality, conflict, and fire signals into a normalized 0–100 score
- **Anomaly detection**: tracks spikes in flights, earthquakes, magnitude, and PM2.5
- **Geofencing**: named bounding boxes with enter/exit notifications
- **Convergence alerts**: warns when 3+ hazard layers overlap in the same grid cell
- **ISS pass predictor**: estimates upcoming passes from the user’s location

### Playback and telemetry
- In-memory SQLite ring buffer for recent snapshots
- Telemetry charts showing recent aircraft count, quake count, and max quake magnitude
- Snapshot and frame listing endpoints for replay / inspection

### Cameras
- Supports up to 4 MJPEG camera sources
- Live camera swapping from the UI
- Fallback image shown when a source is offline

### Search and POI tools
- Location search via Nominatim
- Nearby POI discovery via OpenStreetMap Overpass
- Nearby event lookup using Ticketmaster and OSM venues

## Data sources

GSF can work with mostly free public APIs, with some optional keys:

- OpenSky
- USGS
- NASA FIRMS
- AISStream
- Open-Meteo
- RainViewer
- OpenAQ
- GDELT
- CelesTrak + SGP4
- Ticketmaster Discovery API
- OpenStreetMap / Overpass / Nominatim

Optional environment variables and config keys include:

- `AIS_API_KEY`
- `FIRMS_MAP_KEY`
- `OPENAQ_API_KEY`
- `TICKETMASTER_API_KEY`

## Configuration

Run this once to create `gsf.ini`:

```bash
python3 gsf_v12.py --init-config
```

Then edit the generated file and fill in any API keys you need.

## Installation

Install the dependencies used by the script:

```bash
pip install dash dash-bootstrap-components pandas plotly requests opencv-python numpy websocket-client sgp4
```

## Running

```bash
python3 gsf_v12.py
```

Default host and port come from `gsf.ini`:

- Host: `0.0.0.0`
- Port: `8050`

## Production / WSGI

The app exposes a Flask server object for WSGI deployment:

```bash
gunicorn -w 1 -b 0.0.0.0:8050 gsf_v12:application
```

## Important endpoints

- `/` — main dashboard
- `/deck` — WebGL view
- `/api/v1/` — API index
- `/api/stream` — SSE live stream
- `/api/snapshot` — current snapshot
- `/api/alerts` — active alerts
- `/api/export.geojson` — GeoJSON export
- `/api/weather?lat=&lon=` — weather at a point
- `/api/trail/<icao>` — aircraft trail
- `/api/frames/list` — ring-buffer frame list
- `/api/frame/<idx>` — single stored frame
- `/api/buffer/stats` — ring-buffer stats
- `/healthz` — health check
- `/metrics` — Prometheus-style metrics

## Notes

- The app is designed around public OSINT-style data feeds.
- Some modules are optional and degrade gracefully when a key or dependency is missing.
- If `sgp4` is unavailable, Starlink and ISS pass prediction features will be limited.
- The interface is optimized for a dark terminal-style dashboard.

## License

Add your preferred license here before publishing.

---

Built from `gsf_v12.py`.
