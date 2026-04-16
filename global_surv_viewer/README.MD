# GEO-SURVEILLANCE FEED  ·  v0.1

> Public OSINT Edition — Test Build for Kali Linux

---

## Quick Start (Kali Linux)

```bash
# 1. Make the installer executable and run it
chmod +x install_kali.sh && bash install_kali.sh

# 2. Launch the app
python3 gsf_v01.py

# 3. Open your browser
http://127.0.0.1:8050
```

---

## What's Live in v0.1

| Layer | Source | Notes |
|---|---|---|
| ✈ ADS-B Aircraft | OpenSky Network | ~400 live targets, no API key |
| ⚡ Earthquakes M2.5+ | USGS Feed | Last 24h, auto-refreshes |
| 📹 Public Camera | Open MJPEG streams | Auto-fallback if offline |

---

## UI Features

- **Flat map** or **Orthographic globe** toggle (tabs)
- Aircraft colored by **altitude** (dark green → bright cyan)
- Earthquakes sized and colored by **magnitude**
- **Signal intel log** — top 60 aircraft by altitude
- Live **UTC clock** and stat counters
- Camera with **UTC timestamp overlay** and LIVE badge
- Scanline CRT aesthetic

---

## Roadmap (next versions)

- [ ] v0.2 — AIS maritime ship tracking (aisstream.io WebSocket)
- [ ] v0.3 — NASA FIRMS active wildfire layer
- [ ] v0.4 — Weather radar overlay (Open-Meteo)
- [ ] v0.5 — GDELT event heatmap
- [ ] v0.6 — Time slider / playback
- [ ] v1.0 — CesiumJS 3D globe migration + Gaussian Splat terrain

---

## Notes

- All data sources are **public domain / unauthenticated** free tiers
- OpenSky may rate-limit after ~10 req/min without an account
- Camera streams are public OSINT sources; they go offline randomly — the app auto-retries
- Tested on: Kali Linux 2024.x, Python 3.11+






















/

















# Eagle Eye Feature Gap Analysis

This document compares the current Geo-Surveillance Feed v0.2 prototype against the full Eagle Eye 3D Earth viewer specification.

| Feature Category | Present in v0.2? | Missing / Next Steps |
| :--- | :---: | :--- |
| **Core Rendering** | | |
| Global 3D Globe (CesiumJS) | ❌ | Plotly Scattergeo is 2D map projection. Need full 3D globe with terrain. |
| Seamless Level-of-Detail | ❌ | Not applicable in 2D. |
| Atmospheric scattering / clouds | ❌ | Not in scope for Plotly. |
| **Data Layers** | | |
| Live Air Traffic | ✅ | Good coverage via OpenSky. |
| Live Earthquakes | ✅ | USGS feed integrated. |
| Live Wildfires | ✅ | NASA FIRMS (VIIRS) working. |
| Maritime Traffic (AIS) | ⚠️ | Placeholder (stub). Need WebSocket integration. |
| Weather Overlay (radar, clouds, wind) | ❌ | Entirely missing. |
| Traffic Flow (road congestion) | ❌ | Missing. |
| Points of Interest (POI) | ❌ | Missing. |
| Air Quality (AQI) | ❌ | Missing. |
| **User Interaction** | | |
| Click-to-Inspect | ✅ | Implemented for aircraft/earthquakes. |
| Time Slider (historical/forecast) | ❌ | Critical for "Eagle Eye" analysis. |
| Search / Geocoding | ❌ | Missing. |
| Measurement Tools | ❌ | Missing. |
| URL-based state sharing | ❌ | Missing. |
| **Advanced Technicals** | | |
| View-dependent data loading | ⚠️ | Partially via altitude filter, but not viewport bounds. |
| WebGPU acceleration | ❌ | Dash/Plotly uses WebGL under the hood, not WebGPU. |
| WebSocket streaming | ⚠️ | Planned for AIS; not yet implemented. |
| **Differentiators** | | |
| AR Pass-Through | ❌ | Requires mobile integration. |
| Procedural trees / streetlights | ❌ | Not applicable. |
| AI Visual Narration | ❌ | Not yet. |
