# 🌍 Global Surveillance Feed – The All-Seeing Earth Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![Cesium](https://img.shields.io/badge/CesiumJS-1.113+-brightgreen.svg)](https://cesium.com/platform/cesiumjs/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **The most advanced open‑source 3D Earth viewer.**  
> Combines real‑time satellite imagery, 3D terrain, global building models, street‑level photos, and thermal overlays into a single, high‑performance application.

![Global Surveillance Banner](./public/banner.png)

---

## 🚀 **Vision**

Transform a simple 3D globe script into a **full‑fledged, insanely detailed digital twin of Earth**.  
We aggregate free and public geospatial data from NASA, ESA, OpenStreetMap, Mapillary, and weather satellites to deliver an experience that rivals commercial platforms—completely open and extensible.

**Goal:** *"The most advanced global thermal sonar live feed of all Earth."*

---

## ✨ **Key Features**

- **🌐 High‑Performance 3D Globe** – Powered by [CesiumJS](https://cesium.com/platform/cesiumjs/) with global 3D terrain and smooth camera controls.
- **🏢 Worldwide 3D Buildings** – Over 350 million buildings extruded from OpenStreetMap data.
- **🛰️ Daily Satellite Imagery** – NASA GIBS layers (MODIS, VIIRS) and Sentinel‑2 cloudless mosaics.
- **🚶 Street‑Level View** – Integration with [Mapillary](https://www.mapillary.com/developer) for crowdsourced ground‑level photos.
- **🌡️ Thermal & Real‑Time Overlays** – Near real‑time infrared from geostationary satellites (Geosatellite API) and active fire/flood layers from NASA.
- **🔍 Global Search** – Geocoding via OpenStreetMap Nominatim.
- **⚡ Modern Stack** – React + TypeScript + Vite for rapid development and deployment.

---

## 🛠️ **Tech Stack**

| Area            | Technology                                                                 |
| :-------------- | :-------------------------------------------------------------------------- |
| **3D Engine**   | [CesiumJS](https://cesium.com/platform/cesiumjs/)                           |
| **Framework**   | React 18 + TypeScript                                                       |
| **Build Tool**  | Vite (with `vite-plugin-cesium`)                                            |
| **Data Sources**| NASA GIBS, OpenStreetMap, Mapillary, Geosatellite API, Sentinel Hub, USGS   |
| **Backend**     | (Optional) Node.js / Python Flask for API aggregation and caching            |

---

## 📦 **Installation**

### **Prerequisites**
- Node.js 18+ and npm
- A free [Cesium ion](https://ion.cesium.com/) access token (for terrain and OSM buildings)
- (Optional) API keys for Mapillary and other services

### **Setup**

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/global-surveillance-feed.git
   cd global-surveillance-feed
