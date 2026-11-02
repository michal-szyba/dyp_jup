# Visual Absolute Positioning System for UAVs in GNSS-Denied Environments

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-SIFT-green.svg)
![Rasterio](https://img.shields.io/badge/Rasterio-GIS-orange.svg)
![Domain](https://img.shields.io/badge/Domain-UAV%20%2F%20GNSS--Denied%20Navigation-red.svg)

##  Context & Motivation

Unmanned Aerial Vehicles (UAVs) operating in modern tactical, industrial, or search-and-rescue scenarios face critical vulnerabilities due to **GNSS signal disruption, spoofing, and electronic warfare**. Traditional navigation reliance on GNSS creates single-point failure vectors. While alternative systems like Inertial Navigation Systems (INS) provide short-term autonomy, they suffer from cumulative drift over time. 

This project addresses these constraints by building an **autonomous, completely passive visual navigation framework** that localizes the platform without relying on external radio signals or ground infrastructure.

---

##  System Principles & Trade-offs

* **Passive Operational Security:** The system operates purely on onboard optical sensors—emitting no radio signature, making it fully immune to electronic warfare and signal jamming.
* **Absolute Position Drift Reset:** By continuously matching live aerial frames against a georeferenced reference map, the system computes absolute coordinates (Latitude/Longitude or UTM) and resets INS cumulative drift.
* **Reference Map Requirements:** Requires prior availability of a georeferenced satellite/orthophoto base map of the flight corridor, where localization performance directly correlates with map recency and terrain feature density.

---

##  Architecture & Pipeline

'''
 [ Onboard UAV Camera ] 
          │
          ▼ (Real-time Aerial Image Stream)
 [ Dynamic Cropping & Multi-Scale Feature Detection ] <── (Georeferenced Reference Base Map)
          │
          ▼
 [ Robust Feature Matching & Homography ]
          │
          ▼
 [ Spatial Geo-Transformation ]
          │
          ▼
 [ Absolute Position Output & INS Correction ]
'''

### Key Technical Mechanics:
* **Spatial Cropping & Georeferencing:** To enable high-performance matching without processing massive base maps in real time, reference tiles are cropped dynamically while preserving native Spatial Reference Systems (CRS).
* **Advanced Error Handling & Validation:** Includes verification functions to handle edge cases—such as flying over low-feature terrain (water, dense forest, snow), detecting track deviations, and estimating vector direction relative to known geographical landmarks.

---

##  Tech Stack

* **Language:** **Python 3.10+** — Used for high-level prototyping, algorithm evaluation, and data pipeline design. *(Note: Performance-critical loops are targeted for future C++ refactoring).*
* **Computer Vision:** **OpenCV** — High-performance image processing, keypoint detection (SIFT), and vector-matrix calculations.
* **GIS & Spatial Operations:** **Rasterio** — Geospatial raster I/O, dynamic cropping, CRS transformations, and geo-coordinate resolution.
* **Numerical Processing:** **NumPy & SciPy** — Multi-dimensional matrix transformations and spatial calculations.

---

##  Experimental & Evaluation Framework

The repository incorporates a flexible testing module to benchmark algorithmic resilience under real-world conditions:
1. **Multi-GSD & Resolution Testing:** Evaluates matching accuracy across varying Ground Sample Distance (GSD) ratios between onboard camera frames and reference maps.
2. **Environmental & Noise Simulations:** Benchmarks matching confidence against synthetic noise, motion blur, rotation, and lighting changes.
3. **Algorithm Comparison Pipeline:** Modular architecture allowing head-to-head performance evaluations of classic feature extractors against modern deep learning feature pipelines.

---

##  Author & Engineering Thesis

* **Developer:** Michał Szyba
* **Degree Program:** B.Sc. Eng. in Geoinformatics (Geoinformatyka), Cracow University of Technology (Politechnika Krakowska)
* **Thesis Title:** *Absolute positioning system for unmanned aerial vehicles in GNSS-denied environments*
* **Contact:** mszyba@onet.pl | [LinkedIn](https://www.linkedin.com/in/mszyba/) | [GitHub](https://github.com/michal-szyba)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
