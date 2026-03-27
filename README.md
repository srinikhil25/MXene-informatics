# Materials Informatics

**From Raw Spectra to Cross-Technique Insight: An Autonomous Informatics Platform for Multi-Modal Materials Characterization**

An open-source Python platform that transforms raw experimental characterization data (XRD, XPS, SEM, EDS/EDX) from multiple instrument vendors into structured, FAIR-compliant datasets with automated scientific analysis, cross-technique feature extraction, and interactive visualizations.

> **Paper:** *"From Raw Spectra to Cross-Technique Insight: An Autonomous Informatics Platform for Multi-Modal Materials Characterization"* — manuscript in preparation for Digital Discovery (RSC)

---

## Highlights

- **957 raw files** from 4 instrument vendors parsed autonomously
- **185 unique samples** across 6+ material families (MXene, CF, CAF, BFO, Bi2Se3, Bi2Te3, ...)
- **4 characterization techniques** (XRD + XPS + SEM + EDX) in one unified pipeline
- **Universal ETL**: Rigaku XRD, JEOL FE-SEM, Hitachi HR-FE-SEM, Bruker Quantax EDX, PHI XPS
- **Rietveld refinement** with full atomic site parameters and lattice parameter determination
- **XPS deconvolution** with spin-orbit coupling, Shirley/Tougaard backgrounds, and DOI-linked references
- **Automated SEM morphological analysis** — particle/flake sizing, layer thickness, surface roughness
- **Cross-technique ML**: 33 features x 308 samples, PCA clustering, family-level correlation
- **Interactive Streamlit dashboard** with material family filters and real-time parameter tuning
- **~5,500 lines** of modular, extensible Python code
- **Replaces ~$8k** in commercial software (CasaXPS, HighScore, ImageJ)

---

## Architecture

```
+-------------------------------------------------------------------+
|  Layer 1: Universal Data Engineering (ETL)                        |
|  957 raw files from 4 vendors -> Standardized JSON/CSV            |
|  Rigaku XRD | JEOL SEM | Hitachi SEM | Bruker EDX | PHI XPS      |
+-------------------------------------------------------------------+
|  Layer 2: Scientific Analysis                                     |
|  +----------------+ +----------------+ +----------------+         |
|  |  XRD Analysis   | |  XPS Analysis  | |  SEM Analysis  |         |
|  |  - Peak fitting | |  - Shirley BG  | |  - Otsu/Adapt  |         |
|  |  - Rietveld     | |  - GL deconv   | |  - Watershed   |         |
|  |  - Phase ID     | |  - Spin-orbit  | |  - Flake size  |         |
|  |  - Scherrer     | |  - Chem state  | |  - Layer thick |         |
|  |  - d-spacing    | |  - Export CSV  | |  - Roughness   |         |
|  +----------------+ +----------------+ +----------------+         |
+-------------------------------------------------------------------+
|  Layer 3: Cross-Technique ML                                      |
|  33 features x 308 samples | PCA clustering | Family correlation  |
|  Parallel coordinates | Feature distributions | Material families |
+-------------------------------------------------------------------+
|  Layer 4: Interactive Dashboard (Streamlit)                       |
|  6 pages | Material family filters | Color customization          |
+-------------------------------------------------------------------+
|  Layer 5: Agentic Interface (Planned)                             |
|  RAG over literature | Dynamic reference assignment               |
+-------------------------------------------------------------------+
```

---

## Key Results

| Analysis | Result |
|----------|--------|
| **Universal ETL** | 957 files from 4 vendors -> 185 samples, 18 material families |
| **XRD Patterns** | 124 patterns available for interactive multi-family comparison |
| **Rietveld Refinement** | a = 3.074 A, c = 18.572 A (delta < 0.01 A from ICDD 52-0875) |
| **XPS Composition** | C: 64.06%, O: 23.06%, Ti: 6.08%, F: 6.80% (MXene Ti3C2Tx) |
| **Ti 2p Deconvolution** | 10 components (5 x 2p3/2 + 5 x 2p1/2), R2 = 0.962 |
| **SEM Images** | 169 images (132 JEOL + 37 Hitachi) across multiple families |
| **Flake Size** | Mean 1.68 um, 267 particles (MXene, adaptive segmentation) |
| **EDX Spectra** | 27 Bruker spectra with 40+ X-ray line identification |
| **Cross-Technique ML** | 33 features extracted, PCA separates material families |

---

## Directory Structure

```
Materials-Informatics/
+-- src/
|   +-- etl/                        # Data parsers (~1,500 lines)
|   |   +-- xrd_parser.py           # Rigaku .txt format
|   |   +-- xps_parser.py           # PHI XPS spectra + quantification
|   |   +-- sem_parser.py           # Hitachi SU8600 UTF-16-LE metadata
|   |   +-- jeol_sem_parser.py      # JEOL FE-SEM $KEY VALUE format
|   |   +-- bruker_edx_parser.py    # Bruker .spx (XML) + .xls (Excel)
|   |   +-- eds_parser.py           # EMSA/MAS format with peak ID
|   |   +-- universal_etl.py        # Master orchestrator for all formats
|   +-- analysis/                    # Scientific analysis (~2,200 lines)
|   |   +-- xrd_analysis.py         # Peak fitting, Scherrer, phase ID
|   |   +-- xps_analysis.py         # Shirley BG, GL deconvolution, quantification
|   |   +-- sem_analysis.py         # Segmentation, flake sizing, layer thickness
|   |   +-- rietveld.py             # Whole-pattern refinement, atomic sites
|   +-- ml/                          # Cross-technique ML (~700 lines)
|   |   +-- feature_extraction.py   # 33 features from XRD/SEM/EDX
|   |   +-- sample_matcher.py       # Material family classification + correlation
|   |   +-- correlation_plots.py    # Heatmap, radar, bar visualizations
|   |   +-- __init__.py
+-- data/
|   +-- raw/                         # 957 raw instrument files
|   +-- processed/
|       +-- universal/               # Universal ETL output
|       |   +-- xrd/                 # 124 patterns (JSON + CSV)
|       |   +-- sem/                 # JEOL + Hitachi catalogs
|       |   +-- edx/                 # 27 Bruker spectra + quantification
|       |   +-- sample_catalog.json  # 185 unified sample records
|       +-- features/                # ML feature matrices + correlations
|       +-- xps/                     # XPS spectra (MXene only)
+-- paper/
|   +-- paper_structure.md           # Full paper outline + reviewer Q&A
+-- app.py                           # Streamlit dashboard (~2,700 lines)
+-- run_etl.py                       # ETL orchestrator
+-- agents.md                        # Planned AI agents (TODO)
+-- CLAUDE.md                        # AI assistant context
+-- requirements.txt                 # Python dependencies
+-- .gitignore
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/srinikhil25/Materials-Informatics.git
cd Materials-Informatics

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Universal ETL pipeline
python -c "from src.etl.universal_etl import run_universal_etl; run_universal_etl()"

# 5. Run cross-technique feature extraction
python -c "from src.ml.feature_extraction import build_feature_matrix; build_feature_matrix()"
python -c "from src.ml.sample_matcher import run_cross_technique_analysis; run_cross_technique_analysis()"

# 6. Launch the dashboard
streamlit run app.py
```

---

## Dashboard Pages

### Overview
- Platform summary with live metrics (124 XRD, 169 SEM, 27 EDX, 185 total samples)
- Pipeline architecture visualization
- XPS surface composition (MXene case study)

### XRD Analysis
- **124 patterns** across all material families with dynamic selection
- Material family filter — compare MXene vs CF vs CAF vs BFO patterns
- Up to 6 patterns overlaid with stacking and normalization controls
- **Rietveld refinement**: Y_obs, Y_calc, Y_diff, Bragg tick marks, atomic site table
- Automated peak fitting: Gaussian, Lorentzian, Pseudo-Voigt profiles
- Phase identification, Scherrer crystallite size, d-spacing calculator

### XPS Analysis
- Survey + 4 high-resolution spectra (Ti 2p, C 1s, O 1s, F 1s)
- Peak deconvolution: Shirley/Linear/Tougaard backgrounds, adjustable GL mixing
- Spin-orbit coupling: Ti 2p3/2 + 2p1/2 doublets (delta_BE = 5.7 eV, area ratio 2:1)
- Chemical state assignment with literature DOI references
- Component editor: add/remove/toggle components and re-fit
- CSV export of fitted deconvolution data

### SEM Gallery
- **169 images** from JEOL FE-SEM + Hitachi HR-FE-SEM
- Filter by instrument, material family, accelerating voltage
- Magnification vs pixel size scatter (colored by family, shaped by instrument)
- Morphological analysis: particle sizing, aspect ratio, layer thickness, roughness
- Segmentation visualization: binary masks + Canny edge detection

### EDS/EDX Analysis
- **EMSA + Bruker SPX** data sources merged in one view
- 27 Bruker EDX spectra with 40+ X-ray line identification
- 16 Bruker quantification entries (wt%, at%, error)
- Elemental composition stacked bar chart
- Universal element tracking across all spectra

### Cross-Technique ML
- Universal ETL summary with pipeline statistics
- Material family distribution table with technique availability
- Cross-technique correlation heatmap
- **PCA Clustering**: 3-panel scatter plots (XRD, SEM, EDX) colored by family
- **Parallel Coordinates**: top features by variance across families
- **Feature Distributions**: 6 individual box plots per technique in 3x2 grid

---

## Instruments Supported

| Technique | Instrument | Format | Encoding |
|-----------|-----------|--------|----------|
| XRD | Rigaku Ultima3 | .txt (Rigaku) | ASCII |
| SEM | JEOL FE-SEM | .txt ($KEY VALUE) | ASCII |
| SEM | Hitachi HR-FE-SEM (SU8600) | .txt | UTF-16-LE with BOM |
| EDX | Bruker Quantax XFlash 5010 | .spx (XML) + .xls | Windows-1252 / Excel |
| XPS | PHI | .txt | ASCII |
| EDS | JEOL TEM-EDX | .emsa | ASCII (EMSA standard) |

---

## Material Families

The platform processes samples from 18+ material families, including:

- **MXene_Ti3C2**: Ti3AlC2 MAX phase and Ti3C2Tx MXene
- **CF_Conductive_Fabric**: Conductive fabric samples (CF-1 through CF-7)
- **CAF_Carbon_Fabric**: Carbon/activated carbon fabrics
- **BFO_BiFeO3**: Bismuth ferrite and variants (CoBFO, ZnBFO)
- **Bi2Se3, Bi2Te3**: Bismuth chalcogenides
- **NiCu_Fabric**: Nickel-copper conductive fabrics
- **AgCu_Alloy**: Silver-copper alloys
- And more (Mo compounds, Zn compounds, PVA, Gel, Cotton, Nylon, Polyester...)

---

## Scientific Methods

### XRD
- Peak detection: Savitzky-Golay smoothing + `scipy.signal.find_peaks()`
- Peak fitting: Gaussian / Lorentzian / Pseudo-Voigt via `scipy.optimize.curve_fit()`
- Rietveld refinement: `scipy.optimize.least_squares()` with crystal structure models
- Phase ID: 6-phase reference database (Ti3AlC2, Ti3C2Tx, TiO2, TiC, Al2O3)
- Scherrer equation: L = Kl / (b*cos(theta)), K = 0.9

### XPS
- Backgrounds: Shirley (iterative), Linear, Tougaard (universal cross-section)
- Peak function: Gaussian-Lorentzian product with adjustable mixing ratio
- References: 17 chemical states across Ti 2p, C 1s, O 1s, F 1s (with DOI links)
- Spin-orbit: Automatic 2p3/2 / 2p1/2 doublet generation

### SEM
- Segmentation: Otsu, Adaptive thresholding, Watershed
- Preprocessing: CLAHE, Gaussian denoising, scale bar cropping
- Measurements: Equivalent diameter, aspect ratio, circularity, solidity
- Layer thickness: Gradient-based boundary detection on cross-section images

### EDX
- Bruker SPX: 4096-channel XML parsing with energy calibration
- Peak identification: 40+ known X-ray emission lines (B through Mo)
- Quantification: Bruker XLS parsing (element, wt%, at%, error)

### Cross-Technique ML
- Feature extraction: 33 automated features (XRD 12, EDX 11, SEM 4, metadata 6)
- Family classification: 25 regex patterns -> 18 canonical material families
- PCA clustering: StandardScaler + 2-component PCA per technique
- Correlation: Pearson cross-technique correlation at family level

---

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `scikit-image`, `scikit-learn`
Visualization: `plotly`
Dashboard: `streamlit`
EDX parsing: `xlrd` (for Bruker .xls files)

See `requirements.txt` for full list.

---

## Future Work

- **Agent 1**: Dynamic XPS reference assignment via NIST XPS Database + literature queries
- **Agent 2**: RAG-powered literature Q&A
- **Agent 3**: Synthesis optimization recommendations
- **ML surrogate model**: Synthesis parameters -> property prediction
- **GSAS-II integration**: Full Rietveld refinement with preferred orientation
- **ML segmentation**: U-Net / Mask R-CNN for SEM instance segmentation
- **Technique expansion**: Raman, FTIR, TGA parser modules
- **Multi-material scaling**: Grow dataset with publicly available characterization data

---

## Related Project

[MXDiscovery](https://github.com/srinikhil25/MXDiscovery) — Computational discovery pipeline for novel non-toxic MXene composites using literature mining, ML screening, DFT validation (Quantum ESPRESSO), and TOPSIS ranking for wearable thermoelectric applications.

---

## Authors

**Gudibandi Sri Nikhil Reddy** (1st Author)
Masters Student,
Ikeda - Hamasaki Laboratory,
Research Institute of Electronics,
Shizuoka University, Japan
Research Focus: AI Development in Materials Science, MXenes, Wearable Thermoelectrics, Materials Informatics

**[Senior's Name]** (2nd Author)
*Raw experimental data provider*

---

## License

MIT License
