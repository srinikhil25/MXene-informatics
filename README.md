# MXene-Informatics

**Autonomous Multi-Modal Characterization Pipeline for Ti₃C₂Tₓ MXene**

An open-source Python pipeline that transforms raw experimental characterization data (XRD, XPS, SEM, EDS) from Ti₃AlC₂ → Ti₃C₂Tₓ MXene synthesis into structured, FAIR-compliant datasets with automated scientific analysis, Rietveld refinement, and interactive visualizations.

> **Paper:** *"MXene-Informatics: An Open-Source Autonomous Multi-Modal Characterization Pipeline for Ti₃C₂Tₓ MXene"* — manuscript in preparation

---

## Highlights

- **4 characterization techniques** integrated in one pipeline (XRD + XPS + SEM + EDS)
- **Rietveld refinement** with full atomic site parameters and lattice parameter determination
- **XPS deconvolution** with spin-orbit coupling, Shirley/Tougaard backgrounds, and literature-linked chemical state assignments
- **Automated SEM morphological analysis** — particle/flake sizing, layer thickness, surface roughness
- **Interactive Streamlit dashboard** with real-time parameter tuning and color customization
- **~4,500 lines** of modular, extensible Python code
- **Replaces ~$8k** in commercial software (CasaXPS, HighScore, ImageJ)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Engineering (ETL)                                │
│  Raw instrument files → Standardized JSON/CSV                   │
│  Parsers: XRD (.txt), XPS (.txt), SEM (.txt), EDS (.emsa)      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Scientific Analysis                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ XRD Analysis  │ │ XPS Analysis │ │ SEM Analysis │            │
│  │ • Peak fitting│ │ • Shirley BG │ │ • Otsu/Adapt │            │
│  │ • Rietveld    │ │ • GL deconv  │ │ • Watershed  │            │
│  │ • Phase ID    │ │ • Spin-orbit │ │ • Flake size │            │
│  │ • Scherrer    │ │ • Chem state │ │ • Layer thick│            │
│  │ • d-spacing   │ │ • Export CSV │ │ • Roughness  │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Interactive Dashboard (Streamlit)                     │
│  6 pages: Overview | XRD | XPS | SEM | EDS | Data Export       │
│  Color customization | Real-time parameter tuning | Downloads   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Agentic Interface (Planned)                           │
│  RAG over MXene literature | Dynamic reference assignment       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results (Ti₃AlC₂ → Ti₃C₂Tₓ)


| Analysis                 | Result                                                              |
| ------------------------ | ------------------------------------------------------------------- |
| **XRD Phase ID**         | 13 peaks detected, 10 matched to Ti₃AlC₂ MAX (76% avg confidence)   |
| **Rietveld Refinement**  | a = 3.074 Å, c = 18.572 Å (Δ < 0.01 Å from ICDD 52-0875)            |
| **Crystallite Size**     | ~20.7 nm at (002) peak via Scherrer equation                        |
| **XPS Composition**      | C: 64.06%, O: 23.06%, Ti: 6.08%, F: 6.80%                           |
| **Ti 2p Deconvolution**  | 10 components (5 × 2p₃/₂ + 5 × 2p₁/₂), R² = 0.962                   |
| **Surface Terminations** | -O/-OH dominant (23% O), -F secondary (7% F)                        |
| **Flake Size**           | Mean 1.68 μm, median 1.23 μm (267 particles, adaptive segmentation) |
| **Layer Thickness**      | Mean 6.34 ± 4.62 nm (135 boundaries at 100k×)                       |
| **EDS Al/Ti Ratio**      | Decreasing trend confirms successful Al removal                     |


---

## Directory Structure

```
MXene-Informatics/
├── src/
│   ├── etl/                    # Data parsers (864 lines)
│   │   ├── xrd_parser.py       # Rigaku .txt format
│   │   ├── xps_parser.py       # PHI XPS spectra + quantification
│   │   ├── sem_parser.py       # Hitachi SU8600 UTF-16-LE metadata
│   │   └── eds_parser.py       # EMSA/MAS format with peak ID
│   ├── analysis/               # Scientific analysis (~1,900 lines)
│   │   ├── xrd_analysis.py     # Peak fitting, Scherrer, phase ID
│   │   ├── xps_analysis.py     # Shirley BG, GL deconvolution, quantification
│   │   ├── sem_analysis.py     # Segmentation, flake sizing, layer thickness
│   │   └── rietveld.py         # Whole-pattern refinement, atomic sites
│   ├── ml/                     # Machine learning (planned)
│   └── visualization/          # (integrated into Streamlit dashboard)
├── data/
│   ├── raw/                    # Symlinks to instrument files
│   └── processed/              # Standardized JSON + CSV output
│       ├── xrd/                # 2 patterns (4,251 points each)
│       ├── xps/                # 6 spectra + quantification
│       ├── sem/                # 18 images cataloged
│       └── eds/                # 19 spectra with auto-detected peaks
├── paper/
│   └── paper_structure.md      # Full paper outline + reviewer Q&A
├── app.py                      # Streamlit dashboard (~1,800 lines)
├── run_etl.py                  # ETL orchestrator
├── agents.md                   # Planned AI agents (TODO)
├── CLAUDE.md                   # AI assistant context
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/srinikhil25/MXene-Informatics.git
cd MXene-Informatics

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the ETL pipeline
python run_etl.py              # All stages
python run_etl.py --stage xrd  # Specific stage

# 5. Launch the dashboard
streamlit run app.py
```

---

## Dashboard Features

### XRD Analysis

- **Pattern comparison**: Ti₃AlC₂ (MAX) vs Ti₃C₂Tₓ (MXene) with reference peak overlays
- **Rietveld refinement**: Y_obs, Y_calc, Y_obs−Y_calc, Bragg tick marks
- **Atomic site parameters**: Wyckoff positions, fractional coordinates, occupancy, U_iso
- **Automated peak fitting**: Gaussian, Lorentzian, Pseudo-Voigt profiles
- **Phase identification**: 6-phase reference database with confidence scoring
- **Scherrer crystallite size**: Bar chart visualization
- **d-spacing calculator**: Interactive Bragg's law tool

### XPS Analysis

- **Survey + 4 high-resolution spectra** (Ti 2p, C 1s, O 1s, F 1s)
- **Peak deconvolution**: Shirley/Linear/Tougaard backgrounds, adjustable GL mixing
- **Spin-orbit coupling**: Ti 2p₃/₂ + 2p₁/₂ doublets (ΔBE = 5.7 eV, area ratio 2:1)
- **Chemical state assignment**: Literature DOI-linked references for every component
- **Component quantification**: Pie chart + data table with gradient color scales
- **CSV export**: Download fitted deconvolution data (BE, raw, background, envelope, components)

### SEM Gallery

- **Multi-scale imaging**: 1k–200k× magnification coverage
- **Morphological analysis**: Automated particle/flake segmentation (Otsu, Adaptive, Watershed)
- **Flake size distribution**: Histogram with mean/median markers
- **Layer thickness**: Vertical profile analysis at high magnification
- **Surface roughness**: Ra, Rq, Rsk, Rku metrics
- **Segmentation visualization**: Binary masks + Canny edge detection
- **Column definitions**: Full imaging conditions table with parameter descriptions

### EDS Analysis

- **19 spectra** with automatic element identification (12 X-ray lines)
- **Al Kα / Ti Kα ratio tracking**: Color-coded etching completeness indicator
- **Interactive energy range** and log-scale controls

### Global Features

- **🎨 Color customization**: Per-graph color pickers via popover buttons
- **Gradient color scale bars**: Visual legends on all data tables
- **Data export**: CSV/JSON downloads for all processed data
- **Greek symbols**: Proper notation throughout (θ, λ, α, β, σ, η, χ²)

---

## Instruments


| Technique | Instrument     | Parameters                                        |
| --------- | -------------- | ------------------------------------------------- |
| XRD       | Rigaku Ultima3 | Cu Kα₁ (λ = 1.54056 Å), 40 kV / 40 mA, 2θ = 5–90° |
| XPS       | PHI            | 0–1200 eV, high-res: Ti 2p, C 1s, O 1s, F 1s      |
| SEM       | Hitachi SU8600 | 20 kV, 2560×1920 px, 1k–200k× magnification       |
| TEM-EDX   | JEOL           | 200 kV, 2048 channels (0–20 keV)                  |


## File Formats Supported


| Format           | Instrument         | Encoding              | Parser          |
| ---------------- | ------------------ | --------------------- | --------------- |
| `.txt` (Rigaku)  | Rigaku Ultima3 XRD | ASCII                 | `xrd_parser.py` |
| `.txt` (PHI)     | PHI XPS            | ASCII                 | `xps_parser.py` |
| `.txt` (Hitachi) | Hitachi SU8600 SEM | UTF-16-LE with BOM    | `sem_parser.py` |
| `.emsa` (JEOL)   | JEOL TEM-EDX       | ASCII (EMSA standard) | `eds_parser.py` |


---

## Material System

- **Precursor:** Ti₃AlC₂ (MAX phase, hexagonal P6₃/mmc)
- **Product:** Ti₃C₂Tₓ MXene (via selective etching of Al layer)
- **Terminations (Tₓ):** -O (23%), -F (7%), -OH (confirmed by XPS)
- **Synthesis conditions:** 30°C, compared N₂ vs Ar atmospheres

---

## Scientific Methods

### XRD

- **Peak detection**: Savitzky-Golay smoothing + `scipy.signal.find_peaks()`
- **Peak fitting**: Gaussian / Lorentzian / Pseudo-Voigt via `scipy.optimize.curve_fit()`
- **Rietveld refinement**: Whole-pattern fitting with crystal structure models, `scipy.optimize.least_squares()`
- **Phase ID**: 6-phase reference database (Ti₃AlC₂, Ti₃C₂Tₓ, TiO₂ Anatase/Rutile, TiC, Al₂O₃)
- **Scherrer equation**: L = Kλ / (β·cosθ), K = 0.9

### XPS

- **Backgrounds**: Shirley (iterative), Linear, Tougaard (universal cross-section)
- **Peak function**: Gaussian-Lorentzian product with adjustable mixing ratio
- **References**: 17 chemical states across Ti 2p, C 1s, O 1s, F 1s (with DOI links)
- **Spin-orbit**: Automatic 2p₃/₂ / 2p₁/₂ doublet generation

### SEM

- **Segmentation**: Otsu, Adaptive thresholding, Watershed
- **Preprocessing**: CLAHE contrast enhancement, Gaussian denoising
- **Measurements**: Equivalent diameter, aspect ratio, circularity, solidity
- **Layer thickness**: Gradient-based boundary detection on cross-section images

---

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `scikit-image`
Visualization: `plotly`, `matplotlib`, `seaborn`
Dashboard: `streamlit`
ML (planned): `scikit-learn`, `xgboost`

See `requirements.txt` for full list.

---

## Future Work

- **Agent 1**: Dynamic XPS reference assignment via NIST XPS Database + literature queries
- **Agent 2**: RAG-powered literature Q&A over MXene papers
- **Agent 3**: Synthesis optimization recommendations from characterization results
- **ML surrogate model**: Synthesis parameters → thermoelectric property prediction
- **GSAS-II integration**: Full Rietveld refinement with preferred orientation + thermal parameters
- **ML segmentation**: U-Net / Mask R-CNN for SEM instance segmentation
- **Multi-MXene support**: V₂CTₓ, Mo₂CTₓ, Nb₂CTₓ via configurable reference databases
- **Expand the Scope**: Project adaption to analyze different materials other than MXenes itself

---

## Related Project

[MXDiscovery](https://github.com/srinikhil25/MXDiscovery) — Computational discovery pipeline for novel non-toxic MXene composites using literature mining, ML screening, DFT validation (Quantum ESPRESSO), and TOPSIS ranking for wearable thermoelectric applications.

---

## Author

**Gudibandi Sri Nikhil Reddy**  
Masters Student,  
Ikeda - Hamasaki Laboratory,  
Research Institute of Electronics,  
Shizuoka University, Japan  
Research Focus: AI Development in Materials Science, MXenes, Wearable Thermoelectrics, Materials Informatics

## License

MIT License