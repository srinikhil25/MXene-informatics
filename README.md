# MXene-Informatics

**Autonomous Materials Informatics Pipeline for Ti3C2Tx MXene Characterization**

A data-driven pipeline that transforms raw experimental characterization data (XRD, XPS, SEM, TEM, EDS) from Ti3AlC2 -> Ti3C2Tx MXene synthesis into structured, ML-ready datasets with predictive surrogate models and interactive visualizations.

## Project Motivation

MXene synthesis involves complex parameter spaces (etching time, temperature, acid concentration) that influence the resulting material properties (conductivity, termination ratios, interlayer spacing). This project applies materials informatics to:

1. **Standardize** messy instrument data into reproducible, version-controlled formats
2. **Analyze** characterization results programmatically (peak fitting, phase ID, quantification)
3. **Predict** synthesis-property relationships using machine learning surrogate models
4. **Visualize** results through interactive dashboards and 3D structural renderings

## Architecture

```
Layer 1: ETL (Data Engineering)
    Raw instrument files -> Standardized JSON/CSV
    Parsers: XRD (Rigaku), XPS, SEM (Hitachi), EDS (EMSA)

Layer 2: Analysis + ML (Materials Science + Surrogate Models)
    XRD peak fitting, d-spacing, phase identification
    XPS deconvolution, surface chemistry quantification
    ML: synthesis parameters -> property prediction (XGBoost/RF)

Layer 3: Visualization (3D + Dashboard)
    Streamlit interactive dashboard
    Blender 3D MXene structure renders
    Publication-quality matplotlib/plotly figures

Layer 4: RAG Assistant (Intelligent Query)
    Local LLM (Ollama) + indexed protocols/literature
    Natural language queries over experimental data
```

## Data Summary

| Technique | Instrument | Samples | Key Findings |
|-----------|-----------|---------|--------------|
| XRD | Rigaku Ultima3, Cu Ka | Ti3AlC2 (MAX), Ti3C2 (MXene) | Phase confirmation, d-spacing expansion |
| XPS | - | Ti3C2Tx @ 30C | C: 64%, O: 23%, Ti: 6%, F: 7% |
| SEM | Hitachi SU8600, 20kV | 2 samples (N2 vs Ar atm) | Flake morphology, 1k-200k magnification |
| TEM | JEOL, 200kV | Ti3AlC2, Ti3C2Tx | Lattice imaging, FFT analysis |
| EDS/EDX | TEM-EDX + SEM-EDS | Multiple regions | Elemental mapping (Ti, C, O, F, Al, Cl) |

## Directory Structure

```
MXene-Informatics/
|-- src/
|   |-- etl/              # Data parsers (XRD, XPS, SEM, EDS)
|   |-- analysis/         # Scientific analysis (peak fitting, deconvolution)
|   |-- ml/               # Machine learning models
|   |-- visualization/    # Plotting and dashboard code
|-- data/
|   |-- raw/              # Symlinks to original instrument files
|   |-- processed/        # Standardized JSON + CSV output
|   |-- features/         # ML-ready feature matrices
|-- notebooks/            # Jupyter analysis notebooks
|-- docs/                 # Technical documentation
|-- models/               # Saved ML model artifacts
|-- outputs/
|   |-- figures/          # Publication-quality plots
|   |-- reports/          # Generated analysis reports
|-- run_etl.py            # Master ETL pipeline runner
|-- requirements.txt      # Python dependencies
|-- CLAUDE.md             # AI assistant context
```

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

# 4. Run the ETL pipeline (parse all raw data)
python run_etl.py

# 5. Run specific stages
python run_etl.py --stage xrd
python run_etl.py --stage xps
python run_etl.py --stage sem
python run_etl.py --stage eds
```

## Raw Data Source

Original experimental data is located at `D:/MXDiscovery/Mxene_Analysis/` and organized by characterization technique (XRD, XPS, SEM, TEM). Data was collected at Shizuoka University, Japan.

### Material System

- **Precursor:** Ti3AlC2 (MAX phase)
- **Product:** Ti3C2Tx MXene (via selective etching of Al layer)
- **Terminations (Tx):** -O, -F, -OH (confirmed by XPS)
- **Synthesis conditions:** 30C, compared N2 vs Ar atmospheres

## ETL Pipeline Output

After running `python run_etl.py`:

| Data Type | Files Generated | Description |
|-----------|----------------|-------------|
| XRD | 2 JSON + 2 CSV | 2-theta vs intensity, instrument metadata |
| XPS | 6 JSON + 5 CSV + quant | Survey + high-res spectra + atomic concentrations |
| SEM | catalog JSON + summary CSV | 18 images with full acquisition metadata |
| EDS | 19 JSON + 19 CSV + peaks | Spectra with auto-identified elemental peaks |

## Key Technical Details

### Instruments
- **XRD:** Rigaku Ultima3 Inplane, Cu Ka (1.54056 A), 40kV/40mA, 2theta range 5-90 deg
- **SEM:** Hitachi SU8600, 20kV accelerating voltage, multiple magnifications
- **TEM:** JEOL, 200kV, with EDX capability
- **XPS:** Binding energy range 0-1200 eV, high-resolution regions for C 1s, O 1s, Ti 2p, F 1s

### File Formats Handled
- `.txt` (Rigaku XRD, tab/space-delimited)
- `.raw` (Rigaku binary XRD)
- `.txt` (XPS binding energy vs intensity CSV)
- `.txt` (Hitachi SEM metadata, UTF-16-LE encoded)
- `.emsa` (EMSA/MAS standard for EDS/EDX spectra)
- `.tif` (SEM micrographs)
- `.jpg` (TEM micrographs)
- `.bmp` (EDS elemental maps)

## Related Project

[MXDiscovery](https://github.com/srinikhil25/MXDiscovery) - Computational discovery pipeline for novel MXene composites using literature mining, ML screening, and DFT validation.

## Author

**Gudibandi Sri Nikhil Reddy**
Masters Student, Shizuoka University, Japan
Research Focus: MXenes, Wearable Thermoelectrics, Materials Informatics

## License

MIT License
