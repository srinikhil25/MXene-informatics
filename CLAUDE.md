# MXene-Informatics

## Project Overview
Autonomous Materials Informatics Pipeline for Ti3C2Tx MXene characterization.
Converts raw experimental data (XRD, XPS, SEM, TEM, EDS) from Ti3AlC2 → Ti3C2Tx
synthesis into a structured, ML-ready dataset with predictive models.

## Directory Structure
```
D:/MXene-Informatics/
├── src/
│   ├── etl/           # Data extraction, transformation, loading
│   ├── analysis/      # XRD peak fitting, XPS deconvolution, etc.
│   ├── ml/            # Surrogate models (XGBoost, Random Forest)
│   └── visualization/ # Plotting, Blender scripts, Streamlit dashboard
├── data/
│   ├── raw/           # Original experimental files (symlinked from Mxene_Analysis)
│   ├── processed/     # Cleaned, standardized JSON/CSV
│   └── features/      # ML-ready feature matrices
├── notebooks/         # Jupyter analysis notebooks
├── docs/              # Technical documentation
├── models/            # Saved ML model artifacts
└── outputs/
    ├── figures/       # Publication-quality plots
    └── reports/       # Generated reports
```

## Raw Data Source
Original data: D:/MXDiscovery/Mxene_Analysis/
- XRD: Ti3AlC2 (MAX phase) and Ti3C2 (MXene) powder diffraction, Cu Ka, 5-90°
- XPS: Wide survey + C 1s, O 1s, Ti 2p, F 1s high-resolution spectra
- SEM: Hitachi SU8600, multiple magnifications (1k-200k), two samples
- TEM: Two sessions (2024-06-13, 2025-05-26), with EDX elemental maps
- EDS: EMSA format spectral data with elemental quantification
- Protocols: Synthesis procedures (.docx)

## Key Technical Details
- Material: Ti3C2Tx MXene (from Ti3AlC2 MAX phase precursor)
- Terminations: O, F, OH (confirmed by XPS: C=64%, O=23%, Ti=6%, F=7%)
- XRD instrument: Rigaku Ultima3, Cu Ka (1.54056 A), 40kV/40mA
- SEM instrument: Hitachi SU8600, 20kV
- TEM: JEOL, 200kV (from EMSA headers)
- Samples: mx-ticn2@30 (MXene Ti3C2 N2 at 30°), mx-ticar@30 (MXene Ti3C2 Ar at 30°)

## Planned Agents
See [agents.md](agents.md) for future Layer 4 (Agentic Interface) work:
- **Agent 1**: XPS Reference Assignment — dynamically assign literature DOIs instead of hardcoded references
- **Agent 2**: Literature Q&A (RAG) — natural language querying over MXene papers
- **Agent 3**: Synthesis Optimization — suggest parameters based on characterization results

## TODO — Pending Fixes & Improvements
- [ ] **SEM Gallery: Filter out missing images** — Records with `has_image=FALSE` should be hidden from the Full Imaging Conditions table (metadata .txt exists but .tif image is missing/not copied). Remove `has_image` column from display after filtering.
- [ ] **SEM Gallery: Resolve missing .tif files** — Investigate why some metadata records don't have corresponding .tif images (e.g., `mx-ticar@30 20kV x5000.tif`). Check if files exist in original raw directory and need copying, or if filenames have mismatches.

## Related Project
MXDiscovery (D:/MXDiscovery/) — Computational discovery pipeline (separate project)
