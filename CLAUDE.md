# Materials Informatics

## Project Overview
Autonomous multi-modal materials characterization platform. Researchers drop ANY folder of raw instrument data. The tool auto-detects techniques, identifies samples, parses files, and unlocks analysis pages. No file reorganization required from the user.

**Goal:** Professional lab tool + publishable paper (target: journal-quality informatics tool).
**Users:** Multi-user local install (no auth needed). Researchers in the Ikeda-Hamasaki Lab.

## Tech Stack
- **Frontend:** Streamlit (multipage app, dark theme, plotly charts)
- **Backend:** Python 3.13, venv at `D:/Materials-Informatics/venv`
- **Key packages:** mp-api, pymatgen, scipy, numpy, pandas, plotly, xlrd, openpyxl
- **API keys:** `.env` file (Materials Project API key via `src/config.py`)

## Architecture

### Data Model
```
Project -> Sample -> TechniqueData
```
- `Project`: name, root_path, samples dict, manifest (file scan results), unassigned files
- `Sample`: sample_id, aliases (e.g. "CS Pure" = "CS"), techniques dict
- `TechniqueData`: technique name, source files, parsed data (raw parser output), analysis results
- Session state: `st.session_state.project` is the single source of truth
- Use `@st.cache_data` on expensive parse/fetch operations

### App Structure (Multipage Streamlit)
```
app.py                          (~80 lines: config, theme, session init)
pages/
  01_Overview.py                (project load, sample-technique matrix, file intelligence)
  02_XRD.py                     (pattern view, phase ID, Scherrer)
  03_XPS.py                     (survey + HR regions, deconvolution)
  04_UV_DRS.py                  (reflectance, Kubelka-Munk, Tauc bandgap)
  05_Microscopy.py              (TEM/SEM/STEM gallery, SAED, elemental maps)
  06_EDS.py                     (EMSA spectra + quantification)
  07_Transport.py               (Hall + Thermoelectric, T-dependent plots)
  08_Cross_Correlation.py       (XRD vs HRTEM, XPS vs EDS, purity checks)
  09_Report.py                  (per-sample summary, export figures)
```

### Source Code Structure
```
src/
  models.py                     (Project, Sample, TechniqueData dataclasses)
  project_builder.py            (scan -> parse -> build Project object)
  sample_resolver.py            (3-tier sample ID: directory > filename > content)
  config.py                     (.env loader for API keys)

  agents/
    file_intelligence.py        (file scanning + technique classification)
    xrd_analysis.py             (phase ID: Materials Project, zero-shift, greedy matching)

  etl/                          (parsers - each has can_parse() + parse())
    base_parser.py              (BaseParser ABC)
    asc_xrd_parser.py           (.ASC Rigaku)
    panalytical_xrd_parser.py   (.xrdml PANalytical)
    xrd_parser.py               (.txt Rigaku)
    xps_csv_parser.py           (.csv XPS)
    eds_parser.py               (.emsa EDS)
    bruker_edx_parser.py        (.spx Bruker)
    uv_drs_parser.py            (.txt UV DRS - wavelength, R%)
    hall_parser.py              (.xls Hall measurement)
    thermoelectric_parser.py    (.xlsx multi-sheet TE properties)
    universal_etl.py            (legacy dispatcher - being replaced by project_builder)

  analysis/
    xrd_analysis.py             (legacy, being replaced by agents/xrd_analysis.py)
    xps_analysis.py             (peak fitting, deconvolution)
    uv_drs_analysis.py          (Kubelka-Munk + Tauc plot)
    transport_analysis.py       (Hall + TE property analysis)

  ml/                           (cross-technique analysis)
    sample_matcher.py           (material family classification)
    feature_extraction.py
    correlation_plots.py

  visualization/                (reusable plot builders)
```

## Key Design Decisions

1. **No forced folder structure** - tool adapts to researcher's file organization via File Intelligence Agent
2. **Sample detection via 3-tier fallback**: directory names > filenames > file content headers
3. **Parsers stay as plain functions** - BaseParser ABC wraps existing functions, no rewrite
4. **No premature Agent abstraction** - plain functions until polymorphic dispatch is truly needed
5. **Thermoelectric xlsx with multiple sheets** -> decomposed into per-sample TechniqueData
6. **One-to-one peak matching** (greedy algorithm) for XRD phase ID
7. **Zero-shift correction** for XRD (sample displacement error)

## Migration Plan (from 3382-line app.py monolith)

### Phase 0: Foundation (current)
1. Create `src/models.py` - data model dataclasses
2. Create `src/sample_resolver.py` - multi-tier sample ID detection
3. Create `src/project_builder.py` - orchestrator (scan -> parse -> Project)
4. Create `pages/` skeleton + shrink `app.py` to entry point

### Phase 1: Migrate Existing Pages (one at a time)
5. Overview -> `pages/01_Overview.py`
6. XRD -> `pages/02_XRD.py`
7. XPS -> `pages/03_XPS.py`
8. Microscopy -> `pages/05_Microscopy.py`
9. EDS -> `pages/06_EDS.py`

### Phase 2: New Features
10. UV-DRS parser + `pages/04_UV_DRS.py` (Kubelka-Munk + Tauc bandgap)
11. Hall + Thermoelectric parser + `pages/07_Transport.py`
12. Cross-correlation enhancements
13. Report/export page

## Test Data
Primary test data: `data_raw/dhivya_data/` (161 files, 4 samples)
- **CS** (CuSe pure): XRD, XPS, UV-DRS, Hall, TEM, STEM/EDS, SEM, Thermoelectric
- **CS-1**: XRD, UV-DRS, Hall, Thermoelectric
- **CS-3**: XRD, XPS, UV-DRS, Hall, TEM, STEM/EDS, Thermoelectric
- **CS-5**: XRD, UV-DRS, Hall, Thermoelectric

## Coding Guidelines
- Keep page files under 400 lines
- Use dataclasses for structured data, not raw dicts
- Cache reference data locally (e.g., `data/xrd_cache/` for Materials Project)
- Dark theme Plotly charts (`template="plotly_dark"`)
- Pagination: use `on_click` callbacks, not inline session state updates
- Handle Windows cp1252 encoding (avoid unicode in print statements)
- Handle hexagonal 4-index Miller notation (h,k,i,l) alongside standard (h,k,l)

## Related Project
MXDiscovery (D:/MXDiscovery/) - Computational MXene thermoelectric discovery pipeline (separate project)
