# Materials Informatics — TODO

## Completed
- [x] Phase 0: Foundation refactor (3382-line monolith → multipage app)
- [x] Phase 1: Migrate all existing pages (Overview, XRD, XPS, Microscopy, EDS)
- [x] Phase 2: New parsers (UV-DRS, Hall, Thermoelectric)
- [x] Phase 2: New analysis pages (UV-DRS bandgap, Transport properties)
- [x] UV-DRS Kubelka-Munk Wavelength/Energy x-axis toggle
- [x] E2E test pass — 5 bugs found and fixed:
  - XPS: negative peak areas → use abs()
  - EDS: element markers too crowded → filter to peaks with actual signal
  - EDS: full file paths in title and comparison table → show filename only
  - Hall: negative carrier concentration/mobility for n-type → show absolute values
  - Thermoelectric: CS sheet zT column had intermediate calc values → sanity check + fallback to adjacent column

---

## Priority 1 — VLM-based File Labeling Agent
**Problem:** The #1 reliability bottleneck. Researchers use inconsistent naming:
`CS Pure`, `CSCBI-1`, `cskbi3`, `CS (Pure)` are all different conventions for 4 samples.
Current regex heuristics in `sample_resolver.py` are fragile.

**Architecture: Orchestrator-Worker**
```
ORCHESTRATOR (brain — sequential, sees everything)
├── Plans work batches
├── Dispatches workers in parallel
├── Cross-references results
├── Resolves aliases ("CS Pure" = "CS")
├── Detects anomalies ("XRD file in TEM folder?")
└── Builds final sample map

WORKERS (parallel, each sees one file)
├── Text Worker (Python, ~5ms/file)
│   .xy .txt .csv .emsa .asc .xrdml
│   → technique fingerprint + sample hints from headers
│
├── Vision Worker (Ollama VLM, ~2-5s/image)
│   .tif .png .jpg (TEM/SEM/STEM)
│   → reads scale bars, instrument info, sample names from image footers
│   → only 2-3 images per folder (smart batching)
│
└── Excel Worker (Python, ~50ms/file)
    .xlsx .xls
    → sheet names, column headers, embedded sample names
```

**VLM Model:** Qwen3-VL 8B via Ollama (free, local, fits RTX 4060 8GB VRAM)
- Best OCR accuracy among local models (88.8% OCRBench)
- Fallback: Gemma 3 4B (~3GB VRAM) for quick triage

**Human-in-the-Loop:** Streamlit confirmation UI before final assignment
- Show proposed sample map with aliases (editable)
- Flag unassigned/ambiguous files
- One-click accept or manual corrections

---

## Priority 2 — New Analysis Pages
- [ ] `08_Cross_Correlation.py` — Cross-technique validation
  - XRD crystallite size vs TEM particle size comparison
  - XPS composition vs EDS composition check
  - Bandgap vs thermoelectric property correlations
  - Purity assessment (expected phases vs detected)
- [ ] `09_Report.py` — Per-sample summary & export
  - Auto-generated sample report with key findings from each technique
  - Exportable figures (publication-quality PNG/SVG)
  - Summary table across all samples
  - One-click LaTeX/Word export for paper drafts

---

## Priority 3 — New Parsers & Techniques
- [ ] Raman parser (data coming from lab)
- [ ] XPS .spe parser (PHI native binary format)
- [ ] SAED ring/spot indexing (from TEM diffraction patterns)
- [ ] SEM image parser (currently only .docx with embedded images)

---

## Priority 4 — Polish & Paper-Ready
- [ ] Publication-quality plot themes (Nature/ACS style options)
- [ ] Batch export all figures for a sample in one click
- [ ] Detailed documentation for all algorithms (Kubelka-Munk, Tauc, Shirley background, etc.)
- [ ] Add uncertainty/error bars where applicable
- [ ] Statistical analysis helpers (mean, std across samples)

---

## Ideas (Someday/Maybe)
- [ ] Multi-user project sharing (save/load project as JSON)
- [ ] Database backend (SQLite) instead of session state
- [ ] Cloud deployment (Streamlit Cloud or self-hosted)
- [ ] Integration with electronic lab notebook (ELN)
- [ ] Auto-generate Methods section text for papers
- [ ] ML-based phase identification (train on lab's own XRD library)
