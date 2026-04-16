# -*- coding: utf-8 -*-
"""
Materials Informatics Interactive Dashboard
========================================
Autonomous multi-modal materials characterization platform.
Supports XRD, XPS, SEM, and EDS/EDX from multiple instrument vendors.
Users can upload their own data or explore pre-loaded demo datasets.

Run:  streamlit run app.py
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from PIL import Image
import tempfile
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# Data Registry — central store for all loaded data
# ---------------------------------------------------------------------------
def _empty_registry():
    """Return a fresh, empty registry dict."""
    return {
        "xrd": {},
        "xps": {},
        "xps_quant": {},
        "sem": [],
        "eds": {"spectra": [], "quantifications": []},
        "tem": [],
        "tem_eds": [],
        "images": {
            "tem_raw": [],         # Raw TEM/STEM images
            "tem_processed": [],   # FFT, SAED, composite images
            "elemental_maps": [],  # EDS elemental maps
            "sem_raw": [],         # Raw SEM images
        },
    }


def init_data_registry():
    """Initialize the data registry in session_state — starts empty (clean slate)."""
    if "data_registry" not in st.session_state:
        st.session_state["data_registry"] = _empty_registry()
        st.session_state["upload_log"] = []


def _detect_tem(text: str) -> bool:
    """Detect if a JEOL $CM_FORMAT text file is TEM/STEM rather than SEM.

    Heuristics:
    1. Accelerating voltage >= 100 kV (TEM: 100-300 kV, SEM: 1-30 kV)
    2. Signal name contains TEM/STEM/BF/DF keywords
    """
    import re
    # Check accelerating voltage
    volt_match = re.search(r'\$CM_ACCEL_VOLT\s+([\d.]+)', text)
    if volt_match:
        try:
            voltage_kv = float(volt_match.group(1))
            if voltage_kv >= 100:
                return True
        except ValueError:
            pass

    # Check signal name for TEM/STEM indicators
    signal_match = re.search(r'\$CM_SIGNAL\s+(.+)', text)
    if signal_match:
        signal = signal_match.group(1).strip().upper()
        tem_keywords = ["STEM", "TEM", " BF", " DF", "HAADF", "ABF"]
        if any(kw in signal or signal.startswith(kw.strip()) for kw in tem_keywords):
            return True

    return False


def detect_and_parse(uploaded_file):
    """Auto-detect file format and parse an uploaded file.
    Returns (data_type, parsed_data) or (None, None) on failure.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    # Write to temp file (parsers expect file paths)
    suffix = Path(name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        # --- XRD formats ---
        if name.endswith(".xrdml"):
            from src.etl.panalytical_xrd_parser import parse_panalytical_xrdml
            data = parse_panalytical_xrdml(tmp_path)
            return ("xrd", data)

        if name.endswith(".asc"):
            from src.etl.asc_xrd_parser import parse_asc_xrd
            data = parse_asc_xrd(tmp_path)
            return ("xrd", data)

        # --- EDX formats ---
        if name.endswith(".spx"):
            from src.etl.bruker_edx_parser import parse_bruker_spx
            data = parse_bruker_spx(tmp_path)
            return ("eds_spectrum", data)

        if name.endswith(".xls") and not name.endswith(".xlsx"):
            from src.etl.bruker_edx_parser import parse_bruker_xls
            data = parse_bruker_xls(tmp_path)
            return ("eds_quant", data)

        if name.endswith(".emsa"):
            from src.etl.eds_parser import parse_emsa
            data = parse_emsa(tmp_path)
            beam_kv = data.get("metadata", {}).get("beam_kv", data.get("beam_kv", 0))
            if beam_kv >= 100:
                return ("tem_eds", data)
            return ("eds_spectrum", data)

        # --- .txt files — need content sniffing ---
        if name.endswith(".txt"):
            try:
                text = raw_bytes.decode("utf-8", errors="replace")[:2000]
            except Exception:
                text = raw_bytes.decode("latin-1", errors="replace")[:2000]

            # JEOL: detect TEM vs SEM by voltage/signal
            if "$CM_FORMAT" in text or "$SEM_DATA_VERSION" in text or "$JEOL_SEM" in text.upper():
                from src.etl.jeol_sem_parser import parse_jeol_metadata
                data = parse_jeol_metadata(tmp_path)
                if data:
                    if _detect_tem(text):
                        return ("tem", data)
                    else:
                        return ("sem", data)

            # Rigaku XRD: header with *TYPE, *MEAS_COND
            if "*TYPE" in text or "*MEAS_COND" in text or ";KAlpha1" in text:
                from src.etl.xrd_parser import parse_rigaku_txt
                data = parse_rigaku_txt(tmp_path)
                return ("xrd", data)

            # Hitachi SEM: try UTF-16
            try:
                text16 = raw_bytes.decode("utf-16-le", errors="replace")[:2000]
                if "InstructName" in text16 or "Magnification" in text16:
                    from src.etl.sem_parser import parse_sem_metadata
                    data = parse_sem_metadata(tmp_path)
                    if data:
                        return ("sem", data)
            except Exception:
                pass

        # --- XPS CSV format ---
        if name.endswith(".csv"):
            try:
                text = raw_bytes.decode("utf-8", errors="replace")
                lines = text.strip().split("\n")
                if len(lines) >= 5:
                    # PHI MultiPak CSV: line1=int, line2=blank, line3=identifier, line4=int, line5=float,float
                    try:
                        int(lines[0].strip())
                        if "," in lines[4]:
                            parts = lines[4].strip().split(",")
                            float(parts[0])
                            float(parts[1])
                            from src.etl.xps_csv_parser import parse_xps_csv
                            data = parse_xps_csv(tmp_path)
                            return ("xps", data)
                    except (ValueError, IndexError):
                        pass
            except Exception:
                pass

        # --- Image files (SEM/TEM) ---
        if name.endswith((".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".png")):
            return ("image", {"path": tmp_path, "name": uploaded_file.name})

        return (None, None)

    except Exception as e:
        st.warning(f"Failed to parse {uploaded_file.name}: {e}")
        return (None, None)
    finally:
        try:
            if not name.endswith((".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".png")):
                os.unlink(tmp_path)
        except Exception:
            pass


def merge_uploaded_data(data_type, parsed_data, filename=""):
    """Merge parsed uploaded data into the registry."""
    reg = st.session_state["data_registry"]

    if data_type == "xrd":
        sample_name = parsed_data.get("metadata", {}).get("sample_name", Path(filename).stem)
        reg["xrd"][sample_name] = {
            "two_theta": parsed_data["two_theta"],
            "intensity": parsed_data["intensity"],
            "metadata": parsed_data.get("metadata", {}),
            "source": "uploaded",
        }
        return f"XRD: {sample_name}"

    elif data_type == "xps":
        region = parsed_data.get("region", Path(filename).stem)
        region_key = region.replace(" ", "_").lower()
        reg["xps"][region_key] = {
            "binding_energy": parsed_data["binding_energy"],
            "intensity": parsed_data["intensity"],
            "metadata": parsed_data.get("metadata", {}),
            "source": "uploaded",
        }
        return f"XPS: {region}"

    elif data_type == "sem":
        parsed_data["source"] = "uploaded"
        reg["sem"].append(parsed_data)
        return f"SEM: {parsed_data.get('sample_name', filename)}"

    elif data_type == "eds_spectrum":
        parsed_data["source"] = "uploaded"
        parsed_data["format"] = parsed_data.get("format", "Uploaded")
        reg["eds"]["spectra"].append(parsed_data)
        return f"EDS: {parsed_data.get('source_file', filename)}"

    elif data_type == "eds_quant":
        if isinstance(parsed_data, list):
            reg["eds"]["quantifications"].extend(parsed_data)
        return f"EDX Quantification: {filename}"

    elif data_type == "tem":
        parsed_data["source"] = "uploaded"
        reg["tem"].append(parsed_data)
        return f"TEM: {parsed_data.get('sample_name', filename)}"

    elif data_type == "tem_eds":
        parsed_data["source"] = "uploaded"
        reg["tem_eds"].append(parsed_data)
        return f"TEM-EDS: {parsed_data.get('source_file', filename)}"

    elif data_type == "image":
        # Store for potential SEM gallery display
        return f"Image: {filename} (stored for gallery)"

    return None


def _technique_has_data(reg, key):
    """Check if a technique section has data (handles dict, list, nested dict)."""
    val = reg.get(key)
    if val is None:
        return False
    if isinstance(val, dict):
        if key == "eds":
            return bool(val.get("spectra"))
        return bool(val)
    return bool(val)  # list


# Single source of truth: technique key → page name
TECHNIQUE_PAGES = {
    "xrd": "XRD Analysis",
    "xps": "XPS Analysis",
    "sem": "SEM Gallery",
    "eds": "EDS Analysis",
    "tem": "TEM Analysis",
}


def get_available_pages():
    """Determine which pages to show based on available data — fully dynamic."""
    reg = st.session_state["data_registry"]
    pages = ["Overview"]

    for key, page_name in TECHNIQUE_PAGES.items():
        if _technique_has_data(reg, key):
            pages.append(page_name)

    # TEM page also shows if TEM-EDS data or TEM images exist
    if "TEM Analysis" not in pages:
        _img_reg = reg.get("images", {})
        _has_tem_images = bool(_img_reg.get("tem_raw")) or bool(_img_reg.get("tem_processed"))
        _has_emaps = bool(_img_reg.get("elemental_maps"))
        if reg.get("tem_eds") or _has_tem_images or _has_emaps:
            pages.append("TEM Analysis")

    # Cross-technique ML: show only when 2+ techniques have data
    n_techniques = sum(1 for k in TECHNIQUE_PAGES if _technique_has_data(reg, k))
    if reg.get("tem_eds"):
        n_techniques += 1
    if n_techniques >= 2:
        pages.append("Cross-Technique ML")
    return pages


# Supported extensions for folder scanning
_SUPPORTED_EXTENSIONS = {
    ".txt", ".xrdml", ".asc", ".csv", ".spx", ".xls",
    ".emsa", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".png",
    ".spe", ".xlsx", ".xlsm", ".pdf",
}


def _scan_and_parse_folder(folder: Path):
    """
    Scan a folder using the File Intelligence Agent, then parse all parseable files.

    Two-phase approach:
        Phase 1 — Intelligence: scan_directory() classifies every file
        Phase 2 — Parsing: for each parseable file, call the appropriate parser
        Phase 3 — Images: store categorized images in the registry
    """
    from src.agents.file_intelligence import scan_directory, Technique, FileType

    # ── Phase 1: Scan & Classify ──
    with st.spinner("🔍 Phase 1 — File Intelligence Agent scanning directory…"):
        manifest = scan_directory(folder)

    if manifest.total_files == 0:
        return []

    # Store manifest for use by other pages
    st.session_state["file_manifest"] = manifest

    # Show summary
    summary = manifest.summary()
    st.info(
        f"**File Intelligence:** {summary['total_files']} files found — "
        f"{summary['parseable']} parseable, {summary['images']} images, "
        f"{summary['skipped']} skipped | "
        f"Samples: {', '.join(summary['samples'][:6])}"
    )

    # ── Phase 2: Parse parseable files ──
    new_uploads = []
    parseable = manifest.parseable_files
    if parseable:
        progress = st.progress(0, text="Phase 2 — Parsing data files…")
        for i, entry in enumerate(parseable):
            progress.progress(
                (i + 1) / len(parseable),
                text=f"Parsing {entry.filename} ({i+1}/{len(parseable)})"
            )
            try:
                data_type, parsed_data = _parse_local_file(entry.path)
                if data_type and parsed_data:
                    # Enrich with agent context
                    if isinstance(parsed_data, dict):
                        if entry.sample_name and "sample_name" not in parsed_data:
                            parsed_data["sample_name"] = entry.sample_name
                        if entry.sub_technique:
                            parsed_data["sub_technique"] = entry.sub_technique
                        parsed_data["_agent_technique"] = entry.technique
                        parsed_data["_agent_session"] = entry.session_id
                    result = merge_uploaded_data(data_type, parsed_data, entry.filename)
                    if result:
                        new_uploads.append(result)
                        st.session_state["upload_log"].append(result)
            except Exception:
                pass
        progress.empty()

    # ── Phase 3: Store categorized images ──
    reg = st.session_state["data_registry"]
    images_reg = reg["images"]

    for entry in manifest.images:
        img_info = {
            "path": str(entry.path),
            "filename": entry.filename,
            "sample_name": entry.sample_name,
            "sub_technique": entry.sub_technique,
            "session_id": entry.session_id,
            "format_name": entry.format_name,
            "element": entry.element,
        }
        if entry.file_type == FileType.IMAGE_ELEMENTAL_MAP:
            images_reg["elemental_maps"].append(img_info)
        elif entry.technique in (Technique.TEM, Technique.TEM_EDS):
            if entry.file_type == FileType.IMAGE_PROCESSED:
                images_reg["tem_processed"].append(img_info)
            else:
                images_reg["tem_raw"].append(img_info)
        elif entry.technique == Technique.SEM:
            images_reg["sem_raw"].append(img_info)
        else:
            # Default: assign based on directory context
            images_reg["tem_raw"].append(img_info)

    n_images = sum(len(v) for v in images_reg.values())
    if n_images:
        new_uploads.append(f"Images: {n_images} categorized ({len(images_reg['tem_raw'])} TEM raw, {len(images_reg['tem_processed'])} processed, {len(images_reg['elemental_maps'])} elemental maps)")
        st.session_state["upload_log"].append(new_uploads[-1])

    return new_uploads


def _parse_local_file(filepath: Path):
    """Parse a local file (on disk) by auto-detecting format. Returns (data_type, parsed_data)."""
    ext = filepath.suffix.lower()
    name = filepath.name.lower()

    try:
        # --- XRD formats ---
        if ext == ".xrdml":
            from src.etl.panalytical_xrd_parser import parse_panalytical_xrdml
            return ("xrd", parse_panalytical_xrdml(str(filepath)))

        if ext == ".asc":
            from src.etl.asc_xrd_parser import parse_asc_xrd
            return ("xrd", parse_asc_xrd(str(filepath)))

        # --- EDX formats ---
        if ext == ".spx":
            from src.etl.bruker_edx_parser import parse_bruker_spx
            return ("eds_spectrum", parse_bruker_spx(str(filepath)))

        if ext == ".xls":
            from src.etl.bruker_edx_parser import parse_bruker_xls
            data = parse_bruker_xls(str(filepath))
            return ("eds_quant", data)

        if ext == ".emsa":
            from src.etl.eds_parser import parse_emsa
            data = parse_emsa(str(filepath))
            # Classify by beam voltage: TEM-EDS (>= 100 kV) vs SEM-EDS
            beam_kv = data.get("metadata", {}).get("beam_kv", data.get("beam_kv", 0))
            if beam_kv >= 100:
                return ("tem_eds", data)
            return ("eds_spectrum", data)

        # --- .txt files: content sniffing ---
        if ext == ".txt":
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                text = filepath.read_text(encoding="latin-1", errors="replace")[:2000]

            # JEOL: detect TEM vs SEM by voltage/signal
            if "$CM_FORMAT" in text or "$SEM_DATA_VERSION" in text:
                from src.etl.jeol_sem_parser import parse_jeol_metadata
                data = parse_jeol_metadata(str(filepath))
                if data:
                    if _detect_tem(text):
                        return ("tem", data)
                    else:
                        return ("sem", data)

            # Rigaku XRD
            if "*TYPE" in text or "*MEAS_COND" in text or ";KAlpha1" in text:
                from src.etl.xrd_parser import parse_rigaku_txt
                return ("xrd", parse_rigaku_txt(str(filepath)))

            # Hitachi SEM (UTF-16)
            try:
                raw = filepath.read_bytes()
                text16 = raw.decode("utf-16-le", errors="replace")[:2000]
                if "InstructName" in text16 or "Magnification" in text16:
                    from src.etl.sem_parser import parse_sem_metadata
                    data = parse_sem_metadata(str(filepath))
                    if data:
                        return ("sem", data)
            except Exception:
                pass

        # --- XPS CSV ---
        if ext == ".csv":
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                lines = text.strip().split("\n")
                if len(lines) >= 5:
                    int(lines[0].strip())
                    parts = lines[4].strip().split(",")
                    float(parts[0]); float(parts[1])
                    from src.etl.xps_csv_parser import parse_xps_csv
                    return ("xps", parse_xps_csv(str(filepath)))
            except (ValueError, IndexError):
                pass

        # --- Image files ---
        if ext in {".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".png"}:
            return ("image", {"path": str(filepath), "name": filepath.name})

    except Exception:
        pass

    return (None, None)


st.set_page_config(
    page_title="Materials Informatics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #3b3b5c;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0ff !important;
    }
    .finding-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-left: 4px solid #06b6d4;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
def color_scale_bar(label, low_label, high_label, colors):
    """Render an HTML color scale bar below tables."""
    gradient = ", ".join(colors)
    st.markdown(
        f"""<div style="display:flex; align-items:center; gap:10px; margin:6px 0 12px 0;">
            <span style="color:#888; font-size:0.82em; font-weight:600;">{label}</span>
            <span style="color:#aaa; font-size:0.78em;">{low_label}</span>
            <div style="flex:1; max-width:220px; height:14px; border-radius:7px;
                 background:linear-gradient(to right, {gradient});
                 border:1px solid rgba(255,255,255,0.15);"></div>
            <span style="color:#aaa; font-size:0.78em;">{high_label}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def color_customizer(graph_id: str, trace_names: list, default_colors: list) -> list:
    """
    Inline color customization expander for any graph/table.

    Parameters:
        graph_id: Unique key for this graph (used in session_state)
        trace_names: List of trace/column names shown in the graph
        default_colors: Default color for each trace (hex strings)

    Returns:
        List of colors (user-customized or defaults)
    """
    # Initialize session state with defaults
    state_key = f"colors_{graph_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(default_colors)

    colors = st.session_state[state_key]

    with st.popover("🎨", help="Customize graph colors"):
        st.caption("**Customize Colors**")
        n = len(trace_names)
        cols_per_row = 3
        for row_start in range(0, n, cols_per_row):
            row_end = min(row_start + cols_per_row, n)
            cols = st.columns(row_end - row_start)
            for j, idx in enumerate(range(row_start, row_end)):
                with cols[j]:
                    new_color = st.color_picker(
                        trace_names[idx],
                        value=colors[idx] if idx < len(colors) else "#888888",
                        key=f"{graph_id}_color_{idx}",
                    )
                    if idx < len(colors):
                        colors[idx] = new_color

        if st.button("↩ Reset", key=f"{graph_id}_reset"):
            st.session_state[state_key] = list(default_colors)
            st.rerun()

    st.session_state[state_key] = colors
    return colors


@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def load_xrd(sample):
    """Load XRD data from registry (covers preloaded + uploaded)."""
    reg = st.session_state.get("data_registry", {})
    if sample in reg.get("xrd", {}):
        d = reg["xrd"][sample]
        return np.array(d["two_theta"]), np.array(d["intensity"]), d.get("metadata", {})
    # Fallback to disk
    universal_path = DATA_DIR / "universal" / "xrd" / f"xrd_{sample}.json"
    old_path = DATA_DIR / "xrd" / f"xrd_{sample}.json"
    path = universal_path if universal_path.exists() else old_path
    j = load_json(str(path))
    return np.array(j["two_theta"]), np.array(j["intensity"]), j.get("metadata", {})


def get_xrd_samples():
    """Get list of all available XRD sample names from registry."""
    reg = st.session_state.get("data_registry", {})
    return sorted(reg.get("xrd", {}).keys())


def load_xps_hr(element_key):
    """Load XPS high-res spectrum from registry or disk."""
    reg = st.session_state.get("data_registry", {})
    xps_reg = reg.get("xps", {})
    # Try exact key, then case-insensitive match
    matched_key = None
    if element_key in xps_reg:
        matched_key = element_key
    else:
        el_lower = element_key.lower()
        for k in xps_reg:
            if k.lower() == el_lower:
                matched_key = k
                break
    if matched_key:
        d = xps_reg[matched_key]
        return np.array(d["binding_energy"]), np.array(d["intensity"]), d.get("metadata", {})
    # Fall back to disk
    path = DATA_DIR / "xps" / f"xps_{element_key}.json"
    if not path.exists():
        # Try lowercase
        path = DATA_DIR / "xps" / f"xps_{element_key.lower()}.json"
    j = load_json(str(path))
    return np.array(j["binding_energy_ev"]), np.array(j["intensity_cps"]), j["metadata"]


def load_eds_spectrum(name):
    j = load_json(str(DATA_DIR / "eds" / f"{name}.json"))
    return np.array(j["energy_ev"]), np.array(j["counts"]), j["metadata"]


try:
    from src.ml.sample_matcher import classify_family as _classify_family
except ImportError:
    def _classify_family(name):
        return "Other"


# ---------------------------------------------------------------------------
# Initialize data registry (loads pre-existing processed data)
# ---------------------------------------------------------------------------
init_data_registry()

# ---------------------------------------------------------------------------
# Sidebar navigation — dynamic based on available data
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Materials Informatics")

_sb_reg = st.session_state["data_registry"]
_sb_has_data = (
    bool(_sb_reg["xrd"]) or bool(_sb_reg["xps"]) or bool(_sb_reg["sem"])
    or bool(_sb_reg["eds"]["spectra"]) or bool(_sb_reg.get("tem"))
    or bool(_sb_reg.get("tem_eds"))
)

available_pages = get_available_pages()
page = st.sidebar.radio(
    "Select Analysis",
    available_pages,
    index=0,
)

# Data summary — only show when there is data
if _sb_has_data:
    st.sidebar.markdown("---")
    _src = "Uploaded"
    st.sidebar.caption(f"Data Source: **{_src}**")
    _sb_counts = {
        "XRD": len(_sb_reg["xrd"]),
        "XPS": len(_sb_reg["xps"]),
        "SEM": len(_sb_reg["sem"]),
        "EDS": len(_sb_reg["eds"]["spectra"]),
        "TEM": len(_sb_reg.get("tem", [])),
        "TEM-EDS": len(_sb_reg.get("tem_eds", [])),
    }
    _parts = [f"**{v}** {k}" for k, v in _sb_counts.items() if v > 0]
    st.sidebar.markdown(" · ".join(_parts))
else:
    st.sidebar.markdown("---")
    st.sidebar.caption("No data loaded — upload files on the Overview page to get started.")


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "Overview":
    st.markdown('<h1 class="main-header">Materials Informatics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">From Raw Spectra to Cross-Technique Insight: '
        'An Autonomous Informatics Platform for Multi-Modal Materials Characterization</p>',
        unsafe_allow_html=True,
    )

    reg = st.session_state["data_registry"]
    _has_any_data = (
        bool(reg["xrd"]) or bool(reg["xps"]) or bool(reg["sem"])
        or bool(reg["eds"]["spectra"]) or bool(reg.get("tem"))
        or bool(reg.get("tem_eds"))
    )

    # =====================================================================
    # STATE A: No data loaded — show landing page with clear instructions
    # =====================================================================
    if not _has_any_data:
        st.markdown("---")

        # Hero message
        st.markdown(
            '<div style="text-align:center;padding:30px 0 10px;">'
            '<div style="font-size:2.5rem;margin-bottom:8px;">Welcome</div>'
            '<div style="color:#94a3b8;font-size:1.05rem;max-width:600px;margin:0 auto;">'
            'Upload your raw instrument data and this platform will automatically detect '
            'the technique, parse the files, and unlock the corresponding analysis pages.'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ── Three clear action cards ──
        st.markdown("### Get Started")
        gc1, gc2, gc3 = st.columns(3)

        with gc1:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#06b6d422,#06b6d411);'
                'border:1px solid #06b6d444;border-radius:12px;padding:20px;min-height:180px;">'
                '<div style="color:#06b6d4;font-weight:700;font-size:1rem;margin-bottom:8px;">'
                'Step 1 &mdash; Upload Data</div>'
                '<div style="color:#94a3b8;font-size:0.85rem;">'
                'Point to a folder containing your raw instrument files, or drag & drop individual files below. '
                'Supported: XRD (.txt, .xrdml, .asc), XPS (.csv), SEM/TEM (.txt), EDS (.spx, .emsa), Images (.tif, .jpg)'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with gc2:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#8b5cf622,#8b5cf611);'
                'border:1px solid #8b5cf644;border-radius:12px;padding:20px;min-height:180px;">'
                '<div style="color:#8b5cf6;font-weight:700;font-size:1rem;margin-bottom:8px;">'
                'Step 2 &mdash; Auto-Detection</div>'
                '<div style="color:#94a3b8;font-size:0.85rem;">'
                'The platform auto-detects each file\'s technique and instrument vendor. '
                'SEM vs TEM is classified by accelerating voltage (&ge;100 kV = TEM). '
                'EDS spectra are split into SEM-EDS and TEM-EDS automatically.'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with gc3:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#ec489922,#ec489911);'
                'border:1px solid #ec489944;border-radius:12px;padding:20px;min-height:180px;">'
                '<div style="color:#ec4899;font-weight:700;font-size:1rem;margin-bottom:8px;">'
                'Step 3 &mdash; Analyze</div>'
                '<div style="color:#94a3b8;font-size:0.85rem;">'
                'Analysis pages appear in the sidebar based on your data. '
                'XRD &rarr; pattern fitting, peak ID. '
                'XPS &rarr; deconvolution, quantification. '
                'SEM/TEM &rarr; image gallery, metadata. '
                'EDS &rarr; elemental spectra.'
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown("---")

        # ── Upload Zone ──
        st.markdown("### Upload Your Data")
        upload_tab1, upload_tab2 = st.tabs(["Folder Path (recommended)", "Drag & Drop Files"])

        with upload_tab1:
            folder_col1, folder_col2 = st.columns([4, 1])
            with folder_col1:
                folder_path = st.text_input(
                    "Enter folder path containing raw instrument data:",
                    placeholder="e.g., D:/MyData/Sample_A/",
                    key="folder_path_input",
                )
            with folder_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                scan_clicked = st.button("Scan & Parse", type="primary", key="scan_folder")

            if scan_clicked and folder_path and folder_path.strip():
                folder = Path(folder_path.strip())
                if not folder.exists():
                    st.error(f"Folder not found: `{folder_path}`")
                elif not folder.is_dir():
                    st.error(f"Not a directory: `{folder_path}`")
                else:
                    new_uploads = _scan_and_parse_folder(folder)
                    if new_uploads:
                        st.success(f"Parsed **{len(new_uploads)}** file(s) from `{folder.name}/`")
                        st.rerun()
                    else:
                        st.warning("No supported files found in this folder.")

        with upload_tab2:
            uploaded_files = st.file_uploader(
                "Upload raw instrument files",
                accept_multiple_files=True,
                type=["txt", "xrdml", "asc", "csv", "spx", "xls", "emsa",
                      "tif", "tiff", "jpg", "jpeg", "bmp", "png"],
                key="file_uploader",
                label_visibility="collapsed",
            )
            if uploaded_files:
                new_uploads = []
                for uf in uploaded_files:
                    already = any(uf.name in log for log in st.session_state.get("upload_log", []))
                    if already:
                        continue
                    data_type, parsed_data = detect_and_parse(uf)
                    if data_type and parsed_data:
                        result = merge_uploaded_data(data_type, parsed_data, uf.name)
                        if result:
                            new_uploads.append(result)
                            st.session_state["upload_log"].append(result)
                if new_uploads:
                    st.success(f"Parsed {len(new_uploads)} file(s)")
                    st.rerun()

    # =====================================================================
    # STATE B: Data loaded — show dashboard with metrics & workflow
    # =====================================================================
    else:
        st.caption("Showing: **Your Data**")

        # ── Key metrics ──
        n_xrd = len(reg["xrd"])
        n_sem = len(reg["sem"])
        n_edx = len(reg["eds"]["spectra"])
        n_xps = len(reg["xps"])
        n_tem = len(reg.get("tem", []))
        n_tem_eds = len(reg.get("tem_eds", []))

        _img_reg = reg.get("images", {})
        n_tem_images = len(_img_reg.get("tem_raw", [])) + len(_img_reg.get("tem_processed", []))
        n_elemental_maps = len(_img_reg.get("elemental_maps", []))

        _metrics = []
        if n_xrd:
            _metrics.append(("XRD Patterns", n_xrd))
        if n_sem:
            _metrics.append(("SEM Images", n_sem))
        if n_tem or n_tem_images:
            _metrics.append(("TEM Images", max(n_tem, n_tem_images)))
        if n_edx or n_tem_eds:
            _metrics.append(("EDS Spectra", n_edx + n_tem_eds))
        if n_elemental_maps:
            _metrics.append(("Elemental Maps", n_elemental_maps))
        if n_xps:
            _metrics.append(("XPS Regions", n_xps))
        all_sample_names = set(reg["xrd"].keys())
        all_sample_names.update(r.get("sample_name", "") for r in reg["sem"])
        all_sample_names.update(r.get("sample_name", "") for r in reg.get("tem", []))
        all_sample_names.discard("")
        n_total = len(all_sample_names) if all_sample_names else sum(v for _, v in _metrics)
        _metrics.append(("Total Samples", n_total))

        metric_cols = st.columns(len(_metrics))
        for col, (label, val) in zip(metric_cols, _metrics):
            col.metric(label, val)

        st.markdown("---")

        # ── Available Analysis — tells user what they can do next ──
        st.markdown("### Available Analysis")
        st.caption("Based on your loaded data, the following analysis pages are available in the sidebar:")

        _available_techniques = []
        if n_xrd:
            _available_techniques.append(
                ("XRD Analysis", "#06b6d4",
                 f"**{n_xrd}** patterns",
                 "Interactive pattern explorer, peak identification, Rietveld refinement results, "
                 "phase comparison across samples"))
        if n_xps:
            _available_techniques.append(
                ("XPS Analysis", "#8b5cf6",
                 f"**{n_xps}** regions",
                 "High-resolution spectra, peak deconvolution, surface composition quantification"))
        if n_sem:
            _available_techniques.append(
                ("SEM Gallery", "#10b981",
                 f"**{n_sem}** images",
                 "Image metadata catalog, imaging conditions overview, magnification & voltage distributions"))
        if n_tem:
            _available_techniques.append(
                ("TEM Analysis", "#f59e0b",
                 f"**{n_tem}** images" + (f", **{n_tem_eds}** EDS" if n_tem_eds else ""),
                 "TEM/STEM metadata, image gallery, TEM-EDS spectra at high voltage"))
        elif n_tem_eds:
            _available_techniques.append(
                ("TEM Analysis", "#f59e0b",
                 f"**{n_tem_eds}** TEM-EDS spectra",
                 "EDS spectra collected at beam voltages >= 100 kV (TEM/STEM mode)"))
        if n_edx:
            _available_techniques.append(
                ("EDS Analysis", "#ec4899",
                 f"**{n_edx}** spectra",
                 "Elemental spectra from SEM-EDS, peak identification, quantification tables"))

        if _available_techniques:
            _n_cols = min(3, len(_available_techniques))
            for row_start in range(0, len(_available_techniques), _n_cols):
                row_items = _available_techniques[row_start:row_start + _n_cols]
                cols = st.columns(_n_cols)
                for idx, (tech_name, color, count_str, desc) in enumerate(row_items):
                    with cols[idx]:
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,{color}22,{color}11);'
                            f'border:1px solid {color}44;border-radius:12px;padding:16px;min-height:140px;">'
                            f'<div style="color:{color};font-weight:700;font-size:1rem;">{tech_name}</div>'
                            f'<div style="color:#e2e8f0;font-size:0.85rem;margin:6px 0;">{count_str}</div>'
                            f'<div style="color:#64748b;font-size:0.78rem;">{desc}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("---")

        # ── Upload more data / manage ──
        with st.expander("Upload More Data"):
            upload_tab1, upload_tab2 = st.tabs(["Folder Path", "Drag & Drop Files"])
            with upload_tab1:
                folder_col1, folder_col2 = st.columns([4, 1])
                with folder_col1:
                    folder_path = st.text_input(
                        "Enter folder path:",
                        placeholder="e.g., D:/MoreData/",
                        key="folder_path_input",
                    )
                with folder_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    scan_clicked = st.button("Scan & Parse", type="primary", key="scan_folder")
                if scan_clicked and folder_path and folder_path.strip():
                    folder = Path(folder_path.strip())
                    if not folder.exists():
                        st.error(f"Folder not found: `{folder_path}`")
                    elif not folder.is_dir():
                        st.error(f"Not a directory: `{folder_path}`")
                    else:
                        new_uploads = _scan_and_parse_folder(folder)
                        if new_uploads:
                            st.success(f"Parsed **{len(new_uploads)}** file(s)")
                            st.rerun()
                        else:
                            st.warning("No supported files found.")

            with upload_tab2:
                uploaded_files = st.file_uploader(
                    "Upload raw instrument files",
                    accept_multiple_files=True,
                    type=["txt", "xrdml", "asc", "csv", "spx", "xls", "emsa",
                          "tif", "tiff", "jpg", "jpeg", "bmp", "png"],
                    key="file_uploader",
                    label_visibility="collapsed",
                )
                if uploaded_files:
                    new_uploads = []
                    for uf in uploaded_files:
                        already = any(uf.name in log for log in st.session_state.get("upload_log", []))
                        if already:
                            continue
                        data_type, parsed_data = detect_and_parse(uf)
                        if data_type and parsed_data:
                            result = merge_uploaded_data(data_type, parsed_data, uf.name)
                            if result:
                                new_uploads.append(result)
                                st.session_state["upload_log"].append(result)
                    if new_uploads:
                        st.success(f"Parsed {len(new_uploads)} file(s)")
                        st.rerun()

            # Reset button
            if st.button("Clear All Data & Start Fresh", key="reset_data"):
                st.session_state["data_registry"] = _empty_registry()
                st.session_state["upload_log"] = []
                st.session_state.pop("file_manifest", None)
                st.rerun()

        # Show upload history
        if st.session_state.get("upload_log"):
            with st.expander(f"Upload Log ({len(st.session_state['upload_log'])} files)", expanded=False):
                for entry in st.session_state["upload_log"]:
                    st.markdown(f"- {entry}")

        # ── File Intelligence Summary ──
        _manifest = st.session_state.get("file_manifest")
        if _manifest:
            with st.expander("📁 File Intelligence Report", expanded=False):
                _summ = _manifest.summary()
                st.markdown(f"**Scanned:** `{_manifest.root}`")
                fi_cols = st.columns(4)
                fi_cols[0].metric("Total Files", _summ["total_files"])
                fi_cols[1].metric("Parseable", _summ["parseable"])
                fi_cols[2].metric("Images", _summ["images"])
                fi_cols[3].metric("Samples", _summ["n_samples"])

                st.markdown("**By Technique:**")
                _fi_rows = []
                for tech, info in _summ["techniques"].items():
                    _fi_rows.append({
                        "Technique": tech.upper(),
                        "Files": info["count"],
                        "Parseable": info["parseable"],
                        "Formats": ", ".join(info["formats"][:3]),
                    })
                if _fi_rows:
                    st.dataframe(pd.DataFrame(_fi_rows), use_container_width=True, hide_index=True)

                # Show images breakdown
                _img_reg = reg.get("images", {})
                _n_tem_raw = len(_img_reg.get("tem_raw", []))
                _n_tem_proc = len(_img_reg.get("tem_processed", []))
                _n_emaps = len(_img_reg.get("elemental_maps", []))
                _n_sem_raw = len(_img_reg.get("sem_raw", []))
                if _n_tem_raw + _n_tem_proc + _n_emaps + _n_sem_raw > 0:
                    st.markdown("**Image Classification:**")
                    st.markdown(
                        f"- 🔬 TEM Raw: **{_n_tem_raw}** | "
                        f"FFT/SAED/Processed: **{_n_tem_proc}** | "
                        f"Elemental Maps: **{_n_emaps}** | "
                        f"SEM: **{_n_sem_raw}**"
                    )

        # ── XPS Composition (only if available) ──
        xps_quant = reg.get("xps_quant", {})
        if xps_quant.get("elements"):
            st.markdown("---")
            st.markdown("### Surface Composition — XPS Overview")
            comp_df = pd.DataFrame(xps_quant["elements"])
            col1, col2 = st.columns(2)
            with col1:
                fig_comp = px.pie(
                    comp_df, values="atomic_conc_pct", names="peak",
                    title="Atomic Composition (%)",
                    color_discrete_sequence=["#06b6d4", "#f43f5e", "#8b5cf6", "#10b981"],
                    hole=0.4,
                )
                fig_comp.update_layout(height=350)
                st.plotly_chart(fig_comp, width="stretch")
            with col2:
                fig_bar = px.bar(
                    comp_df, x="peak", y=["atomic_conc_pct", "mass_conc_pct"],
                    barmode="group",
                    title="Atomic vs Mass Concentration",
                    labels={"value": "Concentration (%)", "peak": "Element"},
                    color_discrete_sequence=["#06b6d4", "#8b5cf6"],
                )
                fig_bar.update_layout(height=350, legend_title="Type")
                st.plotly_chart(fig_bar, width="stretch")


# ===========================================================================
# PAGE: XRD Analysis
# ===========================================================================
elif page == "XRD Analysis":
    st.markdown("## XRD Analysis - Interactive Pattern Explorer")

    all_xrd_samples = get_xrd_samples()

    if not all_xrd_samples:
        st.error("No XRD data found. Run the Universal ETL first.")
    else:
        # Classify samples into families for filtering
        sample_families = {s: _classify_family(s) for s in all_xrd_samples}
        family_list = sorted(set(sample_families.values()))

        # --- Sidebar Controls ---
        st.sidebar.markdown("### XRD Settings")

        # Family filter
        selected_families = st.sidebar.multiselect(
            "Material Family", family_list, default=family_list, key="xrd_family_filter"
        )
        filtered_samples = [s for s in all_xrd_samples if sample_families[s] in selected_families]

        # Sample selector - multiselect up to 6 patterns
        # Default: pick first 2 or look for Ti2ALC3/Ti2C3 if they exist
        default_samples = []
        for preferred in ["Ti2ALC3", "Ti2C3"]:
            if preferred in filtered_samples:
                default_samples.append(preferred)
        if not default_samples and filtered_samples:
            default_samples = filtered_samples[:2]

        selected_samples = st.sidebar.multiselect(
            "Select Patterns (max 6)",
            filtered_samples,
            default=default_samples[:6],
            max_selections=6,
            key="xrd_sample_select",
        )

        # Display controls
        normalize = st.sidebar.checkbox("Normalize intensities", value=False, key="xrd_norm")
        log_scale = st.sidebar.checkbox("Log scale (Y-axis)", value=False, key="xrd_log")
        stack_offset = st.sidebar.checkbox("Stack patterns (offset)", value=len(selected_samples) > 2, key="xrd_stack")
        range_min, range_max = st.sidebar.slider(
            "2θ Range (°)", 5.0, 90.0, (5.0, 70.0), step=0.5, key="xrd_range"
        )
        smoothing = st.sidebar.slider("Smoothing (window)", 1, 21, 1, step=2, key="xrd_smooth")

        def smooth(y, window):
            if window <= 1:
                return y
            return np.convolve(y, np.ones(window) / window, mode="same")

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total XRD Patterns", len(all_xrd_samples))
        c2.metric("Material Families", len(family_list))
        c3.metric("Selected Patterns", len(selected_samples))
        c4.metric("Filtered by Family", len(filtered_samples))

        st.markdown("---")

        if not selected_samples:
            st.info("Select at least one XRD pattern from the sidebar.")
        else:
            # Color palette for up to 6 traces
            palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]
            xrd_trace_names = []
            xrd_default_colors = []

            fig_xrd = go.Figure()

            for i, sample_name in enumerate(selected_samples):
                try:
                    tt, intensity, meta = load_xrd(sample_name)
                except Exception as e:
                    st.warning(f"Could not load {sample_name}: {e}")
                    continue

                mask = (tt >= range_min) & (tt <= range_max)
                y = smooth(intensity[mask], smoothing)

                if normalize and y.max() > 0:
                    y = y / y.max() * 100

                # Stack offset
                offset = 0
                if stack_offset and i > 0:
                    offset = i * (y.max() * 0.3 if y.max() > 0 else 100)

                family = sample_families.get(sample_name, "Other")
                color = palette[i % len(palette)]
                label = f"{sample_name} ({family.replace('_', ' ')})"

                fig_xrd.add_trace(go.Scatter(
                    x=tt[mask], y=y + offset,
                    name=label,
                    line=dict(color=color, width=1.3),
                    hovertemplate=f"<b>{sample_name}</b><br>2θ = %{{x:.2f}}°<br>Intensity = %{{y:.0f}}<extra>{family}</extra>",
                ))
                xrd_trace_names.append(label)
                xrd_default_colors.append(color)

            y_title = "Normalized Intensity" if normalize else "Intensity (counts)"
            if stack_offset:
                y_title += " (stacked)"

            fig_xrd.update_layout(
                title=f"XRD Pattern Comparison — {len(selected_samples)} pattern(s)",
                xaxis_title="2θ (°)",
                yaxis_title=y_title,
                height=600,
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(x=0.65, y=0.95, font=dict(size=10)),
            )
            if log_scale:
                fig_xrd.update_yaxes(type="log")

            st.plotly_chart(fig_xrd, width="stretch")
            if xrd_trace_names:
                _xrd_key = "colors_xrd_pattern"
                if _xrd_key not in st.session_state:
                    st.session_state[_xrd_key] = list(xrd_default_colors)
                xrd_colors = color_customizer("xrd_pattern", xrd_trace_names, xrd_default_colors)

            # d-Spacing calculator (compact inline)
            with st.expander("d-Spacing Calculator (Bragg's Law)"):
                calc_col1, calc_col2 = st.columns([2, 3])
                with calc_col1:
                    calc_angle = st.number_input("Enter 2θ (°):", value=9.5, step=0.1, key="xrd_dspacing")
                with calc_col2:
                    wavelength = 1.54056  # Cu Ka1
                    d_spacing = wavelength / (2 * np.sin(np.radians(calc_angle / 2)))
                    st.metric("d-spacing", f"{d_spacing:.4f} Å", help="nλ = 2d·sin(θ), Cu Kα₁")

    # --- PHASE IDENTIFICATION ---
    st.markdown("---")
    st.markdown("## Phase Identification")
    st.markdown(
        "Match your experimental pattern against reference phases from the "
        "**Materials Project** database. Enter the phases you expect, and the agent "
        "will fetch reference stick patterns and assign your peaks."
    )

    _pid_xrd_samples = get_xrd_samples()
    if _pid_xrd_samples:
        # ── Controls ──
        pid_col1, pid_col2 = st.columns([1, 2])
        with pid_col1:
            pid_sample = st.selectbox(
                "Pattern to identify:",
                _pid_xrd_samples,
                key="pid_sample",
            )
        with pid_col2:
            pid_phases_str = st.text_input(
                "Target phases (comma-separated formulas):",
                placeholder="e.g., CuSe, Cs3Bi2I9",
                key="pid_phases",
                help="Enter the phases you expect. The agent will fetch reference "
                     "patterns from Materials Project for each formula.",
            )

        pid_adv1, pid_adv2 = st.columns(2)
        with pid_adv1:
            pid_tolerance = st.number_input(
                "Peak tolerance (°2θ)", value=0.3, min_value=0.05,
                max_value=1.0, step=0.05, key="pid_tol",
            )
        with pid_adv2:
            pid_prominence = st.number_input(
                "Peak prominence (%)", value=3.0, min_value=1.0,
                max_value=15.0, step=0.5, key="pid_prom",
                help="Minimum prominence as % of max intensity for peak detection.",
            )

        pid_run = st.button("Run Phase Identification", type="primary", key="pid_run")

        if pid_run:
            phase_formulas = [p.strip() for p in pid_phases_str.split(",") if p.strip()]
            if not phase_formulas:
                st.error("Enter at least one target phase (e.g., `CuSe, Cs3Bi2I9`).")
            else:
                from src.agents.xrd_analysis import (
                    fetch_reference_pattern, find_peaks, assign_peaks,
                )
                try:
                    tt, inten, meta = load_xrd(pid_sample)
                except Exception as e:
                    st.error(f"Could not load XRD data: {e}")
                    tt, inten = np.array([]), np.array([])

                if len(tt) > 0:
                    # Fetch references
                    all_refs = []
                    with st.spinner("Fetching reference patterns from Materials Project..."):
                        for formula in phase_formulas:
                            try:
                                refs = fetch_reference_pattern(formula)
                                if refs:
                                    # Use the most stable polymorph by default
                                    all_refs.append(refs[0])
                                    if len(refs) > 1:
                                        st.info(
                                            f"**{formula}**: {len(refs)} polymorphs found. "
                                            f"Using most stable: {refs[0].space_group} "
                                            f"({refs[0].material_id})"
                                        )
                                else:
                                    st.warning(f"No reference found for **{formula}**.")
                            except Exception as e:
                                st.warning(f"Failed to fetch **{formula}**: {e}")

                    if all_refs:
                        # Find peaks
                        exp_peaks = find_peaks(tt, inten, prominence_pct=pid_prominence)
                        # Assign peaks
                        assignments, summary = assign_peaks(
                            exp_peaks, all_refs, tolerance_deg=pid_tolerance,
                        )
                        st.session_state["pid_result"] = {
                            "assignments": assignments,
                            "summary": summary,
                            "refs": all_refs,
                            "exp_peaks": exp_peaks,
                            "tt": tt,
                            "inten": inten,
                            "sample": pid_sample,
                        }

        # ── Display results ──
        if "pid_result" in st.session_state:
            r = st.session_state["pid_result"]
            assignments = r["assignments"]
            summary = r["summary"]
            refs = r["refs"]
            exp_peaks = r["exp_peaks"]
            tt = r["tt"]
            inten = r["inten"]

            # Summary metrics
            st.markdown("### Results")
            _zs = summary.get("zero_shift", 0.0)
            if abs(_zs) > 0.001:
                st.info(
                    f"**Zero-shift correction applied:** {_zs:+.4f}° "
                    f"(systematic sample displacement detected)"
                )
            _sm_cols = st.columns(2 + len(refs))
            _sm_cols[0].metric("Experimental Peaks", summary["total"])
            for i, ref in enumerate(refs):
                count = summary["matched"].get(ref.formula, 0)
                _sm_cols[1 + i].metric(
                    ref.formula,
                    f"{count} matched",
                    help=f"{ref.space_group} | {ref.material_id}",
                )
            _sm_cols[-1].metric("Unmatched", summary["unmatched"])

            # ── Publication-style overlay plot ──
            st.markdown("### Pattern Overlay")
            fig_pid = go.Figure()

            # Experimental pattern (normalised)
            i_norm = inten / inten.max() * 100.0
            fig_pid.add_trace(go.Scatter(
                x=tt, y=i_norm,
                name=f"Experimental ({r['sample']})",
                line=dict(color="#e2e8f0", width=1.2),
                hovertemplate="2θ=%{x:.2f}° I=%{y:.1f}<extra>Experimental</extra>",
            ))

            # Peak markers coloured by assigned phase
            _phase_colors = {}
            _color_palette = ["#ef4444", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7"]
            for i, ref in enumerate(refs):
                _phase_colors[ref.formula] = _color_palette[i % len(_color_palette)]
            _phase_colors["Unmatched"] = "#6b7280"

            _phase_symbols = {}
            _symbol_list = ["diamond", "triangle-up", "square", "star", "cross"]
            for i, ref in enumerate(refs):
                _phase_symbols[ref.formula] = _symbol_list[i % len(_symbol_list)]
            _phase_symbols["Unmatched"] = "circle"

            # Group peaks by phase for legend
            _peaks_by_phase = {}
            for a in assignments:
                _peaks_by_phase.setdefault(a.matched_phase, []).append(a)

            for phase, peaks in _peaks_by_phase.items():
                fig_pid.add_trace(go.Scatter(
                    x=[p.exp_two_theta for p in peaks],
                    y=[p.exp_intensity for p in peaks],
                    mode="markers+text",
                    name=phase,
                    marker=dict(
                        symbol=_phase_symbols.get(phase, "circle"),
                        size=10,
                        color=_phase_colors.get(phase, "#888"),
                        line=dict(width=1, color="#000"),
                    ),
                    text=[p.hkl if p.hkl else "" for p in peaks],
                    textposition="top center",
                    textfont=dict(size=8, color=_phase_colors.get(phase, "#888")),
                    hovertemplate=(
                        "2θ=%{x:.2f}° I=%{y:.1f}<br>"
                        "%{text}<extra>" + phase + "</extra>"
                    ),
                ))

            # Reference stick patterns (below x-axis)
            _stick_base = -5  # start below zero
            _stick_gap = -25  # spacing between phases
            for i, ref in enumerate(refs):
                color = _phase_colors[ref.formula]
                y_base = _stick_base + i * _stick_gap

                # Only show peaks with intensity > 3%
                sig_peaks = [p for p in ref.peaks if p.intensity > 3.0]

                # Draw sticks
                for p in sig_peaks:
                    fig_pid.add_trace(go.Scatter(
                        x=[p.two_theta, p.two_theta],
                        y=[y_base, y_base + p.intensity / 100.0 * abs(_stick_gap) * 0.8],
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        hovertemplate=(
                            f"2θ={p.two_theta:.2f}° I={p.intensity:.1f} "
                            f"{p.hkl}<extra>{ref.formula}</extra>"
                        ),
                    ))

                # Label for this reference
                fig_pid.add_annotation(
                    x=0.01, y=y_base + abs(_stick_gap) * 0.4,
                    xref="paper", yref="y",
                    text=f"<b>{ref.formula}</b> ({ref.space_group}) — {ref.material_id}",
                    showarrow=False,
                    font=dict(size=10, color=color),
                    xanchor="left",
                )

            fig_pid.update_layout(
                xaxis_title="2θ (°)",
                yaxis_title="Intensity (normalised)",
                height=650,
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(x=0.55, y=0.98, font=dict(size=11)),
                yaxis=dict(
                    zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.2)",
                    zerolinewidth=1,
                ),
            )
            st.plotly_chart(fig_pid, use_container_width=True)

            # ── Peak Assignment Table ──
            st.markdown("### Peak Assignment Table")
            _tbl = []
            for a in assignments:
                _tbl.append({
                    "Exp 2θ (°)": f"{a.exp_two_theta:.3f}",
                    "d (Å)": f"{a.d_spacing:.4f}",
                    "Intensity": f"{a.exp_intensity:.1f}",
                    "Phase": a.matched_phase,
                    "Ref 2θ (°)": f"{a.ref_two_theta:.3f}" if a.ref_two_theta > 0 else "—",
                    "Δ2θ (°)": f"{a.delta_two_theta:.3f}" if a.ref_two_theta > 0 else "—",
                    "hkl": a.hkl or "—",
                })
            _tbl_df = pd.DataFrame(_tbl)

            # Color the Phase column
            def _color_phase(val):
                c = _phase_colors.get(val, "#888")
                return f"color: {c}; font-weight: bold"

            st.dataframe(
                _tbl_df.style.map(_color_phase, subset=["Phase"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── Reference Details ──
            with st.expander("Reference Pattern Details"):
                for ref in refs:
                    st.markdown(
                        f"**{ref.formula}** — {ref.space_group} ({ref.crystal_system}) "
                        f"| MP: `{ref.material_id}` | E_hull: {ref.energy_above_hull:.3f} eV/atom"
                    )
                    latt = ref.lattice
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;a={latt['a']:.4f} Å, "
                        f"b={latt['b']:.4f} Å, "
                        f"c={latt['c']:.4f} Å, "
                        f"α={latt['alpha']:.1f}°, β={latt['beta']:.1f}°, γ={latt['gamma']:.1f}°"
                    )
                    top_peaks = [p for p in ref.peaks if p.intensity > 10][:10]
                    if top_peaks:
                        st.caption(f"Top {len(top_peaks)} peaks (I > 10%):")
                        _rp_rows = [
                            {"2θ (°)": f"{p.two_theta:.3f}", "I (%)": f"{p.intensity:.1f}",
                             "d (Å)": f"{p.d_spacing:.4f}", "hkl": p.hkl}
                            for p in top_peaks
                        ]
                        st.dataframe(pd.DataFrame(_rp_rows), use_container_width=True, hide_index=True)

# ===========================================================================
# PAGE: XPS Analysis
# ===========================================================================
elif page == "XPS Analysis":
    st.markdown("## XPS Analysis - Interactive Spectroscopy")

    # Build available regions dynamically from registry
    _xps_reg = st.session_state["data_registry"]["xps"]
    _xps_region_keys = sorted(_xps_reg.keys())
    _xps_has_survey = any("survey" in k.lower() for k in _xps_region_keys)
    _xps_hr_keys = [k for k in _xps_region_keys if "survey" not in k.lower()]
    _xps_hr_labels = [k.replace("_", " ").title() for k in _xps_hr_keys]

    # Build radio options dynamically
    _xps_view_options = []
    if _xps_has_survey:
        _xps_view_options.append("Survey")
    _xps_view_options.extend(_xps_hr_labels)
    if len(_xps_hr_keys) > 1:
        _xps_view_options.append("All High-Res")

    # Load quantification if available (may not exist for uploaded data)
    _xps_quant_path = DATA_DIR / "xps" / "xps_quantification.json"
    xps_quant = load_json(str(_xps_quant_path)) if _xps_quant_path.exists() else {}
    if not xps_quant:
        xps_quant = st.session_state["data_registry"].get("xps_quant", {})

    # Composition summary (only if quantification data exists)
    if xps_quant.get("elements"):
        quant_cols = st.columns(min(len(xps_quant["elements"]), 6))
        for col, el in zip(quant_cols, xps_quant["elements"]):
            col.metric(el["peak"], f"{el['atomic_conc_pct']}%",
                       help=f"BE = {el['position_be_ev']} eV, FWHM = {el['fwhm_ev']} eV")
        st.markdown("---")

    # Sidebar controls
    st.sidebar.markdown("### XPS Settings")
    xps_view = st.sidebar.radio(
        "Spectrum View",
        _xps_view_options if _xps_view_options else ["No data"],
    )
    xps_normalize = st.sidebar.checkbox("Normalize spectra", value=False)

    if xps_view == "Survey":
        be, intensity, meta = load_xps_hr("survey")
        if xps_normalize:
            intensity = intensity / intensity.max() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=be, y=intensity,
            fill="tozeroy",
            fillcolor="rgba(6,182,212,0.2)",
            line=dict(color="#06b6d4", width=1.5),
            name="Survey",
            hovertemplate="BE = %{x:.1f} eV<br>Intensity = %{y:.0f} CPS<extra></extra>",
        ))

        # Mark element positions from quantification (if available)
        if xps_quant.get("elements"):
            for el in xps_quant["elements"]:
                fig.add_vline(
                    x=el["position_be_ev"], line_dash="dash",
                    line_color="rgba(255,255,255,0.4)",
                    annotation_text=el["peak"],
                    annotation_font_size=11,
                    annotation_font_color="#e2e8f0",
                )

        fig.update_layout(
            title="XPS Survey Spectrum",
            xaxis_title="Binding Energy (eV)",
            yaxis_title="Intensity (CPS)" if not xps_normalize else "Normalized Intensity",
            xaxis=dict(autorange="reversed"),
            height=550,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")

    elif xps_view == "All High-Res":
        _n_hr = len(_xps_hr_keys)
        _n_rows = (_n_hr + 1) // 2
        _n_c = min(2, _n_hr)
        _palette = ["#06b6d4", "#8b5cf6", "#f43f5e", "#10b981", "#f59e0b", "#ec4899",
                     "#14b8a6", "#a855f7"]
        fig = make_subplots(rows=_n_rows, cols=_n_c,
                           subplot_titles=_xps_hr_labels,
                           horizontal_spacing=0.08, vertical_spacing=0.12)

        for idx, el_key in enumerate(_xps_hr_keys):
            row, col = divmod(idx, _n_c)
            color = _palette[idx % len(_palette)]
            be, intensity, _ = load_xps_hr(el_key)
            if xps_normalize:
                intensity = intensity / intensity.max() * 100
            fig.add_trace(go.Scatter(
                x=be, y=intensity,
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
                line=dict(color=color, width=1.5),
                name=el_key.replace("_", " "),
                hovertemplate="BE = %{x:.1f} eV<br>%{y:.0f} CPS<extra></extra>",
            ), row=row + 1, col=col + 1)

        fig.update_xaxes(autorange="reversed")
        fig.update_layout(height=max(400, 350 * _n_rows), template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    elif xps_view != "No data":
        # Individual high-res spectrum
        el_key = xps_view.replace(" ", "_").lower()
        # Try exact key first, then case-insensitive match
        if el_key not in _xps_reg:
            el_key = next((k for k in _xps_reg if k.lower() == el_key), el_key)
        be, intensity, meta = load_xps_hr(el_key)
        if xps_normalize:
            intensity = intensity / intensity.max() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=be, y=intensity,
            fill="tozeroy",
            fillcolor="rgba(139,92,246,0.2)",
            line=dict(color="#8b5cf6", width=1.5),
            name=xps_view,
            hovertemplate="BE = %{x:.1f} eV<br>Intensity = %{y:.0f} CPS<extra></extra>",
        ))

        fig.update_layout(
            title=f"XPS High-Resolution - {xps_view}",
            xaxis_title="Binding Energy (eV)",
            yaxis_title="Intensity (CPS)" if not xps_normalize else "Normalized",
            xaxis=dict(autorange="reversed"),
            height=550,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")

        # Peak analysis info
        el_label = xps_view
        quant_el = next((e for e in xps_quant["elements"] if e["peak"] == el_label), None)
        if quant_el:
            st.markdown("### Peak Parameters")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Position", f"{quant_el['position_be_ev']} eV")
            pc2.metric("FWHM", f"{quant_el['fwhm_ev']} eV")
            pc3.metric("Atomic %", f"{quant_el['atomic_conc_pct']}%")
            pc4.metric("RSF", f"{quant_el['rsf']}")

    # Quantification table (only if data exists)
    if xps_quant.get("elements"):
        st.markdown("---")
        st.markdown("### Full Quantification Table")
        quant_df = pd.DataFrame(xps_quant["elements"])
        st.dataframe(
            quant_df.style.format({
                "position_be_ev": "{:.1f}",
                "fwhm_ev": "{:.3f}",
                "raw_area_cps_ev": "{:.1f}",
                "atomic_conc_pct": "{:.2f}",
                "mass_conc_pct": "{:.2f}",
            }).background_gradient(subset=["atomic_conc_pct"], cmap="YlOrRd"),
            width="stretch",
        )
        color_scale_bar("Atomic %", "Low", "High", ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"])

    # --- LAYER 2: XPS Peak Deconvolution ---
    if not _xps_hr_keys:
        st.info("Upload high-resolution XPS spectra to enable peak deconvolution.")
    else:
        st.markdown("---")
        st.markdown("## Layer 2: Peak Deconvolution")

        from src.analysis.xps_analysis import full_xps_analysis, gl_peak, XPS_REFERENCES

        deconv_element = st.selectbox(
            "Element to deconvolve",
            _xps_hr_labels,
            key="xps_deconv_el",
        )

        dcol1, dcol2, dcol3 = st.columns(3)
        bg_type = dcol1.selectbox("Background", ["shirley", "linear"], key="xps_bg")
        gl_frac = dcol2.slider("GL mixing (0=Gauss, 1=Lorentz)", 0.0, 1.0, 0.3, step=0.1, key="xps_gl")
        auto_n = dcol3.checkbox("Auto-detect components", value=True, key="xps_auto")

        el_key = deconv_element.replace(" ", "_")
        d_be, d_int, _ = load_xps_hr(el_key)

        n_comp = None
        if not auto_n:
            n_comp = st.number_input("Number of components", 2, 8, 3, key="xps_ncomp")

        # --- Initial deconvolution (auto-detect) ---
        _init_key = f"xps_init_result_{deconv_element}"
        with st.spinner("Running deconvolution..."):
            xps_result_init = full_xps_analysis(d_be, d_int, deconv_element,
                                                background_type=bg_type, gl_fraction=gl_frac)

        deconv_init = xps_result_init["deconvolution"]

        # --- Component Editor ---
        refit = False
        comp_states = {}
        _custom_key = f"xps_custom_peaks_{deconv_element}"
        if _custom_key not in st.session_state:
            st.session_state[_custom_key] = []

        if deconv_init.n_components > 0:
            with st.expander("✏️ Edit Components (add/remove peaks, then re-fit)", expanded=False):
                st.caption("Toggle components on/off or add custom peaks. Click **Re-fit** to update.")

                # Checkboxes for each auto-detected component
                comp_states = {}
                cols_per_row = 4
                comp_list = deconv_init.components
                for row_start in range(0, len(comp_list), cols_per_row):
                    row_cols = st.columns(min(cols_per_row, len(comp_list) - row_start))
                    for j, col in enumerate(row_cols):
                        idx = row_start + j
                        if idx < len(comp_list):
                            c = comp_list[idx]
                            label = f"{c.assignment} ({c.center_ev:.1f} eV)"
                            comp_states[idx] = col.checkbox(
                                label, value=True,
                                key=f"xps_comp_{deconv_element}_{idx}",
                            )

                st.markdown("---")
                st.markdown("**Add Custom Component**")
                add_cols = st.columns([2, 2, 1])
                custom_be = add_cols[0].number_input(
                    "Center BE (eV)", min_value=float(min(d_be)), max_value=float(max(d_be)),
                    value=float(np.mean(d_be)), step=0.5, key=f"xps_custom_be_{deconv_element}",
                )
                custom_label = add_cols[1].text_input(
                    "Label", value="Custom", key=f"xps_custom_label_{deconv_element}",
                )

                # Build custom positions from selections
                refit = st.button("🔄 Re-fit with selected components", key=f"xps_refit_{deconv_element}")

                # Store custom additions in session state
                _custom_key = f"xps_custom_peaks_{deconv_element}"
                if _custom_key not in st.session_state:
                    st.session_state[_custom_key] = []

                add_peak = add_cols[2].button("➕ Add", key=f"xps_add_{deconv_element}")
                if add_peak:
                    st.session_state[_custom_key].append({
                        "be": custom_be, "label": custom_label,
                    })
                    st.rerun()

                # Show custom peaks added
                if st.session_state[_custom_key]:
                    st.markdown("**Custom peaks added:**")
                    for ci, cp in enumerate(st.session_state[_custom_key]):
                        cc1, cc2 = st.columns([4, 1])
                        cc1.write(f"• {cp['label']} at {cp['be']:.1f} eV")
                        if cc2.button("❌", key=f"xps_rm_{deconv_element}_{ci}"):
                            st.session_state[_custom_key].pop(ci)
                            st.rerun()

            # Determine if re-fit is needed
            selected_positions = [comp_list[i].center_ev for i, on in comp_states.items() if on]
            # Add custom peaks
            for cp in st.session_state.get(_custom_key, []):
                selected_positions.append(cp["be"])

            # Check if user modified anything
            original_positions = [c.center_ev for c in comp_list]
            custom_peaks_exist = len(st.session_state.get(_custom_key, [])) > 0
            some_disabled = any(not on for on in comp_states.values())
            needs_refit = refit or custom_peaks_exist or some_disabled

            if needs_refit and selected_positions:
                with st.spinner("Re-fitting with edited components..."):
                    xps_result = full_xps_analysis(
                        d_be, d_int, deconv_element,
                        background_type=bg_type, gl_fraction=gl_frac,
                        initial_positions=selected_positions,
                        n_components=len(selected_positions),
                    )
            else:
                xps_result = xps_result_init
        else:
            xps_result = xps_result_init

        deconv = xps_result["deconvolution"]

        if deconv.n_components > 0 and deconv.binding_energy:
            be_arr = np.array(deconv.binding_energy)
            raw_arr = np.array(deconv.raw_intensity)
            bg_arr = np.array(deconv.background)

            # Build trace names and defaults for color customizer
            xps_trace_names = ["Raw", f"Background ({bg_type})", "Envelope"]
            xps_default_colors = ["#94a3b8", "#475569", "#ef4444"]
            comp_defaults = ["#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899",
                             "#3b82f6", "#14b8a6", "#f43f5e"]
            for i, comp in enumerate(deconv.components):
                xps_trace_names.append(f"{comp.assignment} ({comp.center_ev:.1f} eV)")
                xps_default_colors.append(comp_defaults[i % len(comp_defaults)])

            _xps_key = f"colors_xps_deconv_{deconv_element}"
            if _xps_key not in st.session_state:
                st.session_state[_xps_key] = list(xps_default_colors)
            xps_colors = st.session_state[_xps_key]

            fig_deconv = go.Figure()

            # Raw spectrum
            fig_deconv.add_trace(go.Scatter(
                x=be_arr, y=raw_arr, name="Raw",
                line=dict(color=xps_colors[0], width=1),
            ))

            # Background
            fig_deconv.add_trace(go.Scatter(
                x=be_arr, y=bg_arr, name=f"Background ({bg_type})",
                line=dict(color=xps_colors[1], width=1, dash="dot"),
            ))

            # Envelope
            if deconv.envelope:
                fig_deconv.add_trace(go.Scatter(
                    x=be_arr, y=np.array(deconv.envelope),
                    name="Envelope", line=dict(color=xps_colors[2], width=2),
                ))

            # Individual components (filled to background, not to zero)
            for i, (comp, curve) in enumerate(zip(deconv.components, deconv.component_curves)):
                curve_arr = np.array(curve)
                comp_color = xps_colors[3 + i] if (3 + i) < len(xps_colors) else comp_defaults[i % len(comp_defaults)]
                # Add background baseline trace (invisible) so fill goes to background
                fig_deconv.add_trace(go.Scatter(
                    x=be_arr, y=bg_arr,
                    showlegend=False, mode="lines",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ))
                fig_deconv.add_trace(go.Scatter(
                    x=be_arr, y=curve_arr + bg_arr,
                    name=f"{comp.assignment} ({comp.center_ev:.1f} eV)",
                    fill="tonexty", opacity=0.4,
                    line=dict(color=comp_color, width=1.5),
                ))

            fig_deconv.update_layout(
                title=f"XPS Deconvolution - {deconv_element} ({deconv.n_components} components, R2={deconv.envelope_r_squared:.3f})",
                xaxis_title="Binding Energy (eV)",
                yaxis_title="Intensity (CPS)",
                xaxis=dict(autorange="reversed"),
                height=550, template="plotly_dark",
            )
            st.plotly_chart(fig_deconv, width="stretch")
            xps_colors = color_customizer(f"xps_deconv_{deconv_element}", xps_trace_names, xps_default_colors)

            # Export fitted data as CSV
            export_df = pd.DataFrame({
                "binding_energy_eV": be_arr,
                "raw_intensity_CPS": raw_arr,
                f"background_{bg_type}_CPS": bg_arr,
            })
            if deconv.envelope:
                export_df["envelope_CPS"] = np.array(deconv.envelope)
            for comp, curve in zip(deconv.components, deconv.component_curves):
                col_name = f"{comp.assignment}_{comp.center_ev:.1f}eV_CPS"
                export_df[col_name] = np.array(curve) + bg_arr

            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Export {deconv_element} fitted data as CSV",
                data=csv_data,
                file_name=f"xps_{deconv_element.replace(' ', '_')}_deconvolution.csv",
                mime="text/csv",
            )

            # Component quantification — table only (pie removed to reduce clutter)
            st.markdown("### Component Quantification")
            if xps_result["quantification"]:
                q_df = pd.DataFrame(xps_result["quantification"])
                st.dataframe(
                    q_df.style.format({
                        "center_ev": "{:.1f}",
                        "fwhm_ev": "{:.2f}",
                        "area": "{:.1f}",
                        "relative_pct": "{:.1f}",
                    }).background_gradient(subset=["relative_pct"], cmap="Blues"),
                    width="stretch",
                )
                color_scale_bar("Relative %", "Low", "High", ["#f7fbff", "#6baed6", "#2171b5", "#08306b"])

            # Chemical state interpretation
            st.markdown("### Chemical State Interpretation")
            if deconv_element in XPS_REFERENCES:
                ref = XPS_REFERENCES[deconv_element]
                for comp_ref in ref["components"]:
                    matched = next((c for c in deconv.components
                                   if c.assignment == comp_ref["name"]), None)
                    if matched:
                        rel_pct = next((q["relative_pct"] for q in xps_result["quantification"]
                                       if q["component"] == comp_ref["name"]), 0)
                        doi = comp_ref.get("doi", "")
                        ref_text = comp_ref.get("reference", "")
                        ref_link = f" — [{ref_text}]({doi})" if doi else ""
                        st.markdown(
                            f"**{comp_ref['name']}** ({matched.center_ev:.1f} eV, "
                            f"{rel_pct:.1f}%): {comp_ref['description']}{ref_link}"
                        )
        else:
            st.warning("Deconvolution did not converge. Try adjusting parameters.")


# ===========================================================================
# PAGE: SEM Gallery
# ===========================================================================
elif page == "SEM Gallery":
    st.markdown("## SEM Gallery - Morphology Analysis")

    # Load from universal ETL: merge JEOL + Hitachi catalogs
    @st.cache_data
    def load_sem_catalog():
        sem_records = []
        # Try universal ETL first
        jeol_path = DATA_DIR / "universal" / "sem" / "jeol_sem_catalog.json"
        hitachi_path = DATA_DIR / "universal" / "sem" / "hitachi_sem_catalog.json"
        old_path = DATA_DIR / "sem" / "sem_catalog.json"

        if jeol_path.exists():
            jeol_data = load_json(str(jeol_path))
            for rec in jeol_data:
                # Normalize keys to unified format
                sem_records.append({
                    "image_name": rec.get("source_file", ""),
                    "sample_name": rec.get("sample_name", ""),
                    "instrument": rec.get("instrument", "JEOL FE-SEM"),
                    "magnification": rec.get("magnification", 0),
                    "accelerating_voltage_kv": rec.get("accelerating_voltage_kv", 0),
                    "working_distance_um": rec.get("working_distance_um", 0),
                    "pixel_size_nm": rec.get("pixel_size_nm", 0),
                    "field_of_view_um": rec.get("field_of_view_um", 0),
                    "emission_current": rec.get("emission_current", 0),
                    "signal": rec.get("signal_name", ""),
                    "date": rec.get("date", ""),
                    "image_path": rec.get("image_path", ""),
                    "has_image": rec.get("has_image", False),
                })

        if hitachi_path.exists():
            hitachi_data = load_json(str(hitachi_path))
            for rec in hitachi_data:
                sem_records.append({
                    "image_name": rec.get("image_name", rec.get("source_file", "")),
                    "sample_name": rec.get("sample_name", ""),
                    "instrument": rec.get("instrument", "Hitachi"),
                    "magnification": rec.get("magnification", 0),
                    "accelerating_voltage_kv": rec.get("accelerating_voltage_v", 0) / 1000
                        if rec.get("accelerating_voltage_v", 0) > 100
                        else rec.get("accelerating_voltage_v", 0),
                    "working_distance_um": rec.get("working_distance_um", 0),
                    "pixel_size_nm": rec.get("pixel_size_nm", 0),
                    "field_of_view_um": 0,
                    "emission_current": rec.get("emission_current_na", 0),
                    "signal": rec.get("signal", ""),
                    "date": rec.get("date", ""),
                    "image_path": rec.get("image_path", ""),
                    "has_image": rec.get("has_image", False),
                })

        # Fallback to old catalog
        if not sem_records and old_path.exists():
            old_data = load_json(str(old_path))
            for rec in old_data:
                sem_records.append({
                    "image_name": rec.get("image_name", ""),
                    "sample_name": rec.get("sample_name", ""),
                    "instrument": rec.get("instrument", "Unknown"),
                    "magnification": rec.get("magnification", 0),
                    "accelerating_voltage_kv": rec.get("accelerating_voltage_v", 0) / 1000
                        if rec.get("accelerating_voltage_v", 0) > 100
                        else rec.get("accelerating_voltage_v", 0),
                    "working_distance_um": rec.get("working_distance_um", 0),
                    "pixel_size_nm": rec.get("pixel_size_nm", 0),
                    "field_of_view_um": 0,
                    "emission_current": rec.get("emission_current_na", 0),
                    "signal": rec.get("signal", ""),
                    "date": rec.get("date", ""),
                    "image_path": rec.get("image_path", ""),
                    "has_image": rec.get("has_image", False),
                })

        return sem_records

    sem_catalog = load_sem_catalog()
    sem_df = pd.DataFrame(sem_catalog)

    if sem_df.empty:
        st.error("No SEM data found. Run the Universal ETL first.")
        st.stop()

    # Classify into material families
    sem_df["family"] = sem_df["sample_name"].apply(_classify_family)

    # Filters
    st.sidebar.markdown("### SEM Filters")

    # Instrument filter
    instruments = sorted(sem_df["instrument"].unique().tolist())
    selected_instruments = st.sidebar.multiselect(
        "Instrument", instruments, default=instruments, key="sem_instrument"
    )

    # Material family filter
    sem_families = sorted(sem_df["family"].unique().tolist())
    selected_sem_families = st.sidebar.multiselect(
        "Material Family", sem_families, default=sem_families, key="sem_family"
    )

    # Voltage filter
    voltage_options = sorted(sem_df["accelerating_voltage_kv"].unique())
    if len(voltage_options) > 1:
        voltage_range = st.sidebar.select_slider(
            "Accelerating Voltage (kV)",
            options=voltage_options,
            value=(voltage_options[0], voltage_options[-1]),
            key="sem_voltage",
        )
    else:
        voltage_range = (voltage_options[0], voltage_options[0])

    filtered = sem_df[
        (sem_df["instrument"].isin(selected_instruments)) &
        (sem_df["family"].isin(selected_sem_families)) &
        (sem_df["accelerating_voltage_kv"] >= voltage_range[0]) &
        (sem_df["accelerating_voltage_kv"] <= voltage_range[1])
    ]

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", len(filtered))
    c2.metric("With Image", int(filtered["has_image"].sum()))
    c3.metric("Instruments", len(filtered["instrument"].unique()))
    c4.metric("Material Families", len(filtered["family"].unique()))
    mag_range_str = f"{filtered['magnification'].min():.0f}x – {filtered['magnification'].max():.0f}x" if len(filtered) > 0 else "N/A"
    c5.metric("Mag Range", mag_range_str)

    st.markdown("---")

    # Image gallery
    st.markdown("### Image Gallery")
    images_with_tif = filtered[filtered["has_image"] == True].sort_values("magnification")

    if len(images_with_tif) == 0:
        st.warning("No images found with the current filters.")
    else:
        # Let user select magnification
        mag_options = sorted(images_with_tif["magnification"].unique())
        selected_mag = st.select_slider(
            "Select Magnification",
            options=mag_options,
            value=mag_options[0],
            format_func=lambda x: f"{x:,.0f}x",
        )

        mag_images = images_with_tif[images_with_tif["magnification"] == selected_mag]

        cols = st.columns(min(len(mag_images), 3))
        for idx, (_, row) in enumerate(mag_images.iterrows()):
            with cols[idx % len(cols)]:
                img_path = Path(row["image_path"])
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=f"{row['image_name']}", use_container_width=True)
                    except Exception:
                        st.info(f"Cannot load: {row['image_name']}")
                else:
                    st.info(f"File not found: {row['image_name']}")

                st.markdown(
                    f"**Sample:** {row['sample_name']} ({row.get('family', '')})  \n"
                    f"**Instrument:** {row.get('instrument', 'N/A')} | "
                    f"**Voltage:** {row['accelerating_voltage_kv']:.0f} kV | "
                    f"**Pixel:** {row['pixel_size_nm']:.1f} nm"
                )

    # --- LAYER 2: SEM Morphological Analysis ---
    st.markdown("---")
    st.markdown("## Layer 2: Morphological Analysis")
    st.markdown("*Automated flake/particle size measurement and layer thickness estimation using image segmentation.*")

    from src.analysis.sem_analysis import (
        full_sem_analysis, measure_layer_thickness, surface_roughness,
        load_sem_image, preprocess,
    )

    # Only analyze images that exist
    analyzable = filtered[filtered["has_image"] == True].copy()

    if len(analyzable) == 0:
        st.warning("No images available for analysis.")
    else:
        sem_col1, sem_col2 = st.columns(2)

        with sem_col1:
            sem_img_select = st.selectbox(
                "Select image for analysis",
                analyzable["image_name"].tolist(),
                index=0,
                key="sem_analysis_img",
            )

        with sem_col2:
            seg_method = st.selectbox(
                "Segmentation method",
                ["adaptive", "otsu", "watershed"],
                index=0,
                help="Adaptive: best for uneven illumination. Otsu: global threshold. Watershed: overlapping particles.",
            )

        sem_row = analyzable[analyzable["image_name"] == sem_img_select].iloc[0]

        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            denoise_sig = st.slider("Denoise σ", 0.5, 5.0, 1.5, 0.5, key="sem_denoise")
        with adv_col2:
            min_area = st.slider("Min area (px)", 50, 1000, 200, 50, key="sem_min_area")
        with adv_col3:
            invert_seg = st.checkbox("Invert segmentation", value=False, key="sem_invert")
        with adv_col4:
            n_bins = st.slider("Histogram bins", 5, 50, 20, 5, key="sem_bins")

        img_path = str(Path(sem_row["image_path"]))
        pixel_nm = sem_row["pixel_size_nm"]
        mag = sem_row["magnification"]

        try:
            sem_result = full_sem_analysis(
                image_path=img_path,
                pixel_size_nm=pixel_nm,
                magnification=mag,
                image_name=sem_img_select,
                method=seg_method,
                denoise_sigma=denoise_sig,
                min_area_px=min_area,
                invert=invert_seg,
                n_bins=n_bins,
            )

            if sem_result and sem_result.n_particles > 0:
                # Summary metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Particles Detected", sem_result.n_particles)

                # Auto-select best unit
                mean_d = sem_result.mean_diameter_nm
                if mean_d >= 1000:
                    m2.metric("Mean Diameter", f"{mean_d/1000:.2f} μm")
                    m3.metric("Median Diameter", f"{sem_result.median_diameter_nm/1000:.2f} μm")
                    m4.metric("Std Dev", f"{sem_result.std_diameter_nm/1000:.2f} μm")
                else:
                    m2.metric("Mean Diameter", f"{mean_d:.0f} nm")
                    m3.metric("Median Diameter", f"{sem_result.median_diameter_nm:.0f} nm")
                    m4.metric("Std Dev", f"{sem_result.std_diameter_nm:.0f} nm")
                m5.metric("Mean Aspect Ratio", f"{sem_result.mean_aspect_ratio:.2f}")

                # Two-column layout: histogram + segmentation overlay
                # SEM color defaults
                _sem_h_key = "colors_sem_histogram"
                _sem_h_defaults = ["#06b6d4", "#ef4444", "#22c55e"]
                if _sem_h_key not in st.session_state:
                    st.session_state[_sem_h_key] = list(_sem_h_defaults)
                sem_hist_colors = st.session_state[_sem_h_key]

                # Particle size distribution histogram (full width)
                if sem_result.size_bins_nm and sem_result.size_counts:
                    bins_display = sem_result.size_bins_nm
                    x_label = "Equivalent Diameter (nm)"
                    if mean_d >= 1000:
                        bins_display = [b / 1000 for b in sem_result.size_bins_nm]
                        x_label = "Equivalent Diameter (μm)"

                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Bar(
                        x=bins_display,
                        y=sem_result.size_counts,
                        marker_color=sem_hist_colors[0],
                        opacity=0.85,
                        hovertemplate=f"{x_label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                    ))
                    fig_hist.update_layout(
                        title="Flake/Particle Size Distribution",
                        xaxis_title=x_label,
                        yaxis_title="Count",
                        template="plotly_dark",
                        height=400,
                        bargap=0.05,
                    )
                    mean_line = mean_d / 1000 if mean_d >= 1000 else mean_d
                    median_line = sem_result.median_diameter_nm / 1000 if mean_d >= 1000 else sem_result.median_diameter_nm
                    fig_hist.add_vline(x=mean_line, line_dash="dash",
                                       line_color=sem_hist_colors[1],
                                       annotation_text=f"Mean: {mean_line:.2f}")
                    fig_hist.add_vline(x=median_line, line_dash="dot",
                                       line_color=sem_hist_colors[2],
                                       annotation_text=f"Median: {median_line:.2f}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    sem_hist_colors = color_customizer("sem_histogram",
                        ["Bars", "Mean line", "Median line"], _sem_h_defaults)

                # Segmentation visualization
                with st.expander("Segmentation Visualization", expanded=False):
                    seg_col1, seg_col2 = st.columns(2)
                    with seg_col1:
                        st.markdown("**Binary Segmentation**")
                        if sem_result.binary is not None:
                            st.image(sem_result.binary.astype(np.uint8) * 255,
                                     caption="Segmented regions (white = detected features)",
                                     use_container_width=True)
                    with seg_col2:
                        st.markdown("**Edge Detection (Canny)**")
                        if sem_result.edges is not None:
                            st.image(sem_result.edges.astype(np.uint8) * 255,
                                     caption="Feature boundaries",
                                     use_container_width=True)

                # Detailed particle table
                with st.expander("Particle Measurements Table"):
                    particle_data = []
                    for p in sem_result.particles:
                        if mean_d >= 1000:
                            particle_data.append({
                                "label": p.label,
                                "diameter_μm": round(p.equivalent_diameter_nm / 1000, 3),
                                "major_axis_μm": round(p.major_axis_nm / 1000, 3),
                                "minor_axis_μm": round(p.minor_axis_nm / 1000, 3),
                                "aspect_ratio": round(p.aspect_ratio, 2),
                                "circularity": round(p.circularity, 3),
                                "solidity": round(p.solidity, 3),
                                "area_μm²": round(p.area_nm2 / 1e6, 4),
                            })
                        else:
                            particle_data.append({
                                "label": p.label,
                                "diameter_nm": round(p.equivalent_diameter_nm, 1),
                                "major_axis_nm": round(p.major_axis_nm, 1),
                                "minor_axis_nm": round(p.minor_axis_nm, 1),
                                "aspect_ratio": round(p.aspect_ratio, 2),
                                "circularity": round(p.circularity, 3),
                                "solidity": round(p.solidity, 3),
                                "area_nm²": round(p.area_nm2, 0),
                            })
                    pdf = pd.DataFrame(particle_data)
                    st.dataframe(pdf.reset_index(drop=True), width="stretch")

                    st.markdown("""
                    <div style="margin-top:10px; padding:12px 16px; background:rgba(255,255,255,0.03);
                         border-radius:8px; border:1px solid rgba(255,255,255,0.08); font-size:0.82em; color:#aaa;">
                    <b style="color:#ccc;">Column Definitions</b><br>
                    <b>diameter</b> — Equivalent circular diameter: diameter of a circle with the same area as the particle<br>
                    <b>major_axis</b> — Length of the longest axis of the best-fit ellipse<br>
                    <b>minor_axis</b> — Length of the shortest axis of the best-fit ellipse<br>
                    <b>aspect_ratio</b> — major/minor axis. 1.0 = circular, >2 = elongated flake<br>
                    <b>circularity</b> — 4π·area/perimeter². 1.0 = perfect circle, <0.5 = irregular shape<br>
                    <b>solidity</b> — area/convex_hull_area. 1.0 = convex, <0.8 = concave or branched shape<br>
                    <b>area</b> — Physical area of the particle (calibrated from pixel size)
                    </div>
                    """, unsafe_allow_html=True)

                # Surface roughness
                with st.expander("Surface Roughness Metrics"):
                    raw_img = load_sem_image(img_path)
                    if raw_img is not None:
                        roughness = surface_roughness(raw_img, pixel_nm)
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        rc1.metric("Ra (arithmetic)", f"{roughness['Ra']:.4f}")
                        rc2.metric("Rq (RMS)", f"{roughness['Rq']:.4f}")
                        rc3.metric("Rsk (skewness)", f"{roughness['Rsk']:.3f}")
                        rc4.metric("Rku (kurtosis)", f"{roughness['Rku']:.3f}")
                        st.markdown("""
                        <div style="margin-top:10px; padding:12px 16px; background:rgba(255,255,255,0.03);
                             border-radius:8px; border:1px solid rgba(255,255,255,0.08); font-size:0.82em; color:#aaa;">
                        <b style="color:#ccc;">Roughness Parameters</b> (intensity-based proxy — quantitative roughness requires AFM)<br>
                        <b>Ra</b> — Arithmetic average of absolute deviations from mean. Lower = smoother surface<br>
                        <b>Rq</b> — Root-mean-square roughness. More sensitive to peaks/valleys than Ra<br>
                        <b>Rsk</b> — Skewness. Positive = surface has more peaks, Negative = more valleys<br>
                        <b>Rku</b> — Kurtosis. >3 = sharp peaks/valleys (spiky), <3 = rounded features
                        </div>
                        """, unsafe_allow_html=True)

                # Layer thickness (for high magnification images)
                if mag >= 50000:
                    with st.expander("Layer Thickness Estimation", expanded=True):
                        st.markdown("*For cross-section SEM at high magnification (≥50k×)*")
                        raw_img = load_sem_image(img_path)
                        if raw_img is not None:
                            processed = preprocess(raw_img)
                            lt_result = measure_layer_thickness(
                                processed, pixel_size_nm=pixel_nm, prominence=0.15
                            )
                            lt_result.image_name = sem_img_select

                            if lt_result.n_layers > 1 and lt_result.thicknesses_nm:
                                lt1, lt2, lt3 = st.columns(3)
                                lt1.metric("Layers Detected", lt_result.n_layers)
                                lt2.metric("Mean Thickness", f"{lt_result.mean_thickness_nm:.2f} nm")
                                lt3.metric("Std Dev", f"{lt_result.std_thickness_nm:.2f} nm")

                                # Profile plot
                                fig_lt = go.Figure()
                                fig_lt.add_trace(go.Scatter(
                                    x=lt_result.profile_position,
                                    y=lt_result.profile_intensity,
                                    mode="lines",
                                    name="Intensity Profile",
                                    line=dict(color="#06b6d4"),
                                ))
                                # Mark layer boundaries
                                for pk_nm in lt_result.peak_positions_nm:
                                    fig_lt.add_vline(
                                        x=pk_nm, line_dash="dot",
                                        line_color="rgba(239,68,68,0.5)",
                                        line_width=1,
                                    )
                                fig_lt.update_layout(
                                    title="Vertical Intensity Profile with Layer Boundaries",
                                    xaxis_title="Position (nm)",
                                    yaxis_title="Normalized Intensity",
                                    template="plotly_dark",
                                    height=350,
                                )
                                st.plotly_chart(fig_lt, use_container_width=True)
                            else:
                                st.info("Insufficient layer contrast for thickness measurement at this position.")

            elif sem_result:
                st.warning(f"No particles detected. Try adjusting: lower min area, change segmentation method, or toggle invert.")
            else:
                st.error("Analysis failed. Check image path.")

        except Exception as e:
            st.error(f"SEM analysis error: {e}")

    st.markdown("---")

    # Imaging conditions table removed — data available via sidebar filters and hover info


# ===========================================================================
# PAGE: EDS Analysis
# ===========================================================================
elif page == "EDS Analysis":
    st.markdown("## EDS/EDX Analysis - Elemental Identification")

    # Load data from both sources: old EMSA + universal Bruker EDX
    @st.cache_data
    def load_all_eds_data():
        """Load EDS data from both old EMSA and universal Bruker EDX."""
        spectra = []  # unified list of spectrum info

        # Source 1: Old EMSA data
        old_path = DATA_DIR / "eds" / "eds_peaks_summary.json"
        if old_path.exists():
            emsa_data = load_json(str(old_path))
            for item in emsa_data:
                spectra.append({
                    "source": item.get("source", ""),
                    "label": item.get("source", "").replace(".emsa", ""),
                    "format": "EMSA",
                    "beam_kv": item.get("beam_kv", 0),
                    "live_time_s": item.get("live_time_s", 0),
                    "peaks": item.get("peaks", []),
                    "peak_key": "element_line",
                    "energy_unit": "ev",  # EMSA uses eV
                })

        # Source 2: Universal Bruker EDX
        bruker_path = DATA_DIR / "universal" / "edx" / "bruker_edx_spectra_summary.json"
        if bruker_path.exists():
            bruker_data = load_json(str(bruker_path))
            for item in bruker_data:
                spectra.append({
                    "source": item.get("source_file", ""),
                    "label": f"Bruker_{item.get('source_file', '').replace('.spx', '')}",
                    "format": "Bruker SPX",
                    "beam_kv": 0,  # Not in SPX metadata
                    "live_time_s": (item.get("live_time_ms", 0) or 0) / 1000,
                    "real_time_s": (item.get("real_time_ms", 0) or 0) / 1000,
                    "dead_time_pct": item.get("dead_time_pct", 0),
                    "detector": item.get("detector", ""),
                    "total_counts": item.get("total_counts", 0),
                    "n_channels": item.get("n_channels", 0),
                    "peaks": item.get("detected_peaks", []),
                    "peak_key": "element_line",
                    "energy_unit": "kev",  # Bruker uses keV
                    "sample_group": item.get("sample_group", ""),
                    "json_file": item.get("source_file", "").replace(".spx", ""),
                })

        # Load Bruker quantification data
        quant_data = []
        quant_path = DATA_DIR / "universal" / "edx" / "bruker_edx_quantification.json"
        if quant_path.exists():
            quant_data = load_json(str(quant_path))

        return spectra, quant_data

    eds_spectra, edx_quant_data = load_all_eds_data()

    if not eds_spectra:
        st.error("No EDS/EDX data found.")
        st.stop()

    # Element reference lines for EDS (expanded)
    EDS_LINES = {
        "B Kα": 0.183, "C Kα": 0.277, "N Kα": 0.392, "O Kα": 0.525,
        "F Kα": 0.677, "Na Kα": 1.041, "Mg Kα": 1.254, "Al Kα": 1.487,
        "Si Kα": 1.740, "P Kα": 2.013, "S Kα": 2.307, "Cl Kα": 2.622,
        "K Kα": 3.314, "Ca Kα": 3.691, "Ti Kα": 4.511, "Ti Kβ": 4.932,
        "Ti Lα": 0.452, "Fe Kα": 6.404, "Co Kα": 6.930,
        "Ni Kα": 7.471, "Cu Kα": 8.048, "Cu Lα": 0.930,
        "Zn Kα": 8.638, "Ag Lα": 2.984, "Bi Lα": 10.839,
        "Se Kα": 11.222, "Te Lα": 3.769,
    }

    # Summary metrics
    n_emsa = sum(1 for s in eds_spectra if s["format"] == "EMSA")
    n_bruker = sum(1 for s in eds_spectra if s["format"] == "Bruker SPX")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Spectra", len(eds_spectra))
    m2.metric("EMSA", n_emsa)
    m3.metric("Bruker EDX", n_bruker)
    m4.metric("Quantifications", len(edx_quant_data))

    st.markdown("---")

    # Sidebar controls
    st.sidebar.markdown("### EDS/EDX Settings")
    format_filter = st.sidebar.multiselect(
        "Data Source", ["EMSA", "Bruker SPX"],
        default=["EMSA", "Bruker SPX"] if n_bruker > 0 else ["EMSA"],
        key="eds_format",
    )
    filtered_spectra = [s for s in eds_spectra if s["format"] in format_filter]
    spectrum_labels = [s["label"] for s in filtered_spectra]

    selected_spectrum = st.sidebar.selectbox("Select Spectrum", spectrum_labels, key="eds_select")
    eds_log = st.sidebar.checkbox("Log scale (Y)", value=False, key="eds_log")
    eds_range = st.sidebar.slider("Energy Range (keV)", 0.0, 20.0, (0.0, 12.0), step=0.1, key="eds_range")
    show_element_lines = st.sidebar.checkbox("Show element markers", value=True, key="eds_markers")

    # Get selected spectrum info
    spec_info = next((s for s in filtered_spectra if s["label"] == selected_spectrum), None)

    if spec_info:
        # Try to load full spectrum data
        spectrum_loaded = False
        energy = None
        counts = None

        if spec_info["format"] == "Bruker SPX":
            # Load individual Bruker spectrum JSON (has full energy/counts arrays)
            edx_dir = DATA_DIR / "universal" / "edx"
            # Find the matching JSON file
            bruker_files = sorted(edx_dir.glob("bruker_edx_*.json"))
            for bf in bruker_files:
                try:
                    bdata = load_json(str(bf))
                    if bdata.get("source_file") == spec_info["source"]:
                        energy = np.array(bdata["energy_kev"])
                        counts = np.array(bdata["counts"])
                        spectrum_loaded = True
                        break
                except Exception:
                    continue

        elif spec_info["format"] == "EMSA":
            # Load from old CSV files
            csv_candidates = list((DATA_DIR / "eds").glob("*.csv"))
            spec_digits = "".join(c for c in selected_spectrum if c.isdigit())
            for c in csv_candidates:
                stem_digits = "".join(ch for ch in c.stem.split("_")[-1] if ch.isdigit())
                if stem_digits == spec_digits:
                    df_eds = pd.read_csv(c)
                    energy_col = [col for col in df_eds.columns if "energy" in col.lower() or "ev" in col.lower()][0]
                    counts_col = [col for col in df_eds.columns if "count" in col.lower() or "intensity" in col.lower()][0]
                    energy = df_eds[energy_col].values / 1000  # eV to keV
                    counts = df_eds[counts_col].values
                    spectrum_loaded = True
                    break
            if not spectrum_loaded:
                for c in csv_candidates:
                    if spec_digits and spec_digits in c.stem:
                        df_eds = pd.read_csv(c)
                        energy_col = [col for col in df_eds.columns if "energy" in col.lower() or "ev" in col.lower()][0]
                        counts_col = [col for col in df_eds.columns if "count" in col.lower() or "intensity" in col.lower()][0]
                        energy = df_eds[energy_col].values / 1000
                        counts = df_eds[counts_col].values
                        spectrum_loaded = True
                        break

        if spectrum_loaded and energy is not None:
            mask = (energy >= eds_range[0]) & (energy <= eds_range[1])

            _eds_defaults = ["#10b981", "#fbbf24"]
            _eds_key = "colors_eds_spectrum"
            if _eds_key not in st.session_state:
                st.session_state[_eds_key] = list(_eds_defaults)
            eds_colors = st.session_state[_eds_key]

            fig_eds = go.Figure()
            fig_eds.add_trace(go.Scatter(
                x=energy[mask], y=counts[mask],
                fill="tozeroy",
                fillcolor=f"rgba({int(eds_colors[0][1:3],16)},{int(eds_colors[0][3:5],16)},{int(eds_colors[0][5:7],16)},0.2)",
                line=dict(color=eds_colors[0], width=1),
                name="EDX Spectrum",
                hovertemplate="%{x:.3f} keV<br>%{y:.0f} counts<extra></extra>",
            ))

            if show_element_lines:
                for elem, pos_kev in EDS_LINES.items():
                    if eds_range[0] <= pos_kev <= eds_range[1]:
                        fig_eds.add_vline(
                            x=pos_kev, line_dash="dot",
                            line_color=eds_colors[1],
                            annotation_text=elem,
                            annotation_font_size=9,
                            annotation_font_color=eds_colors[1],
                            annotation_position="top",
                        )

            fig_eds.update_layout(
                title=f"EDS Spectrum — {selected_spectrum} ({spec_info['format']})",
                xaxis_title="Energy (keV)",
                yaxis_title="Counts",
                height=550,
                template="plotly_dark",
            )
            if eds_log:
                fig_eds.update_yaxes(type="log")

            st.plotly_chart(fig_eds, width="stretch")
            eds_colors = color_customizer("eds_spectrum",
                ["Spectrum", "Element markers"], _eds_defaults)
        else:
            st.warning(f"Could not load full spectrum data for: {selected_spectrum}")

        # Detected peaks table
        if spec_info.get("peaks"):
            st.markdown("### Detected Peaks")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Format", spec_info["format"])
            if spec_info.get("live_time_s"):
                pc2.metric("Live Time", f"{spec_info['live_time_s']:.1f} s")
            if spec_info.get("total_counts"):
                pc3.metric("Total Counts", f"{spec_info['total_counts']:,}")
            pc4.metric("Peaks Found", len(spec_info["peaks"]))

            peaks_df = pd.DataFrame(spec_info["peaks"])
            # Format columns based on data source
            format_dict = {}
            for col in peaks_df.columns:
                if "kev" in col.lower():
                    format_dict[col] = "{:.3f}"
                elif "ev" in col.lower() and "kev" not in col.lower():
                    format_dict[col] = "{:.0f}"
                elif col == "intensity":
                    format_dict[col] = "{:.0f}"
            if "intensity" in peaks_df.columns:
                st.dataframe(
                    peaks_df.style.format(format_dict).background_gradient(subset=["intensity"], cmap="YlGn"),
                    width="stretch",
                )
                color_scale_bar("Intensity", "Low", "High", ["#ffffcc", "#addd8e", "#41ab5d", "#006837"])
            else:
                st.dataframe(peaks_df.style.format(format_dict), width="stretch")

    # Bruker EDX Quantification section
    if edx_quant_data:
        st.markdown("---")
        st.markdown("### Elemental Quantification (Bruker EDX)")
        st.caption(f"{len(edx_quant_data)} quantification entries from Bruker XLS files")

        quant_df = pd.DataFrame(edx_quant_data)
        # Clean column names for display
        display_quant = quant_df.copy()
        rename_map = {c: c.replace("norm._", "Norm ").replace("error_in_", "Error ") for c in display_quant.columns}
        display_quant = display_quant.rename(columns=rename_map)

        st.dataframe(display_quant.head(50), width="stretch")

        # Composition bar chart
        if "element" in quant_df.columns and "norm._at.%" in quant_df.columns:
            comp_fig = px.bar(
                quant_df, x="source_file", y="norm._at.%", color="element",
                title="Elemental Composition Across Samples (Atomic %)",
                template="plotly_dark",
                labels={"norm._at.%": "Atomic %", "source_file": "Sample"},
            )
            comp_fig.update_layout(height=450, barmode="stack", xaxis_tickangle=-45)
            st.plotly_chart(comp_fig, width="stretch")

    # Element tracking across all spectra
    st.markdown("---")
    st.markdown("### Element Tracking Across All Spectra")
    st.caption("Track specific element peaks across all available spectra")

    # Collect all unique elements from peaks
    all_elements = set()
    for spec in eds_spectra:
        for peak in spec.get("peaks", []):
            all_elements.add(peak.get("element_line", ""))
    all_elements = sorted(all_elements)

    if all_elements:
        track_element = st.selectbox("Track element", all_elements,
                                     index=all_elements.index("Al Kα") if "Al Kα" in all_elements
                                     else (all_elements.index("Al Ka") if "Al Ka" in all_elements else 0),
                                     key="eds_track_elem")

        track_data = []
        for spec in eds_spectra:
            matched_peak = next(
                (p for p in spec.get("peaks", []) if p.get("element_line") == track_element), None
            )
            if matched_peak:
                track_data.append({
                    "spectrum": spec["label"],
                    "format": spec["format"],
                    "intensity": matched_peak.get("intensity", 0),
                })

        if track_data:
            track_df = pd.DataFrame(track_data)
            fig_track = px.bar(
                track_df, x="spectrum", y="intensity", color="format",
                title=f"{track_element} Peak Intensity Across Spectra",
                color_discrete_map={"EMSA": "#10b981", "Bruker SPX": "#3b82f6"},
                template="plotly_dark",
            )
            fig_track.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_track, width="stretch")
        else:
            st.info(f"No spectra contain a detectable {track_element} peak.")


# ===========================================================================
# PAGE: TEM Analysis
# ===========================================================================
elif page == "TEM Analysis":
    st.markdown("## TEM / STEM Analysis")
    st.markdown(
        "Transmission Electron Microscopy metadata and spectra — "
        "auto-detected from JEOL files (voltage ≥ 100 kV) and TEM-EDS EMSA data."
    )

    reg = st.session_state["data_registry"]
    tem_records = reg.get("tem", [])
    tem_eds = reg.get("tem_eds", [])

    _img_reg_tem = reg.get("images", {})
    _has_any_tem = (tem_records or tem_eds
                    or _img_reg_tem.get("tem_raw")
                    or _img_reg_tem.get("tem_processed")
                    or _img_reg_tem.get("elemental_maps"))

    if not _has_any_tem:
        st.info("No TEM data loaded. Upload JEOL TEM .txt files or TEM-EDS .emsa files.")
    else:
        # ── TEM Image Metadata ──
        if tem_records:
            st.markdown("### TEM Image Metadata")
            st.caption(f"**{len(tem_records)}** TEM/STEM records detected (accelerating voltage ≥ 100 kV)")
            df_tem = pd.DataFrame(tem_records)
            display_cols = [c for c in [
                "source_file", "sample_name", "accelerating_voltage_kv",
                "magnification", "signal_name", "date", "has_image"
            ] if c in df_tem.columns]
            if display_cols:
                st.dataframe(
                    df_tem[display_cols].rename(columns={
                        "source_file": "File",
                        "sample_name": "Sample",
                        "accelerating_voltage_kv": "Voltage (kV)",
                        "magnification": "Magnification",
                        "signal_name": "Signal",
                        "date": "Date",
                        "has_image": "Has Image",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(df_tem, use_container_width=True, hide_index=True)

            # Summary metrics
            mc1, mc2, mc3 = st.columns(3)
            voltages = df_tem.get("accelerating_voltage_kv", pd.Series())
            mags = df_tem.get("magnification", pd.Series())
            signals = df_tem.get("signal_name", pd.Series(dtype=str))
            if not voltages.empty:
                mc1.metric("Voltage", f"{voltages.iloc[0]:.0f} kV")
            if not mags.empty and mags.max() > 0:
                mc2.metric("Mag Range", f"{mags.min():.0f}× – {mags.max():.0f}×")
            if not signals.empty:
                unique_sig = signals.dropna().unique()
                mc3.metric("Signals", ", ".join(str(s) for s in unique_sig[:4]))

            # ── Image Gallery (from agent-classified images) ──
            images_reg = reg.get("images", {})
            tem_raw = images_reg.get("tem_raw", [])
            tem_processed = images_reg.get("tem_processed", [])
            elemental_maps = images_reg.get("elemental_maps", [])

            # Also include metadata-linked images (legacy)
            meta_images = [r for r in tem_records
                           if r.get("image_path") and Path(r["image_path"]).exists()]

            has_any_images = tem_raw or tem_processed or meta_images

            if has_any_images:
                st.markdown("### TEM Image Gallery")

                # Group images by sample
                _sample_filter_options = sorted(set(
                    img.get("sample_name", "Unknown") for img in (tem_raw + tem_processed)
                    if img.get("sample_name")
                ))
                if len(_sample_filter_options) > 1:
                    _sample_filter_options.insert(0, "All Samples")
                    _sel_sample = st.selectbox("Filter by sample:", _sample_filter_options, key="tem_img_sample")
                else:
                    _sel_sample = "All Samples"

                # Tabs for different image types
                _img_tabs = []
                if tem_raw:
                    _img_tabs.append("Raw Images")
                if tem_processed:
                    _img_tabs.append("Processed (FFT/SAED)")
                if not _img_tabs:
                    _img_tabs = ["Images"]

                img_tab_widgets = st.tabs(_img_tabs)

                _IMGS_PER_PAGE = 12

                def _show_image_grid(img_list, tab_widget, grid_key="raw"):
                    with tab_widget:
                        filtered = img_list
                        if _sel_sample != "All Samples":
                            filtered = [im for im in img_list if im.get("sample_name") == _sel_sample]
                        if not filtered:
                            st.info("No images for this sample.")
                            return

                        total = len(filtered)
                        n_pages = max(1, (total + _IMGS_PER_PAGE - 1) // _IMGS_PER_PAGE)

                        # ── Pagination: ◀ Previous | Page X | Next ▶ ──
                        _page_key = f"tem_page_{grid_key}"
                        if _page_key not in st.session_state:
                            st.session_state[_page_key] = 1
                        # Clamp
                        st.session_state[_page_key] = max(1, min(n_pages, st.session_state[_page_key]))
                        _cur = st.session_state[_page_key]

                        if n_pages > 1:
                            prev_col, num_col, next_col, info_col = st.columns([1, 1, 1, 2])
                            with prev_col:
                                st.button(
                                    "◀ Previous", key=f"prev_{grid_key}",
                                    disabled=_cur <= 1,
                                    on_click=lambda k=_page_key: st.session_state.update({k: st.session_state[k] - 1}),
                                )
                            with num_col:
                                def _on_page_change(k=_page_key, wk=f"pginp_{grid_key}"):
                                    st.session_state[k] = int(st.session_state[wk])
                                st.number_input(
                                    "Page", min_value=1, max_value=n_pages,
                                    value=_cur,
                                    key=f"pginp_{grid_key}",
                                    label_visibility="collapsed",
                                    on_change=_on_page_change,
                                )
                            with next_col:
                                st.button(
                                    "Next ▶", key=f"next_{grid_key}",
                                    disabled=_cur >= n_pages,
                                    on_click=lambda k=_page_key: st.session_state.update({k: st.session_state[k] + 1}),
                                )
                            with info_col:
                                st.markdown(
                                    f"<div style='padding-top:8px;color:#94a3b8;font-size:0.85rem;'>"
                                    f"Page {_cur} of {n_pages} &nbsp;·&nbsp; "
                                    f"{total} images</div>",
                                    unsafe_allow_html=True,
                                )

                        page_num = st.session_state[_page_key]
                        start = (page_num - 1) * _IMGS_PER_PAGE
                        page_items = filtered[start:start + _IMGS_PER_PAGE]

                        st.caption(f"Showing {start+1}–{start+len(page_items)} of {total}")
                        n_cols = min(4, len(page_items))
                        cols = st.columns(n_cols)
                        for idx, im in enumerate(page_items):
                            with cols[idx % n_cols]:
                                try:
                                    pil_img = Image.open(im["path"])
                                    caption = im.get("sub_technique") or im.get("format_name", "")
                                    if im.get("sample_name"):
                                        caption = f"{im['sample_name']} | {caption}"
                                    st.image(pil_img, caption=caption, use_container_width=True)
                                except Exception:
                                    st.warning(f"Cannot display {im.get('filename', '')}")

                tab_idx = 0
                if tem_raw:
                    _show_image_grid(tem_raw, img_tab_widgets[tab_idx], "raw")
                    tab_idx += 1
                if tem_processed:
                    _show_image_grid(tem_processed, img_tab_widgets[tab_idx], "processed")

            # ── EDS Elemental Maps ──
            if elemental_maps:
                st.markdown("### EDS Elemental Maps")
                # Group by session (View number)
                _map_sessions = {}
                for em in elemental_maps:
                    sid = em.get("session_id", "unknown")
                    _map_sessions.setdefault(sid, []).append(em)

                for session_id, maps in sorted(_map_sessions.items()):
                    sample = maps[0].get("sample_name", "")
                    elements = [m.get("element", "?") for m in maps]
                    st.markdown(f"**{sample} — {session_id}** ({', '.join(elements)})")
                    n_cols = min(len(maps), 6)
                    cols = st.columns(n_cols)
                    for idx, em in enumerate(maps):
                        with cols[idx % n_cols]:
                            try:
                                pil_img = Image.open(em["path"])
                                st.image(pil_img, caption=em.get("element", em["filename"]),
                                         use_container_width=True)
                            except Exception:
                                st.warning(f"Cannot display {em.get('filename', '')}")

        st.markdown("---")

        # ── TEM-EDS Spectra ──
        if tem_eds:
            st.markdown("### TEM-EDS Spectra")
            st.caption(
                f"**{len(tem_eds)}** EDS spectra collected at beam voltages ≥ 100 kV "
                f"(TEM/STEM mode)"
            )

            for i, spec in enumerate(tem_eds):
                meta = spec.get("metadata", {})
                label = meta.get("title", spec.get("source_file", f"Spectrum {i+1}"))
                beam_kv = meta.get("beam_kv", spec.get("beam_kv", "?"))
                live_t = meta.get("live_time_s", spec.get("live_time_s", "?"))

                with st.expander(f"{Path(str(label)).name if '/' in str(label) or '\\\\' in str(label) else label} — {beam_kv} kV, {live_t}s"):
                    # Plot spectrum if energy/intensity arrays present
                    energies = spec.get("energy_ev", [])
                    intensities = spec.get("intensity", [])
                    if energies and intensities:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=energies, y=intensities,
                            mode="lines", name="TEM-EDS",
                            line=dict(color="#06b6d4", width=1.2),
                        ))
                        fig.update_layout(
                            xaxis_title="Energy (eV)",
                            yaxis_title="Intensity (counts)",
                            height=350,
                            margin=dict(l=40, r=20, t=30, b=40),
                            template="plotly_white",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show peaks if available
                    peaks = spec.get("peaks", [])
                    if peaks:
                        st.markdown("**Identified Peaks:**")
                        pdf = pd.DataFrame(peaks)
                        st.dataframe(pdf, use_container_width=True, hide_index=True)

                    # Metadata summary
                    meta_display = {k: v for k, v in meta.items()
                                    if v and k not in ("data_type",)}
                    if meta_display:
                        st.json(meta_display)


# ===========================================================================
# PAGE: Cross-Technique ML
# ===========================================================================
elif page == "Cross-Technique ML":
    st.markdown("## Cross-Technique Materials Informatics")
    st.markdown(
        '<p>'
        "Unified analysis across XRD, SEM, and EDX characterization techniques. "
        "Explore feature correlations, material family distributions, and cross-technique insights."
        "</p>",
        unsafe_allow_html=True,
    )

    # -- Data Summary (from registry) ------------------------------------------
    st.markdown("### Data Summary")
    _ml_reg = st.session_state["data_registry"]
    _ml_counts = {
        "XRD": len(_ml_reg["xrd"]),
        "XPS": len(_ml_reg["xps"]),
        "SEM": len(_ml_reg["sem"]),
        "EDS": len(_ml_reg["eds"]["spectra"]),
        "TEM": len(_ml_reg.get("tem", [])),
        "TEM-EDS": len(_ml_reg.get("tem_eds", [])),
    }
    _ml_active = {k: v for k, v in _ml_counts.items() if v > 0}
    _ml_cols = st.columns(len(_ml_active) + 1)
    for col, (tech, cnt) in zip(_ml_cols, _ml_active.items()):
        col.metric(f"{tech}", cnt)
    _ml_cols[-1].metric("Techniques", len(_ml_active))

    st.markdown("---")

    # -- Material Family Distribution -----------------------------------------
    st.markdown("### Material Family Distribution")
    family_csv_path = DATA_DIR / "features" / "family_feature_matrix.csv"
    if family_csv_path.exists():
        family_df = load_csv(str(family_csv_path))
        # Build a long-form dataframe for grouped bar chart
        tech_cols = []
        col_map = {"xrd_sample_count": "XRD", "edx_sample_count": "EDX", "sem_image_count": "SEM"}
        for col_name in col_map:
            if col_name in family_df.columns:
                tech_cols.append(col_name)
        family_col = "family" if "family" in family_df.columns else "material_family"
        if family_col in family_df.columns and tech_cols:
            melt_df = family_df.melt(
                id_vars=family_col,
                value_vars=tech_cols,
                var_name="Technique",
                value_name="Count",
            )
            melt_df["Technique"] = melt_df["Technique"].map(col_map).fillna(melt_df["Technique"])
            melt_df = melt_df.dropna(subset=["Count"])
            fig_family = px.bar(
                melt_df, x=family_col, y="Count", color="Technique",
                barmode="group", template="plotly_dark",
                labels={family_col: "Material Family", "Count": "Sample Count"},
                color_discrete_map={"XRD": "#3b82f6", "EDX": "#22c55e", "SEM": "#f59e0b"},
            )
            fig_family.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
            st.plotly_chart(fig_family, width="stretch")
        else:
            st.info("Family feature matrix does not have expected columns.")
    else:
        st.warning("Family feature matrix not found.")

    st.markdown("---")

    # -- Cross-Technique Correlation Heatmap ----------------------------------
    st.markdown("### Cross-Technique Correlation Heatmap")
    corr_csv_path = DATA_DIR / "features" / "cross_technique_correlation.csv"
    if corr_csv_path.exists():
        corr_df = load_csv(str(corr_csv_path))
        # Use first column as index if it contains feature names
        if corr_df.columns[0] in ("Unnamed: 0", "feature", "index"):
            corr_df = corr_df.set_index(corr_df.columns[0])

        # Clean column and index names
        def _clean_feature_name(name):
            return (
                str(name)
                .replace("xrd_", "XRD: ")
                .replace("edx_", "EDX: ")
                .replace("sem_", "SEM: ")
            )

        corr_df.columns = [_clean_feature_name(c) for c in corr_df.columns]
        corr_df.index = [_clean_feature_name(c) for c in corr_df.index]

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns.tolist(),
            y=corr_df.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
        ))
        fig_corr.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=140, b=140),
            xaxis_tickangle=-45,
        )

        heatmap_colors = color_customizer(
            "corr_heatmap",
            ["Low (-1)", "Mid (0)", "High (+1)"],
            ["#2166ac", "#f7f7f7", "#b2182b"],
        )
        st.plotly_chart(fig_corr, width="stretch")
    else:
        st.warning("Cross-technique correlation matrix not found.")

    st.markdown("---")

    # -- Top Cross-Technique Correlations Table --------------------------------
    st.markdown("### Top Cross-Technique Correlations")
    results_json_path = DATA_DIR / "features" / "cross_technique_results.json"
    if results_json_path.exists():
        ct_results = load_json(str(results_json_path))
        top_corrs = ct_results.get("top_cross_correlations", ct_results.get("top_correlations", []))
        if isinstance(top_corrs, list) and len(top_corrs) > 0:
            top_df = pd.DataFrame(top_corrs[:15])
            if "correlation" in top_df.columns:
                display_cols = ["feature_1", "feature_2", "correlation"]
                display_cols = [c for c in display_cols if c in top_df.columns]
                styled = top_df[display_cols].style.background_gradient(
                    subset=["correlation"], cmap="RdBu_r", vmin=-1, vmax=1,
                ).format({"correlation": "{:.3f}"})
                st.dataframe(styled, width="stretch")
            else:
                st.dataframe(top_df, width="stretch")
        else:
            st.info("No cross-technique correlations found in results.")
    else:
        st.warning("Cross-technique results file not found.")

    st.markdown("---")

    # -- Feature Space Visualization (PCA + Parallel Coordinates) -------------
    st.markdown("### Feature Space Visualization")
    st.caption("Automatic dimensionality reduction — no manual axis selection needed")
    feat_csv_path = DATA_DIR / "features" / "feature_matrix.csv"
    if feat_csv_path.exists():
        feat_df = load_csv(str(feat_csv_path))
        numeric_cols = feat_df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) >= 2:
            # Assign technique label
            tech_col = "technique" if "technique" in feat_df.columns else None
            if tech_col is None:
                feat_df["technique"] = "Unknown"
                tech_col = "technique"

            # Assign material family using sample_matcher
            try:
                from src.ml.sample_matcher import classify_family
                feat_df["family"] = feat_df["sample_name"].apply(classify_family)
            except Exception:
                feat_df["family"] = "Unknown"

            # Let user choose color grouping
            pca_tab, parcoord_tab, dist_tab = st.tabs([
                "PCA Clustering", "Parallel Coordinates", "Feature Distributions"
            ])

            # ── Tab 1: PCA per technique ──
            with pca_tab:
                color_by = st.radio(
                    "Color by", ["Technique", "Material Family"],
                    horizontal=True, key="pca_color"
                )
                color_col = tech_col if color_by == "Technique" else "family"

                # Run PCA per technique group (each technique has different feature columns)
                tech_groups = {"XRD": "xrd_", "EDX": "edx_", "SEM": "sem_"}
                pca_col1, pca_col2, pca_col3 = st.columns(3)
                pca_containers = {"XRD": pca_col1, "EDX": pca_col2, "SEM": pca_col3}

                for tech_name, prefix in tech_groups.items():
                    with pca_containers[tech_name]:
                        tech_rows = feat_df[feat_df[tech_col] == tech_name].copy()
                        tech_features = [c for c in numeric_cols if c.startswith(prefix)]

                        if len(tech_rows) < 3 or len(tech_features) < 2:
                            st.info(f"**{tech_name}**: Not enough data for PCA "
                                    f"({len(tech_rows)} samples, {len(tech_features)} features)")
                            continue

                        # Drop columns that are all NaN and fill remaining NaN with column mean
                        sub = tech_rows[tech_features].copy()
                        sub = sub.dropna(axis=1, how="all")
                        if sub.shape[1] < 2:
                            st.info(f"**{tech_name}**: Not enough non-null features")
                            continue
                        sub = sub.fillna(sub.mean())

                        # Standardize
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(sub.values)

                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        var1 = pca.explained_variance_ratio_[0] * 100
                        var2 = pca.explained_variance_ratio_[1] * 100

                        pca_plot_df = pd.DataFrame({
                            "PC1": X_pca[:, 0],
                            "PC2": X_pca[:, 1],
                            "sample": tech_rows["sample_name"].values,
                            "group": tech_rows[color_col].values,
                        })

                        fig_pca = px.scatter(
                            pca_plot_df, x="PC1", y="PC2", color="group",
                            hover_data=["sample"],
                            template="plotly_dark",
                            labels={
                                "PC1": f"PC1 ({var1:.1f}%)",
                                "PC2": f"PC2 ({var2:.1f}%)",
                                "group": color_by,
                            },
                        )
                        fig_pca.update_traces(marker=dict(size=7, opacity=0.8,
                                                          line=dict(width=0.5, color="white")))
                        fig_pca.update_layout(
                            title=dict(text=f"{tech_name} Feature Space",
                                       font=dict(size=14)),
                            height=400,
                            legend=dict(font=dict(size=9), itemsizing="constant"),
                            margin=dict(t=40, b=30),
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                        st.caption(f"{len(tech_rows)} samples | {sub.shape[1]} features | "
                                   f"Total variance explained: {var1 + var2:.1f}%")

            # ── Tab 2: Parallel Coordinates ──
            with parcoord_tab:
                _avail_techs = sorted(tech_groups.keys())
                parcoord_tech = st.selectbox(
                    "Technique", _avail_techs, key="parcoord_tech"
                )
                prefix = tech_groups[parcoord_tech]
                tech_rows = feat_df[feat_df[tech_col] == parcoord_tech].copy()
                tech_features = [c for c in numeric_cols if c.startswith(prefix)]

                if len(tech_rows) < 2 or len(tech_features) < 2:
                    st.info(f"Not enough {parcoord_tech} data for parallel coordinates.")
                else:
                    # Pick top features by variance (most informative)
                    sub = tech_rows[tech_features].copy()
                    sub = sub.dropna(axis=1, how="all").fillna(0)
                    variances = sub.var().sort_values(ascending=False)
                    top_feats = variances.head(min(8, len(variances))).index.tolist()

                    # Normalize to 0-1 for comparable axes
                    sub_norm = sub[top_feats].copy()
                    for col in top_feats:
                        cmin, cmax = sub_norm[col].min(), sub_norm[col].max()
                        if cmax > cmin:
                            sub_norm[col] = (sub_norm[col] - cmin) / (cmax - cmin)
                        else:
                            sub_norm[col] = 0.5

                    # Map families to numeric for colorscale
                    families = tech_rows["family"].values
                    unique_fam = sorted(set(families))
                    fam_to_num = {f: i for i, f in enumerate(unique_fam)}
                    fam_nums = [fam_to_num[f] for f in families]

                    # Clean labels
                    clean_labels = {c: c.replace(prefix, "").replace("_", " ").title() for c in top_feats}

                    dims = []
                    for col in top_feats:
                        dims.append(dict(
                            range=[0, 1],
                            label=clean_labels[col],
                            values=sub_norm[col].values,
                        ))

                    fig_parcoord = go.Figure(data=go.Parcoords(
                        line=dict(
                            color=fam_nums,
                            colorscale="Turbo",
                            showscale=True,
                            colorbar=dict(
                                title="Family",
                                tickvals=list(range(len(unique_fam))),
                                ticktext=[f.replace("_", " ") for f in unique_fam],
                                len=0.8,
                            ),
                        ),
                        dimensions=dims,
                    ))
                    fig_parcoord.update_layout(
                        template="plotly_dark",
                        height=450,
                        title=f"{parcoord_tech} Feature Profiles Across Material Families",
                        margin=dict(l=80, r=80, t=50, b=30),
                    )
                    st.plotly_chart(fig_parcoord, use_container_width=True)
                    st.caption(f"Top {len(top_feats)} features by variance | "
                               f"{len(tech_rows)} samples | Normalized to [0, 1]")

            # ── Tab 3: Feature Distributions (6 separate charts in 3×2 grid) ──
            with dist_tab:
                dist_tech = st.selectbox(
                    "Technique", _avail_techs, key="dist_tech"
                )
                prefix = tech_groups[dist_tech]
                tech_rows = feat_df[feat_df[tech_col] == dist_tech].copy()
                tech_features = [c for c in numeric_cols if c.startswith(prefix)]

                if len(tech_rows) < 2 or len(tech_features) < 1:
                    st.info(f"Not enough {dist_tech} data for distribution plots.")
                else:
                    # Pick top features by variance
                    sub = tech_rows[tech_features + ["family", "sample_name"]].copy()
                    sub = sub.dropna(axis=1, how="all")
                    valid_feats = [c for c in sub.columns if c.startswith(prefix)]
                    variances = sub[valid_feats].var().sort_values(ascending=False)
                    top_feats = variances.head(min(6, len(variances))).index.tolist()

                    # Auto-generate colors for whatever families exist
                    sorted_families = sorted(sub["family"].unique())
                    _auto_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
                    family_colors = {fam: _auto_palette[i % len(_auto_palette)]
                                     for i, fam in enumerate(sorted_families)}

                    st.markdown(f"**{dist_tech} Feature Distributions by Material Family**")
                    st.caption(f"Top {len(top_feats)} features by variance | "
                               f"{len(tech_rows)} samples across {len(sorted_families)} families")

                    # Render each feature as its own chart in a 3-column grid
                    for row_start in range(0, len(top_feats), 3):
                        row_feats = top_feats[row_start:row_start + 3]
                        cols = st.columns(len(row_feats))
                        for col_container, feat in zip(cols, row_feats):
                            with col_container:
                                clean_name = feat.replace(prefix, "").replace("_", " ").title()
                                fig_box = go.Figure()
                                for fam in sorted_families:
                                    fam_data = sub[sub["family"] == fam][feat].dropna()
                                    if len(fam_data) == 0:
                                        continue
                                    color = family_colors.get(fam, "#888888")
                                    fig_box.add_trace(go.Box(
                                        y=fam_data.values,
                                        name=fam.replace("_", " "),
                                        marker_color=color,
                                        boxpoints="all",
                                        jitter=0.3,
                                        pointpos=-1.5,
                                        line=dict(width=1.5),
                                    ))
                                fig_box.update_layout(
                                    template="plotly_dark",
                                    title=dict(text=clean_name, font=dict(size=13)),
                                    height=350,
                                    showlegend=False,
                                    margin=dict(t=35, b=60, l=50, r=15),
                                    xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                                    yaxis=dict(title=None, tickfont=dict(size=9)),
                                )
                                st.plotly_chart(fig_box, use_container_width=True)

                    # Shared legend below the grid
                    legend_items = []
                    for fam in sorted_families:
                        color = family_colors.get(fam, "#888888")
                        label = fam.replace("_", " ")
                        legend_items.append(
                            f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
                            f'<span style="width:12px;height:12px;border-radius:50%;'
                            f'background:{color};display:inline-block;margin-right:5px;"></span>'
                            f'<span style="font-size:12px;color:#cbd5e1;">{label}</span></span>'
                        )
                    st.markdown(
                        f'<div style="text-align:center;padding:8px 0;">{"".join(legend_items)}</div>',
                        unsafe_allow_html=True,
                    )

        else:
            st.info("Feature matrix has fewer than 2 numeric columns.")
    else:
        st.warning("Feature matrix not found. Run cross-technique analysis first.")


# ===========================================================================
# PAGE: Data Export
# ===========================================================================
elif page == "Data Export":
    st.markdown("## Data Export")
    st.markdown("Download processed data in various formats for further analysis.")

    st.markdown("### Available Datasets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### XRD Data")
        for sample in ["Ti2ALC3", "Ti2C3"]:
            csv_path = DATA_DIR / "xrd" / f"xrd_{sample}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    f"Download XRD - {sample} (CSV)",
                    csv_data, f"xrd_{sample}.csv", "text/csv",
                )

        st.markdown("#### XPS Data")
        for el in ["survey", "Ti_2p", "C_1s", "O_1s", "F_1s"]:
            csv_path = DATA_DIR / "xps" / f"xps_{el}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    f"Download XPS - {el} (CSV)",
                    csv_data, f"xps_{el}.csv", "text/csv",
                )

    with col2:
        st.markdown("#### XPS Quantification")
        quant = load_json(str(DATA_DIR / "xps" / "xps_quantification.json"))
        st.download_button(
            "Download Quantification (JSON)",
            json.dumps(quant, indent=2), "xps_quantification.json", "application/json",
        )

        st.markdown("#### SEM Catalog")
        sem = load_json(str(DATA_DIR / "sem" / "sem_catalog.json"))
        st.download_button(
            "Download SEM Catalog (JSON)",
            json.dumps(sem, indent=2), "sem_catalog.json", "application/json",
        )

        st.markdown("#### EDS Peaks Summary")
        eds = load_json(str(DATA_DIR / "eds" / "eds_peaks_summary.json"))
        st.download_button(
            "Download EDS Peaks (JSON)",
            json.dumps(eds, indent=2), "eds_peaks_summary.json", "application/json",
        )

    st.markdown("---")
    st.markdown("### Regenerate Data")
    st.info("Run the ETL pipeline to regenerate all processed data from raw files: `python run_etl.py`")


# ===========================================================================
# SIDEBAR FOOTER (always at the bottom, across all pages)
# ===========================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Materials Informatics v1.0<br>"
    "Gudibandi Sri Nikhil Reddy<br>"
    "Ikeda - Hamasaki Laboratory<br>"
    "Research Institute of Electronics<br>"
    "Shizuoka University, Japan</small>",
    unsafe_allow_html=True,
)
