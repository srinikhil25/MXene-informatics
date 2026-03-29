# -*- coding: utf-8 -*-
"""
Materials Informatics Interactive Dashboard
========================================
Interactive analysis platform for Ti₃AlC₂ → Ti₃C₂Tₓ MXene characterization data.
Supports XRD, XPS, SEM, and EDS analysis with user-adjustable parameters.

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"

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
    # Try universal ETL path first, fall back to old path
    universal_path = DATA_DIR / "universal" / "xrd" / f"xrd_{sample}.json"
    old_path = DATA_DIR / "xrd" / f"xrd_{sample}.json"
    path = universal_path if universal_path.exists() else old_path
    j = load_json(str(path))
    return np.array(j["two_theta"]), np.array(j["intensity"]), j.get("metadata", {})


@st.cache_data
def get_xrd_samples():
    """Get list of all available XRD sample names from universal ETL."""
    universal_dir = DATA_DIR / "universal" / "xrd"
    if universal_dir.exists():
        samples = sorted([
            f.stem.replace("xrd_", "")
            for f in universal_dir.glob("xrd_*.json")
        ])
        return samples
    # Fallback to old directory
    old_dir = DATA_DIR / "xrd"
    if old_dir.exists():
        return sorted([f.stem.replace("xrd_", "") for f in old_dir.glob("xrd_*.json")])
    return []


def load_xps_hr(element_key):
    j = load_json(str(DATA_DIR / "xps" / f"xps_{element_key}.json"))
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
# Ti3AlC2 reference peaks (JCPDS 52-0875)
# ---------------------------------------------------------------------------
MAX_PEAKS = {
    "Ti3AlC2": [
        (9.5, "(002)"), (19.1, "(004)"), (33.9, "(100)"),
        (34.1, "(101)"), (36.8, "(102)"), (38.9, "(103)"),
        (39.0, "(104)"), (41.8, "(006)"), (48.5, "(105)"),
        (52.4, "(106)"), (56.5, "(110)"), (60.3, "(108)"),
        (65.6, "(112)"), (70.4, "(114)"), (74.0, "(200)"),
    ],
}

MXENE_PEAKS = {
    "Ti3C2Tx": [
        (6.6, "(002)"), (9.0, "(002)*"), (18.3, "(004)"),
        (27.5, "(006)"), (34.0, "(100)"), (36.8, "(008)"),
        (41.8, "(101)"), (60.5, "(110)"),
    ],
}

# XPS reference binding energies for Ti₃C₂Tₓ
XPS_REFS = {
    "Ti 2p": [
        (455.0, "Ti-C (2p₃/₂)"), (455.8, "Ti²⁺ (2p₃/₂)"),
        (457.0, "Ti³⁺ (2p₃/₂)"), (458.8, "TiO₂ (2p₃/₂)"),
        (461.0, "Ti-C (2p₁/₂)"), (464.0, "TiO₂ (2p₁/₂)"),
    ],
    "C 1s": [
        (282.0, "Ti-C-Tₓ"), (284.8, "C-C/C=C"),
        (286.4, "C-O"), (288.8, "O-C=O"),
    ],
    "O 1s": [
        (529.8, "TiO₂"), (531.2, "Ti-OH/C=O"),
        (532.5, "C-O/H₂O"), (533.5, "adsorbed H₂O"),
    ],
    "F 1s": [
        (685.0, "Ti-F"), (686.5, "Al-F"),
        (688.5, "C-F"),
    ],
}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Materials Informatics")
page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "XRD Analysis",
        "XPS Analysis",
        "SEM Gallery",
        "EDS Analysis",
        "Cross-Technique ML",
        # "Data Export",  # Hidden until senior's paper is published
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Summary")
st.sidebar.markdown("**Families:** MXene, CF, CAF, BFO, ...")
st.sidebar.markdown("**XRD:** Rigaku Ultima3")
st.sidebar.markdown("**SEM:** JEOL FE-SEM + Hitachi HR-FE-SEM")
st.sidebar.markdown("**EDX:** Bruker Quantax")
st.sidebar.markdown("**XPS:** PHI (MXene only)")


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

    # Key metrics — pull from universal ETL stats if available
    etl_stats_overview = {}
    _etl_stats_path = DATA_DIR / "universal" / "universal_etl_stats.json"
    if _etl_stats_path.exists():
        etl_stats_overview = load_json(str(_etl_stats_path))

    n_xrd = etl_stats_overview.get("xrd_datasets", len(get_xrd_samples()) if 'get_xrd_samples' in dir() else 0)
    n_sem = etl_stats_overview.get("sem_images_total",
                                    etl_stats_overview.get("jeol_sem_images", 0) + etl_stats_overview.get("hitachi_sem_images", 0))
    n_edx = etl_stats_overview.get("edx_spectra", 0)

    xps_quant = {}
    _xps_path = DATA_DIR / "xps" / "xps_quantification.json"
    if _xps_path.exists():
        xps_quant = load_json(str(_xps_path))
    n_xps = len(xps_quant.get("elements", []))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("XRD Patterns", n_xrd)
    c2.metric("SEM Images", n_sem)
    c3.metric("EDX Spectra", n_edx)
    c4.metric("XPS Elements", n_xps)
    c5.metric("Total Samples", etl_stats_overview.get("total_unique_samples", "185"))

    st.markdown("---")

    # Pipeline architecture
    st.markdown("### Pipeline Architecture")
    layers = [
        ("Layer 1", "Data Engineering", "Universal ETL", "#06b6d4",
         "957 raw files (XRD/SEM/EDX/XPS) → Standardized JSON/CSV"),
        ("Layer 2", "Scientific Analysis", "Peak Fitting & Morphology", "#8b5cf6",
         "Rietveld refinement, XPS deconvolution, SEM segmentation"),
        ("Layer 3", "Cross-Technique ML", "Feature Correlation", "#ec4899",
         "33 features × 185 samples, PCA clustering, family comparison"),
        ("Layer 4 (Planned)", "Agentic Interface", "RAG + LLM", "#f59e0b",
         "Ask questions about your data and literature"),
    ]
    layer_html = '<div style="display:flex;gap:12px;padding:10px 0;">'
    for lid, title, subtitle, color, desc in layers:
        layer_html += f'''
        <div style="flex:1;background:linear-gradient(135deg,{color}22,{color}11);
            border:1px solid {color}44;border-radius:12px;padding:16px;text-align:center;">
            <div style="background:{color};color:white;border-radius:8px;padding:4px 12px;
                display:inline-block;font-weight:700;font-size:0.8rem;margin-bottom:8px;">{lid}</div>
            <div style="color:#e2e8f0;font-weight:700;font-size:0.9rem;">{title}</div>
            <div style="color:{color};font-size:0.75rem;font-weight:600;margin:4px 0;">{subtitle}</div>
            <div style="color:#64748b;font-size:0.7rem;">{desc}</div>
        </div>'''
    layer_html += '</div>'
    st.markdown(layer_html, unsafe_allow_html=True)

    st.markdown("---")

    # XPS Composition at a glance (only if XPS data available)
    if xps_quant.get("elements"):
        st.markdown("### Surface Composition — MXene XPS")
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

    # Key findings
    st.markdown("### Key Findings")
    findings = [
        f"Universal ETL processed {n_xrd} XRD patterns, {n_sem} SEM images, {n_edx} EDX spectra across 6+ material families",
        "Cross-technique feature extraction: 33 features per sample enabling PCA clustering and family-level correlation",
        "Multi-vendor support: Rigaku XRD, JEOL FE-SEM, Hitachi HR-FE-SEM, Bruker EDX — all parsed autonomously",
        "Rietveld whole-pattern refinement with crystal structure models (Ti₃AlC₂, Ti₃C₂Tₓ, TiO₂, TiC, Al₂O₃)",
        "XPS deconvolution with spin-orbit coupling (Ti 2p₃/₂ + 2p₁/₂) and DOI-referenced chemical state assignments",
    ]
    for f in findings:
        st.markdown(f'<div class="finding-box">{f}</div>', unsafe_allow_html=True)


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

        # Reference peaks toggle
        show_ref_max = st.sidebar.checkbox("Show Ti₃AlC₂ reference peaks", value=False, key="xrd_ref_max")
        show_ref_mx = st.sidebar.checkbox("Show Ti₃C₂Tₓ reference peaks", value=False, key="xrd_ref_mx")

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

            # Reference peaks
            if show_ref_max:
                for pos, label in MAX_PEAKS["Ti3AlC2"]:
                    if range_min <= pos <= range_max:
                        fig_xrd.add_vline(
                            x=pos, line_dash="dot", line_color="#60a5fa",
                            annotation_text=label, annotation_position="top",
                            annotation_font_size=9, annotation_font_color="#60a5fa",
                        )

            if show_ref_mx:
                for pos, label in MXENE_PEAKS["Ti3C2Tx"]:
                    if range_min <= pos <= range_max:
                        fig_xrd.add_vline(
                            x=pos, line_dash="dot", line_color="#f87171",
                            annotation_text=label, annotation_position="bottom",
                            annotation_font_size=9, annotation_font_color="#f87171",
                        )

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

            # Key observations - dynamic based on selection
            st.markdown("### Pattern Details")
            detail_cols = st.columns(min(len(selected_samples), 3))
            for i, sample_name in enumerate(selected_samples[:3]):
                with detail_cols[i]:
                    try:
                        tt, intensity, meta = load_xrd(sample_name)
                        st.markdown(f"**{sample_name}**")
                        st.markdown(f"- Family: {sample_families.get(sample_name, 'Other').replace('_', ' ')}")
                        st.markdown(f"- Instrument: {meta.get('instrument', 'N/A')}")
                        st.markdown(f"- Range: {tt.min():.1f}° – {tt.max():.1f}°")
                        st.markdown(f"- Points: {len(tt):,}")
                        st.markdown(f"- Max intensity: {intensity.max():.0f}")
                    except Exception:
                        st.info(f"No metadata for {sample_name}")

            with st.expander("d-Spacing Calculator"):
                st.markdown("**Bragg's Law:** nλ = 2d·sin(θ)")
                calc_angle = st.number_input("Enter 2θ (°):", value=9.5, step=0.1, key="xrd_dspacing")
                wavelength = 1.54056  # Cu Ka1
                d_spacing = wavelength / (2 * np.sin(np.radians(calc_angle / 2)))
                st.markdown(f"**d-spacing = {d_spacing:.4f} Å** (λ = {wavelength} Å, Cu Kα₁)")

            # Instrument metadata
            with st.expander("Instrument Metadata"):
                if selected_samples:
                    try:
                        _, _, meta_show = load_xrd(selected_samples[0])
                        meta_rows = []
                        for k, v in meta_show.items():
                            label = k.replace("_", " ").title()
                            meta_rows.append({"Parameter": label, "Value": str(v)})
                        if meta_rows:
                            st.table(pd.DataFrame(meta_rows))
                    except Exception:
                        st.info("No metadata available.")

    # --- RIETVELD REFINEMENT ---
    st.markdown("---")
    st.markdown("## Rietveld Whole-Pattern Refinement")
    st.markdown(
        "Whole-pattern fitting using crystal structure models. Refines lattice "
        "parameters, phase fractions, profile shape, and preferred orientation."
    )

    from src.analysis.rietveld import (
        rietveld_refine, bragg_peak_table, atom_site_table,
        structure_summary, CRYSTAL_PHASES,
    )

    # Phase selection
    riet_col1, riet_col2 = st.columns(2)
    with riet_col1:
        riet_samples_avail = get_xrd_samples()
        # Put known MXene samples at top if available
        riet_preferred = []
        for p in ["Ti2ALC3", "Ti2C3"]:
            if p in riet_samples_avail:
                riet_preferred.append(p)
        riet_samples_ordered = riet_preferred + [s for s in riet_samples_avail if s not in riet_preferred]
        riet_sample = st.selectbox(
            "Refine sample:",
            riet_samples_ordered,
            index=0,
            key="riet_sample",
        )
    with riet_col2:
        available_phases = list(CRYSTAL_PHASES.keys())
        # Default phases based on family
        riet_family = _classify_family(riet_sample) if riet_sample else ""
        if "MAX" in riet_family or "Ti3AlC2" in riet_sample:
            default_phases = ["Ti3AlC2", "TiC"]
        elif "MXene" in riet_family or "Ti3C2" in riet_sample or "Ti2C3" in riet_sample:
            default_phases = ["Ti3C2Tx", "TiO2_Anatase"]
        else:
            default_phases = available_phases[:2] if len(available_phases) >= 2 else available_phases
        selected_phases = st.multiselect(
            "Phases to refine:",
            available_phases,
            default=default_phases,
            key="riet_phases",
        )

    # Refinement controls
    riet_c1, riet_c2, riet_c3, riet_c4 = st.columns(4)
    n_bg = riet_c1.number_input("Background terms", 3, 12, 6, key="riet_bg")
    max_iter = riet_c2.number_input("Max iterations", 50, 500, 200, step=50, key="riet_iter")
    refine_orient = riet_c3.checkbox("Refine preferred orientation", value=True, key="riet_orient")
    riet_range = riet_c4.slider("2θ range for refinement (°)", 5.0, 90.0, (5.0, 70.0), step=0.5, key="riet_range")

    if selected_phases:
        # Get data for selected sample
        try:
            r_tt_raw, r_int_raw, _ = load_xrd(riet_sample)
            r_tt, r_int = r_tt_raw.copy(), r_int_raw.copy()
        except Exception as e:
            st.error(f"Could not load XRD data for {riet_sample}: {e}")
            r_tt, r_int = np.array([]), np.array([])

        # Apply range
        r_mask = (r_tt >= riet_range[0]) & (r_tt <= riet_range[1])
        r_tt = r_tt[r_mask]
        r_int = r_int[r_mask]

        run_rietveld = st.button("Run Rietveld Refinement", type="primary", key="riet_run")

        if run_rietveld:
            with st.spinner("Running Rietveld refinement... (this may take a moment)"):
                riet_result = rietveld_refine(
                    r_tt, r_int,
                    phase_names=selected_phases,
                    n_bg_coeffs=n_bg,
                    refine_orientation=refine_orient,
                    max_iterations=max_iter,
                )

            # Store in session state for persistence
            st.session_state["riet_result"] = riet_result
            st.session_state["riet_tt"] = r_tt
            st.session_state["riet_int"] = r_int

        # Display results if available
        if "riet_result" in st.session_state:
            riet_result = st.session_state["riet_result"]
            r_tt = st.session_state["riet_tt"]
            r_int = st.session_state["riet_int"]

            # --- R-factor metrics ---
            st.markdown("### Refinement Quality")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rwp", f"{riet_result.Rwp:.2f}%",
                       help="Weighted profile R-factor (lower is better, <15%% is good)")
            m2.metric("Rp", f"{riet_result.Rp:.2f}%",
                       help="Profile R-factor")
            m3.metric("χ²", f"{riet_result.chi_squared:.3f}",
                       help="Reduced chi-squared (ideally ~1)")
            m4.metric("GoF", f"{riet_result.GoF:.3f}",
                       help="Goodness of fit = √(χ²)")

            # --- Classic Rietveld Plot ---
            st.markdown("### Rietveld Plot")
            from plotly.subplots import make_subplots

            # Build phase names for color customizer
            riet_phase_names = list(riet_result.bragg_positions.keys())
            riet_trace_names = ["Y_obs", "Y_calc", "Background", "Difference"] + [f"Bragg: {p}" for p in riet_phase_names]
            riet_default_colors = ["#94a3b8", "#ef4444", "#6366f1", "#22d3ee"] + ["#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#8b5cf6"][:len(riet_phase_names)]
            _riet_key = "colors_rietveld"
            if _riet_key not in st.session_state:
                st.session_state[_riet_key] = list(riet_default_colors)
            riet_colors = st.session_state[_riet_key]

            fig_riet = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.75, 0.25],
                subplot_titles=("Observed vs Calculated", "Difference (Yobs − Ycalc)"),
            )

            # Top panel: Observed (dots) + Calculated (red line) + Background (dashed)
            fig_riet.add_trace(go.Scatter(
                x=r_tt, y=riet_result.y_obs,
                name="Y_obs (observed)",
                mode="markers",
                marker=dict(size=2, color=riet_colors[0]),
                hovertemplate="2θ=%{x:.2f}°<br>I=%{y:.0f}<extra>Observed</extra>",
            ), row=1, col=1)

            fig_riet.add_trace(go.Scatter(
                x=r_tt, y=riet_result.y_calc,
                name="Y_calc (calculated)",
                line=dict(color=riet_colors[1], width=1.5),
                hovertemplate="2θ=%{x:.2f}°<br>I=%{y:.0f}<extra>Calculated</extra>",
            ), row=1, col=1)

            fig_riet.add_trace(go.Scatter(
                x=r_tt, y=riet_result.y_background,
                name="Background",
                line=dict(color=riet_colors[2], width=1, dash="dash"),
                hovertemplate="2θ=%{x:.2f}°<br>BG=%{y:.0f}<extra>Background</extra>",
            ), row=1, col=1)

            # Bragg tick marks (vertical lines at bottom of top panel)
            y_bragg_base = -riet_result.y_obs.max() * 0.03
            for idx, (phase_name, bragg_list) in enumerate(riet_result.bragg_positions.items()):
                color = riet_colors[4 + idx] if (4 + idx) < len(riet_colors) else "#22c55e"
                tick_y = y_bragg_base - idx * riet_result.y_obs.max() * 0.025
                bragg_x = [b[0] for b in bragg_list]
                bragg_hkl = [b[1] for b in bragg_list]

                fig_riet.add_trace(go.Scatter(
                    x=bragg_x,
                    y=[tick_y] * len(bragg_x),
                    name=f"Bragg: {phase_name}",
                    mode="markers",
                    marker=dict(symbol="line-ns", size=12, line_width=2, color=color),
                    text=bragg_hkl,
                    hovertemplate="2θ=%{x:.2f}° %{text}<extra>" + phase_name + "</extra>",
                ), row=1, col=1)

            # Bottom panel: Difference curve
            fig_riet.add_trace(go.Scatter(
                x=r_tt, y=riet_result.y_diff,
                name="Difference",
                line=dict(color=riet_colors[3], width=1),
                fill="tozeroy",
                fillcolor=f"rgba({int(riet_colors[3][1:3],16)},{int(riet_colors[3][3:5],16)},{int(riet_colors[3][5:7],16)},0.15)",
                hovertemplate="2θ=%{x:.2f}°<br>Δ=%{y:.0f}<extra>Difference</extra>",
                showlegend=False,
            ), row=2, col=1)

            fig_riet.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=2, col=1)

            fig_riet.update_layout(
                height=700,
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(x=0.55, y=0.98, font_size=11),
                margin=dict(t=40),
            )
            fig_riet.update_xaxes(title_text="2θ (degrees)", row=2, col=1)
            fig_riet.update_yaxes(title_text="Intensity (counts)", row=1, col=1)
            fig_riet.update_yaxes(title_text="Yobs − Ycalc", row=2, col=1)

            st.plotly_chart(fig_riet, width="stretch")
            riet_colors = color_customizer("rietveld", riet_trace_names, riet_default_colors)

            # --- Phase Fractions ---
            st.markdown("### Refined Phase Fractions")
            phase_df = pd.DataFrame(riet_result.phases)
            display_phase_cols = ["name", "space_group", "weight_fraction_pct",
                                  "a_refined", "c_refined", "delta_a", "delta_c"]
            if "march_dollase_r" in phase_df.columns:
                display_phase_cols.append("march_dollase_r")
            display_phase_cols = [c for c in display_phase_cols if c in phase_df.columns]

            st.dataframe(
                phase_df[display_phase_cols].style.format({
                    "weight_fraction_pct": "{:.1f}%",
                    "a_refined": "{:.4f} Å",
                    "c_refined": "{:.4f} Å",
                    "delta_a": "{:+.4f} Å",
                    "delta_c": "{:+.4f} Å",
                    "march_dollase_r": "{:.3f}",
                }),
                width="stretch",
            )

            # Phase pie chart
            phase_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
            if len(riet_result.phases) > 1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[p["name"] for p in riet_result.phases],
                    values=[p["weight_fraction_pct"] for p in riet_result.phases],
                    hole=0.4,
                    marker_colors=phase_colors[:len(riet_result.phases)],
                    textinfo="label+percent",
                )])
                fig_pie.update_layout(
                    title="Phase Weight Fractions",
                    height=350, template="plotly_dark",
                )
                st.plotly_chart(fig_pie, width="stretch")

            # --- Lattice Parameters ---
            st.markdown("### Refined Lattice Parameters")
            for p in riet_result.phases:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        f"{p['name']} — a",
                        f"{p['a_refined']:.4f} Å",
                        delta=f"{p['delta_a']:+.4f} Å",
                        help=f"Initial: {p['a_initial']:.4f} Å",
                    )
                with col_b:
                    st.metric(
                        f"{p['name']} — c",
                        f"{p['c_refined']:.4f} Å",
                        delta=f"{p['delta_c']:+.4f} Å",
                        help=f"Initial: {p['c_initial']:.4f} Å",
                    )

            # --- Crystal Structure & Atom Sites ---
            st.markdown("### Crystal Structure Parameters")

            # Structure summary
            struct_sum = structure_summary(selected_phases)
            if struct_sum:
                sum_df = pd.DataFrame(struct_sum)
                st.dataframe(
                    sum_df.style.format({
                        "a": "{:.4f} Å",
                        "c": "{:.4f} Å",
                        "atoms_per_cell": "{:.2f}",
                    }),
                    width="stretch",
                )

            # Atom site parameters table
            st.markdown("### Reitveld Refinement Parameters")
            st.markdown(
                "Fractional coordinates (x, y, z), site occupancy, and "
                "isotropic thermal displacement parameters (U_iso in Å², "
                "B_iso = 8π²·U_iso in Å²)."
            )
            atom_table = atom_site_table(selected_phases)
            if atom_table:
                atom_df = pd.DataFrame(atom_table)
                st.dataframe(
                    atom_df.style.format({
                        "x": "{:.5f}",
                        "y": "{:.5f}",
                        "z": "{:.5f}",
                        "occupancy": "{:.3f}",
                        "U_iso": "{:.4f}",
                        "B_iso": "{:.4f}",
                    }).background_gradient(
                        subset=["occupancy"], cmap="YlOrRd", vmin=0, vmax=1
                    ).background_gradient(
                        subset=["U_iso"], cmap="Blues"
                    ).background_gradient(
                        subset=["B_iso"], cmap="Purples"
                    ).background_gradient(
                        subset=["mult"], cmap="YlGn"
                    ),
                    width="stretch",
                )
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    color_scale_bar("Occupancy", "0", "1", ["#ffffb2", "#fecc5c", "#fd8d3c", "#e31a1c"])
                    color_scale_bar("Multiplicity", "Low", "High", ["#ffffcc", "#addd8e", "#41ab5d", "#006837"])
                with col_s2:
                    color_scale_bar("U_iso", "Low", "High", ["#f7fbff", "#6baed6", "#2171b5", "#08306b"])
                    color_scale_bar("B_iso", "Low", "High", ["#fcfbfd", "#9e9ac8", "#6a51a3", "#3f007d"])

                # Visual: atom count per element per phase
                st.markdown("#### Atoms per Unit Cell by Element")
                elem_data = {}
                for row in atom_table:
                    key = (row["phase"], row["element"])
                    count = row["mult"] * row["occupancy"]
                    elem_data[key] = elem_data.get(key, 0) + count

                elem_rows = [{"Phase": k[0], "Element": k[1], "Count": v}
                             for k, v in elem_data.items()]
                if elem_rows:
                    elem_df = pd.DataFrame(elem_rows)
                    fig_elem = px.bar(
                        elem_df, x="Element", y="Count", color="Phase",
                        barmode="group",
                        color_discrete_sequence=["#22c55e", "#f59e0b", "#3b82f6",
                                                  "#ec4899", "#8b5cf6"],
                    )
                    fig_elem.update_layout(
                        height=350, template="plotly_dark",
                        title="Element Distribution in Unit Cell",
                    )
                    st.plotly_chart(fig_elem, width="stretch")

            # --- Bragg Peak List ---
            with st.expander("Bragg Peak Positions"):
                bragg_table = bragg_peak_table(riet_result)
                if bragg_table:
                    st.dataframe(
                        pd.DataFrame(bragg_table).style.format({
                            "two_theta": "{:.3f}°",
                            "d_spacing": "{:.4f} Å",
                            "intensity": "{:.1f}",
                        }),
                        width="stretch",
                    )

            # --- Profile Parameters ---
            with st.expander("Refined Profile & Background Parameters"):
                rp = riet_result.refined_params
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("U (Caglioti)", f"{rp['U']:.4f}")
                pc2.metric("V (Caglioti)", f"{rp['V']:.4f}")
                pc3.metric("W (Caglioti)", f"{rp['W']:.4f}")
                pc4.metric("η (pV mixing)", f"{rp['eta']:.3f}")

                st.markdown(f"**Background coefficients (Chebyshev):** "
                            f"{[f'{c:.1f}' for c in rp['background_coeffs']]}")
                st.markdown(f"**Convergence:** {rp['message']}")
                st.markdown(f"**Function evaluations:** {rp['n_iterations']}")

    else:
        st.warning("Please select at least one phase to refine.")

    # --- LAYER 2: Automated Peak Analysis ---
    st.markdown("---")
    st.markdown("## Layer 2: Automated Peak Analysis")

    # Use the same samples available from universal ETL
    peak_analysis_samples = get_xrd_samples()
    if not peak_analysis_samples:
        st.warning("No XRD samples available for peak analysis.")
        st.stop()

    # Default to first selected sample or first available
    default_peak_sample = selected_samples[0] if selected_samples else peak_analysis_samples[0]
    default_idx = peak_analysis_samples.index(default_peak_sample) if default_peak_sample in peak_analysis_samples else 0

    analysis_sample = st.selectbox(
        "Analyze sample:", peak_analysis_samples, index=default_idx,
        key="xrd_analysis_sample",
    )

    try:
        a_tt, a_int, _ = load_xrd(analysis_sample)
        a_tt, a_int = a_tt.copy(), a_int.copy()
    except Exception as e:
        st.error(f"Could not load {analysis_sample}: {e}")
        st.stop()

    # Analysis controls
    acol1, acol2, acol3, acol4 = st.columns(4)
    a_prominence = acol1.number_input("Peak prominence", 50, 5000, 200, step=50, key="xrd_prom")
    a_height_pct = acol2.number_input("Min height (%)", 1, 50, 3, key="xrd_hpct")
    a_profile = acol3.selectbox("Fit profile", ["pseudo_voigt", "gaussian", "lorentzian"], key="xrd_prof")
    a_fit_window = acol4.number_input("Fit window (°)", 0.3, 3.0, 1.0, step=0.1, key="xrd_fw")

    from src.analysis.xrd_analysis import full_xrd_analysis, gaussian as gauss_fn, pseudo_voigt as pv_fn, lorentzian as lor_fn

    with st.spinner("Running peak analysis..."):
        xrd_result = full_xrd_analysis(
            a_tt, a_int, profile=a_profile,
            prominence=a_prominence, height_pct=a_height_pct,
            fit_window=a_fit_window,
        )

    st.success(f"Detected **{xrd_result['peaks_detected']}** peaks, fitted **{len(xrd_result['fitted_peaks'])}**")

    # Peak fit overlay plot
    fig_fit = go.Figure()
    fig_fit.add_trace(go.Scatter(
        x=a_tt, y=a_int, name="Raw data",
        line=dict(color="#64748b", width=1),
    ))

    # Plot individual fitted peaks
    profile_fn = {"gaussian": gauss_fn, "lorentzian": lor_fn, "pseudo_voigt": pv_fn}[a_profile]
    for i, peak in enumerate(xrd_result["fitted_peaks"]):
        x_fit = np.linspace(peak["center_2theta"] - a_fit_window,
                            peak["center_2theta"] + a_fit_window, 200)
        params = peak["params"]
        if a_profile == "pseudo_voigt":
            y_fit = pv_fn(x_fit, params["amp"], params["center"], params["sigma"], params["eta"])
        elif a_profile == "gaussian":
            y_fit = gauss_fn(x_fit, params["amp"], params["center"], params["sigma"])
        else:
            y_fit = lor_fn(x_fit, params["amp"], params["center"], params["gamma"])

        label = peak.get("miller_index", "") or ""
        phase_short = peak.get("phase", "")[:10]
        fig_fit.add_trace(go.Scatter(
            x=x_fit, y=y_fit, name=f"{peak['center_2theta']:.1f} {label} {phase_short}",
            line=dict(width=1.5, dash="dash"),
            fill="tozeroy", opacity=0.4,
        ))

    fig_fit.update_layout(
        title="Peak Fitting Results",
        xaxis_title="2θ (°)", yaxis_title="Intensity",
        height=500, template="plotly_dark", hovermode="x unified",
    )
    st.plotly_chart(fig_fit, width="stretch")

    # Phase identification results
    st.markdown("### Phase Identification")
    if xrd_result["phases"]:
        for phase_name, info in xrd_result["phases"].items():
            conf_color = "#10b981" if info["avg_confidence"] > 0.7 else "#f59e0b" if info["avg_confidence"] > 0.4 else "#ef4444"
            st.markdown(
                f'<div style="display:inline-block;background:{conf_color}22;border:1px solid {conf_color};'
                f'border-radius:8px;padding:8px 16px;margin:4px;">'
                f'<strong style="color:{conf_color}">{phase_name}</strong> - '
                f'{info["count"]} peaks matched, avg confidence: {info["avg_confidence"]:.0%}</div>',
                unsafe_allow_html=True,
            )

    # Fitted peaks table
    st.markdown("### Fitted Peak Parameters")
    if xrd_result["fitted_peaks"]:
        peaks_table = pd.DataFrame(xrd_result["fitted_peaks"])
        display_cols = ["center_2theta", "intensity", "fwhm", "d_spacing",
                        "area", "r_squared", "phase", "miller_index"]
        display_cols = [c for c in display_cols if c in peaks_table.columns]
        st.dataframe(
            peaks_table[display_cols].style.format({
                "center_2theta": "{:.2f}",
                "intensity": "{:.0f}",
                "fwhm": "{:.4f}",
                "d_spacing": "{:.4f}",
                "area": "{:.1f}",
                "r_squared": "{:.4f}",
            }),
            width="stretch",
        )

    # Scherrer crystallite size
    st.markdown("### Crystallite Size (Scherrer Equation)")
    if xrd_result["scherrer"]:
        sch_df = pd.DataFrame(xrd_result["scherrer"])
        fig_sch = px.bar(
            sch_df, x="peak_2theta", y="crystallite_size_nm",
            title="Estimated Crystallite Size by Peak Position",
            labels={"peak_2theta": "2θ (°)", "crystallite_size_nm": "Size (nm)"},
            color="crystallite_size_nm", color_continuous_scale="Viridis",
        )
        fig_sch.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_sch, width="stretch")

        avg_size = np.mean([s["crystallite_size_nm"] for s in xrd_result["scherrer"]])
        st.metric("Average Crystallite Size", f"{avg_size:.1f} nm",
                  help="Scherrer equation: L = Kλ / (β·cos(θ)), K=0.9")


# ===========================================================================
# PAGE: XPS Analysis
# ===========================================================================
elif page == "XPS Analysis":
    st.markdown("## XPS Analysis - Interactive Spectroscopy")

    # Load quantification
    xps_quant = load_json(str(DATA_DIR / "xps" / "xps_quantification.json"))

    # Composition summary
    c1, c2, c3, c4 = st.columns(4)
    for col, el in zip([c1, c2, c3, c4], xps_quant["elements"]):
        col.metric(el["peak"], f"{el['atomic_conc_pct']}%",
                   help=f"BE = {el['position_be_ev']} eV, FWHM = {el['fwhm_ev']} eV")

    st.markdown("---")

    # Sidebar controls
    st.sidebar.markdown("### XPS Settings")
    xps_view = st.sidebar.radio(
        "Spectrum View",
        ["Survey", "Ti 2p", "C 1s", "O 1s", "F 1s", "All High-Res"],
    )
    show_refs = st.sidebar.checkbox("Show reference peak positions", value=True)
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

        # Mark element positions
        for el in xps_quant["elements"]:
            fig.add_vline(
                x=el["position_be_ev"], line_dash="dash",
                line_color="rgba(255,255,255,0.4)",
                annotation_text=el["peak"],
                annotation_font_size=11,
                annotation_font_color="#e2e8f0",
            )

        fig.update_layout(
            title="XPS Survey Spectrum - Ti3C2Tx MXene",
            xaxis_title="Binding Energy (eV)",
            yaxis_title="Intensity (CPS)" if not xps_normalize else "Normalized Intensity",
            xaxis=dict(autorange="reversed"),
            height=550,
            template="plotly_dark",
        )
        st.plotly_chart(fig, width="stretch")

    elif xps_view == "All High-Res":
        elements = ["Ti_2p", "C_1s", "O_1s", "F_1s"]
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=["Ti 2p", "C 1s", "O 1s", "F 1s"],
                           horizontal_spacing=0.08, vertical_spacing=0.12)
        colors = ["#06b6d4", "#8b5cf6", "#f43f5e", "#10b981"]

        for idx, (el_key, color) in enumerate(zip(elements, colors)):
            row, col = divmod(idx, 2)
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

            # Add reference lines
            if show_refs:
                el_label = el_key.replace("_", " ")
                if el_label in XPS_REFS:
                    for pos, label in XPS_REFS[el_label]:
                        fig.add_vline(
                            x=pos, line_dash="dot",
                            line_color="rgba(255,255,255,0.25)",
                            annotation_text=label,
                            annotation_font_size=8,
                            row=row + 1, col=col + 1,
                        )

        fig.update_xaxes(autorange="reversed")
        fig.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    else:
        # Individual high-res spectrum
        el_key = xps_view.replace(" ", "_")
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

        # Reference peaks
        if show_refs and xps_view in XPS_REFS:
            for pos, label in XPS_REFS[xps_view]:
                fig.add_vline(
                    x=pos, line_dash="dash",
                    line_color="rgba(255,255,255,0.35)",
                    annotation_text=label,
                    annotation_font_size=10,
                    annotation_font_color="#fbbf24",
                )

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

    # Quantification table
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
    st.markdown("---")
    st.markdown("## Layer 2: Peak Deconvolution")

    from src.analysis.xps_analysis import full_xps_analysis, gl_peak, XPS_REFERENCES

    deconv_element = st.selectbox(
        "Element to deconvolve",
        ["Ti 2p", "C 1s", "O 1s", "F 1s"],
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

        # Component quantification
        st.markdown("### Component Quantification")
        if xps_result["quantification"]:
            q_df = pd.DataFrame(xps_result["quantification"])
            qcol1, qcol2 = st.columns(2)

            with qcol1:
                fig_qpie = px.pie(
                    q_df, values="relative_pct", names="component",
                    title=f"{deconv_element} - Chemical State Distribution",
                    color_discrete_sequence=xps_colors[3:],
                    hole=0.35,
                )
                fig_qpie.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig_qpie, width="stretch")

            with qcol2:
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

    # Magnification comparison - color by instrument
    st.markdown("### Magnification Overview")
    fig_mag = px.scatter(
        filtered, x="magnification", y="pixel_size_nm",
        color="family", symbol="instrument",
        hover_name="sample_name",
        hover_data=["image_name", "accelerating_voltage_kv"],
        log_x=True, log_y=True,
        title="Magnification vs Pixel Size — colored by family, shaped by instrument",
        labels={"magnification": "Magnification (×)", "pixel_size_nm": "Pixel Size (nm)",
                "family": "Material Family", "instrument": "Instrument"},
    )
    fig_mag.update_layout(height=450, template="plotly_dark")
    st.plotly_chart(fig_mag, width="stretch")

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
                viz_col1, viz_col2 = st.columns(2)

                # SEM color defaults (managed via session state)
                _sem_h_key = "colors_sem_histogram"
                _sem_h_defaults = ["#06b6d4", "#ef4444", "#22c55e"]
                if _sem_h_key not in st.session_state:
                    st.session_state[_sem_h_key] = list(_sem_h_defaults)
                sem_hist_colors = st.session_state[_sem_h_key]

                _sem_a_key = "colors_sem_aspect_ratio"
                _sem_a_defaults = ["#a855f7", "#666666"]
                if _sem_a_key not in st.session_state:
                    st.session_state[_sem_a_key] = list(_sem_a_defaults)
                sem_ar_colors = st.session_state[_sem_a_key]

                with viz_col1:
                    # Particle size distribution histogram
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
                        # Add mean and median lines
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

                with viz_col2:
                    # Aspect ratio distribution
                    aspect_ratios = [p.aspect_ratio for p in sem_result.particles]
                    fig_ar = go.Figure()
                    fig_ar.add_trace(go.Histogram(
                        x=aspect_ratios,
                        nbinsx=15,
                        marker_color=sem_ar_colors[0],
                        opacity=0.85,
                        hovertemplate="Aspect Ratio: %{x:.2f}<br>Count: %{y}<extra></extra>",
                    ))
                    fig_ar.update_layout(
                        title="Aspect Ratio Distribution",
                        xaxis_title="Aspect Ratio (major/minor axis)",
                        yaxis_title="Count",
                        template="plotly_dark",
                        height=400,
                    )
                    fig_ar.add_vline(x=1.0, line_dash="dot", line_color=sem_ar_colors[1],
                                     annotation_text="Circle (1.0)")
                    st.plotly_chart(fig_ar, use_container_width=True)
                    sem_ar_colors = color_customizer("sem_aspect_ratio",
                        ["Bars", "Circle ref line"], _sem_a_defaults)

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

    # Imaging conditions table
    with st.expander("Full Imaging Conditions Table"):
        display_cols = [c for c in ["image_name", "sample_name", "family", "instrument",
                        "magnification", "accelerating_voltage_kv", "pixel_size_nm",
                        "working_distance_um", "field_of_view_um", "has_image"]
                        if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].sort_values("magnification").reset_index(drop=True),
            width="stretch",
        )
        st.markdown("""
        <div style="margin-top:10px; padding:12px 16px; background:rgba(255,255,255,0.03);
             border-radius:8px; border:1px solid rgba(255,255,255,0.08); font-size:0.82em; color:#aaa;">
        <b style="color:#ccc;">Column Definitions</b><br>
        <b>image_name</b> — Filename of the SEM micrograph<br>
        <b>sample_name</b> — Sample identifier from the instrument metadata<br>
        <b>family</b> — Material family classified by the sample matcher<br>
        <b>instrument</b> — SEM instrument (JEOL FE-SEM or Hitachi HR-FE-SEM)<br>
        <b>magnification</b> — Optical magnification (×). Higher = finer detail, smaller field of view<br>
        <b>accelerating_voltage_kv</b> — Electron beam energy (kV). Controls penetration depth and signal type<br>
        <b>pixel_size_nm</b> — Physical size each pixel represents (nm). Determines spatial resolution<br>
        <b>working_distance_um</b> — Distance between sample and objective lens (μm). Affects depth of field<br>
        <b>field_of_view_um</b> — Field of View (μm) — total area captured in the image<br>
        <b>has_image</b> — Whether the corresponding image file exists on disk
        </div>
        """, unsafe_allow_html=True)


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

    # -- Universal ETL Summary ------------------------------------------------
    st.markdown("### Universal ETL Summary")
    etl_stats_path = DATA_DIR / "universal" / "universal_etl_stats.json"
    if etl_stats_path.exists():
        etl_stats = load_json(str(etl_stats_path))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("XRD Patterns", etl_stats.get("xrd_datasets", "N/A"))
        m2.metric("SEM Images", etl_stats.get("sem_images_total", "N/A"))
        m3.metric("EDX Spectra", etl_stats.get("edx_spectra", "N/A"))
        m4.metric("Material Families", 18)  # from family classification
        m5.metric("Total Samples", etl_stats.get("total_samples", "N/A"))
    else:
        st.warning("Universal ETL stats not found. Run the universal ETL pipeline first.")

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

    # -- Family Feature Comparison --------------------------------------------
    st.markdown("### Family Feature Comparison")
    if family_csv_path.exists():
        family_df2 = load_csv(str(family_csv_path))
        if "n_techniques" in family_df2.columns:
            multi_tech = family_df2[family_df2["n_techniques"] >= 2].copy()
        else:
            multi_tech = family_df2.copy()

        compare_features = [
            f for f in ["xrd_n_peaks", "xrd_crystallite_size_nm", "xrd_peak_density", "sem_magnification"]
            if f in multi_tech.columns
        ]
        fam_col2 = "family" if "family" in multi_tech.columns else "material_family"
        if len(compare_features) > 0 and fam_col2 in multi_tech.columns and len(multi_tech) > 0:
            rename_compare = {
                "xrd_n_peaks": "XRD: Peak Count",
                "xrd_crystallite_size_nm": "XRD: Crystallite Size (nm)",
                "xrd_peak_density": "XRD: Peak Density (peaks/deg)",
                "sem_magnification": "SEM: Avg Magnification",
            }
            comp_melt = multi_tech.melt(
                id_vars=fam_col2,
                value_vars=compare_features,
                var_name="Feature",
                value_name="Value",
            )
            comp_melt["Feature"] = comp_melt["Feature"].map(rename_compare).fillna(comp_melt["Feature"])
            comp_melt = comp_melt.dropna(subset=["Value"])
            fig_comp = px.bar(
                comp_melt, x=fam_col2, y="Value", color="Feature",
                barmode="group", template="plotly_dark",
                labels={fam_col2: "Material Family", "Value": "Feature Value"},
            )
            fig_comp.update_layout(xaxis_tickangle=-45, margin=dict(b=100))
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("Not enough multi-technique families or features for comparison.")
    else:
        st.warning("Family feature matrix not found.")

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

                        tech_color_map = {
                            "MXene_Ti3C2": "#ef4444", "Ti3AlC2_MAX": "#f97316",
                            "CF_Conductive_Fabric": "#8b5cf6", "NiCu_Fabric": "#a78bfa",
                            "CAF_Carbon_Fabric": "#06b6d4", "BFO_BiFeO3": "#22c55e",
                            "CoBFO": "#16a34a", "ZnBFO": "#84cc16",
                            "Bi2Se3": "#eab308", "Bi2Te3": "#f59e0b",
                            "Other": "#94a3b8",
                            "XRD": "#3b82f6", "EDX": "#22c55e", "SEM": "#f59e0b",
                        }

                        fig_pca = px.scatter(
                            pca_plot_df, x="PC1", y="PC2", color="group",
                            hover_data=["sample"],
                            color_discrete_map=tech_color_map,
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
                parcoord_tech = st.selectbox(
                    "Technique", ["XRD", "SEM", "EDX"], key="parcoord_tech"
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
                    "Technique", ["XRD", "SEM", "EDX"], key="dist_tech"
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

                    family_colors = {
                        "MXene_Ti3C2": "#ef4444", "Ti3AlC2_MAX": "#f97316",
                        "CF_Conductive_Fabric": "#8b5cf6", "CAF_Carbon_Fabric": "#06b6d4",
                        "BFO_BiFeO3": "#22c55e", "CoBFO": "#16a34a", "ZnBFO": "#84cc16",
                        "Bi2Se3": "#eab308", "Bi2Te3": "#f59e0b",
                        "NiCu_Fabric": "#a78bfa", "AgCu_Alloy": "#fb923c",
                        "Other": "#94a3b8",
                    }
                    sorted_families = sorted(sub["family"].unique())

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
