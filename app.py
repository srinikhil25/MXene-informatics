# -*- coding: utf-8 -*-
"""
MXene-Informatics Interactive Dashboard
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
    page_title="MXene-Informatics",
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
    j = load_json(str(DATA_DIR / "xrd" / f"xrd_{sample}.json"))
    return np.array(j["two_theta"]), np.array(j["intensity"]), j["metadata"]


def load_xps_hr(element_key):
    j = load_json(str(DATA_DIR / "xps" / f"xps_{element_key}.json"))
    return np.array(j["binding_energy_ev"]), np.array(j["intensity_cps"]), j["metadata"]


def load_eds_spectrum(name):
    j = load_json(str(DATA_DIR / "eds" / f"{name}.json"))
    return np.array(j["energy_ev"]), np.array(j["counts"]), j["metadata"]


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
st.sidebar.markdown("## MXene-Informatics")
page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "XRD Analysis",
        "XPS Analysis",
        "SEM Gallery",
        "EDS Analysis",
        "Data Export",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Summary")
st.sidebar.markdown("**Material:** Ti₃AlC₂ → Ti₃C₂Tₓ")
st.sidebar.markdown("**Instrument (XRD):** Ultima3")
st.sidebar.markdown("**Instrument (SEM):** SU8600")
st.sidebar.markdown("**Instrument (XPS):** PHI")


# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "Overview":
    st.markdown('<h1 class="main-header">MXene-Informatics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Autonomous Materials Informatics Pipeline for '
        'Ti&#8323;C&#8322;T&#8339; MXene Characterization</p>',
        unsafe_allow_html=True,
    )

    # Key metrics
    sem_catalog = load_json(str(DATA_DIR / "sem" / "sem_catalog.json"))
    xps_quant = load_json(str(DATA_DIR / "xps" / "xps_quantification.json"))
    eds_peaks = load_json(str(DATA_DIR / "eds" / "eds_peaks_summary.json"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("XRD Samples", "2")
    c2.metric("XPS Elements", f"{len(xps_quant['elements'])}")
    c3.metric("SEM Images", f"{len(sem_catalog)}")
    c4.metric("EDS Spectra", f"{len(eds_peaks)}")
    c5.metric("Data Points", "~15,000+")

    st.markdown("---")

    # Pipeline architecture
    st.markdown("### Pipeline Architecture")
    layers = [
        ("Layer 1", "Data Engineering", "ETL Pipeline", "#06b6d4",
         "Raw XRD/XPS/SEM/EDS -> Standardized JSON/CSV"),
        ("Layer 2", "Scientific Analysis", "Peak Fitting & Phase ID", "#8b5cf6",
         "Interactive XRD, XPS deconvolution, SEM viewer"),
        ("Layer 3 (Coming Soon)", "Machine Learning", "Surrogate Model", "#ec4899",
         "Predict properties from synthesis parameters"),
        ("Layer 4 (Coming Soon)", "Agentic Interface", "RAG + LLM", "#f59e0b",
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

    # XPS Composition at a glance
    st.markdown("### Surface Composition (XPS)")
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
        "Ti₃AlC₂ MAX phase successfully etched to Ti₃C₂Tₓ MXene — XRD confirms loss of (104) peak and shift of (002)",
        "Surface terminations: -O (23.1 at%) and -F (6.8 at%) confirmed by XPS — typical of HF/LiF+HCl etching",
        "Ti 2p binding energy at 455.4 eV indicates Ti–C bonds preserved; no significant TiO₂ contamination",
        "SEM shows characteristic accordion-like morphology under both Ar and N₂ atmospheres",
        "EDS confirms Al Kα peak intensity — key indicator for monitoring etching completion",
    ]
    for f in findings:
        st.markdown(f'<div class="finding-box">{f}</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: XRD Analysis
# ===========================================================================
elif page == "XRD Analysis":
    st.markdown("## XRD Analysis - Interactive Peak Identification")

    # Load both samples
    two_theta_max, intensity_max, meta_max = load_xrd("Ti2ALC3")
    two_theta_mx, intensity_mx, meta_mx = load_xrd("Ti2C3")

    # --- User Controls ---
    st.sidebar.markdown("### XRD Settings")
    show_max = st.sidebar.checkbox("Show Ti₃AlC₂ (MAX)", value=True)
    show_mxene = st.sidebar.checkbox("Show Ti₃C₂Tₓ (MXene)", value=True)
    show_ref_max = st.sidebar.checkbox("Show MAX reference peaks", value=True)
    show_ref_mx = st.sidebar.checkbox("Show MXene reference peaks", value=True)
    normalize = st.sidebar.checkbox("Normalize intensities", value=False)
    log_scale = st.sidebar.checkbox("Log scale (Y-axis)", value=False)

    range_min, range_max = st.sidebar.slider(
        "2θ Range (°)", 5.0, 90.0, (5.0, 70.0), step=0.5
    )
    smoothing = st.sidebar.slider("Smoothing (window)", 1, 21, 1, step=2)

    def smooth(y, window):
        if window <= 1:
            return y
        return np.convolve(y, np.ones(window) / window, mode="same")

    # Mask to range
    mask_max = (two_theta_max >= range_min) & (two_theta_max <= range_max)
    mask_mx = (two_theta_mx >= range_min) & (two_theta_mx <= range_max)

    i_max = smooth(intensity_max[mask_max], smoothing)
    i_mx = smooth(intensity_mx[mask_mx], smoothing)

    if normalize:
        i_max = i_max / i_max.max() * 100 if i_max.max() > 0 else i_max
        i_mx = i_mx / i_mx.max() * 100 if i_mx.max() > 0 else i_mx

    # Build plot — colors managed via session state
    xrd_trace_names = ["Ti₃AlC₂ (MAX)", "Ti₃C₂Tₓ (MXene)", "MAX ref peaks", "MXene ref peaks"]
    xrd_default_colors = ["#3b82f6", "#ef4444", "#60a5fa", "#f87171"]
    _xrd_key = "colors_xrd_pattern"
    if _xrd_key not in st.session_state:
        st.session_state[_xrd_key] = list(xrd_default_colors)
    xrd_colors = st.session_state[_xrd_key]

    fig_xrd = go.Figure()

    if show_max:
        fig_xrd.add_trace(go.Scatter(
            x=two_theta_max[mask_max], y=i_max,
            name="Ti₃AlC₂ (MAX phase)",
            line=dict(color=xrd_colors[0], width=1.5),
            hovertemplate="2θ = %{x:.2f}°<br>Intensity = %{y:.0f}<extra>MAX</extra>",
        ))

    if show_mxene:
        offset = i_max.max() * 0.05 if show_max and not normalize else 0
        fig_xrd.add_trace(go.Scatter(
            x=two_theta_mx[mask_mx], y=i_mx + offset,
            name="Ti₃C₂Tₓ (MXene)",
            line=dict(color=xrd_colors[1], width=1.5),
            hovertemplate="2θ = %{x:.2f}°<br>Intensity = %{y:.0f}<extra>MXene</extra>",
        ))

    # Reference peaks
    if show_ref_max:
        for pos, label in MAX_PEAKS["Ti3AlC2"]:
            if range_min <= pos <= range_max:
                fig_xrd.add_vline(
                    x=pos, line_dash="dot", line_color=xrd_colors[2],
                    annotation_text=label, annotation_position="top",
                    annotation_font_size=9, annotation_font_color=xrd_colors[2],
                )

    if show_ref_mx:
        for pos, label in MXENE_PEAKS["Ti3C2Tx"]:
            if range_min <= pos <= range_max:
                fig_xrd.add_vline(
                    x=pos, line_dash="dot", line_color=xrd_colors[3],
                    annotation_text=label, annotation_position="bottom",
                    annotation_font_size=9, annotation_font_color=xrd_colors[3],
                )

    y_title = "Normalized Intensity" if normalize else "Intensity (counts)"
    fig_xrd.update_layout(
        title="XRD Pattern Comparison: Ti₃AlC₂ (MAX) vs Ti₃C₂Tₓ (MXene)",
        xaxis_title="2θ (°)",
        yaxis_title=y_title,
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0.7, y=0.95),
    )
    if log_scale:
        fig_xrd.update_yaxes(type="log")

    st.plotly_chart(fig_xrd, width="stretch")
    xrd_colors = color_customizer("xrd_pattern", xrd_trace_names, xrd_default_colors)

    # Key observations
    st.markdown("### Key Observations")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAX (002) Peak", "9.5°", help="Characteristic MAX phase basal plane peak")
    col2.metric("MXene (002) Peak", "~6.6–9.0°", help="Shifts lower after etching due to increased d-spacing")
    col3.metric("Al Removal", "Loss of (104)", help="Disappearance of 38.9° peak confirms Al layer removal")

    with st.expander("d-Spacing Calculator"):
        st.markdown("**Bragg's Law:** nλ = 2d·sin(θ)")
        calc_angle = st.number_input("Enter 2θ (°):", value=9.5, step=0.1)
        wavelength = 1.54056  # Cu Ka1
        d_spacing = wavelength / (2 * np.sin(np.radians(calc_angle / 2)))
        st.markdown(f"**d-spacing = {d_spacing:.4f} Å** (λ = {wavelength} Å, Cu Kα₁)")
        st.markdown(f"For MAX (002) at 9.5°: d = {1.54056 / (2 * np.sin(np.radians(9.5/2))):.4f} Å")
        st.markdown(f"For MXene (002) at 6.6°: d = {1.54056 / (2 * np.sin(np.radians(6.6/2))):.4f} Å")
        st.info("d-spacing increase from ~9.3 Å to ~13.4 Å confirms successful intercalation/etching")

    # Metadata
    with st.expander("Instrument Metadata"):
        meta_df = pd.DataFrame([
            {"Parameter": "Instrument", "Value": meta_max["instrument"]},
            {"Parameter": "Target", "Value": meta_max["target"]},
            {"Parameter": "Wavelength (Kα₁)", "Value": f"{meta_max['wavelength_ka1']} Å"},
            {"Parameter": "Voltage", "Value": f"{meta_max['voltage_kv']} kV"},
            {"Parameter": "Current", "Value": f"{meta_max['current_ma']} mA"},
            {"Parameter": "Step Width", "Value": f"{meta_max['step_width']}°"},
            {"Parameter": "Monochromator", "Value": meta_max["monochromator"]},
        ])
        st.table(meta_df)

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
        riet_sample = st.radio(
            "Refine sample:",
            ["Ti₃AlC₂ (MAX)", "Ti₃C₂Tₓ (MXene)"],
            horizontal=True,
            key="riet_sample",
        )
    with riet_col2:
        available_phases = list(CRYSTAL_PHASES.keys())
        # Default phases based on sample
        if riet_sample == "Ti₃AlC₂ (MAX)":
            default_phases = ["Ti3AlC2", "TiC"]
        else:
            default_phases = ["Ti3C2Tx", "TiO2_Anatase"]
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
        if riet_sample == "Ti₃AlC₂ (MAX)":
            r_tt, r_int = two_theta_max.copy(), intensity_max.copy()
        else:
            r_tt, r_int = two_theta_mx.copy(), intensity_mx.copy()

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
                    ),
                    width="stretch",
                )

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

    analysis_sample = st.radio(
        "Analyze sample:", ["Ti₃AlC₂ (MAX)", "Ti₃C₂Tₓ (MXene)"], horizontal=True,
        key="xrd_analysis_sample",
    )

    if analysis_sample == "Ti₃AlC₂ (MAX)":
        a_tt, a_int = two_theta_max.copy(), intensity_max.copy()
    else:
        a_tt, a_int = two_theta_mx.copy(), intensity_mx.copy()

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

    with st.spinner("Running deconvolution..."):
        xps_result = full_xps_analysis(d_be, d_int, deconv_element,
                                       background_type=bg_type, gl_fraction=gl_frac)

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

    sem_catalog = load_json(str(DATA_DIR / "sem" / "sem_catalog.json"))
    sem_df = pd.DataFrame(sem_catalog)

    # Filters
    st.sidebar.markdown("### SEM Filters")
    sample_types = sem_df["sample_type"].unique().tolist()
    selected_type = st.sidebar.multiselect(
        "Sample Type", sample_types, default=sample_types
    )
    voltage_options = sorted(sem_df["accelerating_voltage_v"].unique())
    voltage_range = st.sidebar.select_slider(
        "Accelerating Voltage (V)",
        options=voltage_options,
        value=(voltage_options[0], voltage_options[-1]),
    )

    filtered = sem_df[
        (sem_df["sample_type"].isin(selected_type)) &
        (sem_df["accelerating_voltage_v"] >= voltage_range[0]) &
        (sem_df["accelerating_voltage_v"] <= voltage_range[1])
    ]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", len(filtered))
    c2.metric("With TIF", int(filtered["has_image"].sum()))
    c3.metric("Samples", len(filtered["sample_type"].unique()))
    c4.metric("Mag Range", f"{filtered['magnification'].min():.0f}x - {filtered['magnification'].max():.0f}x")

    st.markdown("---")

    # Magnification comparison
    st.markdown("### Magnification Overview")
    fig_mag = px.scatter(
        filtered, x="magnification", y="pixel_size_nm",
        color="sample_type", size="emission_current_na",
        hover_name="image_name",
        log_x=True, log_y=True,
        title="Magnification vs Pixel Size (nm)",
        labels={"magnification": "Magnification (x)", "pixel_size_nm": "Pixel Size (nm)"},
        color_discrete_sequence=["#06b6d4", "#ec4899"],
    )
    fig_mag.update_layout(height=400, template="plotly_dark")
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
                    f"**Sample:** {row['sample_type']}  \n"
                    f"**Voltage:** {row['accelerating_voltage_v']/1000:.0f} kV | "
                    f"**FOV:** {row['fov']} | "
                    f"**Pixel:** {row['pixel_size_nm']:.2f} nm"
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
        display_cols = ["image_name", "sample_type", "magnification",
                        "accelerating_voltage_v", "pixel_size_nm",
                        "working_distance_um", "emission_current_na",
                        "fov", "has_image"]
        st.dataframe(
            filtered[display_cols].sort_values("magnification").reset_index(drop=True),
            width="stretch",
        )
        st.markdown("""
        <div style="margin-top:10px; padding:12px 16px; background:rgba(255,255,255,0.03);
             border-radius:8px; border:1px solid rgba(255,255,255,0.08); font-size:0.82em; color:#aaa;">
        <b style="color:#ccc;">Column Definitions</b><br>
        <b>image_name</b> — Filename of the SEM micrograph<br>
        <b>sample_type</b> — Sample atmosphere during synthesis (Ar or N₂)<br>
        <b>magnification</b> — Optical magnification (×). Higher = finer detail, smaller field of view<br>
        <b>accelerating_voltage_v</b> — Electron beam energy (V). Controls penetration depth and signal type<br>
        <b>pixel_size_nm</b> — Physical size each pixel represents (nm). Determines spatial resolution<br>
        <b>working_distance_um</b> — Distance between sample and objective lens (μm). Affects depth of field<br>
        <b>emission_current_na</b> — Electron beam current (nA). Higher = brighter image, more beam damage<br>
        <b>fov</b> — Field of View — total area captured in the image<br>
        <b>has_image</b> — Whether the corresponding .tif image file exists on disk
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: EDS Analysis
# ===========================================================================
elif page == "EDS Analysis":
    st.markdown("## EDS Analysis - Elemental Identification")

    eds_peaks = load_json(str(DATA_DIR / "eds" / "eds_peaks_summary.json"))

    # Sidebar controls
    st.sidebar.markdown("### EDS Settings")
    eds_files = [p["source"].replace(".emsa", "") for p in eds_peaks]
    selected_spectrum = st.sidebar.selectbox("Select Spectrum", eds_files)
    eds_log = st.sidebar.checkbox("Log scale (Y)", value=False)
    eds_range = st.sidebar.slider("Energy Range (keV)", 0.0, 20.0, (0.0, 10.0), step=0.1)
    show_element_lines = st.sidebar.checkbox("Show element markers", value=True)

    # Element reference lines for EDS
    EDS_LINES = {
        "C Kα": 0.277, "N Kα": 0.392, "O Kα": 0.525,
        "F Kα": 0.677, "Ti Lα": 0.452, "Ti Kα": 4.511,
        "Ti Kβ": 4.932, "Al Kα": 1.487, "Cu Kα": 8.048,
        "Cu Lα": 0.930, "Cl Kα": 2.622, "Au Mα": 2.123,
    }

    # Find and load the matching spectrum
    spec_info = next((p for p in eds_peaks if p["source"].replace(".emsa", "") == selected_spectrum), None)

    # Find matching CSV file
    csv_candidates = list((DATA_DIR / "eds").glob("*.csv"))
    matched_csv = None

    # Extract digits from selected spectrum for matching
    spec_digits = "".join(c for c in selected_spectrum if c.isdigit())
    for c in csv_candidates:
        stem_digits = "".join(ch for ch in c.stem.split("_")[-1] if ch.isdigit())
        if stem_digits == spec_digits:
            matched_csv = c
            break

    # Fallback: try substring match
    if matched_csv is None:
        for c in csv_candidates:
            if spec_digits and spec_digits in c.stem:
                matched_csv = c
                break

    if matched_csv and matched_csv.exists():
        df_eds = pd.read_csv(matched_csv)
        energy_col = [c for c in df_eds.columns if "energy" in c.lower() or "ev" in c.lower()][0]
        counts_col = [c for c in df_eds.columns if "count" in c.lower() or "intensity" in c.lower()][0]

        energy = df_eds[energy_col].values / 1000  # eV to keV
        counts = df_eds[counts_col].values

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
            name="EDS",
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
            title=f"EDS Spectrum - {selected_spectrum}",
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
        st.warning(f"Could not find CSV data for spectrum: {selected_spectrum}")

    # Peaks table
    if spec_info:
        st.markdown("### Detected Peaks")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Beam Voltage", f"{spec_info['beam_kv']} kV")
        pc2.metric("Live Time", f"{spec_info['live_time_s']:.1f} s")
        pc3.metric("Peaks Found", len(spec_info["peaks"]))

        peaks_df = pd.DataFrame(spec_info["peaks"])
        st.dataframe(
            peaks_df.style.format({
                "expected_ev": "{:.0f}",
                "measured_ev": "{:.0f}",
                "intensity": "{:.0f}",
                "shift_ev": "{:.0f}",
            }).background_gradient(subset=["intensity"], cmap="YlGn"),
            width="stretch",
        )
        color_scale_bar("Intensity", "Low", "High", ["#ffffcc", "#addd8e", "#41ab5d", "#006837"])

    # Al tracking across spectra
    st.markdown("---")
    st.markdown("### Al Kα Peak Tracking Across Spectra")
    st.markdown("*Al removal is the key indicator of successful MAX to MXene etching*")

    al_data = []
    for spec in eds_peaks:
        al_peak = next((p for p in spec["peaks"] if p["element_line"] == "Al Ka"), None)
        ti_peak = next((p for p in spec["peaks"] if p["element_line"] == "Ti Ka"), None)
        if al_peak and ti_peak and ti_peak["intensity"] > 0:
            al_data.append({
                "spectrum": spec["source"],
                "Al_Ka_intensity": al_peak["intensity"],
                "Ti_Ka_intensity": ti_peak["intensity"],
                "Al_Ti_ratio": al_peak["intensity"] / ti_peak["intensity"],
            })

    if al_data:
        al_df = pd.DataFrame(al_data)
        fig_al = px.bar(
            al_df, x="spectrum", y="Al_Ti_ratio",
            title="Al Kα / Ti Kα Intensity Ratio — Lower = More Complete Etching",
            color="Al_Ti_ratio",
            color_continuous_scale="RdYlGn_r",
        )
        fig_al.update_layout(height=400, template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig_al, width="stretch")


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
    "<small>MXene-Informatics v1.0<br>"
    "Gudibandi Sri Nikhil Reddy<br>"
    "Ikeda - Hamasaki Laboratory<br>"
    "Research Institute of Electronics<br>"
    "Shizuoka University, Japan</small>",
    unsafe_allow_html=True,
)
