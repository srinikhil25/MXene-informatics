# -*- coding: utf-8 -*-
"""
EDS Analysis Page
==================
Elemental spectra from EMSA/SPX files, peak identification, quantification.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Guard ──
project = st.session_state.get("project")
if project is None:
    st.warning("No project loaded. Go to Overview to load data.")
    st.stop()

eds_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("EDS")}
if not eds_samples:
    st.info("No EDS data available in the current project.")
    st.stop()

# Common EDS reference lines (keV)
EDS_LINES = {
    "B K\u03b1": 0.183, "C K\u03b1": 0.277, "N K\u03b1": 0.392, "O K\u03b1": 0.525,
    "Na K\u03b1": 1.041, "Mg K\u03b1": 1.254, "Al K\u03b1": 1.487,
    "Si K\u03b1": 1.740, "P K\u03b1": 2.013, "S K\u03b1": 2.307, "Cl K\u03b1": 2.622,
    "K K\u03b1": 3.314, "Ca K\u03b1": 3.691, "Ti K\u03b1": 4.511,
    "Fe K\u03b1": 6.404, "Cu K\u03b1": 8.048, "Cu L\u03b1": 0.930,
    "Zn K\u03b1": 8.638, "Se K\u03b1": 11.222, "Bi L\u03b1": 10.839,
    "Cs L\u03b1": 4.286, "I L\u03b1": 3.937,
}

# ── Header ──
st.markdown("## EDS/EDX Analysis - Elemental Identification")

# ── Sidebar ──
st.sidebar.markdown("### EDS Settings")

eds_sample_ids = sorted(eds_samples.keys())
selected_eds_sample = st.sidebar.selectbox("Sample", eds_sample_ids, key="eds_sample")

eds_log = st.sidebar.checkbox("Log scale (Y)", value=False, key="eds_log")
eds_range = st.sidebar.slider(
    "Energy Range (keV)", 0.0, 20.0, (0.0, 12.0), step=0.1, key="eds_range"
)
show_element_lines = st.sidebar.checkbox("Show element markers", value=True, key="eds_markers")

# ── Get EDS data for selected sample ──
td = project.samples[selected_eds_sample].techniques["EDS"]
spectra = td.parsed.get("spectra", [])

# Summary metrics
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Sample", selected_eds_sample)
mc2.metric("Spectra", len(spectra))
mc3.metric("Samples with EDS", len(eds_samples))

st.markdown("---")

if not spectra:
    st.warning("No parsed EDS spectra available for this sample.")
    st.stop()

# If multiple spectra, let user select
if len(spectra) > 1:
    spec_labels = [s.get("source_file", f"Spectrum {i+1}") for i, s in enumerate(spectra)]
    selected_idx = st.selectbox("Select Spectrum", range(len(spectra)),
                                format_func=lambda i: spec_labels[i], key="eds_spec_select")
else:
    selected_idx = 0

spec = spectra[selected_idx]

# Try to find energy/counts arrays
energy = None
counts = None

# EMSA format
if "energy_ev" in spec:
    energy = np.array(spec["energy_ev"]) / 1000.0  # eV -> keV
    counts = np.array(spec.get("counts", spec.get("intensity", [])))
elif "energy_kev" in spec:
    energy = np.array(spec["energy_kev"])
    counts = np.array(spec.get("counts", spec.get("intensity", [])))
elif "energy" in spec:
    energy = np.array(spec["energy"])
    # Heuristic: if max > 100, assume eV
    if energy is not None and len(energy) > 0 and energy.max() > 100:
        energy = energy / 1000.0
    counts = np.array(spec.get("counts", spec.get("intensity", [])))

if energy is not None and counts is not None and len(energy) > 0:
    mask = (energy >= eds_range[0]) & (energy <= eds_range[1])

    fig_eds = go.Figure()
    fig_eds.add_trace(go.Scatter(
        x=energy[mask], y=counts[mask],
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.2)",
        line=dict(color="#10b981", width=1),
        name="EDS Spectrum",
        hovertemplate="%{x:.3f} keV<br>%{y:.0f} counts<extra></extra>",
    ))

    if show_element_lines:
        # Only show markers where the spectrum has a significant peak nearby
        counts_in_range = counts[mask]
        energy_in_range = energy[mask]
        peak_threshold = np.max(counts_in_range) * 0.03 if len(counts_in_range) > 0 else 0

        for elem, pos_kev in EDS_LINES.items():
            if not (eds_range[0] <= pos_kev <= eds_range[1]):
                continue
            # Check if there's significant signal within ±0.15 keV of marker
            nearby = (energy >= pos_kev - 0.15) & (energy <= pos_kev + 0.15)
            if nearby.any() and np.max(counts[nearby]) > peak_threshold:
                fig_eds.add_vline(
                    x=pos_kev, line_dash="dot",
                    line_color="#fbbf24",
                    annotation_text=elem,
                    annotation_font_size=9,
                    annotation_font_color="#fbbf24",
                    annotation_position="top",
                )

    source = spec.get("source_file", "Unknown")
    source_name = Path(source).name if source != "Unknown" else source
    fig_eds.update_layout(
        title=f"EDS Spectrum \u2014 {selected_eds_sample} ({source_name})",
        xaxis_title="Energy (keV)",
        yaxis_title="Counts",
        height=550,
        template="plotly_dark",
    )
    if eds_log:
        fig_eds.update_yaxes(type="log")

    st.plotly_chart(fig_eds, use_container_width=True)

    # Metadata
    meta = spec.get("metadata", {})
    if meta:
        with st.expander("Spectrum Metadata"):
            for k, v in meta.items():
                st.markdown(f"- **{k}**: {v}")

    # Detected elements
    elements = spec.get("elements", [])
    if elements:
        st.markdown("### Detected Elements")
        st.markdown(", ".join(f"**{e}**" for e in elements))
else:
    st.warning("Could not load spectrum data. The EDS parser may need updating for this file format.")

# ── Cross-sample comparison ──
if len(eds_samples) > 1:
    st.markdown("---")
    st.markdown("### Cross-Sample EDS Comparison")
    st.caption("Compare detected elements across all samples with EDS data")

    comparison_data = []
    for sid in sorted(eds_samples.keys()):
        sample_td = project.samples[sid].techniques["EDS"]
        for spec_data in sample_td.parsed.get("spectra", []):
            elems = spec_data.get("elements", [])
            comparison_data.append({
                "Sample": sid,
                "Source": Path(spec_data.get("source_file", "")).name if spec_data.get("source_file") else "",
                "Elements": ", ".join(elems) if elems else "N/A",
                "N Elements": len(elems),
            })

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
