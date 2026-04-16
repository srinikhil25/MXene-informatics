# -*- coding: utf-8 -*-
"""
XRD Analysis Page
==================
Pattern viewer, phase identification, Scherrer crystallite size.
Migrated from app_legacy.py lines 1108-1517.
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

xrd_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("XRD")}
if not xrd_samples:
    st.info("No XRD data available in the current project.")
    st.stop()


# ── Helpers ──
def _load_xrd(sample_id: str):
    """Load XRD data from the project. Returns (two_theta, intensity, metadata, pattern_key)."""
    td = project.samples[sample_id].techniques["XRD"]
    patterns = td.parsed.get("patterns", {})
    if not patterns:
        return np.array([]), np.array([]), {}, ""
    # Use the first (usually only) pattern
    key = next(iter(patterns))
    pat = patterns[key]
    return np.array(pat["two_theta"]), np.array(pat["intensity"]), pat.get("metadata", {}), key


def _smooth(y, window):
    if window <= 1:
        return y
    return np.convolve(y, np.ones(window) / window, mode="same")


# ── Header ──
st.markdown("## XRD Analysis - Interactive Pattern Explorer")

all_xrd_ids = sorted(xrd_samples.keys())

# ── Sidebar Controls ──
st.sidebar.markdown("### XRD Settings")

selected_samples = st.sidebar.multiselect(
    "Select Patterns (max 6)",
    all_xrd_ids,
    default=all_xrd_ids[:min(2, len(all_xrd_ids))],
    max_selections=6,
    key="xrd_sample_select",
)

normalize = st.sidebar.checkbox("Normalize intensities", value=False, key="xrd_norm")
log_scale = st.sidebar.checkbox("Log scale (Y-axis)", value=False, key="xrd_log")
stack_offset = st.sidebar.checkbox(
    "Stack patterns (offset)",
    value=len(selected_samples) > 2,
    key="xrd_stack",
)
range_min, range_max = st.sidebar.slider(
    "2\u03b8 Range (\u00b0)", 5.0, 90.0, (5.0, 70.0), step=0.5, key="xrd_range"
)
smoothing = st.sidebar.slider("Smoothing (window)", 1, 21, 1, step=2, key="xrd_smooth")

# ── Summary metrics ──
c1, c2, c3 = st.columns(3)
c1.metric("Total XRD Patterns", len(all_xrd_ids))
c2.metric("Selected Patterns", len(selected_samples))
c3.metric("Samples in Project", len(project.samples))

st.markdown("---")

if not selected_samples:
    st.info("Select at least one XRD pattern from the sidebar.")
else:
    palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]

    fig_xrd = go.Figure()
    for i, sid in enumerate(selected_samples):
        try:
            tt, intensity, meta, pat_key = _load_xrd(sid)
        except Exception as e:
            st.warning(f"Could not load {sid}: {e}")
            continue

        if len(tt) == 0:
            continue

        mask = (tt >= range_min) & (tt <= range_max)
        y = _smooth(intensity[mask], smoothing)

        if normalize and y.max() > 0:
            y = y / y.max() * 100

        offset = 0
        if stack_offset and i > 0:
            offset = i * (y.max() * 0.3 if y.max() > 0 else 100)

        color = palette[i % len(palette)]
        fig_xrd.add_trace(go.Scatter(
            x=tt[mask], y=y + offset,
            name=sid,
            line=dict(color=color, width=1.3),
            hovertemplate=(
                f"<b>{sid}</b><br>"
                "2\u03b8 = %{x:.2f}\u00b0<br>"
                "Intensity = %{y:.0f}<extra></extra>"
            ),
        ))

    y_title = "Normalized Intensity" if normalize else "Intensity (counts)"
    if stack_offset:
        y_title += " (stacked)"

    fig_xrd.update_layout(
        title=f"XRD Pattern Comparison \u2014 {len(selected_samples)} pattern(s)",
        xaxis_title="2\u03b8 (\u00b0)",
        yaxis_title=y_title,
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(x=0.65, y=0.95, font=dict(size=10)),
    )
    if log_scale:
        fig_xrd.update_yaxes(type="log")

    st.plotly_chart(fig_xrd, use_container_width=True)

    # ── d-Spacing Calculator ──
    with st.expander("d-Spacing Calculator (Bragg's Law)"):
        calc_col1, calc_col2 = st.columns([2, 3])
        with calc_col1:
            calc_angle = st.number_input(
                "Enter 2\u03b8 (\u00b0):", value=9.5, step=0.1, key="xrd_dspacing"
            )
        with calc_col2:
            wavelength = 1.54056  # Cu Ka1
            if calc_angle > 0:
                d_spacing = wavelength / (2 * np.sin(np.radians(calc_angle / 2)))
                st.metric("d-spacing", f"{d_spacing:.4f} \u00c5", help="n\u03bb = 2d\u00b7sin(\u03b8), Cu K\u03b11")

# ═══════════════════════════════════════════════════════════════════════════
# Phase Identification
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Phase Identification")
st.markdown(
    "Match your experimental pattern against reference phases from the "
    "**Materials Project** database. Enter the phases you expect, and the agent "
    "will fetch reference stick patterns and assign your peaks."
)

if all_xrd_ids:
    # ── Controls ──
    pid_col1, pid_col2 = st.columns([1, 2])
    with pid_col1:
        pid_sample = st.selectbox(
            "Pattern to identify:",
            all_xrd_ids,
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
            "Peak tolerance (\u00b02\u03b8)", value=0.5, min_value=0.05,
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
                tt, inten, meta, _ = _load_xrd(pid_sample)
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
                    exp_peaks = find_peaks(tt, inten, prominence_pct=pid_prominence)
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
        tt = r["tt"]
        inten = r["inten"]

        # Summary metrics
        st.markdown("### Results")
        _zs = summary.get("zero_shift", 0.0)
        if abs(_zs) > 0.001:
            st.info(
                f"**Zero-shift correction applied:** {_zs:+.4f}\u00b0 "
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
            hovertemplate="2\u03b8=%{x:.2f}\u00b0 I=%{y:.1f}<extra>Experimental</extra>",
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
                    "2\u03b8=%{x:.2f}\u00b0 I=%{y:.1f}<br>"
                    "%{text}<extra>" + phase + "</extra>"
                ),
            ))

        # Reference stick patterns (below x-axis)
        _stick_base = -5
        _stick_gap = -25
        for i, ref in enumerate(refs):
            color = _phase_colors[ref.formula]
            y_base = _stick_base + i * _stick_gap

            sig_peaks = [p for p in ref.peaks if p.intensity > 3.0]

            for p in sig_peaks:
                fig_pid.add_trace(go.Scatter(
                    x=[p.two_theta, p.two_theta],
                    y=[y_base, y_base + p.intensity / 100.0 * abs(_stick_gap) * 0.8],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    hovertemplate=(
                        f"2\u03b8={p.two_theta:.2f}\u00b0 I={p.intensity:.1f} "
                        f"{p.hkl}<extra>{ref.formula}</extra>"
                    ),
                ))

            fig_pid.add_annotation(
                x=0.01, y=y_base + abs(_stick_gap) * 0.4,
                xref="paper", yref="y",
                text=f"<b>{ref.formula}</b> ({ref.space_group}) \u2014 {ref.material_id}",
                showarrow=False,
                font=dict(size=10, color=color),
                xanchor="left",
            )

        fig_pid.update_layout(
            xaxis_title="2\u03b8 (\u00b0)",
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
                "Exp 2\u03b8 (\u00b0)": f"{a.exp_two_theta:.3f}",
                "d (\u00c5)": f"{a.d_spacing:.4f}",
                "Intensity": f"{a.exp_intensity:.1f}",
                "Phase": a.matched_phase,
                "Ref 2\u03b8 (\u00b0)": f"{a.ref_two_theta:.3f}" if a.ref_two_theta > 0 else "\u2014",
                "\u03942\u03b8 (\u00b0)": f"{a.delta_two_theta:.3f}" if a.ref_two_theta > 0 else "\u2014",
                "hkl": a.hkl or "\u2014",
            })
        _tbl_df = pd.DataFrame(_tbl)

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
                    f"**{ref.formula}** \u2014 {ref.space_group} ({ref.crystal_system}) "
                    f"| MP: `{ref.material_id}` | E_hull: {ref.energy_above_hull:.3f} eV/atom"
                )
                latt = ref.lattice
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;a={latt['a']:.4f} \u00c5, "
                    f"b={latt['b']:.4f} \u00c5, "
                    f"c={latt['c']:.4f} \u00c5, "
                    f"\u03b1={latt['alpha']:.1f}\u00b0, \u03b2={latt['beta']:.1f}\u00b0, \u03b3={latt['gamma']:.1f}\u00b0"
                )
                top_peaks = [p for p in ref.peaks if p.intensity > 10][:10]
                if top_peaks:
                    st.caption(f"Top {len(top_peaks)} peaks (I > 10%):")
                    _rp_rows = [
                        {"2\u03b8 (\u00b0)": f"{p.two_theta:.3f}", "I (%)": f"{p.intensity:.1f}",
                         "d (\u00c5)": f"{p.d_spacing:.4f}", "hkl": p.hkl}
                        for p in top_peaks
                    ]
                    st.dataframe(pd.DataFrame(_rp_rows), use_container_width=True, hide_index=True)
