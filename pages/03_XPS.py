# -*- coding: utf-8 -*-
"""
XPS Analysis Page
==================
Survey + high-resolution spectra, peak deconvolution, quantification.
Multi-sample support via project.samples.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Guard ──
project = st.session_state.get("project")
if project is None:
    st.warning("No project loaded. Go to Overview to load data.")
    st.stop()

xps_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("XPS")}
if not xps_samples:
    st.info("No XPS data available in the current project.")
    st.stop()


def _get_regions(sample_id: str) -> dict:
    """Get XPS regions for a sample. Returns {region_name: {binding_energy, intensity, ...}}."""
    td = project.samples[sample_id].techniques["XPS"]
    return td.parsed.get("regions", {})


def _load_region(sample_id: str, region_name: str):
    """Load a specific XPS region. Returns (binding_energy, intensity, metadata)."""
    regions = _get_regions(sample_id)
    if region_name not in regions:
        return np.array([]), np.array([]), {}
    r = regions[region_name]
    return np.array(r["binding_energy"]), np.array(r["intensity"]), r.get("metadata", {})


# ── Header ──
st.markdown("## XPS Analysis - Interactive Spectroscopy")

# ── Sidebar Controls ──
st.sidebar.markdown("### XPS Settings")

# Sample selector
xps_sample_ids = sorted(xps_samples.keys())
selected_xps_sample = st.sidebar.selectbox(
    "Sample", xps_sample_ids, key="xps_sample_select"
)

# Get regions for selected sample
regions = _get_regions(selected_xps_sample)
region_names = sorted(regions.keys())
has_survey = "Survey" in region_names
hr_regions = [r for r in region_names if r != "Survey"]

# Build view options
view_options = []
if has_survey:
    view_options.append("Survey")
view_options.extend(hr_regions)
if len(hr_regions) > 1:
    view_options.append("All High-Res")

xps_view = st.sidebar.radio(
    "Spectrum View",
    view_options if view_options else ["No data"],
    key="xps_view",
)
xps_normalize = st.sidebar.checkbox("Normalize spectra", value=False, key="xps_normalize")

# ── Summary metrics ──
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Sample", selected_xps_sample)
mc2.metric("Regions", len(region_names))
mc3.metric("Samples with XPS", len(xps_samples))

st.markdown("---")

# ── Spectrum Display ──
_palette = ["#06b6d4", "#8b5cf6", "#f43f5e", "#10b981", "#f59e0b", "#ec4899",
            "#14b8a6", "#a855f7"]

if xps_view == "Survey":
    be, intensity, meta = _load_region(selected_xps_sample, "Survey")
    if len(be) == 0:
        st.warning("Survey data not available.")
    else:
        if xps_normalize and intensity.max() > 0:
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
        fig.update_layout(
            title=f"XPS Survey Spectrum \u2014 {selected_xps_sample}",
            xaxis_title="Binding Energy (eV)",
            yaxis_title="Normalized Intensity" if xps_normalize else "Intensity (CPS)",
            xaxis=dict(autorange="reversed"),
            height=550,
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

elif xps_view == "All High-Res":
    n_hr = len(hr_regions)
    n_rows = (n_hr + 1) // 2
    n_c = min(2, n_hr)
    fig = make_subplots(
        rows=n_rows, cols=n_c,
        subplot_titles=hr_regions,
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    for idx, region_name in enumerate(hr_regions):
        row, col = divmod(idx, n_c)
        color = _palette[idx % len(_palette)]
        be, intensity, _ = _load_region(selected_xps_sample, region_name)
        if len(be) == 0:
            continue
        if xps_normalize and intensity.max() > 0:
            intensity = intensity / intensity.max() * 100
        fig.add_trace(go.Scatter(
            x=be, y=intensity,
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.2)",
            line=dict(color=color, width=1.5),
            name=region_name,
            hovertemplate="BE = %{x:.1f} eV<br>%{y:.0f} CPS<extra></extra>",
        ), row=row + 1, col=col + 1)

    fig.update_xaxes(autorange="reversed")
    fig.update_layout(
        height=max(400, 350 * n_rows),
        template="plotly_dark",
        showlegend=False,
        title=f"XPS High-Resolution Regions \u2014 {selected_xps_sample}",
    )
    st.plotly_chart(fig, use_container_width=True)

elif xps_view != "No data":
    # Individual high-res spectrum
    be, intensity, meta = _load_region(selected_xps_sample, xps_view)
    if len(be) == 0:
        st.warning(f"No data for region: {xps_view}")
    else:
        if xps_normalize and intensity.max() > 0:
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
            title=f"XPS High-Resolution \u2014 {xps_view} ({selected_xps_sample})",
            xaxis_title="Binding Energy (eV)",
            yaxis_title="Normalized Intensity" if xps_normalize else "Intensity (CPS)",
            xaxis=dict(autorange="reversed"),
            height=550,
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Peak Deconvolution
# ═══════════════════════════════════════════════════════════════════════════
if not hr_regions:
    st.info("No high-resolution XPS spectra available for deconvolution.")
else:
    st.markdown("---")
    st.markdown("## Peak Deconvolution")

    try:
        from src.analysis.xps_analysis import full_xps_analysis, gl_peak, XPS_REFERENCES
    except ImportError:
        st.warning("XPS analysis module not available. Install dependencies.")
        st.stop()

    deconv_element = st.selectbox(
        "Element to deconvolve",
        hr_regions,
        key="xps_deconv_el",
    )

    dcol1, dcol2, dcol3 = st.columns(3)
    bg_type = dcol1.selectbox("Background", ["shirley", "linear"], key="xps_bg")
    gl_frac = dcol2.slider("GL mixing (0=Gauss, 1=Lorentz)", 0.0, 1.0, 0.3, step=0.1, key="xps_gl")
    auto_n = dcol3.checkbox("Auto-detect components", value=True, key="xps_auto")

    d_be, d_int, _ = _load_region(selected_xps_sample, deconv_element)

    if len(d_be) == 0:
        st.warning(f"No data for {deconv_element}")
    else:
        n_comp = None
        if not auto_n:
            n_comp = st.number_input("Number of components", 2, 8, 3, key="xps_ncomp")

        with st.spinner("Running deconvolution..."):
            xps_result_init = full_xps_analysis(
                d_be, d_int, deconv_element,
                background_type=bg_type, gl_fraction=gl_frac,
            )

        deconv_init = xps_result_init["deconvolution"]

        # Component editor
        refit = False
        comp_states = {}
        _custom_key = f"xps_custom_peaks_{deconv_element}"
        if _custom_key not in st.session_state:
            st.session_state[_custom_key] = []

        if deconv_init.n_components > 0:
            with st.expander("Edit Components (add/remove peaks, then re-fit)", expanded=False):
                st.caption("Toggle components on/off or add custom peaks. Click **Re-fit** to update.")

                comp_list = deconv_init.components
                cols_per_row = 4
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

                refit = st.button("Re-fit with selected components", key=f"xps_refit_{deconv_element}")

                add_peak = add_cols[2].button("Add", key=f"xps_add_{deconv_element}")
                if add_peak:
                    st.session_state[_custom_key].append({"be": custom_be, "label": custom_label})
                    st.rerun()

                if st.session_state[_custom_key]:
                    st.markdown("**Custom peaks added:**")
                    for ci, cp in enumerate(st.session_state[_custom_key]):
                        cc1, cc2 = st.columns([4, 1])
                        cc1.write(f"- {cp['label']} at {cp['be']:.1f} eV")
                        if cc2.button("Remove", key=f"xps_rm_{deconv_element}_{ci}"):
                            st.session_state[_custom_key].pop(ci)
                            st.rerun()

            # Determine if re-fit is needed
            selected_positions = [comp_list[i].center_ev for i, on in comp_states.items() if on]
            for cp in st.session_state.get(_custom_key, []):
                selected_positions.append(cp["be"])

            some_disabled = any(not on for on in comp_states.values())
            custom_peaks_exist = len(st.session_state.get(_custom_key, [])) > 0
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

            comp_defaults = ["#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899",
                             "#3b82f6", "#14b8a6", "#f43f5e"]

            fig_deconv = go.Figure()

            # Raw spectrum
            fig_deconv.add_trace(go.Scatter(
                x=be_arr, y=raw_arr, name="Raw",
                line=dict(color="#94a3b8", width=1),
            ))

            # Background
            fig_deconv.add_trace(go.Scatter(
                x=be_arr, y=bg_arr, name=f"Background ({bg_type})",
                line=dict(color="#475569", width=1, dash="dot"),
            ))

            # Envelope
            if deconv.envelope:
                fig_deconv.add_trace(go.Scatter(
                    x=be_arr, y=np.array(deconv.envelope),
                    name="Envelope", line=dict(color="#ef4444", width=2),
                ))

            # Individual components
            for i, (comp, curve) in enumerate(zip(deconv.components, deconv.component_curves)):
                curve_arr = np.array(curve)
                comp_color = comp_defaults[i % len(comp_defaults)]
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
                title=(
                    f"XPS Deconvolution \u2014 {deconv_element} ({selected_xps_sample}, "
                    f"{deconv.n_components} components, R\u00b2={deconv.envelope_r_squared:.3f})"
                ),
                xaxis_title="Binding Energy (eV)",
                yaxis_title="Intensity (CPS)",
                xaxis=dict(autorange="reversed"),
                height=550, template="plotly_dark",
            )
            st.plotly_chart(fig_deconv, use_container_width=True)

            # Export fitted data
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
                label=f"Export {deconv_element} fitted data as CSV",
                data=csv_data,
                file_name=f"xps_{selected_xps_sample}_{deconv_element.replace(' ', '_')}_deconvolution.csv",
                mime="text/csv",
            )

            # Component quantification
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
                    use_container_width=True,
                    hide_index=True,
                )

            # Chemical state interpretation
            st.markdown("### Chemical State Interpretation")
            if deconv_element in XPS_REFERENCES:
                ref = XPS_REFERENCES[deconv_element]
                for comp_ref in ref["components"]:
                    matched = next(
                        (c for c in deconv.components if c.assignment == comp_ref["name"]),
                        None,
                    )
                    if matched:
                        rel_pct = next(
                            (q["relative_pct"] for q in xps_result["quantification"]
                             if q["component"] == comp_ref["name"]),
                            0,
                        )
                        doi = comp_ref.get("doi", "")
                        ref_text = comp_ref.get("reference", "")
                        ref_link = f" \u2014 [{ref_text}]({doi})" if doi else ""
                        st.markdown(
                            f"**{comp_ref['name']}** ({matched.center_ev:.1f} eV, "
                            f"{rel_pct:.1f}%): {comp_ref['description']}{ref_link}"
                        )
        else:
            st.warning("Deconvolution did not converge. Try adjusting parameters.")
