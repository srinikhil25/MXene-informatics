# -*- coding: utf-8 -*-
"""
UV-DRS Analysis Page
=====================
Diffuse reflectance spectra, Kubelka-Munk transform, Tauc bandgap estimation.

Theory:
    Kubelka-Munk: F(R) = (1-R)^2 / (2R)
    Tauc plot:    (F(R) * hv)^n vs hv
        n = 2   for direct allowed transition
        n = 1/2 for indirect allowed transition
    Bandgap is estimated from the x-intercept of the linear region.
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

uv_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("UV-DRS")}
if not uv_samples:
    st.info("No UV-DRS data available in the current project.")
    st.stop()


def _load_uvdrs(sample_id: str):
    """Load UV-DRS data. Returns (wavelength_nm, reflectance_fraction)."""
    td = project.samples[sample_id].techniques["UV-DRS"]
    p = td.parsed
    wl = np.array(p.get("wavelength_nm", []))
    r_pct = np.array(p.get("reflectance_pct", []))
    return wl, r_pct / 100.0  # convert % to fraction


def kubelka_munk(R):
    """Kubelka-Munk function: F(R) = (1-R)^2 / (2R). Clips R to avoid div-by-zero."""
    R = np.clip(R, 1e-6, 1.0 - 1e-6)
    return (1 - R) ** 2 / (2 * R)


def tauc_transform(wavelength_nm, F_R, n=2):
    """
    Compute Tauc plot data: (F(R) * hv)^n vs hv.
    hv = 1240 / wavelength_nm (eV).
    Returns (hv, tauc_y).
    """
    hv = 1240.0 / wavelength_nm
    tauc_y = (F_R * hv) ** n
    return hv, tauc_y


def estimate_bandgap(hv, tauc_y, fit_range=(1.0, 3.0)):
    """
    Estimate bandgap from Tauc plot by fitting a line to the steepest region.
    Returns (bandgap_eV, slope, intercept, fit_mask).
    """
    mask = (hv >= fit_range[0]) & (hv <= fit_range[1]) & np.isfinite(tauc_y)
    hv_fit = hv[mask]
    ty_fit = tauc_y[mask]

    if len(hv_fit) < 10:
        return None, None, None, mask

    # Find the steepest region using a sliding window derivative
    window = max(5, len(ty_fit) // 20)
    gradient = np.gradient(ty_fit, hv_fit)

    # Find the peak of the gradient (steepest rise)
    smooth_grad = np.convolve(gradient, np.ones(window) / window, mode="same")
    peak_idx = np.argmax(smooth_grad)

    # Fit a line around the steepest region (+/- window)
    fit_start = max(0, peak_idx - window * 2)
    fit_end = min(len(hv_fit), peak_idx + window * 2)
    if fit_end - fit_start < 5:
        return None, None, None, mask

    hv_linear = hv_fit[fit_start:fit_end]
    ty_linear = ty_fit[fit_start:fit_end]

    # Linear regression
    coeffs = np.polyfit(hv_linear, ty_linear, 1)
    slope, intercept = coeffs

    if slope <= 0:
        return None, None, None, mask

    # x-intercept = -intercept / slope
    bandgap = -intercept / slope

    # Build mask for the fitted region in original arrays
    fit_hv_min = hv_linear.min()
    fit_hv_max = hv_linear.max()
    linear_mask = (hv >= fit_hv_min) & (hv <= fit_hv_max)

    return bandgap, slope, intercept, linear_mask


# ── Header ──
st.markdown("## UV-DRS Analysis")
st.markdown(
    "Diffuse reflectance spectroscopy with **Kubelka-Munk** transform "
    "and **Tauc plot** for optical bandgap estimation."
)

# ── Sidebar ──
st.sidebar.markdown("### UV-DRS Settings")

all_uv_ids = sorted(uv_samples.keys())
selected_samples = st.sidebar.multiselect(
    "Select Samples (max 6)",
    all_uv_ids,
    default=all_uv_ids[:min(4, len(all_uv_ids))],
    max_selections=6,
    key="uv_sample_select",
)

transition_type = st.sidebar.radio(
    "Transition Type",
    ["Direct allowed (n=2)", "Indirect allowed (n=1/2)"],
    key="uv_transition",
)
n_exponent = 2.0 if "Direct" in transition_type else 0.5

wl_range = st.sidebar.slider(
    "Wavelength Range (nm)", 200, 1600, (200, 900), step=10, key="uv_wl_range"
)
tauc_range = st.sidebar.slider(
    "Tauc Fit Range (eV)", 0.5, 6.0, (1.0, 3.5), step=0.1, key="uv_tauc_range"
)

# ── Metrics ──
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Samples with UV-DRS", len(uv_samples))
mc2.metric("Selected", len(selected_samples))
mc3.metric("Transition", "Direct" if n_exponent == 2 else "Indirect")

st.markdown("---")

if not selected_samples:
    st.info("Select at least one sample from the sidebar.")
    st.stop()

palette = ["#06b6d4", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#ec4899"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. Reflectance Spectra
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### Diffuse Reflectance Spectra")

fig_refl = go.Figure()
for i, sid in enumerate(selected_samples):
    wl, R = _load_uvdrs(sid)
    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    color = palette[i % len(palette)]
    fig_refl.add_trace(go.Scatter(
        x=wl[mask], y=R[mask] * 100,
        name=sid,
        line=dict(color=color, width=1.5),
        hovertemplate=f"<b>{sid}</b><br>\u03bb = %{{x:.0f}} nm<br>R = %{{y:.1f}}%<extra></extra>",
    ))

fig_refl.update_layout(
    xaxis_title="Wavelength (nm)",
    yaxis_title="Reflectance (%)",
    height=450,
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(x=0.7, y=0.95),
)
st.plotly_chart(fig_refl, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Kubelka-Munk Transform
# ═══════════════════════════════════════════════════════════════════════════
km_col1, km_col2 = st.columns([3, 1])
with km_col1:
    st.markdown("### Kubelka-Munk Transform")
    st.caption("F(R) = (1\u2212R)\u00b2 / (2R)")
with km_col2:
    km_xaxis = st.radio(
        "X-axis", ["Wavelength (nm)", "Energy (eV)"],
        horizontal=True, key="km_xaxis",
    )

use_energy = km_xaxis == "Energy (eV)"
fig_km = go.Figure()
for i, sid in enumerate(selected_samples):
    wl, R = _load_uvdrs(sid)
    F_R = kubelka_munk(R)
    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    color = palette[i % len(palette)]
    if use_energy:
        x_vals = 1240.0 / wl[mask]
        hover = f"<b>{sid}</b><br>h\u03bd = %{{x:.3f}} eV<br>F(R) = %{{y:.3f}}<extra></extra>"
    else:
        x_vals = wl[mask]
        hover = f"<b>{sid}</b><br>\u03bb = %{{x:.0f}} nm<br>F(R) = %{{y:.3f}}<extra></extra>"
    fig_km.add_trace(go.Scatter(
        x=x_vals, y=F_R[mask],
        name=sid,
        line=dict(color=color, width=1.5),
        hovertemplate=hover,
    ))

fig_km.update_layout(
    xaxis_title="Photon Energy (eV)" if use_energy else "Wavelength (nm)",
    yaxis_title="F(R)",
    height=450,
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(x=0.7, y=0.95),
)
st.plotly_chart(fig_km, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Tauc Plot + Bandgap Estimation
# ═══════════════════════════════════════════════════════════════════════════
n_label = "2" if n_exponent == 2 else "1/2"
st.markdown(f"### Tauc Plot \u2014 (F(R)\u00b7h\u03bd)^{n_label} vs h\u03bd")

fig_tauc = go.Figure()
bandgap_results = []

for i, sid in enumerate(selected_samples):
    wl, R = _load_uvdrs(sid)
    F_R = kubelka_munk(R)
    hv, tauc_y = tauc_transform(wl, F_R, n=n_exponent)

    # Sort by hv ascending
    sort_idx = np.argsort(hv)
    hv = hv[sort_idx]
    tauc_y = tauc_y[sort_idx]

    color = palette[i % len(palette)]

    # Plot Tauc data
    plot_mask = (hv >= tauc_range[0] - 0.5) & (hv <= tauc_range[1] + 0.5) & np.isfinite(tauc_y)
    fig_tauc.add_trace(go.Scatter(
        x=hv[plot_mask], y=tauc_y[plot_mask],
        name=sid,
        line=dict(color=color, width=1.5),
        hovertemplate=f"<b>{sid}</b><br>h\u03bd = %{{x:.2f}} eV<br>Tauc = %{{y:.2f}}<extra></extra>",
    ))

    # Estimate bandgap
    bg, slope, intercept, lin_mask = estimate_bandgap(hv, tauc_y, fit_range=tauc_range)
    if bg is not None and 0.5 < bg < 5.0:
        bandgap_results.append({"Sample": sid, "Bandgap (eV)": round(bg, 3), "Transition": transition_type.split(" (")[0]})

        # Plot linear fit extrapolation
        hv_line = np.linspace(bg - 0.2, tauc_range[1], 100)
        y_line = slope * hv_line + intercept
        y_line = np.clip(y_line, 0, None)
        fig_tauc.add_trace(go.Scatter(
            x=hv_line, y=y_line,
            name=f"{sid} fit (Eg={bg:.2f} eV)",
            line=dict(color=color, width=1, dash="dash"),
            showlegend=True,
        ))

        # Mark the bandgap point on x-axis
        fig_tauc.add_trace(go.Scatter(
            x=[bg], y=[0],
            mode="markers",
            marker=dict(symbol="x", size=12, color=color, line=dict(width=2)),
            name=f"{sid}: {bg:.2f} eV",
            showlegend=False,
        ))
    else:
        bandgap_results.append({"Sample": sid, "Bandgap (eV)": "N/A", "Transition": transition_type.split(" (")[0]})

fig_tauc.update_layout(
    xaxis_title="Photon Energy h\u03bd (eV)",
    yaxis_title=f"(F(R)\u00b7h\u03bd)^{n_label}",
    height=550,
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(x=0.02, y=0.98, font=dict(size=10)),
)
st.plotly_chart(fig_tauc, use_container_width=True)

# ── Bandgap Results Table ──
if bandgap_results:
    st.markdown("### Bandgap Estimation Results")
    bg_df = pd.DataFrame(bandgap_results)
    st.dataframe(bg_df, use_container_width=True, hide_index=True)

    # Quick interpretation
    numeric_bgs = [r["Bandgap (eV)"] for r in bandgap_results if isinstance(r["Bandgap (eV)"], float)]
    if numeric_bgs:
        avg_bg = np.mean(numeric_bgs)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1e293b,#0f172a);'
            f'border-left:4px solid #06b6d4;padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0;">'
            f'Average bandgap: <strong>{avg_bg:.2f} eV</strong> ({1240/avg_bg:.0f} nm) '
            f'&mdash; {"UV" if avg_bg > 3.1 else "visible" if avg_bg > 1.65 else "IR"} region'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Export ──
with st.expander("Export Data"):
    export_sample = st.selectbox("Sample to export", selected_samples, key="uv_export_sample")
    wl, R = _load_uvdrs(export_sample)
    F_R = kubelka_munk(R)
    hv, tauc_y = tauc_transform(wl, F_R, n=n_exponent)

    export_df = pd.DataFrame({
        "wavelength_nm": wl,
        "reflectance_pct": R * 100,
        "F_R": F_R,
        "photon_energy_eV": 1240.0 / wl,
        f"tauc_n{n_label}": tauc_y,
    })
    csv = export_df.to_csv(index=False)
    st.download_button(
        f"Export {export_sample} UV-DRS data as CSV",
        data=csv,
        file_name=f"uvdrs_{export_sample}.csv",
        mime="text/csv",
    )
