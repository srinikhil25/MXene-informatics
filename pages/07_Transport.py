# -*- coding: utf-8 -*-
"""
Transport Properties Page
==========================
Hall measurement (room-temperature) + Thermoelectric properties (T-dependent).
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

hall_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("Hall")}
te_samples = {sid: s for sid, s in project.samples.items() if s.has_technique("Thermoelectric")}

if not hall_samples and not te_samples:
    st.info("No transport data available in the current project.")
    st.stop()

palette = ["#06b6d4", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#ec4899"]

# ── Header ──
st.markdown("## Transport Properties")
st.markdown("Hall effect measurements and thermoelectric properties.")

# ── Metrics ──
mc1, mc2 = st.columns(2)
mc1.metric("Samples with Hall", len(hall_samples))
mc2.metric("Samples with Thermoelectric", len(te_samples))

# ═══════════════════════════════════════════════════════════════════════════
# Hall Measurement
# ═══════════════════════════════════════════════════════════════════════════
if hall_samples:
    st.markdown("---")
    st.markdown("### Hall Effect Measurements")
    st.caption("Room-temperature carrier concentration, mobility, and resistivity")

    # Build comparison table
    hall_rows = []
    for sid in sorted(hall_samples.keys()):
        p = project.samples[sid].techniques["Hall"].parsed
        T = p.get("temperature_C", [None])
        rho = p.get("resistivity_ohm_cm", [None])
        sigma = p.get("conductivity_1_ohm_cm", [None])
        n = p.get("carrier_concentration_cm3", [None])
        mu = p.get("mobility_cm2_Vs", [None])
        R_H = p.get("hall_coefficient_cm3_C", [None])
        R_s = p.get("sheet_resistance_ohm", [None])

        # Determine carrier type from Hall coefficient sign
        carrier_type = "unknown"
        if R_H and R_H[0] is not None:
            carrier_type = "p-type" if R_H[0] > 0 else "n-type"

        hall_rows.append({
            "Sample": sid,
            "T (\u00b0C)": f"{T[0]:.1f}" if T[0] is not None else "\u2014",
            "Resistivity (\u03a9\u00b7cm)": f"{rho[0]:.6f}" if rho[0] is not None else "\u2014",
            "Conductivity (S/cm)": f"{sigma[0]:.1f}" if sigma[0] is not None else "\u2014",
            "Carrier Conc. (cm\u207b\u00b3)": f"{abs(n[0]):.2e}" if n[0] is not None else "\u2014",
            "Mobility (cm\u00b2/V\u00b7s)": f"{abs(mu[0]):.2f}" if mu[0] is not None else "\u2014",
            "Hall Coeff. (cm\u00b3/C)": f"{R_H[0]:.4f}" if R_H[0] is not None else "\u2014",
            "Carrier Type": carrier_type,
        })

    hall_df = pd.DataFrame(hall_rows)

    def _color_carrier(val):
        if val == "p-type":
            return "color: #ef4444; font-weight: bold"
        elif val == "n-type":
            return "color: #3b82f6; font-weight: bold"
        return ""

    st.dataframe(
        hall_df.style.map(_color_carrier, subset=["Carrier Type"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Bar chart comparison ──
    if len(hall_rows) > 1:
        fig_hall = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Resistivity (\u03a9\u00b7cm)", "Mobility (cm\u00b2/V\u00b7s)", "Carrier Conc. (cm\u207b\u00b3)"],
        )

        sids = [r["Sample"] for r in hall_rows]
        colors = [palette[i % len(palette)] for i in range(len(sids))]

        rho_vals = []
        mu_vals = []
        n_vals = []
        for sid in sids:
            p = project.samples[sid].techniques["Hall"].parsed
            rho_vals.append(abs(p.get("resistivity_ohm_cm", [0])[0] or 0))
            mu_vals.append(abs(p.get("mobility_cm2_Vs", [0])[0] or 0))
            n_vals.append(abs(p.get("carrier_concentration_cm3", [0])[0] or 0))

        fig_hall.add_trace(go.Bar(x=sids, y=rho_vals, marker_color=colors, showlegend=False), row=1, col=1)
        fig_hall.add_trace(go.Bar(x=sids, y=mu_vals, marker_color=colors, showlegend=False), row=1, col=2)
        fig_hall.add_trace(go.Bar(x=sids, y=n_vals, marker_color=colors, showlegend=False), row=1, col=3)

        fig_hall.update_yaxes(type="log", row=1, col=1)
        fig_hall.update_yaxes(type="log", row=1, col=2)
        fig_hall.update_yaxes(type="log", row=1, col=3)
        fig_hall.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_hall, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Thermoelectric Properties
# ═══════════════════════════════════════════════════════════════════════════
if te_samples:
    st.markdown("---")
    st.markdown("### Thermoelectric Properties")
    st.caption("Temperature-dependent Seebeck coefficient, thermal conductivity, resistivity, and ZT")

    # Sidebar controls
    st.sidebar.markdown("### TE Settings")
    te_sample_ids = sorted(te_samples.keys())
    selected_te = st.sidebar.multiselect(
        "Select Samples",
        te_sample_ids,
        default=te_sample_ids[:min(4, len(te_sample_ids))],
        key="te_samples",
    )

    if not selected_te:
        st.info("Select at least one sample from the sidebar.")
    else:
        # ── 2x2 grid of T-dependent plots ──
        fig_te = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Seebeck Coefficient (\u03bcV/K)",
                "Thermal Conductivity (W/m\u00b7K)",
                "Resistivity (\u03a9\u00b7m)",
                "ZT (Figure of Merit)",
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.12,
        )

        for i, sid in enumerate(selected_te):
            p = project.samples[sid].techniques["Thermoelectric"].parsed
            T = np.array(p.get("temperature_K", []))
            color = palette[i % len(palette)]

            # Sort by temperature
            sort_idx = np.argsort(T)
            T = T[sort_idx]

            props = [
                ("seebeck_uV_K", 1, 1),
                ("thermal_conductivity_W_mK", 1, 2),
                ("resistivity_ohm_m", 2, 1),
                ("zT", 2, 2),
            ]

            for key, row, col in props:
                vals = np.array(p.get(key, []))
                if len(vals) == 0:
                    continue
                vals = vals[sort_idx]

                fig_te.add_trace(go.Scatter(
                    x=T, y=vals,
                    name=sid if (row == 1 and col == 1) else None,
                    showlegend=(row == 1 and col == 1),
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                    marker=dict(size=5),
                    hovertemplate=f"<b>{sid}</b><br>T = %{{x:.0f}} K<br>%{{y:.4g}}<extra></extra>",
                ), row=row, col=col)

        fig_te.update_xaxes(title_text="Temperature (K)", row=2, col=1)
        fig_te.update_xaxes(title_text="Temperature (K)", row=2, col=2)
        fig_te.update_layout(
            height=700,
            template="plotly_dark",
            legend=dict(x=0.02, y=0.98, font=dict(size=11)),
        )
        st.plotly_chart(fig_te, use_container_width=True)

        # ── Power Factor plot ──
        st.markdown("### Power Factor")
        fig_pf = go.Figure()
        for i, sid in enumerate(selected_te):
            p = project.samples[sid].techniques["Thermoelectric"].parsed
            T = np.array(p.get("temperature_K", []))
            pf = np.array(p.get("power_factor", []))
            if len(T) == 0 or len(pf) == 0:
                continue
            sort_idx = np.argsort(T)
            color = palette[i % len(palette)]
            fig_pf.add_trace(go.Scatter(
                x=T[sort_idx], y=pf[sort_idx],
                name=sid,
                line=dict(color=color, width=2),
                mode="lines+markers",
                marker=dict(size=5),
            ))

        fig_pf.update_layout(
            xaxis_title="Temperature (K)",
            yaxis_title="Power Factor (W/m\u00b7K\u00b2)",
            height=400,
            template="plotly_dark",
        )
        st.plotly_chart(fig_pf, use_container_width=True)

        # ── Peak ZT Summary ──
        st.markdown("### Peak ZT Summary")
        zt_rows = []
        for sid in selected_te:
            p = project.samples[sid].techniques["Thermoelectric"].parsed
            T = np.array(p.get("temperature_K", []))
            zT = np.array(p.get("zT", []))
            seebeck = np.array(p.get("seebeck_uV_K", []))
            kappa = np.array(p.get("thermal_conductivity_W_mK", []))

            if len(zT) > 0:
                peak_idx = np.argmax(zT)
                zt_rows.append({
                    "Sample": sid,
                    "Peak ZT": f"{zT[peak_idx]:.4f}",
                    "at T (K)": f"{T[peak_idx]:.0f}",
                    "Seebeck (\u03bcV/K)": f"{seebeck[peak_idx]:.1f}" if len(seebeck) > peak_idx else "\u2014",
                    "\u03ba (W/m\u00b7K)": f"{kappa[peak_idx]:.3f}" if len(kappa) > peak_idx else "\u2014",
                })

        if zt_rows:
            st.dataframe(pd.DataFrame(zt_rows), use_container_width=True, hide_index=True)

        # ── Export ──
        with st.expander("Export Thermoelectric Data"):
            export_sid = st.selectbox("Sample", selected_te, key="te_export")
            p = project.samples[export_sid].techniques["Thermoelectric"].parsed
            export_df = pd.DataFrame({
                "temperature_K": p.get("temperature_K", []),
                "seebeck_uV_K": p.get("seebeck_uV_K", []),
                "resistivity_ohm_m": p.get("resistivity_ohm_m", []),
                "thermal_conductivity_W_mK": p.get("thermal_conductivity_W_mK", []),
                "power_factor": p.get("power_factor", []),
                "zT": p.get("zT", []),
            })
            csv = export_df.to_csv(index=False)
            st.download_button(
                f"Export {export_sid} TE data as CSV",
                data=csv,
                file_name=f"thermoelectric_{export_sid}.csv",
                mime="text/csv",
            )
