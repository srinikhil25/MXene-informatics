"""
Cross-Technique Correlation Visualization
==========================================
Generates publication-quality figures for the Materials Informatics paper:
1. Cross-technique correlation heatmap
2. Multi-technique family comparison radar chart
3. Feature importance across material families
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_correlation_heatmap(corr_csv: str, output_dir: str = None):
    """
    Generate cross-technique correlation heatmap.
    This is the KEY figure for the paper.
    """
    corr = pd.read_csv(corr_csv, index_col=0)

    # Clean column names for display
    def clean_name(n):
        n = n.replace("xrd_", "XRD: ").replace("edx_", "EDX: ").replace("sem_", "SEM: ")
        n = n.replace("_", " ").replace("at pct", "at%")
        return n

    labels = [clean_name(c) for c in corr.columns]

    # Color-code by technique
    colors = []
    for c in corr.columns:
        if c.startswith("xrd_"):
            colors.append("rgba(59,130,246,0.3)")   # blue
        elif c.startswith("edx_"):
            colors.append("rgba(16,185,129,0.3)")   # green
        elif c.startswith("sem_"):
            colors.append("rgba(245,158,11,0.3)")   # amber
        else:
            colors.append("rgba(148,163,184,0.3)")

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Cross-Technique Feature Correlation Matrix<br><sub>4 material families (MXene, CF, CAF, Other) | XRD + EDX + SEM</sub>",
        width=1200, height=1000,
        template="plotly_dark",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )

    if output_dir:
        out = Path(output_dir)
        fig.write_html(str(out / "correlation_heatmap.html"))
        try:
            fig.write_image(str(out / "correlation_heatmap.png"), scale=2)
        except Exception:
            pass  # kaleido not installed

    return fig


def plot_family_comparison(family_csv: str, output_dir: str = None):
    """
    Radar/spider chart comparing material families across key features.
    """
    df = pd.read_csv(family_csv)
    multi = df[df["n_techniques"] >= 2].copy()

    # Select key features for comparison
    features = [
        ("xrd_n_peaks", "XRD Peak Count"),
        ("xrd_crystallite_size_nm", "Crystallite Size (nm)"),
        ("xrd_peak_density", "Peak Density"),
        ("xrd_strongest_peak_d_spacing", "d-spacing (A)"),
        ("sem_magnification", "SEM Magnification"),
        ("sem_pixel_size_nm", "Pixel Size (nm)"),
    ]

    # Normalize to 0-1 range for radar
    fig = go.Figure()
    theta_labels = [f[1] for f in features]

    for _, row in multi.iterrows():
        vals = []
        for col, _ in features:
            v = row.get(col, np.nan)
            if pd.isna(v):
                vals.append(0)
            else:
                # Normalize within column
                col_vals = multi[col].dropna()
                if col_vals.max() > col_vals.min():
                    vals.append((v - col_vals.min()) / (col_vals.max() - col_vals.min()))
                else:
                    vals.append(0.5)

        vals.append(vals[0])  # close the polygon
        labels = theta_labels + [theta_labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels,
            fill="toself",
            name=row["family"],
            opacity=0.5,
        ))

    fig.update_layout(
        title="Material Family Characterization Profiles<br><sub>Normalized features across XRD + SEM</sub>",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        width=800, height=600,
    )

    if output_dir:
        out = Path(output_dir)
        fig.write_html(str(out / "family_radar_chart.html"))

    return fig


def plot_family_bar_comparison(family_csv: str, output_dir: str = None):
    """
    Grouped bar chart comparing key features across material families.
    Better than radar for publication.
    """
    df = pd.read_csv(family_csv)
    multi = df[df["n_techniques"] >= 2].copy()

    # Create subplots for key features
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "XRD Peak Count", "Crystallite Size (nm)", "Peak Density (peaks/deg)",
            "d-spacing (A)", "SEM Magnification", "Pixel Size (nm)"
        ],
        vertical_spacing=0.15,
    )

    features = [
        ("xrd_n_peaks", 1, 1), ("xrd_crystallite_size_nm", 1, 2),
        ("xrd_peak_density", 1, 3), ("xrd_strongest_peak_d_spacing", 2, 1),
        ("sem_magnification", 2, 2), ("sem_pixel_size_nm", 2, 3),
    ]

    family_colors = {
        "CAF_Carbon_Fabric": "#06b6d4",
        "CF_Conductive_Fabric": "#8b5cf6",
        "MXene_Ti3C2": "#ef4444",
        "Other": "#94a3b8",
    }

    for col, row_n, col_n in features:
        for _, row in multi.iterrows():
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0
            color = family_colors.get(row["family"], "#888888")
            fig.add_trace(
                go.Bar(
                    x=[row["family"].replace("_", " ")],
                    y=[val],
                    marker_color=color,
                    name=row["family"],
                    showlegend=(row_n == 1 and col_n == 1),
                ),
                row=row_n, col=col_n,
            )

    fig.update_layout(
        title="Multi-Technique Feature Comparison Across Material Families",
        template="plotly_dark",
        width=1200, height=700,
        showlegend=True,
        barmode="group",
    )

    if output_dir:
        out = Path(output_dir)
        fig.write_html(str(out / "family_bar_comparison.html"))

    return fig


def generate_all_figures(features_dir: str = None):
    """Generate all paper figures."""
    if features_dir is None:
        features_dir = "D:/MXene-Informatics/data/processed/features"

    out = Path(features_dir) / "figures"
    out.mkdir(exist_ok=True)

    print("Generating correlation heatmap...")
    fig1 = plot_correlation_heatmap(
        str(Path(features_dir) / "cross_technique_correlation.csv"),
        str(out),
    )

    print("Generating family radar chart...")
    fig2 = plot_family_comparison(
        str(Path(features_dir) / "family_feature_matrix.csv"),
        str(out),
    )

    print("Generating family bar comparison...")
    fig3 = plot_family_bar_comparison(
        str(Path(features_dir) / "family_feature_matrix.csv"),
        str(out),
    )

    print(f"All figures saved to: {out}")
    return fig1, fig2, fig3


if __name__ == "__main__":
    generate_all_figures()
