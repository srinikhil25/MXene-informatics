# -*- coding: utf-8 -*-
"""
Overview Page
==============
Project loading via folder path, sample-technique matrix, file intelligence report.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.project_builder import build_project
from src.models import KNOWN_TECHNIQUES


# ── Header ──
st.markdown('<h1 class="main-header">Materials Informatics</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">From Raw Spectra to Cross-Technique Insight: '
    'An Autonomous Informatics Platform for Multi-Modal Materials Characterization</p>',
    unsafe_allow_html=True,
)

project = st.session_state.get("project")

# ═══════════════════════════════════════════════════════════════════════════
# STATE A: No project loaded — landing page
# ═══════════════════════════════════════════════════════════════════════════
if project is None:
    st.markdown("---")

    # Hero message
    st.markdown(
        '<div style="text-align:center;padding:30px 0 10px;">'
        '<div style="font-size:2.5rem;margin-bottom:8px;">Welcome</div>'
        '<div style="color:#94a3b8;font-size:1.05rem;max-width:600px;margin:0 auto;">'
        'Point to a folder of raw instrument data and this platform will automatically '
        'detect techniques, identify samples, parse files, and unlock analysis pages.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Three action cards ──
    st.markdown("### Get Started")
    gc1, gc2, gc3 = st.columns(3)

    with gc1:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#06b6d422,#06b6d411);'
            'border:1px solid #06b6d444;border-radius:12px;padding:20px;min-height:180px;">'
            '<div style="color:#06b6d4;font-weight:700;font-size:1rem;margin-bottom:8px;">'
            'Step 1 &mdash; Point to Data</div>'
            '<div style="color:#94a3b8;font-size:0.85rem;">'
            'Enter the path to a folder containing your raw instrument files. '
            'Any folder structure is fine &mdash; the platform adapts to your organization.'
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
            'The platform scans every file, detects the characterization technique, '
            'identifies which sample each file belongs to, and parses the data automatically.'
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
            'Analysis pages unlock based on your data: '
            'XRD, XPS, UV-DRS, Microscopy, EDS, Transport properties, and more.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("---")

    # ── Upload Zone ──
    st.markdown("### Load Your Data")
    folder_col1, folder_col2 = st.columns([4, 1])
    with folder_col1:
        folder_path = st.text_input(
            "Enter folder path containing raw instrument data:",
            placeholder="e.g., D:/MyData/Sample_A/",
            key="folder_path_input",
        )
    with folder_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_clicked = st.button("Scan & Build Project", type="primary", key="scan_folder")

    if scan_clicked and folder_path and folder_path.strip():
        folder = Path(folder_path.strip())
        if not folder.exists():
            st.error(f"Folder not found: `{folder_path}`")
        elif not folder.is_dir():
            st.error(f"Not a directory: `{folder_path}`")
        else:
            progress_bar = st.progress(0, text="Scanning files...")

            def _progress(msg, frac):
                progress_bar.progress(min(frac, 1.0), text=msg)

            try:
                proj = build_project(folder, progress_callback=_progress)
                st.session_state["project"] = proj
                progress_bar.progress(100, text="Done!")
                st.rerun()
            except Exception as e:
                st.error(f"Error building project: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# STATE B: Project loaded — dashboard
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.caption(f"Project: **{project.name}** | Source: `{project.root_path}`")

    # ── Key metrics ──
    n_samples = len(project.samples)
    n_files = project.manifest.total_files
    n_unassigned = len(project.unassigned)

    # Count techniques across all samples
    all_techs = set()
    for s in project.samples.values():
        all_techs.update(s.available_techniques)
    n_techniques = len(all_techs)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Samples", n_samples)
    mc2.metric("Techniques", n_techniques)
    mc3.metric("Files Parsed", n_files - n_unassigned)
    mc4.metric("Total Files", n_files)

    st.markdown("---")

    # ── Sample-Technique Matrix ──
    st.markdown("### Sample-Technique Matrix")
    matrix = project.technique_matrix
    if matrix:
        # Build DataFrame
        techs_sorted = sorted(all_techs)
        rows = []
        for sid in project.sample_ids:
            row = {"Sample": sid}
            sample = project.samples[sid]
            for t in techs_sorted:
                has = sample.has_technique(t)
                row[t] = has
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Sample")

        # Color-coded display
        TECH_COLORS = {
            "XRD": "#06b6d4", "XPS": "#8b5cf6", "UV-DRS": "#f59e0b",
            "Raman": "#10b981", "FTIR": "#64748b", "Hall": "#ec4899",
            "Thermoelectric": "#ef4444", "TEM": "#f97316", "SEM": "#84cc16",
            "EDS": "#a855f7", "STEM": "#f97316", "PL": "#6366f1",
        }

        def _style_bool(val):
            if val:
                return "background-color: #06b6d433; color: #06b6d4; font-weight: 700;"
            return "color: #475569;"

        display_df = df.map(lambda v: "Y" if v else "-")
        styled = display_df.style.map(
            lambda v: _style_bool(v == "Y")
        )
        st.dataframe(styled, use_container_width=True)

        # Per-sample detail counts
        st.markdown("### Per-Sample Details")
        for sid in project.sample_ids:
            sample = project.samples[sid]
            techs = sample.available_techniques
            aliases = sample.aliases
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""

            with st.expander(f"**{sid}**{alias_str} — {len(techs)} techniques"):
                for t in techs:
                    td = sample.techniques[t]
                    n_files_t = len(td.files)
                    parsed_info = ""
                    p = td.parsed
                    if t == "XRD" and "patterns" in p:
                        n_pat = len(p["patterns"])
                        parsed_info = f"{n_pat} pattern(s)"
                    elif t == "XPS":
                        if "regions" in p:
                            parsed_info = f"{len(p['regions'])} region(s)"
                        elif "region" in p:
                            parsed_info = f"region: {p['region']}"
                    elif t == "UV-DRS":
                        n_pts = len(p.get("wavelength_nm", []))
                        parsed_info = f"{n_pts} wavelength points"
                    elif t == "Hall":
                        n_pts = len(p.get("temperature_C", []))
                        parsed_info = f"{n_pts} measurement(s)"
                    elif t == "Thermoelectric":
                        n_pts = len(p.get("temperature_K", []))
                        parsed_info = f"{n_pts} temperature points"
                    elif t == "EDS" and "spectra" in p:
                        parsed_info = f"{len(p['spectra'])} spectrum/spectra"
                    elif t == "TEM":
                        parsed_info = f"{n_files_t} image files"

                    st.markdown(f"- **{t}**: {n_files_t} file(s)" + (f" | {parsed_info}" if parsed_info else ""))

    st.markdown("---")

    # ── Available Analysis Pages ──
    st.markdown("### Available Analysis")
    st.caption("Navigate to technique pages in the sidebar:")

    _cards = []
    _card_defs = [
        ("XRD", "#06b6d4", "Phase identification, Scherrer crystallite size, pattern comparison"),
        ("XPS", "#8b5cf6", "Survey + high-resolution spectra, peak deconvolution, quantification"),
        ("UV-DRS", "#f59e0b", "Diffuse reflectance, Kubelka-Munk transform, Tauc bandgap estimation"),
        ("TEM", "#f97316", "Image gallery, SAED patterns, HRTEM"),
        ("EDS", "#a855f7", "Elemental spectra, quantification tables"),
        ("Hall", "#ec4899", "Carrier concentration, mobility, resistivity"),
        ("Thermoelectric", "#ef4444", "Seebeck, thermal conductivity, ZT figure of merit"),
    ]

    for tech, color, desc in _card_defs:
        # Check if any sample has this technique
        has = any(s.has_technique(tech) for s in project.samples.values())
        if has:
            n = sum(1 for s in project.samples.values() if s.has_technique(tech))
            _cards.append((tech, color, f"<strong>{n}</strong> sample(s)", desc))

    if _cards:
        n_cols = min(3, len(_cards))
        for row_start in range(0, len(_cards), n_cols):
            row_items = _cards[row_start:row_start + n_cols]
            cols = st.columns(n_cols)
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

    # ── File Intelligence Report ──
    with st.expander("File Intelligence Report", expanded=False):
        manifest = project.manifest
        fi_cols = st.columns(4)
        fi_cols[0].metric("Total Files", manifest.total_files)
        fi_cols[1].metric("Assigned", manifest.total_files - len(project.unassigned))
        fi_cols[2].metric("Unassigned", len(project.unassigned))
        fi_cols[3].metric("Skipped", manifest.skipped)

        st.markdown("**By Technique:**")
        by_tech = manifest.by_technique
        fi_rows = []
        for tech, entries in sorted(by_tech.items()):
            formats = set(e.file_type for e in entries)
            fi_rows.append({
                "Technique": tech,
                "Files": len(entries),
                "Formats": ", ".join(sorted(formats)),
            })
        if fi_rows:
            st.dataframe(pd.DataFrame(fi_rows), use_container_width=True, hide_index=True)

        if project.unassigned:
            st.markdown("**Unassigned Files:**")
            for f in project.unassigned:
                rel = f.path.relative_to(project.root_path) if project.root_path else f.path
                st.markdown(f"- `{rel}` ({f.technique})")

    # ── Manage project ──
    with st.expander("Manage Project"):
        if st.button("Clear Project & Start Fresh", key="reset_project"):
            st.session_state["project"] = None
            st.rerun()

        st.markdown("---")
        st.markdown("**Load a different folder:**")
        new_folder = st.text_input(
            "Folder path:",
            placeholder="e.g., D:/OtherData/",
            key="new_folder_input",
        )
        if st.button("Load New Folder", key="load_new"):
            if new_folder and new_folder.strip():
                folder = Path(new_folder.strip())
                if folder.exists() and folder.is_dir():
                    progress_bar = st.progress(0, text="Scanning...")

                    def _progress(pct, msg):
                        progress_bar.progress(pct / 100, text=msg)

                    try:
                        proj = build_project(folder, progress_callback=_progress)
                        st.session_state["project"] = proj
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Invalid folder path.")
