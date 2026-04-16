# -*- coding: utf-8 -*-
"""
Microscopy Page
================
TEM/SEM/STEM image gallery with sample filtering and image categorization.
"""

import streamlit as st
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Guard ──
project = st.session_state.get("project")
if project is None:
    st.warning("No project loaded. Go to Overview to load data.")
    st.stop()

micro_samples = {
    sid: s for sid, s in project.samples.items()
    if s.has_technique("TEM") or s.has_technique("SEM") or s.has_technique("STEM")
}
if not micro_samples:
    st.info("No microscopy data available in the current project.")
    st.stop()

# Image extensions we can display
_IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".png"}

# ── Header ──
st.markdown("## Microscopy - TEM / SEM / STEM")

# ── Sidebar ──
st.sidebar.markdown("### Microscopy Settings")

sample_ids = sorted(micro_samples.keys())
selected_sample = st.sidebar.selectbox("Sample", sample_ids, key="micro_sample")

# Determine which techniques this sample has
sample = project.samples[selected_sample]
micro_techniques = [t for t in ["TEM", "SEM", "STEM"] if sample.has_technique(t)]
selected_technique = st.sidebar.selectbox("Technique", micro_techniques, key="micro_tech")

imgs_per_page = st.sidebar.selectbox("Images per page", [6, 12, 24], index=1, key="micro_per_page")

# ── Get files ──
td = sample.techniques[selected_technique]
all_files = td.files

# Separate image files from other files
image_files = [f for f in all_files if f.suffix.lower() in _IMAGE_EXTS and f.exists()]
other_files = [f for f in all_files if f.suffix.lower() not in _IMAGE_EXTS]

# Summary
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Sample", selected_sample)
mc2.metric("Images", len(image_files))
mc3.metric("Total Files", len(all_files))

st.markdown("---")

if not image_files:
    st.info(f"No displayable images found for {selected_sample} / {selected_technique}. "
            f"{len(other_files)} non-image files present.")
    if other_files:
        with st.expander("Non-image files"):
            for f in other_files:
                st.markdown(f"- `{f.name}` ({f.suffix})")
    st.stop()

# ── Pagination ──
total_pages = max(1, (len(image_files) + imgs_per_page - 1) // imgs_per_page)
page_key = f"micro_page_{selected_sample}_{selected_technique}"
if page_key not in st.session_state:
    st.session_state[page_key] = 0

current_page = st.session_state[page_key]

# Page navigation
if total_pages > 1:
    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        if st.button("Previous", key="micro_prev", disabled=current_page == 0):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with nav_cols[1]:
        st.markdown(
            f"<div style='text-align:center;color:#94a3b8;'>Page {current_page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with nav_cols[2]:
        if st.button("Next", key="micro_next", disabled=current_page >= total_pages - 1):
            st.session_state[page_key] = current_page + 1
            st.rerun()

# ── Image Grid ──
start = current_page * imgs_per_page
end = min(start + imgs_per_page, len(image_files))
page_images = image_files[start:end]

n_cols = 3
for row_start in range(0, len(page_images), n_cols):
    row_images = page_images[row_start:row_start + n_cols]
    cols = st.columns(n_cols)
    for j, img_path in enumerate(row_images):
        with cols[j]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_container_width=True)
            except Exception as e:
                st.markdown(
                    f'<div style="background:#1e1e2e;border:1px solid #3b3b5c;border-radius:8px;'
                    f'padding:40px 16px;text-align:center;color:#64748b;">'
                    f'{img_path.name}<br><small>Cannot display</small></div>',
                    unsafe_allow_html=True,
                )

# ── File listing ──
with st.expander(f"All {selected_technique} files ({len(all_files)})"):
    for f in sorted(all_files, key=lambda p: p.name):
        is_img = f.suffix.lower() in _IMAGE_EXTS
        icon = "img" if is_img else "file"
        size_kb = f.stat().st_size / 1024 if f.exists() else 0
        st.markdown(f"- `{f.name}` ({f.suffix}, {size_kb:.0f} KB)")
