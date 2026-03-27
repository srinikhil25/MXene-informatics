"""
JEOL SEM Data Parser
====================
Parses JEOL FE-SEM metadata from .txt companion files.
Format: $KEY VALUE and $$KEY VALUE pairs (ASCII encoding).

Instruments: JEOL JSM-series FE-SEM
Input: .txt metadata files + .bmp/.jpg image files
Output: Standardized JSON catalog compatible with Hitachi parser output
"""

import json
import re
from pathlib import Path
from typing import Optional


def parse_jeol_metadata(filepath: str) -> dict:
    """
    Parse JEOL FE-SEM metadata .txt file.

    Parameters
    ----------
    filepath : str
        Path to the .txt metadata file.

    Returns
    -------
    dict with standardized SEM metadata fields
    """
    path = Path(filepath)
    meta = {
        "source_file": path.name,
        "instrument": "JEOL FE-SEM",
        "format": "JEOL",
    }

    # Read file (ASCII encoding)
    for enc in ["utf-8", "ascii", "latin-1"]:
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return meta

    # Parse $KEY VALUE and $$KEY VALUE pairs
    raw_fields = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match $KEY VALUE or $$KEY VALUE
        m = re.match(r'^(\$\$?[\w_]+)\s*(.*)', line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            raw_fields[key] = val

    # Extract standardized fields
    meta["accelerating_voltage_kv"] = _safe_float(raw_fields.get("$CM_ACCEL_VOLT", ""))
    meta["magnification"] = _safe_float(raw_fields.get("$CM_MAG", ""))
    meta["working_distance_um"] = _safe_float(raw_fields.get("$$SM_WD", ""))
    meta["signal_name"] = raw_fields.get("$CM_SIGNAL_NAME", "")
    meta["detector_name"] = raw_fields.get("$CM_DETECTOR_NAME", "")
    meta["date"] = raw_fields.get("$CM_DATE", "")
    meta["sample_name"] = raw_fields.get("$CM_TITLE", "")
    meta["contrast"] = _safe_float(raw_fields.get("$CM_CONTRAST", ""))
    meta["brightness"] = _safe_float(raw_fields.get("$CM_BRIGHTNESS", ""))
    meta["scan_rotation"] = _safe_float(raw_fields.get("$$SM_SCAN_ROTATION", ""))
    meta["emission_current"] = _safe_float(raw_fields.get("$SM_ARRIVAL_EMI", ""))
    meta["gun_voltage_kv"] = _safe_float(raw_fields.get("$SM_GB_GUN_VOLT", ""))

    # Image dimensions
    full_size = raw_fields.get("$CM_FULL_SIZE", "")
    if full_size:
        parts = full_size.split()
        if len(parts) >= 2:
            meta["image_width"] = int(float(parts[0]))
            meta["image_height"] = int(float(parts[1]))

    # Calculate pixel size from magnification and image width
    if meta.get("magnification") and meta["magnification"] > 0 and meta.get("image_width"):
        # Standard SEM: FOV (μm) ≈ reference_width / magnification
        # For typical JEOL at HFW reference ~127mm for 1x
        hfw_um = 127000.0 / meta["magnification"]  # horizontal field width in μm
        meta["pixel_size_nm"] = (hfw_um / meta["image_width"]) * 1000  # nm per pixel
        meta["field_of_view_um"] = hfw_um

    # Find associated image file
    image_path = None
    for ext in [".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".png"]:
        candidate = path.with_suffix(ext)
        if candidate.exists():
            image_path = candidate
            break
    # Also check for image with same stem in same directory
    if image_path is None:
        for ext in [".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".png"]:
            candidates = list(path.parent.glob(f"{path.stem}*{ext}"))
            if candidates:
                image_path = candidates[0]
                break

    meta["image_path"] = str(image_path) if image_path else None
    meta["has_image"] = image_path is not None

    # Store raw fields for reference
    meta["raw_fields"] = raw_fields

    return meta


def _safe_float(val: str) -> Optional[float]:
    """Safely convert string to float, handling units."""
    if not val:
        return None
    # Remove common units
    val = re.sub(r'\s*(Volt|kV|um|mm|deg|A).*', '', val, flags=re.IGNORECASE)
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_all_jeol_sem(raw_dir: str) -> list:
    """
    Recursively find and parse all JEOL FE-SEM .txt files.

    Parameters
    ----------
    raw_dir : str
        Root directory containing JEOL SEM data.

    Returns
    -------
    list of dict — parsed metadata for each image
    """
    raw_path = Path(raw_dir)
    results = []

    for txt_file in sorted(raw_path.rglob("*.txt")):
        try:
            with open(txt_file, encoding="utf-8") as f:
                first_line = f.readline().strip()
            # JEOL files start with $SEM_DATA_VERSION
            if not first_line.startswith("$SEM"):
                continue
        except Exception:
            continue

        meta = parse_jeol_metadata(str(txt_file))
        if meta.get("magnification") and meta["magnification"] > 0:
            meta["relative_path"] = str(txt_file.relative_to(raw_path))
            results.append(meta)

    print(f"  Parsed {len(results)} JEOL FE-SEM metadata files")
    return results


def save_jeol_sem_catalog(results: list, output_dir: str):
    """Save parsed JEOL SEM catalog to JSON and CSV."""
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON catalog
    # Remove raw_fields for clean output
    clean = []
    for r in results:
        c = {k: v for k, v in r.items() if k != "raw_fields"}
        clean.append(c)

    with open(out / "jeol_sem_catalog.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=str)

    # CSV summary
    if clean:
        df = pd.DataFrame(clean)
        cols = ["source_file", "sample_name", "magnification",
                "accelerating_voltage_kv", "working_distance_um",
                "pixel_size_nm", "date", "has_image"]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(out / "jeol_sem_summary.csv", index=False)

    print(f"  Saved JEOL SEM catalog: {len(clean)} images")
    return clean
