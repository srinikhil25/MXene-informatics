"""
Bruker EDX/EDS Parser
=====================
Parses Bruker Quantax EDX data from two formats:
  1. .spx files — XML-based spectrum files (4096 channels)
  2. .xls files — Excel quantification results

Instruments: Bruker XFlash series (e.g., XFlash 5010)
Input: .spx (spectrum) and .xls (quantification) files
Output: Standardized JSON with energy-intensity arrays + elemental composition
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np


# ── Known X-ray emission lines (keV) for peak identification ──
XRAY_LINES = {
    "B Kα": 0.183, "C Kα": 0.277, "N Kα": 0.392, "O Kα": 0.525,
    "F Kα": 0.677, "Na Kα": 1.041, "Mg Kα": 1.254, "Al Kα": 1.487,
    "Si Kα": 1.740, "P Kα": 2.013, "S Kα": 2.307, "Cl Kα": 2.622,
    "K Kα": 3.314, "Ca Kα": 3.691, "Ti Kα": 4.510, "Ti Kβ": 4.932,
    "Ti Lα": 0.452, "V Kα": 4.952, "Cr Kα": 5.414, "Mn Kα": 5.898,
    "Fe Kα": 6.404, "Fe Kβ": 7.058, "Fe Lα": 0.705,
    "Co Kα": 6.930, "Ni Kα": 7.471, "Ni Kβ": 8.265, "Ni Lα": 0.851,
    "Cu Kα": 8.048, "Cu Kβ": 8.905, "Cu Lα": 0.930,
    "Zn Kα": 8.638, "Zn Kβ": 9.572, "Zn Lα": 1.012,
    "Ag Lα": 2.984, "Ag Lβ": 3.151,
    "Bi Lα": 10.839, "Bi Mα": 2.423,
    "Se Kα": 11.222, "Se Lα": 1.379,
    "Te Lα": 3.769, "Te Lβ": 4.030,
    "Mo Kα": 17.479, "Mo Lα": 2.293,
}


def parse_bruker_spx(filepath: str) -> dict:
    """
    Parse a Bruker .spx (XML) EDX spectrum file.

    Parameters
    ----------
    filepath : str
        Path to the .spx file.

    Returns
    -------
    dict with keys: metadata, energy_kev, counts, n_channels, detected_peaks
    """
    path = Path(filepath)
    result = {
        "source_file": path.name,
        "format": "Bruker SPX",
        "instrument": "Bruker Quantax",
    }

    # Try multiple encodings
    content = None
    for enc in ["windows-1252", "utf-8", "latin-1"]:
        try:
            with open(path, encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        return result

    # Parse XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        # Try stripping BOM or invalid chars
        content = content.lstrip('\ufeff')
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return result

    # Extract hardware header
    hw = root.find(".//ClassInstance[@Type='TRTSpectrumHardwareHeader']")
    if hw is not None:
        result["real_time_ms"] = _xml_float(hw, "RealTime")
        result["live_time_ms"] = _xml_float(hw, "LifeTime")
        result["dead_time_pct"] = _xml_float(hw, "DeadTime")

    # Extract detector info
    det = root.find(".//ClassInstance[@Type='TRTDetectorHeader']")
    if det is not None:
        det_type = det.findtext("Type", "")
        result["detector"] = det_type

    # Extract calibration
    calib_abs = None
    calib_lin = None
    cal_el = root.find(".//CalibAbs")
    if cal_el is not None and cal_el.text:
        calib_abs = float(cal_el.text)
    cal_lin = root.find(".//CalibLin")
    if cal_lin is not None and cal_lin.text:
        calib_lin = float(cal_lin.text)

    result["calib_abs_kev"] = calib_abs
    result["calib_lin_kev_per_ch"] = calib_lin

    # Extract channel data
    channels_el = root.find(".//Channels")
    if channels_el is not None and channels_el.text:
        counts_str = channels_el.text.strip().rstrip(",")
        counts = np.array([int(round(float(x))) for x in counts_str.split(",") if x.strip()])
        n_channels = len(counts)

        # Build energy axis from calibration
        if calib_abs is not None and calib_lin is not None:
            energy = calib_abs + calib_lin * np.arange(n_channels)
        else:
            # Default: 10 eV/channel, 0-40.96 keV
            energy = np.arange(n_channels) * 0.01

        result["n_channels"] = n_channels
        result["energy_kev"] = energy.tolist()
        result["counts"] = counts.tolist()
        result["energy_range_kev"] = [float(energy[0]), float(energy[-1])]
        result["total_counts"] = int(counts.sum())

        # Identify peaks
        result["detected_peaks"] = identify_peaks_bruker(energy, counts)

    return result


def identify_peaks_bruker(energy: np.ndarray, counts: np.ndarray,
                          threshold_pct: float = 5.0) -> list:
    """
    Identify elemental peaks in EDX spectrum by matching to known X-ray lines.

    Parameters
    ----------
    energy : array — energy in keV
    counts : array — intensity counts
    threshold_pct : float — minimum peak intensity as % of max

    Returns
    -------
    list of dict with element_line, expected_kev, measured_kev, intensity
    """
    from scipy.signal import find_peaks

    threshold = counts.max() * (threshold_pct / 100.0)
    peaks_idx, props = find_peaks(counts, height=threshold, distance=10, prominence=threshold * 0.5)

    detected = []
    for idx in peaks_idx:
        peak_energy = energy[idx]
        peak_intensity = int(counts[idx])

        # Match to nearest known line within ±0.15 keV
        best_match = None
        best_dist = 0.15
        for line_name, line_ev in XRAY_LINES.items():
            dist = abs(peak_energy - line_ev)
            if dist < best_dist:
                best_dist = dist
                best_match = line_name

        if best_match:
            detected.append({
                "element_line": best_match,
                "expected_kev": XRAY_LINES[best_match],
                "measured_kev": round(float(peak_energy), 3),
                "shift_kev": round(float(peak_energy - XRAY_LINES[best_match]), 3),
                "intensity": peak_intensity,
            })

    return detected


def parse_bruker_xls(filepath: str) -> dict:
    """
    Parse Bruker EDX quantification .xls file.

    Parameters
    ----------
    filepath : str
        Path to the .xls file.

    Returns
    -------
    dict with elemental composition (wt%, at%, errors)
    """
    try:
        import xlrd
    except ImportError:
        print("  WARNING: xlrd not installed. Cannot parse .xls files.")
        return {}

    path = Path(filepath)
    result = {
        "source_file": path.name,
        "format": "Bruker XLS",
    }

    try:
        wb = xlrd.open_workbook(str(path))
        sheet = wb.sheet_by_index(0)
    except Exception as e:
        result["error"] = str(e)
        return result

    # Find header row (contains "Element")
    header_row = None
    for r in range(sheet.nrows):
        for c in range(sheet.ncols):
            val = str(sheet.cell_value(r, c)).strip()
            if val.lower() == "element":
                header_row = r
                break
        if header_row is not None:
            break

    if header_row is None:
        return result

    # Read headers
    headers = [str(sheet.cell_value(header_row, c)).strip() for c in range(sheet.ncols)]

    # Read data rows
    elements = []
    for r in range(header_row + 1, sheet.nrows):
        row_data = {}
        element_name = str(sheet.cell_value(r, 0)).strip()
        if not element_name or element_name.lower() in ["sum:", "sum", ""]:
            continue

        row_data["element"] = element_name

        for c in range(1, sheet.ncols):
            header = headers[c].strip() if c < len(headers) else f"col_{c}"
            val = sheet.cell_value(r, c)
            if isinstance(val, float):
                row_data[_clean_header(header)] = round(val, 4)
            elif isinstance(val, str) and val.strip():
                row_data[_clean_header(header)] = val.strip()

        elements.append(row_data)

    # Extract metadata from top rows
    result["date"] = ""
    result["spectrum_name"] = ""
    for r in range(header_row):
        for c in range(sheet.ncols):
            val = str(sheet.cell_value(r, c)).strip()
            if re.match(r'\d{4}/\d{2}/\d{2}', val):
                result["date"] = val
            if "map" in val.lower() or "spectrum" in val.lower():
                result["spectrum_name"] = val

    result["elements"] = elements
    result["n_elements"] = len(elements)

    return result


def _xml_float(parent, tag: str) -> Optional[float]:
    """Extract float from XML element."""
    el = parent.find(tag)
    if el is not None and el.text:
        try:
            return float(el.text)
        except ValueError:
            return None
    return None


def _clean_header(h: str) -> str:
    """Clean Excel header for use as dict key."""
    h = h.strip().strip("[]").strip()
    h = re.sub(r'[^\w%.]', '_', h)
    h = re.sub(r'_+', '_', h).strip('_').lower()
    return h


def parse_all_bruker_edx(raw_dir: str) -> dict:
    """
    Recursively find and parse all Bruker EDX files (.spx + .xls).

    Returns
    -------
    dict with keys 'spectra' (from .spx) and 'quantifications' (from .xls)
    """
    raw_path = Path(raw_dir)
    spectra = []
    quantifications = []

    # Parse .spx spectrum files
    for spx_file in sorted(raw_path.rglob("*.spx")):
        try:
            result = parse_bruker_spx(str(spx_file))
            if result.get("n_channels"):
                result["relative_path"] = str(spx_file.relative_to(raw_path))
                # Infer sample from parent folder
                result["sample_group"] = spx_file.parent.name
                spectra.append(result)
        except Exception as e:
            print(f"  WARNING: Failed to parse {spx_file.name}: {e}")

    # Parse .xls quantification files
    for xls_file in sorted(raw_path.rglob("*.xls")):
        try:
            result = parse_bruker_xls(str(xls_file))
            if result.get("elements"):
                result["relative_path"] = str(xls_file.relative_to(raw_path))
                result["sample_group"] = xls_file.parent.name
                quantifications.append(result)
        except Exception as e:
            print(f"  WARNING: Failed to parse {xls_file.name}: {e}")

    print(f"  Parsed {len(spectra)} Bruker SPX spectra")
    print(f"  Parsed {len(quantifications)} Bruker XLS quantifications")

    return {"spectra": spectra, "quantifications": quantifications}


def save_bruker_edx(parsed: dict, output_dir: str):
    """Save parsed Bruker EDX data to JSON files."""
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save spectra (without full arrays for summary)
    spectra_summary = []
    for i, spec in enumerate(parsed["spectra"]):
        # Save individual spectrum
        fname = f"bruker_edx_{i:03d}_{spec.get('sample_group', 'unknown')}.json"
        with open(out / fname, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

        # Summary entry (no arrays)
        summary = {k: v for k, v in spec.items()
                   if k not in ("energy_kev", "counts")}
        summary["n_detected_peaks"] = len(spec.get("detected_peaks", []))
        spectra_summary.append(summary)

    # Save quantifications
    all_quant = []
    for q in parsed["quantifications"]:
        for el in q.get("elements", []):
            entry = {
                "source_file": q["source_file"],
                "sample_group": q.get("sample_group", ""),
                "date": q.get("date", ""),
            }
            entry.update(el)
            all_quant.append(entry)

    if all_quant:
        df = pd.DataFrame(all_quant)
        df.to_csv(out / "bruker_edx_quantification.csv", index=False)
        with open(out / "bruker_edx_quantification.json", "w", encoding="utf-8") as f:
            json.dump(all_quant, f, indent=2)

    # Save spectra summary
    if spectra_summary:
        with open(out / "bruker_edx_spectra_summary.json", "w", encoding="utf-8") as f:
            json.dump(spectra_summary, f, indent=2)

    print(f"  Saved {len(spectra_summary)} EDX spectra + {len(all_quant)} quantification entries")
    return spectra_summary, all_quant
