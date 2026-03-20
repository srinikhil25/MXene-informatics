"""
SEM Data Parser
===============
Parses Hitachi SU8600 SEM metadata from .txt companion files.
Extracts imaging conditions (voltage, magnification, WD, detector, FOV).

Input: SEM .txt metadata files (UTF-16 encoded from Hitachi instruments)
Output: Standardized JSON catalog of SEM images
"""

import json
import re
from pathlib import Path
from typing import Optional


def parse_sem_metadata(filepath: str) -> dict:
    """
    Parse Hitachi SU8600 SEM metadata .txt file.

    These files are UTF-16 encoded with space-padded characters.
    Format: Key = Value pairs describing imaging conditions.

    Returns dict with extracted metadata.
    """
    metadata = {}

    # Read file as bytes and decode (Hitachi uses UTF-16-LE with BOM)
    content = None
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        for enc in ["utf-16", "utf-16-le", "utf-8", "latin-1"]:
            try:
                decoded = raw.decode(enc, errors="replace")
                if "InstructName" in decoded or "Magnification" in decoded or \
                   "AcceleratingVoltage" in decoded:
                    content = decoded
                    break
            except Exception:
                continue
    except Exception:
        return {}

    if not content:
        return {}

    # Remove BOM
    content = content.lstrip("\ufeff")

    for line in content.split("\n"):
        line = line.strip().strip("\r")
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip().strip("[]").strip()
            value = value.strip()
            if key and value:
                metadata[key] = value

    # Extract key parameters
    result = {
        "instrument": metadata.get("InstructName", "Hitachi SU8600"),
        "serial_number": metadata.get("SerialNumber", ""),
        "image_name": metadata.get("ImageName", ""),
        "sample_name": metadata.get("SampleName", ""),
        "date": metadata.get("Date", ""),
        "time": metadata.get("Time", ""),
        "format": metadata.get("Format", "tif"),
        "data_size": metadata.get("DataSize", ""),
        "pixel_size_nm": _safe_float(metadata.get("PixelSize", "0")),
        "signal": metadata.get("SignalName", ""),
        "accelerating_voltage_v": _parse_voltage(metadata.get("AcceleratingVoltage", "")),
        "magnification": _safe_float(metadata.get("Magnification", "0")),
        "working_distance_um": _safe_float(metadata.get("WorkingDistance", "0").replace("um", "")),
        "emission_current_na": _safe_float(metadata.get("EmissionCurrent", "0").replace("nA", "")),
        "lens_mode": metadata.get("LensMode", ""),
        "brightness": _safe_float(metadata.get("Brightness", "0")),
        "contrast": _safe_float(metadata.get("Contrast", "0")),
        "fov": metadata.get("FOV", ""),
        "vacuum": metadata.get("Vacuum", ""),
        "deceleration_voltage_v": _parse_voltage(metadata.get("DecelerationVoltage", "")),
        "specimen_bias_v": _safe_float(metadata.get("SpecimenBias", "0").replace("V", "")),
        "source_file": str(filepath),
    }

    # Derive sample type from image name
    img_name = result["image_name"].lower()
    if "ticn2" in img_name:
        result["sample_type"] = "MXene_Ti3C2_N2"
        result["sample_description"] = "Ti3C2Tx MXene, N2 atmosphere at 30C"
    elif "ticar" in img_name:
        result["sample_type"] = "MXene_Ti3C2_Ar"
        result["sample_description"] = "Ti3C2Tx MXene, Ar atmosphere at 30C"
    else:
        result["sample_type"] = "Unknown"
        result["sample_description"] = ""

    # Find corresponding .tif image
    txt_path = Path(filepath)
    tif_path = txt_path.with_suffix(".tif")
    result["has_image"] = tif_path.exists()
    result["image_path"] = str(tif_path) if tif_path.exists() else ""

    return result


def _safe_float(s: str) -> float:
    """Safely convert string to float."""
    try:
        # Remove units and extra text
        s = re.sub(r"[a-zA-Z\s°µ]", "", s.strip())
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def _parse_voltage(s: str) -> float:
    """Parse voltage string like '20000 Volt' to float."""
    match = re.search(r"([\d.]+)", s)
    return float(match.group(1)) if match else 0.0


def parse_all_sem(data_dir: str) -> list:
    """
    Parse all SEM metadata .txt files in a directory tree.

    Returns list of image metadata dicts.
    """
    results = []
    data_path = Path(data_dir)

    for txt_file in sorted(data_path.rglob("*.txt")):
        try:
            parsed = parse_sem_metadata(str(txt_file))
            if parsed and parsed.get("magnification", 0) > 0:
                results.append(parsed)
                print(f"  Parsed SEM: {txt_file.name} -> {parsed['sample_type']} "
                      f"({parsed['magnification']:.0f}x, {parsed['accelerating_voltage_v']:.0f}V)")
        except Exception as e:
            print(f"  Warning: Could not parse {txt_file}: {e}")

    return results


def save_sem_catalog(sem_data: list, output_dir: str):
    """Save parsed SEM catalog as JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save full catalog
    json_path = out_path / "sem_catalog.json"
    with open(json_path, "w") as f:
        json.dump(sem_data, f, indent=2)
    print(f"  Saved: sem_catalog.json ({len(sem_data)} images)")

    # Save summary CSV
    csv_path = out_path / "sem_summary.csv"
    with open(csv_path, "w") as f:
        f.write("image_name,sample_type,magnification,voltage_v,working_distance_um,pixel_size_nm,has_image\n")
        for img in sem_data:
            f.write(f"{img['image_name']},{img['sample_type']},{img['magnification']:.0f},"
                    f"{img['accelerating_voltage_v']:.0f},{img['working_distance_um']:.1f},"
                    f"{img['pixel_size_nm']:.4f},{img['has_image']}\n")
    print(f"  Saved: sem_summary.csv")


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/MXDiscovery/Mxene_Analysis/SEM"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/MXene-Informatics/data/processed/sem"

    print("=" * 60)
    print("  SEM Metadata Parser — MXene-Informatics")
    print("=" * 60)
    print(f"\nSource: {raw_dir}")
    print(f"Output: {out_dir}\n")

    print("Parsing SEM metadata files...")
    sem_data = parse_all_sem(raw_dir)
    print(f"\nFound {len(sem_data)} SEM images.\n")

    print("Saving catalog...")
    save_sem_catalog(sem_data, out_dir)
    print("\nDone!")
