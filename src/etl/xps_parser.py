"""
XPS Data Parser
===============
Parses X-ray Photoelectron Spectroscopy data from the MXene characterization.
Handles both:
  - Survey/wide scan data (binding energy vs intensity CSV)
  - High-resolution region scans (C 1s, O 1s, Ti 2p, F 1s)
  - Quantification summary (dens.txt with atomic concentrations)

Input: Raw XPS .txt files
Output: Standardized JSON with metadata + arrays
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional


def parse_xps_spectrum(filepath: str) -> dict:
    """
    Parse an XPS spectrum file.

    Two formats handled:
    1. Simple CSV: binding_energy,intensity (no header)
    2. Tab-separated with header: KE, BE, Intensity, Transmission

    Returns:
        dict with binding_energy, intensity arrays and metadata
    """
    filepath = Path(filepath)
    filename = filepath.stem

    # Determine scan type from filename
    scan_type = "unknown"
    element = ""
    if "wide" in filename.lower():
        scan_type = "survey"
    elif "C 1s" in filename or "C_1s" in filename:
        scan_type = "high_resolution"
        element = "C 1s"
    elif "O 1s" in filename or "O_1s" in filename:
        scan_type = "high_resolution"
        element = "O 1s"
    elif "Ti 2p" in filename or "Ti_2p" in filename:
        scan_type = "high_resolution"
        element = "Ti 2p"
    elif "F 1s" in filename or "F_1s" in filename:
        scan_type = "high_resolution"
        element = "F 1s"

    binding_energy = []
    intensity = []
    kinetic_energy = []
    metadata_lines = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for header/metadata lines
            if line.startswith("Dataset") or line.startswith("Dwell") or \
               line.startswith("Number") or line.startswith("Kinetic"):
                metadata_lines.append(line)
                continue

            # Parse data
            parts = line.split(",") if "," in line else line.split("\t")
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) >= 2:
                try:
                    vals = [float(p) for p in parts]
                    if len(vals) == 2:
                        # Simple CSV: BE, intensity
                        binding_energy.append(vals[0])
                        intensity.append(vals[1])
                    elif len(vals) >= 3:
                        # Tab format: KE, BE, intensity [, transmission]
                        kinetic_energy.append(vals[0])
                        binding_energy.append(vals[1])
                        intensity.append(vals[2])
                except ValueError:
                    continue

    # Parse metadata
    dataset_name = ""
    dwell_time = 0.0
    n_sweeps = 0
    for ml in metadata_lines:
        if ml.startswith("Dataset"):
            dataset_name = ml.split("Dataset")[-1].strip().lstrip(":")
        elif "Dwell" in ml:
            try:
                dwell_time = float(ml.split("\t")[-1])
            except (ValueError, IndexError):
                pass
        elif "sweeps" in ml.lower():
            try:
                n_sweeps = int(ml.split("\t")[-1])
            except (ValueError, IndexError):
                pass

    result = {
        "metadata": {
            "scan_type": scan_type,
            "element": element,
            "dataset_name": dataset_name or filename,
            "dwell_time_s": dwell_time,
            "n_sweeps": n_sweeps,
            "be_range": [min(binding_energy), max(binding_energy)] if binding_energy else [],
        },
        "n_points": len(binding_energy),
        "binding_energy_ev": binding_energy,
        "intensity_cps": intensity,
        "source_file": str(filepath),
    }

    if kinetic_energy:
        result["kinetic_energy_ev"] = kinetic_energy

    return result


def parse_xps_quantification(filepath: str) -> dict:
    """
    Parse XPS quantification summary (dens.txt format).

    Format: Tab-separated table with columns:
    Peak, Type, Position BE (eV), FWHM (eV), Raw Area (cps eV), RSF, Atomic Mass,
    Atomic Conc %, Mass Conc %

    Returns dict with element-level quantification.
    """
    elements = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Peak") or "BE (eV)" in line or "---" in line:
            continue

        # Parse element data lines
        parts = line.split()
        if len(parts) >= 9:
            try:
                element_data = {
                    "peak": f"{parts[0]} {parts[1]}",
                    "type": parts[2],
                    "position_be_ev": float(parts[3]),
                    "fwhm_ev": float(parts[4]),
                    "raw_area_cps_ev": float(parts[5]),
                    "rsf": float(parts[6]),
                    "atomic_mass": float(parts[7]),
                    "atomic_conc_pct": float(parts[8]),
                    "mass_conc_pct": float(parts[9]) if len(parts) > 9 else None,
                }
                elements.append(element_data)
            except (ValueError, IndexError):
                continue

    result = {
        "metadata": {
            "type": "quantification",
            "n_elements": len(elements),
        },
        "elements": elements,
        "summary": {
            e["peak"]: {
                "atomic_pct": e["atomic_conc_pct"],
                "mass_pct": e["mass_conc_pct"],
                "position_ev": e["position_be_ev"],
            }
            for e in elements
        },
        "source_file": str(filepath),
    }

    return result


def parse_all_xps(data_dir: str) -> dict:
    """
    Parse all XPS data from a directory.

    Returns dict with 'spectra' (list) and 'quantification' (dict).
    """
    spectra = []
    quantification = None
    data_path = Path(data_dir)

    for txt_file in sorted(data_path.rglob("*.txt")):
        try:
            fname = txt_file.name.lower()

            # Quantification file
            if fname == "dens.txt":
                quantification = parse_xps_quantification(str(txt_file))
                print(f"  Parsed XPS quant: {txt_file.name} -> "
                      f"{len(quantification['elements'])} elements")
                continue

            # Spectrum files (check if it has binding energy data)
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                first_lines = f.read(500)

            # XPS files have comma-separated float data or "Binding Energy" header
            if ("Binding Energy" in first_lines or
                "Dataset" in first_lines or
                re.search(r"\d+\.\d+,\d+\.\d+", first_lines)):
                parsed = parse_xps_spectrum(str(txt_file))
                if parsed["n_points"] > 10:  # Valid spectrum
                    spectra.append(parsed)
                    print(f"  Parsed XPS: {txt_file.name} -> {parsed['metadata']['scan_type']} "
                          f"({parsed['metadata']['element'] or 'survey'}, {parsed['n_points']} points)")

        except Exception as e:
            print(f"  Warning: Could not parse {txt_file}: {e}")

    return {
        "spectra": spectra,
        "quantification": quantification,
    }


def save_xps_processed(xps_data: dict, output_dir: str):
    """Save parsed XPS data as JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save spectra
    for spec in xps_data["spectra"]:
        scan_type = spec["metadata"]["scan_type"]
        element = spec["metadata"]["element"].replace(" ", "_") or "survey"
        name = f"xps_{element}"

        json_path = out_path / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(spec, f, indent=2)

        csv_path = out_path / f"{name}.csv"
        with open(csv_path, "w") as f:
            f.write("binding_energy_ev,intensity_cps\n")
            for be, inten in zip(spec["binding_energy_ev"], spec["intensity_cps"]):
                f.write(f"{be:.6f},{inten:.6f}\n")

        print(f"  Saved: {name}.json, {name}.csv")

    # Save quantification
    if xps_data["quantification"]:
        json_path = out_path / "xps_quantification.json"
        with open(json_path, "w") as f:
            json.dump(xps_data["quantification"], f, indent=2)
        print(f"  Saved: xps_quantification.json")


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/MXDiscovery/Mxene_Analysis/XPS"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/MXene-Informatics/data/processed/xps"

    print("=" * 60)
    print("  XPS Data Parser — MXene-Informatics")
    print("=" * 60)
    print(f"\nSource: {raw_dir}")
    print(f"Output: {out_dir}\n")

    print("Parsing XPS files...")
    xps_data = parse_all_xps(raw_dir)
    print(f"\nFound {len(xps_data['spectra'])} spectra, "
          f"{'1 quantification table' if xps_data['quantification'] else 'no quantification'}.\n")

    print("Saving processed data...")
    save_xps_processed(xps_data, out_dir)
    print("\nDone!")
