"""
XRD Data Parser
===============
Parses Rigaku Ultima3 XRD .txt and .raw files into standardized format.
Extracts metadata (instrument, wavelength, scan range) and 2theta-intensity data.

Input: Raw XRD .txt files from Rigaku instrument
Output: Standardized JSON with metadata + numpy-compatible arrays
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional


def parse_rigaku_txt(filepath: str) -> dict:
    """
    Parse a Rigaku Ultima3 XRD text file.

    Format:
        - Header lines start with ';' containing metadata
        - Data lines: '2theta intensity' (space-separated)

    Returns:
        dict with keys:
            metadata: instrument settings, sample name, wavelength, etc.
            two_theta: list of 2theta angles (degrees)
            intensity: list of counts
            scan_range: (start, finish) in degrees
            step_size: angular step in degrees
    """
    metadata = {}
    two_theta = []
    intensity = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse metadata lines (;Key = Value)
            if line.startswith(";"):
                match = re.match(r";(\w[\w\s]*)=\s*(.*)", line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    metadata[key] = value
                continue

            # Parse data lines (2theta intensity)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue

    # Extract key parameters
    result = {
        "metadata": {
            "sample_name": metadata.get("SampleName", "Unknown"),
            "instrument": metadata.get("Gonio", "Unknown"),
            "target": metadata.get("Target", "Cu"),
            "wavelength_ka1": float(metadata.get("KAlpha1", "1.54056")),
            "wavelength_ka2": float(metadata.get("KAlpha2", "1.5444")),
            "voltage_kv": float(metadata.get("KV", "0")),
            "current_ma": float(metadata.get("mA", "0")),
            "axis": metadata.get("AxisName", "2Theta/Omega"),
            "monochromator": metadata.get("IncidentMonochro", ""),
            "counter": metadata.get("Counter", ""),
            "attachment": metadata.get("Attachment", ""),
            "scan_speed": metadata.get("Speed", ""),
            "step_width": metadata.get("Width", ""),
            "x_unit": metadata.get("Xunit", "deg."),
            "y_unit": metadata.get("Yunit", "Count"),
        },
        "scan_range": {
            "start": float(metadata.get("Start", two_theta[0] if two_theta else 0)),
            "finish": float(metadata.get("Finish", two_theta[-1] if two_theta else 0)),
        },
        "step_size": float(metadata.get("Width", "0.02")),
        "n_points": len(two_theta),
        "two_theta": two_theta,
        "intensity": intensity,
        "source_file": str(filepath),
    }

    return result


def parse_all_xrd(data_dir: str) -> list:
    """
    Parse all XRD .txt files in a directory tree.

    Returns list of parsed XRD datasets.
    """
    results = []
    data_path = Path(data_dir)

    for txt_file in sorted(data_path.rglob("*.txt")):
        # Only parse files that look like XRD data (have ;SampleName header)
        try:
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                first_lines = f.read(1500)
            if "SampleName" in first_lines or "KAlpha1" in first_lines:
                parsed = parse_rigaku_txt(str(txt_file))
                results.append(parsed)
                print(f"  Parsed XRD: {txt_file.name} -> {parsed['metadata']['sample_name']} "
                      f"({parsed['n_points']} points, {parsed['scan_range']['start']}-{parsed['scan_range']['finish']}°)")
        except Exception as e:
            print(f"  Warning: Could not parse {txt_file}: {e}")

    return results


def save_xrd_processed(xrd_data: list, output_dir: str):
    """Save parsed XRD data as JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(xrd_data):
        sample = data["metadata"]["sample_name"].replace(" ", "_")

        # Save full JSON (with metadata)
        json_path = out_path / f"xrd_{sample}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save CSV (just 2theta, intensity for easy plotting)
        csv_path = out_path / f"xrd_{sample}.csv"
        with open(csv_path, "w") as f:
            f.write("two_theta_deg,intensity_counts\n")
            for tt, inten in zip(data["two_theta"], data["intensity"]):
                f.write(f"{tt:.4f},{inten:.0f}\n")

        print(f"  Saved: {json_path.name}, {csv_path.name}")


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/MXDiscovery/Mxene_Analysis/XRD"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/Materials Informatics/data/processed/xrd"

    print("=" * 60)
    print("  XRD Data Parser — Materials Informatics")
    print("=" * 60)
    print(f"\nSource: {raw_dir}")
    print(f"Output: {out_dir}\n")

    print("Parsing XRD files...")
    xrd_data = parse_all_xrd(raw_dir)
    print(f"\nFound {len(xrd_data)} XRD datasets.\n")

    print("Saving processed data...")
    save_xrd_processed(xrd_data, out_dir)
    print("\nDone!")
