"""
ASC XRD Data Parser
===================
Parses ASC format XRD files (simple 2-column space-separated: 2theta, intensity).
No header — just scientific notation data. Commonly exported from various diffractometers.
Negative intensities are preserved as-is (background-subtracted data).

Input: .ASC / .asc files with two columns (2theta in degrees, intensity in counts)
Output: Standardized dict matching xrd_parser.py format for dashboard compatibility
"""

import json
import numpy as np
from pathlib import Path


def parse_asc_xrd(filepath: str) -> dict:
    """
    Parse an ASC format XRD file (2-column, space-separated, scientific notation).

    Format:
        - No header lines
        - Each line: 2theta_value   intensity_value  (scientific notation)
        - Negative intensities are kept as-is (background-subtracted)

    Returns:
        dict with keys matching parse_rigaku_txt() output:
            metadata: sample_name, instrument, x_unit, y_unit
            two_theta: list of 2theta angles (degrees)
            intensity: list of counts (may be negative)
            scan_range: {start, finish} in degrees
            step_size: angular step in degrees
            n_points: number of data points
            source_file: path to original file
    """
    two_theta = []
    intensity = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue

    # Derive sample name from filename (strip extension)
    sample_name = Path(filepath).stem

    # Calculate step size from first two data points
    step_size = round(two_theta[1] - two_theta[0], 6) if len(two_theta) >= 2 else 0.0

    result = {
        "metadata": {
            "sample_name": sample_name,
            "instrument": "Unknown (ASC format)",
            "x_unit": "deg.",
            "y_unit": "Count",
        },
        "scan_range": {
            "start": two_theta[0] if two_theta else 0,
            "finish": two_theta[-1] if two_theta else 0,
        },
        "step_size": step_size,
        "n_points": len(two_theta),
        "two_theta": two_theta,
        "intensity": intensity,
        "source_file": str(filepath),
    }

    return result


def parse_all_asc_xrd(base_dir: str) -> list:
    """
    Recursively find and parse all .ASC/.asc XRD files in a directory tree.

    Returns list of parsed XRD datasets.
    """
    results = []
    data_path = Path(base_dir)

    # Collect both .ASC and .asc (rglob is case-sensitive on some OS)
    asc_files = sorted(set(data_path.rglob("*.ASC")) | set(data_path.rglob("*.asc")))

    for asc_file in asc_files:
        try:
            parsed = parse_asc_xrd(str(asc_file))
            results.append(parsed)
            print(f"  Parsed ASC XRD: {asc_file.name} -> {parsed['metadata']['sample_name']} "
                  f"({parsed['n_points']} points, {parsed['scan_range']['start']:.2f}-{parsed['scan_range']['finish']:.2f}°)")
        except Exception as e:
            print(f"  Warning: Could not parse {asc_file}: {e}")

    return results


def save_asc_xrd_processed(xrd_data: list, output_dir: str):
    """Save parsed ASC XRD data as JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for data in xrd_data:
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
                f.write(f"{tt:.4f},{inten:.2f}\n")

        print(f"  Saved: {json_path.name}, {csv_path.name}")


if __name__ == "__main__":
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/Materials-Informatics/data_raw/dhivya_data/XRD"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/Materials-Informatics/data/processed/xrd_asc"

    print("=" * 60)
    print("  ASC XRD Data Parser — Materials Informatics")
    print("=" * 60)
    print(f"\nSource: {raw_dir}")
    print(f"Output: {out_dir}\n")

    print("Parsing ASC XRD files...")
    xrd_data = parse_all_asc_xrd(raw_dir)
    print(f"\nFound {len(xrd_data)} ASC XRD datasets.\n")

    if xrd_data:
        print("Saving processed data...")
        save_asc_xrd_processed(xrd_data, out_dir)
    print("\nDone!")
