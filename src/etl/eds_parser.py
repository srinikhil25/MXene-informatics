"""
EDS/EDX Data Parser
===================
Parses Energy Dispersive X-ray Spectroscopy data in EMSA format.
EMSA (Electron Microscopy Society of America) is the standard format
for spectral data from electron microscopy instruments.

Input: .emsa files from TEM-EDX and SEM-EDS measurements
Output: Standardized JSON with energy-intensity spectra
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional


def parse_emsa(filepath: str) -> dict:
    """
    Parse an EMSA/MAS spectral data file.

    Format:
        - Header lines start with '#' containing metadata
        - Data begins after '#SPECTRUM : DATA BEGINS HERE'
        - Data lines: intensity values (one per channel)

    Returns:
        dict with metadata, energy array, and intensity array
    """
    metadata = {}
    intensities = []
    data_started = False

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header/metadata lines
            if line.startswith("#"):
                if "DATA BEGINS HERE" in line.upper():
                    data_started = True
                    continue
                # Parse key-value: #KEY : VALUE
                match = re.match(r"#(\w+)\s*:\s*(.*)", line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    metadata[key] = value
                continue

            # Data lines (after header)
            if data_started:
                # Values may be comma-separated or space-separated
                value = line.rstrip(",").strip()
                try:
                    intensities.append(float(value))
                except ValueError:
                    # May have multiple values per line
                    for v in re.split(r"[,\s]+", line):
                        v = v.strip().rstrip(",")
                        if v:
                            try:
                                intensities.append(float(v))
                            except ValueError:
                                pass

    # Build energy axis from metadata
    n_points = int(float(metadata.get("NPOINTS", len(intensities))))
    ev_per_channel = float(metadata.get("XPERCHAN", "10"))
    offset = float(metadata.get("OFFSET", "0"))
    energies = [offset + i * ev_per_channel for i in range(len(intensities))]

    result = {
        "metadata": {
            "title": metadata.get("TITLE", ""),
            "date": metadata.get("DATE", ""),
            "time": metadata.get("TIME", ""),
            "owner": metadata.get("OWNER", ""),
            "beam_kv": float(metadata.get("BEAMKV", "0")),
            "probe_current": float(metadata.get("PROBECUR", "0")),
            "live_time_s": float(metadata.get("LIVETIME", "0")),
            "real_time_s": float(metadata.get("REALTIME", "0")),
            "dead_time_pct": 0.0,
            "ev_per_channel": ev_per_channel,
            "offset_ev": offset,
            "n_channels": n_points,
            "x_units": metadata.get("XUNITS", ""),
            "y_units": metadata.get("YUNITS", ""),
            "data_type": metadata.get("DATATYPE", ""),
        },
        "n_points": len(intensities),
        "energy_ev": energies,
        "intensity": intensities,
        "source_file": str(filepath),
    }

    # Calculate dead time
    lt = result["metadata"]["live_time_s"]
    rt = result["metadata"]["real_time_s"]
    if rt > 0:
        result["metadata"]["dead_time_pct"] = (1 - lt / rt) * 100

    return result


def identify_peaks(energy: list, intensity: list, threshold_pct: float = 5.0) -> list:
    """
    Simple peak identification for EDS spectra.

    Identifies peaks above threshold_pct of maximum intensity
    and matches to known characteristic X-ray energies.

    Known MXene-relevant energies (Ka lines):
        C Ka: 277 eV, N Ka: 392 eV, O Ka: 525 eV, F Ka: 677 eV,
        Al Ka: 1487 eV, Ti Ka: 4511 eV, Ti La: 452 eV,
        Cu Ka: 8048 eV (grid), Cl Ka: 2622 eV
    """
    known_lines = {
        "C Ka": 277, "N Ka": 392, "O Ka": 525, "F Ka": 677,
        "Al Ka": 1487, "Si Ka": 1740, "Cl Ka": 2622,
        "Ti La": 452, "Ti Ka": 4511, "Ti Kb": 4932,
        "Cu Ka": 8048, "Cu La": 930, "Au Ma": 2123,
    }

    e = np.array(energy)
    i = np.array(intensity)

    if len(i) == 0 or max(i) == 0:
        return []

    threshold = max(i) * threshold_pct / 100

    peaks = []
    for name, line_ev in known_lines.items():
        # Find channels within ±100 eV of known line
        mask = (e >= line_ev - 100) & (e <= line_ev + 100)
        if mask.any():
            region = i[mask]
            if max(region) > threshold:
                peak_idx = np.argmax(region)
                peak_energy = e[mask][peak_idx]
                peak_intensity = region[peak_idx]
                peaks.append({
                    "element_line": name,
                    "expected_ev": line_ev,
                    "measured_ev": float(peak_energy),
                    "intensity": float(peak_intensity),
                    "shift_ev": float(peak_energy - line_ev),
                })

    return sorted(peaks, key=lambda x: x["measured_ev"])


def parse_all_eds(data_dir: str) -> list:
    """Parse all EMSA files in a directory tree."""
    results = []
    data_path = Path(data_dir)

    for emsa_file in sorted(data_path.rglob("*.emsa")):
        try:
            parsed = parse_emsa(str(emsa_file))
            # Add peak identification
            parsed["peaks"] = identify_peaks(
                parsed["energy_ev"], parsed["intensity"]
            )
            results.append(parsed)
            peak_names = [p["element_line"] for p in parsed["peaks"]]
            print(f"  Parsed EDS: {emsa_file.name} -> {parsed['n_points']} channels, "
                  f"peaks: {', '.join(peak_names) if peak_names else 'none'}")
        except Exception as e:
            print(f"  Warning: Could not parse {emsa_file}: {e}")

    return results


def save_eds_processed(eds_data: list, output_dir: str):
    """Save parsed EDS data as JSON and CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(eds_data):
        src = Path(data["source_file"])
        name = f"eds_{src.parent.name}_{src.stem}"

        json_path = out_path / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        csv_path = out_path / f"{name}.csv"
        with open(csv_path, "w") as f:
            f.write("energy_ev,intensity\n")
            for e, inten in zip(data["energy_ev"], data["intensity"]):
                f.write(f"{e:.1f},{inten:.1f}\n")

        print(f"  Saved: {name}.json, {name}.csv")

    # Save peak summary
    if eds_data:
        peak_path = out_path / "eds_peaks_summary.json"
        summary = []
        for data in eds_data:
            summary.append({
                "source": Path(data["source_file"]).name,
                "beam_kv": data["metadata"]["beam_kv"],
                "live_time_s": data["metadata"]["live_time_s"],
                "peaks": data["peaks"],
            })
        with open(peak_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: eds_peaks_summary.json")


if __name__ == "__main__":
    import sys
    # Search both TEM and XRD directories for EMSA files
    raw_dirs = [
        "D:/MXDiscovery/Mxene_Analysis/TEM",
        "D:/MXDiscovery/Mxene_Analysis/XRD",
    ]
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/MXene-Informatics/data/processed/eds"

    print("=" * 60)
    print("  EDS/EDX Data Parser — MXene-Informatics")
    print("=" * 60)

    all_eds = []
    for raw_dir in raw_dirs:
        if Path(raw_dir).exists():
            print(f"\nSource: {raw_dir}")
            eds_data = parse_all_eds(raw_dir)
            all_eds.extend(eds_data)

    print(f"\nTotal: {len(all_eds)} EDS spectra.\n")

    print("Saving processed data...")
    save_eds_processed(all_eds, out_dir)
    print("\nDone!")
