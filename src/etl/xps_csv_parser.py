"""
XPS CSV Data Parser
===================
Parses X-ray Photoelectron Spectroscopy data exported as CSV from PHI MultiPak
software. Handles both survey scans and high-resolution region scans.

CSV format (PHI MultiPak export):
    Line 1: version/format number (e.g., 3 or 6)
    Line 2: blank
    Line 3: spectrum identifier (e.g., Su1s, Se3d, C1s, Cu2p3, I3d5)
    Line 4: number of spectra (typically 1)
    Lines 5+: binding_energy,intensity (comma-separated, descending BE)

Input: Raw XPS .csv files from PHI MultiPak
Output: Standardized dicts with metadata + binding energy/intensity arrays
"""

import re
from pathlib import Path
from typing import Optional


# Regex to split identifiers like "Su1s", "Se3d", "Cu2p3", "I3d5"
# into element/prefix and orbital. Captures:
#   group 1: element symbol or "Su" (1-2 chars, first uppercase)
#   group 2: orbital label like "1s", "2p", "3d", "4f" (digit + letter)
#   group 3: optional trailing sub-shell digit (e.g., "3" in Cu2p3, "5" in I3d5)
_IDENTIFIER_RE = re.compile(r"^([A-Z][a-z]?)(\d[a-z])(\d?)$")


def _parse_region_name(identifier: str) -> str:
    """
    Convert a PHI MultiPak spectrum identifier to a human-readable region name.

    Examples:
        "Su1s" -> "Survey"
        "Su"   -> "Survey"
        "Se3d" -> "Se 3d"
        "C1s"  -> "C 1s"
        "Cu2p3"-> "Cu 2p"
        "I3d5" -> "I 3d"
        "Bi4f" -> "Bi 4f"
        "Cs3d" -> "Cs 3d"
        "O1s"  -> "O 1s"
        "Ti2p" -> "Ti 2p"
        "F1s"  -> "F 1s"

    Args:
        identifier: Raw spectrum identifier string from the CSV file.

    Returns:
        Human-readable region name string.
    """
    identifier = identifier.strip()

    # Survey scan
    if identifier.lower().startswith("su"):
        return "Survey"

    # Try regex match for "ElementOrbital[SubShell]" pattern
    match = _IDENTIFIER_RE.match(identifier)
    if match:
        element = match.group(1)
        orbital = match.group(2)
        # Sub-shell digit (group 3) is dropped for the display name
        return f"{element} {orbital}"

    # Fallback: return as-is
    return identifier


def parse_xps_csv(filepath: str) -> dict:
    """
    Parse a single XPS CSV file exported from PHI MultiPak.

    Args:
        filepath: Path to the .csv file.

    Returns:
        dict with keys:
            region: str (e.g., "Survey", "Se 3d", "C 1s")
            binding_energy: list of float (eV)
            intensity: list of float (counts/CPS)
            n_points: int
            be_range: dict with "start" and "end" (first and last BE values)
            metadata: dict with source_file, region, instrument
    """
    filepath = Path(filepath)
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    # Parse header
    # Line 1: version/format number
    # Line 2: blank
    # Line 3: spectrum identifier
    # Line 4: number of spectra
    identifier = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i <= 1:
            continue  # skip version number and blank line
        if i == 2:
            identifier = stripped
            break

    region = _parse_region_name(identifier)

    # Parse data lines (from line 5 onward, index 4+)
    binding_energy = []
    intensity = []

    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) >= 2:
            try:
                be = float(parts[0])
                inten = float(parts[1])
                binding_energy.append(be)
                intensity.append(inten)
            except ValueError:
                continue

    be_range = {}
    if binding_energy:
        be_range = {"start": binding_energy[0], "end": binding_energy[-1]}

    return {
        "region": region,
        "binding_energy": binding_energy,
        "intensity": intensity,
        "n_points": len(binding_energy),
        "be_range": be_range,
        "metadata": {
            "source_file": str(filepath),
            "region": region,
            "instrument": "PHI XPS (CSV export)",
        },
    }


def parse_xps_sample_folder(folder_path: str) -> dict:
    """
    Parse all XPS CSV files in a single sample folder.

    Expects a folder containing one or more .csv files, each being a survey
    or high-resolution region scan for the same sample.

    Args:
        folder_path: Path to the sample folder.

    Returns:
        dict with keys:
            sample_name: str (derived from folder name)
            survey: dict with binding_energy and intensity, or None
            regions: dict mapping region name to {binding_energy, intensity}
    """
    folder = Path(folder_path)
    sample_name = folder.name

    survey = None
    regions = {}

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        # Also check for uppercase extension
        csv_files = sorted(folder.glob("*.CSV"))

    for csv_file in csv_files:
        try:
            parsed = parse_xps_csv(str(csv_file))
        except Exception as e:
            print(f"  Warning: Could not parse {csv_file}: {e}")
            continue

        if parsed["n_points"] < 5:
            continue

        region = parsed["region"]
        spectrum_data = {
            "binding_energy": parsed["binding_energy"],
            "intensity": parsed["intensity"],
        }

        if region == "Survey":
            survey = spectrum_data
        else:
            regions[region] = spectrum_data

    return {
        "sample_name": sample_name,
        "survey": survey,
        "regions": regions,
    }


def parse_all_xps_csv(base_dir: str) -> list:
    """
    Recursively find all folders containing XPS CSV files and parse them.

    Walks the directory tree under base_dir. Any folder that directly contains
    at least one .csv file with valid XPS data is treated as a sample folder.

    Args:
        base_dir: Root directory to search.

    Returns:
        List of sample dicts (same format as parse_xps_sample_folder output).
    """
    base = Path(base_dir)
    results = []
    visited = set()

    for csv_file in sorted(base.rglob("*.csv")):
        folder = csv_file.parent
        if folder in visited:
            continue

        # Quick check: read first few lines to see if it matches PHI CSV format
        try:
            with open(csv_file, encoding="utf-8", errors="replace") as f:
                first_lines = []
                for _ in range(5):
                    line = f.readline()
                    if line is None:
                        break
                    first_lines.append(line.strip())

            # Validate: line 1 should be a small integer, line 3 should be
            # an identifier, line 5 should be "float,float"
            if len(first_lines) < 5:
                continue
            try:
                int(first_lines[0])
            except ValueError:
                continue
            if not re.match(r"^\d+\.\d+,\d+", first_lines[4]):
                continue

        except Exception:
            continue

        visited.add(folder)

        try:
            sample = parse_xps_sample_folder(str(folder))
            n_regions = len(sample["regions"])
            has_survey = sample["survey"] is not None
            if n_regions > 0 or has_survey:
                results.append(sample)
                print(f"  Parsed XPS CSV: {sample['sample_name']} -> "
                      f"{'survey + ' if has_survey else ''}"
                      f"{n_regions} region{'s' if n_regions != 1 else ''}")
        except Exception as e:
            print(f"  Warning: Could not parse folder {folder}: {e}")

    return results


if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/Materials-Informatics/data_raw/dhivya_data/XPS"

    print("=" * 60)
    print("  XPS CSV Parser (PHI MultiPak) — Materials Informatics")
    print("=" * 60)
    print(f"\nSource: {base_dir}\n")

    print("Parsing XPS CSV files...")
    samples = parse_all_xps_csv(base_dir)
    print(f"\nFound {len(samples)} sample(s).\n")

    for sample in samples:
        print(f"  Sample: {sample['sample_name']}")
        if sample["survey"]:
            n_pts = len(sample["survey"]["binding_energy"])
            print(f"    Survey: {n_pts} points")
        for region, data in sample["regions"].items():
            n_pts = len(data["binding_energy"])
            print(f"    {region}: {n_pts} points")

    print("\nDone!")
