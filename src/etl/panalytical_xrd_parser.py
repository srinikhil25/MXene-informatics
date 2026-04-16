"""
PANalytical XRDML Parser
=========================
Parses PANalytical Empyrean XRD .xrdml files (XML format) into standardized format.
Extracts metadata (instrument, wavelength, scan range) and 2theta-intensity data.

Input: .xrdml files from PANalytical Empyrean instruments
Output: Standardized dict with metadata + 2theta/intensity arrays
        (same keys as xrd_parser.py so the dashboard loads without changes)
"""

import json
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Known XRDML namespace versions — the parser tries each one
_XRDML_NAMESPACES = [
    "http://www.xrdml.com/XRDMeasurement/1.7",
    "http://www.xrdml.com/XRDMeasurement/1.5",
    "http://www.xrdml.com/XRDMeasurement/2.1",
]


def _detect_namespace(root: ET.Element) -> str:
    """
    Detect the XRDML namespace from the root element's tag.

    The tag comes back as '{http://...}xrdMeasurements', so we strip the
    braces and return the URI.  Falls back to the 1.7 namespace if detection
    fails.
    """
    tag = root.tag
    if tag.startswith("{"):
        ns = tag.split("}")[0].lstrip("{")
        return ns
    # Fallback
    return _XRDML_NAMESPACES[0]


def _find(element: ET.Element, path: str, ns: str) -> Optional[ET.Element]:
    """Namespace-aware find helper."""
    # Build the namespaced XPath: each bare tag gets the {ns} prefix
    parts = path.split("/")
    ns_path = "/".join(f"{{{ns}}}{p}" for p in parts)
    return element.find(ns_path)


def _findall(element: ET.Element, path: str, ns: str) -> list:
    """Namespace-aware findall helper."""
    parts = path.split("/")
    ns_path = "/".join(f"{{{ns}}}{p}" for p in parts)
    return element.findall(ns_path)


def _text(element: ET.Element, path: str, ns: str, default: str = "") -> str:
    """Return text content of a sub-element, or *default* if not found."""
    el = _find(element, path, ns)
    if el is not None and el.text:
        return el.text.strip()
    return default


def parse_panalytical_xrdml(filepath: str) -> dict:
    """
    Parse a PANalytical XRDML (.xrdml) file.

    The XRDML format is XML-based and contains:
      - Sample ID and metadata
      - Wavelength information (K-Alpha1, K-Alpha2, K-Beta)
      - X-ray tube settings (voltage, current, anode material)
      - Scan data: 2Theta start/end positions and space-separated intensity counts

    Two-theta values are linearly interpolated from startPosition / endPosition
    using the number of intensity data points.

    Returns
    -------
    dict
        Keys: metadata, two_theta, intensity, scan_range, step_size, n_points,
              source_file  (matches xrd_parser.py output format)
    """
    filepath = Path(filepath)
    logger.info("Parsing XRDML: %s", filepath.name)

    try:
        tree = ET.parse(str(filepath))
    except ET.ParseError as exc:
        logger.error("XML parse error in %s: %s", filepath, exc)
        raise ValueError(f"Could not parse XRDML file {filepath}: {exc}") from exc

    root = tree.getroot()
    ns = _detect_namespace(root)

    # ── Sample ────────────────────────────────────────────────────────
    sample_id = _text(root, "sample/id", ns, default=filepath.stem)

    # ── xrdMeasurement block ─────────────────────────────────────────
    measurement = _find(root, "xrdMeasurement", ns)
    if measurement is None:
        raise ValueError(f"No <xrdMeasurement> element found in {filepath}")

    # ── Wavelength ───────────────────────────────────────────────────
    wavelength_el = _find(measurement, "usedWavelength", ns)
    ka1 = float(_text(wavelength_el, "kAlpha1", ns, default="1.5405980"))
    ka2 = float(_text(wavelength_el, "kAlpha2", ns, default="1.5444260"))

    # ── X-ray tube (inside incidentBeamPath) ─────────────────────────
    incident = _find(measurement, "incidentBeamPath", ns)
    tube = _find(incident, "xRayTube", ns) if incident is not None else None

    if tube is not None:
        instrument_name = tube.get("name", "PANalytical Empyrean")
        anode = _text(tube, "anodeMaterial", ns, default="Cu")
        voltage = float(_text(tube, "tension", ns, default="0"))
        current = float(_text(tube, "current", ns, default="0"))
    else:
        instrument_name = "PANalytical Empyrean"
        anode = "Cu"
        voltage = 0.0
        current = 0.0

    # ── Scan block (first scan) ──────────────────────────────────────
    scan = _find(measurement, "scan", ns)
    if scan is None:
        raise ValueError(f"No <scan> element found in {filepath}")

    scan_axis = scan.get("scanAxis", "Gonio")

    # ── dataPoints ───────────────────────────────────────────────────
    data_points = _find(scan, "dataPoints", ns)
    if data_points is None:
        raise ValueError(f"No <dataPoints> element found in {filepath}")

    # Find the 2Theta positions element (there may be multiple <positions>
    # elements for different axes; we want axis="2Theta")
    two_theta_start = None
    two_theta_end = None
    x_unit = "deg"

    for pos_el in _findall(data_points, "positions", ns):
        if pos_el.get("axis") == "2Theta":
            two_theta_start = float(_text(pos_el, "startPosition", ns))
            two_theta_end = float(_text(pos_el, "endPosition", ns))
            x_unit = pos_el.get("unit", "deg")
            break

    if two_theta_start is None or two_theta_end is None:
        raise ValueError(f"No 2Theta positions found in {filepath}")

    # ── Intensities / Counts ─────────────────────────────────────────
    # PANalytical files use either <intensities> or <counts> for the data
    intensities_el = _find(data_points, "intensities", ns)
    if intensities_el is None:
        intensities_el = _find(data_points, "counts", ns)
    if intensities_el is None:
        raise ValueError(f"No <intensities> or <counts> element found in {filepath}")

    y_unit = intensities_el.get("unit", "counts")
    raw_counts = intensities_el.text.strip().split()
    intensity = [float(c) for c in raw_counts]
    n_points = len(intensity)

    # ── Build two-theta array ────────────────────────────────────────
    # The counts are evenly spaced between startPosition and endPosition
    two_theta = np.linspace(two_theta_start, two_theta_end, n_points).tolist()

    step_size = (two_theta_end - two_theta_start) / (n_points - 1) if n_points > 1 else 0.0

    # ── Assemble output (matches xrd_parser.py format) ───────────────
    result = {
        "metadata": {
            "sample_name": sample_id,
            "instrument": instrument_name,
            "target": anode,
            "wavelength_ka1": ka1,
            "wavelength_ka2": ka2,
            "voltage_kv": voltage,
            "current_ma": current,
            "scan_axis": scan_axis,
            "x_unit": x_unit,
            "y_unit": y_unit,
        },
        "scan_range": {
            "start": round(two_theta_start, 6),
            "finish": round(two_theta_end, 6),
        },
        "step_size": round(step_size, 6),
        "n_points": n_points,
        "two_theta": two_theta,
        "intensity": intensity,
        "source_file": str(filepath),
    }

    logger.info(
        "  Parsed: %s — %d points, %.2f–%.2f° 2theta",
        sample_id, n_points, two_theta_start, two_theta_end,
    )
    return result


def parse_all_panalytical_xrdml(base_dir: str) -> list:
    """
    Recursively find and parse all .xrdml files under *base_dir*.

    Returns
    -------
    list[dict]
        One standardized dict per file (same schema as parse_panalytical_xrdml).
    """
    results = []
    base_path = Path(base_dir)

    if not base_path.exists():
        logger.warning("Directory does not exist: %s", base_dir)
        return results

    xrdml_files = sorted(base_path.rglob("*.xrdml"))
    logger.info("Found %d .xrdml files in %s", len(xrdml_files), base_dir)

    for xrdml_file in xrdml_files:
        try:
            parsed = parse_panalytical_xrdml(str(xrdml_file))
            results.append(parsed)
            print(
                f"  Parsed XRDML: {xrdml_file.name} -> {parsed['metadata']['sample_name']} "
                f"({parsed['n_points']} points, "
                f"{parsed['scan_range']['start']}-{parsed['scan_range']['finish']}°)"
            )
        except Exception as e:
            logger.error("Could not parse %s: %s", xrdml_file, e)
            print(f"  Warning: Could not parse {xrdml_file}: {e}")

    return results


def save_xrdml_processed(xrdml_data: list, output_dir: str):
    """Save parsed XRDML data as JSON and CSV files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for data in xrdml_data:
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "D:/Materials-Informatics/data_raw/dhivya_data/XRD"
    )
    out_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "D:/Materials-Informatics/data/processed/xrd"
    )

    print("=" * 60)
    print("  PANalytical XRDML Parser — Materials Informatics")
    print("=" * 60)
    print(f"\nSource: {raw_dir}")
    print(f"Output: {out_dir}\n")

    print("Parsing XRDML files...")
    xrdml_data = parse_all_panalytical_xrdml(raw_dir)
    print(f"\nFound {len(xrdml_data)} XRDML datasets.\n")

    if xrdml_data:
        print("Saving processed data...")
        save_xrdml_processed(xrdml_data, out_dir)

    print("\nDone!")
