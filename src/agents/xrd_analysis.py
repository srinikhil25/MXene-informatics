# -*- coding: utf-8 -*-
"""
XRD Phase Identification Agent
===============================
Follows the REAL researcher workflow: you know what you made, you fetch
reference patterns for your target phases, overlay them on your experimental
data, assign peaks, and flag anything unmatched as a potential impurity.

Workflow:
    1. User selects target phases (e.g. "CuSe", "Cs3Bi2I9")
    2. Agent fetches reference patterns from Materials Project (cached)
    3. Experimental peaks are found via scipy
    4. Each experimental peak is assigned to the closest reference peak
    5. Unmatched peaks are flagged

Usage:
    from src.agents.xrd_analysis import (
        fetch_reference_pattern, find_peaks, assign_peaks,
    )

    refs = fetch_reference_pattern("CuSe")
    exp_peaks = find_peaks(two_theta, intensity)
    assignments, summary = assign_peaks(exp_peaks, refs)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks

from src.config import get_mp_api_key

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "xrd_cache"

# Common X-ray wavelengths in Angstroms
WAVELENGTHS = {
    "CuKa": 1.5406,
    "CuKa1": 1.5405,
    "CuKa2": 1.5443,
    "MoKa": 0.7107,
    "CoKa": 1.7889,
    "FeKa": 1.9373,
    "CrKa": 2.2897,
    "AgKa": 0.5594,
}

# Default two-theta range for XRD simulation
TWO_THETA_RANGE = (5, 90)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RefPeak:
    """A single peak in a computed reference pattern."""
    two_theta: float
    intensity: float   # 0-100
    d_spacing: float
    hkl: str           # e.g. "(1,0,1)"


@dataclass
class ReferencePattern:
    """Complete reference pattern for one polymorph of a material."""
    formula: str
    material_id: str
    space_group: str
    crystal_system: str
    lattice: dict       # {a, b, c, alpha, beta, gamma}
    peaks: list[RefPeak]            # sorted by intensity descending
    energy_above_hull: float        # 0 = thermodynamically stable


@dataclass
class ExpPeak:
    """A single experimentally observed peak."""
    two_theta: float
    intensity: float   # 0-100
    d_spacing: float


@dataclass
class PeakAssignment:
    """An experimental peak assigned (or not) to a reference phase."""
    exp_two_theta: float
    exp_intensity: float
    matched_phase: str      # formula or "Unmatched"
    ref_two_theta: float    # 0 if unmatched
    delta_two_theta: float
    hkl: str                # Miller indices or ""
    d_spacing: float


# ──────────────────────────────────────────────────────────────────────────────
# Helper: format hkl tuples
# ──────────────────────────────────────────────────────────────────────────────

def _format_hkl(hkl_list: list[dict]) -> str:
    """
    Format pymatgen hkl data into a readable string.

    pymatgen XRDPattern.hkls is a list of dicts, each like:
        {'hkl': (1, 0, 1), 'multiplicity': 4}
    We take the first entry (strongest contributor) and format it.
    """
    if not hkl_list:
        return "(?,?,?)"
    first = hkl_list[0]
    indices = first["hkl"]
    # Handle both 3-index (h,k,l) and 4-index hexagonal (h,k,i,l)
    return "(" + ",".join(str(x) for x in indices) + ")"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: sanitize formula for filename
# ──────────────────────────────────────────────────────────────────────────────

def _clean_formula(formula: str) -> str:
    """Remove characters that are invalid in filenames."""
    return re.sub(r"[^\w]", "_", formula)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Fetch reference pattern from Materials Project (with caching)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_reference_pattern(
    formula: str,
    wavelength: str = "CuKa",
    two_theta_range: tuple[float, float] = TWO_THETA_RANGE,
) -> list[ReferencePattern]:
    """
    Fetch XRD reference patterns for a given formula from Materials Project.

    Queries MP for all structures matching the formula, computes the XRD
    stick pattern for each polymorph using pymatgen's XRDCalculator, and
    returns them sorted by energy above hull (most stable first).

    Results are cached as JSON in data/xrd_cache/ so repeated calls are free.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. "CuSe", "Cs3Bi2I9", "Ti3C2".
    wavelength : str
        X-ray source name (key in WAVELENGTHS dict). Default "CuKa".
    two_theta_range : tuple
        Min and max 2-theta for pattern computation.

    Returns
    -------
    list[ReferencePattern]
        One entry per polymorph, sorted by energy_above_hull ascending.
        Empty list if nothing found on MP.
    """
    from mp_api.client import MPRester
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    wl = WAVELENGTHS.get(wavelength, 1.5406)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check cache first ────────────────────────────────────────────────
    formula_clean = _clean_formula(formula)
    cached = _load_cached_patterns(formula_clean)
    if cached:
        logger.info("Loaded %d cached pattern(s) for %s", len(cached), formula)
        return cached

    # ── Query Materials Project ──────────────────────────────────────────
    api_key = get_mp_api_key()
    logger.info("Querying Materials Project for '%s' ...", formula)

    results = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=[
                "material_id",
                "formula_pretty",
                "symmetry",
                "structure",
                "energy_above_hull",
            ],
        )

        if not docs:
            logger.warning("No MP entries found for formula '%s'", formula)
            return []

        logger.info("Found %d polymorph(s) for %s", len(docs), formula)

        calc = XRDCalculator(wavelength=wl)

        for doc in docs:
            try:
                structure = doc.structure
                pattern = calc.get_pattern(
                    structure,
                    two_theta_range=two_theta_range,
                )

                # Build peak list
                peaks = []
                max_intensity = max(pattern.y) if len(pattern.y) > 0 else 1.0
                for i, (tt, inten) in enumerate(zip(pattern.x, pattern.y)):
                    d = wl / (2.0 * np.sin(np.radians(tt / 2.0)))
                    hkl_str = _format_hkl(pattern.hkls[i])
                    peaks.append(RefPeak(
                        two_theta=round(float(tt), 4),
                        intensity=round(float(inten / max_intensity * 100.0), 2),
                        d_spacing=round(float(d), 4),
                        hkl=hkl_str,
                    ))

                # Sort by intensity descending
                peaks.sort(key=lambda p: p.intensity, reverse=True)

                # Extract lattice parameters
                latt = structure.lattice
                lattice_dict = {
                    "a": round(latt.a, 4),
                    "b": round(latt.b, 4),
                    "c": round(latt.c, 4),
                    "alpha": round(latt.alpha, 4),
                    "beta": round(latt.beta, 4),
                    "gamma": round(latt.gamma, 4),
                }

                symmetry = doc.symmetry
                ref = ReferencePattern(
                    formula=doc.formula_pretty,
                    material_id=str(doc.material_id),
                    space_group=str(symmetry.symbol) if symmetry else "Unknown",
                    crystal_system=str(symmetry.crystal_system) if symmetry else "Unknown",
                    lattice=lattice_dict,
                    peaks=peaks,
                    energy_above_hull=round(float(doc.energy_above_hull), 4),
                )
                results.append(ref)

            except Exception as exc:
                logger.warning(
                    "Failed to compute XRD for %s: %s", doc.material_id, exc
                )
                continue

    # Sort by energy above hull (most stable first)
    results.sort(key=lambda r: r.energy_above_hull)

    # ── Cache results ────────────────────────────────────────────────────
    for ref in results:
        _save_cached_pattern(formula_clean, ref)

    logger.info("Cached %d pattern(s) for %s", len(results), formula)
    return results


def _cache_path(formula_clean: str, material_id: str) -> Path:
    """Build the cache file path for a reference pattern."""
    mid = material_id.replace("-", "_")
    return _CACHE_DIR / f"ref_{formula_clean}_{mid}.json"


def _save_cached_pattern(formula_clean: str, ref: ReferencePattern) -> None:
    """Serialize a ReferencePattern to JSON cache."""
    path = _cache_path(formula_clean, ref.material_id)
    data = {
        "formula": ref.formula,
        "material_id": ref.material_id,
        "space_group": ref.space_group,
        "crystal_system": ref.crystal_system,
        "lattice": ref.lattice,
        "energy_above_hull": ref.energy_above_hull,
        "peaks": [asdict(p) for p in ref.peaks],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_cached_patterns(formula_clean: str) -> list[ReferencePattern]:
    """Load all cached patterns matching a formula prefix."""
    if not _CACHE_DIR.exists():
        return []

    prefix = f"ref_{formula_clean}_"
    files = sorted(_CACHE_DIR.glob(f"{prefix}*.json"))
    if not files:
        return []

    results = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            peaks = [
                RefPeak(
                    two_theta=p["two_theta"],
                    intensity=p["intensity"],
                    d_spacing=p["d_spacing"],
                    hkl=p["hkl"],
                )
                for p in data["peaks"]
            ]
            ref = ReferencePattern(
                formula=data["formula"],
                material_id=data["material_id"],
                space_group=data["space_group"],
                crystal_system=data["crystal_system"],
                lattice=data["lattice"],
                peaks=peaks,
                energy_above_hull=data["energy_above_hull"],
            )
            results.append(ref)
        except Exception as exc:
            logger.warning("Failed to load cache file %s: %s", fp.name, exc)
            continue

    results.sort(key=lambda r: r.energy_above_hull)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Find peaks in experimental data
# ──────────────────────────────────────────────────────────────────────────────

def find_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    prominence_pct: float = 3.0,
    wavelength_angstrom: float = 1.5406,
) -> list[ExpPeak]:
    """
    Detect peaks in an experimental XRD pattern.

    Parameters
    ----------
    two_theta : array-like
        2-theta values in degrees.
    intensity : array-like
        Measured intensity values (arbitrary units).
    prominence_pct : float
        Minimum peak prominence as a percentage of the maximum intensity.
        Default 3.0 means peaks must be at least 3% of max to be detected.
    wavelength_angstrom : float
        X-ray wavelength in Angstroms (for d-spacing via Bragg's law).

    Returns
    -------
    list[ExpPeak]
        Detected peaks, sorted by 2-theta ascending.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # Normalize intensity to 0-100
    i_max = intensity.max()
    if i_max <= 0:
        return []
    intensity_norm = intensity / i_max * 100.0

    # Find peaks with scipy
    min_prominence = prominence_pct  # on 0-100 scale
    indices, properties = scipy_find_peaks(
        intensity_norm,
        prominence=min_prominence,
    )

    peaks = []
    for idx in indices:
        tt = float(two_theta[idx])
        inten = float(intensity_norm[idx])
        # Bragg's law: d = lambda / (2 * sin(theta))
        theta_rad = np.radians(tt / 2.0)
        d = wavelength_angstrom / (2.0 * np.sin(theta_rad)) if theta_rad > 0 else 0.0

        peaks.append(ExpPeak(
            two_theta=round(tt, 4),
            intensity=round(inten, 2),
            d_spacing=round(d, 4),
        ))

    # Sort by 2-theta ascending
    peaks.sort(key=lambda p: p.two_theta)
    return peaks


# ──────────────────────────────────────────────────────────────────────────────
# 3. Assign experimental peaks to reference phases
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_zero_shift(
    exp_peaks: list[ExpPeak],
    ref_lookup: list[tuple[str, "RefPeak"]],
    coarse_tolerance: float = 1.0,
    min_ref_intensity: float = 5.0,
    top_n: int = 8,
) -> float:
    """
    Estimate the systematic zero-shift between experimental and reference
    patterns using the strongest experimental peaks.

    XRD patterns commonly have a systematic angular offset (sample
    displacement error). This function matches the strongest *top_n*
    experimental peaks against strong reference peaks (I > min_ref_intensity)
    with a coarse tolerance, then returns the **median** shift so that
    outliers (impurity peaks, misidentifications) are suppressed.

    Returns
    -------
    float
        Estimated shift in degrees (experimental - reference). A negative
        value means the experimental pattern is shifted to lower angles.
        Returns 0.0 if no reliable matches are found.
    """
    # Sort experimental peaks by intensity (strongest first)
    strong_exp = sorted(exp_peaks, key=lambda p: p.intensity, reverse=True)[:top_n]
    # Filter reference to significant peaks only
    strong_ref = [(f, rp) for f, rp in ref_lookup if rp.intensity >= min_ref_intensity]

    shifts = []
    for ep in strong_exp:
        best_delta = float("inf")
        best_shift = 0.0
        for _formula, rp in strong_ref:
            delta = abs(ep.two_theta - rp.two_theta)
            if delta < coarse_tolerance and delta < best_delta:
                best_delta = delta
                best_shift = ep.two_theta - rp.two_theta
        if best_delta < coarse_tolerance:
            shifts.append(best_shift)

    if len(shifts) < 2:
        return 0.0

    # Use median to suppress outliers (impurity peaks)
    median_shift = float(np.median(shifts))
    logger.info(
        "Zero-shift estimation: %d/%d strong peaks matched, "
        "median shift = %.4f deg (shifts: %s)",
        len(shifts), len(strong_exp),
        median_shift,
        [f"{s:.4f}" for s in shifts],
    )
    return median_shift


def assign_peaks(
    exp_peaks: list[ExpPeak],
    ref_patterns: list[ReferencePattern],
    tolerance_deg: float = 0.5,
    min_ref_intensity: float = 1.0,
    auto_zero_shift: bool = True,
) -> tuple[list[PeakAssignment], dict]:
    """
    Assign each experimental peak to the closest reference peak.

    Uses a two-pass approach:
      1. **Zero-shift estimation** — The strongest experimental peaks are
         matched against strong reference peaks with a coarse tolerance to
         estimate the systematic angular offset (sample displacement error).
      2. **Assignment pass** — All peaks are matched using the shift-corrected
         reference positions within ``tolerance_deg``.

    Parameters
    ----------
    exp_peaks : list[ExpPeak]
        Peaks found in the experimental pattern.
    ref_patterns : list[ReferencePattern]
        Reference patterns for the target phases.
    tolerance_deg : float
        Maximum allowed difference in 2-theta (degrees) to consider a match.
        Default 0.5 (increased from 0.3 to accommodate real-world shifts).
    min_ref_intensity : float
        Minimum reference peak intensity (%) to consider for matching.
        Prevents spurious matches to very weak calculated peaks. Default 1.0.
    auto_zero_shift : bool
        If True (default), automatically estimate and correct for systematic
        angular offset before matching.

    Returns
    -------
    assignments : list[PeakAssignment]
        One entry per experimental peak.
    summary : dict
        Keys: "matched" (dict of formula -> count), "unmatched" (int),
        "total" (int), "zero_shift" (float, estimated shift in degrees).
    """
    # Build a flat lookup: (formula, RefPeak) for all reference peaks
    # Filter out extremely weak peaks that lead to false matches
    ref_lookup: list[tuple[str, RefPeak]] = []
    for ref in ref_patterns:
        for rp in ref.peaks:
            if rp.intensity >= min_ref_intensity:
                ref_lookup.append((ref.formula, rp))

    # ── Pass 1: Estimate zero-shift ──────────────────────────────────
    zero_shift = 0.0
    if auto_zero_shift and exp_peaks and ref_lookup:
        zero_shift = _estimate_zero_shift(exp_peaks, ref_lookup)

    # ── Pass 2: Greedy one-to-one assignment with shift correction ────
    #
    # Build all candidate (exp_idx, ref_idx, delta) pairs within tolerance,
    # sort by delta ascending, then greedily assign — each ref peak can only
    # be claimed by one experimental peak (the one with the smallest delta).
    # This prevents multiple exp peaks mapping to the same reference peak.
    #
    candidates: list[tuple[int, int, float, str, RefPeak]] = []
    for ei, ep in enumerate(exp_peaks):
        ep_corrected = ep.two_theta - zero_shift
        for ri, (formula, rp) in enumerate(ref_lookup):
            delta = abs(ep_corrected - rp.two_theta)
            if delta <= tolerance_deg:
                candidates.append((ei, ri, delta, formula, rp))

    # Sort by delta (smallest first) for greedy assignment
    candidates.sort(key=lambda c: c[2])

    # Greedy: claim each (exp, ref) pair only if neither is already taken
    claimed_exp: set[int] = set()
    claimed_ref: set[int] = set()
    assignment_map: dict[int, PeakAssignment] = {}
    match_counts: dict[str, int] = {}

    for ei, ri, delta, formula, rp in candidates:
        if ei in claimed_exp or ri in claimed_ref:
            continue
        claimed_exp.add(ei)
        claimed_ref.add(ri)
        ep = exp_peaks[ei]
        assignment_map[ei] = PeakAssignment(
            exp_two_theta=ep.two_theta,
            exp_intensity=ep.intensity,
            matched_phase=formula,
            ref_two_theta=rp.two_theta,
            delta_two_theta=round(delta, 4),
            hkl=rp.hkl,
            d_spacing=ep.d_spacing,
        )
        match_counts[formula] = match_counts.get(formula, 0) + 1

    # Fill in unmatched peaks
    assignments: list[PeakAssignment] = []
    n_unmatched = 0
    for ei, ep in enumerate(exp_peaks):
        if ei in assignment_map:
            assignments.append(assignment_map[ei])
        else:
            # Find closest ref for reporting (even though it's outside tolerance)
            ep_corrected = ep.two_theta - zero_shift
            best_delta = float("inf")
            for _formula, rp in ref_lookup:
                delta = abs(ep_corrected - rp.two_theta)
                if delta < best_delta:
                    best_delta = delta
            assignments.append(PeakAssignment(
                exp_two_theta=ep.two_theta,
                exp_intensity=ep.intensity,
                matched_phase="Unmatched",
                ref_two_theta=0.0,
                delta_two_theta=round(best_delta, 4) if best_delta < float("inf") else 0.0,
                hkl="",
                d_spacing=ep.d_spacing,
            ))
            n_unmatched += 1

    summary = {
        "matched": match_counts,
        "unmatched": n_unmatched,
        "total": len(exp_peaks),
        "zero_shift": round(zero_shift, 4),
    }

    return assignments, summary


# ──────────────────────────────────────────────────────────────────────────────
# 4. Extract elements from a sample registry (XPS/EDS)
# ──────────────────────────────────────────────────────────────────────────────

def extract_elements_from_registry(registry: dict) -> tuple[list[str], str]:
    """
    Pull element lists from XPS or EDS data in a sample registry.

    Searches the registry dict for keys related to XPS survey or EDS
    quantification and extracts the element symbols found.

    Parameters
    ----------
    registry : dict
        A sample registry dictionary (as built by the ETL pipeline).
        Expected to contain keys like "xps_survey", "eds_quant", etc.

    Returns
    -------
    elements : list[str]
        Sorted list of element symbols (e.g. ["Bi", "Cs", "I"]).
    source : str
        Which data source the elements came from (e.g. "XPS survey",
        "EDS quantification").
    """
    # Try XPS survey first
    for key in ("xps_survey", "xps", "xps_elements"):
        if key in registry and registry[key]:
            data = registry[key]
            if isinstance(data, dict):
                elements = sorted(data.keys())
            elif isinstance(data, list):
                elements = sorted(data)
            else:
                continue
            return elements, "XPS survey"

    # Try EDS quantification
    for key in ("eds_quant", "eds", "eds_elements", "sem_eds"):
        if key in registry and registry[key]:
            data = registry[key]
            if isinstance(data, dict):
                elements = sorted(data.keys())
            elif isinstance(data, list):
                elements = sorted(data)
            else:
                continue
            return elements, "EDS quantification"

    return [], "none"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load XRDML file
# ──────────────────────────────────────────────────────────────────────────────

def _load_xrdml(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a PANalytical XRDML file and return (two_theta, intensity).

    This is a lightweight parser that extracts the scan data from the XML.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()

    # XRDML uses a namespace
    ns_match = re.match(r"\{(.+?)\}", root.tag)
    ns = {"xrdml": ns_match.group(1)} if ns_match else {}

    # Find the scan data
    if ns:
        data_points = root.find(".//xrdml:dataPoints", ns)
    else:
        data_points = root.find(".//dataPoints")

    if data_points is None:
        raise ValueError(f"No dataPoints element found in {path}")

    # Get 2-theta start/end positions
    if ns:
        positions = data_points.findall(".//xrdml:positions", ns)
    else:
        positions = data_points.findall(".//positions")

    tt_start = tt_end = None
    for pos in positions:
        axis = pos.get("axis", "")
        if "2Theta" in axis:
            if ns:
                start_el = pos.find("xrdml:startPosition", ns)
                end_el = pos.find("xrdml:endPosition", ns)
            else:
                start_el = pos.find("startPosition")
                end_el = pos.find("endPosition")
            if start_el is not None and end_el is not None:
                tt_start = float(start_el.text)
                tt_end = float(end_el.text)
                break

    if tt_start is None or tt_end is None:
        raise ValueError(f"Could not find 2Theta positions in {path}")

    # Get intensity counts
    if ns:
        counts_el = data_points.find(".//xrdml:counts", ns)
        intensities_el = data_points.find(".//xrdml:intensities", ns)
    else:
        counts_el = data_points.find(".//counts")
        intensities_el = data_points.find(".//intensities")

    count_el = counts_el if counts_el is not None else intensities_el
    if count_el is None:
        raise ValueError(f"No counts/intensities element found in {path}")

    intensity = np.array([float(x) for x in count_el.text.strip().split()])
    two_theta = np.linspace(tt_start, tt_end, len(intensity))

    return two_theta, intensity


# ──────────────────────────────────────────────────────────────────────────────
# CLI test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    # ── 1. Fetch reference patterns ──────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Fetching reference patterns from Materials Project")
    print("=" * 70)

    cuse_refs = fetch_reference_pattern("CuSe")
    print(f"\nCuSe: {len(cuse_refs)} polymorph(s) found")
    for ref in cuse_refs:
        print(f"  {ref.material_id:12s}  {ref.space_group:12s}  "
              f"E_hull={ref.energy_above_hull:.3f} eV/atom  "
              f"{len(ref.peaks)} peaks")

    csbi_refs = fetch_reference_pattern("Cs3Bi2I9")
    print(f"\nCs3Bi2I9: {len(csbi_refs)} polymorph(s) found")
    for ref in csbi_refs:
        print(f"  {ref.material_id:12s}  {ref.space_group:12s}  "
              f"E_hull={ref.energy_above_hull:.3f} eV/atom  "
              f"{len(ref.peaks)} peaks")

    # ── 2. Load experimental data ────────────────────────────────────────
    xrdml_path = PROJECT_ROOT / "data_raw" / "dhivya_data" / "XRD" / "Dr.MN-dhivya-cscbi1.xrdml"
    print(f"\n{'=' * 70}")
    print(f"STEP 2: Loading experimental XRD data")
    print(f"  File: {xrdml_path}")
    print("=" * 70)

    if not xrdml_path.exists():
        print(f"  ERROR: File not found at {xrdml_path}")
        print("  Skipping experimental analysis.")
        sys.exit(0)

    two_theta, intensity = _load_xrdml(xrdml_path)
    print(f"  Loaded {len(two_theta)} data points")
    print(f"  2-theta range: {two_theta[0]:.2f} - {two_theta[-1]:.2f} deg")

    # ── 3. Find experimental peaks ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STEP 3: Finding experimental peaks")
    print("=" * 70)

    exp_peaks = find_peaks(two_theta, intensity)
    print(f"  Found {len(exp_peaks)} peaks")
    for ep in exp_peaks[:10]:
        print(f"    2theta={ep.two_theta:7.3f}  I={ep.intensity:6.1f}  d={ep.d_spacing:.3f} A")
    if len(exp_peaks) > 10:
        print(f"    ... and {len(exp_peaks) - 10} more")

    # ── 4. Assign peaks ──────────────────────────────────────────────────
    # Use Cs3Bi2I9 references (this is the expected phase for the sample)
    all_refs = csbi_refs  # adjust as needed
    print(f"\n{'=' * 70}")
    print("STEP 4: Assigning peaks to reference phases")
    print(f"  Using {len(all_refs)} reference pattern(s)")
    print("=" * 70)

    assignments, summary = assign_peaks(exp_peaks, all_refs)

    print(f"\n  Summary:")
    print(f"    Total peaks:    {summary['total']}")
    for phase, count in summary["matched"].items():
        print(f"    Matched {phase}: {count}")
    print(f"    Unmatched:      {summary['unmatched']}")

    print(f"\n  {'Exp 2theta':>10s}  {'I':>6s}  {'Phase':>14s}  "
          f"{'Ref 2theta':>10s}  {'delta':>6s}  {'hkl':>10s}  {'d (A)':>7s}")
    print(f"  {'-' * 10}  {'-' * 6}  {'-' * 14}  "
          f"{'-' * 10}  {'-' * 6}  {'-' * 10}  {'-' * 7}")

    for a in assignments:
        ref_tt = f"{a.ref_two_theta:10.3f}" if a.ref_two_theta > 0 else "         -"
        print(f"  {a.exp_two_theta:10.3f}  {a.exp_intensity:6.1f}  {a.matched_phase:>14s}  "
              f"{ref_tt}  {a.delta_two_theta:6.3f}  {a.hkl:>10s}  {a.d_spacing:7.3f}")

    print("\nDone.")
