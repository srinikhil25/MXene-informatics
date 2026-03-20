"""
XPS Analysis Module
====================
Peak deconvolution, chemical state identification, Shirley background
subtraction, and quantification for MXene surface characterization.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class XPSPeakFit:
    """Result of fitting a single XPS component peak."""
    center_ev: float
    intensity: float
    fwhm_ev: float
    area: float
    profile: str  # gaussian, lorentzian, voigt, GL
    assignment: str  # e.g. "Ti-C", "TiO2", "C-C"
    r_squared: float
    params: dict = field(default_factory=dict)


@dataclass
class XPSDeconvolution:
    """Full deconvolution result for one XPS region."""
    element: str
    n_components: int
    components: List[XPSPeakFit]
    envelope_r_squared: float
    background_type: str
    binding_energy: list = field(default_factory=list)
    raw_intensity: list = field(default_factory=list)
    background: list = field(default_factory=list)
    envelope: list = field(default_factory=list)
    component_curves: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# XPS reference database for Ti3C2Tx MXene
# ---------------------------------------------------------------------------
XPS_REFERENCES = {
    "Ti 2p": {
        "spin_orbit_split": 5.7,  # eV between 2p3/2 and 2p1/2
        "area_ratio": 0.5,  # 2p1/2 / 2p3/2 = 1:2
        "components": [
            {"name": "Ti-C", "be_range": (454.5, 455.5), "typical_be": 455.0,
             "description": "Ti bonded to C in MXene backbone"},
            {"name": "Ti(II)", "be_range": (455.5, 456.5), "typical_be": 455.8,
             "description": "Ti2+ from partial oxidation or Ti-OH"},
            {"name": "Ti(III)", "be_range": (456.5, 458.0), "typical_be": 457.0,
             "description": "Ti3+ from surface -OH/-O termination"},
            {"name": "TiO2", "be_range": (458.0, 459.5), "typical_be": 458.7,
             "description": "Ti4+ from TiO2 surface oxide"},
            {"name": "Ti-F", "be_range": (459.5, 461.0), "typical_be": 460.0,
             "description": "Ti bonded to F termination"},
        ],
    },
    "C 1s": {
        "components": [
            {"name": "Ti-C-Tx", "be_range": (281.0, 283.0), "typical_be": 282.0,
             "description": "C in MXene Ti-C bond"},
            {"name": "C-C/C=C", "be_range": (284.0, 285.5), "typical_be": 284.8,
             "description": "Adventitious carbon / graphitic C"},
            {"name": "C-O", "be_range": (285.5, 287.0), "typical_be": 286.4,
             "description": "C-O bonds (surface contamination)"},
            {"name": "C=O", "be_range": (287.0, 288.5), "typical_be": 287.8,
             "description": "Carbonyl groups"},
            {"name": "O-C=O", "be_range": (288.5, 290.0), "typical_be": 289.0,
             "description": "Carboxyl groups"},
        ],
    },
    "O 1s": {
        "components": [
            {"name": "TiO2", "be_range": (529.0, 530.5), "typical_be": 529.8,
             "description": "Lattice oxygen in TiO2"},
            {"name": "Ti-OH/C=O", "be_range": (530.5, 532.0), "typical_be": 531.2,
             "description": "Hydroxyl termination on MXene or carbonyl"},
            {"name": "C-O/Ti-OH2", "be_range": (532.0, 533.5), "typical_be": 532.5,
             "description": "C-O bonds or adsorbed water on Ti"},
            {"name": "H2O(ads)", "be_range": (533.5, 535.0), "typical_be": 533.8,
             "description": "Adsorbed water"},
        ],
    },
    "F 1s": {
        "components": [
            {"name": "Ti-F", "be_range": (684.0, 686.0), "typical_be": 685.0,
             "description": "F termination on MXene surface"},
            {"name": "Al-F", "be_range": (686.0, 687.5), "typical_be": 686.5,
             "description": "AlF3 residue from etching"},
            {"name": "C-F", "be_range": (687.5, 690.0), "typical_be": 688.5,
             "description": "C-F bonds"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------
def shirley_background(binding_energy, intensity, n_iter=50, tol=1e-6):
    """
    Shirley (iterative) background subtraction.

    The Shirley background accounts for inelastically scattered electrons
    and is the standard for XPS analysis.
    """
    n = len(intensity)
    bg = np.zeros(n)

    # Endpoints
    i_left = np.mean(intensity[:5])   # high BE side
    i_right = np.mean(intensity[-5:])  # low BE side

    # If BE is decreasing (typical XPS convention), swap
    if binding_energy[0] > binding_energy[-1]:
        i_left, i_right = i_right, i_left

    for _ in range(n_iter):
        old_bg = bg.copy()
        total_area = np.trapezoid(intensity - bg, binding_energy)
        if abs(total_area) < 1e-10:
            break

        for i in range(n):
            partial = np.trapezoid((intensity - bg)[:i + 1], binding_energy[:i + 1])
            bg[i] = i_right + (i_left - i_right) * partial / total_area

        if np.max(np.abs(bg - old_bg)) < tol:
            break

    return bg


def linear_background(binding_energy, intensity):
    """Simple linear background between endpoints."""
    bg = np.linspace(intensity[0], intensity[-1], len(intensity))
    return bg


def tougaard_background(binding_energy, intensity, B=2866, C=1643, D=1.0):
    """
    Simplified Tougaard background (3-parameter universal cross-section).
    S(E) = B * E / (C + E^2)^2
    """
    n = len(intensity)
    bg = np.zeros(n)
    de = abs(binding_energy[1] - binding_energy[0]) if n > 1 else 1.0

    for i in range(n):
        s = 0.0
        for j in range(i + 1, n):
            E_loss = abs(binding_energy[j] - binding_energy[i])
            cross_section = B * E_loss / (C + E_loss ** 2) ** 2
            s += intensity[j] * cross_section * de
        bg[i] = D * s

    return bg


BACKGROUND_FUNCS = {
    "shirley": shirley_background,
    "linear": linear_background,
}


# ---------------------------------------------------------------------------
# Peak functions for XPS
# ---------------------------------------------------------------------------
def gl_peak(x, amp, center, sigma, fraction=0.5):
    """
    Gaussian-Lorentzian product function (GL).
    fraction: 0 = pure Gaussian, 1 = pure Lorentzian.
    """
    fraction = np.clip(fraction, 0, 1)
    g = np.exp(-4 * np.log(2) * ((x - center) / sigma) ** 2)
    l = 1 / (1 + 4 * ((x - center) / sigma) ** 2)
    return amp * ((1 - fraction) * g + fraction * l)


def multi_gl(x, *params):
    """Sum of N GL peaks. params = [amp1, center1, sigma1, frac1, ...]"""
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        amp = params[4 * i]
        center = params[4 * i + 1]
        sigma = params[4 * i + 2]
        frac = params[4 * i + 3]
        y += gl_peak(x, amp, center, sigma, frac)
    return y


# ---------------------------------------------------------------------------
# XPS Deconvolution
# ---------------------------------------------------------------------------
def deconvolve_xps(binding_energy, intensity, element,
                   n_components=None, initial_positions=None,
                   background_type="shirley", gl_fraction=0.3,
                   max_fwhm=3.0, min_fwhm=0.5):
    """
    Deconvolve an XPS spectrum into component peaks.

    Parameters
    ----------
    binding_energy : array - BE values in eV
    intensity : array - intensity in CPS
    element : str - e.g. "Ti 2p", "C 1s"
    n_components : int - number of peaks to fit (auto if None)
    initial_positions : list of float - initial BE guesses
    background_type : str - 'shirley' or 'linear'
    gl_fraction : float - Gaussian-Lorentzian mixing (0=G, 1=L)
    max_fwhm : float - maximum allowed FWHM in eV
    min_fwhm : float - minimum allowed FWHM in eV

    Returns
    -------
    XPSDeconvolution
    """
    be = np.array(binding_energy)
    raw = np.array(intensity)

    # Background subtraction
    if background_type in BACKGROUND_FUNCS:
        bg = BACKGROUND_FUNCS[background_type](be, raw)
    else:
        bg = linear_background(be, raw)

    y = raw - bg
    y = np.maximum(y, 0)  # no negative values after background

    # Determine initial peak positions
    if initial_positions is None:
        if element in XPS_REFERENCES:
            ref = XPS_REFERENCES[element]
            initial_positions = [c["typical_be"] for c in ref["components"]
                                 if be.min() <= c["typical_be"] <= be.max()]
        else:
            # Auto-detect peaks
            peaks_idx, _ = find_peaks(y, prominence=y.max() * 0.05, distance=5)
            initial_positions = be[peaks_idx].tolist()

    if n_components is not None:
        initial_positions = initial_positions[:n_components]

    n_peaks = len(initial_positions)
    if n_peaks == 0:
        return XPSDeconvolution(
            element=element, n_components=0, components=[],
            envelope_r_squared=0, background_type=background_type,
        )

    # Build initial guesses and bounds
    p0 = []
    lower = []
    upper = []

    for pos in initial_positions:
        amp_guess = float(np.interp(pos, be[::-1] if be[0] > be[-1] else be,
                                    y[::-1] if be[0] > be[-1] else y))
        amp_guess = max(amp_guess, y.max() * 0.1)

        p0.extend([amp_guess, pos, 1.2, gl_fraction])
        lower.extend([0, pos - 2.0, min_fwhm, 0.0])
        upper.extend([y.max() * 3, pos + 2.0, max_fwhm, 1.0])

    try:
        popt, pcov = curve_fit(multi_gl, be, y, p0=p0,
                               bounds=(lower, upper), maxfev=20000)

        envelope = multi_gl(be, *popt)
        ss_res = np.sum((y - envelope) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        components = []
        component_curves = []
        ref_components = XPS_REFERENCES.get(element, {}).get("components", [])

        for i in range(n_peaks):
            amp = popt[4 * i]
            center = popt[4 * i + 1]
            sigma = popt[4 * i + 2]
            frac = popt[4 * i + 3]

            curve = gl_peak(be, amp, center, sigma, frac)
            area = float(np.trapezoid(curve, be))
            fwhm = float(sigma)  # GL sigma is approximately FWHM

            # Find assignment from reference
            assignment = "Unknown"
            for ref in ref_components:
                if ref["be_range"][0] <= center <= ref["be_range"][1]:
                    assignment = ref["name"]
                    break

            components.append(XPSPeakFit(
                center_ev=float(center),
                intensity=float(amp),
                fwhm_ev=fwhm,
                area=area,
                profile="GL",
                assignment=assignment,
                r_squared=float(r_sq),
                params={"amp": float(amp), "center": float(center),
                        "sigma": float(sigma), "gl_fraction": float(frac)},
            ))
            component_curves.append(curve.tolist())

        return XPSDeconvolution(
            element=element,
            n_components=n_peaks,
            components=components,
            envelope_r_squared=float(r_sq),
            background_type=background_type,
            binding_energy=be.tolist(),
            raw_intensity=raw.tolist(),
            background=bg.tolist(),
            envelope=(envelope + bg).tolist(),
            component_curves=component_curves,
        )

    except (RuntimeError, ValueError) as e:
        return XPSDeconvolution(
            element=element, n_components=0, components=[],
            envelope_r_squared=0, background_type=background_type,
            binding_energy=be.tolist(), raw_intensity=raw.tolist(),
            background=bg.tolist(),
        )


def quantify_components(deconv: XPSDeconvolution):
    """
    Calculate relative atomic percentages from peak areas.
    Returns list of dicts with component name, area, and relative %.
    """
    total_area = sum(c.area for c in deconv.components)
    if total_area == 0:
        return []

    result = []
    for c in deconv.components:
        result.append({
            "component": c.assignment,
            "center_ev": c.center_ev,
            "fwhm_ev": c.fwhm_ev,
            "area": c.area,
            "relative_pct": 100.0 * c.area / total_area,
        })

    return sorted(result, key=lambda x: -x["relative_pct"])


# ---------------------------------------------------------------------------
# Full XPS analysis pipeline
# ---------------------------------------------------------------------------
def full_xps_analysis(binding_energy, intensity, element,
                      background_type="shirley", gl_fraction=0.3):
    """
    Run complete XPS analysis: background subtraction, deconvolution,
    chemical state assignment, quantification.
    """
    deconv = deconvolve_xps(
        binding_energy, intensity, element,
        background_type=background_type,
        gl_fraction=gl_fraction,
    )

    quant = quantify_components(deconv)

    return {
        "element": element,
        "n_components": deconv.n_components,
        "envelope_r_squared": deconv.envelope_r_squared,
        "background_type": deconv.background_type,
        "components": [asdict(c) for c in deconv.components],
        "quantification": quant,
        "deconvolution": deconv,
    }
