"""
XRD Analysis Module
====================
Peak detection, fitting (Gaussian/Lorentzian/Voigt/PseudoVoigt),
d-spacing calculation, crystallite size estimation (Scherrer),
and phase identification for MXene characterization.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PeakResult:
    """Result of fitting a single XRD peak."""
    center_2theta: float
    intensity: float
    fwhm: float
    area: float
    d_spacing: float
    profile: str  # gaussian, lorentzian, voigt, pseudo_voigt
    r_squared: float
    params: dict = field(default_factory=dict)
    miller_index: str = ""
    phase: str = ""


@dataclass
class ScherrerResult:
    """Crystallite size from Scherrer equation."""
    peak_2theta: float
    fwhm_rad: float
    crystallite_size_nm: float
    k_factor: float
    wavelength: float


@dataclass
class PhaseMatch:
    """Result of matching a detected peak to a reference phase."""
    detected_2theta: float
    reference_2theta: float
    phase_name: str
    miller_index: str
    delta_2theta: float
    confidence: float  # 0-1


# ---------------------------------------------------------------------------
# Peak profile functions
# ---------------------------------------------------------------------------
def gaussian(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, amp, center, gamma):
    return amp * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def pseudo_voigt(x, amp, center, sigma, eta):
    """Pseudo-Voigt: linear combination of Gaussian and Lorentzian."""
    eta = np.clip(eta, 0, 1)
    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    l = sigma ** 2 / ((x - center) ** 2 + sigma ** 2)
    return amp * (eta * l + (1 - eta) * g)


PROFILE_FUNCS = {
    "gaussian": (gaussian, ["amp", "center", "sigma"]),
    "lorentzian": (lorentzian, ["amp", "center", "gamma"]),
    "pseudo_voigt": (pseudo_voigt, ["amp", "center", "sigma", "eta"]),
}


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------
def baseline_als(y, lam=1e6, p=0.01, n_iter=10):
    """Asymmetric Least Squares baseline estimation (Eilers & Boelens 2005)."""
    from scipy.sparse import diags, spdiags
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def simple_baseline(two_theta, intensity, percentile=10, window=101):
    """Simple rolling-minimum baseline."""
    from scipy.ndimage import minimum_filter1d
    baseline = minimum_filter1d(intensity, size=window)
    # Smooth it
    if len(baseline) > window:
        baseline = savgol_filter(baseline, min(window, len(baseline) - 1 if len(baseline) % 2 == 0 else len(baseline)), 2)
    return baseline


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------
def detect_peaks(two_theta, intensity, prominence=50, distance=10,
                 height_pct=5, smooth_window=5):
    """
    Detect peaks in XRD pattern.

    Parameters
    ----------
    two_theta : array-like
    intensity : array-like
    prominence : float - minimum prominence for peak detection
    distance : int - minimum distance between peaks (in data points)
    height_pct : float - minimum height as percentage of max intensity
    smooth_window : int - Savitzky-Golay smoothing window (odd, 0=off)

    Returns
    -------
    peak_positions : array of 2-theta values
    peak_indices : array of indices
    properties : dict from scipy.signal.find_peaks
    """
    y = intensity.copy()
    if smooth_window > 2:
        if smooth_window % 2 == 0:
            smooth_window += 1
        y = savgol_filter(y, smooth_window, 2)

    min_height = y.max() * (height_pct / 100.0)

    indices, props = find_peaks(
        y,
        prominence=prominence,
        distance=distance,
        height=min_height,
    )

    positions = two_theta[indices]
    return positions, indices, props


# ---------------------------------------------------------------------------
# Peak fitting
# ---------------------------------------------------------------------------
def fit_peak(two_theta, intensity, center_guess, window_deg=1.0,
             profile="pseudo_voigt"):
    """
    Fit a single peak with the specified profile.

    Parameters
    ----------
    two_theta : full 2-theta array
    intensity : full intensity array
    center_guess : approximate 2-theta of peak center
    window_deg : half-width of fitting window in degrees
    profile : 'gaussian', 'lorentzian', or 'pseudo_voigt'

    Returns
    -------
    PeakResult or None if fitting fails
    """
    mask = (two_theta >= center_guess - window_deg) & \
           (two_theta <= center_guess + window_deg)
    x = two_theta[mask]
    y = intensity[mask]

    if len(x) < 5:
        return None

    func, param_names = PROFILE_FUNCS[profile]
    amp_guess = y.max()
    sigma_guess = 0.15  # typical FWHM/2.355 for XRD

    try:
        if profile == "gaussian":
            p0 = [amp_guess, center_guess, sigma_guess]
            bounds = ([0, center_guess - window_deg, 0.01],
                      [amp_guess * 3, center_guess + window_deg, 2.0])
        elif profile == "lorentzian":
            p0 = [amp_guess, center_guess, sigma_guess]
            bounds = ([0, center_guess - window_deg, 0.01],
                      [amp_guess * 3, center_guess + window_deg, 2.0])
        elif profile == "pseudo_voigt":
            p0 = [amp_guess, center_guess, sigma_guess, 0.5]
            bounds = ([0, center_guess - window_deg, 0.01, 0.0],
                      [amp_guess * 3, center_guess + window_deg, 2.0, 1.0])
        else:
            return None

        popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=5000)

        # Calculate R-squared
        y_pred = func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Extract parameters
        center = popt[1]
        amp = popt[0]
        sigma = popt[2]

        # Calculate FWHM
        if profile == "gaussian":
            fwhm = 2.355 * sigma
        elif profile == "lorentzian":
            fwhm = 2 * sigma
        elif profile == "pseudo_voigt":
            fwhm = 2.355 * sigma  # approximation
        else:
            fwhm = 2 * sigma

        # Area under the curve
        area = np.trapezoid(y_pred, x)

        # d-spacing from Bragg's law
        wavelength = 1.54056  # Cu Ka1
        d = wavelength / (2 * np.sin(np.radians(center / 2)))

        params_dict = {name: float(val) for name, val in zip(param_names, popt)}

        return PeakResult(
            center_2theta=float(center),
            intensity=float(amp),
            fwhm=float(fwhm),
            area=float(area),
            d_spacing=float(d),
            profile=profile,
            r_squared=float(r_sq),
            params=params_dict,
        )
    except (RuntimeError, ValueError):
        return None


def fit_all_peaks(two_theta, intensity, peak_positions, window_deg=1.0,
                  profile="pseudo_voigt"):
    """Fit all detected peaks and return list of PeakResult."""
    results = []
    for pos in peak_positions:
        result = fit_peak(two_theta, intensity, pos, window_deg, profile)
        if result and result.r_squared > 0.5:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Multi-peak fitting (simultaneous)
# ---------------------------------------------------------------------------
def multi_gaussian(x, *params):
    """Sum of N Gaussians. params = [amp1, center1, sigma1, amp2, ...]"""
    n_peaks = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        amp, center, sigma = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        y += amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return y


def fit_multi_peak(two_theta, intensity, center_guesses, window_deg=2.0):
    """
    Simultaneously fit multiple overlapping peaks as sum of Gaussians.
    Useful for deconvolving overlapping XPS or XRD peaks.

    Returns list of PeakResult.
    """
    x_min = min(center_guesses) - window_deg
    x_max = max(center_guesses) + window_deg
    mask = (two_theta >= x_min) & (two_theta <= x_max)
    x = two_theta[mask]
    y = intensity[mask]

    if len(x) < 5:
        return []

    n_peaks = len(center_guesses)
    p0 = []
    lower = []
    upper = []

    for cg in center_guesses:
        amp_guess = y.max()
        p0.extend([amp_guess, cg, 0.15])
        lower.extend([0, cg - 1.0, 0.01])
        upper.extend([amp_guess * 5, cg + 1.0, 3.0])

    try:
        popt, pcov = curve_fit(multi_gaussian, x, y, p0=p0,
                               bounds=(lower, upper), maxfev=10000)

        y_pred = multi_gaussian(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results = []
        for i in range(n_peaks):
            amp = popt[3 * i]
            center = popt[3 * i + 1]
            sigma = popt[3 * i + 2]
            fwhm = 2.355 * sigma

            single_peak = gaussian(x, amp, center, sigma)
            area = np.trapezoid(single_peak, x)

            wavelength = 1.54056
            d = wavelength / (2 * np.sin(np.radians(center / 2)))

            results.append(PeakResult(
                center_2theta=float(center),
                intensity=float(amp),
                fwhm=float(fwhm),
                area=float(area),
                d_spacing=float(d),
                profile="gaussian",
                r_squared=float(r_sq),
                params={"amp": float(amp), "center": float(center), "sigma": float(sigma)},
            ))

        return results
    except (RuntimeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Scherrer crystallite size
# ---------------------------------------------------------------------------
def scherrer_size(two_theta_deg, fwhm_deg, wavelength=1.54056, k=0.9):
    """
    Estimate crystallite size using the Scherrer equation.

    L = K * lambda / (beta * cos(theta))

    Parameters
    ----------
    two_theta_deg : float - peak position in degrees (2-theta)
    fwhm_deg : float - FWHM in degrees (2-theta)
    wavelength : float - X-ray wavelength in Angstroms
    k : float - Scherrer constant (0.9 for spherical crystallites)

    Returns
    -------
    ScherrerResult with crystallite size in nm
    """
    theta_rad = np.radians(two_theta_deg / 2)
    beta_rad = np.radians(fwhm_deg)

    if beta_rad <= 0 or np.cos(theta_rad) <= 0:
        return ScherrerResult(
            peak_2theta=two_theta_deg,
            fwhm_rad=beta_rad,
            crystallite_size_nm=0.0,
            k_factor=k,
            wavelength=wavelength,
        )

    # Size in Angstroms, convert to nm
    size_angstrom = (k * wavelength) / (beta_rad * np.cos(theta_rad))
    size_nm = size_angstrom / 10.0

    return ScherrerResult(
        peak_2theta=two_theta_deg,
        fwhm_rad=float(beta_rad),
        crystallite_size_nm=float(size_nm),
        k_factor=k,
        wavelength=wavelength,
    )


# ---------------------------------------------------------------------------
# Phase identification
# ---------------------------------------------------------------------------
# Reference database for MXene-related phases
REFERENCE_DB = {
    "Ti3AlC2 (MAX)": [
        (9.52, "(002)"), (19.15, "(004)"), (33.97, "(100)"),
        (34.06, "(101)"), (36.77, "(102)"), (38.99, "(103)"),
        (39.05, "(104)"), (41.76, "(006)"), (48.49, "(105)"),
        (52.36, "(106)"), (56.46, "(110)"), (60.27, "(108)"),
        (65.60, "(112)"), (70.36, "(114)"), (74.02, "(200)"),
    ],
    "Ti3C2Tx (MXene)": [
        (6.60, "(002)"), (9.00, "(002)*"), (18.30, "(004)"),
        (27.50, "(006)"), (34.00, "(100)"), (36.80, "(008)"),
        (41.80, "(101)"), (60.50, "(110)"),
    ],
    "TiO2 (Anatase)": [
        (25.28, "(101)"), (37.80, "(004)"), (48.05, "(200)"),
        (53.89, "(105)"), (55.06, "(211)"), (62.69, "(204)"),
    ],
    "TiO2 (Rutile)": [
        (27.45, "(110)"), (36.09, "(101)"), (39.19, "(200)"),
        (41.23, "(111)"), (54.32, "(211)"), (56.64, "(220)"),
    ],
    "TiC": [
        (35.93, "(111)"), (41.73, "(200)"), (60.49, "(220)"),
        (72.41, "(311)"), (76.17, "(222)"),
    ],
    "Al2O3 (Corundum)": [
        (25.58, "(012)"), (35.15, "(104)"), (37.78, "(110)"),
        (43.36, "(113)"), (52.55, "(024)"), (57.50, "(116)"),
    ],
}


def identify_phases(detected_peaks, tolerance_deg=0.5,
                    reference_db=None):
    """
    Match detected peaks against reference database.

    Parameters
    ----------
    detected_peaks : list of float (2-theta positions)
    tolerance_deg : float - maximum allowed deviation in degrees
    reference_db : dict - custom reference database (uses built-in if None)

    Returns
    -------
    list of PhaseMatch
    """
    if reference_db is None:
        reference_db = REFERENCE_DB

    matches = []
    for det_pos in detected_peaks:
        best_match = None
        best_delta = tolerance_deg + 1

        for phase_name, ref_peaks in reference_db.items():
            for ref_pos, miller in ref_peaks:
                delta = abs(det_pos - ref_pos)
                if delta < best_delta:
                    best_delta = delta
                    confidence = max(0, 1 - delta / tolerance_deg)
                    best_match = PhaseMatch(
                        detected_2theta=float(det_pos),
                        reference_2theta=float(ref_pos),
                        phase_name=phase_name,
                        miller_index=miller,
                        delta_2theta=float(delta),
                        confidence=float(confidence),
                    )

        if best_match and best_match.confidence > 0:
            matches.append(best_match)

    return matches


def phase_summary(matches):
    """
    Summarize which phases are present based on peak matches.
    Returns dict of phase_name -> {count, avg_confidence, peaks}
    """
    phases = {}
    for m in matches:
        if m.phase_name not in phases:
            phases[m.phase_name] = {
                "count": 0,
                "total_confidence": 0,
                "peaks": [],
            }
        phases[m.phase_name]["count"] += 1
        phases[m.phase_name]["total_confidence"] += m.confidence
        phases[m.phase_name]["peaks"].append({
            "detected": m.detected_2theta,
            "reference": m.reference_2theta,
            "miller": m.miller_index,
            "confidence": m.confidence,
        })

    for phase in phases.values():
        phase["avg_confidence"] = phase["total_confidence"] / phase["count"]
        del phase["total_confidence"]

    return dict(sorted(phases.items(), key=lambda x: -x[1]["count"]))


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------
def full_xrd_analysis(two_theta, intensity, profile="pseudo_voigt",
                      prominence=50, distance=10, height_pct=5,
                      fit_window=1.0, smooth_window=5):
    """
    Run complete XRD analysis: detect peaks, fit them, identify phases,
    calculate crystallite sizes.

    Returns dict with all results.
    """
    # 1. Detect peaks
    positions, indices, props = detect_peaks(
        two_theta, intensity,
        prominence=prominence,
        distance=distance,
        height_pct=height_pct,
        smooth_window=smooth_window,
    )

    # 2. Fit peaks
    fitted = fit_all_peaks(two_theta, intensity, positions,
                           window_deg=fit_window, profile=profile)

    # 3. Phase identification
    matches = identify_phases(positions)
    phases = phase_summary(matches)

    # Annotate fitted peaks with phase info
    for peak in fitted:
        for match in matches:
            if abs(peak.center_2theta - match.detected_2theta) < 0.5:
                peak.phase = match.phase_name
                peak.miller_index = match.miller_index
                break

    # 4. Scherrer crystallite sizes
    scherrer_results = []
    for peak in fitted:
        if peak.fwhm > 0:
            sr = scherrer_size(peak.center_2theta, peak.fwhm)
            scherrer_results.append(sr)

    return {
        "peaks_detected": len(positions),
        "peak_positions": positions.tolist(),
        "fitted_peaks": [asdict(p) for p in fitted],
        "phases": phases,
        "phase_matches": [asdict(m) for m in matches],
        "scherrer": [asdict(s) for s in scherrer_results],
    }
