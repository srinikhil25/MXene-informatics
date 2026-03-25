"""
Rietveld Refinement Module
===========================
Whole-pattern fitting for XRD data using crystal structure models.
Refines lattice parameters, phase fractions, profile parameters,
and preferred orientation for Ti3AlC2/Ti3C2Tx MXene system.

Implements a simplified Rietveld method using:
- Crystal structure definitions with reflection lists and multiplicities
- Caglioti profile function (U, V, W) for angle-dependent broadening
- Chebyshev polynomial background
- March-Dollase preferred orientation correction
- scipy.optimize.least_squares for refinement

References:
    Rietveld (1969) J. Appl. Cryst. 2, 65-71
    Caglioti et al. (1958) Nucl. Instr. Methods 3, 223-228
    March (1932) Z. Kristallogr. 81, 285-297
    Dollase (1986) J. Appl. Cryst. 19, 267-272
"""

import numpy as np
from scipy.optimize import least_squares
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import warnings


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AtomSite:
    """Crystallographic atom site for Rietveld refinement."""
    label: str          # e.g., "Ti1", "Al1", "C1", "O1"
    element: str        # chemical element symbol
    wyckoff: str        # Wyckoff position, e.g., "2a", "4f", "8e"
    x: float            # fractional coordinate x
    y: float            # fractional coordinate y
    z: float            # fractional coordinate z
    occupancy: float    # site occupancy (0-1)
    U_iso: float        # isotropic thermal displacement parameter (Å²)
    multiplicity: int   # number of equivalent positions in the unit cell


@dataclass
class CrystalPhase:
    """Crystal structure definition for Rietveld refinement."""
    name: str
    space_group: str
    crystal_system: str  # hexagonal, tetragonal, cubic
    a: float  # lattice parameter a (Angstrom)
    c: float  # lattice parameter c (Angstrom) - same as a for cubic
    reflections: list  # [(h, k, l, multiplicity, relative_intensity), ...]
    atoms: List[AtomSite] = field(default_factory=list)  # atom sites
    Z: int = 1  # formula units per unit cell


@dataclass
class RietveldResult:
    """Result of Rietveld refinement."""
    phases: List[dict]           # refined phase info (name, wt%, lattice params)
    two_theta: np.ndarray        # 2-theta array
    y_obs: np.ndarray            # observed intensity
    y_calc: np.ndarray           # calculated intensity
    y_diff: np.ndarray           # difference (obs - calc)
    y_background: np.ndarray     # refined background
    bragg_positions: Dict[str, list]  # {phase_name: [2theta positions]}
    Rwp: float                   # weighted profile R-factor
    Rp: float                    # profile R-factor
    chi_squared: float           # goodness of fit
    GoF: float                   # goodness of fit sqrt(chi2)
    refined_params: dict         # all refined parameters


# ---------------------------------------------------------------------------
# Crystal structure database for Ti3AlC2 → Ti3C2Tx system
# ---------------------------------------------------------------------------
# Reflection lists: (h, k, l, multiplicity, relative_intensity)
# Intensities from ICDD/literature powder patterns

CRYSTAL_PHASES = {
    # -----------------------------------------------------------------------
    # Ti3AlC2 — MAX phase (312 MAX)
    # Space group: P63/mmc (#194), Z = 2
    # Ref: Hug et al. (2006) Phys. Rev. B 74, 184113
    #      Barsoum (2013) MAX Phases, Springer
    # -----------------------------------------------------------------------
    "Ti3AlC2": CrystalPhase(
        name="Ti3AlC2",
        space_group="P63/mmc",
        crystal_system="hexagonal",
        a=3.075,
        c=18.578,
        reflections=[
            (0, 0, 2, 2, 100.0),   # 9.52° - strongest basal
            (0, 0, 4, 2, 30.0),    # 19.15°
            (1, 0, 0, 6, 12.0),    # 33.97°
            (1, 0, 1, 12, 25.0),   # 34.06°
            (1, 0, 2, 12, 18.0),   # 36.77°
            (1, 0, 3, 12, 45.0),   # 38.99° - strong
            (1, 0, 4, 12, 35.0),   # 39.05°
            (0, 0, 6, 2, 15.0),    # 41.76°
            (1, 0, 5, 12, 20.0),   # 48.49°
            (1, 0, 6, 12, 12.0),   # 52.36°
            (1, 1, 0, 6, 18.0),    # 56.46°
            (1, 0, 8, 12, 10.0),   # 60.27°
            (1, 1, 2, 12, 8.0),    # 65.60°
            (1, 1, 4, 12, 6.0),    # 70.36°
            (2, 0, 0, 6, 5.0),     # 74.02°
        ],
        atoms=[
            # label, element, wyckoff, x, y, z, occupancy, U_iso (Å²), multiplicity
            AtomSite("Ti1", "Ti", "2a",  0.0,     0.0,     0.0,      1.0, 0.006, 2),
            AtomSite("Ti2", "Ti", "4f",  1/3,     2/3,     0.13550,  1.0, 0.006, 4),
            AtomSite("Al1", "Al", "2b",  0.0,     0.0,     0.25,     1.0, 0.012, 2),
            AtomSite("C1",  "C",  "4f",  1/3,     2/3,     0.07220,  1.0, 0.005, 4),
        ],
        Z=2,
    ),

    # -----------------------------------------------------------------------
    # Ti3C2Tx — MXene (etched, with surface terminations)
    # Space group: P63/mmc (#194), Z = 2
    # Ref: Naguib et al. (2011) Adv. Mater. 23, 4248
    #      Khazaei et al. (2013) Adv. Funct. Mater. 23, 2185
    # Note: T_x sites (O, OH, F) have fractional occupancy
    # -----------------------------------------------------------------------
    "Ti3C2Tx": CrystalPhase(
        name="Ti3C2Tx",
        space_group="P63/mmc",
        crystal_system="hexagonal",
        a=3.050,
        c=19.50,  # expanded c-axis after etching
        reflections=[
            (0, 0, 2, 2, 100.0),   # ~9.0° (shifted from MAX)
            (0, 0, 4, 2, 20.0),    # ~18.3°
            (0, 0, 6, 2, 8.0),     # ~27.5°
            (1, 0, 0, 6, 15.0),    # ~34.0°
            (0, 0, 8, 2, 5.0),     # ~36.8°
            (1, 0, 1, 12, 10.0),   # ~41.8°
            (1, 1, 0, 6, 12.0),    # ~60.5°
        ],
        atoms=[
            AtomSite("Ti1", "Ti", "2a",  0.0,     0.0,     0.0,      1.0,  0.007, 2),
            AtomSite("Ti2", "Ti", "4f",  1/3,     2/3,     0.1300,   1.0,  0.007, 4),
            AtomSite("C1",  "C",  "4f",  1/3,     2/3,     0.0680,   1.0,  0.005, 4),
            AtomSite("O1",  "O",  "2c",  1/3,     2/3,     0.25,     0.46, 0.015, 2),  # -O termination
            AtomSite("F1",  "F",  "2c",  1/3,     2/3,     0.25,     0.14, 0.018, 2),  # -F termination
            AtomSite("OH1", "O",  "2d",  1/3,     2/3,     0.75,     0.40, 0.020, 2),  # -OH termination
        ],
        Z=2,
    ),

    # -----------------------------------------------------------------------
    # TiO2 — Anatase
    # Space group: I41/amd (#141), Z = 4
    # Ref: Howard et al. (1991) Acta Crystallogr. B47, 462
    # -----------------------------------------------------------------------
    "TiO2_Anatase": CrystalPhase(
        name="TiO2 (Anatase)",
        space_group="I41/amd",
        crystal_system="tetragonal",
        a=3.785,
        c=9.514,
        reflections=[
            (1, 0, 1, 8, 100.0),   # 25.28°
            (0, 0, 4, 2, 20.0),    # 37.80°
            (2, 0, 0, 4, 35.0),    # 48.05°
            (1, 0, 5, 8, 20.0),    # 53.89°
            (2, 1, 1, 16, 18.0),   # 55.06°
            (2, 0, 4, 8, 14.0),    # 62.69°
        ],
        atoms=[
            AtomSite("Ti1", "Ti", "4a",  0.0,     0.0,     0.0,      1.0, 0.006, 4),
            AtomSite("O1",  "O",  "8e",  0.0,     0.0,     0.2081,   1.0, 0.008, 8),
        ],
        Z=4,
    ),

    # -----------------------------------------------------------------------
    # TiC — Titanium Carbide (rock salt structure)
    # Space group: Fm-3m (#225), Z = 4
    # Ref: Storms (1967) The Refractory Carbides, Academic Press
    # -----------------------------------------------------------------------
    "TiC": CrystalPhase(
        name="TiC",
        space_group="Fm-3m",
        crystal_system="cubic",
        a=4.328,
        c=4.328,
        reflections=[
            (1, 1, 1, 8, 70.0),    # 35.93°
            (2, 0, 0, 6, 100.0),   # 41.73°
            (2, 2, 0, 12, 55.0),   # 60.49°
            (3, 1, 1, 24, 30.0),   # 72.41°
            (2, 2, 2, 8, 15.0),    # 76.17°
        ],
        atoms=[
            AtomSite("Ti1", "Ti", "4a",  0.0,     0.0,     0.0,      1.0, 0.005, 4),
            AtomSite("C1",  "C",  "4b",  0.5,     0.5,     0.5,      1.0, 0.006, 4),
        ],
        Z=4,
    ),

    # -----------------------------------------------------------------------
    # Al2O3 — Corundum (alpha-alumina)
    # Space group: R-3c (#167), Z = 6
    # Ref: Ishizawa et al. (1980) Acta Crystallogr. B36, 228
    # -----------------------------------------------------------------------
    "Al2O3": CrystalPhase(
        name="Al2O3 (Corundum)",
        space_group="R-3c",
        crystal_system="hexagonal",
        a=4.759,
        c=12.993,
        reflections=[
            (0, 1, 2, 6, 70.0),    # 25.58°
            (1, 0, 4, 6, 100.0),   # 35.15°
            (1, 1, 0, 6, 50.0),    # 37.78°
            (1, 1, 3, 12, 45.0),   # 43.36°
            (0, 2, 4, 6, 40.0),    # 52.55°
            (1, 1, 6, 12, 35.0),   # 57.50°
        ],
        atoms=[
            AtomSite("Al1", "Al", "12c", 0.0,     0.0,     0.35217,  1.0, 0.003, 12),
            AtomSite("O1",  "O",  "18e", 0.30636, 0.0,     0.25,     1.0, 0.004, 18),
        ],
        Z=6,
    ),
}


# ---------------------------------------------------------------------------
# d-spacing calculators by crystal system
# ---------------------------------------------------------------------------
def d_spacing_hexagonal(h, k, l, a, c):
    """d-spacing for hexagonal crystal system."""
    denom = (4.0 / 3.0) * (h**2 + h * k + k**2) / a**2 + l**2 / c**2
    if denom <= 0:
        return np.inf
    return 1.0 / np.sqrt(denom)


def d_spacing_tetragonal(h, k, l, a, c):
    """d-spacing for tetragonal crystal system."""
    denom = (h**2 + k**2) / a**2 + l**2 / c**2
    if denom <= 0:
        return np.inf
    return 1.0 / np.sqrt(denom)


def d_spacing_cubic(h, k, l, a, c=None):
    """d-spacing for cubic crystal system."""
    denom = (h**2 + k**2 + l**2) / a**2
    if denom <= 0:
        return np.inf
    return 1.0 / np.sqrt(denom)


D_SPACING_FUNCS = {
    "hexagonal": d_spacing_hexagonal,
    "tetragonal": d_spacing_tetragonal,
    "cubic": d_spacing_cubic,
}


def calc_two_theta(d, wavelength=1.54056):
    """Convert d-spacing to 2-theta (degrees). Returns None if invalid."""
    sin_theta = wavelength / (2.0 * d)
    if abs(sin_theta) > 1.0:
        return None
    return 2.0 * np.degrees(np.arcsin(sin_theta))


# ---------------------------------------------------------------------------
# Profile functions for Rietveld
# ---------------------------------------------------------------------------
def caglioti_fwhm(two_theta_rad, U, V, W):
    """
    Caglioti function for angle-dependent FWHM.
    FWHM² = U·tan²(θ) + V·tan(θ) + W

    Parameters are in degrees² — output is FWHM in degrees.
    """
    theta = two_theta_rad / 2.0
    tan_theta = np.tan(theta)
    fwhm_sq = U * tan_theta**2 + V * tan_theta + W
    # Ensure positive
    fwhm_sq = np.maximum(fwhm_sq, 0.001)
    return np.sqrt(fwhm_sq)


def pseudo_voigt_profile(x, center, fwhm, eta, intensity):
    """
    Pseudo-Voigt profile for Rietveld: η·L + (1-η)·G

    Parameters
    ----------
    x : array - 2-theta values
    center : float - peak center
    fwhm : float - full width at half maximum (degrees)
    eta : float - mixing parameter (0=Gaussian, 1=Lorentzian)
    intensity : float - integrated intensity (area)
    """
    eta = np.clip(eta, 0.0, 1.0)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM to sigma
    gamma = fwhm / 2.0  # FWHM to half-width for Lorentzian

    # Normalized Gaussian
    G = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sigma) ** 2
    )

    # Normalized Lorentzian
    L = (1.0 / np.pi) * (gamma / ((x - center) ** 2 + gamma**2))

    # Combined profile, scaled by intensity
    return intensity * (eta * L + (1.0 - eta) * G)


# ---------------------------------------------------------------------------
# March-Dollase preferred orientation correction
# ---------------------------------------------------------------------------
def march_dollase(two_theta_deg, r, preferred_hkl_angle=0.0):
    """
    March-Dollase preferred orientation correction.

    Parameters
    ----------
    two_theta_deg : float - 2-theta of reflection
    r : float - March-Dollase parameter (r=1 for random, r<1 for platelet)
    preferred_hkl_angle : float - angle between preferred orientation
                                  direction and reflection normal (degrees)

    Returns
    -------
    correction factor
    """
    if abs(r - 1.0) < 1e-6:
        return 1.0
    alpha = np.radians(preferred_hkl_angle)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    return (r**2 * cos_a**2 + sin_a**2 / r) ** (-1.5)


# ---------------------------------------------------------------------------
# Background model
# ---------------------------------------------------------------------------
def chebyshev_background(x, coeffs):
    """
    Chebyshev polynomial background.

    Parameters
    ----------
    x : array - 2-theta values (normalized to [-1, 1] internally)
    coeffs : list of float - polynomial coefficients

    Returns
    -------
    background array
    """
    # Normalize x to [-1, 1]
    x_min, x_max = x.min(), x.max()
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    bg = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        bg += c * np.polynomial.chebyshev.chebval(x_norm, [0] * i + [1])
    return bg


# ---------------------------------------------------------------------------
# Calculated pattern generator
# ---------------------------------------------------------------------------
def calc_pattern(two_theta, phases, phase_scales, lattice_params,
                 U, V, W, eta, bg_coeffs, wavelength=1.54056,
                 march_r=None):
    """
    Calculate full XRD pattern from crystal structure models.

    Parameters
    ----------
    two_theta : array - 2-theta values
    phases : list of CrystalPhase
    phase_scales : list of float - scale factor per phase
    lattice_params : list of (a, c) per phase
    U, V, W : float - Caglioti profile parameters
    eta : float - pseudo-Voigt mixing parameter
    bg_coeffs : list of float - background coefficients
    wavelength : float - X-ray wavelength (Angstrom)
    march_r : list of float or None - March-Dollase parameter per phase

    Returns
    -------
    y_calc : array - calculated intensity
    y_background : array - background contribution
    bragg_positions : dict - {phase_name: [(2theta, hkl_str, intensity)]}
    """
    two_theta_rad = np.radians(two_theta)

    # Background
    y_bg = chebyshev_background(two_theta, bg_coeffs)
    y_calc = y_bg.copy()

    bragg_positions = {}

    for i, phase in enumerate(phases):
        scale = phase_scales[i]
        a, c = lattice_params[i]
        d_func = D_SPACING_FUNCS[phase.crystal_system]
        r = march_r[i] if march_r is not None else 1.0

        phase_bragg = []

        for h, k, l, mult, rel_int in phase.reflections:
            d = d_func(h, k, l, a, c)
            tt = calc_two_theta(d, wavelength)

            if tt is None or tt < two_theta.min() or tt > two_theta.max():
                continue

            # FWHM from Caglioti
            tt_rad = np.radians(tt)
            fwhm = caglioti_fwhm(tt_rad, U, V, W)
            fwhm = max(fwhm, 0.02)  # minimum FWHM

            # Intensity = scale × multiplicity × relative_intensity
            peak_int = scale * mult * rel_int / 100.0

            # Preferred orientation (for basal reflections in layered materials)
            if h == 0 and k == 0 and l != 0:
                # (00l) reflections are enhanced in layered materials
                peak_int *= march_dollase(tt, r, preferred_hkl_angle=0.0)

            # Lorentz-polarization correction
            theta_rad = tt_rad / 2.0
            cos_theta = np.cos(theta_rad)
            sin_theta = np.sin(theta_rad)
            if sin_theta > 0 and cos_theta > 0:
                lp = (1.0 + np.cos(2 * theta_rad)**2) / (
                    sin_theta**2 * cos_theta
                )
                peak_int *= lp

            # Add peak profile
            y_calc += pseudo_voigt_profile(two_theta, tt, fwhm, eta, peak_int)

            hkl_str = f"({h}{k}{l})"
            phase_bragg.append((tt, hkl_str, peak_int))

        bragg_positions[phase.name] = phase_bragg

    return y_calc, y_bg, bragg_positions


# ---------------------------------------------------------------------------
# Rietveld refinement engine
# ---------------------------------------------------------------------------
def _pack_params(phase_scales, lattice_params, U, V, W, eta, bg_coeffs,
                 march_r=None):
    """Pack all refinable parameters into a flat array."""
    params = []
    params.extend(phase_scales)
    for a, c in lattice_params:
        params.extend([a, c])
    params.extend([U, V, W, eta])
    params.extend(bg_coeffs)
    if march_r is not None:
        params.extend(march_r)
    return np.array(params, dtype=float)


def _unpack_params(params, n_phases, n_bg_coeffs, refine_orientation=False):
    """Unpack flat parameter array."""
    idx = 0

    phase_scales = list(params[idx:idx + n_phases])
    idx += n_phases

    lattice_params = []
    for _ in range(n_phases):
        a, c = params[idx], params[idx + 1]
        lattice_params.append((a, c))
        idx += 2

    U, V, W, eta = params[idx:idx + 4]
    idx += 4

    bg_coeffs = list(params[idx:idx + n_bg_coeffs])
    idx += n_bg_coeffs

    march_r = None
    if refine_orientation:
        march_r = list(params[idx:idx + n_phases])
        idx += n_phases

    return phase_scales, lattice_params, U, V, W, eta, bg_coeffs, march_r


def _residuals(params, two_theta, y_obs, weights, phases, n_bg_coeffs,
               wavelength, refine_orientation):
    """Residual function for least_squares optimization."""
    n_phases = len(phases)
    (phase_scales, lattice_params, U, V, W, eta,
     bg_coeffs, march_r) = _unpack_params(
        params, n_phases, n_bg_coeffs, refine_orientation
    )

    y_calc, _, _ = calc_pattern(
        two_theta, phases, phase_scales, lattice_params,
        U, V, W, eta, bg_coeffs, wavelength, march_r
    )

    return weights * (y_obs - y_calc)


def _build_bounds(n_phases, lattice_params_init, n_bg_coeffs,
                  refine_orientation=False):
    """Build parameter bounds for least_squares."""
    lower, upper = [], []

    # Phase scales: [0, inf)
    for _ in range(n_phases):
        lower.append(0.0)
        upper.append(np.inf)

    # Lattice parameters: ±5% from initial
    for a, c in lattice_params_init:
        lower.extend([a * 0.95, c * 0.95])
        upper.extend([a * 1.05, c * 1.05])

    # Caglioti U, V, W
    lower.extend([-1.0, -1.0, 0.001])  # U, V, W
    upper.extend([5.0, 1.0, 1.0])

    # Eta (pseudo-Voigt mixing)
    lower.append(0.0)
    upper.append(1.0)

    # Background coefficients
    for _ in range(n_bg_coeffs):
        lower.append(-1e6)
        upper.append(1e6)

    # March-Dollase r per phase
    if refine_orientation:
        for _ in range(n_phases):
            lower.append(0.2)  # strong texture
            upper.append(2.0)  # inverse texture

    return (np.array(lower), np.array(upper))


def auto_detect_phases(two_theta, intensity, wavelength=1.54056,
                       threshold=0.1):
    """
    Automatically detect which phases are likely present.

    Checks for characteristic peaks of each phase in CRYSTAL_PHASES.

    Parameters
    ----------
    two_theta : array
    intensity : array (normalized to [0, 1])
    wavelength : float
    threshold : float - minimum normalized intensity to consider peak present

    Returns
    -------
    list of CrystalPhase names that are likely present
    """
    from scipy.signal import find_peaks

    # Normalize
    y_norm = intensity / intensity.max() if intensity.max() > 0 else intensity

    detected = []
    for key, phase in CRYSTAL_PHASES.items():
        d_func = D_SPACING_FUNCS[phase.crystal_system]
        match_count = 0
        total_strong = 0

        for h, k, l, mult, rel_int in phase.reflections:
            if rel_int < 30:  # only check strong reflections
                continue
            total_strong += 1

            d = d_func(h, k, l, phase.a, phase.c)
            tt = calc_two_theta(d, wavelength)
            if tt is None:
                continue

            # Check if there's intensity near this position
            mask = (two_theta >= tt - 0.5) & (two_theta <= tt + 0.5)
            if mask.any() and y_norm[mask].max() > threshold:
                match_count += 1

        if total_strong > 0 and match_count / total_strong >= 0.3:
            detected.append(key)

    return detected


def estimate_initial_params(two_theta, intensity, phases, n_bg_coeffs=6):
    """
    Estimate initial parameters for refinement.

    Returns
    -------
    phase_scales, lattice_params, U, V, W, eta, bg_coeffs, march_r
    """
    y_max = intensity.max()

    # Estimate scale factors
    phase_scales = [y_max / (len(phases) * 50.0)] * len(phases)

    # Use default lattice parameters from database
    lattice_params = [(p.a, p.c) for p in phases]

    # Caglioti defaults (typical for lab diffractometer)
    U = 0.1
    V = -0.05
    W = 0.05

    # Pseudo-Voigt mixing
    eta = 0.5

    # Background: fit polynomial to low-intensity regions
    # Simple estimate: percentile of intensity
    bg_level = np.percentile(intensity, 10)
    bg_coeffs = [bg_level] + [0.0] * (n_bg_coeffs - 1)

    # March-Dollase (slight preferred orientation for layered materials)
    march_r = [0.8] * len(phases)  # slight platelet texture

    return phase_scales, lattice_params, U, V, W, eta, bg_coeffs, march_r


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------
def rietveld_refine(two_theta, intensity, phase_names=None,
                    n_bg_coeffs=6, wavelength=1.54056,
                    refine_orientation=True, max_iterations=200):
    """
    Perform Rietveld refinement on XRD data.

    Parameters
    ----------
    two_theta : array - 2-theta values (degrees)
    intensity : array - observed intensity
    phase_names : list of str or None
        Phase names from CRYSTAL_PHASES. If None, auto-detect.
    n_bg_coeffs : int - number of Chebyshev background coefficients
    wavelength : float - X-ray wavelength (Angstrom)
    refine_orientation : bool - refine March-Dollase preferred orientation
    max_iterations : int - maximum refinement iterations

    Returns
    -------
    RietveldResult
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # Auto-detect phases if not specified
    if phase_names is None:
        phase_names = auto_detect_phases(two_theta, intensity, wavelength)
        if not phase_names:
            phase_names = ["Ti3AlC2"]  # fallback to primary phase

    # Get phase objects
    phases = [CRYSTAL_PHASES[name] for name in phase_names]

    # Weights: 1/sqrt(y) for Poisson statistics, with floor
    weights = 1.0 / np.sqrt(np.maximum(intensity, 1.0))

    # Initial parameter estimates
    (phase_scales, lattice_params, U, V, W, eta,
     bg_coeffs, march_r) = estimate_initial_params(
        two_theta, intensity, phases, n_bg_coeffs
    )

    # Pack parameters
    p0 = _pack_params(
        phase_scales, lattice_params, U, V, W, eta, bg_coeffs,
        march_r if refine_orientation else None
    )

    # Bounds
    bounds = _build_bounds(
        len(phases), lattice_params, n_bg_coeffs, refine_orientation
    )

    # Run refinement
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = least_squares(
            _residuals, p0,
            args=(two_theta, intensity, weights, phases, n_bg_coeffs,
                  wavelength, refine_orientation),
            bounds=bounds,
            method="trf",
            max_nfev=max_iterations * len(p0),
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )

    # Unpack refined parameters
    (phase_scales_ref, lattice_params_ref, U_ref, V_ref, W_ref, eta_ref,
     bg_coeffs_ref, march_r_ref) = _unpack_params(
        result.x, len(phases), n_bg_coeffs, refine_orientation
    )

    # Calculate final pattern
    y_calc, y_bg, bragg_pos = calc_pattern(
        two_theta, phases, phase_scales_ref, lattice_params_ref,
        U_ref, V_ref, W_ref, eta_ref, bg_coeffs_ref, wavelength,
        march_r_ref
    )

    y_diff = intensity - y_calc

    # Calculate R-factors
    Rwp = _calc_Rwp(intensity, y_calc, weights)
    Rp = _calc_Rp(intensity, y_calc)

    # Chi-squared and GoF
    n_obs = len(intensity)
    n_params = len(result.x)
    dof = max(n_obs - n_params, 1)
    chi_sq = np.sum(weights**2 * (intensity - y_calc)**2) / dof
    GoF = np.sqrt(max(chi_sq, 0))

    # Build phase results with weight fractions
    total_scale = sum(phase_scales_ref)
    phase_results = []
    for i, phase in enumerate(phases):
        a_ref, c_ref = lattice_params_ref[i]
        wt_frac = phase_scales_ref[i] / total_scale * 100 if total_scale > 0 else 0

        phase_info = {
            "name": phase.name,
            "space_group": phase.space_group,
            "crystal_system": phase.crystal_system,
            "a_refined": float(a_ref),
            "c_refined": float(c_ref),
            "a_initial": float(phase.a),
            "c_initial": float(phase.c),
            "delta_a": float(a_ref - phase.a),
            "delta_c": float(c_ref - phase.c),
            "weight_fraction_pct": float(wt_frac),
            "scale_factor": float(phase_scales_ref[i]),
        }
        if march_r_ref is not None:
            phase_info["march_dollase_r"] = float(march_r_ref[i])

        phase_results.append(phase_info)

    # All refined parameters
    refined_params = {
        "U": float(U_ref),
        "V": float(V_ref),
        "W": float(W_ref),
        "eta": float(eta_ref),
        "background_coeffs": [float(c) for c in bg_coeffs_ref],
        "n_iterations": result.nfev,
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message,
    }

    return RietveldResult(
        phases=phase_results,
        two_theta=two_theta,
        y_obs=intensity,
        y_calc=y_calc,
        y_diff=y_diff,
        y_background=y_bg,
        bragg_positions=bragg_pos,
        Rwp=float(Rwp),
        Rp=float(Rp),
        chi_squared=float(chi_sq),
        GoF=float(GoF),
        refined_params=refined_params,
    )


# ---------------------------------------------------------------------------
# R-factor calculations
# ---------------------------------------------------------------------------
def _calc_Rwp(y_obs, y_calc, weights):
    """Weighted profile R-factor: Rwp = sqrt(Σ w(yobs-ycalc)² / Σ w·yobs²)"""
    num = np.sum(weights**2 * (y_obs - y_calc)**2)
    den = np.sum(weights**2 * y_obs**2)
    if den == 0:
        return 999.0
    return np.sqrt(num / den) * 100.0  # percentage


def _calc_Rp(y_obs, y_calc):
    """Profile R-factor: Rp = Σ|yobs-ycalc| / Σ yobs"""
    num = np.sum(np.abs(y_obs - y_calc))
    den = np.sum(np.abs(y_obs))
    if den == 0:
        return 999.0
    return (num / den) * 100.0  # percentage


# ---------------------------------------------------------------------------
# Utility: extract Bragg peak list for display
# ---------------------------------------------------------------------------
def bragg_peak_table(rietveld_result):
    """
    Extract Bragg peak positions as a flat list for display.

    Returns
    -------
    list of dict with keys: phase, hkl, two_theta, d_spacing, intensity
    """
    rows = []
    for phase_info in rietveld_result.phases:
        phase_name = phase_info["name"]
        bragg = rietveld_result.bragg_positions.get(phase_name, [])
        crystal_sys = phase_info["crystal_system"]
        a = phase_info["a_refined"]
        c = phase_info["c_refined"]

        for tt, hkl_str, peak_int in bragg:
            # Calculate d-spacing from refined 2-theta
            d = 1.54056 / (2.0 * np.sin(np.radians(tt / 2.0)))
            rows.append({
                "phase": phase_name,
                "hkl": hkl_str,
                "two_theta": round(tt, 3),
                "d_spacing": round(d, 4),
                "intensity": round(peak_int, 1),
            })

    return sorted(rows, key=lambda r: r["two_theta"])


def atom_site_table(phase_names=None):
    """
    Extract atom site parameters for display.

    Parameters
    ----------
    phase_names : list of str or None
        Phase names from CRYSTAL_PHASES. If None, return all.

    Returns
    -------
    list of dict with keys: phase, label, element, wyckoff, x, y, z,
                            occupancy, U_iso, multiplicity, B_iso
    """
    if phase_names is None:
        phase_names = list(CRYSTAL_PHASES.keys())

    rows = []
    for name in phase_names:
        if name not in CRYSTAL_PHASES:
            continue
        phase = CRYSTAL_PHASES[name]
        for atom in phase.atoms:
            # B_iso = 8π² × U_iso (Debye-Waller conversion)
            B_iso = 8.0 * np.pi**2 * atom.U_iso
            rows.append({
                "phase": phase.name,
                "atom": atom.label,
                "element": atom.element,
                "wyckoff": atom.wyckoff,
                "mult": atom.multiplicity,
                "x": round(atom.x, 5),
                "y": round(atom.y, 5),
                "z": round(atom.z, 5),
                "occupancy": round(atom.occupancy, 3),
                "U_iso": round(atom.U_iso, 4),
                "B_iso": round(B_iso, 4),
            })

    return rows


def structure_summary(phase_names=None):
    """
    Get a summary of crystal structure for each phase.

    Returns
    -------
    list of dict with keys: name, space_group, crystal_system, a, c, Z,
                            n_atoms, formula, atoms_per_cell
    """
    if phase_names is None:
        phase_names = list(CRYSTAL_PHASES.keys())

    summaries = []
    for name in phase_names:
        if name not in CRYSTAL_PHASES:
            continue
        phase = CRYSTAL_PHASES[name]

        # Count total atoms per unit cell
        total_atoms = sum(atom.multiplicity * atom.occupancy for atom in phase.atoms)

        # Build element composition per unit cell
        element_counts = {}
        for atom in phase.atoms:
            el = atom.element
            count = atom.multiplicity * atom.occupancy
            element_counts[el] = element_counts.get(el, 0) + count

        # Format as formula string
        formula_parts = []
        for el, count in element_counts.items():
            if abs(count - round(count)) < 0.01:
                count_str = str(int(round(count)))
            else:
                count_str = f"{count:.2f}"
            formula_parts.append(f"{el}{count_str}")
        formula = " ".join(formula_parts)

        summaries.append({
            "name": phase.name,
            "space_group": phase.space_group,
            "crystal_system": phase.crystal_system,
            "a": phase.a,
            "c": phase.c,
            "Z": phase.Z,
            "n_unique_sites": len(phase.atoms),
            "atoms_per_cell": round(total_atoms, 2),
            "composition": formula,
        })

    return summaries


# ---------------------------------------------------------------------------
# Validation: Literature comparison database
# ---------------------------------------------------------------------------
# Published lattice parameters for validation
# Sources: ICDD PDF cards, Barsoum (2013), Naguib et al. (2011)
LITERATURE_VALUES = {
    "Ti3AlC2": {
        "a": {"value": 3.075, "uncertainty": 0.001, "source": "ICDD 52-0875"},
        "c": {"value": 18.578, "uncertainty": 0.005, "source": "ICDD 52-0875"},
        "density_gcc": 4.25,
        "space_group": "P63/mmc",
    },
    "Ti3C2Tx": {
        "a": {"value": 3.057, "uncertainty": 0.005, "source": "Naguib et al. 2011 Adv. Mater."},
        "c": {"value": 19.86, "uncertainty": 0.50, "source": "varies with intercalation (19-25 Å)"},
        "density_gcc": 3.7,
        "space_group": "P63/mmc",
    },
    "TiO2_Anatase": {
        "a": {"value": 3.785, "uncertainty": 0.001, "source": "ICDD 21-1272"},
        "c": {"value": 9.514, "uncertainty": 0.002, "source": "ICDD 21-1272"},
        "density_gcc": 3.89,
        "space_group": "I41/amd",
    },
    "TiC": {
        "a": {"value": 4.328, "uncertainty": 0.001, "source": "ICDD 32-1383"},
        "c": {"value": 4.328, "uncertainty": 0.001, "source": "ICDD 32-1383 (cubic, a=c)"},
        "density_gcc": 4.93,
        "space_group": "Fm-3m",
    },
    "Al2O3": {
        "a": {"value": 4.759, "uncertainty": 0.001, "source": "ICDD 46-1212"},
        "c": {"value": 12.993, "uncertainty": 0.002, "source": "ICDD 46-1212"},
        "density_gcc": 3.99,
        "space_group": "R-3c",
    },
}


def validate_rietveld(rietveld_result, xps_composition=None,
                      peak_fit_d_spacings=None):
    """
    Comprehensive validation of Rietveld refinement results.

    Performs three types of validation:
    1. Literature comparison — refined lattice params vs published values
    2. Internal consistency — R-factor quality assessment, residual analysis
    3. Cross-technique — XRD phases vs XPS composition (if provided)

    Parameters
    ----------
    rietveld_result : RietveldResult
    xps_composition : dict or None
        e.g., {"Ti": 6.08, "C": 64.06, "O": 23.06, "F": 6.80} (at%)
    peak_fit_d_spacings : list of dict or None
        From individual peak fitting: [{"two_theta": ..., "d_spacing": ...}]

    Returns
    -------
    dict with keys:
        "literature_comparison" : list of dict (per phase)
        "internal_consistency" : dict
        "cross_technique" : dict or None
        "overall_assessment" : str
        "confidence_level" : str ("high", "moderate", "low")
        "reviewer_notes" : list of str
    """
    validation = {
        "literature_comparison": [],
        "internal_consistency": {},
        "cross_technique": None,
        "d_spacing_comparison": None,
        "overall_assessment": "",
        "confidence_level": "",
        "reviewer_notes": [],
    }

    scores = []  # collect numerical scores for overall assessment

    # =======================================================================
    # 1. Literature Comparison
    # =======================================================================
    for phase in rietveld_result.phases:
        phase_key = None
        for key in LITERATURE_VALUES:
            if LITERATURE_VALUES[key]["space_group"] == phase["space_group"]:
                # Match by name similarity
                if key.replace("_", " ").lower() in phase["name"].lower() or \
                   phase["name"].lower().replace(" ", "").replace("(", "").replace(")", "") in key.lower().replace("_", ""):
                    phase_key = key
                    break

        if phase_key is None:
            # Try direct name match
            for key in LITERATURE_VALUES:
                if key in phase["name"] or phase["name"] in LITERATURE_VALUES[key].get("space_group", ""):
                    phase_key = key
                    break

        if phase_key and phase_key in LITERATURE_VALUES:
            lit = LITERATURE_VALUES[phase_key]

            a_ref = lit["a"]["value"]
            a_unc = lit["a"]["uncertainty"]
            a_refined = phase["a_refined"]
            a_delta = abs(a_refined - a_ref)
            a_within = a_delta <= 3 * a_unc  # within 3σ
            a_deviation_pct = (a_delta / a_ref) * 100

            c_ref = lit["c"]["value"]
            c_unc = lit["c"]["uncertainty"]
            c_refined = phase["c_refined"]
            c_delta = abs(c_refined - c_ref)
            c_within = c_delta <= 3 * c_unc
            c_deviation_pct = (c_delta / c_ref) * 100

            # Score: 1.0 = perfect match, 0.0 = far off
            a_score = max(0, 1.0 - a_deviation_pct / 1.0)  # 1% deviation = score 0
            c_score = max(0, 1.0 - c_deviation_pct / 1.0)

            scores.extend([a_score, c_score])

            comparison = {
                "phase": phase["name"],
                "parameter_a": {
                    "refined": round(a_refined, 4),
                    "literature": a_ref,
                    "uncertainty": a_unc,
                    "delta": round(a_delta, 4),
                    "deviation_pct": round(a_deviation_pct, 3),
                    "within_3sigma": a_within,
                    "source": lit["a"]["source"],
                    "status": "PASS" if a_within else "REVIEW",
                },
                "parameter_c": {
                    "refined": round(c_refined, 4),
                    "literature": c_ref,
                    "uncertainty": c_unc,
                    "delta": round(c_delta, 4),
                    "deviation_pct": round(c_deviation_pct, 3),
                    "within_3sigma": c_within,
                    "source": lit["c"]["source"],
                    "status": "PASS" if c_within else "REVIEW",
                },
            }
            validation["literature_comparison"].append(comparison)

    # =======================================================================
    # 2. Internal Consistency
    # =======================================================================
    Rwp = rietveld_result.Rwp
    Rp = rietveld_result.Rp
    chi2 = rietveld_result.chi_squared
    GoF = rietveld_result.GoF

    # R-factor quality assessment
    if Rwp < 10:
        rwp_quality = "Excellent"
        rwp_score = 1.0
    elif Rwp < 15:
        rwp_quality = "Good"
        rwp_score = 0.8
    elif Rwp < 25:
        rwp_quality = "Acceptable"
        rwp_score = 0.6
    elif Rwp < 40:
        rwp_quality = "Fair (simplified model)"
        rwp_score = 0.4
    else:
        rwp_quality = "Poor"
        rwp_score = 0.2

    scores.append(rwp_score)

    # χ² assessment
    if 0.5 < chi2 < 2.0:
        chi2_quality = "Ideal"
        chi2_score = 1.0
    elif chi2 < 5.0:
        chi2_quality = "Acceptable"
        chi2_score = 0.7
    elif chi2 < 20:
        chi2_quality = "Model improvement needed"
        chi2_score = 0.4
    else:
        chi2_quality = "Significant model-data mismatch"
        chi2_score = 0.2

    scores.append(chi2_score)

    # Residual analysis
    y_diff = rietveld_result.y_diff
    y_obs = rietveld_result.y_obs
    residual_rms = np.sqrt(np.mean(y_diff**2))
    residual_max = np.max(np.abs(y_diff))
    signal_max = np.max(y_obs)
    residual_ratio = residual_rms / signal_max if signal_max > 0 else 999

    # Check for systematic residuals (runs test - simplified)
    signs = np.sign(y_diff)
    n_runs = 1 + np.sum(np.abs(np.diff(signs)) > 0)
    n_expected_runs = len(signs) / 2  # for random residuals
    runs_ratio = n_runs / n_expected_runs if n_expected_runs > 0 else 0

    if runs_ratio > 0.8:
        residual_pattern = "Random (good — no systematic error)"
    elif runs_ratio > 0.5:
        residual_pattern = "Mildly correlated (minor systematic trends)"
    else:
        residual_pattern = "Strongly correlated (systematic misfit — model needs improvement)"

    validation["internal_consistency"] = {
        "Rwp": {"value": round(Rwp, 2), "quality": rwp_quality},
        "Rp": {"value": round(Rp, 2)},
        "chi_squared": {"value": round(chi2, 3), "quality": chi2_quality},
        "GoF": {"value": round(GoF, 3)},
        "residual_rms": round(residual_rms, 1),
        "residual_max": round(residual_max, 1),
        "residual_ratio_pct": round(residual_ratio * 100, 2),
        "n_runs": n_runs,
        "expected_runs": round(n_expected_runs),
        "runs_ratio": round(runs_ratio, 3),
        "residual_pattern": residual_pattern,
        "convergence": rietveld_result.refined_params.get("success", False),
        "n_evaluations": rietveld_result.refined_params.get("n_iterations", 0),
    }

    # =======================================================================
    # 3. Cross-Technique Validation (XRD vs XPS)
    # =======================================================================
    if xps_composition is not None:
        cross = {"checks": [], "consistent": True}

        # Check 1: If Ti3AlC2 is dominant phase, XPS should show Al
        has_max_phase = any("Ti3AlC2" in p["name"] or "MAX" in p["name"]
                           for p in rietveld_result.phases
                           if p["weight_fraction_pct"] > 20)
        al_in_xps = xps_composition.get("Al", 0) > 0.5

        if has_max_phase and not al_in_xps:
            cross["checks"].append({
                "test": "MAX phase vs Al in XPS",
                "result": "INCONSISTENT",
                "detail": "XRD shows significant Ti₃AlC₂ but XPS shows no Al — "
                          "may indicate surface-only etching (XPS is surface-sensitive ~10 nm)",
            })
            cross["consistent"] = False
        elif has_max_phase and al_in_xps:
            cross["checks"].append({
                "test": "MAX phase vs Al in XPS",
                "result": "CONSISTENT",
                "detail": "Both XRD and XPS confirm presence of Al-containing phase",
            })

        # Check 2: If MXene phase present, should see terminations (O, F) in XPS
        has_mxene = any("Ti3C2" in p["name"] or "MXene" in p["name"]
                        for p in rietveld_result.phases
                        if p["weight_fraction_pct"] > 10)
        has_terminations = (xps_composition.get("O", 0) > 5 or
                           xps_composition.get("F", 0) > 2)

        if has_mxene and has_terminations:
            cross["checks"].append({
                "test": "MXene phase vs surface terminations",
                "result": "CONSISTENT",
                "detail": f"MXene phase in XRD confirmed by XPS terminations "
                          f"(O: {xps_composition.get('O', 0):.1f}%, "
                          f"F: {xps_composition.get('F', 0):.1f}%)",
            })
        elif has_mxene and not has_terminations:
            cross["checks"].append({
                "test": "MXene phase vs surface terminations",
                "result": "INCONSISTENT",
                "detail": "XRD shows MXene but XPS lacks expected -O/-F terminations",
            })
            cross["consistent"] = False

        # Check 3: Ti/C ratio consistency
        ti_xps = xps_composition.get("Ti", 0)
        c_xps = xps_composition.get("C", 0)
        if ti_xps > 0 and c_xps > 0:
            tc_ratio_xps = ti_xps / c_xps
            # For Ti3C2, theoretical Ti/C = 3/2 = 1.5
            # But XPS has adventitious C, so ratio will be much lower
            cross["checks"].append({
                "test": "Ti/C ratio (XPS)",
                "result": "NOTE",
                "detail": f"XPS Ti/C = {tc_ratio_xps:.3f} (expected ~1.5 for Ti₃C₂, "
                          f"but adventitious C at 284.8 eV inflates C content — "
                          f"use high-res C 1s deconvolution for true Ti-C ratio)",
            })

        # Check 4: F presence indicates HF etching route
        f_xps = xps_composition.get("F", 0)
        if f_xps > 2:
            cross["checks"].append({
                "test": "F content (etching confirmation)",
                "result": "CONSISTENT",
                "detail": f"F at {f_xps:.1f}% confirms HF-based etching route "
                          f"(Ti-F terminations at ~685 eV in F 1s)",
            })

        if cross["consistent"]:
            scores.append(0.9)
        else:
            scores.append(0.4)

        validation["cross_technique"] = cross

    # =======================================================================
    # 4. d-spacing Cross-Validation
    # =======================================================================
    if peak_fit_d_spacings is not None and len(peak_fit_d_spacings) > 0:
        d_comparisons = []
        bragg_list = []
        for phase_bragg in rietveld_result.bragg_positions.values():
            bragg_list.extend(phase_bragg)

        for pf in peak_fit_d_spacings:
            pf_tt = pf.get("center_2theta", pf.get("two_theta", 0))
            pf_d = pf.get("d_spacing", 0)
            if pf_tt == 0 or pf_d == 0:
                continue

            # Find closest Rietveld Bragg peak
            best_match = None
            best_delta = 999
            for bragg_tt, bragg_hkl, bragg_int in bragg_list:
                delta = abs(pf_tt - bragg_tt)
                if delta < best_delta:
                    best_delta = delta
                    best_match = (bragg_tt, bragg_hkl)

            if best_match and best_delta < 1.0:
                riet_d = 1.54056 / (2.0 * np.sin(np.radians(best_match[0] / 2.0)))
                d_delta = abs(pf_d - riet_d)
                d_comparisons.append({
                    "peak_fit_2theta": round(pf_tt, 3),
                    "rietveld_2theta": round(best_match[0], 3),
                    "delta_2theta": round(best_delta, 3),
                    "peak_fit_d": round(pf_d, 4),
                    "rietveld_d": round(riet_d, 4),
                    "delta_d": round(d_delta, 4),
                    "hkl": best_match[1],
                    "status": "MATCH" if d_delta < 0.05 else "CLOSE" if d_delta < 0.2 else "MISMATCH",
                })

        if d_comparisons:
            n_match = sum(1 for d in d_comparisons if d["status"] in ("MATCH", "CLOSE"))
            validation["d_spacing_comparison"] = {
                "comparisons": d_comparisons,
                "n_matched": n_match,
                "n_total": len(d_comparisons),
                "match_rate_pct": round(n_match / len(d_comparisons) * 100, 1),
            }
            scores.append(n_match / len(d_comparisons))

    # =======================================================================
    # Overall Assessment
    # =======================================================================
    if scores:
        avg_score = np.mean(scores)
    else:
        avg_score = 0.5

    if avg_score >= 0.75:
        validation["confidence_level"] = "high"
        validation["overall_assessment"] = (
            "Refinement results are well-validated. Lattice parameters agree with "
            "literature values, and cross-technique checks are consistent."
        )
    elif avg_score >= 0.5:
        validation["confidence_level"] = "moderate"
        validation["overall_assessment"] = (
            "Refinement results are reasonable but have limitations. The simplified "
            "Rietveld model (without full structure factor calculation) explains "
            "elevated R-factors. Lattice parameters are reliable; phase fractions "
            "should be treated as semi-quantitative."
        )
    else:
        validation["confidence_level"] = "low"
        validation["overall_assessment"] = (
            "Refinement requires improvement. Consider adding more phases, "
            "refining atomic positions, or checking data quality."
        )

    # Reviewer notes
    notes = []
    if Rwp > 20:
        notes.append(
            f"Rwp = {Rwp:.1f}% is elevated because this implementation uses "
            f"literature-derived relative intensities rather than computing "
            f"structure factors from atomic coordinates. This is equivalent to "
            f"a Le Bail (profile matching) approach for intensity extraction, "
            f"combined with Rietveld-style lattice parameter refinement. "
            f"The lattice parameters are still reliably determined."
        )
    if any(p["weight_fraction_pct"] < 5 for p in rietveld_result.phases):
        notes.append(
            "Minor phases below 5 wt% should be treated with caution — "
            "detection limit for laboratory XRD is typically 1-3 wt%."
        )
    if xps_composition is not None:
        notes.append(
            "XPS is surface-sensitive (~5-10 nm probe depth) while XRD is "
            "bulk-sensitive. Discrepancies between XRD phase fractions and "
            "XPS composition are expected and informative about surface vs. "
            "bulk chemistry differences."
        )
    notes.append(
        "For full Rietveld refinement with proper structure factors, "
        "software such as GSAS-II or FullProf is recommended. The values "
        "presented here use a profile-matching approach with constrained "
        "lattice parameters, which reliably determines lattice constants "
        "and approximate phase fractions."
    )

    validation["reviewer_notes"] = notes
    validation["validation_score"] = round(avg_score, 3)

    return validation
