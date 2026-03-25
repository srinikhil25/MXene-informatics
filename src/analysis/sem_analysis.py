"""
SEM Image Analysis Module
=========================
Automated morphological analysis of SEM micrographs for MXene characterization.

Capabilities:
- Edge detection (Canny) for feature boundary identification
- Otsu / adaptive thresholding for segmentation
- Particle/flake detection via contour analysis
- Size measurement calibrated by pixel_size_nm
- Flake size distribution and layer thickness estimation
- Surface roughness / texture metrics

Input:  .tif SEM images + pixel_size_nm from metadata
Output: Particle size distributions, morphological statistics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from scipy import ndimage
    from scipy.signal import savgol_filter
except ImportError:
    ndimage = None

try:
    from skimage import filters, morphology, measure, exposure, feature, segmentation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ParticleResult:
    """Single detected particle/flake measurement."""
    label: int
    area_px: float          # Area in pixels
    area_nm2: float         # Area in nm²
    equivalent_diameter_nm: float  # Diameter of circle with same area
    major_axis_nm: float    # Length of major axis (Feret-like)
    minor_axis_nm: float    # Length of minor axis
    aspect_ratio: float     # major/minor
    perimeter_nm: float     # Boundary perimeter
    circularity: float      # 4π·area/perimeter² (1=perfect circle)
    centroid: Tuple[float, float]  # (row, col)
    solidity: float         # area/convex_hull_area
    orientation_deg: float  # Angle of major axis


@dataclass
class SEMAnalysisResult:
    """Complete SEM image analysis output."""
    image_name: str
    pixel_size_nm: float
    magnification: float
    n_particles: int
    particles: List[ParticleResult]
    # Summary statistics
    mean_diameter_nm: float
    median_diameter_nm: float
    std_diameter_nm: float
    min_diameter_nm: float
    max_diameter_nm: float
    mean_aspect_ratio: float
    # Size distribution
    size_bins_nm: List[float]
    size_counts: List[int]
    # Image arrays for visualization
    edges: Optional[np.ndarray] = field(default=None, repr=False)
    labeled: Optional[np.ndarray] = field(default=None, repr=False)
    binary: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class LayerThicknessResult:
    """Layer thickness measurement from cross-section SEM."""
    image_name: str
    pixel_size_nm: float
    n_layers: int
    thicknesses_nm: List[float]
    mean_thickness_nm: float
    std_thickness_nm: float
    profile_position: List[float]  # Position along profile line (nm)
    profile_intensity: List[float]  # Intensity along profile line
    peak_positions_nm: List[float]  # Detected layer boundaries


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def load_sem_image(image_path: str) -> Optional[np.ndarray]:
    """Load SEM .tif image as grayscale numpy array."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")
    try:
        img = Image.open(image_path)
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def crop_scale_bar(image: np.ndarray, bar_fraction: float = 0.05) -> np.ndarray:
    """
    Remove the bottom scale bar region from SEM image.
    Hitachi SU8600 typically has info bar in bottom ~5% of image.
    """
    h = image.shape[0]
    crop_rows = int(h * (1 - bar_fraction))
    return image[:crop_rows, :]


def preprocess(image: np.ndarray,
               denoise_sigma: float = 1.5,
               clahe: bool = True) -> np.ndarray:
    """
    Preprocess SEM image for analysis.

    Steps:
    1. Crop scale bar region
    2. Gaussian denoising
    3. CLAHE contrast enhancement (adaptive histogram equalization)
    4. Normalize to [0, 1]
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required: pip install scikit-image")

    # Crop scale bar
    img = crop_scale_bar(image)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    # Gaussian denoise
    if denoise_sigma > 0:
        img = filters.gaussian(img, sigma=denoise_sigma)

    # CLAHE for local contrast enhancement
    if clahe:
        img = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img


# ---------------------------------------------------------------------------
# Segmentation methods
# ---------------------------------------------------------------------------
def segment_otsu(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Otsu thresholding for binary segmentation.

    For MXene SEM: bright features on dark background → invert=False
    For cross-sections: may need invert=True
    """
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    if invert:
        binary = ~binary
    return binary


def segment_adaptive(image: np.ndarray,
                     block_size: int = 51,
                     offset: float = 0.0,
                     invert: bool = False) -> np.ndarray:
    """
    Adaptive thresholding — better for uneven illumination (common in SEM).

    Parameters:
        block_size: Size of local neighborhood (must be odd)
        offset: Subtracted from local mean threshold
        invert: Flip foreground/background
    """
    # Ensure image is in [0, 1] range
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)
    thresh = filters.threshold_local(img_norm, block_size=block_size, offset=offset)
    binary = img_norm > thresh
    if invert:
        binary = ~binary
    return binary


def segment_watershed(image: np.ndarray,
                      min_distance: int = 20,
                      compactness: float = 0.001) -> np.ndarray:
    """
    Watershed segmentation for touching/overlapping particles.

    Steps:
    1. Otsu threshold for initial binary
    2. Distance transform
    3. Find local maxima as markers
    4. Watershed from markers
    """
    # Binary mask
    binary = segment_otsu(image)

    # Clean up
    binary = morphology.remove_small_objects(binary, max_size=100)
    binary = morphology.remove_small_holes(binary, max_size=100)

    # Distance transform
    distance = ndimage.distance_transform_edt(binary)

    # Find peaks in distance map as markers
    from skimage.feature import peak_local_max
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)

    # Create marker image
    markers = np.zeros_like(binary, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers = ndimage.label(morphology.dilation(markers > 0, morphology.disk(3)))[0]

    # Watershed
    labels = segmentation.watershed(-distance, markers, mask=binary,
                                     compactness=compactness)
    return labels


def detect_edges(image: np.ndarray,
                 sigma: float = 2.0,
                 low_threshold: float = 0.05,
                 high_threshold: float = 0.15) -> np.ndarray:
    """
    Canny edge detection for feature boundary identification.
    """
    return feature.canny(image, sigma=sigma,
                         low_threshold=low_threshold,
                         high_threshold=high_threshold)


# ---------------------------------------------------------------------------
# Morphological cleaning
# ---------------------------------------------------------------------------
def clean_binary(binary: np.ndarray,
                 min_size: int = 200,
                 max_hole: int = 500,
                 closing_disk: int = 3,
                 opening_disk: int = 2) -> np.ndarray:
    """
    Morphological cleanup of binary segmentation.

    Steps:
    1. Closing (fill small gaps)
    2. Opening (remove small protrusions)
    3. Remove small objects
    4. Fill small holes
    """
    # Morphological closing then opening
    if closing_disk > 0:
        binary = morphology.closing(binary, morphology.disk(closing_disk))
    if opening_disk > 0:
        binary = morphology.opening(binary, morphology.disk(opening_disk))

    # Remove small objects and fill holes
    binary = morphology.remove_small_objects(binary, max_size=min_size)
    binary = morphology.remove_small_holes(binary, max_size=max_hole)

    return binary


# ---------------------------------------------------------------------------
# Particle measurement
# ---------------------------------------------------------------------------
def measure_particles(labeled: np.ndarray,
                      pixel_size_nm: float,
                      min_area_px: int = 100,
                      max_area_px: Optional[int] = None,
                      border_exclusion: int = 10) -> List[ParticleResult]:
    """
    Measure morphological properties of labeled particles.

    Parameters:
        labeled: Label image from segmentation
        pixel_size_nm: Physical size of each pixel in nm
        min_area_px: Minimum particle area (pixels) to include
        max_area_px: Maximum particle area (pixels) to include
        border_exclusion: Exclude particles within N pixels of image border
    """
    results = []

    regions = measure.regionprops(labeled)
    h, w = labeled.shape

    if max_area_px is None:
        max_area_px = h * w * 0.5  # Max 50% of image area

    for prop in regions:
        # Skip very small or very large
        if prop.area < min_area_px or prop.area > max_area_px:
            continue

        # Skip particles touching border
        r0, c0, r1, c1 = prop.bbox
        if (r0 < border_exclusion or c0 < border_exclusion or
            r1 > h - border_exclusion or c1 > w - border_exclusion):
            continue

        area_nm2 = prop.area * (pixel_size_nm ** 2)
        equiv_d = np.sqrt(4 * area_nm2 / np.pi)
        major = prop.axis_major_length * pixel_size_nm
        minor = prop.axis_minor_length * pixel_size_nm if prop.axis_minor_length > 0 else pixel_size_nm
        perimeter = prop.perimeter * pixel_size_nm

        # Circularity: 4π·area/perimeter²
        circ = (4 * np.pi * area_nm2) / (perimeter ** 2 + 1e-10)
        circ = min(circ, 1.0)  # Cap at 1.0

        aspect = major / (minor + 1e-10)

        results.append(ParticleResult(
            label=prop.label,
            area_px=prop.area,
            area_nm2=area_nm2,
            equivalent_diameter_nm=equiv_d,
            major_axis_nm=major,
            minor_axis_nm=minor,
            aspect_ratio=aspect,
            perimeter_nm=perimeter,
            circularity=circ,
            centroid=prop.centroid,
            solidity=prop.solidity,
            orientation_deg=np.degrees(prop.orientation),
        ))

    return results


# ---------------------------------------------------------------------------
# Layer thickness analysis (for cross-section SEM at high magnification)
# ---------------------------------------------------------------------------
def measure_layer_thickness(image: np.ndarray,
                            pixel_size_nm: float,
                            profile_col: Optional[int] = None,
                            profile_width: int = 20,
                            prominence: float = 0.1,
                            min_distance: int = 5) -> LayerThicknessResult:
    """
    Measure MXene layer thickness from cross-section SEM.

    Method:
    1. Take a vertical intensity profile (averaged over profile_width columns)
    2. Detect periodic brightness variations (layer boundaries)
    3. Measure peak-to-peak distances = layer spacing

    Best for: High magnification (50k-200k×) cross-section images

    Parameters:
        image: Preprocessed SEM image
        pixel_size_nm: Physical size per pixel
        profile_col: Column position for vertical profile (default: center)
        profile_width: Number of columns to average for smoother profile
        prominence: Minimum peak prominence for layer detection
        min_distance: Minimum peak spacing in pixels
    """
    from scipy.signal import find_peaks

    img = crop_scale_bar(image)
    h, w = img.shape

    if profile_col is None:
        profile_col = w // 2

    # Average vertical profile over several columns
    col_start = max(0, profile_col - profile_width // 2)
    col_end = min(w, profile_col + profile_width // 2)
    profile = img[:, col_start:col_end].mean(axis=1)

    # Normalize
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-10)

    # Smooth
    if len(profile) > 11:
        profile = savgol_filter(profile, window_length=11, polyorder=3)

    # Detect layer boundaries as peaks in gradient magnitude
    gradient = np.abs(np.gradient(profile))
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-10)

    peaks, properties = find_peaks(gradient,
                                    prominence=prominence,
                                    distance=min_distance)

    # Calculate spacings
    position_nm = np.arange(len(profile)) * pixel_size_nm
    peak_positions_nm = peaks * pixel_size_nm

    thicknesses = []
    if len(peaks) > 1:
        thicknesses = np.diff(peak_positions_nm).tolist()

    mean_t = float(np.mean(thicknesses)) if thicknesses else 0.0
    std_t = float(np.std(thicknesses)) if len(thicknesses) > 1 else 0.0

    return LayerThicknessResult(
        image_name="",
        pixel_size_nm=pixel_size_nm,
        n_layers=len(peaks),
        thicknesses_nm=thicknesses,
        mean_thickness_nm=mean_t,
        std_thickness_nm=std_t,
        profile_position=position_nm.tolist(),
        profile_intensity=profile.tolist(),
        peak_positions_nm=peak_positions_nm.tolist(),
    )


# ---------------------------------------------------------------------------
# Size distribution
# ---------------------------------------------------------------------------
def compute_size_distribution(particles: List[ParticleResult],
                              n_bins: int = 20,
                              size_key: str = "equivalent_diameter_nm"
                              ) -> Tuple[List[float], List[int]]:
    """
    Compute particle size distribution histogram.

    Parameters:
        particles: List of measured particles
        n_bins: Number of histogram bins
        size_key: Which size metric to use
    """
    if not particles:
        return [], []

    sizes = [getattr(p, size_key) for p in particles]
    counts, bin_edges = np.histogram(sizes, bins=n_bins)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    return bin_centers, counts.tolist()


# ---------------------------------------------------------------------------
# Surface texture / roughness metrics
# ---------------------------------------------------------------------------
def surface_roughness(image: np.ndarray, pixel_size_nm: float) -> dict:
    """
    Calculate surface roughness metrics from SEM image intensity.

    Uses intensity as a proxy for surface height (brighter = higher in SE mode).
    Note: This is semi-quantitative — true roughness requires AFM/profilometry.

    Returns:
        Ra: Arithmetic mean roughness (intensity units)
        Rq: Root-mean-square roughness
        Rsk: Skewness (asymmetry of surface profile)
        Rku: Kurtosis (peakedness)
    """
    from scipy.stats import skew, kurtosis

    img = crop_scale_bar(image)
    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)

    mean_h = img.mean()
    deviations = img - mean_h

    ra = float(np.abs(deviations).mean())
    rq = float(np.sqrt(np.mean(deviations**2)))
    rsk = float(skew(deviations.ravel()))
    rku = float(kurtosis(deviations.ravel()))

    return {
        "Ra": ra,
        "Rq": rq,
        "Rsk": rsk,
        "Rku": rku,
        "mean_intensity": float(mean_h),
        "std_intensity": float(img.std()),
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------
def full_sem_analysis(image_path: str,
                      pixel_size_nm: float,
                      magnification: float = 0,
                      image_name: str = "",
                      method: str = "adaptive",
                      denoise_sigma: float = 1.5,
                      min_area_px: int = 200,
                      adaptive_block: int = 51,
                      adaptive_offset: float = -0.01,
                      invert: bool = False,
                      n_bins: int = 20) -> Optional[SEMAnalysisResult]:
    """
    Complete SEM image analysis pipeline.

    Steps:
    1. Load and preprocess image
    2. Segment features (Otsu / adaptive / watershed)
    3. Label connected components
    4. Measure particle properties
    5. Compute size distribution
    6. Return comprehensive results

    Parameters:
        image_path: Path to .tif SEM image
        pixel_size_nm: Physical size per pixel (from metadata)
        magnification: Magnification level
        image_name: Name for identification
        method: Segmentation method — 'otsu', 'adaptive', 'watershed'
        denoise_sigma: Gaussian smoothing σ
        min_area_px: Minimum particle area to include
        adaptive_block: Block size for adaptive threshold
        adaptive_offset: Offset for adaptive threshold
        invert: Invert binary (dark features on bright background)
        n_bins: Number of bins for size distribution
    """
    # 1. Load
    raw = load_sem_image(image_path)
    if raw is None:
        return None

    # 2. Preprocess
    processed = preprocess(raw, denoise_sigma=denoise_sigma)

    # 3. Segment
    if method == "otsu":
        binary = segment_otsu(processed, invert=invert)
    elif method == "adaptive":
        binary = segment_adaptive(processed,
                                   block_size=adaptive_block,
                                   offset=adaptive_offset,
                                   invert=invert)
    elif method == "watershed":
        labeled = segment_watershed(processed)
        binary = labeled > 0
    else:
        binary = segment_otsu(processed, invert=invert)

    # 4. Clean binary
    binary = clean_binary(binary, min_size=min_area_px)

    # 5. Label
    if method != "watershed":
        labeled = measure.label(binary)

    # 6. Edge detection (for visualization)
    edges = detect_edges(processed)

    # 7. Measure
    particles = measure_particles(labeled, pixel_size_nm,
                                   min_area_px=min_area_px)

    # 8. Size distribution
    if particles:
        size_bins, size_counts = compute_size_distribution(particles, n_bins=n_bins)
        diameters = [p.equivalent_diameter_nm for p in particles]
        mean_d = float(np.mean(diameters))
        median_d = float(np.median(diameters))
        std_d = float(np.std(diameters))
        min_d = float(np.min(diameters))
        max_d = float(np.max(diameters))
        mean_ar = float(np.mean([p.aspect_ratio for p in particles]))
    else:
        size_bins, size_counts = [], []
        mean_d = median_d = std_d = min_d = max_d = mean_ar = 0.0

    return SEMAnalysisResult(
        image_name=image_name or Path(image_path).name,
        pixel_size_nm=pixel_size_nm,
        magnification=magnification,
        n_particles=len(particles),
        particles=particles,
        mean_diameter_nm=mean_d,
        median_diameter_nm=median_d,
        std_diameter_nm=std_d,
        min_diameter_nm=min_d,
        max_diameter_nm=max_d,
        mean_aspect_ratio=mean_ar,
        size_bins_nm=size_bins,
        size_counts=size_counts,
        edges=edges,
        labeled=labeled,
        binary=binary,
    )
