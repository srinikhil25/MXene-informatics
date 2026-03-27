"""
Feature Extraction Module
=========================
Extracts features from all processed MXene characterization data (XRD, EDX, SEM)
and builds a unified feature matrix for downstream ML tasks.

Features extracted:
- XRD: peak count, top 3 peak positions, max intensity, mean FWHM,
  crystallite size (Scherrer), d-spacing, background level, peak density
- EDX: atomic % per element, element count, dominant/secondary elements
- SEM: magnification, accelerating voltage, working distance, pixel size

Outputs:
- feature_matrix.csv: full feature matrix (rows=samples, cols=features)
- correlation_matrix.csv: cross-feature correlations
- feature_summary.json: summary statistics
"""

import sys
import os
import json
import glob
import warnings
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = "D:/Materials Informatics"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from analysis.xrd_analysis import detect_peaks, fit_peak, scherrer_size

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UNIVERSAL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "universal")
XRD_DIR = os.path.join(UNIVERSAL_DIR, "xrd")
EDX_DIR = os.path.join(UNIVERSAL_DIR, "edx")
SEM_DIR = os.path.join(UNIVERSAL_DIR, "sem")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "features")


# ---------------------------------------------------------------------------
# XRD Feature Extraction
# ---------------------------------------------------------------------------
def extract_xrd_features(xrd_json_path):
    """
    Extract features from a single XRD JSON file.

    Returns dict of features or None if extraction fails.
    """
    try:
        with open(xrd_json_path, "r") as f:
            data = json.load(f)

        two_theta = np.array(data["two_theta"], dtype=float)
        intensity = np.array(data["intensity"], dtype=float)
        sample_name = data.get("sample_name", Path(xrd_json_path).stem)

        if len(two_theta) < 20 or len(intensity) < 20:
            return None

        features = {"sample_name": sample_name, "technique": "XRD"}

        # Background level: mean of first 50 points
        n_bg = min(50, len(intensity))
        features["xrd_background_level"] = float(np.mean(intensity[:n_bg]))

        # Max intensity
        features["xrd_max_intensity"] = float(np.max(intensity))

        # Peak detection - try with progressively lower prominence
        peak_positions = np.array([])
        peak_indices = np.array([])
        props = {}
        for prom in [50, 30, 15, 5]:
            try:
                peak_positions, peak_indices, props = detect_peaks(
                    two_theta, intensity,
                    prominence=prom, distance=8, height_pct=3, smooth_window=5
                )
                if len(peak_positions) >= 1:
                    break
            except Exception:
                continue

        n_peaks = len(peak_positions)
        features["xrd_n_peaks"] = n_peaks

        # Peak density (peaks per degree of 2-theta range)
        two_theta_range = two_theta[-1] - two_theta[0]
        features["xrd_peak_density"] = float(n_peaks / two_theta_range) if two_theta_range > 0 else 0.0

        # Top 3 peak positions (sorted by intensity at peak)
        if n_peaks > 0:
            peak_intensities = intensity[peak_indices]
            sorted_idx = np.argsort(peak_intensities)[::-1]
            for i in range(3):
                col = f"xrd_top_peak_{i+1}_2theta"
                if i < n_peaks:
                    features[col] = float(peak_positions[sorted_idx[i]])
                else:
                    features[col] = np.nan

            # Fit peaks and extract FWHM, crystallite size, d-spacing
            strongest_pos = peak_positions[sorted_idx[0]]
            fwhm_values = []

            # Fit up to 10 strongest peaks
            for j in range(min(10, n_peaks)):
                pos = peak_positions[sorted_idx[j]]
                try:
                    result = fit_peak(two_theta, intensity, pos,
                                      window_deg=1.0, profile="pseudo_voigt")
                    if result is not None and result.r_squared > 0.3:
                        fwhm_values.append(result.fwhm)

                        # Strongest peak features
                        if j == 0:
                            features["xrd_strongest_peak_fwhm"] = float(result.fwhm)
                            features["xrd_strongest_peak_d_spacing"] = float(result.d_spacing)

                            # Scherrer crystallite size
                            try:
                                sr = scherrer_size(result.center_2theta, result.fwhm)
                                features["xrd_crystallite_size_nm"] = float(sr.crystallite_size_nm)
                            except Exception:
                                features["xrd_crystallite_size_nm"] = np.nan
                except Exception:
                    continue

            # Mean FWHM
            if fwhm_values:
                features["xrd_mean_fwhm"] = float(np.mean(fwhm_values))
            else:
                features["xrd_mean_fwhm"] = np.nan

            # Fill in if strongest peak fitting failed
            if "xrd_strongest_peak_fwhm" not in features:
                features["xrd_strongest_peak_fwhm"] = np.nan
            if "xrd_strongest_peak_d_spacing" not in features:
                features["xrd_strongest_peak_d_spacing"] = np.nan
            if "xrd_crystallite_size_nm" not in features:
                features["xrd_crystallite_size_nm"] = np.nan
        else:
            for i in range(3):
                features[f"xrd_top_peak_{i+1}_2theta"] = np.nan
            features["xrd_strongest_peak_fwhm"] = np.nan
            features["xrd_strongest_peak_d_spacing"] = np.nan
            features["xrd_crystallite_size_nm"] = np.nan
            features["xrd_mean_fwhm"] = np.nan

        return features

    except Exception as e:
        print(f"  [WARN] XRD extraction failed for {xrd_json_path}: {e}")
        return None


def extract_all_xrd_features():
    """Extract features from all XRD JSON files."""
    xrd_files = sorted([
        f for f in glob.glob(os.path.join(XRD_DIR, "xrd_*.json"))
        if not os.path.basename(f).startswith("xrd_summary")
    ])
    print(f"Found {len(xrd_files)} XRD files")

    results = []
    failed = 0
    for i, fpath in enumerate(xrd_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing XRD {i+1}/{len(xrd_files)}...")
        feats = extract_xrd_features(fpath)
        if feats is not None:
            results.append(feats)
        else:
            failed += 1

    print(f"  XRD extraction complete: {len(results)} succeeded, {failed} failed")
    return results


# ---------------------------------------------------------------------------
# EDX Feature Extraction
# ---------------------------------------------------------------------------
def extract_edx_features():
    """
    Extract features from bruker_edx_quantification.json.
    Groups rows by source_file (each file = one measurement/sample).

    Returns list of feature dicts.
    """
    quant_path = os.path.join(EDX_DIR, "bruker_edx_quantification.json")
    if not os.path.exists(quant_path):
        print("  [WARN] EDX quantification file not found")
        return []

    try:
        with open(quant_path, "r") as f:
            rows = json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to load EDX quantification: {e}")
        return []

    # Group by source_file
    by_sample = defaultdict(list)
    for row in rows:
        key = row.get("source_file", "unknown")
        by_sample[key].append(row)

    print(f"Found {len(by_sample)} EDX samples (from {len(rows)} element rows)")

    results = []
    for source_file, elements in by_sample.items():
        try:
            # Build a sample name from sample_group or source_file
            sample_group = elements[0].get("sample_group", "")
            sample_name = f"edx_{Path(source_file).stem}"
            if sample_group:
                sample_name = f"edx_{sample_group}_{Path(source_file).stem}"

            features = {"sample_name": sample_name, "technique": "EDX"}

            # Number of elements
            features["edx_n_elements"] = len(elements)

            # Atomic % for each element
            at_pcts = {}
            for elem in elements:
                el_name = elem.get("element", "Unknown")
                at_pct = elem.get("norm._at.%", 0.0)
                col_name = f"edx_at_pct_{el_name}"
                features[col_name] = float(at_pct)
                at_pcts[el_name] = float(at_pct)

            # Dominant and secondary elements
            sorted_elements = sorted(at_pcts.items(), key=lambda x: -x[1])
            if len(sorted_elements) >= 1:
                features["edx_dominant_element"] = sorted_elements[0][0]
                features["edx_dominant_at_pct"] = sorted_elements[0][1]
            if len(sorted_elements) >= 2:
                features["edx_secondary_element"] = sorted_elements[1][0]
                features["edx_secondary_at_pct"] = sorted_elements[1][1]
            else:
                features["edx_secondary_element"] = np.nan
                features["edx_secondary_at_pct"] = np.nan

            results.append(features)

        except Exception as e:
            print(f"  [WARN] EDX extraction failed for {source_file}: {e}")
            continue

    print(f"  EDX extraction complete: {len(results)} samples")
    return results


# ---------------------------------------------------------------------------
# SEM Feature Extraction
# ---------------------------------------------------------------------------
def extract_sem_features():
    """
    Extract features from JEOL and Hitachi SEM catalog JSONs.

    Returns list of feature dicts.
    """
    results = []

    for catalog_file, instrument_label in [
        ("jeol_sem_catalog.json", "JEOL"),
        ("hitachi_sem_catalog.json", "Hitachi"),
    ]:
        catalog_path = os.path.join(SEM_DIR, catalog_file)
        if not os.path.exists(catalog_path):
            print(f"  [WARN] SEM catalog not found: {catalog_path}")
            continue

        try:
            with open(catalog_path, "r") as f:
                entries = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load SEM catalog {catalog_file}: {e}")
            continue

        print(f"Found {len(entries)} {instrument_label} SEM entries")

        for entry in entries:
            try:
                sample_name = entry.get("sample_name", "")
                source_file = entry.get("source_file", "")
                if not sample_name:
                    sample_name = Path(source_file).stem if source_file else "unknown"

                sample_name = f"sem_{instrument_label}_{sample_name}"

                features = {
                    "sample_name": sample_name,
                    "technique": "SEM",
                }

                # Magnification
                mag = entry.get("magnification", None)
                features["sem_magnification"] = float(mag) if mag is not None else np.nan

                # Accelerating voltage (JEOL uses kV, Hitachi uses V)
                acc_kv = entry.get("accelerating_voltage_kv", None)
                acc_v = entry.get("accelerating_voltage_v", None)
                if acc_kv is not None:
                    features["sem_accelerating_voltage_kv"] = float(acc_kv)
                elif acc_v is not None:
                    features["sem_accelerating_voltage_kv"] = float(acc_v) / 1000.0
                else:
                    features["sem_accelerating_voltage_kv"] = np.nan

                # Working distance (both use _um but Hitachi values may be large)
                wd = entry.get("working_distance_um", None)
                if wd is not None:
                    wd_val = float(wd)
                    # Hitachi stores in um but sometimes values are in nm-scale
                    # JEOL is consistently in um; Hitachi can have large values (mm-scale)
                    if instrument_label == "Hitachi" and wd_val > 1000:
                        wd_val = wd_val / 1000.0  # convert um to mm then back to um
                    features["sem_working_distance_um"] = wd_val
                else:
                    features["sem_working_distance_um"] = np.nan

                # Pixel size
                px = entry.get("pixel_size_nm", None)
                features["sem_pixel_size_nm"] = float(px) if px is not None else np.nan

                results.append(features)

            except Exception as e:
                print(f"  [WARN] SEM extraction failed for entry: {e}")
                continue

    print(f"  SEM extraction complete: {len(results)} entries")
    return results


# ---------------------------------------------------------------------------
# Build Unified Feature Matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(xrd_features, edx_features, sem_features):
    """
    Build a unified feature matrix combining all techniques.

    Each row = one sample/measurement, columns = all features.
    NaN for missing values.
    """
    all_records = []
    all_records.extend(xrd_features)
    all_records.extend(edx_features)
    all_records.extend(sem_features)

    if not all_records:
        print("[WARN] No features extracted from any technique!")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Reorder columns: sample_name, technique first, then sorted feature columns
    meta_cols = ["sample_name", "technique"]
    feature_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + feature_cols]

    return df


# ---------------------------------------------------------------------------
# Cross-Technique Correlation Matrix
# ---------------------------------------------------------------------------
def build_correlation_matrix(df):
    """
    Build correlation matrix using only numeric features.
    Only considers samples with >=2 techniques if possible,
    but since each row is a single-technique measurement,
    we compute correlations across the full numeric feature space.
    """
    # Select only numeric columns
    meta_cols = ["sample_name", "technique"]
    numeric_cols = [c for c in df.columns if c not in meta_cols
                    and df[c].dtype in [np.float64, np.int64, float, int]]

    # Also exclude string-valued columns that pandas might have as object
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])

    if numeric_df.empty or numeric_df.shape[1] < 2:
        print("[WARN] Not enough numeric features for correlation matrix")
        return pd.DataFrame()

    # Drop columns that are all NaN
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Drop columns with zero variance
    std = numeric_df.std()
    nonzero_var_cols = std[std > 0].index.tolist()
    numeric_df = numeric_df[nonzero_var_cols]

    if numeric_df.shape[1] < 2:
        print("[WARN] Not enough variable features for correlation matrix")
        return pd.DataFrame()

    # Compute pairwise correlation, using pairwise complete observations
    corr = numeric_df.corr(method="pearson", min_periods=3)

    return corr


# ---------------------------------------------------------------------------
# Feature Summary Statistics
# ---------------------------------------------------------------------------
def build_feature_summary(df, corr_df):
    """Build summary statistics for the feature extraction."""
    summary = {
        "total_samples": len(df),
        "techniques": {},
        "feature_counts": {},
        "numeric_features": 0,
        "correlation_matrix_shape": list(corr_df.shape) if not corr_df.empty else [0, 0],
    }

    # Per-technique stats
    if "technique" in df.columns:
        for tech in df["technique"].unique():
            tech_df = df[df["technique"] == tech]
            summary["techniques"][tech] = {
                "n_samples": len(tech_df),
                "n_features_with_data": int(tech_df.notna().sum().sum()),
            }

    # Numeric feature stats
    meta_cols = ["sample_name", "technique"]
    numeric_cols = [c for c in df.columns if c not in meta_cols]
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
    summary["numeric_features"] = len(numeric_df.columns)

    feature_stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) > 0:
            feature_stats[col] = {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "missing": int(numeric_df[col].isna().sum()),
            }
    summary["feature_stats"] = feature_stats

    # Top correlations (absolute value)
    if not corr_df.empty:
        top_corrs = []
        for i in range(len(corr_df.columns)):
            for j in range(i + 1, len(corr_df.columns)):
                val = corr_df.iloc[i, j]
                if not np.isnan(val):
                    top_corrs.append({
                        "feature_1": corr_df.columns[i],
                        "feature_2": corr_df.columns[j],
                        "correlation": float(val),
                        "abs_correlation": float(abs(val)),
                    })
        top_corrs.sort(key=lambda x: -x["abs_correlation"])
        summary["top_correlations"] = top_corrs[:20]
    else:
        summary["top_correlations"] = []

    return summary


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_feature_extraction():
    """Run the complete feature extraction pipeline."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 60)
    print("MXene Feature Extraction Pipeline")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Extract XRD features
    print("\n[1/3] Extracting XRD features...")
    xrd_features = extract_all_xrd_features()

    # 2. Extract EDX features
    print("\n[2/3] Extracting EDX features...")
    edx_features = extract_edx_features()

    # 3. Extract SEM features
    print("\n[3/3] Extracting SEM features...")
    sem_features = extract_sem_features()

    # 4. Build unified feature matrix
    print("\n[4/6] Building unified feature matrix...")
    df = build_feature_matrix(xrd_features, edx_features, sem_features)
    print(f"  Feature matrix shape: {df.shape}")
    if not df.empty:
        print(f"  Techniques: {df['technique'].value_counts().to_dict()}")

    # 5. Build correlation matrix
    print("\n[5/6] Computing correlation matrix...")
    corr_df = build_correlation_matrix(df)
    print(f"  Correlation matrix shape: {corr_df.shape}")

    # 6. Build summary and save
    print("\n[6/6] Saving outputs...")
    summary = build_feature_summary(df, corr_df)

    # Save feature matrix
    matrix_path = os.path.join(OUTPUT_DIR, "feature_matrix.csv")
    df.to_csv(matrix_path, index=False)
    print(f"  Saved: {matrix_path}")

    # Save correlation matrix
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr_df.to_csv(corr_path)
    print(f"  Saved: {corr_path}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "feature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Numeric features: {summary['numeric_features']}")
    for tech, info in summary["techniques"].items():
        print(f"  {tech}: {info['n_samples']} samples")

    if summary["top_correlations"]:
        print("\n  Top 5 feature correlations:")
        for c in summary["top_correlations"][:5]:
            print(f"    {c['feature_1']} <-> {c['feature_2']}: "
                  f"{c['correlation']:.3f}")

    print("=" * 60)

    return df, corr_df, summary


if __name__ == "__main__":
    df, corr_df, summary = run_feature_extraction()
