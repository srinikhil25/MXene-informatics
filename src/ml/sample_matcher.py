"""
Cross-Technique Sample Matcher
==============================
Matches samples across XRD, SEM, and EDX by material family,
since individual sample names differ between instruments.

Strategy: Group by material family (CF, CAF, BFO, MXene, etc.)
and aggregate features per family for cross-technique correlation.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ── Material family rules ──
# Maps name patterns to canonical family names
FAMILY_RULES = [
    # MXene variants
    (r"ti2alc3|ti3alc2|ti2c3|ti3c2|mx-ene|mxene|mx-tic|cmx", "MXene_Ti3C2"),
    (r"before.etching|ti3alc2", "Ti3AlC2_MAX"),
    # Conductive fabrics
    (r"cf[-_]?\d+|cf[-_]new", "CF_Conductive_Fabric"),
    (r"ncf|ni[-_]?cu", "NiCu_Fabric"),
    # Carbon fabrics
    (r"caf[-_]?\d*|carbon.?fabric|carbon.?yarn|carbonyarn", "CAF_Carbon_Fabric"),
    # Bismuth compounds
    (r"bfo|bismuth.ferr", "BFO_BiFeO3"),
    (r"cobfo|co.?bfo", "CoBFO"),
    (r"znbfo|zn.?bfo", "ZnBFO"),
    (r"bi2se3|bise", "Bi2Se3"),
    (r"bi2te3|bite|bi2te", "Bi2Te3"),
    # Copper oxide
    (r"cuo2|cuo_?2", "CuO2"),
    # Zinc compounds
    (r"zn[cm_]|zinc", "Zn_Compound"),
    # Fabrics
    (r"cotton", "Cotton"),
    (r"nylon", "Nylon"),
    (r"polyester", "Polyester"),
    # Silver-copper
    (r"agcuy|ag.?cu", "AgCu_Alloy"),
    (r"agf", "AgF"),
    # Molybdenum
    (r"mo[24]", "Mo_Compound"),
    # Other
    (r"zolteck", "Zolteck_Fabric"),
    (r"wasteteapowder|tea", "Tea_Powder"),
    (r"pva", "PVA"),
    (r"gel", "Gel"),
    (r"freshpowder|old", "Powder_Sample"),
]


def classify_family(sample_name: str) -> str:
    """Classify a sample name into a material family."""
    name = str(sample_name).lower().strip()
    # Remove common prefixes
    name = re.sub(r'^(sem_hitachi_|sem_jeol_|edx_|xrd_)', '', name)
    # Remove folder paths
    name = re.sub(r'venkata.sai.varma.*?_', '', name)

    for pattern, family in FAMILY_RULES:
        if re.search(pattern, name, re.IGNORECASE):
            return family

    return "Other"


def build_family_feature_matrix(feature_csv: str) -> pd.DataFrame:
    """
    Build a family-level feature matrix by:
    1. Classifying each sample into a material family
    2. Aggregating features (mean) per family per technique
    3. Merging across techniques

    Returns DataFrame where each row = one material family
    with features from all available techniques.
    """
    df = pd.read_csv(feature_csv)

    # Classify families
    df["family"] = df["sample_name"].apply(classify_family)

    print("Sample family distribution:")
    family_counts = df.groupby(["family", "technique"]).size().unstack(fill_value=0)
    print(family_counts.to_string())
    print()

    # Separate by technique
    xrd_df = df[df["technique"] == "XRD"].copy()
    edx_df = df[df["technique"] == "EDX"].copy()
    sem_df = df[df["technique"] == "SEM"].copy()

    # Get numeric columns per technique
    xrd_cols = [c for c in df.columns if c.startswith("xrd_") and df[c].dtype in [np.float64, np.int64]]
    edx_cols = [c for c in df.columns if c.startswith("edx_") and df[c].dtype in [np.float64, np.int64]]
    sem_cols = [c for c in df.columns if c.startswith("sem_") and df[c].dtype in [np.float64, np.int64]]

    # Aggregate by family (mean)
    merged = pd.DataFrame()

    if len(xrd_df) > 0:
        xrd_agg = xrd_df.groupby("family")[xrd_cols].mean()
        xrd_agg["xrd_sample_count"] = xrd_df.groupby("family").size()
        merged = xrd_agg

    if len(edx_df) > 0:
        edx_agg = edx_df.groupby("family")[edx_cols].mean()
        edx_agg["edx_sample_count"] = edx_df.groupby("family").size()
        if merged.empty:
            merged = edx_agg
        else:
            merged = merged.join(edx_agg, how="outer")

    if len(sem_df) > 0:
        sem_agg = sem_df.groupby("family")[sem_cols].mean()
        sem_agg["sem_image_count"] = sem_df.groupby("family").size()
        if merged.empty:
            merged = sem_agg
        else:
            merged = merged.join(sem_agg, how="outer")

    # Add technique availability flags
    merged["has_xrd"] = merged.get("xrd_sample_count", pd.Series(dtype=float)).notna()
    merged["has_edx"] = merged.get("edx_sample_count", pd.Series(dtype=float)).notna()
    merged["has_sem"] = merged.get("sem_image_count", pd.Series(dtype=float)).notna()
    merged["n_techniques"] = merged[["has_xrd", "has_edx", "has_sem"]].sum(axis=1).astype(int)

    merged = merged.reset_index()
    merged = merged.rename(columns={"index": "family"})
    if "family" not in merged.columns and merged.index.name == "family":
        merged = merged.reset_index()

    return merged


def build_cross_technique_correlations(family_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-technique correlations using only families
    that have features from 2+ techniques.
    """
    # Filter to multi-technique families
    multi = family_df[family_df["n_techniques"] >= 2].copy()

    if len(multi) < 3:
        print(f"WARNING: Only {len(multi)} multi-technique families. Need >= 3 for correlations.")
        # Fall back to all families
        multi = family_df.copy()

    # Get numeric columns
    numeric_cols = multi.select_dtypes(include=[np.number]).columns
    # Exclude count/flag columns
    feature_cols = [c for c in numeric_cols
                    if not c.endswith("_count") and c != "n_techniques"
                    and not c.startswith("has_")]

    # Only keep columns with >50% non-null values
    valid_cols = [c for c in feature_cols if multi[c].notna().sum() > len(multi) * 0.3]

    corr = multi[valid_cols].corr()
    return corr


def extract_top_cross_correlations(corr_df: pd.DataFrame, top_n: int = 30) -> list:
    """Extract top cross-technique correlations (XRD vs EDX, XRD vs SEM, etc.)."""
    results = []

    for i, col1 in enumerate(corr_df.columns):
        for j, col2 in enumerate(corr_df.columns):
            if i >= j:
                continue
            # Only cross-technique pairs
            tech1 = col1.split("_")[0]
            tech2 = col2.split("_")[0]
            if tech1 == tech2:
                continue

            r = corr_df.loc[col1, col2]
            if pd.notna(r):
                results.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "technique_1": tech1.upper(),
                    "technique_2": tech2.upper(),
                    "correlation": round(float(r), 4),
                    "abs_correlation": round(abs(float(r)), 4),
                })

    results.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return results[:top_n]


def run_cross_technique_analysis(feature_csv: str = None, output_dir: str = None):
    """
    Full cross-technique analysis pipeline.
    """
    if feature_csv is None:
        feature_csv = "D:/Materials Informatics/data/processed/features/feature_matrix.csv"
    if output_dir is None:
        output_dir = "D:/Materials Informatics/data/processed/features"

    out_path = Path(output_dir)

    print("=" * 60)
    print("  CROSS-TECHNIQUE CORRELATION ANALYSIS")
    print("=" * 60)

    # Step 1: Build family-level features
    print("\n[1] Building family-level feature matrix...")
    family_df = build_family_feature_matrix(feature_csv)
    print(f"\nFamilies: {len(family_df)}")
    print(f"Multi-technique families (>=2): {(family_df['n_techniques'] >= 2).sum()}")

    # Show multi-technique families
    multi = family_df[family_df["n_techniques"] >= 2]
    if len(multi) > 0:
        print("\nMulti-technique families:")
        for _, row in multi.iterrows():
            techs = []
            if row.get("has_xrd", False):
                techs.append(f"XRD({int(row.get('xrd_sample_count', 0))})")
            if row.get("has_edx", False):
                techs.append(f"EDX({int(row.get('edx_sample_count', 0))})")
            if row.get("has_sem", False):
                techs.append(f"SEM({int(row.get('sem_image_count', 0))})")
            print(f"  {row['family']:30s} -> {' + '.join(techs)}")

    # Step 2: Cross-technique correlations
    print("\n[2] Computing cross-technique correlations...")
    corr_df = build_cross_technique_correlations(family_df)

    # Step 3: Extract top cross-technique correlations
    top_cross = extract_top_cross_correlations(corr_df, top_n=30)

    if top_cross:
        print("\nTop 15 CROSS-technique correlations:")
        for c in top_cross[:15]:
            print(f"  {c['feature_1']:35s} vs {c['feature_2']:35s}  r={c['correlation']:+.3f}")

    # Save outputs
    family_df.to_csv(out_path / "family_feature_matrix.csv", index=False)
    corr_df.to_csv(out_path / "cross_technique_correlation.csv")

    import json
    with open(out_path / "cross_technique_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_families": len(family_df),
            "n_multi_technique": int((family_df["n_techniques"] >= 2).sum()),
            "top_cross_correlations": top_cross,
            "families": family_df[["family", "n_techniques", "has_xrd", "has_edx", "has_sem"]].to_dict(orient="records"),
        }, f, indent=2)

    print(f"\nSaved: family_feature_matrix.csv, cross_technique_correlation.csv")
    print("=" * 60)

    return family_df, corr_df, top_cross


if __name__ == "__main__":
    run_cross_technique_analysis()
