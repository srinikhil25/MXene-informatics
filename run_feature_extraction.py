"""
Runner script for MXene Feature Extraction Pipeline
====================================================
Executes the feature extraction module and prints results.

Usage:
    python run_feature_extraction.py
"""

import sys
import os
import time

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ml.feature_extraction import run_feature_extraction

def main():
    print("Starting MXene Feature Extraction...")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    t0 = time.time()

    try:
        df, corr_df, summary = run_feature_extraction()
    except Exception as e:
        print(f"\n[ERROR] Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed:.1f} seconds")

    # Print detailed results
    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)

    if df is not None and not df.empty:
        print(f"\nFeature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df)} non-null")

        # Show a few sample rows per technique
        if "technique" in df.columns:
            for tech in df["technique"].unique():
                tech_df = df[df["technique"] == tech]
                print(f"\n--- {tech} samples (first 3) ---")
                print(tech_df.head(3).to_string(max_cols=10))

    if corr_df is not None and not corr_df.empty:
        print(f"\nCorrelation matrix: {corr_df.shape[0]} x {corr_df.shape[1]}")

    print("\nOutputs saved to: D:/Materials Informatics/data/processed/features/")
    print("  - feature_matrix.csv")
    print("  - correlation_matrix.csv")
    print("  - feature_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
