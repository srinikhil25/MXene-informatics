"""
Materials Informatics: Master ETL Pipeline
=======================================
Runs all data parsers to convert raw experimental data
into standardized, ML-ready format.

Usage:
    python run_etl.py                    # Run all parsers
    python run_etl.py --stage xrd        # Run specific parser
    python run_etl.py --stage xps
    python run_etl.py --stage sem
    python run_etl.py --stage eds

Input:  D:/MXDiscovery/Mxene_Analysis/  (raw experimental data)
Output: D:/Materials Informatics/data/processed/  (standardized JSON + CSV)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from etl.xrd_parser import parse_all_xrd, save_xrd_processed
from etl.xps_parser import parse_all_xps, save_xps_processed
from etl.sem_parser import parse_all_sem, save_sem_catalog
from etl.eds_parser import parse_all_eds, save_eds_processed


RAW_BASE = Path("D:/MXDiscovery/Mxene_Analysis")
PROCESSED_BASE = Path("D:/Materials Informatics/data/processed")


def run_xrd():
    """Parse XRD data."""
    print("\n" + "=" * 60)
    print("  STAGE: XRD Data Extraction")
    print("=" * 60)
    raw_dir = str(RAW_BASE / "XRD")
    out_dir = str(PROCESSED_BASE / "xrd")
    data = parse_all_xrd(raw_dir)
    save_xrd_processed(data, out_dir)
    return {"xrd_datasets": len(data)}


def run_xps():
    """Parse XPS data."""
    print("\n" + "=" * 60)
    print("  STAGE: XPS Data Extraction")
    print("=" * 60)
    raw_dir = str(RAW_BASE / "XPS")
    out_dir = str(PROCESSED_BASE / "xps")
    data = parse_all_xps(raw_dir)
    save_xps_processed(data, out_dir)
    return {
        "xps_spectra": len(data["spectra"]),
        "xps_quantification": bool(data["quantification"]),
    }


def run_sem():
    """Parse SEM metadata."""
    print("\n" + "=" * 60)
    print("  STAGE: SEM Metadata Extraction")
    print("=" * 60)
    raw_dir = str(RAW_BASE / "SEM")
    out_dir = str(PROCESSED_BASE / "sem")
    data = parse_all_sem(raw_dir)
    save_sem_catalog(data, out_dir)
    return {"sem_images": len(data)}


def run_eds():
    """Parse EDS/EDX data."""
    print("\n" + "=" * 60)
    print("  STAGE: EDS/EDX Data Extraction")
    print("=" * 60)
    out_dir = str(PROCESSED_BASE / "eds")
    all_eds = []
    for subdir in ["TEM", "XRD"]:
        raw_dir = str(RAW_BASE / subdir)
        if Path(raw_dir).exists():
            data = parse_all_eds(raw_dir)
            all_eds.extend(data)
    save_eds_processed(all_eds, out_dir)
    return {"eds_spectra": len(all_eds)}


def main():
    parser = argparse.ArgumentParser(description="Materials Informatics ETL Pipeline")
    parser.add_argument("--stage", choices=["xrd", "xps", "sem", "eds", "all"],
                        default="all", help="Which data to process")
    args = parser.parse_args()

    print("=" * 60)
    print("  Materials Informatics: ETL Pipeline")
    print("  Raw Data -> Standardized JSON + CSV")
    print("=" * 60)
    print(f"  Source: {RAW_BASE}")
    print(f"  Output: {PROCESSED_BASE}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Ensure output dirs exist
    PROCESSED_BASE.mkdir(parents=True, exist_ok=True)

    stats = {}
    stages = {
        "xrd": run_xrd,
        "xps": run_xps,
        "sem": run_sem,
        "eds": run_eds,
    }

    if args.stage == "all":
        for name, func in stages.items():
            try:
                result = func()
                stats.update(result)
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}")
                stats[f"{name}_error"] = str(e)
    else:
        result = stages[args.stage]()
        stats.update(result)

    # Save pipeline stats
    stats["timestamp"] = datetime.now().isoformat()
    stats["raw_source"] = str(RAW_BASE)
    stats_path = PROCESSED_BASE / "etl_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("  ETL Pipeline Complete!")
    print("=" * 60)
    for k, v in stats.items():
        if k not in ("timestamp", "raw_source"):
            print(f"  {k}: {v}")
    print(f"\n  Stats saved to: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
