"""
Universal ETL Pipeline
======================
Processes ALL raw characterization data across multiple material families
and instrument formats. Extends the original MXene-specific ETL to handle:

  - XRD: Rigaku .txt (all samples)
  - FE-SEM: JEOL .txt + .bmp (CF, KALI, Bi₂Se₃/Bi₂Te₃, etc.)
  - HR-FE-SEM: Hitachi .txt + .tif (MXene, recent samples)
  - EDX: Bruker .spx (spectra) + .xls (quantification)
  - EDS: EMSA .emsa (TEM-EDX, MXene only)

Output: Unified JSON dataset in data/processed/universal/
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def _print(msg: str):
    """Print with safe encoding for Windows console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def run_universal_etl(raw_base: str, output_dir: str = None) -> dict:
    """
    Run the universal ETL pipeline on all raw data.

    Parameters
    ----------
    raw_base : str
        Path to the root raw data directory
        (e.g., 'D:/MXene-Informatics/data_raw/Analysis Raw DATA')
    output_dir : str
        Output directory for processed data

    Returns
    -------
    dict with counts and summary statistics
    """
    raw_path = Path(raw_base)
    if output_dir is None:
        output_dir = str(raw_path.parent.parent / "data" / "processed" / "universal")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "timestamp": datetime.now().isoformat(),
        "raw_source": str(raw_path),
        "output_dir": str(out_path),
    }

    print("=" * 60)
    print("  UNIVERSAL MATERIALS INFORMATICS ETL PIPELINE")
    print("=" * 60)

    # ── Stage 1: XRD (Rigaku .txt) ──
    print("\n[1/4] Processing XRD data (Rigaku format)...")
    xrd_results = _process_xrd(raw_path, out_path)
    stats["xrd_datasets"] = len(xrd_results)

    # ── Stage 2: SEM — JEOL FE-SEM + Hitachi HR-FE-SEM ──
    print("\n[2/4] Processing SEM data (JEOL + Hitachi)...")
    sem_results = _process_sem(raw_path, out_path)
    stats["sem_images_jeol"] = sem_results.get("jeol_count", 0)
    stats["sem_images_hitachi"] = sem_results.get("hitachi_count", 0)
    stats["sem_images_total"] = stats["sem_images_jeol"] + stats["sem_images_hitachi"]

    # ── Stage 3: EDX — Bruker .spx/.xls ──
    print("\n[3/4] Processing EDX data (Bruker format)...")
    edx_results = _process_edx(raw_path, out_path)
    stats["edx_spectra"] = edx_results.get("spectra_count", 0)
    stats["edx_quantifications"] = edx_results.get("quant_count", 0)

    # ── Stage 4: EDS — EMSA (existing MXene data) ──
    print("\n[4/4] Processing EDS data (EMSA format)...")
    eds_results = _process_eds(raw_path, out_path)
    stats["eds_emsa_spectra"] = eds_results

    # ── Build unified sample catalog ──
    print("\n[*] Building unified sample catalog...")
    catalog = _build_sample_catalog(out_path, stats)
    stats["total_samples"] = len(catalog)

    # Save stats
    with open(out_path / "universal_etl_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  ETL COMPLETE: {stats['total_samples']} samples processed")
    print(f"  XRD: {stats['xrd_datasets']} patterns")
    print(f"  SEM: {stats['sem_images_total']} images ({stats['sem_images_jeol']} JEOL + {stats['sem_images_hitachi']} Hitachi)")
    print(f"  EDX: {stats['edx_spectra']} spectra + {stats['edx_quantifications']} quantifications")
    print(f"  EDS: {stats['eds_emsa_spectra']} EMSA spectra")
    print(f"  Output: {out_path}")
    print("=" * 60)

    return stats


def _process_xrd(raw_path: Path, out_path: Path) -> list:
    """Process all Rigaku XRD .txt files."""
    from src.etl.xrd_parser import parse_rigaku_txt

    xrd_dir = out_path / "xrd"
    xrd_dir.mkdir(exist_ok=True)

    results = []
    # Search in XRD folder
    xrd_folders = list(raw_path.rglob("*XRD*")) + list(raw_path.rglob("*xrd*"))
    txt_files = set()
    for folder in xrd_folders:
        if folder.is_dir():
            for f in folder.rglob("*.txt"):
                txt_files.add(f)
        elif folder.suffix == ".txt":
            txt_files.add(folder)

    for txt_file in sorted(txt_files):
        try:
            # Check if it's Rigaku format (starts with ;)
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                first_line = f.readline().strip()
            if not first_line.startswith(";"):
                continue

            data = parse_rigaku_txt(str(txt_file))
            if data and data.get("n_points", 0) > 10:
                # Infer sample name from filename or metadata
                sample_name = data.get("metadata", {}).get("SampleName", txt_file.stem)
                data["sample_name"] = sample_name
                data["relative_path"] = str(txt_file.relative_to(raw_path))

                # Save individual JSON
                safe_name = _safe_filename(sample_name)
                with open(xrd_dir / f"xrd_{safe_name}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                # Also save CSV
                df = pd.DataFrame({
                    "two_theta_deg": data["two_theta"],
                    "intensity_counts": data["intensity"],
                })
                df.to_csv(xrd_dir / f"xrd_{safe_name}.csv", index=False)

                results.append({
                    "sample_name": sample_name,
                    "n_points": data["n_points"],
                    "scan_range": data.get("scan_range", {}),
                    "relative_path": data["relative_path"],
                })
                _print(f"    OK {sample_name}: {data['n_points']} points")

        except Exception as e:
            _print(f"    FAIL {txt_file.name}: {e}")

    # Save XRD summary
    if results:
        with open(xrd_dir / "xrd_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"  Total: {len(results)} XRD patterns processed")
    return results


def _process_sem(raw_path: Path, out_path: Path) -> dict:
    """Process all SEM data (JEOL + Hitachi formats)."""
    from src.etl.jeol_sem_parser import parse_all_jeol_sem, save_jeol_sem_catalog

    sem_dir = out_path / "sem"
    sem_dir.mkdir(exist_ok=True)

    result = {"jeol_count": 0, "hitachi_count": 0}

    # Process JEOL FE-SEM
    fesem_dirs = [d for d in raw_path.rglob("*FE-SEM*") if d.is_dir()]
    jeol_results = []
    for fesem_dir in fesem_dirs:
        parsed = parse_all_jeol_sem(str(fesem_dir))
        jeol_results.extend(parsed)

    if jeol_results:
        save_jeol_sem_catalog(jeol_results, str(sem_dir))
        result["jeol_count"] = len(jeol_results)

    # Process Hitachi HR-FE-SEM (using existing parser)
    try:
        from src.etl.sem_parser import parse_all_sem, save_sem_catalog
        hrfesem_dirs = [d for d in raw_path.rglob("*HR-FE-SEM*") if d.is_dir()]
        hitachi_results = []
        for hr_dir in hrfesem_dirs:
            parsed = parse_all_sem(str(hr_dir))
            hitachi_results.extend(parsed)

        if hitachi_results:
            # Save with different name to avoid overwriting
            hitachi_clean = []
            for r in hitachi_results:
                c = {k: v for k, v in r.items() if k != "raw_fields"}
                hitachi_clean.append(c)
            with open(sem_dir / "hitachi_sem_catalog.json", "w", encoding="utf-8") as f:
                json.dump(hitachi_clean, f, indent=2, default=str)
            result["hitachi_count"] = len(hitachi_results)
    except Exception as e:
        print(f"  WARNING: Hitachi parser error: {e}")

    return result


def _process_edx(raw_path: Path, out_path: Path) -> dict:
    """Process all Bruker EDX data."""
    from src.etl.bruker_edx_parser import parse_all_bruker_edx, save_bruker_edx

    edx_dir = out_path / "edx"
    edx_dir.mkdir(exist_ok=True)

    result = {"spectra_count": 0, "quant_count": 0}

    # Find EDX folders
    edx_folders = [d for d in raw_path.rglob("*EDX*") if d.is_dir()]
    edx_folders += [d for d in raw_path.rglob("*edx*") if d.is_dir()]
    # Remove duplicates
    edx_folders = list(set(edx_folders))

    for edx_folder in edx_folders:
        parsed = parse_all_bruker_edx(str(edx_folder))
        if parsed["spectra"] or parsed["quantifications"]:
            save_bruker_edx(parsed, str(edx_dir))
            result["spectra_count"] += len(parsed["spectra"])
            result["quant_count"] += len(parsed["quantifications"])

    return result


def _process_eds(raw_path: Path, out_path: Path) -> int:
    """Process EMSA format EDS files (existing parser)."""
    try:
        from src.etl.eds_parser import parse_all_eds, save_eds_processed

        eds_dir = out_path / "eds_emsa"
        eds_dir.mkdir(exist_ok=True)

        # Look for .emsa files in TEM or other folders
        emsa_count = 0
        for emsa_dir in raw_path.rglob("*"):
            if emsa_dir.is_dir():
                emsa_files = list(emsa_dir.glob("*.emsa"))
                if emsa_files:
                    parsed = parse_all_eds(str(emsa_dir))
                    if parsed:
                        save_eds_processed(parsed, str(eds_dir))
                        emsa_count += len(parsed)

        return emsa_count
    except Exception as e:
        print(f"  WARNING: EMSA parser error: {e}")
        return 0


def _build_sample_catalog(out_path: Path, stats: dict) -> list:
    """
    Build a unified sample catalog linking data across techniques.
    Each sample gets a unique ID and lists which techniques are available.
    """
    catalog = {}

    # Collect XRD samples
    xrd_summary = out_path / "xrd" / "xrd_summary.json"
    if xrd_summary.exists():
        with open(xrd_summary, encoding="utf-8") as f:
            xrd_data = json.load(f)
        for entry in xrd_data:
            name = _normalize_sample_name(entry["sample_name"])
            if name not in catalog:
                catalog[name] = {"sample_id": name, "techniques": {}}
            catalog[name]["techniques"]["xrd"] = {
                "n_points": entry["n_points"],
                "file": entry.get("relative_path", ""),
            }

    # Collect JEOL SEM samples
    jeol_catalog = out_path / "sem" / "jeol_sem_catalog.json"
    if jeol_catalog.exists():
        with open(jeol_catalog, encoding="utf-8") as f:
            jeol_data = json.load(f)
        for entry in jeol_data:
            name = _normalize_sample_name(entry.get("sample_name", entry["source_file"]))
            if name not in catalog:
                catalog[name] = {"sample_id": name, "techniques": {}}
            if "sem" not in catalog[name]["techniques"]:
                catalog[name]["techniques"]["sem"] = {"images": []}
            catalog[name]["techniques"]["sem"]["images"].append({
                "magnification": entry.get("magnification"),
                "voltage_kv": entry.get("accelerating_voltage_kv"),
                "instrument": "JEOL",
            })

    # Collect Hitachi SEM samples
    hitachi_catalog = out_path / "sem" / "hitachi_sem_catalog.json"
    if hitachi_catalog.exists():
        with open(hitachi_catalog, encoding="utf-8") as f:
            hitachi_data = json.load(f)
        for entry in hitachi_data:
            name = _normalize_sample_name(entry.get("image_name", entry.get("source_file", "")))
            if name not in catalog:
                catalog[name] = {"sample_id": name, "techniques": {}}
            if "sem" not in catalog[name]["techniques"]:
                catalog[name]["techniques"]["sem"] = {"images": []}
            catalog[name]["techniques"]["sem"]["images"].append({
                "magnification": entry.get("magnification"),
                "voltage_kv": entry.get("accelerating_voltage_kv"),
                "instrument": "Hitachi",
            })

    # Collect EDX quantifications
    edx_quant = out_path / "edx" / "bruker_edx_quantification.json"
    if edx_quant.exists():
        with open(edx_quant, encoding="utf-8") as f:
            edx_data = json.load(f)
        # Group by sample
        from itertools import groupby
        for entry in edx_data:
            group = entry.get("sample_group", "unknown")
            name = _normalize_sample_name(group)
            if name not in catalog:
                catalog[name] = {"sample_id": name, "techniques": {}}
            if "edx" not in catalog[name]["techniques"]:
                catalog[name]["techniques"]["edx"] = {"elements": []}
            catalog[name]["techniques"]["edx"]["elements"].append({
                "element": entry.get("element", ""),
                "wt_pct": entry.get("norm__wt", entry.get("wt", 0)),
                "at_pct": entry.get("norm__at", entry.get("at", 0)),
            })

    # Convert to list and add technique counts
    catalog_list = []
    for name, data in sorted(catalog.items()):
        data["n_techniques"] = len(data["techniques"])
        data["has_xrd"] = "xrd" in data["techniques"]
        data["has_sem"] = "sem" in data["techniques"]
        data["has_edx"] = "edx" in data["techniques"]
        data["has_xps"] = False  # XPS only for MXene
        catalog_list.append(data)

    # Save catalog
    with open(out_path / "sample_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog_list, f, indent=2)

    # Save summary CSV
    summary_rows = []
    for entry in catalog_list:
        summary_rows.append({
            "sample_id": entry["sample_id"],
            "n_techniques": entry["n_techniques"],
            "has_xrd": entry["has_xrd"],
            "has_sem": entry["has_sem"],
            "has_edx": entry["has_edx"],
            "has_xps": entry["has_xps"],
        })
    pd.DataFrame(summary_rows).to_csv(out_path / "sample_catalog.csv", index=False)

    print(f"  Unified catalog: {len(catalog_list)} unique samples")
    multi = sum(1 for c in catalog_list if c["n_techniques"] >= 2)
    print(f"  Multi-technique samples (>=2): {multi}")

    return catalog_list


def _normalize_sample_name(name: str) -> str:
    """Normalize sample name for matching across techniques."""
    import re
    name = str(name).strip()
    # Remove file extensions
    name = re.sub(r'\.(txt|bmp|jpg|tif|raw|spx|xls)$', '', name, flags=re.IGNORECASE)
    # Remove magnification suffixes
    name = re.sub(r'\s*\d+[kKxX]+\s*$', '', name)
    # Remove voltage suffixes
    name = re.sub(r'\s*\d+\s*[kK]?[vV]\s*', ' ', name)
    # Clean up
    name = name.strip().lower()
    name = re.sub(r'[_\-\s]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name if name else "unknown"


def _safe_filename(name: str) -> str:
    """Create a safe filename from sample name."""
    import re
    name = re.sub(r'[^\w\-.]', '_', str(name))
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:50] if name else "unknown"


if __name__ == "__main__":
    import sys

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/MXene-Informatics/data_raw/Analysis Raw DATA"
    output = sys.argv[2] if len(sys.argv) > 2 else None

    run_universal_etl(raw_dir, output)
