# -*- coding: utf-8 -*-
"""
File Intelligence Agent
=======================
Scans a directory, reads every file, determines what it is, and builds a
structured manifest that groups related files by sample and technique.

Design:
    1. Walk the directory tree
    2. For each file: sniff content (magic bytes, headers, text patterns)
    3. Combine content sniffing with directory-structure context
    4. Group files into sessions/samples
    5. Return a FileManifest with categorized entries

Usage:
    from src.agents.file_intelligence import scan_directory
    manifest = scan_directory("D:/data/dhivya_data")
    print(manifest.summary())
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# File categories
# ──────────────────────────────────────────────────────────────────────────────

class Technique:
    """Known characterization techniques."""
    XRD = "xrd"
    XPS = "xps"
    SEM = "sem"
    TEM = "tem"
    EDS = "eds"            # generic EDS (will be sub-typed to SEM-EDS or TEM-EDS)
    TEM_EDS = "tem_eds"
    SEM_EDS = "sem_eds"
    TRANSPORT = "transport"  # electrical/thermal transport (zT, Seebeck, etc.)
    UNKNOWN = "unknown"


class FileType:
    """What kind of file this is."""
    SPECTRUM = "spectrum"           # 1D data (XRD pattern, XPS spectrum, EDS spectrum)
    IMAGE_RAW = "image_raw"         # Raw instrument image (TIF from microscope)
    IMAGE_PROCESSED = "image_processed"  # Processed image (FFT, SAED, composite)
    IMAGE_ELEMENTAL_MAP = "image_elemental_map"  # EDS elemental map
    METADATA = "metadata"           # Instrument metadata (JEOL .txt)
    REPORT = "report"               # Generated report (PDF, JPG summary)
    SPREADSHEET = "spreadsheet"     # Excel/CSV with tabular data
    NATIVE_BINARY = "native_binary" # Instrument-native binary (SPE, RAW, etc.)
    SYSTEM = "system"               # System files (Thumbs.db, .DS_Store)
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# File entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    """One file in the manifest."""
    path: Path
    relative_path: str           # relative to scan root
    filename: str
    extension: str
    size_bytes: int

    # Classification
    technique: str = Technique.UNKNOWN
    file_type: str = FileType.UNKNOWN
    format_name: str = ""        # e.g. "PANalytical XRDML", "PHI MultiPak CSV"
    parser: str = ""             # parser function to use

    # Context from directory structure
    sample_name: str = ""
    sub_technique: str = ""      # e.g. "STEM", "HRTEM", "SAED"

    # Grouping
    session_id: str = ""         # groups related files (e.g. "STEM/CS/002")
    element: str = ""            # for elemental maps (e.g. "Cu K", "Se L")

    # Sniffed details
    details: dict = field(default_factory=dict)

    # Can we parse this?
    parseable: bool = False
    skip_reason: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Content sniffers
# ──────────────────────────────────────────────────────────────────────────────

def _sniff_text_head(path: Path, n_bytes: int = 4000) -> Optional[str]:
    """Read first N bytes of a file as text, trying utf-8 then latin-1."""
    try:
        raw = path.read_bytes()[:n_bytes]
    except Exception:
        return None

    # Try UTF-16 LE (Hitachi SEM)
    if raw[:2] == b'\xff\xfe':
        try:
            return raw.decode("utf-16-le", errors="replace")
        except Exception:
            pass

    # Try UTF-8
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        pass

    return raw.decode("latin-1", errors="replace")


def _sniff_binary_magic(path: Path) -> Optional[str]:
    """Check magic bytes for known binary formats."""
    try:
        raw = path.read_bytes()[:16]
    except Exception:
        return None

    # TIFF: II (little-endian) or MM (big-endian)
    if raw[:2] in (b'II', b'MM'):
        return "tiff"
    # JPEG
    if raw[:2] == b'\xff\xd8':
        return "jpeg"
    # PNG
    if raw[:4] == b'\x89PNG':
        return "png"
    # BMP
    if raw[:2] == b'BM':
        return "bmp"
    # PDF
    if raw[:4] == b'%PDF':
        return "pdf"
    # PK (ZIP, XLSX, DOCX)
    if raw[:2] == b'PK':
        return "zip_archive"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Directory structure analyzer
# ──────────────────────────────────────────────────────────────────────────────

# Common technique folder names
_TECHNIQUE_KEYWORDS = {
    "xrd": Technique.XRD,
    "xps": Technique.XPS,
    "sem": Technique.SEM,
    "tem": Technique.TEM,
    "stem": Technique.TEM,
    "eds": Technique.EDS,
    "edx": Technique.EDS,
    "eels": Technique.TEM,  # electron energy loss → TEM family
    "saed": Technique.TEM,
    "hrtem": Technique.TEM,
}

_SUB_TECHNIQUE_KEYWORDS = {
    "stem": "STEM",
    "hrtem": "HRTEM",
    "saed": "SAED",
    "bf": "BF",
    "df": "DF",
    "haadf": "HAADF",
}


def _extract_context_from_path(filepath: Path, root: Path) -> dict:
    """Extract technique, sample, sub-technique from directory structure."""
    rel = filepath.relative_to(root)
    parts = [p.lower() for p in rel.parts[:-1]]  # directory parts only

    ctx = {
        "technique_hint": None,
        "sub_technique": "",
        "sample_name": "",
        "path_parts": list(rel.parts[:-1]),
    }

    # Walk parts and identify technique/sample
    for i, part in enumerate(parts):
        # Check for technique keyword
        if part in _TECHNIQUE_KEYWORDS:
            ctx["technique_hint"] = _TECHNIQUE_KEYWORDS[part]
            # Check for sub-technique
            if part in _SUB_TECHNIQUE_KEYWORDS:
                ctx["sub_technique"] = _SUB_TECHNIQUE_KEYWORDS[part]
            # Remaining parts after technique are likely sample names
            remaining = [p for p in rel.parts[i+1:-1]
                         if p.lower() not in _TECHNIQUE_KEYWORDS
                         and p.lower() not in _SUB_TECHNIQUE_KEYWORDS]
            if remaining:
                ctx["sample_name"] = remaining[0]  # First non-technique part = sample
            continue

        if part in _SUB_TECHNIQUE_KEYWORDS:
            ctx["sub_technique"] = _SUB_TECHNIQUE_KEYWORDS[part]
            continue

    # If no technique found from directory, try the parent folder name
    if not ctx["technique_hint"] and len(parts) >= 1:
        # The immediate parent might be the sample
        ctx["sample_name"] = rel.parts[-2] if len(rel.parts) >= 2 else ""

    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# File classifiers (one per format/type)
# ──────────────────────────────────────────────────────────────────────────────

def _classify_xrdml(entry: FileEntry, text: str) -> None:
    entry.technique = Technique.XRD
    entry.file_type = FileType.SPECTRUM
    entry.format_name = "PANalytical XRDML"
    entry.parser = "parse_panalytical_xrdml"
    entry.parseable = True
    # Extract instrument info
    if "<fixedSlitIntensity>" in text:
        entry.details["has_fixed_slit"] = True


def _classify_asc(entry: FileEntry, text: str) -> None:
    # ASC = simple 2-column XRD (scientific notation)
    lines = text.strip().split("\n")
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    if len(data_lines) > 10:
        # Check if it's 2-column numeric
        try:
            parts = data_lines[0].split()
            float(parts[0])
            float(parts[1])
            entry.technique = Technique.XRD
            entry.file_type = FileType.SPECTRUM
            entry.format_name = "ASC XRD (2-column)"
            entry.parser = "parse_asc_xrd"
            entry.parseable = True
            entry.details["n_points"] = len(data_lines)
        except (ValueError, IndexError):
            pass


def _classify_txt(entry: FileEntry, text: str) -> None:
    """Classify .txt files by content sniffing."""
    # JEOL $CM_FORMAT (SEM or TEM)
    if "$CM_FORMAT" in text or "$SEM_DATA_VERSION" in text:
        # Detect TEM vs SEM by voltage and signal
        volt_match = re.search(r'\$CM_ACCEL_VOLT\s+([\d.]+)', text)
        signal_match = re.search(r'\$CM_SIGNAL\s+(.+)', text)

        voltage = 0
        if volt_match:
            try:
                voltage = float(volt_match.group(1))
            except ValueError:
                pass

        signal = signal_match.group(1).strip().upper() if signal_match else ""
        tem_signals = ("STEM", "TEM", "BF", "DF", "HAADF", "ABF")
        is_tem = voltage >= 100 or any(s in signal for s in tem_signals)

        if is_tem:
            entry.technique = Technique.TEM
            entry.sub_technique = signal if signal else "TEM"
        else:
            entry.technique = Technique.SEM

        entry.file_type = FileType.METADATA
        entry.format_name = "JEOL Metadata"
        entry.parser = "parse_jeol_metadata"
        entry.parseable = True
        entry.details["voltage_kv"] = voltage
        entry.details["signal"] = signal
        return

    # Rigaku XRD
    if "*TYPE" in text or "*MEAS_COND" in text or ";KAlpha1" in text:
        entry.technique = Technique.XRD
        entry.file_type = FileType.SPECTRUM
        entry.format_name = "Rigaku XRD TXT"
        entry.parser = "parse_rigaku_txt"
        entry.parseable = True
        return

    # Hitachi SEM (UTF-16-LE encoded)
    if "InstructName" in text or "Magnification" in text:
        entry.technique = Technique.SEM
        entry.file_type = FileType.METADATA
        entry.format_name = "Hitachi SEM Metadata"
        entry.parser = "parse_sem_metadata"
        entry.parseable = True
        return


def _classify_csv(entry: FileEntry, text: str) -> None:
    """Classify CSV files."""
    lines = text.strip().split("\n")
    if len(lines) < 5:
        return

    # PHI MultiPak XPS CSV: line1=int, line2=blank/text, line3=identifier, line4=int
    try:
        int(lines[0].strip())
        if "," in lines[4]:
            parts = lines[4].strip().split(",")
            float(parts[0])
            float(parts[1])
            entry.technique = Technique.XPS
            entry.file_type = FileType.SPECTRUM
            entry.format_name = "PHI MultiPak CSV"
            entry.parser = "parse_xps_csv"
            entry.parseable = True

            # Detect region from identifier line
            identifier = lines[2].strip() if len(lines) > 2 else ""
            entry.details["xps_identifier"] = identifier
            if identifier.lower().startswith("su"):
                entry.details["xps_region"] = "Survey"
            else:
                entry.details["xps_region"] = identifier
            return
    except (ValueError, IndexError):
        pass

    # Generic CSV — could be transport data, etc.
    # Check for common header patterns
    header = lines[0].lower()
    if any(kw in header for kw in ["temperature", "seebeck", "resistivity", "zt", "conductivity"]):
        entry.technique = Technique.TRANSPORT
        entry.file_type = FileType.SPREADSHEET
        entry.format_name = "Transport Properties CSV"
        entry.parseable = False  # TODO: add transport parser
        return


def _classify_emsa(entry: FileEntry, text: str) -> None:
    """Classify EMSA spectral data files."""
    entry.file_type = FileType.SPECTRUM
    entry.format_name = "EMSA/MAS Spectral Data"
    entry.parser = "parse_emsa"
    entry.parseable = True

    # Check beam voltage to distinguish TEM-EDS vs SEM-EDS
    beam_match = re.search(r'#BEAMKV\s*:\s*([\d.]+)', text)
    if beam_match:
        beam_kv = float(beam_match.group(1))
        entry.details["beam_kv"] = beam_kv
        if beam_kv >= 100:
            entry.technique = Technique.TEM_EDS
        else:
            entry.technique = Technique.SEM_EDS
    else:
        entry.technique = Technique.EDS


def _classify_spx(entry: FileEntry) -> None:
    entry.technique = Technique.SEM_EDS
    entry.file_type = FileType.SPECTRUM
    entry.format_name = "Bruker SPX"
    entry.parser = "parse_bruker_spx"
    entry.parseable = True


def _classify_xls(entry: FileEntry) -> None:
    entry.technique = Technique.SEM_EDS
    entry.file_type = FileType.SPREADSHEET
    entry.format_name = "Bruker EDX Quantification (XLS)"
    entry.parser = "parse_bruker_xls"
    entry.parseable = True


def _classify_spe(entry: FileEntry, raw_bytes: bytes) -> None:
    """PHI .spe files — native binary XPS format."""
    entry.technique = Technique.XPS
    entry.file_type = FileType.NATIVE_BINARY
    entry.format_name = "PHI Multipak SPE (binary)"
    entry.parseable = False  # TODO: add SPE binary parser
    entry.skip_reason = "Binary format; CSV export available"


def _classify_xlsx(entry: FileEntry) -> None:
    """Excel files — could be transport data, calculations, etc."""
    name_lower = entry.filename.lower()
    if any(kw in name_lower for kw in ["zt", "seebeck", "transport", "calculation"]):
        entry.technique = Technique.TRANSPORT
        entry.file_type = FileType.SPREADSHEET
        entry.format_name = "Excel Spreadsheet"
        entry.parseable = False  # TODO: add transport parser
        entry.skip_reason = "Transport property data (parser not yet implemented)"
    else:
        entry.file_type = FileType.SPREADSHEET
        entry.format_name = "Excel Spreadsheet"
        entry.parseable = False


def _classify_image(entry: FileEntry, magic: str, ctx: dict) -> None:
    """Classify image files using context from path and filename."""
    name_lower = entry.filename.lower()
    name_no_ext = Path(entry.filename).stem.lower()

    # ── EDS Elemental Maps ──
    # Pattern: "View002 Cu K.bmp" → elemental map of Cu K-line
    view_match = re.match(r'view\s*(\d+)\s+(.+)', name_no_ext, re.IGNORECASE)
    if view_match:
        view_num = view_match.group(1)
        element_or_signal = view_match.group(2).strip()
        if element_or_signal.upper() in ("BF", "DF", "HAADF"):
            # This is a STEM BF/DF reference image, not an elemental map
            entry.technique = Technique.TEM
            entry.file_type = FileType.IMAGE_RAW
            entry.sub_technique = element_or_signal.upper()
        else:
            # Elemental map
            entry.technique = Technique.TEM_EDS
            entry.file_type = FileType.IMAGE_ELEMENTAL_MAP
            entry.element = element_or_signal
        entry.session_id = f"view_{view_num}"
        entry.format_name = f"STEM {magic.upper()} Image"
        entry.parseable = False  # images are displayed, not parsed
        return

    # ── STEM BF/DF paired images ──
    # Pattern: "01bf.bmp", "01df.bmp"
    bf_df_match = re.match(r'(\d+)(bf|df)\b', name_no_ext, re.IGNORECASE)
    if bf_df_match:
        scan_num = bf_df_match.group(1)
        mode = bf_df_match.group(2).upper()
        entry.technique = Technique.TEM
        entry.file_type = FileType.IMAGE_RAW
        entry.sub_technique = f"STEM {mode}"
        entry.session_id = f"stem_{scan_num}"
        entry.format_name = f"STEM {mode} Image"
        entry.parseable = False
        return

    # ── Processed/analyzed images ──
    processed_keywords = ["fft", "inverse fft", "saed", "composite", "dislocation",
                          "diffraction", "pattern", "overlay"]
    if any(kw in name_lower for kw in processed_keywords):
        entry.technique = ctx.get("technique_hint") or Technique.TEM
        entry.file_type = FileType.IMAGE_PROCESSED
        # Sub-categorize
        if "saed" in name_lower or "diffraction" in name_lower:
            entry.sub_technique = "SAED"
        elif "fft" in name_lower:
            entry.sub_technique = "FFT"
        elif "composite" in name_lower:
            entry.sub_technique = "Composite"
        entry.format_name = f"Processed {magic.upper()} Image"
        entry.parseable = False
        return

    # ── Raw instrument images ──
    # Pattern: "Image_318665.tif" (JEOL TEM)
    if re.match(r'image_\d+', name_no_ext, re.IGNORECASE):
        entry.technique = ctx.get("technique_hint") or Technique.TEM
        entry.file_type = FileType.IMAGE_RAW
        entry.format_name = f"Instrument {magic.upper()} Image"
        entry.parseable = False
        return

    # ── Default: use directory context ──
    entry.technique = ctx.get("technique_hint") or Technique.UNKNOWN
    entry.file_type = FileType.IMAGE_RAW
    entry.format_name = f"{magic.upper()} Image"
    entry.parseable = False


def _classify_pdf(entry: FileEntry, ctx: dict) -> None:
    """PDF files — usually EDS reports or analysis exports."""
    entry.file_type = FileType.REPORT
    entry.format_name = "PDF Report"
    entry.parseable = False
    entry.skip_reason = "PDF report (visual reference)"
    # If in a STEM/EDS directory, it's an EDS report
    if ctx.get("technique_hint") in (Technique.TEM, Technique.EDS):
        entry.technique = Technique.TEM_EDS


# ──────────────────────────────────────────────────────────────────────────────
# System file detection
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_FILES = {"thumbs.db", ".ds_store", "desktop.ini", ".gitkeep"}


# ──────────────────────────────────────────────────────────────────────────────
# Main scanner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FileManifest:
    """Complete scan result."""
    root: Path
    entries: list[FileEntry] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.entries)

    @property
    def parseable_files(self) -> list[FileEntry]:
        return [e for e in self.entries if e.parseable]

    @property
    def images(self) -> list[FileEntry]:
        return [e for e in self.entries
                if e.file_type in (FileType.IMAGE_RAW, FileType.IMAGE_PROCESSED,
                                   FileType.IMAGE_ELEMENTAL_MAP)]

    @property
    def skipped(self) -> list[FileEntry]:
        return [e for e in self.entries if e.file_type == FileType.SYSTEM]

    def by_technique(self) -> dict[str, list[FileEntry]]:
        """Group entries by technique."""
        groups: dict[str, list[FileEntry]] = {}
        for e in self.entries:
            if e.file_type == FileType.SYSTEM:
                continue
            groups.setdefault(e.technique, []).append(e)
        return groups

    def by_sample(self) -> dict[str, list[FileEntry]]:
        """Group entries by sample name."""
        groups: dict[str, list[FileEntry]] = {}
        for e in self.entries:
            if e.file_type == FileType.SYSTEM or not e.sample_name:
                continue
            groups.setdefault(e.sample_name, []).append(e)
        return groups

    def summary(self) -> dict:
        """Return a summary dict suitable for display."""
        tech_groups = self.by_technique()
        sample_groups = self.by_sample()
        return {
            "total_files": self.total_files,
            "parseable": len(self.parseable_files),
            "images": len(self.images),
            "skipped": len(self.skipped),
            "techniques": {
                tech: {
                    "count": len(entries),
                    "parseable": sum(1 for e in entries if e.parseable),
                    "file_types": list(set(e.file_type for e in entries)),
                    "formats": list(set(e.format_name for e in entries if e.format_name)),
                }
                for tech, entries in sorted(tech_groups.items())
            },
            "samples": sorted(sample_groups.keys()),
            "n_samples": len(sample_groups),
        }

    def summary_text(self) -> str:
        """Human-readable summary."""
        s = self.summary()
        lines = [
            f"Scanned: {self.root}",
            f"Total files: {s['total_files']}",
            f"Parseable: {s['parseable']} | Images: {s['images']} | Skipped: {s['skipped']}",
            f"Samples: {s['n_samples']} ({', '.join(s['samples'][:8])})",
            "",
            "By technique:",
        ]
        for tech, info in s["techniques"].items():
            lines.append(
                f"  {tech:10s}: {info['count']:3d} files "
                f"({info['parseable']} parseable) — {', '.join(info['formats'][:3])}"
            )
        return "\n".join(lines)


def scan_directory(root: str | Path) -> FileManifest:
    """
    Scan a directory recursively, classify every file, and return a manifest.

    This is the main entry point for the File Intelligence Agent.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    manifest = FileManifest(root=root)

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue

        # Build basic entry
        entry = FileEntry(
            path=filepath,
            relative_path=str(filepath.relative_to(root)),
            filename=filepath.name,
            extension=filepath.suffix.lower(),
            size_bytes=filepath.stat().st_size,
        )

        # ── System files ──
        if entry.filename.lower() in _SYSTEM_FILES:
            entry.file_type = FileType.SYSTEM
            entry.skip_reason = "System file"
            manifest.entries.append(entry)
            continue

        # ── Extract context from directory structure ──
        ctx = _extract_context_from_path(filepath, root)
        if ctx["technique_hint"]:
            entry.technique = ctx["technique_hint"]
        if ctx["sample_name"]:
            entry.sample_name = ctx["sample_name"]
        if ctx["sub_technique"]:
            entry.sub_technique = ctx["sub_technique"]

        # ── Classify by extension + content ──
        ext = entry.extension

        # --- XRD ---
        if ext == ".xrdml":
            text = _sniff_text_head(filepath) or ""
            _classify_xrdml(entry, text)

        elif ext == ".asc":
            text = _sniff_text_head(filepath) or ""
            _classify_asc(entry, text)

        # --- XPS ---
        elif ext == ".spe":
            raw = filepath.read_bytes()[:64]
            _classify_spe(entry, raw)

        # --- EDS ---
        elif ext == ".spx":
            _classify_spx(entry)

        elif ext == ".xls" and ext != ".xlsx":
            _classify_xls(entry)

        elif ext == ".emsa":
            text = _sniff_text_head(filepath) or ""
            _classify_emsa(entry, text)

        # --- Text files (need content sniffing) ---
        elif ext == ".txt":
            text = _sniff_text_head(filepath) or ""
            _classify_txt(entry, text)
            # If still unknown, try UTF-16 for Hitachi
            if entry.technique == Technique.UNKNOWN:
                try:
                    raw = filepath.read_bytes()[:4000]
                    text16 = raw.decode("utf-16-le", errors="replace")
                    _classify_txt(entry, text16)
                except Exception:
                    pass

        # --- CSV ---
        elif ext == ".csv":
            text = _sniff_text_head(filepath) or ""
            _classify_csv(entry, text)

        # --- Excel ---
        elif ext in (".xlsx", ".xlsm"):
            _classify_xlsx(entry)

        # --- PDF ---
        elif ext == ".pdf":
            _classify_pdf(entry, ctx)

        # --- Images ---
        elif ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"):
            magic = _sniff_binary_magic(filepath) or ext.lstrip(".")
            _classify_image(entry, magic, ctx)

        # --- Unknown ---
        else:
            entry.skip_reason = f"Unrecognized extension: {ext}"

        # ── Build session_id for grouping ──
        if not entry.session_id:
            stem = filepath.stem
            # Strip bf/df suffix for grouping: "01bf" and "01df" → session "01"
            session_stem = re.sub(r'(bf|df)$', '', stem, flags=re.IGNORECASE).rstrip()
            # Include sample in session
            if entry.sample_name:
                entry.session_id = f"{entry.sample_name}/{session_stem}"
            else:
                entry.session_id = session_stem

        # ── Override technique from directory context if still unknown ──
        if entry.technique == Technique.UNKNOWN and ctx.get("technique_hint"):
            entry.technique = ctx["technique_hint"]

        # ── Derive sample_name from parent folder if not set ──
        if not entry.sample_name and len(ctx["path_parts"]) > 0:
            # Use deepest non-technique folder as sample
            for part in reversed(ctx["path_parts"]):
                if part.lower() not in _TECHNIQUE_KEYWORDS and part.lower() not in _SUB_TECHNIQUE_KEYWORDS:
                    entry.sample_name = part
                    break

        manifest.entries.append(entry)

    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# CLI for testing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    manifest = scan_directory(path)
    print(manifest.summary_text())
    print()

    # Detailed listing
    for entry in manifest.entries:
        status = "PARSE" if entry.parseable else ("SKIP" if entry.file_type == FileType.SYSTEM else "IMAGE" if "image" in entry.file_type else "-----")
        print(f"  [{status:5s}] {entry.technique:8s} | {entry.file_type:20s} | {entry.format_name:30s} | {entry.sample_name:12s} | {entry.relative_path}")
