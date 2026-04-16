# -*- coding: utf-8 -*-
"""
Sample Resolver
================
Identifies which sample each file belongs to using a three-tier fallback:

    Tier 1: Directory context  (e.g. XPS/CS-3/ -> "CS-3")
    Tier 2: Filename patterns  (e.g. CS Pure.ASC -> "CS", CS-1.txt -> "CS-1")
    Tier 3: File content       (e.g. UV-DRS header "CS - RawData" -> "CS")

After collecting all hints, a SampleRegistry normalizes aliases to canonical
IDs (e.g. "CS (Pure)" = "CS Pure" = "CS").

This module does NOT parse file data — it only resolves identity. Parsing
is handled by the parsers in src/etl/.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants — technique folder names to strip when looking for sample names
# ──────────────────────────────────────────────────────────────────────────────

_TECHNIQUE_DIRS = {
    "xrd", "xps", "sem", "tem", "stem", "eds", "edx", "hrtem", "saed",
    "haadf", "raman", "ftir", "uv-vis", "uv drs", "uv_drs", "uvdrs",
    "uv-drs", "hall", "hall measurement", "thermoelectric",
    "thermoelectric properties", "images", "stem-eds",
    "pl", "photoluminescence", "transport",
}

# Patterns that indicate a date/session ID, not a sample name
_DATE_PATTERNS = [
    re.compile(r"^\d{4,8}[-/]\d{2}[-/]\d{2}$"),         # 20250621
    re.compile(r"^\d+[-]\d+[-]\d+([-]\d+)*$"),           # 43175-21-06-25 (any digit-dash chain)
    re.compile(r"^Divya|^divya|^Dr\.MN", re.I),          # operator names
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),                  # 28-06-2025
]

# Extensions to ignore for filename-based sample extraction
_SKIP_EXTENSIONS = {".db", ".ini", ".sys", ".dll"}


# ──────────────────────────────────────────────────────────────────────────────
# SampleRegistry — alias normalization
# ──────────────────────────────────────────────────────────────────────────────

class SampleRegistry:
    """
    Collects raw sample hints, normalizes them to canonical IDs.

    The normalization strategy:
    1. Strip parentheses: "CS (Pure)" -> "CS Pure"
    2. Collapse whitespace: "CS  Pure" -> "CS Pure"
    3. Common suffixes: "Pure" is dropped -> "CS Pure" -> "CS"
    4. Case-insensitive grouping: "cs-3" = "CS-3"
    5. Known prefix patterns: "cUsE3" -> "CS-3" (from Dhivya's TEM folders)
    """

    def __init__(self):
        self._raw_hints: list[str] = []           # all raw hints collected
        self._alias_map: dict[str, str] = {}      # normalized -> canonical
        self._canonical_set: set[str] = set()     # final canonical IDs

    def add_hint(self, raw_hint: str) -> None:
        """Record a raw sample hint for later resolution."""
        if raw_hint and raw_hint.strip():
            self._raw_hints.append(raw_hint.strip())

    def resolve(self) -> dict[str, str]:
        """
        Process all hints and build the alias -> canonical mapping.
        Returns the mapping dict.
        """
        # Step 1: Normalize all hints
        normalized = [self._normalize(h) for h in self._raw_hints]
        normalized = [n for n in normalized if n]  # drop empties

        # Step 2: Count occurrences to find the most common form
        counts = Counter(normalized)

        # Step 3: Group by canonical form
        canon_groups: dict[str, list[str]] = defaultdict(list)
        for norm in set(normalized):
            canon = self._to_canonical(norm)
            canon_groups[canon].append(norm)

        # Step 4: Build alias map
        self._alias_map.clear()
        self._canonical_set.clear()
        for canon, aliases in canon_groups.items():
            self._canonical_set.add(canon)
            for alias in aliases:
                self._alias_map[alias] = canon
            # Also map the canonical to itself
            self._alias_map[canon] = canon

        # Map raw hints through normalization
        for raw in set(self._raw_hints):
            norm = self._normalize(raw)
            if norm in self._alias_map:
                self._alias_map[raw] = self._alias_map[norm]

        logger.info(
            "SampleRegistry resolved %d hints -> %d canonical samples: %s",
            len(self._raw_hints), len(self._canonical_set),
            sorted(self._canonical_set),
        )
        return dict(self._alias_map)

    def get_canonical(self, raw_hint: str) -> str:
        """Map a raw hint to its canonical sample ID. Returns '' if unknown."""
        if not raw_hint:
            return ""
        # Try direct lookup
        if raw_hint in self._alias_map:
            return self._alias_map[raw_hint]
        # Try normalized lookup
        norm = self._normalize(raw_hint)
        return self._alias_map.get(norm, "")

    @property
    def canonical_ids(self) -> list[str]:
        """Sorted list of all canonical sample IDs."""
        return sorted(self._canonical_set)

    def get_aliases(self, canonical_id: str) -> list[str]:
        """Get all aliases that map to a canonical ID."""
        return sorted(
            k for k, v in self._alias_map.items()
            if v == canonical_id and k != canonical_id
        )

    # ── Internal normalization methods ─────────────────────────────────

    @staticmethod
    def _normalize(raw: str) -> str:
        """Normalize a raw sample hint to a standard form."""
        s = raw.strip()
        # Remove parenthetical suffixes: "CS (Pure)" -> "CS"
        s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # Remove "Pure" suffix: "CS Pure" -> "CS"
        s = re.sub(r"\s+Pure$", "", s, flags=re.IGNORECASE)
        # Remove "RawData" suffix: "CS - RawData" -> "CS"
        s = re.sub(r"\s*-?\s*RawData$", "", s, flags=re.IGNORECASE)
        return s.strip()

    @staticmethod
    def _to_canonical(normalized: str) -> str:
        """
        Convert a normalized name to canonical form.
        Uppercase the base, preserve number suffixes.
        """
        s = normalized.strip()

        # Handle known irregular names (Dhivya-specific)
        # "cUsE3" -> "CS-3", "CSCBI-1" -> "CS-1", etc.
        irregular = _match_irregular(s)
        if irregular:
            return irregular

        # Standard: uppercase
        return s.upper() if s else ""


# ──────────────────────────────────────────────────────────────────────────────
# Irregular name matching
# ──────────────────────────────────────────────────────────────────────────────

def _match_irregular(name: str) -> Optional[str]:
    """
    Match known irregular sample name patterns to canonical forms.
    Returns canonical form or None if no match.
    """
    low = name.lower().strip()

    # "cUsE3", "cuse3" -> "CS-3"
    m = re.match(r"^cuse[-_]?(\d+)$", low)
    if m:
        return f"CS-{m.group(1)}"

    # "CSCBI-1", "cscbi1", "CSCBI 5" -> "CS-1", "CS-5"
    m = re.match(r"^cscbi[-_ ]?(\d+)$", low)
    if m:
        return f"CS-{m.group(1)}"

    # "cskbi3" (typo in XRD filename) -> "CS-3"
    m = re.match(r"^cskbi[-_ ]?(\d+)$", low)
    if m:
        return f"CS-{m.group(1)}"

    # "Cs3Bi2I9" — this is the precursor material, not a sample
    # Keep as-is; user can merge or separate in confirmation UI
    if low == "cs3bi2i9":
        return "Cs3Bi2I9"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: Directory-based sample extraction
# ──────────────────────────────────────────────────────────────────────────────

def sample_from_directory(filepath: Path, root: Path) -> str:
    """
    Extract sample hint from the directory structure.

    Strategy: Walk directory parts from root to file. Skip known technique
    folder names. The first remaining part that isn't a date/operator is
    likely the sample name.
    """
    try:
        rel = filepath.relative_to(root)
    except ValueError:
        return ""

    parts = list(rel.parts[:-1])  # directories only, no filename

    for part in parts:
        low = part.lower().strip()
        # Skip technique directories
        if low in _TECHNIQUE_DIRS:
            continue
        # Skip date/operator patterns
        if any(p.match(low) for p in _DATE_PATTERNS):
            continue
        # Skip very short or generic names
        if len(low) <= 1:
            continue
        # This looks like a sample name
        return part

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: Filename-based sample extraction
# ──────────────────────────────────────────────────────────────────────────────

def sample_from_filename(filepath: Path) -> str:
    """
    Extract sample hint from the filename itself.

    Strategy: Strip the extension and known prefixes/suffixes, then look
    for sample-like patterns.
    """
    stem = filepath.stem
    ext = filepath.suffix.lower()
    if ext in _SKIP_EXTENSIONS:
        return ""

    # Known prefix patterns from instrument software
    # "Dr.MN-dhivya-cscbi1" -> "cscbi1"
    m = re.match(r"(?:Dr\.?\w*[-.])*(?:dhivya|pavi)[-.](.+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # "28-06-2025.104.Dr.MN-Pavi - CS" -> "CS"
    m = re.match(r"\d{2}-\d{2}-\d{4}\.\d+\..*?[-–]\s*(.+)", stem)
    if m:
        return m.group(1).strip()

    # XPS region files: "Cu 2p", "Se 3d", "Su" -> not sample names
    if re.match(r"^[A-Z][a-z]?\s*\d[a-z]$", stem) or stem.lower() in {"su", "survey"}:
        return ""

    # Image files with just numbers: "Image_318665" -> not useful
    if re.match(r"^Image_\d+$", stem, re.IGNORECASE):
        return ""

    # STEM files: "002", "01bf", "View002 Cu K" -> not sample names
    if re.match(r"^\d{2,3}(bf|df)?$", stem, re.IGNORECASE):
        return ""
    if re.match(r"^View\d+", stem, re.IGNORECASE):
        return ""

    # "zT calculation" -> not a sample name
    if "calculation" in stem.lower() or "conv" in stem.lower():
        return ""

    # Processed TEM image descriptions -> not sample names
    # "2nm image", "FFT of ...", "Inverse FFT of ...", "SAED pattern ...",
    # "Composite overall ...", "Dislocation from ...", "for dislocation ..."
    _PROCESSED_IMAGE_KEYWORDS = [
        "fft", "inverse", "saed", "dislocation", "composite",
        "disloc", "pattern", "image j", "nm image", "nm ",
    ]
    if any(kw in stem.lower() for kw in _PROCESSED_IMAGE_KEYWORDS):
        return ""

    # SEM/EDX doc files with technique names embedded -> extract sample prefix
    # "CS SEM  EDX" -> "CS"
    m = re.match(r"^([A-Za-z0-9][\w-]*?)\s+(?:SEM|EDX|EDS|TEM|XRD|XPS)", stem, re.I)
    if m:
        return m.group(1).strip()

    # Generic file: try the whole stem as the sample name
    # But only if it looks like a plausible sample ID (short, not a sentence)
    clean = stem.strip()
    if len(clean) <= 12 and re.match(r"^[A-Za-z0-9][\w\s\-().]*$", clean):
        return clean

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: Content-based sample extraction
# ──────────────────────────────────────────────────────────────────────────────

def sample_from_content(filepath: Path) -> str:
    """
    Extract sample hint from file content (first few lines).

    Only reads the file for specific formats where the sample name is
    embedded in the content header.
    """
    ext = filepath.suffix.lower()

    # UV-DRS .txt: first line is "CS - RawData" or "CS-3 - RawData"
    if ext == ".txt":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                line1 = f.readline().strip().strip('"')
            # Pattern: "SampleName - RawData" or "SampleName - ..."
            m = re.match(r"^(.+?)\s*-\s*RawData", line1, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        except OSError:
            pass

    # .xrdml: sample name might be in XML
    if ext == ".xrdml":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(2000)
            m = re.search(r"<sampleId>(.+?)</sampleId>", head)
            if m:
                return m.group(1).strip()
        except OSError:
            pass

    # .xlsx: sheet names might be sample names (thermoelectric)
    # This is handled at the project_builder level, not here,
    # because one file produces multiple samples.

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Combined resolver
# ──────────────────────────────────────────────────────────────────────────────

def resolve_sample_hint(filepath: Path, root: Path) -> str:
    """
    Resolve the sample hint for a file using the 3-tier fallback.
    Returns the raw hint (not yet canonical — feed to SampleRegistry for that).
    """
    # Tier 1: Directory
    hint = sample_from_directory(filepath, root)
    if hint:
        return hint

    # Tier 2: Filename
    hint = sample_from_filename(filepath)
    if hint:
        return hint

    # Tier 3: Content (only for specific formats)
    hint = sample_from_content(filepath)
    if hint:
        return hint

    return ""
