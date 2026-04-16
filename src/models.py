# -*- coding: utf-8 -*-
"""
Core Data Model
================
Defines the central data structures for the Materials Informatics platform.

Hierarchy:
    Project -> Sample -> TechniqueData

A Project is built by scanning a user's raw data folder. The project_builder
module detects techniques, resolves sample identities, parses files, and
assembles these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Technique constants
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_TECHNIQUES = {
    "XRD",
    "XPS",
    "UV-DRS",
    "Raman",
    "FTIR",
    "SEM",
    "TEM",
    "STEM",
    "HRTEM",
    "SAED",
    "EDS",
    "Hall",
    "Thermoelectric",
    "PL",          # Photoluminescence
    "UV-Vis",      # Transmission UV-Vis (distinct from DRS)
}


# ──────────────────────────────────────────────────────────────────────────────
# File-level entry (from File Intelligence Agent scan)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    """One file discovered during the folder scan."""
    path: Path
    technique: str                    # e.g. "XRD", "XPS", "TEM", "image", "skip"
    file_type: str                    # e.g. "xrdml", "csv", "tif", "emsa"
    sample_hint: str = ""             # best-guess sample ID (resolved later)
    parseable: bool = False           # can we extract numeric data?
    image_category: str = ""          # "tem_raw", "elemental_map", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileManifest:
    """Result of scanning an entire folder tree."""
    root: Path
    entries: list[FileEntry] = field(default_factory=list)
    scan_time: Optional[datetime] = None
    total_files: int = 0
    skipped: int = 0

    @property
    def by_technique(self) -> dict[str, list[FileEntry]]:
        """Group entries by technique."""
        groups: dict[str, list[FileEntry]] = {}
        for e in self.entries:
            groups.setdefault(e.technique, []).append(e)
        return groups

    @property
    def by_sample(self) -> dict[str, list[FileEntry]]:
        """Group entries by resolved sample_hint."""
        groups: dict[str, list[FileEntry]] = {}
        for e in self.entries:
            key = e.sample_hint or "_unassigned"
            groups.setdefault(key, []).append(e)
        return groups


# ──────────────────────────────────────────────────────────────────────────────
# Technique-level data (parsed + analysis results)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TechniqueData:
    """
    Parsed data and analysis results for one technique on one sample.

    Attributes
    ----------
    technique : str
        Technique name (e.g. "XRD", "XPS", "UV-DRS").
    files : list[Path]
        Source files that contributed to this data.
    parsed : dict[str, Any]
        Raw parser output. Format depends on technique:
        - XRD: {"two_theta": [...], "intensity": [...], "metadata": {...}}
        - XPS: {"binding_energy": [...], "intensity": [...], ...}
        - UV-DRS: {"wavelength": [...], "reflectance": [...], ...}
        - Hall: {"temperature": ..., "resistivity": ..., "mobility": ..., ...}
        - EDS: {"energy": [...], "counts": [...], "elements": [...], ...}
        - Images: {"paths": [...], "categories": {...}}
    analysis : dict[str, Any]
        Results from analysis agents. Populated after the user runs analysis.
        - XRD: {"phase_id": {...}, "scherrer": {...}}
        - XPS: {"peak_fits": {...}, "quantification": {...}}
        - UV-DRS: {"bandgap": float, "tauc_data": {...}}
    """
    technique: str
    files: list[Path] = field(default_factory=list)
    parsed: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Sample-level container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    """
    One material sample with all its characterization data.

    Attributes
    ----------
    sample_id : str
        Canonical sample identifier (e.g. "CS", "CS-3").
    aliases : list[str]
        All name variants seen for this sample (e.g. ["CS Pure", "CS (Pure)", "CuSe"]).
    techniques : dict[str, TechniqueData]
        Technique name -> parsed data + analysis results.
        For techniques with multiple sub-datasets (e.g. XPS survey + high-res),
        the TechniqueData.parsed dict holds all of them.
    """
    sample_id: str
    aliases: list[str] = field(default_factory=list)
    techniques: dict[str, TechniqueData] = field(default_factory=dict)

    @property
    def available_techniques(self) -> list[str]:
        """Sorted list of technique names that have parsed data."""
        return sorted(
            t for t, td in self.techniques.items()
            if td.parsed or td.files
        )

    def has_technique(self, technique: str) -> bool:
        return technique in self.techniques and (
            self.techniques[technique].parsed or self.techniques[technique].files
        )


# ──────────────────────────────────────────────────────────────────────────────
# Project-level container (top of hierarchy)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Project:
    """
    Top-level container: one user project = one data folder.

    Attributes
    ----------
    name : str
        Project name (derived from folder name or user input).
    root_path : Path
        Absolute path to the data folder the user pointed at.
    samples : dict[str, Sample]
        sample_id -> Sample. Ordered by sample_id.
    manifest : FileManifest
        Raw file scan results (for the file intelligence report).
    unassigned : list[FileEntry]
        Files that could not be mapped to any sample.
    created_at : datetime
        When this project was built.
    """
    name: str
    root_path: Path
    samples: dict[str, Sample] = field(default_factory=dict)
    manifest: FileManifest = field(default_factory=lambda: FileManifest(root=Path(".")))
    unassigned: list[FileEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def sample_ids(self) -> list[str]:
        """Sorted list of all sample IDs."""
        return sorted(self.samples.keys())

    @property
    def technique_matrix(self) -> dict[str, dict[str, bool]]:
        """
        Build a sample x technique availability matrix.
        Returns {sample_id: {technique: bool}}.
        """
        all_techniques: set[str] = set()
        for s in self.samples.values():
            all_techniques.update(s.techniques.keys())

        matrix = {}
        for sid, sample in sorted(self.samples.items()):
            matrix[sid] = {t: sample.has_technique(t) for t in sorted(all_techniques)}
        return matrix

    def get_all_for_technique(self, technique: str) -> dict[str, TechniqueData]:
        """Get TechniqueData for a specific technique across all samples."""
        result = {}
        for sid, sample in self.samples.items():
            if sample.has_technique(technique):
                result[sid] = sample.techniques[technique]
        return result
