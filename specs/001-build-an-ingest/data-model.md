# Data Model: FITS to Zarr Ingest Package

**Date**: October 20, 2025 **Feature**: 001-build-an-ingest

## Overview

This document defines the key data structures and entities for the FITS to Zarr
ingestion pipeline. All models are derived from the functional requirements and
key entities identified in the specification.

## Core Entities

### 1. FITSImageFile

**Purpose**: Represents a single OVRO-LWA FITS image file containing one time
step and one frequency subband.

**Attributes**:

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class FITSImageFile:
    """Radio astronomy image file in FITS format."""

    path: Path                          # Absolute path to FITS file
    observation_date: str               # YYYYMMDD format
    observation_time: str               # HHMMSS format
    subband_mhz: int                    # Frequency subband in MHz
    is_fixed: bool                      # Whether _fixed.fits suffix present

    @property
    def time_step_key(self) -> str:
        """Unique identifier for time step grouping."""
        return f"{self.observation_date}_{self.observation_time}"

    @property
    def filename_pattern_match(self) -> bool:
        """Validates filename matches OVRO-LWA convention."""
        # Pattern: YYYYMMDD_HHMMSS_<SB>MHz_averaged_*-I-image[_fixed].fits
        pass

    def requires_fixing(self) -> bool:
        """Checks if FITS file needs header correction."""
        return not self.is_fixed
```

**Validation Rules** (from FR-005):

- Filename MUST match pattern:
  `YYYYMMDD_HHMMSS_<SB>MHz_averaged_*-I-image[_fixed].fits`
- `observation_date` MUST be 8 digits
- `observation_time` MUST be 6 digits
- `subband_mhz` MUST be positive integer

**State Transitions**: Immutable (frozen dataclass)

### 2. FixedFITSFile

**Purpose**: Represents a corrected FITS file with header corrections applied.

**Attributes**:

```python
@dataclass(frozen=True)
class FixedFITSFile:
    """Corrected FITS file with BSCALE/BZERO applied and WCS keywords added."""

    source_file: FITSImageFile          # Original FITS file
    fixed_path: Path                    # Path to _fixed.fits file
    corrections_applied: list[str]      # List of corrections: ["BSCALE", "BZERO", "RESTFREQ", ...]
    created_at: datetime                # Timestamp of fix operation

    @property
    def exists(self) -> bool:
        """Check if fixed file already exists on disk."""
        return self.fixed_path.exists()
```

**Validation Rules** (from FR-010, FR-011):

- `fixed_path` MUST end with `_fixed.fits`
- `corrections_applied` MUST include: BSCALE/BZERO materialization, RESTFREQ,
  SPECSYS, TIMESYS, RADESYS, PC matrix, BMAJ/BMIN

**Lifecycle**:

1. Created: When `fix_fits_headers()` generates fixed file
2. Reused: When fixed file already exists (FR-013)

### 3. TimeStepGroup

**Purpose**: Collection of FITS files sharing the same observation timestamp
across frequency subbands.

**Attributes**:

```python
@dataclass
class TimeStepGroup:
    """Collection of FITS files for a single temporal snapshot."""

    time_step_key: str                  # YYYYMMDD_HHMMSS
    files: list[FITSImageFile]          # FITS files in this time step
    frequency_range_mhz: tuple[int, int]  # (min_freq, max_freq)

    def __post_init__(self):
        """Sort files by frequency for deterministic ordering."""
        self.files.sort(key=lambda f: f.subband_mhz)

    @property
    def file_count(self) -> int:
        """Number of FITS files in this time step."""
        return len(self.files)

    @property
    def has_frequency_gaps(self) -> bool:
        """Detect gaps in frequency coverage."""
        frequencies = sorted(f.subband_mhz for f in self.files)
        expected_count = (frequencies[-1] - frequencies[0]) // frequency_step + 1
        return len(frequencies) < expected_count
```

**Validation Rules** (from FR-006, FR-007):

- All files MUST have same `time_step_key`
- Files MUST be sorted by `subband_mhz` (deterministic ordering)
- Frequency gaps logged as warnings (edge case)

**Relationships**:

- Aggregates multiple `FITSImageFile` instances
- One-to-many: One time step → many FITS files

### 4. SpatialGrid

**Purpose**: Represents the l,m coordinate grid for spatial dimensions.

**Attributes**:

```python
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class SpatialGrid:
    """Spatial coordinate grid (l, m) for image dimensions."""

    l_coords: NDArray[np.floating]      # L coordinate array
    m_coords: NDArray[np.floating]      # M coordinate array

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of spatial grid (l_size, m_size)."""
        return (len(self.l_coords), len(self.m_coords))

    @property
    def l_checksum(self) -> str:
        """MD5 checksum of L coordinates for grid validation."""
        import hashlib
        return hashlib.md5(self.l_coords.tobytes()).hexdigest()

    @property
    def m_checksum(self) -> str:
        """MD5 checksum of M coordinates for grid validation."""
        import hashlib
        return hashlib.md5(self.m_coords.tobytes()).hexdigest()

    def matches(self, other: 'SpatialGrid') -> bool:
        """Check if grids are equivalent (FR-009)."""
        return (
            self.l_checksum == other.l_checksum
            and self.m_checksum == other.m_checksum
        )
```

**Validation Rules** (from FR-008, FR-009):

- L and M coordinate arrays MUST be consistent across all time steps
- Grid mismatch MUST abort processing with detailed error
- Checksums used for efficient comparison

**Relationships**:

- Referenced by `ConversionMetadata.spatial_grid_reference`
- Validated against every `TimeStepGroup`

### 4a. WCSHeader

**Purpose**: Represents the FITS celestial WCS header for coordinate
transformations and plotting.

**Attributes**:

```python
from astropy.io.fits import Header
from astropy.wcs import WCS

@dataclass(frozen=True)
class WCSHeader:
    """FITS celestial WCS header for RA/Dec coordinate transformations."""

    header_string: str                  # WCS header as formatted string
    ra_coords: NDArray[np.floating]     # 2D right ascension array (m, l) in degrees
    dec_coords: NDArray[np.floating]    # 2D declination array (m, l) in degrees
    frame: str = "fk5"                  # Coordinate frame (default FK5)
    equinox: str = "J2000"              # Equinox (default J2000)

    @property
    def wcs(self) -> WCS:
        """Reconstruct astropy WCS object from header string."""
        return WCS(Header.fromstring(self.header_string, sep="\n"))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of coordinate arrays (m_size, l_size)."""
        return self.ra_coords.shape

    def to_dict(self) -> dict:
        """Serialize WCS header for storage in Zarr attrs."""
        return {
            "header_string": self.header_string,
            "frame": self.frame,
            "equinox": self.equinox,
        }

    @classmethod
    def from_fits(cls, fits_path: Path, spatial_grid: SpatialGrid) -> 'WCSHeader':
        """Extract WCS header from FITS file and compute RA/Dec coordinates."""
        from astropy.io import fits
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
        w2d = WCS(header).celestial

        # Compute RA/Dec at pixel centers (origin=0)
        ny, nx = spatial_grid.shape
        yy, xx = np.indices((ny, nx), dtype=float)
        ra2d, dec2d = w2d.all_pix2world(xx, yy, 0)

        header_str = w2d.to_header().tostring(sep="\n")

        return cls(
            header_string=header_str,
            ra_coords=ra2d,
            dec_coords=dec2d,
        )
```

**Validation Rules** (from FR-019a-f):

- Header string MUST be valid FITS WCS header format
- RA/Dec coordinates MUST match spatial grid shape
- RA/Dec MUST be computed at pixel centers (origin=0) for FITS standard
  alignment
- Coordinates MUST be in degrees, FK5/J2000 frame

**Storage in Zarr** (FR-019c; updated for per-time incremental ingest):

**Canonical on disk** when the store has a `time` dimension:

1. **`wcs_header_str(time)`** — one FITS header string per time step (promoted
   from 0-D per step during `_align_time_dimension_for_zarr_write` / append).
   This is the only celestial WCS metadata that updates on each
   `append_dim='time'` write.

**In-memory only** during combine/regrid (`_load_for_combine`); **not**
persisted to Zarr when `wcs_header_str` is present (stripped by
`strip_redundant_fits_wcs_header_attrs` before `_write_or_append_zarr` and on
`open_dataset`):

- Dataset global attrs: `xds.attrs['fits_wcs_header']`
- Per-variable attrs: `xds[var].attrs['fits_wcs_header']`
- Coordinate attrs on `right_ascension` / `declination`

**Why:** Zarr array-level attrs (e.g. on `SKY`) do not vary per time slice;
writing `fits_wcs_header` there after the first append freezes time-0 `CRVAL`
and breaks QA tools that read attrs instead of `wcs_header_str[time_idx]`. See
**Per-Time WCS and CRVAL** in `AGENTS.md`.

**Regression tests:**
`test_write_or_append_omits_fits_wcs_header_when_wcs_header_str_present`,
`test_open_dataset_strips_stale_fits_wcs_header_with_wcs_header_str`.

**Relationships**:

- Extracted from first FITS file in each dataset
- Provides RA/Dec coordinates for Zarr store
- Used by plotting and analysis tools (WCSAxes)

### 5. ZarrStore

**Purpose**: Represents the output Zarr data store with metadata.

**Attributes**:

```python
@dataclass
class ZarrStore:
    """Cloud-optimized Zarr storage for multi-dimensional radio astronomy data."""

    path: Path                          # Absolute path to .zarr directory
    dimensions: dict[str, int]          # {"time": 50, "frequency": 320, "l": 2048, "m": 2048}
    chunk_sizes: dict[str, int]         # {"l": 1024, "m": 1024}
    exists: bool                        # Whether store already exists on disk
    locked: bool = False                # Whether store is locked for writing

    @property
    def lock_file_path(self) -> Path:
        """Path to lock file for concurrent access control."""
        return self.path.with_suffix(self.path.suffix + ".lock")

    @property
    def state_file_path(self) -> Path:
        """Path to pipeline state file for resume/rebuild."""
        return self.path.with_suffix(self.path.suffix + ".state.json")

    def is_locked(self) -> bool:
        """Check if another process has locked this Zarr store."""
        return self.lock_file_path.exists()

    def has_incomplete_state(self) -> bool:
        """Check if pipeline was interrupted (state file exists)."""
        if not self.state_file_path.exists():
            return False
        state = PipelineState.load(self.state_file_path)
        return state.status == "in_progress"
```

**Validation Rules** (from FR-014, FR-015, FR-016):

- Path MUST end with `.zarr`
- Compatible with xarray and xradio libraries
- Support append operations (FR-015)
- Safe atomic writes (FR-017)

**State Transitions**:

1. **New**: `exists=False`, no lock, no state
2. **Writing**: `exists=True`, `locked=True`, state="in_progress"
3. **Complete**: `exists=True`, `locked=False`, state="completed"
4. **Interrupted**: `exists=True`, `locked=False`, state="in_progress"

**Relationships**:

- Output target for `ConversionMetadata`
- Managed by `PipelineState`

### 6. PipelineState

**Purpose**: Tracks conversion pipeline execution state for resume/rebuild
functionality.

**Attributes**:

```python
from enum import Enum

class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PipelineState:
    """Persistent state for pipeline execution and recovery."""

    version: str = "1.0.0"              # State schema version
    zarr_path: Path                     # Target Zarr store path
    start_time: datetime                # Pipeline start timestamp
    last_update: datetime               # Last state update timestamp
    status: PipelineStatus              # Current pipeline status
    completed_time_steps: list[str]     # List of completed time_step_keys
    current_time_step: Optional[str]    # Currently processing time step
    spatial_grid_reference: Optional[dict]  # Reference grid checksums
    total_files_processed: int = 0      # Cumulative file count
    error_message: Optional[str] = None # Last error if status=FAILED

    def save(self, path: Path) -> None:
        """Save state to JSON file with atomic write."""
        import json
        import tempfile
        # Atomic write: temp file + rename
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        temp_path.rename(path)

    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        """Load state from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        pass

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineState':
        """Deserialize from dictionary."""
        pass
```

**Validation Rules** (from FR-027, NFR-002):

- State file updated after each time step completion
- Atomic writes prevent corruption
- Version field for future schema evolution

**State Transitions**:

1. NOT_STARTED → IN_PROGRESS (pipeline starts)
2. IN_PROGRESS → IN_PROGRESS (time step completed)
3. IN_PROGRESS → COMPLETED (all time steps done)
4. IN_PROGRESS → FAILED (error encountered)
5. FAILED → IN_PROGRESS (user chooses resume)

### 7. ConversionMetadata

**Purpose**: Tracks metadata during pipeline execution for logging and
validation.

**Attributes**:

```python
@dataclass
class ConversionMetadata:
    """Metadata tracked during pipeline execution."""

    discovered_time_steps: list[TimeStepGroup]
    spatial_grid_reference: Optional[SpatialGrid] = None
    files_processed: int = 0
    errors_encountered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def add_time_step(self, group: TimeStepGroup) -> None:
        """Add a time step group to metadata."""
        self.discovered_time_steps.append(group)

    def set_reference_grid(self, grid: SpatialGrid) -> None:
        """Set the spatial grid reference for validation."""
        if self.spatial_grid_reference is not None:
            raise ValueError("Reference grid already set")
        self.spatial_grid_reference = grid

    def validate_grid(self, grid: SpatialGrid) -> None:
        """Validate grid against reference (FR-009)."""
        if self.spatial_grid_reference is None:
            self.set_reference_grid(grid)
        elif not self.spatial_grid_reference.matches(grid):
            raise ValidationError(
                "Spatial grid mismatch detected. "
                f"Expected shape {self.spatial_grid_reference.shape}, "
                f"got {grid.shape}"
            )

    @property
    def total_files(self) -> int:
        """Total number of FITS files across all time steps."""
        return sum(ts.file_count for ts in self.discovered_time_steps)

    @property
    def frequency_range_mhz(self) -> tuple[int, int]:
        """Overall frequency range across all time steps."""
        all_freqs = []
        for ts in self.discovered_time_steps:
            all_freqs.extend(f.subband_mhz for f in ts.files)
        return (min(all_freqs), max(all_freqs))
```

**Relationships**:

- Aggregates `TimeStepGroup` instances
- References `SpatialGrid` for validation
- Provides summary statistics (FR-021, FR-024)

## Value Objects

### ConversionOptions

**Purpose**: Configuration parameters for conversion process.

```python
@dataclass(frozen=True)
class ConversionOptions:
    """Configuration for FITS to Zarr conversion."""

    input_dir: Path                     # FITS input directory (FR-002)
    output_dir: Path                    # Zarr output directory (FR-003)
    zarr_name: str                      # Zarr store name (FR-004)
    fixed_dir: Path                     # Fixed FITS directory (FR-012)
    chunk_lm: int = 1024                # LM chunk size (FR-018)
    rebuild: bool = False               # Overwrite existing Zarr (FR-016)
    use_prefect: bool = False           # Enable Prefect orchestration (FR-022)
    log_level: str = "INFO"             # Logging verbosity (FR-028)

    def __post_init__(self):
        """Validate configuration."""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory not found: {self.input_dir}")
        if self.chunk_lm <= 0:
            raise ValueError(f"chunk_lm must be positive, got {self.chunk_lm}")
```

## Exceptions

### ValidationError

```python
class ValidationError(Exception):
    """Raised when validation fails (e.g., grid mismatch)."""
    pass
```

### LockAcquisitionError

```python
class LockAcquisitionError(Exception):
    """Raised when Zarr store is locked by another process (FR-019)."""

    def __init__(self, zarr_path: Path, lock_holder_pid: Optional[int] = None):
        self.zarr_path = zarr_path
        self.lock_holder_pid = lock_holder_pid
        message = (
            f"Zarr store is locked by another process: {zarr_path}\n"
            f"Lock holder PID: {lock_holder_pid or 'unknown'}\n"
            "Wait for the other process to finish or manually remove the lock file."
        )
        super().__init__(message)
```

## Entity Relationships

```text
ConversionOptions
    ↓ provides config
ConversionMetadata
    ↓ aggregates
TimeStepGroup (many)
    ↓ contains
FITSImageFile (many)
    ↓ may require
FixedFITSFile (optional)

ConversionMetadata
    ↓ validates against
SpatialGrid (reference)

ConversionOptions
    ↓ specifies target
ZarrStore
    ↓ manages state via
PipelineState
```

## Data Flow

1. **Discovery**: `input_dir` → `list[FITSImageFile]` → `list[TimeStepGroup]`
2. **Fixing**: `FITSImageFile` → `FixedFITSFile` (if needed)
3. **Validation**: `TimeStepGroup` → extract `SpatialGrid` → validate against
   reference
4. **Conversion**: `TimeStepGroup` + `FixedFITSFile` → xarray Dataset
5. **Writing**: xarray Dataset → `ZarrStore` (append or create)
6. **State**: `PipelineState` updated after each time step

## Summary

All data models align with:

- **Functional Requirements**: FR-001 through FR-032 coverage
- **Non-Functional Requirements**: NFR-001 through NFR-005 support
- **Constitution**: Type hints required, immutable where appropriate, validation
  rules enforced
- **Testing**: All models are testable with clear validation rules
