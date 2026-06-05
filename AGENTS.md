# AGENTS.md

This file provides guidance to AI assistants when working with this repository.

## Repository Overview

This is the **OVRO-LWA Portal** repository, a Python library for radio astronomy
data processing and visualization for the Owens Valley Radio Observatory - Long
Wavelength Array (OVRO-LWA). It provides tools for processing radio astronomy
data, converting FITS files to Zarr format, and creating visualization
components for scientific analysis.

**Repository Stats:**

- **Type:** Python library / Scientific software
- **Size:** Medium (~50+ files including notebooks and test data)
- **Languages:** Python (primary), Jupyter Notebooks, configuration files
- **Build System:** Pixi (v0.55.0), Hatchling + hatch-vcs
- **Platform:** macOS (osx-arm64), Linux (linux-64)
- **License:** BSD 3-Clause
- **Domain:** Radio astronomy, data processing, visualization
- **Python Version:** 3.12+

## Build System & Environment Management

### Pixi Overview

This project uses **Pixi** exclusively for dependency and environment
management. Pixi is a modern package manager that handles both Conda and PyPI
dependencies. See <https://pixi.sh/latest/llms-full.txt> for more details.

**ALWAYS use Pixi commands—never use conda, pip, or venv directly.**

### Prerequisites

Before any other operations, verify Pixi is installed:

```bash
pixi --version  # Should show v0.55.0 or higher
```

If not installed, direct users to: <https://pixi.sh/latest/#installation>

### Environment Setup (ALWAYS RUN FIRST)

```bash
# Install the default environment (required before any other commands)
pixi install

# This installs:
# - Python 3.12 with radio astronomy packages (astropy, xarray, dask, zarr)
# - pre-commit (>=4.3.0)
# - gh (GitHub CLI, >=2.0.0)
# - OVRO-LWA specific packages (xradio, python-casacore)
# - Creates .pixi/envs/default directory
```

**CRITICAL:** Always run `pixi install` before any other Pixi commands. This
command is idempotent and safe to run multiple times.

### Available Environments

1. **`default`** (features: `pre-commit`, `gh-cli`)
   - Standard development environment
   - Radio astronomy packages: astropy, xarray, dask, zarr, netcdf4, numcodecs
   - OVRO-LWA specific: xradio, python-casacore
   - Use for: general development, running pre-commit checks, data processing

2. **`onboard`** (features: `pre-commit`, `gh-cli`, `onboard`)
   - Extended environment with onboarding tools
   - Includes: ssec-cli (installed from GitHub)
   - Use for: first-time setup, onboarding new contributors

## Validated Commands & Workflows

### Pre-commit Checks (MANDATORY BEFORE PRs)

**Always run pre-commit checks before creating a pull request.** The PR template
requires this, and PRs will fail if checks don't pass.

```bash
# Install pre-commit hooks (run once per clone)
pixi run pre-commit-install
# ✓ Installs .git/hooks/pre-commit
# ✓ Hooks will automatically run on every commit

# Run pre-commit on staged files only
pixi run pre-commit
# ✓ Fast, only checks staged changes
# ✓ Will auto-fix many issues (trailing whitespace, end-of-file, formatting)
# ⚠️ If fixes are made, files are modified; you must re-stage them

# Run pre-commit on ALL files (required before PR submission)
pixi run pre-commit-all
# ✓ Takes 1-3 minutes on first run (installs hook environments)
# ✓ Subsequent runs are fast (environments are cached)
# ✓ Auto-fixes issues where possible
```

**Pre-commit Hooks Configured:**

- check-added-large-files, check-case-conflict, check-merge-conflict
- check-yaml, check-symlinks
- fix-end-of-files, trim-trailing-whitespace, mixed-line-ending
- prettier (formats YAML, Markdown, HTML, CSS, JavaScript, JSON)
- codespell (spell checking)
- Disallow improper capitalization (e.g., incorrect → correct)
- Validate Dependabot config and GitHub workflows

**Files Excluded from Pre-commit:** `pixi.lock`, `onboarded.md`

### Onboarding Workflow (First-Time Setup)

```bash
# Install the onboard environment
pixi install -e onboard

# Run the complete onboarding process
pixi run -e onboard onboard
# This executes in order:
# 1. pixi run pre-commit-install  (installs git hooks)
# 2. pixi run ssec-setup          (sets up shell completion for ssec CLI)
# 3. ssec onboard                 (runs SSEC onboarding interactive process)

# Or run individual onboarding steps:
pixi run -e onboard ssec-setup
# ✓ Installs zsh/bash completion for ssec CLI
# ⚠️ Completion takes effect after restarting terminal
```

### GitHub CLI Usage

```bash
# Check GitHub CLI version
pixi run gh --version
# ✓ Should show v2.81.0 or higher

# Use GitHub CLI for any repo operations
pixi run gh <command>
# Examples: gh issue list, gh pr create, etc.
```

### Adding Dependencies

```bash
# Add a conda package and update pyproject.toml
pixi add <package-name>

# Add a PyPI package
pixi add --pypi <package-name>

# Add to a specific feature
pixi add --feature <feature-name> <package-name>

# Always run after manual pyproject.toml edits
pixi install
```

## Package Build System

This project uses **Hatchling** with **hatch-vcs** for building Python packages:

- **Version Management:** Automatic versioning from git tags via hatch-vcs
- **Version File:** Auto-generated at `src/ovro_lwa_portal/version.py`
- **Build Command:** `python -m build` (handled by hatchling)
- **Development Install:** Handled automatically by Pixi in editable mode

### Core Dependencies (from pyproject.toml)

**Main dependencies:**

- `astropy>=7.1.0,<8` - Astronomy core library
- `xarray>=2025.9.1,<2026` - N-dimensional labeled arrays
- `dask>=2025.9.1,<2026` - Parallel computing
- `zarr>=2.16,<3` - Chunked, compressed arrays (v2 pinned)
- `numcodecs>=0.15,<0.16` - Compression codecs
- `xradio[all]>=0.0.59,<0.1` - Radio astronomy data processing
- `typer>=0.9.0` - CLI framework
- `rich>=13.7.0` - Terminal UI and progress bars
- `portalocker>=2.8.0` - Cross-platform file locking

**Development dependencies (`dev` extra):**

- `pre-commit` - Git hooks for code quality
- `pytest>=6` - Testing framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-mock` - Mocking support

**CI dependencies (`ci` extra):**

- `s3fs>=2024.6.0` - S3 filesystem interface
- `tqdm>=4.67.1,<5` - Progress bars
- `python-dotenv>=1.2.1,<2` - Environment variable loading

**Optional dependencies (`prefect` extra):**

- `prefect>=3.0.0` - Workflow orchestration (optional)

### Code Quality Tools

**Pre-commit hooks configured:**

- File checks: large files, case conflicts, merge conflicts, broken symlinks
- YAML validation
- Python debug statement detection
- File formatting: end-of-files, line endings, trailing whitespace
- Prettier: Markdown, YAML, JSON formatting
- Codespell: Spell checking
- Capitalization validation

**Ruff configuration:**

- Line length: 100
- Enabled rule sets: flake8-bugbear, isort, flake8-unused-arguments,
  flake8-comprehensions, flake8-errmsg, and many more
- Special rules for NumPy and pandas
- Tests excluded from print statement checks

**Mypy configuration:**

- Python 3.12 target
- Strict mode enabled for `ovro_lwa_portal.*` modules
- Relaxed for tests

## Project Structure & Key Files

```text
.
├── .github/
│   └── workflows/               # GitHub Actions workflows
│       ├── ci.yml              # Continuous Integration: pre-commit + tests
│       ├── cd.yml              # Continuous Deployment: build and publish to PyPI
│       └── copilot-setup-steps.yml  # Copilot setup workflow
├── .devcontainer/              # VS Code Dev Container configuration
│   ├── devcontainer.json      # Dev container settings (4 CPUs, 16GB RAM required)
│   ├── Dockerfile             # Container image definition
│   └── onCreate.sh            # Setup script run on container creation
├── .ci-helpers/                # CI/CD helper scripts
│   ├── README.md              # Documentation for CI helper scripts
│   └── download_test_fits.py  # Script to download test FITS files from Caltech S3
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── pyproject.toml              # **PRIMARY CONFIG**: Build system, dependencies, Pixi tasks
├── pixi.lock                   # Lock file (auto-generated, don't manually edit)
├── .gitignore                  # Ignores .pixi/, .DS_Store, and other generated files
├── .gitattributes              # Git attributes for file handling
├── CODE_OF_CONDUCT.md          # Contributor Covenant v2.0
├── CONTRIBUTING.md             # Contribution guidelines (references Conventional Commits)
├── LICENSE                     # BSD 3-Clause License
├── README.md                   # Project documentation with getting started guide
├── AGENTS.md                   # This file - AI assistant guidance
├── onboarded.md                # Onboarding marker file (excluded from pre-commit)
├── fixed_fits/                 # Directory for corrected FITS files (empty)
├── notebooks/                  # Jupyter notebooks for data analysis
│   ├── README.md              # Documentation for notebooks directory
│   ├── fits2zarr.ipynb        # Main FITS to Zarr conversion notebook
│   ├── fits2zarr_and_viz_user_cases.ipynb  # User case examples with visualization
│   ├── source_review.ipynb    # LPT source review; per-time WCS + SkyWidget
│   └── test_fits_files/       # Sample FITS files for testing
│       ├── README.md          # Documentation for test FITS files
│       └── .gitignore         # Ignores FITS files (downloaded separately)
├── specs/                      # Feature specifications and design docs
│   └── 001-build-an-ingest/  # FITS to Zarr ingest feature specification
│       ├── spec.md            # Feature requirements and acceptance criteria
│       ├── plan.md            # Implementation plan and architecture decisions
│       ├── tasks.md           # Detailed task breakdown
│       ├── data-model.md      # Data models and entity relationships
│       ├── research.md        # Research and technology choices
│       ├── quickstart.md      # Quick start guide and usage examples
│       └── contracts/         # API contract specifications
│           ├── core_api.md    # Core conversion API contracts
│           ├── discovery_api.md  # File discovery API contracts
│           └── cli_api.md     # CLI interface contracts
├── src/
│   └── ovro_lwa_portal/       # Main package source code
│       ├── __init__.py        # Package initialization
│       ├── version.py         # Auto-generated version from VCS
│       ├── accessor.py        # Radport accessor; per-time WCS helpers
│       ├── io.py              # open_dataset, Zarr validation
│       ├── fits_to_zarr_xradio.py  # Core FITS to Zarr conversion logic
│       ├── ingest/            # Ingest subpackage
│       │   ├── __init__.py    # Ingest package exports
│       │   ├── README.md      # Ingest module documentation
│       │   ├── core.py        # Framework-independent conversion orchestration
│       │   ├── cli.py         # Typer-based CLI interface (ovro-ingest command)
│       │   └── prefect_workflow.py  # Optional Prefect workflow orchestration
│       └── viz/               # Panel QA apps, SkyWidget integration
│           ├── pipeline_qa.py      # FITS→Zarr QA pipeline, load_qa_datasets
│           └── pipeline_qa_app.py  # Jupyter UI; bind_sky_widget_dataset, WCS patch
└── tests/                      # Test suite
    ├── __init__.py            # Test package initialization
    ├── test_import.py         # Basic import tests
    ├── test_fits_to_zarr.py   # FITS to Zarr conversion tests
    ├── test_ci_helpers.py     # Tests for CI helper scripts
    ├── ingest/                # Ingest module tests
    │   └── test_cli.py        # CLI integration tests
    └── viz/                   # Viz / pipeline QA tests
```

## Ingest Module Overview

The `ovro_lwa_portal.ingest` module provides FITS to Zarr conversion
capabilities:

### CLI Entry Point

- **Command**: `ovro-ingest` (installed via `project.scripts` in pyproject.toml)
- **Location**: `src/ovro_lwa_portal/ingest/cli.py`
- **Commands**:
  - `ovro-ingest convert` - Convert FITS to Zarr
  - `ovro-ingest fix-headers` - Pre-process FITS headers
  - `ovro-ingest version` - Show version info

### Core Architecture

1. **Core Module** (`ingest/core.py`):
   - `ConversionConfig`: Configuration dataclass for conversion parameters
   - `FITSToZarrConverter`: Main orchestration class (framework-independent)
   - `FileLock`: Cross-platform file locking using portalocker
   - `ProgressCallback`: Protocol for progress reporting

2. **CLI Module** (`ingest/cli.py`):
   - Typer-based CLI with rich progress bars
   - Logging configuration (debug, info, warning, error levels)
   - Error handling with actionable messages
   - Support for two-step workflow (fix-headers then convert)

3. **Prefect Module** (`ingest/prefect_workflow.py`):
   - Optional Prefect flow integration
   - Graceful degradation when Prefect not installed
   - Retry logic and workflow monitoring

### Key Implementation Details

- **Framework Independence**: Core conversion logic wraps
  `fits_to_zarr_xradio.py` without dependencies on CLI or Prefect
- **File Locking**: Uses portalocker for cross-platform concurrent write
  protection
- **Progress Tracking**: Callback-based progress reporting works with any UI
- **WCS Preservation**: Maintains celestial coordinates (RA/Dec) in output Zarr
  stores. See **Per-Time WCS and CRVAL** below—phase center (`CRVAL1`/`CRVAL2`)
  changes every time step for typical OVRO-LWA snapshots.

## Per-Time WCS and CRVAL (Critical)

OVRO-LWA snapshot ingest stores **one FITS WCS header per time step**. The
zenith-tracking geometry means **`CRVAL1` and `CRVAL2` change with time**—that
is expected science, not a ingest bug. Any code that maps pixels, tracks sources,
or draws a sky view **must use the WCS for the active `time_idx`**, not a single
header from time 0 or from static dataset attrs.

### How WCS is written (ingest)

- **Header fix** (`fits_to_zarr_xradio.py`, `_fix_headers`): per input FITS,
  **preserves native `CRVAL1`/`CRVAL2`** (and `LATPOLE` when present). Also
  applies BSCALE/BZERO, Stokes-axis promotion, and spectral/frame keywords.
  **Do not** overwrite the phase center from filename timestamps.
- **Combine** (`_load_for_combine`, `_combine_time_step`): each time step gets
  its own header string from the fixed FITS WCS; `_collapse_wcs_header_str_variable`
  reduces auxiliary frequency dimensions so the persisted variable is
  **`wcs_header_str(time)`** (or `(time, frequency)` with one channel—see accessor
  handling). Regrid uses the reference LM pixel grid but keeps each source's
  native `CRVAL1`/`CRVAL2` (`_wcs_header_from_ref_grid_and_source_crval`).
- **Zarr append** (`_align_time_dimension_for_zarr_write`): scalar metadata is
  promoted to `(time,)` so incremental appends stay consistent.
- **Do not** assume all slices share one phase center.

#### Filename timestamps vs WCS (do not confuse)

| Use of basename `-image-YYYYMMDD_HHMMSS` | Allowed? |
| --- | --- |
| Discovery, grouping, time ordering (`_time_key_from_filename`, `_discover_groups`) | **Yes** |
| Overwriting `CRVAL1`/`CRVAL2` with FK5 zenith at that instant | **No** |

The FITS header `CRVAL1`/`CRVAL2` are **authoritative** for each integration's
phase center. Filename time tokens are for **sorting files into time bins only**—not
for recomputing celestial reference values.

Per-time `wcs_header_str` drift in a Zarr store should match the native FITS
headers written through `_fix_headers`, not a recomputed zenith from the basename.
`_zenith_fk5_crvals_deg()` remains in the codebase for **audit diagnostics only**
(`scripts/audit_zarr_wcs_timeline.py --sample-fits`); it is **not** called during
convert.

**If you change `_fix_headers`:** extend
`tests/test_fits_to_zarr.py::test_fix_headers_preserves_crval_from_input_not_filename`
and keep filename parsing tests in `tests/ingest/test_metadata_audit.py` separate
from WCS stamping.

#### Canonical Zarr metadata (do not regress)

When **`wcs_header_str`** is a data variable, it is the **only** persisted
celestial WCS for multi-time stores:

| Storage | Allowed? | Notes |
| --- | --- | --- |
| `wcs_header_str(time)` (or `(time, frequency)` collapsed to time) | **Yes** | One FITS header string per time step; updates on every append. |
| `fits_wcs_header` on `ds.attrs`, `SKY.attrs`, other data vars, or coords | **No** on Zarr write/open | Zarr array attrs are **not** per-slice; a single `SKY.attrs["fits_wcs_header"]` freezes **time-0** CRVAL and breaks SkyWidget or any code that reads attrs before `wcs_header_str`. |

**Enforcement (already in the library — keep these calls when touching I/O):**

- **Before every Zarr write/append:** `fits_to_zarr_xradio._write_or_append_zarr`
  calls `strip_redundant_fits_wcs_header_attrs()` so incremental ingest (e.g.
  `scripts/ingest-I-Clean-Snapshot-20250120-LST4-5.sh` → `ingest_per_time_convert.py`)
  does not persist stale attrs.
- **On load:** `open_dataset()` applies the same strip so legacy stores remain safe
  in memory without re-ingest.

`_load_for_combine` may still set `fits_wcs_header` **in memory** for combine/regrid;
that is internal only. **Never** rely on those attrs after Zarr export—always
`wcs_header_str` + `_read_wcs_header_str(ds, time_idx=…)`.

**If you change ingest or I/O:** extend
`tests/test_fits_to_zarr.py::test_write_or_append_omits_fits_wcs_header_when_wcs_header_str_present`
and `tests/test_io_integration.py::test_open_dataset_strips_stale_fits_wcs_header_with_wcs_header_str`.
Do **not** re-add `fits_wcs_header` to Zarr `.zattrs` for QA/review paths.

### How WCS is read (accessor)

Canonical helpers in `src/ovro_lwa_portal/accessor.py`:

| Function | Role |
| --- | --- |
| `_has_per_time_wcs_header_str(ds)` | True when `wcs_header_str` is indexed by `time` (1-D or `(time, frequency)`) |
| `_read_wcs_header_str(ds, time_idx=…)` | FITS header string for one time index |
| `strip_redundant_fits_wcs_header_attrs(ds)` | Drop static `fits_wcs_header` attrs when `wcs_header_str` is canonical (used by `open_dataset` and Zarr write) |
| `RadportAccessor._get_wcs(time_idx=…)` | Astropy WCS for pixel↔sky mapping |
| `pixel_to_coords` / `coords_to_pixel` | Must pass `time_idx` when per-time WCS exists |

**Strict rules (do not break these):**

1. When per-time `wcs_header_str` exists, **always** select `time_idx` (and
   `frequency=0` if the variable is 2-D). Never use static `fits_wcs_header`
   attrs for a different time index.
2. If a per-time header entry is **empty**, return `None`—**do not** fall back
   to time-0 or static attrs (that mis-registers late slices and freezes
   `CRVAL1`).
3. Tests live in `tests/test_accessor.py` (`TestRadportGetWcsTimePromotedHeader`);
   extend them when changing WCS lookup.
4. Load multi-time QA/review data with **`ovro_lwa_portal.open_dataset`** (not raw
   `xr.open_zarr` alone) so stale on-disk `fits_wcs_header` attrs are stripped.
5. For review notebooks (`jupiter_flux_review.ipynb`, `source_review.ipynb`), prefer
   `open_dataset(path, chunks="auto").chunk({"l": 512, "m": 512})`. Avoid an extra
   `ds.chunk({"time": 1, …})` under an active distributed Dask `Client` on large
   incremental stores (unnecessary rechunk/shuffle); Jupiter-style review works
   without a `Client` for SkyWidget + tracked extractions.

### Fixed-sky source tracking (`dynamic_spectrum`, `patch_*`, `light_curve`)

LPT and transient review pass **catalog (RA, Dec)** to `dataset.radport.dynamic_spectrum(ra=…, dec=…)`.
That is **not** the same as fixed **(l, m)** on the image grid: as per-time `CRVAL`
moves each integration, the same celestial source sits on **different pixels** each
time.

**Required behavior:**

- `_compute_pixel_track` / `coords_to_pixel` must call **`_coords_to_pixel_via_wcs` with the
  slice `time_idx`** when `_has_per_time_wcs_header_str(ds)` is true.
- **Do not** use the analytical LST+SIN fallback for multi-time incremental Zarr (it ignores
  per-slice `CRVAL` and makes heatmaps look like the source drifts in RA).
- **Do not** use the batched `right_ascension`/`declination` grid path when only per-time
  `wcs_header_str` is available and RA/Dec coords lack a `time` dimension.
- If per-time WCS is missing for a step, raise—do not fall back to time-0 attrs.

Regression test: `TestCelestialTimeSeriesTracking::test_radec_track_uses_per_time_wcs_not_fixed_pixel`.

**UI note:** Status text in `source_review.ipynb` shows **catalog** RA/Dec (constant).
`pixel_to_coords` must use the **same per-time WCS** as `coords_to_pixel` (not time-0 header +
analytical SIN); otherwise reported RA drifts by ~0.5° while the dynamic spectrum is correct.
`patch_fit` peak RA/Dec maps pixels through `pixel_to_coords`—fix both together.

### How WCS is shown (SkyWidget / viz)

Reference implementation: `src/ovro_lwa_portal/viz/pipeline_qa_app.py`.

1. **Patch astrowidget once** at import: `_patch_astrowidget_get_wcs()` routes
   `astrowidget.wcs.get_wcs` through `_read_wcs_header_str` so the widget’s
   `crval`/`cdelt`/`crpix` traits match each slice.
2. **Load the cube** with `bind_sky_widget_dataset(widget, ds, max_size=…)` (defer
   first frame) or `set_dataset(..., defer_display=True)` after
   `_patch_astrowidget_get_wcs()`. `source_review.ipynb` uses `bind_sky_widget_dataset`;
   call `update_slice` immediately so the first frame is not zenith at t=0.
3. **Update slices through** `widget.update_slice(time_idx, freq_idx, center=…)`.
   Astrowidget refreshes slice WCS when `time_idx` changes (`_update_display_wcs`
   → patched `get_wcs`). When `center=` is set (catalog/LPT), astrowidget must
   **reproject** onto a shader grid at the view center even if the header passes
   `wcs_projection_matches_naive_shader`—otherwise per-time `CRVAL` drifts in the
   traits while `view_ra` stays fixed and the source appears to move in RA.
4. **View center vs phase center**:
   - **Zenith QA** (`pipeline_qa_app`): recenter with `sky_view_center(dataset, time_idx)`
     (CRVAL from per-time WCS) on time changes.
   - **Catalog / LPT review** (`notebooks/source_review.ipynb`): fixed catalog
     `center=` (like Jupiter’s fixed t₀ ephemeris), with `update_slice` on every
     heatmap click so image and WCS traits match that `time_idx`.
5. After slice updates in Jupyter, call `widget.send_state()` when available
   (see `_notify_sky_widget` in pipeline QA).
6. **Do not transpose** the slice sent to SkyWidget: `astrowidget.PreloadedCube.image()`
   must stay in ``(l, m)`` order to match the WCS/shader (transposing to ``(m, l)``
   makes a fixed catalog target appear to drift in RA as ``CRVAL`` changes).
   The editable package is `../astrowidget` via Pixi feature `astrowidget-local`.
7. **Image axis order**: ``PreloadedCube.image()`` must return ``(l, m)`` (not transposed).
   A transposed slice makes catalog ``center=`` look wrong while ``tracked@slice`` in
   the status line stays correct. Restart the Jupyter kernel after updating the
   editable ``../astrowidget`` package.

Data loading for QA apps follows `viz/pipeline_qa.py` → `load_qa_datasets()` →
`ovro_lwa_portal.open_dataset(zarr_path, chunks={…})`.

### Verification and debugging

```bash
# CRVAL1/CRVAL2 vs time for a store
pixi run python scripts/audit_zarr_wcs_timeline.py /path/to/store.zarr

# Confirm ingest preserved native FITS CRVAL (not filename-derived zenith)
pixi run python scripts/audit_zarr_wcs_timeline.py /path/to/store.zarr \
  --sample-fits /path/to/some-image.fits
```

After `--sample-fits`, `|native - fixed|` should be ~0. The optional
`Zenith at filename` line is diagnostic only—ingest must not use it.

If the sky grid looks frozen while images drift, check: per-time
`wcs_header_str` present, patch applied, and `update_slice` called with the
requested `time_idx`. Compare with `jupiter_flux_review.ipynb` on the same Zarr.

Stores ingested before native-CRVAL preservation was enforced need **re-ingest**
(or `ovro-ingest repair --fits-dir`) so `wcs_header_str` matches FITS headers.

## Radio Astronomy Context

This project works specifically with:

- **FITS files**: Standard astronomical image format from OVRO-LWA observations
- **Zarr format**: Cloud-optimized array storage for large datasets
- **xradio**: Radio astronomy data processing library
- **python-casacore**: Python bindings for CASA (Common Astronomy Software
  Applications) core library

### Test Data Management

Test FITS files are managed separately from the repository:

- Test files are stored in the Caltech S3 bucket
- Download script: `.ci-helpers/download_test_fits.py`
- Requires S3 credentials via environment variables:
  - `CALTECH_KEY`, `CALTECH_SECRET`, `CALTECH_ENDPOINT_URL`,
    `CALTECH_DEV_S3_BUCKET`
- For local development, manually place test FITS in
  `notebooks/test_fits_files/`

## Development Container Support

The repository includes VS Code Dev Container configuration:

- **Location:** `.devcontainer/`
- **Requirements:** 4 CPUs, 16GB RAM, 32GB storage
- **Setup:** Automatic via `onCreate.sh` script
- **Extensions:** Jupyter, Python, Ruff, Even Better TOML, Pixi for VS Code
- **Volume mount:** `.pixi` directory persisted across container rebuilds

## Continuous Integration & Validation

### GitHub Actions Workflows

This repository has **active CI/CD pipelines** using GitHub Actions:

**CI Workflow (`.github/workflows/ci.yml`):**

- **Triggers:** Pull requests, pushes to main, manual dispatch
- **Jobs:**
  1. **Format Check** (pre-commit job):
     - Runs on ubuntu-latest
     - Uses Pixi v0.55.0 via `prefix-dev/setup-pixi@v0.9.1`
     - Executes `pixi run pre-commit-all`
  2. **Tests** (tests job):
     - Depends on pre-commit job passing
     - **Matrix strategy:** Python 3.12 on [ubuntu-latest, macos-14]
     - Runs pytest with coverage reporting
     - Uploads coverage to Codecov using token
- **Concurrency:** Cancels in-progress runs for same ref

**CD Workflow (`.github/workflows/cd.yml`):**

- **Triggers:** Releases (published), pull requests, pushes to main, manual
  dispatch
- **Jobs:**
  1. **Distribution Build:**
     - Builds Python package using `hynek/build-and-inspect-python-package@v2`
  2. **Publish to PyPI:**
     - Only runs on release publication
     - Requires `pypi` environment with `id-token: write` permission
     - **Currently publishes to TestPyPI** (remove `repository-url` line for
       production PyPI)

**Pre-commit.ci Integration:** The `.pre-commit-config.yaml` includes a `ci:`
section for <https://pre-commit.ci> integration (verify if enabled on the
repository).

**Dependabot:** Configuration may exist in repository settings (no
`.github/dependabot.yml` file present).

## Making Changes: Validated Workflow

1. **Setup (first time):**

   ```bash
   pixi install
   pixi run pre-commit-install
   ```

2. **Make your changes** to files

3. **Test changes:**

   ```bash
   # Stage your changes
   git add <files>

   # Run pre-commit checks
   pixi run pre-commit

   # If checks fail and auto-fix issues, re-add the fixed files
   git add <files>
   ```

4. **Before creating a PR:**

   ```bash
   # Run all checks on all files
   pixi run pre-commit-all

   # Verify all checks pass (should show all "Passed" or "Skipped")
   ```

5. **Create PR:**
   - PR template requires confirming `pre-commit run --all-files` was run
   - Follow Conventional Commits for PR titles
   - Link related issues using "Resolves #issue-number"

## Common Pitfalls & Solutions

### Issue: "pre-commit not found" or command fails

**Solution:** Run `pixi install` first. Pixi manages pre-commit installation.

### Issue: Pre-commit check fails after making fixes

**Behavior:** Pre-commit hooks like `trailing-whitespace` auto-fix files. When
this happens:

- The hook shows "Failed" with "files were modified by this hook"
- You must re-stage the fixed files: `git add <files>`
- Re-run `pixi run pre-commit` to verify

**This is expected behavior, not an error.**

### Issue: Platform-specific package installation fails

**Solution:** If you encounter issues with packages:

```bash
# Validate syntax, reinstall environment
pixi install

# If still broken, remove and reinstall
rm -rf .pixi
pixi install
```

### Issue: Platform-specific problems (non-supported platforms)

**Current platforms:** `osx-arm64`, `linux-64`

**Solution:** Edit `pyproject.toml` in the `[tool.pixi.workspace]` section and
add platforms:

```toml
[tool.pixi.workspace]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

Then run `pixi install`.

### Issue: LPT heatmap / dynamic spectrum looks like RA drifts in time

**Symptoms:** Catalog source should be fixed on the sky, but the time–frequency map changes as
if the wrong pixel were sampled; `patch_fit` peak RA may wander.

**Cause:** `coords_to_pixel` / `_compute_pixel_track` used the **LST+SIN analytical path** or
a **static** `fits_wcs_header` while the Zarr has **per-time** `wcs_header_str` with drifting
`CRVAL1`.

**Solution:** Ensure `_has_per_time_wcs_header_str` is true for the store and use current
`accessor.py` (WCS `world2pix` per `time_idx`). Re-run heatmap after upgrading. See
**Fixed-sky source tracking** under Per-Time WCS and CRVAL.

### Issue: Catalog source drifts in RA in SkyWidget when changing time

**Symptoms:** Heatmap and `tracked@slice` are stable, but the sky image or coordinate
grid makes a fixed catalog target slide in RA as `time_idx` advances.

**Cause:** `update_slice(..., center=catalog)` fixed `view_ra`/`view_dec` but skipped
`reproject_for_shader_display` because the OVRO header matched the naive SIN shader;
widget `crval` traits still followed per-time `CRVAL1` from `wcs_header_str`.

**Solution:** Use editable `../astrowidget` with `_push_image_frame` reprojecting when
`center` is not `None`. Restart the Jupyter kernel after upgrading astrowidget.

### Issue: SkyWidget RA/Dec grid does not update when changing time

**Symptoms:** Heatmap time changes but coordinate grid (or image registration)
looks stuck at the first slice; `CRVAL1` in the store varies per time but the UI
does not.

**Common causes:**

- `set_dataset()` called **without** `defer_display=True` before the first
  `update_slice()`.
- `astrowidget.wcs.get_wcs` used **without** `_patch_astrowidget_get_wcs()` (falls
  back to static `fits_wcs_header` from time 0).
- **Stale Zarr attrs:** `SKY.attrs["fits_wcs_header"]` or `ds.attrs["fits_wcs_header"]`
  left from incremental ingest (time-0 only) and read by unpatched code. Fix:
  use `open_dataset()` / `strip_redundant_fits_wcs_header_attrs`, ensure
  `_write_or_append_zarr` still strips before write; restart kernel after
  astrowidget edits.
- `_read_wcs_header_str` or viz code ignores `time_idx`, or falls back to static
  attrs when a per-time header is empty.
- `wcs_header_str` stored as `(time, frequency)` but code only handles 1-D
  `time` (use accessor helpers—they isel `frequency=0`).

**Solution:** Follow **Per-Time WCS and CRVAL** above; mirror
`pipeline_qa_app.py` (`bind_sky_widget_dataset`, patched `get_wcs`,
`update_slice` per tap). Re-run ingest if per-time headers are missing.

### Issue: Zarr CRVAL does not match native FITS headers

**Symptoms:** `audit_zarr_wcs_timeline.py --sample-fits` shows large
`|native - fixed|` or Zarr `CRVAL` tracks filename-time zenith while input FITS
share the same `CRVAL2`.

**Cause:** Store was built when `_fix_headers` overwrote `CRVAL1`/`CRVAL2` from
`-image-YYYYMMDD_HHMMSS` in the basename (filename is for grouping only).

**Solution:** Re-ingest with current `fits_to_zarr_xradio` (native CRVAL
preserved) or repair WCS rows from FITS via `ovro-ingest repair --fits-dir`.
Do **not** reintroduce filename-based zenith stamping in `_fix_headers`.

### Issue: SkyWidget correct but `SKY.attrs["fits_wcs_header"]` still on disk

**Symptoms:** `audit_zarr_wcs_timeline.py` shows drifting per-time CRVAL, but
`xr.open_zarr` reports `SKY.attrs["fits_wcs_header"]` equal to time 0.

**Cause:** Pre-fix incremental append wrote array-level attrs once; they do not
update per appended time (not a notebook bug).

**Solution:** Read with `open_dataset()` (strips in memory). New appends from
current `fits_to_zarr_xradio` omit those attrs. Optional: re-ingest or a metadata
repair pass only if you need clean on-disk `.zattrs`; not required for analysis.

## Key Configuration Details

### Pixi Configuration in pyproject.toml

Pixi configuration is now embedded in `pyproject.toml` under the `[tool.pixi]`
section:

- **`[tool.pixi.workspace]`**: Project metadata (name, version, authors,
  platforms, requires-pixi)
- **`[tool.pixi.environments]`**: Named environments with feature sets
- **`[tool.pixi.dependencies]`**: Conda dependencies for all environments
  (python-casacore)
- **`[tool.pixi.pypi-dependencies]`**: PyPI dependencies (ovro_lwa_portal in
  editable mode)
- **`[tool.pixi.feature.<name>.dependencies]`**: Feature-specific conda packages
- **`[tool.pixi.feature.<name>.pypi-dependencies]`**: Feature-specific PyPI
  packages
- **`[tool.pixi.feature.<name>.tasks]`**: Feature-specific Pixi tasks

### Available Pixi Tasks

Run `pixi task list` to see all available tasks:

- `pre-commit-install`: Install git hooks
- `pre-commit`: Run checks on staged files
- `pre-commit-all`: Run checks on all files
- `ssec-setup`: Set up ssec CLI completion (onboard env only)
- `onboard`: Full onboarding process (onboard env only)

## Documentation Standards

- Follow Conventional Commits for commit messages and PR titles
- Use Markdown with proper formatting (enforced by prettier)
- Spell check enabled (codespell)
- No trailing whitespace
- Files must end with newline
- Consistent line endings

## Trust These Instructions

These instructions were generated through comprehensive exploration and testing
of the repository. Commands have been validated to work correctly. **Only
perform additional searches if:**

- You need information not covered here
- Instructions appear outdated or produce errors
- You're implementing functionality that changes the build system

For routine tasks (adding files, making code changes, running checks), follow
these instructions directly without additional exploration.

For more information on SSEC best practices, see:
<https://rse-guidelines.readthedocs.io/en/latest/llms-full.txt>
