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
│           ├── pipeline_qa_app.py  # Jupyter UI; bind_sky_widget_dataset, WCS patch
│           ├── source_review.py      # Pure Center/load orchestration (testable)
│           ├── source_review_data.py # Heatmap helpers, known-sources I/O
│           ├── source_review_app.py  # SourceReview Panel app class
│           └── panel_ui_session.py   # PanelUISession backends (Jupyter vs headless)
└── tests/                      # Test suite
    ├── __init__.py            # Test package initialization
    ├── test_import.py         # Basic import tests
    ├── test_fits_to_zarr.py   # FITS to Zarr conversion tests
    ├── test_ci_helpers.py     # Tests for CI helper scripts
    ├── ingest/                # Ingest module tests
    │   └── test_cli.py        # CLI integration tests
    └── viz/                   # Viz / pipeline QA tests
        ├── panel_ui_testkit.py        # PanelUITestHarness, QueuedIOLoop
        ├── test_panel_ui_session.py   # PanelUISession + harness
        ├── test_source_review_ui_integration.py  # Comm integration (inline + queued io_loop)
        ├── test_source_review.py      # Center/load orchestration tests
        ├── test_source_review_app.py  # SourceReview Panel app smoke tests
        └── test_source_review_data.py # Heatmap data helper tests
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
is expected science, not a ingest bug. Any code that maps pixels, tracks
sources, or draws a sky view **must use the WCS for the active `time_idx`**, not
a single header from time 0 or from static dataset attrs.

### How WCS is written (ingest)

- **Header fix** (`fits_to_zarr_xradio.py`, `_fix_headers`): per input FITS,
  **preserves native `CRVAL1`/`CRVAL2`** (and `LATPOLE` when present). Also
  applies BSCALE/BZERO, Stokes-axis promotion, and spectral/frame keywords. **Do
  not** overwrite the phase center from filename timestamps.
- **Combine** (`_load_for_combine`, `_combine_time_step`): each time step gets
  its own header string from the fixed FITS WCS;
  `_collapse_wcs_header_str_variable` reduces auxiliary frequency dimensions so
  the persisted variable is **`wcs_header_str(time)`** (or `(time, frequency)`
  with one channel—see accessor handling). Regrid uses the reference LM pixel
  grid but keeps each source's native `CRVAL1`/`CRVAL2`
  (`_wcs_header_from_ref_grid_and_source_crval`).
- **Zarr append** (`_align_time_dimension_for_zarr_write`): scalar metadata is
  promoted to `(time,)` so incremental appends stay consistent.
- **Do not** assume all slices share one phase center.

#### Filename timestamps vs WCS (do not confuse)

| Use of basename `-image-YYYYMMDD_HHMMSS`                                           | Allowed? |
| ---------------------------------------------------------------------------------- | -------- |
| Discovery, grouping, time ordering (`_time_key_from_filename`, `_discover_groups`) | **Yes**  |
| Overwriting `CRVAL1`/`CRVAL2` with FK5 zenith at that instant                      | **No**   |

The FITS header `CRVAL1`/`CRVAL2` are **authoritative** for each integration's
phase center. Filename time tokens are for **sorting files into time bins
only**—not for recomputing celestial reference values.

Per-time `wcs_header_str` drift in a Zarr store should match the native FITS
headers written through `_fix_headers`, not a recomputed zenith from the
basename. `_zenith_fk5_crvals_deg()` remains in the codebase for **audit
diagnostics only** (`scripts/audit_zarr_wcs_timeline.py --sample-fits`); it is
**not** called during convert.

**If you change `_fix_headers`:** extend
`tests/test_fits_to_zarr.py::test_fix_headers_preserves_crval_from_input_not_filename`
and keep filename parsing tests in `tests/ingest/test_metadata_audit.py`
separate from WCS stamping.

#### Canonical Zarr metadata (do not regress)

When **`wcs_header_str`** is a data variable, it is the **only** persisted
celestial WCS for multi-time stores:

| Storage                                                                  | Allowed?                  | Notes                                                                                                                                                                               |
| ------------------------------------------------------------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wcs_header_str(time)` (or `(time, frequency)` collapsed to time)        | **Yes**                   | One FITS header string per time step; updates on every append.                                                                                                                      |
| `fits_wcs_header` on `ds.attrs`, `SKY.attrs`, other data vars, or coords | **No** on Zarr write/open | Zarr array attrs are **not** per-slice; a single `SKY.attrs["fits_wcs_header"]` freezes **time-0** CRVAL and breaks SkyWidget or any code that reads attrs before `wcs_header_str`. |

**Enforcement (already in the library — keep these calls when touching I/O):**

- **Before every Zarr write/append:**
  `fits_to_zarr_xradio._write_or_append_zarr` calls
  `strip_redundant_fits_wcs_header_attrs()` so incremental ingest (e.g.
  `scripts/ingest-I-Clean-Snapshot-20250120-LST4-5.sh` →
  `ingest_per_time_convert.py`) does not persist stale attrs.
- **On load:** `open_dataset()` applies the same strip so legacy stores remain
  safe in memory without re-ingest.

`_load_for_combine` may still set `fits_wcs_header` **in memory** for
combine/regrid; that is internal only. **Never** rely on those attrs after Zarr
export—always `wcs_header_str` + `_read_wcs_header_str(ds, time_idx=…)`.

**If you change ingest or I/O:** extend
`tests/test_fits_to_zarr.py::test_write_or_append_omits_fits_wcs_header_when_wcs_header_str_present`
and
`tests/test_io_integration.py::test_open_dataset_strips_stale_fits_wcs_header_with_wcs_header_str`.
Do **not** re-add `fits_wcs_header` to Zarr `.zattrs` for QA/review paths.

### How WCS is read (accessor)

Canonical helpers in `src/ovro_lwa_portal/accessor.py`:

| Function                                    | Role                                                                                                           |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `_has_per_time_wcs_header_str(ds)`          | True when `wcs_header_str` is indexed by `time` (1-D or `(time, frequency)`)                                   |
| `_read_wcs_header_str(ds, time_idx=…)`      | FITS header string for one time index                                                                          |
| `strip_redundant_fits_wcs_header_attrs(ds)` | Drop static `fits_wcs_header` attrs when `wcs_header_str` is canonical (used by `open_dataset` and Zarr write) |
| `RadportAccessor._get_wcs(time_idx=…)`      | Astropy WCS for pixel↔sky mapping                                                                              |
| `pixel_to_coords` / `coords_to_pixel`       | Must pass `time_idx` when per-time WCS exists                                                                  |

**Strict rules (do not break these):**

1. When per-time `wcs_header_str` exists, **always** select `time_idx` (and
   `frequency=0` if the variable is 2-D). Never use static `fits_wcs_header`
   attrs for a different time index.
2. If a per-time header entry is **empty**, return `None`—**do not** fall back
   to time-0 or static attrs (that mis-registers late slices and freezes
   `CRVAL1`).
3. Tests live in `tests/test_accessor.py`
   (`TestRadportGetWcsTimePromotedHeader`); extend them when changing WCS
   lookup.
4. Load multi-time QA/review data with **`ovro_lwa_portal.open_dataset`** (not
   raw `xr.open_zarr` alone) so stale on-disk `fits_wcs_header` attrs are
   stripped.
5. For review notebooks (`jupiter_flux_review.ipynb`, `source_review.ipynb`),
   prefer `open_dataset(path, chunks="auto").chunk({"l": 512, "m": 512})`. Avoid
   an extra `ds.chunk({"time": 1, …})` under an active distributed Dask `Client`
   on large incremental stores (unnecessary rechunk/shuffle); Jupiter-style
   review works without a `Client` for SkyWidget + tracked extractions.

### Fixed-sky source tracking (`dynamic_spectrum`, `patch_*`, `light_curve`)

LPT and transient review pass **catalog (RA, Dec)** to
`dataset.radport.dynamic_spectrum(ra=…, dec=…)`. That is **not** the same as
fixed **(l, m)** on the image grid: as per-time `CRVAL` moves each integration,
the same celestial source sits on **different pixels** each time.

**Required behavior:**

- `_compute_pixel_track` / `coords_to_pixel` must call
  **`_coords_to_pixel_via_wcs` with the slice `time_idx`** when
  `_has_per_time_wcs_header_str(ds)` is true.
- **Do not** use the analytical LST+SIN fallback for multi-time incremental Zarr
  (it ignores per-slice `CRVAL` and makes heatmaps look like the source drifts
  in RA).
- **Do not** use the batched `right_ascension`/`declination` grid path when only
  per-time `wcs_header_str` is available and RA/Dec coords lack a `time`
  dimension.
- If per-time WCS is missing for a step, raise—do not fall back to time-0 attrs.

Regression test:
`TestCelestialTimeSeriesTracking::test_radec_track_uses_per_time_wcs_not_fixed_pixel`.

**UI note:** Status text in `source_review.ipynb` shows **catalog** RA/Dec
(constant). `pixel_to_coords` must use the **same per-time WCS** as
`coords_to_pixel` (not time-0 header + analytical SIN); otherwise reported RA
drifts by ~0.5° while the dynamic spectrum is correct. `patch_fit` peak RA/Dec
maps pixels through `pixel_to_coords`—fix both together.

### How WCS is shown (SkyWidget / viz)

Reference implementation: `src/ovro_lwa_portal/viz/pipeline_qa_app.py`.

1. **Patch astrowidget once** at import: `_patch_astrowidget_get_wcs()` routes
   `astrowidget.wcs.get_wcs` through `_read_wcs_header_str` so the widget’s
   `crval`/`cdelt`/`crpix` traits match each slice.
2. **Load the cube** with `bind_sky_widget_dataset(widget, ds, max_size=…)`
   (defer first frame) or `set_dataset(..., defer_display=True)` after
   `_patch_astrowidget_get_wcs()`. `source_review.ipynb` uses
   `bind_sky_widget_dataset`; call `update_slice` immediately so the first frame
   is not zenith at t=0.
3. **Update slices through**
   `widget.update_slice(time_idx, freq_idx, center=…)`. Astrowidget refreshes
   slice WCS when `time_idx` changes (`_update_display_wcs` → patched
   `get_wcs`). When `center=` is set (catalog/LPT), astrowidget must
   **reproject** onto a shader grid at the view center even if the header passes
   `wcs_projection_matches_naive_shader`—otherwise per-time `CRVAL` drifts in
   the traits while `view_ra` stays fixed and the source appears to move in RA.
4. **View center vs phase center**:
   - **Zenith QA** (`pipeline_qa_app`): recenter with
     `sky_view_center(dataset, time_idx)` (CRVAL from per-time WCS) on time
     changes.
   - **Catalog / LPT review** (`notebooks/source_review.ipynb`): fixed catalog
     `center=` (like Jupiter’s fixed t₀ ephemeris), with `update_slice` on every
     heatmap click so image and WCS traits match that `time_idx`.
5. After slice updates in Jupyter, call `widget.send_state()` when available
   (see `_notify_sky_widget` in pipeline QA).
6. **Do not transpose** the slice sent to SkyWidget:
   `astrowidget.PreloadedCube.image()` must return **`(l, m)`** (numpy row = WCS
   axis 1, column = axis 2). Transposing to `(m, l)` mis-registers the overlay
   (fixed catalog targets drift in RA as `CRVAL` changes). The editable package
   is `../astrowidget` via Pixi feature `astrowidget-local`.
7. **WebGL texture UV vs numpy `(l, m)`** — the JS shader uploads
   `image_shape = (n_l, n_m)` as texture **height × width** (`n_l` rows, `n_m`
   columns). WCS `world2pix` yields `px` on axis 1 and `py` on axis 2; texture
   sampling must be **`uv = (py/n_m, px/n_l)`** (axis 2 → `u`, axis 1 → `v`).
   Swapping `px`/`py` in the UV assignment transposes the texture on square OVRO
   images and reads as a **flip + ~90° rotation** relative to HiPS. Regression:
   `../astrowidget/tests/test_wcs_shader_reproject.py::test_shader_texture_axes_match_numpy_lm_order`.
   After shader edits: `cd ../astrowidget && pixi run build`, then **restart the
   Jupyter kernel** (browser caches the bundled `widget.js`).

Data loading for QA apps follows `viz/pipeline_qa.py` → `load_qa_datasets()` →
`ovro_lwa_portal.open_dataset(zarr_path, chunks={…})`.

### Panel notebook comm management (Jupyter)

**Canonical reference:** When guidance differs, follow **`source_review.ipynb`** /
``viz/source_review_app.py`` + ``viz/panel_ui_session.py``. That app is extracted,
headless-tested (``test_source_review_ui_integration.py``, ``test_panel_ui_session.py``),
and validated in live Jupyter. **`jupiter_flux_review.ipynb`** is a legacy inline
notebook with minimal automated comm coverage — use it for science/workflow examples
(Zarr chunking, Jupiter ephemeris, flux methods), not as the comm architecture template.

Two notebook UIs share low-level helpers from ``viz/pipeline_qa_app`` but differ in structure:

| | ``source_review.ipynb`` **(canonical)** | ``jupiter_flux_review.ipynb`` (legacy) |
| --- | --- | --- |
| Test coverage | App module + UI integration tests | Manual notebook only |
| App code | ``viz/source_review_app.py`` + ``JupyterPanelUISession`` | Inline ``JupiterFluxReview`` class in the notebook |
| Activity log | ipywidgets ``HTML`` in ``pn.pane.IPyWidget`` | ``pn.pane.HTML`` + ``_push_panel_layout`` per ``_log()`` |
| Panel/Bokeh updates | ``_dispatch`` → ``JupyterPanelUISession`` (batch assign + push) | Direct assign + ``_push_panel_layout`` (explicit pane list) |
| Zarr open timing | ``schedule_when_panel_loaded`` + ``configure_source_review_notebook()`` | Loads on init when a default store is selected |
| ``hold_and_push`` | **Not** used in production | **Not** used |

**On conflict:** port **from** ``source_review`` **to** Jupiter (or new notebooks), not the
reverse. Shared primitive only: ``assign model`` + ``_push_panel_layout(*views)`` — both
avoid ``hold_and_push`` in the notebook UI.

**``source_review`` comm channels** (validated on calim10):

| UI piece | Technology | Comm | Update path |
| -------- | ---------- | ---- | ----------- |
| SkyWidget | ipywidgets | Widget comm | Trait updates; works from observers/workers |
| Activity log | ipywidgets ``HTML`` in ``pn.pane.IPyWidget`` | Widget comm | ``_refresh_log_widget()`` on every ``_log()`` |
| Status, spinner, controls | Panel/Bokeh | One notebook comm | ``_dispatch`` → ``JupyterPanelUISession.dispatch`` → assign + ``_push_panel_layout`` |
| Heatmap (Bokeh figure) | Panel/Bokeh | Same comm | ``publish_bokeh_pane_to_notebook`` **after** batch push |

**Bottom line (``source_review``):** Production UI goes through
:class:`~ovro_lwa_portal.viz.panel_ui_session.JupyterPanelUISession` (``SourceReview._ui``).
Schedule worker/io-loop callbacks via ``_dispatch`` / ``_schedule_ipython_main``, mutate
Python models, then ``_push_panel_layout`` over ``SourceReview._notebook_ui_views()``
(layout + status + spinner + heatmap + coord field). **Do not** use ``hold_and_push`` in
production ``SourceReview``. **Do not** copy ``jupiter_flux_review``'s Panel HTML log or
inline-class structure when they differ — follow ``source_review_app.py`` instead.

**Activity log:** ``source_review`` uses ipywidgets (validated). ``jupiter_flux_review``
still uses Panel HTML log; treat that as legacy unless you re-validate with
``source_review``-level tests.

#### Three sync tiers (do not collapse)

Implemented in ``viz/panel_ui_session.py``; app code calls ``self._ui.*`` only:

1. **Dispatch batch** — spinner, status markdown inside
   :meth:`~ovro_lwa_portal.viz.panel_ui_session.JupyterPanelUISession.dispatch`:
   assign on Python models, one ``_push_panel_layout`` sweep over
   ``SourceReview._notebook_ui_views()`` (layout + status + spinner + heatmap +
   coord field).
2. **Direct Bokeh publish** — heatmap figures: assign ``pane.object``, then
   ``_push_panel_layout`` via ``publish_bokeh_pane_to_notebook`` — **never** inside
   ``doc.hold('combine')``, **never** ``discard_events``.
3. **Deferred widget publish** — coordinate field when a dispatch batch is active:
   queue ``publish_panel_widget_to_notebook`` with ``defer_after_notebook_hold`` and
   flush it **after** the batch push in the same io-loop turn.

Worker threads must never mutate Panel panes directly — always ``self._dispatch(...)``
(which schedules on the io_loop) or ``self._ui.schedule(...)``.

#### What did **not** work (do not reintroduce)

These were tried repeatedly; Python state updated but the browser UI stayed frozen
(activity log often still worked via ipywidgets):

- **`hold_and_push` / `dispatch_notebook_ui` in production `SourceReview`** — explicit
  ``doc.hold('combine')`` + nested sync froze spinner, status, heatmap, and coordinate
  field in live Jupyter even when pytest passed (see **Test coverage vs Jupyter**
  below). **Validated fix:** ``JupyterPanelUISession`` (assign + push on ``io_loop``).
- **`discard_events` + `_push_panel_layout` for Bokeh heatmap** — suppresses Panel's
  watcher; browser stays on the placeholder figure.
- **Assigning `pn.pane.Bokeh.object` inside any hold cycle** — mid-hold watcher push is
  lost; zeros grid persists after Generate heatmap.
- **`defer_dispatch(_ensure_heatmap_grid)` after Zarr open** — a second io-loop dispatch
  can run **after** Generate finishes; the zeros grid republish overwrites the computed
  spectrum (activity log shows finite range; browser stays zeros). Build the grid inside
  ``finalize_dataset_load`` instead (see **Deferred UI ordering** below).
- **Activity log as `pn.pane.HTML`** with Panel push in **`source_review`** — nested-ref /
  comm issues; log pane stale while ipywidgets log works. (``jupiter_flux_review`` still
  uses Panel HTML log + ``_push_panel_layout`` successfully — layout-specific, not a
  universal ban on Panel logging.)
- **Starting Zarr open synchronously** on first ``review.panel`` access — races the
  notebook comm; use ``schedule_when_panel_loaded(_open_dataset)``.
- **`_run_on_main_thread` / inline `pn.state.execute` from worker threads** — no live
  Bokeh session on the worker; Panel updates never reach the browser.
- **`notebook_views_registered(review._layout) is True` as the only gate** — layout root
  can be registered while nested pane refs are absent from ``state._views``; pass
  mutated panes explicitly in ``_notebook_ui_views()``.
- **Assuming green pytest implies a working notebook** — ``InlinePanelUISession`` uses
  ``hold_and_push`` synchronously with a fully registered view tree; that is **not** the
  production Jupyter comm path.

#### What **did** work (current `source_review_app.py` + `panel_ui_session.py`)

1. **ipywidgets activity log** — ``_log_widget = widgets.HTML(...)`` in
   ``pn.pane.IPyWidget``; ``_refresh_log_widget()`` on every ``_log()``. Independent
   of Panel push (symptom: log updates while spinner/heatmap stay frozen → Panel path
   bug, not logging).
2. **`JupyterPanelUISession`** — production backend for ``source_review`` (default in
   ``SourceReview``): ``dispatch`` on ``io_loop`` → mutate models → ``_push_panel_layout``
   → flush deferred Bokeh/widget publishes in the same turn. This is the validated
   comm architecture; do not replace it with ``jupiter_flux_review``'s inline direct-push
   class when the two differ.
3. **`configure_source_review_notebook()`** — call once before launch; captures kernel
   ``io_loop`` via ``_capture_ipython_io_loop()``.
4. **`schedule_when_panel_loaded(_open_dataset)`** — defer Zarr worker until
   ``pn.state.loaded`` so open callbacks run against a live comm.
5. **`publish_bokeh_pane_to_notebook`** — heatmap via ``_publish_heatmap_figure`` →
   ``self._ui.publish_bokeh_figure`` (deferred after batch push, not inside hold).
6. **`finalize_dataset_load` builds the zeros grid synchronously** — call
   ``_ensure_heatmap_grid()`` from the ``build_heatmap_grid`` step in ``_finish_open``,
   not ``defer_dispatch`` afterward. ``_apply_heatmap`` sets ``_heatmap_grid_ready`` so
   late grid passes cannot clobber a computed spectrum.
7. **`publish_panel_widget_to_notebook`** — coordinate field after sky click (deferred
   when ``notebook_ui_hold_active()`` during a dispatch batch).
8. **Module extraction** — ``source_review_app.py`` / ``source_review_data.py`` /
   ``panel_ui_session.py`` so threading and comm intent are headless-testable.

#### Low-level helpers (`viz/pipeline_qa_app.py`)

Used by ``JupyterPanelUISession``, ``InlinePanelUISession``, and legacy QA code:

| Function | Role |
| -------- | ---- |
| `_schedule_ipython_main(callback)` | Schedule on kernel ``io_loop`` (never inline from workers) |
| `_push_panel_layout(*views)` | Push layout comm after model assign (shared primitive) |
| `publish_bokeh_pane_to_notebook(pane, value, *root_views)` | Bokeh figure: assign + push, no hold (`source_review` heatmap path) |
| `publish_panel_widget_to_notebook(widget, *root_views, **params)` | Widget assign + push |
| `defer_after_notebook_hold(callback)` | Queue publish after active dispatch batch |
| `notebook_ui_hold_active()` | True inside ``JupyterPanelUISession.dispatch`` batch |
| `hold_and_push(*views)` | **Headless / ``InlinePanelUISession`` only** — not production Jupyter |
| `dispatch_notebook_ui(callback, *views)` | Legacy hold/push entry; prefer ``JupyterPanelUISession`` |
| `set_notebook_widget_params` / `sync_pane_to_notebook` | Used inside ``InlinePanelUISession`` + unit tests |
| `schedule_when_panel_loaded(callback)` | Run after Panel notebook comm is ready |
| `notebook_views_registered(*views)` | Comm retry gate; not sufficient alone |

Tests: ``tests/viz/test_pipeline_qa.py`` (low-level hold/sync/publish helpers),
``tests/viz/test_panel_ui_session.py``, ``tests/viz/test_source_review_ui_integration.py``,
``tests/viz/test_source_review_app.py``.

#### Headless testing architecture

App code must not branch on ``notebook_ui_hold_active()`` in production — inject a
:class:`~ovro_lwa_portal.viz.panel_ui_session.PanelUISession`:

| Backend | Comm strategy | Use |
| ------- | ------------- | --- |
| ``JupyterPanelUISession`` | Assign + ``_push_panel_layout`` on ``io_loop`` + deferred publish | Production ``source_review`` (default) |
| ``InlinePanelUISession`` | Synchronous ``hold_and_push`` on mounted document | Headless harness tests |
| ``RecordingPanelUISession`` | Operation log; optional ``InlinePanelUISession`` delegate | Behavioral / ordering tests |
| ``CallbackPanelUISession`` | Inline ``dispatch_override`` | Python state only (no comm) |

**Harness:** ``PanelUITestHarness.mount(layout)`` registers a real
``bokeh.document.Document`` and full nested view tree in ``state._views``.
``mount_layout_only`` keeps only the layout root registered (notebook-like gap).
``capture_notebook_pushes(monkeypatch)`` records ``push`` calls. Assert on **Bokeh
models** via ``harness.bokeh_model(viewable, layout)``, not only Python Param values.

**What pytest covers:**

| Test module | What it proves |
| ----------- | -------------- |
| ``test_panel_ui_session.py`` | ``PanelUISession`` API + harness publish/spinner/coord |
| ``test_source_review_ui_integration.py`` | End-to-end heatmap/spinner/coord via inline **and** ``QueuedIOLoop`` + ``JupyterPanelUISession``; includes inverted-order grid race regression |
| ``test_pipeline_qa.py`` | ``hold_and_push``, ``sync_*``, ``publish_*`` helpers on real documents |
| ``test_source_review.py`` | Pure logic (Center, load threading) — no browser comm |

**Test coverage vs Jupyter (critical):**

- ``InlinePanelUISession`` runs **synchronously** with ``hold_and_push`` and a **fully
  registered** view tree. Passing these tests does **not** prove live Jupyter works.
- ``test_jupyter_session_open_generate_and_sky_click`` uses ``QueuedIOLoop`` with
  monkeypatched ``_schedule_ipython_main`` and **synchronous flush** — closer to the
  notebook, but still not async browser comm timing.
- After any change to ``JupyterPanelUISession``, ``_push_panel_layout``, or io-loop
  scheduling: run pytest **and** manual ``source_review.ipynb`` smoke (kernel restart,
  confirm spinner, heatmap grid after open, coordinate field on sky click).
- SkyWidget, HiPS, and WebGL overlay still require manual notebook validation.

#### Deferred UI ordering (heatmap grid race)

Once Panel comm delivery works, the next class of bugs is **ordering and overwrite** —
not “nothing reaches the browser.”

**Symptom:** Activity log shows ``Finished … in N s`` and a finite value range, but the
Bokeh heatmap still looks like the initial zeros grid (correct ``time × frequency``
shape, no colormap).

**Cause:** ``_ensure_heatmap_grid()`` republishes the same ``_heatmap_pane`` with a zeros
array. If that ran **after** ``_apply_heatmap`` (e.g. ``defer_dispatch(_ensure_heatmap_grid)``
queued at end of ``_finish_open`` while Generate finished on an earlier/later io-loop
turn), the browser received the computed figure and then the zeros figure. Python
``_heatmap_values`` could already hold real data.

**Enforcement (do not regress):**

| Rule | Allowed? |
| ---- | -------- |
| Build zeros grid in ``finalize_dataset_load`` ``build_heatmap_grid`` step (same open dispatch batch as mount/clear-loading) | **Yes** |
| ``defer_dispatch(_ensure_heatmap_grid)`` after open | **No** |
| ``_apply_heatmap`` sets ``_heatmap_grid_ready = True`` before publish | **Yes** |
| ``_ensure_heatmap_grid`` skips when a non-zero computed spectrum is already loaded (unless ``force=True``) | **Yes** |

**Defer only** Bokeh **publish** (``publish_bokeh_figure`` → after batch push), not whole
workflow steps that republish the same pane.

**Testing:** Sequential open → grid → generate tests are insufficient. Keep
``test_jupyter_session_generate_before_deferred_grid_does_not_reset`` in
``test_source_review_ui_integration.py`` — it runs Generate **before** a deferred grid
pass and asserts values are not reset.

**Triage:** Log shows finite range but heatmap is zeros → check overwrite race before
debugging ``publish_bokeh_pane_to_notebook``. Log frozen → Panel comm path. Log works,
Panel frozen → comm path, not logging.

### `source_review` Panel app and notebook

The **SourceReview** UI lives in `src/ovro_lwa_portal/viz/source_review_app.py` (extracted
from the notebook for testability). `notebooks/source_review.ipynb` keeps path/config
constants, calls `configure_source_review_notebook()`, and constructs
`SourceReview` with a `SourceReviewConfig`.

Notebook launch pattern (``notebooks/source_review.ipynb``):

```python
configure_source_review_notebook()  # astrowidget patch, pn.extension, _capture_ipython_io_loop
review = SourceReview(ZARR_PATH, ..., config=SourceReviewConfig(...), validate_zarr=False)
review.panel  # schedule_when_panel_loaded → Zarr open
```

``jupiter_flux_review.ipynb`` defines ``JupiterFluxReview`` in-notebook with a simpler
comm stack (direct ``_push_panel_layout``, Panel HTML log). **Do not treat that notebook
as the template for new Panel comm work** — align Jupiter with ``source_review`` when
updating comm behavior, and add tests under ``tests/viz/`` before relying on changes.

Pure decision logic remains in `viz/source_review.py` (`plan_center_action`,
`run_dataset_load`, `finalize_dataset_load`). Data helpers (heatmap computation,
known-sources YAML) are in `viz/source_review_data.py`.

Headless tests: `tests/viz/test_source_review.py`, `test_source_review_app.py`,
`test_source_review_data.py`, `test_source_review_ui_integration.py`,
`test_panel_ui_session.py`, and `test_pipeline_qa.py` (low-level comm helpers).
Run `pixi run python -m pytest tests/viz/test_source_review_ui_integration.py`
after Panel comm or `JupyterPanelUISession` changes; still smoke-test the notebook.

### `source_review.ipynb` coordinate UI and SkyWidget actions

Reference: `SourceReview` in `viz/source_review_app.py` (launched from
`notebooks/source_review.ipynb`). The **Center**
button's decision logic is a pure function — `plan_center_action` in
`src/ovro_lwa_portal/viz/source_review.py`, tested in
`tests/viz/test_source_review.py`. Keep decisions there, not inline in the
notebook: the UI cannot be exercised headlessly, and several "Center recenters
on the wrong position" bugs were only pinned down once the decision became a
testable function. When changing Center semantics, change `plan_center_action` +
its tests first, then the notebook call site.

**Coordinate field (`pn.widgets.AutocompleteInput`):**

- Typing / tab completion / dropdown pick **log** RA/Dec only
  (`_log_coordinate_resolution`); they do **not** load Zarr or build a heatmap.
- **Center** — resolves the field, computes a `CenterPlan`, then
  `widget.goto(field_coord, fov=SKY_FOV_DEG)`. **Never clear an existing overlay
  on Center** — reproject it onto the field coordinate instead
  (`update_slice(center=field_coord)`). Users click a source in the overlay, hit
  Center, and expect to see the _same source_ centered; clearing reads as data
  loss. If the field no longer matches the generated heatmap target, reset the
  heatmap to a **zeros grid** (`_reset_heatmap_to_zeros`) — do not hide the
  heatmap pane (`object = None`), it must stay clickable.
- **Generate heatmap** — resolves the field, then `_load_heatmap()`; sets
  `_heatmap_coord` (the target the _computed_ spectrum belongs to, distinct from
  `_coord`, the current overlay center).
- **Sky click** — observe `SkyWidget.click_tick`; read `clicked_coord`, format
  with `format_icrs_degree_pair` (`ovro_lwa_portal.name_resolution`), fill the
  field via `_set_coordinate_field_from_text`. Schedule with
  `_dispatch(...)` → `JupyterPanelUISession.dispatch` (io_loop + batch push),
  not `_schedule_ipython_main` alone from a worker.

**Always-on heatmap grid and overlay toggle:**

- A clickable **zeros heatmap** spanning the full Zarr `time × frequency` shape
  is shown as soon as the store opens (`_ensure_heatmap_grid` via
  ``finalize_dataset_load`` in ``_finish_open`` — **not** ``defer_dispatch``);
  Generate replaces the zeros in place. No notebook action may leave
  `_heatmap_pane.object = None` — rebuild a zeros grid instead.
- **Heatmap cell click** loads that Zarr slice as the overlay and **turns the
  overlay on** (`_set_overlay_toggle_display(True)`), even if the user toggled
  it off. It must **preserve pan/zoom**: call `update_slice(view_lock=True)`
  with **no `center` and no `fov`** (`_update_sky(..., preserve_view=True)`).
  Passing `fov=` on every slice change resets the user's zoom and reads as the
  view "jumping".
- **Overlay toggle off** calls `widget.clear_image()`; the JS side must clear
  the GPU texture (`clearImageTexture()` uploads a 1×1 transparent texture),
  otherwise the stale overlay keeps rendering even though Python state says it
  is gone.
- When syncing a `pn.widgets.Toggle` programmatically, set `.value` under a
  suppress flag (`_suppress_overlay_toggle`) checked at the top of the watcher —
  same pattern as the coordinate field below; `discard_events` would break
  browser sync.
- Long operations (overlay slice loads, heatmap computation) must log start
  _and_ finish to the activity log (`_update_sky(..., log_loading=True)`); a
  silent multi-second load reads as a dead UI.

**Panel widget updates from sky clicks and io-loop callbacks:**

- Do **not** wrap `AutocompleteInput` value updates in
  `param.parameterized.discard_events` when the browser must show the new text —
  that suppresses Param events and the field stays blank while the activity log
  still updates. Use a `_suppress_coord_value_handler` flag to skip duplicate
  resolution logging on programmatic writes instead.
- Sky-click coordinate writes: status and spinner mutate inside
  `JupyterPanelUISession.dispatch`; the coordinate field uses
  **`publish_panel_widget_to_notebook`** via `defer_after_notebook_hold` (flush
  after the batch push, same io-loop turn).
- **Bokeh heatmap figures** from io-loop callbacks (e.g. `_ensure_heatmap_grid`,
  Generate) use **`publish_bokeh_pane_to_notebook`** via `_publish_heatmap_figure`
  → `self._ui.publish_bokeh_figure` — deferred after batch push, **not** inside
  `hold_and_push`. Use `publish_bokeh_pane_to_notebook` (assign `pane.object` then
  `_push_panel_layout`) as in `source_review_app.py`.

**Testing the controller:** compare numpy-derived booleans with
`bool(...) is True`, not `np.bool_ is True` (`assert np.True_ is True` fails).
Run `pixi run python -m pytest tests/viz/test_source_review.py` after any
Center-semantics change.

### astrowidget HiPS + WebGL overlay sync (editable `../astrowidget`)

The default Pixi env uses **`astrowidget-local`** (`../astrowidget`, editable).
After JS changes: `cd ../astrowidget && pixi run build`, then **restart the
Jupyter kernel**.

Layout: Aladin HiPS (`z-index: 0`) under a transparent WebGL canvas
(`z-index: 1`). Pan/zoom on the canvas must keep both layers on the same
celestial view.

**Python-driven view** (`goto`, `update_slice(..., center=, fov=)`):

1. Traits `view_ra` / `view_dec` / `view_fov` update in Python.
2. JS `onPythonViewChange` (only when `userInteracting` is false):
   `applyViewFromModel()` → `syncAladin()` → `scheduleDraw()` (deferred
   `requestAnimationFrame`, not synchronous `draw()`). `change:image_revision`
   uses the same deferred draw and calls `applyViewFromModel()` directly (not
   `syncView`, which skips during `userInteracting`) so
   `update_slice(..., center=)` cannot paint the new view center before
   `image_data`/`crval` sync — that mismatch projected zenith data onto the
   wrong sky (e.g. southern hemisphere).
3. **Do not** call `syncViewFromAladin()` inside `updateViewPlaneScales()`
   before `syncAladin()` runs. That reads the **stale** HiPS center and reverts
   the overlay before Python’s target is applied (breaks **Slew** and makes the
   field appear not to move).

**User drag / wheel on the canvas:**

- While `userInteracting` is true, ignore trait `change:view_*` echo (would
  redraw with stale Aladin scales mid-gesture).
- On pan/zoom end, `finishViewGesture()`: `syncViewFromAladin()` →
  `syncAladin()` → `draw()`.
- Pan uses Aladin WASM `goFromTo` when available; refresh
  `measureViewPlaneScales` each drag frame with current `viewRotation`
  (`-aladin.getRotation()`).

**Projection / overlay appearance:**

- **Two coordinate layers** — do not conflate them:
  - **View / HiPS** (Aladin): screen ↔ sky via measured `pix2world` scales,
    `view_ra`/`view_dec`/`view_fov`, and `viewRotation` (`-aladin.getRotation()`).
  - **Radio texture** (WebGL): sky ↔ numpy `(l, m)` via shader `crval`/`cdelt`/
    `crpix` after optional `reproject_for_shader_display`. Small FOV only shrinks
    the visible patch; it does **not** bound reprojection error or HiPS↔WCS
    disagreement (arcsecond offsets span more pixels when zoomed in).
- `measureViewPlaneScales` must pass **rotation-aware** scales (inverse-rotate
  measured `l,m` into the view plane) so overlay and HiPS stay registered when
  north is not up.
- With a HiPS background, use **measured** scales from `pix2world` directly; cap
  zoom-out with `maxSinViewFov(aspect)` so SIN view-disk corners stay inside
  `r ≤ 1`.
- Crosshair after sky click: fixed **screen-space** size via
  `celestialToScreen`, not angular FOV scaling. Sky clicks use **Aladin**
  `pix2world` when HiPS is active (`clicked_coord_debug` logs HiPS vs WebGL for
  the same pixel).
- `update_slice(..., center=catalog)` reprojects so shader `crval` matches the
  catalog view; **panning** moves `view_ra/dec` away from that `crval`. A curved
  clip at the view edge is expected after large pans; do not re-add the shader
  “horizon circle” overlay (it drew a second great circle and looked like a
  wedge intersection).
- **View-lock vs catalog-center reproject** — `update_slice(view_lock=True)`
  warps the slice to the **current pan/zoom** center (can look aligned with HiPS
  while browsing). **Center** / explicit `center=` re-samples onto the
  **field/catalog** tangent plane; a source that sat under the crosshair in
  view-lock mode can shift when Center exposes the true FITS WCS position vs the
  HiPS-reported click coordinate.
- **SIN two-to-one ambiguity:** `reproject_for_shader_display` must mask output
  pixels whose world coordinate is ≥ 90° from the source tangent point
  (`cos_sep ≤ 0 → NaN`). Without the mask, `all_world2pix` maps far-hemisphere
  world points back onto valid source pixels and real data appears mirrored near
  the opposite celestial pole when zooming out from an empty field. Regression
  tests: `test_reproject_rejects_far_hemisphere_mirror_ghost`,
  `test_reproject_masks_southern_ghost_for_northern_snapshot` in
  `../astrowidget/tests/test_wcs_shader_reproject.py`.

**View-locked overlay (HiPS pan/zoom):**

- `overlay_view_lock=True` + `view_gesture_revision` (incremented by JS
  `finishViewGesture`) drive a **debounced** Python-side reproject of the
  current slice onto the new view center.
- `update_slice(view_lock=True)` with no `center` reprojects to the current
  `view_ra`/`view_dec` — this is the path that preserves user pan/zoom across
  time/frequency changes. An explicit `center=` always overrides view lock.
- Python `clear_image()` empties `image_data`/`image_shape` and bumps
  `image_revision`; JS `syncImage()` must detect the empty payload and call
  `clearImageTexture()` (1×1 transparent texture). A Python-side clear without
  the GPU clear leaves a stale overlay on screen.
- `clicked_coord_debug` carries both the Aladin `pix2world` and WebGL
  `screenToRaDec` results for the same click — use it to localize
  click-coordinate bugs to one projection layer.

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

**Symptoms:** Catalog source should be fixed on the sky, but the time–frequency
map changes as if the wrong pixel were sampled; `patch_fit` peak RA may wander.

**Cause:** `coords_to_pixel` / `_compute_pixel_track` used the **LST+SIN
analytical path** or a **static** `fits_wcs_header` while the Zarr has
**per-time** `wcs_header_str` with drifting `CRVAL1`.

**Solution:** Ensure `_has_per_time_wcs_header_str` is true for the store and
use current `accessor.py` (WCS `world2pix` per `time_idx`). Re-run heatmap after
upgrading. See **Fixed-sky source tracking** under Per-Time WCS and CRVAL.

### Issue: Catalog source drifts in RA in SkyWidget when changing time

**Symptoms:** Heatmap and `tracked@slice` are stable, but the sky image or
coordinate grid makes a fixed catalog target slide in RA as `time_idx` advances.

**Cause:** `update_slice(..., center=catalog)` fixed `view_ra`/`view_dec` but
skipped `reproject_for_shader_display` because the OVRO header matched the naive
SIN shader; widget `crval` traits still followed per-time `CRVAL1` from
`wcs_header_str`.

**Solution:** Use editable `../astrowidget` with `_push_image_frame`
reprojecting when `center` is not `None`. Restart the Jupyter kernel after
upgrading astrowidget.

### Issue: SkyWidget RA/Dec grid does not update when changing time

**Symptoms:** Heatmap time changes but coordinate grid (or image registration)
looks stuck at the first slice; `CRVAL1` in the store varies per time but the UI
does not.

**Common causes:**

- `set_dataset()` called **without** `defer_display=True` before the first
  `update_slice()`.
- `astrowidget.wcs.get_wcs` used **without** `_patch_astrowidget_get_wcs()`
  (falls back to static `fits_wcs_header` from time 0).
- **Stale Zarr attrs:** `SKY.attrs["fits_wcs_header"]` or
  `ds.attrs["fits_wcs_header"]` left from incremental ingest (time-0 only) and
  read by unpatched code. Fix: use `open_dataset()` /
  `strip_redundant_fits_wcs_header_attrs`, ensure `_write_or_append_zarr` still
  strips before write; restart kernel after astrowidget edits.
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
preserved) or repair WCS rows from FITS via `ovro-ingest repair --fits-dir`. Do
**not** reintroduce filename-based zenith stamping in `_fix_headers`.

### Issue: Sky click logs coordinates but Coordinate field stays empty

**Symptoms:** Activity log shows `Sky click → 'RA, Dec'`; AutocompleteInput does
not update.

**Cause:** Coordinate field publish ran inside a failed Panel comm path (e.g.
legacy `hold_and_push`, or `discard_events` suppressing browser sync). Activity
log still updates because it uses ipywidgets.

**Solution:** Use `self._ui.sync_coordinate_field` → `publish_panel_widget_to_notebook`
(deferred after dispatch batch push). Schedule sky-click handling through
`_dispatch`, not `_schedule_ipython_main` alone from a worker.

### Issue: Activity log frozen after `HiPS background`; no heatmap after Zarr open

**Symptoms:** Activity log stops at the last `__init__` line (often `HiPS background`);
`review.log_text` in Python is correct but the browser log pane is stale. No
`Opening…`, Zarr progress, `Opened — …`, or zeros heatmap. SkyWidget overlay and
Center may still work.

**Cause:** Dual (triple) comm architecture. Pre-display `__init__` logs are baked
into the first Panel render snapshot unless ipywidgets log is used. Post-display
Panel updates need assign + `_push_panel_layout` on the kernel `io_loop` via
`JupyterPanelUISession`. Legacy `hold_and_push` froze Panel widgets in live Jupyter
while pytest still passed.

**Solution (validated):**

1. Activity log → ipywidgets `HTML` + `_refresh_log_widget()` (see
   `source_review_app.py`).
2. Panel panes → `_dispatch` → `JupyterPanelUISession` (assign + push, not
   production `hold_and_push`).
3. Defer open → `schedule_when_panel_loaded(_open_dataset)`; call
   `configure_source_review_notebook()` before launch.
4. After kernel restart, re-run all cells; confirm log continues past HiPS and
   heatmap grid appears after open.

Do **not** revert to `pn.pane.HTML` for the activity log. Do **not** assume green
pytest replaces a notebook smoke test after comm changes.

### Issue: Generate heatmap runs but the plot stays zeros

**Symptoms:** Activity log shows computation progress, finite-cell stats, and
`Finished … in N s`, but the time–frequency pane still shows the initial zeros
grid (correct shape, no colormap).

**Two distinct causes:**

1. **Panel comm path broken** — Bokeh figure update went through `hold_and_push`,
   `discard_events`, or `set_notebook_pane_object` instead of the validated
   `source_review` publish path. Python `pane.object` may update but the browser
   does not.
2. **Deferred grid overwrote computed heatmap** — `_apply_heatmap` published real
   data, then a later `_ensure_heatmap_grid` (often from `defer_dispatch` after
   open) republished zeros to the same pane. Activity log and `_heatmap_values`
   can show finite stats while the browser shows zeros. See **Deferred UI
   ordering (heatmap grid race)** above.

**Solution:**

- Route all heatmap figure updates through `_publish_heatmap_figure` →
  `self._ui.publish_bokeh_figure` → `publish_bokeh_pane_to_notebook` (assign
  `pane.object`, then `_push_panel_layout`, deferred after dispatch batch push).
- Build the open-time zeros grid inside `finalize_dataset_load`, not
  `defer_dispatch(_ensure_heatmap_grid)`.
- Ensure `_apply_heatmap` sets `_heatmap_grid_ready` and `_ensure_heatmap_grid`
  does not replace a loaded computed spectrum unless `force=True`.
- Spinner/status sync in the same dispatch batch via direct assign + batch push.

### Issue: pytest passes but `source_review.ipynb` Panel UI is frozen

**Symptoms:** Activity log updates (ipywidgets) on sky click and after Zarr open,
but spinner, status markdown, heatmap, and coordinate field stay on their initial
render. `test_source_review_ui_integration.py` is green.

**Cause:** Headless tests use ``InlinePanelUISession`` (synchronous ``hold_and_push``
+ fully registered view tree) or ``QueuedIOLoop`` with synchronous flush — neither
matches async live Jupyter comm timing. Production previously used ``hold_and_push``,
which froze the browser while tests still passed.

**Solution:** Production must use ``JupyterPanelUISession`` (assign +
``_push_panel_layout`` on the kernel ``io_loop``). After comm changes, restart the
kernel and smoke-test ``source_review.ipynb`` even when pytest passes. Symptom triage: log works,
Panel frozen → Panel comm path, not logging.

### Issue: Heatmap clicks do nothing / toggle shows no feedback

**Symptoms:** The zeros heatmap renders but clicking cells produces no log entry
or overlay; the Overlay toggle flips visually but nothing else changes.

**Cause:** The Bokeh figure was never published to the browser (placeholder still
shown), or tap handler was wired before the live figure reached the comm. Often
the same root cause as “Generate heatmap runs but plot stays zeros” — Panel path
broken while activity log still works.

**Solution:** Ensure `_ensure_heatmap_grid` / Generate use `_publish_heatmap_figure`
(`source_review` path). Log/status/spinner from io-loop paths use `_dispatch` →
`JupyterPanelUISession`. After comm fixes, restart the kernel and confirm the
heatmap title changes from “Heatmap loads…” before testing clicks.

### Issue: Radio overlay flipped or rotated relative to HiPS

**Symptoms:** DSS/WISE background looks correct, but the Zarr radio overlay
appears mirrored, rotated ~90°, or sheared — worst when zoomed on an off-center
source. May have looked “almost right” at the phase center on square 512×512
fields.

**Cause:** WebGL texture UV mapped WCS axis 1/2 onto the wrong texture
dimensions (correct: `uv = (py/n_m, px/n_l)`; the bug used `uv = (px/n_m,
py/n_l)`, which transposes numpy `(l, m)` on upload). This is independent of
`reproject_for_shader_display` and per-time `CRVAL`; it is a **shader sampling**
bug in `../astrowidget/js/inline_widget.js` (and `renderer.js`).

**Solution:** Use current `../astrowidget` with the corrected UV line; run
`cd ../astrowidget && pixi run build`; restart the Jupyter kernel. Extend
`test_shader_texture_axes_match_numpy_lm_order` if you touch the fragment shader.

### Issue: Source moves away from crosshair after Center (small FOV)

**Symptoms:** User clicks a radio feature, crosshair marks the spot, **Center**
recenters — the bright source no longer sits under the crosshair. Worse at small
FOV, so it feels like a “zoom math” bug.

**Cause:** Not FOV-scaled. **Center** runs `goto` + `update_slice(center=field)`
(full spherical reproject onto the click/catalog tangent plane). The crosshair
stays at the **HiPS click** coordinate; the overlay peak follows **FITS/Zarr
WCS** after reproject. View-lock display before Center can hide a HiPS↔WCS
offset. Check activity-log diagnostics:
`[diag] Click … hips↔webgl Δ=…″` and
`[diag] Center[…]: intended … | view=… Δ=…″ | crval=… Δ=…″`.

**Solution:** Compare click coords to
`dataset.radport.pixel_to_coords(l, m, time_idx=…)` at the feature pixel. Large
`hips↔webgl Δ` → projection-layer mismatch; small Δ but post-Center offset →
view-lock vs catalog reproject or stale `astrowidget`. Heatmap taps should keep
`preserve_view=True` (no `center=`/`fov=`); reserve `center=` for explicit
Center/Generate.

### Issue: Center clears the overlay or makes the heatmap disappear

**Symptoms:** Click a source in the overlay, press **Center** — the overlay
vanishes; or pressing Center wipes the (zeros) heatmap so there is nothing left
to click.

**Cause:** Old Center logic cleared the overlay when the field coordinate was
far from the heatmap target, and set `_heatmap_pane.object = None`. Both read as
data loss; the zeros grid is the entry point for loading slices, so hiding it
dead-ends the workflow.

**Solution:** Center **always keeps and reprojects** the overlay onto the field
coordinate (`plan_center_action` returns `overlay_center=field_coord` whenever
`has_overlay`), and resets the heatmap to a zeros grid
(`_reset_heatmap_to_zeros`) instead of hiding it. Note the Python
`clear_image()` → JS `clearImageTexture()` pairing: without the GPU texture
clear, a "cleared" overlay keeps rendering at its old position, which presented
as "overlay shows the wrong position" after Center.

### Issue: Zoom/FOV resets when changing time/frequency via heatmap click

**Symptoms:** User zooms in, clicks another heatmap cell, and the view jumps
back to the default FOV or recenters.

**Cause:** `_update_sky` passed `center=` and/or `fov=` on every slice change.

**Solution:** Heatmap taps use `_update_sky(..., preserve_view=True)` →
`update_slice(view_lock=True)` with no `center`/`fov`: only the slice changes,
the view stays put. Reserve `center=`/`fov=` for explicit Center/Generate
actions.

### Issue: Overlay shows mirrored data near the opposite celestial pole

**Symptoms:** Click an empty field (e.g. far southern sky for a northern
snapshot), zoom out — radio data appears near the south celestial pole where no
data exists.

**Cause:** SIN projection is two-to-one; without masking, `all_world2pix` in
`reproject_for_shader_display` maps far-hemisphere world points onto valid
source pixels.

**Solution:** Use current `../astrowidget`: pixels ≥ 90° from the source tangent
point are set to NaN (`cos_sep ≤ 0` mask). Covered by
`test_wcs_shader_reproject.py` regressions.

### Issue: Slew button does not move the sky view

**Symptoms:** Log says “Slewed HiPS background…”, but the widget view does not
change.

**Cause:** `updateViewPlaneScales()` called `syncViewFromAladin()` before
`syncAladin()` applied Python `goto()`, reverting `view_ra/dec` to the old HiPS
center.

**Solution:** Use current `../astrowidget` (`onPythonViewChange`: apply model →
`syncAladin` → `draw`; no `syncViewFromAladin` inside `updateViewPlaneScales`).
Rebuild astrowidget JS and restart the kernel.

### Issue: Zarr overlay misaligned or “two great circles” after panning

**Symptoms:** After dragging the background, the field recenters unexpectedly;
radio overlay covers only a wedge; edges look like two great circles
intersecting.

**Causes:**

- Trait `change:view_*` fired `draw()` mid-drag with stale Aladin scales while
  `userInteracting` was true.
- View center (`view_ra/dec`) moved away from shader `crval` after pan (catalog
  slice keeps reprojected `crval` fixed on the sky).
- (Historical) shader drew the image SIN horizon as a visible arc crossing the
  view disk.

**Solution:** Use current `../astrowidget` (`finishViewGesture` on pan end; skip
trait echo while `userInteracting`; rotation-aware `measureViewPlaneScales`).
Rebuild and restart kernel. After large pans off the catalog center, some curved
clipping at the view edge is expected.

### Issue: SkyWidget correct but `SKY.attrs["fits_wcs_header"]` still on disk

**Symptoms:** `audit_zarr_wcs_timeline.py` shows drifting per-time CRVAL, but
`xr.open_zarr` reports `SKY.attrs["fits_wcs_header"]` equal to time 0.

**Cause:** Pre-fix incremental append wrote array-level attrs once; they do not
update per appended time (not a notebook bug).

**Solution:** Read with `open_dataset()` (strips in memory). New appends from
current `fits_to_zarr_xradio` omit those attrs. Optional: re-ingest or a
metadata repair pass only if you need clean on-disk `.zattrs`; not required for
analysis.

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
