# ovro-lwa-portal Development Guidelines

Auto-generated from AGENTS.md and feature plans. Last updated: 2025-10-23

## Repository Overview

OVRO-LWA Portal is a Python library for radio astronomy data processing and
visualization for the Owens Valley Radio Observatory - Long Wavelength Array
(OVRO-LWA). It provides tools for processing radio astronomy data, converting
FITS files to Zarr format, and creating visualization components for scientific
analysis.

**Repository Type:** Python library / Scientific software **License:** BSD
3-Clause **Python Version:** 3.12+ **Build System:** Pixi (v0.55.0), Hatchling

- hatch-vcs **Platforms:** macOS (osx-arm64), Linux (linux-64)

## Active Technologies

### Core Dependencies

- **Language**: Python 3.12+
- **Radio Astronomy**: xradio[all]>=0.0.59,<0.1 (radio astronomy data
  processing), python-casacore (CASA bindings)
- **Core Libraries**: astropy>=7.1.0,<8 (FITS I/O), xarray>=2025.9.1,<2026 (data
  structures), dask>=2025.9.1,<2026 (parallel processing), zarr>=2.16,<3
  (storage, v2 pinned), numcodecs>=0.15,<0.16 (compression), netcdf4, numpy
  (numerical operations)
- **Data Validation**: Pydantic v2.0+, pydantic-settings (planned for
  001-build-an-ingest)
- **CLI & UI**: typer v0.12+ (CLI with Pydantic integration), rich (terminal UI,
  progress bars) (planned for 001-build-an-ingest)
- **Orchestration**: prefect (optional orchestration layer) (planned for
  001-build-an-ingest)
- **Storage**: File system (FITS input, Zarr output directories), file locking
  for concurrent access control (planned for 001-build-an-ingest)

### Development & Testing

- **Testing**: pytest>=6, pytest-cov, pytest-xdist, pytest-mock
- **Code Quality**: pre-commit>=4.3.0, ruff (line length: 100), mypy (strict
  mode)
- **CI/CD**: s3fs>=2024.6.0, GitHub Actions
- **Version Control**: gh (GitHub CLI) >=2.0.0

## Build System & Environment Management

**CRITICAL: Use Pixi exclusively for dependency management. Never use conda,
pip, or venv directly.**

### Prerequisites

```bash
# Verify Pixi is installed (v0.55.0 or higher)
pixi --version
```

If not installed: <https://pixi.sh/latest/#installation>

### Environment Setup (ALWAYS RUN FIRST)

```bash
# Install the default environment (required before any other commands)
pixi install

# This installs:
# - Python 3.12 with radio astronomy packages
# - pre-commit, gh CLI, and all dependencies
# - Creates .pixi/envs/default directory
```

### Available Environments

1. **`default`** (features: `pre-commit`, `gh-cli`)

   - Standard development environment
   - Radio astronomy packages: astropy, xarray, dask, zarr, xradio,
     python-casacore
   - Use for: general development, running pre-commit checks, data processing

2. **`onboard`** (features: `pre-commit`, `gh-cli`, `onboard`)
   - Extended environment with ssec-cli
   - Use for: first-time setup, onboarding new contributors

### Key Pixi Commands

```bash
# Add dependencies
pixi add <package-name>              # Add conda package
pixi add --pypi <package-name>       # Add PyPI package
pixi add --feature <name> <package>  # Add to specific feature
pixi install                         # After manual pyproject.toml edits

# Pre-commit
pixi run pre-commit-install          # Install git hooks (once per clone)
pixi run pre-commit                  # Check staged files
pixi run pre-commit-all              # Check all files (REQUIRED before PR)

# GitHub CLI
pixi run gh --version                # Check version
pixi run gh <command>                # Use GitHub CLI

# Onboarding
pixi install -e onboard              # Install onboard environment
pixi run -e onboard onboard          # Run complete onboarding
pixi run -e onboard ssec-setup       # Set up ssec CLI completion
```

### Available Pixi Tasks

Run `pixi task list` to see all available tasks:

- `pre-commit-install`: Install git hooks
- `pre-commit`: Run checks on staged files
- `pre-commit-all`: Run checks on all files
- `ssec-setup`: Set up ssec CLI completion (onboard env only)
- `onboard`: Full onboarding process (onboard env only)

## Project Structure

```
.
├── .github/
│   └── workflows/               # GitHub Actions workflows
│       ├── ci.yml              # CI: pre-commit + tests
│       ├── cd.yml              # CD: build and publish to PyPI
│       └── copilot-setup-steps.yml
├── .devcontainer/              # VS Code Dev Container config
│   ├── devcontainer.json      # 4 CPUs, 16GB RAM required
│   ├── Dockerfile
│   └── onCreate.sh
├── .ci-helpers/                # CI/CD helper scripts
│   ├── README.md
│   └── download_test_fits.py  # Download test FITS from Caltech S3
├── pyproject.toml              # PRIMARY CONFIG: build, deps, Pixi tasks
├── pixi.lock                   # Auto-generated lock file
├── .pre-commit-config.yaml     # Pre-commit hook configuration
├── AGENTS.md                   # AI assistant guidance
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # Contribution guidelines
├── CODE_OF_CONDUCT.md          # Contributor Covenant v2.0
├── LICENSE                     # BSD 3-Clause License
├── onboarded.md                # Onboarding marker file
├── fixed_fits/                 # Directory for corrected FITS files
├── notebooks/                  # Jupyter notebooks
│   ├── fits2zarr.ipynb        # Main FITS to Zarr conversion
│   ├── fits2zarr_and_viz_user_cases.ipynb
│   └── test_fits_files/       # Test FITS (downloaded separately)
├── src/ovro_lwa_portal/       # Main package source
│   ├── __init__.py
│   ├── version.py             # Auto-generated from git tags
│   └── fits_to_zarr_xradio.py # FITS to Zarr conversion
└── tests/                      # Test suite
    ├── test_import.py
    ├── test_fits_to_zarr.py
    └── test_ci_helpers.py
```

## Commands

### Pre-commit Checks (MANDATORY BEFORE PRs)

```bash
# Install hooks (run once per clone)
pixi run pre-commit-install

# Run on staged files
pixi run pre-commit

# Run on ALL files (required before PR submission)
pixi run pre-commit-all
```

### Running Tests

```bash
# Run all tests
pixi run pytest

# Run with coverage
pixi run pytest --cov=ovro_lwa_portal

# Run specific test file
pixi run pytest tests/test_fits_to_zarr.py
```

### Working with Test Data

Test FITS files are stored in Caltech S3 bucket. Download via:

```bash
# Requires S3 credentials: CALTECH_KEY, CALTECH_SECRET, CALTECH_ENDPOINT_URL, CALTECH_DEV_S3_BUCKET
python .ci-helpers/download_test_fits.py
```

For local development, manually place test FITS in `notebooks/test_fits_files/`

### Code Quality

```bash
# Check code with ruff
pixi run ruff check .

# Format code with ruff
pixi run ruff format .

# Type check with mypy
pixi run mypy src/ovro_lwa_portal
```

## Code Style

### General Standards

- **Python Version**: 3.12+
- **Line Length**: 100 characters (Ruff)
- **Type Hints**: Required for all functions
- **Docstrings**: Required for public APIs
- **Formatting**: Follow PEP 8

### Ruff Configuration

- **Enabled rule sets**: flake8-bugbear, isort, flake8-unused-arguments,
  flake8-comprehensions, flake8-errmsg, NumPy/pandas rules
- **Tests excluded from**: print statement checks

### Mypy Configuration

- **Target**: Python 3.12
- **Strict mode**: Enabled for `ovro_lwa_portal.*` modules
- **Relaxed for**: tests

### Pre-commit Hooks

- File checks: large files, case conflicts, merge conflicts, broken symlinks
- YAML validation
- Python debug statement detection
- File formatting: end-of-files, line endings, trailing whitespace
- prettier: Markdown, YAML, JSON formatting
- codespell: Spell checking
- Capitalization validation
- Dependabot config and GitHub workflows validation

**Files Excluded**: `pixi.lock`, `onboarded.md`

## Validated Workflows

### Making Changes

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

   # Verify all checks pass
   ```

5. **Create PR:**
   - Follow Conventional Commits for PR titles
   - Link related issues: "Resolves #123"
   - Confirm `pre-commit run --all-files` was run (required by PR template)

## Radio Astronomy Context

This project works specifically with:

- **FITS files**: Standard astronomical image format from OVRO-LWA observations
- **Zarr format**: Cloud-optimized array storage for large datasets
- **xradio**: Primary library for radio astronomy data processing
- **python-casacore**: Python bindings for CASA (Common Astronomy Software
  Applications) core library

## CI/CD Pipelines

### CI Workflow (`.github/workflows/ci.yml`)

- **Triggers**: Pull requests, pushes to main, manual dispatch
- **Jobs**:
  1. **Format Check**: Runs `pixi run pre-commit-all` on ubuntu-latest
  2. **Tests**: Matrix strategy [ubuntu-latest, macos-14], pytest with coverage,
     uploads to Codecov
- **Concurrency**: Cancels in-progress runs for same ref

### CD Workflow (`.github/workflows/cd.yml`)

- **Triggers**: Releases (published), PRs, pushes to main, manual dispatch
- **Jobs**:
  1. **Build**: Builds Python package
  2. **Publish**: Publishes to TestPyPI on release (ready for production PyPI)

## Common Issues & Solutions

### Issue: "pre-commit not found" or command fails

**Solution:** Run `pixi install` first. Pixi manages pre-commit installation.

### Issue: Pre-commit check fails after making fixes

**Behavior:** Pre-commit hooks auto-fix files and show "files were modified by
this hook"

**Solution:** This is expected. Re-stage fixed files: `git add <files>` and
re-run `pixi run pre-commit`

### Issue: Platform-specific package installation fails

**Solution:**

```bash
# Reinstall environment
pixi install

# If still broken, remove and reinstall
rm -rf .pixi
pixi install
```

### Issue: Platform not supported

**Current platforms:** `osx-arm64`, `linux-64`

**Solution:** Edit `pyproject.toml`:

```toml
[tool.pixi.workspace]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

Then run `pixi install`

## Documentation Standards

- Follow Conventional Commits for commit messages and PR titles
- Use Markdown with proper formatting (prettier enforced)
- Spell check enabled (codespell)
- No trailing whitespace
- Files must end with newline
- Consistent line endings

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
