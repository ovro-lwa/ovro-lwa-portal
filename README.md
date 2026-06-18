# OVRO-LWA Portal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20128330.svg)](https://doi.org/10.5281/zenodo.20128330)
[![PyPI](https://img.shields.io/pypi/v/ovro-lwa-portal)](https://pypi.org/project/ovro-lwa-portal/)

A Python library for radio astronomy data processing and visualization for the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA).

## Features

- **Unified Data Loading**: Load OVRO-LWA data from local paths, remote URLs
  (S3, HTTPS), or DOI identifiers with a single `open_dataset()` function
- **FITS to Zarr Conversion**: Convert OVRO-LWA FITS image files to
  cloud-optimized Zarr format
- **Command-Line Interface**: User-friendly `ovro-ingest` CLI with progress
  tracking
- **WCS Coordinate Preservation**: Maintain celestial coordinates (RA/Dec) for
  FITS-free analysis
- **Incremental Processing**: Append new observations to existing Zarr stores
- **Concurrent Write Protection**: File locking prevents data corruption from
  simultaneous processes
- **Optional Workflow Orchestration**: Prefect integration for production
  deployments

## Installation

Choose the path that matches how you work with the project. **Users** should
install with **pip** into a Python 3.12+ environment (conda, venv, or system).
**Contributors** should use **Pixi** so dependencies match CI and the locked
`pixi.lock` file.

### For users (pip)

Requires **Python 3.12 or newer**.

```bash
# Latest release from PyPI
pip install ovro-lwa-portal

# Or install from GitHub main
pip install git+https://github.com/uw-ssec/ovro-lwa-portal.git
```

**Optional extras** (install only what you need):

```bash
# Interactive notebooks (Panel, Bokeh, SkyWidget, ipyaladin) — see Jupyter section below
pip install 'ovro-lwa-portal[visualization]'

# Remote data (S3, GCS, HTTP)
pip install 'ovro-lwa-portal[remote]'

# Prefect workflow orchestration
pip install 'ovro-lwa-portal[prefect]'
```

For notebooks such as `notebooks/source_review.ipynb`, Panel and Bokeh versions
matter for live UI updates. If heatmaps or other Panel panes look frozen while
Python state is correct, align with the versions used in this repo's Pixi lock:

```bash
pip install 'ovro-lwa-portal[visualization]' 'panel>=1.8.10,<2' 'bokeh>=3.9,<4'
```

Verify:

```bash
python -c "import ovro_lwa_portal; print(ovro_lwa_portal.__version__)"
ovro-ingest --help
```

### For developers (Pixi)

Contributors should use [Pixi](https://pixi.sh) so the environment matches CI.
Install Pixi from the
[Pixi installation guide](https://pixi.sh/latest/#installation), then:

```bash
git clone https://github.com/uw-ssec/ovro-lwa-portal.git
cd ovro-lwa-portal
pixi install
pixi run pre-commit-install
```

First-time SSEC onboarding (optional):

```bash
pixi install -e onboard
pixi run -e onboard onboard
```

Run commands through Pixi (`pixi run pytest`, `pixi run pre-commit`, etc.) or
enter the environment with `pixi shell`. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the full development workflow, adding dependencies, and pull request
guidelines.

**Pixi environments:**

| Environment | Use                                                     |
| ----------- | ------------------------------------------------------- |
| `default`   | Day-to-day development, tests, notebooks, visualization |
| `onboard`   | First-time setup with SSEC onboarding tools             |
| `ci`        | CI-only dependencies (used in GitHub Actions)           |

## Jupyter notebooks and kernels

Interactive review notebooks (`notebooks/source_review.ipynb`,
`notebooks/jupiter_flux_review.ipynb`) need the **`[visualization]`** extra and
a Jupyter kernel that points at the **same environment** where the package is
installed. Start JupyterLab from that environment—not from a different conda env
or an old kernel left over from another install.

Those notebooks combine three front-end stacks:

| Stack           | Packages                          | Role in review notebooks                    |
| --------------- | --------------------------------- | ------------------------------------------- |
| Panel / Bokeh   | `panel`, `bokeh`, `jupyter_bokeh` | Heatmap, controls, status                   |
| anywidget       | `anywidget`, `ipywidgets`         | Custom widget comm (SkyWidget, HiPS)        |
| SkyWidget       | `astrowidget`                     | Radio image overlay (WebGL)                 |
| HiPS background | `ipyaladin`                       | Aladin Lite survey tiles behind the overlay |

`[visualization]` pulls in `astrowidget` and `ipyaladin`; both depend on
**anywidget**. JupyterLab must load the **anywidget** and **jupyter-widgets**
lab extensions from the same environment (see verification below).

### Pip / conda users

Use one environment for install, kernel registration, and launching Jupyter:

```bash
# Example: conda env named py312 (adjust to your env name)
conda activate py312

pip install 'ovro-lwa-portal[visualization]' 'panel>=1.8.10,<2' 'bokeh>=3.9,<4'
pip install jupyterlab ipykernel

# Register a kernel so the notebook UI lists this interpreter
python -m ipykernel install --user \
  --name ovro-lwa-portal \
  --display-name "OVRO-LWA Portal (pip)"

# Launch Jupyter from the same env
jupyter lab
```

**astrowidget source:** `pip install 'ovro-lwa-portal[visualization]'` installs
**astrowidget from PyPI** (`>=0.1.1`). Pixi development uses **astrowidget
`main` from GitHub**, which can be ahead of PyPI. If SkyWidget misbehaves after
a portal upgrade, align with the Git version:

```bash
pip install 'git+https://github.com/ovro-lwa/astrowidget.git'
```

**Editable astrowidget (pip, optional):** for local SkyWidget JS/Python work,
clone a sibling checkout, build the JS bundle, and install editable:

```bash
git clone https://github.com/ovro-lwa/astrowidget.git ../astrowidget
cd ../astrowidget
npm ci && npm run build   # bundles js/ → src/astrowidget/static/widget.js
pip install -e .
```

Requires **Node.js** for the build step. After any astrowidget JS change, rerun
`npm run build`, restart the Jupyter kernel, and hard-refresh the browser.

In JupyterLab: **Kernel → Change Kernel → OVRO-LWA Portal (pip)**, then run all
cells from the top. For `source_review.ipynb`, the first setup cell calls
`configure_source_review_notebook()`—run it once per kernel session before
displaying `review.panel`.

**Verify Jupyter front ends** (from the same env you use for `jupyter lab`):

```bash
jupyter labextension list
```

Expect **`anywidget`** and **`@jupyter-widgets/jupyterlab-manager`** to show
`enabled` and `OK`. If anywidget is missing:

```bash
pip install 'anywidget>=0.9' jupyterlab ipywidgets
```

**Sanity checks** (run in a notebook cell after setup):

```python
import sys
import ovro_lwa_portal
import panel, bokeh
import astrowidget, anywidget
from astrowidget import SkyWidget

print("python:", sys.executable)
print("ovro_lwa_portal:", ovro_lwa_portal.__file__)
print("panel:", panel.__version__, "| bokeh:", bokeh.__version__)
print("astrowidget:", astrowidget.__version__, "@", astrowidget.__file__)
print("anywidget:", anywidget.__version__)
_ = SkyWidget()  # should construct without import errors
```

- `ovro_lwa_portal.__file__` should be under `site-packages` for a normal
  install, or under `src/` if you used `pip install -e .`.
- If Panel panes do not update in the browser after **Generate heatmap**,
  restart the kernel, confirm the kernel name above, and check Panel/Bokeh
  versions.
- If the sky pane is blank, frozen, or missing HiPS, confirm **anywidget** lab
  extensions, **astrowidget** version/path, and that you restarted the kernel
  after reinstalling or rebuilding astrowidget.

More detail:
[Interactive visualization guide](docs/user-guide/interactive-visualization.md).

### Pixi developers

```bash
cd ovro-lwa-portal
pixi install

# Expose the Pixi env as a Jupyter kernel (one-time per machine/user)
pixi run python -m ipykernel install --user \
  --name ovro-lwa-portal-pixi \
  --display-name "ovro-lwa-portal (pixi)"

# Start JupyterLab with locked deps from pixi.lock
pixi run jupyter lab
```

Select **ovro-lwa-portal (pixi)** as the notebook kernel.

**astrowidget in Pixi:** the default environment installs astrowidget from
`github.com/ovro-lwa/astrowidget.git` (`main`), not PyPI—this is what CI and
most contributors test against. The package is editable in the Pixi env but the
**JS bundle is prebuilt in the wheel/git checkout**; after pulling astrowidget
or changing its `js/` sources, rebuild from the astrowidget repo:

```bash
cd ../astrowidget
pixi run build    # or: npm run build
```

Then restart the Jupyter kernel (and hard-refresh the browser if the overlay
still looks stale).

**Editable astrowidget sibling (optional):** to develop astrowidget alongside
the portal, add the `astrowidget-local` Pixi feature (see
[CONTRIBUTING.md](CONTRIBUTING.md)), run `pixi install`, rebuild JS, and restart
the kernel. Portal Python changes under `src/ovro_lwa_portal` are picked up
automatically via editable install; astrowidget JS changes are not until you
rebuild.

After changing `src/ovro_lwa_portal` or Python in astrowidget, restart the
kernel so the notebook picks up your edits.

## Quick Start

### Loading OVRO-LWA Data

Load data from various sources with a unified interface:

```python
import ovro_lwa_portal

# Load from local zarr store
ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

# Load from remote URL
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

# Load via DOI
ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")

# Customize chunking for large datasets
ds = ovro_lwa_portal.open_dataset(
    "path/to/data.zarr",
    chunks={"time": 100, "frequency": 50}  # or chunks="auto" (default), chunks=None
)
```

For remote data access, install with the remote extra:

```bash
pip install 'ovro-lwa-portal[remote]'
```

See the [open_dataset documentation](docs/open_dataset.md) for more details.

### Using the FITS to Zarr Ingest CLI

After installation, convert OVRO-LWA FITS files to Zarr format:

```bash
# Basic conversion
ovro-ingest convert /path/to/fits /path/to/output

# With custom options
ovro-ingest convert /path/to/fits /path/to/output \
    --zarr-name my_data.zarr \
    --chunk-lm 2048 \
    --rebuild

# Show help
ovro-ingest convert --help
```

For detailed documentation on the ingest module, see the
[Ingest Module README](src/ovro_lwa_portal/ingest/README.md).

### Using the Python API

```python
from pathlib import Path
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

# Configure conversion
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
    chunk_lm=1024,
)

# Execute conversion
converter = FITSToZarrConverter(config)
result = converter.convert()
print(f"Created: {result}")
```

## Technology Stack

- **Core**: Python 3.12, xarray, dask, zarr
- **Astronomy**: astropy, xradio, python-casacore
- **CLI**: typer, rich (progress bars and formatted output)
- **Workflow**: prefect (optional orchestration)
- **Storage**: Zarr format optimized for cloud access
- **Environment management**: pip (users) or Pixi (developers)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct and the process for submitting pull requests.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE)
file.

## Project Resources

- eScience Slack channel: 🔒
  [#ssec-ovro-lwa-portal](https://escience-institute.slack.com/archives/C098GJYLNBW)
- SSEC Sharepoint (**INTERNAL SSEC ONLY**): 🔒
  [Projects/OVROXarrraySciPlt](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt?csf=1&web=1&e=P5QKAc)
- Shared Sharepoint Directory: 🔒
  [UW SSEC Caltech OVRO-LWA Portal Shared Folder](https://uwnetid.sharepoint.com/:f:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt/UW%20SSEC%20Caltech%20OVRO-LWA%20Portal%20Shared%20Folder?csf=1&web=1&e=siXUk2)
- [User Stories Document 🔒](https://uwnetid.sharepoint.com/:w:/r/sites/og_ssec_escience/Shared%20Documents/Projects/OVROXarrraySciPlt/UW%20SSEC%20Caltech%20OVRO-LWA%20Portal%20Shared%20Folder/SSEC%20OVRO-LWA%20Portal%20User%20Stories.docx?d=w15624ab2d3c0475e95a2865a346e359b&csf=1&web=1&e=ImDH96)

## General Discussions

For general discussion, ideas, and resources please use the
[GitHub Discussions](https://github.com/uw-ssec/ovro-lwa-portal/discussions).
However, if there's an internal discussion that need to happen, please use the
slack channel provided.

- Meeting Notes in GitHub:
  [discussions/meetings](https://github.com/uw-ssec/ovro-lwa-portal/discussions/categories/meetings)

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{ovro_lwa_portal,
  title = {OVRO-LWA Portal},
  author = {Core, Cordero and Setiawan, Don and Tambay, Anshul T. and Kosogorov, Nikita and Johari, Ishika},
  url = {https://github.com/uw-ssec/ovro-lwa-portal},
  license = {BSD-3-Clause}
}
```

## Questions

If you have any questions about our process, or locations of SSEC resources,
please ask [Anshul Tambay](https://github.com/atambay37).
