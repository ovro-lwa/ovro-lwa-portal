# Implementation Roadmap: Issue #71 - xarray Accessor for OVRO-LWA Portal

**Issue**: https://github.com/uw-ssec/ovro-lwa-portal/issues/71 **Branch**:
`cdcore09/feat/xarray-accessor` **Status**: In Progress

---

## Overview

Implement a custom xarray accessor called `radport` to streamline access to
plotting and visualization features in the `ovro_lwa_portal` package.

---

## Current Status

**IMPLEMENTATION COMPLETE** - All acceptance criteria met plus enhanced
features. Ready for final review and PR preparation.

The `radport` xarray accessor has been fully implemented with:

- Accessor registration and auto-import
- Dataset validation with informative error messages
- `plot()` method with full customization options
- Selection helpers (`nearest_freq_idx`, `nearest_time_idx`, `nearest_lm_idx`)
- Enhanced plot parameters (`freq_mhz`, `time_mjd`, `mask_radius`)
- Advanced visualization: cutouts, dynamic spectra, difference maps
- Data quality methods: `find_valid_frame()`, `finite_fraction()`
- Grid plots: `plot_grid()`, `plot_frequency_grid()`, `plot_time_grid()`
- 1D analysis: `light_curve()`, `spectrum()`, `time_average()`,
  `frequency_average()`
- WCS coordinate support: `pixel_to_coords()`, `coords_to_pixel()`, `plot_wcs()`
- Animation & export: `animate_time()`, `animate_frequency()`, `export_frames()`
- Source detection: `rms_map()`, `snr_map()`, `find_peaks()`, `peak_flux_map()`
- 206 unit tests (all passing)
- Comprehensive documentation in `docs/open_dataset.md`

---

## Acceptance Criteria (from Issue #71)

- [x] Accessor `radport` is registered and available on xarray Datasets after
      importing
- [x] `ds.radport.plot()` successfully creates and displays a plot
- [x] Documentation with usage examples
- [x] Unit test coverage for registration and basic functionality
- [x] Compatibility with file paths and DOI-based dataset loading

---

## Implementation Phases

### Phase 1: Accessor Foundation

**Goal**: Create the basic accessor scaffold with registration and validation.

**Status**: COMPLETE

#### Tasks

- [x] Create `src/ovro_lwa_portal/accessor.py` with `RadportAccessor` class
- [x] Register with `@xr.register_dataset_accessor("radport")`
- [x] Update `src/ovro_lwa_portal/__init__.py` to auto-register accessor on
      import

#### Implementation Details

```python
@xr.register_dataset_accessor("radport")
class RadportAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._validate_structure()

    def _validate_structure(self):
        """Validate dataset has required dimensions and variables."""
        # Check for required dimensions: time, frequency, polarization, l, m
        # Check for required variable: SKY
        # Raise informative errors if validation fails
```

---

### Phase 2: Dataset Validation

**Goal**: Implement robust validation with informative error messages.

**Status**: COMPLETE (implemented as part of Phase 1)

#### Validation Requirements

Based on the existing data model:

**Required Dimensions**:

- `time` - Observation timestamps
- `frequency` - Frequency channels (Hz)
- `polarization` - Polarization indices
- `l`, `m` - Direction cosines (spatial)

**Required Data Variables**:

- `SKY` - 5D array `[time, frequency, polarization, l, m]`

**Optional Variables**:

- `BEAM` - 5D array (same shape as SKY)

**Required Coordinates**:

- `right_ascension` - 2D `[m, l]`
- `declination` - 2D `[m, l]`
- `fits_wcs_header` - WCS metadata string

#### Tasks

- [x] Implement `_validate_structure()` method
- [x] Add informative error messages for missing dimensions
- [x] Add informative error messages for missing variables
- [x] Handle optional variables gracefully (via `has_beam` property)

---

### Phase 3: Plotting Implementation

**Goal**: Implement `plot()` method for default visualization.

**Status**: COMPLETE

#### Core `plot()` Method Signature

```python
def plot(self, var='SKY', time_idx=0, freq_idx=0, pol=0,
         cmap='inferno', vmin=None, vmax=None, robust=False,
         figsize=(8, 6), add_colorbar=True, **kwargs):
    """
    Create default visualization of radio data.

    Parameters
    ----------
    var : str
        Data variable to plot ('SKY' or 'BEAM')
    time_idx : int
        Time index for snapshot
    freq_idx : int
        Frequency index for snapshot
    pol : int
        Polarization index
    cmap : str
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    robust : bool
        Use 2nd/98th percentile for color scaling
    figsize : tuple
        Figure size in inches
    add_colorbar : bool
        Whether to add colorbar

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
```

#### Tasks

- [x] Implement basic `plot()` method using matplotlib
- [x] Support selecting time/frequency/polarization indices
- [x] Support colormap and scale customization
- [x] Add axis labels with coordinate information
- [x] Return figure object for further customization
- [x] Add robust scaling option (2nd/98th percentile)

#### Dependencies

- [x] Added matplotlib>=3.8,<4 to `pyproject.toml`

---

### Phase 4: Unit Tests

**Goal**: Comprehensive test coverage for accessor functionality.

**Status**: COMPLETE

#### Tasks

- [x] Create `tests/test_accessor.py` (28 tests)
- [x] Add fixtures for valid/invalid datasets in `tests/conftest.py`
- [x] Test accessor registration (3 tests)
- [x] Test validation error messages (5 tests)
- [x] Test plot() method returns figure (16 tests)
- [x] Test plot() with NaN values (2 tests)

#### Test Structure

```python
class TestRadportAccessorRegistration:
    """Test accessor is properly registered."""

    def test_accessor_available_after_import(self, valid_dataset):
        """Accessor 'radport' is available on xarray Datasets."""

    def test_accessor_type(self, valid_dataset):
        """Accessor returns RadportAccessor instance."""

class TestRadportValidation:
    """Test dataset validation."""

    def test_missing_dimension_raises(self):
        """Missing required dimension raises ValueError."""

    def test_missing_variable_raises(self):
        """Missing SKY variable raises ValueError."""

class TestRadportPlot:
    """Test plot() method."""

    def test_plot_returns_figure(self, valid_dataset):
        """plot() returns matplotlib Figure."""

    def test_plot_with_options(self, valid_dataset):
        """plot() accepts visualization options."""
```

---

### Phase 5: Documentation

**Goal**: Usage examples and API documentation.

**Status**: COMPLETE

#### Tasks

- [x] Update `docs/open_dataset.md` with accessor usage section
- [x] Add NumPy-style docstrings to all public methods
- [x] Include example code snippets

#### Documentation Example

````markdown
## Using the radport Accessor

After loading a dataset, access plotting and analysis features via the `radport`
accessor:

```python
import ovro_lwa_portal
from ovro_lwa_portal import open_dataset

ds = open_dataset("doi:10.22002/...")

# Create default visualization
ds.radport.plot()

# Customize plot options
ds.radport.plot(time_idx=5, freq_idx=100, cmap='viridis')
```
````

````

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/ovro_lwa_portal/accessor.py` | **CREATE** | Accessor class with validation and plot() |
| `src/ovro_lwa_portal/__init__.py` | **MODIFY** | Import accessor for auto-registration |
| `tests/test_accessor.py` | **CREATE** | Unit tests for accessor |
| `tests/conftest.py` | **MODIFY** | Add test fixtures for valid datasets |
| `docs/open_dataset.md` | **MODIFY** | Add accessor usage documentation |
| `pyproject.toml` | **MODIFY** | Add matplotlib dependency if not present |

---

## Out of Scope

Per Issue #71, the following are explicitly excluded from this implementation:

- Advanced plotting methods (spectra, waterfalls, beam patterns)
- Interactive visualization options
- Export/save functionality
- Plot styling configuration

These can be added as future enhancements after the core accessor is merged.

---

## Phase A: Enhanced Selection & Masking

**Goal**: Add selection helpers and enhanced plot parameters identified from notebook analysis.

**Status**: COMPLETE

### New Selection Helper Methods

```python
def nearest_freq_idx(self, freq_mhz: float) -> int:
    """Find the index of the frequency nearest to the given value in MHz."""

def nearest_time_idx(self, mjd: float) -> int:
    """Find the index of the time nearest to the given MJD value."""

def nearest_lm_idx(self, l: float, m: float) -> tuple[int, int]:
    """Find the indices of the (l, m) pixel nearest to the given coordinates."""
````

### Enhanced plot() Parameters

New parameters added to `plot()`:

- `freq_mhz: float | None` - Select frequency by value in MHz (overrides
  `freq_idx`)
- `time_mjd: float | None` - Select time by MJD value (overrides `time_idx`)
- `mask_radius: int | None` - Apply circular mask to hide invalid edge pixels

### Tests Added

16 new tests added for Phase A functionality:

- 9 tests for selection helpers (`TestRadportSelectionHelpers`)
- 4 tests for frequency/time selection (`TestRadportPlotFrequencySelection`)
- 3 tests for circular masking (`TestRadportPlotMasking`)

---

## Phase B: Advanced Visualization Methods

**Goal**: Add cutout, dynamic spectrum, and difference map capabilities
identified from notebook analysis.

**Status**: COMPLETE

### New Methods

#### Spatial Cutouts

```python
def cutout(self, l_center, m_center, dl, dm, ...) -> xr.DataArray:
    """Extract a spatial cutout (rectangular region) from the data."""

def plot_cutout(self, l_center, m_center, dl, dm, ...) -> Figure:
    """Extract and plot a spatial cutout."""
```

#### Dynamic Spectrum

```python
def dynamic_spectrum(self, l, m, ...) -> xr.DataArray:
    """Extract a dynamic spectrum (time vs frequency) for a single pixel."""

def plot_dynamic_spectrum(self, l, m, ...) -> Figure:
    """Plot a dynamic spectrum (time vs frequency) for a single pixel."""
```

#### Difference Maps

```python
def diff(self, mode='time', ...) -> xr.DataArray:
    """Compute a difference map between adjacent time or frequency slices."""

def plot_diff(self, mode='time', ...) -> Figure:
    """Plot a difference map between adjacent time or frequency slices."""
```

### Tests Added

31 new tests added for Phase B functionality:

- 7 tests for cutout() (`TestRadportCutout`)
- 3 tests for plot_cutout() (`TestRadportPlotCutout`)
- 5 tests for dynamic_spectrum() (`TestRadportDynamicSpectrum`)
- 3 tests for plot_dynamic_spectrum() (`TestRadportPlotDynamicSpectrum`)
- 8 tests for diff() (`TestRadportDiff`)
- 5 tests for plot_diff() (`TestRadportPlotDiff`)

---

## Phase C: Data Quality & Grid Plots

**Goal**: Add data quality assessment and multi-panel grid visualization.

**Status**: COMPLETE

### New Methods

#### Data Quality Assessment

```python
def find_valid_frame(self, var="SKY", pol=0, min_finite_fraction=0.1) -> tuple[int, int]:
    """Find the first (time, freq) frame with sufficient finite data."""

def finite_fraction(self, var="SKY", pol=0) -> xr.DataArray:
    """Compute the fraction of finite (non-NaN) pixels for each (time, freq)."""
```

#### Grid Plot Methods

```python
def plot_grid(self, time_indices=None, freq_indices=None, freq_mhz_list=None,
              var="SKY", pol=0, ncols=4, ...) -> Figure:
    """Create a grid of plots showing multiple time/frequency combinations."""

def plot_frequency_grid(self, time_idx=0, freq_mhz_list=None, ...) -> Figure:
    """Create a grid showing all frequencies at a fixed time."""

def plot_time_grid(self, freq_idx=None, freq_mhz=None, time_indices=None, ...) -> Figure:
    """Create a grid showing all times at a fixed frequency."""
```

### Tests Added

24 new tests added for Phase C functionality:

- 4 tests for find_valid_frame() (`TestRadportFindValidFrame`)
- 5 tests for finite_fraction() (`TestRadportFiniteFraction`)
- 8 tests for plot_grid() (`TestRadportPlotGrid`)
- 3 tests for plot_frequency_grid() (`TestRadportPlotFrequencyGrid`)
- 4 tests for plot_time_grid() (`TestRadportPlotTimeGrid`)

---

## Phase D: 1D Analysis Methods

**Goal**: Add light curves, spectra, and averaging methods for time/frequency
analysis.

**Status**: COMPLETE

### New Methods

#### Light Curves (Time Series)

```python
def light_curve(self, l, m, freq_idx=None, freq_mhz=None, var="SKY", pol=0) -> xr.DataArray:
    """Extract a light curve (time series) at a specific spatial location."""

def plot_light_curve(self, l, m, freq_idx=None, freq_mhz=None, ...) -> Figure:
    """Plot a light curve (time series) at a specific spatial location."""
```

#### Frequency Spectra

```python
def spectrum(self, l, m, time_idx=None, time_mjd=None, var="SKY", pol=0) -> xr.DataArray:
    """Extract a frequency spectrum at a specific spatial location and time."""

def plot_spectrum(self, l, m, time_idx=None, time_mjd=None, freq_unit="MHz", ...) -> Figure:
    """Plot a frequency spectrum at a specific spatial location and time."""
```

#### Averaging Methods

```python
def time_average(self, var="SKY", pol=0, time_indices=None) -> xr.DataArray:
    """Compute the time-averaged image."""

def frequency_average(self, var="SKY", pol=0, freq_indices=None,
                      freq_min_mhz=None, freq_max_mhz=None) -> xr.DataArray:
    """Compute the frequency-averaged image."""

def plot_time_average(self, freq_idx=None, freq_mhz=None, ...) -> Figure:
    """Plot the time-averaged image at a specific frequency."""

def plot_frequency_average(self, time_idx=None, time_mjd=None,
                           freq_min_mhz=None, freq_max_mhz=None, ...) -> Figure:
    """Plot the frequency-averaged image at a specific time."""
```

### Tests Added

42 new tests added for Phase D functionality:

- 7 tests for light_curve() (`TestRadportLightCurve`)
- 3 tests for plot_light_curve() (`TestRadportPlotLightCurve`)
- 6 tests for spectrum() (`TestRadportSpectrum`)
- 4 tests for plot_spectrum() (`TestRadportPlotSpectrum`)
- 6 tests for time_average() (`TestRadportTimeAverage`)
- 8 tests for frequency_average() (`TestRadportFrequencyAverage`)
- 4 tests for plot_time_average() (`TestRadportPlotTimeAverage`)
- 4 tests for plot_frequency_average() (`TestRadportPlotFrequencyAverage`)

---

## Additional Phases

The following phases extend the core accessor functionality with advanced
features.

---

### Phase E: WCS & Coordinate Plotting

**Goal**: Add World Coordinate System (WCS) support for RA/Dec plotting.

**Status**: COMPLETE

#### Implemented Methods

```python
@property
def has_wcs(self) -> bool:
    """Check if WCS coordinate information is available in the dataset."""

def pixel_to_coords(self, l_idx: int, m_idx: int) -> tuple[float, float]:
    """Convert pixel indices to celestial coordinates (RA, Dec)."""

def coords_to_pixel(self, ra: float, dec: float) -> tuple[int, int]:
    """Convert celestial coordinates (RA, Dec) to pixel indices."""

def plot_wcs(self, var="SKY", time_idx=0, freq_idx=0, freq_mhz=None, pol=0,
             cmap="inferno", vmin=None, vmax=None, robust=True,
             mask_radius=None, figsize=(10, 10), add_colorbar=True,
             grid_color="white", grid_alpha=0.6, grid_linestyle=":",
             label_color="white", facecolor="black", **kwargs) -> Figure:
    """Plot with WCS projection and celestial coordinate grid overlay."""
```

#### WCS Header Sources

The accessor looks for WCS information in the following locations (in order):

1. Variable attributes: `ds["SKY"].attrs["fits_wcs_header"]`
2. Dataset attributes: `ds.attrs["fits_wcs_header"]`
3. Dedicated variable: `ds["wcs_header_str"]`

#### Tests Added

20 new tests added for Phase E functionality:

- 2 tests for has_wcs property (`TestRadportHasWcs`)
- 6 tests for pixel_to_coords() (`TestRadportPixelToCoords`)
- 5 tests for coords_to_pixel() (`TestRadportCoordsToPixel`)
- 7 tests for plot_wcs() (`TestRadportPlotWcs`)

#### Dependencies

- Uses existing `astropy>=7.1.0,<8` dependency for WCS handling
- Uses `fits_wcs_header` attribute from dataset variables or attributes

---

### Phase F: Animation & Export

**Goal**: Add animation generation and frame export capabilities.

**Status**: COMPLETE

#### Implemented Methods

```python
def animate_time(self, freq_idx=None, freq_mhz=None, var="SKY", pol=0,
                 output_file=None, fps=5, cmap="inferno", vmin=None, vmax=None,
                 robust=True, mask_radius=None, figsize=(8, 6), dpi=100,
                 **kwargs) -> FuncAnimation:
    """Create animation showing time evolution at a fixed frequency."""

def animate_frequency(self, time_idx=None, time_mjd=None, var="SKY", pol=0,
                      output_file=None, fps=5, cmap="inferno", vmin=None, vmax=None,
                      robust=True, mask_radius=None, figsize=(8, 6), dpi=100,
                      **kwargs) -> FuncAnimation:
    """Create animation showing frequency sweep at a fixed time."""

def export_frames(self, output_dir, var="SKY", pol=0, time_indices=None,
                  freq_indices=None, format="png", cmap="inferno", vmin=None,
                  vmax=None, robust=True, mask_radius=None, figsize=(8, 6),
                  dpi=150, filename_template="...") -> list[str]:
    """Export all (time, freq) frames as individual image files."""
```

#### Features

- **Time animations**: Animate across all time steps at a fixed frequency
- **Frequency animations**: Animate across all frequencies at a fixed time
- **Multiple output formats**: Support for MP4 (via ffmpeg) and GIF (via pillow)
- **Frame export**: Export individual frames as PNG, JPG, or PDF
- **Customizable filenames**: Template-based filename generation with
  placeholders
- **Global color scaling**: Consistent color scale across all frames
- **Circular masking**: Optional mask for all-sky images

#### Tests Added

19 new tests added for Phase F functionality:

- 6 tests for animate_time() (`TestRadportAnimateTime`)
- 5 tests for animate_frequency() (`TestRadportAnimateFrequency`)
- 8 tests for export_frames() (`TestRadportExportFrames`)

#### Dependencies

- Uses `matplotlib.animation.FuncAnimation` (included with matplotlib)
- MP4 export requires `ffmpeg` installed on the system
- GIF export uses `pillow` writer (no additional dependencies)

---

### Phase G: Source Detection

**Goal**: Add basic source finding and signal-to-noise analysis.

**Status**: COMPLETE

#### Implemented Methods

```python
def rms_map(self, time_idx=0, freq_idx=None, freq_mhz=None, var="SKY",
            pol=0, box_size=50) -> xr.DataArray:
    """Compute local RMS noise estimate map using a sliding box."""

def snr_map(self, time_idx=0, freq_idx=None, freq_mhz=None, var="SKY",
            pol=0, box_size=50) -> xr.DataArray:
    """Compute signal-to-noise ratio map."""

def find_peaks(self, time_idx=0, freq_idx=None, freq_mhz=None, var="SKY",
               pol=0, threshold_sigma=5.0, box_size=50,
               min_separation=5) -> list[dict]:
    """Find peaks above threshold in the image.
    Returns list of dicts with keys: l, m, l_idx, m_idx, flux, snr."""

def peak_flux_map(self, var="SKY", pol=0, freq_idx=None,
                  freq_mhz=None) -> xr.DataArray:
    """Compute peak flux at each pixel across all times."""

def plot_snr_map(self, time_idx=0, freq_idx=None, freq_mhz=None, var="SKY",
                 pol=0, box_size=50, cmap="RdBu_r", ...) -> Figure:
    """Plot the signal-to-noise ratio map with symmetric colorscale."""
```

#### Features

- **Local RMS computation**: Sliding box approach with NaN handling
- **SNR maps**: Signal divided by local RMS noise
- **Peak detection**: Local maximum finding with SNR threshold and minimum
  separation
- **Peak flux maps**: Maximum across time dimension
- **SNR plotting**: Diverging colormap with symmetric scaling

#### Tests Added

26 new tests added for Phase G functionality:

- 6 tests for rms_map() (`TestRadportRmsMap`)
- 5 tests for snr_map() (`TestRadportSnrMap`)
- 6 tests for find_peaks() (`TestRadportFindPeaks`)
- 5 tests for peak_flux_map() (`TestRadportPeakFluxMap`)
- 4 tests for plot_snr_map() (`TestRadportPlotSnrMap`)

#### Dependencies

- Uses `scipy.ndimage.uniform_filter` for efficient local statistics
- Uses `scipy.ndimage.maximum_filter` for local maximum detection

---

## Future Phases (Not Yet Implemented)

The following phases are potential future enhancements. They are **not
required** for the initial PR but could be added later.

---

### Phase H: Spectral Analysis (Low Priority)

**Goal**: Add spectral index and flux density calculations.

**Status**: COMPLETE

#### Implemented Methods

1. **`spectral_index(l, m, ...)`** - Compute spectral index (power-law slope) at
   a location

   - Returns α where S ∝ ν^α, computed as log(S₂/S₁) / log(ν₂/ν₁)
   - Supports frequency selection via MHz or index

2. **`spectral_index_map(...)`** - Compute spectral index map across the full
   image

   - Returns DataArray with dims (l, m)
   - Includes frequency metadata in attrs

3. **`integrated_flux(l, m, ...)`** - Compute integrated flux density over
   frequency band

   - Uses trapezoidal integration (numpy.trapezoid)
   - Returns flux in Jy·Hz

4. **`plot_spectral_index_map(...)`** - Plot spectral index map with diverging
   colormap
   - RdBu_r colormap centered on 0
   - Optional horizon masking support

#### Tests (19 new tests)

- `TestRadportSpectralIndex` - 5 tests
- `TestRadportSpectralIndexMap` - 5 tests
- `TestRadportIntegratedFlux` - 5 tests
- `TestRadportPlotSpectralIndexMap` - 4 tests

#### Notes

- Non-positive flux values result in NaN spectral indices
- At least two frequency channels required
- Typical spectral index values: ~-0.7 (synchrotron), ~+2 (thermal), ~-0.1
  (free-free)

---

## Progress Tracking

### Completed

- [x] Analyze current codebase state
- [x] Create implementation roadmap
- [x] Phase 1: Accessor Foundation
- [x] Phase 2: Dataset Validation (merged into Phase 1)
- [x] Phase 3: Plotting Implementation
- [x] Phase 4: Unit Tests (28 tests, all passing)
- [x] Phase 5: Documentation
- [x] Phase A: Enhanced Selection & Masking (44 tests total, all passing)
- [x] Phase B: Advanced Visualization Methods (75 tests total, all passing)
- [x] Phase C: Data Quality & Grid Plots (99 tests total, all passing)
- [x] Phase D: 1D Analysis Methods (141 tests total, all passing)
- [x] Phase E: WCS & Coordinate Plotting (161 tests total, all passing)
- [x] Phase F: Animation & Export (180 tests total, all passing)
- [x] Phase G: Source Detection (206 tests total, all passing)
- [x] Phase H: Spectral Analysis (225 tests total, all passing)

### In Progress

(none)

### Planned (Future Enhancements)

(none - all phases complete)

### Ready for PR

- [x] Final review complete - all checks pass
- [x] All Issue #71 acceptance criteria met
- [x] 225 unit tests passing
- [x] Documentation complete
