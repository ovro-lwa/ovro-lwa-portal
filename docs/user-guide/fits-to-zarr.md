# FITS to Zarr Conversion

The OVRO-LWA Portal provides tools for converting FITS image files to
cloud-optimized Zarr format.

## Why Zarr?

Zarr offers several advantages over FITS for large radio astronomy datasets:

- **Cloud-optimized**: Efficient access to data stored in cloud object stores
- **Chunked storage**: Read only the data you need
- **Parallel I/O**: Multiple processes can read simultaneously
- **Compression**: Reduce storage requirements
- **Incremental updates**: Append new observations to existing stores

## Command-Line Interface

### Basic Conversion

Convert a directory of FITS files to Zarr:

```bash
ovro-ingest convert /path/to/fits /path/to/output
```

This will:

1. Scan the input directory for FITS files
2. Convert each file to Zarr format
3. Create a single consolidated Zarr store
4. Display progress with a rich progress bar

### Advanced Options

```bash
ovro-ingest convert /path/to/fits /path/to/output \
    --zarr-name custom_name.zarr \
    --chunk-lm 2048 \
    --rebuild
```

#### Options

- `--zarr-name`: Name of the output Zarr store (default: derived from input
  path)
- `--chunk-lm`: Chunk size for the l and m dimensions (default: 1024)
- `--rebuild`: Remove existing Zarr store and rebuild from scratch

### Get Help

```bash
ovro-ingest convert --help
```

## Python API

For more control, use the Python API directly:

### Basic Usage

```python
from pathlib import Path
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

# Configure conversion
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
)

# Execute conversion
converter = FITSToZarrConverter(config)
result = converter.convert()
print(f"Created: {result}")
```

### Configuration Options

```python
config = ConversionConfig(
    input_dir=Path("/path/to/fits"),
    output_dir=Path("/path/to/output"),
    zarr_name="ovro_lwa_data.zarr",
    chunk_lm=2048,              # Chunk size for l/m dimensions
    rebuild=False,              # Whether to rebuild existing store
    compressor=None,            # Custom compression (optional)
)
```

### Incremental Processing

Append new observations to an existing Zarr store:

```python
# First conversion
config1 = ConversionConfig(
    input_dir=Path("/path/to/fits/batch1"),
    output_dir=Path("/path/to/output"),
    zarr_name="observations.zarr",
)
converter1 = FITSToZarrConverter(config1)
converter1.convert()

# Append more data
config2 = ConversionConfig(
    input_dir=Path("/path/to/fits/batch2"),
    output_dir=Path("/path/to/output"),
    zarr_name="observations.zarr",  # Same name
    rebuild=False,                   # Don't rebuild
)
converter2 = FITSToZarrConverter(config2)
converter2.convert()
```

## Concurrent Write Protection

The converter uses file locking to prevent data corruption when multiple
processes write to the same Zarr store:

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ovro_lwa_portal.ingest import FITSToZarrConverter
from ovro_lwa_portal.ingest.core import ConversionConfig

def convert_batch(batch_dir):
    config = ConversionConfig(
        input_dir=batch_dir,
        output_dir=Path("/path/to/output"),
        zarr_name="shared_observations.zarr",
    )
    converter = FITSToZarrConverter(config)
    return converter.convert()

# Safe to run in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    batches = [Path(f"/path/to/batch{i}") for i in range(4)]
    results = executor.map(convert_batch, batches)
```

## Preserving WCS Coordinates

The converter automatically preserves World Coordinate System (WCS) information:

- Right Ascension (RA) and Declination (Dec)
- Frequency and time coordinates
- Observation metadata

After conversion, you can work with celestial coordinates directly:

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset("observations.zarr")

# Access WCS coordinates
ra = ds.coords['ra']
dec = ds.coords['dec']

# Select by celestial coordinates
region = ds.sel(ra=slice(10, 20), dec=slice(-5, 5))
```

## Best Practices

1. **Chunk Size**: Choose chunk sizes that match your access patterns

   - For time series analysis: larger time chunks
   - For spatial analysis: larger l/m chunks

2. **Compression**: Use compression for reduced storage (enabled by default)

3. **Incremental Updates**: Use `rebuild=False` to append new data

4. **Parallel Processing**: The converter is safe for concurrent writes

5. **Cloud Storage**: Convert data once, then access efficiently from anywhere
