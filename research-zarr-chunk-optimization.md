# Research: Zarr Chunk Optimization for Cloud-Based Object Stores (OVRO-LWA)

---

**Date:** 2026-02-26 **Author:** AI Assistant **Status:** Active **Related
Documents:** `docs/open_dataset.md` (PR #95), external references listed in
References section

---

## Research Question

How should the OVRO-LWA portal documentation guide users through optimizing Zarr
chunk sizes for read performance on cloud-based object stores? What are the best
practices, benchmarking techniques, strategies for finding optimal chunk sizes,
and how do they apply to the existing codebase?

## Executive Summary

Zarr chunk optimization for cloud object stores is fundamentally different from
local filesystem optimization. Each chunk in a Zarr store maps to a single
object in cloud storage, meaning every chunk read requires an HTTP GET request
with ~50-100ms latency. The key tradeoff is between chunks that are too small
(excessive HTTP requests, high overhead) and too large (downloading unnecessary
data). Industry consensus targets **10-100 MB compressed chunks** for cloud
workloads, with the sweet spot depending on access patterns.

The OVRO-LWA codebase currently uses a fixed `chunk_lm=1024` default for spatial
dimensions during FITS-to-Zarr ingest, with no explicit compression or codec
configuration (relying on xradio/Zarr defaults). The `open_dataset()` function
defaults to `chunks="auto"` on read. This is a reasonable starting point, but
the documentation should help users understand when and how to tune these
parameters. For typical OVRO-LWA data (5D arrays: time x frequency x
polarization x l x m), proper chunk alignment can yield 10-60x read performance
improvements for cloud access, based on published benchmarks from similar radio
astronomy datasets.

This research synthesizes findings from 5 external references (Zarr
documentation, ESIP cloud optimization guides, HackMD chunking tutorials, and
cloud data engineering articles) with a detailed analysis of the OVRO-LWA
codebase's current chunking implementation. The goal is to inform a set of
documentation issues that AI agents will convert into comprehensive, actionable
documentation.

## Scope

**What This Research Covers:**

- Current chunking implementation in the OVRO-LWA codebase (write and read
  paths)
- Cloud object store performance characteristics affecting chunk design
- Industry best practices for Zarr chunk sizing
- Compression strategies and their interaction with chunking
- Benchmarking techniques for measuring chunk performance
- Access pattern analysis for OVRO-LWA data workflows
- Zarr metadata optimization (consolidated metadata, sharding)

**What This Research Does NOT Cover:**

- Zarr v3 specification (project is pinned to Zarr v2)
- Non-Zarr storage formats (HDF5, NetCDF, FITS)
- Network infrastructure tuning (TCP, DNS, CDN)
- Dask cluster configuration and distributed scheduling
- Write-path optimization beyond initial ingest

## Key Findings

### 1. Current Codebase Chunking Implementation

The OVRO-LWA portal has two distinct chunking contexts: **write-time** (during
FITS-to-Zarr ingest) and **read-time** (via `open_dataset()`).

**Write-Time Chunking (Ingest Pipeline):**

Relevant Files:

- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:241-316` -- `_load_for_combine()`
  applies `chunk_lm` to l and m dimensions only
- `src/ovro_lwa_portal/fits_to_zarr_xradio.py:498-598` --
  `convert_fits_dir_to_zarr()` accepts `chunk_lm` parameter (default 1024)
- `src/ovro_lwa_portal/ingest/core.py:43-58` -- `ConversionConfig` class with
  `chunk_lm: int = 1024`
- `src/ovro_lwa_portal/ingest/cli.py` -- CLI exposes `--chunk-lm` (default 1024,
  min 0)

Key observations:

- Only spatial dimensions (l, m) are chunked; time and frequency are not
  explicitly chunked at write time
- No explicit compression codecs are configured -- relies entirely on xradio
  `write_image()` defaults
- The `write_image()` call passes through to Zarr with whatever encoding xradio
  applies
- Variable encodings are explicitly cleared (`xds[v].encoding = {}`) before
  writes at lines 313, 416, 472
- A `chunk_lm=0` disables spatial chunking entirely

**Read-Time Chunking (open_dataset):**

Relevant Files:

- `src/ovro_lwa_portal/io.py:487-718` -- `open_dataset()` function with
  `chunks="auto"` default
- `docs/open_dataset.md:93-137` -- User-facing documentation for chunk parameter

Key observations:

- Default is `chunks="auto"` which delegates to xarray/dask heuristics
- Users can pass a dict for explicit per-dimension chunks, `"auto"`, or `None`
  (load into memory)
- Existing docs show two access pattern examples (time-series vs spatial
  analysis) but don't explain _why_ those values were chosen
- No guidance on chunk alignment between write-time and read-time chunks

### 2. Cloud Object Store Performance Characteristics

Each Zarr chunk maps to one object in cloud storage (S3, GCS, OSN). Performance
is governed by:

**Latency per request:** ~50-100ms for S3/OSN HTTP GET requests. This makes many
small requests extremely expensive. A dataset with 10,000 small chunks = 10,000
HTTP round-trips = ~500-1000 seconds of pure latency.

**Throughput:** Single GET request throughput is ~50-100 MB/s for large objects.
S3 supports parallel GET requests, so total throughput scales with concurrency.

**Optimal chunk size:** Industry consensus from multiple sources:

- **10-100 MB compressed** per chunk is the sweet spot
- Below 1 MB: HTTP overhead dominates
- Above 500 MB: Wastes bandwidth when accessing subsets
- Zarr documentation recommends "at least several MB" for remote stores

**OVRO-LWA specific calculations:**

- A single SKY variable slice: `float32` x 1024 x 1024 = 4 MB uncompressed per
  (time, frequency, polarization) chunk
- With the default `chunk_lm=1024` on a 4096x4096 image: 16 spatial chunks per
  (t, f, pol)
- For a typical observation with 10 time steps x 20 frequencies x 1
  polarization: 3,200 chunks
- At ~4 MB each = ~12.5 GB total, which is reasonable but chunk access patterns
  matter

### 3. Chunk Sizing Best Practices

**The "Rule of Thumb" Framework:**

1. Start with 10-100 MB compressed chunks
2. Align chunks with dominant access patterns
3. Avoid chunks smaller than 1 MB on cloud stores
4. Keep the number of chunks per variable manageable (<10,000 for typical
   workloads)

**Access Pattern Alignment:**

- **Time-series analysis** (accessing all times at one spatial location): Chunk
  heavily in spatial, keep time contiguous
- **Spatial analysis** (accessing full images at one time/frequency): Chunk
  heavily in time/frequency, keep spatial contiguous
- **Spectral analysis** (accessing all frequencies at one location): Chunk
  heavily in spatial, keep frequency contiguous
- The default `chunk_lm=1024` favors spatial access (full or large image tiles)

**Compression Interaction:**

- Compression ratios of 2-10x are typical for astronomical data
- A 4 MB uncompressed chunk might compress to 0.5-2 MB -- potentially below the
  ideal range
- Recommended stack: **Blosc (with Zstd or LZ4) + optional Bitshuffle** for
  float32 array data
- The OVRO-LWA codebase currently has no explicit compression configuration

**Consolidated Metadata:**

- For remote stores, Zarr's `.zmetadata` (consolidated metadata) avoids N+1
  small requests for array metadata
- xarray's `xr.open_zarr()` uses `consolidated=True` by default
- The ingest pipeline does not explicitly write consolidated metadata (relies on
  xradio behavior)

### 4. Benchmarking Techniques

**Key Metrics:**

- Time to first byte (TTFB) -- measures initial access latency
- Total transfer time for a representative workload
- Number of HTTP requests per operation
- Effective throughput (data transferred / total wall time)
- Chunk utilization ratio (bytes used / bytes transferred)

**Benchmarking Approach:**

1. Define representative access patterns (time-series, spatial, spectral)
2. Measure baseline with current chunk configuration
3. Vary chunk sizes systematically (powers of 2)
4. Measure with and without compression
5. Compare local vs. cloud performance
6. Use `dask.diagnostics` or the Dask dashboard for profiling

**Tools:**

- `time.perf_counter()` for wall-clock timing
- `dask.distributed.performance_report()` for detailed Dask profiling
- `fsspec.implementations.http.HTTPFileSystem` access logs for request counting
- `zarr.storage.MemoryStore` vs `zarr.storage.FSStore` for isolating I/O from
  compute

**Published Benchmarks:**

- ESIP cloud computing guide: reports up to 63x improvement from chunk alignment
  on similar multidimensional geoscience data
- Zarr documentation: recommends benchmarking with `%timeit` on representative
  slicing operations

### 5. Zarr v2 Constraints and Configuration

The OVRO-LWA project is pinned to Zarr v2 (`zarr>=2.16,<3`). Key v2 constraints:

- No built-in sharding (each chunk = one file/object)
- Compression configured per-array via `encoding` dict or `compressor` parameter
- Consolidated metadata via `zarr.convenience.consolidate_metadata()`
- Chunk shape is immutable after array creation (cannot rechunk in-place)
- `numcodecs` provides the compression codec registry (project pins
  `numcodecs>=0.15,<0.16`)

**Configuration points in the codebase:**

- `xr.Dataset.to_zarr(encoding={...})` -- not currently used explicitly
- `zarr.open(compressor=...)` -- not used (writes go through xradio)
- `write_empty_chunks=False` -- not currently set (relevant for sparse data)

### 6. OVRO-LWA Data Model and Typical Dimensions

Understanding data shapes is critical for chunk sizing:

| Dimension    | Typical Range  | Notes                       |
| ------------ | -------------- | --------------------------- |
| time         | 1-100+ steps   | One step per FITS snapshot  |
| frequency    | 20-48 channels | 27-88 MHz, ~1.3 MHz spacing |
| polarization | 1-4            | Usually 1 (Stokes I only)   |
| l            | 512-4096       | Direction cosine, spatial   |
| m            | 512-4096       | Direction cosine, spatial   |

**Typical data sizes:**

- Single frame (1 time, 1 freq, 1 pol, 4096x4096): 64 MB (float32)
- Full observation (10 time, 48 freq, 1 pol, 4096x4096): ~30 GB
- Large campaign dataset: 100s of GB to TBs

### 7. Integration Points for Documentation

The documentation needs to bridge several integration seams:

1. **Ingest → Storage:** How `chunk_lm` at write time affects cloud read
   performance
2. **Storage → Read:** How `chunks=` parameter in `open_dataset()` interacts
   with on-disk chunks
3. **Read → Compute:** How Dask task graph is shaped by chunk boundaries
4. **xradio → Zarr:** What xradio's `write_image()` does with
   compression/encoding under the hood

## Architecture Overview

```
FITS Files                     Zarr Store (Cloud/Local)            User Code
-----------                    -----------------------            ----------

[*.fits] ──fix_headers──> [*_fixed.fits]
                               │
         ┌─────────────────────┘
         │
   _load_for_combine()         ┌─────────────────┐
         │                     │  .zmetadata      │   consolidated
   chunk(l=1024, m=1024)       │  .zattrs         │   metadata
         │                     │  SKY/            │
   write_image() ─────────────>│    .zarray       │   chunk shape,
                               │    .zattrs       │   compressor,
                               │    0.0.0.0.0     │   dtype
                               │    0.0.0.0.1     │
                               │    ...           │   one file per
                               │  BEAM/           │   chunk
                               │    ...           │
                               └────────┬────────┘
                                        │
                               open_dataset()
                               chunks="auto"
                                        │
                               xr.open_zarr()
                               ├── read .zmetadata
                               ├── create dask graph
                               └── lazy load chunks on .compute()
                                        │
                               ds.SKY.sel(...).compute()
                               ├── identify needed chunks
                               ├── parallel HTTP GETs
                               ├── decompress
                               └── assemble result
```

## Component Interactions

**Write Path Flow:**

1. `ingest/cli.py` parses `--chunk-lm` (default 1024) into `ConversionConfig`
2. `ingest/core.py:FITSToZarrConverter` calls `convert_fits_dir_to_zarr()`
3. `fits_to_zarr_xradio.py:_load_for_combine()` applies
   `xds.chunk({"l": chunk_lm, "m": chunk_lm})` at line 316
4. `fits_to_zarr_xradio.py:_write_or_append_zarr()` calls `write_image()` which
   writes to Zarr
5. Chunk boundaries and compression are determined by xradio/Zarr defaults at
   write time

**Read Path Flow:**

1. `io.py:open_dataset()` normalizes the source (DOI → URL → path)
2. For remote sources, creates fsspec mapper with `storage_options`
3. Calls `xr.open_zarr(mapper, chunks=chunks)` where `chunks` defaults to
   `"auto"`
4. xarray reads `.zmetadata` to get array shapes and chunk info
5. Creates dask task graph aligned with on-disk chunk boundaries
6. Data is fetched lazily: only when `.compute()`, `.values`, or `.plot()` is
   called
7. Each chunk fetch = one HTTP GET to cloud storage

**Critical Alignment Point:** Read-time `chunks` should be multiples of
write-time chunk shapes to avoid "partial chunk" reads where multiple on-disk
chunks must be fetched and stitched together.

## Code Examples

```python
# src/ovro_lwa_portal/fits_to_zarr_xradio.py:315-316
# Write-time chunking: only spatial dimensions, hardcoded to chunk_lm
if chunk_lm and {"l", "m"} <= set(xds.dims):
    xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})
```

```python
# src/ovro_lwa_portal/io.py (open_dataset default)
# Read-time chunking: delegates to xarray/dask "auto" heuristic
def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
```

```python
# Current docs example (docs/open_dataset.md:275-286)
# Shows two access patterns but doesn't explain the reasoning
# For time-series analysis:
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1000, "frequency": 10, "l": 256, "m": 256}
)
# For spatial analysis:
ds = ovro_lwa_portal.open_dataset(
    "data.zarr",
    chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}
)
```

## Technical Decisions

- **Decision:** Fixed `chunk_lm=1024` default for spatial dimensions only

  - **Rationale:** Balances spatial tile size with memory; 1024x1024 float32 = 4
    MB per tile
  - **Trade-offs:** Good for full-image spatial access; suboptimal for
    time-series at a single pixel (must read full 1024x1024 tile to get one
    pixel)

- **Decision:** No explicit compression configuration

  - **Rationale:** Relies on xradio `write_image()` defaults
  - **Trade-offs:** Simplifies ingest code but may leave performance on the
    table; users cannot currently control compression without modifying code

- **Decision:** Zarr v2 pinned (no sharding)

  - **Rationale:** Required by xradio compatibility (`xradio/issues/355`)
  - **Trade-offs:** Each chunk = one object; no sub-chunk access. Must balance
    chunk count vs. chunk size without sharding.

- **Decision:** `chunks="auto"` default on read
  - **Rationale:** Delegates to xarray/dask heuristics which consider available
    memory and on-disk chunk shape
  - **Trade-offs:** Good general default, but "auto" doesn't account for cloud
    latency or access pattern specifics

## Dependencies and Integrations

- **zarr v2 (2.16-2.x):** On-disk chunk storage, metadata format, compression
  integration
- **numcodecs (0.15.x):** Compression codec registry (Zstd, LZ4, Blosc, etc.)
- **xarray (2025.x):** Lazy loading via `open_zarr()`, chunk parameter handling,
  dask integration
- **dask (2025.x):** Task graph construction from chunk boundaries, parallel
  execution
- **xradio (0.0.59+):** `read_image()`/`write_image()` for FITS I/O and Zarr
  writing
- **fsspec/s3fs/gcsfs:** Cloud filesystem abstraction, HTTP request handling
- **universal-pathlib:** Unified path handling for local/remote stores

## Edge Cases and Constraints

- Clearing variable encodings (`xds[v].encoding = {}`) at write time may discard
  compression settings that xradio applied during `read_image()`
- When appending time steps, the entire existing Zarr is read and re-written
  (`_write_or_append_zarr`), making chunk size changes between appends
  potentially inconsistent
- `chunk_lm=0` disables spatial chunking entirely, which could create very large
  single chunks
- OSN endpoints have different latency characteristics than AWS S3 proper
- `write_empty_chunks` is not explicitly set; for sparse data (many NaN/zero
  frames), this could waste storage and bandwidth

## Open Questions

1. What compression does xradio's `write_image()` actually apply by default?
   (Needs empirical check on a real Zarr store)
2. What is the actual compressed chunk size for typical OVRO-LWA data? (Need to
   inspect `.zarray` metadata)
3. Should the ingest pipeline support user-configurable compression codecs?
4. What are the actual latency and throughput characteristics of the Caltech OSN
   endpoint vs. other S3 providers?
5. Should rechunking utilities (e.g., `rechunker`) be documented for users who
   need different chunk layouts for different access patterns?
6. How does xarray's `chunks="auto"` heuristic perform specifically with
   OVRO-LWA data on cloud stores?

## References

- Files analyzed: 8 total files

  - `src/ovro_lwa_portal/io.py` -- open_dataset(), resolve_source(),
    DataSourceError
  - `src/ovro_lwa_portal/fits_to_zarr_xradio.py` -- FITS-to-Zarr ingest with
    chunk_lm
  - `src/ovro_lwa_portal/ingest/core.py` -- ConversionConfig,
    FITSToZarrConverter
  - `src/ovro_lwa_portal/ingest/cli.py` -- CLI --chunk-lm parameter
  - `src/ovro_lwa_portal/accessor.py` -- radport xarray accessor
  - `src/ovro_lwa_portal/__init__.py` -- Package exports
  - `docs/open_dataset.md` -- Existing user documentation (PR #95)
  - `pyproject.toml` -- Dependencies and build configuration

- External references:
  - [Zarr Performance User Guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
    -- Official Zarr documentation on chunk sizing, compression codecs, and
    performance tuning
  - [ESIP Cloud Computing Optimization Resources](https://esipfed.github.io/cloud-computing-cluster/resources-for-optimization.html)
    -- Cloud-optimized data access checklist, chunk alignment strategies
  - [HackMD Zarr Chunking Guide](https://hackmd.io/@brivadeneira/rkqm_XYHgg) --
    Practical chunking examples, Dask integration, rechunking workflows
  - [MarkTechPost: Zarr for Large-Scale Data](https://www.marktechpost.com/2025/09/16/a-coding-guide-to-implement-zarr-for-large-scale-data-chunking-compression-indexing-and-visualization-techniques/)
    -- End-to-end coding guide for Zarr chunking, compression, and visualization
  - [Xarray Intermediate: Intro to Zarr](https://tutorial.xarray.dev/intermediate/intro-to-zarr.html)
    -- Xarray+Zarr integration tutorial, cloud-native data access patterns
  - [CloudFront Preprint on Cloud-Optimized Zarr](https://d197for5662m48.cloudfront.net/documents/publicationstatus/129928/preprint_pdf/aa1ef041a20d28b38262dc6bca5ed109.pdf)
    -- Academic paper on cloud-native chunking strategies, reports 63x
    improvement from chunk alignment

---

## Proposed GitHub Issue Set

The following issues are designed so that each one can be independently assigned
to an AI agent to produce a complete, standalone documentation page. Together
they form a comprehensive documentation suite on Zarr chunk optimization for the
OVRO-LWA portal.

### Issue 1: "Docs: Zarr Chunking Fundamentals for Radio Astronomy Data"

**Scope:** Foundational concepts document explaining what Zarr chunks are, how
they map to cloud objects, and why chunk size matters for OVRO-LWA data.

**Content outline:**

- What is a Zarr chunk? (visual diagram of array → chunks → files/objects)
- How chunks map to cloud object store objects (one chunk = one HTTP GET)
- The latency-throughput tradeoff (small chunks = many requests; large chunks =
  wasted bandwidth)
- OVRO-LWA data model overview (5D: time x frequency x polarization x l x m)
- How float32 dtype and dimension sizes determine uncompressed chunk sizes
- The "10-100 MB compressed" sweet spot and why
- Glossary: chunk shape, chunk size, compression ratio, consolidated metadata

**Key references:** Zarr docs performance guide, xarray Zarr tutorial, ESIP
resources **Target audience:** Users new to Zarr or cloud-native data

---

### Issue 2: "Docs: Understanding the OVRO-LWA Chunking Pipeline (Write Path)"

**Scope:** Document how the OVRO-LWA ingest pipeline writes chunks, what
defaults are used, and what configuration options exist.

**Content outline:**

- The FITS-to-Zarr ingest flow: `fix_fits_headers()` → `_load_for_combine()` →
  `write_image()` → Zarr store
- The `chunk_lm` parameter: what it controls, default value (1024), how to
  change it via CLI (`--chunk-lm`) or API
- What happens with dimensions NOT explicitly chunked (time, frequency,
  polarization)
- What xradio's `write_image()` does under the hood with encoding and
  compression
- The encoding clearing pattern (`xds[v].encoding = {}`) and its implications
- How append mode (`_write_or_append_zarr`) interacts with chunk boundaries
- Inspecting chunk metadata: reading `.zarray` to verify chunk shape and
  compressor

**Key code references:**

- `fits_to_zarr_xradio.py:315-316` (chunk application)
- `fits_to_zarr_xradio.py:498-598` (convert function)
- `ingest/core.py:43-58` (ConversionConfig)
- `ingest/cli.py` (CLI interface)

**Target audience:** Data engineers, ingest pipeline operators

---

### Issue 3: "Docs: Optimizing Read-Time Chunk Configuration"

**Scope:** Guide users on choosing the right `chunks=` parameter in
`open_dataset()` for their specific access pattern.

**Content outline:**

- How `open_dataset(chunks=...)` works: `"auto"`, explicit dict, or `None`
- How xarray/dask "auto" heuristic works (considers available memory and on-disk
  chunk shape)
- **Access pattern recipes** with worked examples and rationale:
  - Time-series at a single pixel:
    `chunks={"time": -1, "frequency": 1, "l": 256, "m": 256}`
  - Full spatial images:
    `chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}`
  - Spectral analysis: `chunks={"time": 1, "frequency": -1, "l": 256, "m": 256}`
  - Broadband averaging:
    `chunks={"time": 1, "frequency": -1, "l": 512, "m": 512}`
- **Chunk alignment:** Why read-time chunks should be multiples of write-time
  chunks
- Impact of misaligned chunks (partial reads, unnecessary data transfer)
- When to use `chunks=None` (small datasets, one-shot analysis)
- Practical decision flowchart for choosing chunk configuration

**Key code references:**

- `io.py` open_dataset() function signature
- `docs/open_dataset.md:93-137` (existing parameter docs)

**Target audience:** Researchers and data analysts

---

### Issue 4: "Docs: Compression Strategies for Cloud-Stored Zarr Data"

**Scope:** Document compression options, how they interact with chunking, and
recommendations for OVRO-LWA data.

**Content outline:**

- Why compression matters for cloud storage (reduces transfer size, but adds CPU
  time)
- Available codecs in numcodecs: Zstd, LZ4, Blosc (with shuffle modes), GZip
- Recommended stack for float32 astronomy data: **Blosc(cname='zstd', clevel=3,
  shuffle=Blosc.BITSHUFFLE)**
- Compression ratio expectations for OVRO-LWA data (empirical measurements if
  possible)
- How compression interacts with chunk size (compressed chunk size is what
  matters for cloud)
- Configuring compression at write time via xarray encoding:
  ```python
  encoding = {"SKY": {"compressor": Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)}}
  ds.to_zarr("store.zarr", encoding=encoding)
  ```
- Current state: what the ingest pipeline uses by default
- `write_empty_chunks=False` for sparse OVRO-LWA data
- Filters vs. compressors: when to use Delta coding or other filters

**Key references:** Zarr performance docs, numcodecs documentation, HackMD guide
**Target audience:** Data engineers, advanced users

---

### Issue 5: "Docs: Benchmarking Zarr Read Performance"

**Scope:** Provide a practical benchmarking guide with scripts and methodology
for measuring chunk performance.

**Content outline:**

- Why benchmarking is necessary (no universal optimal chunk size)
- **Benchmarking methodology:**
  1. Define representative access patterns
  2. Measure baseline configuration
  3. Vary one parameter at a time (chunk shape, compression, read-time chunks)
  4. Measure metrics: wall time, HTTP requests, data transferred, effective
     throughput
  5. Statistical rigor: multiple runs, warm cache vs. cold cache
- **Benchmarking script template** for OVRO-LWA:

  ```python
  import time
  import ovro_lwa_portal

  # Cold-cache spatial access benchmark
  def benchmark_spatial_access(source, chunks, n_runs=5):
      times = []
      for _ in range(n_runs):
          ds = ovro_lwa_portal.open_dataset(source, chunks=chunks)
          t0 = time.perf_counter()
          _ = ds.SKY.isel(time=0, frequency=0).compute()
          times.append(time.perf_counter() - t0)
      return {"mean": np.mean(times), "std": np.std(times)}
  ```

- Using Dask diagnostics: `performance_report()`, task stream, worker memory
- Interpreting results: what "good" looks like for cloud reads
- Common pitfalls: caching effects, network variability, not testing at scale
- Comparison matrix template: chunk config vs. access pattern vs. latency

**Target audience:** Performance-conscious users, data engineers

---

### Issue 6: "Docs: Cloud Storage Configuration and Optimization"

**Scope:** Document cloud-specific considerations for optimal Zarr access from
S3, GCS, and OSN endpoints.

**Content outline:**

- How fsspec/s3fs/gcsfs handle Zarr chunk requests (one GET per chunk object)
- **Consolidated metadata:** what it is, why it matters (avoids N small
  requests), how to verify it exists
  - `zarr.convenience.consolidate_metadata()` for existing stores
  - How `xr.open_zarr(consolidated=True)` works (the default)
- **Concurrent access:** How Dask parallelizes chunk fetches, configuring worker
  count
- **Caching strategies:**
  - fsspec `simplecache` for frequently accessed remote data
  - `filecache` for persistent local mirrors
  - When to just download the whole dataset locally
- **OSN-specific considerations:**
  - Endpoint URLs and region settings
  - HTTPS vs. S3 protocol access (performance differences)
  - Authentication setup (storage_options)
- **Request tuning:** s3fs retry configuration, timeout settings,
  max_concurrency
- **Cost considerations:** egress costs, request pricing (GET vs. LIST)

**Key code references:**

- `io.py` -- fsspec mapper construction, storage_options handling
- `open_dataset.md:150-182` -- existing cloud storage docs

**Target audience:** Users accessing data from cloud stores

---

### Issue 7: "Docs: Chunk Optimization Decision Guide and Reference Card"

**Scope:** Quick-reference decision guide that synthesizes all other
documentation into actionable recommendations.

**Content outline:**

- **Decision flowchart:** "What chunks should I use?" based on:
  - Data size (< 1 GB: just use `chunks=None`; > 1 GB: use the guide)
  - Access location (local vs. cloud)
  - Primary access pattern (spatial, temporal, spectral, mixed)
  - Available memory
- **Quick reference table:**

  | Access Pattern     | Write `chunk_lm` | Read `chunks=`                                      | Rationale                                           |
  | ------------------ | ---------------- | --------------------------------------------------- | --------------------------------------------------- |
  | Full images, cloud | 1024             | `{"time": 1, "frequency": 1, "l": 1024, "m": 1024}` | One chunk per spatial tile, minimal requests        |
  | Time series, cloud | 512              | `{"time": -1, "frequency": 1, "l": 512, "m": 512}`  | Smaller spatial tiles reduce waste for point access |
  | Spectral, cloud    | 1024             | `{"time": 1, "frequency": -1, "l": 256, "m": 256}`  | All frequencies per tile                            |
  | Local analysis     | 1024             | `"auto"`                                            | Local I/O is fast; auto works well                  |

- **Anti-patterns:** Common mistakes and how to avoid them
  - Using `chunks=None` on large remote datasets
  - Using very small chunks (< 1 MB) on cloud stores
  - Misaligned read chunks (e.g., reading 512x512 from 1024x1024 on-disk chunks)
  - Forgetting consolidated metadata for remote stores
- **Troubleshooting FAQ:**
  - "My cloud reads are slow" -- diagnostic steps
  - "I'm running out of memory" -- chunk size reduction guide
  - "Dask is creating too many tasks" -- chunk consolidation
- Links to all other documentation pages in the series

**Target audience:** All users (quick reference)

---

### Issue Dependencies and Ordering

```
Issue 1 (Fundamentals) ──────────────> Issue 3 (Read Optimization)
       │                                       │
       └──> Issue 2 (Write Pipeline) ──────────┤
                                               │
Issue 4 (Compression) ────────────────────────>│
                                               │
Issue 5 (Benchmarking) ───────────────────────>│
                                               │
Issue 6 (Cloud Configuration) ────────────────>│
                                               │
                                               └──> Issue 7 (Decision Guide)
```

Issues 1-6 can be worked in parallel (they are self-contained). Issue 7 should
be last as it synthesizes all others.
