# Benchmarking Zarr Read Performance

Optimizing chunk configurations requires measuring actual performance on
representative data and access patterns. Published recommendations provide
starting points, but the optimal configuration depends on your specific data
characteristics, query patterns, network infrastructure, and storage backend.

This guide provides methodology, scripts, and interpretation guidance for
benchmarking OVRO-LWA Zarr datasets on both local storage and cloud object
stores like OSN.

## Why Benchmarking is Necessary

There is no universal optimal chunk size. The best chunking strategy depends on:

**Data characteristics:** Array dimensions, data types, sparsity, and
compressibility vary between observations. A configuration optimized for
4096×4096 full-field images may perform poorly on smaller cutouts or different
spatial resolutions.

**Access patterns:** Workflows that extract time series at fixed sky positions
require different chunking than workflows that generate full spatial maps. Chunk
configurations optimized for one access pattern often degrade performance for
others.

**Infrastructure:** Network bandwidth, latency, and concurrency limits differ
dramatically between local filesystems, institutional clusters, cloud providers,
and regions. The Open Storage Network (OSN) endpoint used by OVRO-LWA has
different latency characteristics than AWS S3 proper, including different
geographic routing and CDN caching behavior.

**Temporal variability:** Cloud storage performance varies by time of day,
regional congestion, and instance load. A configuration that performs well
during off-peak hours may show different behavior under load.

Published recommendations like the 10-100 MB compressed chunk size target from
[Chunking Fundamentals](chunking-fundamentals.md) provide good starting points,
but they represent industry averages rather than guarantees for specific
deployments. Benchmarking verifies that a chosen configuration meets performance
requirements for your workflow.

## Key Metrics to Measure

### Wall-Clock Time

The total elapsed time for a representative operation, measured from start to
result availability. This is the most user-relevant metric: how long does my
analysis take?

**How to measure:**

```python
import time

t0 = time.perf_counter()
result = ds.SKY.isel(time=0, frequency=0).compute()
elapsed = time.perf_counter() - t0
print(f"Wall-clock time: {elapsed:.3f} seconds")
```

Use `time.perf_counter()` rather than `time.time()` for higher resolution and
monotonic behavior (not affected by system clock adjustments).

### Time to First Byte (TTFB)

The latency before data transfer begins, measuring network round-trip time and
storage system response. TTFB dominates performance when fetching many small
chunks.

**How to measure:**

For HTTP-based storage, TTFB appears in request logs. Enable fsspec logging to
capture per-request timing:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("fsspec")
logger.setLevel(logging.DEBUG)

# Now HTTP requests will be logged with timing
ds = ovro_lwa_portal.open_dataset("s3://bucket/obs.zarr")
result = ds.SKY.isel(time=0, frequency=0).compute()
```

Look for log entries showing request timing. TTFB typically appears as the time
between request initiation and first response.

### Number of HTTP Requests

The count of individual HTTP GET requests made to cloud storage. Each request
incurs ~50-100ms latency overhead regardless of payload size, making request
count a critical metric for cloud performance.

**How to measure:**

Enable fsspec request logging as shown above, then count GET requests in the
logs. Alternatively, inspect the Dask task graph:

```python
# Estimate HTTP requests from Dask chunk count
n_chunks = ds.SKY.isel(time=0, frequency=0).data.npartitions
print(f"Chunks to fetch: {n_chunks}")
```

Each chunk typically corresponds to one HTTP request (assuming no caching).

### Effective Throughput

The ratio of data transferred to wall-clock time, measured in MB/s. Effective
throughput accounts for both network bandwidth and latency overhead.

**How to measure:**

```python
import time

# Estimate data volume
n_elements = ds.SKY.isel(time=0, frequency=0).size
data_mb = (n_elements * 4) / (1024**2)  # float32 = 4 bytes

t0 = time.perf_counter()
result = ds.SKY.isel(time=0, frequency=0).compute()
elapsed = time.perf_counter() - t0

throughput = data_mb / elapsed
print(f"Effective throughput: {throughput:.1f} MB/s")
```

This measures uncompressed data volume divided by time. For compressed Zarr
stores, multiply the chunk count by the actual compressed chunk size (inspect
with `.zarray` metadata) for a more accurate transfer volume estimate.

### Chunk Utilization Ratio

The fraction of transferred bytes that are actually used by the operation. Low
utilization indicates wasted bandwidth from misaligned chunks.

**How to measure:**

```python
# Bytes actually needed for the operation
region = ds.SKY.isel(time=0, frequency=0, l=slice(0, 512), m=slice(0, 512))
used_mb = (region.size * 4) / (1024**2)

# Bytes transferred (depends on chunk boundaries)
# With chunk_lm=1024, a 512×512 region falls within one 1024×1024 chunk
transferred_mb = (1024 * 1024 * 4) / (1024**2)  # 4 MB uncompressed

utilization = used_mb / transferred_mb
print(f"Chunk utilization: {utilization:.1%}")
```

For cloud reads, poor utilization (< 50%) suggests chunk size is too large for
the access pattern.

### Dask Task Count

The number of tasks in the computation graph. Excessive tasks create scheduling
overhead; too few tasks limit parallelism.

**How to measure:**

```python
task_count = len(ds.SKY.isel(time=0, frequency=0).__dask_graph__())
print(f"Dask tasks: {task_count}")
```

As a rule of thumb, aim for 100-10,000 tasks per operation. Fewer than 100
limits parallelism; more than 10,000 creates excessive scheduling overhead.

## Benchmarking Methodology

Follow this systematic approach to isolate performance factors and ensure
reproducible results:

1. **Define 3–5 representative access patterns** matching real user workflows.
   Use the recipes from [Read-Time Optimization](chunking-read-optimization.md)
   as templates:

   - Recipe A: Full spatial images (snapshot analysis)
   - Recipe B: Time series at a point
   - Recipe C: Spectral analysis
   - Recipe D: Full dataset batch processing

2. **Create a controlled environment** to minimize external variability:

   - Consistent network: Use the same network connection for all runs
   - No background jobs: Close applications that may compete for bandwidth or
     CPU
   - Stable instance: If benchmarking on cloud VMs, use dedicated instances
     rather than shared/spot instances

3. **Distinguish warm cache vs. cold cache measurements:**

   - **Warm cache:** Data is already in OS page cache or fsspec cache. Measures
     compute and memory performance without I/O latency. To warm cache: run the
     operation once, then measure subsequent runs.
   - **Cold cache:** Data must be fetched from storage. Measures full I/O +
     compute performance. To clear cache:
     - **macOS:** `sudo purge`
     - **Linux:** `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
     - **fsspec cache:** Delete `~/.cache/fsspec/` or use
       `fsspec.config.set(cache_storage=None)`

   Label all results as "warm" or "cold" cache. Cold cache measurements reflect
   production behavior for first-time access; warm cache measurements reflect
   repeated access patterns.

4. **Run minimum 5 iterations per configuration** for statistical reliability.
   Compute mean, standard deviation, min, and max to characterize variability.
   Outliers (> 2× median) may indicate network issues or background
   interference.

5. **Vary one parameter at a time:** Test chunk shape OR compression OR
   read-time chunks in isolation. Changing multiple parameters simultaneously
   makes it impossible to attribute performance changes to specific factors.

6. **Record environment metadata:**
   - Python version: `python --version`
   - Dependency versions: `pip list | grep -E '(xarray|dask|zarr)'`
   - Network type: wired, WiFi, cloud instance
   - Instance type: local workstation, AWS m5.xlarge, etc.
   - Storage endpoint: OSN region, S3 bucket region

!!! note "Clearing OS Page Cache Requires Root"

    The Linux and macOS commands to clear OS page cache require root/sudo
    access. If running on shared infrastructure without root, use fsspec cache
    clearing instead, or note that measurements include some cache warming
    effects. For most cloud benchmarking scenarios, fsspec cache clearing is
    sufficient since cloud data doesn't persist in OS cache between runs.

## Benchmarking Script Template

This script benchmarks three access patterns against OVRO-LWA data, comparing
multiple chunk configurations and outputting a results table.

```python
"""Benchmark OVRO-LWA Zarr chunk configurations.

Usage:
    python benchmark_chunks.py /path/to/obs.zarr
    python benchmark_chunks.py s3://bucket/obs.zarr --cloud
"""

import argparse
import time
from pathlib import Path

import numpy as np
import ovro_lwa_portal


def benchmark(name, fn, n_runs=5):
    """Run a benchmark function multiple times and report statistics.

    Parameters
    ----------
    name : str
        Name of the benchmark for reporting
    fn : callable
        Function to benchmark (should call .compute() internally)
    n_runs : int
        Number of iterations to run

    Returns
    -------
    times : np.ndarray
        Array of wall-clock times for each run
    """
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {i+1}/{n_runs}: {elapsed:.3f}s")

    times = np.array(times)
    print(
        f"{name}: mean={times.mean():.3f}s std={times.std():.3f}s "
        f"min={times.min():.3f}s max={times.max():.3f}s"
    )
    return times


def benchmark_spatial(source, chunks, n_runs=5):
    """Benchmark full spatial image access (Recipe A)."""
    def load_spatial():
        ds = ovro_lwa_portal.open_dataset(source, chunks=chunks)
        _ = ds.SKY.isel(time=0, frequency=0).compute()
    return benchmark("Spatial access (full image)", load_spatial, n_runs)


def benchmark_timeseries(source, chunks, n_runs=5):
    """Benchmark time series extraction at one location (Recipe B)."""
    def load_timeseries():
        ds = ovro_lwa_portal.open_dataset(source, chunks=chunks)
        # Extract a 256×256 region across all times and frequencies
        _ = ds.SKY.isel(l=slice(0, 256), m=slice(0, 256)).compute()
    return benchmark("Time series (256×256 region)", load_timeseries, n_runs)


def benchmark_spectral(source, chunks, n_runs=5):
    """Benchmark spectral analysis across all frequencies (Recipe C)."""
    def load_spectral():
        ds = ovro_lwa_portal.open_dataset(source, chunks=chunks)
        # Extract a 512×512 region at one time, all frequencies
        region = ds.SKY.isel(time=0, l=slice(0, 512), m=slice(0, 512))
        _ = region.mean(dim="frequency").compute()
    return benchmark("Spectral (512×512, all freq)", load_spectral, n_runs)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OVRO-LWA Zarr chunk configurations"
    )
    parser.add_argument("source", help="Path or URL to Zarr store")
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Source is cloud storage (enables special handling)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per test"
    )
    args = parser.parse_args()

    # Define configurations to test
    configs = {
        "Default (auto)": "auto",
        "Spatial-optimized": {
            "time": 1,
            "frequency": 1,
            "l": 1024,
            "m": 1024,
        },
        "Time-optimized": {"time": -1, "frequency": 1, "l": 256, "m": 256},
        "Spectral-optimized": {
            "time": 1,
            "frequency": -1,
            "l": 256,
            "m": 256,
        },
    }

    print(f"Benchmarking: {args.source}")
    print(f"Runs per configuration: {args.runs}")
    print("=" * 60)

    results = {}
    for config_name, chunks in configs.items():
        print(f"\nConfiguration: {config_name}")
        print(f"  chunks={chunks}")

        spatial_times = benchmark_spatial(args.source, chunks, args.runs)
        timeseries_times = benchmark_timeseries(args.source, chunks, args.runs)
        spectral_times = benchmark_spectral(args.source, chunks, args.runs)

        results[config_name] = {
            "spatial": spatial_times.mean(),
            "timeseries": timeseries_times.mean(),
            "spectral": spectral_times.mean(),
        }

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY (mean times in seconds)")
    print("=" * 60)
    print(
        f"{'Configuration':<25} {'Spatial':>10} {'Time-series':>12} "
        f"{'Spectral':>10}"
    )
    print("-" * 60)
    for config_name, times in results.items():
        print(
            f"{config_name:<25} {times['spatial']:>10.3f} "
            f"{times['timeseries']:>12.3f} {times['spectral']:>10.3f}"
        )


if __name__ == "__main__":
    main()
```

**Example usage:**

```bash
# Local storage
python benchmark_chunks.py /data/ovro/obs.zarr

# Cloud storage (OSN or S3)
python benchmark_chunks.py s3://ovro-lwa/obs.zarr --cloud --runs 3
```

**Example output:**

```text
Benchmarking: s3://ovro-lwa/obs.zarr
Runs per configuration: 5
============================================================

Configuration: Default (auto)
  chunks=auto
  Run 1/5: 12.456s
  Run 2/5: 11.892s
  Run 3/5: 12.103s
  Run 4/5: 12.234s
  Run 5/5: 11.987s
Spatial access (full image): mean=12.134s std=0.213s min=11.892s max=12.456s
...

============================================================
SUMMARY (mean times in seconds)
============================================================
Configuration              Spatial  Time-series   Spectral
------------------------------------------------------------
Default (auto)              12.134       18.456      8.234
Spatial-optimized            8.567       22.103      9.012
Time-optimized              15.234       14.567     10.123
Spectral-optimized          13.456       19.234      6.789
```

## Using Dask Diagnostics

Dask provides multiple diagnostic tools for profiling computation and I/O
patterns. These tools help identify whether performance is limited by network
I/O, CPU computation, or Dask scheduling overhead.

### Performance Reports

For detailed profiling, use `dask.distributed.performance_report()` to generate
an HTML report showing task execution, memory usage, and worker communication:

```python
from dask.distributed import Client, performance_report

# Start a local cluster
client = Client()

# Profile a computation
with performance_report(filename="profile.html"):
    ds = ovro_lwa_portal.open_dataset(
        "s3://ovro-lwa/obs.zarr",
        chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024},
    )
    result = ds.SKY.isel(time=0, frequency=0).compute()

print("Profile saved to profile.html")
```

Open `profile.html` in a browser to view:

- **Task stream:** Timeline showing when each task executed and on which worker
- **Memory:** Memory allocation over time, identifying memory spikes or leaks
- **CPU:** CPU utilization per worker
- **Network:** Bytes transferred between workers and from storage

**What to look for:**

- Long gaps between task execution → network I/O is the bottleneck
- High CPU usage → computation is the bottleneck
- Many small tasks executing serially → Dask scheduling overhead
- Memory spikes → chunks may be too large for available RAM

### Dask Dashboard

For real-time monitoring, the Dask distributed scheduler provides a web
dashboard at `http://localhost:8787/status` by default:

```python
from dask.distributed import Client

client = Client()
print(f"Dashboard: {client.dashboard_link}")

# Work proceeds, monitor in real-time via browser
ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa/obs.zarr")
result = ds.SKY.mean(dim=["l", "m"]).compute()
```

The dashboard updates in real-time, showing:

- **Progress:** Active computations and task completion percentage
- **Graph:** Visualization of task dependencies
- **Workers:** Resource usage per worker (CPU, memory, network)
- **Task Stream:** Real-time task execution timeline

**Interpreting the task stream:**

- **Red bars:** Tasks waiting for data (I/O bound)
- **Green bars:** Tasks executing computation (CPU bound)
- **Gray bars:** Tasks in Dask scheduler queue (scheduling overhead)

If task stream shows mostly red bars, you're I/O bound — optimize chunking or
network. If mostly green, you're CPU bound — optimize computation or add
workers.

### Progress Bars

!!! tip "Simple Progress Tracking in Notebooks"

    For Jupyter notebook users who don't need full distributed diagnostics,
    `dask.diagnostics.ProgressBar` provides a simple text progress indicator:
    ```python
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        result = ds.SKY.isel(time=0).compute()
    ```
    This shows completion percentage and estimated time remaining. See
    [Read-Time Optimization section 6](chunking-read-optimization.md#monitoring-dask-scheduler-activity)
    for more details on using the ProgressBar in analysis workflows.

## Comparing Configurations

Use this template to organize benchmark results. Fill in mean times for each
configuration and access pattern:

| Configuration      | Spatial (s) | Time-series (s) | Spectral (s) | HTTP Requests |
| ------------------ | ----------- | --------------- | ------------ | ------------- |
| Default (auto)     |             |                 |              |               |
| Spatial-optimized  |             |                 |              |               |
| Time-optimized     |             |                 |              |               |
| Spectral-optimized |             |                 |              |               |

**How to fill it in:**

1. Run the benchmark script with each configuration
2. Record mean times in the corresponding columns
3. Count HTTP requests from Dask chunk count or fsspec logs
4. Look for patterns:
   - Does one configuration outperform others for all patterns? (rare)
   - Does each configuration excel at its intended pattern?
   - Are there configurations that perform poorly across the board?

**Interpreting patterns:**

- **Diagonal dominance:** Spatial-optimized is fastest for spatial,
  time-optimized for time-series, etc. This is expected and validates that
  chunking strategies work as intended.
- **One configuration wins everywhere:** Either data is small enough that
  chunking doesn't matter, or one configuration happens to align well with all
  patterns. Verify chunk alignment with on-disk layout.
- **All configurations perform similarly:** Likely bottlenecked by network
  bandwidth or non-I/O factors (computation, decompression). Chunking
  optimization may have limited impact.
- **High variability (std > 50% of mean):** Network instability or background
  interference. Repeat benchmarks at a different time or with more runs.

## Interpreting Results

### What "Good" Looks Like

**Effective throughput targets:**

Typical industry starting points for effective throughput are:

- **Local SSD storage:** Tens to hundreds of MB/s depending on hardware (modern
  SATA SSDs ~100-500 MB/s, NVMe drives 500+ MB/s)
- **Cloud object storage (OSN, S3):** Single-digit to tens of MB/s for
  sequential chunk reads, higher with parallelism
- **Networked filesystems:** Tens to hundreds of MB/s depending on protocol and
  infrastructure (NFS, Lustre, GPFS)

These are rough orientation points — your actual targets depend on your
workflow's latency tolerance and data volume. Throughput significantly below
typical values for your storage type suggests I/O bottlenecks from misaligned
chunks, excessive HTTP requests, or network limitations.

**HTTP request counts:**

For cloud access, request count drives latency overhead. Target < 100 requests
for interactive operations (< 10 seconds total), < 1000 requests for batch
operations (< 60 seconds total). Each request adds ~50-100ms latency.

**Task count:**

Aim for 100-10,000 tasks per operation. Fewer than 100 limits parallelism on
multi-core systems; more than 10,000 creates excessive scheduling overhead in
the Dask scheduler.

### When to Stop Optimizing

**Diminishing returns:** If a configuration achieves 2× improvement over
baseline, further optimization typically yields < 20% additional gain. The
cost-benefit of continued tuning rarely justifies the effort beyond 2×
improvement.

**Good enough for workflow:** If operations complete within acceptable time for
your workflow (e.g., < 10 seconds for interactive plotting, < 5 minutes for
batch jobs), optimization is complete. Perfect is the enemy of good.

**Bottleneck shifted:** If you've optimized chunking but performance is still
poor, the bottleneck may have shifted to:

- Compression/decompression CPU time
- Network bandwidth saturation
- Worker memory constraints
- Non-I/O computation (algorithms, array operations)

Profile with Dask diagnostics to identify the new bottleneck.

### Translating Benchmark Results to Production

After benchmarking, choose a configuration based on:

1. **Primary access pattern:** If 80%+ of queries follow one pattern, optimize
   for that pattern using the corresponding recipe from
   [Read-Time Optimization](chunking-read-optimization.md)
2. **Mixed patterns:** If no single pattern dominates, choose the configuration
   with the lowest average time across all benchmarked patterns
3. **Chunk alignment:** Verify the chosen configuration aligns with on-disk
   chunk layout to avoid partial chunk reads (see
   [Chunking Fundamentals](chunking-fundamentals.md) for alignment principles)

!!! note "ESIP's 63× Improvement Upper Bound"

    The ESIP Cloud Optimization guide reports up to 63× improvement from chunk
    alignment on similar multidimensional geoscience data. This represents an
    upper bound from highly misaligned baseline configurations (e.g., reading
    1D slices from arrays chunked orthogonally). Typical improvements for
    already-reasonable chunk configurations are more modest, in the 2-5×
    range. Use 63× as motivation to benchmark rather than an expected outcome.

## Common Pitfalls

### OS Page Cache Masking I/O Time

**Symptom:** First run is slow (10+ seconds), subsequent runs are fast (< 1
second).

**Cause:** Operating system caches file reads in RAM. The second run reads from
cache rather than disk/network, giving artificially fast times.

**Solution:** Clear OS cache between runs (requires root):

```bash
# macOS
sudo purge

# Linux
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

Or explicitly note results as "warm cache" measurements.

### fsspec Caching Between Runs

**Symptom:** Cloud benchmarks show fast times even with cold OS cache.

**Cause:** fsspec HTTP cache stores fetched chunks in `~/.cache/fsspec/`,
persisting across runs.

**Solution:** Clear fsspec cache before benchmarking:

```bash
rm -rf ~/.cache/fsspec/
```

Or disable caching programmatically:

```python
import fsspec

fsspec.config.set(cache_storage=None)

# Now cloud reads bypass cache
ds = ovro_lwa_portal.open_dataset("s3://bucket/obs.zarr")
```

### Network Variability on Shared Cloud Instances

**Symptom:** Benchmark results vary by > 2× between runs.

**Cause:** Cloud instances share network bandwidth with other tenants.
Congestion varies unpredictably.

**Solution:**

- Run benchmarks at consistent times (e.g., always off-peak hours)
- Use dedicated instances rather than shared/spot instances
- Record network conditions (run `ping` or `traceroute` to storage endpoint)
- Increase `n_runs` to 10+ to smooth out variability

### Not Testing with Realistic Data Sizes

**Symptom:** Small test datasets show no performance difference between
configurations.

**Cause:** Small data fits entirely in memory regardless of chunking. Chunking
optimization only matters when data exceeds available RAM.

**Solution:** Benchmark with production-scale data (10+ GB uncompressed) on
representative datasets. For OVRO-LWA, use observations with 10+ time steps, 48
frequency channels, and 4096×4096 spatial resolution.

### Comparing Warm-Cache and Cold-Cache Results

**Symptom:** Conflicting conclusions about which configuration is best.

**Cause:** Warm cache measurements reflect compute performance; cold cache
reflects I/O performance. Configurations may rank differently.

**Solution:** Always label results as "warm" or "cold" cache. Report both
measurements separately. For production planning, cold-cache results reflect
first-time access behavior; warm-cache results reflect repeated access to the
same data.

## External References

- [ESIP Cloud Computing Optimization](https://esipfed.github.io/cloud-computing-cluster/resources-for-optimization.html) -
  Cloud-optimized data formats and benchmarking best practices (reports up to
  63× improvement from chunk alignment on geoscience data)
- [Zarr Performance User Guide](https://zarr.readthedocs.io/en/stable/user-guide/performance.html) -
  Official Zarr performance tuning recommendations

## See Also

- [Chunking Fundamentals](chunking-fundamentals.md) - Chunk size sweet spot and
  cloud storage performance characteristics
- [Read-Time Optimization](chunking-read-optimization.md) - Access pattern
  recipes and chunk configuration guide
- [Write Path Pipeline](chunking-write-path.md) - How on-disk chunks are created
  during FITS-to-Zarr conversion
