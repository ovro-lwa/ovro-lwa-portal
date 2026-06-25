# Cloud Storage Configuration

Cloud object stores like AWS S3, Google Cloud Storage, and the Open Storage
Network (OSN) provide scalable storage for OVRO-LWA datasets, but accessing Zarr
data from cloud storage requires understanding HTTP request patterns, metadata
consolidation, caching strategies, and provider-specific configuration.

This guide explains how cloud Zarr access works internally, provides
configuration examples for major cloud providers, and documents optimization
strategies for reducing latency and costs.

## How Cloud Zarr Access Works

### The fsspec Abstraction Layer

OVRO-LWA's `open_dataset()` uses the
[fsspec](https://filesystem-spec.readthedocs.io/) library as a filesystem
abstraction layer sitting between xarray and cloud storage:

```text
User Code
    │
    ├─ open_dataset("s3://bucket/data.zarr")
    │
    └─> fsspec.filesystem("s3", ...)
            │
            ├─ s3fs (AWS S3)
            ├─ gcsfs (Google Cloud Storage)
            └─ http (HTTPS URLs)
                │
                └─> Cloud Storage (S3, GCS, OSN)
```

When you call `open_dataset()` with a cloud URL, the function:

1. Detects the protocol (`s3://`, `gs://`, `https://`)
2. Constructs an fsspec filesystem with `storage_options` (credentials,
   endpoint)
3. Creates a mapper object via `fs.get_mapper(path)` that presents cloud objects
   as a dict-like interface
4. Passes the mapper to `xr.open_zarr()` which reads metadata and constructs
   Dask task graphs
5. On `.compute()`, Dask workers fetch chunks via individual HTTP GET requests

See `src/ovro_lwa_portal/io.py:622-656` for the mapper construction logic.

### One Chunk = One HTTP Request

In Zarr v2 (the version used by OVRO-LWA), each chunk is stored as a separate
object in cloud storage. There are no range requests within chunks — fetching
any part of a chunk requires downloading the entire chunk object.

This one-to-one mapping has critical implications:

- A dataset with 10,000 chunks requires 10,000 HTTP GET requests to read fully
- Each request incurs ~50-100ms latency regardless of chunk size
- Small chunks (< 1 MB) waste time on HTTP overhead rather than data transfer
- Large chunks (> 100 MB) waste bandwidth when accessing subsets

See [Chunking Fundamentals](chunking-fundamentals.md) for the 10-100 MB
compressed chunk size sweet spot that balances these factors.

### Metadata Requests: The N+1 Problem

Before reading any chunk data, Zarr must fetch metadata describing array shapes,
chunk layouts, compression settings, and data types. Without optimization, this
creates an "N+1 problem":

**Without consolidated metadata:**

```text
1. GET .zgroup                  (root group metadata)
2. GET SKY/.zarray              (SKY array metadata)
3. GET SKY/.zattrs              (SKY attributes)
4. GET BEAM/.zarray             (BEAM array metadata)
5. GET BEAM/.zattrs             (BEAM attributes)
... one request per array + attributes
```

For a store with 10 arrays, this is 20+ HTTP requests before any actual data is
read. With typical 50-100ms latency per request, metadata alone adds 1-2 seconds
of overhead.

**With consolidated metadata:**

```text
1. GET .zmetadata               (all metadata in one file)
```

A single request fetches all metadata at once, reducing open time from seconds
to milliseconds. Consolidated metadata is essential for production cloud
deployments.

### How xr.open_zarr() Constructs the Mapper

When you call `open_dataset()`, it internally calls `xr.open_zarr()` with a
configured mapper. The process:

```python
# Simplified from io.py:622-656
import fsspec

# Create filesystem with credentials
fs = fsspec.filesystem("s3", key="...", secret="...",
                       client_kwargs={"endpoint_url": "https://..."})

# Create mapper: dict-like interface to cloud objects
path = "bucket/prefix/data.zarr"
store = fs.get_mapper(path)

# Pass to xarray
ds = xr.open_zarr(store, consolidated=True, chunks="auto")
```

The mapper presents cloud storage as a Python dictionary where keys are object
paths like `SKY/0.0.0.0.1` and values are the compressed chunk bytes.

## Consolidated Metadata

### What It Is

Consolidated metadata is a single `.zmetadata` file containing all array
metadata, group metadata, and attributes for an entire Zarr store. Instead of
dozens of small metadata requests, a single GET retrieves everything.

The `.zmetadata` file is a JSON document with this structure:

```json
{
  "metadata": {
    ".zgroup": {...},
    "SKY/.zarray": {...},
    "SKY/.zattrs": {...},
    "BEAM/.zarray": {...},
    "BEAM/.zattrs": {...}
  },
  "zarr_consolidated_format": 1
}
```

### Why It Matters

Without consolidated metadata, opening a Zarr store from cloud storage requires
one HTTP request per array plus one per group. For a store with 10 arrays, this
is 20+ requests totaling 1-2 seconds of latency before any data is read.

With consolidated metadata, opening the same store requires one request and
completes in ~50-100ms.

**Performance impact example:**

- Store with 15 arrays, no consolidated metadata: ~1.5s to open (15 arrays × 2
  files × 50ms)
- Same store with consolidated metadata: ~50ms to open (1 request)

For interactive workflows where datasets are opened repeatedly, this 30× speedup
in open time significantly improves user experience.

### Checking for Consolidated Metadata

Verify whether a store has consolidated metadata:

```python
import zarr

# For cloud stores, provide storage_options
store = zarr.open(
    "s3://bucket/data.zarr",
    mode="r",
    storage_options={"key": "...", "secret": "..."}
)

# Check for .zmetadata file
if ".zmetadata" in store.store:
    print("Consolidated metadata present")
else:
    print("No consolidated metadata — performance will be degraded")
```

For local stores, check the filesystem directly:

```bash
ls /path/to/data.zarr/.zmetadata
```

### Creating Consolidated Metadata

If a store lacks consolidated metadata, create it with
`zarr.consolidate_metadata()`:

```python
import zarr

# For local stores
zarr.consolidate_metadata("/path/to/data.zarr")

# For cloud stores (requires write access)
store = zarr.open(
    "s3://bucket/data.zarr",
    mode="r+",
    storage_options={"key": "...", "secret": "..."}
)
zarr.consolidate_metadata(store.store)
```

!!! warning "Write Access Required"

    Creating consolidated metadata requires write access to the store. For
    read-only stores or stores you don't control, contact the data provider to
    request consolidated metadata.

### How xr.open_zarr(consolidated=True) Works

OVRO-LWA's `open_dataset()` passes `consolidated=True` to `xr.open_zarr()` by
default. This enables consolidated metadata loading with graceful fallback:

1. xarray attempts to read `.zmetadata`
2. If `.zmetadata` exists, all metadata is loaded from this single file
3. If `.zmetadata` is missing, xarray falls back to reading individual `.zarray`
   and `.zattrs` files

The fallback behavior means `consolidated=True` never causes errors, but
performance degrades when `.zmetadata` is absent.

### OVRO-LWA Ingest Pipeline Behavior

The OVRO-LWA FITS-to-Zarr conversion pipeline in
`ovro_lwa_portal.ingest.fits_to_zarr_xradio` writes Zarr stores via xradio's
`write_image()` function. As of the current implementation, consolidated
metadata is not explicitly created during ingest.

!!! tip "Verify Consolidated Metadata Before Production Deployment"

    Always verify that a Zarr store has consolidated metadata before deploying
    it for production use or sharing with users. Open time can be 30× slower
    without it. Create consolidated metadata manually using
    `zarr.consolidate_metadata()` if needed.

## Provider-Specific Configuration

### AWS S3

AWS S3 is the most widely used cloud object store. Authentication supports
environment variables, IAM roles, or explicit credentials.

**Basic S3 access:**

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket-name/observation.zarr",
    storage_options={
        "key": "<AWS_ACCESS_KEY_ID>",
        "secret": "<AWS_SECRET_ACCESS_KEY>",
    }
)
```

**Authentication methods:**

1. **Explicit credentials** (shown above): Pass `key` and `secret` in
   `storage_options`
2. **Environment variables:** Set `AWS_ACCESS_KEY_ID` and
   `AWS_SECRET_ACCESS_KEY`, then omit `storage_options`
3. **IAM roles:** When running on EC2/ECS/Lambda, omit `storage_options` and
   credentials are automatically retrieved from the instance metadata service
4. **Shared credentials file:** Configure `~/.aws/credentials` and omit
   `storage_options`

**Specifying region:**

S3 buckets exist in specific regions. Accessing from the same region
significantly reduces latency per request compared to cross-region access —
choose the region closest to where your compute runs. Specify region in
`storage_options`:

```python
ds = ovro_lwa_portal.open_dataset(
    "s3://bucket-name/observation.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {"region_name": "us-west-2"},
    }
)
```

If your bucket is in `us-east-1` but you're accessing from `us-west-2`, expect
higher latency. Use `aws s3api get-bucket-location --bucket bucket-name` to
check bucket region.

### Google Cloud Storage

Google Cloud Storage (GCS) uses different authentication but follows similar
patterns.

**Basic GCS access:**

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "gs://bucket-name/observation.zarr",
    storage_options={
        "token": "/path/to/service-account-key.json"
    }
)
```

**Authentication methods:**

1. **Service account key file** (shown above): Download a JSON key from GCP
   console and pass the path in `storage_options`
2. **Application default credentials:** Set
   `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` environment variable and
   omit `storage_options`
3. **Compute Engine default service account:** When running on GCP Compute
   Engine, omit `storage_options` to use the instance service account

**Alternative URL format:**

GCS supports both `gs://` and `gcs://` protocols:

```python
# Both are equivalent
ds = ovro_lwa_portal.open_dataset("gs://bucket/data.zarr")
ds = ovro_lwa_portal.open_dataset("gcs://bucket/data.zarr")
```

### Open Storage Network (OSN)

OSN provides S3-compatible object storage for academic research. OSN endpoints
differ from AWS S3 in endpoint URLs and latency characteristics.

**Basic OSN access:**

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket-name/observation.zarr",
    storage_options={
        "key": "OSN_ACCESS_KEY",
        "secret": "OSN_SECRET_KEY",
        "client_kwargs": {
            "endpoint_url": "https://caltech1.osn.mghpcc.org"
        },
    }
)
```

**OSN endpoints:**

Different OSN pods have different endpoint URLs. Common endpoints:

- `https://uma1.osn.mghpcc.org` (UMass Amherst)
- `https://caltech1.osn.mghpcc.org` (Caltech)
- `https://renc1.osn.mghpcc.org` (RENCI)

Verify the correct endpoint for your bucket with the OSN allocation manager or
bucket owner.

**DOI Resolution to OSN:**

OVRO-LWA datasets archived on OSN may be assigned DOIs. The `resolve_source()`
function handles DOI → HTTPS → S3 resolution automatically:

```python
from ovro_lwa_portal import resolve_source

# Resolve DOI to see the full chain
info = resolve_source(
    "doi:10.33569/9wsys-h7b71",
    production=False,
    storage_options={"key": "...", "secret": "..."}
)

print(info["resolved_url"])    # HTTPS OSN URL
print(info["s3_url"])          # S3 URL (if credentials provided)
print(info["endpoint"])        # OSN endpoint
```

When credentials are provided, `open_dataset()` automatically converts OSN HTTPS
URLs to S3 URLs for better performance (S3 protocol is faster than HTTPS for
multi-object access). See `src/ovro_lwa_portal/io.py:168-213` for the conversion
logic in `_convert_osn_https_to_s3()`.

!!! note "OSN Latency Characteristics"

    OSN endpoints have different latency characteristics than AWS S3 proper due
    to different geographic routing and network paths. The actual difference is
    not quantified — benchmark your specific OSN endpoint using the techniques
    in [Benchmarking Performance](chunking-benchmarking.md) to establish
    baseline expectations for your infrastructure.

## Caching Strategies

Caching reduces repeated HTTP requests by storing previously fetched chunks
locally. Multiple caching strategies exist, each suited for different access
patterns.

### fsspec simplecache — Transparent Per-Chunk Caching

The simplest caching strategy: fsspec automatically caches chunks as they're
fetched, storing them in a local directory.

**How to configure:**

```python
import fsspec
import ovro_lwa_portal

# Create a caching filesystem
fs = fsspec.filesystem(
    "simplecache",
    target_protocol="s3",
    target_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {"endpoint_url": "https://..."},
    },
    cache_storage="/tmp/zarr_cache",
)

# Create mapper from the caching filesystem
mapper = fs.get_mapper("bucket-name/observation.zarr")

# Open with xarray
import xarray as xr
ds = xr.open_zarr(mapper, consolidated=True, chunks="auto")
```

**When to use:**

- Repeatedly accessing the same chunks within a single session
- Interactive analysis where you load, analyze, and reload the same data regions
- Memory constraints prevent loading full dataset with `chunks=None`

**Behavior:**

- First access: Fetches chunk from cloud, stores in `cache_storage`
- Subsequent access: Reads chunk from local cache (no HTTP request)
- Cache persists until manually deleted or system reboot (if using `/tmp`)

**Limitations:**

- Cache is not shared across processes
- No automatic eviction — cache grows unbounded
- Cache directory must have sufficient disk space

### fsspec filecache — Persistent Local Cache

Similar to `simplecache` but with persistent storage that survives restarts and
smarter cache management.

**How to configure:**

```python
import fsspec

fs = fsspec.filesystem(
    "filecache",
    target_protocol="s3",
    target_options={"key": "...", "secret": "..."},
    cache_storage="/home/user/.cache/ovro_lwa_zarr",
    expiry_time=86400,  # Cache expires after 24 hours (seconds)
)

mapper = fs.get_mapper("bucket-name/observation.zarr")
```

**When to use:**

- Repeated analyses on the same dataset over multiple days
- Workflows where the dataset doesn't change but you run different queries
- Shared cache directory across multiple scripts/notebooks

**Behavior:**

- Chunks cached to `cache_storage` with metadata (timestamp, size)
- Cache survives system restarts
- Chunks older than `expiry_time` are automatically re-fetched on next access
- Cache directory can be shared across processes (filesystem locking)

**Configuration options:**

- `expiry_time`: Seconds before cached chunks are considered stale (default:
  604800 = 7 days)
- `cache_check`: How often to check for cache validity (default: 60s)
- `same_names`: If True, cache files use original names (default: False uses
  hashed names)

### Full Local Download

For the ultimate in read performance, download the entire Zarr store locally
using `aws s3 sync`, `gsutil rsync`, or `rclone`.

**Using aws CLI:**

```bash
# Sync entire store to local directory
aws s3 sync s3://bucket-name/observation.zarr /data/local/observation.zarr

# Configure endpoint for OSN
aws s3 sync s3://bucket-name/observation.zarr /data/local/observation.zarr \
    --endpoint-url https://caltech1.osn.mghpcc.org
```

**Using rclone (works for all cloud providers):**

```bash
# Configure remote in ~/.config/rclone/rclone.conf first
rclone sync remote:bucket-name/observation.zarr /data/local/observation.zarr
```

**When to use:**

- Dataset fits on local storage (typically < 100 GB)
- Workflow requires reading most or all chunks
- Performance is critical (local NVMe SSD → 500+ MB/s vs cloud → 10-50 MB/s)
- Network is unreliable or slow

**After downloading:**

```python
import ovro_lwa_portal

# Read from local copy — no cloud access
ds = ovro_lwa_portal.open_dataset("/data/local/observation.zarr")
```

### Decision Guide

Choose a caching strategy based on your access pattern and storage constraints:

| Access Pattern                      | Recommended Strategy | Rationale                                      |
| ----------------------------------- | -------------------- | ---------------------------------------------- |
| One-time exploratory analysis       | No caching           | Overhead not worth it for single access        |
| Interactive session, same data      | `simplecache`        | Fast repeated access within session            |
| Daily workflow, same dataset        | `filecache`          | Persistent cache survives restarts             |
| Processing all chunks, local space  | Full download        | Maximum performance, no network variability    |
| Large dataset, small subsets        | `simplecache`        | Cache only the chunks you use                  |
| Shared analysis across team members | Full download        | One download benefits multiple users           |
| Dataset > local storage capacity    | `filecache`          | Selective caching with automatic eviction      |
| Unreliable network                  | Full download        | Eliminate network dependency for critical work |

## Concurrent Access Tuning

### How Dask Parallelizes Chunk Fetches

When you call `.compute()` on a Dask array, Dask creates a task graph where each
task corresponds to fetching and processing one or more chunks. Dask workers
execute these tasks in parallel, fetching multiple chunks concurrently.

**Parallelism example:**

```python
import ovro_lwa_portal
from dask.distributed import Client

# Start local cluster with 4 workers
client = Client(n_workers=4)

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/data.zarr",
    storage_options={"key": "...", "secret": "..."},
    chunks={"time": 1, "frequency": 1, "l": 1024, "m": 1024}
)

# Compute triggers parallel chunk fetches across 4 workers
result = ds.SKY.isel(time=slice(0, 10)).mean(dim=["l", "m"]).compute()
```

Each of the 4 workers fetches chunks in parallel, issuing multiple HTTP requests
simultaneously. This parallelism is critical for good cloud performance: a
single worker fetching chunks serially would take ~50ms × N chunks, while 4
workers reduce this to ~50ms × (N / 4).

### s3fs max_concurrency Parameter

The `s3fs` library (underlying fsspec for S3 access) limits the number of
concurrent requests per filesystem instance. The default is `max_concurrency=1`,
meaning requests are serialized even when Dask has multiple workers.

**Increasing concurrency:**

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/data.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "max_concurrency": 10,
    }
)
```

**Guidelines:**

- For single-worker Dask: `max_concurrency=5-10` improves throughput
- For multi-worker Dask (4+ workers): `max_concurrency=1` per worker (each
  worker gets its own filesystem instance)
- Avoid `max_concurrency > 20` without benchmarking — diminishing returns and
  potential rate limiting

### fsspec Retry and Timeout Configuration

Cloud networks are unreliable. Configure retries and timeouts to handle
transient failures gracefully.

**Configuring retries:**

```python
import ovro_lwa_portal

ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/data.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "config_kwargs": {
            "retries": {"max_attempts": 5},
        },
    }
)
```

**Configuring timeouts:**

```python
ds = ovro_lwa_portal.open_dataset(
    "s3://bucket/data.zarr",
    storage_options={
        "key": "...",
        "secret": "...",
        "client_kwargs": {
            "config": {
                "connect_timeout": 10,  # seconds
                "read_timeout": 60,     # seconds
            }
        },
    }
)
```

**When to adjust:**

- Unreliable network: Increase `max_attempts` to 5-10
- Fast network, small chunks: Decrease `read_timeout` to 10-20s to fail fast
- Slow network, large chunks: Increase `read_timeout` to 120-300s

### Balancing Concurrency vs. Rate Limiting

Cloud providers throttle requests at high rates to prevent abuse and ensure fair
resource allocation. Excessive concurrency triggers rate limiting errors.

**Common rate limits:**

- AWS S3: 5,500 GET requests per second per prefix
- Google Cloud Storage: 5,000 requests per second per bucket
- OSN: Provider-dependent, typically lower than commercial clouds

**Symptoms of rate limiting:**

- HTTP 503 "Slow Down" errors
- Exponential backoff warnings in logs
- Degraded performance at high worker counts

!!! warning "Avoid Excessive Concurrency Without Benchmarking"

    Setting `max_concurrency` too high or using too many Dask workers can
    trigger rate limiting, degrading performance below single-worker baseline.
    Always benchmark with
    [the techniques in Benchmarking Performance](chunking-benchmarking.md) to
    verify that increased concurrency improves throughput.

**Mitigation strategies:**

- Reduce Dask worker count (fewer concurrent requests)
- Implement request throttling:
  `dask.config.set({"distributed.worker.connections.incoming": 2})`
- Use caching to reduce request volume (see Caching Strategies above)
- Consolidate small chunks into larger chunks to reduce total request count (see
  [Chunking Fundamentals](chunking-fundamentals.md))

## Cost Considerations

Cloud storage pricing includes three components: storage fees, request fees, and
data transfer (egress) fees. Zarr access patterns directly affect request and
egress costs.

### GET Request Pricing

Cloud providers charge per request. For AWS S3 as of publication:

- GET request: $0.0004 per 1,000 requests
- PUT request: $0.005 per 1,000 requests (10× more expensive than GET)

!!! note "Prices Change — Verify Current Pricing"

    The prices listed here are examples from AWS S3 standard tier as of early
    2024. Verify current pricing at
    [AWS S3 Pricing](https://aws.amazon.com/s3/pricing/),
    [GCS Pricing](https://cloud.google.com/storage/pricing), or your provider's
    documentation.

**Example cost calculation:**

- Dataset with 10,000 chunks
- Full dataset scan: 10,000 GET requests
- Cost: 10,000 / 1,000 × $0.0004 = $0.004 (less than half a cent)

Request costs are negligible for individual analyses but scale with usage:

- 1,000 full scans: $4
- 10,000 full scans: $40

For large-scale batch processing or public datasets with thousands of users,
request costs accumulate.

### Data Transfer (Egress) Costs

Egress fees apply when transferring data out of the cloud provider's network.
For AWS S3:

- First 100 GB per month: Free
- Next 10 TB per month: $0.09 per GB
- Next 40 TB per month: $0.085 per GB

**Example egress cost:**

- Dataset: 30 GB uncompressed, 10 GB compressed
- Full download: 10 GB egress
- Cost: $0 (within free tier) or $0.90 if over free tier

Egress costs dominate for large datasets or frequent full downloads.

### How Chunk Count Affects Costs

More chunks = more HTTP requests = higher request costs.

**Chunking strategy cost impact:**

- Dataset: 30 GB, chunk size 10 MB → 3,000 chunks
- Full scan: 3,000 GET requests → $0.0012
- Same dataset, chunk size 100 MB → 300 chunks
- Full scan: 300 GET requests → $0.00012 (10× cheaper)

Larger chunks reduce request costs but may waste egress bandwidth if you only
need subsets. Balance chunk size based on typical access patterns (see
[Chunking Fundamentals](chunking-fundamentals.md) for the 10-100 MB sweet spot).

### Cost Minimization Strategies

1. **Use consolidated metadata** — Saves 10-50 requests per dataset open
2. **Enable caching** — Repeated chunk access costs zero after first fetch
3. **Optimize chunk sizes** — Larger chunks reduce request counts
4. **Download locally for intensive work** — One-time egress cost, unlimited
   local access
5. **Process in-cloud** — Run analysis on cloud VMs in the same region (no
   egress fees)

!!! note "OSN Does Not Charge Egress Fees for Academic Use"

    The Open Storage Network (OSN) does not charge egress fees for academic and
    research use, a significant cost advantage compared to commercial cloud
    providers. OVRO-LWA users accessing data from OSN endpoints avoid the
    egress costs that would apply on AWS S3 or GCS, making OSN particularly
    attractive for large dataset analysis and public data distribution.

## Troubleshooting Cloud Access

### Connection Timeout

**Symptom:**

```text
TimeoutError: Connection to endpoint timed out
```

**Causes and solutions:**

1. **Incorrect endpoint URL** — Verify endpoint matches bucket location:

   ```python
   from ovro_lwa_portal import resolve_source
   info = resolve_source("s3://bucket/data.zarr", storage_options={...})
   print(info["endpoint"])  # Check resolved endpoint
   ```

2. **Network firewall blocking cloud access** — Test with `curl`:

   ```bash
   curl -I https://endpoint.osn.mghpcc.org
   ```

   If curl fails, check firewall rules or proxy settings.

3. **DNS resolution failure** — Verify endpoint DNS:
   ```bash
   nslookup caltech1.osn.mghpcc.org
   ```

**Fix:**

- Verify `endpoint_url` in `storage_options["client_kwargs"]`
- Test network connectivity outside Python
- Increase `connect_timeout` if network is slow

### 403 Forbidden

**Symptom:**

```text
ClientError: An error occurred (403) when calling HeadObject: Forbidden
```

**Causes and solutions:**

1. **Invalid credentials** — Access key or secret key is incorrect:

   ```python
   # Verify credentials by listing buckets
   import boto3
   s3 = boto3.client('s3', endpoint_url='https://...',
                     aws_access_key_id='...', aws_secret_access_key='...')
   s3.list_buckets()  # Should succeed if credentials valid
   ```

2. **Credentials expired** — Temporary credentials have expiration times. Check
   expiry and regenerate if needed.

3. **Insufficient bucket permissions** — Credentials are valid but lack
   permission for this bucket. Contact bucket owner to verify IAM policy grants
   `s3:GetObject` permission.

**Fix:**

- Verify `key` and `secret` in `storage_options`
- Check credential expiration dates
- Confirm bucket policy allows your account

### NoSuchBucket

**Symptom:**

```text
NoSuchBucket: The specified bucket does not exist
```

**Causes and solutions:**

1. **Typo in bucket name** — Verify exact bucket name (case-sensitive):

   ```python
   from ovro_lwa_portal import resolve_source
   info = resolve_source("s3://bucket/data.zarr", storage_options={...})
   print(info["bucket"])  # Check parsed bucket name
   ```

2. **Wrong endpoint** — Bucket exists on different OSN pod or AWS region:

   - OSN buckets are pod-specific (uma1 vs caltech1)
   - AWS buckets are region-specific (us-east-1 vs us-west-2)

3. **Bucket not yet created** — Verify bucket exists with cloud provider console

**Fix:**

- Double-check bucket name for typos
- Verify `endpoint_url` matches bucket location
- Use `resolve_source()` to debug DOI → URL resolution (see
  `src/ovro_lwa_portal/io.py:374-484`)

### Slow Reads

**Symptom:**

Operations take minutes instead of seconds, or throughput is < 1 MB/s.

**Causes and solutions:**

1. **Misaligned chunks** — Read chunks don't match write chunks, causing excess
   HTTP requests:

   ```python
   import zarr
   store = zarr.open("s3://bucket/data.zarr", mode="r", storage_options={...})
   print(store.SKY.chunks)  # Check on-disk chunk shape
   ```

   Compare to your `chunks=` parameter in `open_dataset()`. See
   [Chunking Fundamentals](chunking-fundamentals.md) for alignment principles.

2. **Too many small chunks** — Each chunk incurs ~50-100ms latency:

   ```python
   # Count chunks
   ds = ovro_lwa_portal.open_dataset("s3://bucket/data.zarr", chunks=...)
   n_chunks = ds.SKY.data.npartitions
   print(f"Chunks to fetch: {n_chunks}")
   ```

   If `n_chunks > 1000` for a simple operation, chunk sizes are likely too
   small.

3. **Wrong region** — Accessing S3 bucket from different region adds 50-100ms
   per request. Specify `region_name` in `storage_options`:

   ```python
   ds = ovro_lwa_portal.open_dataset(
       "s3://bucket/data.zarr",
       storage_options={
           "key": "...",
           "secret": "...",
           "client_kwargs": {"region_name": "us-west-2"},
       }
   )
   ```

4. **No consolidated metadata** — Opening store requires dozens of metadata
   requests. Verify `.zmetadata` exists (see Consolidated Metadata section).

**Fix:**

- Align read chunks with on-disk chunks (see
  [Read-Time Optimization](chunking-read-optimization.md) for recipes)
- Use caching for repeated access (see Caching Strategies section)
- Benchmark with
  [techniques in Benchmarking Performance](chunking-benchmarking.md) to isolate
  bottleneck

### Using resolve_source() for Debugging

The `resolve_source()` function exposes the full DOI → URL → S3 resolution chain
without loading data, useful for debugging cloud access issues:

```python
from ovro_lwa_portal import resolve_source

info = resolve_source(
    "doi:10.33569/9wsys-h7b71",
    production=False,
    storage_options={"key": "...", "secret": "..."}
)

print("Source type:", info["source_type"])           # "doi"
print("Original source:", info["original_source"])   # "doi:10.33569/..."
print("Resolved URL:", info["resolved_url"])         # HTTPS OSN URL
print("Final URL:", info["final_url"])               # S3 URL if converted
print("S3 URL:", info["s3_url"])                     # S3 URL
print("Endpoint:", info["endpoint"])                 # OSN endpoint
print("Bucket:", info["bucket"])                     # Bucket name
print("Path:", info["path"])                         # Path within bucket
```

Use this to verify:

- DOI resolution succeeds
- HTTPS → S3 conversion applies when expected
- Endpoint URL matches bucket location
- Bucket and path are parsed correctly

See `src/ovro_lwa_portal/io.py:374-484` for implementation details.

## See Also

- [Chunking Fundamentals](chunking-fundamentals.md) - One chunk = one HTTP
  request, 10-100 MB sweet spot
- [Read-Time Optimization](chunking-read-optimization.md) - Chunk configuration
  recipes for different access patterns
- [Benchmarking Performance](chunking-benchmarking.md) - Measure and optimize
  cloud read performance
- [Getting Started: Loading Data](../../getting-started/loading-data.md) - Basic
  `open_dataset()` usage
