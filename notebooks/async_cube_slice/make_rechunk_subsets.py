"""Write rechunked subset copies of the benchmark store (stage 2 layouts R0–R3).

The subset is the center native tile (``l``/``m`` 1173:1564), Stokes I, all
time and frequency. Each copy keeps the source Blosc codec so only the chunk
shape differs. R4 is not written here; it pairs R0 (sky) with R3 (dynspec).

Usage::

    pixi run python notebooks/async_cube_slice/make_rechunk_subsets.py \
        --source /fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr \
        --out-dir /fast/claw/async-test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

L0, L1 = 1173, 1564
POL = 0

SCHEMES: dict[str, tuple[int, int, int, int, int]] = {
    "R0": (1, 2, 1, 391, 391),
    "R1": (1, 15, 1, 391, 391),
    "R2": (32, 15, 1, 128, 128),
    "R3": (415, 15, 1, 32, 32),
}


def _dir_size(path: Path) -> int:
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(path)
        for name in names
    )


def read_subset(source: str, *, workers: int = 16) -> xr.Dataset:
    group = zarr.open_consolidated(source, mode="r")
    sky = group["SKY"]
    n_time = sky.shape[0]
    out = np.empty((n_time, sky.shape[1], 1, L1 - L0, L1 - L0), dtype=sky.dtype)

    def _read(ti: int) -> None:
        out[ti] = sky[ti, :, POL : POL + 1, L0:L1, L0:L1]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_read, range(n_time)))

    coords = {
        "time": np.asarray(group["time"][:]),
        "frequency": np.asarray(group["frequency"][:]),
        "polarization": np.asarray(group["polarization"][POL : POL + 1]),
        "l": np.asarray(group["l"][L0:L1]),
        "m": np.asarray(group["m"][L0:L1]),
    }
    ds = xr.Dataset(
        {"SKY": (("time", "frequency", "polarization", "l", "m"), out)},
        coords=coords,
    )
    for name in ("time", "frequency", "polarization", "l", "m"):
        ds[name].attrs.update(dict(group[name].attrs))
    ds.attrs["subset_of"] = source
    ds.attrs["subset_l_m_slice"] = [L0, L1]
    return ds


def write_scheme(ds: xr.Dataset, name: str, chunks: tuple[int, ...], out_dir: Path, compressor) -> dict:
    path = out_dir / f"{name}.zarr"
    encoding = {"SKY": {"chunks": chunks, "compressor": compressor}}
    for var in ds.coords:
        ds[var].encoding.clear()
    t0 = time.perf_counter()
    ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
    elapsed = time.perf_counter() - t0
    return {
        "scheme": name,
        "chunks": list(chunks),
        "path": str(path),
        "write_s": round(elapsed, 2),
        "bytes_on_disk": _dir_size(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr")
    parser.add_argument("--out-dir", default="/fast/claw/async-test")
    parser.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compressor = zarr.open_consolidated(args.source, mode="r")["SKY"].compressor

    t0 = time.perf_counter()
    ds = read_subset(args.source)
    read_s = time.perf_counter() - t0
    print(f"read subset {ds.SKY.shape} in {read_s:.1f} s ({ds.SKY.nbytes / 1e9:.2f} GB)", flush=True)

    report = {"source": args.source, "read_subset_s": round(read_s, 2), "schemes": []}
    for name in args.schemes:
        info = write_scheme(ds, name, SCHEMES[name], out_dir, compressor)
        print(json.dumps(info), flush=True)
        report["schemes"].append(info)

    (out_dir / "rechunk_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
