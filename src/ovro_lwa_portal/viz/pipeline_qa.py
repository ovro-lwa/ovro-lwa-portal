"""Pipeline tree discovery and FITS→Zarr QA conversion for one observation day."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits

import ovro_lwa_portal
from ovro_lwa_portal.fits_to_zarr_xradio import (
    _lm_reference_from_existing_zarr,
    convert_fits_dir_to_zarr,
)

PIPELINE_ROOT = Path("/lustre/pipeline/exopipe/phase1/")
SYMLINK_ROOT = Path("/lustre/claw")
ZARR_ROOT = Path("/fast/claw")

RUN_PATTERN = re.compile(r"Run_(\d{8})_(\d{6})")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HOUR_PATTERN = re.compile(r"^\d{2}h$")

WIDEBAND_QA = Path("Wideband/thermal_noise_vs_subband.png")
FLUX_CHECK_HYBRID_CSV = Path("QA/flux_check_hybrid.csv")
I_FITS_GLOB = "*I-Deep-Taper-Robust-0.75-image*.pbcorr.fits"
V_FITS_GLOB = "*V-Taper-Deep-image*.pbcorr.fits"
REF_SUBBAND = "82MHz"
I_QA_ZARR_STEM = "pipelineQA-I-Deep-Taper-Robust-0.75"
V_QA_ZARR_STEM = "pipelineQA-V-Taper-Deep"

FITS_KEY_PATTERN = re.compile(
    r"^(?P<subband>\d+MHz)-.*-image-(?P<stamp>\d{8}_\d{6})"
)

LogFn = Callable[[str], None]


def run_sort_key(run_dir: Path) -> str:
    """Sort key from Run_YYYYMMDD_HHMMSS directory name."""
    match = RUN_PATTERN.match(run_dir.name)
    if match is None:
        return run_dir.name
    return match.group(1) + match.group(2)


def select_run_dir(day_dir: Path) -> Path | None:
    """Pick the newest Run_* that has Wideband thermal-noise QA."""
    runs = [
        path
        for path in day_dir.iterdir()
        if path.is_dir() and path.name.startswith("Run_") and (path / WIDEBAND_QA).is_file()
    ]
    if not runs:
        return None
    return max(runs, key=run_sort_key)


def list_subbands(run_dir: Path) -> list[str]:
    """List subband directories (e.g. 23MHz) under a run."""
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.endswith("MHz")
    )


def discover_hour_bins(root: Path) -> list[str]:
    return sorted(
        (path.name for path in root.iterdir() if path.is_dir() and HOUR_PATTERN.match(path.name)),
        key=lambda name: int(name[:-1]),
    )


def scan_coverage(root: Path | None = None) -> pd.DataFrame:
    """Build one row per (LST hour bin, observation day) from directory names."""
    scan_root = PIPELINE_ROOT if root is None else root
    rows: list[dict[str, object]] = []

    for hour in discover_hour_bins(scan_root):
        hour_dir = scan_root / hour
        for day_dir in sorted(hour_dir.iterdir()):
            if not day_dir.is_dir() or not DATE_PATTERN.match(day_dir.name):
                continue

            all_runs = [p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("Run_")]
            selected = select_run_dir(day_dir)
            subbands = list_subbands(selected) if selected is not None else []

            rows.append(
                {
                    "lst_hour": hour,
                    "obs_date": day_dir.name,
                    "n_runs": len(all_runs),
                    "n_wideband_runs": sum(
                        1 for p in all_runs if (p / WIDEBAND_QA).is_file()
                    ),
                    "latest_run": selected.name if selected is not None else pd.NA,
                    "n_subbands": len(subbands),
                    "subbands": ", ".join(subbands),
                    "run_path": str(selected) if selected is not None else pd.NA,
                    "thermal_noise_png": str(selected / WIDEBAND_QA)
                    if selected is not None
                    else pd.NA,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["lst_hour_num"] = df["lst_hour"].str.replace("h", "", regex=False).astype(int)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df.sort_values(["obs_date", "lst_hour_num"]).reset_index(drop=True)


def qa_days(coverage: pd.DataFrame) -> list[str]:
    """Observation days with at least one Wideband-qualified run."""
    if coverage.empty:
        return []
    mask = coverage["latest_run"].notna()
    return (
        coverage.loc[mask, "obs_date"]
        .dt.strftime("%Y-%m-%d")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def qa_zarr_path(pol: str, select_day: str) -> Path:
    """Output Zarr path for pipeline QA products on one observation day."""
    day_tag = select_day.replace("-", "")
    stem = I_QA_ZARR_STEM if pol == "I" else V_QA_ZARR_STEM
    return ZARR_ROOT / f"{stem}-{day_tag}.zarr"


def zarr_status(select_day: str) -> dict[str, bool]:
    """Return whether Stokes I/V QA Zarr stores exist for one day."""
    return {
        "I": qa_zarr_path("I", select_day).exists(),
        "V": qa_zarr_path("V", select_day).exists(),
    }


def default_select_day(coverage: pd.DataFrame) -> str | None:
    """Earliest QA day with Stokes I Zarr, else earliest QA day."""
    days = qa_days(coverage)
    if not days:
        return None
    for day in days:
        if qa_zarr_path("I", day).exists():
            return day
    return days[0]


def day_rows(select_day: str, coverage: pd.DataFrame) -> pd.DataFrame:
    """Rows for one observation day with a Wideband-qualified run."""
    mask = coverage["obs_date"].dt.strftime("%Y-%m-%d") == select_day
    return (
        coverage.loc[mask & coverage["latest_run"].notna()]
        .sort_values("lst_hour_num")
        .reset_index(drop=True)
    )


def day_summary_table(select_day: str, coverage: pd.DataFrame) -> pd.DataFrame:
    """Minimal per-hour summary for the QA plot grid."""
    rows = day_rows(select_day, coverage)
    if rows.empty:
        return pd.DataFrame(columns=["lst_hour", "n_subbands", "thermal_noise_png"])
    return rows[["lst_hour", "n_subbands", "thermal_noise_png"]].reset_index(drop=True)


def frequency_mhz_from_subdir(subdir_name: str) -> float:
    """Parse a subband directory name such as ``82MHz`` into MHz."""
    if not subdir_name.endswith("MHz"):
        msg = f"Expected a frequency subdirectory name ending in MHz, got {subdir_name!r}"
        raise ValueError(msg)
    return float(subdir_name.removesuffix("MHz"))


def collect_flux_check_hybrid_paths(run_dir: Path) -> list[Path]:
    """Return ``flux_check_hybrid.csv`` files under each frequency subband in one run."""
    return sorted(run_dir.glob(f"*MHz/{FLUX_CHECK_HYBRID_CSV.as_posix()}"))


def load_flux_check_hybrid_dataframe(
    select_day: str,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Load and combine all ``flux_check_hybrid.csv`` files for one observation day.

    Each CSV row is augmented with ``lst_hour``, ``frequency_mhz``, ``obs_date``,
    and ``flux_ratio`` (= ``imfit_flux`` / ``model_flux``).
    """
    chunks: list[pd.DataFrame] = []
    for _, row in day_rows(select_day, coverage).iterrows():
        run_dir = Path(str(row["run_path"]))
        if not run_dir.is_dir():
            continue
        for csv_path in collect_flux_check_hybrid_paths(run_dir):
            subband = csv_path.parent.parent.name
            freq_from_dir = frequency_mhz_from_subdir(subband)
            chunk = pd.read_csv(csv_path)
            chunk["lst_hour"] = row["lst_hour"]
            chunk["lst_hour_num"] = int(row["lst_hour_num"])
            chunk["obs_date"] = select_day
            chunk["run_path"] = str(run_dir)
            chunk["subband"] = subband
            if "freq" in chunk.columns:
                chunk["frequency_mhz"] = chunk["freq"].astype(float)
            else:
                chunk["frequency_mhz"] = freq_from_dir
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(
            columns=[
                "imfit_flux",
                "imfit_err",
                "elevation",
                "model_flux",
                "source",
                "freq",
                "lst_hour",
                "lst_hour_num",
                "obs_date",
                "run_path",
                "subband",
                "frequency_mhz",
                "flux_ratio",
            ]
        )

    df = pd.concat(chunks, ignore_index=True)
    model = df["model_flux"].astype(float)
    df["flux_ratio"] = np.where(model != 0, df["imfit_flux"].astype(float) / model, np.nan)
    return df.sort_values(["source", "lst_hour_num", "frequency_mhz"]).reset_index(drop=True)


def flux_ratio_grids(flux_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot flux ratios into one LST × frequency grid per calibrator source."""
    if flux_df.empty or "source" not in flux_df.columns:
        return {}

    grids: dict[str, pd.DataFrame] = {}
    for source, group in flux_df.groupby("source", sort=True):
        grid = group.pivot_table(
            index="lst_hour_num",
            columns="frequency_mhz",
            values="flux_ratio",
            aggfunc="mean",
        )
        grids[str(source)] = grid.sort_index().sort_index(axis=1)
    return grids


def collect_pol_fits(select_day: str, pol: str, coverage: pd.DataFrame) -> list[Path]:
    """Collect deep pbcorr FITS for one Stokes parameter across all hours."""
    glob_pattern = I_FITS_GLOB if pol == "I" else V_FITS_GLOB
    fits_paths: list[Path] = []
    for run_path in day_rows(select_day, coverage)["run_path"]:
        run_dir = Path(run_path)
        fits_paths.extend(sorted(run_dir.glob(f"*/{pol}/deep/{glob_pattern}")))
    return fits_paths


def infer_target_size_from_82mhz(select_day: str, coverage: pd.DataFrame) -> int:
    """Return the square pixel size of the 82 MHz I deep image for this day."""
    ref_glob = f"{REF_SUBBAND}/I/deep/{I_FITS_GLOB}"
    for run_path in day_rows(select_day, coverage)["run_path"]:
        matches = sorted(Path(run_path).glob(ref_glob))
        if not matches:
            continue
        with fits.open(matches[0]) as hdul:
            shape = hdul[0].data.shape
            return int(max(shape[-2], shape[-1]))
    msg = f"No {REF_SUBBAND} I deep image found for {select_day}"
    raise FileNotFoundError(msg)


def stage_symlinks(fits_paths: Sequence[Path], staging_dir: Path) -> Path:
    """Symlink FITS files into a flat staging directory."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for src in fits_paths:
        dst = staging_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return staging_dir


def fits_group_key(path: Path) -> tuple[str, str]:
    match = FITS_KEY_PATTERN.match(path.name)
    if match is None:
        msg = f"Cannot parse subband/stamp from {path.name}"
        raise ValueError(msg)
    return match.group("subband"), match.group("stamp")


def beam_header_from_i(i_path: Path) -> dict[str, float]:
    with fits.open(i_path) as hdul:
        hdr = hdul[0].header
        return {key: float(hdr[key]) for key in ("BMAJ", "BMIN", "BPA") if key in hdr}


def stage_v_fits_with_beam_from_i(
    v_paths: Sequence[Path],
    i_paths: Sequence[Path],
    staging_dir: Path,
) -> Path:
    """Stage Stokes V FITS with beam headers copied from matching I images."""
    i_index = {fits_group_key(path): path for path in i_paths}
    staging_dir.mkdir(parents=True, exist_ok=True)
    for v_path in v_paths:
        key = fits_group_key(v_path)
        i_path = i_index.get(key)
        if i_path is None:
            msg = f"No matching I image for {v_path.name} ({key})"
            raise FileNotFoundError(msg)
        beam = beam_header_from_i(i_path)
        dst = staging_dir / v_path.name
        if dst.exists():
            dst.unlink()
        with fits.open(v_path) as hdul:
            hdr = hdul[0].header
            hdr.update(beam)
            hdul.writeto(dst, overwrite=True)
    return staging_dir


def convert_missing_zarr(
    select_day: str,
    coverage: pd.DataFrame,
    log: LogFn,
) -> dict[str, Path]:
    """Convert only missing Stokes I/V Zarr stores for one observation day."""
    if day_rows(select_day, coverage).empty:
        msg = f"No Wideband-qualified runs for {select_day}"
        raise RuntimeError(msg)

    zarr_paths: dict[str, Path] = {
        "I": qa_zarr_path("I", select_day),
        "V": qa_zarr_path("V", select_day),
    }

    if zarr_paths["I"].exists() and zarr_paths["V"].exists():
        log(f"Using existing Zarr stores for {select_day}")
        for pol, path in zarr_paths.items():
            log(f"  Stokes {pol}: {path}")
        return zarr_paths

    target_size: int | None = None
    fits_by_pol: dict[str, list[Path]] | None = None
    day_tag = select_day.replace("-", "")

    if not zarr_paths["I"].exists() or not zarr_paths["V"].exists():
        target_size = infer_target_size_from_82mhz(select_day, coverage)
        log(f"LM reference target size from {REF_SUBBAND}: {target_size} px")
        fits_by_pol = {
            "I": collect_pol_fits(select_day, "I", coverage),
            "V": collect_pol_fits(select_day, "V", coverage),
        }
        for pol, paths in fits_by_pol.items():
            log(f"Stokes {pol}: {len(paths)} FITS files")

    if not zarr_paths["I"].exists():
        assert fits_by_pol is not None
        i_paths = fits_by_pol["I"]
        if not i_paths:
            msg = f"No Stokes I FITS found for {select_day}"
            raise FileNotFoundError(msg)

        staging_i = SYMLINK_ROOT / f"{I_QA_ZARR_STEM}-{day_tag}-fits"
        stage_symlinks(i_paths, staging_i)
        log(f"Staged {len(i_paths)} Stokes I files -> {staging_i}")

        zarr_paths["I"] = convert_fits_dir_to_zarr(
            input_dir=staging_i,
            out_dir=ZARR_ROOT,
            zarr_name=zarr_paths["I"].name,
            fixed_dir=SYMLINK_ROOT / f"{I_QA_ZARR_STEM}-{day_tag}-fixed",
            chunk_lm=1024,
            rebuild=True,
            lm_reference_target_size=target_size,
        )
        log(f"Wrote {zarr_paths['I']}")
    else:
        log(f"Using existing Stokes I Zarr: {zarr_paths['I']}")

    lm_ref_ds = _lm_reference_from_existing_zarr(zarr_paths["I"])

    if not zarr_paths["V"].exists():
        assert fits_by_pol is not None
        v_paths = fits_by_pol["V"]
        if not v_paths:
            msg = f"No Stokes V FITS found for {select_day}"
            raise FileNotFoundError(msg)

        staging_v = SYMLINK_ROOT / f"{V_QA_ZARR_STEM}-{day_tag}-fits"
        stage_v_fits_with_beam_from_i(v_paths, fits_by_pol["I"], staging_v)
        log(f"Staged {len(v_paths)} Stokes V files -> {staging_v}")

        zarr_paths["V"] = convert_fits_dir_to_zarr(
            input_dir=staging_v,
            out_dir=ZARR_ROOT,
            zarr_name=zarr_paths["V"].name,
            fixed_dir=SYMLINK_ROOT / f"{V_QA_ZARR_STEM}-{day_tag}-fixed",
            chunk_lm=1024,
            rebuild=True,
            lm_reference_ds=lm_ref_ds,
        )
        log(f"Wrote {zarr_paths['V']}")
    else:
        log(f"Using existing Stokes V Zarr: {zarr_paths['V']}")

    return zarr_paths


def load_qa_datasets(
    select_day: str,
    log: LogFn,
    *,
    flush: Callable[[], None] | None = None,
) -> dict[str, xr.Dataset]:
    """Load available Stokes I/V datasets for one day."""
    status = zarr_status(select_day)
    datasets: dict[str, xr.Dataset] = {}
    for pol, available in status.items():
        if not available:
            continue
        zarr_path = qa_zarr_path(pol, select_day)
        log(f"Opening Stokes {pol} Zarr at {zarr_path}…")
        if flush is not None:
            flush()
        started = time.perf_counter()
        ds = ovro_lwa_portal.open_dataset(zarr_path, chunks={"l": 512, "m": 512})
        elapsed = time.perf_counter() - started
        datasets[pol] = ds
        log(
            f"Opened Stokes {pol} in {elapsed:.1f}s "
            f"(time={ds.sizes.get('time', '?')}, freq={ds.sizes.get('frequency', '?')}, "
            f"l={ds.sizes.get('l', '?')}, m={ds.sizes.get('m', '?')})"
        )
        if flush is not None:
            flush()
    return datasets


def convert_button_label(status: dict[str, bool]) -> str:
    """Label for the conversion action button."""
    if status["I"] and status["V"]:
        return "Convert FITS → Zarr (complete)"
    if status["I"] and not status["V"]:
        return "Convert Stokes V"
    return "Convert FITS → Zarr"


def convert_button_disabled(status: dict[str, bool], *, converting: bool) -> bool:
    """Whether the conversion button should be disabled."""
    if converting:
        return True
    return status["I"] and status["V"]
