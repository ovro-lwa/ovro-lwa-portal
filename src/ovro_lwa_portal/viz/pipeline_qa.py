"""Pipeline tree discovery and FITS→Zarr QA conversion for one observation day."""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits

import ovro_lwa_portal
from ovro_lwa_portal.fits_to_zarr_xradio import (
    _lm_reference_from_existing_zarr,
    _zarr_store_exists,
    convert_fits_dir_to_zarr,
)

PIPELINE_ROOT = Path("/lustre/pipeline/exopipe/phase1/")
PIPELINE_ROOT_PHASE2 = Path("/lustre/pipeline/exopipe/phase2/")
SYMLINK_ROOT = Path("/lustre/claw")
ZARR_ROOT = Path("/fast/claw")

RUN_PATTERN = re.compile(r"Run_(\d{8})_(\d{6})")
SCIENCE_RUN_PATTERN = re.compile(r"Science_(\d{8})_(\d{6})")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HOUR_PATTERN = re.compile(r"^\d{2}h$")

WIDEBAND_QA = Path("Wideband/thermal_noise_vs_subband.png")
FLUX_CHECK_HYBRID_CSV = Path("QA/flux_check_hybrid.csv")
I_FITS_GLOB = "*I-Deep-Taper-Robust-0.75-image*.pbcorr.fits"
V_FITS_GLOB = "*V-Taper-Deep-image*.pbcorr.fits"
I_FITS_GLOB_PHASE2 = "*I-NoTaper-*-Robust-0-*-image.pbcorr_dewarped.fits"
V_FITS_GLOB_PHASE2 = "*V-Taper-*-Robust-0-*-image.pbcorr_dewarped.fits"
REF_SUBBAND = "82MHz"
I_QA_ZARR_STEM = "pipelineQA-I-Deep-Taper-Robust-0.75"
V_QA_ZARR_STEM = "pipelineQA-V-Taper-Deep"
I_QA_ZARR_STEM_PHASE2 = "pipelineQA-phase2-I-NoTaper-Robust-0"
V_QA_ZARR_STEM_PHASE2 = "pipelineQA-phase2-V-Taper-Robust-0"
DEWARP_FREQ_COLUMN = "freq_mhz"
DEWARP_SHIFT_COLUMN = "median_shift_arcmin"

FITS_KEY_PATTERN = re.compile(r"^(?P<subband>\d+MHz)-.*?(?P<stamp>\d{8}_\d{6})")

LogFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class PipelineQAConfig:
    """Notebook-overridable paths and FITS discovery patterns for pipeline QA."""

    pipeline_root: Path
    symlink_root: Path
    zarr_root: Path
    i_fits_glob: str
    v_fits_glob: str
    run_dir_prefix: str = "Run_"
    run_dir_pattern: str = r"Run_(\d{8})_(\d{6})"
    qa_thermal_noise_glob: str = "Wideband/thermal_noise_vs_subband.png"
    flux_check_csv_glob: str = "*MHz/QA/flux_check_hybrid.csv"
    flux_check_csv_per_run: bool = False
    dewarp_summary_csv_glob: str = "QA/*dewarp_summary.csv"
    ref_subband: str = REF_SUBBAND
    i_qa_zarr_stem: str = I_QA_ZARR_STEM
    v_qa_zarr_stem: str = V_QA_ZARR_STEM
    thermal_noise_grid_cols: int = 4
    thermal_noise_plot_name: str = "thermal_noise_vs_subband"
    qa_run_label: str = "Wideband"

    @classmethod
    def default(cls) -> PipelineQAConfig:
        """Return phase1 module-default pipeline, staging, Zarr, and FITS settings."""
        return cls.phase1_default()

    @classmethod
    def phase1_default(cls) -> PipelineQAConfig:
        """Exopipe phase1: ``Run_*`` dirs, per-subband flux CSV, Wideband thermal-noise grid."""
        return cls(
            pipeline_root=PIPELINE_ROOT,
            symlink_root=SYMLINK_ROOT,
            zarr_root=ZARR_ROOT,
            i_fits_glob=I_FITS_GLOB,
            v_fits_glob=V_FITS_GLOB,
        )

    @classmethod
    def phase2_default(cls) -> PipelineQAConfig:
        """Exopipe phase2: ``Science_*`` dirs, run-level QA CSV/PNG, dewarped FITS."""
        return cls(
            pipeline_root=PIPELINE_ROOT_PHASE2,
            symlink_root=SYMLINK_ROOT,
            zarr_root=ZARR_ROOT,
            i_fits_glob=I_FITS_GLOB_PHASE2,
            v_fits_glob=V_FITS_GLOB_PHASE2,
            run_dir_prefix="Science_",
            run_dir_pattern=r"Science_(\d{8})_(\d{6})",
            qa_thermal_noise_glob="QA/*_thermal_noise_vs_freq.png",
            flux_check_csv_glob="QA/*_flux_check_hybrid.csv",
            flux_check_csv_per_run=True,
            i_qa_zarr_stem=I_QA_ZARR_STEM_PHASE2,
            v_qa_zarr_stem=V_QA_ZARR_STEM_PHASE2,
            thermal_noise_grid_cols=4,
            thermal_noise_plot_name="thermal_noise_vs_freq",
            dewarp_summary_csv_glob="QA/*dewarp_summary.csv",
            qa_run_label="Science",
        )


def resolve_pipeline_qa_config(
    *,
    config: PipelineQAConfig | None = None,
    pipeline_root: Path | str | None = None,
    symlink_root: Path | str | None = None,
    zarr_root: Path | str | None = None,
    i_fits_glob: str | None = None,
    v_fits_glob: str | None = None,
) -> PipelineQAConfig:
    """Build a :class:`PipelineQAConfig`, applying only the arguments that are set."""
    base = config or PipelineQAConfig.default()
    if (
        pipeline_root is None
        and symlink_root is None
        and zarr_root is None
        and i_fits_glob is None
        and v_fits_glob is None
    ):
        return base
    return replace(
        base,
        pipeline_root=Path(pipeline_root) if pipeline_root is not None else base.pipeline_root,
        symlink_root=Path(symlink_root) if symlink_root is not None else base.symlink_root,
        zarr_root=Path(zarr_root) if zarr_root is not None else base.zarr_root,
        i_fits_glob=i_fits_glob if i_fits_glob is not None else base.i_fits_glob,
        v_fits_glob=v_fits_glob if v_fits_glob is not None else base.v_fits_glob,
    )


def _run_pattern(config: PipelineQAConfig) -> re.Pattern[str]:
    return re.compile(config.run_dir_pattern)


def run_sort_key(run_dir: Path, *, config: PipelineQAConfig | None = None) -> str:
    """Sort key from ``Run_`` or ``Science_`` YYYYMMDD_HHMMSS directory names."""
    cfg = config or PipelineQAConfig.default()
    match = _run_pattern(cfg).match(run_dir.name)
    if match is None:
        return run_dir.name
    return match.group(1) + match.group(2)


def thermal_noise_png_for_run(
    run_dir: Path,
    *,
    config: PipelineQAConfig | None = None,
) -> Path | None:
    """Resolve the thermal-noise QA PNG for one pipeline run directory."""
    cfg = config or PipelineQAConfig.default()
    matches = sorted(run_dir.glob(cfg.qa_thermal_noise_glob))
    return matches[0] if matches else None


def run_has_thermal_noise_qa(
    run_dir: Path,
    *,
    config: PipelineQAConfig | None = None,
) -> bool:
    """Return whether a run directory contains the configured thermal-noise QA plot."""
    return thermal_noise_png_for_run(run_dir, config=config) is not None


def _science_runs_in_day(day_dir: Path, *, config: PipelineQAConfig) -> list[Path]:
    """List ``Run_*`` or ``Science_*`` directories under one observation day."""
    return [
        path
        for path in day_dir.iterdir()
        if path.is_dir() and path.name.startswith(config.run_dir_prefix)
    ]


def _thermal_noise_status_for_runs(
    runs: Sequence[Path],
    *,
    config: PipelineQAConfig | None = None,
) -> dict[Path, Path | None]:
    """Resolve thermal-noise QA PNG paths with one glob per run directory."""
    cfg = config or PipelineQAConfig.default()
    return {run_dir: thermal_noise_png_for_run(run_dir, config=cfg) for run_dir in runs}


def select_run_dir(
    day_dir: Path,
    *,
    config: PipelineQAConfig | None = None,
    runs: Sequence[Path] | None = None,
    thermal_pngs: Mapping[Path, Path | None] | None = None,
) -> Path | None:
    """Pick the newest run directory that has thermal-noise QA for this product."""
    cfg = config or PipelineQAConfig.default()
    run_dirs = list(runs) if runs is not None else _science_runs_in_day(day_dir, config=cfg)
    if not run_dirs:
        return None
    png_by_run = (
        dict(thermal_pngs)
        if thermal_pngs is not None
        else _thermal_noise_status_for_runs(run_dirs, config=cfg)
    )
    qualified = [path for path in run_dirs if png_by_run.get(path) is not None]
    if not qualified:
        return None
    return max(qualified, key=lambda path: run_sort_key(path, config=cfg))


def list_subbands(run_dir: Path) -> list[str]:
    """List subband directories (e.g. 23MHz) under a run."""
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.endswith("MHz")
    )


def populate_subbands_for_day(coverage: pd.DataFrame, select_day: str) -> None:
    """Fill ``n_subbands`` and ``subbands`` for one observation day (lazy scan).

    :func:`scan_coverage` leaves these columns unset (``NA``) until a day is
    loaded so the initial pipeline-tree walk avoids thousands of Lustre ``stat``
    calls. Updates ``coverage`` in place.
    """
    if coverage.empty or "n_subbands" not in coverage.columns:
        return
    day_mask = coverage["obs_date"].dt.strftime("%Y-%m-%d") == select_day
    day_mask &= coverage["latest_run"].notna()
    missing = day_mask & coverage["n_subbands"].isna()
    for idx in coverage.index[missing]:
        run_path = coverage.at[idx, "run_path"]
        if pd.isna(run_path):
            coverage.at[idx, "n_subbands"] = 0
            coverage.at[idx, "subbands"] = ""
            continue
        subbands = list_subbands(Path(str(run_path)))
        coverage.at[idx, "n_subbands"] = len(subbands)
        coverage.at[idx, "subbands"] = ", ".join(subbands)


def discover_hour_bins(root: Path) -> list[str]:
    return sorted(
        (path.name for path in root.iterdir() if path.is_dir() and HOUR_PATTERN.match(path.name)),
        key=lambda name: int(name[:-1]),
    )


def scan_coverage(
    root: Path | str | None = None,
    *,
    config: PipelineQAConfig | None = None,
) -> pd.DataFrame:
    """Build one row per (LST hour bin, observation day) from directory names."""
    cfg = config or PipelineQAConfig.default()
    scan_root = Path(root) if root is not None else cfg.pipeline_root
    rows: list[dict[str, object]] = []

    for hour in discover_hour_bins(scan_root):
        hour_dir = scan_root / hour
        for day_dir in sorted(hour_dir.iterdir()):
            if not day_dir.is_dir() or not DATE_PATTERN.match(day_dir.name):
                continue

            all_runs = _science_runs_in_day(day_dir, config=cfg)
            thermal_by_run = _thermal_noise_status_for_runs(all_runs, config=cfg)
            qualified_runs = [path for path in all_runs if thermal_by_run.get(path) is not None]
            selected = select_run_dir(
                day_dir,
                config=cfg,
                runs=all_runs,
                thermal_pngs=thermal_by_run,
            )
            thermal_png = thermal_by_run.get(selected) if selected is not None else None

            rows.append(
                {
                    "lst_hour": hour,
                    "obs_date": day_dir.name,
                    "n_runs": len(all_runs),
                    "n_wideband_runs": len(qualified_runs),
                    "latest_run": selected.name if selected is not None else pd.NA,
                    # Deferred: populated by populate_subbands_for_day when a day loads.
                    "n_subbands": pd.NA if selected is not None else 0,
                    "subbands": pd.NA if selected is not None else "",
                    "run_path": str(selected) if selected is not None else pd.NA,
                    "thermal_noise_png": str(thermal_png) if thermal_png is not None else pd.NA,
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
        .map(str)
        .tolist()
    )


def qa_zarr_path(
    pol: str,
    select_day: str,
    *,
    config: PipelineQAConfig | None = None,
) -> Path:
    """Output Zarr path for pipeline QA products on one observation day."""
    cfg = config or PipelineQAConfig.default()
    day_tag = select_day.replace("-", "")
    stem = cfg.i_qa_zarr_stem if pol == "I" else cfg.v_qa_zarr_stem
    return cfg.zarr_root / f"{stem}-{day_tag}.zarr"


def zarr_status(
    select_day: str,
    *,
    config: PipelineQAConfig | None = None,
) -> dict[str, bool]:
    """Return whether Stokes I/V QA Zarr stores exist for one day."""
    return {
        "I": _zarr_store_exists(qa_zarr_path("I", select_day, config=config)),
        "V": _zarr_store_exists(qa_zarr_path("V", select_day, config=config)),
    }


def default_select_day(
    coverage: pd.DataFrame,
    *,
    config: PipelineQAConfig | None = None,
) -> str | None:
    """Earliest QA day with Stokes I Zarr, else earliest QA day."""
    days = qa_days(coverage)
    if not days:
        return None
    for day in days:
        if qa_zarr_path("I", day, config=config).exists():
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
    populate_subbands_for_day(coverage, select_day)
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


def collect_flux_check_hybrid_paths(
    run_dir: Path,
    *,
    config: PipelineQAConfig | None = None,
) -> list[Path]:
    """Return flux-check CSV paths for one run (per subband or single run-level file)."""
    cfg = config or PipelineQAConfig.default()
    return sorted(run_dir.glob(cfg.flux_check_csv_glob))


def load_flux_check_hybrid_dataframe(
    select_day: str,
    coverage: pd.DataFrame,
    *,
    config: PipelineQAConfig | None = None,
) -> pd.DataFrame:
    """Load and combine all ``flux_check_hybrid.csv`` files for one observation day.

    Each CSV row is augmented with ``lst_hour``, ``frequency_mhz``, ``obs_date``,
    and ``flux_ratio`` (= ``imfit_flux`` / ``model_flux``).
    """
    cfg = config or PipelineQAConfig.default()
    chunks: list[pd.DataFrame] = []
    for _, row in day_rows(select_day, coverage).iterrows():
        run_dir = Path(str(row["run_path"]))
        if not run_dir.is_dir():
            continue
        for csv_path in collect_flux_check_hybrid_paths(run_dir, config=cfg):
            chunk = pd.read_csv(csv_path)
            chunk["lst_hour"] = row["lst_hour"]
            chunk["lst_hour_num"] = int(row["lst_hour_num"])
            chunk["obs_date"] = select_day
            chunk["run_path"] = str(run_dir)
            if cfg.flux_check_csv_per_run:
                if "freq" in chunk.columns:
                    chunk["subband"] = chunk["freq"].map(
                        lambda f: f"{float(f):.0f}MHz" if pd.notna(f) else ""
                    )
                else:
                    chunk["subband"] = ""
            else:
                subband = csv_path.parent.parent.name
                chunk["subband"] = subband
            if "freq" in chunk.columns:
                chunk["frequency_mhz"] = chunk["freq"].astype(float)
            elif not cfg.flux_check_csv_per_run:
                chunk["frequency_mhz"] = frequency_mhz_from_subdir(str(chunk["subband"].iloc[0]))
            else:
                chunk["frequency_mhz"] = np.nan
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


def collect_dewarp_summary_paths(
    run_dir: Path,
    *,
    config: PipelineQAConfig | None = None,
) -> list[Path]:
    """Return dewarp summary CSV paths for one run (under ``QA/``)."""
    cfg = config or PipelineQAConfig.default()
    return sorted(run_dir.glob(cfg.dewarp_summary_csv_glob))


def load_dewarp_summary_dataframe(
    select_day: str,
    coverage: pd.DataFrame,
    *,
    config: PipelineQAConfig | None = None,
) -> pd.DataFrame:
    """Load and combine ``*dewarp_summary.csv`` files for one observation day."""
    cfg = config or PipelineQAConfig.default()
    chunks: list[pd.DataFrame] = []
    for _, row in day_rows(select_day, coverage).iterrows():
        run_dir = Path(str(row["run_path"]))
        if not run_dir.is_dir():
            continue
        for csv_path in collect_dewarp_summary_paths(run_dir, config=cfg):
            chunk = pd.read_csv(csv_path, usecols=[DEWARP_FREQ_COLUMN, DEWARP_SHIFT_COLUMN])
            chunks.append(
                pd.DataFrame(
                    {
                        "lst_hour": row["lst_hour"],
                        "lst_hour_num": int(row["lst_hour_num"]),
                        "obs_date": select_day,
                        "run_path": str(run_dir),
                        "frequency_mhz": pd.to_numeric(
                            chunk[DEWARP_FREQ_COLUMN], errors="coerce"
                        ),
                        "median_shift": pd.to_numeric(
                            chunk[DEWARP_SHIFT_COLUMN], errors="coerce"
                        ),
                    }
                )
            )

    if not chunks:
        return pd.DataFrame(
            columns=[
                "lst_hour",
                "lst_hour_num",
                "obs_date",
                "run_path",
                "frequency_mhz",
                "median_shift",
            ]
        )
    return pd.concat(chunks, ignore_index=True).sort_values(
        ["lst_hour_num", "frequency_mhz"]
    ).reset_index(drop=True)


def dewarp_shift_grid(dewarp_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot median dewarp shift into an LST × frequency grid."""
    if dewarp_df.empty:
        return pd.DataFrame()
    return (
        dewarp_df.pivot_table(
            index="lst_hour_num",
            columns="frequency_mhz",
            values="median_shift",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )


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


def collect_pol_fits(
    select_day: str,
    pol: str,
    coverage: pd.DataFrame,
    *,
    config: PipelineQAConfig | None = None,
) -> list[Path]:
    """Collect deep pbcorr FITS for one Stokes parameter across all hours."""
    cfg = config or PipelineQAConfig.default()
    glob_pattern = cfg.i_fits_glob if pol == "I" else cfg.v_fits_glob
    fits_paths: list[Path] = []
    for run_path in day_rows(select_day, coverage)["run_path"]:
        run_dir = Path(run_path)
        fits_paths.extend(sorted(run_dir.glob(f"*/{pol}/deep/{glob_pattern}")))
    return fits_paths


def infer_target_size_from_82mhz(
    select_day: str,
    coverage: pd.DataFrame,
    *,
    config: PipelineQAConfig | None = None,
) -> int:
    """Return the square pixel size of a reference Stokes I deep image for this day."""
    cfg = config or PipelineQAConfig.default()
    ref_globs = (
        f"{cfg.ref_subband}/I/deep/{cfg.i_fits_glob}",
        f"*/I/deep/{cfg.i_fits_glob}",
    )
    for ref_glob in ref_globs:
        for run_path in day_rows(select_day, coverage)["run_path"]:
            matches = sorted(Path(run_path).glob(ref_glob))
            if not matches:
                continue
            with fits.open(matches[0]) as hdul:
                shape = hdul[0].data.shape
                return int(max(shape[-2], shape[-1]))
    msg = f"No Stokes I deep reference image found for {select_day}"
    raise FileNotFoundError(msg)


def _convert_progress_callback(log: LogFn) -> Callable[[str, int, int, str], None]:
    """Build a progress callback that writes conversion status to the QA activity log."""

    def _callback(stage: str, current: int, total: int, message: str) -> None:
        del stage
        log(f"{message} ({current}/{total})")

    return _callback


def _remove_staging_dir(path: Path, log: LogFn) -> None:
    """Remove a FITS symlink or fixed-header staging directory after Zarr conversion."""
    if not path.exists():
        return
    shutil.rmtree(path)
    log(f"Removed staging directory {path}")


def stage_symlinks(fits_paths: Sequence[Path], staging_dir: Path) -> Path:
    """Symlink FITS files into a flat staging directory."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    for src in fits_paths:
        link_name = src.name
        if link_name in seen_names:
            run_name = src.parents[3].name if len(src.parents) > 3 else src.parent.name
            link_name = f"{run_name}__{src.name}"
        seen_names.add(link_name)
        dst = staging_dir / link_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    return staging_dir


def fits_group_key(path: Path) -> tuple[str, str]:
    name = path.name
    if "__" in name:
        name = name.split("__", 1)[1]
    match = FITS_KEY_PATTERN.match(name)
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
    *,
    config: PipelineQAConfig | None = None,
) -> dict[str, Path]:
    """Convert only missing Stokes I/V Zarr stores for one observation day."""
    if day_rows(select_day, coverage).empty:
        msg = f"No Wideband-qualified runs for {select_day}"
        raise RuntimeError(msg)

    cfg = config or PipelineQAConfig.default()

    zarr_paths: dict[str, Path] = {
        "I": qa_zarr_path("I", select_day, config=cfg),
        "V": qa_zarr_path("V", select_day, config=cfg),
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
        target_size = infer_target_size_from_82mhz(select_day, coverage, config=cfg)
        log(f"LM reference target size from {cfg.ref_subband}: {target_size} px")
        fits_by_pol = {
            "I": collect_pol_fits(select_day, "I", coverage, config=cfg),
            "V": collect_pol_fits(select_day, "V", coverage, config=cfg),
        }
        for pol, paths in fits_by_pol.items():
            log(f"Stokes {pol}: {len(paths)} FITS files")

    if not zarr_paths["I"].exists():
        assert fits_by_pol is not None
        i_paths = fits_by_pol["I"]
        if not i_paths:
            msg = f"No Stokes I FITS found for {select_day}"
            raise FileNotFoundError(msg)

        staging_i = cfg.symlink_root / f"{cfg.i_qa_zarr_stem}-{day_tag}-fits"
        fixed_i = cfg.symlink_root / f"{cfg.i_qa_zarr_stem}-{day_tag}-fixed"
        stage_symlinks(i_paths, staging_i)
        log(f"Staged {len(i_paths)} Stokes I files -> {staging_i}")
        log(f"Converting Stokes I -> {zarr_paths['I'].name} …")
        progress = _convert_progress_callback(log)

        zarr_paths["I"] = convert_fits_dir_to_zarr(
            input_dir=staging_i,
            out_dir=cfg.zarr_root,
            zarr_name=zarr_paths["I"].name,
            fixed_dir=fixed_i,
            chunk_lm=1024,
            rebuild=True,
            lm_reference_target_size=target_size,
            progress_callback=progress,
        )
        log(f"Wrote {zarr_paths['I']}")
        _remove_staging_dir(staging_i, log)
        _remove_staging_dir(fixed_i, log)
    else:
        log(f"Using existing Stokes I Zarr: {zarr_paths['I']}")

    lm_ref_ds = _lm_reference_from_existing_zarr(zarr_paths["I"])

    if not zarr_paths["V"].exists():
        assert fits_by_pol is not None
        v_paths = fits_by_pol["V"]
        if not v_paths:
            msg = f"No Stokes V FITS found for {select_day}"
            raise FileNotFoundError(msg)

        staging_v = cfg.symlink_root / f"{cfg.v_qa_zarr_stem}-{day_tag}-fits"
        fixed_v = cfg.symlink_root / f"{cfg.v_qa_zarr_stem}-{day_tag}-fixed"
        stage_v_fits_with_beam_from_i(v_paths, fits_by_pol["I"], staging_v)
        log(f"Staged {len(v_paths)} Stokes V files -> {staging_v}")
        log(f"Converting Stokes V -> {zarr_paths['V'].name} …")
        progress = _convert_progress_callback(log)

        zarr_paths["V"] = convert_fits_dir_to_zarr(
            input_dir=staging_v,
            out_dir=cfg.zarr_root,
            zarr_name=zarr_paths["V"].name,
            fixed_dir=fixed_v,
            chunk_lm=1024,
            rebuild=True,
            lm_reference_ds=lm_ref_ds,
            progress_callback=progress,
        )
        log(f"Wrote {zarr_paths['V']}")
        _remove_staging_dir(staging_v, log)
        _remove_staging_dir(fixed_v, log)
    else:
        log(f"Using existing Stokes V Zarr: {zarr_paths['V']}")

    return zarr_paths


def load_qa_datasets(
    select_day: str,
    log: LogFn,
    *,
    flush: Callable[[], None] | None = None,
    config: PipelineQAConfig | None = None,
) -> dict[str, xr.Dataset]:
    """Load available Stokes I/V datasets for one day."""
    status = zarr_status(select_day, config=config)
    datasets: dict[str, xr.Dataset] = {}
    for pol, available in status.items():
        if not available:
            continue
        zarr_path = qa_zarr_path(pol, select_day, config=config)
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
