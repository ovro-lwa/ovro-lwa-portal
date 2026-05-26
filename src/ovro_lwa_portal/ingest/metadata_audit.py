"""Audit FITS header metadata consistency across subbands for one observation time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from astropy.io import fits
from astropy.time import Time
from rich.console import Console
from rich.table import Table

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DISCOVERY_FREQ_BIN_HZ,
    _canonical_stack_frequency_hz,
    _combine_time_step,
    _obstime_from_fits_filename,
    _time_key_from_filename,
    _time_key_from_header,
)
from ovro_lwa_portal.ingest.discovery import (
    IngestDiscoveryConfig,
    discover_time_grouped_fits,
)

__all__ = [
    "CombineProbeResult",
    "TimeGroupAuditReport",
    "audit_directory",
    "audit_time_group_files",
    "print_audit_reports",
]

# Header keys that must match across every subband in one observation.
_SUBBAND_CONSISTENT_FIELDS: tuple[str, ...] = (
    "DATE-OBS",
    "TIME-OBS",
    "TIMESYS",
    "MJD-OBS",
    "SPECSYS",
)

# Reported for context; variation per subband is normal.
_SPECTRAL_FIELDS: tuple[str, ...] = ("RESTFREQ", "RESTFRQ", "CRVAL3", "CTYPE3")
_SHAPE_FIELDS: tuple[str, ...] = ("NAXIS1", "NAXIS2")
_BEAM_FIELDS: tuple[str, ...] = ("BMAJ", "BMIN", "BPA")


@dataclass(frozen=True)
class FieldSummary:
    """Summary of one FITS keyword across subbands."""

    name: str
    n_files: int
    n_unique: int
    consistent: bool
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimeKeySummary:
    """Filename vs header observation-time keys."""

    discovery_time_key: str
    filename_keys: frozenset[str]
    header_keys: frozenset[str]
    filename_header_agree: bool
    header_mjd_spread_s: float
    max_filename_header_delta_s: float | None


@dataclass(frozen=True)
class CombineProbeResult:
    """Optional result from :func:`_combine_time_step` on the audited files."""

    velocity_dims: tuple[str, ...] | None = None
    sky_dims: tuple[str, ...] | None = None
    time_size: int | None = None
    time_mjd_values: tuple[float, ...] = ()
    per_file_mjd_spread_s: float | None = None
    error: str | None = None


@dataclass
class TimeGroupAuditReport:
    """Audit of one observation-time group."""

    time_key: str
    label: str
    paths: list[Path]
    field_summaries: list[FieldSummary] = field(default_factory=list)
    time_keys: TimeKeySummary | None = None
    spectral_note: str = ""
    shape_note: str = ""
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    combine: CombineProbeResult | None = None

    @property
    def n_files(self) -> int:
        return len(self.paths)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


def _header_mjd(hdr: fits.Header) -> float | None:
    mjd_obs = hdr.get("MJD-OBS")
    if mjd_obs is not None:
        try:
            return float(mjd_obs)
        except (TypeError, ValueError):
            pass
    tk = _time_key_from_header(hdr)
    if tk is None:
        return None
    try:
        return float(Time.strptime(tk, "%Y%m%d_%H%M%S", scale="utc").mjd)
    except Exception:
        return None


def _format_header_value(value: object) -> str:
    if value is None:
        return "<missing>"
    return repr(value)


def _summarize_field(paths: Sequence[Path], key: str) -> FieldSummary:
    vals: list[str] = []
    for fp in paths:
        hdr = fits.getheader(fp, ext=0)
        vals.append(_format_header_value(hdr.get(key)))
    unique = tuple(sorted(set(vals)))
    return FieldSummary(
        name=key,
        n_files=len(paths),
        n_unique=len(unique),
        consistent=len(unique) == 1,
        values=unique if len(unique) <= 8 else (*unique[:8], f"... +{len(unique) - 8} more"),
    )


def _audit_time_keys(paths: Sequence[Path], discovery_time_key: str) -> TimeKeySummary:
    filename_keys: set[str] = set()
    header_keys: set[str] = set()
    header_mjds: list[float] = []
    deltas: list[float] = []

    for fp in paths:
        hdr = fits.getheader(fp, ext=0)
        tk_fn = _time_key_from_filename(fp)
        tk_hdr = _time_key_from_header(hdr)
        if tk_fn is not None:
            filename_keys.add(tk_fn)
        if tk_hdr is not None:
            header_keys.add(tk_hdr)
        mjd_hdr = _header_mjd(hdr)
        if mjd_hdr is not None:
            header_mjds.append(mjd_hdr)
        obstime_fn = _obstime_from_fits_filename(fp)
        if obstime_fn is not None and mjd_hdr is not None:
            deltas.append(abs(float(obstime_fn.mjd) - mjd_hdr) * 86400.0)

    spread_s = 0.0
    if header_mjds:
        spread_s = (max(header_mjds) - min(header_mjds)) * 86400.0

    agree = filename_keys == header_keys and len(filename_keys) == 1
    return TimeKeySummary(
        discovery_time_key=discovery_time_key,
        filename_keys=frozenset(filename_keys),
        header_keys=frozenset(header_keys),
        filename_header_agree=agree,
        header_mjd_spread_s=spread_s,
        max_filename_header_delta_s=max(deltas) if deltas else None,
    )


def _spectral_note(paths: Sequence[Path], *, group_metadata_source: Literal["fits", "filename"]) -> str:
    rest_vals: set[str] = set()
    crval3: set[str] = set()
    canon_mhz: list[float] = []
    for fp in paths:
        hdr = fits.getheader(fp, ext=0)
        rest_vals.add(_format_header_value(hdr.get("RESTFREQ")))
        crval3.add(_format_header_value(hdr.get("CRVAL3")))
        hz = _canonical_stack_frequency_hz(fp, group_metadata_source=group_metadata_source)
        if hz is not None:
            canon_mhz.append(float(hz) / 1e6)
    parts = [
        f"RESTFREQ unique={len(rest_vals)}",
        f"CRVAL3 unique={len(crval3)}",
    ]
    if canon_mhz:
        parts.append(
            f"canonical stack labels (MHz): {', '.join(f'{v:.3f}' for v in sorted(canon_mhz))}"
        )
    return "; ".join(parts)


def _shape_note(paths: Sequence[Path]) -> str:
    shapes: set[tuple[int, int]] = set()
    for fp in paths:
        hdr = fits.getheader(fp, ext=0)
        n1 = int(hdr.get("NAXIS1", 0))
        n2 = int(hdr.get("NAXIS2", 0))
        shapes.add((n1, n2))
    if len(shapes) == 1:
        n1, n2 = next(iter(shapes))
        return f"uniform NAXIS1×NAXIS2 = {n1}×{n2}"
    return f"mixed NAXIS1×NAXIS2 shapes: {sorted(shapes)}"


def _probe_combine(
    paths: Sequence[Path],
    fixed_dir: Path,
    *,
    group_metadata_source: Literal["fits", "filename"],
) -> CombineProbeResult:
    try:
        xds_t, _, _ = _combine_time_step(
            list(paths),
            fixed_dir,
            chunk_lm=0,
            fix_headers_on_demand=True,
            group_metadata_source=group_metadata_source,
        )
    except Exception as exc:
        return CombineProbeResult(error=str(exc))

    mjds = [float(v) for v in np.asarray(xds_t["time"].values).ravel()]
    spread_s = (max(mjds) - min(mjds)) * 86400.0 if mjds else None
    return CombineProbeResult(
        velocity_dims=tuple(xds_t["velocity"].dims),
        sky_dims=tuple(xds_t["SKY"].dims),
        time_size=int(xds_t.sizes.get("time", 0)),
        time_mjd_values=tuple(mjds),
        per_file_mjd_spread_s=spread_s,
    )


def audit_time_group_files(
    paths: Sequence[Path],
    time_key: str,
    *,
    label: str = "input",
    group_metadata_source: Literal["fits", "filename"] = "fits",
    probe_combine: bool = False,
    fixed_dir: Path | None = None,
    warn_filename_header_delta_s: float = 1.0,
) -> TimeGroupAuditReport:
    """Audit header metadata consistency for one observation-time file set."""
    sorted_paths = sorted(paths, key=lambda p: p.name)
    report = TimeGroupAuditReport(time_key=time_key, label=label, paths=list(sorted_paths))

    if not sorted_paths:
        report.issues.append("no FITS files in this group")
        return report

    for key in _SUBBAND_CONSISTENT_FIELDS:
        summary = _summarize_field(sorted_paths, key)
        report.field_summaries.append(summary)
        if not summary.consistent:
            report.issues.append(
                f"{key} differs across subbands ({summary.n_unique} distinct values)"
            )

    report.time_keys = _audit_time_keys(sorted_paths, time_key)
    tk = report.time_keys
    if not tk.filename_header_agree:
        delta = tk.max_filename_header_delta_s
        delta_s = f"{delta:.3f}" if delta is not None else "unknown"
        report.warnings.append(
            "filename time key "
            f"{sorted(tk.filename_keys)} differs from header DATE-OBS key "
            f"{sorted(tk.header_keys)} (max delta {delta_s} s)"
        )
        if delta is not None and delta > warn_filename_header_delta_s:
            report.warnings.append(
                f"filename vs header time offset exceeds {warn_filename_header_delta_s:g} s "
                f"({delta:.3f} s); Zarr uses header MJD via xradio while discovery uses filename"
            )
    if tk.header_mjd_spread_s > 0.0:
        report.issues.append(
            f"header time MJD differs across subbands (spread {tk.header_mjd_spread_s:.6f} s)"
        )

    report.spectral_note = _spectral_note(sorted_paths, group_metadata_source=group_metadata_source)
    report.shape_note = _shape_note(sorted_paths)

    if probe_combine:
        combine_dir = fixed_dir if fixed_dir is not None else sorted_paths[0].parent
        report.combine = _probe_combine(
            sorted_paths,
            combine_dir,
            group_metadata_source=group_metadata_source,
        )
        if report.combine.error:
            report.warnings.append(f"combine probe failed: {report.combine.error}")
        elif report.combine.time_size not in (None, 1):
            report.issues.append(
                f"combine produced time dimension size {report.combine.time_size} "
                "(expected 1 for one observation)"
            )
        elif (
            report.combine.per_file_mjd_spread_s is not None
            and report.combine.per_file_mjd_spread_s > 0.0
        ):
            report.issues.append(
                "combine time coordinate spread "
                f"{report.combine.per_file_mjd_spread_s:.6f} s across subbands"
            )

    return report


def audit_directory(
    input_dir: Path,
    time_keys: Sequence[str] | None = None,
    *,
    staging_dir: Path | None = None,
    group_metadata_source: Literal["fits", "filename"] = "fits",
    time_key_source: Literal["header", "filename"] = "filename",
    discovery_freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
    probe_combine: bool = False,
    fixed_dir: Path | None = None,
    warn_filename_header_delta_s: float = 1.0,
) -> list[TimeGroupAuditReport]:
    """Discover and audit observation-time groups under *input_dir*."""
    input_dir = Path(input_dir)
    discovery = IngestDiscoveryConfig(
        freq_bin_hz=discovery_freq_bin_hz,
        group_metadata_source=group_metadata_source,
        time_key_source=time_key_source,
    )
    by_time = discover_time_grouped_fits(input_dir, discovery=discovery)
    if not by_time:
        msg = f"No groupable FITS files found in {input_dir}"
        raise FileNotFoundError(msg)

    selected = sorted(by_time.keys()) if time_keys is None else list(time_keys)
    missing = [k for k in selected if k not in by_time]
    if missing:
        msg = f"Time key(s) not found in {input_dir}: {', '.join(missing)}"
        raise KeyError(msg)

    reports: list[TimeGroupAuditReport] = []
    for tkey in selected:
        reports.append(
            audit_time_group_files(
                by_time[tkey],
                tkey,
                label="input",
                group_metadata_source=group_metadata_source,
                probe_combine=probe_combine,
                fixed_dir=fixed_dir,
                warn_filename_header_delta_s=warn_filename_header_delta_s,
            )
        )
        if staging_dir is not None:
            staged = sorted(Path(staging_dir).glob(f"{tkey}__*.fits"))
            if staged:
                reports.append(
                    audit_time_group_files(
                        staged,
                        tkey,
                        label="staging",
                        group_metadata_source=group_metadata_source,
                        probe_combine=probe_combine,
                        fixed_dir=fixed_dir,
                        warn_filename_header_delta_s=warn_filename_header_delta_s,
                    )
                )
    return reports


def print_audit_reports(
    reports: Sequence[TimeGroupAuditReport],
    *,
    console: Console | None = None,
) -> None:
    """Print human-readable audit output with Rich."""
    out = console or Console()
    for report in reports:
        out.print()
        out.print(f"[bold cyan]Time {report.time_key}[/bold cyan] ({report.label}, {report.n_files} files)")
        if report.n_files == 0:
            for issue in report.issues:
                out.print(f"  [red]ISSUE[/red] {issue}")
            continue

        table = Table(show_header=True, header_style="bold")
        table.add_column("Keyword")
        table.add_column("Consistent")
        table.add_column("Unique")
        table.add_column("Values")
        for fs in report.field_summaries:
            status = "yes" if fs.consistent else "no"
            style = "green" if fs.consistent else "red"
            table.add_row(fs.name, f"[{style}]{status}[/{style}]", str(fs.n_unique), ", ".join(fs.values))
        out.print(table)

        if report.time_keys is not None:
            tk = report.time_keys
            out.print(
                f"  Discovery key: {tk.discovery_time_key}; "
                f"filename keys: {sorted(tk.filename_keys)}; "
                f"header keys: {sorted(tk.header_keys)}"
            )
            if tk.max_filename_header_delta_s is not None:
                out.print(
                    f"  Max filename↔header Δt: {tk.max_filename_header_delta_s:.3f} s; "
                    f"header MJD spread: {tk.header_mjd_spread_s:.6f} s"
                )

        out.print(f"  Spectral: {report.spectral_note}")
        out.print(f"  Shape: {report.shape_note}")

        if report.combine is not None and report.combine.error is None:
            cb = report.combine
            out.print(
                f"  Combine probe: velocity{cb.velocity_dims} SKY{cb.sky_dims} "
                f"time_size={cb.time_size} MJD={list(cb.time_mjd_values)}"
            )
        elif report.combine is not None and report.combine.error:
            out.print(f"  Combine probe: [yellow]failed[/yellow] ({report.combine.error})")

        for issue in report.issues:
            out.print(f"  [red]ISSUE[/red] {issue}")
        for warning in report.warnings:
            out.print(f"  [yellow]WARN[/yellow] {warning}")
        if not report.issues and not report.warnings:
            out.print("  [green]No subband metadata issues detected.[/green]")
