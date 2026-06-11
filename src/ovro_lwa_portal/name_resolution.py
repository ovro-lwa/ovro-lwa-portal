"""Resolve sky coordinates from degree strings or source names."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import astropy.units as u
import requests
from astropy.coordinates import SkyCoord

NED_OBJECT_LOOKUP_URL = "https://ned.ipac.caltech.edu/srs/ObjectLookup"

ResolverName = Literal["degrees", "j2000_name", "from_name", "ned"]

# Survey-style names: optional catalog prefix, J2000 marker, RA hhmmss[.s], Dec [sign]ddmmss[.s].
_J2000_NAME_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"[\w./+-]+\s+"
    r")?"
    r"J"
    r"(?P<ra_h>\d{2})(?P<ra_m>\d{2})(?P<ra_s>\d{2}(?:\.\d+)?)"
    r"(?P<dec_sign>[+-])"
    r"(?P<dec_d>\d{2})(?P<dec_m>\d{2})(?P<dec_s>\d{2}(?:\.\d+)?)"
    r"\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CoordinateResolution:
    """A successfully resolved ICRS sky position."""

    coord: SkyCoord
    resolver: ResolverName
    canonical_name: str | None = None


def resolve_coordinate_string(
    coord_str: str,
    *,
    use_ned_fallback: bool = True,
    ned_timeout: float = 10.0,
    known_source_names: Collection[str] | None = None,
) -> tuple[CoordinateResolution | None, list[str]]:
    """Resolve ``coord_str`` as ICRS degrees or a source name.

    Resolution order: explicit (RA, Dec) degrees; survey-style ``J`` sexagesimal
    names (when not listed in ``known_source_names``); :func:`~astropy.coordinates.SkyCoord.from_name`;
    then NED ObjectLookup when ``use_ned_fallback`` is true.

    Returns
    -------
    resolution
        The resolved coordinate and metadata, or ``None`` on failure.
    messages
        Log lines (warnings, resolver notes) for activity logs.
    """
    text = coord_str.strip()
    if not text:
        return None, []

    has_letters = bool(re.search(r"[A-Za-z]", text))
    coord = _try_degree_pair(text)
    if coord is not None and not has_letters:
        return CoordinateResolution(coord=coord, resolver="degrees"), []

    if not _is_known_source_name(text, known_source_names):
        coord = _try_j2000_embedded_sexagesimal(text)
        if coord is not None:
            return CoordinateResolution(coord=coord, resolver="j2000_name"), []

    try:
        coord = SkyCoord.from_name(text)
    except Exception:
        coord = None
    else:
        return CoordinateResolution(coord=coord, resolver="from_name"), []

    if not use_ned_fallback:
        return None, [
            f"WARNING: Could not resolve {text!r} via Simbad/name databases.",
        ]

    ned_coord, ned_messages, canonical = resolve_via_ned(text, timeout=ned_timeout)
    if ned_coord is None:
        return None, ned_messages

    return (
        CoordinateResolution(
            coord=ned_coord,
            resolver="ned",
            canonical_name=canonical,
        ),
        ned_messages,
    )


def resolve_via_ned(
    name: str,
    *,
    timeout: float = 10.0,
) -> tuple[SkyCoord | None, list[str], str | None]:
    """Resolve a name with NED ObjectLookup.

    Only ``ResultCode == 3`` returns a coordinate. Ambiguous names (code 1) fail
    strictly with alias hints in the message list.
    """
    try:
        payload = _fetch_ned_object_lookup(name, timeout=timeout)
    except requests.RequestException as exc:
        return None, [f"WARNING: NED request failed for {name!r}: {exc}"], None

    result_code = payload.get("ResultCode")
    if result_code == 3:
        position = payload["Preferred"]["Position"]
        canonical = _ned_canonical_from_payload(payload)
        coord = SkyCoord(
            ra=float(position["RA"]) * u.deg,
            dec=float(position["Dec"]) * u.deg,
            frame="icrs",
        )
        note = f"Resolved via NED ({canonical})." if canonical else "Resolved via NED."
        return coord, [note], canonical

    if result_code == 1:
        aliases = payload.get("Interpreted", {}).get("Aliases") or []
        return None, [_format_ned_ambiguous_message(name, aliases)], None

    if result_code == 2:
        interpreted = payload.get("Interpreted", {}).get("Name")
        detail = f" (interpreted as {interpreted!r})" if interpreted else ""
        return None, [f"WARNING: NED parsed {name!r}{detail} but the object is unknown."], None

    if result_code == 0:
        return None, [f"WARNING: NED could not interpret {name!r} as an object name."], None

    return None, [f"WARNING: NED lookup failed for {name!r} (ResultCode={result_code})."], None


def format_icrs_degree_pair(ra_deg: float, dec_deg: float, *, precision: int = 4) -> str:
    """Format ICRS RA/Dec as a decimal-degree pair for coordinate entry.

    The returned string round-trips through :func:`resolve_coordinate_string`
    when parsed as a degree pair (no letters in the string).
    """
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        msg = f"RA/Dec must be finite, got RA={ra_deg}, Dec={dec_deg}"
        raise ValueError(msg)
    if not (-90.0 <= dec_deg <= 90.0):
        msg = f"Dec must be in [-90, 90]°, got {dec_deg}"
        raise ValueError(msg)
    return f"{ra_deg:.{precision}f}, {dec_deg:.{precision}f}"


def _try_degree_pair(text: str) -> SkyCoord | None:
    cleaned = text.strip().strip("()[]")
    match = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[,;\s]+\s*"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
        cleaned,
    )
    if match is None:
        return None
    try:
        ra_deg = float(match.group(1))
        dec_deg = float(match.group(2))
    except ValueError:
        return None
    if not (-360.0 <= ra_deg <= 360.0 and -90.0 <= dec_deg <= 90.0):
        return None
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")


def _is_known_source_name(text: str, known_source_names: Collection[str] | None) -> bool:
    if known_source_names is None:
        return False
    key = text.strip().casefold()
    return any(name.strip().casefold() == key for name in known_source_names)


def _try_j2000_embedded_sexagesimal(text: str) -> SkyCoord | None:
    """Parse survey-style ``J`` names as J2000 ICRS hhmmss[.s] ± ddmmss[.s]."""
    match = _J2000_NAME_RE.match(text.strip())
    if match is None:
        return None

    parts = match.groupdict()
    try:
        ra_h = int(parts["ra_h"])
        ra_m = int(parts["ra_m"])
        ra_s = float(parts["ra_s"])
        dec_d = int(parts["dec_d"])
        dec_m = int(parts["dec_m"])
        dec_s = float(parts["dec_s"])
    except ValueError:
        return None

    if not (0 <= ra_h <= 23 and 0 <= ra_m <= 59 and 0.0 <= ra_s < 60.0):
        return None
    if not (0 <= dec_d <= 90 and 0 <= dec_m <= 59 and 0.0 <= dec_s < 60.0):
        return None

    dec_sign = parts["dec_sign"]
    ra_str = f"{ra_h}:{ra_m}:{ra_s}"
    dec_str = f"{dec_sign}{dec_d}:{dec_m}:{dec_s}"
    try:
        return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
    except (ValueError, u.UnitConversionError):
        return None


def _format_ned_ambiguous_message(name: str, aliases: list[str]) -> str:
    preview = ", ".join(str(alias) for alias in aliases[:8])
    if len(aliases) > 8:
        preview = f"{preview}, … ({len(aliases)} aliases total)"
    return (
        f"WARNING: NED ambiguous name {name!r}; specify a unique alias"
        f"{f': {preview}' if preview else ''}."
    )


@lru_cache(maxsize=256)
def _fetch_ned_object_lookup(name: str, *, timeout: float) -> dict[str, object]:
    response = requests.post(
        NED_OBJECT_LOOKUP_URL,
        data={"json": json.dumps({"name": {"v": name}})},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        msg = f"unexpected NED response type: {type(data).__name__}"
        raise requests.RequestException(msg)
    return data


def _ned_canonical_from_payload(payload: dict[str, object]) -> str | None:
    preferred = payload.get("Preferred")
    if isinstance(preferred, dict):
        canonical = preferred.get("Name")
        if isinstance(canonical, str) and canonical.strip():
            return canonical.strip()
    interpreted = payload.get("Interpreted")
    if isinstance(interpreted, dict):
        canonical = interpreted.get("Name")
        if isinstance(canonical, str) and canonical.strip():
            return canonical.strip()
    return None
