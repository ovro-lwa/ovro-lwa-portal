"""Read ExoPipe transient-review HTML into a candidate table.

The review apps are separate HTTP servers. This module parses their index
table and the combined-position field on ``/source/{id}``. It does not import
the ExoPipe generator.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "stokes",
    "source",
    "base_url",
    "rank",
    "date",
    "bands",
    "peak_jy",
    "detections",
    "snr",
    "elevation",
    "n_epoch",
    "night_noise",
    "tess",
    "verdict",
    "fp_reason",
    "cross_match",
    "note",
)

_NUMERIC_COLUMNS = ("peak_jy", "detections", "snr", "elevation")
# Rank and epoch counts stay text in the index parse. Range filters coerce them.
RANGE_COLUMNS: tuple[str, ...] = (
    "rank",
    "peak_jy",
    "detections",
    "snr",
    "elevation",
    "n_epoch",
)
DATE_RANGE_COLUMNS: tuple[str, ...] = ("date",)
_BAND_MHZ_RE = re.compile(r"^(\d+(?:\.\d+)?)MHz$", re.IGNORECASE)
_LST_HOUR_RE = re.compile(r"^(\d{1,2})h$", re.IGNORECASE)
_DETECTION_HEADERS = ("lst", "date", "band", "snr", "ra", "dec")

_HEADER_TO_COLUMN = {
    "#": "rank",
    "source": "source",
    "date": "date",
    "bands": "bands",
    "pb-corr peak (jy)": "peak_jy",
    "detections": "detections",
    "snr": "snr",
    "elevation": "elevation",
    "n_i_epoch": "n_epoch",
    "n_v_epoch": "n_epoch",
    "night noise (x cell)": "night_noise",
    "tess": "tess",
    "verdict": "verdict",
    "fp reason": "fp_reason",
    "cross-match": "cross_match",
    "note": "note",
}


@dataclass(frozen=True)
class SourceDetection:
    """One row of the source page's ``Detections of this source`` table."""

    lst_hour: int
    on_date: date
    band: str
    snr: float
    ra_deg: float
    dec_deg: float


def parse_source_detections(html: str) -> list[SourceDetection]:
    """Read per-detection LST, date, band, and SNR from a source page.

    The index table has one date and a list of bands. This table is the
    specific sample: ``15h``, ``2025-05-05``, ``32MHz``.
    """
    parser = _DetectionTableParser()
    parser.feed(html)
    parser.close()
    detections: list[SourceDetection] = []
    for cells in parser.rows:
        parsed = _detection_from_cells(cells)
        if parsed is not None:
            detections.append(parsed)
    return detections


def peak_detection(detections: Sequence[SourceDetection]) -> SourceDetection:
    """Highest-SNR detection. That row is the one the index date refers to."""
    if len(detections) == 0:
        msg = "Source page has no detection rows."
        raise ValueError(msg)
    return max(detections, key=lambda item: item.snr)


def subband_mhz_for_overlay(
    peak: SourceDetection,
    detections: Sequence[SourceDetection],
) -> tuple[float | None, str]:
    """Frequency channel for the overlay, and a short explanation.

    ``32MHz`` is that channel. ``Full``, ``Red``, ``Green``, and ``Blue`` are
    wideband images, so the overlay uses the brightest ``NNMHz`` detection at
    the same date and LST. ``None`` means no subband shares that time.
    """
    own = _band_mhz(peak.band)
    if own is not None:
        return own, f"{peak.band} at {peak.on_date.isoformat()} {peak.lst_hour:02d}h"
    same_time = [
        item
        for item in detections
        if item.on_date == peak.on_date
        and item.lst_hour == peak.lst_hour
        and _band_mhz(item.band) is not None
    ]
    if same_time:
        best = max(same_time, key=lambda item: item.snr)
        mhz = _band_mhz(best.band)
        return (
            mhz,
            (
                f"{peak.band} is wideband; overlay uses {best.band} "
                f"(SNR {best.snr:.1f}) at {peak.on_date.isoformat()} {peak.lst_hour:02d}h"
            ),
        )
    return (
        None,
        (
            f"{peak.band} at {peak.on_date.isoformat()} {peak.lst_hour:02d}h "
            "has no subband detection"
        ),
    )


def displayed_columns(columns: Sequence[str] | None = None) -> tuple[str, ...]:
    """Columns shown in the candidate table. ``base_url`` stays hidden."""
    names = CANDIDATE_COLUMNS if columns is None else columns
    return tuple(column for column in names if column != "base_url")


def is_range_column(column: str) -> bool:
    """True when a displayed column filters with an inclusive min and max."""
    return column in RANGE_COLUMNS or column in DATE_RANGE_COLUMNS


def filter_candidates(
    frame: pd.DataFrame,
    *,
    contains: Mapping[str, str] | None = None,
    ranges: Mapping[str, tuple[object, object]] | None = None,
) -> pd.DataFrame:
    """Keep rows that match every displayed-column filter.

    ``contains`` is a case-insensitive substring for text columns. A blank
    string does not filter. ``ranges`` is an inclusive ``(low, high)`` for
    numeric columns and ``date``. Either side may be ``None`` to leave that
    end open. The result keeps ``frame.index``, so a Tabulator selection can
    still be read with ``iloc`` on the unfiltered table.

    Raises
    ------
    ValueError
        When a column is hidden, missing, or paired with the other filter kind,
        or when a bound is not a number or an ISO date.
    """
    text_filters = {
        column: text.strip()
        for column, text in (contains or {}).items()
        if text is not None and str(text).strip()
    }
    range_filters: dict[str, tuple[float | date | None, float | date | None]] = {}
    for column, bounds in (ranges or {}).items():
        _require_filter_column(column, frame.columns, kind="range")
        low, high = _range_bounds(column, bounds)
        if low is None and high is None:
            continue
        range_filters[column] = (low, high)
    for column in text_filters:
        _require_filter_column(column, frame.columns, kind="contains")
    for column in range_filters:
        _require_filter_column(column, frame.columns, kind="range")
    if not text_filters and not range_filters:
        return frame

    mask = pd.Series(True, index=frame.index)
    for column, text in text_filters.items():
        values = frame[column].astype("string")
        mask &= values.str.contains(text, case=False, regex=False, na=False)
    for column, (low, high) in range_filters.items():
        mask &= _range_mask(frame[column], column, low, high)
    return frame.loc[mask]


def candidate_from_selection(frame: pd.DataFrame, selection: Sequence[int]) -> pd.Series:
    """Return the last selected candidate row.

    ``selection`` is integer positions into ``frame``, matching Panel
    Tabulator's ``selection``. An empty selection raises ``ValueError``.
    """
    if len(selection) == 0:
        msg = "Select a candidate row, then press Load."
        raise ValueError(msg)
    return frame.iloc[int(selection[-1])]


def source_page_url(base_url: str, source_id: str) -> str:
    """URL of one candidate page. ``+`` in the id is encoded as ``%2B``."""
    base = base_url.rstrip("/")
    return f"{base}/source/{quote(source_id, safe='')}"


def parse_candidate_index(html: str, *, stokes: str, base_url: str) -> pd.DataFrame:
    """Build a candidate frame from one index page (``table.idx``).

    Rows come from ``tr[data-sid]``. RA/Dec is not on this page; this function
    does not request ``/source/{id}``.
    """
    parser = _IndexTableParser()
    parser.feed(html)
    parser.close()
    column_for_cell = [_HEADER_TO_COLUMN.get(_header_key(header)) for header in parser.headers]
    origin = base_url.rstrip("/")
    records: list[dict[str, str]] = []
    for source_id, cells in parser.rows:
        record = dict.fromkeys(CANDIDATE_COLUMNS, "")
        record["stokes"] = stokes
        record["source"] = source_id
        record["base_url"] = origin
        for index, cell in enumerate(cells):
            if index >= len(column_for_cell):
                break
            column = column_for_cell[index]
            if column is None or column == "source":
                continue
            record[column] = cell
        records.append(record)
    frame = pd.DataFrame.from_records(records, columns=list(CANDIDATE_COLUMNS))
    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def parse_combined_position(html: str) -> str:
    """Return the first locbar degree pair (``input.lv``).

    Raises
    ------
    ValueError
        When the page has no combined-position field.
    """
    parser = _LocbarParser()
    parser.feed(html)
    parser.close()
    if parser.value is None or not parser.value.strip():
        msg = "ExoPipe source page has no combined position in div.locbar input.lv"
        raise ValueError(msg)
    return parser.value.strip()


def fetch_text(url: str, *, timeout: float = 60.0) -> str:
    """GET ``url`` and decode the body as text.

    Raises
    ------
    URLError
        When the connection fails, including a missing SSH forward.
    """
    with urlopen(url, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset)


def load_candidate_table(
    queues: Mapping[str, str],
    *,
    fetch: Callable[[str], str] = fetch_text,
) -> pd.DataFrame:
    """Fetch each queue index and concatenate them.

    ``queues`` maps a Stokes label (``"I"``, ``"V"``) to the index origin,
    for example ``{"V": "http://127.0.0.1:8766"}``. ``fetch`` is injectable
    so tests do not open a socket. Connection errors propagate as
    :class:`URLError`.
    """
    frames: list[pd.DataFrame] = []
    for stokes, base_url in queues.items():
        origin = base_url.rstrip("/")
        html = fetch(f"{origin}/")
        frames.append(parse_candidate_index(html, stokes=str(stokes), base_url=origin))
    if not frames:
        return pd.DataFrame(columns=list(CANDIDATE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _blank_bound(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _range_bounds(
    column: str,
    bounds: tuple[object, object],
) -> tuple[float | date | None, float | date | None]:
    if len(bounds) != 2:
        msg = f"{column} range must be a (low, high) pair"
        raise ValueError(msg)
    low, high = bounds
    if column in DATE_RANGE_COLUMNS:
        return _date_bound(column, low), _date_bound(column, high)
    return _float_bound(column, low), _float_bound(column, high)


def _float_bound(column: str, value: object) -> float | None:
    if _blank_bound(value):
        return None
    msg = f"{column} range bound {value!r} is not a number"
    try:
        if isinstance(value, bool):
            raise TypeError(msg)
        if isinstance(value, (int, float, str)):
            return float(value)
        raise TypeError(msg)
    except (TypeError, ValueError) as exc:
        raise ValueError(msg) from exc


def _date_bound(column: str, value: object) -> date | None:
    if _blank_bound(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            msg = f"{column} range bound {value!r} is not a YYYY-MM-DD date"
            raise ValueError(msg) from exc
    msg = f"{column} range bound {value!r} is not a YYYY-MM-DD date"
    raise ValueError(msg)


def _require_filter_column(column: str, columns: pd.Index, *, kind: str) -> None:
    if column not in displayed_columns(list(columns)):
        msg = f"{column} is not a displayed candidate column"
        raise ValueError(msg)
    ranged = is_range_column(column)
    if kind == "range" and not ranged:
        msg = f"{column} is not a range column"
        raise ValueError(msg)
    if kind == "contains" and ranged:
        msg = f"{column} uses a min/max range, not a text filter"
        raise ValueError(msg)


def _range_mask(
    values: pd.Series,
    column: str,
    low: float | date | None,
    high: float | date | None,
) -> pd.Series:
    if column in DATE_RANGE_COLUMNS:
        parsed = pd.to_datetime(values, errors="coerce")
        mask = parsed.notna()
        if low is not None:
            mask &= parsed >= pd.Timestamp(low)
        if high is not None:
            mask &= parsed <= pd.Timestamp(high)
        return mask
    parsed_numbers = pd.to_numeric(values, errors="coerce")
    mask = parsed_numbers.notna()
    if low is not None:
        mask &= parsed_numbers >= low
    if high is not None:
        mask &= parsed_numbers <= high
    return mask


def _header_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _class_tokens(attrs: list[tuple[str, str | None]]) -> set[str]:
    for key, value in attrs:
        if key == "class" and value:
            return set(value.split())
    return set()


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for key, value in attrs:
        if key == name:
            return value
    return None


class _IndexTableParser(HTMLParser):
    """Collect ``table.idx`` header labels and ``data-sid`` body rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headers: list[str] = []
        self.rows: list[tuple[str, list[str]]] = []
        self._idx_depth = 0
        self._in_cell = False
        self._cell_chunks: list[str] = []
        self._row_cells: list[str] = []
        self._row_sid: str | None = None
        self._row_is_header = False
        self._in_row = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = _class_tokens(attrs)
        if tag == "table":
            if self._idx_depth or "idx" in classes:
                self._idx_depth += 1
            return
        if not self._idx_depth:
            return
        if tag == "tr":
            self._in_row = True
            self._row_cells = []
            self._row_sid = _attr(attrs, "data-sid")
            self._row_is_header = False
            return
        if self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_chunks = []
            if tag == "th":
                self._row_is_header = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._idx_depth:
            self._idx_depth -= 1
            return
        if not self._idx_depth:
            return
        if tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._cell_chunks).split())
            self._row_cells.append(text)
            self._in_cell = False
            self._cell_chunks = []
            return
        if tag == "tr" and self._in_row:
            if self._row_is_header and not self.headers:
                self.headers = list(self._row_cells)
            elif self._row_sid:
                self.rows.append((self._row_sid, list(self._row_cells)))
            self._in_row = False
            self._row_sid = None
            self._row_cells = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)


def _band_mhz(band: str) -> float | None:
    match = _BAND_MHZ_RE.fullmatch(band.strip())
    if match is None:
        return None
    return float(match.group(1))


def _detection_from_cells(cells: list[str]) -> SourceDetection | None:
    if len(cells) < 6:
        return None
    lst = _LST_HOUR_RE.fullmatch(cells[0].strip())
    if lst is None:
        return None
    hour = int(lst.group(1))
    if hour > 23:
        return None
    try:
        on_date = date.fromisoformat(cells[1].strip())
        snr = float(cells[3].replace(",", ""))
        ra_deg = float(cells[4])
        dec_deg = float(cells[5])
    except ValueError:
        return None
    return SourceDetection(
        lst_hour=hour,
        on_date=on_date,
        band=cells[2].strip(),
        snr=snr,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )


class _DetectionTableParser(HTMLParser):
    """Rows of the detections table (headers LST, date, band, SNR, RA, Dec)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._depth = 0
        self._in_row = False
        self._in_cell = False
        self._cell_chunks: list[str] = []
        self._row_cells: list[str] = []
        self._capture = False
        self._saw_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._capture = False
                self._saw_header = False
            return
        if not self._depth:
            return
        if tag == "tr":
            self._in_row = True
            self._row_cells = []
            return
        if self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._cell_chunks).split())
            self._row_cells.append(text)
            self._in_cell = False
            self._cell_chunks = []
            return
        if tag == "tr" and self._in_row:
            cells = list(self._row_cells)
            keys = tuple(cell.strip().casefold() for cell in cells[:6])
            if not self._saw_header and keys == _DETECTION_HEADERS:
                self._capture = True
                self._saw_header = True
            elif self._capture and cells:
                self.rows.append(cells)
            self._in_row = False
            self._row_cells = []
            return
        if tag == "table" and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)


class _LocbarParser(HTMLParser):
    """First ``input.lv`` value inside ``div.locbar``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.value: str | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = _class_tokens(attrs)
        if self._depth:
            self._depth += 1
            if self.value is None and tag == "input" and "lv" in classes:
                raw = _attr(attrs, "value")
                self.value = "" if raw is None else raw
            return
        if "locbar" in classes:
            self._depth = 1

    def handle_endtag(self, tag: str) -> None:
        del tag
        if self._depth:
            self._depth -= 1


__all__ = [
    "CANDIDATE_COLUMNS",
    "DATE_RANGE_COLUMNS",
    "RANGE_COLUMNS",
    "SourceDetection",
    "URLError",
    "candidate_from_selection",
    "displayed_columns",
    "fetch_text",
    "filter_candidates",
    "is_range_column",
    "load_candidate_table",
    "parse_candidate_index",
    "parse_combined_position",
    "parse_source_detections",
    "peak_detection",
    "source_page_url",
    "subband_mhz_for_overlay",
]
