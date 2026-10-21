"""Read ExoPipe transient-review HTML into a candidate table.

The review apps are separate HTTP servers. This module parses their index
table and the combined-position field on ``/source/{id}``. It does not import
the ExoPipe generator.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import panel as pn
import param
from panel.io.model import JSCode

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
# Rank and epoch counts stay text in the index parse. The header filter coerces them.
_RANGE_FILTER_COLUMNS = (
    "rank",
    "peak_jy",
    "detections",
    "snr",
    "elevation",
    "n_epoch",
)
_DATE_FILTER_COLUMNS = ("date",)
# Tabulator stores a function-typed header filter as ``type: <function>``.
# Bokeh then raises ``[object Function] is not serializable`` and drops the
# next cell click, so Load keeps the previous candidate. A named string type
# serializes. The editor registers that name on the filter module.
_RANGE_FILTER_NAME = "minMax"
_RANGE_HEADER_FILTER = """
function(headerValue, rowValue, rowData, filterParams) {
  if (!headerValue || typeof headerValue !== "object") {
    return true;
  }
  var start = headerValue.start;
  var end = headerValue.end;
  var hasStart = start !== undefined && start !== null && String(start) !== "";
  var hasEnd = end !== undefined && end !== null && String(end) !== "";
  if (!hasStart && !hasEnd) {
    return true;
  }
  if (rowValue === null || rowValue === undefined || rowValue === "") {
    return false;
  }
  var kind = filterParams && filterParams.kind === "date" ? "date" : "number";
  if (kind === "date") {
    var text = String(rowValue).slice(0, 10);
    if (hasStart && text < String(start)) {
      return false;
    }
    if (hasEnd && text > String(end)) {
      return false;
    }
    return true;
  }
  var value = Number(rowValue);
  if (!Number.isFinite(value)) {
    return false;
  }
  if (hasStart && value < Number(start)) {
    return false;
  }
  if (hasEnd && value > Number(end)) {
    return false;
  }
  return true;
}
"""
_RANGE_HEADER_EDITOR = """
function(cell, onRendered, success, cancel, editorParams) {
  var filterModule = cell.getTable().modules.filter;
  var registry = filterModule && (filterModule.constructor.filters || filterModule.filters);
  if (registry && !registry.minMax) {
    registry.minMax = __RANGE_FILTER__;
  }
  var kind = editorParams && editorParams.kind === "date" ? "date" : "number";
  var box = document.createElement("span");
  box.style.display = "flex";
  box.style.gap = "2px";
  function makeInput(placeholder) {
    var input = document.createElement("input");
    input.type = kind;
    input.placeholder = placeholder;
    input.style.width = "50%";
    input.style.minWidth = "0";
    input.style.boxSizing = "border-box";
    input.style.padding = "2px";
    function stop(event) { event.stopPropagation(); }
    input.addEventListener("mousedown", stop);
    input.addEventListener("click", stop);
    input.addEventListener("keydown", stop);
    input.addEventListener("keyup", stop);
    input.addEventListener("keypress", stop);
    return input;
  }
  var start = makeInput(kind === "date" ? "from" : "min");
  var end = makeInput(kind === "date" ? "to" : "max");
  var current = cell.getValue();
  if (current && typeof current === "object") {
    if (current.start) start.value = current.start;
    if (current.end) end.value = current.end;
  }
  function commit() {
    success({start: start.value, end: end.value});
  }
  start.addEventListener("input", commit);
  end.addEventListener("input", commit);
  box.appendChild(start);
  box.appendChild(end);
  return box;
}
""".replace("__RANGE_FILTER__", _RANGE_HEADER_FILTER.strip())
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


def candidate_tabulator_filters(
    columns: Sequence[str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, object]]]]:
    """Header filters for the displayed candidate columns.

    Text columns match a substring. Numeric columns and ``date`` use a
    min/max editor in that column's header. ``base_url`` is omitted.

    Returns
    -------
    header_filters, configuration
        Pass these to ``pn.widgets.Tabulator``. The configuration columns
        carry the range editor; Tabulator runs it in the browser.
    """
    header_filters: dict[str, dict[str, str]] = {}
    range_columns: list[dict[str, object]] = []
    for column in columns:
        if column == "base_url":
            continue
        if column in _DATE_FILTER_COLUMNS or column in _RANGE_FILTER_COLUMNS:
            kind = "date" if column in _DATE_FILTER_COLUMNS else "number"
            params = {"kind": kind}
            range_columns.append(
                {
                    "field": column,
                    "headerFilter": JSCode(_RANGE_HEADER_EDITOR),
                    "headerFilterParams": params,
                    "headerFilterFunc": _RANGE_FILTER_NAME,
                    "headerFilterFuncParams": params,
                    # Tabulator's default live filter reads the wrapper node's
                    # .value. That node is a <span>, so .value is undefined and
                    # the filter is deleted ~300 ms after it is applied.
                    "headerFilterLiveFilter": False,
                }
            )
            continue
        header_filters[column] = {"type": "input", "func": "like"}
    return header_filters, {"columns": range_columns}


def _only_filter_events(events: tuple[object, ...]) -> bool:
    return all(getattr(event, "name", None) == "filters" for event in events)


class CandidateTabulator(pn.widgets.Tabulator):
    """Candidate table whose header filters stay in the browser.

    ``clicked_iloc`` is the last row index reported by the browser.

    With local pagination, echoing a header filter through ``_update_cds``
    replaces every row. A min/max value is ``{start, end}``, and Panel then
    tries to cast that dict with the column dtype. Leaving the filter on the
    client keeps the visible rows and the row selection.
    """

    clicked_iloc: int | None = None
    clicked_filter_key: tuple[object, ...] | None = None
    clicked_row: pd.Series | None = None

    def _update_cds(self, *events: object) -> None:
        if self.pagination != "remote" and events and _only_filter_events(events):
            return
        super()._update_cds(*events)  # type: ignore[misc]

    def _get_header_filters(self, df: pd.DataFrame) -> list[pd.Series]:
        current = list(self.filters or [])
        kept = [item for item in current if not _is_client_range_filter(item)]
        if len(kept) == len(current):
            return super()._get_header_filters(df)  # type: ignore[misc]
        with param.discard_events(self):
            self.filters = kept
            try:
                return super()._get_header_filters(df)  # type: ignore[misc]
            finally:
                self.filters = current

    def _process_event(self, event: object) -> None:
        # The browser ``_index`` is the position in the full frame. Record that
        # row before Panel remaps it. A header filter can make the remap fail,
        # and Load would otherwise keep the previous candidate.
        raw = getattr(event, "row", None)
        name = getattr(event, "event_name", None)
        recorded = False
        if (
            name == "cell-click"
            and isinstance(raw, int)
            and self.value is not None
            and 0 <= raw < len(self.value)
        ):
            self.clicked_iloc = raw
            self.clicked_filter_key = _filter_key(self.filters)
            self.clicked_row = self.value.iloc[raw]
            recorded = True
        try:
            super()._process_event(event)  # type: ignore[misc]
        except (KeyError, IndexError, ValueError):
            if not recorded:
                raise
            for callback in self._on_click_callbacks.get(None, []):
                callback(event)
            return
        if name == "cell-click" and isinstance(getattr(event, "row", None), int):
            iloc = int(event.row)
            if self.value is not None and 0 <= iloc < len(self.value):
                self.clicked_iloc = iloc
                self.clicked_row = self.value.iloc[iloc]


def _is_client_range_filter(item: Mapping[str, object]) -> bool:
    """True when the header value is a min/max pair applied in the browser.

    ``detections`` is an integer column. Panel casts a header value with the
    column dtype, and ``int({start, end})`` raises. That exception aborts the
    click that should select the filtered row.
    """
    op = item.get("type")
    if isinstance(item.get("value"), Mapping):
        return True
    return not isinstance(op, str) or op in {"function", _RANGE_FILTER_NAME}


def _header_filter_mask(frame: pd.DataFrame, filters: Sequence[Mapping[str, object]]) -> pd.Series:
    """Rows that pass the Tabulator header filters currently on the table."""
    mask = pd.Series(True, index=frame.index)
    for item in filters:
        field = item.get("field")
        if not isinstance(field, str) or field not in frame.columns:
            continue
        op = item.get("type")
        value = item.get("value")
        column = frame[field]
        if op == "like":
            mask &= column.astype(str).str.contains(
                "" if value is None else str(value),
                case=False,
                regex=False,
                na=False,
            )
            continue
        if isinstance(value, Mapping) and _is_client_range_filter(item):
            mask &= _range_filter_mask(column, value, date=field == "date")
    return mask


def _range_filter_mask(
    column: pd.Series,
    value: Mapping[str, object],
    *,
    date: bool,
) -> pd.Series:
    start = value.get("start")
    end = value.get("end")
    has_start = start is not None and str(start) != ""
    has_end = end is not None and str(end) != ""
    if not has_start and not has_end:
        return pd.Series(True, index=column.index)
    if date:
        text = column.astype(str).str.slice(0, 10)
        keep = pd.Series(True, index=column.index)
        if has_start:
            keep &= text >= str(start)
        if has_end:
            keep &= text <= str(end)
        return keep.fillna(False)
    numbers = pd.to_numeric(column, errors="coerce")
    keep = numbers.notna()
    if has_start:
        try:
            keep &= numbers >= float(str(start))
        except ValueError:
            return pd.Series(False, index=column.index)
    if has_end:
        try:
            keep &= numbers <= float(str(end))
        except ValueError:
            return pd.Series(False, index=column.index)
    return keep.fillna(False)


def _filter_key(filters: Sequence[Mapping[str, object]] | None) -> tuple[object, ...]:
    key: list[object] = []
    for item in filters or []:
        op = item.get("type")
        op_key = "function" if _is_client_range_filter(item) else op
        value = item.get("value")
        if isinstance(value, Mapping):
            frozen: object = tuple(
                sorted((str(name), "" if raw is None else str(raw)) for name, raw in value.items())
            )
        else:
            frozen = "" if value is None else str(value)
        key.append((item.get("field"), op_key, frozen))
    return tuple(key)


def candidate_row_for_load(
    frame: pd.DataFrame,
    selection: Sequence[int],
    filters: Sequence[Mapping[str, object]] | None = None,
    *,
    allow_visible_position: bool = True,
) -> pd.Series:
    """Return the candidate Load should open.

    ``selection`` is a Tabulator row index. With no header filter that index is
    a position in ``frame``. After a filter it may instead be a position in the
    visible rows. A hidden row is not reused: that left the sky on the previous
    candidate.
    """
    if len(selection) == 0:
        msg = "Select a candidate row, then press Load."
        raise ValueError(msg)
    iloc = int(selection[-1])
    visible = frame.loc[_header_filter_mask(frame, list(filters or []))]
    if 0 <= iloc < len(frame) and frame.index[iloc] in visible.index:
        return frame.iloc[iloc]
    if allow_visible_position and 0 <= iloc < len(visible):
        return visible.iloc[iloc]
    msg = "Select a visible row, then press Load."
    raise ValueError(msg)


def load_selected_candidate(
    frame: pd.DataFrame,
    *,
    clicked_iloc: int | None,
    clicked_filter_key: tuple[object, ...] | None,
    selection: Sequence[int],
    filters: Sequence[Mapping[str, object]] | None,
    picked_iloc: int | None = None,
) -> pd.Series:
    """Open the row clicked under the current header filter.

    A click from before the filter is ignored when that row is now hidden.
    The visible-page position is used only when the click happened with these
    filters already applied.
    """
    current = list(filters or [])
    chosen = list(selection)
    if not chosen and picked_iloc is not None:
        chosen = [picked_iloc]
    if chosen:
        try:
            return candidate_row_for_load(
                frame,
                chosen,
                current,
                allow_visible_position=False,
            )
        except ValueError:
            pass
    if clicked_iloc is not None:
        try:
            return candidate_row_for_load(
                frame,
                [clicked_iloc],
                current,
                allow_visible_position=False,
            )
        except ValueError:
            if clicked_filter_key == _filter_key(current):
                return candidate_row_for_load(frame, [clicked_iloc], current)
    msg = "Select a visible row, then press Load."
    raise ValueError(msg)


@dataclass(frozen=True)
class CandidateCatalogLayout:
    """Candidate column placed beside a Source Review sky pane."""

    table: CandidateTabulator
    load_button: pn.widgets.Button
    catalog: pn.Column
    sky_row: pn.Row


def _detection_time_freq(review: Any, html: str) -> tuple[int | None, int | None]:
    """Time and frequency indices for the brightest detection on this page."""
    from ovro_lwa_portal.viz.source_review_data import match_detection_slice

    detections = parse_source_detections(html)
    dataset = review._dataset
    lst_hours = review._lst_hours
    freq_mhz = review._freq_mhz
    if not detections or dataset is None or lst_hours is None or freq_mhz is None:
        return None, None
    peak = peak_detection(detections)
    band_mhz, band_note = subband_mhz_for_overlay(peak, detections)
    time_idx, freq_idx, slice_note = match_detection_slice(
        dataset.coords["time"].values,
        freq_mhz,
        lst_hours,
        on_date=peak.on_date,
        lst_hour=peak.lst_hour,
        band_mhz=band_mhz,
    )
    review._log(f"{peak.band} SNR {peak.snr:.1f}: {band_note}. Store sample {slice_note}.")
    return time_idx, freq_idx


def _fetch_source_page_on_worker(
    review: Any,
    key: tuple[str, str],
    apply: Callable[[str, str], None],
) -> None:
    """GET the source page off the Bokeh thread, then schedule ``apply``."""

    def _work() -> None:
        try:
            html = fetch_text(source_page_url(key[0], key[1]))
            degrees = parse_combined_position(html)
        except (URLError, ValueError) as exc:
            message = f"WARNING: Could not open {key[1]}: {exc}"

            def _report(text: str = message) -> None:
                review._log(text)

            review._ui.schedule(_report)
            return

        def _finish(parsed: str = degrees, page: str = html) -> None:
            apply(parsed, page)

        review._ui.schedule(_finish)

    threading.Thread(target=_work, name="exopipe-source-page", daemon=True).start()


def attach_candidate_catalog(review: Any, candidates: pd.DataFrame) -> CandidateCatalogLayout:
    """Place the candidate table beside ``review``'s sky pane.

    The sky pane stays a child of :attr:`CandidateCatalogLayout.sky_row` so
    ``panel serve`` can replace it when the Zarr store opens. Load fetches the
    ExoPipe source page and calls ``review.review_catalog_position``. Under
    :class:`~ovro_lwa_portal.viz.panel_ui_session.ServedPanelUISession` that
    fetch runs on a worker and the review call is scheduled back onto the
    Bokeh document. Header filters stay in the browser.
    """
    objects = list(review._layout.objects)
    try:
        sky_index = objects.index(review._sky_pane)
    except ValueError as exc:
        msg = "Sky pane is not in the Source Review layout."
        raise ValueError(msg) from exc

    header_filters, filter_configuration = candidate_tabulator_filters(candidates.columns)
    table = CandidateTabulator(
        candidates,
        show_index=False,
        sizing_mode="stretch_width",
        height=620,
        header_filters=header_filters,
        configuration=filter_configuration,
        hidden_columns=["base_url"],
        editors={column: None for column in candidates.columns},
        pagination="local",
        page_size=50,
        selectable=1,
    )
    position_cache: dict[tuple[str, str], tuple[str, str]] = {}

    def _on_click(_event: object) -> None:
        row = table.clicked_row
        if row is None:
            review._log("WARNING: Select a visible row, then press Load.")
            return
        review._log(f"Selected {row.stokes} {row.source}. Press Load.")

    def _load(_event: object | None = None) -> None:
        try:
            row = load_selected_candidate(
                candidates,
                clicked_iloc=table.clicked_iloc,
                clicked_filter_key=table.clicked_filter_key,
                selection=list(table.selection or []),
                filters=table.filters,
                picked_iloc=table.clicked_iloc,
            )
        except ValueError as exc:
            review._log(f"WARNING: {exc}")
            return
        request_id = getattr(review, "_catalog_load_generation", 0) + 1
        review._catalog_load_generation = request_id
        key = (str(row.base_url), str(row.source))
        review._log(f"Loading {row.stokes} {row.source}…")

        def _apply(degrees: str, html: str) -> None:
            if request_id != review._catalog_load_generation:
                return
            position_cache[key] = (degrees, html)
            time_idx, freq_idx = _detection_time_freq(review, html)
            review.review_catalog_position(
                degrees,
                name=str(row.source),
                stokes=str(row.stokes),
                time_idx=time_idx,
                freq_idx=freq_idx,
            )

        cached = position_cache.get(key)
        if cached is not None:
            _apply(*cached)
            return
        if getattr(review, "_panel_serve_mode", False):
            _fetch_source_page_on_worker(review, key, _apply)
            return
        try:
            html = fetch_text(source_page_url(key[0], key[1]))
            _apply(parse_combined_position(html), html)
        except (URLError, ValueError) as exc:
            review._log(f"WARNING: Could not open {key[1]}: {exc}")

    table.on_click(_on_click)
    load_button = pn.widgets.Button(name="Load", button_type="primary", width=120)
    load_button.on_click(_load)
    catalog = pn.Column(
        pn.pane.Markdown("**Candidates** — select one row, then **Load**."),
        load_button,
        table,
        sizing_mode="stretch_width",
    )
    sky_row = pn.Row(catalog, review._sky_pane, sizing_mode="stretch_width")
    objects[sky_index] = sky_row
    review._layout.objects = objects
    review._layout.max_width = None
    return CandidateCatalogLayout(
        table=table,
        load_button=load_button,
        catalog=catalog,
        sky_row=sky_row,
    )


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
    "CandidateCatalogLayout",
    "CandidateTabulator",
    "attach_candidate_catalog",
    "SourceDetection",
    "URLError",
    "candidate_from_selection",
    "candidate_row_for_load",
    "candidate_tabulator_filters",
    "fetch_text",
    "load_candidate_table",
    "load_selected_candidate",
    "parse_candidate_index",
    "parse_combined_position",
    "parse_source_detections",
    "peak_detection",
    "source_page_url",
    "subband_mhz_for_overlay",
]
