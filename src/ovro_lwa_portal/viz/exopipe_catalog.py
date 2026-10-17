"""Read ExoPipe transient-review HTML into a candidate table.

The review apps are separate HTTP servers. This module parses their index
table and the combined-position field on ``/source/{id}``. It does not import
the ExoPipe generator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    "URLError",
    "fetch_text",
    "load_candidate_table",
    "parse_candidate_index",
    "parse_combined_position",
    "source_page_url",
]
