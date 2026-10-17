"""ExoPipe index and source-page HTML parsing. No network."""

from __future__ import annotations

from datetime import date

import pandas as pd
import panel as pn
import pytest
from panel.io.model import JSCode

from ovro_lwa_portal.name_resolution import resolve_coordinate_string
from ovro_lwa_portal.viz.exopipe_catalog import (
    CandidateTabulator,
    candidate_from_selection,
    candidate_row_for_load,
    candidate_tabulator_filters,
    load_candidate_table,
    load_selected_candidate,
    parse_candidate_index,
    parse_combined_position,
    parse_source_detections,
    peak_detection,
    source_page_url,
    subband_mhz_for_overlay,
)

_INDEX_I = """
<table class="idx">
<tr>
<th>#</th><th>source</th><th>date</th><th>bands</th>
<th>PB-corr peak (Jy)</th><th>detections</th><th>SNR</th><th>elevation</th>
<th>n_i_epoch</th><th>night noise (x cell)</th><th>TESS</th>
<th>verdict</th><th>FP reason</th><th>cross-match</th><th>note</th>
</tr>
<tr data-sid="J1841-0157">
<td>2</td><td><a href="/source/J1841-0157">J1841-0157</a></td>
<td>2025-04-06</td><td class="bands">Full</td><td>21.91</td><td>2</td>
<td>72.3</td><td>30.8</td><td>1</td><td></td><td></td><td></td><td></td>
<td class="note">pulsars PSR J1841-0157</td><td></td>
</tr>
<tr data-sid="J1939+2136">
<td>1</td><td><a href="/source/J1939%2B2136">J1939+2136</a></td>
<td>2025-05-05</td><td>Full, Red</td><td>39.90</td><td>6</td>
<td>92.6</td><td>57.9</td><td>16</td><td></td><td></td><td>real</td><td></td>
<td>pulsars</td><td>note text</td>
</tr>
</table>
"""

_INDEX_V = """
<table class="idx">
<tr><th>#</th><th>source</th><th>n_v_epoch</th><th>SNR</th></tr>
<tr data-sid="J0953+0755"><td>1</td><td>J0953+0755</td><td>4</td><td>11.5</td></tr>
</table>
"""

_LOCBAR = """
<div class="locbar">
<span class="lk">combined position</span>
<input class="lv" readonly value="280.46630 -1.96395">
<input class="lv" readonly value="18:41:51.91 -01:57:50.2">
<span class="lk">&plusmn; 81″</span>
</div>
"""


def test_parse_candidate_index_reads_data_sid_and_snr() -> None:
    frame = parse_candidate_index(
        _INDEX_I,
        stokes="I",
        base_url="http://127.0.0.1:8768/",
    )
    assert list(frame["source"]) == ["J1841-0157", "J1939+2136"]
    row = frame.iloc[0]
    assert row.stokes == "I"
    assert row.base_url == "http://127.0.0.1:8768"
    assert row.date == "2025-04-06"
    assert row.bands == "Full"
    assert row.snr == 72.3
    assert row.peak_jy == 21.91
    assert row.detections == 2
    assert row.elevation == 30.8
    assert row.n_epoch == "1"
    assert row.cross_match == "pulsars PSR J1841-0157"
    assert "ra" not in frame.columns
    assert "dec" not in frame.columns


def test_source_page_url_encodes_plus() -> None:
    assert source_page_url("http://127.0.0.1:8768", "J1939+2136") == (
        "http://127.0.0.1:8768/source/J1939%2B2136"
    )


def test_parse_combined_position_uses_first_degree_pair() -> None:
    degrees = parse_combined_position(_LOCBAR)
    assert degrees == "280.46630 -1.96395"
    resolution, messages = resolve_coordinate_string(degrees, use_ned_fallback=False)
    assert resolution is not None
    assert resolution.resolver == "degrees"
    assert messages == []
    assert resolution.coord.ra.deg == 280.46630
    assert resolution.coord.dec.deg == -1.96395


_DETECTIONS = """
<h2>Detections of this source</h2>
<table class="idx">
<tr><th>LST</th><th>date</th><th>band</th><th>SNR</th><th>RA</th><th>Dec</th></tr>
<tr><td>17h</td><td>2025-05-05</td><td>32MHz</td><td>68.27</td><td>294.90696</td><td>21.5738</td></tr>
<tr><td>17h</td><td>2025-05-05</td><td>Full</td><td>92.61</td><td>294.91343</td><td>21.57423</td></tr>
<tr><td>15h</td><td>2025-04-10</td><td>23MHz</td><td>33.77</td><td>294.82922</td><td>21.5583</td></tr>
</table>
"""


def test_parse_source_detections_peak_uses_subband_at_same_time() -> None:
    detections = parse_source_detections(_DETECTIONS)
    assert len(detections) == 3
    peak = peak_detection(detections)
    assert peak.band == "Full"
    assert peak.on_date == date(2025, 5, 5)
    assert peak.lst_hour == 17
    mhz, note = subband_mhz_for_overlay(peak, detections)
    assert mhz == 32.0
    assert "32MHz" in note
    subband = next(item for item in detections if item.band == "32MHz")
    own, own_note = subband_mhz_for_overlay(subband, detections)
    assert own == 32.0
    assert "32MHz at 2025-05-05 17h" in own_note


def test_candidate_tabulator_filters_put_ranges_on_column_headers() -> None:
    frame = parse_candidate_index(
        _INDEX_I,
        stokes="I",
        base_url="http://127.0.0.1:8768",
    )
    header_filters, configuration = candidate_tabulator_filters(frame.columns)
    assert "base_url" not in header_filters
    assert "snr" not in header_filters
    assert header_filters["source"] == {"type": "input", "func": "like"}
    table = pn.widgets.Tabulator(
        frame,
        show_index=False,
        header_filters=header_filters,
        configuration=configuration,
        editors=dict.fromkeys(frame.columns),
        pagination="local",
    )
    specs = {
        column["field"]: column
        for column in table._get_configuration(table._get_columns())["columns"]
    }
    assert isinstance(specs["snr"]["headerFilter"], JSCode)
    assert isinstance(specs["snr"]["headerFilterFunc"], JSCode)
    assert specs["snr"]["headerFilterFuncParams"] == {"kind": "number"}
    assert specs["date"]["headerFilterParams"] == {"kind": "date"}
    assert specs["rank"]["headerFilterFuncParams"] == {"kind": "number"}
    assert specs["source"]["headerFilter"] == "input"
    assert specs["source"]["headerFilterFunc"] == "like"
    assert specs["snr"]["headerFilterLiveFilter"] is False
    assert "keyup" in specs["snr"]["headerFilter"].js_code
    assert "headerFilter" not in specs["base_url"]


def test_range_header_filter_stays_on_the_client() -> None:
    frame = parse_candidate_index(
        _INDEX_I,
        stokes="I",
        base_url="http://127.0.0.1:8768",
    )
    header_filters, configuration = candidate_tabulator_filters(frame.columns)
    table = CandidateTabulator(
        frame,
        show_index=False,
        header_filters=header_filters,
        configuration=configuration,
        editors=dict.fromkeys(frame.columns),
        pagination="local",
    )
    before = len(table._processed)
    table.filters = [
        {"field": "snr", "type": "function", "value": {"start": "5", "end": ""}},
    ]
    assert len(table._processed) == before
    assert table._get_header_filters(frame) == []


def test_cell_click_records_the_full_frame_row() -> None:
    frame = pd.DataFrame({"source": ["a", "b", "c"], "snr": [1.0, 5.0, 9.0]})
    table = CandidateTabulator(frame, selectable=1, pagination="local")
    seen: list[str] = []

    def _click(_event: object) -> None:
        assert table.clicked_row is not None
        seen.append(str(table.clicked_row.source))

    table.on_click(_click)

    class _Click:
        event_name = "cell-click"
        row = 2
        column = "source"

    table._process_event(_Click())
    assert seen == ["c"]
    assert table.clicked_iloc == 2


def test_candidate_row_for_load_uses_a_visible_filtered_row() -> None:
    frame = pd.DataFrame({"source": ["a", "b", "c"], "snr": [1.0, 5.0, 9.0]})
    filters = [{"field": "snr", "type": "function", "value": {"start": "4", "end": ""}}]
    assert candidate_row_for_load(frame, [2], filters).source == "c"
    # Original row 0 is hidden, so 0 is the first visible row.
    assert candidate_row_for_load(frame, [0], filters).source == "b"
    with pytest.raises(ValueError, match="visible row"):
        candidate_row_for_load(frame, [0], filters, allow_visible_position=False)
    with pytest.raises(ValueError, match="visible row"):
        candidate_row_for_load(frame, [5], filters)


def test_load_selected_candidate_ignores_a_hidden_prefilter_click() -> None:
    frame = pd.DataFrame({"source": ["a", "b", "c"], "snr": [1.0, 5.0, 9.0]})
    filters = [{"field": "snr", "type": "function", "value": {"start": "4", "end": ""}}]
    with pytest.raises(ValueError, match="visible row"):
        load_selected_candidate(
            frame,
            clicked_iloc=0,
            clicked_filter_key=(),
            selection=[0],
            filters=filters,
        )
    loaded = load_selected_candidate(
        frame,
        clicked_iloc=0,
        clicked_filter_key=(("snr", "function", (("end", ""), ("start", "4"))),),
        selection=[0],
        filters=filters,
    )
    assert loaded.source == "b"
    preferred = load_selected_candidate(
        frame,
        clicked_iloc=1,
        clicked_filter_key=(("snr", "function", (("end", ""), ("start", "4"))),),
        selection=[2],
        filters=filters,
    )
    assert preferred.source == "c"


def test_candidate_from_selection_uses_last_iloc() -> None:
    frame = pd.DataFrame(
        {
            "stokes": ["V", "I"],
            "source": ["J0953+0755", "J1841-0157"],
            "base_url": ["http://127.0.0.1:8766", "http://127.0.0.1:8768"],
        }
    )
    row = candidate_from_selection(frame, [0, 1])
    assert row.source == "J1841-0157"
    assert row.stokes == "I"
    assert row.base_url == "http://127.0.0.1:8768"
    with pytest.raises(ValueError, match="press Load"):
        candidate_from_selection(frame, [])


def test_parse_combined_position_missing_locbar_raises() -> None:
    with pytest.raises(ValueError, match="combined position"):
        parse_combined_position("<html><p>no position</p></html>")


def test_load_candidate_table_concatenates_queues_without_network() -> None:
    pages = {
        "http://127.0.0.1:8766/": _INDEX_V,
        "http://127.0.0.1:8768/": _INDEX_I,
    }
    fetched: list[str] = []

    def _fetch(url: str) -> str:
        fetched.append(url)
        return pages[url]

    frame = load_candidate_table(
        {"V": "http://127.0.0.1:8766", "I": "http://127.0.0.1:8768"},
        fetch=_fetch,
    )
    assert fetched == ["http://127.0.0.1:8766/", "http://127.0.0.1:8768/"]
    assert len(frame) == 3
    stokes_v = frame[frame["stokes"] == "V"]
    assert list(stokes_v["source"]) == ["J0953+0755"]
    assert stokes_v.iloc[0].n_epoch == "4"
    assert stokes_v.iloc[0].base_url == "http://127.0.0.1:8766"
    assert isinstance(frame, pd.DataFrame)
