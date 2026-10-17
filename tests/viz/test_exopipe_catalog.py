"""ExoPipe index and source-page HTML parsing. No network."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ovro_lwa_portal.name_resolution import resolve_coordinate_string
from ovro_lwa_portal.viz.exopipe_catalog import (
    candidate_from_selection,
    displayed_columns,
    filter_candidates,
    load_candidate_table,
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


def test_filter_candidates_combines_text_and_ranges() -> None:
    frame = parse_candidate_index(
        _INDEX_I,
        stokes="I",
        base_url="http://127.0.0.1:8768",
    )
    assert "base_url" not in displayed_columns(frame.columns)
    narrowed = filter_candidates(
        frame,
        contains={"source": "1939", "bands": "red"},
        ranges={"snr": (80, None), "rank": ("1", "1"), "date": ("2025-05-01", date(2025, 5, 31))},
    )
    assert list(narrowed["source"]) == ["J1939+2136"]
    assert narrowed.index.tolist() == [1]
    assert frame.iloc[int(narrowed.index[0])].source == "J1939+2136"
    assert filter_candidates(frame, contains={"source": "  "}, ranges={"snr": (None, None)}) is frame


def test_filter_candidates_rejects_hidden_and_wrong_kind() -> None:
    frame = parse_candidate_index(
        _INDEX_I,
        stokes="I",
        base_url="http://127.0.0.1:8768",
    )
    with pytest.raises(ValueError, match="base_url"):
        filter_candidates(frame, contains={"base_url": "8768"})
    with pytest.raises(ValueError, match="not a range column"):
        filter_candidates(frame, ranges={"source": ("a", "z")})
    with pytest.raises(ValueError, match="min/max"):
        filter_candidates(frame, contains={"snr": "70"})
    with pytest.raises(ValueError, match="not a number"):
        filter_candidates(frame, ranges={"snr": ("high", None)})
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        filter_candidates(frame, ranges={"date": ("05-05-2025", None)})
    assert filter_candidates(frame, ranges={"elevation": (60, 90)}).empty


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
