"""ExoPipe index and source-page HTML parsing. No network."""

from __future__ import annotations

import pandas as pd
import pytest

from ovro_lwa_portal.name_resolution import resolve_coordinate_string
from ovro_lwa_portal.viz.exopipe_catalog import (
    load_candidate_table,
    parse_candidate_index,
    parse_combined_position,
    source_page_url,
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
