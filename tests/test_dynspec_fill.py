"""Chunked dynamic-spectrum reads."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ovro_lwa_portal.dynspec_fill import (
    TrackedPixelReader,
    plan_fill_units,
    spectrum_chunk_lengths,
)


def _sky() -> xr.DataArray:
    values = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    return xr.DataArray(
        values,
        dims=("time", "frequency", "l", "m"),
        coords={
            "time": np.arange(4),
            "frequency": np.arange(3),
            "l": np.arange(2),
            "m": np.arange(2),
        },
    )


def test_plan_fill_units_covers_the_grid() -> None:
    units = plan_fill_units(5, 3, 2, 2)
    covered = np.zeros((5, 3), dtype=bool)
    for unit in units:
        covered[unit.t0 : unit.t1, unit.f0 : unit.f1] = True
    assert covered.all()
    assert units[0].t1 - units[0].t0 == 2
    assert units[-1].t0 == 4


def test_tracked_reader_uses_one_pixel_and_skips_hidden_times() -> None:
    sky = _sky()
    visible = np.array([True, False, True, True])
    reader = TrackedPixelReader(
        sky,
        pol_idx=0,
        l_indices=np.zeros(4, dtype=int),
        m_indices=np.ones(4, dtype=int),
        visible=visible,
    )
    block = reader(plan_fill_units(4, 3, 4, 3)[0])
    expected = np.asarray(sky.isel(l=0, m=1).values, dtype=np.float32)
    expected[1] = np.nan
    assert block.shape == (4, 3)
    assert np.allclose(block, expected, equal_nan=True)


def test_tracked_reader_follows_a_moving_pixel() -> None:
    sky = _sky()
    reader = TrackedPixelReader(
        sky,
        pol_idx=0,
        l_indices=np.array([0, 1, 0, 1]),
        m_indices=np.array([0, 0, 1, 1]),
        visible=np.ones(4, dtype=bool),
    )
    block = reader(plan_fill_units(4, 3, 4, 3)[0])
    for t, (l_idx, m_idx) in enumerate(zip([0, 1, 0, 1], [0, 0, 1, 1], strict=True)):
        column = np.asarray(sky.isel(time=t, l=l_idx, m=m_idx).values, dtype=np.float32)
        assert np.allclose(block[t], column)


def test_spectrum_chunk_lengths_follow_dask_chunks() -> None:
    sky = _sky().chunk({"time": 2, "frequency": 1, "l": 2, "m": 2})
    assert spectrum_chunk_lengths(sky) == (2, 1)
