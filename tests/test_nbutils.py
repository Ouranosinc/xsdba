from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsdba import nbutils as nbu


class TestQuantiles:
    @pytest.mark.parametrize("uses_dask", [True, False])
    def test_quantile(self, uses_dask, gosset):
        da = (xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc")).sel(time=slice("1950", "1955")).pr).load()
        if uses_dask:
            da = da.chunk({"location": 1})
        else:
            da = da.load()
        q = np.linspace(0.1, 0.99, 50)
        out_nbu = nbu.quantile(da, q, dim="time").transpose("location", ...)
        out_xr = da.quantile(q=q, dim="time").transpose("location", ...)
        np.testing.assert_array_almost_equal(out_nbu.values, out_xr.values)

    def test_edge_cases(self):
        q = np.linspace(0.1, 0.99, 50)

        # only 1 non-null value
        da = xr.DataArray([1] + [np.nan] * 100, dims="dim_0")
        out_nbu = nbu.quantile(da, q, dim="dim_0")
        np.testing.assert_array_equal(out_nbu.values, np.full_like(q, 1))

        # only NANs
        da = xr.DataArray([np.nan] * 100, dims="dim_0")
        out_nbu = nbu.quantile(da, q, dim="dim_0")
        np.testing.assert_array_equal(out_nbu.values, np.full_like(q, np.nan))


def test_rank(random):
    arr = random.random((10, 10, 1000))
    da = xr.DataArray(arr, dims=("x", "y", "time"))

    ranks = nbu.rank(da, dim="time", pct=False)

    exp = arr.argsort().argsort() + 1

    np.testing.assert_array_equal(ranks.values, exp)


def test_rank_tiebreak(random):
    arr = [1, 26, 2, 4.0, 6, 2, 2]
    da = xr.DataArray(arr, dims=("time"))
    ranks = nbu.rank(da, dim="time", pct=False)
    sorted_ranks = np.asarray(sorted(list(ranks.values)))
    ranks_t = nbu.rank(da, dim="time", pct=False, use_random_tiebreak=True)
    sorted_ranks_t = np.asarray(sorted(list(ranks_t.values)))
    exp_ranks_with_tiebreaks = np.arange(1, len(arr) + 1)
    assert not all(sorted_ranks == exp_ranks_with_tiebreaks)
    assert all(sorted_ranks_t == exp_ranks_with_tiebreaks)
