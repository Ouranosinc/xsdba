from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsdba import measures


def test_bias(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    test = measures.bias(sim, ref).values
    np.testing.assert_array_almost_equal(
        test, np.array([[6.430237, 39.088974, 5.2402344]])
    )


def test_relative_bias(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    test = measures.relative_bias(sim, ref).values
    np.testing.assert_array_almost_equal(
        test, np.array([[0.02366494, 0.16392256, 0.01920133]])
    )


def test_circular_bias():
    sim = xr.DataArray(
        data=np.array([1, 1, 1, 2, 365, 300]), attrs={"units": "", "long_name": "test"}
    )
    ref = xr.DataArray(
        data=np.array([2, 365, 300, 1, 1, 1]), attrs={"units": "", "long_name": "test"}
    )
    test = measures.circular_bias(sim, ref).values
    np.testing.assert_array_almost_equal(test, [1, 1, 66, -1, -1, -66])


def test_ratio(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time="1950-01-01")
        .tasmax
    )
    test = measures.ratio(sim, ref).values
    np.testing.assert_array_almost_equal(
        test, np.array([[1.023665, 1.1639225, 1.0192013]])
    )


def test_rmse(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    test = measures.rmse(sim, ref).values
    np.testing.assert_array_almost_equal(test, [5.4499755, 18.124086, 12.387193], 4)


def test_rmse_nan(timeseries):
    sim = timeseries([1, 1, 1], start="2000-01-01")
    sim.attrs["units"] = "K"

    ref = timeseries([1, 1, np.nan], start="2000-01-01")
    ref.attrs["units"] = "K"

    test = measures.rmse(sim, ref).values
    np.testing.assert_array_almost_equal(test, [0], 4)


def test_mae_nan(timeseries):
    sim = timeseries([1, 1, 1], start="2000-01-01")
    sim.attrs["units"] = "K"

    ref = timeseries([1, 1, np.nan], start="2000-01-01")
    ref.attrs["units"] = "K"

    test = measures.mae(sim, ref).values
    np.testing.assert_array_almost_equal(test, [0], 4)


def test_mae(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    test = measures.mae(sim, ref).values
    np.testing.assert_array_almost_equal(test, [4.159672, 14.2148, 9.768536], 4)


def test_annual_cycle_correlation(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time=slice("1950", "1953"))
        .tasmax
    )
    test = (
        measures.annual_cycle_correlation(sim, ref, window=31)
        .sel(location="Vancouver")
        .values
    )
    np.testing.assert_array_almost_equal(test, [0.94580488], 4)


@pytest.mark.slow
def test_scorr(gosset):
    ref = xr.open_dataset(
        gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmin_1990.nc")
    ).tasmin
    sim = xr.open_dataset(
        gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
    ).tasmax
    scorr = measures.scorr(sim.isel(lon=slice(0, 50)), ref.isel(lon=slice(0, 50)))

    np.testing.assert_allclose(scorr, [97374.2146243])


def test_taylordiagram(gosset):
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .sel(time=slice("1950", "1953"), location="Amos")
        .tasmax
    )
    ref = (
        xr.open_dataset(gosset.fetch("sdba/nrcan_1950-2013.nc"))
        .sel(time=slice("1950", "1953"), location="Amos")
        .tasmax
    )
    test = measures.taylordiagram(sim, ref).values
    np.testing.assert_array_almost_equal(test, [13.12244701, 6.76166582, 0.73230199], 4)

    # test normalization option
    test_normalize = measures.taylordiagram(sim, ref, normalize=True).values
    np.testing.assert_array_almost_equal(
        test_normalize,
        [13.12244701 / 13.12244701, 6.76166582 / 13.12244701, 0.73230199],
        4,
    )
