from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xclim

from xsdba._processing import _adapt_freq
from xsdba.adjustment import EmpiricalQuantileMapping
from xsdba.base import Grouper
from xsdba.processing import (
    _normalized_radial_wavenumber,
    adapt_freq,
    escore,
    from_additive_space,
    jitter,
    jitter_over_thresh,
    jitter_under_thresh,
    normalize,
    reordering,
    spectral_filter,
    stack_variables,
    standardize,
    to_additive_space,
    unstack_variables,
    unstandardize,
)
from xsdba.units import convert_units_to, pint_multiply, units


def test_jitter_both():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "K"})
    out = jitter(da, lower="1 K", upper="2 K", maximum="3 K")

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0

    assert da[1] != out[1]
    assert da[1] < 3
    assert da[1] > 2


def test_jitter_under_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "K"})
    out = jitter_under_thresh(da, "1 K")

    assert da[0] != out[0]
    assert da[0] < 1
    assert da[0] > 0
    np.testing.assert_allclose(da[1:], out[1:])
    assert (
        "jitter(x=<array>, lower='1 K', upper=None, minimum=None, maximum=None) - xsdba version"
        in out.attrs["history"]
    )


def test_jitter_over_thresh():
    da = xr.DataArray([0.5, 2.1, np.nan], attrs={"units": "m"})
    out = jitter_over_thresh(da, "200 cm", "0.003 km")

    assert da[1] != out[1]
    assert da[1] < 3
    assert da[1] > 2
    np.testing.assert_allclose(da[[0, 2]], out[[0, 2]])
    assert out.units == "m"


@pytest.mark.parametrize("test_val", [1e-6, 1, 100, 1e6])
@pytest.mark.parametrize("dtype, delta", [("f8", 1e-8), ("f4", 1e-6), ("f16", 1e-8)])
def test_jitter_other_dtypes(dtype, delta, test_val):
    # below, narrow intervals are meant to increase likely hood of rounding issues
    da = xr.DataArray(test_val + np.zeros(1000, dtype=dtype), attrs={"units": "%"})
    out_high = jitter(
        da, upper=f"{test_val * (1 - delta):.20f} %", maximum=f"{test_val:.20f} %"
    )
    out_low = jitter(
        da, lower=f"{test_val * (1 + delta):.20f} %", minimum=f"{test_val:.20f} %"
    )
    assert (out_high < test_val).all()
    assert (out_low > test_val).all()


@pytest.mark.parametrize("test", ["lower", "upper"])
@pytest.mark.parametrize("dtype, delta", [("f8", 1e-8), ("f4", 1e-6), ("f16", 1e-8)])
def test_jitter_log(dtype, delta, test):
    # below, narrow intervals are meant to increase likely hood of rounding issues
    test_val = delta / 2 if test == "lower" else 1 - delta / 2
    da = xr.DataArray(test_val + np.zeros(1000, dtype=dtype), attrs={"units": "%"})
    if test == "lower":
        out = jitter(da, lower=f"{delta:.20f} %", minimum=f"{test_val:.20f} %")
    else:
        out = jitter(da, upper=f"{1 - delta:.20f} %", maximum=f"{test_val:.20f} %")
    assert (np.isfinite(np.log(out / (1 - out)))).all()


@pytest.mark.parametrize("use_dask", [True, False])
def test_adapt_freq(use_dask, random):
    time = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    prvals = random.integers(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals,
        coords={"time": time, "lat": [0, 1, 2]},
        dims=("time", "lat"),
        attrs={"units": "mm d-1"},
    )

    if use_dask:
        pr = pr.chunk({"lat": 1})
    group = Grouper("time.month")
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)

    # Where the input is considered zero
    input_zeros = sim_ad.where(prsim <= 1)

    # The proportion of corrected values (time.size * 3 * 0.2 is the theoretical number of values under 1 in prsim)
    dP0_out = (input_zeros > 1).sum() / (time.size * 3 * 0.2)
    np.testing.assert_allclose(dP0_out, 0.5, atol=0.1)

    # Assert that corrected values were generated in the range ]1, 20 + tol[
    corrected = (
        input_zeros.where(input_zeros > 1)
        .stack(flat=["lat", "time"])
        .reset_index("flat")
        .dropna("flat")
    )
    assert ((corrected < 20.1) & (corrected > 1)).all()

    # Assert that non-corrected values are untouched
    # Again we add a 0.5 tol because of randomness.
    xr.testing.assert_equal(
        sim_ad.where(prsim > 20.1),
        prsim.where(prsim > 20.5).transpose("lat", "time"),
    )
    # Assert that Pth and dP0 are approx the good values
    np.testing.assert_allclose(pth, 20, rtol=0.05)
    np.testing.assert_allclose(dP0, 0.5, atol=0.25)
    assert sim_ad.units == "mm d-1"
    assert sim_ad.attrs["references"].startswith("Themeßl")
    assert pth.units == "mm d-1"


def test_adapt_freq_adjust(gosset):
    past = {"time": slice("1950", "1969")}
    future = {"time": slice("1970", "1989")}
    all_time = {"time": slice("1950", "1989")}
    ref = (
        xr.open_dataset(gosset.fetch("sdba/ahccd_1950-2013.nc")).loc[past].pr.fillna(0)
    )
    sim = (
        xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        .loc[all_time]
        .pr.fillna(0)
    )
    sim = xclim.core.units.convert_units_to(sim, ref)  # mm/d
    sim.loc[{"time": slice("1950", "1965")}] = 0
    sim.loc[{"time": slice("1970", "1980")}] = 0
    sim = jitter_under_thresh(sim, "1 mm/d")
    hist = sim.loc[past]
    # this is just to make sure the example works, some adaptation is needed
    assert ((hist <= 1).sum(dim="time") > (ref <= 1).sum(dim="time")).all()

    outh = _adapt_freq.func(xr.Dataset(dict(ref=ref, sim=hist)), dim="time", thresh=1)
    hist_ad = outh.sim_ad
    outs = _adapt_freq.func(
        xr.merge([sim.to_dataset(name="sim"), outh]),
        dim="time",
        thresh=1,
    )
    sim_ad = outs.sim_ad
    sim_f = sim.loc[future]
    sim_ad_f = sim_ad.loc[future]
    assert ((sim_ad_f <= 1).sum(dim="time") < (sim_ad <= 1).sum(dim="time")).all()


@pytest.mark.parametrize("use_dask", [True, False])
def test_adapt_freq_add_dims(use_dask, random):
    time = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    prvals = random.integers(0, 100, size=(time.size, 3))
    pr = xr.DataArray(
        prvals,
        coords={"time": time, "lat": [0, 1, 2]},
        dims=("time", "lat"),
        attrs={"units": "mm d-1"},
    )

    if use_dask:
        pr = pr.chunk()
    group = Grouper("time.month", add_dims=["lat"])
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, _dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)
    assert set(sim_ad.dims) == set(prsim.dims)
    assert "lat" not in pth.dims

    group = Grouper("time.dayofyear", window=5)
    with xr.set_options(keep_attrs=True):
        prsim = xr.where(pr < 20, pr / 20, pr)
        prref = xr.where(pr < 10, pr / 20, pr)
    sim_ad, pth, _dP0 = adapt_freq(prref, prsim, thresh="1 mm d-1", group=group)
    assert set(sim_ad.dims) == set(prsim.dims)


def test_escore():
    x = np.array([1, 4, 3, 6, 4, 7, 5, 8, 4, 5, 3, 7]).reshape(2, 6)
    y = np.array([6, 6, 3, 8, 5, 7, 3, 7, 3, 6, 4, 3]).reshape(2, 6)

    x = xr.DataArray(x, dims=("variables", "time")).astype(np.float64)
    y = xr.DataArray(y, dims=("variables", "time")).astype(np.float64)

    # Value taken from escore of Cannon's MBC R package.
    out = escore(x, y)
    np.testing.assert_allclose(out, 1.90018550338863)
    assert "escore(" in out.attrs["history"]
    assert out.attrs["references"].startswith("Székely")


def test_standardize(random):
    x = random.standard_normal((2, 10000))
    x[0, 50] = np.nan
    x = xr.DataArray(x, dims=("x", "y"), attrs={"units": "m"})

    xp, avg, std = standardize(x, dim="y")

    np.testing.assert_allclose(avg, 0, atol=4e-2)
    np.testing.assert_allclose(std, 1, atol=2e-2)

    xp, avg, std = standardize(x, mean=avg, dim="y")
    np.testing.assert_allclose(std, 1, atol=2e-2)

    y = unstandardize(xp, 0, 1)

    np.testing.assert_allclose(x, y, atol=0.1)
    assert avg.units == xp.units


def test_reordering():
    y = xr.DataArray(np.arange(1, 11), dims=("time",), attrs={"a": 1, "units": "K"})
    x = xr.DataArray(np.arange(10, 20)[::-1], dims=("time",))

    out = reordering(x, y, group="time")

    np.testing.assert_array_equal(out, np.arange(1, 11)[::-1])
    out.attrs.pop("history")
    assert out.attrs == y.attrs


def test_reordering_with_window():
    time = list(
        xr.date_range("2000-01-01", "2000-01-04", freq="D", calendar="noleap")
    ) + list(xr.date_range("2001-01-01", "2001-01-04", freq="D", calendar="noleap"))

    x = xr.DataArray(
        np.arange(1, 9, 1),
        dims=("time"),
        coords={"time": time},
    )

    y = xr.DataArray(
        np.arange(8, 0, -1),
        dims=("time"),
        coords={"time": time},
    )

    group = Grouper(group="time.dayofyear", window=3)
    out = reordering(x, y, group=group)

    np.testing.assert_array_equal(out, [3.0, 3.0, 2.0, 2.0, 7.0, 7.0, 6.0, 6.0])
    out.attrs.pop("history")
    assert out.attrs == y.attrs


def test_to_additive(timeseries):
    # log
    pr = timeseries(np.array([0, 1e-5, 1, np.e**10]), units="kg m^-2 s^-1")
    prlog = to_additive_space(pr, lower_bound="0 kg m^-2 s^-1", trans="log")
    np.testing.assert_allclose(prlog, [-np.inf, -11.512925, 0, 10])
    assert prlog.attrs["xsdba_transform"] == "log"
    assert prlog.attrs["xsdba_transform_units"] == "kg m^-2 s^-1"

    with xr.set_options(keep_attrs=True):
        pr1 = pr + 1
    lower_bound = "1 kg m^-2 s^-1"
    prlog2 = to_additive_space(pr1, trans="log", lower_bound=lower_bound)
    np.testing.assert_allclose(prlog2, [-np.inf, -11.512925, 0, 10])
    assert prlog2.attrs["xsdba_transform_lower"] == 1.0

    # logit
    hurs = timeseries(np.array([0, 1e-3, 90, 100]), units="%")

    hurslogit = to_additive_space(
        hurs, lower_bound="0 %", trans="logit", upper_bound="100 %"
    )
    np.testing.assert_allclose(
        hurslogit, [-np.inf, -11.5129154649, 2.197224577, np.inf]
    )
    assert hurslogit.attrs["xsdba_transform"] == "logit"
    assert hurslogit.attrs["xsdba_transform_units"] == "%"

    with xr.set_options(keep_attrs=True):
        hursscl = hurs * 4 + 200
    hurslogit2 = to_additive_space(
        hursscl, trans="logit", lower_bound="2", upper_bound="6"
    )
    np.testing.assert_allclose(
        hurslogit2, [-np.inf, -11.5129154649, 2.197224577, np.inf]
    )
    assert hurslogit2.attrs["xsdba_transform_lower"] == 200.0
    assert hurslogit2.attrs["xsdba_transform_upper"] == 600.0


def test_to_additive_clipping(timeseries):
    # log
    pr = timeseries(np.array([0]), units="kg m^-2 s^-1")
    prlog = to_additive_space(
        pr, lower_bound="0 kg m^-2 s^-1", trans="log", clip_next_to_bounds=True
    )
    assert np.isfinite(prlog).all()

    with xr.set_options(keep_attrs=True):
        pr1 = pr + 1
    lower_bound = "1 kg m^-2 s^-1"
    prlog2 = to_additive_space(
        pr1, trans="log", lower_bound=lower_bound, clip_next_to_bounds=True
    )
    assert np.isfinite(prlog2).all()

    # logit
    hurs = timeseries(np.array([0, 100]), units="%")
    hurslogit = to_additive_space(
        hurs,
        lower_bound="0 %",
        trans="logit",
        upper_bound="100 %",
        clip_next_to_bounds=True,
    )
    assert np.isfinite(hurslogit).all()


def test_from_additive(timeseries):
    # log
    pr = timeseries(np.array([0, 1e-5, 1, np.e**10]), units="mm/d")
    pr2 = from_additive_space(to_additive_space(pr, lower_bound="0 mm/d", trans="log"))
    np.testing.assert_allclose(pr[1:], pr2[1:])
    pr2.attrs.pop("history")
    assert pr.attrs == pr2.attrs

    # logit
    hurs = timeseries(np.array([0, 1e-5, 0.9, 1]), units="%")
    hurs2 = from_additive_space(
        to_additive_space(hurs, lower_bound="0 %", trans="logit", upper_bound="100 %")
    )
    np.testing.assert_allclose(hurs[1:-1], hurs2[1:-1])


def test_from_additive_with_args(timeseries):
    pr = timeseries(np.array([0, 1e-5, 1, np.e**10]), units="mm/d")
    prlog = np.log(pr).assign_attrs({"units": 1})
    pr2 = from_additive_space(prlog, lower_bound="0 mm/d", trans="log", units="mm/d")
    np.testing.assert_allclose(pr[1:], pr2[1:])
    pr2.attrs.pop("history")
    assert pr.attrs == pr2.attrs

    # logit
    hurs = timeseries(np.array([0, 1e-5, 0.9, 1]), units="%")
    hurslogit = (np.log(hurs / (100 - hurs))).assign_attrs({"units": 1})
    hurs2 = from_additive_space(
        hurslogit, lower_bound="0 %", trans="logit", upper_bound="100 %", units="%"
    )
    np.testing.assert_allclose(hurs[1:-1], hurs2[1:-1])
    assert hurs2.attrs["units"] == "%"


def test_normalize(timeseries, random):
    tas = timeseries(
        random.standard_normal((int(365.25 * 36),)) + 273.15,
        units="K",
        start="2000-01-01",
    )

    xp, norm = normalize(tas, group="time.dayofyear")
    np.testing.assert_allclose(norm, 273.15, atol=1)

    xp2, norm = normalize(tas, norm=norm, group="time.dayofyear")
    np.testing.assert_allclose(xp, xp2)


def test_stack_variables(gosset):
    ds1 = xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
    ds2 = xr.open_dataset(gosset.fetch("sdba/ahccd_1950-2013.nc"))

    da1 = stack_variables(ds1)
    da2 = stack_variables(ds2)

    # FIXME: These test for variable order; use a membership test instead
    assert list(da1.multivar.values) == ["pr", "tasmax"]
    assert da1.multivar.attrs["_standard_name"] == [
        "precipitation_flux",
        "air_temperature",
    ]
    assert da2.multivar.attrs["is_variables"]
    assert da1.multivar.equals(da2.multivar)

    da1p = da1.sortby("multivar", ascending=False)

    with pytest.raises(ValueError, match="Inputs have different multivariate"):
        EmpiricalQuantileMapping.train(da1p, da2)

    ds1p = unstack_variables(da1)

    xr.testing.assert_equal(ds1, ds1p)


class TestSpectralUtils:
    @pytest.mark.parametrize(
        "expected",
        # values obtained in xsdba v0.5
        [
            (
                [
                    267.061139,
                    267.347475,
                    267.58364,
                    267.816278,
                    268.093685,
                    268.326505,
                    268.485636,
                    268.684414,
                    268.863002,
                    268.969267,
                ]
            ),
        ],
    )
    def test_spectral_filter(self, gosset, expected):
        ds = xr.open_dataset(
            gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"),
            engine="h5netcdf",
        )
        # select lat/lon without nan values
        # spectral_filter not working if nan values are present (for now?)
        tx = ds.tasmax.isel(time=0).sel(lat=slice(50, 47), lon=slice(-80, -74))
        # using the default filter
        tx_filt = spectral_filter(
            tx,
            lam_long=None,
            lam_short=None,
            dims=["lon", "lat"],
            alpha_low_high=[0.9, 0.99],  # dummy value
        ).isel(lon=0)
        # performing dctn & idctn has a small inherent imprecision
        np.testing.assert_allclose(expected, tx_filt.values[0:10], rtol=1e-5)

    def test_spectral_filter_identity(self, gosset):
        ds = xr.open_dataset(
            gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"),
            engine="h5netcdf",
        )
        # select lat/lon without nan values
        # spectral_filter not working in this case, for now
        tx = ds.tasmax.isel(time=0).sel(lat=slice(50, 47), lon=slice(-80, -74))
        tx_filt = spectral_filter(
            tx,
            lam_long=None,
            lam_short=None,
            dims=["lon", "lat"],
            alpha_low_high=[0.9, 0.99],  # dummy value
            mask_func=lambda da, _1, _2: 0 * da + 1,  # identity function, mask =1
        )
        # performing dctn & idctn has a small inherent imprecision
        np.testing.assert_allclose(tx.values, tx_filt.values, rtol=1e-5)

    def test_spectral_filter_everthing(self, gosset):
        ds = xr.open_dataset(
            gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"),
            engine="h5netcdf",
        )
        tx = ds.tasmax.isel(time=0).sel(lat=slice(50, 47), lon=slice(-80, -74))
        tx_filt = spectral_filter(
            tx,
            lam_long=None,
            lam_short=None,
            dims=["lon", "lat"],
            alpha_low_high=[0.9, 0.99],  # dummy value
            mask_func=lambda da, _1, _2: 0 * da,  # mask =0
        )
        assert ((0 * tx).values == tx_filt.values).all()

    def test_normalized_radial_wavenumber(self, gosset):
        ds = xr.open_dataset(
            gosset.fetch("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"),
            engine="h5netcdf",
        )

        ds_sim = ds.tasmax.isel(time=0).sel(lat=slice(50, 49.5), lon=slice(-80, -79.5))
        alpha = _normalized_radial_wavenumber(ds_sim, ["lat", "lon"])
        # it is similar to the function itself, but a bit more human readable with the numpy notation
        # I think it's a good check
        alpha_by_hand = np.array(
            [
                [
                    np.sqrt((i / ds_sim.lon.size) ** 2 + (j / ds_sim.lat.size) ** 2)
                    for i in np.arange(ds_sim.lon.size)
                ]
                for j in np.arange(ds_sim.lat.size)
            ]
        )
        np.testing.assert_allclose(alpha.values, alpha_by_hand)
