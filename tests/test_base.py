# pylint: disable=missing-kwoa
from __future__ import annotations

import jsonpickle
import numpy as np
import pytest
import xarray as xr

from xsdba import set_options
from xsdba.base import Grouper, Parametrizable, map_blocks, map_groups


class ATestSubClass(Parametrizable):
    pass


def test_param_class():
    gr = Grouper(group="time.month")
    in_params = dict(
        anint=4, abool=True, astring="a string", adict={"key": "val"}, group=gr
    )
    obj = Parametrizable(**in_params)

    assert obj.parameters == in_params

    assert repr(obj).startswith(
        "Parametrizable(anint=4, abool=True, astring='a string', adict={'key': 'val'}, "
        "group=Grouper("
    )

    s = jsonpickle.encode(obj)
    obj2 = jsonpickle.decode(s)  # noqa: S301
    assert obj.parameters == obj2.parameters


@pytest.mark.parametrize(
    "group,window,nvals",
    [("time", 1, 366), ("time.month", 1, 31), ("time.dayofyear", 5, 1)],
)
def test_grouper_group(timeseries, group, window, nvals):
    da = timeseries(np.ones(366), start="2000-01-01")

    grouper = Grouper(group, window=window)
    grpd = grouper.group(da)

    if window > 1:
        assert "window" in grpd.dims

    assert grpd.count().max() == nvals


@pytest.mark.parametrize(
    "group,interp,val90,calendar",
    [
        ("time", False, True, None),
        ("time.month", False, 3, None),
        ("time.month", True, 3.5, None),
        ("time.season", False, 1, None),
        ("time.season", True, 0.8278688524590164, None),
        ("time.month", True, 3.533333333333333, "360_day"),
        ("time.month", True, 3.533333333333333, "noleap"),
        ("time.season", True, 0.8444444444444444, "360_day"),
        ("time.season", True, 0.8305936073059361, "noleap"),
    ],
)
def test_grouper_get_index(timeseries, group, interp, val90, calendar):
    da = timeseries(np.ones(366), start="2000-01-01", calendar=calendar)
    grouper = Grouper(group)
    indx = grouper.get_index(da, interp=interp)
    # 90 is March 31st
    assert indx[90] == val90


# xarray does not yet access "week" or "weekofyear" with groupby in a pandas-compatible way for cftime objects.
# See: https://github.com/pydata/xarray/discussions/6375
@pytest.mark.filterwarnings("ignore:dt.weekofyear and dt.week have been deprecated")
@pytest.mark.slow
@pytest.mark.parametrize(
    "group,n",
    [("time", 1), ("time.month", 12), ("time.week", 52)],
)
@pytest.mark.parametrize("use_dask", [True, False])
def test_grouper_apply(timeseries, use_dask, group, n):
    da1 = timeseries(np.arange(366), start="2000-01-01")
    da2 = timeseries(np.zeros(366), start="2000-01-01")
    da0 = xr.concat((da1, da2), dim="lat")

    grouper = Grouper(group)
    if not group.startswith("time"):
        da0 = da0.rename(time=grouper.dim)
        da1 = da1.rename(time=grouper.dim)
        da2 = da2.rename(time=grouper.dim)

    if use_dask:
        da0 = da0.chunk({"lat": 1, grouper.dim: -1})
        da1 = da1.chunk({grouper.dim: -1})
        da2 = da2.chunk({grouper.dim: -1})

    # Normal monthly mean
    out_mean = grouper.apply("mean", da0)
    if grouper.prop != "group":
        exp = da0.groupby(group).mean()
    else:
        exp = da0.mean(dim=grouper.dim).expand_dims("group").T
    np.testing.assert_array_equal(out_mean, exp)

    # With additional dimension included
    grouper = Grouper(group, add_dims=["lat"])
    out = grouper.apply("mean", da0)
    assert out.ndim == 1
    np.testing.assert_array_equal(out, exp.mean("lat"))
    assert out.attrs["group"] == group
    assert out.attrs["group_compute_dims"] == [grouper.dim, "lat"]
    assert out.attrs["group_window"] == 1

    # Additional but main_only
    out = grouper.apply("mean", da0, main_only=True)
    np.testing.assert_array_equal(out, out_mean)

    # With window (not applicable for group = 'time')
    if group != "time":
        win_grouper = Grouper(group, window=5)
        out = win_grouper.apply("mean", da0)
        rolld = da0.rolling({win_grouper.dim: 5}, center=True).construct(
            window_dim="window"
        )
        if grouper.prop != "group":
            exp = rolld.groupby(group).mean(dim=[win_grouper.dim, "window"])
        else:
            exp = rolld.mean(dim=[grouper.dim, "window"]).expand_dims("group").T
        np.testing.assert_array_equal(out, exp)

    # With function + nongrouping-grouped
    grouper = Grouper(group)

    def normalize(grp, dim):
        return grp / grp.mean(dim=dim)

    normed = grouper.apply(normalize, da0)
    assert normed.shape == da0.shape
    if use_dask:
        assert normed.chunks == ((1, 1), (366,))

    # With window + nongrouping-grouped
    if group != "time":
        out = win_grouper.apply(normalize, da0)
        assert out.shape == da0.shape

    # Mixed output
    def mixed_reduce(grdds, dim=None):
        da1 = grdds.da1.mean(dim=dim)
        da2 = grdds.da2 / grdds.da2.mean(dim=dim)
        da1.attrs["_group_apply_reshape"] = True
        return xr.Dataset(data_vars={"da1_mean": da1, "norm_da2": da2})

    out = grouper.apply(mixed_reduce, {"da1": da1, "da2": da2})
    assert grouper.prop not in out.norm_da2.dims
    assert grouper.prop in out.da1_mean.dims

    if use_dask:
        assert out.da1_mean.chunks == ((n,),)
        assert out.norm_da2.chunks == ((366,),)

    # Mixed input
    def normalize_from_precomputed(grpds, dim=None):
        return (grpds.da0 / grpds.da1_mean).mean(dim=dim)

    out = grouper.apply(
        normalize_from_precomputed, {"da0": da0, "da1_mean": out.da1_mean}
    ).isel(lat=0)
    if grouper.prop == "group":
        exp = normed.mean("time").isel(lat=0)
    else:
        exp = normed.groupby(group).mean().isel(lat=0)
    assert grouper.prop in out.dims
    np.testing.assert_allclose(out, exp, rtol=1e-10)


class TestMapBlocks:
    def test_lat_lon(self, timeseries):
        da0 = timeseries(np.arange(366), start="2000-01-01")
        da0 = da0.expand_dims(lat=[1, 2, 3, 4]).chunk()

        # Test dim parsing
        @map_blocks(reduces=["lat"], data=["lon"])
        def func(ds, *, group, lon=None):
            assert group.window == 5
            d = ds.da0.rename(lat="lon")
            return d.rename("data").to_dataset()

        # Raises on missing coords
        with pytest.raises(ValueError, match="This function adds the lon dimension*"):
            data = func(xr.Dataset(dict(da0=da0)), group="time.dayofyear", window=5)

        data = func(
            xr.Dataset(dict(da0=da0)),
            group="time.dayofyear",
            window=5,
            lon=[1, 2, 3, 4],
        ).load()
        assert set(data.data.dims) == {"time", "lon"}

    def test_grouper_prop(self, timeseries):
        da0 = timeseries(np.arange(366), start="2000-01-01")
        da0 = da0.expand_dims(lat=[1, 2, 3, 4]).chunk()

        @map_groups(data=[Grouper.PROP])
        def func(ds, *, dim):
            assert isinstance(dim, list)
            d = ds.da0.mean(dim)
            return d.rename("data").to_dataset()

        data = func(
            xr.Dataset(dict(da0=da0)),
            group="time.dayofyear",
            window=5,
            add_dims=["lat"],
        ).load()
        assert set(data.data.dims) == {"dayofyear"}

    def test_grouper_prop_main_only(self, timeseries):
        da0 = timeseries(np.arange(366), start="2000-01-01")
        da0 = da0.expand_dims(lat=[1, 2, 3, 4]).chunk()

        @map_groups(data=[Grouper.PROP], main_only=True)
        def func(ds, *, dim):
            assert isinstance(dim, str)
            data = ds.da0.mean(dim)
            return data.rename("data").to_dataset()

        # with a scalar aux coord
        data = func(
            xr.Dataset(dict(da0=da0.isel(lat=0, drop=True)), coords=dict(leftover=1)),
            group="time.dayofyear",
        ).load()
        assert set(data.data.dims) == {"dayofyear"}
        assert "leftover" in data

    def test_raises_error(self, timeseries):
        da0 = timeseries(np.arange(366), start="2000-01-01")
        da0 = da0.expand_dims(lat=[1, 2, 3, 4]).chunk(lat=1)

        # Test dim parsing
        @map_blocks(reduces=["lat"], data=[])
        def func(ds, *, group, lon=None):
            return ds.da0.rename("data").to_dataset()

        with pytest.raises(ValueError, match="cannot be chunked"):
            func(xr.Dataset(dict(da0=da0)), group="time")

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_dataarray_cfencode(self, use_dask, gosset):
        ds = xr.open_dataset(gosset.fetch("sdba/CanESM2_1950-2100.nc"))
        if use_dask:
            ds = ds.chunk()

        @map_blocks(reduces=["location"], data=[])
        def func(ds, *, group):
            d = ds.mean("location")
            return d.rename("data").to_dataset()

        func(ds.convert_calendar("noleap").tasmax, group=Grouper("time"))
