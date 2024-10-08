from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from cf_xarray import __version__ as __cfxr_version__
from packaging.version import Version

from xsdba.units import (
    harmonize_units,
    pint2str,
    str2pint,
    to_agg_units,
    units,
    units2pint,
)


class TestUnits:
    def test_temperature(self):
        assert 4 * units.d == 4 * units.day
        Q_ = units.Quantity
        assert Q_(1, units.C) == Q_(1, units.degC)

    def test_lat_lon(self):
        assert 100 * units.degreeN == 100 * units.degree

    def test_fraction(self):
        q = 5 * units.percent
        assert q.to("dimensionless") == 0.05

        q = 5 * units.parse_units("pct")
        assert q.to("dimensionless") == 0.05


class TestConvertUnitsTo:
    @pytest.mark.parametrize(
        "alias", [units("Celsius"), units("degC"), units("C"), units("deg_C")]
    )
    def test_temperature_aliases(self, alias):
        assert alias == units("celsius")


class TestUnitConversion:
    def test_pint2str(self):
        pytest.importorskip("cf-xarray")
        u = units("mm/d")
        assert pint2str(u.units) == "mm d-1"

        u = units("percent")
        assert pint2str(u.units) == "%"

        u = units("pct")
        assert pint2str(u.units) == "%"

    def test_units2pint(self, timelonlatseries):
        pytest.importorskip("cf-xarray")
        u = units2pint(timelonlatseries([1, 2], attrs={"units": "kg m-2 s-1"}))
        assert pint2str(u) == "kg m-2 s-1"

        u = units2pint("m^3 s-1")
        assert pint2str(u) == "m3 s-1"

        u = units2pint("%")
        assert pint2str(u) == "%"

        u = units2pint("1")
        assert pint2str(u) == ""

    def test_str2pint(self):
        Q_ = units.Quantity
        assert str2pint("-0.78 m") == Q_(-0.78, units="meter")
        assert str2pint("m kg/s") == Q_(1, units="meter kilogram/second")
        assert str2pint("11.8 degC days") == Q_(11.8, units="delta_degree_Celsius days")
        assert str2pint("nan m^2 K^-3").units == Q_(1, units="m²/K³").units


@pytest.mark.parametrize(
    "in_u,opfunc,op,exp,exp_u",
    [
        ("m/h", "sum", "integral", 8760, "m"),
        ("m/h", "sum", "sum", 365, "m/h"),
        ("K", "mean", "mean", 1, "K"),
        ("", "sum", "count", 365, "d"),
        ("", "sum", "count", 365, "d"),
        ("kg m-2", "var", "var", 0, "kg2 m-4"),
        ("°C", "argmax", "doymax", 0, ("", "1")),  # dependent on numpy/pint version
        (
            "°C",
            "sum",
            "integral",
            365,
            ("K d", "d K"),
        ),  # dependent on numpy/pint version
        ("°F", "sum", "integral", 365, "d °R"),  # not sure why the order is different
    ],
)
def test_to_agg_units(in_u, opfunc, op, exp, exp_u):
    da = xr.DataArray(
        np.ones((365,)),
        dims=("time",),
        coords={"time": xr.cftime_range("1993-01-01", periods=365, freq="D")},
        attrs={"units": in_u},
    )

    out = to_agg_units(getattr(da, opfunc)(), da, op)
    np.testing.assert_allclose(out, exp)

    if isinstance(exp_u, tuple):
        if Version(__cfxr_version__) < Version("0.9.3"):
            assert out.attrs["units"] == exp_u[0]
        else:
            assert out.attrs["units"] == exp_u[1]
    else:
        assert out.attrs["units"] == exp_u


class TestHarmonizeUnits:
    def test_simple(self):
        da = xr.DataArray([1, 2], attrs={"units": "K"})
        thr = "1 K"

        @harmonize_units(["d", "t"])
        def gt(d, t):
            return (d > t).sum().values

        assert gt(da, thr) == 1

    def test_no_units(self):
        da = xr.DataArray([1, 2])
        thr = 1

        @harmonize_units(["d", "t"])
        def gt(d, t):
            return (d > t).sum().values

        assert gt(da, thr) == 1

    def test_wrong_decorator(self):
        da = xr.DataArray([1, 2], attrs={"units": "K"})
        thr = "1 K"

        @harmonize_units(["d", "this_is_clearly_wrong"])
        def gt(d, t):
            return (d > t).sum().values

        with pytest.raises(TypeError, match="should be a subset of"):
            gt(da, thr)

    def test_wrong_input_catched_by_decorator(self):
        da = xr.DataArray([1, 2], attrs={"units": "K"})
        thr = "1 K"

        @harmonize_units(["d", "t"])
        def gt(d, t):
            return (d > t).sum().values

        with pytest.raises(TypeError, match="were passed but only"):
            gt(da)
