from __future__ import annotations

import pytest
import xarray as xr

from xsdba.units import harmonize_units, str2pint, units, units2pint


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
        assert str(u.units) == "mm d-1"

        u = units("percent")
        assert str(u.units) == "%"

    def test_units2pint(self, timelonlatseries):
        pytest.importorskip("cf-xarray")
        u = units2pint(timelonlatseries([1, 2], attrs={"units": "kg m-2 s-1"}))
        assert str(u) == "kg m-2 s-1"

        u = units2pint("m^3 s-1")
        assert str(u) == "m3 s-1"

        u = units2pint("%")
        assert str(u) == "%"

        u = units2pint("1")
        assert str(u) == ""

    def test_str2pint(self):
        Q_ = units.Quantity
        assert str2pint("-0.78 m") == Q_(-0.78, units="meter")
        assert str2pint("m kg/s") == Q_(1, units="meter kilogram/second")
        assert str2pint("11.8 degC days") == Q_(11.8, units="delta_degree_Celsius days")
        assert str2pint("nan m^2 K^-3").units == Q_(1, units="m²/K³").units


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
