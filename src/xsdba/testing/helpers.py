"""Helper functions for testing purposes."""

from __future__ import annotations
import collections

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gamma

from xsdba.utils import equally_spaced_nodes


__all__ = [
    "test_cannon_2015_dist",
    "test_cannon_2015_rvs",
    "test_timelonlatseries",
    "test_timeseries",
]


def test_cannon_2015_dist():  # noqa: D103
    # ref ~ gamma(k=4, theta=7.5)  mu: 30, sigma: 15
    ref = gamma(4, scale=7.5)

    # hist ~ gamma(k=8.15, theta=3.68) mu: 30, sigma: 10.5
    hist = gamma(8.15, scale=3.68)

    # sim ~ gamma(k=16, theta=2.63) mu: 42, sigma: 10.5
    sim = gamma(16, scale=2.63)

    return ref, hist, sim


def test_cannon_2015_rvs(n, random=True):  # noqa: D103
    # Frozen distributions
    fd = test_cannon_2015_dist()

    if random:
        r = [d.rvs(n) for d in fd]
    else:
        u = equally_spaced_nodes(n, None)
        r = [d.ppf(u) for d in fd]

    return map(lambda x: test_timelonlatseries(x, attrs={"units": "kg/m/m/s"}), r)


def test_timelonlatseries(values, attrs=None, start="2000-01-01"):
    """Create a DataArray with time, lon and lat dimensions."""
    attrs = {} if attrs is None else attrs
    coords = collections.OrderedDict()
    for dim, n in zip(("time", "lon", "lat"), values.shape, strict=False):
        if dim == "time":
            coords[dim] = pd.date_range(start, periods=n, freq="D")
        else:
            coords[dim] = xr.IndexVariable(dim, np.arange(n))

    return xr.DataArray(
        values,
        coords=coords,
        dims=list(coords.keys()),
        attrs=attrs,
    )


# XC
def test_timeseries(
    values,
    start: str = "2000-07-01",
    units: str | None = None,
    freq: str = "D",
    as_dataset: bool = False,
    cftime: bool = False,
    calendar: str | None = None,
) -> xr.DataArray | xr.Dataset:
    """
    Create a generic timeseries object based on pre-defined dictionaries of existing variables.

    Parameters
    ----------
    values : np.ndarray
        The values of the DataArray.
    start : str
        The start date of the time dimension. Default is "2000-07-01".
    units : str or None
        The units of the DataArray. Default is None.
    freq : str
        The frequency of the time dimension. Default is daily/"D".
    as_dataset : bool
        Whether to return a Dataset or a DataArray. Default is False.
    cftime : bool
        Whether to use cftime or not. Default is False.
    calendar : str or None
        Whether to use a calendar. If a calendar is provided, cftime is used.

    Returns
    -------
    xr.DataArray or xr.Dataset
        A DataArray or Dataset with time, lon and lat dimensions.
    """
    if calendar or cftime:
        coords = xr.date_range(start, periods=len(values), freq=freq, calendar=calendar or "standard", use_cftime=True)
    else:
        coords = pd.date_range(start, periods=len(values), freq=freq)

    attrs = {} if units is None else {"units": units}

    da = xr.DataArray(values, coords=[coords], dims="time", attrs=attrs)

    if as_dataset:
        return da.to_dataset()
    else:
        return da
