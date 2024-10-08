"""Testing utilities for xsdba."""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import xarray as xr
from platformdirs import user_cache_dir
from scipy.stats import gamma
from xarray import open_dataset as _open_dataset

from xsdba.utils import equally_spaced_nodes

__all__ = ["nancov", "test_timelonlatseries", "test_timeseries"]

# keeping xclim-testdata for now, since it's still this on gitHub
_default_cache_dir = Path(user_cache_dir("xclim-testdata"))

# XC
TESTDATA_BRANCH = os.getenv("XCLIM_TESTDATA_BRANCH", "main")
"""Sets the branch of Ouranosinc/xclim-testdata to use when fetching testing datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XCLIM_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XCLIM_TESTDATA_BRANCH="my_testing_branch" pytest

"""

logger = logging.getLogger("xsdba")

try:
    from pytest_socket import SocketBlockedError
except ImportError:
    SocketBlockedError = None


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
    for dim, n in zip(("time", "lon", "lat"), values.shape):
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
) -> xr.DataArray | xr.Dataset:
    """Create a generic timeseries object based on pre-defined dictionaries of existing variables."""
    if cftime:
        coords = xr.cftime_range(start, periods=len(values), freq=freq)
    else:
        coords = pd.date_range(start, periods=len(values), freq=freq)

    attrs = {} if units is None else {"units": units}

    da = xr.DataArray(values, coords=[coords], dims="time", attrs=attrs)

    if as_dataset:
        return da.to_dataset()
    else:
        return da


# XC
def nancov(X):
    """Drop observations with NaNs from Numpy's cov."""
    X_na = np.isnan(X).any(axis=0)
    return np.cov(X[:, ~X_na])
