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

from xsdba.calendar import percentile_doy
from xsdba.utils import equally_spaced_nodes

__all__ = ["test_timelonlatseries", "test_timeseries"]

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
def file_md5_checksum(f_name):
    hash_md5 = hashlib.md5()  # noqa: S324
    with open(f_name, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


# XC
def audit_url(url: str, context: str | None = None) -> str:
    """Check if the URL is well-formed.

    Raises
    ------
    URLError
        If the URL is not well-formed.
    """
    msg = ""
    result = urlparse(url)
    if result.scheme == "http":
        msg = f"{context if context else ''} URL is not using secure HTTP: '{url}'".strip()
    if not all([result.scheme, result.netloc]):
        msg = f"{context if context else ''} URL is not well-formed: '{url}'".strip()

    if msg:
        logger.error(msg)
        raise URLError(msg)
    return url


# XC (oh dear)
def _get(
    fullname: Path,
    github_url: str,
    branch: str,
    suffix: str,
    cache_dir: Path,
) -> Path:
    cache_dir = cache_dir.absolute()
    local_file = cache_dir / branch / fullname
    md5_name = fullname.with_suffix(f"{suffix}.md5")
    md5_file = cache_dir / branch / md5_name

    if not github_url.lower().startswith("http"):
        raise ValueError(f"GitHub URL not safe: '{github_url}'.")

    if local_file.is_file():
        local_md5 = file_md5_checksum(local_file)
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            msg = f"Attempting to fetch remote file md5: {md5_name.as_posix()}"
            logger.info(msg)
            urlretrieve(url, md5_file)  # nosec
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"MD5 checksum for {local_file.as_posix()} does not match upstream md5. "
                    "Attempting new download."
                )
                warnings.warn(msg)
        except HTTPError:
            msg = (
                f"{md5_name.as_posix()} not accessible in remote repository. "
                "Unable to determine validity with upstream repo."
            )
            warnings.warn(msg)
        except URLError:
            msg = (
                f"{md5_name.as_posix()} not found in remote repository. "
                "Unable to determine validity with upstream repo."
            )
            warnings.warn(msg)
        except SocketBlockedError:
            msg = f"Unable to access {md5_name.as_posix()} online. Testing suite is being run with `--disable-socket`."
            warnings.warn(msg)

    if not local_file.is_file():
        # This will always leave this directory on disk.
        # We may want to add an option to remove it.
        local_file.parent.mkdir(exist_ok=True, parents=True)

        url = "/".join((github_url, "raw", branch, fullname.as_posix()))
        msg = f"Fetching remote file: {fullname.as_posix()}"
        logger.info(msg)
        try:
            urlretrieve(url, local_file)  # nosec
        except HTTPError as e:
            msg = f"{fullname.as_posix()} not accessible in remote repository. Aborting file retrieval."
            raise FileNotFoundError(msg) from e
        except URLError as e:
            msg = (
                f"{fullname.as_posix()} not found in remote repository. "
                "Verify filename and repository address. Aborting file retrieval."
            )
            raise FileNotFoundError(msg) from e
        # gives TypeError: catching classes that do not inherit from BaseException is not allowed
        except SocketBlockedError as e:
            msg = (
                f"Unable to access {fullname.as_posix()} online. Testing suite is being run with `--disable-socket`. "
                f"If you intend to run tests with this option enabled, please download the file beforehand with the "
                f"following console command: `xclim prefetch_testing_data`."
            )
            raise FileNotFoundError(msg) from e
        try:
            url = "/".join((github_url, "raw", branch, md5_name.as_posix()))
            msg = f"Fetching remote file md5: {md5_name.as_posix()}"
            logger.info(msg)
            urlretrieve(url, md5_file)  # nosec
        except (HTTPError, URLError) as e:
            msg = (
                f"{md5_name.as_posix()} not accessible online. "
                "Unable to determine validity of file from upstream repo. "
                "Aborting file retrieval."
            )
            local_file.unlink()
            raise FileNotFoundError(msg) from e

        local_md5 = file_md5_checksum(local_file)
        try:
            with open(md5_file) as f:
                remote_md5 = f.read()
            if local_md5.strip() != remote_md5.strip():
                local_file.unlink()
                msg = (
                    f"{local_file.as_posix()} and md5 checksum do not match. "
                    "There may be an issue with the upstream origin data."
                )
                raise OSError(msg)
        except OSError as e:
            logger.error(e)

    return local_file


# XC
# idea copied from xclim that it borrowed from raven that it borrowed from xclim that borrowed it from xarray that was borrowed from Seaborn
def open_dataset(
    name: str | os.PathLike[str],
    suffix: str | None = None,
    dap_url: str | None = None,
    github_url: str = "https://github.com/Ouranosinc/xclim-testdata",
    branch: str = "main",
    cache: bool = True,
    cache_dir: Path = _default_cache_dir,
    **kwargs,
) -> xr.Dataset:
    r"""Open a dataset from the online GitHub-like repository.

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str or os.PathLike
        Name of the file containing the dataset.
    suffix : str, optional
        If no suffix is given, assumed to be netCDF ('.nc' is appended). For no suffix, set "".
    dap_url : str, optional
        URL to OPeNDAP folder where the data is stored. If supplied, supersedes github_url.
    github_url : str
        URL to GitHub repository where the data is stored.
    branch : str, optional
        For GitHub-hosted files, the branch to download from.
    cache : bool
        If True, then cache data locally for use on subsequent calls.
    cache_dir : Path
        The directory in which to search for and write cached data.
    \*\*kwargs : dict
        For NetCDF files, keywords passed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    Union[Dataset, Path]

    See Also
    --------
    xarray.open_dataset
    """
    if isinstance(name, (str, os.PathLike)):
        name = Path(name)
    if suffix is None:
        suffix = ".nc"
    fullname = name.with_suffix(suffix)

    if dap_url is not None:
        dap_file_address = urljoin(dap_url, str(name))
        try:
            ds = _open_dataset(audit_url(dap_file_address, context="OPeNDAP"), **kwargs)
            return ds
        except URLError:
            raise
        except OSError:
            msg = f"OPeNDAP file not read. Verify that the service is available: '{dap_file_address}'"
            logger.error(msg)
            raise OSError(msg)

    local_file = _get(
        fullname=fullname,
        github_url=github_url,
        branch=branch,
        suffix=suffix,
        cache_dir=cache_dir,
    )

    try:
        ds = _open_dataset(local_file, **kwargs)
        if not cache:
            ds = ds.load()
            local_file.unlink()
        return ds
    except OSError as err:
        raise err


# XC
def nancov(X):
    """Drop observations with NaNs from Numpy's cov."""
    X_na = np.isnan(X).any(axis=0)
    return np.cov(X[:, ~X_na])


# XC
def generate_atmos(cache_dir: Path) -> dict[str, xr.DataArray]:
    """Create the `atmosds` synthetic testing dataset."""
    with open_dataset(
        "ERA5/daily_surface_cancities_1990-1993.nc",
        cache_dir=cache_dir,
        branch=TESTDATA_BRANCH,
        engine="h5netcdf",
    ) as ds:
        tn10 = percentile_doy(ds.tasmin, per=10)
        t10 = percentile_doy(ds.tas, per=10)
        t90 = percentile_doy(ds.tas, per=90)
        tx90 = percentile_doy(ds.tasmax, per=90)

        # rsus = shortwave_upwelling_radiation_from_net_downwelling(ds.rss, ds.rsds)
        # rlus = longwave_upwelling_radiation_from_net_downwelling(ds.rls, ds.rlds)

        ds = ds.assign(
            # rsus=rsus,
            # rlus=rlus,
            tn10=tn10,
            t10=t10,
            t90=t90,
            tx90=tx90,
        )

        # Create a file in session scoped temporary directory
        atmos_file = cache_dir.joinpath("atmosds.nc")
        ds.to_netcdf(atmos_file, engine="h5netcdf")

    # Give access to dataset variables by name in namespace
    namespace = dict()
    with open_dataset(
        atmos_file, branch=TESTDATA_BRANCH, cache_dir=cache_dir, engine="h5netcdf"
    ) as ds:
        for variable in ds.data_vars:
            namespace[f"{variable}_dataset"] = ds.get(variable)
    return namespace
