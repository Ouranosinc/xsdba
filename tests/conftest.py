# noqa: D104
# XC: Many things deactivated, not sure what will be necessary
from __future__ import annotations

import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# from filelock import FileLock
from packaging.version import Version

from xsdba.testing import TESTDATA_BRANCH, generate_atmos
from xsdba.testing import open_dataset as _open_dataset
from xsdba.testing import (
    test_cannon_2015_dist,
    test_cannon_2015_rvs,
    test_timelonlatseries,
    test_timeseries,
)
from xsdba.utils import apply_correction, equally_spaced_nodes

# import xclim
# from xclim import __version__ as __xclim_version__
# from xclim.core.calendar import max_doy
# from xclim.testing import helpers
# from xclim.testing.utils import _default_cache_dir
# from xclim.testing.utils import get_file
# from xclim.testing.utils import open_dataset as _open_dataset

# ADAPT
# if (
#     re.match(r"^\d+\.\d+\.\d+$", __xclim_version__)
#     and helpers.TESTDATA_BRANCH == "main"
# ):
#     # This does not need to be emitted on GitHub Workflows and ReadTheDocs
#     if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
#         warnings.warn(
#             f'`xclim` {__xclim_version__} is running tests against the "main" branch of `Ouranosinc/xclim-testdata`. '
#             "It is possible that changes in xclim-testdata may be incompatible with test assertions in this version. "
#             "Please be sure to check https://github.com/Ouranosinc/xclim-testdata for more information.",
#             UserWarning,
#         )

# if re.match(r"^v\d+\.\d+\.\d+", helpers.TESTDATA_BRANCH):
#     # Find the date of last modification of xclim source files to generate a calendar version
#     install_date = dt.strptime(
#         time.ctime(os.path.getmtime(xclim.__file__)),
#         "%a %b %d %H:%M:%S %Y",
#     )
#     install_calendar_version = (
#         f"{install_date.year}.{install_date.month}.{install_date.day}"
#     )

#     if Version(helpers.TESTDATA_BRANCH) > Version(install_calendar_version):
#         warnings.warn(
#             f"Installation date of `xclim` ({install_date.ctime()}) "
#             f"predates the last release of `xclim-testdata` ({helpers.TESTDATA_BRANCH}). "
#             "It is very likely that the testing data is incompatible with this build of `xclim`.",
#             UserWarning,
#         )


@pytest.fixture
def cannon_2015_rvs():
    return test_cannon_2015_rvs


@pytest.fixture
def cannon_2015_dist():
    return test_cannon_2015_dist


@pytest.fixture
# FIXME: can't find `socket_enable` fixture
# def ref_hist_sim_tuto(socket_enabled):  # noqa: F841
def ref_hist_sim_tuto():  # noqa: F841
    """Return ref, hist, sim time series of air temperature.

    socket_enabled is a fixture that enables the use of the internet to download the tutorial dataset while the
    `--disable-socket` flag has been called. This fixture will crash if the `air_temperature` tutorial file is
    not on disk while the internet is unavailable.
    """

    def _ref_hist_sim_tuto(sim_offset=3, delta=0.1, smth_win=3, trend=True):
        ds = xr.tutorial.open_dataset("air_temperature")
        ref = ds.air.resample(time="D").mean(keep_attrs=True)
        hist = ref.rolling(time=smth_win, min_periods=1).mean(keep_attrs=True) + delta
        hist.attrs["units"] = ref.attrs["units"]
        sim_time = hist.time + np.timedelta64(730 + sim_offset * 365, "D").astype(
            "<m8[ns]"
        )
        sim = hist + (
            0
            if not trend
            else xr.DataArray(
                np.linspace(0, 2, num=hist.time.size),
                dims=("time",),
                coords={"time": hist.time},
                attrs={"units": hist.attrs["units"]},
            )
        )
        sim["time"] = sim_time
        return ref, hist, sim

    return _ref_hist_sim_tuto


@pytest.fixture
def random() -> np.random.Generator:
    return np.random.default_rng(seed=list(map(ord, "𝕽𝔞𝖓𝔡𝖔𝔪")))


# ADAPT
# @pytest.fixture
# def tmp_netcdf_filename(tmpdir) -> Path:
#     yield Path(tmpdir).joinpath("testfile.nc")


@pytest.fixture(autouse=True, scope="session")
def threadsafe_data_dir(tmp_path_factory) -> Path:
    yield Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def open_dataset(threadsafe_data_dir):
    def _open_session_scoped_file(
        file: str | os.PathLike, branch: str = TESTDATA_BRANCH, **xr_kwargs
    ):
        xr_kwargs.setdefault("engine", "h5netcdf")
        return _open_dataset(
            file, cache_dir=threadsafe_data_dir, branch=branch, **xr_kwargs
        )

    return _open_session_scoped_file


# XC
@pytest.fixture
def mon_triangular():
    return np.array(list(range(1, 7)) + list(range(7, 1, -1))) / 7


# XC (name changed)
@pytest.fixture
def mon_timelonlatseries(timelonlatseries, mon_triangular):
    def _mon_timelonlatseries(values, attrs):
        """Random time series whose mean varies over a monthly cycle."""
        x = timelonlatseries(values, attrs)
        m = mon_triangular
        factor = timelonlatseries(m[x.time.dt.month - 1], attrs)

        with xr.set_options(keep_attrs=True):
            return apply_correction(x, factor, x.kind)

    return _mon_timelonlatseries


@pytest.fixture
def timelonlatseries():
    return test_timelonlatseries


# ADAPT
# @pytest.fixture
# def per_doy():
#     def _per_doy(values, calendar="standard", units="kg m-2 s-1"):
#         n = max_doy[calendar]
#         if len(values) != n:
#             raise ValueError(
#                 "Values must be same length as number of days in calendar."
#             )
#         coords = xr.IndexVariable("dayofyear", np.arange(1, n + 1))
#         return xr.DataArray(
#             values, coords=[coords], attrs={"calendar": calendar, "units": units}
#         )

#     return _per_doy


@pytest.fixture
def areacella() -> xr.DataArray:
    """Return a rectangular grid of grid cell area."""
    r = 6100000
    lon_bnds = np.arange(-180, 181, 1)
    lat_bnds = np.arange(-90, 91, 1)
    d_lon = np.diff(lon_bnds)
    d_lat = np.diff(lat_bnds)
    lon = np.convolve(lon_bnds, [0.5, 0.5], "valid")
    lat = np.convolve(lat_bnds, [0.5, 0.5], "valid")
    area = (
        r
        * np.radians(d_lat)[:, np.newaxis]
        * r
        * np.cos(np.radians(lat)[:, np.newaxis])
        * np.radians(d_lon)
    )
    return xr.DataArray(
        data=area,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        attrs={"r": r, "units": "m2", "standard_name": "cell_area"},
    )


areacello = areacella


# ADAPT?
# @pytest.fixture(scope="session")
# def open_dataset(threadsafe_data_dir):
#     def _open_session_scoped_file(
#         file: str | os.PathLike, branch: str = helpers.TESTDATA_BRANCH, **xr_kwargs
#     ):
#         xr_kwargs.setdefault("engine", "h5netcdf")
#         return _open_dataset(
#             file, cache_dir=threadsafe_data_dir, branch=branch, **xr_kwargs
#         )

#     return _open_session_scoped_file


# ADAPT?
# @pytest.fixture(autouse=True, scope="session")
# def add_imports(xdoctest_namespace, threadsafe_data_dir) -> None:
#     """Add these imports into the doctests scope."""
#     ns = xdoctest_namespace
#     ns["np"] = np
#     ns["xr"] = xclim.testing  # xr.open_dataset(...) -> xclim.testing.open_dataset(...)
#     ns["xclim"] = xclim
#     ns["open_dataset"] = partial(
#         _open_dataset,
#         cache_dir=threadsafe_data_dir,
#         branch=helpers.TESTDATA_BRANCH,
#         engine="h5netcdf",
#     )  # Needed for modules where xarray is imported as `xr`


@pytest.fixture(autouse=True, scope="function")
def add_example_dataarray(xdoctest_namespace, timeseries) -> None:
    ns = xdoctest_namespace
    ns["da"] = timeseries(np.random.rand(365) * 20 + 253.15)


@pytest.fixture(autouse=True, scope="session")
def is_matplotlib_installed(xdoctest_namespace) -> None:
    def _is_matplotlib_installed():
        try:
            import matplotlib

            return
        except ImportError:
            return pytest.skip("This doctest requires matplotlib to be installed.")

    ns = xdoctest_namespace
    ns["is_matplotlib_installed"] = _is_matplotlib_installed


# ADAPT or REMOVE?
@pytest.fixture(scope="function")
def atmosds(threadsafe_data_dir) -> xr.Dataset:
    return _open_dataset(
        threadsafe_data_dir.joinpath("atmosds.nc"),
        cache_dir=threadsafe_data_dir,
        branch=TESTDATA_BRANCH,
        engine="h5netcdf",
    ).load()


# @pytest.fixture(scope="function")
# def ensemble_dataset_objects() -> dict:
#     edo = dict()
#     edo["nc_files_simple"] = [
#         "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
#         "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
#         "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
#         "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
#     ]
#     edo["nc_files_extra"] = [
#         "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc"
#     ]
#     edo["nc_files"] = edo["nc_files_simple"] + edo["nc_files_extra"]
#     return edo


# @pytest.fixture(scope="session")
# def lafferty_sriver_ds() -> xr.Dataset:
#     """Get data from Lafferty & Sriver unit test.

#     Notes
#     -----
#     https://github.com/david0811/lafferty-sriver_2023_npjCliAtm/tree/main/unit_test
#     """
#     fn = get_file(
#         "uncertainty_partitioning/seattle_avg_tas.csv",
#         cache_dir=_default_cache_dir,
#         branch=helpers.TESTDATA_BRANCH,
#     )

#     df = pd.read_csv(fn, parse_dates=["time"]).rename(
#         columns={"ssp": "scenario", "ensemble": "downscaling"}
#     )

#     # Make xarray dataset
#     return xr.Dataset.from_dataframe(
#         df.set_index(["scenario", "model", "downscaling", "time"])
#     )


@pytest.fixture(scope="session", autouse=True)
def gather_session_data(threadsafe_data_dir):
    """Gather testing data on pytest run.

    When running pytest with multiple workers, one worker will copy data remotely to _default_cache_dir while
    other workers wait using lockfile. Once the lock is released, all workers will then copy data to their local
    threadsafe_data_dir.As this fixture is scoped to the session, it will only run once per pytest run.

    Additionally, this fixture is also used to generate the `atmosds` synthetic testing dataset as well as add the
    example file paths to the xdoctest_namespace, used when running doctests.
    """
    generate_atmos(threadsafe_data_dir)


#     if (
#         not _default_cache_dir.joinpath(helpers.TESTDATA_BRANCH).exists()
#         or helpers.PREFETCH_TESTING_DATA
#     ):
#         if helpers.PREFETCH_TESTING_DATA:
#             print("`XCLIM_PREFETCH_TESTING_DATA` set. Prefetching testing data...")
#         if sys.platform == "win32":
#             raise OSError(
#                 "UNIX-style file-locking is not supported on Windows. "
#                 "Consider running `$ xclim prefetch_testing_data` to download testing data."
#             )
#         elif worker_id in ["master"]:
#             helpers.populate_testing_data(branch=helpers.TESTDATA_BRANCH)
#         else:
#             _default_cache_dir.mkdir(exist_ok=True, parents=True)
#             lockfile = _default_cache_dir.joinpath(".lock")
#             test_data_being_written = FileLock(lockfile)
#             with test_data_being_written:
#                 # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
#                 helpers.populate_testing_data(branch=helpers.TESTDATA_BRANCH)
#                 _default_cache_dir.joinpath(".data_written").touch()
#             with test_data_being_written.acquire():
#                 if lockfile.exists():
#                     lockfile.unlink()
#     shutil.copytree(_default_cache_dir, threadsafe_data_dir)
#     xdoctest_namespace.update(helpers.add_example_file_paths(threadsafe_data_dir))


# @pytest.fixture(scope="session", autouse=True)
# def cleanup(request):
#     """Cleanup a testing file once we are finished.

#     This flag prevents remote data from being downloaded multiple times in the same pytest run.
#     """

#     def remove_data_written_flag():
#         flag = _default_cache_dir.joinpath(".data_written")
#         if flag.exists():
#             flag.unlink()

#     request.addfinalizer(remove_data_written_flag)


@pytest.fixture
def timeseries():
    return test_timeseries
