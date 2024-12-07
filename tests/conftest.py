# noqa: D104
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xsdba.testing import (
    test_cannon_2015_dist,
    test_cannon_2015_rvs,
    test_timelonlatseries,
    test_timeseries,
)
from xsdba.testing.utils import (
    TESTDATA_BRANCH,
    TESTDATA_CACHE_DIR,
    TESTDATA_REPO_URL,
    default_testdata_cache,
    gather_testing_data,
)
from xsdba.testing.utils import gosset as _gosset
from xsdba.testing.utils import testing_setup_warnings
from xsdba.utils import apply_correction


@pytest.fixture
def cannon_2015_dist():
    return test_cannon_2015_dist


@pytest.fixture
def cannon_2015_rvs():
    return test_cannon_2015_rvs


@pytest.fixture
def timelonlatseries():
    return test_timelonlatseries


@pytest.fixture
def timeseries():
    return test_timeseries


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


@pytest.fixture(autouse=True, scope="function")
def add_example_dataarray(xdoctest_namespace, timeseries) -> None:
    ns = xdoctest_namespace
    ns["da"] = timeseries(np.random.rand(365) * 20 + 253.15)


@pytest.fixture(scope="session")
def threadsafe_data_dir(tmp_path_factory):
    return Path(tmp_path_factory.getbasetemp().joinpath("data"))


@pytest.fixture(scope="session")
def gosset(threadsafe_data_dir, worker_id):
    return _gosset(
        repo=TESTDATA_REPO_URL,
        branch=TESTDATA_BRANCH,
        cache_dir=(
            TESTDATA_CACHE_DIR if worker_id == "master" else threadsafe_data_dir
        ),
    )


@pytest.fixture
def tmp_netcdf_filename(tmpdir) -> Path:
    yield Path(tmpdir).joinpath("testfile.nc")


@pytest.fixture(autouse=True, scope="session")
def gather_session_data(request, gosset, worker_id):
    """Gather testing data on pytest run.

    When running pytest with multiple workers, one worker will copy data remotely to default cache dir while
    other workers wait using lockfile. Once the lock is released, all workers will then copy data to their local
    threadsafe_data_dir. As this fixture is scoped to the session, it will only run once per pytest run.
    """
    testing_setup_warnings()
    gather_testing_data(worker_cache_dir=gosset.path, worker_id=worker_id)

    def remove_data_written_flag():
        """Cleanup cache folder once we are finished."""
        flag = default_testdata_cache.joinpath(".data_written")
        if flag.exists():
            flag.unlink()

    request.addfinalizer(remove_data_written_flag)
