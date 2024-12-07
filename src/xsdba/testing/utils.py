"""Testing utilities for xsdba."""

from __future__ import annotations

import importlib.resources as ilr
import logging
import os
import platform
import re
import time
import warnings
from datetime import datetime as dt
from pathlib import Path
from shutil import copytree
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

from filelock import FileLock
from packaging.version import Version

import xsdba
from xsdba import __version__ as __xsdba_version__

try:
    import pooch
except ImportError:
    warnings.warn(
        "The `pooch` library is not installed. "
        "The default cache directory for testing data will not be set."
    )
    pooch = None


logger = logging.getLogger("xclim")

__all__ = [
    "TESTDATA_BRANCH",
    "TESTDATA_CACHE_DIR",
    "TESTDATA_REPO_URL",
    "audit_url",
    "default_testdata_cache",
    "gather_testing_data",
    "gosset",
    "load_registry",
    "populate_testing_data",
    "testing_setup_warnings",
]

default_testdata_repo_url = (
    "https://raw.githubusercontent.com/Ouranosinc/xclim-testdata/"
)
"""Default URL of the testing data repository to use when fetching datasets."""

default_testdata_version = "v2024.8.23"
"""Default version of the testing data to use when fetching datasets."""

try:
    default_testdata_cache = Path(pooch.os_cache("xclim-testdata"))
    """Default location for the testing data cache."""
except AttributeError:
    default_testdata_cache = None


TESTDATA_REPO_URL = str(os.getenv("XSDBA_TESTDATA_REPO_URL", default_testdata_repo_url))
"""Sets the URL of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XSDBA_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata"

or setting the variable at runtime:

.. code-block:: console

    $ env XSDBA_TESTDATA_REPO_URL="https://github.com/my_username/xclim-testdata" pytest
"""

TESTDATA_BRANCH = str(os.getenv("XSDBA_TESTDATA_BRANCH", default_testdata_version))
"""Sets the branch of the testing data repository to use when fetching datasets.

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XSDBA_TESTDATA_BRANCH="my_testing_branch"

or setting the variable at runtime:

.. code-block:: console

    $ env XSDBA_TESTDATA_BRANCH="my_testing_branch" pytest
"""

TESTDATA_CACHE_DIR = os.getenv("XSDBA_TESTDATA_CACHE_DIR", default_testdata_cache)
"""Sets the directory to store the testing datasets.

If not set, the default location will be used (based on ``platformdirs``, see :func:`pooch.os_cache`).

Notes
-----
When running tests locally, this can be set for both `pytest` and `tox` by exporting the variable:

.. code-block:: console

    $ export XSDBA_TESTDATA_CACHE_DIR="/path/to/my/data"

or setting the variable at runtime:

.. code-block:: console

    $ env XSDBA_TESTDATA_CACHE_DIR="/path/to/my/data" pytest
"""


def testing_setup_warnings():
    """Warn users about potential incompatibilities between xsdba and xclim-testdata versions."""
    if (
        re.match(r"^\d+\.\d+\.\d+$", __xsdba_version__)
        and TESTDATA_BRANCH != default_testdata_version
    ):
        # This does not need to be emitted on GitHub Workflows and ReadTheDocs
        if not os.getenv("CI") and not os.getenv("READTHEDOCS"):
            warnings.warn(
                f"`xsdba` stable ({__xsdba_version__}) is running tests against a non-default branch of the testing data. "
                "It is possible that changes to the testing data may be incompatible with some assertions in this version. "
                f"Please be sure to check {TESTDATA_REPO_URL} for more information.",
            )

    if re.match(r"^v\d+\.\d+\.\d+", TESTDATA_BRANCH):
        # Find the date of last modification of xsdba source files to generate a calendar version
        install_date = dt.strptime(
            time.ctime(Path(xsdba.__file__).stat().st_mtime),
            "%a %b %d %H:%M:%S %Y",
        )
        install_calendar_version = (
            f"{install_date.year}.{install_date.month}.{install_date.day}"
        )

        if Version(TESTDATA_BRANCH) > Version(install_calendar_version):
            warnings.warn(
                f"The installation date of `xsdba` ({install_date.ctime()}) "
                f"predates the last release of testing data ({TESTDATA_BRANCH}). "
                "It is very likely that the testing data is incompatible with this build of `xsdba`.",
            )


def load_registry(
    branch: str = TESTDATA_BRANCH, repo: str = TESTDATA_REPO_URL
) -> dict[str, str]:
    """Load the registry file for the test data.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    remote_registry = audit_url(f"{repo}/{branch}/data/registry.txt")

    if branch != default_testdata_version:
        custom_registry_folder = Path(
            str(ilr.files("xsdba").joinpath(f"testing/{branch}"))
        )
        custom_registry_folder.mkdir(parents=True, exist_ok=True)
        registry_file = custom_registry_folder.joinpath("registry.txt")
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    elif repo != default_testdata_repo_url:
        registry_file = Path(str(ilr.files("xsdba").joinpath("testing/registry.txt")))
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    registry_file = Path(str(ilr.files("xsdba").joinpath("testing/registry.txt")))
    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    with registry_file.open(encoding="utf-8") as f:
        registry = {line.split()[0]: line.split()[1] for line in f}
    return registry


def gosset(  # noqa: PR01
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    cache_dir: str | Path = TESTDATA_CACHE_DIR,
    data_updates: bool = True,
):
    """Pooch registry instance for xsdba test data.

    Parameters
    ----------
    repo : str
        URL of the repository to use when fetching testing datasets.
    branch : str
        Branch of repository to use when fetching testing datasets.
    cache_dir : str or Path
        The path to the directory where the data files are stored.
    data_updates : bool
        If True, allow updates to the data files. Default is True.

    Returns
    -------
    pooch.Pooch
        The Pooch instance for accessing the xsdba testing data.

    Notes
    -----
    There are three environment variables that can be used to control the behaviour of this registry:
        - ``XSDBA_TESTDATA_CACHE_DIR``: If this environment variable is set, it will be used as the base directory to
          store the data files. The directory should be an absolute path (i.e., it should start with ``/``).
          Otherwise,the default location will be used (based on ``platformdirs``, see :py:func:`pooch.os_cache`).
        - ``XSDBA_TESTDATA_REPO_URL``: If this environment variable is set, it will be used as the URL of the repository
          to use when fetching datasets. Otherwise, the default repository will be used.
        - ``XSDBA_TESTDATA_BRANCH``: If this environment variable is set, it will be used as the branch of the repository
          to use when fetching datasets. Otherwise, the default branch will be used.

    Examples
    --------
    Using the registry to download a file:

    .. code-block:: python

        import xarray as xr
        from xsdba.testing.utilities import gosset

        example_file = gosset().fetch("example.nc")
        data = xr.open_dataset(example_file)
    """
    if pooch is None:
        raise ImportError(
            "The `pooch` package is required to fetch the xsdba testing data. "
            "You can install it with `pip install pooch` or `pip install xsdba[dev]`."
        )

    remote = audit_url(f"{repo}/{branch}/data")
    return pooch.create(
        path=cache_dir,
        base_url=remote,
        version=default_testdata_version,
        version_dev=branch,
        allow_updates=data_updates,
        registry=load_registry(branch=branch, repo=repo),
    )


def populate_testing_data(
    temp_folder: Path | None = None,
    repo: str = TESTDATA_REPO_URL,
    branch: str = TESTDATA_BRANCH,
    local_cache: Path = TESTDATA_CACHE_DIR,
) -> None:
    """Populate the local cache with the testing data.

    Parameters
    ----------
    temp_folder : Path, optional
        Path to a temporary folder to use as the local cache. If not provided, the default location will be used.
    repo : str, optional
        URL of the repository to use when fetching testing datasets.
    branch : str, optional
        Branch of xclim-testdata to use when fetching testing datasets.
    local_cache : Path
        The path to the local cache. Defaults to the location set by the platformdirs library.
        The testing data will be downloaded to this local cache.

    Returns
    -------
    None
    """
    # Create the Pooch instance
    n = gosset(repo=repo, branch=branch, cache_dir=temp_folder or local_cache)

    # Download the files
    errored_files = []
    for file in load_registry():
        try:
            n.fetch(file)
        except HTTPError:
            msg = f"File `{file}` not accessible in remote repository."
            logging.error(msg)
            errored_files.append(file)
        else:
            logging.info("Files were downloaded successfully.")

    if errored_files:
        logging.error(
            "The following files were unable to be downloaded: %s",
            errored_files,
        )


def gather_testing_data(
    worker_cache_dir: str | os.PathLike[str] | Path,
    worker_id: str,
    _cache_dir: str | os.PathLike[str] | None = TESTDATA_CACHE_DIR,
):
    """Gather testing data across workers."""
    if _cache_dir is None:
        raise ValueError(
            "The cache directory must be set. "
            "Please set the `cache_dir` parameter or the `XSDBA_TESTDATA_CACHE_DIR` environment variable."
        )
    cache_dir = Path(_cache_dir)

    if worker_id == "master":
        populate_testing_data(branch=TESTDATA_BRANCH)
    else:
        if platform.system() == "Windows":
            if not cache_dir.joinpath(default_testdata_version).exists():
                raise FileNotFoundError(
                    "Testing data not found and UNIX-style file-locking is not supported on Windows. "
                    "Consider running `xsdba.testing.utils.populate_testing_data()` to download testing data beforehand."
                )
        else:
            cache_dir.mkdir(exist_ok=True, parents=True)
            lockfile = cache_dir.joinpath(".lock")
            test_data_being_written = FileLock(lockfile)
            with test_data_being_written:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=TESTDATA_BRANCH)
                cache_dir.joinpath(".data_written").touch()
            with test_data_being_written.acquire():
                if lockfile.exists():
                    lockfile.unlink()
        copytree(cache_dir.joinpath(default_testdata_version), worker_cache_dir)


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
