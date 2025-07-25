=========
Changelog
=========

..
    `Unreleased <https://github.com/Ouranosinc/xsdba>`_ (latest)
    ------------------------------------------------------------

    Contributors:

    Changes
    ^^^^^^^
    * No change.

    Fixes
    ^^^^^
    * No change.

.. _changes_0.5.0:

`v0.5.0 <https://github.com/Ouranosinc/xsdba/tree/0.5.0>`_ (2025-07-21)
-----------------------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`), Éric Dupuis (:user:`coxipi`), Juliette Lavoie (:user:`juliettelavoie`), Pascal Bourgault (:user:`aulemahal`).

Changes
^^^^^^^
* Make additional grouping dimensions optional for methods accepting the ``group`` argument, except ``Loci`` and ``PrincipalComponents``. (:issue:`99`, :issue:`144`, :pull:`151`).
* Speed up import by activating `cache=True` for in numba-accelerated functions from ``xsdba.nbutils``. (:pull:`135`).
* Added a new installation recipe (``pip install xsdba[sbck]``) for installing the `SBCK` package. (:pull:`139`):
    * Note that `SBCK` support is experimental and that the `pybind11` library must be installed prior to installing `SBCK`.
* New functions related to spectral properties in Fourier space:
    * Perform a spectral filter with ``xsdba.processing.spectral_filter`` with a low-pass filter with a cosine-squared profile by default. (:pull:`88`).
    * New spatial diagnostic to compute the spectral variance of a given field ``xsdba.properties.spectral_variance``. (:pull:`88`).
* ``xsdba.units.convert_units_to`` now wraps a private function ``_convert_units_to``. (:pull:`145`).
* ``xsdba.jitter_over_thresh`` is available directly in training methods by passing the `jitter_over_thresh_value` and `jitter_over_thresh_upper_bnd`  arguments. (:pull:`110`).
* Throw an error if `group=Grouper('5D',window)` is used with a biasadjust method other than `MBCn`.
* ``xsdba.processing.to_additive_space`` accepts `clip_next_to_bounds`, which avoids infinities by ensuring `lower_bound < data < upper_bound`. (:issue:`164`, :pull:`165`).
* Allow nan values in ``xsdba.measures.rmse`` and ``xsdba.measures.mae``. (:pull:`170`).
* The adaptation of frequencies through `adapt_freq_thresh_value` is now applied in the adjusting step as well. (:pull:`160`).
* ``xsdba.adjustment.ExtremeValues`` now accepts a DataArray for `cluster_thresh`, letting specify distinct thresholds for multiple locations. (:issue:`177`, :pull:`179`).
* Updated minimum supported versions of `SBCK` (v1.4.2) and `numpy` (v1.25). (:pull:`180`).

Fixes
^^^^^
* Fix ``xsdba.base.get_coordinates`` to avoid using a private xarray function.(:pull:`147`, :issue:`148`).
* Fix ``xsdba.processing.from_additive_space`` to handles units correctly by using `convert_units_to` instead of `harmonize_units`. (:pull:`146`).
* Fix the order of `clip_next_to_bounds` in ``xsdba.processing.to_additive_space``. (:pull:`169`).

Internal changes
^^^^^^^^^^^^^^^^
* The `tox` and CI configurations now support the installation of `SBCK` and `Eigen3` for testing purposes. (:pull:`139`).
* The `coveralls` tox keyword has been renamed to `coverage` to avoid confusion with the `coveralls` service. (:pull:`139`).
* The order of arguments in the following private functions was changed: ``xsdba._adjustment.{_fit_on_cluster,_fit_cluster_and_cdf, _extremes_train_1d}``.
* Updated the package metadata to reflect development progress and list user :user:`aulemahal` as a primary developer (:pull:`180`).

.. _changes_0.4.0:

`v0.4.0 <https://github.com/Ouranosinc/xsdba/tree/0.4.0>`_ (2025-04-03)
-----------------------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`), Jan Haacker (:user:`j-haacker`), Éric Dupuis (:user:`coxipi`).

Changes
^^^^^^^
* `xsdba` now supports Python3.13. Metadata and CI have been adjusted. (:pull:`105`).
* Unpinned `numpy` and raised minimum supported versions of a few scientific libraries. (:pull:`105`).
* More code that needed to be ported from `xclim` has been added. This includes mainly documentation, as well as testing utilities and a benchmark notebook. (:pull:`107`).

Fixes
^^^^^
* For `fastnanquantile`, `POT`, and `xclim` have been added to a new `extras` install recipe. All dependencies can be installed using the ``$ python -m pip install xsdba[all]`` command. Documentation has been added. (:pull:`105`).
* Several small `dask`-related issues (chunking behaviour, dimension order when broadcasting variables, lazy array preservation) have been fixed. (:issue:`112`, :issue:`113`, :pull:`114`).
* ``xsdba.processing.escore`` now correctly handles all-nan slices. (:issue:`109`, :pull:`108`).
* `xsdba` now uses directly `operator` instead of using `xarray`'s derived `get_op` function. A refactoring in `xarray` had changed the position of `get_op` which caused a bug. (:pull:`120`).
* For more than 1000 quantiles, `fastnanquantile` is not used anymore, as it would throw an error. (:issue:`119`, :pull:`123`).
* `Grouper` now throws an error if `group='time'` is used  with `window>1`. (:issue:`104`, :pull:`122`).
* Slightly reduce "maximum" in `jitter` to fix dtype conversion issue. (:issue:`124`, :pull:`125`).

Internal changes
^^^^^^^^^^^^^^^^
* `tox` has been configured to test Python3.10 builds against `numpy >=1.24.0,<2.0` in the GitHub Workflow pipeline. Passing the `numpy` keyword to `tox` (``$ tox -e py3.10-numpy``) will adjust the build. (:pull:`105`).
* Authorship and Zenodo metadata have been updated. Order of contributions is now developers followed by contributors in alphabetical order. (:pull:`116`).
* `MBCn.adjust` now re-performs the check on `ref` and `hist` to ensure they have compatible time arrays (the check is done a second time in `adjust` since `ref` and `hist` are given again). (:pull:`118`).
* Updated `docs` dependencies to use `sphinx>=8.2.2`. (:pull:`133`).

.. _changes_0.3.2:

`v0.3.2 <https://github.com/Ouranosinc/xsdba/tree/0.3.2>`_ (2025-03-06)
-----------------------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`).

Fixes
^^^^^
* Packaging and security adjustments. (:pull:`106`):
    * Added `deptry`, `codespell`, `vulture`, and `yamllint` to the dev dependencies.
    * Added a few transitive dependencies (`packaging`, `pandas`) to the core dependencies.
    * Added `fastnanquantile` to the `dev` dependencies (to be placed in an `extras` recipe for `xsdba` v0.4.0+).
    * Configured `deptry` to handle optional imports.
    * A new Makefile command `lint/security` has been added (called when running `$ make lint`).
    * Updated `tox.ini` with new linting dependencies.

.. _changes_0.3.1:

`v0.3.1 <https://github.com/Ouranosinc/xsdba/tree/0.3.1>`_ (2025-03-04)
-----------------------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`).

Changes
^^^^^^^
* Added `POT` to the development dependencies. (:pull:`96`).

Fixes
^^^^^
* Adjusted the documentation dependencies and the `sphinx` configuration to fix the ReadTheDocs build. (:pull:`96`).

.. _changes_0.3.0:

`v0.3.0 <https://github.com/Ouranosinc/xsdba/tree/0.3.0>`_ (2025-03-04)
-----------------------------------------------------------------------

Contributors: Pascal Bourgault (:user:`aulemahal`), Éric Dupuis (:user:`coxipi`), Trevor James Smith (:user:`Zeitsperre`).

Announcements
^^^^^^^^^^^^^
* `xsdba` is now available as a package on the Anaconda `conda-forge` channel. (:pull:`82`).

Changes
^^^^^^^
* Remove the units registry declaration and instead use whatever is set as pint's application registry.
  Code still assumes it is a registry based upon the one in cf-xarray (which exports the `cf` formatter). (:issue:`44`, :pull:`57`).
* Updated the cookiecutter template to use the latest version of `cookiecutter-pypackage`. (:pull:`71`):
    * Python and GitHub Actions versions have been updated.
    * Now using advanced CodeQL configuration.
    * New pre-commit hooks for `vulture` (find dead code), `codespell` (grammatical errors), `zizmor` (workflow security), and `gitleaks` (token commit prevention).
    * Corrected some minor spelling and security issues.
* Added `upstream` testing to the CI pipeline for both daily and push events. (:pull:`61`).
* Import last changes in xclim before the embargo (:pull:`80`).
* `xsdba` has begun the process of adoption of the OpenSSF Best Practices checklist. (:pull:`82`).
* `xclim` migration guide added. (:issue:`62`, :pull:`86`).
* Add a missing `dOTC` example to documentation. (:pull:`86`).
* Add a new grouping method specific for `MBCn` which called by passing `group=Grouper("5D", window=n)` where `n` is an odd positive integer. (:pull:`79`).

Fixes
^^^^^
* Gave credits to the package to all previous contributors of ``xclim.sdba``. (:issue:`58`, :pull:`59`).
* Pin `sphinx-codeautolink` to fix ReadTheDocs and correct some docs errors. (:pull:`40`).
* Removed reliance on the `netcdf4` package for testing purposes. The `h5netcdf` engine is now used for file IO operations. (:pull:`71`).
* Changes to reflect the change of library name `xsdba`. (:pull:`72`).
* Revert changes to allow using `group="time.dayofyear"` and `interp="linear"` in adjustment methods. (:pull:`86`).

.. _changes_0.2.0:

`v0.2.0 <https://github.com/Ouranosinc/xsdba/tree/0.2.0>`_ (2025-01-09)
-----------------------------------------------------------------------

Contributors: Éric Dupuis (:user:`coxipi`), Trevor James Smith (:user:`Zeitsperre`).

Changes
^^^^^^^
* Split `sdba` from `xclim` into its own standalone package. Where needed, some common functionalities were duplicated: (:pull:`8`)
    * ``xsdba.units`` is an adaptation of the ``xclim.core.units`` modules.
    * Many functions and definitions found in ``xclim.core.calendar`` have been adapted to ``xsdba.base``.
* Dependencies have been updated to reflect the new package structure. (:pull:`45`).
* Updated documentation configuration: (:pull:`46`)
    * Significant improvements to the documentation content and layout.
    * Now using the `furo` theme for `sphinx`.
    * Notebooks are now linted and formatted with `nbstripout` and `nbqa-black`.
    * CSS configurations have been added for better rendering of the documentation and logos.
* Added the `vulture` linter (for identifying dead code) to the pre-commit configuration. (:pull:`46`).

.. _changes_0.1.0:

`v0.1.0 <https://github.com/Ouranosinc/xsdba/tree/0.1.0>`_
----------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`)

Changes
^^^^^^^
* First release on PyPI.
