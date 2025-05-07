=========
Changelog
=========

`Unreleased <https://github.com/Ouranosinc/xsdba>`_ (latest)
------------------------------------------------------------

Contributors: Éric Dupuis (:user:`coxipi`)

Changes
^^^^^^^
* Speed up import by activating `cache=True` for in numba-accelerated functions from ``xsdba.nbutils``. (:pull:`135`).

Fixes
^^^^^
* No change.

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
