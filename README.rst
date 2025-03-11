===========================================================
xsdba: Statistical Downscaling and Bias Adjustment library
===========================================================

+----------------------------+-----------------------------------------------------+
| Versions                   | |pypi| |conda| |versions|                           |
+----------------------------+-----------------------------------------------------+
| Documentation and Support  | |docs|                                              |
+----------------------------+-----------------------------------------------------+
| Open Source                | |license| |ossf-score|                              |
+----------------------------+-----------------------------------------------------+
| Coding Standards           | |black| |isort| |ruff| |ossf-bp| |pre-commit|       |
+----------------------------+-----------------------------------------------------+
| Development Status         | |status| |build| |coveralls|                        |
+----------------------------+-----------------------------------------------------+

Statistical correction and bias adjustment tools for xarray.

* Free software: Apache Software License 2.0
* Documentation: https://xsdba.readthedocs.io.

Features
--------

* The `xsdba` submodule provides a collection of bias-adjustment methods meant to correct for systematic biases found in climate model simulations relative to observations.
  Almost all adjustment algorithms conform to the `train` - `adjust` scheme, meaning that adjustment factors are first estimated on training data sets, then applied in a distinct step to the data to be adjusted.
  Given a reference time series (`ref`), historical simulations (`hist`) and simulations to be adjusted (`sim`), any bias-adjustment method would be applied by first estimating the adjustment factors between the historical simulation and the observation series, and then applying these factors to `sim``, which could be a future simulation:

* Time grouping (months, day of year, season) can be done within bias adjustment methods.

* Properties and measures utilities can be used to assess the quality of adjustments.

Quick Install
-------------
`xsdba` can be installed from PyPI:

.. code-block:: shell

    $ pip install xsdba

Documentation
-------------
The official documentation is at https://xsdba.readthedocs.io/

How to make the most of `xsdba`: `Basic Usage Examples`_ and `In-Depth Examples`_.

`xsdba` was first developed as the submodule `sdba` of `xclim`_. A `migration guide`_  is available for users of `sdba`.

.. _Basic Usage Examples: https://xsdba.readthedocs.io/en/latest/notebooks/example.html
.. _In-Depth Examples: https://xsdba.readthedocs.io/en/latest/notebooks/advanced_example.html
.. _xclim: https://xclim.readthedocs.io/
.. _migration guide: https://xsdba.readthedocs.io/en/stable/xclim_migration_guide.html

Credits
-------

This package was created with Cookiecutter_ and the `Ouranosinc/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter
.. _`Ouranosinc/cookiecutter-pypackage`: https://github.com/Ouranosinc/cookiecutter-pypackage

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: Python Black

.. |build| image:: https://github.com/Ouranosinc/xsdba/actions/workflows/main.yml/badge.svg
        :target: https://github.com/Ouranosinc/xsdba/actions
        :alt: Build Status

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/xsdba.svg
        :target: https://anaconda.org/conda-forge/xsdba
        :alt: Conda-forge Build Version

.. |coveralls| image:: https://coveralls.io/repos/github/Ouranosinc/xsdba/badge.svg
        :target: https://coveralls.io/github/Ouranosinc/xsdba
        :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/xsdba/badge/?version=latest
        :target: https://xsdba.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
        :target: https://pycqa.github.io/isort/
        :alt: Isort

.. |license| image:: https://img.shields.io/github/license/Ouranosinc/xsdba.svg
        :target: https://github.com/Ouranosinc/xsdba/blob/main/LICENSE
        :alt: License

.. |ossf-bp| image:: https://bestpractices.coreinfrastructure.org/projects/10045/badge
        :target: https://bestpractices.coreinfrastructure.org/projects/10045
        :alt: Open Source Security Foundation Best Practices

.. |ossf-score| image:: https://api.securityscorecards.dev/projects/github.com/Ouranosinc/xsdba/badge
        :target: https://securityscorecards.dev/viewer/?uri=github.com/Ouranosinc/xsdba
        :alt: OpenSSF Scorecard

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/Ouranosinc/xsdba/main.svg
        :target: https://results.pre-commit.ci/latest/github/Ouranosinc/xsdba/main
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/xsdba.svg
        :target: https://pypi.python.org/pypi/xsdba
        :alt: PyPI

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://github.com/astral-sh/ruff
        :alt: Ruff

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/xsdba.svg
        :target: https://pypi.python.org/pypi/xsdba
        :alt: Supported Python Versions

.. |logo| image:: https://raw.githubusercontent.com/Ouranosinc/xsdba/main/docs/logos/xsdba-logo-small-light.png
        :target: https://github.com/Ouranosinc/xsdba
        :alt: Xsdba
        :class: xsdba-logo-small no-theme

.. |logo-light| image:: https://raw.githubusercontent.com/Ouranosinc/xsdba/main/docs/logos/xsdba-logo-small-light.png
        :target: https://github.com/Ouranosinc/xsdba
        :alt:
        :class: xclim-logo-small only-light-inline

.. |logo-dark| image:: https://raw.githubusercontent.com/Ouranosinc/xsdba/main/docs/logos/xsdba-logo-small-dark.png
        :target: https://github.com/Ouranosinc/xsdba
        :alt:
        :class: xclim-logo-small only-dark-inline
