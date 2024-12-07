============
Installation
============

..
    We strongly recommend installing xsdba in an Anaconda Python environment.
    Furthermore, due to the complexity of some packages, the default dependency solver can take a long time to resolve the environment.
    If `mamba` is not already your default solver, consider running the following commands in order to speed up the process:

        .. code-block:: console

            conda install -n base conda-libmamba-solver
            conda config --set solver libmamba

If you don't have `pip`_ installed, this `Python installation guide`_ can guide you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Stable release
--------------

To install xsdba, run this command in your terminal:

.. code-block:: console

    python -m pip install xsdba

..
    .. code-block:: console

        conda install xsdba

This is the preferred method to install xsdba, as it will always install the most recent stable release.


From sources
------------

The sources for xsdba can be downloaded from the `Github repo`_.

#. Download the source code from the `Github repo`_ using one of the following methods:

    * Clone the public repository:

        .. code-block:: console

            git clone git@github.com:Ouranosinc/xsdba.git

    * Download the `tarball <https://github.com/Ouranosinc/xsdba/tarball/main>`_:

        .. code-block:: console

            curl -OJL https://github.com/Ouranosinc/xsdba/tarball/main

#. Once you have a copy of the source, you can install it with:

    .. code-block:: console

        python -m pip install .

    ..
        .. code-block:: console

            conda env create -f environment-dev.yml
            conda activate xsdba-dev
            make dev

        If you are on Windows, replace the ``make dev`` command with the following:

        .. code-block:: console

            python -m pip install -e .[dev]

        Even if you do not intend to contribute to `xsdba`, we favor using `environment-dev.yml` over `environment.yml` because it includes additional packages that are used to run all the examples provided in the documentation.
        If for some reason you wish to install the `PyPI` version of `xsdba` into an existing Anaconda environment (*not recommended if requirements are not met*), only run the last command above.

#. When new changes are made to the `Github repo`_, if using a clone, you can update your local copy using the following commands from the root of the repository:

    .. code-block:: console

        git fetch
        git checkout main
        git pull origin main
        python -m pip install .

    ..
        .. code-block:: console

            git fetch
            git checkout main
            git pull origin main
            conda env update -n xsdba-dev -f environment-dev.yml
            conda activate xsdba-dev
            make dev

    These commands should work most of the time, but if big changes are made to the repository, you might need to remove the environment and create it again.

.. _Github repo: https://github.com/Ouranosinc/xsdba


.. _extra-dependencies:

Extra Dependencies
------------------

Experimental SDBA Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`xsdba` also offers support for a handful of experimental adjustment methods to extend, available only if some additional libraries are installed. These libraries are completely optional.

One experimental library is `SBCK`_. `SBCK` is available from PyPI but has one complex dependency: `Eigen3`_.
As `SBCK` is compiled at installation time, a **C++** compiler (`GCC`, `Clang`, `MSVC`, etc.) must also be available.

On Debian/Ubuntu, `Eigen3` can be installed via `apt`:

.. code-block:: shell

    $ sudo apt-get install libeigen3-dev

Eigen3 is also available on conda-forge, so, if already using Anaconda, one can do:

.. code-block:: shell

    $ conda install -c conda-forge eigen

Afterwards, `SBCK` can be installed from PyPI using `pip`:

.. code-block:: shell

    $ python -m pip install pybind11 sbck

.. _SBCK: https://github.com/yrobink/SBCK
.. _Eigen3: https://eigen.tuxfamily.org/index.php
