=========
Changelog
=========

`Unreleased <https://github.com/Ouranosinc/xsdba>`_ (latest)
------------------------------------------------------------

Contributors: Pascal Bourgault (:user:`aulemahal`).

Changes
^^^^^^^
* Remove the units registry declaration and instead use whatever is set as pint's application registry.
  Code still assumes it is a registry based upon the one in cf-xarray (which exports the `cf` formatter). (:issue:`44`, :pull:`57`).

Fixes
^^^^^
* No change.

.. _changes_0.2.0:

`v0.2.0 <https://github.com/Ouranosinc/xsdba/tree/0.2.0>`_ (2025-01-09)
-----------------------------------------------------------------------

Contributors: Éric Dupuis (:user:`coxipi`), Trevor James Smith (:user:`Zeitsperre`).

Changes
^^^^^^^
* Split `sdba` from `xclim` into its own standalone package. Where needed, some common functionalities were duplicated: (:pull:`8`)
    * ``xsdba.units`` is an adaptation of the ``xclim.core.units`` modules.
    * Many functions and definitions found in ``xclim.core.calendar`` have been adapted to ``xsdba.base``.
* Dependencies have been updated to reflect the new package structure. (:pull:`45`)
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
