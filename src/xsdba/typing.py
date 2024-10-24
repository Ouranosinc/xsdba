"""# noqa: SS01
Typing Utilities
===================================
"""

from __future__ import annotations

from enum import IntEnum
from typing import NewType, TypeVar

import xarray as xr
from pint import Quantity

# XC:
#: Type annotation for strings representing full dates (YYYY-MM-DD), may include time.
DateStr = NewType("DateStr", str)

#: Type annotation for strings representing dates without a year (MM-DD).
DayOfYearStr = NewType("DayOfYearStr", str)

#: Type annotation for thresholds and other not-exactly-a-variable quantities
Quantified = TypeVar("Quantified", xr.DataArray, str, Quantity)


# XC
class InputKind(IntEnum):
    """Constants for input parameter kinds.

    For use by external parses to determine what kind of data the indicator expects.
    On the creation of an indicator, the appropriate constant is stored in
    :py:attr:`xsdba.indicator.Indicator.parameters`. The integer value is what gets stored in the output
    of :py:meth:`xsdba.indicator.Indicator.json`.

    For developers : for each constant, the docstring specifies the annotation a parameter of an indice function
    should use in order to be picked up by the indicator constructor. Notice that we are using the annotation format
    as described in `PEP 604 <https://peps.python.org/pep-0604/>`_, i.e. with '|' indicating a union and without import
    objects from `typing`.
    """

    VARIABLE = 0
    """A data variable (DataArray or variable name).

       Annotation : ``xr.DataArray``.
    """
    OPTIONAL_VARIABLE = 1
    """An optional data variable (DataArray or variable name).

       Annotation : ``xr.DataArray | None``. The default should be None.
    """
    QUANTIFIED = 2
    """A quantity with units, either as a string (scalar), a pint.Quantity (scalar) or a DataArray (with units set).

       Annotation : ``xsdba.typing.Quantified`` and an entry in the :py:func:`xsdba.units.declare_units`
       decorator. "Quantified" translates to ``str | xr.DataArray | pint.util.Quantity``.
    """
    FREQ_STR = 3
    """A string representing an "offset alias", as defined by pandas.

       See the Pandas documentation on :ref:`timeseries.offset_aliases` for a list of valid aliases.

       Annotation : ``str`` + ``freq`` as the parameter name.
    """
    NUMBER = 4
    """A number.

       Annotation : ``int``, ``float`` and unions thereof, potentially optional.
    """
    STRING = 5
    """A simple string.

       Annotation : ``str`` or ``str | None``. In most cases, this kind of parameter makes sense
       with choices indicated in the docstring's version of the annotation with curly braces.
       # TOOO : what about this notebook? removed reference to extendxclim
    """
    DAY_OF_YEAR = 6
    """A date, but without a year, in the MM-DD format.

       Annotation : :py:obj:`xsdba.typing.DayOfYearStr` (may be optional).
    """
    DATE = 7
    """A date in the YYYY-MM-DD format, may include a time.

       Annotation : :py:obj:`xsdba.typing.DateStr` (may be optional).
    """
    NUMBER_SEQUENCE = 8
    """A sequence of numbers

       Annotation : ``Sequence[int]``, ``Sequence[float]`` and unions thereof, may include single ``int`` and ``float``,
       may be optional.
    """
    BOOL = 9
    """A boolean flag.

       Annotation : ``bool``, may be optional.
    """
    DICT = 10
    """A dictionary.

       Annotation : ``dict`` or ``dict | None``, may be optional.
    """
    KWARGS = 50
    """A mapping from argument name to value.

       Developers : maps the ``**kwargs``. Please use as little as possible.
    """
    DATASET = 70
    """An xarray dataset.

       Developers : as indices only accept DataArrays, this should only be added on the indicator's constructor.
    """
    OTHER_PARAMETER = 99
    """An object that fits None of the previous kinds.

       Developers : This is the fallback kind, it will raise an error in xsdba's unit tests if used.
    """


KIND_ANNOTATION = {
    InputKind.VARIABLE: "str or DataArray",
    InputKind.OPTIONAL_VARIABLE: "str or DataArray, optional",
    InputKind.QUANTIFIED: "quantity (string or DataArray, with units)",
    InputKind.FREQ_STR: "offset alias (string)",
    InputKind.NUMBER: "number",
    InputKind.NUMBER_SEQUENCE: "number or sequence of numbers",
    InputKind.STRING: "str",
    InputKind.DAY_OF_YEAR: "date (string, MM-DD)",
    InputKind.DATE: "date (string, YYYY-MM-DD)",
    InputKind.BOOL: "boolean",
    InputKind.DICT: "dict",
    InputKind.DATASET: "Dataset, optional",
    InputKind.KWARGS: "",
    InputKind.OTHER_PARAMETER: "Any",
}
