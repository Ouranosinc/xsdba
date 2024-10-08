"""# noqa: SS01
Units Handling Submodule
========================
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from functools import wraps
from typing import Any, cast

import pint

# this dependency is "necessary" for convert_units_to
# if we only do checks, we could get rid of it

# XC : units

try:
    # allows to use cf units
    import cf_xarray.units
except ImportError:  # noqa: S110
    # cf-xarray is not installed, this will not be used
    pass
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .base import get_calendar, parse_offset
from .typing import Quantified
from .utils import copy_all_attrs

__all__ = [
    "compare_units",
    "convert_units_to",
    "ensure_absolute_temperature",
    "ensure_cf_units",
    "ensure_delta",
    "harmonize_units",
    "infer_context",
    "infer_sampling_units",
    "pint2cfunits",
    "pint_multiply",
    "str2pint",
    "to_agg_units",
    "units",
    "units2pint",
]

units = pint.get_application_registry()
# Another alias not included by cf_xarray
units.define("@alias percent = pct")

FREQ_UNITS = {
    "D": "d",
    "W": "week",
}
"""
Resampling frequency units for :py:func:`xsdba.units.infer_sampling_units`.

Mapping from offset base to CF-compliant unit. Only constant-length frequencies are included.
"""


def infer_sampling_units(
    da: xr.DataArray,
    deffreq: str | None = "D",
    dim: str = "time",
) -> tuple[int, str]:
    """Infer a multiplier and the units corresponding to one sampling period.

    Parameters
    ----------
    da : xr.DataArray
        A DataArray from which to take coordinate `dim`.
    deffreq : str, optional
        If no frequency is inferred from `da[dim]`, take this one.
    dim : str
        Dimension from which to infer the frequency.

    Raises
    ------
    ValueError
        If the frequency has no exact corresponding units.

    Returns
    -------
    int
        The magnitude (number of base periods per period).
    str
        Units as a string, understandable by pint.
    """
    dimmed = getattr(da, dim)
    freq = xr.infer_freq(dimmed)
    if freq is None:
        freq = deffreq

    multi, base, _, _ = parse_offset(freq)
    try:
        out = multi, FREQ_UNITS.get(base, base)
    except KeyError as err:
        raise ValueError(
            f"Sampling frequency {freq} has no corresponding units."
        ) from err
    if out == (7, "d"):
        # Special case for weekly frequency. xarray's CFTimeOffsets do not have "W".
        return 1, "week"
    return out


# XC
def units2pint(value: xr.DataArray | str | units.Quantity | units.Unit) -> pint.Unit:
    """Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : xr.DataArray or str or pint.Quantity or pint.Unit
        Input data array or string representing a unit (with no magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.
    """
    if isinstance(value, str):
        unit = value
    elif isinstance(value, xr.DataArray):
        unit = value.attrs["units"]
    elif isinstance(value, units.Quantity):
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value.units)
    elif isinstance(value, units.Quantity):
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value.units)
    elif isinstance(value, units.Unit):
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value)
    else:
        raise NotImplementedError(f"Value of type `{type(value)}` not supported.")

    # Catch user errors undetected by Pint
    degree_ex = ["deg", "degree", "degrees"]
    unit_ex = [
        "C",
        "K",
        "F",
        "Celsius",
        "Kelvin",
        "Fahrenheit",
        "celsius",
        "kelvin",
        "fahrenheit",
    ]
    possibilities = [f"{d} {u}" for d in degree_ex for u in unit_ex]
    if unit.strip() in possibilities:
        raise ValidationError(
            "Remove white space from temperature units, e.g. use `degC`."
        )

    return units.parse_units(unit)


def units2str(value: xr.DataArray | str | units.Quantity | units.Unit) -> str:
    """Return a str unit from various inputs.

    Parameters
    ----------
    value : xr.DataArray or str or pint.Quantity or pint.Unit
        Input data array or string representing a unit (with no magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.
    """
    return value if isinstance(value, str) else pint2str(units2pint(value))


# XC
def str2pint(val: str) -> pint.Quantity:
    """Convert a string to a pint.Quantity, splitting the magnitude and the units.

    Parameters
    ----------
    val : str
        A quantity in the form "[{magnitude} ]{units}", where magnitude can be cast to a float and
        units is understood by `units2pint`.

    Returns
    -------
    pint.Quantity
        Magnitude is 1 if no magnitude was present in the string.
    """
    mstr, *ustr = val.split(" ", maxsplit=1)
    try:
        if ustr:
            return units.Quantity(float(mstr), units=units2pint(ustr[0]))
        return units.Quantity(float(mstr))
    except ValueError:
        return units.Quantity(1, units2pint(val))


def pint2str(value: units.Quantity | units.Unit) -> str:
    """A unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
        Input unit.

    Returns
    -------
    str
        Units.

    Notes
    -----
    If cf-xarray is installed, the units will be converted to cf units.
    """
    if isinstance(value, (pint.Quantity | units.Quantity)):
        value = value.units

    # Issue originally introduced in https://github.com/hgrecco/pint/issues/1486
    # Should be resolved in pint v0.24. See: https://github.com/hgrecco/pint/issues/1913
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return f"{value:cf}".replace("dimensionless", "")


def pint_multiply(
    da: xr.DataArray, q: pint.Quantity | str, out_units: str | None = None
) -> xr.DataArray:
    """Multiply xarray.DataArray by pint.Quantity.

    Parameters
    ----------
    da : xr.DataArray
        Input array.
    q : pint.Quantity or str
        Multiplicative factor.
    out_units : str, optional
        Units the output array should be converted into.

    Returns
    -------
    xr.DataArray
    """
    q = q if isinstance(q, pint.Quantity) else str2pint(q)
    a = 1 * units2pint(da)
    f = a * q.to_base_units()
    if out_units:
        f = f.to(out_units)
    else:
        f = f.to_reduced_units()
    out: xr.DataArray = da * f.magnitude
    out = out.assign_attrs(units=pint2str(f.units))
    return out


DELTA_ABSOLUTE_TEMP = {
    units.delta_degC: units.kelvin,
    units.delta_degF: units.rankine,
}


def ensure_absolute_temperature(units: str):
    """Convert temperature units to their absolute counterpart, assuming they represented a difference (delta).

    Celsius becomes Kelvin, Fahrenheit becomes Rankine. Does nothing for other units.
    """
    a = str2pint(units)
    # ensure a delta pint unit
    a = a - 0 * a
    if a.units in DELTA_ABSOLUTE_TEMP:
        return pint2str(DELTA_ABSOLUTE_TEMP[a.units])
    return units


def ensure_cf_units(ustr: str) -> str:
    """Ensure the passed unit string is CF-compliant.

    The string will be parsed to pint then recast to a string by xclim's `pint2cfunits`.
    """
    return pint2cfunits(units2pint(ustr))


def pint2cfunits(value: units.Quantity | units.Unit) -> str:
    """Return a CF-compliant unit string from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
        Input unit.

    Returns
    -------
    str
        Units following CF-Convention, using symbols.
    """
    if isinstance(value, pint.Quantity | units.Quantity):
        value = value.units

    # Force "1" if the formatted string is "" (pint < 0.24)
    return f"{value:~cf}" or "1"


def ensure_delta(unit: str) -> str:
    """Return delta units for temperature.

    For dimensions where delta exist in pint (Temperature), it replaces the temperature unit by delta_degC or
    delta_degF based on the input unit. For other dimensionality, it just gives back the input units.

    Parameters
    ----------
    unit : str
        unit to transform in delta (or not).
    """
    u = units2pint(unit)
    d = 1 * u
    #
    delta_unit = pint2str(d - d)
    # replace kelvin/rankine by delta_degC/F
    if "kelvin" in u._units:
        delta_unit = pint2str(u / units2pint("K") * units2pint("delta_degC"))
    if "degree_Rankine" in u._units:
        delta_unit = pint2str(u / units2pint("°R") * units2pint("delta_degF"))
    return delta_unit


def extract_units(arg):
    """Extract units from a string, DataArray, or scalar."""
    if not (
        isinstance(arg, (str | xr.DataArray | pint.Unit | units.Unit))
        or np.isscalar(arg)
    ):
        raise TypeError(
            f"Argument must be a str, DataArray, or scalar. Got {type(arg)}"
        )
    elif isinstance(arg, xr.DataArray):
        ustr = None if "units" not in arg.attrs else arg.attrs["units"]
    elif isinstance(arg, (pint.Unit | units.Unit)):
        ustr = pint2str(arg)  # XC: from pint2str
    elif isinstance(arg, str):
        ustr = pint2str(str2pint(arg).units)
    else:  # (scalar case)
        ustr = None
    return ustr if ustr is None else pint.Quantity(1, ustr).units


# TODO: Is this really needed?
def compare_units(args_to_check):
    """Decorator to check that all arguments have the same units (or no units)."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            # dictionnary {arg_name:arg} for all args of func
            arg_dict = dict(zip(inspect.getfullargspec(func).args, args))
            # Obtain units (or None if no units) of all args
            units = []
            for arg_name in args_to_check:
                if isinstance(arg_name, str):
                    value = arg_dict[arg_name]
                    key = arg_name
                if isinstance(
                    arg_name, dict
                ):  # support for Dataset, or a dict of thresholds
                    key, val = list(arg_name.keys())[0], list(arg_name.values())[0]
                    value = arg_dict[key][val]
                if value is None:  # optional argument, should be ignored
                    args_to_check.remove(arg_name)
                    continue
                if key not in arg_dict:
                    raise ValueError(
                        f"Argument '{arg_name}' not found in function arguments."
                    )
                units.append(extract_units(value))
            # Check that units are consistent
            if len(set(units)) > 1:
                raise ValueError(
                    f"{args_to_check} must have the same units (or no units). Got {units}"
                )
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


# XC simplified
def convert_units_to(  # noqa: C901
    source: Quantified,
    target: Quantified | units.Unit,
) -> xr.DataArray | float:
    """Convert a mathematical expression into a value with the same units as a DataArray.

    If the dimensionalities of source and target units differ, automatic CF conversions
    will be applied when possible.

    Parameters
    ----------
    source : str or xr.DataArray or units.Quantity
        The value to be converted, e.g. '4C' or '1 mm/d'.
    target : str or xr.DataArray or units.Quantity or units.Unit
        Target array of values to which units must conform.

    Returns
    -------
    xr.DataArray or float
        The source value converted to target's units.
        The outputted type is always similar to `source` initial type.
        Attributes are preserved unless an automatic CF conversion is performed,
        in which case only the new `standard_name` appears in the result.
    """
    # Target units
    target_unit = extract_units(target)
    source_unit = extract_units(source)
    if target_unit == source_unit:
        return source if isinstance(source, str) is False else str2pint(source).m
    else:  # Convert units
        if isinstance(source, xr.DataArray):
            out = source.copy(data=units.convert(source.data, source_unit, target_unit))
            out = out.assign_attrs(units=target_unit)
        else:
            out = str2pint(source).to(target_unit).m
        return out


def _add_default_kws(params_dict, params_to_check, func):
    """Combine args and kwargs into a dict."""
    args_dict = {}
    signature = inspect.signature(func)
    for ik, (k, v) in enumerate(signature.parameters.items()):
        if k not in params_dict and k in params_to_check:
            if v.default != inspect._empty:
                params_dict[k] = v.default
    return params_dict


# TODO: this changes the type of some variables (e.g. thresh : str -> float). This should probably not be allowed
# TODO: support for Datasets and dict like in compare_units?
def harmonize_units(params_to_check):
    """Compare units and perform a conversion if possible, otherwise raise a `ValidationError`."""

    # if no units are present (DataArray without units attribute or float), then no check is performed
    # if units are present, then check is performed
    # in mixed cases, an error is raised
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            params_func = inspect.signature(func).parameters.keys()
            if set(params_to_check).issubset(set(params_func)) is False:
                raise TypeError(
                    f"`harmonize_units' inputs `{params_to_check}` should be a subset of "
                    f"`{func.__name__}`'s arguments: `{params_func}` (arguments that can contain units)"
                )
            arg_names = inspect.getfullargspec(func).args
            args_dict = dict(zip(arg_names, args))
            params_dict = args_dict | {k: v for k, v in kwargs.items()}
            params_dict = {k: v for k, v in params_dict.items() if k in params_to_check}
            params_dict = _add_default_kws(params_dict, params_to_check, func)
            if set(params_dict.keys()) != set(params_to_check):
                raise TypeError(
                    f"{params_to_check} were passed but only {params_dict.keys()} were found "
                    f"in `{func.__name__}`'s arguments"
                )
            first_param = params_dict[params_to_check[0]]
            for param_name in params_dict.keys():
                value = params_dict[param_name]
                if value is None:  # optional argument, should be ignored
                    continue
                params_dict[param_name] = convert_units_to(value, first_param)
            # reassign keyword arguments
            for k in [k for k in params_dict.keys() if k not in args_dict.keys()]:
                kwargs[k] = params_dict[k]
                params_dict.pop(k)
            # reassign remaining arguments (passed as arg)
            args = list(args)
            for iarg in range(len(args)):
                if arg_names[iarg] in params_dict.keys():
                    args[iarg] = params_dict[arg_names[iarg]]
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


def to_agg_units(
    out: xr.DataArray, orig: xr.DataArray, op: str, dim: str = "time"
) -> xr.DataArray:
    """Set and convert units of an array after an aggregation operation along the sampling dimension (time).

    Parameters
    ----------
    out : xr.DataArray
        The output array of the aggregation operation, no units operation done yet.
    orig : xr.DataArray
        The original array before the aggregation operation,
        used to infer the sampling units and get the variable units.
    op : {'min', 'max', 'mean', 'std', 'var', 'doymin', 'doymax',  'count', 'integral', 'sum'}
        The type of aggregation operation performed. "integral" is mathematically equivalent to "sum",
        but the units are multiplied by the timestep of the data (requires an inferrable frequency).
    dim : str
        The time dimension along which the aggregation was performed.

    Returns
    -------
    xr.DataArray

    Examples
    --------
    Take a daily array of temperature and count number of days above a threshold.
    `to_agg_units` will infer the units from the sampling rate along "time", so
    we ensure the final units are correct:

    >>> time = xr.cftime_range("2001-01-01", freq="D", periods=365)
    >>> tas = xr.DataArray(
    ...     np.arange(365),
    ...     dims=("time",),
    ...     coords={"time": time},
    ...     attrs={"units": "degC"},
    ... )
    >>> cond = tas > 100  # Which days are boiling
    >>> Ndays = cond.sum("time")  # Number of boiling days
    >>> Ndays.attrs.get("units")
    None
    >>> Ndays = to_agg_units(Ndays, tas, op="count")
    >>> Ndays.units
    'd'

    Similarly, here we compute the total heating degree-days, but we have weekly data:

    >>> time = xr.cftime_range("2001-01-01", freq="7D", periods=52)
    >>> tas = xr.DataArray(
    ...     np.arange(52) + 10,
    ...     dims=("time",),
    ...     coords={"time": time},
    ... )
    >>> dt = (tas - 16).assign_attrs(units="delta_degC")
    >>> degdays = dt.clip(0).sum("time")  # Integral of temperature above a threshold
    >>> degdays = to_agg_units(degdays, dt, op="integral")
    >>> degdays.units
    'K week'

    Which we can always convert to the more common "K days":

    >>> degdays = convert_units_to(degdays, "K days")
    >>> degdays.units
    'K d'.
    """
    if op in ["amin", "min", "amax", "max", "mean", "sum"]:
        out.attrs["units"] = orig.attrs["units"]

    elif op in ["std"]:
        out.attrs["units"] = ensure_absolute_temperature(orig.attrs["units"])

    elif op in ["var"]:
        out.attrs["units"] = pint2cfunits(
            str2pint(ensure_absolute_temperature(orig.units)) ** 2
        )

    elif op in ["doymin", "doymax"]:
        out.attrs.update(
            units="1", is_dayofyear=np.int32(1), calendar=get_calendar(orig)
        )

    elif op in ["count", "integral"]:
        m, freq_u_raw = infer_sampling_units(orig[dim])
        orig_u = str2pint(ensure_absolute_temperature(orig.units))
        freq_u = str2pint(freq_u_raw)
        out = out * m

        if op == "count":
            out.attrs["units"] = freq_u_raw
        elif op == "integral":
            if "[time]" in orig_u.dimensionality:
                # We need to simplify units after multiplication
                out_units = (orig_u * freq_u).to_reduced_units()
                out = out * out_units.magnitude
                out.attrs["units"] = pint2cfunits(out_units)
            else:
                out.attrs["units"] = pint2cfunits(orig_u * freq_u)
    else:
        raise ValueError(
            f"Unknown aggregation op {op}. "
            "Known ops are [min, max, mean, std, var, doymin, doymax, count, integral, sum]."
        )

    return out
