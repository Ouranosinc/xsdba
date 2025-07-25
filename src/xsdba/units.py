"""
# noqa: SS01
Units Handling Submodule
========================
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import cast

# this dependency is "necessary" for convert_units_to and all unit printing (which use the CF formatter)
# if we only do checks, we could get rid of it
import cf_xarray.units  # noqa: F401
import numpy as np
import pint
import xarray as xr

from xsdba.base import parse_offset
from xsdba.typing import Quantified

__all__ = [
    "convert_units_to",
    "harmonize_units",
    "infer_sampling_units",
    "pint2cfattrs",
    "pint_multiply",
    "str2pint",
    "units",
    "units2pint",
    "units2str",
]

units = pint.get_application_registry()
# CF-xarray forces numpy arrays even for scalar values, not sure why.
# We don't want that in xsdba, the magnitude of a scalar is a scalar (float).
units.force_ndarray_like = False
FREQ_UNITS = {
    "D": "d",
    "W": "week",
}
"""
Resampling frequency units for :py:func:`xsdba.units.infer_sampling_units`.

Mapping from offset base to CF-compliant unit. Only constant-length frequencies are included.
"""


# XC
def infer_sampling_units(
    da: xr.DataArray,
    deffreq: str | None = "D",
    dim: str = "time",
) -> tuple[int, str]:
    """
    Infer a multiplier and the units corresponding to one sampling period.

    Parameters
    ----------
    da : xr.DataArray
        A DataArray from which to take coordinate `dim`.
    deffreq : str, optional
        If no frequency is inferred from `da[dim]`, take this one.
    dim : str
        Dimension from which to infer the frequency.

    Returns
    -------
    int
        The magnitude (number of base periods per period).
    str
        Units as a string, understandable by pint.

    Raises
    ------
    ValueError
        If the frequency has no exact corresponding units.
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


def _parse_str(value: str) -> tuple[str, str]:
    """
    Parse a str as a number and a unit.

    Parameters
    ----------
    value : str
        Input string representing a unit (may contain a magnitude or not).

    Returns
    -------
    tuple[str, str]
        Magntitude and unit strings. If no magntiude is found, "1" is used by default.
    """
    mstr, *ustr = value.split(" ", maxsplit=1)
    try:
        mstr = str(float(mstr))
    except ValueError:
        mstr = "1"
        ustr = [value]
    ustr = "dimensionless" if len(ustr) == 0 else ustr[0]
    return mstr, ustr


# XC
def units2pint(
    value: xr.DataArray | units.Unit | units.Quantity | dict | str,
) -> pint.Unit:
    """
    Return the pint Unit for the DataArray units.

    Parameters
    ----------
    value : xr.DataArray or pint.Unit or pint.Quantity or dict or str
        Input data array or string representing a unit (may contain a magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.

    Notes
    -----
    To avoid ambiguity related to differences in temperature vs absolute temperatures, set the `units_metadata`
    attribute to `"temperature: difference"` or `"temperature: on_scale"` on the DataArray.
    """
    # Value is already a pint unit or a pint quantity
    if isinstance(value, units.Unit):
        return value

    if isinstance(value, units.Quantity):
        # This is a pint.PlainUnit, which is not the same as a pint.Unit
        return cast(pint.Unit, value.units)

    # We only need the attributes
    if isinstance(value, xr.DataArray):
        value = value.attrs

    if isinstance(value, str):
        _, unit = _parse_str(value)
        metadata = None
    elif isinstance(value, dict):
        unit = value["units"]
        metadata = value.get("units_metadata", None)
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

    pu = units.parse_units(unit)
    if metadata == "temperature: difference":
        return (1 * pu - 1 * pu).units
    return pu


def units2str(value: xr.DataArray | str | units.Quantity | units.Unit) -> str:
    """
    Return a str unit from various inputs.

    Parameters
    ----------
    value : xr.DataArray or str or pint.Quantity or pint.Unit
        Input data array or string representing a unit (with no magnitude).

    Returns
    -------
    pint.Unit
        Units of the data array.
    """
    # Ensure we use CF's formatter. (default with xclim, but not with only cf-xarray)
    return f"{units2pint(value):cf}"


# XC
def str2pint(val: str) -> pint.Quantity:
    """
    Convert a string to a pint.Quantity, splitting the magnitude and the units.

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
    mstr, ustr = _parse_str(val)
    return units.Quantity(float(mstr), units=units2pint(ustr))


# XC
def pint_multiply(
    da: xr.DataArray, q: pint.Quantity | str, out_units: str | None = None
) -> xr.DataArray:
    """
    Multiply xarray.DataArray by pint.Quantity.

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
    out: xr.DataArray = da * float(f.magnitude)
    out = out.assign_attrs(units=f"{f.units:cf}")
    return out


DELTA_ABSOLUTE_TEMP = {
    units.delta_degC: units.kelvin,
    units.delta_degF: units.rankine,
}


# XC
def pint2cfattrs(value: units.Quantity | units.Unit, is_difference=None) -> dict:
    """
    Return CF-compliant units attributes from a `pint` unit.

    Parameters
    ----------
    value : pint.Unit
        Input unit.
    is_difference : bool
        Whether the value represent a difference in temperature, which is ambiguous in the case of absolute
        temperature scales like Kelvin or Rankine. It will automatically be set to True if units are "delta_*"
        units.

    Returns
    -------
    dict
        Units following CF-Convention, using symbols.
    """
    value = value if isinstance(value, pint.Unit | units.Unit) else value.units
    s = f"{value:cf}"
    if "delta_" in s:
        is_difference = True
        s = s.replace("delta_", "")

    attrs = {"units": s}
    if "[temperature]" in value.dimensionality:
        if is_difference:
            attrs["units_metadata"] = "temperature: difference"
        elif is_difference is False:
            attrs["units_metadata"] = "temperature: on_scale"
        else:
            attrs["units_metadata"] = "temperature: unknown"

    return attrs


# Private function so it can be patched
def _convert_units_to(  # noqa: C901
    source: Quantified,
    target: Quantified | units.Unit,
) -> xr.DataArray | float:
    target_unit = units2str(target)
    source_unit = units2str(source)
    if target_unit == source_unit:
        return source if not isinstance(source, str) else float(str2pint(source).m)
    else:  # Convert units
        if isinstance(source, xr.DataArray):
            out = source.copy(data=units.convert(source.data, source_unit, target_unit))
            out = out.assign_attrs(units=target_unit)
        else:  # scalar
            # explicit float cast because cf-xarray registry outputting 0-dim arrays
            out = str2pint(source).to(target_unit).m
        return out


def convert_units_to(  # noqa: C901
    source: Quantified,
    target: Quantified | units.Unit,
) -> xr.DataArray | float:
    """
    Convert a mathematical expression into a value with the same units as a DataArray.

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
    return _convert_units_to(source, target)


def extract_units(arg):
    """
    Extract units from a string, DataArray, or scalar.

    Wrapper that can also yield `None`.
    """
    if isinstance(arg, xr.DataArray):
        # arg becomes str | None
        arg = arg.attrs.get("units", None)
    # "2" is assumed to be "2 dimensionless", like a DataArray with units ""
    if isinstance(arg, pint.Unit | units.Unit | str):
        arg = units2str(arg)
    # 2 is assumed to be 2, no dimension (None), like a DataArray without units attribute
    elif np.isscalar(arg):
        arg = None
    if isinstance(arg, str | None):
        return arg
    raise TypeError(
        f"Argument must be a str | DataArray | pint.Unit | units.Unit | scalar. Got {type(arg)}"
    )


def _add_default_kws(params_dict, params_to_check, func):
    """Combine args and kwargs into a dict."""
    args_dict = {}
    signature = inspect.signature(func)
    for ik, (k, v) in enumerate(signature.parameters.items()):
        if k not in params_dict and k in params_to_check:
            if v.default != inspect._empty:
                params_dict[k] = v.default
    return params_dict


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
            # # Passing datasets or thresh as float (i.e. assign no units) is accepted
            has_units = {
                extract_units(p) is not None
                for p in params_dict.values()
                if p is not None
            }
            if len(has_units) > 1:
                raise ValueError(
                    "All arguments passed to `harmonize_units` must have units, or no units. Mixed cases "
                    "are not allowed. `None` values are ignored."
                )
            if has_units == {True}:
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


def wavelength_to_normalized_wavenumber(
    lam: xr.DataArray | str,
    delta: str | None = None,
) -> xr.DataArray | float:
    """
    Convert a wavelength `lam` to a normalized wavenumber.

    Parameters
    ----------
    lam : xr.DataArray or float
        Wavelength.
    delta: str, Optional
        Nominal resolution of the grid.

    Returns
    -------
    xr.DataArray or float
        Normalized wavenumber.
    """
    if isinstance(lam, str):
        lam, u = _parse_str(lam)
        lam = float(lam)
    else:
        u = lam.units
    delta = convert_units_to(delta, u)
    alpha = 2 * delta / lam
    if isinstance(lam, xr.DataArray):
        alpha.attrs["units"] = ""
    return alpha


def normalized_wavenumber_to_wavelength(
    alpha: xr.DataArray | float, delta: str | None = None, out_units: str | None = None
) -> xr.DataArray | str:
    """
    Convert a normalized wavenumber `alpha` to a wavelength.

    Parameters
    ----------
    alpha : xr.DataArray or float
        Normalized wavelength number.
    delta: str, Optional
        Nominal resolution of the grid.

    Returns
    -------
    xr.DataArray or float
        Wavelength.
    """
    delta, u = (
        _parse_str(delta) if out_units is None else convert_units_to(delta, out_units)
    ), out_units
    delta = np.abs(delta)
    lam = 2 * delta / alpha
    if isinstance(alpha, xr.DataArray):
        lam = lam.assign_attrs({"units": u})
    else:
        lam = f"{lam} {u}"
    return lam
