"""# noqa: SS01
Formatting Utilities
===================================
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable
from inspect import signature

import xarray as xr
from boltons.funcutils import wraps


# XC
def merge_attributes(
    attribute: str,
    *inputs_list: xr.DataArray | xr.Dataset,
    new_line: str = "\n",
    missing_str: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""Merge attributes from several DataArrays or Datasets.

    If more than one input is given, its name (if available) is prepended as: "<input name> : <input attribute>".

    Parameters
    ----------
    attribute : str
        The attribute to merge.
    \*inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their "name" attribute if available.
    new_line : str
        The character to put between each instance of the attributes. Usually, in CF-conventions,
        the history attributes uses '\\n' while cell_methods uses ' '.
    missing_str : str
        A string that is printed if an input doesn't have the attribute. Defaults to None, in which
        case the input is simply skipped.
    \*\*inputs_kws : xr.DataArray or xr.Dataset
        Mapping from names to the datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
        The new attribute made from the combination of the ones from all the inputs.
    """
    inputs = [(getattr(in_ds, "name", None), in_ds) for in_ds in inputs_list]
    inputs += list(inputs_kws.items())

    merged_attr = ""
    for in_name, in_ds in inputs:
        if attribute in in_ds.attrs or missing_str is not None:
            if in_name is not None and len(inputs) > 1:
                merged_attr += f"{in_name}: "
            merged_attr += in_ds.attrs.get(
                attribute, "" if in_name is None else missing_str
            )
            merged_attr += new_line

    if len(new_line) > 0:
        return merged_attr[: -len(new_line)]  # Remove the last added new_line
    return merged_attr


# XC
def update_history(
    hist_str: str,
    *inputs_list: xr.DataArray | xr.Dataset,
    new_name: str | None = None,
    **inputs_kws: xr.DataArray | xr.Dataset,
) -> str:
    r"""Return a history string with the timestamped message and the combination of the history of all inputs.

    The new history entry is formatted as "[<timestamp>] <new_name>: <hist_str> - xsdba version: <xsdba.__version__>."

    Parameters
    ----------
    hist_str : str
        The string describing what has been done on the data.
    \*inputs_list : xr.DataArray or xr.Dataset
        The datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixed by their "name" attribute if available.
    new_name : str, optional
        The name of the newly created variable or dataset to prefix hist_msg.
    \*\*inputs_kws : xr.DataArray or xr.Dataset
        Mapping from names to the datasets or variables that were used to produce the new object.
        Inputs given that way will be prefixes by the passed name.

    Returns
    -------
    str
        The combine history of all inputs starting with `hist_str`.

    See Also
    --------
    merge_attributes
    """
    from xsdba import (  # pylint: disable=cyclic-import,import-outside-toplevel
        __version__,
    )

    merged_history = merge_attributes(
        "history",
        *inputs_list,
        new_line="\n",
        missing_str="",
        **inputs_kws,
    )
    if len(merged_history) > 0 and not merged_history.endswith("\n"):
        merged_history += "\n"
    merged_history += (
        f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {new_name or ''}: "
        f"{hist_str} - xsdba version: {__version__}"
    )
    return merged_history


# XC
def update_xsdba_history(func: Callable):
    """Decorator that auto-generates and fills the history attribute.

    The history is generated from the signature of the function and added to the first output.
    Because of a limitation of the `boltons` wrapper, all arguments passed to the wrapped function
    will be printed as keyword arguments.
    """

    @wraps(func)
    def _call_and_add_history(*args, **kwargs):
        """Call the function and then generate and add the history attr."""
        outs = func(*args, **kwargs)

        if isinstance(outs, tuple):
            out = outs[0]
        else:
            out = outs

        if not isinstance(out, (xr.DataArray | xr.Dataset)):
            raise TypeError(
                f"Decorated `update_xsdba_history` received a non-xarray output from {func.__name__}."
            )

        da_list = [arg for arg in args if isinstance(arg, xr.DataArray)]
        da_dict = {
            name: arg for name, arg in kwargs.items() if isinstance(arg, xr.DataArray)
        }

        # The wrapper hides how the user passed the arguments (positional or keyword)
        # Instead of having it all position, we have it all keyword-like for explicitness.
        bound_args = signature(func).bind(*args, **kwargs)
        attr = update_history(
            gen_call_string(func.__name__, **bound_args.arguments),
            *da_list,
            new_name=out.name if isinstance(out, xr.DataArray) else None,
            **da_dict,
        )
        out.attrs["history"] = attr
        return outs

    return _call_and_add_history


# XC
def gen_call_string(
    funcname: str,
    *args,
    **kwargs,
) -> str:
    r"""Generate a signature string for use in the history attribute.

    DataArrays and Dataset are replaced with their name, while Nones, floats, ints and strings are printed directly.
    All other objects have their type printed between < >.

    Arguments given through positional arguments are printed positionnally and those
    given through keywords are printed prefixed by their name.

    Parameters
    ----------
    funcname : str
        Name of the function.

    Example
    -------
    >>> A = xr.DataArray([1], dims=("x",), name="A")
    >>> gen_call_string("func", A, b=2.0, c="3", d=[10] * 100)
    "func(A, b=2.0, c='3', d=<list>)"
    """
    elements = []
    chain = itertools.chain(zip([None] * len(args), args), kwargs.items())
    for name, val in chain:
        if isinstance(val, xr.DataArray):
            rep = val.name or "<array>"
        elif isinstance(val, (int | float | str | bool)) or val is None:
            rep = repr(val)
        else:
            rep = repr(val)
            if len(rep) > 50:
                rep = f"<{type(val).__name__}>"

        if name is not None:
            rep = f"{name}={rep}"

        elements.append(rep)

    return f"{funcname}({', '.join(elements)})"
