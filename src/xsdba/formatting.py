"""# noqa: SS01
Formatting Utilities
===================================
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
import string
import warnings
from ast import literal_eval
from collections.abc import Callable, Sequence
from fnmatch import fnmatch
from inspect import _empty, signature
from typing import Any

import xarray as xr
from boltons.funcutils import wraps

from .typing import KIND_ANNOTATION, InputKind


class AttrFormatter(string.Formatter):
    """A formatter for frequently used attribute values.

    See the doc of format_field() for more details.
    """

    def __init__(
        self,
        mapping: dict[str, Sequence[str]],
        modifiers: Sequence[str],
    ) -> None:
        """Initialize the formatter.

        Parameters
        ----------
        mapping : dict[str, Sequence[str]]
            A mapping from values to their possible variations.
        modifiers : Sequence[str]
            The list of modifiers, must be the as long as the longest value of `mapping`.
            Cannot include reserved modifier 'r'.
        """
        super().__init__()
        if "r" in modifiers:
            raise ValueError("Modifier 'r' is reserved for default raw formatting.")
        self.modifiers = modifiers
        self.mapping = mapping

    def format(self, format_string: str, /, *args: Any, **kwargs: Any) -> str:
        r"""Format a string.

        Parameters
        ----------
        format_string: str
        \*args: Any
        \*\*kwargs: Any

        Returns
        -------
        str
        """
        # ADAPT: THIS IS VERY CLIMATE, WILL BE REMOVED
        #     for k, v in DEFAULT_FORMAT_PARAMS.items():
        #         if k not in kwargs:
        #             kwargs.update({k: v})
        return super().format(format_string, *args, **kwargs)

    def format_field(self, value, format_spec):
        """Format a value given a formatting spec.

        If `format_spec` is in this Formatter's modifiers, the corresponding variation
        of value is given. If `format_spec` is 'r' (raw), the value is returned unmodified.
        If `format_spec` is not specified but `value` is in the mapping, the first variation is returned.

        Examples
        --------
        Let's say the string "The dog is {adj1}, the goose is {adj2}" is to be translated
        to French and that we know that possible values of `adj` are `nice` and `evil`.
        In French, the genre of the noun changes the adjective (cat = chat is masculine,
        and goose = oie is feminine) so we initialize the formatter as:

        >>> fmt = AttrFormatter(
        ...     {
        ...         "nice": ["beau", "belle"],
        ...         "evil": ["méchant", "méchante"],
        ...         "smart": ["intelligent", "intelligente"],
        ...     },
        ...     ["m", "f"],
        ... )
        >>> fmt.format(
        ...     "Le chien est {adj1:m}, l'oie est {adj2:f}, le gecko est {adj3:r}",
        ...     adj1="nice",
        ...     adj2="evil",
        ...     adj3="smart",
        ... )
        "Le chien est beau, l'oie est méchante, le gecko est smart"

        The base values may be given using unix shell-like patterns:

        >>> fmt = AttrFormatter(
        ...     {"YS-*": ["annuel", "annuelle"], "MS": ["mensuel", "mensuelle"]},
        ...     ["m", "f"],
        ... )
        >>> fmt.format(
        ...     "La moyenne {freq:f} est faite sur un échantillon {src_timestep:m}",
        ...     freq="YS-JUL",
        ...     src_timestep="MS",
        ... )
        'La moyenne annuelle est faite sur un échantillon mensuel'
        """
        baseval = self._match_value(value)
        if baseval is None:  # Not something we know how to translate
            if format_spec in self.modifiers + [
                "r"
            ]:  # Woops, however a known format spec was asked
                warnings.warn(
                    f"Requested formatting `{format_spec}` for unknown string `{value}`."
                )
                format_spec = ""
            return super().format_field(value, format_spec)
        # Thus, known value

        if not format_spec:  # (None or '') No modifiers, return first
            return self.mapping[baseval][0]

        if format_spec == "r":  # Raw modifier
            return super().format_field(value, "")

        if format_spec in self.modifiers:  # Known modifier
            if len(self.mapping[baseval]) == 1:  # But unmodifiable entry
                return self.mapping[baseval][0]
            # Known modifier, modifiable entry
            return self.mapping[baseval][self.modifiers.index(format_spec)]
        # Known value but unknown modifier, must be a built-in one, only works for the default val...
        return super().format_field(self.mapping[baseval][0], format_spec)

    def _match_value(self, value):
        if isinstance(value, str):
            for mapval in self.mapping.keys():
                if fnmatch(value, mapval):
                    return mapval
        return None


# Tag mappings between keyword arguments and long-form text.
default_formatter = AttrFormatter(
    {
        # Arguments to "freq"
        "D": ["daily", "days"],
        "YS": ["annual", "years"],
        "YS-*": ["annual", "years"],
        "MS": ["monthly", "months"],
        "QS-*": ["seasonal", "seasons"],
        # Arguments to "indexer"
        "DJF": ["winter"],
        "MAM": ["spring"],
        "JJA": ["summer"],
        "SON": ["fall"],
        "norm": ["Normal"],
        "m1": ["january"],
        "m2": ["february"],
        "m3": ["march"],
        "m4": ["april"],
        "m5": ["may"],
        "m6": ["june"],
        "m7": ["july"],
        "m8": ["august"],
        "m9": ["september"],
        "m10": ["october"],
        "m11": ["november"],
        "m12": ["december"],
        # Arguments to "op / reducer / stat" (for example for generic.stats)
        "integral": ["integrated", "integral"],
        "count": ["count"],
        "doymin": ["day of minimum"],
        "doymax": ["day of maximum"],
        "mean": ["average"],
        "max": ["maximal", "maximum"],
        "min": ["minimal", "minimum"],
        "sum": ["total", "sum"],
        "std": ["standard deviation"],
        "var": ["variance"],
        "absamp": ["absolute amplitude"],
        "relamp": ["relative amplitude"],
        # For when we are formatting indicator classes with empty options
        "<class 'inspect._empty'>": ["<empty>"],
    },
    ["adj", "noun"],
)


# XC
def prefix_attrs(source: dict, keys: Sequence, prefix: str) -> dict:
    """Rename some keys of a dictionary by adding a prefix.

    Parameters
    ----------
    source : dict
        Source dictionary, for example data attributes.
    keys : sequence
        Names of keys to prefix.
    prefix : str
        Prefix to prepend to keys.

    Returns
    -------
    dict
        Dictionary of attributes with some keys prefixed.
    """
    out = {}
    for key, val in source.items():
        if key in keys:
            out[f"{prefix}{key}"] = val
        else:
            out[key] = val
    return out


# XC
def unprefix_attrs(source: dict, keys: Sequence, prefix: str) -> dict:
    """Remove prefix from keys in a dictionary.

    Parameters
    ----------
    source : dict
        Source dictionary, for example data attributes.
    keys : sequence
        Names of original keys for which prefix should be removed.
    prefix : str
        Prefix to remove from keys.

    Returns
    -------
    dict
        Dictionary of attributes whose keys were prefixed, with prefix removed.
    """
    out = {}
    n = len(prefix)
    for key, val in source.items():
        k = key[n:]
        if (k in keys) and key.startswith(prefix):
            out[k] = val
        elif key not in out:
            out[key] = val
    return out


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
