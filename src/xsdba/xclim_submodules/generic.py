"""Generic functions adapted from xclim."""

from __future__ import annotations

from collections.abc import Callable

import xarray as xr

__all__ = ["default_freq", "get_op", "select_resample_op"]


# XC
binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


# XC
def default_freq(**indexer) -> str:
    """Return the default frequency."""
    freq = "YS-JAN"
    if indexer:
        group, value = indexer.popitem()
        if group == "season":
            month = 12  # The "season" scheme is based on YS-DEC
        elif group == "month":
            month = np.take(value, 0)
        elif group == "doy_bounds":
            month = cftime.num2date(value[0] - 1, "days since 2004-01-01").month
        elif group == "date_bounds":
            month = int(value[0][:2])
        else:
            raise ValueError(f"Unknown group `{group}`.")
        freq = "YS-" + _MONTH_ABBREVIATIONS[month]
    return freq


# XC
def get_op(op: str, constrain: Sequence[str] | None = None) -> Callable:
    """Get python's comparing function according to its name of representation and validate allowed usage.

    Accepted op string are keys and values of xclim.indices.generic.binary_ops.

    Parameters
    ----------
    op : str
        Operator.
    constrain : sequence of str, optional
        A tuple of allowed operators.
    """
    if op == "gteq":
        warnings.warn(f"`{op}` is being renamed `ge` for compatibility.")
        op = "ge"
    if op == "lteq":
        warnings.warn(f"`{op}` is being renamed `le` for compatibility.")
        op = "le"

    if op in binary_ops:
        binary_op = binary_ops[op]
    elif op in binary_ops.values():
        binary_op = op
    else:
        raise ValueError(f"Operation `{op}` not recognized.")

    constraints = []
    if isinstance(constrain, list | tuple | set):
        constraints.extend([binary_ops[c] for c in constrain])
        constraints.extend(constrain)
    elif isinstance(constrain, str):
        constraints.extend([binary_ops[constrain], constrain])

    if constrain:
        if op not in constraints:
            raise ValueError(f"Operation `{op}` not permitted for indice.")

    return xr.core.ops.get_op(binary_op)


# XC
def select_resample_op(
    da: xr.DataArray,
    op: str | Callable,
    freq: str = "YS",
    out_units: str | None = None,
    **indexer,
) -> xr.DataArray:
    """Apply operation over each period that is part of the index selection.

    Parameters
    ----------
    da : xr.DataArray
        Input data.
    op : str {'min', 'max', 'mean', 'std', 'var', 'count', 'sum', 'integral', 'argmax', 'argmin'} or func
        Reduce operation. Can either be a DataArray method or a function that can be applied to a DataArray.
    freq : str
        Resampling frequency defining the periods as defined in :ref:`timeseries.resampling`.
    out_units : str, optional
        Output units to assign. Only necessary if `op` is function not supported by :py:func:`xclim.core.units.to_agg_units`.
    indexer : {dim: indexer, }, optional
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
        month=1 to select January, or month=[6,7,8] to select summer months. If not indexer is given, all values are
        considered.

    Returns
    -------
    xr.DataArray
        The maximum value for each period.
    """
    da = select_time(da, **indexer)
    r = da.resample(time=freq)
    if isinstance(op, str):
        op = _xclim_ops.get(op, op)
    if isinstance(op, str):
        out = getattr(r, op.replace("integral", "sum"))(dim="time", keep_attrs=True)
    else:
        with xr.set_options(keep_attrs=True):
            out = r.map(op)
        op = op.__name__
    if out_units is not None:
        return out.assign_attrs(units=out_units)
    return to_agg_units(out, da, op)
