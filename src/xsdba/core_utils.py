"""
Core Utilities
===========================

Various functions.
"""

import operator
from collections.abc import Callable, Sequence

import dask.array as dsk
import xarray as xr


# XC: core.utils
def uses_dask(*das: xr.DataArray | xr.Dataset) -> bool:
    r"""
    Evaluate whether dask is installed and array is loaded as a dask array.

    Parameters
    ----------
    *das : xr.DataArray or xr.Dataset
        DataArrays or Datasets to check.

    Returns
    -------
    bool
        True if any of the passed objects is using dask.
    """
    if len(das) > 1:
        return any([uses_dask(da) for da in das])
    da = das[0]
    if isinstance(da, xr.DataArray) and isinstance(da.data, dsk.Array):
        return True
    if isinstance(da, xr.Dataset) and any(isinstance(var.data, dsk.Array) for var in da.variables.values()):
        return True
    return False


# XC: core
def get_op(op: str, constrain: Sequence[str] | None = None) -> Callable:
    """
    Get python's comparing function according to its name of representation and validate allowed usage.

    Parameters
    ----------
    op : str
        Operator.
    constrain : sequence of str, optional
        A tuple of allowed operators.
    """
    # XC
    binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}
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

    return getattr(operator, f"__{binary_op}__")
