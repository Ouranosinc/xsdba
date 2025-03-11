"""
# noqa: SS01
Base Classes and Developer Tools
================================
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, Sequence
from inspect import _empty, signature

import cftime
import dask.array as dsk
import jsonpickle
import numpy as np
import pandas as pd
import xarray as xr
from boltons.funcutils import wraps
from xarray.core import dtypes

from xsdba.options import OPTIONS

# TODO : Redistributes some functions in existing/new scripts


# ## Base class for the sdba module
class Parametrizable(UserDict):
    """
    Helper base class resembling a dictionary.

    This object is _completely_ defined by the content of its internal dictionary, accessible through item access
    (`self['attr']`) or in `self.parameters`. When serializing and restoring this object, only members of that internal
    dict are preserved. All other attributes set directly with `self.attr = value` will not be preserved upon
    serialization and restoration of the object with `[json]pickle` dictionary. Other variables set with
    `self.var = data` will be lost in the serialization process.
    This class is best serialized and restored with `jsonpickle`.
    """

    _repr_hide_params = []

    def __getstate__(self):
        """For (json)pickle, a Parametrizable should be defined by its internal dict only."""
        return self.data

    def __setstate__(self, state):
        """For (json)pickle, a Parametrizable in only defined by its internal dict."""
        # Unpickling skips the init, so we must _set_ data, we can't just update it - it's not there yet
        self.data = {**state}

    def __getattr__(self, attr):
        """Get attributes."""
        if attr == "data" or attr not in self.data:
            return self.__getattribute__(attr)
        return self.data[attr]

    @property
    def parameters(self) -> dict:
        """All parameters as a dictionary. Read-only."""
        return {**self.data}

    def __repr__(self) -> str:
        """Return a string representation."""
        # Get default values from the init signature
        defaults = {
            # A default value of None could mean an empty mutable object
            n: [p.default] if p.default is not None else [[], {}, set(), None]
            for n, p in signature(self.__init__).parameters.items()
            if p.default is not _empty
        }
        # The representation only includes the parameters with a value different from their default
        # and those not explicitly excluded.
        params = ", ".join(
            [
                f"{k}={v!r}"
                for k, v in self.items()
                if k not in self._repr_hide_params and v not in defaults.get(k, [])
            ]
        )
        return f"{self.__class__.__name__}({params})"


class ParametrizableWithDataset(Parametrizable):
    """Parametrizable class that also has a `ds` attribute storing a dataset."""

    _attribute = "_xsdba_parameters"

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """
        Create an instance from a dataset.

        The dataset must have a global attribute with a name corresponding to `cls._attribute`,
        and that attribute must be the result of `jsonpickle.encode(object)` where object is
        of the same type as this object.
        """
        obj = jsonpickle.decode(ds.attrs[cls._attribute])  # noqa: S301
        obj.set_dataset(ds)
        return obj

    def set_dataset(self, ds: xr.Dataset) -> None:
        """
        Store an xarray dataset in the `ds` attribute.

        Useful with custom object initialization or if some external processing was performed.
        """
        self.ds = ds
        self.ds.attrs[self._attribute] = jsonpickle.encode(self)


class Grouper(Parametrizable):
    """Grouper inherited class for parameterizable classes."""

    _repr_hide_params = ["dim", "prop"]  # For a concise repr
    # Two constants for use of `map_blocks` and `map_groups`.
    # They provide better code readability, nothing more
    PROP = "<PROP>"
    DIM = "<DIM>"
    ADD_DIMS = "<ADD_DIMS>"

    def __init__(
        self,
        group: str,
        window: int = 1,
        add_dims: Sequence[str] | set[str] | None = None,
    ):
        """
        Create the Grouper object.

        Parameters
        ----------
        group : str
            The usual grouping name as xarray understands it. Ex: "time.month" or "time".
            The dimension name before the dot is the "main dimension" stored in `Grouper.dim` and
            the property name after is stored in `Grouper.prop`.
        window : int
            If larger than 1, a centered rolling window along the main dimension is created when grouping data.
            Units are the sampling frequency of the data along the main dimension.
        add_dims : Optional[Union[Sequence[str], str]]
            Additional dimensions that should be reduced in grouping operations. This behaviour is also controlled
            by the `main_only` parameter of the `apply` method. If any of these dimensions are absent from the
            DataArrays, they will be omitted.
        """
        if "." in group:
            dim, prop = group.split(".")
        else:
            dim, prop = group, "group"
        # TODO : Remove this special workaround
        # This will only work with MBCn
        if group == "5D":
            dim = "time"
        if isinstance(add_dims, str):
            add_dims = [add_dims]

        add_dims = add_dims or []
        super().__init__(
            dim=dim,
            add_dims=add_dims,
            prop=prop,
            name=group,
            window=window,
        )

    @classmethod
    def from_kwargs(cls, **kwargs) -> dict[str, Grouper]:
        """Parameterize groups using kwargs."""
        kwargs["group"] = cls(
            group=kwargs.pop("group"),
            window=kwargs.pop("window", 1),
            add_dims=kwargs.pop("add_dims", []),
        )
        return kwargs

    @property
    def freq(self):
        """
        Format a frequency string corresponding to the group.

        For use with xarray's resampling functions.
        """
        return {
            "group": "YS",
            "season": "QS-DEC",
            "month": "MS",
            "week": "W",
            "dayofyear": "D",
        }.get(self.prop, None)

    @property
    def prop_name(self):
        """Create a significant name for the grouping."""
        return "year" if self.prop == "group" else self.prop

    def get_coordinate(self, ds: xr.Dataset | None = None) -> xr.DataArray:
        """
        Return the coordinate as in the output of group.apply.

        Currently, only implemented for groupings with prop == `month` or `dayofyear`.
        For prop == `dayfofyear`, a ds (Dataset or DataArray) can be passed to infer
        the max day of year from the available years and calendar.
        """
        if self.prop == "month":
            return xr.DataArray(np.arange(1, 13), dims=("month",), name="month")
        if self.prop == "season":
            return xr.DataArray(
                ["DJF", "MAM", "JJA", "SON"], dims=("season",), name="season"
            )
        if self.prop == "dayofyear":
            if ds is not None:
                cal = ds.time.dt.calendar
                mdoy = max(
                    xr.coding.calendar_ops._days_in_year(yr, cal)
                    for yr in np.unique(ds[self.dim].dt.year)
                )
            else:
                mdoy = 365
            return xr.DataArray(
                np.arange(1, mdoy + 1), dims="dayofyear", name="dayofyear"
            )
        if self.prop == "group":
            return xr.DataArray([1], dims=("group",), name="group")
        # TODO: woups what happens when there is no group? (prop is None)
        raise NotImplementedError("No grouping found.")

    def group(
        self,
        da: xr.DataArray | xr.Dataset | None = None,
        main_only: bool = False,
        **das: xr.DataArray,
    ) -> xr.core.groupby.GroupBy:  # pylint: disable=no-member
        """
        Return a xr.core.groupby.GroupBy object.

        More than one array can be combined to a dataset before grouping using the `das` kwargs.
        A new `window` dimension is added if `self.window` is larger than 1.
        If `Grouper.dim` is 'time', but 'prop' is None, the whole array is grouped together.

        When multiple arrays are passed, some of them can be grouped along the same group as self.
        They are broadcast, merged to the grouping dataset and regrouped in the output.
        """
        if das:
            from .utils import (  # pylint: disable=cyclic-import,import-outside-toplevel
                broadcast,
            )

            if da is not None:
                das[da.name] = da

            da = xr.Dataset(
                data_vars={
                    name: das.pop(name)
                    for name in list(das.keys())
                    if self.dim in das[name].dims
                }
            )

            # "Ungroup" the grouped arrays
            da = da.assign(
                {
                    name: broadcast(var, da[self.dim], group=self, interp="nearest")
                    for name, var in das.items()
                }
            )

        if not main_only and self.window > 1:
            da = da.rolling(center=True, **{self.dim: self.window}).construct(
                window_dim="window"
            )
            if uses_dask(da):
                # Rechunk. There might be padding chunks.
                da = da.chunk({self.dim: -1})

        if self.prop == "group":
            group = self.get_index(da)
        else:
            group = self.name

        return da.groupby(group)

    def get_index(
        self,
        da: xr.DataArray | xr.Dataset,
        interp: bool | None = None,
    ) -> xr.DataArray:
        """
        Return the group index of each element along the main dimension.

        Parameters
        ----------
        da : xr.DataArray or xr.Dataset
            The input array/dataset for which the group index is returned.
            It must have `Grouper.dim` as a coordinate.
        interp : bool, optional
            If True, the returned index can be used for interpolation. Only value for month
            grouping, where integer values represent the middle of the month, all other
            days are linearly interpolated in between.

        Returns
        -------
        xr.DataArray
            The index of each element along `Grouper.dim`.
            If `Grouper.dim` is `time` and `Grouper.prop` is None, a uniform array of True is returned.
            If `Grouper.prop` is a time accessor (month, dayofyear, etc.), a numerical array is returned,
            with a special case of `month` and `interp=True`.
            If `Grouper.dim` is not `time`, the dim is simply returned.
        """
        if self.prop == "group":
            if self.dim == "time":
                return xr.full_like(da[self.dim], 1, dtype=int).rename("group")
            return da[self.dim].rename("group")

        ind = da.indexes[self.dim]

        if interp and self.dim == "time":
            if self.prop == "month":
                i = ind.month - 0.5 + ind.day / ind.days_in_month

            elif self.prop == "season":
                calendar = ind.calendar if hasattr(ind, "calendar") else "standard"
                length_year = (
                    360
                    if calendar == "360_day"
                    else 365 + (0 if calendar == "noleap" else ind.is_leap_year)
                )
                # This is assuming that seasons have the same length. The factor 1/6 comes from the fact that
                # the first season is shifted by 1 month the but the middle of the season is shifted in the other direction
                # by half a month so -(1/12-1/24)*4 = -1/6
                i = ind.dayofyear / length_year * 4 - 1 / 6
            elif self.prop == "dayofyear":
                i = ind.dayofyear
            else:
                raise ValueError(
                    f"Interpolation is not supported for {self.dim}.{self.prop}."
                )
        else:
            if self.prop == "week":
                i = da[self.dim].copy(data=ind.isocalendar().week).astype(int)
            elif self.prop == "season":
                i = da[self.dim].copy(data=ind.month % 12 // 3)
            else:
                i = getattr(ind, self.prop)

            if not np.issubdtype(i.dtype, np.integer):
                raise ValueError(
                    f"Index {self.name} is not of type int (rather {i.dtype}), "
                    f"but {self.__class__.__name__} requires integer indexes."
                )

        xi = xr.DataArray(
            i,
            dims=self.dim,
            coords={self.dim: da.coords[self.dim]},
            name=self.dim + " group index",
        )

        # Expand dimensions of index to match the dimensions of da
        # We want vectorized indexing with no broadcasting
        # xi = xi.broadcast_like(da)
        xi.name = self.prop
        return xi

    def apply(
        self,
        func: Callable | str,
        da: xr.DataArray | dict[str, xr.DataArray] | xr.Dataset,
        main_only: bool = False,
        **kwargs,
    ) -> xr.DataArray | xr.Dataset:
        r"""
        Apply a function group-wise on DataArrays.

        Parameters
        ----------
        func : Callable or str
            The function to apply to the groups, either a callable or a `xr.core.groupby.GroupBy` method name as a string.
            The function will be called as `func(group, dim=dims, **kwargs)`. See `main_only` for the behaviour of `dims`.
        da : xr.DataArray or dict[str, xr.DataArray] or xr.Dataset
            The DataArray on which to apply the function. Multiple arrays can be passed through a dictionary.
            A dataset will be created before grouping.
        main_only : bool
            Whether to call the function with the main dimension only (if True) or with all grouping dims
            (if False, default) (including the window and dimensions given through `add_dims`).
            The dimensions used are also written in the "group_compute_dims" attribute.
            If all the input arrays are missing one of the 'add_dims', it is silently omitted.
        \*\*kwargs
            Other keyword arguments to pass to the function.

        Returns
        -------
        xr.DataArray or xr.Dataset
            Attributes "group", "group_window" and "group_compute_dims" are added.

            If the function did not reduce the array:

            - The output is sorted along the main dimension.
            - The output is rechunked to match the chunks on the input
              If multiple inputs with differing chunking were given as inputs,
              the chunking with the smallest number of chunks is used.

            If the function reduces the array:

            - If there is only one group, the singleton dimension is squeezed out of the output
            - The output is rechunked as to have only 1 chunk along the new dimension.

        Notes
        -----
        For the special case where a Dataset is returned, but only some of its variable where reduced by the grouping,
        xarray's `GroupBy.map` will broadcast everything back to the ungrouped dimensions. To overcome this issue,
        function may add a "_group_apply_reshape" attribute set to `True` on the variables that should be reduced and
        these will be re-grouped by calling `da.groupby(self.name).first()`.
        """
        if isinstance(da, dict | xr.Dataset):
            grpd = self.group(main_only=main_only, **da)
            dim_chunks = min(  # Get smallest chunking to rechunk if the operation is non-grouping
                [
                    d.chunks[d.get_axis_num(self.dim)]
                    for d in da.values()
                    if uses_dask(d) and self.dim in d.dims
                ]
                or [[]],  # pass [[]] if no DataArrays have chunks so min doesn't fail
                key=len,
            )
        else:
            grpd = self.group(da, main_only=main_only)
            # Get chunking to rechunk is the operation is non-grouping
            # To match the behaviour of the case above, an empty list signifies that dask is not used for the input.
            dim_chunks = (
                [] if not uses_dask(da) else da.chunks[da.get_axis_num(self.dim)]
            )

        if main_only:
            dims = self.dim
        else:
            dims = [self.dim] + self.add_dims
            if self.window > 1:
                dims += ["window"]

        if isinstance(func, str):
            out = getattr(grpd, func)(dim=dims, **kwargs)
        else:
            out = grpd.map(func, dim=dims, **kwargs)

        # Case where the function wants to return more than one variable.
        # and that some have grouped dims and other have the same dimensions as the input.
        # In that specific case, groupby broadcasts everything back to the input's dim, copying the grouped data.
        if isinstance(out, xr.Dataset):
            for name, outvar in out.data_vars.items():
                if "_group_apply_reshape" in outvar.attrs:
                    out[name] = self.group(outvar, main_only=True).first(
                        skipna=False, keep_attrs=True
                    )
                    del out[name].attrs["_group_apply_reshape"]

        # Save input parameters as attributes of output DataArray.
        out.attrs["group"] = self.name
        out.attrs["group_compute_dims"] = dims
        out.attrs["group_window"] = self.window

        # On non-reducing ops, drop the constructed window
        if self.window > 1 and "window" in out.dims:
            out = out.isel(window=self.window // 2, drop=True)

        # If the grouped operation did not reduce the array, the result is sometimes unsorted along dim
        if self.dim in out.dims:
            out = out.sortby(self.dim)
            # The expected behavior for downstream methods would be to conserve chunking along dim
            if uses_dask(out):
                # or -1 in case dim_chunks is [], when no input is chunked
                # (only happens if the operation is chunking the output)
                out = out.chunk({self.dim: dim_chunks or -1})
        if self.prop == "season" and self.prop in out.coords:
            # Special case for "DIM.season", it is often returned in alphabetical order,
            # but that doesn't fit the coord given in get_coordinate
            out = out.sel(season=np.array(["DJF", "MAM", "JJA", "SON"]))
        if self.prop in out.dims and uses_dask(out):
            # Same as above : downstream methods expect only one chunk along the group
            out = out.chunk({self.prop: -1})

        return out


def parse_group(func: Callable, kwargs=None, allow_only=None) -> Callable:
    """
    Parse the kwargs given to a function to set the `group` arg with a Grouper object.

    This function can be used as a decorator, in which case the parsing and updating of the kwargs is done at call time.
    It can also be called with a function from which extract the default group and kwargs to update,
    in which case it returns the updated kwargs.

    If `allow_only` is given, an exception is raised when the parsed group is not within that list.
    """
    sig = signature(func)
    if "group" in sig.parameters:
        default_group = sig.parameters["group"].default
    else:
        default_group = None

    def _update_kwargs(_kwargs, allowed=None):
        if default_group or "group" in _kwargs:
            _kwargs.setdefault("group", default_group)
            if not isinstance(_kwargs["group"], Grouper):
                _kwargs = Grouper.from_kwargs(**_kwargs)
        if (
            allowed is not None
            and "group" in _kwargs
            and _kwargs["group"].prop not in allowed
        ):
            raise ValueError(
                f"Grouping on {_kwargs['group'].prop_name} is not allowed for this "
                f"function. Should be one of {allowed}."
            )
        return _kwargs

    if kwargs is not None:  # Not used as a decorator
        return _update_kwargs(kwargs, allowed=allow_only)

    # else (then it's a decorator)
    @wraps(func)
    def _parse_group(*f_args, **f_kwargs):
        f_kwargs = _update_kwargs(f_kwargs, allowed=allow_only)
        return func(*f_args, **f_kwargs)

    return _parse_group


def duck_empty(
    dims: xr.DataArray.dims, sizes, dtype="float64", chunks=None
) -> xr.DataArray:
    """Return an empty DataArray based on a numpy or dask backend, depending on the "chunks" argument."""
    shape = [sizes[dim] for dim in dims]
    if chunks:
        chnks = [chunks.get(dim, (sizes[dim],)) for dim in dims]
        content = dsk.empty(shape, chunks=chnks, dtype=dtype)
    else:
        content = np.empty(shape, dtype=dtype)
    return xr.DataArray(content, dims=dims)


def _decode_cf_coords(ds: xr.Dataset):
    """Decode coords in-place."""
    crds = xr.decode_cf(ds.coords.to_dataset())
    for crdname in list(ds.coords.keys()):
        ds[crdname] = crds[crdname]
        # decode_cf introduces an encoding key for the dtype, which can confuse the netCDF writer
        dtype = ds[crdname].encoding.get("dtype")
        if np.issubdtype(dtype, np.timedelta64) or np.issubdtype(dtype, np.datetime64):
            del ds[crdname].encoding["dtype"]


def map_blocks(  # noqa: C901
    reduces: Sequence[str] | None = None, **out_vars
) -> Callable:
    r"""
    Decorator for declaring functions and wrapping them into a map_blocks.

    Takes care of constructing the template dataset. Dimension order is not preserved.
    The decorated function must always have the signature: ``func(ds, **kwargs)``, where ds is a DataArray or a Dataset.
    It must always output a dataset matching the mapping passed to the decorator.

    Parameters
    ----------
    reduces : sequence of strings
        Name of the dimensions that are removed by the function.
    \*\*out_vars
        Mapping from variable names in the output to their *new* dimensions.
        The placeholders ``Grouper.PROP``, ``Grouper.DIM`` and ``Grouper.ADD_DIMS`` can be used to signify
        ``group.prop``,``group.dim`` and ``group.add_dims`` respectively.
        If an output keeps a dimension that another loses, that dimension name must be given in ``reduces`` and in
        the list of new dimensions of the first output.
    """

    def merge_dimensions(*seqs):
        """Merge several dimensions lists while preserving order."""
        out = seqs[0].copy()
        for seq in seqs[1:]:
            last_index = 0
            for e in seq:
                if e in out:
                    indx = out.index(e)
                    if indx < last_index:
                        raise ValueError(
                            "Dimensions order mismatch, lists are not mergeable."
                        )
                    last_index = indx
                else:
                    out.insert(last_index + 1, e)
        return out

    # Ordered list of all added dimensions
    out_dims = merge_dimensions(*out_vars.values())
    # List of dimensions reduced by the function.
    red_dims = reduces or []

    def _decorator(func):  # noqa: C901
        # @wraps(func, hide_wrapped=True)
        @parse_group
        def _map_blocks(ds, **kwargs):  # noqa: C901
            if isinstance(ds, xr.Dataset):
                ds = ds.unify_chunks()

            # Get group if present
            group = kwargs.get("group")

            # Ensure group is given as it might not be in the signature of the wrapped func
            if {Grouper.PROP, Grouper.DIM, Grouper.ADD_DIMS}.intersection(
                out_dims + red_dims
            ) and group is None:
                raise ValueError("Missing required `group` argument.")

            # Make translation dict
            if group is not None:
                placeholders = {
                    Grouper.PROP: [group.prop],
                    Grouper.DIM: [group.dim],
                    Grouper.ADD_DIMS: group.add_dims,
                }
            else:
                placeholders = {}

            # Get new dimensions (in order), translating placeholders to real names.
            new_dims = []
            for dim in out_dims:
                new_dims.extend(placeholders.get(dim, [dim]))

            reduced_dims = []
            for dim in red_dims:
                reduced_dims.extend(placeholders.get(dim, [dim]))

            for dim in new_dims:
                if dim in ds.dims and dim not in reduced_dims:
                    raise ValueError(
                        f"Dimension {dim} is meant to be added by the "
                        "computation but it is already on one of the inputs."
                    )
            if uses_dask(ds):
                # Use dask if any of the input is dask-backed.
                chunks = (
                    dict(ds.chunks)
                    if isinstance(ds, xr.Dataset)
                    else dict(zip(ds.dims, ds.chunks, strict=False))
                )
                badchunks = {}
                if group is not None:
                    badchunks.update(
                        {
                            dim: chunks.get(dim)
                            for dim in group.add_dims + [group.dim]
                            if len(chunks.get(dim, [])) > 1
                        }
                    )
                badchunks.update(
                    {
                        dim: chunks.get(dim)
                        for dim in reduced_dims
                        if len(chunks.get(dim, [])) > 1
                    }
                )
                if badchunks:
                    raise ValueError(
                        f"The dimension(s) over which we group, reduce or interpolate cannot be chunked ({badchunks})."
                    )
            else:
                chunks = None

            # Dimensions untouched by the function.
            base_dims = list(set(ds.dims) - set(new_dims) - set(reduced_dims))

            # All dimensions of the output data, new_dims are added at the end on purpose.
            all_dims = base_dims + new_dims
            # The coordinates of the output data.
            added_coords = []
            coords = {}
            sizes = {}
            for dim in all_dims:
                if dim == group.prop:
                    coords[group.prop] = group.get_coordinate(ds=ds)
                elif dim == group.dim:
                    coords[group.dim] = ds[group.dim]
                elif dim in kwargs:
                    coords[dim] = xr.DataArray(kwargs[dim], dims=(dim,), name=dim)
                elif dim in ds.dims:
                    # If a dim has no coords : some sdba function will add them, so to be safe we add them right now
                    # and note them to remove them afterwards.
                    if dim not in ds.coords:
                        added_coords.append(dim)
                    ds[dim] = ds[dim]
                    coords[dim] = ds[dim]
                else:
                    raise ValueError(
                        f"This function adds the {dim} dimension, its coordinate must be provided as a keyword argument."
                    )
            sizes.update({name: crd.size for name, crd in coords.items()})

            # Create the output dataset, but empty
            tmpl = xr.Dataset(coords=coords)
            if isinstance(ds, xr.Dataset):
                # Get largest dtype of the inputs, assign it to the output.
                dtype = max(
                    (da.dtype for da in ds.data_vars.values()), key=lambda d: d.itemsize
                )
            else:
                dtype = ds.dtype

            for var, dims in out_vars.items():
                var_new_dims = []
                for dim in dims:
                    var_new_dims.extend(placeholders.get(dim, [dim]))
                # Out variables must have the base dims + new_dims
                dims = base_dims + var_new_dims
                # duck empty calls dask if chunks is not None
                tmpl[var] = duck_empty(dims, sizes, dtype=dtype, chunks=chunks)

            def _call_and_transpose_on_exit(dsblock, **f_kwargs):
                """Call the decorated func and transpose to ensure the same dim order as on the template."""
                try:
                    _decode_cf_coords(dsblock)
                    func_out = func(dsblock, **f_kwargs).transpose(*all_dims)
                except Exception as err:
                    raise ValueError(
                        f"{func.__name__} failed on block with coords : {dsblock.coords}."
                    ) from err
                return func_out

            # Fancy patching for explicit dask task names
            _call_and_transpose_on_exit.__name__ = f"block_{func.__name__}"

            # Remove all auxiliary coords on both tmpl and ds
            extra_coords = {
                name: crd for name, crd in ds.coords.items() if name not in crd.dims
            }
            ds = ds.drop_vars(extra_coords.keys())
            # Coords not sharing dims with `all_dims` (like scalar aux coord on reduced 1D input) are absent from tmpl
            tmpl = tmpl.drop_vars(extra_coords.keys(), errors="ignore")

            # Call
            out = ds.map_blocks(
                _call_and_transpose_on_exit, template=tmpl, kwargs=kwargs
            )
            # Add back the extra coords, but only those which have compatible dimensions (like xarray would have done)
            out = out.assign_coords(
                {
                    name: crd
                    for name, crd in extra_coords.items()
                    if set(crd.dims).issubset(out.dims)
                }
            )

            # Finally remove coords we added... 'ignore' in case they were already removed.
            out = out.drop_vars(added_coords, errors="ignore")
            return out

        _map_blocks.__dict__["func"] = func
        return _map_blocks

    return _decorator


def map_groups(
    reduces: Sequence[str] | None = None, main_only: bool = False, **out_vars
) -> Callable:
    r"""
    Decorator for declaring functions acting only on groups and wrapping them into a map_blocks.

    This is the same as `map_blocks` but adds a call to `group.apply()` in the mapped func and the default
    value of `reduces` is changed.

    The decorated function must have the signature: ``func(ds, dim, **kwargs)``.
    Where ds is a DataAray or Dataset, dim is the `group.dim` (and add_dims). The `group` argument
    is stripped from the kwargs, but must evidently be provided in the call.

    Parameters
    ----------
    reduces : sequence of str, optional
        Dimensions that are removed from the inputs by the function. Defaults to [Grouper.DIM, Grouper.ADD_DIMS]
        if main_only is False, and [Grouper.DIM] if main_only is True. See :py:func:`map_blocks`.
    main_only : bool
        Same as for :py:meth:`Grouper.apply`.
    \*\*out_vars
        Mapping from variable names in the output to their *new* dimensions.
        The placeholders ``Grouper.PROP``, ``Grouper.DIM`` and ``Grouper.ADD_DIMS`` can be used to signify
        ``group.prop``,``group.dim`` and ``group.add_dims``, respectively.
        If an output keeps a dimension that another loses, that dimension name must be given in `reduces` and in
        the list of new dimensions of the first output.

    See Also
    --------
    map_blocks
    """
    def_reduces = [Grouper.DIM]
    if not main_only:
        def_reduces.append(Grouper.ADD_DIMS)
    reduces = reduces or def_reduces

    def _decorator(func):
        decorator = map_blocks(reduces=reduces, **out_vars)

        def _apply_on_group(dsblock, **kwargs):
            group = kwargs.pop("group")
            return group.apply(func, dsblock, main_only=main_only, **kwargs)

        # Fancy patching for explicit dask task names
        _apply_on_group.__name__ = f"group_{func.__name__}"

        # wraps(func, injected=['dim'], hide_wrapped=True)(
        wrapper = decorator(_apply_on_group)
        wrapper.__dict__["func"] = func
        return wrapper

    return _decorator


# XC: core.utils
def ensure_chunk_size(da: xr.DataArray, **minchunks: int) -> xr.DataArray:
    r"""
    Ensure that the input DataArray has chunks of at least the given size.

    If only one chunk is too small, it is merged with an adjacent chunk.
    If many chunks are too small, they are grouped together by merging adjacent chunks.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray, with or without the dask backend. Does nothing when passed a non-dask array.
    \*\*minchunks : dict[str, int]
        A kwarg mapping from dimension name to minimum chunk size.
        Pass -1 to force a single chunk along that dimension.

    Returns
    -------
    xr.DataArray
    """
    if not uses_dask(da):
        return da

    all_chunks = dict(zip(da.dims, da.chunks))
    chunking = {}
    for dim, minchunk in minchunks.items():
        chunks = all_chunks[dim]
        if minchunk == -1 and len(chunks) > 1:
            # Rechunk to single chunk only if it's not already one
            chunking[dim] = -1

        toosmall = np.array(chunks) < minchunk  # Chunks that are too small
        if toosmall.sum() > 1:
            # Many chunks are too small, merge them by groups
            fac = np.ceil(minchunk / min(chunks)).astype(int)
            chunking[dim] = tuple(
                sum(chunks[i : i + fac]) for i in range(0, len(chunks), fac)
            )
            # Reset counter is case the last chunks are still too small
            chunks = chunking[dim]
            toosmall = np.array(chunks) < minchunk
        if toosmall.sum() == 1:
            # Only one, merge it with adjacent chunk
            ind = np.where(toosmall)[0][0]
            new_chunks = list(chunks)
            sml = new_chunks.pop(ind)
            new_chunks[max(ind - 1, 0)] += sml
            chunking[dim] = tuple(new_chunks)

    if chunking:
        return da.chunk(chunks=chunking)
    return da


# XC: core.utils
def uses_dask(*das: xr.DataArray | xr.Dataset) -> bool:
    r"""
    Evaluate whether dask is installed and array is loaded as a dask array.

    Parameters
    ----------
    \*das : xr.DataArray or xr.Dataset
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
    if isinstance(da, xr.Dataset) and any(
        isinstance(var.data, dsk.Array) for var in da.variables.values()
    ):
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

    return xr.core.ops.get_op(binary_op)


# XC: calendar
# TODO: Do not allow this?
def _interpolate_doy_calendar(
    source: xr.DataArray, doy_max: int, doy_min: int = 1
) -> xr.DataArray:
    """
    Interpolate from one set of dayofyear range to another.

    Interpolate an array defined over a `dayofyear` range (say 1 to 360) to another `dayofyear` range (say 1
    to 365).

    Parameters
    ----------
    source : xr.DataArray
        Array with `dayofyear` coordinates.
    doy_max : int
        The largest day of the year allowed by calendar.
    doy_min : int
        The smallest day of the year in the output.
        This parameter is necessary when the target time series does not span over a full year (e.g. JJA season).
        Default is 1.

    Returns
    -------
    xr.DataArray
        Interpolated source array over coordinates spanning the target `dayofyear` range.
    """
    if "dayofyear" not in source.coords.keys():
        raise AttributeError("Source should have `dayofyear` coordinates.")

    # Interpolate to fill na values
    da = source
    if uses_dask(source):
        # interpolate_na cannot run on chunked dayofyear.
        da = source.chunk({"dayofyear": -1})
    filled_na = da.interpolate_na(dim="dayofyear")

    # Interpolate to target dayofyear range
    filled_na.coords["dayofyear"] = np.linspace(
        start=doy_min, stop=doy_max, num=len(filled_na.coords["dayofyear"])
    )

    return filled_na.interp(dayofyear=range(doy_min, doy_max + 1))


# XC: calendar
def parse_offset(freq: str) -> tuple[int, str, bool, str | None]:
    """
    Parse an offset string.

    Parse a frequency offset and, if needed, convert to cftime-compatible components.

    Parameters
    ----------
    freq : str
        Frequency offset.

    Returns
    -------
    multiplier : int
        Multiplier of the base frequency. "[n]W" is always replaced with "[7n]D",
        as xarray doesn't support "W" for cftime indexes.
    offset_base : str
        Base frequency.
    is_start_anchored : bool
        Whether coordinates of this frequency should correspond to the beginning of the period (`True`)
        or its end (`False`). Can only be False when base is Y, Q or M; in other words, xsdba assumes frequencies finer
        than monthly are all start-anchored.
    anchor : str, optional
        Anchor date for bases Y or Q. As xarray doesn't support "W",
        neither does xsdba (anchor information is lost when given).
    """
    # Useful to raise on invalid freqs, convert Y to A and get default anchor (A, Q)
    offset = pd.tseries.frequencies.to_offset(freq)
    base, *anchor = offset.name.split("-")
    anchor = anchor[0] if len(anchor) > 0 else None
    start = ("S" in base) or (base[0] not in "AYQM")
    if base.endswith("S") or base.endswith("E"):
        base = base[:-1]
    mult = offset.n
    if base == "W":
        mult = 7 * mult
        base = "D"
        anchor = None
    return mult, base, start, anchor


# XC : calendar
def compare_offsets(freqA: str, op: str, freqB: str) -> bool:
    """
    Compare offsets string based on their approximate length, according to a given operator.

    Offset are compared based on their length approximated for a period starting
    after 1970-01-01 00:00:00. If the offsets are from the same category (same first letter),
    only the multiplier prefix is compared (QS-DEC == QS-JAN, MS < 2MS).
    "Business" offsets are not implemented.

    Parameters
    ----------
    freqA : str
        RHS Date offset string ('YS', '1D', 'QS-DEC', ...).
    op : {'<', '<=', '==', '>', '>=', '!='}
        Operator to use.
    freqB : str
        LHS Date offset string ('YS', '1D', 'QS-DEC', ...).

    Returns
    -------
    bool
        Return freqA op freqB.
    """
    # Get multiplier and base frequency
    t_a, b_a, _, _ = parse_offset(freqA)
    t_b, b_b, _, _ = parse_offset(freqB)

    if b_a != b_b:
        # Different base freq, compare length of first period after beginning of time.
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqA)
        t_a = (t[1] - t[0]).total_seconds()
        t = pd.date_range("1970-01-01T00:00:00.000", periods=2, freq=freqB)
        t_b = (t[1] - t[0]).total_seconds()
    # else Same base freq, compare multiplier only.

    return get_op(op)(t_a, t_b)


# XC: calendar
def construct_offset(mult: int, base: str, start_anchored: bool, anchor: str | None):
    """
    Reconstruct an offset string from its parts.

    Parameters
    ----------
    mult : int
        The period multiplier (>= 1).
    base : str
        The base period string (one char).
    start_anchored : bool
        If True and base in [Y, Q, M], adds the "S" flag, False add "E".
    anchor : str, optional
        The month anchor of the offset. Defaults to JAN for bases YS and QS and to DEC for bases YE and QE.

    Returns
    -------
    str
        An offset string, conformant to pandas-like naming conventions.

    Notes
    -----
    This provides the mirror opposite functionality of :py:func:`parse_offset`.
    """
    start = ("S" if start_anchored else "E") if base in "YAQM" else ""
    if anchor is None and base in "AQY":
        anchor = "JAN" if start_anchored else "DEC"
    return (
        f"{mult if mult > 1 else ''}{base}{start}{'-' if anchor else ''}{anchor or ''}"
    )


# XC: calendar
# Names of calendars that have the same number of days for all years
uniform_calendars = ("noleap", "all_leap", "365_day", "366_day", "360_day")


# XC: calendar
def _month_is_first_period_month(time, freq):
    """Return True if the given time is from the first month of freq."""
    if isinstance(time, cftime.datetime):
        frq_monthly = xr.coding.cftime_offsets.to_offset("MS")
        frq = xr.coding.cftime_offsets.to_offset(freq)
        if frq_monthly.onOffset(time):
            return frq.onOffset(time)
        return frq.onOffset(frq_monthly.rollback(time))
    # Pandas
    time = pd.Timestamp(time)
    frq_monthly = pd.tseries.frequencies.to_offset("MS")
    frq = pd.tseries.frequencies.to_offset(freq)
    if frq_monthly.is_on_offset(time):
        return frq.is_on_offset(time)
    return frq.is_on_offset(frq_monthly.rollback(time))


# XC: calendar
# TODO: implement needed functions in stack_periods
# move to processing
def stack_periods(
    da: xr.Dataset | xr.DataArray,
    window: int = 30,
    stride: int | None = None,
    min_length: int | None = None,
    freq: str = "YS",
    dim: str = "period",
    start: str = "1970-01-01",
    align_days: bool = True,
    pad_value=dtypes.NA,
):
    """
    Construct a multi-period array.

    Stack different equal-length periods of `da` into a new 'period' dimension.

    This is similar to ``da.rolling(time=window).construct(dim, stride=stride)``, but adapted for arguments
    in terms of a base temporal frequency that might be non-uniform (years, months, etc.).
    It is reversible for some cases (see `stride`).
    A rolling-construct method will be much more performant for uniform periods (days, weeks).

    Parameters
    ----------
    da : xr.Dataset or xr.DataArray
        An xarray object with a `time` dimension.
        Must have a uniform timestep length.
        Output might be strange if this does not use a uniform calendar (noleap, 360_day, all_leap).
    window : int
        The length of the moving window as a multiple of ``freq``.
    stride : int, optional
        At which interval to take the windows, as a multiple of ``freq``.
        For the operation to be reversible with :py:func:`unstack_periods`, it must divide `window` into an odd number of parts.
        Default is `window` (no overlap between periods).
    min_length : int, optional
        Windows shorter than this are not included in the output.
        Given as a multiple of ``freq``. Default is ``window`` (every window must be complete).
        Similar to the ``min_periods`` argument of  ``da.rolling``.
        If ``freq`` is annual or quarterly and ``min_length == ``window``, the first period is considered complete
        if the first timestep is in the first month of the period.
    freq : str
        Units of ``window``, ``stride`` and ``min_length``, as a frequency string.
        Must be larger or equal to the data's sampling frequency.
        Note that this function offers an easier interface for non-uniform period (like years or months)
        but is much slower than a rolling-construct method.
    dim : str
        The new dimension name.
    start : str
        The `start` argument passed to :py:func:`xarray.date_range` to generate the new placeholder
        time coordinate.
    align_days : bool
        When True (default), an error is raised if the output would have unaligned days across periods.
        If `freq = 'YS'`, day-of-year alignment is checked and if `freq` is "MS" or "QS", we check day-in-month.
        Only uniform-calendar will pass the test for `freq='YS'`.
        For other frequencies, only the `360_day` calendar will work.
        This check is ignored if the sampling rate of the data is coarser than "D".
    pad_value : Any
        When some periods are shorter than others, this value is used to pad them at the end.
        Passed directly as argument ``fill_value`` to :py:func:`xarray.concat`,
        the default is the same as on that function.

    Returns
    -------
    xr.DataArray
        A DataArray with a new `period` dimension and a `time` dimension with the length of the longest window.
        The new time coordinate has the same frequency as the input data but is generated using
        :py:func:`xarray.date_range` with the given `start` value.
        That coordinate is the same for all periods, depending on the choice of ``window`` and ``freq``, it might make sense.
        But for unequal periods or non-uniform calendars, it will certainly not.
        If ``stride`` is a divisor of ``window``, the correct timeseries can be reconstructed with :py:func:`unstack_periods`.
        The coordinate of `period` is the first timestep of each window.
    """
    # Import in function to avoid cyclical imports
    from xsdba.units import (  # pylint: disable=import-outside-toplevel
        infer_sampling_units,
        units2str,
    )

    stride = stride or window
    min_length = min_length or window
    if stride > window:
        raise ValueError(
            f"Stride must be less than or equal to window. Got {stride} > {window}."
        )

    srcfreq = xr.infer_freq(da.time)
    cal = da.time.dt.calendar
    use_cftime = da.time.dtype == "O"

    if (
        # if srcfreq in ("D", "h", "min", "s", "ms", "us", "ns")
        # TODO: Can we remove compare_offsets, only used here
        compare_offsets(srcfreq, "<=", "D")
        and align_days
        and (
            (freq.startswith(("Y", "A")) and cal not in uniform_calendars)
            or (freq.startswith(("Q", "M")) and window > 1 and cal != "360_day")
        )
    ):
        if freq.startswith(("Y", "A")):
            u = "year"
        else:
            u = "month"
        raise ValueError(
            f"Stacking {window}{freq} periods will result in unaligned day-of-{u}. "
            f"Consider converting the calendar of your data to one with uniform {u} lengths, "
            "or pass `align_days=False` to disable this check."
        )

    # Convert integer inputs to freq strings
    mult, *args = parse_offset(freq)
    # TODO: remove construct?  (hard code  construct-offset)
    win_frq = construct_offset(mult * window, *args)
    strd_frq = construct_offset(mult * stride, *args)
    minl_frq = construct_offset(mult * min_length, *args)

    # The same time coord as da, but with one extra element.
    # This way, the last window's last index is not returned as None by xarray's grouper.
    time2 = xr.DataArray(
        xr.date_range(
            da.time[0].item(),
            freq=srcfreq,
            calendar=cal,
            periods=da.time.size + 1,
            use_cftime=use_cftime,
        ),
        dims=("time",),
        name="time",
    )

    periods = []
    # longest = 0
    # Iterate over strides, but recompute the full window for each stride start
    for strd_slc in da.resample(time=strd_frq).groups.values():
        win_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(time=win_frq)
        # Get slice for first group
        win_slc = list(win_resamp.groups.values())[0]
        if min_length < window:
            # If we ask for a min_length period instead is it complete ?
            min_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(
                time=minl_frq
            )
            min_slc = list(min_resamp.groups.values())[0]
            open_ended = min_slc.stop is None
        else:
            # The end of the group slice is None if no outside-group value was found after the last element
            # As we added an extra step to time2, we avoid the case where a group ends exactly on the last element of ds
            open_ended = win_slc.stop is None
        if open_ended:
            # Too short, we got to the end
            break
        if (
            strd_slc.start == 0
            and parse_offset(freq)[1] in "YAQ"
            and min_length == window
            and not _month_is_first_period_month(da.time[0].item(), freq)
        ):
            # For annual or quarterly frequencies (which can be anchor-based),
            # if the first time is not in the first month of the first period,
            # then the first period is incomplete but by a fractional amount.
            continue
        periods.append(
            slice(
                strd_slc.start + win_slc.start,
                (
                    (strd_slc.start + win_slc.stop)
                    if win_slc.stop is not None
                    else da.time.size
                ),
            )
        )

    # Make coordinates
    lengths = xr.DataArray(
        [slc.stop - slc.start for slc in periods],
        dims=(dim,),
        attrs={"long_name": "Length of each period"},
    )
    longest = lengths.max().item()
    # Length as a pint-ready array : with proper units, but values are not usable as indexes anymore
    m, u = infer_sampling_units(da)
    lengths = lengths * m
    lengths.attrs["units"] = units2str(u)
    # Start points for each period and remember parameters for unstacking
    starts = xr.DataArray(
        [da.time[slc.start].item() for slc in periods],
        dims=(dim,),
        attrs={
            "long_name": "Start of the period",
            # Save parameters so that we can unstack.
            "window": window,
            "stride": stride,
            "freq": freq,
            "unequal_lengths": int(len(np.unique(lengths)) > 1),
        },
    )
    # The "fake" axis that all periods share
    fake_time = xr.date_range(
        start, periods=longest, freq=srcfreq, calendar=cal, use_cftime=use_cftime
    )
    # Slice and concat along new dim. We drop the index and add a new one so that xarray can concat them together.
    out = xr.concat(
        [
            da.isel(time=slc)
            .drop_vars("time")
            .assign_coords(time=np.arange(slc.stop - slc.start))
            for slc in periods
        ],
        dim,
        join="outer",
        fill_value=pad_value,
    )
    out = out.assign_coords(
        time=(("time",), fake_time, da.time.attrs.copy()),
        **{f"{dim}_length": lengths, dim: starts},
    )
    out.time.attrs.update(long_name="Placeholder time axis")
    return out


# XC: calendar
def unstack_periods(da: xr.DataArray | xr.Dataset, dim: str = "period"):
    """
    Unstack an array constructed with :py:func:`stack_periods`.

    Can only work with periods stacked with a ``stride`` that divides ``window`` in an odd number of sections.
    When ``stride`` is smaller than ``window``, only the center-most stride of each window is kept,
    except for the beginning and end which are taken from the first and last windows.

    Parameters
    ----------
    da : xr.DataArray
        As constructed by :py:func:`stack_periods`, attributes of the period coordinates must have been preserved.
    dim : str
        The period dimension name.

    Notes
    -----
    The following table shows which strides are included (``o``) in the unstacked output.

    In this example, ``stride`` was a fifth of ``window`` and ``min_length`` was four (4) times ``stride``.
    The row index ``i`` the period index in the stacked dataset,
    columns are the stride-long section of the original timeseries.

    .. table:: Unstacking example with ``stride < window``.

        === === === === === === === ===
         i   0   1   2   3   4   5   6
        === === === === === === === ===
         3               x   x   o   o
         2           x   x   o   x   x
         1       x   x   o   x   x
         0   o   o   o   x   x
        === === === === === === === ===
    """
    from xsdba.units import (  # pylint: disable=import-outside-toplevel
        infer_sampling_units,
    )

    try:
        starts = da[dim]
        window = starts.attrs["window"]
        stride = starts.attrs["stride"]
        freq = starts.attrs["freq"]
        unequal_lengths = bool(starts.attrs["unequal_lengths"])
    except (AttributeError, KeyError) as err:
        raise ValueError(
            f"`unstack_periods` can't find the window, stride and freq attributes on the {dim} coordinates."
        ) from err

    if unequal_lengths:
        try:
            lengths = da[f"{dim}_length"]
        except KeyError as err:
            raise ValueError(
                f"`unstack_periods` can't find the `{dim}_length` coordinate."
            ) from err
        # Get length as number of points
        m, _ = infer_sampling_units(da.time)
        lengths = lengths // m
    else:
        # It is acceptable to lose "{dim}_length" if they were all equal
        lengths = xr.DataArray([da.time.size] * da[dim].size, dims=(dim,))

    # Convert from the fake axis to the real one
    time_as_delta = da.time - da.time[0]
    if da.time.dtype == "O":
        # cftime can't add with np.timedelta64 (restriction comes from numpy which refuses to add O with m8)
        time_as_delta = pd.TimedeltaIndex(
            time_as_delta
        ).to_pytimedelta()  # this array is O, numpy complies
    else:
        # Xarray will return int when iterating over datetime values, this returns timestamps
        starts = pd.DatetimeIndex(starts)

    def _reconstruct_time(_time_as_delta, _start):
        times = _time_as_delta + _start
        return xr.DataArray(times, dims=("time",), coords={"time": times}, name="time")

    # Easy case:
    if window == stride:
        # just concat them all
        periods = []
        for i, (start, length) in enumerate(
            zip(starts.values, lengths.values, strict=False)
        ):
            real_time = _reconstruct_time(time_as_delta, start)
            periods.append(
                da.isel(**{dim: i}, drop=True)
                .isel(time=slice(0, length))
                .assign_coords(time=real_time.isel(time=slice(0, length)))
            )
        return xr.concat(periods, "time")

    # Difficult and ambiguous case
    if (window / stride) % 2 != 1:
        raise NotImplementedError(
            "`unstack_periods` can't work with strides that do not divide the window into an odd number of parts."
            f"Got {window} / {stride} which is not an odd integer."
        )

    # Non-ambiguous overlapping case
    Nwin = window // stride
    mid = (Nwin - 1) // 2  # index of the center window

    mult, *args = parse_offset(freq)
    strd_frq = construct_offset(mult * stride, *args)

    periods = []
    for i, (start, length) in enumerate(
        zip(starts.values, lengths.values, strict=False)
    ):
        real_time = _reconstruct_time(time_as_delta, start)
        slices = list(real_time.resample(time=strd_frq).groups.values())
        if i == 0:
            slc = slice(slices[0].start, min(slices[mid].stop, length))
        elif i == da.period.size - 1:
            slc = slice(slices[mid].start, min(slices[Nwin - 1].stop or length, length))
        else:
            slc = slice(slices[mid].start, min(slices[mid].stop, length))
        periods.append(
            da.isel(**{dim: i}, drop=True)
            .isel(time=slc)
            .assign_coords(time=real_time.isel(time=slc))
        )

    return xr.concat(periods, "time")
