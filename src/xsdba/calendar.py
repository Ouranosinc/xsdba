"""
Calendar Handling Utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import dtypes

from xsdba.core_utils import get_op, uses_dask
from xsdba.typing import DataType


_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_MONTHS_ABBREVIATIONS = dict(zip(range(1, len(_MONTHS) + 1), _MONTHS, strict=False))
_MONTHS_NUMBERS = dict(zip(_MONTHS, range(1, len(_MONTHS) + 1), strict=False))


def get_gen_seasons(freq):
    """
    Parse an offset string and return season coordinates.

    Parameters
    ----------
    freq : str
        Frequency offset.

    Returns
    -------
    array-like
        Generalized season coordinates.
    """
    mult, _, _, anchor = parse_offset(freq)
    month_str = "".join([s[0] for s in _MONTHS])  # JFMAMJJSOND
    # reorder month string to start on the anchor
    # QS-NOV -> NDJFMAMJJASO
    # QS-MAY -> MJJASONDJFMA
    ii = _MONTHS_NUMBERS[anchor] - 1
    month_str = month_str[ii:] + month_str[:ii]
    if mult == 1:
        # QS-MAY -> MJJ, ASO, NDJ, FMA
        return [month_str[3 * i : 3 * i + 3] for i in range(4)]
    if mult == 2:
        # 2QS-MAY -> MJJASO, NDJFMA
        return [month_str[6 * i : 6 * i + 6] for i in range(2)]


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
    return f"{mult if mult > 1 else ''}{base}{start}{'-' if anchor else ''}{anchor or ''}"


def add_gen_season_coord(ds: DataType, freq: str) -> DataType:
    """
    Add a season coordinates on a resampled dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
      The xarray object with a "time" coordinate.
      Only supports daily or coarser frequencies (excluding weekly).
      The time axis must be complete and regular (`xr.infer_freq(ds.time)` doesn't fail).
    freq : str
      Resampling frequency. Must be between "MS" and "YS" and divide a year evenly.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Input dataset with season coordinate.
    """
    dsr = ds[{d: 0 for d in set(ds.dims) - {"time"}}].resample(time=freq).first()
    mult, base, isstart, anchor = parse_offset(freq)
    if base not in "YAQM":
        raise ValueError(f"Only daily frequencies or coarser are supported. Got: {freq}.")
    if (base == "M" and 12 % mult != 0) or (base == "Q" and mult not in [1, 2, 4]) or (base in "YA" and mult > 1):
        raise ValueError(f"Only periods  that divide the year evenly are supported. Got {freq}.")
    if base in "YA":
        season_coords = ["annual"] * ds.time.size
    elif base == "Q" or (base == "M" and mult > 1):
        months = np.array(list("JFMAMJJASOND"))
        n = mult * {"M": 1, "Q": 3}[base]
        seasons = {}
        for m in dsr.time.dt.month.values:
            label = "".join(months[np.array(range(m - 1, m + n - 1)) % 12])
            for i in range(n):
                seasons[(m - 1 + i) % 12 + 1] = label
        season_coords = [seasons[m] for m in ds.time.dt.month.values]
    else:  # M or MS
        seasons = dict(zip(_MONTHS_NUMBERS.values(), _MONTHS_NUMBERS.keys(), strict=False))
        season_coords = [seasons[m] for m in ds.time.dt.month.values]
    season_length = len(season_coords[0]) if base != "M" else 1
    attrs = dict(mult=mult, base=base, isstart=isstart, anchor=anchor or "JAN", season_length=season_length)
    return ds.assign_coords(gen_season=("time", season_coords, attrs))


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


# TODO: Do not allow this?
def _interpolate_doy_calendar(source: xr.DataArray, doy_max: int, doy_min: int = 1) -> xr.DataArray:
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
    filled_na.coords["dayofyear"] = np.linspace(start=doy_min, stop=doy_max, num=len(filled_na.coords["dayofyear"]))

    return filled_na.interp(dayofyear=range(doy_min, doy_max + 1))


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
        raise ValueError(f"Stride must be less than or equal to window. Got {stride} > {window}.")

    srcfreq = xr.infer_freq(da.time)
    cal = da.time.dt.calendar
    use_cftime = da.time.dtype == "O"

    if (
        # if srcfreq in ("D", "h", "min", "s", "ms", "us", "ns")
        # TODO: Can we remove compare_offsets, only used here
        compare_offsets(srcfreq, "<=", "D")
        and align_days
        and ((freq.startswith(("Y", "A")) and cal not in uniform_calendars) or (freq.startswith(("Q", "M")) and window > 1 and cal != "360_day"))
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
            min_resamp = time2.isel(time=slice(strd_slc.start, None)).resample(time=minl_frq)
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
                ((strd_slc.start + win_slc.stop) if win_slc.stop is not None else da.time.size),
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
    fake_time = xr.date_range(start, periods=longest, freq=srcfreq, calendar=cal, use_cftime=use_cftime)
    # Slice and concat along new dim. We drop the index and add a new one so that xarray can concat them together.
    out = xr.concat(
        [da.isel(time=slc).drop_vars("time").assign_coords(time=np.arange(slc.stop - slc.start)) for slc in periods],
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
        raise ValueError(f"`unstack_periods` can't find the window, stride and freq attributes on the {dim} coordinates.") from err

    if unequal_lengths:
        try:
            lengths = da[f"{dim}_length"]
        except KeyError as err:
            raise ValueError(f"`unstack_periods` can't find the `{dim}_length` coordinate.") from err
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
        time_as_delta = pd.TimedeltaIndex(time_as_delta).to_pytimedelta()  # this array is O, numpy complies
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
        for i, (start, length) in enumerate(zip(starts.values, lengths.values, strict=False)):
            real_time = _reconstruct_time(time_as_delta, start)
            periods.append(da.isel(**{dim: i}, drop=True).isel(time=slice(0, length)).assign_coords(time=real_time.isel(time=slice(0, length))))
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
    for i, (start, length) in enumerate(zip(starts.values, lengths.values, strict=False)):
        real_time = _reconstruct_time(time_as_delta, start)
        slices = list(real_time.resample(time=strd_frq).groups.values())
        if i == 0:
            slc = slice(slices[0].start, min(slices[mid].stop, length))
        elif i == da.period.size - 1:
            slc = slice(slices[mid].start, min(slices[Nwin - 1].stop or length, length))
        else:
            slc = slice(slices[mid].start, min(slices[mid].stop, length))
        periods.append(da.isel(**{dim: i}, drop=True).isel(time=slc).assign_coords(time=real_time.isel(time=slc)))

    return xr.concat(periods, "time")
