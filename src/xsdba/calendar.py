"""
Calendar Handling Utilities
===========================

Helper function to handle dates, times and different calendars with xarray.
"""

import cftime
import numpy as np
import pandas as pd
import xarray as xr

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
