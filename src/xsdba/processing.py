# pylint: disable=missing-kwoa
"""
# noqa: SS01
Pre- and Post-Processing Submodule
==================================
"""
from __future__ import annotations

import types
from collections.abc import Sequence
from typing import cast

import cftime
import dask.array as dsk
import numpy as np
import xarray as xr
from scipy.fft import dctn, idctn
from xarray.core import dtypes
from xarray.core.utils import get_temp_dimname

from xsdba._processing import _adapt_freq, _normalize, _reordering
from xsdba.base import Grouper, uses_dask
from xsdba.formatting import update_xsdba_history
from xsdba.nbutils import _escore
from xsdba.units import (
    convert_units_to,
    harmonize_units,
    normalized_wavenumber_to_wavelength,
    wavelength_to_normalized_wavenumber,
)
from xsdba.utils import ADDITIVE, copy_all_attrs

__all__ = [
    "adapt_freq",
    "escore",
    "from_additive_space",
    "grouped_time_indexes",
    "jitter",
    "jitter_over_thresh",
    "jitter_under_thresh",
    "normalize",
    "reordering",
    "spectral_filter",
    "stack_variables",
    "standardize",
    "to_additive_space",
    "uniform_noise_like",
    "unstack_variables",
    "unstandardize",
]


@update_xsdba_history
@harmonize_units(["ref", "sim", "thresh"])
def adapt_freq(
    ref: xr.DataArray,
    sim: xr.DataArray,
    *,
    group: Grouper | str,
    thresh: str = "0 mm d-1",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    r"""
    Adapt frequency of values under thresh of `sim`, in order to match ref.

    This is useful when the dry-day frequency in the simulations is higher than in the references. This function
    will create new non-null values for `sim`/`hist`, so that adjustment factors are less wet-biased.
    Based on :cite:t:`themesl_empirical-statistical_2012`.

    Parameters
    ----------
    ref : xr.Dataset
        Target/reference data, usually observed data, with a "time" dimension.
    sim : xr.Dataset
        Simulated data, with a "time" dimension.
    group : str or Grouper
        Grouping information, see base.Grouper.
    thresh : str
        Threshold below which values are considered zero, a quantity with units.

    Returns
    -------
    sim_adj : xr.DataArray
        Simulated data with the same frequency of values under threshold than ref.
        Adjustment is made group-wise.
    pth : xr.DataArray
        For each group, the smallest value of sim that was not frequency-adjusted.
        All values smaller were either left as zero values or given a random value between thresh and pth.
        NaN where frequency adaptation wasn't needed.
    dP0 : xr.DataArray
        For each group, the percentage of values that were corrected in sim.

    Notes
    -----
    With :math:`P_0^r` the frequency of values under threshold :math:`T_0` in the reference (ref) and
    :math:`P_0^s` the same for the simulated values, :math:`\\Delta P_0 = \\frac{P_0^s - P_0^r}{P_0^s}`,
    when positive, represents the proportion of values under :math:`T_0` that need to be corrected.

    The correction replaces a proportion :math:`\\Delta P_0` of the values under :math:`T_0` in sim by a uniform random
    number between :math:`T_0` and :math:`P_{th}`, where :math:`P_{th} = F_{ref}^{-1}( F_{sim}( T_0 ) )` and
    `F(x)` is the empirical cumulative distribution function (CDF).

    References
    ----------
    :cite:cts:`themesl_empirical-statistical_2012`
    """
    out = _adapt_freq(xr.Dataset(dict(sim=sim, ref=ref)), group=group, thresh=thresh)

    # Set some metadata
    copy_all_attrs(out, sim)
    out.sim_ad.attrs.update(sim.attrs)
    out.sim_ad.attrs.update(
        references="Themeßl et al. (2012), Empirical-statistical downscaling and error correction of regional climate "
        "models and its impact on the climate change signal, Climatic Change, DOI 10.1007/s10584-011-0224-4."
    )
    out.pth.attrs.update(
        long_name="Smallest value of the timeseries not corrected by frequency adaptation.",
        units=sim.units,
    )
    out.dP0.attrs.update(
        long_name=f"Proportion of values smaller than {thresh} in the timeseries corrected by frequency adaptation",
    )

    return out.sim_ad, out.pth, out.dP0


def jitter_under_thresh(x: xr.DataArray, thresh: str) -> xr.DataArray:
    """
    Replace values smaller than threshold by a uniform random noise.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold under which to add uniform random noise to values, a quantity with units.

    Returns
    -------
    xr.DataArray.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Notes
    -----
    If thresh is high, this will change the mean value of x.
    """
    j: xr.DataArray = jitter(x, lower=thresh, upper=None, minimum=None, maximum=None)
    return j


def jitter_over_thresh(x: xr.DataArray, thresh: str, upper_bnd: str) -> xr.DataArray:
    """
    Replace values greater than threshold by a uniform random noise.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    thresh : str
        Threshold over which to add uniform random noise to values, a quantity with units.
    upper_bnd : str
        Maximum possible value for the random noise, a quantity with units.

    Returns
    -------
    xr.DataArray.

    Warnings
    --------
    Not to be confused with R's jitter, which adds uniform noise instead of replacing values.

    Notes
    -----
    If thresh is low, this will change the mean value of x.
    """
    j: xr.DataArray = jitter(
        x, lower=None, upper=thresh, minimum=None, maximum=upper_bnd
    )
    return j


@update_xsdba_history
@harmonize_units(["x", "lower", "upper", "minimum", "maximum"])
def jitter(
    x: xr.DataArray,
    lower: str | None = None,
    upper: str | None = None,
    minimum: str | None = None,
    maximum: str | None = None,
) -> xr.DataArray:
    """
    Replace values under a threshold and values above another by a uniform random noise.

    Parameters
    ----------
    x : xr.DataArray
        Values.
    lower : str, optional
        Threshold under which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the lower end.
    upper : str, optional
        Threshold over which to add uniform random noise to values, a quantity with units.
        If None, no jittering is performed on the upper end.
    minimum : str, optional
        Lower limit (excluded) for the lower end random noise, a quantity with units.
        If None but `lower` is not None, 0 is used.
    maximum : str, optional
        Upper limit (excluded) for the upper end random noise, a quantity with units.
        If `upper` is not None, it must be given.

    Returns
    -------
    xr.DataArray
        Same as  `x` but values < lower are replaced by a uniform noise in range (minimum, lower)
        and values >= upper are replaced by a uniform noise in range [upper, maximum).
        The two noise distributions are independent.

    Warnings
    --------
    Not to be confused with R's `jitter`, which adds uniform noise instead of replacing values.
    """
    out: xr.DataArray = x
    notnull = x.notnull()
    if lower is not None:
        jitter_lower = np.array(lower).astype(float)
        jitter_min = np.array(minimum if minimum is not None else 0).astype(float)
        jitter_min = np.nextafter(jitter_min.astype(x.dtype), np.inf, dtype=x.dtype)
        if uses_dask(x):
            jitter_dist = dsk.random.uniform(
                low=dsk.from_array(jitter_min),
                high=dsk.from_array(jitter_lower),
                size=x.shape,
                chunks=x.chunks,
            )
        else:
            jitter_dist = np.random.uniform(
                low=jitter_min, high=jitter_lower, size=x.shape
            )
        out = out.where(~((x < jitter_lower) & notnull), jitter_dist.astype(x.dtype))
    if upper is not None:
        if maximum is None:
            raise ValueError("If 'upper' is given, so must 'maximum'.")
        jitter_upper = np.array(upper).astype(float)
        jitter_max = np.array(maximum).astype(float)
        # for float64 (dtype.itemsize==8), `np.random.uniform`
        # already excludes the upper limit
        if x.dtype.itemsize < 8:
            jitter_max = np.nextafter(
                jitter_max.astype(x.dtype), -np.inf, dtype=x.dtype
            )
        if uses_dask(x):
            jitter_dist = dsk.random.uniform(
                low=dsk.from_array(jitter_upper),
                high=dsk.from_array(jitter_max),
                size=x.shape,
                chunks=x.chunks,
            )
        else:
            jitter_dist = np.random.uniform(
                low=jitter_upper, high=jitter_max, size=x.shape
            )
        out = out.where(~((x >= jitter_upper) & notnull), jitter_dist.astype(x.dtype))

    copy_all_attrs(out, x)  # copy attrs and same units
    return out


@update_xsdba_history
@harmonize_units(["data", "norm"])
def normalize(
    data: xr.DataArray,
    norm: xr.DataArray | None = None,
    *,
    group: Grouper | str,
    kind: str = ADDITIVE,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Normalize an array by removing its mean.

    Normalization if performed group-wise and according to `kind`.

    Parameters
    ----------
    data : xr.DataArray
        The variable to normalize.
    norm : xr.DataArray, optional
        If present, it is used instead of computing the norm again.
    group : str or Grouper
        Grouping information. See :py:class:`xsdba.base.Grouper` for details..
    kind : {'+', '*'}
        If `kind` is "+", the mean is subtracted from the mean and if it is '*', it is divided from the data.

    Returns
    -------
    xr.DataArray
        Groupwise anomaly.
    norm : xr.DataArray
        Mean over each group.
    """
    ds = xr.Dataset({"data": data})

    if norm is not None:
        ds = ds.assign(norm=norm)

    out = _normalize(ds, group=group, kind=kind)
    copy_all_attrs(out, ds)
    out.data.attrs.update(data.attrs)
    out.norm.attrs["units"] = data.attrs["units"]
    return out.data.rename(data.name), out.norm


def uniform_noise_like(
    da: xr.DataArray, low: float = 1e-6, high: float = 1e-3
) -> xr.DataArray:
    """
    Return a uniform noise array of the same shape as da.

    Noise is uniformly distributed between low and high.
    Alternative method to `jitter_under_thresh` for avoiding zeroes.
    """
    mod: types.ModuleType
    kw: dict
    if uses_dask(da):
        mod = dsk
        kw = {"chunks": da.chunks}
    else:
        mod = np
        kw = {}

    return da.copy(
        data=(high - low) * mod.random.random_sample(size=da.shape, **kw) + low
    )


@update_xsdba_history
def standardize(
    da: xr.DataArray,
    mean: xr.DataArray | None = None,
    std: xr.DataArray | None = None,
    dim: str = "time",
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray, xr.DataArray]:
    """
    Standardize a DataArray by centering its mean and scaling it by its standard deviation.

    Either of both of mean and std can be provided if need be.

    Returns
    -------
    out : xr.DataArray or xr.Dataset
        Standardized data.
    mean : xr.DataArray
        Mean.
    std : xr.DataArray
        Standard Deviation.
    """
    if mean is None:
        mean = da.mean(dim, keep_attrs=True)
    if std is None:
        std = da.std(dim, keep_attrs=True)
    out = (da - mean) / std
    copy_all_attrs(out, da)
    return out, mean, std


@update_xsdba_history
def unstandardize(da: xr.DataArray, mean: xr.DataArray, std: xr.DataArray):
    """Rescale a standardized array by performing the inverse operation of `standardize`."""
    out = (std * da) + mean
    copy_all_attrs(out, da)
    return out


@update_xsdba_history
def reordering(ref: xr.DataArray, sim: xr.DataArray, group: str = "time") -> xr.Dataset:
    """
    Reorder data in `sim` following the order of ref.

    The rank structure of `ref` is used to reorder the elements of `sim` along dimension "time", optionally doing the
    operation group-wise.

    Parameters
    ----------
    ref : xr.DataArray
        Array whose rank order sim should replicate.
    sim : xr.DataArray
        Array to reorder.
    group : str
        Grouping information. See :py:class:`xsdba.base.Grouper` for details.

    Returns
    -------
    xr.Dataset
        Sim reordered according to ref's rank order.

    References
    ----------
    :cite:cts:`cannon_multivariate_2018`.
    """
    ds = xr.Dataset({"sim": sim, "ref": ref})
    out: xr.Dataset = _reordering(ds, group=group).reordered
    copy_all_attrs(out, sim)
    return out


@update_xsdba_history
def escore(
    tgt: xr.DataArray,
    sim: xr.DataArray,
    dims: Sequence[str] = ("variables", "time"),
    N: int = 0,
    scale: bool = False,
) -> xr.DataArray:
    r"""
    Energy score, or energy dissimilarity metric, based on :cite:t:`szekely_testing_2004` and :cite:t:`cannon_multivariate_2018`.

    Parameters
    ----------
    tgt: xr.DataArray
        Target observations.
    sim: xr.DataArray
        Candidate observations. Must have the same dimensions as `tgt`.
    dims: sequence of 2 strings
        The name of the dimensions along which the variables and observation points are listed.
        `tgt` and `sim` can have different length along the second one, but must be equal along the first one.
        The result will keep all other dimensions.
    N : int
        If larger than 0, the number of observations to use in the score computation. The points are taken
        evenly distributed along `obs_dim`.
    scale : bool
        Whether to scale the data before computing the score. If True, both arrays as scaled according
        to the mean and standard deviation of `tgt` along `obs_dim`. (std computed with `ddof=1` and both
        statistics excluding NaN values).

    Returns
    -------
    xr.DataArray
        Return e-score with dimensions not in `dims`.

    Notes
    -----
    Explanation adapted from the "energy" R package documentation.
    The e-distance between two clusters :math:`C_i`, :math:`C_j` (tgt and sim) of size :math:`n_i,n_j`
    proposed by :cite:t:`szekely_testing_2004` is defined by:

    .. math::

        e(C_i,C_j) = \frac{1}{2}\frac{n_i n_j}{n_i + n_j} \left[2 M_{ij} − M_{ii} − M_{jj}\right]

    where

    .. math::

        M_{ij} = \frac{1}{n_i n_j} \sum_{p = 1}^{n_i} \sum_{q = 1}^{n_j} \left\Vert X_{ip} − X{jq} \right\Vert.

    :math:`\Vert\cdot\Vert` denotes Euclidean norm, :math:`X_{ip}` denotes the p-th observation in the i-th cluster.

    The input scaling and the factor :math:`\frac{1}{2}` in the first equation are additions of
    :cite:t:`cannon_multivariate_2018` to the metric. With that factor, the test becomes identical to the one
    defined by :cite:t:`baringhaus_new_2004`.
    This version is tested against values taken from Alex Cannon's MBC R package :cite:p:`cannon_mbc_2020`.

    References
    ----------
    :cite:cts:`baringhaus_new_2004,cannon_multivariate_2018,cannon_mbc_2020,szekely_testing_2004`.
    """
    pts_dim, obs_dim = dims

    if N > 0:
        # If N non-zero we only take around N points, evenly distributed
        sim_step = int(np.ceil(sim[obs_dim].size / N))
        sim = sim.isel({obs_dim: slice(None, None, sim_step)})
        tgt_step = int(np.ceil(tgt[obs_dim].size / N))
        tgt = tgt.isel({obs_dim: slice(None, None, tgt_step)})

    if scale:
        tgt, avg, std = standardize(tgt)
        sim, _, _ = standardize(sim, avg, std)

    # The dimension renaming is to allow different coordinates.
    # Otherwise, apply_ufunc tries to align both obs_dim together.
    new_dim = get_temp_dimname(tgt.dims, obs_dim)
    sim = sim.rename({obs_dim: new_dim})
    out: xr.DataArray = xr.apply_ufunc(
        _escore,
        tgt,
        sim,
        input_core_dims=[[pts_dim, obs_dim], [pts_dim, new_dim]],
        output_dtypes=[sim.dtype],
        dask="parallelized",
        vectorize=True,
    )

    out.name = "escores"
    out = out.assign_attrs(
        {
            "long_name": "Energy dissimilarity metric",
            "description": f"Escores computed from {N or 'all'} points.",
            "references": "Székely, G. J. and Rizzo, M. L. (2004) Testing for Equal Distributions in High Dimension, InterStat, November (5)",
        }
    )
    return out


def _get_number_of_elements_by_year(time):
    """
    Get the number of elements in time in a year by inferring its sampling frequency.

    Only calendar with uniform year lengths are supported : 360_day, noleap, all_leap.
    """
    mult, freq, _, _ = parse_offset(xr.infer_freq(time))
    days_in_year = time.dt.days_in_year.max()
    elements_in_year = {"Q": 4, "M": 12, "D": days_in_year, "h": days_in_year * 24}
    N_in_year = elements_in_year.get(freq, 1) / mult
    if N_in_year % 1 != 0:
        raise ValueError(
            f"Sampling frequency of the data must be Q, M, D or h and evenly divide a year (got {mult}{freq})."
        )

    return int(N_in_year)


@update_xsdba_history
@harmonize_units(["data", "lower_bound", "upper_bound"])
def to_additive_space(
    data: xr.DataArray,
    lower_bound: str,
    upper_bound: str | None = None,
    trans: str = "log",
    clip_next_to_bounds: bool = False,
):
    r"""
    Transform a non-additive variable into an additive space by the means of a log or logit transformation.

    Based on :cite:t:`alavoine_distinct_2022`.

    Parameters
    ----------
    data : xr.DataArray
        A variable that can't usually be bias-adjusted by additive methods.
    lower_bound : str
        The smallest physical value of the variable, excluded, as a Quantity string.
        The data should only have values strictly larger than this bound.
    upper_bound : str, optional
        The largest physical value of the variable, excluded, as a Quantity string.
        Only relevant for the logit transformation.
        The data should only have values strictly smaller than this bound.
    trans : {'log', 'logit'}
        The transformation to use. See notes.
    clip_next_to_bounds : bool
        If `True`, values are clipped to ensure `data > lower_bound`  and `data < upper_bound` (if specified).
        Defaults to `False`. `data` must be in the range [lower_bound, upper_bound], else an error is thrown.

    See Also
    --------
    from_additive_space : For the inverse transformation.
    jitter_under_thresh : Remove values exactly equal to the lower bound.
    jitter_over_thresh : Remove values exactly equal to the upper bound.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, this applies a transformation to a space where
    additive methods are sensible. Given :math:`X` the variable, :math:`b_-` the lower physical bound of that variable
    and :math:`b_+` the upper physical bound, two transformations are currently implemented to get :math:`Y`,
    the additive-ready variable. :math:`\ln` is the natural logarithm.

    - `log`

        .. math::

            Y = \ln\left( X - b_- \right)

        Usually used for variables with only a lower bound, like precipitation (`pr`,  `prsn`, etc)
        and daily temperature range (`dtr`). Both have a lower bound of 0.

    - `logit`

        .. math::

            X' = (X - b_-) / (b_+ - b_-)
            Y = \ln\left(\frac{X'}{1 - X'} \right)

        Usually used for variables with both a lower and a upper bound, like relative and specific humidity,
        cloud cover fraction, etc.

    This will thus produce `Infinity` and `NaN` values where :math:`X == b_-` or :math:`X == b_+`.
    We recommend using :py:func:`jitter_under_thresh` and :py:func:`jitter_over_thresh` to remove those issues.

    If :math:`X \in [b_-, b_+]`, `clip_next_to_bounds` can be set to `True`, and boundary values will be slightly changed (with the smallest float32
    increment) to ensure that :math:`X \in ]b_-, b_+[`.

    References
    ----------
    :cite:cts:`alavoine_distinct_2022`.
    """
    lower_bound_array = np.array(lower_bound).astype(float)
    if upper_bound is not None:
        upper_bound_array = np.array(upper_bound).astype(float)

    # clip bounds
    if clip_next_to_bounds:
        if (data < lower_bound).any() or (data > (upper_bound or np.nan)).any():
            raise ValueError(
                "The input dataset contains values outside of the range [lower_bound, upper_bound] "
                "(with upper_bound given by infinity if it is not specified). Clipping the values to the range "
                "]lower_bound, upper_bound[ is not allowed in this case. Check if the bounds are taken appropriately or "
                "if your input dataset has unphysical values."
            )

        low = np.nextafter(lower_bound, np.inf, dtype=np.float32)
        high = (
            None
            if upper_bound is None
            else np.nextafter(upper_bound, -np.inf, dtype=np.float32)
        )
        data = data.clip(low, high)

    with xr.set_options(keep_attrs=True), np.errstate(divide="ignore"):
        if trans == "log":
            out = cast(xr.DataArray, np.log(data - lower_bound_array))
        elif trans == "logit" and upper_bound is not None:
            data_prime = (data - lower_bound_array) / (
                upper_bound_array - lower_bound_array  # pylint: disable=E0606
            )
            out = cast(xr.DataArray, np.log(data_prime / (1 - data_prime)))
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Attributes to remember all this.
    out = out.assign_attrs(xsdba_transform=trans)
    out = out.assign_attrs(xsdba_transform_lower=lower_bound_array)
    if upper_bound is not None:
        out = out.assign_attrs(xsdba_transform_upper=upper_bound_array)
    if "units" in out.attrs:
        out = out.assign_attrs(xsdba_transform_units=out.attrs.pop("units"))
        out = out.assign_attrs(units="")
    return out


@update_xsdba_history
def from_additive_space(
    data: xr.DataArray,
    lower_bound: str | None = None,
    upper_bound: str | None = None,
    trans: str | None = None,
    units: str | None = None,
):
    r"""
    Transform back to the physical space a variable that was transformed with `to_additive_space`.

    Based on :cite:t:`alavoine_distinct_2022`.
    If parameters are not present on the attributes of the data, they must be all given are arguments.

    Parameters
    ----------
    data : xr.DataArray
        A variable that was transformed by :py:func:`to_additive_space`.
    lower_bound : str, optional
        The smallest physical value of the variable, as a Quantity string.
        The final data will have no value smaller or equal to this bound.
        If None (default), the `xsdba_transform_lower` attribute is looked up on `data`.
    upper_bound : str, optional
        The largest physical value of the variable, as a Quantity string.
        Only relevant for the logit transformation.
        The final data will have no value larger or equal to this bound.
        If None (default), the `xsdba_transform_upper` attribute is looked up on `data`.
    trans : {'log', 'logit'}, optional
        The transformation to use. See notes.
        If None (the default), the `xsdba_transform` attribute is looked up on `data`.
    units : str, optional
        The units of the data before transformation to the additive space.
        If None (the default), the `xsdba_transform_units` attribute is looked up on `data`.

    Returns
    -------
    xr.DataArray
        The physical variable. Attributes are conserved, even if some might be incorrect.
        Except units which are taken from `xsdba_transform_units` if available.
        All `xsdba_transform*` attributes are deleted.

    See Also
    --------
    to_additive_space : For the original transformation.

    Notes
    -----
    Given a variable that is not usable in an additive adjustment, :py:func:`to_additive_space` applied a transformation
    to a space where additive methods are sensible. Given :math:`Y` the transformed variable, :math:`b_-` the
    lower physical bound of that variable and :math:`b_+` the upper physical bound, two back-transformations are
    currently implemented to get :math:`X`, the physical variable.

    - `log`

        .. math::

            X = e^{Y} + b_-

    - `logit`

        .. math::

            X' = \frac{1}{1 + e^{-Y}}
            X = X * (b_+ - b_-) + b_-

    References
    ----------
    :cite:cts:`alavoine_distinct_2022`.
    """
    if trans is None and lower_bound is None and units is None:
        try:
            trans = data.attrs["xsdba_transform"]
            units = data.attrs["xsdba_transform_units"]
            lower_bound_array = np.array(data.attrs["xsdba_transform_lower"]).astype(
                float
            )
            if trans == "logit":
                upper_bound_array = np.array(
                    data.attrs["xsdba_transform_upper"]
                ).astype(float)
        except KeyError as err:
            raise ValueError(
                f"Attribute {err!s} must be present on the input data "
                "or all parameters must be given as arguments."
            ) from err
    elif (
        trans is not None
        and lower_bound is not None
        and units is not None
        and (upper_bound is not None or trans == "log")
    ):
        # FIXME: convert_units_to is causing issues since it can't handle all variations of Quantified here
        lower_bound_array = np.array(convert_units_to(lower_bound, units)).astype(float)
        if trans == "logit":
            upper_bound_array = np.array(convert_units_to(upper_bound, units)).astype(
                float
            )
    else:
        raise ValueError(
            "Parameters missing. Either all parameters are given as attributes of data, "
            "or all of them are given as input arguments."
        )

    with xr.set_options(keep_attrs=True):
        if trans == "log":
            out = np.exp(data) + lower_bound_array
        elif trans == "logit":
            out_prime = 1 / (1 + np.exp(-data))
            out = (
                out_prime
                * (upper_bound_array - lower_bound_array)  # pylint: disable=E0606
                + lower_bound_array
            )
        else:
            raise NotImplementedError("`trans` must be one of 'log' or 'logit'.")

    # Remove unneeded attributes, put correct units back.
    out.attrs.pop("xsdba_transform", None)
    out.attrs.pop("xsdba_transform_lower", None)
    out.attrs.pop("xsdba_transform_upper", None)
    out.attrs.pop("xsdba_transform_units", None)
    out = out.assign_attrs(units=units)
    return out


def stack_variables(ds: xr.Dataset, rechunk: bool = True, dim: str = "multivar"):
    """
    Stack different variables of a dataset into a single DataArray with a new "variables" dimension.

    Variable attributes are all added as lists of attributes to the new coordinate, prefixed with "_".
    Variables are concatenated in the new dimension in alphabetical order, to ensure
    coherent behaviour with different datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    rechunk : bool
        If True (default), dask arrays are rechunked with `variables : -1`.
    dim : str
        Name of dimension along which variables are indexed.

    Returns
    -------
    xr.DataArray
        The transformed variable. Attributes are conserved, even if some might be incorrect, except for units,
        which are replaced with `""`. Old units are stored in `xsdba_transformation_units`.
        A `xsdba_transform` attribute is added, set to the transformation method. `xsdba_transform_lower` and
        `xsdba_transform_upper` are also set if the requested bounds are different from the defaults.

        Array with variables stacked along `dim` dimension. Units are set to "".
    """
    # Store original arrays' attributes
    attrs: dict = {}
    # sort to have coherent order with different datasets
    data_vars = sorted(ds.data_vars.items(), key=lambda e: e[0])
    nvar = len(data_vars)
    for i, (_nm, var) in enumerate(data_vars):
        for name, attr in var.attrs.items():
            attrs.setdefault(f"_{name}", [None] * nvar)[i] = attr

    # Special key used for later `unstacking`
    attrs["is_variables"] = True
    var_crd = xr.DataArray([nm for nm, vr in data_vars], dims=(dim,), name=dim)

    da = xr.concat([vr for nm, vr in data_vars], var_crd, combine_attrs="drop")

    if uses_dask(da) and rechunk:
        da = da.chunk({dim: -1})

    da.attrs.update(ds.attrs)
    da.attrs["units"] = ""
    da[dim].attrs.update(attrs)
    return da.rename("multivariate")


def unstack_variables(da: xr.DataArray, dim: str | None = None) -> xr.Dataset:
    """
    Unstack a DataArray created by `stack_variables` to a dataset.

    Parameters
    ----------
    da : xr.DataArray
        Array holding different variables along `dim` dimension.
    dim : str, optional
        Name of dimension along which the variables are stacked.
        If not specified (default), `dim` is inferred from attributes of the coordinate.

    Returns
    -------
    xr.Dataset
        Dataset holding each variable in an individual DataArray.
    """
    if dim is None:
        for _dim, _crd in da.coords.items():
            if _crd.attrs.get("is_variables"):
                dim = str(_dim)
                break
        else:
            raise ValueError("No variable coordinate found, were attributes removed?")

    ds = xr.Dataset(
        {name.item(): da.sel({dim: name.item()}, drop=True) for name in da[dim]},
        attrs=da.attrs,
    )
    del ds.attrs["units"]

    # Reset attributes
    for name, attr_list in da[dim].attrs.items():
        if not name.startswith("_"):
            continue
        for attr, var in zip(attr_list, da[dim], strict=False):
            if attr is not None:
                ds[var.item()].attrs[name[1:]] = attr

    return ds


def grouped_time_indexes(times, group):
    """
    Time indexes for every group blocks

    Time indexes can be used to implement a pseudo-"numpy.groupies" approach to grouping.

    Parameters
    ----------
    times : xr.DataArray
        Time dimension in the dataset of interest.
    group : str or Grouper
        Grouping information, see base.Grouper.

    Returns
    -------
    g_idxs : xr.DataArray
        Time indexes of the blocks (only using `group.name` and not `group.window`).
    gw_idxs : xr.DataArray
        Time indexes of the blocks (built with a rolling window of `group.window` if any).
    """

    def _get_group_complement(da, group):
        # complement of "dayofyear": "year", etc.
        gr = group if isinstance(group, str) else group.name
        if gr == "time.dayofyear":
            return da.time.dt.year
        if gr == "time.month":
            return da.time.dt.strftime("%Y-%d")
        raise NotImplementedError(f"Grouping {gr} not implemented.")

    # does not work with group == "time.month"
    group = group if isinstance(group, Grouper) else Grouper(group)
    gr, win = group.name, group.window
    # get time indices (0,1,2,...) for each block
    timeind = xr.DataArray(np.arange(times.size), coords={"time": times})
    win_dim0, win_dim = (
        get_temp_dimname(timeind.dims, lab) for lab in ["win_dim0", "win_dim"]
    )
    if gr == "time.dayofyear":
        # time indices for each block with window = 1
        g_idxs = timeind.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr)).rename(
                {"time": "year"}
            )
        )
        # time indices for each block with general window
        da = timeind.rolling(time=win, center=True).construct(window_dim=win_dim0)
        gw_idxs = da.groupby(gr).apply(
            lambda da: da.assign_coords(time=_get_group_complement(da, gr)).stack(
                {win_dim: ["time", win_dim0]}
            )
        )
        gw_idxs = gw_idxs.transpose(..., win_dim)
    elif gr == "time":
        gw_idxs = timeind.rename({"time": win_dim}).expand_dims({win_dim0: [-1]})
        g_idxs = gw_idxs.copy()
    # TODO : Implement a proper Grouper treatment
    # This would normally not be allowed with sdba.Grouper.
    # A proper implementation in Grouper may be given in the future, but here is the implementation
    # that I used for a project
    elif gr == "5D":
        if win % 2 == 0:
            raise ValueError(
                f"Group 5D only works with an odd window, got `window` = {win}"
            )

        gr_dim = "five_days"
        imin, imax = 0, times.size - 1

        def _get_idxs(win):
            block0 = np.concatenate(
                [
                    np.arange(5) + iwin * 5 + iyear * 365
                    for iyear in range(len(set(times.dt.year.values)))
                    for iwin in range(-(win - 1) // 2, (win - 1) // 2 + 1)
                ]
            )
            base = xr.DataArray(
                block0, dims=[win_dim], coords={win_dim: np.arange(len(block0))}
            )
            idxs = xr.concat(
                [(base + i * 5).expand_dims({gr_dim: [i]}) for i in range(365 // 5)],
                dim=gr_dim,
            )
            return idxs.where((idxs >= imin) & (idxs <= imax), -1)

        gw_idxs, g_idxs = _get_idxs(win), _get_idxs(1)

    else:
        raise NotImplementedError(f"Grouping {gr} not implemented.")
    gw_idxs.attrs["group"] = (gr, win)
    gw_idxs.attrs["time_dim"] = win_dim
    gw_idxs.attrs["group_dim"] = [d for d in g_idxs.dims if d != win_dim][0]
    return g_idxs, gw_idxs


# spectral utils
def _make_mask(template, cond_vals):
    """
    Create a mask from a series of conditions.

    Parameters
    ----------
    template: xr.DataArray
        Array with the dimensions to be filtered.
    cond_vals: tuple
        The list of (condition, value) pairs applied to create the mask.

    Returns
    -------
    xarray.DataArray, [unitless]
        Mask based on the condition values.

    Notes
    -----
    Conditions are allowed to have any values. The idea is to create a
    soft mask with values between 0 and 1, which allows to implement smooth
    filters.
    """
    mask = xr.full_like(template, 1)
    for cond, val in cond_vals:
        mask = mask.where(cond == False, val)
    return mask


def cos2_mask_func(da, low, high):
    """
    Create a mask applied Fourier coefficient with a cosine squared filter
    between given thresholds .

    Parameters
    ----------
    da : np.ndarray
        Fourier coefficients.
    low : float
        Low frequency threshold (Long wavelength).
    high : float
        High frequency threshold (Short wavelength).

    Returns
    -------
    np.ndarray
        A mask used to apply a low-pass filter.

    Notes
    -----
    The mask is 1 below `low`, 0 above `high`, and transitions from 1 to 0
    following a cosine profile between `low` and `high`.
    """
    cond_vals = [
        # This first condition could be remove, the mask starts as an array of 1's
        (da < low, 1),
        (da > high, 0),
        (
            (da >= low) & (da <= high),
            np.cos(((da - low) / (high - low)) * (np.pi / 2)) ** 2,
        ),
    ]
    return _make_mask(da, cond_vals)


def _normalized_radial_wavenumber(da, dims):
    r"""
    Compute a normalized radial wavenumber.

    Parameters
    ----------
    da: xr.DataArray or xr.Dataset
        Input field to be transformed in reciprocal space.
    dims: list[str]
        Dimensions on which to perform the Discrete Cosine Transform.

    Returns
    -------
    xr.DataArray
        Normalized radial wavenumber.

    Notes
    -----
    The normalized radial wavenumber is obtained at each point of the lattice in reciprocal space following
    the Fourier transformation along dimensions `dims`. For example, if :math:`i,j` are the wavenumbers of
    the discrete cosine transform along longitude and latitude, respectively, and :math:`N_i, N_j` are
    the total number of grid points along longitude and latitude, then the normalized wavenumber
    :math:`\alpha` is given by:

    .. math::

        \alpha = \sqrt{\left(\frac{i}{N_i}\right)^2 + \left(\frac{j}{N_j}\right)^2}

    Each coordinate point takes integer values.

    References
    ----------
    :cite:cts:`denis_spectral_2002`
    """
    # Replace lat/lon coordinates with integers (wavenumbers in reciprocal space)
    ds_dims = da[dims] if isinstance(da, xr.Dataset) else (da.to_dataset())[dims]
    da0 = xr.Dataset(coords={d: range(sh) for d, sh in ds_dims.dims.items()})
    # Radial distance in Fourier space
    alpha = sum([da0[d] ** 2 / da0[d].size ** 2 for d in da0.dims]) ** 0.5
    alpha = alpha.assign_coords({d: ds_dims[d] for d in ds_dims.dims}).rename("alpha")
    alpha = alpha.assign_attrs(
        {
            "units": "",
            "standard_name": "normalized_wavenumber",
            "long_name": "Normalized wavenumber",
        }
    )
    return alpha


def _dctn_filter(arr, mask):
    """Multiply the Fourier (Discrete cosine transform) coefficients by a filter which takes values between 0 and 1."""
    coeffs = (dctn(arr, norm="ortho"),)
    return idctn(coeffs * mask, norm="ortho")


def spectral_filter(
    da,
    lam_long,
    lam_short,
    dims=["lat", "lon"],
    delta=None,
    mask_func=cos2_mask_func,
    alpha_low_high=None,
):
    """
    Filter coefficients of a Discrete Cosine Fourier transform between given thresholds and invert back to real space.

    Parameters
    ----------
    da : xr.DataArray
        Input physical field.
    lam_long : str | optional
        Long wavelength threshold.
    lam_short : str | optional
        Short wavelength threshold.
    dims: list
        Dimensions on which to perform the spectral filter.
    delta: str, Optional
        Nominal resolution of the grid. A string with units, e.g. `delta=="55.5 km"`. This converts `alpha` to `wavelength`.
        If `delta` is not specified, a dimension named `rlat` or `lat` is expected to be in `da` and will be used to
        deduce an appropriate length scale.
    mask_func: function
        Function used to create the mask. Default is `cos2_mask_func`, which applies a cosine squared filter
        to Fourier coefficients in momentum space.
    alpha_low_high : tuple[float,float] | optional
        Low and high frequencies threshold (Long and short wavelength) for the
        radial normalized wavenumber (`alpha`). It should be numbers between 0 and 1.

    Returns
    -------
    xr.DataArray
        Filtered physical field.

    Notes
    -----
    * If `delta` is specified, the normalized wavenumber `alpha` will be converted to a `wavelength`.
    * If the input field contains any `nan`, the output will be all `nan` values.

    References
    ----------
    :cite:cts:`denis_spectral_2002`
    """
    dims = [dims] if isinstance(dims, str) else dims

    if isinstance(da, xr.Dataset):
        out = da.copy()
        for v in da.data_vars:
            out[v] = spectral_filter(da[v], lam_long, lam_short, dims, delta=delta)
        return out.assign_attrs(da.attrs)

    if delta is None and alpha_low_high is None:
        if "rlat" in da.dims:
            lat = da.rlat
        else:
            lat = da.lat
        # is this a good approximation?
        delta = f"{(lat[1] - lat[0]).values.item() * 111} km"
    if alpha_low_high is None and None in set(lam_long, lam_short):
        raise ValueError(
            "`lam_long` or `lam_short` can only be None if `alpha_low_high` is provided."
        )
    if alpha_low_high is not None:
        alpha_low, alpha_high = alpha_low_high
    else:
        alpha_low = wavelength_to_normalized_wavenumber(lam_long, delta=delta)
        alpha_high = wavelength_to_normalized_wavenumber(lam_short, delta=delta)
    alpha = _normalized_radial_wavenumber(da, dims)
    mask = mask_func(alpha, alpha_low, alpha_high)
    out = xr.apply_ufunc(
        _dctn_filter,
        da,
        mask,
        input_core_dims=[dims, dims],
        output_core_dims=[dims],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    filter_bounds = alpha_low_high or (lam_long, lam_short)
    out = out.assign_attrs(
        {
            "filter_bounds": filter_bounds,
            "mask_func": mask_func.__name__,
        }
    ).transpose(
        *da.dims
    )  # reimplement original order, if needed
    return out
