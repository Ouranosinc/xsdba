# pylint: disable=missing-kwoa
"""
# noqa: SS01
Properties Submodule
====================
SDBA diagnostic tests are made up of statistical properties and measures. Properties are calculated on both simulation
and reference datasets. They collapse the time dimension to one value.

This framework for the diagnostic tests was inspired by the `VALUE <http://www.value-cost.eu/>`_ project.
Statistical Properties is the xclim term for 'indices' in the VALUE project.

This module depends on `xclim`. Run `pip install xsdba['extras']` to install it.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr
import xclim.indices.run_length as rl
from scipy import stats
from scipy.fft import dctn
from statsmodels.tsa import stattools
from xclim.core.indicator import Indicator, base_registry
from xclim.indices.generic import compare, select_resample_op
from xclim.indices.stats import fit, parametric_quantile

from xsdba.base import Grouper, map_groups, parse_group, uses_dask
from xsdba.nbutils import _pairwise_haversine_and_bins
from xsdba.processing import _normalized_radial_wavenumber
from xsdba.units import (
    convert_units_to,
    infer_sampling_units,
    normalized_wavenumber_to_wavelength,
    pint2cfattrs,
    str2pint,
    units,
    units2pint,
)
from xsdba.utils import _pairwise_spearman, copy_all_attrs


class StatisticalProperty(Indicator):
    """
    Base indicator class for statistical properties used for validating bias-adjusted outputs.

    Statistical properties reduce the time dimension, sometimes adding a grouping dimension
    according to the passed value of `group` (e.g.: group='time.month' means the loss of the
    time dimension and the addition of a month one).

    Statistical properties are generally unit-generic. To use those indicator in a workflow, it
    is recommended to wrap them with a virtual submodule, creating one specific indicator for
    each variable input (or at least for each possible dimensionality).

    Statistical properties may restrict the sampling frequency of the input, they usually take in a
    single variable (named "da" in unit-generic instances).
    """

    aspect = None
    """The aspect the statistical property studies: marginal, temporal, multivariate or spatial."""

    measure = "xsdba.measures.BIAS"
    """The default measure to use when comparing the properties of two datasets.
    This gives the registry id. See :py:meth:`get_measure`."""

    allowed_groups = None
    """A list of allowed groupings. A subset of dayofyear, week, month, season or group.
    The latter stands for no temporal grouping."""

    realm = "generic"

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        if "group" not in parameters:
            raise ValueError(
                f"{cls.__name__} require a 'group' argument, use the base Indicator"
                " class if your computation doesn't perform any regrouping."
            )
        return super()._ensure_correct_parameters(parameters)

    def _preprocess_and_checks(self, das, params):
        """Perform parent's checks and also check if group is allowed."""
        das, params = super()._preprocess_and_checks(das, params)

        # Convert grouping and check if allowed:
        if isinstance(params["group"], str):
            params["group"] = Grouper(params["group"])

        if self.allowed_groups is not None:
            if params["group"].prop not in self.allowed_groups:
                raise ValueError(
                    f"Grouping period {params['group'].prop_name} is not allowed for property "
                    f"{self.identifier} (needs something in "
                    f"{map(lambda g: '<dim>.' + g.replace('group', ''), self.allowed_groups)})."
                )

        return das, params

    def _postprocess(self, outs, das, params):
        """Squeeze `group` dim if needed."""
        outs = super()._postprocess(outs, das, params)

        for ii, out in enumerate(outs):
            if "group" in out.dims:
                outs[ii] = out.squeeze("group", drop=True)

        return outs

    def get_measure(self):
        """Get the statistical measure indicator that is best used with this statistical property."""
        from xclim.core.indicator import registry

        return registry[self.measure].get_instance()


base_registry["StatisticalProperty"] = StatisticalProperty


@parse_group
def _mean(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """
    Mean.

    Mean over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the temporal average is performed separately for each month.

    Returns
    -------
    xr.DataArray, [same as input]
      Mean of the variable.
    """
    u = da.units
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.mean(dim=group.dim)
    return out.assign_attrs(units=u)


mean = StatisticalProperty(
    identifier="mean",
    aspect="marginal",
    cell_methods="time: mean",
    compute=_mean,
)


@parse_group
def _var(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """
    Variance.

    Variance of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the variance is performed separately for each month.

    Returns
    -------
    xr.DataArray, [square of the input units]
        Variance of the variable.
    """
    u = da.units
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.var(dim=group.dim)
    out.attrs["units"] = f"{(units(u) ** 2).units:cf}"
    return out


var = StatisticalProperty(
    identifier="var",
    aspect="marginal",
    cell_methods="time: var",
    compute=_var,
    measure="xsdba.measures.RATIO",
)


@parse_group
def _std(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """
    Standard Deviation.

    Standard deviation of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the standard deviation is performed separately for each month.

    Returns
    -------
    xr.DataArray,
        Standard deviation of the variable.
    """
    u = da.units
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.std(dim=group.dim)
    out.attrs["units"] = u
    return out


std = StatisticalProperty(
    identifier="std",
    aspect="marginal",
    cell_methods="time: std",
    compute=_std,
    measure="xsdba.measures.RATIO",
)


@parse_group
def _skewness(da: xr.DataArray, *, group: str | Grouper = "time") -> xr.DataArray:
    """
    Skewness.

    Skewness of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the skewness is performed separately for each month.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Skewness of the variable.

    See Also
    --------
    scipy.stats.skew
    """
    if group.prop != "group":
        da = da.groupby(group.name)
    out = xr.apply_ufunc(
        stats.skew,
        da,
        input_core_dims=[[group.dim]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs["units"] = ""
    return out


skewness = StatisticalProperty(
    identifier="skewness", aspect="marginal", compute=_skewness, units=""
)


@parse_group
def _quantile(
    da: xr.DataArray, *, q: float = 0.98, group: str | Grouper = "time"
) -> xr.DataArray:
    """
    Quantile.

    Returns the quantile q of the distribution of the variable over all years at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    q : float
        Quantile to be calculated. Should be between 0 and 1.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the quantile is computed separately for each month.

    Returns
    -------
    xr.DataArray, [same as input]
        Quantile {q} of the variable.
    """
    u = da.units
    if group.prop != "group":
        da = da.groupby(group.name)
    out = da.quantile(q, dim=group.dim, keep_attrs=True).drop_vars("quantile")
    return out.assign_attrs(units=u)


quantile = StatisticalProperty(
    identifier="quantile", aspect="marginal", compute=_quantile
)


def _spell_length_distribution(
    da: xr.DataArray,
    *,
    method: str = "amount",
    op: str = ">=",
    thresh: str = "1 mm d-1",
    window: int = 1,
    stat: str = "mean",
    stat_resample: str | None = None,
    group: str | Grouper = "time",
    resample_before_rl: bool = True,
) -> xr.DataArray:
    """
    Spell length distribution.

    Statistic of spell length distribution when the variable respects a condition (defined by an operation, a method and
     a threshold).

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    method : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op} threshold.
    thresh : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    window : int
        Number of consecutive days respecting the constraint in order to begin a spell.
        Default is 1, which is equivalent to `_threshold_count`.
    stat : {'mean', 'sum', 'max','min'}
        Statistics to apply to the remaining time dimension after resampling (e.g. Jan 1980-2010)
    stat_resample : {'mean', 'sum', 'max','min'}, optional
        Statistics to apply to the resampled input at the {group} (e.g. 1-31 Jan 1980).
        If `None`, the same method as `stat` will be used.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the spell lengths are computed separately for each month.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xr.DataArray, [units of the sampling frequency]
        {stat} of spell length distribution when the variable is {op} the {method} {thresh} for {window} consecutive day(s).
    """
    group = group if isinstance(group, Grouper) else Grouper(group)

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _spell_stats(
        ds,
        *,
        dim,
        method,
        thresh,
        window,
        op,
        freq,
        resample_before_rl,
        stat,
        stat_resample,
    ):
        # PB: This prevents an import error in the distributed dask scheduler, but I don't know why.
        import xarray.core.resample_cftime  # noqa: F401, pylint: disable=unused-import

        da = ds.data
        mask = ~(da.isel({dim: 0}).isnull()).drop_vars(
            dim
        )  # mask of the ocean with NaNs
        if method == "quantile":
            thresh = da.quantile(thresh, dim=dim).drop_vars("quantile")

        cond = compare(da, op, thresh)
        out = rl.resample_and_rl(
            cond,
            resample_before_rl,
            rl.rle_statistics,
            reducer=stat_resample,
            window=window,
            dim=dim,
            freq=freq,
        )
        out = getattr(out, stat)(dim=dim)
        out = out.where(mask)
        return out.rename("out").to_dataset()

    # threshold is an amount that will be converted to the right units
    if method == "amount":
        thresh = convert_units_to(thresh, da)
    elif method != "quantile":
        raise ValueError(
            f"{method} is not a valid method. Choose 'amount' or 'quantile'."
        )

    out = _spell_stats(
        da.rename("data").to_dataset(),
        group=group,
        method=method,
        thresh=thresh,
        window=window,
        op=op,
        freq=group.freq,
        resample_before_rl=resample_before_rl,
        stat=stat,
        stat_resample=stat_resample or stat,
    ).out
    # in xclim this was managed by to_agg_units
    # will hard-code this part for now
    m, freq_u_raw = infer_sampling_units(da["time"])
    with xr.set_options(keep_attrs=True):
        out = (out * m).assign_attrs({"units": freq_u_raw})
    return out


spell_length_distribution = StatisticalProperty(
    identifier="spell_length_distribution",
    aspect="temporal",
    compute=_spell_length_distribution,
)


@parse_group
def _threshold_count(
    da: xr.DataArray,
    *,
    method: str = "amount",
    op: str = ">=",
    thresh: str = "1 mm d-1",
    stat: str = "mean",
    stat_resample: str | None = None,
    group: str | Grouper = "time",
) -> xr.DataArray:
    r"""
    Correlation between two variables.

    Spearman or Pearson correlation coefficient between two variables at the time resolution.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    method : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op} threshold.
    thresh : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    stat : {'mean', 'sum', 'max','min'}
        Statistics to apply to the remaining time dimension after resampling (e.g. Jan 1980-2010)
    stat_resample : {'mean', 'sum', 'max','min'}, optional
        Statistics to apply to the resampled input at the {group} (e.g. 1-31 Jan 1980). If `None`, the same method as `stat` will be used.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. For 'time.month', the correlation would be calculated on each month separately,
        but with all the years together.

    Returns
    -------
    xr.DataArray, [dimensionless]
        {stat} number of days when the variable is {op} the {method} {thresh}.

    Notes
    -----
    This corresponds to ``xsdba.properties._spell_length_distribution`` with `window=1`.
    """
    return _spell_length_distribution(
        da,
        method=method,
        op=op,
        thresh=thresh,
        stat=stat,
        stat_resample=stat_resample,
        group=group,
        window=1,
    )


threshold_count = StatisticalProperty(
    identifier="threshold_count", aspect="temporal", compute=_threshold_count
)


@parse_group
def _acf(
    da: xr.DataArray, *, lag: int = 1, group: str | Grouper = "time.season"
) -> xr.DataArray:
    """
    Autocorrelation.

    Autocorrelation with a lag over a time resolution and averaged over all years.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    lag : int
        Lag.
    group : {'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the autocorrelation is calculated over each month separately for all years.
        Then, the autocorrelation for all Jan/Feb/... is averaged over all years, giving 12 outputs for each grid point.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Lag-{lag} autocorrelation of the variable over a {group.prop} and averaged over all years.

    See Also
    --------
    statsmodels.tsa.stattools.acf

    References
    ----------
    :cite:cts:`alavoine_distinct_2022`
    """

    def acf_last(x, nlags):
        """Calculates, like Statsmodels, acf for lag 0 to nlags, but returns only the last one."""
        # As we resample + group, timeseries are quite short and fft=False seems more performant
        out_last = stattools.acf(x, nlags=nlags, fft=False)
        return out_last[-1]

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _acf(ds, *, dim, lag, freq):
        out = xr.apply_ufunc(
            acf_last,
            ds.data.resample({dim: freq}),
            input_core_dims=[[dim]],
            vectorize=True,
            kwargs={"nlags": lag},
        )
        out = out.mean("__resample_dim__")
        return out.rename("out").to_dataset()

    out = _acf(
        da.rename("data").to_dataset(), group=group, lag=lag, freq=group.freq
    ).out
    out.attrs["units"] = ""
    return out


acf = StatisticalProperty(
    identifier="acf",
    aspect="temporal",
    allowed_groups=["season", "month"],
    compute=_acf,
)


# group was kept even though "time" is the only acceptable arg to keep the signature similar to other properties
# @parse_group doesn't work well here because of `window`
def _annual_cycle(
    da: xr.DataArray,
    *,
    stat: str = "absamp",
    window: int = 31,
    group: str | Grouper = "time",
) -> xr.DataArray:
    r"""
    Annual cycle statistics.

    A daily climatology is calculated and optionally smoothed with a (circular) moving average.
    The requested statistic is returned.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    stat : {'absamp','relamp', 'phase', 'min', 'max', 'asymmetry'}
        - 'absamp' is the peak-to-peak amplitude. (max - min). In the same units as the input.
        - 'relamp' is a relative percentage. 100 * (max - min) / mean (Recommended for precipitation). Dimensionless.
        - 'phase' is the day of year of the maximum.
        - 'max' is the maximum. Same units as the input.
        - 'min' is the minimum. Same units as the input.
        - 'asymmetry' is the length of the period going from the minimum to the maximum. In years between 0 and 1.
    window : int
        Size of the window for the moving average filtering. Deactivate this feature by passing window = 1.

    Returns
    -------
    xr.DataArray, [same units as input or dimensionless or time]
        {stat} of the annual cycle.
    """
    group = group if isinstance(group, Grouper) else Grouper(group)
    u = da.units

    ac = da.groupby("time.dayofyear").mean()
    if window > 1:  # smooth the cycle
        # We want the rolling mean to be circular. There's no built-in method to do this in xarray,
        # we'll pad the array and extract the meaningful part.
        ac = (
            ac.pad(dayofyear=(window // 2), mode="wrap")
            .rolling(dayofyear=window, center=True)
            .mean()
            .isel(dayofyear=slice(window // 2, -(window // 2)))
        )
    match stat:
        case "absamp":
            out = ac.max("dayofyear") - ac.min("dayofyear")
            out.attrs.update(pint2cfattrs(units2pint(u), is_difference=True))
        case "relamp":
            out = (
                (ac.max("dayofyear") - ac.min("dayofyear")) * 100 / ac.mean("dayofyear")
            )
            out.attrs["units"] = "%"
        case "phase":
            out = ac.idxmax("dayofyear")
            out.attrs.update(units="", is_dayofyear=np.int32(1))
        case "min":
            out = ac.min("dayofyear")
            out.attrs["units"] = u
        case "max":
            out = ac.max("dayofyear")
            out.attrs["units"] = u
        case "asymmetry":
            out = (ac.idxmax("dayofyear") - ac.idxmin("dayofyear")) % 365 / 365
            out.attrs["units"] = "yr"
        case _:
            raise NotImplementedError(f"{stat} is not a valid annual cycle statistic.")
    return out


annual_cycle_amplitude = StatisticalProperty(
    identifier="annual_cycle_amplitude",
    aspect="temporal",
    compute=_annual_cycle,
    parameters={"stat": "absamp"},
    allowed_groups=["group"],
    cell_methods="time: mean time: range",
)

relative_annual_cycle_amplitude = StatisticalProperty(
    identifier="relative_annual_cycle_amplitude",
    aspect="temporal",
    compute=_annual_cycle,
    units="%",
    parameters={"stat": "relamp"},
    allowed_groups=["group"],
    cell_methods="time: mean time: range",
    measure="xsdba.measures.RATIO",
)

annual_cycle_phase = StatisticalProperty(
    identifier="annual_cycle_phase",
    aspect="temporal",
    units="",
    compute=_annual_cycle,
    parameters={"stat": "phase"},
    cell_methods="time: range",
    allowed_groups=["group"],
    measure="xsdba.measures.CIRCULAR_BIAS",
)

annual_cycle_asymmetry = StatisticalProperty(
    identifier="annual_cycle_asymmetry",
    aspect="temporal",
    compute=_annual_cycle,
    parameters={"stat": "asymmetry"},
    allowed_groups=["group"],
    units="yr",
)

annual_cycle_minimum = StatisticalProperty(
    identifier="annual_cycle_minimum",
    aspect="temporal",
    units="",
    compute=_annual_cycle,
    parameters={"stat": "min"},
    cell_methods="time: mean time: min",
    allowed_groups=["group"],
)

annual_cycle_maximum = StatisticalProperty(
    identifier="annual_cycle_maximum",
    aspect="temporal",
    compute=_annual_cycle,
    parameters={"stat": "max"},
    cell_methods="time: mean time: max",
    allowed_groups=["group"],
)


# @parse_group
def _annual_statistic(
    da: xr.DataArray,
    *,
    stat: str = "absamp",
    window: int = 31,
    group: str | Grouper = "time",
):
    """
    Annual range statistics.

    Compute a statistic on each year of data and return the interannual average. This is similar
    to the annual cycle, but with the statistic and average operations inverted.

    Parameters
    ----------
    da : xr.DataArray
        Data.
    stat : {'absamp', 'relamp', 'phase'}
        The statistic to return.
    window : int
        Size of the window for the moving average filtering. Deactivate this feature by passing window = 1.

    Returns
    -------
    xr.DataArray, [same units as input or dimensionless]
        Average annual {stat}.
    """
    u = da.units

    if window > 1:
        da = da.rolling(time=window, center=True).mean()

    yrs = da.resample(time="YS")

    match stat:
        case "absamp":
            out = yrs.max() - yrs.min()
            out.attrs.update(pint2cfattrs(units2pint(u), is_difference=True))
        case "relamp":
            out = (yrs.max() - yrs.min()) * 100 / yrs.mean()
            out.attrs["units"] = "%"
        case "phase":
            out = yrs.map(xr.DataArray.idxmax).dt.dayofyear
            out.attrs.update(units="", is_dayofyear=np.int32(1))
        case _:
            raise NotImplementedError(f"{stat} is not a valid annual cycle statistic.")

    return out.mean("time", keep_attrs=True)


mean_annual_range = StatisticalProperty(
    identifier="mean_annual_range",
    aspect="temporal",
    compute=_annual_statistic,
    parameters={"stat": "absamp"},
    allowed_groups=["group"],
)

mean_annual_relative_range = StatisticalProperty(
    identifier="mean_annual_relative_range",
    aspect="temporal",
    compute=_annual_statistic,
    parameters={"stat": "relamp"},
    allowed_groups=["group"],
    units="%",
    measure="xsdba.measures.RATIO",
)

mean_annual_phase = StatisticalProperty(
    identifier="mean_annual_phase",
    aspect="temporal",
    compute=_annual_statistic,
    parameters={"stat": "phase"},
    allowed_groups=["group"],
    units="",
    measure="xsdba.measures.CIRCULAR_BIAS",
)


@parse_group
def _corr_btw_var(
    da1: xr.DataArray,
    da2: xr.DataArray,
    *,
    corr_type: str = "Spearman",
    group: str | Grouper = "time",
    output: str = "correlation",
) -> xr.DataArray:
    r"""
    Correlation between two variables.

    Spearman or Pearson correlation coefficient between two variables at the time resolution.

    Parameters
    ----------
    da1 : xr.DataArray
        First variable on which to calculate the diagnostic.
    da2 : xr.DataArray
        Second variable on which to calculate the diagnostic.
    corr_type: {'Pearson','Spearman'}
        Type of correlation to calculate.
    output: {'correlation', 'pvalue'}
        Whether to return the correlation coefficient or the p-value.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. For 'time.month', the correlation would be calculated on each month separately,
        but with all the years together.

    Returns
    -------
    xr.DataArray, [dimensionless]
        {corr_type} correlation coefficient.
    """
    if corr_type.lower() not in {"pearson", "spearman"}:
        raise ValueError(
            f"{corr_type} is not a valid type. Choose 'Pearson' or 'Spearman'."
        )

    index = {"correlation": 0, "pvalue": 1}[output]

    def _first_output_1d(a, b, index, corr_type):
        """Only keep the correlation (first output) from the scipy function."""
        # for points in the water with NaNs
        if np.isnan(a).all():
            return np.nan
        aok = ~np.isnan(a)
        bok = ~np.isnan(b)
        if corr_type == "Pearson":
            return stats.pearsonr(a[aok & bok], b[aok & bok])[index]
        return stats.spearmanr(a[aok & bok], b[aok & bok])[index]

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _first_output(ds, *, dim, index, corr_type):
        out = xr.apply_ufunc(
            _first_output_1d,
            ds.a,
            ds.b,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            kwargs={"index": index, "corr_type": corr_type},
        )
        return out.rename("out").to_dataset()

    out = _first_output(
        xr.Dataset({"a": da1, "b": da2}), group=group, index=index, corr_type=corr_type
    ).out
    out.attrs["units"] = ""
    return out


corr_btw_var = StatisticalProperty(
    identifier="corr_btw_var", aspect="multivariate", compute=_corr_btw_var
)


def _bivariate_spell_length_distribution(
    da1: xr.DataArray,
    da2: xr.DataArray,
    *,
    method1: str = "amount",
    method2: str = "amount",
    op1: str = ">=",
    op2: str = ">=",
    thresh1: str = "1 mm d-1",
    thresh2: str = "1 mm d-1",
    window: int = 1,
    stat: str = "mean",
    stat_resample: str | None = None,
    group: str | Grouper = "time",
    resample_before_rl: bool = True,
) -> xr.DataArray:
    """
    Spell length distribution with bivariate condition.

    Statistic of spell length distribution when two variables respect individual conditions (defined by an operation, a method,
    and a threshold).

    Parameters
    ----------
    da1 : xr.DataArray
        First variable on which to calculate the diagnostic.
    da2 : xr.DataArray
        Second variable on which to calculate the diagnostic.
    method1 : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    method2 : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op1 : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op1} threshold.
    op2 : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op2} threshold.
    thresh1 : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    thresh2 : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    window : int
        Number of consecutive days respecting the constraint in order to begin a spell.
        Default is 1, which is equivalent to `_bivariate_threshold_count`.
    stat : {'mean', 'sum', 'max','min'}
        Statistics to apply to the remaining time dimension after resampling (e.g. Jan 1980-2010)
    stat_resample : {'mean', 'sum', 'max','min'}, optional
        Statistics to apply to the resampled input at the {group} (e.g. 1-31 Jan 1980). If `None`, the same method as `stat` will be used.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. If 'time.month', the spell lengths are computed separately for each month.
    resample_before_rl : bool
        Determines if the resampling should take place before or after the run
        length encoding (or a similar algorithm) is applied to runs.

    Returns
    -------
    xr.DataArray, [units of the sampling frequency]
        {stat} of spell length distribution when the first variable is {op1} the {method1} {thresh1}
        and the second variable is {op2} the {method2} {thresh2} for {window} consecutive day(s).
    """
    group = group if isinstance(group, Grouper) else Grouper(group)
    allowed_ops = [">", "<", ">=", "<="]
    if op1 not in allowed_ops or op2 not in allowed_ops:
        raise ValueError(
            f"`op1` and `op2` must be in {allowed_ops}, but {op1} and {op2} were given."
        )

    @map_groups(out=[Grouper.PROP], main_only=True)
    def _bivariate_spell_stats(
        ds,
        *,
        dim,
        methods,
        threshs,
        ops,
        freq,
        window,
        resample_before_rl,
        stat,
        stat_resample,
    ):
        # PB: This prevents an import error in the distributed dask scheduler, but I don't know why.
        import xarray.core.resample_cftime  # noqa: F401, pylint: disable=unused-import

        conds = []
        masks = []
        for da, thresh, op, method in zip([ds.da1, ds.da2], threshs, ops, methods):
            masks.append(
                ~(da.isel({dim: 0}).isnull()).drop_vars(dim)
            )  # mask of the ocean with NaNs
            if method == "quantile":
                thresh = da.quantile(thresh, dim=dim).drop_vars("quantile")
            conds.append(compare(da, op, thresh))
        mask = masks[0] & masks[1]
        cond = conds[0] & conds[1]
        out = rl.resample_and_rl(
            cond,
            resample_before_rl,
            rl.rle_statistics,
            reducer=stat_resample,
            window=window,
            dim=dim,
            freq=freq,
        )
        out = getattr(out, stat)(dim=dim)
        out = out.where(mask)
        return out.rename("out").to_dataset()

    # threshold is an amount that will be converted to the right units
    methods = [method1, method2]
    threshs = [thresh1, thresh2]
    for i, da in enumerate([da1, da2]):
        if methods[i] == "amount":
            threshs[i] = convert_units_to(threshs[i], da)
        elif methods[i] != "quantile":
            raise ValueError(
                f"{methods[i]} is not a valid method. Choose 'amount' or 'quantile'."
            )

    out = _bivariate_spell_stats(
        xr.Dataset({"da1": da1, "da2": da2}),
        group=group,
        threshs=threshs,
        methods=methods,
        ops=[op1, op2],
        window=window,
        freq=group.freq,
        resample_before_rl=resample_before_rl,
        stat=stat,
        stat_resample=stat_resample or stat,
    ).out
    # in xclim this was managed by to_agg_units
    # will hard-code this part for now
    m, freq_u_raw = infer_sampling_units(da["time"])
    with xr.set_options(keep_attrs=True):
        out = (out * m).assign_attrs({"units": freq_u_raw})
    return out


bivariate_spell_length_distribution = StatisticalProperty(
    identifier="bivariate_spell_length_distribution",
    aspect="temporal",
    compute=_bivariate_spell_length_distribution,
)


@parse_group
def _bivariate_threshold_count(
    da1: xr.DataArray,
    da2: xr.DataArray,
    *,
    method1: str = "amount",
    method2: str = "amount",
    op1: str = ">=",
    op2: str = ">=",
    thresh1: str = "1 mm d-1",
    thresh2: str = "1 mm d-1",
    stat: str = "mean",
    stat_resample: str | None = None,
    group: str | Grouper = "time",
) -> xr.DataArray:
    """
    Count the number of time steps where two variables respect given conditions.

    Statistic of number of time steps when two variables respect individual conditions (defined by an operation, a method,
    and a threshold).

    Parameters
    ----------
    da1 : xr.DataArray
        First variable on which to calculate the diagnostic.
    da2 : xr.DataArray
        Second variable on which to calculate the diagnostic.
    method1 : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    method2 : {'amount', 'quantile'}
        Method to choose the threshold.
        'amount': The threshold is directly the quantity in {thresh}. It needs to have the same units as {da}.
        'quantile': The threshold is calculated as the quantile {thresh} of the distribution.
    op1 : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op} threshold.
    op2 : {">", "<", ">=", "<="}
        Operation to verify the condition for a spell.
        The condition for a spell is variable {op} threshold.
    thresh1 : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    thresh2 : str or float
        Threshold on which to evaluate the condition to have a spell.
        String with units if the method is "amount".
        Float of the quantile if the method is "quantile".
    stat : {'mean', 'sum', 'max','min'}
        Statistics to apply to the remaining time dimension after resampling (e.g. Jan 1980-2010)
    stat_resample : {'mean', 'sum', 'max','min'}, optional
        Statistics to apply to the resampled input at the {group} (e.g. 1-31 Jan 1980).
        If `None`, the same method as `stat` will be used.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output.
        e.g. For 'time.month', the correlation would be calculated on each month separately,
        but with all the years together.

    Returns
    -------
    xr.DataArray, [dimensionless]
        {stat} number of days when the first variable is {op1} the {method1} {thresh1}
        and the second variable is {op2} the {method2} {thresh2} for {window} consecutive day(s).

    Notes
    -----
    This corresponds to ``xsdba.properties._bivariate_spell_length_distribution`` with `window=1`.
    """
    return _bivariate_spell_length_distribution(
        da1,
        da2,
        method1=method1,
        method2=method2,
        op1=op1,
        op2=op2,
        thresh1=thresh1,
        thresh2=thresh2,
        window=1,
        stat=stat,
        stat_resample=stat_resample,
        group=group,
    )


bivariate_threshold_count = StatisticalProperty(
    identifier="bivariate_threshold_count",
    aspect="multivariate",
    compute=_bivariate_threshold_count,
)


@parse_group
def _relative_frequency(
    da: xr.DataArray,
    *,
    op: str = ">=",
    thresh: str = "1 mm d-1",
    group: str | Grouper = "time",
) -> xr.DataArray:
    """
    Relative Frequency.

    Relative Frequency of days with variable respecting a condition (defined by an operation and a threshold) at the
    time resolution. The relative frequency is the number of days that satisfy the condition divided by the total number
    of days.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    op : {">", "<", ">=", "<="}
        Operation to verify the condition.
        The condition is variable {op} threshold.
    thresh : str
        Threshold on which to evaluate the condition.
    group : {'time', 'time.season', 'time.month'}
        Grouping on the output.
        e.g. For 'time.month', the relative frequency would be calculated on each month, with all years included.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Relative frequency of values {op} {thresh}.
    """
    # mask of the ocean with NaNs
    mask = ~(da.isel({group.dim: 0}).isnull()).drop_vars(group.dim)
    allowed_ops = [">", "<", ">=", "<="]
    if op not in allowed_ops:
        raise ValueError(f"`op` must be in {allowed_ops}, but {op} was given.")

    t = convert_units_to(thresh, da)
    length = da.sizes[group.dim]
    cond = compare(da, op, t)
    if group.prop != "group":  # change the time resolution if necessary
        cond = cond.groupby(group.name)
        # length of the groupBy groups
        length = np.array([len(v) for k, v in cond.groups.items()])
        for _ in range(da.ndim - 1):  # add empty dimension(s) to match input
            length = np.expand_dims(length, axis=-1)
    # count days with the condition and divide by total nb of days
    out = cond.sum(dim=group.dim, skipna=False) / length
    out = out.where(mask, np.nan)
    out.attrs["units"] = ""
    return out


relative_frequency = StatisticalProperty(
    identifier="relative_frequency", aspect="temporal", compute=_relative_frequency
)


@parse_group
def _transition_probability(
    da: xr.DataArray,
    *,
    initial_op: str = ">=",
    final_op: str = ">=",
    thresh: str = "1 mm d-1",
    group: str | Grouper = "time",
) -> xr.DataArray:
    """
    Transition probability.

    Probability of transition from the initial state to the final state. The states are
    booleans comparing the value of the day to the threshold with the operator.

    The transition occurs when consecutive days are both in the given states.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    initial_op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Operation to verify the condition for the initial state.
        The condition is variable {op} threshold.
    final_op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Operation to verify the condition for the final state.
        The condition is variable {op} threshold.
    thresh : str
        Threshold on which to evaluate the condition.
    group : {"time", "time.season", "time.month"}
        Grouping on the output.
        e.g. For "time.month", the transition probability would be calculated on each month, with all years included.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Transition probability of values {initial_op} {thresh} to values {final_op} {thresh}.
    """
    # mask of the ocean with NaNs
    mask = ~(da.isel({group.dim: 0}).isnull()).drop_vars(group.dim)

    today = da.isel(time=slice(0, -1))
    tomorrow = da.shift(time=-1).isel(time=slice(0, -1))

    t = convert_units_to(thresh, da)
    cond = compare(today, initial_op, t) * compare(tomorrow, final_op, t)
    out = group.apply("mean", cond)
    out = out.where(mask, np.nan)
    out.attrs["units"] = ""
    return out


transition_probability = StatisticalProperty(
    identifier="transition_probability",
    aspect="temporal",
    compute=_transition_probability,
)


@parse_group
def _trend(
    da: xr.DataArray,
    *,
    group: str | Grouper = "time",
    output: str = "slope",
) -> xr.DataArray:
    """
    Linear Trend.

    The data is averaged over each time resolution and the inter-annual trend is returned.
    This function will rechunk along the grouping dimension.

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    group : {'time', 'time.season', 'time.month'}
        Grouping on the output.
    output : {'slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr'}
        The attributes of the linear regression to return, as defined in scipy.stats.linregress:
        'slope' is the slope of the regression line.
        'intercept' is the intercept of the regression line.
        'rvalue' is The Pearson correlation coefficient.
        The square of rvalue is equal to the coefficient of determination.
        'pvalue' is the p-value for a hypothesis test whose null hypothesis is that the slope is zero,
        using Wald Test with t-distribution of the test statistic.
        'stderr' is the standard error of the estimated slope (gradient), under the assumption of residual normality.
        'intercept_stderr' is the standard error of the estimated intercept, under the assumption of residual normality.

    Returns
    -------
    xr.DataArray, [units of input per year or dimensionless]
        {output} of the interannual linear trend.

    See Also
    --------
    scipy.stats.linregress

    numpy.polyfit
    """
    u = da.units

    da = da.resample({group.dim: group.freq})  # separate all the {group}
    da_mean = da.mean(dim=group.dim)  # avg over all {group}
    if uses_dask(da_mean):
        da_mean = da_mean.chunk({group.dim: -1})
    if group.prop != "group":
        da_mean = da_mean.groupby(group.name)  # group all month/season together

    def modified_lr(
        x,
    ):  # modify linregress to fit into apply_ufunc and only return slope
        return getattr(stats.linregress(list(range(len(x))), x), output)

    out = xr.apply_ufunc(
        modified_lr,
        da_mean,
        input_core_dims=[[group.dim]],
        vectorize=True,
        dask="parallelized",
    )
    out.attrs["units"] = f"{u}/year"
    return out


trend = StatisticalProperty(identifier="trend", aspect="temporal", compute=_trend)


@parse_group
def _return_value(
    da: xr.DataArray,
    *,
    period: int = 20,
    op: str = "max",
    method: str = "ML",
    group: str | Grouper = "time",
) -> xr.DataArray:
    r"""
    Return value.

    Return the value corresponding to a return period. On average, the return value will be exceeded
    (or not exceed for op='min') every return period (e.g. 20 years). The return value is computed by first extracting
    the variable annual maxima/minima, fitting a statistical distribution to the maxima/minima,
    then estimating the percentile associated with the return period (eg. 95th percentile (1/20) for 20 years)

    Parameters
    ----------
    da : xr.DataArray
        Variable on which to calculate the diagnostic.
    period : int
        Return period. Number of years over which to check if the value is exceeded (or not for op='min').
    op : {'max','min'}
        Whether we are looking for a probability of exceedance ('max', right side of the distribution)
        or a probability of non-exceedance (min, left side of the distribution).
    method : {"ML", "PWM"}
        Fitting method, either maximum likelihood (ML) or probability weighted moments (PWM), also called L-Moments.
        The PWM method is usually more robust to outliers.
    group : {'time', 'time.season', 'time.month'}
        Grouping of the output. A distribution of the extremes is done for each group.

    Returns
    -------
    xr.DataArray, [same as input]
        {period}-{group.prop_name} {op} return level of the variable.
    """

    @map_groups(out=[Grouper.PROP], main_only=True)
    def frequency_analysis_method(ds, *, dim, method):
        sub = select_resample_op(ds.x, op=op)
        params = fit(sub, dist="genextreme", method=method)
        out = parametric_quantile(params, q=1 - 1.0 / period)
        return out.isel(quantile=0, drop=True).rename("out").to_dataset()

    out = frequency_analysis_method(
        da.rename("x").to_dataset(), method=method, group=group
    ).out
    return out.assign_attrs(units=da.units)


return_value = StatisticalProperty(
    identifier="return_value", aspect="temporal", compute=_return_value
)


@parse_group
def _spatial_correlogram(
    da: xr.DataArray,
    *,
    dims: Sequence[str] | None = None,
    bins: int = 100,
    group: str = "time",
    method: int = 1,
):
    """
    Spatial correlogram.

    Compute the pairwise spatial correlations (Spearman) and averages them based on the pairwise distances.
    This collapses the spatial and temporal dimensions and returns a distance bins dimension.
    Needs coordinates for longitude and latitude. This property is heavy to compute, and it will
    need to create a NxN array in memory (outside of dask), where N is the number of spatial points.
    There are shortcuts for all-nan time-slices or spatial points, but scipy's nan-omitting algorithm
    is extremely slow, so the presence of any lone NaN will increase the computation time. Based on an idea
    from :cite:p:`francois_multivariate_2020`.

    Parameters
    ----------
    da : xr.DataArray
        Data.
    dims : sequence of strings, optional
        Name of the spatial dimensions. Once these are stacked, the longitude and latitude coordinates must be 1D.
    bins : int
        Same as argument `bins` from :py:meth:`xarray.DataArray.groupby_bins`.
        If given as a scalar, the equal-width bin limits are generated here
        (instead of letting xarray do it) to improve performance.
    group : str
        Useless for now.

    Returns
    -------
    xr.DataArray, [dimensionless]
        Inter-site correlogram as a function of distance.
    """
    if dims is None:
        dims = [d for d in da.dims if d != "time"]

    corr = _pairwise_spearman(da, dims)
    dists, mn, mx = _pairwise_haversine_and_bins(
        corr.cf["longitude"].values, corr.cf["latitude"].values
    )
    dists = xr.DataArray(dists, dims=corr.dims, coords=corr.coords, name="distance")
    if np.isscalar(bins):
        bins = np.linspace(mn * 0.9999, mx * 1.0001, bins + 1)
    if uses_dask(corr):
        dists = dists.chunk()

    w = np.diff(bins)
    centers = xr.DataArray(
        bins[:-1] + w / 2,
        dims=("distance_bins",),
        attrs={
            "units": "km",
            "long_name": f"Centers of the intersite distance bins (width of {w[0]:.3f} km)",
        },
    )

    dists = dists.where(corr.notnull())

    def _bin_corr(corr, distance):
        """Bin and mean."""
        return stats.binned_statistic(
            distance.flatten(), corr.flatten(), statistic="mean", bins=bins
        ).statistic

    # (_spatial, _spatial2) -> (_spatial, distance_bins)
    binned = xr.apply_ufunc(
        _bin_corr,
        corr,
        dists,
        input_core_dims=[["_spatial", "_spatial2"], ["_spatial", "_spatial2"]],
        output_core_dims=[["distance_bins"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {"distance_bins": bins},
        },
    )
    binned = (
        binned.assign_coords(distance_bins=centers)
        .rename(distance_bins="distance")
        .assign_attrs(units="")
        .rename("corr")
    )
    return binned


spatial_correlogram = StatisticalProperty(
    identifier="spatial_correlogram",
    aspect="spatial",
    compute=_spatial_correlogram,
    allowed_groups=["group"],
)


def _decorrelation_length(
    da: xr.DataArray,
    *,
    radius: int | float = 300,
    thresh: float = 0.50,
    dims: Sequence[str] | None = None,
    bins: int = 100,
    group: xr.Coordinate | str | None = "time",  # FIXME: this needs to be clarified
):
    """
    Decorrelation length.

    Distance from a grid cell where the correlation with its neighbours goes below the threshold.
    A correlogram is calculated for each grid cell following the method from
    ``xsdba.properties.spatial_correlogram``. Then, we find the first bin closest to the correlation threshold.

    Parameters
    ----------
    da : xr.DataArray
        Data.
    radius : float
        Radius (in km) defining the region where correlations will be calculated between a point and its neighbours.
    thresh : float
        Threshold correlation defining decorrelation.
        The decorrelation length is defined as the center of the distance bin that has a correlation closest
        to this threshold.
    dims : sequence of strings
        Name of the spatial dimensions. Once these are stacked, the longitude and latitude coordinates must be 1D.
    bins : int
        Same as argument `bins` from :py:meth:`scipy.stats.binned_statistic`.
        If given as a scalar, the equal-width bin limits from 0 to radius are generated here
        (instead of letting scipy do it) to improve performance.
    group : xarray.Coordinate or str, optional
        Useless for now.

    Returns
    -------
    xr.DataArray, [km]
        Decorrelation length.

    Notes
    -----
    Calculating this property requires a lot of memory. It will not work with large datasets.
    """
    if dims is None and group is not None:
        dims = [d for d in da.dims if d != group.dim]

    corr = _pairwise_spearman(da, dims)

    dists, _, _ = _pairwise_haversine_and_bins(
        corr.cf["longitude"].values, corr.cf["latitude"].values, transpose=True
    )

    dists = xr.DataArray(dists, dims=corr.dims, coords=corr.coords, name="distance")

    trans_dists = xr.DataArray(
        dists.T, dims=corr.dims, coords=corr.coords, name="distance"
    )

    if np.isscalar(bins):
        bin_array = np.linspace(0, radius, bins + 1)
    elif isinstance(bins, np.ndarray):
        bin_array = bins
    else:
        raise ValueError("bins must be a scalar or a numpy array.")

    if uses_dask(corr):
        dists = dists.chunk()
        trans_dists = trans_dists.chunk()

    w = np.diff(bin_array)
    centers = xr.DataArray(
        bin_array[:-1] + w / 2,
        dims=("distance_bins",),
        attrs={
            "units": "km",
            "long_name": f"Centers of the intersite distance bins (width of {w[0]:.3f} km)",
        },
    )
    ds = xr.Dataset({"corr": corr, "distance": dists, "distance2": trans_dists})

    # only keep points inside the radius
    ds = ds.where(ds.distance < radius)
    ds = ds.where(ds.distance2 < radius)

    def _bin_corr(_corr, _distance):
        """Bin and mean."""
        mask_nan = ~np.isnan(_corr)
        binned_corr = stats.binned_statistic(
            _distance[mask_nan], _corr[mask_nan], statistic="mean", bins=bin_array
        )
        stat = binned_corr.statistic
        return stat

    # (_spatial, _spatial2) -> (_spatial, distance_bins)
    binned = (
        xr.apply_ufunc(
            _bin_corr,
            ds.corr,
            ds.distance,
            input_core_dims=[["_spatial2"], ["_spatial2"]],
            output_core_dims=[["distance_bins"]],
            dask="parallelized",
            vectorize=True,
            output_dtypes=[float],
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": {"distance_bins": len(bin_array)},
            },
        )
        .rename("corr")
        .to_dataset()
    )

    binned = (
        binned.assign_coords(distance_bins=centers)
        .rename(distance_bins="distance")
        .assign_attrs(units="")
    )

    closest = abs(binned.corr - thresh).idxmin(dim="distance")
    binned["decorrelation_length"] = closest

    # get back to 2d lat and lon
    # if 'lat' in dims and 'lon' in dims:
    if len(dims) > 1:
        binned = binned.set_index({"_spatial": dims})
        out = binned.decorrelation_length.unstack()
    else:
        out = binned.swap_dims({"_spatial": dims[0]}).decorrelation_length

    copy_all_attrs(out, da)

    out.attrs["units"] = "km"
    return out


decorrelation_length = StatisticalProperty(
    identifier="decorrelation_length",
    aspect="spatial",
    compute=_decorrelation_length,
    allowed_groups=["group"],
)


def first_eof():
    """
    EOF Statistical Property (function removed).

    Warnings
    --------
    Due to a licensing issue, eofs-based functionality has been permanently removed.
    Please excuse the inconvenience.
    For more information, see: https://github.com/Ouranosinc/xclim/issues/1620
    """
    raise RuntimeError(
        "Due to a licensing issue, eofs-based functionality has been permanently removed. "
        "Please excuse the inconvenience. "
        "For more information, see: https://github.com/Ouranosinc/xclim/issues/1620"
    )


def _spectral_variance_alpha(da, dims):
    """
    Compute spectral variance with a radial normalized wavenumber.

    References
    ----------
    :cite:cts:`denis_spectral_2002`
    """
    # compute variance as a function of alpha
    Fmn = xr.apply_ufunc(
        lambda x: dctn(x, norm="ortho"),
        da,
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        vectorize=True,  # Errors in values if set to False! investigate?
        keep_attrs=True,
    )
    sizes = [da[d].size for d in dims]
    # \sigma = \sum_{m,n} F_{m,n} / (M*N)
    sigmn = (1 / np.prod(sizes)) * (Fmn**2)
    sigmn["alpha"] = _normalized_radial_wavenumber(da, dims)

    # eq.13 and 14 of the reference
    # alpha should increase in integer steps of 1/min(N_i,N_j)
    sigmn["alpha"] = (sigmn["alpha"] // (1 / min(sizes))) * (1 / min(sizes))
    return sigmn.groupby("alpha").sum(keep_attrs=True)


# TODO: Why can't I make `dims` as positional argument?
def _spectral_variance(
    da: xr.DataArray,
    *,
    dims: str = ["lat", "lon"],
    delta: str | None = None,
    group: xr.Coordinate | str | None = "time",  # FIXME: this needs to be clarified
):
    r"""
    Compute spectral variance

    If `delta` is specified, the normalized wavenumber `alpha` will be converted to a `wavelength`

    Parameters
    ----------
    da : xr.DataArray
        Input physical field.
    dims: list
        Dimensions on which to perform the DCT.
    delta: str, Optional
        Nominal resolution of the grid. It should be a string with units.
    group: xr.Coordinate | str | None = "time"
        Useless for now # FIXME: this needs to be clarified.

    Notes
    -----
    If the input field contains any `nan`, the output will be all `nan` values.

    References
    ----------
    :cite:cts:`denis_spectral_2002`
    """
    var = _spectral_variance_alpha(da, dims)
    # The grid is incomplete for alpha>1, so we restrain to alpha<=1
    # alpha=0 was excluded simply  because it's excluded from
    # the total variance \sigma = \sum_{m,n \neq (0,0)} \sigma_{m,n}
    var = var.where((var.alpha > 0) & (var.alpha <= 1), drop=True)
    if delta is not None:
        var["alpha"] = normalized_wavenumber_to_wavelength(var.alpha, delta=delta)
        var = var.rename({"alpha": "wavelength"})
    var = var.assign_attrs(da.attrs)
    if "units" in da.attrs:
        var.attrs["units"] = str(str2pint(da.units) ** 2)
    name = "variance"
    if var.name:
        name = f"{var.name}_{name}"
    var = var.to_dataset(name=name)[name]
    return var


spectral_variance = StatisticalProperty(
    identifier="spectral_variance",
    aspect="spatial",
    compute=_spectral_variance,
    allowed_groups=["group"],  # FIXME: this needs to be clarified
)
