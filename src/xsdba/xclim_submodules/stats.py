"""Statistic-related functions. See the `frequency_analysis` notebook for examples."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.stats
import xarray as xr

from xsdba.base import uses_dask
from xsdba.formatting import prefix_attrs, unprefix_attrs, update_history
from xsdba.typing import DateStr, Quantified
from xsdba.units import convert_units_to

from . import generic

__all__ = [
    "_fit_start",
    "dist_method",
    "fa",
    "fit",
    "frequency_analysis",
    "get_dist",
    "parametric_cdf",
    "parametric_quantile",
]


# Fit the parameters.
# This would also be the place to impose constraints on the series minimum length if needed.
def _fitfunc_1d(arr, *, dist, nparams, method, **fitkwargs):
    """Fit distribution parameters."""
    x = np.ma.masked_invalid(arr).compressed()  # pylint: disable=no-member

    # Return NaNs if array is empty.
    if len(x) <= 1:
        return np.asarray([np.nan] * nparams)

    # Estimate parameters
    if method in ["ML", "MLE"]:
        args, kwargs = _fit_start(x, dist.name, **fitkwargs)
        params = dist.fit(x, *args, method="mle", **kwargs, **fitkwargs)
    elif method == "MM":
        params = dist.fit(x, method="mm", **fitkwargs)
    elif method == "PWM":
        params = list(dist.lmom_fit(x).values())
    elif method == "APP":
        args, kwargs = _fit_start(x, dist.name, **fitkwargs)
        kwargs.setdefault("loc", 0)
        params = list(args) + [kwargs["loc"], kwargs["scale"]]
    else:
        raise NotImplementedError(f"Unknown method `{method}`.")

    params = np.asarray(params)

    # Fill with NaNs if one of the parameters is NaN
    if np.isnan(params).any():
        params[:] = np.nan

    return params


def fit(
    da: xr.DataArray,
    dist: str | scipy.stats.rv_continuous = "norm",
    method: str = "ML",
    dim: str = "time",
    **fitkwargs: Any,
) -> xr.DataArray:
    r"""Fit an array to a univariate distribution along the time dimension.

    Parameters
    ----------
    da : xr.DataArray
        Time series to be fitted along the time dimension.
    dist : str or rv_continuous distribution object
        Name of the univariate distribution, such as beta, expon, genextreme, gamma, gumbel_r, lognorm, norm
        (see :py:mod:scipy.stats for full list) or the distribution object itself.
    method : {"ML" or "MLE", "MM", "PWM", "APP"}
        Fitting method, either maximum likelihood (ML or MLE), method of moments (MM) or approximate method (APP).
        Can also be the probability weighted moments (PWM), also called L-Moments, if a compatible `dist` object is passed.
        The PWM method is usually more robust to outliers.
    dim : str
        The dimension upon which to perform the indexing (default: "time").
    \*\*fitkwargs
        Other arguments passed directly to :py:func:`_fitstart` and to the distribution's `fit`.

    Returns
    -------
    xr.DataArray
        An array of fitted distribution parameters.

    Notes
    -----
    Coordinates for which all values are NaNs will be dropped before fitting the distribution. If the array still
    contains NaNs, the distribution parameters will be returned as NaNs.
    """
    method = method.upper()
    method_name = {
        "ML": "maximum likelihood",
        "MM": "method of moments",
        "MLE": "maximum likelihood",
        "PWM": "probability weighted moments",
        "APP": "approximative method",
    }
    if method not in method_name:
        raise ValueError(f"Fitting method not recognized: {method}")

    # Get the distribution
    dist = get_dist(dist)

    if method == "PWM" and not hasattr(dist, "lmom_fit"):
        raise ValueError(
            f"The given distribution {dist} does not implement the PWM fitting method. Please pass an instance from the lmoments3 package."
        )

    shape_params = [] if dist.shapes is None else dist.shapes.split(",")
    dist_params = shape_params + ["loc", "scale"]

    data = xr.apply_ufunc(
        _fitfunc_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["dparams"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        kwargs=dict(
            # Don't know how APP should be included, this works for now
            dist=dist,
            nparams=len(dist_params),
            method=method,
            **fitkwargs,
        ),
        dask_gufunc_kwargs={"output_sizes": {"dparams": len(dist_params)}},
    )

    # Add coordinates for the distribution parameters and transpose to original shape (with dim -> dparams)
    dims = [d if d != dim else "dparams" for d in da.dims]
    out = data.assign_coords(dparams=dist_params).transpose(*dims)

    out.attrs = prefix_attrs(
        da.attrs, ["standard_name", "long_name", "units", "description"], "original_"
    )
    attrs = dict(
        long_name=f"{dist.name} parameters",
        description=f"Parameters of the {dist.name} distribution",
        method=method,
        estimator=method_name[method].capitalize(),
        scipy_dist=dist.name,
        units="",
        history=update_history(
            f"Estimate distribution parameters by {method_name[method]} method along dimension {dim}.",
            new_name="fit",
            data=da,
        ),
    )
    out.attrs.update(attrs)
    return out


def parametric_quantile(
    p: xr.DataArray,
    q: float | Sequence[float],
    dist: str | scipy.stats.rv_continuous | None = None,
) -> xr.DataArray:
    """Return the value corresponding to the given distribution parameters and quantile.

    Parameters
    ----------
    p : xr.DataArray
        Distribution parameters returned by the `fit` function.
        The array should have dimension `dparams` storing the distribution parameters,
        and attribute `scipy_dist`, storing the name of the distribution.
    q : float or Sequence of float
        Quantile to compute, which must be between `0` and `1`, inclusive.
    dist: str, rv_continuous instance, optional
        The distribution name or instance if the `scipy_dist` attribute is not available on `p`.

    Returns
    -------
    xarray.DataArray
        An array of parametric quantiles estimated from the distribution parameters.

    Notes
    -----
    When all quantiles are above 0.5, the `isf` method is used instead of `ppf` because accuracy is sometimes better.
    """
    q = np.atleast_1d(q)

    dist = get_dist(dist or p.attrs["scipy_dist"])

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    if np.all(q > 0.5):

        def func(x):
            return dist.isf(1 - q, *x)

    else:

        def func(x):
            return dist.ppf(q, *x)

    data = xr.apply_ufunc(
        func,
        p,
        input_core_dims=[["dparams"]],
        output_core_dims=[["quantile"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes": {"quantile": len(q)}},
    )

    # Assign quantile coordinates and transpose to preserve original dimension order
    dims = [d if d != "dparams" else "quantile" for d in p.dims]
    out = data.assign_coords(quantile=q).transpose(*dims)
    out.attrs = unprefix_attrs(p.attrs, ["units", "standard_name"], "original_")

    attrs = dict(
        long_name=f"{dist.name} quantiles",
        description=f"Quantiles estimated by the {dist.name} distribution",
        cell_methods="dparams: ppf",
        history=update_history(
            "Compute parametric quantiles from distribution parameters",
            new_name="parametric_quantile",
            parameters=p,
        ),
    )
    out.attrs.update(attrs)
    return out


def parametric_cdf(
    p: xr.DataArray,
    v: float | Sequence[float],
    dist: str | scipy.stats.rv_continuous | None = None,
) -> xr.DataArray:
    """Return the cumulative distribution function corresponding to the given distribution parameters and value.

    Parameters
    ----------
    p : xr.DataArray
        Distribution parameters returned by the `fit` function.
        The array should have dimension `dparams` storing the distribution parameters,
        and attribute `scipy_dist`, storing the name of the distribution.
    v : float or Sequence of float
        Value to compute the CDF.
    dist: str, rv_continuous instance, optional
        The distribution name or instance is the `scipy_dist` attribute is not available on `p`.

    Returns
    -------
    xarray.DataArray
        An array of parametric CDF values estimated from the distribution parameters.
    """
    v = np.atleast_1d(v)

    dist = get_dist(dist or p.attrs["scipy_dist"])

    # Create a lambda function to facilitate passing arguments to dask. There is probably a better way to do this.
    def func(x):
        return dist.cdf(v, *x)

    data = xr.apply_ufunc(
        func,
        p,
        input_core_dims=[["dparams"]],
        output_core_dims=[["cdf"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes": {"cdf": len(v)}},
    )

    # Assign quantile coordinates and transpose to preserve original dimension order
    dims = [d if d != "dparams" else "cdf" for d in p.dims]
    out = data.assign_coords(cdf=v).transpose(*dims)
    out.attrs = unprefix_attrs(p.attrs, ["units", "standard_name"], "original_")

    attrs = dict(
        long_name=f"{dist.name} cdf",
        description=f"CDF estimated by the {dist.name} distribution",
        cell_methods="dparams: cdf",
        history=update_history(
            "Compute parametric cdf from distribution parameters",
            new_name="parametric_cdf",
            parameters=p,
        ),
    )
    out.attrs.update(attrs)
    return out


def fa(
    da: xr.DataArray,
    t: int | Sequence,
    dist: str | scipy.stats.rv_continuous = "norm",
    mode: str = "max",
    method: str = "ML",
) -> xr.DataArray:
    """Return the value corresponding to the given return period.

    Parameters
    ----------
    da : xr.DataArray
        Maximized/minimized input data with a `time` dimension.
    t : int or Sequence of int
        Return period. The period depends on the resolution of the input data. If the input array's resolution is
        yearly, then the return period is in years.
    dist : str or rv_continuous instance
        Name of the univariate distribution, such as:
        `beta`, `expon`, `genextreme`, `gamma`, `gumbel_r`, `lognorm`, `norm`
        Or the distribution instance itself.
    mode : {'min', 'max}
        Whether we are looking for a probability of exceedance (max) or a probability of non-exceedance (min).
    method : {"ML", "MLE", "MOM", "PWM", "APP"}
        Fitting method, either maximum likelihood (ML or MLE), method of moments (MOM) or approximate method (APP).
        Also accepts probability weighted moments (PWM), also called L-Moments, if `dist` is an instance from the lmoments3 library.
        The PWM method is usually more robust to outliers.

    Returns
    -------
    xarray.DataArray
        An array of values with a 1/t probability of exceedance (if mode=='max').

    See Also
    --------
    scipy.stats : For descriptions of univariate distribution types.
    """
    # Fit the parameters of the distribution
    p = fit(da, dist, method=method)
    t = np.atleast_1d(t)

    if mode in ["max", "high"]:
        q = 1 - 1.0 / t

    elif mode in ["min", "low"]:
        q = 1.0 / t

    else:
        raise ValueError(f"Mode `{mode}` should be either 'max' or 'min'.")

    # Compute the quantiles
    out = (
        parametric_quantile(p, q, dist)
        .rename({"quantile": "return_period"})
        .assign_coords(return_period=t)
    )
    out.attrs["mode"] = mode
    return out


def frequency_analysis(
    da: xr.DataArray,
    mode: str,
    t: int | Sequence[int],
    dist: str | scipy.stats.rv_continuous,
    window: int = 1,
    freq: str | None = None,
    method: str = "ML",
    **indexer: int | float | str,
) -> xr.DataArray:
    r"""Return the value corresponding to a return period.

    Parameters
    ----------
    da : xarray.DataArray
        Input data.
    mode : {'min', 'max'}
        Whether we are looking for a probability of exceedance (high) or a probability of non-exceedance (low).
    t : int or sequence
        Return period. The period depends on the resolution of the input data. If the input array's resolution is
        yearly, then the return period is in years.
    dist : str or rv_continuous
        Name of the univariate distribution, e.g. `beta`, `expon`, `genextreme`, `gamma`, `gumbel_r`, `lognorm`, `norm`.
        Or an instance of the distribution.
    window : int
        Averaging window length (days).
    freq : str, optional
        Resampling frequency. If None, the frequency is assumed to be 'YS' unless the indexer is season='DJF',
        in which case `freq` would be set to `YS-DEC`.
    method : {"ML" or "MLE", "MOM", "PWM", "APP"}
        Fitting method, either maximum likelihood (ML or MLE), method of moments (MOM) or approximate method (APP).
        Also accepts probability weighted moments (PWM), also called L-Moments, if `dist` is an instance from the lmoments3 library.
        The PWM method is usually more robust to outliers.
    \*\*indexer
        Time attribute and values over which to subset the array. For example, use season='DJF' to select winter values,
        month=1 to select January, or month=[6,7,8] to select summer months. If indexer is not provided, all values are
        considered.

    Returns
    -------
    xarray.DataArray
        An array of values with a 1/t probability of exceedance or non-exceedance when mode is high or low respectively.

    See Also
    --------
    scipy.stats : For descriptions of univariate distribution types.
    """
    # Apply rolling average
    attrs = da.attrs.copy()
    if window > 1:
        da = da.rolling(time=window).mean(skipna=False)
        da.attrs.update(attrs)

    # Assign default resampling frequency if not provided
    freq = freq or generic.default_freq(**indexer)

    # Extract the time series of min or max over the period
    sel = generic.select_resample_op(da, op=mode, freq=freq, **indexer)

    if uses_dask(sel):
        sel = sel.chunk({"time": -1})
    # Frequency analysis
    return fa(sel, t, dist=dist, mode=mode, method=method)


def get_dist(dist: str | scipy.stats.rv_continuous):
    """Return a distribution object from `scipy.stats`."""
    if isinstance(dist, scipy.stats.rv_continuous):
        return dist

    dc = getattr(scipy.stats, dist, None)
    if dc is None:
        e = f"Statistical distribution `{dist}` is not found in scipy.stats."
        raise ValueError(e)
    return dc


def _fit_start(x, dist: str, **fitkwargs: Any) -> tuple[tuple, dict]:
    r"""Return initial values for distribution parameters.

    Providing the ML fit method initial values can help the optimizer find the global optimum.

    Parameters
    ----------
    x : array-like
        Input data.
    dist : str
        Name of the univariate distribution, e.g. `beta`, `expon`, `genextreme`, `gamma`, `gumbel_r`, `lognorm`, `norm`.
        (see :py:mod:scipy.stats). Only `genextreme` and `weibull_exp` distributions are supported.
    \*\*fitkwargs
        Kwargs passed to fit.

    Returns
    -------
    tuple, dict

    References
    ----------
    :cite:cts:`coles_introduction_2001,cohen_parameter_2019, thom_1958, cooke_1979, muralidhar_1992`

    """
    x = np.asarray(x)
    m = x.mean()
    v = x.var()

    if dist == "genextreme":
        s = np.sqrt(6 * v) / np.pi
        return (0.1,), {"loc": m - 0.57722 * s, "scale": s}

    if dist == "genpareto" and "floc" in fitkwargs:
        # Taken from julia' Extremes. Case for when "mu/loc" is known.
        t = fitkwargs["floc"]
        if not np.isclose(t, 0):
            m = (x - t).mean()
            v = (x - t).var()

        c = 0.5 * (1 - m**2 / v)
        scale = (1 - c) * m
        return (c,), {"scale": scale}

    if dist in "weibull_min":
        s = x.std()
        loc = x.min() - 0.01 * s
        chat = np.pi / np.sqrt(6) / (np.log(x - loc)).std()
        scale = ((x - loc) ** chat).mean() ** (1 / chat)
        return (chat,), {"loc": loc, "scale": scale}

    if dist in ["gamma"]:
        if "floc" in fitkwargs:
            loc0 = fitkwargs["floc"]
        else:
            xs = sorted(x)
            x1, x2, xn = xs[0], xs[1], xs[-1]
            # muralidhar_1992 would suggest the following, but it seems more unstable
            # using cooke_1979 for now
            # n = len(x)
            # cv = x.std() / x.mean()
            # p = (0.48265 + 0.32967 * cv) * n ** (-0.2984 * cv)
            # xp = xs[int(p/100*n)]
            xp = x2
            loc0 = (x1 * xn - xp**2) / (x1 + xn - 2 * xp)
            loc0 = loc0 if loc0 < x1 else (0.9999 * x1 if x1 > 0 else 1.0001 * x1)
        x_pos = x - loc0
        x_pos = x_pos[x_pos > 0]
        m = x_pos.mean()
        log_of_mean = np.log(m)
        mean_of_logs = np.log(x_pos).mean()
        A = log_of_mean - mean_of_logs
        a0 = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A)
        scale0 = m / a0
        kwargs = {"scale": scale0, "loc": loc0}
        return (a0,), kwargs

    if dist in ["fisk"]:
        if "floc" in fitkwargs:
            loc0 = fitkwargs["floc"]
        else:
            xs = sorted(x)
            x1, x2, xn = xs[0], xs[1], xs[-1]
            loc0 = (x1 * xn - x2**2) / (x1 + xn - 2 * x2)
            loc0 = loc0 if loc0 < x1 else (0.9999 * x1 if x1 > 0 else 1.0001 * x1)
        x_pos = x - loc0
        x_pos = x_pos[x_pos > 0]
        # method of moments:
        # LHS is computed analytically with the two-parameters log-logistic distribution
        # and depends on alpha,beta
        # RHS is from the sample
        # <x> = m
        # <x^2> / <x>^2 = m2/m**2
        # solving these equations yields
        m = x_pos.mean()
        m2 = (x_pos**2).mean()
        scale0 = 2 * m**3 / (m2 + m**2)
        c0 = np.pi * m / np.sqrt(3) / np.sqrt(m2 - m**2)
        kwargs = {"scale": scale0, "loc": loc0}
        return (c0,), kwargs
    return (), {}


def _dist_method_1D(  # noqa: N802
    *args, dist: str | scipy.stats.rv_continuous, function: str, **kwargs: Any
) -> xr.DataArray:
    r"""Statistical function for given argument on given distribution initialized with params.

    See :py:ref:`scipy.stats.rv_continuous` for all available functions and their arguments.
    Every method where `"*args"` are the distribution parameters can be wrapped.

    Parameters
    ----------
    \*args
        The arguments for the requested scipy function.
    dist : str
        The scipy name of the distribution.
    function : str
        The name of the function to call.
    \*\*kwargs
        Other parameters to pass to the function call.

    Returns
    -------
    array_like
    """
    dist = get_dist(dist)
    return getattr(dist, function)(*args, **kwargs)


def dist_method(
    function: str,
    fit_params: xr.DataArray,
    arg: xr.DataArray | None = None,
    dist: str | scipy.stats.rv_continuous | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    r"""Vectorized statistical function for given argument on given distribution initialized with params.

    Methods where `"*args"` are the distribution parameters can be wrapped, except those that reduce dimensions (
    e.g. `nnlf`) or create new dimensions (eg: 'rvs' with size != 1, 'stats' with more than one moment, 'interval',
    'support').

    Parameters
    ----------
    function : str
        The name of the function to call.
    fit_params : xr.DataArray
        Distribution parameters are along `dparams`, in the same order as given by :py:func:`fit`.
    arg : array_like, optional
        The first argument for the requested function if different from `fit_params`.
    dist : str pr rv_continuous, optional
        The distribution name or instance. Defaults to the `scipy_dist` attribute or `fit_params`.
    \*\*kwargs : dict
        Other parameters to pass to the function call.

    Returns
    -------
    array_like
        Same shape as arg.

    See Also
    --------
    scipy.stats.rv_continuous : for all available functions and their arguments.
    """
    # Typically the data to be transformed
    arg = [arg] if arg is not None else []
    if function == "nnlf":
        raise ValueError(
            "This method is not supported because it reduces the dimensionality of the data."
        )

    # We don't need to set `input_core_dims` because we're explicitly splitting the parameters here.
    args = arg + [fit_params.sel(dparams=dp) for dp in fit_params.dparams.values]

    return xr.apply_ufunc(
        _dist_method_1D,
        *args,
        kwargs={
            "dist": dist or fit_params.attrs["scipy_dist"],
            "function": function,
            **kwargs,
        },
        output_dtypes=[float],
        dask="parallelized",
    )
