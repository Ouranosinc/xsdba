"""
Global or contextual options for xsdba, similar to xarray.set_options.
"""

from __future__ import annotations

from collections.abc import Callable

EXTRA_OUTPUT = "extra_output"
AS_DATASET = "as_dataset"

MISSING_METHODS: dict[str, Callable] = {}

OPTIONS = {
    EXTRA_OUTPUT: False,
    AS_DATASET: False,
}

_VALIDATORS = {
    EXTRA_OUTPUT: lambda opt: isinstance(opt, bool),
    AS_DATASET: lambda opt: isinstance(opt, bool),
}


class set_options:
    """
    Set options for xsdba in a controlled context.

    Attributes
    ----------
    extra_output : bool
        Whether to add diagnostic variables to outputs of sdba's `train`, `adjust`
        and `processing` operations. Details about these additional variables are given in the object's
        docstring. When activated, `adjust` will return a Dataset with `scen` and those extra diagnostics
        For `processing` functions, see the doc, the output type might change, or not depending on the
        algorithm. Default: ``False``.

    Examples
    --------
    You can use ``set_options`` either as a context manager:

    >>> import xclim
    >>> ds = xr.open_dataset(path_to_tas_file).tas
    >>> with xsdba.set_options(extra_output=True):
    ...     out = xsdba.MBCn.train(ref, hist)

    Or to set global options:

    .. code-block:: python

        import xsdba

        xsdba.set_options(extra_output=True)
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                msg = f"Argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}."
                raise ValueError(msg)
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")

            self.old[k] = OPTIONS[k]

        self._update(kwargs)

    def __enter__(self):
        """Context management."""
        return

    @staticmethod
    def _update(kwargs):
        """Update values."""
        OPTIONS.update(kwargs)

    def __exit__(self, option_type, value, traceback):  # noqa: F841
        """Context management."""
        self._update(self.old)
