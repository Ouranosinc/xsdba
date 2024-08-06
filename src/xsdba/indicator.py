"""# noqa: SS01
Indicator Utilities
===================

The `Indicator` class wraps indices computations with pre- and post-processing functionality. Prior to computations,
the class runs data and metadata health checks. After computations, the class masks values that should be considered
missing and adds metadata attributes to the  object.

There are many ways to construct indicators. A good place to start is
`this notebook <notebooks/extendxclim.ipynb#Defining-new-indicators>`_.

Dictionary and YAML parser # noqa: GL06
-------------------------- # noqa: GL06

To construct indicators dynamically, xclim can also use dictionaries and parse them from YAML files.
This is especially useful for generating whole indicator "submodules" from files.
This functionality is inspired by the work of `clix-meta <https://github.com/clix-meta/clix-meta/>`_.

YAML file structure
~~~~~~~~~~~~~~~~~~~

Indicator-defining yaml files are structured in the following way.
Most entries of the `indicators` section are mirroring attributes of
the :py:class:`Indicator`, please refer to its documentation for more
details on each.

.. code-block:: yaml

    module: <module name>  # Defaults to the file name
    realm: <realm>  # If given here, applies to all indicators that do not already provide it.
    keywords: <keywords> # Merged with indicator-specific keywords (joined with a space)
    references: <references> # Merged with indicator-specific references (joined with a new line)
    base: <base indicator class>  # Defaults to "Daily" and applies to all indicators that do not give it.
    doc: <module docstring>  # Defaults to a minimal header, only valid if the module doesn't already exist.
    variables:  # Optional section if indicators declared below rely on variables unknown to xclim
                # (not in `xclim.core.utils.VARIABLES`)
                # The variables are not module-dependent and will overwrite any already existing with the same name.
      <varname>:
        canonical_units: <units> # required
        description: <description> # required
        standard_name: <expected standard_name> # optional
        cell_methods: <expected cell_methods> # optional
    indicators:
      <identifier>:
        # From which Indicator to inherit
        base: <base indicator class>  # Defaults to module-wide base class
                                      # If the name startswith a '.', the base class is taken from the current module
                                      # (thus an indicator declared _above_).
                                      # Available classes are listed in `xsdba.indicator.registry` and
                                      # `xsdba.indicator.base_registry`.

        # General metadata, usually parsed from the `compute`'s docstring when possible.
        realm: <realm>  # defaults to module-wide realm. One of "atmos", "land", "seaIce", "ocean".
        title: <title>
        abstract: <abstract>
        keywords: <keywords>  # Space-separated, merged to module-wide keywords.
        references: <references>  # newline-seperated, merged to module-wide references.
        notes: <notes>

        # Other options
        missing: <missing method name>
        missing_options:
            # missing options mapping
        allowed_periods: [<list>, <of>, <allowed>, <periods>]

        # Compute function
        compute: <function name>  # Referring to a function in `Indices` module (xclim.indices.generic or xclim.indices)
        input:  # When "compute" is a generic function, this is a mapping from argument name to the expected variable.
                # This will allow the input units and CF metadata checks to run on the inputs.
                # Can also be used to modify the expected variable, as long as it has the same dimensionality
                # Ex: tas instead of tasmin.
                # Can refer to a variable declared in the `variables` section above.
          <var name in compute> : <variable official name>
          ...

.. code-block:: restructuredtext

    Parameters  # noqa: D214
    ----------  # noqa: D215
    <param name> : <param data>  # Simplest case, to inject parameters in the compute function.
    <param name> :  # To change parameters metadata or to declare units when "compute" is a generic function.
    units : <param units>  # Only valid if "compute" points to a generic function
    default : <param default>
    description : <param description>
    kind : <param kind> # Override the parameter kind.
                        # This is mostly useful for transforming an optional variable into a required one by passing ``kind: 0``.
    ...

All fields are optional. Other fields found in the yaml file will trigger errors in xclim.
In the following, the section under `<identifier>` is referred to as `data`. When creating indicators from
a dictionary, with :py:meth:`Indicator.from_dict`, the input dict must follow the same structure of `data`.

When a module is built from a yaml file, the yaml is first validated against the schema (see xclim/data/schema.yml)
using the YAMALE library (:cite:p:`lopker_yamale_2022`). See the "Extending xclim" notebook for more info.

Inputs
~~~~~~
As xclim has strict definitions of possible input variables (see :py:data:`xclim.core.utils.variables`),
the mapping of `data.input` simply links an argument name from the function given in "compute"
to one of those official variables.
"""

from __future__ import annotations

import re
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import reduce
from inspect import Parameter as _Parameter
from inspect import Signature
from inspect import _empty as _empty_default
from inspect import signature
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Union

import numpy as np
import xarray
import yamale
from xarray import DataArray, Dataset
from yaml import safe_load

from .base import Grouper, infer_kind_from_parameter
from .calendar import parse_offset
from .datachecks import is_percentile_dataarray
from .formatting import (
    AttrFormatter,
    default_formatter,
    gen_call_string,
    generate_indicator_docstring,
    get_percentile_metadata,
    merge_attributes,
    parse_doc,
    update_history,
)
from .locales import (
    TRANSLATABLE_ATTRS,
    get_local_attrs,
    get_local_formatter,
    load_locale,
    read_locale_file,
)
from .logging import MissingVariableError, ValidationError, raise_warn_or_log
from .options import (
    AS_DATASET,
    CHECK_MISSING,
    KEEP_ATTRS,
    METADATA_LOCALES,
    MISSING_METHODS,
    MISSING_OPTIONS,
    OPTIONS,
)
from .typing import InputKind
from .units import convert_units_to, units
from .utils import load_module

# Indicators registry
registry = {}  # Main class registry
base_registry = {}
_indicators_registry = defaultdict(list)  # Private instance registry


class _empty:
    pass


@dataclass
class Parameter:
    """Class for storing an indicator's controllable parameter.

    For retrocompatibility, this class implements a "getitem" and a special "contains".

    Example
    -------
    >>> p = Parameter(InputKind.NUMBER, default=2, description="A simple number")
    >>> p.units is Parameter._empty  # has not been set
    True
    >>> "units" in p  # Easier/retrocompatible way to test if units are set
    False
    >>> p.description
    'A simple number'
    """

    _empty = _empty

    kind: InputKind
    default: Any = _empty_default
    description: str = ""
    units: str = _empty
    choices: set = _empty
    value: Any = _empty

    def update(self, other: dict) -> None:
        """Update a parameter's values from a dict."""
        for k, v in other.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unexpected parameter field '{k}'.")

    @classmethod
    def is_parameter_dict(cls, other: dict) -> bool:
        """Return whether indicator has a parameter dictionary."""
        return set(other.keys()).issubset(
            cls.__dataclass_fields__.keys()  # pylint: disable=no-member
        )

    # def __getitem__(self, key) -> str:
    #     """Return an item in retro-compatible fashion."""
    #     try:
    #         return str(getattr(self, key))
    #     except AttributeError as err:
    #         raise KeyError(key) from err

    def __contains__(self, key) -> bool:
        """Imitate previous behaviour where "units" and "choices" were missing, instead of being "_empty"."""
        return getattr(self, key, _empty) is not _empty

    def asdict(self) -> dict:
        """Format indicators as a dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not _empty}

    @property
    def injected(self) -> bool:
        """Indicate whether values are injected."""
        return self.value is not _empty


class IndicatorRegistrar:
    """Climate Indicator registering object."""

    def __new__(cls):
        """Add subclass to registry."""
        name = cls.__name__.upper()
        module = cls.__module__
        # If the module is not one of xclim's default, prepend the submodule name.
        if module.startswith("xclim.indicators"):
            submodule = module.split(".")[2]
            if submodule not in ["atmos", "generic", "land", "ocean", "seaIce"]:
                name = f"{submodule}.{name}"
        else:
            name = f"{module}.{name}"
        if name in registry:
            warnings.warn(
                f"Class {name} already exists and will be overwritten.", stacklevel=1
            )
        registry[name] = cls
        cls._registry_id = name
        return super().__new__(cls)

    def __init__(self):
        _indicators_registry[self.__class__].append(weakref.ref(self))

    @classmethod
    def get_instance(cls):
        """Return first found instance.

        Raises `ValueError` if no instance exists.
        """
        for inst_ref in _indicators_registry[cls]:
            inst = inst_ref()
            if inst is not None:
                return inst
        raise ValueError(
            f"There is no existing instance of {cls.__name__}. "
            "Either none were created or they were all garbage-collected."
        )


class Indicator(IndicatorRegistrar):
    r"""Climate indicator base class.

    Climate indicator object that, when called, computes an indicator and assigns its output a number of
    CF-compliant attributes. Some of these attributes can be *templated*, allowing metadata to reflect
    the value of call arguments.

    Instantiating a new indicator returns an instance but also creates and registers a custom subclass
    in :py:data:`xsdba.indicator.registry`.

    Attributes in `Indicator.cf_attrs` will be formatted and added to the output variable(s).
    This attribute is a list of dictionaries. For convenience and retro-compatibility,
    standard CF attributes (names listed in :py:attr:`xsdba.indicator.Indicator._cf_names`)
    can be passed as strings or list of strings directly to the indicator constructor.

    A lot of the Indicator's metadata is parsed from the underlying `compute` function's
    docstring and signature. Input variables and parameters are listed in
    :py:attr:`xsdba.indicator.Indicator.parameters`, while parameters that will be
    injected in the compute function are in  :py:attr:`xsdba.indicator.Indicator.injected_parameters`.
    Both are simply views of :py:attr:`xsdba.indicator.Indicator._all_parameters`.

    Compared to their base `compute` function, indicators add the possibility of using dataset as input,
    with the injected argument `ds` in the call signature. All arguments that were indicated
    by the compute function to be variables (DataArrays) through annotations will be promoted
    to also accept strings that correspond to variable names in the `ds` dataset.

    Parameters
    ----------
    identifier : str
        Unique ID for class registry, should be a valid slug.
    realm : {'atmos', 'seaIce', 'land', 'ocean'}
        General domain of validity of the indicator. Indicators created outside xclim.indicators must set this attribute.
    compute : func
        The function computing the indicators. It should return one or more DataArray.
    cf_attrs : list of dicts
        Attributes to be formatted and added to the computation's output.
        See :py:attr:`xsdba.indicator.Indicator.cf_attrs`.
    title : str
        A succinct description of what is in the computed outputs. Parsed from `compute` docstring if None (first paragraph).
    abstract : str
        A long description of what is in the computed outputs. Parsed from `compute` docstring if None (second paragraph).
    keywords : str
        Comma separated list of keywords. Parsed from `compute` docstring if None (from a "Keywords" section).
    references : str
        Published or web-based references that describe the data or methods used to produce it. Parsed from
        `compute` docstring if None (from the "References" section).
    notes : str
        Notes regarding computing function, for example the mathematical formulation. Parsed from `compute`
        docstring if None (form the "Notes" section).
    src_freq : str, sequence of strings, optional
        The expected frequency of the input data. Can be a list for multiple frequencies, or None if irrelevant.
    context : str
        The `pint` unit context, for example use 'hydro' to allow conversion from kg m-2 s-1 to mm/day.

    Notes
    -----
    All subclasses created are available in the `registry` attribute and can be used to define custom subclasses
    or parse all available instances.
    """

    # Officially-supported metadata attributes on the output variables
    _cf_names = [
        "var_name",
        "standard_name",
        "long_name",
        "units",
        "cell_methods",
        "description",
        "comment",
    ]

    # metadata fields that are formatted as free text (first letter capitalized)
    _text_fields = ["long_name", "description", "comment"]
    # Class attributes that are function (so we know which to convert to static methods)
    _funcs = ["compute"]
    # Mapping from name in the compute function to official (CMIP6) variable name
    _variable_mapping = {}

    # Will become the class's name
    identifier = None

    context = "none"
    src_freq = None

    # Global metadata (must be strings, not attributed to the output)
    realm = None
    title = ""
    abstract = ""
    keywords = ""
    references = ""
    notes = ""
    _version_deprecated = ""

    # Note: typing and class types in this call signature will cause errors with sphinx-autodoc-typehints
    # See: https://github.com/tox-dev/sphinx-autodoc-typehints/issues/186#issuecomment-1450739378
    _all_parameters: dict = {}
    """A dictionary mapping metadata about the input parameters to the indicator.

    Keys are the arguments of the "compute" function. All parameters are listed, even
    those "injected", absent from the indicator's call signature. All are instances of
    :py:class:`xsdba.indicator.Parameter`.
    """

    # Note: typing and class types in this call signature will cause errors with sphinx-autodoc-typehints
    # See: https://github.com/tox-dev/sphinx-autodoc-typehints/issues/186#issuecomment-1450739378
    cf_attrs: list[dict[str, str]] = None
    """A list of metadata information for each output of the indicator.

    It minimally contains a "var_name" entry, and may contain : "standard_name", "long_name",
    "units", "cell_methods", "description" and "comment" on official xclim indicators. Other
    fields could also be present if the indicator was created from outside xclim.

    var_name:
      Output variable(s) name(s). For derived single-output indicators, this field is not
      inherited from the parent indicator and defaults to the identifier.
    standard_name:
      Variable name, must be in the CF standard names table (this is not checked).
    long_name:
      Descriptive variable name. Parsed from `compute` docstring if not given.
      (first line after the output dtype, only works on single output function).
    units:
      Representative units of the physical quantity.
    cell_methods:
      List of blank-separated words of the form "name: method". Must respect the
      CF-conventions and vocabulary (not checked).
    description:
      Sentence(s) meant to clarify the qualifiers of the fundamental quantities, such as which
      surface a quantity is defined on or what the flux sign conventions are.
    comment:
      Miscellaneous information about the data or methods used to produce it.
    """

    def __new__(cls, **kwds):  # noqa: C901
        """Create subclass from arguments."""
        identifier = kwds.get("identifier", cls.identifier)
        if identifier is None:
            raise AttributeError("`identifier` has not been set.")

        if "compute" in kwds:
            # Parsed parameters and metadata override parent's params entirely.
            parameters, docmeta = cls._parse_indice(
                kwds["compute"], kwds.get("parameters", {})
            )
            for name, value in docmeta.items():
                # title, abstract, references, notes, long_name
                kwds.setdefault(name, value)

            # Inject parameters (subclasses can override or extend this through _injected_parameters)
            for name, param in cls._injected_parameters():
                if name in parameters:
                    raise ValueError(
                        f"Class {cls.__name__} can't wrap indices that have a `{name}`"
                        " argument as it conflicts with arguments it injects."
                    )
                parameters[name] = param
        else:  # inherit parameters from base class
            parameters = deepcopy(cls._all_parameters)

        # Update parameters with passed parameters
        cls._update_parameters(parameters, kwds.pop("parameters", {}))

        # Input variable mapping (to change variable names in signature and expected units/cf attrs).
        cls._parse_var_mapping(kwds.pop("input", {}), parameters, kwds)

        # Raise on incorrect params, sort params, modify var defaults in-place if needed
        parameters = cls._ensure_correct_parameters(parameters)

        # If needed, wrap compute with declare units
        if "compute" in kwds:
            if not hasattr(kwds["compute"], "in_units") and "_variable_mapping" in kwds:
                # We actually need the inverse mapping (to get cmip6 name -> arg name)
                inv_var_map = dict(map(reversed, kwds["_variable_mapping"].items()))
                # parameters has already been update above.
                # kwds["compute"] = declare_units(
                #     **{
                #         inv_var_map[k]: m.units
                #         for k, m in parameters.items()
                #         if "units" in m and k in inv_var_map
                #     }
                # )(kwds["compute"])

            if hasattr(kwds["compute"], "in_units"):
                varmap = kwds.get("_variable_mapping", {})
                for name, unit in kwds["compute"].in_units.items():
                    parameters[varmap.get(name, name)].units = unit

        # All updates done.
        kwds["_all_parameters"] = parameters

        # Parse kwds to organize `cf_attrs`
        # And before converting callables to static methods
        kwds["cf_attrs"] = cls._parse_output_attrs(kwds, identifier)
        # Parse keywords
        if "keywords" in kwds:
            kwds["keywords"] = cls.keywords + " " + kwds.get("keywords")

        # Convert function objects to static methods.
        for key in cls._funcs:
            if key in kwds and callable(kwds[key]):
                kwds[key] = staticmethod(kwds[key])

        # ADAPT: Get rid of this
        # Infer realm for built-in xclim instances
        # if cls.__module__.startswith(__package__.split(".", maxsplit=1)[0]):
        #     xclim_realm = cls.__module__.split(".")[2]
        # else:
        xclim_realm = None

        # ADAPT: Get rid of this
        # Priority given to passed realm -> parent's realm -> location of the class declaration (official inds only)
        kwds.setdefault("realm", cls.realm or xclim_realm)
        # if kwds["realm"] not in ["atmos", "seaIce", "land", "ocean", "generic"]:
        #     raise AttributeError(
        #         "Indicator's realm must be given as one of 'atmos', 'seaIce', 'land', 'ocean' or 'generic'"
        #     )

        # Create new class object
        new = type(identifier.upper(), (cls,), kwds)

        # Forcing the module is there so YAML-generated submodules are correctly seen by IndicatorRegistrar.
        if kwds.get("module") is not None:
            new.__module__ = f"xclim.indicators.{kwds['module']}"
        else:
            # If the module was not forced, set the module to the base class' module.
            # Otherwise, all indicators will have module `xsdba.indicator`.
            new.__module__ = cls.__module__

        #  Add the created class to the registry
        # This will create an instance from the new class and call __init__.
        return super().__new__(new)

    @staticmethod
    def _parse_indice(compute, passed_parameters):  # noqa: F841
        """Parse the compute function.

        - Metadata is extracted from the docstring
        - Parameters are parsed from the docstring (description, choices), decorator (units), signature (kind, default)

        'passed_parameters' is only needed when compute is a generic function
        (not decorated by `declare_units`) and it takes a string parameter. In that case
        we need to check if that parameter has units (which have been passed explicitly).
        """
        docmeta = parse_doc(compute.__doc__)
        params_dict = docmeta.pop("parameters", {})  # override parent's parameters

        compute_sig = signature(compute)
        # Check that the `Parameters` section of the docstring does not include parameters
        # that are not in the `compute` function signature.
        if not set(params_dict.keys()).issubset(compute_sig.parameters.keys()):
            raise ValueError(
                f"Malformed docstring on {compute} : the parameters "
                f"{set(params_dict.keys()) - set(compute_sig.parameters.keys())} "
                "are absent from the signature."
            )
        for name, param in compute_sig.parameters.items():
            meta = params_dict.setdefault(name, {})
            meta["default"] = param.default
            meta["kind"] = infer_kind_from_parameter(param)

        parameters = {name: Parameter(**param) for name, param in params_dict.items()}
        return parameters, docmeta

    @classmethod
    def _injected_parameters(cls):
        """Create a list of tuples for arguments to inject, (name, Parameter)."""
        return [
            (
                "ds",
                Parameter(
                    kind=InputKind.DATASET,
                    default=None,
                    description="A dataset with the variables given by name.",
                ),
            )
        ]

    @classmethod
    def _update_parameters(cls, parameters, passed):
        """Update parameters with the ones passed."""
        try:
            for key, val in passed.items():
                if isinstance(val, dict) and Parameter.is_parameter_dict(val):
                    # modified meta
                    parameters[key].update(val)
                elif key in parameters:
                    parameters[key].value = val
                else:
                    raise KeyError(key)
        except KeyError as err:
            raise ValueError(
                f"Parameter {err} was passed but it does not exist on the "
                f"compute function (not one of {parameters.keys()})"
            ) from err

    @classmethod
    def _parse_var_mapping(cls, variable_mapping, parameters, kwds):
        """Parse the variable mapping passed in `input` and update `parameters` in-place."""
        # Update parameters
        for old_name, new_name in variable_mapping.items():
            meta = parameters[new_name] = parameters.pop(old_name)
            # try:
            #     varmeta = VARIABLES[new_name]
            # except KeyError as err:
            #     raise ValueError(
            #         f"Compute argument {old_name} was mapped to variable "
            #         f"{new_name} which is not understood by xclim or CMIP6. Please"
            #         " use names listed in `xclim.core.utils.VARIABLES`."
            #     ) from err
            # if meta.units is not _empty:
            #     try:
            #         check_units(varmeta["canonical_units"], meta.units)
            #     except ValidationError as err:
            #         raise ValueError(
            #             "When changing the name of a variable by passing `input`, "
            #             "the units dimensionality must stay the same. Got: old = "
            #             f"{meta.units}, new = {varmeta['canonical_units']}"
            #         ) from err
            # meta.units = varmeta.get("dimensions", varmeta["canonical_units"])
            # meta.description = varmeta["description"]

        if variable_mapping:
            # Update mapping attribute
            new_variable_mapping = deepcopy(cls._variable_mapping)
            new_variable_mapping.update(variable_mapping)
            kwds["_variable_mapping"] = new_variable_mapping

    @classmethod
    def _ensure_correct_parameters(cls, parameters):
        """Ensure the parameters are correctly set and ordered."""
        # Set default values, otherwise the signature binding chokes
        # on missing arguments when passing only `ds`.
        for name, meta in parameters.items():
            if not meta.injected:
                if meta.kind == InputKind.OPTIONAL_VARIABLE:
                    meta.default = None
                elif meta.kind in [InputKind.VARIABLE]:
                    meta.default = name

        # Sort parameters : Var, Opt Var, all params, ds, injected params.
        def sortkey(kv):
            if not kv[1].injected:
                if kv[1].kind in [
                    InputKind.VARIABLE,
                    InputKind.OPTIONAL_VARIABLE,
                    InputKind.KWARGS,
                ]:
                    return kv[1].kind
                return 2
            return 99

        return dict(sorted(parameters.items(), key=sortkey))

    @classmethod
    def _parse_output_attrs(  # noqa: C901
        cls, kwds: dict[str, Any], identifier: str
    ) -> list[dict[str, str | Callable]]:
        """CF-compliant metadata attributes for all output variables."""
        parent_cf_attrs = cls.cf_attrs
        cf_attrs = kwds.get("cf_attrs")
        if isinstance(cf_attrs, dict):
            # Single output indicator, but we store as a list anyway.
            cf_attrs = [cf_attrs]
        elif cf_attrs is None:
            # Attributes were passed the "old" way, with lists or strings directly (only _cf_names)
            # We need to get the number of outputs first, defaulting to the length of parent's cf_attrs or 1
            n_outs = len(parent_cf_attrs) if parent_cf_attrs is not None else 1
            for name in cls._cf_names:
                arg = kwds.get(name)
                if isinstance(arg, (tuple, list)):
                    n_outs = len(arg)

            # Populate new cf_attrs from parsing cf_names passed directly.
            cf_attrs = [{} for _ in range(n_outs)]
            for name in cls._cf_names:
                values = kwds.pop(name, None)
                if values is None:  # None passed, skip
                    continue
                if not isinstance(values, (tuple, list)):
                    # a single string or callable, same for all outputs
                    values = [values] * n_outs
                elif len(values) != n_outs:  # A sequence of the wrong length.
                    raise ValueError(
                        f"Attribute {name} has {len(values)} elements but xclim expected {n_outs}."
                    )
                for attrs, value in zip(cf_attrs, values):
                    if value:  # Skip the empty ones (None or "")
                        attrs[name] = value
        # else we assume a list of dicts

        # For single output, var_name defaults to identifier.
        if len(cf_attrs) == 1 and "var_name" not in cf_attrs[0]:
            cf_attrs[0]["var_name"] = identifier

        # update from parent, if they have the same length.
        if parent_cf_attrs is not None and len(parent_cf_attrs) == len(cf_attrs):
            for old, new in zip(parent_cf_attrs, cf_attrs):
                for attr, value in old.items():
                    new.setdefault(attr, value)

        # check if we have var_names for everybody
        for i, var in enumerate(cf_attrs, start=1):
            if "var_name" not in var:
                raise ValueError(f"Output #{i} is missing a var_name! Got: {var}.")

        return cf_attrs

    @classmethod
    def from_dict(
        cls,
        data: dict,
        identifier: str,
        module: str | None = None,
    ):
        """Create an indicator subclass and instance from a dictionary of parameters.

        Most parameters are passed directly as keyword arguments to the class constructor, except:

        - "base" : A subclass of Indicator or a name of one listed in
          :py:data:`xsdba.indicator.registry` or
          :py:data:`xsdba.indicator.base_registry`. When passed, it acts as if
          `from_dict` was called on that class instead.
        - "compute" : A string function name translates to a
          :py:mod:`xclim.indices.generic` or :py:mod:`xclim.indices` function.

        Parameters
        ----------
        data: dict
          The exact structure of this dictionary is detailed in the submodule documentation.
        identifier : str
          The name of the subclass and internal indicator name.
        module : str
          The module name of the indicator. This is meant to be used only if the indicator
          is part of a dynamically generated submodule, to override the module of the base class.
        """
        data = data.copy()
        if "base" in data:
            if isinstance(data["base"], str):
                parts = data["base"].split(".")
                registry_id = ".".join([*parts[:-1], parts[-1].upper()])
                cls = registry.get(registry_id, base_registry.get(data["base"]))
                if cls is None:
                    raise ValueError(
                        f"Requested base class {data['base']} is neither in the "
                        "indicators registry nor in base classes registry."
                    )
            else:
                cls = data["base"]

        compute = data.get("compute", None)
        # data.compute refers to a function in xclim.indices.generic or xclim.indices (in this order of priority).
        # It can also directly be a function (like if a module was passed to build_indicator_module_from_yaml)
        if isinstance(compute, str):
            compute_func = getattr(
                indices.generic, compute, getattr(indices, compute, None)
            )
            if compute_func is None:
                raise ImportError(
                    f"Indice function {compute} not found in xclim.indices or "
                    "xclim.indices.generic."
                )
            data["compute"] = compute_func

        return cls(identifier=identifier, module=module, **data)

    def __init__(self, **kwds):
        """Run checks and organizes the metadata."""
        # keywords of kwds that are class attributes have already been set in __new__
        self._check_identifier(self.identifier)

        # Validation is done : register the instance.
        super().__init__()

        self.__signature__ = self._gen_signature()

        # Generate docstring
        self.__doc__ = generate_indicator_docstring(self)

    def _gen_signature(self):
        """Generate the correct signature."""
        # Update call signature
        variables = []
        parameters = []
        compute_sig = signature(self.compute)
        for name, meta in self.parameters.items():
            if meta.kind in [
                InputKind.VARIABLE,
                InputKind.OPTIONAL_VARIABLE,
            ]:
                annot = Union[DataArray, str]
                if meta.kind == InputKind.OPTIONAL_VARIABLE:
                    annot = Optional[annot]
                variables.append(
                    _Parameter(
                        name,
                        kind=_Parameter.POSITIONAL_OR_KEYWORD,
                        default=meta.default,
                        annotation=annot,
                    )
                )
            elif meta.kind == InputKind.KWARGS:
                parameters.append(_Parameter(name, kind=_Parameter.VAR_KEYWORD))
            elif meta.kind == InputKind.DATASET:
                parameters.append(
                    _Parameter(
                        name,
                        kind=_Parameter.KEYWORD_ONLY,
                        annotation=Dataset,
                        default=meta.default,
                    )
                )
            else:
                parameters.append(
                    _Parameter(
                        name,
                        kind=_Parameter.KEYWORD_ONLY,
                        default=meta.default,
                        annotation=compute_sig.parameters[name].annotation,
                    )
                )

        ret_ann = DataArray if self.n_outs == 1 else tuple[(DataArray,) * self.n_outs]
        return Signature(variables + parameters, return_annotation=ret_ann)

    def __call__(self, *args, **kwds):
        """Call function of Indicator class."""
        # Put the variables in `das`, parse them according to the following annotations:
        #     das : OrderedDict of variables (required + non-None optionals)
        #     params : OrderedDict of parameters (var_kwargs as a single argument, if any)

        if self._version_deprecated:
            self._show_deprecation_warning()

        das, params, dsattrs = self._parse_variables_from_call(args, kwds)

        if OPTIONS[KEEP_ATTRS] is True or (
            OPTIONS[KEEP_ATTRS] == "xarray"
            and xarray.core.options._get_keep_attrs(False)
        ):
            out_attrs = xarray.core.merge.merge_attrs(
                [da.attrs for da in das.values()], "drop_conflicts"
            )
            out_attrs.pop("units", None)
        else:
            out_attrs = {}
        out_attrs = [out_attrs.copy() for i in range(self.n_outs)]

        # Get correct variable names for the compute function.
        inv_var_map = dict(map(reversed, self._variable_mapping.items()))
        compute_das = {inv_var_map.get(nm, nm): das[nm] for nm in das}

        # Compute the indicator values, ignoring NaNs and missing values.
        # Filter the passed parameters to only keep the ones needed by compute.
        kwargs = {}
        var_kwargs = {}
        for nm, pa in signature(self.compute).parameters.items():
            if pa.kind == _Parameter.VAR_KEYWORD:
                var_kwargs = params[nm]
            elif nm not in compute_das and nm in params:
                kwargs[nm] = params[nm]

        with xarray.set_options(keep_attrs=False):
            outs = self.compute(**compute_das, **kwargs, **var_kwargs)

        if isinstance(outs, DataArray):
            outs = [outs]

        if len(outs) != self.n_outs:
            raise ValueError(
                f"Indicator {self.identifier} was wrongly defined. Expected "
                f"{self.n_outs} outputs, got {len(outs)}."
            )

        # Metadata attributes from templates
        var_id = None
        for out, attrs, base_attrs in zip(outs, out_attrs, self.cf_attrs):
            if self.n_outs > 1:
                var_id = base_attrs["var_name"]
            attrs.update(units=out.units)
            attrs.update(
                self._update_attrs(
                    params.copy(),
                    das,
                    base_attrs,
                    names=self._cf_names,
                    var_id=var_id,
                )
            )

        # Convert to output units
        outs = [
            convert_units_to(out, attrs["units"]) for out, attrs in zip(outs, out_attrs)
        ]

        outs = self._postprocess(outs, das, params)

        # Update variable attributes
        for out, attrs in zip(outs, out_attrs):
            var_name = attrs.pop("var_name")
            out.attrs.update(attrs)
            out.name = var_name

        if OPTIONS[AS_DATASET]:
            out = Dataset({o.name: o for o in outs})
            if OPTIONS[KEEP_ATTRS] is True or (
                OPTIONS[KEEP_ATTRS] == "xarray"
                and xarray.core.options._get_keep_attrs(False)
            ):
                out.attrs.update(dsattrs)
            out.attrs["history"] = update_history(
                self._history_string(das, params),
                out,
                new_name=self.identifier,
            )
            return out

        # Return a single DataArray in case of single output, otherwise a tuple
        if self.n_outs == 1:
            return outs[0]
        return tuple(outs)

    def _parse_variables_from_call(self, args, kwds) -> tuple[OrderedDict, dict]:
        """Extract variable and optional variables from call arguments."""
        # Bind call arguments to `compute` arguments and set defaults.
        ba = self.__signature__.bind(*args, **kwds)
        ba.apply_defaults()

        # Assign inputs passed as strings from ds.
        self._assign_named_args(ba)

        # Extract variables + inject injected
        das = OrderedDict()
        params = ba.arguments.copy()
        for name, param in self._all_parameters.items():
            if not param.injected:
                # If a variable pop the arg
                if is_percentile_dataarray(params[name]):
                    # duplicate percentiles DA in both das and params
                    das[name] = params[name]
                elif param.kind in [InputKind.VARIABLE, InputKind.OPTIONAL_VARIABLE]:
                    data = params.pop(name)
                    # If a non-optional variable OR None, store the arg
                    if param.kind == InputKind.VARIABLE or data is not None:
                        das[name] = data
            else:
                params[name] = param.value

        ds = ba.arguments.get("ds")
        dsattrs = ds.attrs if ds is not None else {}
        return das, params, dsattrs

    def _assign_named_args(self, ba):
        """Assign inputs passed as strings from ds."""
        ds = ba.arguments.get("ds")

        for name, val in ba.arguments.items():
            kind = self.parameters[name].kind

            if kind <= InputKind.OPTIONAL_VARIABLE:
                if isinstance(val, str) and ds is None:
                    raise ValueError(
                        "Passing variable names as string requires giving the `ds` "
                        f"dataset (got {name}='{val}')"
                    )
                if (isinstance(val, str) or val is None) and ds is not None:
                    # Set default name for DataArray
                    key = val or name

                    if key in ds:
                        ba.arguments[name] = ds[key]
                    elif kind == InputKind.VARIABLE:
                        raise MissingVariableError(
                            f"For input '{name}', variable '{key}' "
                            "was not found in the input dataset."
                        )

    def _postprocess(self, outs, das, params):
        """Run post-computation actions."""
        return outs

    def _bind_call(self, func, **das):
        """Call function using `__call__` `DataArray` arguments.

        This will try to bind keyword arguments to `func` arguments. If this fails,
        `func` is called with positional arguments only.

        Notes
        -----
        This method is used to support two main use cases.

        In use case #1, we have two compute functions with arguments in a different order:
            `func1(tasmin, tasmax)` and `func2(tasmax, tasmin)`

        In use case #2, we have two compute functions with arguments that have different names:
            `generic_func(da)` and `custom_func(tas)`

        Passing a dictionary of arguments will solve #1, but not #2.
        """
        # First try to bind arguments to function.
        try:
            ba = signature(func).bind(**das)
        except TypeError:
            # If this fails, simply call the function using positional arguments
            return func(*das.values())
        else:
            # Call the func using bound arguments
            return func(*ba.args, **ba.kwargs)

    @classmethod
    def _get_translated_metadata(
        cls, locale, var_id=None, names=None, append_locale_name=True
    ):
        """Get raw translated metadata for the current indicator and a given locale.

        All available translated metadata from the current indicator and those it is
        based on are merged, with the highest priority set to the current one.
        """
        var_id = var_id or ""
        if var_id:
            var_id = "." + var_id

        family_tree = []
        cl = cls
        while hasattr(cl, "_registry_id"):
            family_tree.append(cl._registry_id + var_id)
            # The indicator mechanism always has single inheritance.
            cl = cl.__bases__[0]

        return get_local_attrs(
            family_tree,
            locale,
            names=names,
            append_locale_name=append_locale_name,
        )

    def _update_attrs(
        self,
        args: dict[str, Any],
        das: dict[str, DataArray],
        attrs: dict[str, str],
        var_id: str | None = None,
        names: Sequence[str] | None = None,
    ):
        """Format attributes with the run-time values of `compute` call parameters.

        Cell methods and history attributes are updated, adding to existing values.
        The language of the string is taken from the `OPTIONS` configuration dictionary.

        Parameters
        ----------
        args : dict[str, Any]
            Keyword arguments of the `compute` call.
        das : dict[str, DataArray]
            Input arrays.
        attrs : dict[str, str]
            The attributes to format and update.
        var_id : str
            The identifier to use when requesting the attributes translations.
            Defaults to the class name (for the translations) or the `identifier` field of
            the class (for the history attribute).
            If given, the identifier will be converted to uppercase to get the translation
            attributes. This is meant for multi-outputs indicators.
        names : sequence of str, optional
            List of attribute names for which to get a translation.

        Returns
        -------
        dict
            Attributes with {} expressions replaced by call argument values. With updated `cell_methods` and `history`.
            `cell_methods` is not added if `names` is given and those not contain `cell_methods`.
        """
        # FIXME: Some tests fail without this, the groups are not properly parsed before
        # e.g. test_properties::TestProperties::test_return_value fails
        if "group" in args and isinstance(args["group"], str):
            args["group"] = Grouper(args["group"])

        out = self._format(attrs, args)
        for locale in OPTIONS[METADATA_LOCALES]:
            out.update(
                self._format(
                    self._get_translated_metadata(
                        locale, var_id=var_id, names=names or list(attrs.keys())
                    ),
                    args=args,
                    formatter=get_local_formatter(locale),
                )
            )

        # Get history and cell method attributes from source data
        attrs = defaultdict(str)
        if names is None or "cell_methods" in names:
            attrs["cell_methods"] = merge_attributes(
                "cell_methods", new_line=" ", missing_str=None, **das
            )
            if "cell_methods" in out:
                attrs["cell_methods"] += " " + out.pop("cell_methods")

        attrs["history"] = update_history(
            self._history_string(das, args),
            new_name=out.get("var_name"),
            **das,
        )

        attrs.update(out)
        return attrs

    def _history_string(self, das, params):
        kwargs = dict(**das)
        for k, v in params.items():
            if self._all_parameters[k].injected:
                continue
            if self._all_parameters[k].kind == InputKind.KWARGS:
                kwargs.update(**v)
            elif self._all_parameters[k].kind != InputKind.DATASET:
                kwargs[k] = v
        return gen_call_string(self._registry_id, **kwargs)

    @staticmethod
    def _check_identifier(identifier: str) -> None:
        """Verify that the identifier is a proper slug."""
        if not re.match(r"^[-\w]+$", identifier):
            warnings.warn(
                "The identifier contains non-alphanumeric characters. It could make "
                "life difficult for downstream software reusing this class.",
                UserWarning,
            )

    @classmethod
    def translate_attrs(cls, locale: str | Sequence[str], fill_missing: bool = True):
        """Return a dictionary of unformatted translated translatable attributes.

        Translatable attributes are defined in :py:const:`xsdba.locales.TRANSLATABLE_ATTRS`.

        Parameters
        ----------
        locale : str or sequence of str
            The POSIX name of the locale or a tuple of a locale name and a path to a json file defining translations.
            See `xclim.locale` for details.
        fill_missing : bool
            If True (default) fill the missing attributes by their english values.
        """

        def _translate(cf_attrs, names, var_id=None):
            attrs = cls._get_translated_metadata(
                locale,
                var_id=var_id,
                names=names,
                append_locale_name=False,
            )
            if fill_missing:
                for name in names:
                    if name not in attrs and cf_attrs.get(name):
                        attrs[name] = cf_attrs.get(name)
            return attrs

        # Translate global attrs
        attrs = _translate(
            cls.__dict__,
            # Translate only translatable attrs that are not variable attrs
            set(TRANSLATABLE_ATTRS).difference(set(cls._cf_names)),
        )
        # Translate variable attrs
        attrs["cf_attrs"] = []
        var_id = None
        for cf_attrs in cls.cf_attrs:  # Translate for each variable
            if len(cls.cf_attrs) > 1:
                var_id = cf_attrs["var_name"]
            attrs["cf_attrs"].append(
                _translate(
                    cf_attrs,
                    set(TRANSLATABLE_ATTRS).intersection(cls._cf_names),
                    var_id=var_id,
                )
            )
        return attrs

    def json(self, args=None):
        """Return a serializable dictionary representation of the class.

        Parameters
        ----------
        args : mapping, optional
            Arguments as passed to the call method of the indicator.
            If not given, the default arguments will be used when formatting the attributes.

        Notes
        -----
        This is meant to be used by a third-party library wanting to wrap this class into another interface.
        """
        names = ["identifier", "title", "abstract", "keywords"]
        out = {key: getattr(self, key) for key in names}
        out = self._format(out, args)

        # Format attributes
        out["outputs"] = [self._format(attrs, args) for attrs in self.cf_attrs]
        out["notes"] = self.notes

        # We need to deepcopy, otherwise empty defaults get overwritten!
        # All those tweaks are to ensure proper serialization of the returned dictionary.
        out["parameters"] = {
            k: p.asdict() if not p.injected else deepcopy(p.value)
            for k, p in self._all_parameters.items()
        }
        for name, param in list(out["parameters"].items()):
            if not self._all_parameters[name].injected:
                param["kind"] = param["kind"].value  # Get the int.
                if "choices" in param:  # A set is stored, convert to list
                    param["choices"] = list(param["choices"])
                if param["default"] is _empty_default:
                    del param["default"]
            elif callable(param):  # Rare special case (doy_qmax and doy_qmin).
                out["parameters"][name] = f"{param.__module__}.{param.__name__}"

        return out

    @classmethod
    def _format(
        cls,
        attrs: dict,
        args: dict | None = None,
        formatter: AttrFormatter = default_formatter,
    ) -> dict:
        """Format attributes including {} tags with arguments.

        Parameters
        ----------
        attrs : dict
            Attributes containing tags to replace with arguments' values.
        args : dict, optional
            Function call arguments. If not given, the default arguments will be used when formatting the attributes.
        formatter : AttrFormatter
            Plaintext mappings for indicator attributes.

        Returns
        -------
        dict
        """
        # Use defaults
        if args is None:
            args = {
                k: p.default if not p.injected else p.value
                for k, p in cls._all_parameters.items()
            }

        # Prepare arguments
        mba = {}
        # Add formatting {} around values to be able to replace them with _attrs_mapping using format.
        for k, v in args.items():
            if isinstance(v, units.Quantity):
                mba[k] = f"{v:g~P}"
            elif isinstance(v, (int, float)):
                mba[k] = f"{v:g}"
            # TODO: What about InputKind.NUMBER_SEQUENCE
            elif k == "indexer":
                if v and v not in [_empty, _empty_default]:
                    dk, dv = v.copy().popitem()
                    if dk == "month":
                        dv = f"m{dv}"
                    elif dk in ("doy_bounds", "date_bounds"):
                        dv = f"{dv[0]} to {dv[1]}"
                    mba["indexer"] = dv
                else:
                    mba["indexer"] = args.get("freq") or "YS"
            elif is_percentile_dataarray(v):
                mba.update(get_percentile_metadata(v, k))
            elif (
                isinstance(v, DataArray)
                and cls._all_parameters[k].kind == InputKind.QUANTIFIED
            ):
                mba[k] = "<an array>"
            else:
                mba[k] = v
        out = {}
        for key, val in attrs.items():
            if callable(val):
                val = val(**mba)

            out[key] = formatter.format(val, **mba)

            if key in cls._text_fields:
                out[key] = out[key].strip().capitalize()

        return out

    # The following static methods are meant to be replaced to define custom indicators.
    @staticmethod
    def compute(*args, **kwds):
        """Compute the indicator.

        This would typically be a function from `xclim.indices`.
        """
        raise NotImplementedError

    def __getattr__(self, attr):
        """Return the attribute."""
        if attr in self._cf_names:
            out = [meta.get(attr, "") for meta in self.cf_attrs]
            if len(out) == 1:
                return out[0]
            return out
        raise AttributeError(attr)

    @property
    def n_outs(self):
        """Return the length of all cf_attrs."""
        return len(self.cf_attrs)

    @property
    def parameters(self):
        """Create a dictionary of controllable parameters.

        Similar to :py:attr:`Indicator._all_parameters`, but doesn't include injected parameters.
        """
        return {
            name: param
            for name, param in self._all_parameters.items()
            if not param.injected
        }

    @property
    def injected_parameters(self):
        """Return a dictionary of all injected parameters.

        Opposite of :py:meth:`Indicator.parameters`.
        """
        return {
            name: param.value
            for name, param in self._all_parameters.items()
            if param.injected
        }

    @property
    def is_generic(self):
        """Return True if the indicator is "generic", meaning that it can accept variables with any units."""
        return not hasattr(self.compute, "in_units")

    def _show_deprecation_warning(self):
        warnings.warn(
            f"`{self.title}` is deprecated as of `xclim` v{self._version_deprecated} and will be removed "
            "in a future release. See the `xclim` release notes for more information: "
            f"https://xclim.readthedocs.io/en/stable/history.html",
            FutureWarning,
            stacklevel=3,
        )
