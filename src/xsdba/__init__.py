"""Statistical correction and bias adjustment tools for xarray."""

###################################################################################
# Apache Software License 2.0
#
# Copyright (c) 2024, Ouranos Inc., Éric Dupuis, Trevor James Smith
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################

from __future__ import annotations

import importlib.util
import warnings

from xsdba import adjustment, base, detrending, processing, units, utils

xclim_installed = importlib.util.find_spec("xclim") is not None
if xclim_installed:
    from xsdba import (
        measures,
        properties,
    )

from xsdba.adjustment import *
from xsdba.base import Grouper, stack_periods, unstack_periods
from xsdba.options import set_options
from xsdba.processing import stack_variables, unstack_variables

# TODO: ISIMIP ? Used for precip freq adjustment in biasCorrection.R
# Hempel, S., Frieler, K., Warszawski, L., Schewe, J., & Piontek, F. (2013). A trend-preserving bias correction &ndash;
# The ISI-MIP approach. Earth System Dynamics, 4(2), 219–236. https://doi.org/10.5194/esd-4-219-2013
# If SBCK is installed, create adjustment classes wrapping SBCK's algorithms.
if hasattr(adjustment, "_generate_SBCK_classes"):
    for cls in adjustment._generate_SBCK_classes():
        adjustment.__dict__[cls.__name__] = cls

__author__ = """Éric Dupuis"""
__email__ = "dupuis.eric@ouranos.ca"
__version__ = "0.3.2"
