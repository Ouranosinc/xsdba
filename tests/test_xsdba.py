#!/usr/bin/env python
"""Tests for `xsdba` package."""

from __future__ import annotations

import pathlib
from importlib.util import find_spec

from xsdba import xsdba  # noqa: F401


def test_package_metadata():
    """Test the package metadata."""
    project = find_spec("xsdba").submodule_search_locations[0]

    metadata = pathlib.Path(project).resolve().joinpath("__init__.py")

    with metadata.open() as f:
        contents = f.read()
        assert """Trevor James Smith""" in contents
        assert '__email__ = "smith.trevorj@ouranos.ca"' in contents
        assert '__version__ = "0.1.1-dev.0"' in contents
