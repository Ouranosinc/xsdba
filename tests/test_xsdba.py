#!/usr/bin/env python
"""Tests for `xsdba` package."""

from __future__ import annotations

import pathlib
from importlib.util import find_spec

from xsdba import xsdba  # noqa: F401


def test_package_metadata():
    """Test the package metadata."""
    project = find_spec("xsdba")

    assert project is not None
    assert project.submodule_search_locations is not None
    location = project.submodule_search_locations[0]

    metadata = pathlib.Path(location).resolve().joinpath("__init__.py")

    with metadata.open() as f:
        contents = f.read()
        assert """Éric Dupuis""" in contents
        assert '__email__ = "dupuis.eric@ouranos.ca"' in contents
        assert '__version__ = "0.3.2"' in contents
