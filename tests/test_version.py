"""The package version must match pyproject.toml.

`__version__` is a literal, and a release bumps `pyproject.toml`. Nothing
connected the two, so it drifted: 0.10.0 and 0.11.0 both shipped
reporting "0.9.0". The publish workflow reads only `pyproject.toml`, so it
could not catch it either (branes-ai/graphs#268 F4 review).
"""

from __future__ import annotations

import pathlib
import re

import embodied_schemas


def _pyproject_version() -> str:
    text = (
        pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert match, "pyproject.toml has no top-level version"
    return match.group(1)


def test_version_matches_pyproject():
    assert embodied_schemas.__version__ == _pyproject_version(), (
        f"embodied_schemas.__version__ is {embodied_schemas.__version__!r} but "
        f"pyproject.toml says {_pyproject_version()!r}; a release must bump both"
    )


def test_version_is_exported():
    """Consumers read it off the package, so it is part of the API."""
    assert "__version__" in embodied_schemas.__all__
