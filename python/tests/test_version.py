"""Package version metadata."""

from importlib.metadata import version

import refraction


def test_version_matches_distribution_metadata():
    assert refraction.__version__ == version("earth-refraction")
