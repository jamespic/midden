import sys

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    if sys.platform != "linux" and item.get_closest_marker("linux"):
        pytest.skip("Test requires Linux")
    if (
        min_python_version := item.get_closest_marker("min_python")
    ) and sys.version_info < min_python_version.args[0]:
        pytest.skip(f"Test requires Python {min_python_version.args[0]} or higher")
