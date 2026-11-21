# pmd_pycore/__init__.py

from importlib.metadata import version as get_version, PackageNotFoundError
from .version import __version__

try:
    installed_version = get_version("pmd-pycore")
    if installed_version != __version__:
        __version__ = installed_version
except PackageNotFoundError:
    pass  # Use the version from version.py

from . import core

__all__ = ["core", "__version__"]
