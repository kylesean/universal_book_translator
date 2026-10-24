"""Universal Book Translator (UBT) root package."""

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

# Filter known noisy third-party deprecations (e.g. jieba's legacy pkg_resources import)
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    # The packaging metadata is the only version source: restating the
    # version in ubt/cli or the API's OpenAPI info creates copies that are
    # free to drift on every release bump.
    __version__ = _distribution_version("universal-book-translator")
except PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0+unknown"
