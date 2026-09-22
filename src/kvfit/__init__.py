"""KV-cache and inference-memory planning from model metadata."""

from kvfit.architectures import estimate_cache
from kvfit.models import CacheEstimate, UnsupportedArchitecture

__all__ = ["CacheEstimate", "UnsupportedArchitecture", "estimate_cache"]
__version__ = "0.2.4"
