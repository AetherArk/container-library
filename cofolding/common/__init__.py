"""Shared, dependency-light cofolding helpers.

Only pure-numpy modules belong here: the containers must be able to compute
derived metrics on a CPU-only host before the large prediction tensors are
dropped.
"""

from .ipsae import compute_interface_scores

__all__ = ["compute_interface_scores"]
