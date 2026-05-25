"""SCfBF patch: eye confidence stream → PD confidence signal."""

from base.patches import register_patch
from .patch import ConfidenceStreamPatch

register_patch("SCfBF_v1", ConfidenceStreamPatch)
register_patch("SCfBF_v2", ConfidenceStreamPatch)

__all__ = ["ConfidenceStreamPatch"]
