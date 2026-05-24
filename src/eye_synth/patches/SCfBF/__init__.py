"""SCfBF patch: eye confidence stream → PD confidence signal."""

from eye_synth.patches.base import register_patch
from eye_synth.patches.SCfBF.patch import ConfidenceStreamPatch

register_patch("SCfBF_v1", ConfidenceStreamPatch)
register_patch("SCfBF_v2", ConfidenceStreamPatch)

__all__ = ["ConfidenceStreamPatch"]
