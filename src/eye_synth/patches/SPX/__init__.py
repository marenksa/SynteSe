"""SPX patch: gaze coordinates + velocity → pitch/loudness stream."""

from eye_synth.patches.base import register_patch
from eye_synth.patches.SPX.patch import GazeStreamPatch

register_patch("SPX_v1", GazeStreamPatch)
register_patch("SPX_v2", GazeStreamPatch)
register_patch("SPX_v3", GazeStreamPatch)

__all__ = ["GazeStreamPatch"]
