"""SPX patch: gaze coordinates + velocity → pitch/loudness stream."""

from base.patches import register_patch
from .patch import GazeStreamPatch

register_patch("SPX_v1", GazeStreamPatch)
register_patch("SPX_v2", GazeStreamPatch)
register_patch("SPX_v3", GazeStreamPatch)

__all__ = ["GazeStreamPatch"]
