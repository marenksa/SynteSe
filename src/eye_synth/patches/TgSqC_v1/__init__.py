from eye_synth.patches.base import register_patch
from eye_synth.patches.TgSqC_v1.patch import ColorTogglePatch

register_patch("TgSqC_v1", ColorTogglePatch)

__all__ = ["ColorTogglePatch"]
