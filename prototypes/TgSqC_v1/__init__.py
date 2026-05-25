from base.patches import register_patch
from .patch import ColorTogglePatch

register_patch("TgSqC_v1", ColorTogglePatch)

__all__ = ["ColorTogglePatch"]
