"""RAVE_v1 patch: eye signals → RAVE latent dimensions for nn~."""

from base.patches import register_patch
from .patch import RAVEPatch

register_patch("RAVE_v1", RAVEPatch)

__all__ = ["RAVEPatch"]
