"""RAVE_v1 patch: eye signals → RAVE latent dimensions for nn~."""

from eye_synth.patches.base import register_patch
from eye_synth.patches.RAVE_v1.patch import RAVEPatch

register_patch("RAVE_v1", RAVEPatch)

__all__ = ["RAVEPatch"]
