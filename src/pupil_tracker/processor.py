"""Frame processor for mapping gaze to pixels and extracting gaze regions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pupil_tracker.client import FrameData, GazeData


@dataclass(frozen=True)
class GazeRegion:
    """A region of the frame around the gaze point."""

    center_x: int
    center_y: int
    region: NDArray[np.uint8]  # BGR image patch
    frame_width: int
    frame_height: int
    timestamp: float
    confidence: float


class FrameProcessor:
    """Processes frames and gaze data to extract regions around gaze points."""

    def __init__(self, region_size: int = 50) -> None:
        """Initialize the frame processor.

        Args:
            region_size: Size of the square region to extract around gaze point.
        """
        self._region_size = region_size
        self._last_frame: FrameData | None = None
        self._last_gaze: GazeData | None = None

    @property
    def region_size(self) -> int:
        """Get the configured region size."""
        return self._region_size

    @property
    def last_frame(self) -> FrameData | None:
        """Get the last received frame."""
        return self._last_frame

    @property
    def last_gaze(self) -> GazeData | None:
        """Get the last received gaze data."""
        return self._last_gaze

    def update_frame(self, frame: FrameData) -> None:
        """Update the current frame.

        Args:
            frame: The new frame data from Pupil Capture.
        """
        self._last_frame = frame

    def update_gaze(self, gaze: GazeData, min_confidence: float = 0.5) -> bool:
        """Update the current gaze data if it's valid.

        Args:
            gaze: The new gaze data from Pupil Capture.
            min_confidence: Minimum confidence threshold (0-1).

        Returns:
            True if the gaze was accepted, False if rejected.
        """
        # Filter out low-confidence gaze
        if gaze.confidence < min_confidence:
            return False

        # Filter out out-of-bounds gaze (should be 0-1 normalized)
        x, y = gaze.norm_pos
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False

        self._last_gaze = gaze
        return True

    def norm_to_pixel(
        self,
        norm_x: float,
        norm_y: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates.

        Note: Pupil Core uses (0,0) at bottom-left, but OpenCV uses (0,0) at top-left.
        We flip the Y coordinate here.

        Args:
            norm_x: Normalized X coordinate (0-1).
            norm_y: Normalized Y coordinate (0-1).
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            Tuple of (pixel_x, pixel_y) in OpenCV coordinate system.
        """
        pixel_x = int(norm_x * width)
        pixel_y = int((1.0 - norm_y) * height)  # Flip Y for OpenCV
        return pixel_x, pixel_y

    def extract_region(self) -> GazeRegion | None:
        """Extract the region around the current gaze point.

        Returns:
            GazeRegion if both frame and gaze data are available, None otherwise.
        """
        if self._last_frame is None or self._last_gaze is None:
            return None

        frame = self._last_frame
        gaze = self._last_gaze

        # Convert normalized gaze to pixel coordinates
        pixel_x, pixel_y = self.norm_to_pixel(
            gaze.norm_pos[0],
            gaze.norm_pos[1],
            frame.width,
            frame.height,
        )

        # Calculate region bounds with clamping
        half_size = self._region_size // 2

        # Clamp to frame boundaries
        x_start = max(0, pixel_x - half_size)
        x_end = min(frame.width, pixel_x + half_size)
        y_start = max(0, pixel_y - half_size)
        y_end = min(frame.height, pixel_y + half_size)

        # Extract the region
        region = frame.data[y_start:y_end, x_start:x_end].copy()

        return GazeRegion(
            center_x=pixel_x,
            center_y=pixel_y,
            region=region,
            frame_width=frame.width,
            frame_height=frame.height,
            timestamp=gaze.timestamp,
            confidence=gaze.confidence,
        )


