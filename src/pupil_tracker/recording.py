"""Pupil Capture recording loader and data access."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import msgpack
import numpy as np


@dataclass
class RecordingInfo:
    """Metadata about a Pupil Capture recording."""

    path: Path
    duration_s: float
    recording_name: str
    start_time_s: float
    world_resolution: tuple[int, int]
    gaze_count: int
    frame_count: int


@dataclass
class GazeSample:
    """A single gaze sample from the recording."""

    timestamp: float
    norm_pos: tuple[float, float]  # Normalized position (0-1, origin bottom-left)
    confidence: float


class Recording:
    """Loads and provides access to Pupil Capture recording data."""

    def __init__(self, recording_path: str | Path):
        self.path = Path(recording_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Recording not found: {self.path}")

        # Load metadata
        self._load_info()

        # Load timestamps
        self.world_timestamps = np.load(self.path / "world_timestamps.npy")
        self.gaze_timestamps = np.load(self.path / "gaze_timestamps.npy")

        # Load gaze data
        self.gaze_data = self._load_gaze_data()

        # Video capture (lazy loaded)
        self._video: cv2.VideoCapture | None = None
        self._video_pos: int = 0  # Track current video position for efficient sequential reads

    def _load_info(self) -> None:
        """Load recording metadata from info.player.json."""
        info_path = self.path / "info.player.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            self.duration_s = info.get("duration_s", 0)
            self.recording_name = info.get("recording_name", self.path.name)
            self.start_time_s = info.get("start_time_synced_s", 0)
        else:
            self.duration_s = 0
            self.recording_name = self.path.name
            self.start_time_s = 0

        # Get world video resolution
        intrinsics_path = self.path / "world.intrinsics"
        if intrinsics_path.exists():
            with open(intrinsics_path, "rb") as f:
                intrinsics = msgpack.unpack(f, raw=False, strict_map_key=False)
                res = intrinsics.get("resolution", (1280, 720))
                self.world_resolution = (res[0], res[1])
        else:
            self.world_resolution = (1280, 720)

    def _load_gaze_data(self) -> list[GazeSample]:
        """Load all gaze samples from gaze.pldata."""
        gaze_path = self.path / "gaze.pldata"
        if not gaze_path.exists():
            return []

        samples = []
        with open(gaze_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
            for topic, payload in unpacker:
                data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                norm_pos = data.get("norm_pos", [0.5, 0.5])
                samples.append(
                    GazeSample(
                        timestamp=data.get("timestamp", 0),
                        norm_pos=(norm_pos[0], norm_pos[1]),
                        confidence=data.get("confidence", 0),
                    )
                )
        return samples

    @property
    def video(self) -> cv2.VideoCapture:
        """Get the world video capture object (lazy loaded)."""
        if self._video is None:
            video_path = self.path / "world.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"World video not found: {video_path}")
            self._video = cv2.VideoCapture(str(video_path))
        return self._video

    @property
    def fps(self) -> float:
        """Get the video frame rate."""
        return self.video.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Get total number of video frames."""
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_info(self) -> RecordingInfo:
        """Get recording metadata."""
        return RecordingInfo(
            path=self.path,
            duration_s=self.duration_s,
            recording_name=self.recording_name,
            start_time_s=self.start_time_s,
            world_resolution=self.world_resolution,
            gaze_count=len(self.gaze_data),
            frame_count=len(self.world_timestamps),
        )

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """Get a specific video frame by index.

        Optimized for sequential access - only seeks when necessary.
        """
        # Only seek if not at the expected position
        if frame_index != self._video_pos:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._video_pos = frame_index

        ret, frame = self.video.read()
        if ret:
            self._video_pos += 1
            return frame
        return None

    def read_next_frame(self) -> tuple[int, np.ndarray] | None:
        """Read the next frame sequentially (most efficient for playback).

        Returns:
            Tuple of (frame_index, frame) or None if no more frames.
        """
        ret, frame = self.video.read()
        if ret:
            frame_index = self._video_pos
            self._video_pos += 1
            return (frame_index, frame)
        return None

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame position."""
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._video_pos = frame_index

    def get_frame_timestamp(self, frame_index: int) -> float:
        """Get the timestamp for a specific frame."""
        if 0 <= frame_index < len(self.world_timestamps):
            return self.world_timestamps[frame_index]
        return 0

    def get_gaze_at_timestamp(
        self, timestamp: float, max_time_diff: float = 0.05
    ) -> GazeSample | None:
        """Get the closest gaze sample to a given timestamp.

        Args:
            timestamp: The target timestamp
            max_time_diff: Maximum time difference to accept (default 50ms)

        Returns:
            The closest gaze sample, or None if no sample within max_time_diff
        """
        if not self.gaze_data:
            return None

        # Binary search for closest timestamp
        idx = np.searchsorted(self.gaze_timestamps, timestamp)

        # Check neighbors for closest match
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(self.gaze_data):
            candidates.append(idx)

        best_sample = None
        best_diff = float("inf")

        for i in candidates:
            if i < len(self.gaze_data):
                diff = abs(self.gaze_data[i].timestamp - timestamp)
                if diff < best_diff and diff <= max_time_diff:
                    best_diff = diff
                    best_sample = self.gaze_data[i]

        return best_sample

    def get_gaze_for_frame(
        self, frame_index: int, max_time_diff: float = 0.05
    ) -> GazeSample | None:
        """Get gaze data for a specific video frame."""
        timestamp = self.get_frame_timestamp(frame_index)
        return self.get_gaze_at_timestamp(timestamp, max_time_diff)

    def gaze_to_pixel(
        self, gaze: GazeSample, frame_width: int | None = None, frame_height: int | None = None
    ) -> tuple[int, int]:
        """Convert normalized gaze position to pixel coordinates.

        Note: Pupil uses (0,0) at bottom-left, OpenCV uses top-left.
        This function handles the Y-axis flip.
        """
        width = frame_width or self.world_resolution[0]
        height = frame_height or self.world_resolution[1]

        x = int(gaze.norm_pos[0] * width)
        y = int((1 - gaze.norm_pos[1]) * height)  # Flip Y axis

        return (x, y)

    def close(self) -> None:
        """Release video resources."""
        if self._video is not None:
            self._video.release()
            self._video = None

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
