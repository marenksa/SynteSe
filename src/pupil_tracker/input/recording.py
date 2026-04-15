"""Pupil Capture recording loader and data access."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import msgpack
import numpy as np

from pupil_tracker.signals.eye_blinks import (
    BLINK_MAX_MS, FLUTTER_END_TIMEOUT_S, FLUTTER_MIN_BLINKS,
    FLUTTER_WINDOW_S, INTENTIONAL_MIN_MS,
    BlinkSample, BlinkType, FlutterEvent, classify_blink,
)

logger = logging.getLogger(__name__)


@dataclass
class RecordingInfo:
    """Metadata about a Pupil Capture recording."""

    path: Path
    duration_s: float
    recording_name: str
    start_time_s: float
    world_resolution: tuple[int, int]
    gaze_count: int
    fixation_count: int
    blink_count: int
    frame_count: int


@dataclass
class GazeSample:
    """A single gaze sample from the recording."""

    timestamp: float
    norm_pos: tuple[float, float]  # Normalized position (0-1, origin bottom-left)
    confidence: float


@dataclass
class FixationSample:
    """A fixation event from Pupil's fixation detection."""

    id: int
    timestamp: float  # Start timestamp
    duration: float  # Duration in ms
    norm_pos: tuple[float, float]  # Normalized position (0-1, origin bottom-left)
    dispersion: float  # Dispersion in degrees
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

        # Load gaze data (may filter to single eye for binocular recordings)
        self.gaze_data = self._load_gaze_data()

        # Build gaze_timestamps from loaded data (not from file, which may have both eyes)
        self.gaze_timestamps = np.array([g.timestamp for g in self.gaze_data])

        # Load fixation data
        self.fixation_data = self._load_fixation_data()

        # Load blink data (with duration classification)
        self.blink_data = self._load_blink_data()
        self.blink_timestamps = np.array([b.timestamp for b in self.blink_data])

        # Flutter detection from rapid blink bursts
        self.flutter_data = self._detect_flutter_from_blinks()
        self.flutter_timestamps = np.array(
            [f.timestamp for f in self.flutter_data]
        ) if self.flutter_data else np.array([])

        # Eye camera (lazy loaded)
        self._eye_video: cv2.VideoCapture | None = None
        self._eye_video_pos: int = 0
        self.eye_timestamps: np.ndarray | None = None
        self._load_eye_timestamps()

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
        """Load gaze samples from gaze.pldata.

        Handles both monocular and binocular recordings:
        - Prefers combined gaze (gaze.3d.01. or gaze.2d.01.) if available
        - For binocular, picks the eye with higher average confidence
        - Uses whatever is available for monocular recordings
        """
        gaze_path = self.path / "gaze.pldata"
        if not gaze_path.exists():
            return []

        # First pass: collect all samples grouped by topic
        samples_by_topic: dict[str, list[GazeSample]] = {}
        with open(gaze_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
            for topic, payload in unpacker:
                data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                norm_pos = data.get("norm_pos", [0.5, 0.5])
                sample = GazeSample(
                    timestamp=data.get("timestamp", 0),
                    norm_pos=(norm_pos[0], norm_pos[1]),
                    confidence=data.get("confidence", 0),
                )
                if topic not in samples_by_topic:
                    samples_by_topic[topic] = []
                samples_by_topic[topic].append(sample)

        def avg_confidence(samples: list[GazeSample]) -> float:
            if not samples:
                return 0.0
            return sum(s.confidence for s in samples) / len(samples)

        # Choose the best topic (prefer combined, then best single eye)
        # Priority: combined 3d > combined 2d > then pick by confidence
        combined_priority = ["gaze.3d.01.", "gaze.2d.01."]

        for preferred in combined_priority:
            if preferred in samples_by_topic:
                samples = samples_by_topic[preferred]
                if len(samples_by_topic) > 1:
                    other_topics = [t for t in samples_by_topic if t != preferred]
                    print(
                        f"[Recording] Using {preferred} ({len(samples)} samples), "
                        f"ignoring: {other_topics}"
                    )
                return sorted(samples, key=lambda s: s.timestamp)

        # No combined gaze - pick the eye with best average confidence
        if samples_by_topic:
            best_topic = max(
                samples_by_topic.keys(),
                key=lambda t: avg_confidence(samples_by_topic[t]),
            )
            samples = samples_by_topic[best_topic]
            best_conf = avg_confidence(samples)

            if len(samples_by_topic) > 1:
                other_topics = [t for t in samples_by_topic if t != best_topic]
                other_confs = [
                    f"{t}: {avg_confidence(samples_by_topic[t]):.2f}"
                    for t in other_topics
                ]
                print(
                    f"[Recording] Using {best_topic} (conf={best_conf:.2f}, "
                    f"{len(samples)} samples), ignoring: {other_confs}"
                )
            return sorted(samples, key=lambda s: s.timestamp)

        return []

    def _load_fixation_data(self) -> list[FixationSample]:
        """Load all fixation events from fixations.pldata.

        Fixations are deduplicated by ID since Pupil outputs multiple
        updates for each fixation as it progresses.
        """
        fixation_path = self.path / "fixations.pldata"
        if not fixation_path.exists():
            return []

        # Use dict to deduplicate by ID, keeping the last (most complete) entry
        fixations_by_id: dict[int, FixationSample] = {}

        with open(fixation_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
            for topic, payload in unpacker:
                data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                norm_pos = data.get("norm_pos", [0.5, 0.5])
                fixation_id = data.get("id", 0)

                fixations_by_id[fixation_id] = FixationSample(
                    id=fixation_id,
                    timestamp=data.get("timestamp", 0),
                    duration=data.get("duration", 0),
                    norm_pos=(norm_pos[0], norm_pos[1]),
                    dispersion=data.get("dispersion", 0),
                    confidence=data.get("confidence", 0),
                )

        # Return sorted by timestamp
        return sorted(fixations_by_id.values(), key=lambda f: f.timestamp)

    def _load_blink_data(self) -> list[BlinkSample]:
        """Load blink events from blinks.pldata.

        Pupil's blink detector emits separate 'onset' and 'offset' events,
        often duplicated from both eyes and not always in timestamp order.
        This method sorts by timestamp, clusters nearby duplicate events
        (within 100ms), pairs onset→offset when the offset arrives within
        500ms, and keeps unpaired onsets (duration=-1) since many real
        blinks don't get an offset detected.
        """
        blink_path = self.path / "blinks.pldata"
        if not blink_path.exists():
            return []

        # Read all events
        events: list[tuple[str, float, float]] = []  # (type, timestamp, confidence)
        with open(blink_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
            for topic, payload in unpacker:
                data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                events.append((
                    data.get("type", "unknown"),
                    data.get("timestamp", 0),
                    data.get("confidence", 0),
                ))

        # Sort by timestamp — events from both eyes can arrive out of order
        events.sort(key=lambda e: e[1])

        # Cluster duplicate events of the same type within 100ms
        # (these are the same blink detected by both eyes)
        cluster_gap = 0.1  # seconds
        clusters: list[tuple[str, float, float]] = []

        i = 0
        while i < len(events):
            event_type, ts, conf = events[i]
            timestamps = [ts]
            confidences = [conf]

            j = i + 1
            while (
                j < len(events)
                and events[j][0] == event_type
                and events[j][1] - timestamps[0] < cluster_gap
            ):
                timestamps.append(events[j][1])
                confidences.append(events[j][2])
                j += 1

            clusters.append((
                event_type,
                sum(timestamps) / len(timestamps),
                max(confidences),
            ))
            i = j

        # Pair each onset with the next offset (no duration cap — let classification handle it)
        blinks: list[BlinkSample] = []
        i = 0
        while i < len(clusters):
            if clusters[i][0] == "onset":
                onset_ts = clusters[i][1]
                onset_conf = clusters[i][2]

                if (
                    i + 1 < len(clusters)
                    and clusters[i + 1][0] == "offset"
                ):
                    offset_ts = clusters[i + 1][1]
                    offset_conf = clusters[i + 1][2]
                    duration_ms = (offset_ts - onset_ts) * 1000
                    avg_conf = (onset_conf + offset_conf) / 2
                    blinks.append(BlinkSample(
                        timestamp=onset_ts,
                        duration_ms=duration_ms,
                        confidence=avg_conf,
                        blink_type=classify_blink(duration_ms),
                    ))
                    i += 2
                else:
                    # Unpaired onset — real blink but no offset detected
                    blinks.append(BlinkSample(
                        timestamp=onset_ts,
                        duration_ms=-1,
                        confidence=onset_conf,
                        blink_type=BlinkType.BLINK,
                    ))
                    i += 1
            else:
                i += 1

        return blinks

    def _detect_flutter_from_blinks(self) -> list[FlutterEvent]:
        """Detect flutter as bursts of rapid blinks (3+ in 2s window).

        Uses causal detection: simulates what a real-time streaming tracker
        would see. A flutter is detected at the moment the Nth blink arrives
        (not retroactively from the first blink), and ends when the window
        no longer qualifies.
        """
        if len(self.blink_data) < FLUTTER_MIN_BLINKS:
            return []

        blink_ts = self.blink_timestamps
        events: list[FlutterEvent] = []
        flutter_start: float | None = None
        flutter_blink_start_idx: int | None = None

        for i, ts in enumerate(blink_ts):
            # Count blinks in the trailing window ending at this blink
            window_start = ts - FLUTTER_WINDOW_S
            count = int(np.sum((blink_ts >= window_start) & (blink_ts <= ts)))

            if count >= FLUTTER_MIN_BLINKS:
                if flutter_start is None:
                    # Flutter just became detectable at this blink; note the
                    # start of the window so we can count the full burst later
                    flutter_start = ts
                    flutter_blink_start_idx = int(np.searchsorted(blink_ts, window_start))
            else:
                if flutter_start is not None:
                    # Flutter just ended
                    events.append(FlutterEvent(
                        timestamp=flutter_start,
                        duration_s=blink_ts[i - 1] - flutter_start + 0.2,
                        blink_count=i - flutter_blink_start_idx,
                    ))
                    flutter_start = None
                    flutter_blink_start_idx = None

        # Close any open flutter at end of recording
        if flutter_start is not None:
            events.append(FlutterEvent(
                timestamp=flutter_start,
                duration_s=blink_ts[-1] - flutter_start + 0.2,
                blink_count=len(blink_ts) - flutter_blink_start_idx,
            ))

        logger.info("[Recording] Flutter detection: %d event(s)", len(events))
        return events

    def get_flutter_at_timestamp(self, timestamp: float) -> FlutterEvent | None:
        """Get a flutter event active at the given timestamp."""
        if not self.flutter_data:
            return None

        idx = int(np.searchsorted(self.flutter_timestamps, timestamp))

        for i in range(max(0, idx - 1), min(len(self.flutter_data), idx + 1)):
            event = self.flutter_data[i]
            if event.timestamp <= timestamp <= event.timestamp + event.duration_s:
                return event

        return None

    def _load_eye_timestamps(self) -> None:
        """Load eye camera timestamps for sync."""
        eye_ts_path = self.path / "eye0_timestamps.npy"
        if eye_ts_path.exists():
            self.eye_timestamps = np.load(eye_ts_path)

    @property
    def eye_video(self) -> cv2.VideoCapture | None:
        """Get the eye camera video capture object (lazy loaded)."""
        if self._eye_video is None:
            eye_path = self.path / "eye0.mp4"
            if not eye_path.exists():
                return None
            self._eye_video = cv2.VideoCapture(str(eye_path))
        return self._eye_video

    def get_eye_frame_for_world_frame(self, world_frame_index: int) -> np.ndarray | None:
        """Get the eye camera frame closest to a world frame's timestamp."""
        if self.eye_timestamps is None or self.eye_video is None:
            return None

        world_ts = self.get_frame_timestamp(world_frame_index)
        # Binary search for closest eye frame
        eye_idx = int(np.searchsorted(self.eye_timestamps, world_ts))
        eye_idx = max(0, min(eye_idx, len(self.eye_timestamps) - 1))

        # Seek and read
        if eye_idx != self._eye_video_pos:
            self._eye_video.set(cv2.CAP_PROP_POS_FRAMES, eye_idx)
            self._eye_video_pos = eye_idx

        ret, frame = self._eye_video.read()
        if ret:
            self._eye_video_pos += 1
            return frame
        return None

    def get_blink_at_timestamp(self, timestamp: float, window: float = 0.5) -> BlinkSample | None:
        """Get a blink event active at the given timestamp.

        Returns the blink if timestamp falls within the blink's duration window.
        """
        if not self.blink_data:
            return None

        idx = int(np.searchsorted(self.blink_timestamps, timestamp))

        # Check nearby blinks
        for i in range(max(0, idx - 1), min(len(self.blink_data), idx + 1)):
            blink = self.blink_data[i]
            # Use known duration, or assume ~200ms for unpaired onsets
            duration_s = blink.duration_ms / 1000.0 if blink.duration_ms >= 0 else 0.2
            if blink.timestamp <= timestamp <= blink.timestamp + duration_s:
                return blink

        return None

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
            fixation_count=len(self.fixation_data),
            blink_count=len(self.blink_data),
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

    def fixation_to_pixel(
        self,
        fixation: FixationSample,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> tuple[int, int]:
        """Convert fixation normalized position to pixel coordinates."""
        width = frame_width or self.world_resolution[0]
        height = frame_height or self.world_resolution[1]

        x = int(fixation.norm_pos[0] * width)
        y = int((1 - fixation.norm_pos[1]) * height)  # Flip Y axis

        return (x, y)

    def get_fixation_for_frame(self, frame_index: int) -> FixationSample | None:
        """Get the fixation that starts closest to this frame's timestamp.

        Only returns a fixation if its start time is within one frame period
        of the frame timestamp. This ensures we trigger once per fixation.
        """
        if not self.fixation_data:
            return None

        frame_ts = self.get_frame_timestamp(frame_index)
        frame_period = 1.0 / self.fps if self.fps > 0 else 0.033

        for fixation in self.fixation_data:
            # Check if fixation starts within this frame's time window
            if abs(fixation.timestamp - frame_ts) <= frame_period / 2:
                return fixation

        return None

    def close(self) -> None:
        """Release video resources."""
        if self._video is not None:
            self._video.release()
            self._video = None
        if self._eye_video is not None:
            self._eye_video.release()
            self._eye_video = None

    def __enter__(self) -> "Recording":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
