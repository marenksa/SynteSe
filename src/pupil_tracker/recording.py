"""Pupil Capture recording loader and data access."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import msgpack
import numpy as np

logger = logging.getLogger(__name__)

# Eye closure detection tuning constants
CONFIDENCE_THRESHOLD = 0.5
MIN_CLOSURE_MS_BINOCULAR = 50
MIN_CLOSURE_MS_MONOCULAR = 50  # Safe with clean-entry filter
BLINK_MAX_MS = 400
INTENTIONAL_MIN_MS = 500
BINOCULAR_SYNC_TOLERANCE_S = 0.015  # 15ms
GAP_BRIDGE_SAMPLES = 2  # Bridge gaps of ≤2 high-confidence samples between low runs
CLEAN_ENTRY_WINDOW_S = 0.2  # 200ms window before onset to check for stable tracking
CLEAN_ENTRY_MIN_CONFIDENCE = 0.8  # Mean confidence required in pre-onset window

# Flutter detection tuning constants
FLUTTER_WINDOW_S = 3.0  # Sliding window size
FLUTTER_MIN_CROSSINGS = 10  # Min high→low transitions in window to qualify
FLUTTER_STEP_S = 0.5  # Window step size (every 0.5s)
FLUTTER_SMOOTH_S = 0.03  # Median filter width to suppress single-frame noise
FLUTTER_HIGH_THRESH = 0.6  # Confidence must rise above this to "arm" a crossing
FLUTTER_LOW_THRESH = 0.3  # Confidence must drop below this to count a crossing
FLUTTER_MIN_CROSSING_GAP_S = 0.1  # Debounce: min time between counted crossings


class EyeClosureType(Enum):
    """Classification of eye closure events by duration."""

    BLINK = "blink"  # <400ms
    INTENTIONAL = "intentional"  # >500ms
    AMBIGUOUS = "ambiguous"  # 400-500ms


@dataclass
class EyeClosureEvent:
    """An eye closure event detected from pupil confidence drops."""

    timestamp: float
    duration_ms: float
    closure_type: EyeClosureType
    binocular: bool  # True if both eyes confirmed low confidence


@dataclass
class FlutterEvent:
    """A rapid eye flutter burst detected from confidence oscillations."""

    timestamp: float  # Start of the flutter burst
    duration_s: float  # How long the flutter pattern lasted
    crossing_count: int  # Number of high→low crossings detected
    crossing_rate: float  # Crossings per second


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
class BlinkSample:
    """A blink event paired from onset/offset detections."""

    timestamp: float  # Onset timestamp
    duration_ms: float  # Duration in milliseconds (0 if no offset detected)
    confidence: float  # Average confidence of onset/offset


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

        # Load blink data
        self.blink_data = self._load_blink_data()
        self.blink_timestamps = np.array([b.timestamp for b in self.blink_data])

        # Eye closure detection (confidence-based)
        self._pupil_confidence = self._load_pupil_confidence()
        self.eye_closure_data = self._detect_eye_closures()
        self.eye_closure_timestamps = np.array(
            [e.timestamp for e in self.eye_closure_data]
        ) if self.eye_closure_data else np.array([])

        # Flutter detection (confidence oscillation)
        self.flutter_data = self._detect_flutter()
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

        # Pair onset clusters with the next offset cluster (if within 500ms)
        max_blink_duration = 0.5  # seconds — beyond this, don't pair
        blinks: list[BlinkSample] = []
        i = 0
        while i < len(clusters):
            if clusters[i][0] == "onset":
                onset_ts = clusters[i][1]
                onset_conf = clusters[i][2]

                if (
                    i + 1 < len(clusters)
                    and clusters[i + 1][0] == "offset"
                    and clusters[i + 1][1] - onset_ts < max_blink_duration
                ):
                    offset_ts = clusters[i + 1][1]
                    offset_conf = clusters[i + 1][2]
                    duration_ms = (offset_ts - onset_ts) * 1000
                    avg_conf = (onset_conf + offset_conf) / 2
                    blinks.append(BlinkSample(
                        timestamp=onset_ts,
                        duration_ms=duration_ms,
                        confidence=avg_conf,
                    ))
                    i += 2
                else:
                    # Unpaired onset — real blink but no offset detected
                    blinks.append(BlinkSample(
                        timestamp=onset_ts,
                        duration_ms=-1,
                        confidence=onset_conf,
                    ))
                    i += 1
            else:
                i += 1

        return blinks

    def _load_pupil_confidence(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Load per-eye pupil confidence from pupil.pldata.

        Returns:
            Dict mapping eye_id (0 or 1) to (timestamps, confidences) arrays.
        """
        pupil_path = self.path / "pupil.pldata"
        if not pupil_path.exists():
            return {}

        samples: dict[int, list[tuple[float, float]]] = {0: [], 1: []}

        with open(pupil_path, "rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False)
            for topic, payload in unpacker:
                # Only use 2d detector results
                if topic not in ("pupil.0.2d", "pupil.1.2d"):
                    continue
                data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                eye_id = data.get("id", 0)
                if eye_id in samples:
                    samples[eye_id].append((
                        data.get("timestamp", 0.0),
                        data.get("confidence", 0.0),
                    ))

        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for eye_id, eye_samples in samples.items():
            if eye_samples:
                eye_samples.sort(key=lambda s: s[0])
                timestamps = np.array([s[0] for s in eye_samples])
                confidences = np.array([s[1] for s in eye_samples])
                result[eye_id] = (timestamps, confidences)

        return result

    def _detect_eye_closures(self) -> list[EyeClosureEvent]:
        """Detect eye closure events from pupil confidence data.

        Uses binocular confirmation when both eyes are available (requiring both
        eyes to show low confidence simultaneously), falling back to monocular
        detection with a higher minimum duration threshold.
        """
        if not self._pupil_confidence:
            return []

        has_eye0 = 0 in self._pupil_confidence
        has_eye1 = 1 in self._pupil_confidence

        if has_eye0 and has_eye1:
            return self._detect_binocular_closures()
        elif has_eye0:
            logger.warning(
                "Only eye0 available — using monocular closure detection "
                "(higher false-positive rate)"
            )
            return self._detect_monocular_closures(eye_id=0)
        elif has_eye1:
            logger.warning(
                "Only eye1 available — using monocular closure detection "
                "(higher false-positive rate)"
            )
            return self._detect_monocular_closures(eye_id=1)
        return []

    @staticmethod
    def _median_filter_1d(arr: np.ndarray, size: int) -> np.ndarray:
        """Simple 1D median filter (no scipy dependency)."""
        half = size // 2
        result = np.empty_like(arr)
        padded = np.pad(arr, half, mode="edge")
        for i in range(len(arr)):
            result[i] = np.median(padded[i : i + size])
        return result

    def _detect_flutter(self) -> list[FlutterEvent]:
        """Detect rapid eye flutter from confidence signal oscillations.

        Pipeline:
        1. Median-filter the confidence signal (~30ms) to remove single-frame noise
        2. Apply hysteresis thresholding (must rise above 0.6 then drop below 0.3)
        3. Debounce crossings (min 100ms apart — a real blink cycle is ≥150ms)
        4. Count crossings in sliding windows, merge overlapping qualifying windows

        This is conservative: normal blinking, tracking noise, and sustained closure
        should NOT trigger flutter. Only deliberate rapid fluttering (~4 Hz) should.
        """
        if 0 not in self._pupil_confidence:
            logger.info("[Recording] Flutter detection: skipped (no eye0 data)")
            return []

        timestamps, confidences = self._pupil_confidence[0]
        if len(timestamps) < 2:
            logger.info("[Recording] Flutter detection: skipped (insufficient samples)")
            return []

        # Step 1: Median filter to suppress single-frame tracking glitches
        dt = float(np.median(np.diff(timestamps)))
        sample_rate = 1.0 / dt if dt > 0 else 120.0
        kernel = max(3, int(FLUTTER_SMOOTH_S * sample_rate) | 1)  # Ensure odd
        smoothed = self._median_filter_1d(confidences, kernel)

        # Step 2+3: Hysteresis crossing detection with debounce
        armed = smoothed[0] >= FLUTTER_HIGH_THRESH
        crossing_times_list: list[float] = []
        last_crossing_t = -999.0
        for i in range(1, len(smoothed)):
            if armed and smoothed[i] < FLUTTER_LOW_THRESH:
                t = timestamps[i]
                if t - last_crossing_t >= FLUTTER_MIN_CROSSING_GAP_S:
                    crossing_times_list.append(t)
                    last_crossing_t = t
                armed = False
            elif not armed and smoothed[i] >= FLUTTER_HIGH_THRESH:
                armed = True

        crossing_times = np.array(crossing_times_list) if crossing_times_list else np.array([])

        if len(crossing_times) == 0:
            logger.info(
                "[Recording] Flutter detection: 0 events (%.1fs recording, "
                "eye0 only, 0 crossings total)",
                self.duration_s,
            )
            return []

        logger.debug(
            "[Recording] Flutter detection: %d debounced crossings (kernel=%d, %.0f Hz)",
            len(crossing_times),
            kernel,
            sample_rate,
        )

        # Step 4: Slide window and find qualifying windows
        rec_start = timestamps[0]
        rec_end = timestamps[-1]
        qualifying_windows: list[tuple[float, float, int]] = []  # (start, end, count)

        window_start = rec_start
        while window_start + FLUTTER_WINDOW_S <= rec_end:
            window_end = window_start + FLUTTER_WINDOW_S
            # Count crossings in this window
            mask = (crossing_times >= window_start) & (crossing_times < window_end)
            count = int(np.sum(mask))
            if count >= FLUTTER_MIN_CROSSINGS:
                qualifying_windows.append((window_start, window_end, count))
            window_start += FLUTTER_STEP_S

        if not qualifying_windows:
            logger.info(
                "[Recording] Flutter detection: 0 events (%.1fs recording, "
                "eye0 only, %d crossings total)",
                self.duration_s,
                len(crossing_times),
            )
            return []

        # Merge overlapping qualifying windows into contiguous events
        events: list[FlutterEvent] = []
        merge_start, merge_end, merge_max = qualifying_windows[0]
        for w_start, w_end, w_count in qualifying_windows[1:]:
            if w_start <= merge_end:
                # Overlapping — extend
                merge_end = max(merge_end, w_end)
                merge_max = max(merge_max, w_count)
            else:
                # Gap — emit previous and start new
                duration = merge_end - merge_start
                events.append(FlutterEvent(
                    timestamp=merge_start,
                    duration_s=duration,
                    crossing_count=merge_max,
                    crossing_rate=merge_max / FLUTTER_WINDOW_S,
                ))
                merge_start, merge_end, merge_max = w_start, w_end, w_count

        # Emit last merged window
        duration = merge_end - merge_start
        events.append(FlutterEvent(
            timestamp=merge_start,
            duration_s=duration,
            crossing_count=merge_max,
            crossing_rate=merge_max / FLUTTER_WINDOW_S,
        ))

        # Log results
        logger.info(
            "[Recording] Flutter detection: %d event(s)", len(events)
        )
        for evt in events:
            logger.info(
                "  Flutter at %.1fs: %.1fs duration, %d crossings (%.1f/s)",
                evt.timestamp - self.start_time_s,
                evt.duration_s,
                evt.crossing_count,
                evt.crossing_rate,
            )

        return events

    def get_flutter_at_timestamp(self, timestamp: float) -> FlutterEvent | None:
        """Get a flutter event active at the given timestamp."""
        if not self.flutter_data:
            return None

        idx = int(np.searchsorted(self.flutter_timestamps, timestamp))

        # Check nearby events
        for i in range(max(0, idx - 1), min(len(self.flutter_data), idx + 1)):
            event = self.flutter_data[i]
            if event.timestamp <= timestamp <= event.timestamp + event.duration_s:
                return event

        return None

    def _detect_binocular_closures(self) -> list[EyeClosureEvent]:
        """Detect closures requiring both eyes below threshold simultaneously."""
        ts0, conf0 = self._pupil_confidence[0]
        ts1, conf1 = self._pupil_confidence[1]

        # For each eye0 sample, find nearest eye1 sample and check both
        low0 = conf0 < CONFIDENCE_THRESHOLD

        # Find nearest eye1 confidence for each eye0 timestamp
        indices1 = np.searchsorted(ts1, ts0)
        indices1 = np.clip(indices1, 0, len(ts1) - 1)

        # Check if nearest eye1 sample is within sync tolerance
        time_diffs = np.abs(ts1[indices1] - ts0)
        synced = time_diffs < BINOCULAR_SYNC_TOLERANCE_S

        # Both eyes must be low and synced
        low1_at_eye0 = conf1[indices1] < CONFIDENCE_THRESHOLD
        both_low = low0 & low1_at_eye0 & synced

        return self._extract_closure_runs(
            ts0, conf0, both_low, MIN_CLOSURE_MS_BINOCULAR, binocular=True
        )

    def _detect_monocular_closures(self, eye_id: int) -> list[EyeClosureEvent]:
        """Detect closures from a single eye with clean-entry filtering."""
        timestamps, confidences = self._pupil_confidence[eye_id]
        low = confidences < CONFIDENCE_THRESHOLD

        return self._extract_closure_runs(
            timestamps, confidences, low, MIN_CLOSURE_MS_MONOCULAR, binocular=False
        )

    @staticmethod
    def _bridge_gaps(low_mask: np.ndarray) -> np.ndarray:
        """Bridge small gaps of high-confidence samples between low runs.

        A real blink can have 1-2 frame tracking "flickers" mid-closure.
        This merges low runs separated by ≤GAP_BRIDGE_SAMPLES high samples.
        """
        if GAP_BRIDGE_SAMPLES <= 0:
            return low_mask

        result = low_mask.copy()
        # Find transitions: low→high (potential gap start) and high→low (gap end)
        i = 0
        n = len(result)
        while i < n:
            if result[i]:
                # In a low run — find its end
                run_end = i
                while run_end < n and result[run_end]:
                    run_end += 1
                # run_end is now the first high sample after the low run
                # Check if there's a small gap followed by another low run
                gap_end = run_end
                while gap_end < n and not result[gap_end]:
                    gap_end += 1
                gap_size = gap_end - run_end
                if 0 < gap_size <= GAP_BRIDGE_SAMPLES and gap_end < n:
                    # Bridge the gap
                    result[run_end:gap_end] = True
                    # Don't advance i — re-check from the merged run
                    continue
                i = run_end
            else:
                i += 1

        return result

    @staticmethod
    def _has_clean_entry(
        timestamps: np.ndarray,
        confidences: np.ndarray,
        onset_idx: int,
    ) -> bool:
        """Check that tracking was stable before the closure onset.

        Real blinks are sharp drops from confident tracking. Noise periods
        have sustained low/unstable confidence beforehand.
        """
        onset_ts = timestamps[onset_idx]
        window_start_ts = onset_ts - CLEAN_ENTRY_WINDOW_S

        # Find samples in the pre-onset window
        start_idx = int(np.searchsorted(timestamps, window_start_ts))
        pre_window = confidences[start_idx:onset_idx]

        if len(pre_window) < 3:
            # Not enough data before onset — allow it (could be start of recording)
            return True

        return float(np.mean(pre_window)) >= CLEAN_ENTRY_MIN_CONFIDENCE

    @staticmethod
    def _extract_closure_runs(
        timestamps: np.ndarray,
        confidences: np.ndarray,
        low_mask: np.ndarray,
        min_duration_ms: float,
        binocular: bool,
    ) -> list[EyeClosureEvent]:
        """Find contiguous runs of low confidence and classify them."""
        if not np.any(low_mask):
            return []

        # Bridge small gaps to merge fragmented blinks
        bridged = Recording._bridge_gaps(low_mask)

        # Find run boundaries using diff on boolean array
        padded = np.concatenate(([False], bridged, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        events: list[EyeClosureEvent] = []
        for start_idx, end_idx in zip(starts, ends):
            if end_idx > len(timestamps) - 1:
                end_idx = len(timestamps) - 1
            onset_ts = timestamps[start_idx]
            offset_ts = timestamps[min(end_idx, len(timestamps) - 1)]
            duration_ms = (offset_ts - onset_ts) * 1000

            if duration_ms < min_duration_ms:
                continue

            # Clean entry filter: require stable tracking before the dip
            if not Recording._has_clean_entry(timestamps, confidences, start_idx):
                continue

            if duration_ms <= BLINK_MAX_MS:
                closure_type = EyeClosureType.BLINK
            elif duration_ms >= INTENTIONAL_MIN_MS:
                closure_type = EyeClosureType.INTENTIONAL
            else:
                closure_type = EyeClosureType.AMBIGUOUS

            events.append(EyeClosureEvent(
                timestamp=onset_ts,
                duration_ms=duration_ms,
                closure_type=closure_type,
                binocular=binocular,
            ))

        return events

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

    def get_eye_closure_at_timestamp(self, timestamp: float) -> EyeClosureEvent | None:
        """Get an eye closure event active at the given timestamp.

        Returns the closure if timestamp falls within the event's duration.
        """
        if not self.eye_closure_data:
            return None

        idx = int(np.searchsorted(self.eye_closure_timestamps, timestamp))

        # Check nearby events
        for i in range(max(0, idx - 1), min(len(self.eye_closure_data), idx + 1)):
            event = self.eye_closure_data[i]
            duration_s = event.duration_ms / 1000.0
            if event.timestamp <= timestamp <= event.timestamp + duration_s:
                return event

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
