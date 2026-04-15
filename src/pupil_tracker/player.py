"""Video player with gaze overlay for Pupil Capture recordings."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pupil_tracker.analyzer import ColorAnalyzer, ColorReading, Note, NoteEvent
from pupil_tracker.output import MultiSink, PureDataSink
from pupil_tracker.recording import FixationSample, Recording

if TYPE_CHECKING:
    from pupil_tracker.recording import GazeSample


# BGR colors for each note (matching the hue ranges)
NOTE_BGR_COLORS: dict[Note, tuple[int, int, int]] = {
    Note.C: (60, 60, 220),  # Red
    Note.D: (60, 140, 255),  # Orange
    Note.E: (60, 220, 255),  # Yellow
    Note.F: (60, 180, 60),  # Green
    Note.G: (180, 180, 60),  # Cyan
    Note.A: (220, 120, 60),  # Blue
    Note.B: (180, 60, 180),  # Violet
}

NOTE_COLOR_NAMES: dict[Note, str] = {
    Note.C: "Red",
    Note.D: "Orange",
    Note.E: "Yellow",
    Note.F: "Green",
    Note.G: "Cyan",
    Note.A: "Blue",
    Note.B: "Violet",
}


@dataclass
class GazeRegionCompat:
    """Compatibility wrapper to match FrameProcessor.GazeRegion interface."""
    center_x: int
    center_y: int
    region: np.ndarray
    frame_width: int
    frame_height: int
    timestamp: float
    confidence: float


class GazeVideoPlayer:
    """Interactive video player that displays gaze position overlay."""

    def __init__(
        self,
        recording: Recording,
        analyzer: ColorAnalyzer | None = None,
        output: MultiSink | None = None,
        pd_sink: PureDataSink | None = None,
        region_size: int = 50,
        show_overlay: bool = False,
        gamma: float = 1.0,
    ):
        self.recording = recording
        self.analyzer = analyzer
        self.output = output
        self.pd_sink = pd_sink
        self.region_size = region_size
        self.show_overlay = show_overlay
        self.gamma = gamma
        self.frame_index = 0
        self.playing = False
        self.playback_speed = 1.0
        self.last_reading: ColorReading | None = None

        # Build gamma lookup table for performance
        self._gamma_lut = self._build_gamma_lut(gamma)

        # Display settings
        self.gaze_radius = 20
        self.gaze_color = (0, 255, 0)  # Green
        self.gaze_thickness = 2
        self.low_confidence_color = (0, 165, 255)  # Orange
        self.confidence_threshold = 0.6

        # Window name
        self.window_name = f"Gaze Player - {recording.recording_name}"

        # Track which fixations have been triggered (by ID)
        self._triggered_fixations: set[int] = set()

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """Build a lookup table for gamma correction.

        gamma < 1.0 brightens the image (e.g., 0.5)
        gamma > 1.0 darkens the image (e.g., 2.0)
        """
        if gamma == 1.0:
            return np.arange(256, dtype=np.uint8)
        return np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )

    def apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction to a frame using lookup table."""
        if self.gamma == 1.0:
            return frame
        return cv2.LUT(frame, self._gamma_lut)

    def extract_region(
        self, frame: np.ndarray, gaze: GazeSample
    ) -> GazeRegionCompat | None:
        """Extract a region around the gaze point from the frame."""
        height, width = frame.shape[:2]
        gaze_x, gaze_y = self.recording.gaze_to_pixel(gaze, width, height)

        # Calculate region bounds
        half_size = self.region_size // 2
        x1 = max(0, gaze_x - half_size)
        y1 = max(0, gaze_y - half_size)
        x2 = min(width, gaze_x + half_size)
        y2 = min(height, gaze_y + half_size)

        if x2 <= x1 or y2 <= y1:
            return None

        region = frame[y1:y2, x1:x2]
        return GazeRegionCompat(
            center_x=gaze_x,
            center_y=gaze_y,
            region=region,
            frame_width=width,
            frame_height=height,
            timestamp=gaze.timestamp,
            confidence=gaze.confidence,
        )

    def extract_fixation_region(
        self, frame: np.ndarray, fixation: FixationSample
    ) -> GazeRegionCompat | None:
        """Extract a region around the fixation point from the frame."""
        height, width = frame.shape[:2]
        fix_x, fix_y = self.recording.fixation_to_pixel(fixation, width, height)

        # Calculate region bounds
        half_size = self.region_size // 2
        x1 = max(0, fix_x - half_size)
        y1 = max(0, fix_y - half_size)
        x2 = min(width, fix_x + half_size)
        y2 = min(height, fix_y + half_size)

        if x2 <= x1 or y2 <= y1:
            return None

        region = frame[y1:y2, x1:x2]
        return GazeRegionCompat(
            center_x=fix_x,
            center_y=fix_y,
            region=region,
            frame_width=width,
            frame_height=height,
            timestamp=fixation.timestamp,
            confidence=fixation.confidence,
        )

    def draw_color_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw color/note info overlay on frame."""
        if self.last_reading is None:
            return frame

        reading = self.last_reading
        note = reading.note
        octave = reading.octave
        color = NOTE_BGR_COLORS.get(note, (128, 128, 128))
        color_name = NOTE_COLOR_NAMES.get(note, "?")

        x, y, size = 10, 30, 40

        # Draw brightness bar
        bar_x, bar_y, bar_w, bar_h = 10, 80, 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled_w = int(reading.smoothed_brightness / 255 * bar_w)
        if filled_w > 0:
            ratio = reading.smoothed_brightness / 255
            bar_color = (int(50 + ratio * 50), int(50 + ratio * 200), int(50 + ratio * 200))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        cv2.putText(frame, f"Brightness: {reading.smoothed_brightness:.0f}", (bar_x + bar_w + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw color square
        cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
        cv2.rectangle(frame, (x, y), (x + size, y + size), (200, 200, 200), 2)

        # Note name and octave
        note_text = f"{note.name}{octave}"
        cv2.putText(frame, note_text, (x + size + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Color name
        cv2.putText(frame, color_name, (x + size + 10, y + size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # MIDI note
        cv2.putText(frame, f"MIDI: {reading.midi_note}", (x + size + 80, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame

    def draw_gaze(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw gaze position on frame."""
        gaze = self.recording.get_gaze_for_frame(frame_index)

        if gaze is None:
            return frame

        # Get pixel coordinates
        height, width = frame.shape[:2]
        x, y = self.recording.gaze_to_pixel(gaze, width, height)

        # Choose color based on confidence
        if gaze.confidence >= self.confidence_threshold:
            color = self.gaze_color
        else:
            color = self.low_confidence_color

        # Draw crosshair
        cv2.circle(frame, (x, y), self.gaze_radius, color, self.gaze_thickness)
        cv2.line(frame, (x - 30, y), (x + 30, y), color, 1)
        cv2.line(frame, (x, y - 30), (x, y + 30), color, 1)

        # Draw confidence indicator
        conf_text = f"Conf: {gaze.confidence:.2f}"
        cv2.putText(frame, conf_text, (x + 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def draw_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw playback info on frame."""
        height, width = frame.shape[:2]

        # Create info panel at bottom
        panel_height = 60
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        # Frame info
        timestamp = self.recording.get_frame_timestamp(self.frame_index)
        elapsed = timestamp - self.recording.start_time_s
        total_frames = self.recording.frame_count

        info_text = f"Frame: {self.frame_index}/{total_frames - 1}"
        cv2.putText(panel, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        time_text = f"Time: {elapsed:.2f}s / {self.recording.duration_s:.2f}s"
        cv2.putText(panel, time_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Playback status
        status = "PLAYING" if self.playing else "PAUSED"
        status_color = (0, 255, 0) if self.playing else (0, 165, 255)
        cv2.putText(panel, status, (width - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        speed_text = f"Speed: {self.playback_speed:.1f}x"
        cv2.putText(panel, speed_text, (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Progress bar
        progress = self.frame_index / max(1, total_frames - 1)
        bar_width = width - 250
        bar_x = 200
        bar_y = 35
        cv2.rectangle(panel, (bar_x, bar_y - 5), (bar_x + bar_width, bar_y + 5), (80, 80, 80), -1)
        cv2.rectangle(panel, (bar_x, bar_y - 5), (bar_x + int(bar_width * progress), bar_y + 5), (0, 200, 0), -1)

        # Combine frame and panel
        combined = np.vstack([frame, panel])
        return combined

    def draw_help(self, frame: np.ndarray) -> np.ndarray:
        """Draw help overlay on frame."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Semi-transparent background
        cv2.rectangle(overlay, (50, 50), (width - 50, height - 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Help text
        help_lines = [
            "KEYBOARD CONTROLS",
            "",
            "SPACE    - Play/Pause",
            "LEFT     - Previous frame",
            "RIGHT    - Next frame",
            "HOME     - Go to start",
            "END      - Go to end",
            "[        - Decrease speed",
            "]        - Increase speed",
            "0-9      - Jump to 0%-90%",
            "H        - Toggle this help",
            "Q / ESC  - Quit",
        ]

        y_offset = 100
        for line in help_lines:
            cv2.putText(frame, line, (80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y_offset += 35

        return frame

    def seek_to_percent(self, percent: float) -> None:
        """Seek to a percentage of the video."""
        total_frames = self.recording.frame_count
        self.frame_index = int(total_frames * percent)
        self.frame_index = max(0, min(self.frame_index, total_frames - 1))

    def run(self) -> None:
        """Run the interactive video player."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        show_help = False
        total_frames = self.recording.frame_count
        need_seek = True  # Track if we need to seek before reading
        current_frame: np.ndarray | None = None

        print(f"\nPlaying: {self.recording.recording_name}")
        print(f"Duration: {self.recording.duration_s:.1f}s, {total_frames} frames @ {self.recording.fps:.1f} FPS")
        print(f"Gaze samples: {len(self.recording.gaze_data)}")
        print(f"Fixations: {len(self.recording.fixation_data)}")
        print("\nPress 'H' for keyboard controls, 'Q' to quit\n")

        # Initial seek
        self.recording.seek(self.frame_index)
        frame_duration_s = 1.0 / self.recording.fps
        next_frame_time = time.perf_counter()  # Absolute time for next frame

        while True:
            # Get current frame - use sequential read when playing
            if need_seek:
                self.recording.seek(self.frame_index)
                need_seek = False
                next_frame_time = time.perf_counter()  # Reset timing after seek

            result = self.recording.read_next_frame()
            if result is None:
                # End of video or error - try to recover
                if self.frame_index >= total_frames:
                    self.frame_index = 0
                    need_seek = True
                    continue
                print(f"Failed to read frame {self.frame_index}")
                break

            actual_index, current_frame = result
            self.frame_index = actual_index

            # Apply gamma correction to brighten dark footage
            current_frame = self.apply_gamma(current_frame)

            # Color analysis if enabled (only when playing to avoid smoothing lag)
            if self.analyzer is not None and self.playing:
                gaze = self.recording.get_gaze_for_frame(self.frame_index)
                if gaze is not None and gaze.confidence >= self.confidence_threshold:
                    region = self.extract_region(current_frame, gaze)
                    if region is not None:
                        color_reading = self.analyzer.analyze(region)
                        self.last_reading = color_reading
                        if self.output is not None:
                            self.output.emit(color_reading)

                # Check for Pupil fixation events
                fixation = self.recording.get_fixation_for_frame(self.frame_index)
                if (
                    fixation is not None
                    and fixation.id not in self._triggered_fixations
                    and self.pd_sink is not None
                ):
                    # Extract region at fixation point and analyze
                    fix_region = self.extract_fixation_region(current_frame, fixation)
                    if fix_region is not None:
                        reading = self.analyzer.analyze(fix_region)
                        # Create and emit NoteEvent
                        note_event = NoteEvent(
                            timestamp=fixation.timestamp,
                            note=reading.note,
                            octave=reading.octave,
                            midi_note=reading.midi_note,
                            brightness=reading.smoothed_brightness / 255.0,
                            center_x=reading.center_x,
                            center_y=reading.center_y,
                            duration_ms=fixation.duration,
                        )
                        self.pd_sink.emit(note_event)
                        self._triggered_fixations.add(fixation.id)

            # Draw overlays
            display_frame = current_frame.copy()
            if self.show_overlay:
                display_frame = self.draw_color_info(display_frame)
            display_frame = self.draw_gaze(display_frame, self.frame_index)
            display_frame = self.draw_info(display_frame)

            if show_help:
                display_frame = self.draw_help(display_frame)

            # Display
            cv2.imshow(self.window_name, display_frame)

            # Calculate wait time using absolute timing to prevent drift
            if self.playing:
                next_frame_time += frame_duration_s / self.playback_speed
                now = time.perf_counter()
                wait_s = next_frame_time - now
                wait_ms = max(1, int(wait_s * 1000))
            else:
                wait_ms = 50  # Responsive when paused

            # Handle input
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord(" "):  # Space - toggle play/pause
                self.playing = not self.playing
                if not self.playing:
                    # When pausing, we need to re-read current frame on next iteration
                    need_seek = True
            elif key == ord("h"):  # H - toggle help
                show_help = not show_help
            elif key == 81 or key == 2:  # Left arrow
                self.playing = False
                self.frame_index = max(0, self.frame_index - 1)
                need_seek = True
            elif key == 83 or key == 3:  # Right arrow
                self.playing = False
                self.frame_index = min(total_frames - 1, self.frame_index + 1)
                need_seek = True
            elif key == 80 or key == 0:  # Home
                self.frame_index = 0
                need_seek = True
            elif key == 87 or key == 1:  # End
                self.frame_index = total_frames - 1
                need_seek = True
            elif key == ord("["):  # Decrease speed
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
            elif key == ord("]"):  # Increase speed
                self.playback_speed = min(4.0, self.playback_speed + 0.25)
            elif ord("0") <= key <= ord("9"):  # Number keys - jump to percentage
                percent = (key - ord("0")) / 10.0
                self.seek_to_percent(percent)
                need_seek = True

            # If not playing, stay on current frame
            if not self.playing:
                need_seek = True

            # Handle loop at end
            if self.playing and self.frame_index >= total_frames - 1:
                self.frame_index = 0
                need_seek = True

        cv2.destroyAllWindows()


def main() -> None:
    """Main entry point for the gaze video player."""
    parser = argparse.ArgumentParser(
        description="Video player with gaze overlay for Pupil Capture recordings"
    )
    parser.add_argument(
        "recording_path",
        type=str,
        help="Path to the Pupil Capture recording directory",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start playing automatically",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Show color/note/brightness overlay (useful for debugging)",
    )
    parser.add_argument(
        "--pd",
        action="store_true",
        help="Send color-to-music output to Pure Data via FUDI protocol",
    )
    parser.add_argument(
        "--pd-host",
        type=str,
        default="127.0.0.1",
        help="Pure Data host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--pd-port",
        type=int,
        default=9001,
        help="Pure Data FUDI port (default: 9001)",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=50,
        help="Size of gaze region to analyze (default: 50)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction value. Values < 1.0 brighten the image (e.g., 0.5), "
        "values > 1.0 darken it. Default: 1.0 (no correction)",
    )

    args = parser.parse_args()

    # Validate path
    recording_path = Path(args.recording_path)
    if not recording_path.exists():
        print(f"Error: Recording not found: {recording_path}")
        sys.exit(1)

    # Initialize color analyzer and output based on flags
    analyzer = None
    output = None
    pd_sink = None
    show_overlay = args.overlay or args.pd

    if show_overlay:
        analyzer = ColorAnalyzer()

    if args.pd:
        pd_sink = PureDataSink(host=args.pd_host, port=args.pd_port)
        print(f"Pure Data output enabled: {args.pd_host}:{args.pd_port}")
        print("  Using Pupil fixation detection")

    # Load and play
    try:
        with Recording(recording_path) as recording:
            info = recording.get_info()
            print(f"Loaded recording: {info.recording_name}")
            print(f"  Path: {info.path}")
            print(f"  Duration: {info.duration_s:.1f}s")
            print(f"  Resolution: {info.world_resolution[0]}x{info.world_resolution[1]}")
            print(f"  Frames: {info.frame_count}")
            print(f"  Gaze samples: {info.gaze_count}")
            print(f"  Fixations: {info.fixation_count}")

            if args.gamma != 1.0:
                print(f"  Gamma correction: {args.gamma}")

            player = GazeVideoPlayer(
                recording,
                analyzer=analyzer,
                output=output,
                pd_sink=pd_sink,
                region_size=args.region_size,
                show_overlay=show_overlay,
                gamma=args.gamma,
            )
            player.frame_index = args.start_frame
            player.playing = args.autoplay
            player.run()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    finally:
        if output is not None:
            output.close()
        if pd_sink is not None:
            pd_sink.close()


if __name__ == "__main__":
    main()
