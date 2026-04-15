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

from pupil_tracker.analyzer import ColorAnalyzer, ColorReading, NoteEvent, NoteGate
from pupil_tracker.output import MultiSink, PureDataSink, flutter_to_lfo
from pupil_tracker.overlay import (
    apply_gamma,
    build_gamma_lut,
    draw_brightness_bar,
    draw_color_info,
    draw_eye_panel,
    draw_gaze_crosshair,
    draw_region_box,
)
from pupil_tracker.recording import FLUTTER_MIN_BLINKS, BlinkType, FlutterEvent, Recording

if TYPE_CHECKING:
    from pupil_tracker.recording import GazeSample


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
        self._gamma_lut = build_gamma_lut(gamma)

        # Display settings
        self.gaze_radius = 20
        self.gaze_color = (0, 255, 0)  # Green
        self.gaze_thickness = 2
        self.low_confidence_color = (0, 165, 255)  # Orange
        self.confidence_threshold = 0.6

        # Window name
        self.window_name = f"Gaze Player - {recording.recording_name}"

        # Content-based note triggering
        self._note_gate = NoteGate()

        # Blink display state
        self._blink_flash_until = 0.0
        self._last_blink_label: str | None = None

        # Flutter display state
        self._flutter_flash_until = 0.0
        self._last_flutter: FlutterEvent | None = None

        # AM-LFO state: track flutter transitions and intentional blinks
        self._in_flutter = False
        self._last_intentional_ts: float = -1.0

    def apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction to a frame using lookup table."""
        if self.gamma == 1.0:
            return frame
        return apply_gamma(frame, self._gamma_lut)

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

    def draw_color_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw color/note info overlay on frame."""
        if self.last_reading is None:
            return frame

        reading = self.last_reading
        draw_brightness_bar(frame, reading.smoothed_brightness, x=10, y=80)
        draw_color_info(frame, reading, x=10, y=30)

        return frame

    def draw_gaze(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw gaze position on frame."""
        gaze = self.recording.get_gaze_for_frame(frame_index)

        if gaze is None:
            return frame

        height, width = frame.shape[:2]
        x, y = self.recording.gaze_to_pixel(gaze, width, height)

        draw_gaze_crosshair(frame, x, y, gaze.confidence,
                            radius=self.gaze_radius,
                            confidence_threshold=self.confidence_threshold)
        draw_region_box(frame, x, y, self.region_size, gaze.confidence,
                        confidence_threshold=self.confidence_threshold)

        return frame

    def draw_eye_camera(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw eye camera feed and blink indicator in the top-right corner."""
        eye_frame = self.recording.get_eye_frame_for_world_frame(frame_index)

        world_ts = self.recording.get_frame_timestamp(frame_index)

        # Check for active blink
        blink = self.recording.get_blink_at_timestamp(world_ts)
        if blink is not None:
            self._blink_flash_until = world_ts + 0.2
            if blink.duration_ms >= 0:
                self._last_blink_label = (
                    f"{blink.blink_type.value.upper()} {blink.duration_ms:.0f}ms"
                )
            else:
                self._last_blink_label = "BLINK"
        is_blinking = world_ts < self._blink_flash_until

        # Check for flutter
        flutter = self.recording.get_flutter_at_timestamp(world_ts)
        if flutter is not None:
            self._flutter_flash_until = world_ts + 0.2
            self._last_flutter = flutter
        is_flutter = world_ts < self._flutter_flash_until

        flutter_label = None
        if is_flutter and self._last_flutter is not None:
            flutter_label = f"FLUTTER {self._last_flutter.blink_count} blinks"

        draw_eye_panel(
            frame,
            eye_frame,
            is_blink=is_blinking,
            blink_label=self._last_blink_label if is_blinking else None,
            is_flutter=is_flutter,
            flutter_label=flutter_label,
            blink_count=len(self.recording.blink_data),
            flutter_count=len(self.recording.flutter_data),
        )

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
        print(f"Blinks: {len(self.recording.blink_data)}")
        print(f"Flutter events: {len(self.recording.flutter_data)}")
        if self.recording.eye_video is not None:
            print(f"Eye camera: available")
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
                # Reset flash timers so stale indicators don't persist after seek/loop
                self._blink_flash_until = 0.0
                self._flutter_flash_until = 0.0
                self._in_flutter = False
                self._last_intentional_ts = -1.0

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
                # Feed fixation events to the note gate
                fixation = self.recording.get_fixation_for_frame(self.frame_index)
                if fixation is not None:
                    self._note_gate.new_fixation(fixation.id)

                world_ts = self.recording.get_frame_timestamp(self.frame_index)
                flutter = self.recording.get_flutter_at_timestamp(world_ts)
                in_flutter = flutter is not None

                gaze = self.recording.get_gaze_for_frame(self.frame_index)
                if gaze is not None and gaze.confidence >= self.confidence_threshold:
                    region = self.extract_region(current_frame, gaze)
                    if region is not None:
                        color_reading = self.analyzer.analyze(region)
                        self.last_reading = color_reading
                        if self.output is not None:
                            self.output.emit(color_reading)

                        # Content-based note triggering (suppressed during flutter)
                        if (
                            self.pd_sink is not None
                            and not in_flutter
                            and self._note_gate.update(color_reading.midi_note, color_reading.raw_midi_note)
                        ):
                            note_event = NoteEvent(
                                timestamp=color_reading.timestamp,
                                note=color_reading.note,
                                octave=color_reading.octave,
                                midi_note=color_reading.midi_note,
                                brightness=color_reading.smoothed_brightness / 255.0,
                                center_x=color_reading.center_x,
                                center_y=color_reading.center_y,
                            )
                            self.pd_sink.emit(note_event)

                # AM-LFO runs at frame level, independent of gaze confidence
                if self.pd_sink is not None:
                    if self._in_flutter and not in_flutter and self._last_flutter is not None:
                        self.pd_sink.emit_am_lfo(flutter_to_lfo(self._last_flutter.blink_count, FLUTTER_MIN_BLINKS))

                    blink = self.recording.get_blink_at_timestamp(world_ts)
                    if (
                        blink is not None
                        and blink.blink_type == BlinkType.INTENTIONAL
                        and blink.timestamp != self._last_intentional_ts
                    ):
                        self.pd_sink.emit_am_lfo(0)
                        self._last_intentional_ts = blink.timestamp

                if in_flutter and flutter is not None:
                    self._last_flutter = flutter
                self._in_flutter = in_flutter

            # Draw overlays
            display_frame = current_frame.copy()
            if self.show_overlay:
                display_frame = self.draw_color_info(display_frame)
            display_frame = self.draw_gaze(display_frame, self.frame_index)
            display_frame = self.draw_eye_camera(display_frame, self.frame_index)
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
                self.frame_index = max(0, self.frame_index - 30)
                need_seek = True
            elif key == 83 or key == 3:  # Right arrow
                self.playing = False
                self.frame_index = min(total_frames - 1, self.frame_index + 30)
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
        print("  Using content-based note triggering")

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
            print(f"  Blinks: {info.blink_count}")

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
