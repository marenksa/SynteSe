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

from pupil_tracker.input.recording import FlutterEvent, Recording
from pupil_tracker.output import (
    MultiSink, PureDataSink,
    apply_gamma, build_gamma_lut,
    draw_brightness_bar, draw_color_info, draw_eye_panel,
    draw_gaze_crosshair, draw_region_box,
)
from pupil_tracker.patches import Patch, load_patch
from pupil_tracker.signals.bus import OutputBus, SignalBus
from pupil_tracker.signals.env_color import ColorAnalyzer, ColorReading
from pupil_tracker.signals.env_scene_change import SceneChangeDetector
from pupil_tracker.signals.eye_gaze import GazeVelocityTracker

if TYPE_CHECKING:
    from pupil_tracker.input.recording import GazeSample


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
        patch: Patch | None = None,
        region_size: int = 50,
        show_overlay: bool = False,
        gamma: float = 1.0,
    ):
        self.recording = recording
        self.analyzer = analyzer
        self.output = output
        self.region_size = region_size
        self.show_overlay = show_overlay
        self.gamma = gamma
        self.frame_index = 0
        self.playing = False
        self.playback_speed = 1.0
        self.last_reading: ColorReading | None = None

        self._gamma_lut = build_gamma_lut(gamma)

        # Display settings
        self.gaze_radius = 20
        self.gaze_color = (0, 255, 0)
        self.gaze_thickness = 2
        self.low_confidence_color = (0, 165, 255)
        self.confidence_threshold = 0.6
        self.window_name = f"Gaze Player - {recording.recording_name}"

        # Signal pipeline
        self._signals = SignalBus()
        self._outputs = OutputBus(pd_sink)
        self._patch = patch if patch is not None else load_patch("color_music")
        self._scene_detector = SceneChangeDetector()
        self._gaze_vel = GazeVelocityTracker()

        # Flutter transition tracking (recording has no streaming tracker)
        self._prev_in_flutter = False
        self._last_flutter_event: FlutterEvent | None = None

        # Blink/flutter display state
        self._blink_flash_until = 0.0
        self._last_blink_label: str | None = None
        self._flutter_flash_until = 0.0
        self._last_flutter: FlutterEvent | None = None

    def apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        if self.gamma == 1.0:
            return frame
        return apply_gamma(frame, self._gamma_lut)

    def extract_region(
        self, frame: np.ndarray, gaze: GazeSample
    ) -> GazeRegionCompat | None:
        height, width = frame.shape[:2]
        gaze_x, gaze_y = self.recording.gaze_to_pixel(gaze, width, height)

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
        if self.last_reading is None:
            return frame
        reading = self.last_reading
        draw_brightness_bar(frame, reading.smoothed_brightness, x=10, y=80)
        draw_color_info(frame, reading, x=10, y=30)
        return frame

    def draw_gaze(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
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
        eye_frame = self.recording.get_eye_frame_for_world_frame(frame_index)
        world_ts = self.recording.get_frame_timestamp(frame_index)

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
        height, width = frame.shape[:2]
        panel_height = 60
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        timestamp = self.recording.get_frame_timestamp(self.frame_index)
        elapsed = timestamp - self.recording.start_time_s
        total_frames = self.recording.frame_count

        cv2.putText(panel, f"Frame: {self.frame_index}/{total_frames - 1}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, f"Time: {elapsed:.2f}s / {self.recording.duration_s:.2f}s",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        status = "PLAYING" if self.playing else "PAUSED"
        status_color = (0, 255, 0) if self.playing else (0, 165, 255)
        cv2.putText(panel, status, (width - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(panel, f"Speed: {self.playback_speed:.1f}x", (width - 120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        progress = self.frame_index / max(1, total_frames - 1)
        bar_width = width - 250
        bar_x, bar_y = 200, 35
        cv2.rectangle(panel, (bar_x, bar_y - 5), (bar_x + bar_width, bar_y + 5), (80, 80, 80), -1)
        cv2.rectangle(panel, (bar_x, bar_y - 5),
                      (bar_x + int(bar_width * progress), bar_y + 5), (0, 200, 0), -1)

        return np.vstack([frame, panel])

    def draw_help(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        height, width = frame.shape[:2]
        cv2.rectangle(overlay, (50, 50), (width - 50, height - 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        for i, line in enumerate([
            "KEYBOARD CONTROLS", "",
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
        ]):
            cv2.putText(frame, line, (80, 100 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def seek_to_percent(self, percent: float) -> None:
        total_frames = self.recording.frame_count
        self.frame_index = max(0, min(int(total_frames * percent), total_frames - 1))

    def _reset_on_seek(self) -> None:
        """Reset all stateful components after a seek."""
        self._blink_flash_until = 0.0
        self._flutter_flash_until = 0.0
        self._prev_in_flutter = False
        self._last_flutter_event = None
        self._scene_detector.reset()
        self._gaze_vel.reset()
        self._patch.reset()
        if self.analyzer is not None:
            self.analyzer.reset()

    def run(self) -> None:
        """Run the interactive video player."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        show_help = False
        total_frames = self.recording.frame_count
        need_seek = True
        current_frame: np.ndarray | None = None

        print(f"\nPlaying: {self.recording.recording_name}")
        print(f"Duration: {self.recording.duration_s:.1f}s, {total_frames} frames @ {self.recording.fps:.1f} FPS")
        print(f"Gaze samples: {len(self.recording.gaze_data)}")
        print(f"Fixations: {len(self.recording.fixation_data)}")
        print(f"Blinks: {len(self.recording.blink_data)}")
        print(f"Flutter events: {len(self.recording.flutter_data)}")
        if self.recording.eye_video is not None:
            print("Eye camera: available")
        print("\nPress 'H' for keyboard controls, 'Q' to quit\n")

        self.recording.seek(self.frame_index)
        frame_duration_s = 1.0 / self.recording.fps
        next_frame_time = time.perf_counter()

        while True:
            if need_seek:
                self.recording.seek(self.frame_index)
                need_seek = False
                next_frame_time = time.perf_counter()
                self._reset_on_seek()

            result = self.recording.read_next_frame()
            if result is None:
                if self.frame_index >= total_frames:
                    self.frame_index = 0
                    need_seek = True
                    continue
                print(f"Failed to read frame {self.frame_index}")
                break

            actual_index, current_frame = result
            self.frame_index = actual_index
            current_frame = self.apply_gamma(current_frame)

            # --- Analysis and patch dispatch (only when playing) ---
            if self.analyzer is not None and self.playing:
                world_ts = self.recording.get_frame_timestamp(self.frame_index)
                gaze = self.recording.get_gaze_for_frame(self.frame_index)
                blink = self.recording.get_blink_at_timestamp(world_ts)
                flutter = self.recording.get_flutter_at_timestamp(world_ts)
                fixation = self.recording.get_fixation_for_frame(self.frame_index)
                in_flutter = flutter is not None

                # --- Build signal bus ---
                self._signals.clear_events()
                self._signals.timestamp = world_ts
                self._signals.frame_width = current_frame.shape[1]
                self._signals.frame_height = current_frame.shape[0]

                # Eye signals
                if gaze is not None:
                    self._signals.eye.confidence = gaze.confidence
                    self._signals.eye.norm_pos = gaze.norm_pos
                    px = self.recording.gaze_to_pixel(
                        gaze, current_frame.shape[1], current_frame.shape[0]
                    )
                    self._signals.eye.px_pos = px
                    self._gaze_vel.update(
                        gaze.norm_pos, gaze.timestamp,
                        current_frame.shape[1], current_frame.shape[0],
                    )
                    self._signals.eye.velocity_px_s = self._gaze_vel.velocity

                if blink is not None:
                    self._signals.eye.blink = blink
                self._signals.eye.is_eyes_closed = blink is not None

                if fixation is not None:
                    self._signals.eye.fixation_id = fixation.id

                # Flutter: detect end transition
                if self._prev_in_flutter and not in_flutter and self._last_flutter_event is not None:
                    self._signals.eye.flutter = self._last_flutter_event
                self._signals.eye.is_flutter_active = in_flutter
                self._signals.eye.flutter_blink_count = flutter.blink_count if flutter else 0
                if in_flutter and flutter is not None:
                    self._last_flutter_event = flutter
                self._prev_in_flutter = in_flutter

                self._signals.eye.total_blinks = len(self.recording.blink_data)
                self._signals.eye.total_flutters = len(self.recording.flutter_data)

                # Env signals
                self._signals.env.scene_change = self._scene_detector.update(current_frame)

                if gaze is not None and gaze.confidence >= self.confidence_threshold:
                    region = self.extract_region(current_frame, gaze)
                    if region is not None:
                        color_reading = self.analyzer.analyze(region)
                        self.last_reading = color_reading
                        if self.output is not None:
                            self.output.emit(color_reading)

                        self._signals.env.hue = color_reading.smoothed_hue
                        self._signals.env.hue_normalized = color_reading.smoothed_hue / 179.0
                        self._signals.env.saturation = color_reading.saturation
                        self._signals.env.brightness = color_reading.smoothed_brightness
                        self._signals.env.brightness_normalized = (
                            color_reading.smoothed_brightness / 255.0
                        )
                        self._signals.env.note = color_reading.note
                        self._signals.env.octave = color_reading.octave
                        self._signals.env.midi_note = color_reading.midi_note
                        self._signals.env.raw_midi_note = color_reading.raw_midi_note
                        self._signals.has_env_reading = True

                # --- Dispatch to patch ---
                self._patch.update(self._signals, self._outputs)

            # --- Draw overlays ---
            display_frame = current_frame.copy()
            if self.show_overlay:
                display_frame = self.draw_color_info(display_frame)
            display_frame = self.draw_gaze(display_frame, self.frame_index)
            display_frame = self.draw_eye_camera(display_frame, self.frame_index)
            display_frame = self.draw_info(display_frame)
            if show_help:
                display_frame = self.draw_help(display_frame)

            cv2.imshow(self.window_name, display_frame)

            # Timing
            if self.playing:
                next_frame_time += frame_duration_s / self.playback_speed
                wait_ms = max(1, int((next_frame_time - time.perf_counter()) * 1000))
            else:
                wait_ms = 50

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                self.playing = not self.playing
                if not self.playing:
                    need_seek = True
            elif key == ord("h"):
                show_help = not show_help
            elif key == 81 or key == 2:   # Left arrow
                self.playing = False
                self.frame_index = max(0, self.frame_index - 30)
                need_seek = True
            elif key == 83 or key == 3:   # Right arrow
                self.playing = False
                self.frame_index = min(total_frames - 1, self.frame_index + 30)
                need_seek = True
            elif key == 80 or key == 0:   # Home
                self.frame_index = 0
                need_seek = True
            elif key == 87 or key == 1:   # End
                self.frame_index = total_frames - 1
                need_seek = True
            elif key == ord("["):
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
            elif key == ord("]"):
                self.playback_speed = min(4.0, self.playback_speed + 0.25)
            elif ord("0") <= key <= ord("9"):
                self.seek_to_percent((key - ord("0")) / 10.0)
                need_seek = True

            if not self.playing:
                need_seek = True

            if self.playing and self.frame_index >= total_frames - 1:
                self.frame_index = 0
                need_seek = True

        cv2.destroyAllWindows()


def main() -> None:
    """Main entry point for the gaze video player."""
    parser = argparse.ArgumentParser(
        description="Video player with gaze overlay for Pupil Capture recordings"
    )
    parser.add_argument("recording_path", type=str)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--overlay", action="store_true",
                        help="Show color/note/brightness overlay")
    parser.add_argument("--patch", type=str, default="color_music",
                        help="Patch to use for mapping signals to outputs")
    parser.add_argument("--pd", action="store_true",
                        help="Send output to Pure Data via FUDI protocol")
    parser.add_argument("--pd-host", type=str, default="127.0.0.1")
    parser.add_argument("--pd-port", type=int, default=9001)
    parser.add_argument("--region-size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (< 1.0 brightens)")

    args = parser.parse_args()

    recording_path = Path(args.recording_path)
    if not recording_path.exists():
        print(f"Error: Recording not found: {recording_path}")
        sys.exit(1)

    show_overlay = args.overlay or args.pd
    analyzer = ColorAnalyzer() if show_overlay else None
    pd_sink = None
    if args.pd:
        pd_sink = PureDataSink(host=args.pd_host, port=args.pd_port)
        print(f"Pure Data output enabled: {args.pd_host}:{args.pd_port}")

    patch = load_patch(args.patch)

    try:
        with Recording(recording_path) as recording:
            info = recording.get_info()
            print(f"Loaded recording: {info.recording_name}")
            print(f"  Duration: {info.duration_s:.1f}s  Frames: {info.frame_count}")
            print(f"  Resolution: {info.world_resolution[0]}x{info.world_resolution[1]}")
            print(f"  Gaze: {info.gaze_count}  Fixations: {info.fixation_count}  Blinks: {info.blink_count}")
            if args.gamma != 1.0:
                print(f"  Gamma: {args.gamma}")
            print(f"  Patch: {args.patch}")

            player = GazeVideoPlayer(
                recording,
                analyzer=analyzer,
                pd_sink=pd_sink,
                patch=patch,
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
        if pd_sink is not None:
            pd_sink.send("confidence", 1.0)
            pd_sink.send("am_lfo", 0)
            pd_sink.close()


if __name__ == "__main__":
    main()
